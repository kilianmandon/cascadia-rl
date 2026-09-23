import math

import numpy as np
from torch import nn
import torch
from torch.nn import functional as F

from config import Config


neighbor_dirs = np.array([
    (1, 1), # Bottom right
    (0, 1), # Right
    (-1, 0), # Top Right
    (-1, -1), # Top Left
    (0, -1), # Left
    (1, 0), # Bottom Left
])

def hex_dist(di, dj):
    return (abs(di) + abs(dj) + abs(di-dj)) // 2

def hex_ring(d):
    if d==0: return [(0, 0)]

    points = np.zeros((6*d, 2))
    for dir_idx in range(6):
        start_dir = neighbor_dirs[dir_idx]
        end_dir = neighbor_dirs[(dir_idx+1)%6]
        diff = end_dir - start_dir
        points[dir_idx*d:(dir_idx+1)*d] = d * start_dir + np.arange(d)[:, None] * diff

    points_tuple = [(i, j) for i,j in points] 
    return points_tuple


class AttentionGridBias(nn.Module):
    def __init__(self, d_model, n_bias_embs, n_heads=8):
        super().__init__()
        self.c = d_model // n_heads
        self.n_heads = n_heads
        self.layer_norm = nn.LayerNorm(d_model)
        self.embedding_b = nn.Embedding(n_bias_embs, n_heads)

        self.linear_q = nn.Linear(d_model, self.c*n_heads)
        self.linear_k = nn.Linear(d_model, self.c*n_heads)
        self.linear_v = nn.Linear(d_model, self.c*n_heads)

        self.linear_out = nn.Linear(self.c*n_heads, d_model)

    def forward(self, x: torch.Tensor, b, token_mask):
        x = self.layer_norm(x)

        q = self.linear_q(x).unflatten(dim=-1, sizes=(self.n_heads, self.c))
        k = self.linear_k(x).unflatten(dim=-1, sizes=(self.n_heads, self.c))
        v = self.linear_v(x).unflatten(dim=-1, sizes=(self.n_heads, self.c))

        bias = self.embedding_b(b)
        bias += -1e8 * (~token_mask[..., None, :, None]).float()

        att_logits = 1/math.sqrt(self.c) * torch.einsum('...ihc,...jhc->...ijh', q, k)
        att_logits += bias
        attn = torch.softmax(att_logits, dim=-2)

        o = torch.einsum('...ijh,...jhc->...ihc', attn, v)
        o = o.flatten(start_dim=-2)

        o = self.linear_out(o)

        return o

class TransitionBlock(nn.Module):
    def __init__(self, d_model, n=2):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.linear_a1 = nn.Linear(d_model, n*d_model)
        self.linear_a2 = nn.Linear(d_model, n*d_model)
        self.linear_transition = nn.Linear(n*d_model, d_model)

    def forward(self, x):
        x = self.layer_norm(x)
        y = F.silu(self.linear_a1(x)) * self.linear_a2(x)

        x = self.linear_transition(y)
        return x

class Transformer(nn.Module):
    def __init__(self, d_model, n_bias_embs, n_blocks=1):
        super().__init__()
        self.attn_blocks = nn.ModuleList([AttentionGridBias(d_model, n_bias_embs) for _ in range(n_blocks)])
        self.trans_blocks = nn.ModuleList([TransitionBlock(d_model) for _ in range(n_blocks)])
        self.layer_norm_out = nn.LayerNorm(d_model)

    def forward(self, x, b, token_mask):
        for attn_block, trans_block in zip(self.attn_blocks, self.trans_blocks):
            x = x + attn_block(x, b, token_mask)
            x = x + trans_block(x)

        x = self.layer_norm_out(x)

        return x

class ActorCriticTransformer(nn.Module):
    def __init__(self, c_single, c_grid, config: Config, d_model=128):
        super().__init__()
        self.global_mlp = nn.Sequential(
            nn.Linear(c_single, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.max_size = config.MAX_GRID_SIZE

        self.grid_embedding = nn.Linear(c_grid, d_model)

        rel_idx_lookup, self.rel_idx_embedding_count = self.build_rel_idx(d_model)
        self.register_buffer('rel_idx_lookup', rel_idx_lookup)

        self.transformer = Transformer(d_model, self.rel_idx_embedding_count)

        self.land_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 6)
        )

        self.animal_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1)
        )

        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1)
        )

        self.global_action = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 23)
        )

    def build_rel_idx(self, d_model):
        small_offsets = [
            (di, dj) for d in range(4) for di, dj in hex_ring(d)
        ]

        rel_idx = {
            d: i for i, d in enumerate(small_offsets)
        }

        for d in range(4, 23):
            for p in hex_ring(d):
                dist_idx = min(d, 10)-4 + len(small_offsets)
                rel_idx[p] = dist_idx

        basic_embedding_count = len(set(rel_idx.values()))
        rel_idx_embedding_count = basic_embedding_count+1
        # Valid index differences in [-23, 23]
        # Special difference: (24, 24) -> index (47, 47)
        # for attention to and from the global token
        rel_idx_lookup = torch.zeros((48, 48), dtype=torch.long)

        for di in range(-23, 24):
            for dj in range(-23, 24):
                if (di, dj) in rel_idx:
                    rel_idx_lookup[di+23, dj+23] = rel_idx[di, dj]

        # Special index for global tokens
        rel_idx_lookup[47, 47] = basic_embedding_count

        return rel_idx_lookup, rel_idx_embedding_count

    def compute_rel_bias(self, grid_indices):
        batch_shape = grid_indices.shape[:-2]
        n_tokens = grid_indices.shape[-2]
        device = grid_indices.device

        grid_diffs = grid_indices[..., :, None, :] - grid_indices[..., None, :, :]
        # Pad to make space for the global token
        padded_shape = batch_shape + (n_tokens+1, n_tokens+1, 2)
        padded_grid_diffs = torch.full(padded_shape, device=device, dtype=int, fill_value=24)
        padded_grid_diffs[..., 1:, 1:, :] = grid_diffs

        # Offset for lookup table
        padded_grid_diffs = padded_grid_diffs + 23
        rel_idx = self.rel_idx_lookup[padded_grid_diffs[..., 0], padded_grid_diffs[..., 1]]

        return rel_idx

    # Action Space:
    # 1: Reroll All
    # 1: Reroll Three
    # 4: Take Pair
    # 16: Take Mixed
    # max_size**2 * 6: place land
    # max_size**2: place animal
    # 1: reject animal
    def build_action(self, land_logits, animal_logits, global_action_logits, grid_indices, token_mask):
        batch_shape = global_action_logits.shape[:-1]
        device = land_logits.device

        n_actions = 1+1+4+16+6*self.max_size**2 + self.max_size**2 + 1
        action_logits = torch.zeros(batch_shape + (n_actions,), device=device)

        action_logits[..., :22] = global_action_logits[..., :22]
        action_logits[..., -1] = global_action_logits[..., -1]

        grid_level_grid_inds = grid_indices+self.max_size//2
        flat_grid_inds = grid_level_grid_inds[..., 0] * self.max_size + grid_level_grid_inds[..., 1]

        land_logits = land_logits * token_mask[..., None]
        land_scatter_inds = 22 + 6*flat_grid_inds[..., None] + torch.arange(6, device=device)
        action_logits = torch.scatter_add(action_logits, dim=-1, index=land_scatter_inds.flatten(-2), src=land_logits.flatten(-2))

        animal_logits = animal_logits * token_mask[..., None]
        animal_scatter_inds = flat_grid_inds + 22 + 6 * self.max_size**2
        action_logits = torch.scatter_add(action_logits, dim=-1, index=animal_scatter_inds, src=animal_logits.flatten(-2))

        return action_logits

    def crop_features(self, features):
        # mask has shape (**batch_shape, n_token)
        mask = features['mask']
        first_masked = ((~mask).int().argmax(dim=-1)).min().item()
        if first_masked == 0:
            first_masked = mask.shape[-1]

        features['mask'] = features['mask'][..., :first_masked]
        features['grid_encoding'] = features['grid_encoding'][..., :first_masked, :]
        features['grid_indices'] = features['grid_indices'][..., :first_masked, :]

        return features

        

    def forward(self, features):
        features = self.crop_features(features)
        grid_encoding = features['grid_encoding']
        grid_indices = features['grid_indices']
        global_state = features['global_state']
        mask = features['mask']
        device = mask.device
        batch_shape = mask.shape[:-1]

        mask_pad = torch.ones(batch_shape + (1,), dtype=mask.dtype, device=device)
        padded_mask = torch.cat((mask_pad, mask), dim=-1)

        x_global = self.global_mlp(global_state)[..., None, :]
        rel_bias = self.compute_rel_bias(grid_indices)
        x_grid = self.grid_embedding(grid_encoding)

        x = torch.cat((x_global, x_grid), dim=-2)
        x = self.transformer(x, rel_bias, padded_mask)

        land_logits = self.land_head(x[..., 1:, :])
        animal_logits = self.animal_head(x[..., 1:, :])
        value_logits = self.value_head(x[..., 0, :])
        global_action_logits = self.global_action(x[..., 0, :])

        action_logits = self.build_action(land_logits, animal_logits, global_action_logits, grid_indices, mask)

        return action_logits, value_logits

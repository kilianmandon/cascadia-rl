import numpy as np
from base_types import Animal, BuiltPlate, GamePhase
import torch
from torch.nn import functional as F

from base_types import GameState, Action, ActionKind
from config import Config
from game import (reroll_all_mask, reroll_three_mask, take_mixed_mask, take_pair_mask,
                  place_land_plate_mask, place_animal_mask)

neighbor_dirs = [
    (1, 1), # Bottom right
    (0, 1), # Right
    (-1, 0), # Top Right
    (-1, -1), # Top Left
    (0, -1), # Left
    (1, 0), # Bottom Left
]

neighbor_dirs_by_orientation = [
    neighbor_dirs[i:] + neighbor_dirs[:i] for i in range(6)
]

def encode_individual_land_state(game_state: GameState, player: int, config: Config):
    land_state = game_state.players[player].plate_grid
    max_size = config.MAX_GRID_SIZE

    # Land-type-by-side, supported animals, built animal
    out_channels = 6*5 + 5 + 5
    out = np.zeros((max_size, max_size, out_channels), dtype=int)

    for (ii,jj), plate in land_state.items():
        i, j = ii+max_size//2, jj+max_size//2

        right_landscape_inds = (np.array([0, 1, 2]) + plate.orientation) % 6
        left_landscape_inds = (np.array([3, 4, 5]) + plate.orientation) % 6

        landscape_encoding = np.zeros(6, dtype=int)
        landscape_encoding[right_landscape_inds] = plate.right_landscape.value
        landscape_encoding[left_landscape_inds] = plate.left_landscape.value

        landscape_onehot = F.one_hot(torch.tensor(landscape_encoding), num_classes=5).numpy().reshape(-1)

        supported_animals = np.array([a.value for a in plate.animals])

        out[i, j, :30] = landscape_onehot
        out[i, j, supported_animals+30] = 1
        if plate.built_animal is not None:
            out[i, j, 35 + plate.built_animal.value] = 1

    return out


def compute_grid_state(game_state: GameState, player_idx: int, config: Config):
    n_players = len(game_state.players)
    individual_land_states = []
    for i in [(i+player_idx)%n_players for i in range(n_players)]:
        individual_land_states.append(encode_individual_land_state(game_state, i, config))

    # channels n_players * 40
    grid_state = np.concatenate(individual_land_states, axis=-1)
    return grid_state


def encode_individual_single_state(game_state: GameState, player: int):
    player_state = game_state.players[player]
    pine_state = np.zeros((10,), dtype=int)
    pine_cone_idx = min(player_state.pine_cones, 9)
    pine_state[pine_cone_idx] = 1

    turns_left = 23 - len(player_state.plate_grid.keys())
    thermo = np.arange(20) < turns_left
    scalar = np.array([turns_left / 20])

    return np.concatenate([pine_state, thermo, scalar], axis=-1)


def encode_pool_state(game_state: GameState):
    # size 4*2*5 + 4*5 + 4*5 = 80
    pool_state = game_state.pool_state

    landscape_enc = np.zeros((40,), dtype=int)
    for i, plate in enumerate(pool_state.plate_pool):
        if plate is not None:
            for j, landscape_val in enumerate([plate.left_landscape.value, plate.right_landscape.value]):
                landscape_enc[10*i + 5*j + landscape_val] = 1

    supported_animals = np.zeros((20,), dtype=int)
    for i, plate in enumerate(pool_state.plate_pool):
        if plate is not None:
            for a in plate.animals:
                supported_animals[5*i + a.value] = 1

    animals = np.zeros((20,), dtype=int)
    for i, a in enumerate(pool_state.animal_pool):
        if a is not None:
            animals[5*i + a.value] = 1


    return np.concatenate((landscape_enc, supported_animals, animals), axis=-1)

def compute_single_state(game_state: GameState, player_idx: int):
    n_players = len(game_state.players)

    individual_single_states = []
    for i in [(i+player_idx)%n_players for i in range(n_players)]:
        individual_single_states.append(encode_individual_single_state(game_state, i))

    group_single_state = np.concatenate(individual_single_states, axis=-1)

    phase_state = F.one_hot(torch.tensor(game_state.game_phase.value), num_classes=3).numpy()
    pool_state = encode_pool_state(game_state)

    player_state = game_state.players[player_idx]

    if player_state.to_place is None:
        picked_state = np.zeros((20))
    else:
        animal, plate = player_state.to_place
        landscapes = [plate.left_landscape.value, plate.right_landscape.value]
        landscape_enc = F.one_hot(torch.tensor(landscapes), num_classes=5).numpy().reshape(-1)
        supported_animals = np.array([1 if a in plate.animals else 0 for a in Animal])
        animals = F.one_hot(torch.tensor(animal.value), num_classes=5).numpy().reshape(-1)

        picked_state = np.concatenate([landscape_enc, supported_animals, animals], axis=-1)

    # shape n_players*(10+21) + 80 + 3 + 20
    return np.concatenate((group_single_state, pool_state, phase_state, picked_state), axis=-1)


def compute_state(game_state: GameState, player_idx: int, config: Config):
    grid_state = compute_grid_state(game_state, player_idx, config)
    single_state = compute_single_state(game_state, player_idx)

    single_state = np.broadcast_to(single_state[None, None, :], grid_state.shape[:2] + (single_state.shape[-1],))

    return np.concatenate((grid_state, single_state), axis=-1)

## Transformer state
def single_plate_encoding(plate: BuiltPlate):
    out = np.zeros(6*5+5+5)
    right_landscape_inds = (np.array([0, 1, 2]) + plate.orientation) % 6
    left_landscape_inds = (np.array([3, 4, 5]) + plate.orientation) % 6

    landscape_encoding = np.zeros(6, dtype=int)
    landscape_encoding[right_landscape_inds] = plate.right_landscape.value
    landscape_encoding[left_landscape_inds] = plate.left_landscape.value

    landscape_onehot = F.one_hot(torch.tensor(landscape_encoding), num_classes=5).numpy().reshape(-1)

    supported_animals = np.array([a.value for a in plate.animals])

    out[:30] = landscape_onehot
    out[supported_animals+30] = 1
    if plate.built_animal is not None:
        out[35 + plate.built_animal.value] = 1

    return out

def build_grid_for_transformer(game_state: GameState, player: int, config: Config):
    land_state = game_state.players[player].plate_grid
    max_size = config.MAX_GRID_SIZE

    # occupied_flag ([1, 0] or [0, 1]), Land-type-by-side, supported animals, built animal
    out_channels = 2 + 6*5 + 5 + 5

    valid_frontier = lambda i,j: (i,j) not in land_state and (-max_size//2 <= i < max_size//2) and (-max_size//2 <= j < max_size//2)

    frontier = set([
        (i+di, j+dj) for i,j in land_state for di,dj in neighbor_dirs if valid_frontier(i+di, j+dj)
    ])

    indices = []
    encodings = []


    for (i,j), plate in land_state.items():
        enc = single_plate_encoding(plate)
        enc = np.concatenate([np.array([1, 0]), enc], axis=-1)
        indices.append((i,j))
        encodings.append(enc)

    for (i,j) in frontier:
        enc = np.zeros(out_channels)
        enc[1] = 1
        encodings.append(enc)
        indices.append((i, j))

    encodings = np.stack(encodings, axis=0)
    indices = np.stack(indices, axis=0)
    return encodings, indices

def crop_pad_to_shape(v, shape):
    slice_inds = tuple(slice(min(d1, d2)) for d1, d2 in zip(v.shape, shape))
    return pad_to_shape(v[slice_inds], shape)

def pad_to_shape(v, shape):
    out = np.zeros(shape, v.dtype)
    slice_inds = tuple(slice(dim) for dim in v.shape)
    out[slice_inds] = v
    return out

def build_transformer_state(game_state: GameState, config: Config):
    global_state = compute_single_state(game_state, game_state.active_player)
    grid_encoding, grid_indices = build_grid_for_transformer(game_state, game_state.active_player, config)
    n_token = grid_encoding.shape[0]
    n_token_max = 127 # leave one token for the global state
    assert n_token <= n_token_max
    c_grid_enc = grid_encoding.shape[1]
    mask = np.ones((n_token,), dtype=bool)

    return {
        'global_state': global_state,
        'grid_encoding': crop_pad_to_shape(grid_encoding, (n_token_max, c_grid_enc)),
        'grid_indices': crop_pad_to_shape(grid_indices, (n_token_max, 2)),
        'mask': crop_pad_to_shape(mask, (n_token_max,))
    }

def flatten_transformer_state(state, n_players=1):
    # n_players*(10+21) + 80 + 3
    flat_state = np.concatenate([
        state['global_state'],
        state['grid_encoding'].reshape(-1),
        state['grid_indices'].reshape(-1),
        state['mask'].reshape(-1),

    ], dtype=float)
    return flat_state

def unflatten_transformer_state(state, n_players=1):
    batch_shape = state.shape[:-1]
    c_global_state = n_players*(10+21) + 80 + 3 + 20
    c_grid_enc = 2 + 6*5 + 5 + 5
    n_token = 127
    c_total_grid = c_grid_enc * n_token
    c_total_grid_inds = n_token * 2
    out_state = {}
    start = 0
    out_state['global_state'] = state[..., :c_global_state]
    start += c_global_state
    out_state['grid_encoding'] = state[..., start:start+c_total_grid].reshape(batch_shape+(n_token, c_grid_enc))
    start += c_total_grid
    out_state['grid_indices'] = state[..., start:start+c_total_grid_inds].reshape(batch_shape + (n_token, 2))
    start += c_total_grid_inds
    out_state['mask'] = state[..., start:start+n_token]

    out_state['grid_indices'] = out_state['grid_indices'].long()
    out_state['mask'] = out_state['mask'].bool()

    return out_state


##

def allowed_land_placement(game_state: GameState, player_idx: int):
    land_state = game_state.players[player_idx].plate_grid
    neighbors = set()
    for pi, pj in land_state:
        for di, dj in neighbor_dirs:
            i, j = pi+di, pj+dj
            if (i,j) not in land_state:
                neighbors.add((i, j))

    return neighbors

def allowed_animal_placement(game_state: GameState, player_idx: int):
    player_state = game_state.players[player_idx]
    land_state = player_state.plate_grid

    return [p for p, plate in land_state.items() if player_state.to_place[0] in plate.animals]


# Action Space:
# 1: Reroll All
# 1: Reroll Three
# 4: Take Pair
# 16: Take Mixed
# max_size**2 * 6: place land
# max_size**2: place animal
def compute_action_mask(game_state: GameState, player_idx: int, config: Config):
    player_state = game_state.players[player_idx]
    land_state = player_state.plate_grid
    phase = game_state.game_phase
    max_size = config.MAX_GRID_SIZE

    n_actions = np.array([1, 1, 4, 16, 6*max_size**2, max_size**2+1])
    n_actions_prefix = np.concatenate((np.array([0]), np.cumsum(n_actions)), axis=0)
    total_n_actions = np.sum(n_actions)
    mask = np.zeros(total_n_actions, dtype=bool)

    match phase:
        case GamePhase.PICKING:
            if reroll_all_mask(game_state, player_idx):
                mask[n_actions_prefix[0]] = True
            if reroll_three_mask(game_state, player_idx):
                mask[n_actions_prefix[1]] = True
            allowed_take_inds = take_pair_mask(game_state, player_idx)['take_idx']
            for i in allowed_take_inds:
                mask[n_actions_prefix[2]+i] = True

            mixed_mask = take_mixed_mask(game_state, player_idx)
            if mixed_mask:
                for i in mixed_mask['take_idx_land']:
                    for j in mixed_mask['take_idx_animal']:
                        mask[n_actions_prefix[3] + 4*i + j] = True

        case GamePhase.PLACING_LAND:
            for i, j in place_land_plate_mask(game_state, player_idx, config)['index_place']:
                i, j = i + max_size // 2, j + max_size // 2
                lower = n_actions_prefix[4] + (i*max_size + j) * 6
                if config.allow_rotating_land:
                    mask[lower: lower+6] = True
                else:
                    mask[lower] = True

        case GamePhase.PLACING_ANIMAL:
            for cell in place_animal_mask(game_state, player_idx)['index_place']:
                if cell is None:
                    continue
                i, j = cell
                i, j = i + max_size // 2, j + max_size // 2
                idx = n_actions_prefix[5] + (i*max_size + j)
                mask[idx] = True

            # The "reject animal" option
            mask[n_actions_prefix[5] + max_size**2] = True

    if np.all(mask==False):
        ...
    return mask


# Action Space:
# 1: Reroll All
# 1: Reroll Three
# 4: Take Pair
# 16: Take Mixed
# max_size**2 * 6: place land
# max_size**2: place animal
# 1: reject animal
def action_from_index(game_state: GameState, player_idx: int, action_idx: int, config: Config):
    player_state = game_state.players[player_idx]
    land_state = player_state.plate_grid
    phase = game_state.game_phase

    max_size = config.MAX_GRID_SIZE
    n_actions = np.array([1, 1, 4, 16, 6*max_size**2, max_size**2+1])
    n_actions_prefix = np.concatenate((np.array([0]), np.cumsum(n_actions)), axis=0)

    if not 0 <= action_idx < n_actions_prefix[-1]:
        raise ValueError(f"Action index {action_idx} is outside the action space.")

    action_category = np.max(np.nonzero(action_idx>=n_actions_prefix)[0])
    offset = action_idx - n_actions_prefix[action_category]

    match action_category:
        case 0:
            # Reroll All
            return Action(ActionKind.REROLL_ALL, {})
        case 1:
            # Reroll Three
            return Action(ActionKind.REROLL_THREE, {})
        case 2:
            # Take Pair
            return Action(ActionKind.TAKE_PAIR, {'take_idx': offset})

        case 3:
            # Take Mixed
            params = {
                'take_idx_land': offset//4,
                'take_idx_animal': offset%4,
            }
            return Action(ActionKind.TAKE_MIXED, params)
        
        case 4:
            # Place Land
            i,j,k = np.unravel_index(offset, (max_size, max_size, 6))
            params = {
                'index_place': (i - max_size // 2, j - max_size // 2, k),
            }
            return Action(ActionKind.PLACE_LAND, params)

        case 5:
            # Place Animal
            if offset==max_size**2:
                params = {
                    'index_place': None
                }
            else:
                i, j = np.unravel_index(offset, (max_size, max_size))
                params = {
                    'index_place': (i - max_size // 2, j - max_size // 2),
                }

            return Action(ActionKind.PLACE_ANIMAL, params)

        case _:
            raise ValueError("Action Index out of bounds of allowed actions.")


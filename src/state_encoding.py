import numpy as np
from base_types import BuiltPlate, GamePhase
import torch
from torch.nn import functional as F

from base_types import GameState, Action, ActionKind
from config import CONFIG
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

def encode_individual_land_state(game_state: GameState, player: int):
    land_state = game_state.players[player].plate_grid
    max_size = CONFIG.MAX_GRID_SIZE

    # Land-type-by-side, supported animals, built animal
    out_channels = 6*5 + 5 + 5
    out = np.zeros((max_size, max_size, out_channels))

    for (ii,jj), plate in land_state.items():
        i, j = ii+max_size//2, jj+max_size//2

        right_landscape_inds = (np.array([0, 1, 2]) + plate.orientation) % 6
        left_landscape_inds = (np.array([3, 4, 5]) + plate.orientation) % 6

        landscape_encoding = np.zeros(6, dtype=int)
        landscape_encoding[right_landscape_inds] = plate.right_landscape.value
        landscape_encoding[left_landscape_inds] = plate.left_landscape.value

        landscape_onehot = F.one_hot(torch.tensor(landscape_encoding), num_classes=5).numpy()

        supported_animals = np.array([a.value for a in plate.animals])

        out[i, j, :30] = landscape_onehot
        out[i, j, supported_animals+30] = 1
        if plate.built_animal is not None:
            out[i, j, 35 + plate.built_animal.value] = 1

    return out


def compute_grid_state(game_state: GameState, player_idx: int):
    n_players = len(game_state.players)
    individual_land_states = []
    for i in [(i+player_idx)%n_players for i in range(n_players)]:
        individual_land_states.append(encode_individual_land_state(game_state, i))

    # channels n_players * 40
    grid_state = np.concatenate(individual_land_states, axis=-1)
    return grid_state


def encode_individual_single_state(game_state: GameState, player: int):
    player_state = game_state.players[player]
    state = np.zeros((10,))
    pine_cone_idx = min(player_state.pine_cones, 9)
    state[pine_cone_idx] = 1

    return state

def compute_single_state(game_state: GameState, player_idx: int):
    n_players = len(game_state.players)

    individual_single_states = []
    for i in [(i+player_idx)%n_players for i in range(n_players)]:
        individual_single_states.append(encode_individual_single_state(game_state, i))

    group_single_state = np.concatenate(individual_single_states, axis=-1)

    phase_state = F.one_hot(torch.tensor(game_state.game_phase.value), num_classes=3).numpy()

    # shape n_players*10 + 3
    return np.concatenate((group_single_state, phase_state), axis=-1)


def compute_state(game_state: GameState, player_idx: int):
    grid_state = compute_grid_state(game_state, player_idx)
    single_state = compute_single_state(game_state, player_idx)

    single_state = np.broadcast_to(single_state[None, None, :], grid_state.shape[:2] + (single_state.shape[-1],))

    return np.concatenate((grid_state, single_state), axis=-1)

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
def compute_action_mask(game_state: GameState, player_idx: int):
    player_state = game_state.players[player_idx]
    land_state = player_state.plate_grid
    phase = game_state.game_phase
    max_size = CONFIG.MAX_GRID_SIZE

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
            for i, j in place_land_plate_mask(game_state, player_idx)['index_place']:
                i, j = i + max_size // 2, j + max_size // 2
                lower = n_actions_prefix[4] + (i*max_size + j) * 6
                mask[lower: lower+6] = True

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

    return mask


# Action Space:
# 1: Reroll All
# 1: Reroll Three
# 4: Take Pair
# 16: Take Mixed
# max_size**2 * 6: place land
# max_size**2: place animal
def action_from_index(game_state: GameState, player_idx: int, action_idx: int):
    player_state = game_state.players[player_idx]
    land_state = player_state.plate_grid
    phase = game_state.game_phase

    max_size = CONFIG.MAX_GRID_SIZE
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


        

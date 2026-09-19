from collections import Counter
from config import CONFIG

from base_types import Animal, BasePlate, BuiltPlate, GameState, Action, ActionKind

def reroll_all(state: GameState, action: Action, active_player_idx: int) -> GameState:
    player_state = state.players[active_player_idx]
    if player_state.pine_cones < 1:
        raise ValueError("Not enough pine cones to reroll all.")
    player_state.pine_cones -= 1
    new_animal_plates = [state.bag.sample_animal_plate() for _ in range(4)]
    for animal in state.pool_state.animal_pool:
        state.bag.animal_plates[animal] += 1

    state.pool_state.animal_pool = new_animal_plates

def reroll_all_mask(state: GameState, active_player_idx: int) -> bool:
    return state.players[active_player_idx].pine_cones > 0

def reroll_three(state: GameState, action: Action, active_player_idx: int) -> GameState:
    animal_plates = state.pool_state.animal_pool
    counter = Counter(animal_plates)
    player_state = state.players[active_player_idx]
    with_three = [a for a in Animal if counter[a] == 3]
    if len(with_three) != 1 or player_state.has_rerolled or player_state.pine_cones < 1:
        raise ValueError('No three identical animals in pool.')
    to_reroll = with_three[0]
    for i, a in enumerate(animal_plates):
        if a == to_reroll:
            animal_plates[i] = state.bag.sample_animal_plate()
    state.bag.animal_plates[to_reroll] += 3
    player_state.pine_cones -= 1
    player_state.has_rerolled = True

def reroll_three_mask(state: GameState, active_player_idx: int) -> bool:
    animal_plates = state.pool_state.animal_pool
    counter = Counter(animal_plates)
    with_three = [a for a in Animal if counter[a] == 3]
    return len(with_three) == 1 and not state.players[active_player_idx].has_rerolled and state.players[active_player_idx].pine_cones > 0

def take_pair(state: GameState, action: Action, active_player_idx: int) -> GameState:
    take_idx = action.params['take_idx']
    if not (0 <= take_idx < 4):
        raise ValueError(f"take_idx {take_idx} out of bounds.")

    base_plate = state.pool_state.plate_pool[take_idx]
    animal = state.pool_state.animal_pool[take_idx]

    state.pool_state.plate_pool[take_idx] = state.bag.sample_base_plate()
    state.pool_state.animal_pool[take_idx] = state.bag.sample_animal_plate()

    state.players[active_player_idx].to_place = (animal, base_plate)

def take_pair_mask(state: GameState, active_player_idx: int) -> dict:
    return {
        'take_idx': [0, 1, 2, 3]
    }

def take_mixed(state: GameState, action: Action, active_player_idx: int):
    take_idx_animal = action.params['take_idx_animal']
    take_idx_land = action.params['take_idx_land']
    player_state = state.players[active_player_idx]
    if not (0 <= take_idx_animal < 4) or not (0 <= take_idx_land < 4) or player_state.pine_cones<1:
        raise ValueError(f"Take inds out of bounds: {take_idx_animal} | {take_idx_land}")

    base_plate = state.pool_state.plate_pool[take_idx_land]
    animal = state.pool_state.animal_pool[take_idx_animal]

    state.pool_state.plate_pool[take_idx_land] = state.bag.sample_base_plate()
    state.pool_state.animal_pool[take_idx_animal] = state.bag.sample_animal_plate()

    player_state.to_place = (animal, base_plate)
    player_state.pine_cones -= 1

def take_mixed_mask(state: GameState, active_player_idx: int) -> bool:
    player_state = state.players[active_player_idx]
    if player_state.pine_cones < 1:
        return False
    else:
        return {
            'take_idx_land': [0, 1, 2, 3],
            'take_idx_animal': [0, 1, 2, 3],
        }

def place_land_plate_mask(state: GameState, active_player_idx: int):
    player_state = state.players[active_player_idx]
    land_state = player_state.plate_grid
    neighbors = set()
    neighboring = [
        (0, 1),
        (1, 1),
        (1, 0),
        (0, -1),
        (-1, -1),
        (-1, 0)
    ]
    for (i,j) in land_state.keys():
        for (off_i, off_j) in neighboring:
            ci = i + off_i
            cj = j + off_j
            if ci < -CONFIG.MAX_GRID_SIZE//2 or ci > CONFIG.MAX_GRID_SIZE//2:
                continue

            if cj < -CONFIG.MAX_GRID_SIZE//2 or cj > CONFIG.MAX_GRID_SIZE//2:
                continue

            neighbors.add((i+off_i, j+off_j))

    for (i,j) in land_state.keys():
        neighbors.discard((i, j))

    return {
        'index_place': neighbors
    }

def place_land_plate(state: GameState, action: Action, active_player_idx: int):
    player_state = state.players[active_player_idx]
    land_state = player_state.plate_grid
    place_idx = action.params['index_place']

    if place_idx[:2] not in place_land_plate_mask(state, active_player_idx)['index_place'] or not 0 <= place_idx[2] < 6:
        raise ValueError("Invalid location selected for land placement.")

    plate = player_state.to_place[1]
    built_plate = BuiltPlate(animals=plate.animals, left_landscape=plate.left_landscape, right_landscape=plate.right_landscape, orientation=place_idx[2])
    land_state[place_idx[:2]] = built_plate

def place_animal_mask(state: GameState, active_player_idx: int):
    player_state = state.players[active_player_idx]
    land_state = player_state.plate_grid
    animal = player_state.to_place[0]
    placeable = [p for p, plate in land_state.items() if plate.built_animal is None and animal in plate.animals]

    return {
        'index_place': placeable + [None]
    }

def place_animal(state: GameState, action: Action, active_player_idx: int):
    player_state = state.players[active_player_idx]
    land_state = player_state.plate_grid
    animal = player_state.to_place[0]
    place_idx = action.params['index_place']
    if place_idx not in place_animal_mask(state, active_player_idx)['index_place']:
        raise ValueError("Invalid location selected for animal placement.")

    if place_idx is not None:
        land_state[place_idx].built_animal = animal
    else:
        state.bag.animal_plates[animal] += 1


def action_transition(state: GameState, action: Action, active_player_idx: int): 
    match action.kind:
        case ActionKind.REROLL_ALL:
            reroll_all(state, action, active_player_idx)
        case ActionKind.REROLL_THREE:
            reroll_three(state, action, active_player_idx)
        case ActionKind.TAKE_PAIR:
            take_pair(state, action, active_player_idx)
            state.game_phase = state.game_phase.PLACING_LAND
        case ActionKind.TAKE_MIXED:
            take_mixed(state, action, active_player_idx)
            state.game_phase = state.game_phase.PLACING_LAND
        case ActionKind.PLACE_LAND:
            place_land_plate(state, action, active_player_idx)
            state.game_phase = state.game_phase.PLACING_ANIMAL
        case ActionKind.PLACE_ANIMAL:
            place_animal(state, action, active_player_idx)
            state.players[active_player_idx].to_place = None
            state.players[active_player_idx].has_rerolled = False
            state.active_player = (active_player_idx + 1) % len(state.players)
            state.game_phase = state.game_phase.PICKING

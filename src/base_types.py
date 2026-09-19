from dataclasses import dataclass, field
import json
import numpy as np
import random
from pathlib import Path
from enum import Enum



class Animal(Enum):
    GRIZZLY=0
    DEER=1
    SALMON=2
    HAWK=3
    FOX=4

class Landscape(Enum):
    MOUNTAIN=0
    FOREST=1
    PLAINS=2
    WETLAND=3
    RIVER=4


animal_str_to_enum = {
    "GRIZZLY": Animal.GRIZZLY,
    "DEER": Animal.DEER,
    "SALMON": Animal.SALMON,
    "HAWK": Animal.HAWK,
    "FOX": Animal.FOX,
}

landscape_str_to_enum = {
    "MOUNTAIN": Landscape.MOUNTAIN,
    "FOREST": Landscape.FOREST,
    "PLAINS": Landscape.PLAINS,
    "WETLAND": Landscape.WETLAND,
    "RIVER": Landscape.RIVER,
}




@dataclass
class BasePlate:
    animals: list[Animal]
    left_landscape: Landscape
    right_landscape: Landscape


@dataclass
class BuiltPlate(BasePlate):
    orientation: int # [0, 1, 2, 3, 4, 5] * 60 deg
    built_animal: Animal = None


@dataclass
class PlayerState:
    plate_grid: dict[tuple[int, int], BuiltPlate] = field(default_factory=dict)
    to_place: tuple[Animal, BasePlate] = None
    has_rerolled: bool = False
    pine_cones: int = 0

@dataclass 
class PoolState:
    plate_pool: list[BasePlate] = field(default_factory=list)
    animal_pool: list[Animal] = field(default_factory=list)

def parse_base_plate(data: dict) -> BasePlate:
    animals = [animal_str_to_enum[animal] for animal in data['animals']]
    left_landscape = landscape_str_to_enum[data['left_landscape']]
    right_landscape = landscape_str_to_enum[data['right_landscape']]
    return BasePlate(animals=animals, left_landscape=left_landscape, right_landscape=right_landscape)

def load_base_plate_config() -> dict:
    with (Path(__file__).parent / 'data' / 'base_plates.json').open('r') as f:
        data = json.load(f)

    base_plates = []
    for plate_data in data['base_plates']:
        base_plate = parse_base_plate(plate_data)
        base_plates.append(base_plate)

    return base_plates

@dataclass
class PlateBag:
    base_plates: list[BasePlate] = field(default_factory=load_base_plate_config)
    animal_plates : dict[Animal, int] = field(default_factory=lambda: {animal: 20 for animal in Animal})

    def sample_base_plate(self) -> BasePlate:
        if not self.base_plates:
            raise ValueError("No more base plates available in the bag.")

        plate_idx = random.randint(0, len(self.base_plates) - 1)
        return self.base_plates.pop(plate_idx)

    def sample_animal_plate(self) -> Animal:
        if not any(v > 0 for v in self.animal_plates.values()):
            raise ValueError("No more animal plates available in the bag.")

        counts = np.array([self.animal_plates[a] for a in Animal])
        p = counts / np.sum(counts)
        animal = np.random.choice(list(Animal), p=p)
        self.animal_plates[animal] -= 1
        return animal

class GamePhase(Enum):
    PICKING=0
    PLACING_LAND=1
    PLACING_ANIMAL=2


class GameState:
    players: list[PlayerState]
    bag: PlateBag
    pool_state: PoolState
    game_phase: GamePhase
    active_player: int

    def __init__(self, n_players: int):
        self.players: list[PlayerState] = [PlayerState() for _ in range(n_players)]
        self.bag: PlateBag = PlateBag()

    def init_pool_state(self):
        plate_pool = [self.bag.sample_base_plate() for _ in range(4)]
        animal_pool = [self.bag.sample_animal_plate() for _ in range(4)]
        self.pool_state = PoolState(plate_pool=plate_pool, animal_pool=animal_pool)
        

    def init_player_states(self):
        with (Path(__file__).parent / 'data' / 'starting_tiles.json').open('r') as f:
            starting_tiles_data = json.load(f)['starting_tiles']

        for i in range(len(self.players)):
            idx = random.randrange(len(starting_tiles_data))
            starting_tile = starting_tiles_data.pop(idx)
            plates = [parse_base_plate(plate_data) for plate_data in starting_tile['base_plates']]
            self.players[i].plate_grid = {
                (0, 0): BuiltPlate(plates[0].animals, plates[0].left_landscape, plates[0].right_landscape, orientation=0),
                (0, 1): BuiltPlate(plates[1].animals, plates[1].left_landscape, plates[1].right_landscape, orientation=2),
                (1, 1): BuiltPlate(plates[2].animals, plates[2].left_landscape, plates[2].right_landscape, orientation=1),
            }

    def init_game(self):
        self.init_pool_state()
        self.init_player_states()
        self.game_phase = GamePhase.PICKING
        self.active_player = 0


class ActionKind(Enum):
    REROLL_ALL = 0
    REROLL_THREE = 1
    TAKE_PAIR = 2
    TAKE_MIXED = 3
    PLACE_LAND = 4
    PLACE_ANIMAL = 5

class Action:
    def __init__(self, kind: ActionKind, params: dict):
        self.kind = kind
        self.params = params

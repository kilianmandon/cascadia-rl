from dataclasses import dataclass
from enum import Enum

from base_types import Animal

class ScoringGroup(Enum):
    A = 0
    B = 1
    C = 2
    D = 3


@dataclass
class Config:
    MAX_GRID_SIZE = 30
    scoring_groups = {
        Animal.GRIZZLY: ScoringGroup.A,
        Animal.SALMON: ScoringGroup.A,
        Animal.FOX: ScoringGroup.A,
        Animal.HAWK: ScoringGroup.A,
        Animal.DEER: ScoringGroup.A,
    }
    score_land = True
    score_animals = {
        Animal.GRIZZLY: True,
        Animal.SALMON: True,
        Animal.FOX: True,
        Animal.HAWK: True,
        Animal.DEER: True,
    }
    allow_rotating_land = True
    score_land_bonus = False
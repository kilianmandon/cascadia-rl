from abc import ABC, abstractmethod
from functools import lru_cache
from collections import Counter

import numpy as np

from base_types import Animal, BuiltPlate, GameState, Landscape
from config import CONFIG, ScoringGroup

neighbor_dirs = [
    (1, 1), # Bottom right
    (0, 1), # Right
    (-1, 0), # Top Right
    (-1, -1), # Top Left
    (0, -1), # Left
    (1, 0), # Bottom Left
]


def form_rerooting(form):
    rerooted_forms = set()

    new_basis = [
        ((1, 0), (0, 1)), # right points right
        ((1, 1), (-1, 0)), # right points top-right
        ((0, 1), (-1, -1)), # right points top-left
        ((-1, 0), (0, -1)), # right points left
        ((-1, -1), (1, 0)), # right points down-left
        ((0, -1), (1, 1)), # right points down-right
        ]

    rotated_forms = []

    for n1, n2 in new_basis:
        new_form = [
            (i*n1[0]+j*n2[0], i*n1[1]+j*n2[1]) for i,j in form
        ]
        rotated_forms.append(new_form)

    for base_form in rotated_forms:
        for di, dj in base_form:
            new_form = tuple([(i-di, j-dj) for i,j in form])
            rerooted_forms.add(new_form)

    return rerooted_forms


@lru_cache
def caching_form_fit(nodes, forms, scores):
    score_callbacks = [lambda _, i=i: i for i in scores]
    return bruteforce_form_fit_from_callback(nodes, forms, score_callbacks)

def bruteforce_form_fit(nodes, forms, scores):
    def to_tuple(ls):
        return tuple(to_tuple(i) if isinstance(i, list) else i for i in ls)

    return caching_form_fit(to_tuple(sorted(nodes)), to_tuple(forms), to_tuple(scores))


def bruteforce_form_fit_from_callback(nodes, forms, score_callbacks):
    @lru_cache
    def solve(remaining_nodes: frozenset):
        if not remaining_nodes:
            return 0

        root_i, root_j = next(iter(remaining_nodes))

        best_score = 0
        for form, score_callback in zip(forms, score_callbacks):
            for rerooted_form in form_rerooting(form):
                to_test = [(root_i+di, root_j+dj) for di, dj in rerooted_form]
                if all(t in remaining_nodes for t in to_test):
                    score = score_callback(to_test)
                    new_remaining = remaining_nodes - set(to_test)
                    best_score = max(best_score, score + solve(new_remaining))
        return best_score


    return solve(frozenset(nodes))


def flood(start_nodes: list, edges: list):
    to_flood = start_nodes
    groups = []
    while to_flood:
        root = to_flood.pop(0)
        new_group = [root]
        to_expand = [root]
        while to_expand:
            expanding = to_expand.pop(0)
            for (n1, n2) in edges:
                if n1 != expanding:
                    continue
                if n2 not in new_group:
                    new_group.append(n2)
                    to_expand.append(n2)
                    to_flood.remove(n2)
        groups.append(new_group)

    return groups

def flood_fill_animals(land_state: dict[tuple[int, int], BuiltPlate], animal: Animal):
    start_nodes = [p for p, plate in land_state.items() if animal in plate.animals]
    edges = []

    for (i, j) in start_nodes:
        for (di, dj) in neighbor_dirs:
            ii, jj = i+di, j+dj
            if (ii, jj) in start_nodes:
                edges.append(((i, j), (ii, jj)))

    return flood(start_nodes, edges)

def flood_fill_land(land_state: dict[tuple[int, int], BuiltPlate], land: Landscape):
    start_nodes = [p for p, plate in land_state.items() if plate.left_landscape == land or plate.right_landscape == land]

    neighbor_dirs_by_orientation = [
        neighbor_dirs[i:] + neighbor_dirs[:i] for i in range(6)
    ]

    # only outgoing correctness checked
    possible_edges = []

    for (i, j) in start_nodes:
        plate = land_state[(i,j)]
        local_neighbor_dirs = neighbor_dirs_by_orientation[plate.orientation]
        if plate.right_landscape == land:
            for (di, dj) in local_neighbor_dirs[:3]:
                ii, jj = (i+di, j+dj)
                if (ii, jj) in start_nodes:
                    possible_edges.append( ((i, j), (ii, jj)) )

    edges = []
    for (p1, p2) in possible_edges:
        if (p2, p1) in possible_edges:
            edges.append((p1, p2))

    return flood(start_nodes, edges)


def has_triangle(group):
    forms = [
        [(0, 0), (0, 1), (1, 1)]
    ]
    return bruteforce_form_fit(group, forms, [1]) > 0

def inbetween_iterator(p1, p2):
    p1 = np.array(p1)
    p2 = np.array(p2)
    t = np.max(np.abs(p1-p2))
    dp = (p2-p1) / t

    for i in range(1, t):
        p = p1 + i*dp
        yield (p[0], p[1])

def hawk_find_lines_of_sight(land_state: dict[tuple[int, int], BuiltPlate]):
    hawks = [p for p, plate in land_state.items() if plate.built_animal == Animal.HAWK]

    lines_of_sight = []
    for n, (i, j) in enumerate(hawks):
        for (ii, jj) in hawks[n+1:]:
            horizontal = (i-ii) == 0
            diag1 = (j-jj) == 0
            diag2 = (i-ii) == (j-jj)
            if horizontal or diag1 or diag2:
                inbetween = [land_state[p].built_animal if p in land_state else None for p in inbetween_iterator((i,j), (ii,jj))]
                if not Animal.HAWK in inbetween:
                    lines_of_sight.append( ((i,j), (ii,jj)) )

    return lines_of_sight

    


### Group A
def score_grizzly_a(game_state: GameState, active_player_idx: int):
    player_state = game_state.players[active_player_idx]
    land_state = player_state.plate_grid

    groups = flood_fill_animals(land_state, Animal.GRIZZLY)
    couples = [g for g in groups if len(groups) == 2]
    match len(couples):
        case 0:
            return 0
        case 1:
            return 4
        case 2:
            return 11
        case 3:
            return 19
        case _:
            return 27

def score_salmon_a(game_state: GameState, active_player_idx: int):
    player_state = game_state.players[active_player_idx]
    land_state = player_state.plate_grid
    groups = flood_fill_animals(land_state, Animal.SALMON)
    group_sizes = [len(g) for g in groups if not has_triangle(g)]

    def score_for_size(n):
        top_score = (n//7) * 25
        base_scores = [0, 2, 5, 8, 12, 16, 20]
        remaining_score = base_scores[n%7]
        return top_score + remaining_score

    score = sum(score_for_size(n) for n in group_sizes)
    return score

def score_fox_a(game_state: GameState, active_player_idx: int):
    player_state = game_state.players[active_player_idx]
    land_state = player_state.plate_grid
    foxes = [n for n, plate in land_state.items() if plate.built_animal == Animal.FOX]

    total_score = 0
    for i,j in foxes:
        neighbor_animals = set()
        for di, dj in neighbor_dirs:
            neighbor = land_state.get((i+di, j+dj), None)
            if neighbor is not None and neighbor.built_animal is not None:
                neighbor_animals.add(neighbor.built_animal)

        total_score += len(neighbor_animals)

    return total_score

def score_hawk_a(game_state: GameState, active_player_idx: int):
    player_state = game_state.players[active_player_idx]
    land_state = player_state.plate_grid
    groups = flood_fill_animals(land_state, Animal.HAWK)
    single_count = len([g for g in groups if len(g) == 1])
    val_per_single = [0, 2, 5, 8, 11, 14, 18, 22, 26]

    single_count = min(single_count, len(val_per_single)-1)

    return val_per_single[single_count]



def score_deer_a(game_state: GameState, active_player_idx: int):
    player_state = game_state.players[active_player_idx]
    land_state = player_state.plate_grid
    groups = flood_fill_animals(land_state, Animal.DEER)

    scores = [2, 5, 9, 13]
    lines = [
        [(0, i) for i in range(length)]
        for length in range(1, 5)
    ]

    total_score = 0
    for group in groups:
        total_score += bruteforce_form_fit(group, lines, scores)

    return total_score



###

### Group B
def score_grizzly_b(game_state: GameState, active_player_idx: int):
    player_state = game_state.players[active_player_idx]
    land_state = player_state.plate_grid

    groups = flood_fill_animals(land_state, Animal.GRIZZLY)
    triples = [g for g in groups if len(groups) == 3]

    return len(triples) * 10

def score_salmon_b(game_state: GameState, active_player_idx: int):
    player_state = game_state.players[active_player_idx]
    land_state = player_state.plate_grid
    groups = flood_fill_animals(land_state, Animal.SALMON)
    group_sizes = [len(g) for g in groups if not has_triangle(g)]

    def score_for_size(n):
        top_score = (n//5) * 17
        base_scores = [0, 2, 4, 9, 11]
        remaining_score = base_scores[n%5]
        return top_score + remaining_score

    score = sum(score_for_size(n) for n in group_sizes)
    return score

def score_fox_b(game_state: GameState, active_player_idx: int):
    player_state = game_state.players[active_player_idx]
    land_state = player_state.plate_grid
    foxes = [n for n, plate in land_state.items() if plate.built_animal == Animal.FOX]
    score_by_pair = [0, 3, 5, 7]

    total_score = 0
    for i,j in foxes:
        neighbor_animals = []
        for di, dj in neighbor_dirs:
            neighbor = land_state.get((i+di, j+dj), None)
            if neighbor is not None and neighbor.built_animal is not None:
                neighbor_animals.append(neighbor.built_animal)

        counter = Counter(neighbor_animals)
        pairs = len([a for a in Animal if counter[a] >= 2])
        total_score += score_by_pair[pairs]

    return total_score


def score_hawk_b(game_state: GameState, active_player_idx: int):
    player_state = game_state.players[active_player_idx]
    land_state = player_state.plate_grid
    groups = flood_fill_animals(land_state, Animal.HAWK)

    single_hawks = [g[0] for g in groups if len(g) == 1]
    all_hawks = [p for p, plate in land_state.items() if plate.built_animal == Animal.HAWK]

    valid_hawks = 0
    for i, j in single_hawks:
        for ii, jj in all_hawks:
            horizontal = (i-ii) == 0
            diag1 = (j-jj) == 0
            diag2 = (i-ii) == (j-jj)
            if horizontal or diag1 or diag2:
                valid_hawks += 1
                break

    valid_hawk_scoring = [0, 5, 9, 12, 16, 20, 24, 28]
    valid_hawks = min(valid_hawks, len(valid_hawk_scoring)-1)

    return valid_hawk_scoring[valid_hawks]





def score_deer_b(game_state: GameState, active_player_idx: int):
    player_state = game_state.players[active_player_idx]
    land_state = player_state.plate_grid
    groups = flood_fill_animals(land_state, Animal.DEER)

    scores = [2, 5, 9, 13]
    patterns = [
        [(0, 0)],
        [(0, 0), (0, 1)],
        [(0, 0), (1, 0), (1, 1)],
        [(0, 0), (1, 0), (1, 1), (2, 1)],
    ]

    total_score = 0
    for group in groups:
        total_score += bruteforce_form_fit(group, patterns, scores)

    return total_score

###

### Group C
def score_grizzly_c(game_state: GameState, active_player_idx: int):
    player_state = game_state.players[active_player_idx]
    land_state = player_state.plate_grid

    groups = flood_fill_animals(land_state, Animal.GRIZZLY)
    group_sizes = Counter([len(group) for group in groups])

    points = group_sizes[1] * 2 + group_sizes[2] * 5 + group_sizes[3] * 8
    if group_sizes[1]>0 and group_sizes[2]>0 and group_sizes[3]>0:
        points += 3

    return points

def score_salmon_c(game_state: GameState, active_player_idx: int):
    player_state = game_state.players[active_player_idx]
    land_state = player_state.plate_grid
    groups = flood_fill_animals(land_state, Animal.SALMON)
    group_sizes = [len(g) for g in groups if not has_triangle(g)]

    def score_for_size(n):
        top_score = (n//3) * 10
        base_scores = [0, 2, 5] # Extension of last 3-chain
        remaining_score = base_scores[n%3] if top_score>0 else 0 # Only add extension if there is a chain to be extended
        return top_score + remaining_score

    score = sum(score_for_size(n) for n in group_sizes)
    return score

def score_fox_c(game_state: GameState, active_player_idx: int):
    player_state = game_state.players[active_player_idx]
    land_state = player_state.plate_grid
    foxes = [n for n, plate in land_state.items() if plate.built_animal == Animal.FOX]
    score_by_count = [0, 1, 2, 3, 4, 5, 6]

    total_score = 0
    for i,j in foxes:
        neighbor_animals = []
        for di, dj in neighbor_dirs:
            neighbor = land_state.get((i+di, j+dj), None)
            if neighbor is not None and neighbor.built_animal not in [None, Animal.FOX]:
                neighbor_animals.append(neighbor.built_animal)

        counter = Counter(neighbor_animals)
        max_count = max(counter[a] for a in Animal)
        total_score += score_by_count[max_count]

    return total_score


def score_hawk_c(game_state: GameState, active_player_idx: int):
    player_state = game_state.players[active_player_idx]
    land_state = player_state.plate_grid
    sight_lines = hawk_find_lines_of_sight(land_state)

    return 3 * len(sight_lines)


def score_deer_c(game_state: GameState, active_player_idx: int):
    player_state = game_state.players[active_player_idx]
    land_state = player_state.plate_grid
    groups = flood_fill_animals(land_state, Animal.DEER)

    score_lookup = [0, 2, 4, 7, 10, 14, 18, 23, 28]
    group_sizes = [min(len(group), len(score_lookup)-1) for group in groups]

    score = sum(score_lookup[n] for n in group_sizes)
    return score


###

### Group D
def score_grizzly_d(game_state: GameState, active_player_idx: int):
    player_state = game_state.players[active_player_idx]
    land_state = player_state.plate_grid

    groups = flood_fill_animals(land_state, Animal.GRIZZLY)
    group_sizes = Counter([len(group) for group in groups])

    points = group_sizes[2] * 5 + group_sizes[3] * 8 + group_sizes[4] * 13

    return points

def score_salmon_d(game_state: GameState, active_player_idx: int):
    player_state = game_state.players[active_player_idx]
    land_state = player_state.plate_grid
    groups = flood_fill_animals(land_state, Animal.SALMON)
    groups = [g for g in groups if not has_triangle(g) and len(g)>=3]

    def group_score(group):
        neighboring = set()
        for di, dj in neighbor_dirs:
            for i, j in group:
                ii, jj = i+di, j+dj
                if (ii, jj) not in group:
                    neighboring.add((ii, jj))

        return len(group) + len([1 for n in neighboring if n in land_state and land_state[n].built_animal is not None])

    return sum(group_score(g) for g in groups)

def score_fox_d(game_state: GameState, active_player_idx: int):
    player_state = game_state.players[active_player_idx]
    land_state = player_state.plate_grid
    groups = flood_fill_animals(land_state, Animal.SALMON)
    groups = [g for g in groups if len(g) >= 2]

    def score_callback(foxes):
        neighbor_cells = set()
        for i, j in foxes:
            for di, dj in neighbor_dirs:
                ii, jj = i+di, j+dj
                neighbor_cells.add((ii, jj))

        neighbor_animals = []
        for p in neighbor_cells:
            animal = land_state[p].built_animal if p in land_state else None
            if animal not in [None, Animal.FOX]:
                neighbor_animals.append(animal)

        counter = Counter(neighbor_animals)
        pair_count = len([a for a in Animal if counter[a] >= 2])
        pair_value = [0, 5, 7, 9, 11]
        return pair_value[pair_count]

    forms = [[(0, 0), (1, 0)]]
    score_callbacks = [score_callback]

    total_score = 0
    for group in groups:
        total_score += bruteforce_form_fit_from_callback(group, forms, score_callbacks)

    return total_score
        

def score_hawk_d(game_state: GameState, active_player_idx: int):
    player_state = game_state.players[active_player_idx]
    land_state = player_state.plate_grid
    sight_lines = hawk_find_lines_of_sight(land_state)

    sight_lines = frozenset(sight_lines)

    @lru_cache
    def solve(remaining_lines):
        best_score = 0
        for p1, p2 in remaining_lines:
            inbetween =  set([land_state[p].built_animal for p in inbetween_iterator(p1, p2) if p in land_state])
            if None in inbetween:
                inbetween.remove(None)

            scoring = [0, 4, 7, 9]
            inbetween_count = min(len(inbetween), len(scoring)-1)

            new_rem_lines = remaining_lines - [(pp1, pp2) for (pp1, pp2) in remaining_lines if pp1 not in [p1, p2] and pp2 not in [p1, p2]]
            score = scoring[inbetween_count] + solve(new_rem_lines)
            best_score = max(score, best_score)
        return best_score

    return solve(sight_lines)



def score_deer_d(game_state: GameState, active_player_idx: int):
    player_state = game_state.players[active_player_idx]
    land_state = player_state.plate_grid
    groups = flood_fill_animals(land_state, Animal.DEER)

    scores = [2, 5, 8, 12, 16, 21]
    circles = [
        [(0, 0)],
        [(0, 0), (0, 1)],
        [(0, 0), (0, 1), (1, 2)],
        [(0, 0), (0, 1), (1, 2), (2, 2)],
        [(0, 0), (0, 1), (1, 2), (2, 2), (2, 1)],
        [(0, 0), (0, 1), (1, 2), (2, 2), (2, 1), (1, 0)],
    ]

    total_score = 0
    for group in groups:
        total_score += bruteforce_form_fit(group, circles, scores)

    return total_score
###

def get_scoring_method_for(animal: Animal):
    scoring_group = CONFIG.scoring_groups[animal]
    scoring_bear = {
        ScoringGroup.A: score_grizzly_a,
        ScoringGroup.B: score_grizzly_b,
        ScoringGroup.C: score_grizzly_c,
        ScoringGroup.D: score_grizzly_d,
    }

    scoring_salmon = {
        ScoringGroup.A: score_salmon_a,
        ScoringGroup.B: score_salmon_b,
        ScoringGroup.C: score_salmon_c,
        ScoringGroup.D: score_salmon_d,
    }

    scoring_fox = {
        ScoringGroup.A: score_fox_a,
        ScoringGroup.B: score_fox_b,
        ScoringGroup.C: score_fox_c,
        ScoringGroup.D: score_fox_d,
    }

    scoring_hawk = {
        ScoringGroup.A: score_hawk_a,
        ScoringGroup.B: score_hawk_b,
        ScoringGroup.C: score_hawk_c,
        ScoringGroup.D: score_hawk_d,
    }

    scoring_deer = {
        ScoringGroup.A: score_deer_a,
        ScoringGroup.B: score_deer_b,
        ScoringGroup.C: score_deer_c,
        ScoringGroup.D: score_deer_d,
    }

    match animal:
        case Animal.GRIZZLY:
            return scoring_bear[scoring_group]
        case Animal.SALMON:
            return scoring_salmon[scoring_group]
        case Animal.FOX:
            return scoring_fox[scoring_group]
        case Animal.HAWK:
            return scoring_hawk[scoring_group]
        case Animal.DEER:
            return scoring_deer[scoring_group]


def score_land(game_state: GameState, active_player_idx: int):
    player_state = game_state.players[active_player_idx]
    land_state = player_state.plate_grid

    res = dict()

    for land_type in Landscape:
        groups = flood_fill_land(land_state, land_type)
        res[land_type] = max((len(g) for g in groups), default=0)

    return res


def land_extra_points(game_state: GameState):
    n_players = len(game_state.players)
    all_land_scores = {
        land: [] for land in Landscape
    }
    for player_idx in range(n_players):
        land_scores = score_land(game_state, player_idx)
        for land, score in land_scores.items():
            all_land_scores[land].append(score)

    final_player_scores = [
        {
            land: 0 for land in Landscape
        }
        for _ in range(n_players)
    ]


    for land, scores in all_land_scores.items():
        scores = np.asarray(scores)
        values, counts = np.unique(scores, return_counts=True)

        top_scorer = np.nonzero(scores==values[-1])[0]
        top_scorer_bonus = 3 if counts[-1]==1 else 2

        second_top = []
        if values.size > 1:
            second_top = np.nonzero(scores==values[-2])[0]
            second_top_bonus = 1 if counts[-2]==1 else 0

        for i in top_scorer:
            final_player_scores[i][land] = top_scorer_bonus

        for i in second_top:
            final_player_scores[i][land] = second_top_bonus

    return final_player_scores

def full_player_score(game_state: GameState, player_idx: int):
    animal_scores = {}
    total_score = 0

    for animal in Animal:
        animal_score = get_scoring_method_for(animal)(game_state, player_idx)
        animal_scores[animal] = animal_score

    land_scores = score_land(game_state, player_idx)
    land_extra = land_extra_points(game_state)[player_idx]

    total_score += sum(animal_scores.values())
    total_score += sum(land_scores.values())
    total_score += sum(land_extra.values())
    total_score += game_state.players[player_idx].pine_cones

    score_summary = {
        'animal_scores': animal_scores,
        'land_scores': land_scores,
        'land_extra': land_extra,
        'pine_cones': game_state.players[player_idx].pine_cones,
    }

    return total_score, score_summary


        

        
        
        

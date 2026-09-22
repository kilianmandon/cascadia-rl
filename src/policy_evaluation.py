import copy

from base_types import GamePhase, GameState
from config import Config
from game import action_transition
from scoring import full_player_score
from single_player_env import SinglePlayerEnv
import numpy as np

from state_encoding import action_from_index, compute_action_mask

def random_policy(obs, info, game_state, config):
    mask = info['action_mask']
    return np.random.choice(np.nonzero(mask)[0])

def greedy_placement(obs, info, game_state: GameState, config):
    if game_state.game_phase == GamePhase.PICKING:
        return random_policy(obs, info, game_state, config)
    else:
        return greedy_policy(obs, info, game_state, config, depth=2)


def greedy_policy(obs, info, game_state: GameState, config, depth=3):
    def action_and_value(game_state: GameState, depth):
        player_idx = game_state.active_player
        action_mask = compute_action_mask(game_state, player_idx, config)
        opt_action = None
        opt_value = -1
        for action_idx in np.nonzero(action_mask)[0]:
            copy_gamestate = copy.deepcopy(game_state)
            action = action_from_index(copy_gamestate, copy_gamestate.active_player, action_idx, config)
            action_transition(copy_gamestate, action, copy_gamestate.active_player, config)

            if depth>1:
                _, opt_sub_value = action_and_value(copy_gamestate, depth=depth-1)
            else:
                opt_sub_value, _ = full_player_score(copy_gamestate, copy_gamestate.active_player, config)
            if opt_sub_value > opt_value:
                opt_value = opt_sub_value
                opt_action = action_idx

        return opt_value, opt_action

    _, action_idx = action_and_value(game_state, depth)

    return action_idx







def test_policy(policy):
    config = Config()
    config.score_land = False
    config.allow_rotating_land = False

    env = SinglePlayerEnv(config)

    num_steps = 100
    num_tests = 100
    rewards = []

    for _  in range(num_tests):
        observation, info = env.reset()
        game_state = env.game_state
        for i in range(num_steps):
            action = policy(observation, info, game_state, config)
            observation, reward, terminated, truncated, info = env.step(action)

            if 'final_info' in info:
                rew = info['final_info']['episode']['r']
                print(rew)
                rewards.append(rew)

            if terminated:
                break

    print(np.mean(rewards))



def main():
    test_policy(greedy_placement)

if __name__=='__main__':
    main()
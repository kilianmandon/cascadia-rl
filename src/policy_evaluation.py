import copy
from gymnasium.vector import AsyncVectorEnv
import tqdm
import torch

from base_types import GamePhase, GameState
from config import Config
from game import action_transition
from ppo import Agent
from scoring import full_player_score
from single_player_env import SinglePlayerEnv
import numpy as np

from state_encoding import action_from_index, compute_action_mask, unflatten_transformer_state

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


def create_transformer_policy():
    config = Config()
    agent =  Agent(config)
    agent.load_state_dict(torch.load('checkpoints/Cascadia__ppo__1__1790182147/final.pt')['model_state_dict'])
    device = 'cuda'
    agent.to(device)

    def policy(obs, info, game_state: GameState, config):
        obs = torch.tensor(obs, device=device, dtype=torch.float32)
        action_mask = torch.tensor(info['action_mask'], device=device)
        action_idx, _, _, _ = agent.get_action_and_value(obs, action_mask=action_mask)
        action_idx = action_idx.cpu().numpy()
        # feats = unflatten_transformer_state(obs)
        # logits, value = agent.actor_critic(feats)
        # logits[~action_mask] = -1e8
        # action_idx = logits.argmax(dim=-1).cpu().numpy()

        return action_idx

    return policy




def test_policy(policy, async_env=False):
    config = Config()
    # config.score_land = False
    # config.allow_rotating_land = False

    num_steps = 100
    num_tests = 200
    num_parallel = 50
    returns = []

    if async_env:
        env = AsyncVectorEnv([lambda: SinglePlayerEnv(config) for _ in range(num_parallel)])
        num_iterations = num_tests // num_parallel
    else:
        env = SinglePlayerEnv(config)
        num_iterations = num_tests


    for _  in tqdm.tqdm(range(num_iterations)):
        observation, info = env.reset()
        game_state = env.game_state if hasattr(env, 'game_state') else None
        for i in range(num_steps):
            action = policy(observation, info, game_state, config)
            observation, reward, terminated, truncated, info = env.step(action)

            if 'final_info' in info:
                rewards = info['final_info']['episode']['r']
                terminated = info['final_info']['episode']['terminated']
                for r, w in zip(rewards, terminated):
                    if w: returns.append(r)


    print(np.mean(returns), np.std(returns))



def main():
    test_policy(create_transformer_policy(), async_env=True)
    # test_policy(greedy_placement)

if __name__=='__main__':
    main()
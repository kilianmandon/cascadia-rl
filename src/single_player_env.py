import gymnasium as gym
import numpy as np
from game import action_transition
from scoring import full_player_score
from state_encoding import action_from_index, build_transformer_state, compute_action_mask, compute_state, flatten_transformer_state
from base_types import Animal, GamePhase, GameState
from config import Config

# Action Space:
# 1: Reroll All
# 1: Reroll Three
# 4: Take Pair
# 16: Take Mixed
# max_size**2 * 6: place land
# max_size**2+1: place animal
class SinglePlayerEnv(gym.Env):
    def __init__(self, config: Config):
        grid_size = config.MAX_GRID_SIZE
        observation_grid_channels = 40
        observation_single_channels = 1*(10+21) + 80 + 3 + 20
        action_count = 1+1+4+16+6*grid_size**2 + (grid_size**2+1)

        self.stop_when_remaining = 61
        # self.observation_space = gym.spaces.MultiBinary([grid_size, grid_size, observation_grid_channels+observation_single_channels])

        n_players = 1
        c_global_state = n_players*(10+21) + 80 + 3 + 20
        c_grid_enc = 2 + 6*5 + 5 + 5
        n_token = 127
        c_total = c_global_state + n_token * (c_grid_enc + 2 + 1)
        self.observation_space = gym.spaces.Box(-1e5, 1e5, shape=(c_total,))
        self.action_space = gym.spaces.Discrete(action_count)
        self.config = config

    def _get_obs(self):
        # s = compute_state(self.game_state, self.game_state.active_player, self.config)
        s = flatten_transformer_state(build_transformer_state(self.game_state, self.config))
        return s

    def _get_info(self):
        return {
            'action_mask': compute_action_mask(self.game_state, self.game_state.active_player, self.config)
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.generator = np.random.default_rng(seed=seed)
        self.passed_steps = 0

        self.game_state = GameState(n_players=1, generator=self.generator)
        self.game_state.init_game()

        return self._get_obs(), self._get_info()

    def step(self, action_idx):
        player_idx = self.game_state.active_player
        old_score, _ = full_player_score(self.game_state, player_idx, self.config) 


        action_mask = compute_action_mask(self.game_state, player_idx, self.config)
        if not action_mask[action_idx]:
            raise ValueError("Player selected invalid action.")

        action = action_from_index(self.game_state, player_idx, action_idx, self.config)
        action_transition(self.game_state, action, player_idx, self.config)
        self.passed_steps += 1

        new_score, score_info = full_player_score(self.game_state, player_idx, self.config)

        reward = (new_score - old_score) / 10
        observation = self._get_obs()
        terminated = (len(self.game_state.bag.base_plates) <= self.stop_when_remaining) and self.game_state.game_phase==GamePhase.PICKING
        truncated = False
        info = self._get_info()
        info['score_info'] = score_info
        if terminated:
            info['final_info'] = {
                    'episode': {
                        'r': new_score,
                        'l': self.passed_steps,
                        'terminated': terminated,
                    }
                }

        return  observation, reward, terminated, truncated, info





import gymnasium as gym
from base_types import GameState
from config import CONFIG

class SinglePlayerEnv(gym.Env):
    def __init__(self):
        grid_size = CONFIG.MAX_GRID_SIZE
        observation_grid_channels = 16
        observation_single_channels = 16
        action_count = 3000

        self.observation_space = gym.spaces.Dict(
             {
                'grid': gym.spaces.MultiBinary([grid_size, grid_size, observation_grid_channels]),
                'single': gym.spaces.MultiBinary([observation_single_channels]),
             }
        )

        self.action_space = gym.spaces.Discrete(action_count)

    def _get_obs(self):
        # State encoding is still being developed; keeping this explicit makes
        # the environment importable while avoiding a silently invalid obs.
        raise NotImplementedError("Observation encoding has not been implemented yet.")

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.game_state = GameState(n_players=0)
        self.game_state.init_game()




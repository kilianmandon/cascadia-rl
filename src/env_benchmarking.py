# import os
# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["MKL_NUM_THREADS"] = "1"
# os.environ["OPENBLAS_NUM_THREADS"] = "1"

import torch
import time
import gymnasium as gym

from single_player_env import SinglePlayerEnv
import cProfile
import pstats


def main():
    num_envs = 48
    envs = gym.vector.AsyncVectorEnv(
        [lambda: SinglePlayerEnv() for i in range(num_envs)],
        shared_memory=True
    )
    # envs = SinglePlayerEnv()
    _, infos = envs.reset()

    t0 = time.time()
    for step in range(100):
        action_mask = torch.tensor(infos['action_mask'])
        action_scores = torch.rand(action_mask.shape)
        action_scores[~action_mask] = -1e8

        action_inds = action_scores.argmax(dim=-1).numpy()
        _, _, _, _, infos = envs.step(action_inds)

    t = time.time()
    dt = t-t0
    print(f'Took: {dt}')
    print(f'SPS: {100*num_envs / dt}')

if __name__=='__main__':
    cProfile.run('main()', 'output.prof')

    stats = pstats.Stats('output.prof')
    stats.sort_stats('cumulative')  # or 'tottime' for time spent in the function itself
    stats.print_stats(20)  # top 20

    # main()
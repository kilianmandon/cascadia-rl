# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

from config import CONFIG
from single_player_env import SinglePlayerEnv


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = False
    """if toggled, cuda will be enabled by default"""
    track: bool = True
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "Cascadia"
    """the id of the environment"""
    total_timesteps: int = 50_000_000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 48
    """the number of parallel game environments"""
    num_steps: int = 100
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 16
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.025
    """coefficient of the entropy"""
    use_normalized_entropy: bool = True
    """Toggles whether entropy is normalized with the action mask."""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


def save_checkpoint(agent, optimizer, iteration, global_step, path, args, upload_to_wandb=False):
    torch.save({
        "model_state_dict": agent.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iteration": iteration,
        "global_step": global_step,
        "args": vars(args),
    }, path)

    if upload_to_wandb:
        import wandb
        artifact = wandb.Artifact(
            name=f'model-{wandb.run.id}',
            type='model',
            metadata={'iteration': iteration, 'global_step': global_step},
        )
        artifact.add_file(path)
        wandb.log_artifact(artifact)


def make_env(env_id, idx, capture_video, run_name):
    def thunk():
        env = SinglePlayerEnv()
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk

class SpatialActorCriticModel(nn.Module):
    def __init__(self, n_channels, n_actions):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(n_channels, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding='same'),
            nn.ReLU(),
        )

        self.land_place_actor_head = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(128, 6, kernel_size=1)
        )

        self.animal_place_actor_head = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(128, 1, kernel_size=1)
        )

        self.one_dim_action_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(start_dim=-3),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 23)
        )

        self.value_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(start_dim=-3),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        x =  torch.einsum('...ijc->...cij', x)
        x = self.backbone(x)
        land_place_logits = self.land_place_actor_head(x)
        animal_place_logits = self.animal_place_actor_head(x)

        land_place_logits = torch.einsum('...cij->...ijc', land_place_logits)
        animal_place_logits = torch.einsum('...cij->...ijc', animal_place_logits)

        one_dim_action = self.one_dim_action_head(x)

        action_logits = torch.cat([
            one_dim_action[..., :-1],
            torch.flatten(land_place_logits, start_dim=-3),
            torch.flatten(animal_place_logits, start_dim=-3),
            one_dim_action[..., -1:],
        ], dim=-1)

        value = self.value_head(x)

        return action_logits, value




class ActorCriticModel(nn.Module):
    def __init__(self, n_channels, n_actions):
        super().__init__()
        self.backbone = nn.Sequential(
            # nn.Conv2d(n_channels, 128, kernel_size=1),
            # nn.ReLU(),
            nn.Conv2d(n_channels, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

        self.actor_head = nn.Sequential(
            nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, n_actions),
        )

        self.critic_head = nn.Sequential(
            nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 1),
        )

    def forward(self, x):
        x = torch.einsum('...ijc->...cij', x)
        x = self.backbone(x)

        return self.actor_head(x), self.critic_head(x)



class Agent(nn.Module):
    def __init__(self, envs=None):
        super().__init__()
        grid_size = CONFIG.MAX_GRID_SIZE
        observation_grid_channels = 40
        observation_single_channels = 1*10 + 80 + 3
        action_count = 1+1+4+16+6*grid_size**2 + (grid_size**2+1)

        observation_channels = observation_grid_channels + observation_single_channels
        self.actor_critic = SpatialActorCriticModel(observation_channels, action_count)

    def get_value(self, x):
        _, value = self.actor_critic(x)
        return value

    def get_action_and_value(self, x, action=None, action_mask=None):
        logits, value = self.actor_critic(x)
        if action_mask is not None:
            logits[~action_mask] -= 1e8
        probs = Categorical(logits=logits)
        entropy = probs.entropy()
        if action is None:
            action = probs.sample()
        # action = action_mask.int().argmax(dim=-1)
        return action, probs.log_prob(action), probs.entropy(), value


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )

    ckpt_dir = f"checkpoints/{run_name}"
    os.makedirs(ckpt_dir, exist_ok=True)

    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    if torch.cuda.is_available():
        print('Using cuda.')
        device = torch.device('cuda')
    elif torch.mps.is_available():
        print('Using MPS.')
        device = torch.device('mps')
    else:
        print("Using CPU.")
        device = torch.device('cpu')

    # env setup
    envs = gym.vector.AsyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name) for i in range(args.num_envs)],
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    action_masks = torch.zeros((args.num_steps, args.num_envs, envs.single_action_space.n), dtype=bool).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, next_info = envs.reset(seed=args.seed)
    next_action_mask = next_info['action_mask']
    next_obs = torch.Tensor(next_obs).to(device)
    next_action_mask = torch.tensor(next_action_mask, dtype=bool).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    for iteration in range(1, args.num_iterations + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            frac = max(frac, 0.2)
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow


        print('Gathering experience.')
        t0 = time.time()
        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            action_masks[step] = next_action_mask
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs, action_mask=next_action_mask)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.Tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)
            next_action_mask = infos['action_mask']
            next_action_mask = torch.tensor(next_action_mask).to(device)

            if "final_info" in infos:
                info = infos['final_info']
                if info and "episode" in info:
                    all_episodic_returns = []
                    all_episodic_lengths = []
                    
                    for i, (r, l, terminated) in enumerate(zip(info['episode']['r'], info['episode']['l'], info['episode']['terminated'])):
                        if terminated:
                            # print(f"global_step={global_step}, episodic_return={r} terminated={info['episode']['terminated'][i]}")
                            all_episodic_returns.append(r)
                            all_episodic_lengths.append(l)
                    if len(all_episodic_returns) > 0:
                        writer.add_scalar("charts/episodic_return", np.mean(np.array(all_episodic_returns)), global_step)
                        writer.add_scalar("charts/episodic_length", np.mean(np.array(all_episodic_lengths)), global_step)

        t1 = time.time()
        print(f'Env SPS: {100*args.num_envs / (t1-t0)}')
        print('Bootstrapping value')
        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_action_mask = action_masks.reshape((-1,) + (envs.single_action_space.n,))
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        print('Starting update cycle...')
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds], action_mask=b_action_mask[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                unmasked_action_count = b_action_mask[mb_inds].float().sum(dim=-1)
                low_action_entropy_mean = entropy[unmasked_action_count<10].mean()
                high_action_entropy_mean = entropy[unmasked_action_count>=10].mean()
                normalized_entropy = (entropy / torch.log(unmasked_action_count.clamp(min=2))).mean()

                if not args.use_normalized_entropy:
                    entropy_loss = entropy.mean()
                else:
                    entropy_loss = normalized_entropy

                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef
                # loss = -args.ent_coef * entropy_loss + v_loss * args.vf_coef
                # loss = - args.ent_coef * entropy_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        print('Done')

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar("losses/low_action_count_entropy", low_action_entropy_mean, global_step)
        writer.add_scalar("losses/high_action_count_entropy", high_action_entropy_mean, global_step)
        writer.add_scalar("losses/normalized_entropy", normalized_entropy, global_step)

        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

        checkpoint_every = max(1, args.num_iterations // 10)
        if iteration%checkpoint_every==0:
            ckpt_path = f'{ckpt_dir}/iter_{iteration}.pt'
            save_checkpoint(agent, optimizer, iteration, global_step, ckpt_path, args, upload_to_wandb=args.track)


    envs.close()
    final_path = f'{ckpt_dir}/final.pt'
    save_checkpoint(agent, optimizer, iteration, global_step, final_path, args, upload_to_wandb=args.track)
    writer.close()
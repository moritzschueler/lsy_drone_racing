"""A naive RL pipeline for drone racing."""

import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal

import fire
import gymnasium as gym
import jax
import jax.numpy as jp
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from functools import partial
from crazyflow.envs.drone_env import DroneEnv
from crazyflow.envs.norm_actions_wrapper import NormalizeActions
from crazyflow.sim.data import SimData
from crazyflow.sim.physics import Physics
from crazyflow.sim.visualize import draw_line, draw_points
from crazyflow.utils import leaf_replace
from gymnasium import spaces
from gymnasium.spaces import flatten_space
from gymnasium.vector import VectorEnv, VectorObservationWrapper, VectorRewardWrapper
from gymnasium.vector.utils import batch_space
from gymnasium.wrappers.vector.jax_to_torch import JaxToTorch
from jax import Array
from jax.scipy.spatial.transform import Rotation as R
from ml_collections import ConfigDict
from scipy.interpolate import CubicSpline
from torch import Tensor
from torch.distributions.normal import Normal

from lsy_drone_racing.envs.race_core import build_dynamics_disturbance_fn, rng_spec2fn
from lsy_drone_racing.utils import load_config
from lsy_drone_racing.envs.drone_race import VecDroneRaceEnv
from lsy_drone_racing.control.wrappers import (GateObsWrapper, 
                                               CrashPenaltyWrapper,
                                               GatePassageRewardWrapper,
                                               ProximityRewardWrapper,
                                               SpeedRewardWrapper,
                                               GateInViewRewardWrapper,
                                               AltitudeRewardWrapper,
                                               SurviveRewardWrapper,
                                               RPYPenaltyWrapper,
                                               OOBPenaltyWrapper,
                                               VerticalSpeedPenaltyWrapper,
                                               SoftCollisionWrapper, 
                                               RandomGateSpawnWrapper, 
                                               ActionPenalty, 
                                               FlattenJaxObservation, 
                                               NormalizeRaceActions, 
                                               RaceStackObs)


# region Arguments
@dataclass
class Args:
    """Class to store configurations."""

    seed: int = 42
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    jax_device: str = "gpu"  
    """environment device"""
    wandb_project_name: str = "ADR-PPO-Racing"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""

    # Algorithm specific arguments
    total_timesteps: int = 200_000_000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    num_envs: int = 2048
    """the number of parallel game environments"""
    num_steps: int = 32
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.98
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 16
    """the number of mini-batches"""
    update_epochs: int = 10
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.3
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.005
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 1.0
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

    # Wrapper settings
    n_obs: int = 2
    rpy_coef: float = 0.06
    d_act_th_coef: float = 0.4
    d_act_xy_coef: float = 1.0
    act_coef: float = 0.077
    """reward coefficients for training"""

    @staticmethod
    def create(**kwargs: Any) -> "Args":
        """Create arguments class."""
        args = Args(**kwargs)
        args.batch_size = int(args.num_envs * args.num_steps)
        args.minibatch_size = int(args.batch_size // args.num_minibatches)
        args.num_iterations = args.total_timesteps // args.batch_size
        return args


def set_seeds(seed: int):
    """Seed everything."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# region MakeEnvs
def make_race_envs(
    config: str = "level2.toml",
    num_envs: int = 1024,
    jax_device: str = "cpu",
    torch_device=None,
    coefs: dict | None = None,
) -> VectorEnv:
    """Create VecDroneRaceEnv with training wrappers.

    Wrapper stack:
        VecDroneRaceEnv → NormalizeRaceActions → RaceRewardAndObs
        → RaceStackObs → ActionPenalty → FlattenJaxObservation → JaxToTorch
    """

    if torch_device is None:
        torch_device = torch.device("cpu")
    if coefs is None:
        coefs = {}

    cfg = load_config(Path(__file__).parents[2] / "config" / config)
    control_mode = cfg.env.get("control_mode", "attitude")
    n_gates = len(cfg.env.track.gates)

    print(f"[make_race_envs] config={config}, num_envs={num_envs}, "
          f"control_mode={control_mode}, n_gates={n_gates}, device={jax_device}")

    env = VecDroneRaceEnv(
        num_envs=num_envs,
        freq=cfg.env.freq,
        sim_config=cfg.sim,
        track=cfg.env.track,
        sensor_range=cfg.env.sensor_range,
        control_mode=control_mode,
        disturbances=cfg.env.get("disturbances", None),
        randomizations=cfg.env.get("randomizations", None),
        max_episode_steps=coefs.get("max_episode_steps", 1500),
        device=jax_device,
    )
    print(env.observation_space.shape, env.action_space.shape)

    env = NormalizeRaceActions(env)

    # env = RandomGateSpawnWrapper(
    #     env,
    #     random_gate_ratio=coefs.get("random_gate_ratio", 1.0),
    #     spawn_offset=coefs.get("spawn_offset", 0.5),
    #     spawn_pos_noise=coefs.get("spawn_pos_noise", 0.05),
    #     spawn_vel_noise=coefs.get("spawn_vel_noise", 0.05),
    # )

    # env = CrashPenaltyWrapper(env)
 
    env = GatePassageRewardWrapper(env, gate_bonus=coefs.get("gate_bonus", 15.0))
 
    env = ProximityRewardWrapper(
        env,
        n_gates=n_gates,
        proximity_coef=coefs.get("proximity_coef", 0.0),
        progress_coef=coefs.get("progress_coef", 1.0),
        bilateral_progress=coefs.get("bilateral_progress", False),
    )
 
    # env = SpeedRewardWrapper(env, n_gates=n_gates, speed_coef=coefs.get("speed_coef", 0.5))
 
    # if coefs.get("gate_in_view_coef", 0.5) > 0.0:
    #     env = GateInViewRewardWrapper(
    #         env, n_gates=n_gates, gate_in_view_coef=coefs.get("gate_in_view_coef", 0.5)
    #     )
 
    # if coefs.get("alt_coef", 0.5) > 0.0:
    #     env = AltitudeRewardWrapper(env, n_gates=n_gates, alt_coef=coefs.get("alt_coef", 0.5))
 
    
    # env = SurviveRewardWrapper(env, survive_coef=0.1)
 
    # env = RPYPenaltyWrapper(env, rpy_coef=coefs.get("rpy_coef", 0.06))
 
    # if coefs.get("oob_coef", 0.5) > 0.0:
    #     env = OOBPenaltyWrapper(
    #         env,
    #         oob_coef=coefs.get("oob_coef", 0.5),
    #         z_low=coefs.get("z_low", 0.0),
    #         z_high=coefs.get("z_high", 2.0),
    #         terminate_on_oob=True,
    #     )
 
    # if coefs.get("vz_coef", 0.5) > 0.0:
    #     env = VerticalSpeedPenaltyWrapper(
    #         env,
    #         vz_coef=coefs.get("vz_coef", 0.5),
    #         vz_threshold=coefs.get("vz_threshold", 0.5),
    #     )


    # env = SoftCollisionWrapper(
    #     env,
    #     soft_collision_penalty=coefs.get("soft_collision_penalty", 2.0),
    #     soft_collision_steps=coefs.get("soft_collision_steps", 5_000_000),
    # )

    env = GateObsWrapper(env, body_frame_obs = True)

    

    # env = RaceStackObs(env, n_obs=coefs.get("n_obs", 2))
    # env = ActionPenalty(
    #     env,
    #     act_coef=coefs.get("act_coef", 0.002),
    #     d_act_th_coef=coefs.get("d_act_th_coef", 0.05),
    #     d_act_xy_coef=coefs.get("d_act_xy_coef", 0.05),
    # )

    


    env = FlattenJaxObservation(env)
    env = JaxToTorch(env, torch_device)


    print(f"[make_race_envs] obs_space={env.single_observation_space}, "
          f"act_space={env.single_action_space}")
    return env


def layer_init(layer: nn.Module, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Module:
    """Initialize layer."""
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer



# region Agent
class Agent(nn.Module):
    """RL Agent."""

    def __init__(self, obs_shape: tuple, action_shape: tuple, hidden_size: int = 64):
        """Init network structures."""
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(torch.tensor(obs_shape).prod(), hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(torch.tensor(obs_shape).prod(), hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, torch.tensor(action_shape).prod()), std=0.01),
            nn.Tanh(),
        )
        self.actor_logstd = nn.Parameter(torch.Tensor([[-0.5, -0.5, -0.5, -0.5]])
        )

    def get_value(self, x: Tensor) -> Tensor:
        """Value estimation."""
        return self.critic(x)

    def get_action_and_value(
        self, x: Tensor, action: Tensor | None = None, deterministic: bool = False
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Action output."""
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        # During learning the agent explores the environment by sampling actions from a Normal
        # distribution. The standard deviation is a learnable parameter that should decrease during
        # training as the agent gets more confident in its actions.
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample() if not deterministic else action_mean
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)


# region Train
def train_ppo(
    args: Args, model_path: Path, device: torch.device, jax_device: str, wandb_enabled: bool = False
) -> None:
    """Train.

    An implementation of PPO from cleanrl, see https://docs.cleanrl.dev/.
    """
    # train setup
    if wandb_enabled and wandb.run is None:
        wandb.init(project=args.wandb_project_name, entity=args.wandb_entity, config=vars(args))
    train_start_time = time.time()
    set_seeds(args.seed)  # TRY NOT TO MODIFY: seeding
    print("Training on device:", device, "| Environment device:", jax_device)

    # env setup
    r_coefs = {
        "n_obs": args.n_obs,
        "rpy_coef": args.rpy_coef,
        "d_act_xy_coef": args.d_act_xy_coef,
        "d_act_th_coef": args.d_act_th_coef,
        "act_coef": args.act_coef,

    }
    envs = make_race_envs(
        num_envs=args.num_envs, jax_device=jax_device, torch_device=device, coefs=r_coefs
    )

    assert isinstance(envs.single_action_space, gym.spaces.Box), (
        "only continuous action space is supported"
    )

    agent = Agent(envs.single_observation_space.shape, envs.single_action_space.shape).to(device)
    optimizer = optim.AdamW(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(
        device
    )
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(
        device
    )
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    sum_rewards = torch.zeros((args.num_envs)).to(device)
    sum_rewards_hist = []

    try:
        for iteration in range(1, args.num_iterations + 1):
            start_time = time.time()

            # Annealing the rate if instructed to do so.
            if args.anneal_lr:
                frac = 1.0 - (iteration - 1.0) / args.num_iterations
                lrnow = frac * args.learning_rate
                optimizer.param_groups[0]["lr"] = lrnow

            for step in range(0, args.num_steps):
                global_step += args.num_envs
                obs[step] = next_obs
                dones[step] = next_done

                # ALGO LOGIC: action logic
                with torch.no_grad():
                    action, logprob, _, value = agent.get_action_and_value(next_obs)
                    values[step] = value.flatten()
                actions[step] = action
                logprobs[step] = logprob

                # TRY NOT TO MODIFY: execute the game and log data.
                next_obs, reward, terminations, truncations, infos = envs.step(action)
                print(f"\n=== DEBUG iter={iteration} step={step} ===")
                print(f"reward shape:      {reward.shape}")
                print(f"reward stats:      min={reward.min():.4f} max={reward.max():.4f} mean={reward.mean():.4f} nonzero={( reward != 0).sum().item()}")
                print(f"terminations:      shape={terminations.shape} sum={terminations.sum().item()}")
                print(f"truncations:       shape={truncations.shape} sum={truncations.sum().item()}")
                print(f"sum_rewards stats: min={sum_rewards.min():.4f} max={sum_rewards.max():.4f} mean={sum_rewards.mean():.4f}")
                # envs.render()
                rewards[step] = reward
                sum_rewards += reward
                sum_rewards[next_done.bool()] = 0
                next_done = terminations | truncations

                if wandb_enabled and next_done.any():
                    for r in sum_rewards[next_done.bool()]:
                        wandb.log({"train/reward": r.item()}, step=global_step)
                        sum_rewards_hist.append(r.item())

            # bootstrap value if not done
            with torch.no_grad():
                next_value = agent.get_value(next_obs).reshape(1, -1)
                advantages = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done.float()
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = (
                        delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                    )
                returns = advantages + values

            # flatten the batch
            b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
            b_logprobs = logprobs.reshape(-1)
            b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)

            # Optimizing the policy and value network
            b_inds = np.arange(args.batch_size)
            clipfracs = []
            for epoch in range(args.update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, args.batch_size, args.minibatch_size):
                    end = start + args.minibatch_size
                    mb_inds = b_inds[start:end]

                    _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                        b_obs[mb_inds], b_actions[mb_inds]
                    )
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                    mb_advantages = b_advantages[mb_inds]
                    if args.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                            mb_advantages.std() + 1e-8
                        )

                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(
                        ratio, 1 - args.clip_coef, 1 + args.clip_coef
                    )
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    newvalue = newvalue.view(-1)
                    if args.clip_vloss:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds], -args.clip_coef, args.clip_coef
                        )
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                    entropy_loss = entropy.mean()
                    loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                    optimizer.step()

                if args.target_kl is not None and approx_kl > args.target_kl:
                    break

            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

            # TRY NOT TO MODIFY: record rewards for plotting purposes
            if wandb_enabled:
                wandb.log(
                    {
                        "charts/learning_rate": optimizer.param_groups[0]["lr"],
                        "losses/value_loss": v_loss.item(),
                        "losses/policy_loss": pg_loss.item(),
                        "losses/entropy": entropy_loss.item(),
                        "losses/old_approx_kl": old_approx_kl.item(),
                        "losses/approx_kl": approx_kl.item(),
                        "losses/clipfrac": np.mean(clipfracs),
                        "losses/explained_variance": explained_var,
                        "charts/SPS": int(global_step / (time.time() - start_time)),
                    },
                    step=global_step,
                )
            end_time = time.time()
            print(f"Iter {iteration}/{args.num_iterations} took {end_time - start_time:.2f} seconds")
    except KeyboardInterrupt:
        print("\n\n[WARNING] Training interrupted by user (Ctrl+C)!")
        if model_path is not None:
            torch.save(agent.state_dict(), model_path)
            print(f"  [SAFETY] Model emergency saved to: {model_path}")
        print("Closing environment...")
        envs.close()
    train_end_time = time.time()
    print(f"Training for {global_step} steps took {train_end_time - train_start_time:.2f} seconds.")
    if model_path is not None:
        torch.save(agent.state_dict(), model_path)
        print(f"model saved to {model_path}")
    envs.close()

    return sum_rewards_hist


# region Evaluate
def evaluate_ppo(args: Args, n_eval: int, model_path: Path) -> tuple[float, float]:
    """Evaluate."""
    set_seeds(args.seed)
    device = torch.device("cpu")
    r_coefs = {
        "n_obs": args.n_obs,
        "rpy_coef": args.rpy_coef,
        "d_act_xy_coef": args.d_act_xy_coef,
        "d_act_th_coef": args.d_act_th_coef,
        "act_coef": args.act_coef,
    }
    eval_env = make_race_envs(num_envs=1, coefs=r_coefs)
    agent = Agent(eval_env.single_observation_space.shape, eval_env.single_action_space.shape).to(
        device
    )
    agent.load_state_dict(torch.load(model_path))
    with torch.no_grad():
        episode_rewards = []
        episode_lengths = []
        ep_seed = args.seed
        # Evaluate the policy
        for episode in range(n_eval):
            obs, _ = eval_env.reset(seed=(ep_seed := ep_seed + 1))
            done = torch.zeros(1, dtype=bool, device=device)
            episode_reward = 0
            steps = 0
            while not done.any():
                act, _, _, _ = agent.get_action_and_value(obs, deterministic=True)
                obs, reward, terminated, truncated, info = eval_env.step(act)
                eval_env.render()
                done = terminated | truncated
                episode_reward += reward[0].item()
                steps += 1
            episode_rewards.append(episode_reward)
            episode_lengths.append(steps)
            print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}, Length = {steps}")

        print(
            f"Average Reward = {np.mean(episode_rewards):.2f}, Length = {np.mean(episode_lengths)}"
        )
        eval_env.close()

        return episode_rewards, episode_lengths


# region Main
def main(wandb_enabled: bool = True, train: bool = True, eval: int = 1):
    """Main."""
    args = Args.create()
    model_path = Path(__file__).parent / "ppo_drone_racing.ckpt"
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    jax_device = args.jax_device

    if train:  # use "--train False" to skip training
        train_ppo(args, model_path, device, jax_device, wandb_enabled)

    if eval > 0:  # use "--eval <N>" to perform N evaluation episodes
        episode_rewards, episode_lengths = evaluate_ppo(args, eval, model_path)
        if wandb_enabled and train:
            wandb.log(
                {
                    "eval/mean_rewards": np.mean(episode_rewards),
                    "eval/mean_steps": np.mean(episode_lengths),
                }
            )
            wandb.finish()


if __name__ == "__main__":
    fire.Fire(main)

"""Standalone JAX/Flax PPO training for the Crazyflow hover & figure-8 tasks (with envs.py).

Run:
    uv run python train.py --task hover
    uv run python train.py --task figure8 --device gpu [--timesteps N] [--seed S]
"""

from __future__ import annotations

import os

os.environ.setdefault("SCIPY_ARRAY_API", "1")

import fire
import functools
import pickle
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable

import flax
import flax.struct as struct
import jax
import jax.numpy as jp
import matplotlib.pyplot as plt
import numpy as np
import optax
from envs import (
    ActionPenalty,
    AngleReward,
    FigureEightEnv,
    FlattenJaxObservation,
    NormalizeActions,
    NormalizeActionsV2,
    QuatToMatrixObs,
    ReachPosEnv,
    ZeroYaw,
    OpponentWrapper
)
from flax import linen as nn
from flax.linen.initializers import orthogonal, zeros
from flax.training import train_state
from jax import Array

from lsy_drone_racing.envs.multi_drone_race import VecMultiDroneRaceEnv
from lsy_drone_racing.utils.utils import load_config




# region PPO agent


class ActorNet(nn.Module):
    """Actor network: MLP returning a tanh-squashed mean and a learnable log-std."""

    hidden_size: int = 64
    act_dim: int = 4

    @nn.compact
    def __call__(self, obs: Array) -> tuple[Array, Array]:
        """Return ``(mean, logstd)`` of the Gaussian policy."""
        x = obs
        x = nn.Dense(self.hidden_size, kernel_init=orthogonal(), bias_init=zeros)(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_size, kernel_init=orthogonal(), bias_init=zeros)(x)
        x = nn.relu(x)
        mean = nn.Dense(self.act_dim, kernel_init=orthogonal(0.01), bias_init=zeros)(x)
        mean = nn.tanh(mean)
        actor_logstd = self.param(
            "actor_logstd",
            lambda rng, shape: jp.zeros(shape, dtype=jp.float32),
            (1, self.act_dim),
        )
        logstd = jp.broadcast_to(actor_logstd, (mean.shape[0], self.act_dim))
        return mean, logstd


class CriticNet(nn.Module):
    """Critic network: MLP returning a scalar state value."""

    hidden_size: int = 64

    @nn.compact
    def __call__(self, obs: Array) -> Array:
        """Return the scalar value estimate for each observation."""
        x = obs
        x = nn.Dense(self.hidden_size, kernel_init=orthogonal(), bias_init=zeros)(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_size, kernel_init=orthogonal(), bias_init=zeros)(x)
        x = nn.relu(x)
        value = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=zeros)(x)
        return value.squeeze(-1)


class Agent(struct.PyTreeNode):
    """PPO agent class with actor and critic networks."""

    actor_states: train_state.TrainState = struct.field(pytree_node=True)
    critic_states: train_state.TrainState = struct.field(pytree_node=True)
    actor: ActorNet = struct.field(pytree_node=False)
    critic: CriticNet = struct.field(pytree_node=False)
    actor_tx: optax.GradientTransformation = struct.field(pytree_node=False)
    critic_tx: optax.GradientTransformation = struct.field(pytree_node=False)
    init_logstd: float = struct.field(pytree_node=False)
    mean_action: Callable = struct.field(pytree_node=False)
    sample_action: Callable = struct.field(pytree_node=False)
    get_action_logprob: Callable = struct.field(pytree_node=False)
    get_value: Callable = struct.field(pytree_node=False)

    @classmethod
    def create(
        cls,
        key: Array,
        obs_dim: int,
        act_dim: int,
        hidden_size: int = 64,
        actor_lr: float | optax.Schedule = 3e-4,
        critic_lr: float | optax.Schedule = 1e-3,
        init_logstd: float = -1.0,
        max_grad_norm: float = 0.0,
    ) -> "Agent":
        """Initialize the PPO agent. ``max_grad_norm > 0`` adds global grad-norm clipping (0 = off)."""
        actor = ActorNet(hidden_size=hidden_size, act_dim=act_dim)
        critic = CriticNet(hidden_size=hidden_size)
        k1, k2 = jax.random.split(key)
        dummy_obs = jp.zeros((1, obs_dim), dtype=jp.float32)
        actor_params = actor.init(k1, dummy_obs)
        actor_params["params"]["actor_logstd"] = jp.full(
            (1, act_dim), init_logstd, dtype=jp.float32
        )
        critic_params = critic.init(k2, dummy_obs)

        def _make_tx(lr: float | optax.Schedule) -> optax.GradientTransformation:
            adamw = optax.adamw(learning_rate=lr, eps=1e-5)
            if max_grad_norm and max_grad_norm > 0.0:
                return optax.chain(optax.clip_by_global_norm(max_grad_norm), adamw)
            return adamw

        actor_tx = _make_tx(actor_lr)
        critic_tx = _make_tx(critic_lr)
        actor_states = train_state.TrainState.create(
            apply_fn=actor.apply, params=actor_params, tx=actor_tx
        )
        critic_states = train_state.TrainState.create(
            apply_fn=critic.apply, params=critic_params, tx=critic_tx
        )

        def _sample_action(params: dict, obs: Array, key: Array) -> tuple[Array, Array]:
            """Get stochastic action sample for collecting rollout."""
            mean, logstd = actor.apply(params, obs)
            std = jp.exp(logstd)
            new_key, sub = jax.random.split(key)
            eps = jax.random.normal(sub, mean.shape, dtype=mean.dtype)
            action = mean + std * eps
            logp = -0.5 * (eps**2 + 2.0 * logstd + jp.log(2.0 * jp.pi))
            logp = jp.sum(logp, axis=-1)
            entropy = jp.sum(0.5 * (1.0 + jp.log(2.0 * jp.pi)) + logstd, axis=-1)
            return (action, logp, entropy), new_key

        def _get_action_logprob(
            params: dict, obs: Array, action: Array
        ) -> tuple[Array, Array]:
            """Get log probability and entropy of given action for training."""
            mean, logstd = actor.apply(params, obs)
            std = jp.exp(logstd)
            logp = -0.5 * (
                ((action - mean) / (std + 1e-8)) ** 2
                + 2.0 * logstd
                + jp.log(2.0 * jp.pi)
            )
            logp = jp.sum(logp, axis=-1)
            entropy = jp.sum(0.5 * (1.0 + jp.log(2.0 * jp.pi)) + logstd, axis=-1)
            return logp, entropy

        def _mean_action(params: dict, obs: Array) -> Array:
            """Get deterministic action (mean)."""
            mean, logstd = actor.apply(params, obs)
            return mean

        def _get_value(params: dict, obs: Array) -> Array:
            return jp.squeeze(critic.apply(params, obs))

        return cls(
            actor_states=actor_states,
            critic_states=critic_states,
            actor=actor,
            critic=critic,
            actor_tx=actor_tx,
            critic_tx=critic_tx,
            init_logstd=init_logstd,
            mean_action=jax.jit(_mean_action),
            sample_action=jax.jit(_sample_action),
            get_action_logprob=jax.jit(_get_action_logprob),
            get_value=jax.jit(_get_value),
        )


# region Configuration
@dataclass(frozen=True)
class Args:
    """PPO + environment configuration. Only the values differ between the two tasks."""

    task: str = "hover"
    seed: int = 0
    device: str = "cpu"

    # PPO hyperparameters
    total_timesteps: int = 2_000_000
    num_envs: int = 2048
    num_steps: int = 16
    num_minibatches: int = 16
    update_epochs: int = 10
    anneal_lr: bool = True
    actor_lr: float = 8e-4
    critic_lr: float = 5e-3
    gamma: float = 0.92
    gae_lambda: float = 0.94
    clip_coef: float = 0.7
    ent_coef: float = 0.007
    vf_coef: float = 0.7
    norm_adv: bool = False  # per-minibatch advantage normalisation (PPO stabiliser)
    max_grad_norm: float = 0.0  # global gradient-norm clip; 0 disables
    hidden_size: int = 24
    init_logstd: float = -1.0

    # Wrapper / reward-shaping settings
    rpy_coef: float = 0.1
    act_coefs: tuple = (0.1, 0.1, 0.0, 0.02)
    d_act_coefs: tuple = (0.9, 0.9, 0.0, 0.6)
    num_last_actions: int = 1
    n_samples: int = 11
    samples_dt: float = 0.06
    # sharpness of the sharp PRECISION term exp(-k*dist)
    hover_reward_scale: float = 6.0
    # penalty on body angular velocity (damps swaying); 0 -> off
    hover_angvel_weight: float = 0.0

    # Hover (ReachPosEnv) reset / goal randomization ranges
    pos_min: tuple = (-0.5, -0.5, 1.2)
    pos_max: tuple = (0.5, 0.5, 1.8)
    goal_pmin: tuple = (-0.0, -0.0, 1.5)
    goal_pmax: tuple = (0.0, 0.0, 1.5)
    vel_min: float = -0.0898
    vel_max: float = 0.0898
    ang_vel_min: tuple = (-0.55, -0.55, -0.55)
    ang_vel_max: tuple = (0.55, 0.55, 0.55)

    # Derived at runtime
    pos_start: int = 0  # flat-obs offset of the relative position (hover metric)
    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0

    def finalize(self, pos_start: int = 0) -> "Args":
        """Fill in derived batch sizes, iteration count, and the hover pos slice offset."""
        batch_size = int(self.num_envs * self.num_steps)
        return replace(
            self,
            pos_start=pos_start,
            batch_size=batch_size,
            minibatch_size=batch_size // self.num_minibatches,
            num_iterations=self.total_timesteps // batch_size,
        )


# Per-task hyperparameter overrides applied on top of the Args defaults
MULTI_ARGS = dict(
    task="multi",
    total_timesteps=20_000_000,
    num_envs=512,
    num_steps=16,
    num_minibatches=16,
    update_epochs=10,
    actor_lr=8e-4,
    critic_lr=5e-3,
    gamma=0.92,
    gae_lambda=0.94,
    clip_coef=0.7,
    ent_coef=0.007,
    vf_coef=0.7,
    norm_adv=True,
    max_grad_norm=1.0,
    hidden_size=24,
    init_logstd=-1.0,
    rpy_coef=0.1,
    act_coefs=(0.1, 0.1, 0.0, 0.02),
    d_act_coefs=(0.9, 0.9, 0.0, 0.6),
    num_last_actions=1,
    n_samples=11,
    samples_dt=0.06,
)


def make_multi_env(
        args = MULTI_ARGS,
        config: str = "multi_level2.toml",
        num_envs: int = 1024,
        jax_device: str = "cpu",
        coefs: dict | None = None,
        ) -> FlattenJaxObservation:
    """Build the figure-8 PPO environment (copied from the repository's make_jitted_envs)."""
    
    if coefs is None:
        coefs = {}

    cfg = load_config(Path(__file__).parents[2] / "config" / str(config))

    control_mode = cfg.env.kwargs[0]["control_mode"]
    env: VecMultiDroneRaceEnv = VecMultiDroneRaceEnv(
        num_envs=num_envs,
        freq=np.array([kwargs["freq"] for kwargs in cfg.env.kwargs], dtype=np.int64)[0],
        sim_config=cfg.sim,
        track=cfg.env.track,
        sensor_range=cfg.env.kwargs[0]["sensor_range"],
        control_mode=control_mode,
        disturbances=cfg.env.get("disturbances", None),
        randomizations=cfg.env.get("randomizations", None),
        max_episode_steps=coefs.get("max_episode_steps", 1500),
        device=jax_device,
    )
    waypoints = np.array(
            [
                [-1.3, 0.75, 0.01],
                [-1.0, 0.75, 0.4],
                [0.3, 0.35, 0.7],
                [1.3, -0.15, 0.9],
                [0.9, 0.7, 1.2],
                [-0.5, -0.05, 0.7],
                [-1.2, -0.1, 0.8],
                [-1.2, -0.1, 1.2],
                [-0.0, -0.7, 1.2],
                [0.5, -0.75, 1.2],
            ]
        )
    env = OpponentWrapper.create(
        env,
        waypoints,
    )
    env = NormalizeActions.create(env)
    env = ZeroYaw.create(env)
    
    env = ActionPenalty.create(
        env,
        num_actions=args.num_last_actions,
        init_last_actions=jp.array([[0.0, 0.0, 0.0, 0.0]]),
        act_coefs=args.act_coefs,
        d_act_coefs=args.d_act_coefs,
    )
    env = FlattenJaxObservation.create(env)
    return env


# region PPO


@flax.struct.dataclass
class RolloutData:
    """Per-step rollout buffers, each shaped ``(num_steps, num_envs, ...)``."""

    observations: Array
    actions: Array
    logprobs: Array
    rewards: Array
    dones: Array
    values: Array
    advantages: Array
    returns: Array


@functools.partial(jax.jit, static_argnames=("args",))
def collect_rollout(
    env: struct.PyTreeNode,
    args: Args,
    agent: Agent,
    next_obs: Array,
    next_done: Array,
    ep_return: Array,
    key: Array,
) -> tuple:
    """Collect a rollout of length ``args.num_steps``.

    Returns the updated env/obs/done/ep_return/key, the RolloutData, and three per-step metric
    stacks ``(ep_return_at_done, done_now, dist)``.
    """

    def step_once(carry: tuple, _: Any) -> tuple[tuple, tuple]:
        env, key, ep_return, obs, dones = carry
        (action, logprob, _), key = agent.sample_action(
            agent.actor_states.params, obs, key
        )
        value = agent.get_value(agent.critic_states.params, obs)
        env, (next_obs, reward, terminations, truncations, info) = env.step(env, action)
        done_now = terminations | truncations
        # Episodic-return accumulator (reset at episode end).
        ep_return_acc = ep_return + reward
        ret_at_done = jp.where(done_now, ep_return_acc, jp.nan)
        next_ep_return = jp.where(done_now, 0.0, ep_return_acc)
        # Task metric: distance to the goal / trajectory target.
        
        pos = env.unwrapped.data.states.pos[:, 0, :]
        dist = jp.linalg.norm(pos - env.unwrapped.target_pos, axis=-1)
        
        data = RolloutData(
            observations=obs,
            actions=action,
            logprobs=logprob,
            rewards=reward,
            dones=dones,
            values=value,
            advantages=jp.zeros_like(reward),
            returns=jp.zeros_like(reward),
        )
        return (env, key, next_ep_return, next_obs, done_now), (
            data,
            ret_at_done,
            done_now,
            dist,
        )

    (env, key, ep_return, next_obs, next_done), (data, ret_at_done, done_now, dist) = (
        jax.lax.scan(
            step_once, (env, key, ep_return, next_obs, next_done), length=args.num_steps
        )
    )
    return env, data, next_obs, next_done, ep_return, key, ret_at_done, done_now, dist


@functools.partial(jax.jit, static_argnames=("args",))
def compute_gae(
    args: Args, agent: Agent, next_obs: Array, next_done: Array, data: RolloutData
) -> RolloutData:
    """Compute GAE advantages and bootstrapped returns."""
    last_value = agent.get_value(agent.critic_states.params, next_obs)
    advantages = jp.zeros((args.num_envs,))
    dones = jp.concatenate([data.dones, next_done[None, :]], axis=0)
    values = jp.concatenate([data.values, last_value[None, :]], axis=0)

    def compute_gae_once(carry: Array, inp: tuple) -> tuple[Array, Array]:
        advantages = carry
        nextdone, nextvalues, curvalues, reward = inp
        nextnonterminal = 1.0 - nextdone
        delta = reward + args.gamma * nextvalues * nextnonterminal - curvalues
        advantages = delta + args.gamma * args.gae_lambda * nextnonterminal * advantages
        return advantages, advantages

    _, advantages = jax.lax.scan(
        compute_gae_once,
        advantages,
        (dones[1:], values[1:], values[:-1], data.rewards),
        reverse=True,
    )
    return data.replace(advantages=advantages, returns=advantages + data.values)


@functools.partial(jax.jit, static_argnames=("args",))
def update_policy(args: Args, agent: Agent, data: RolloutData, key: Array) -> tuple:
    """Perform the PPO updates. Returns ``((agent, key), (pg, v, entropy, kl, explained_var))``."""

    def loss_fn(
        actor_params: dict,
        critic_params: dict,
        obs_mb: Array,
        acts_mb: Array,
        old_logprobs_mb: Array,
        adv_mb: Array,
        ret_mb: Array,
        val_mb: Array,
    ) -> tuple:
        # ppo policy loss
        if args.norm_adv:  # per-minibatch advantage normalisation
            adv_mb = (adv_mb - jp.mean(adv_mb)) / (jp.std(adv_mb) + 1e-8)
        logp, entropy = agent.get_action_logprob(actor_params, obs_mb, acts_mb)
        ratio = jp.exp(logp - old_logprobs_mb)
        pg_loss1 = -adv_mb * ratio
        pg_loss2 = -adv_mb * jp.clip(ratio, 1.0 - args.clip_coef, 1.0 + args.clip_coef)
        pg_loss = jp.mean(jp.maximum(pg_loss1, pg_loss2))
        entropy_loss = -jp.mean(entropy)
        # ppo value loss (clipped)
        value = agent.get_value(critic_params, obs_mb).reshape(-1)
        v_unclipped = (value - ret_mb) ** 2
        v_clipped = val_mb + jp.clip(value - val_mb, -args.clip_coef, args.clip_coef)
        v_clipped_loss = (v_clipped - ret_mb) ** 2
        v_loss = 0.5 * jp.mean(jp.maximum(v_unclipped, v_clipped_loss))
        loss = pg_loss + args.ent_coef * entropy_loss + args.vf_coef * v_loss
        approx_kl = jp.mean((ratio - 1.0) - (logp - old_logprobs_mb))
        return loss, (pg_loss, v_loss, entropy_loss, approx_kl)

    grad_fn = jax.value_and_grad(loss_fn, argnums=(0, 1), has_aux=True)

    def update_epoch(carry: tuple, inp: int) -> tuple:
        agent, key = carry
        key, subkey = jax.random.split(key)

        def process_data(x: Array) -> Array:
            x = x.reshape((-1,) + x.shape[2:])
            x = jax.random.permutation(subkey, x)
            x = jp.reshape(x, (args.num_minibatches, -1) + x.shape[1:])
            return x

        minibatches = jax.tree_util.tree_map(process_data, data)

        def update_minibatch(carry: Agent, minibatch: RolloutData) -> tuple:
            agent = carry
            (loss, (pg_loss, v_loss, entropy_loss, approx_kl)), (g_actor, g_critic) = (
                grad_fn(
                    agent.actor_states.params,
                    agent.critic_states.params,
                    minibatch.observations,
                    minibatch.actions,
                    minibatch.logprobs,
                    minibatch.advantages,
                    minibatch.returns,
                    minibatch.values,
                )
            )
            return agent.replace(
                actor_states=agent.actor_states.apply_gradients(grads=g_actor),
                critic_states=agent.critic_states.apply_gradients(grads=g_critic),
            ), (pg_loss, v_loss, entropy_loss, approx_kl)

        def run_epoch(agent: Agent) -> tuple:
            return jax.lax.scan(update_minibatch, agent, minibatches)

        agent, (pg_loss, v_loss, entropy_loss, approx_kl) = run_epoch(agent)
        return (agent, key), (pg_loss, v_loss, entropy_loss, approx_kl)

    (agent, key), (pg_loss, v_loss, entropy_loss, approx_kl) = jax.lax.scan(
        update_epoch, (agent, key), (), length=args.update_epochs
    )

    y_pred = data.values.reshape(-1)
    y_true = data.returns.reshape(-1)
    var_y = jp.var(y_true)
    explained_var = jp.where(var_y == 0, jp.nan, 1.0 - jp.var(y_true - y_pred) / var_y)
    return (agent, key), (pg_loss, v_loss, entropy_loss, approx_kl, explained_var)


@functools.partial(jax.jit, static_argnames=("args",))
def train_iteration(
    args: Args,
    env: struct.PyTreeNode,
    agent: Agent,
    next_obs: Array,
    next_done: Array,
    ep_return: Array,
    key: Array,
) -> tuple:
    """One PPO iteration: collect rollout, compute GAE, update policy, aggregate metrics."""
    env, data, next_obs, next_done, ep_return, key, ret_at_done, done_now, dist = (
        collect_rollout(env, args, agent, next_obs, next_done, ep_return, key)
    )
    data = compute_gae(args, agent, next_obs, next_done, data)
    (agent, key), (pg_loss, v_loss, entropy_loss, approx_kl, _) = update_policy(
        args, agent, data, key
    )
    # Aggregate scalar metrics.
    ret_valid = jp.isfinite(ret_at_done)
    ret_sum = jp.sum(jp.where(ret_valid, ret_at_done, 0.0))
    ret_count = jp.sum(ret_valid)
    if args.task == "multi":  # average deviation over all steps
        metric_sum, metric_count = jp.sum(dist), jp.asarray(dist.size, jp.float32)
   
    metrics = {
        "ret_sum": ret_sum,
        "ret_count": ret_count,
        "metric_sum": metric_sum,
        "metric_count": metric_count,
        "approx_kl": jp.mean(approx_kl),
    }
    return (env, agent, next_obs, next_done, ep_return, key), metrics


# region Training


def train(args: Args) -> tuple[Agent, dict]:
    """Train a PPO agent for the configured task. Returns ``(agent, history)``."""
    print(
        f"Training PPO | task={args.task} | {args.num_iterations} iters "
        f"| {args.num_envs} envs x {args.num_steps} steps | device={args.device}"
    )
    print(args)
    env = make_multi_env(args, num_envs=args.num_envs, jax_device=args.device)
    obs_dim = env.single_observation_space.shape[0]
    act_dim = env.single_action_space.shape[0]

    train_steps = args.num_iterations * args.update_epochs * args.num_minibatches
    if args.anneal_lr:
        actor_lr = optax.linear_schedule(args.actor_lr, 0.0, train_steps)
        critic_lr = optax.linear_schedule(args.critic_lr, 0.0, train_steps)
    else:
        actor_lr, critic_lr = args.actor_lr, args.critic_lr

    key = jax.random.PRNGKey(args.seed)
    key, agent_key = jax.random.split(key)
    agent = Agent.create(
        agent_key,
        obs_dim,
        act_dim,
        args.hidden_size,
        actor_lr,
        critic_lr,
        args.init_logstd,
        args.max_grad_norm,
    )

    env, (next_obs, _) = env.reset(env, seed=args.seed)
    next_done = jp.zeros(args.num_envs, dtype=bool)
    ep_return = jp.zeros(args.num_envs)

    metric_name = "final_dist" if args.task == "hover" else "avg_deviation"
    history = {"step": [], "return": [], "metric": []}
    last_return, last_metric = np.nan, np.nan
    start = time.time()
    for it in range(args.num_iterations):
        carry, m = train_iteration(
            args, env, agent, next_obs, next_done, ep_return, key
        )
        env, agent, next_obs, next_done, ep_return, key = carry
        m = {k: float(v) for k, v in m.items()}
        ret = m["ret_sum"] / m["ret_count"] if m["ret_count"] > 0 else last_return
        metric = (
            m["metric_sum"] / m["metric_count"]
            if m["metric_count"] > 0
            else last_metric
        )
        last_return, last_metric = ret, metric
        step = (it + 1) * args.batch_size
        history["step"].append(step)
        history["return"].append(ret)
        history["metric"].append(metric)
        if it % max(1, args.num_iterations // 20) == 0 or it == args.num_iterations - 1:
            print(
                f"  iter {it + 1:4d}/{args.num_iterations} | step {step:>10d} | "
                f"return {ret:8.3f} | {metric_name} {metric:7.4f} m | kl {m['approx_kl']:.4f}"
            )
    elapsed = time.time() - start
    sps = int(args.num_iterations * args.batch_size / elapsed) if elapsed > 0 else 0
    print(f"Training finished in {elapsed:.1f}s ({sps} steps/s).")
    env.close()
    return agent, history


# region evaluation


def evaluate(args: Args, agent: Agent, num_envs: int = 64) -> float:
    """Run a deterministic rollout and return the task metric in meters.

    hover   -> mean final distance to the hover point (distance at episode ends).
    figure8 -> position RMSE sqrt(mean(||pos-target||^2)), matching the reference's calc_rmse (x1000=mm).
    """
    env = make_multi_env(args, num_envs=args.num_envs, jax_device=args.device)


    pos_start = 0
    n_steps = int(env.unwrapped.max_episode_time * env.unwrapped.freq) + 1
    env, (obs, _) = env.reset(env, seed=args.seed + 12345)

    @jax.jit
    def rollout(env: struct.PyTreeNode, obs: Array) -> tuple:
        def step_once(carry: tuple, _: Any) -> tuple[tuple, tuple]:
            env, obs = carry
            action = agent.mean_action(agent.actor_states.params, obs)
            env, (next_obs, _, term, trunc, info) = env.step(env, action)
            done = term | trunc
            if args.task == "multi":
                pos = env.unwrapped.data.states.pos[:, 0, :]
                dist = jp.linalg.norm(pos - env.unwrapped.target_pos, axis=-1)
            else:
                dist = jp.linalg.norm(
                    info["final_obs"][:, pos_start : pos_start + 3], axis=-1
                )
            return (env, next_obs), (dist, done)

        (env, _), (dist, done) = jax.lax.scan(step_once, (env, obs), length=n_steps)
        return dist, done

    dist, done = rollout(env, obs)
    dist, done = np.asarray(dist), np.asarray(done)
    env.close()
    if args.task == "multi":
        return float(np.sqrt(np.mean(dist**2)))  # RMSE, matching reference calc_rmse
    # hover: average the final distance over completed episodes (fall back to last-step mean).
    return float(dist[done].mean()) if done.any() else float(dist[-1].mean())


# region plotting


def plot_history(args: Args, history: dict, out_dir: Path) -> Path:
    """Plot the training reward curve and the task metric, and save the figure."""
    metric_label = (
        "Final distance to hover point [m]"
        if args.task == "hover"
        else "Avg. deviation from trajectory [m]"
    )
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(history["step"], history["return"], color="tab:blue")
    ax1.set_xlabel("Environment steps")
    ax1.set_ylabel("Mean episode return")
    ax1.set_title(f"PPO training curve ({args.task})")
    ax1.grid(alpha=0.3)
    ax2.plot(history["step"], history["metric"], color="tab:red")
    ax2.set_xlabel("Environment steps")
    ax2.set_ylabel(metric_label)
    ax2.set_title(f"Task metric ({args.task})")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    out_path = out_dir / f"ppo_{args.task}_training.png"
    fig.savefig(out_path, dpi=130)
    print(f"Saved training curves to {out_path}")
    return out_path


def save_agent(args: Args, agent: Agent, out_dir: Path) -> Path:
    """Pickle the actor/critic parameters and return the save path."""
    out_path = out_dir / f"ppo_{args.task}_params.pkl"
    params = {"actor": agent.actor_states.params, "critic": agent.critic_states.params}
    with open(out_path, "wb") as f:
        pickle.dump(jax.tree_util.tree_map(np.asarray, params), f)
    print(f"Saved agent parameters to {out_path}")
    return out_path


# region CLI


def build_args(task: str, device: str, seed: int, timesteps: int | None) -> Args:
    """Build the finalised, task-specific Args (resolving the hover pos-slice offset)."""
    overrides = dict(MULTI_ARGS)
    overrides.update(seed=seed, device=device)
    if timesteps is not None:
        overrides["total_timesteps"] = timesteps
    args = Args(**overrides)
    pos_start = 0
    return args.finalize(pos_start=pos_start)


def main(
    task: str = "hover", device: str = "cpu", seed: int = 0, timesteps: int = None
) -> None:
    """Train, evaluate, plot, and save PPO for the given task."""
    args = build_args(task, device, seed, timesteps)
    out_dir = Path(__file__).parent
    agent, history = train(args)
    metric = evaluate(args, agent)
    if args.task == "multi":
        print(f"Evaluation: final distance to hover point = {metric:.4f} m")
    else:  # report RMSE in mm to match crazyflow_experiments' eval logging
        print(
            f"Evaluation: figure-8 position RMSE = {metric * 1000:.3f} mm ({metric:.4f} m)"
        )
    plot_history(args, history, out_dir)
    save_agent(args, agent, out_dir)
    plt.show()


if __name__ == "__main__":
    fire.Fire(main)

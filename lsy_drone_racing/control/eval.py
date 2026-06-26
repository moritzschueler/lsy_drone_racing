"""Live-render a trained PPO policy in the Crazyflow sim. Run: XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 MUJOCO_GL=glfw uv run python render_eval.py --task hover."""

from __future__ import annotations

import os

os.environ.setdefault("SCIPY_ARRAY_API", "1")
os.environ.setdefault("MUJOCO_GL", "glfw")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.5")

import pickle
from pathlib import Path
from typing import Any

import fire
import jax
import jax.numpy as jp
import numpy as np
from jax import Array
from crazyflow.sim.visualize import draw_line, draw_points

from envs import FigureEightEnv, ReachPosEnv
from train_jax_fast import Agent, Args, FIGURE8_ARGS, HOVER_ARGS, hover_pos_offset, make_env


def build_eval_args(task: str, seed: int, device: str) -> Args:
    """Build the finalised, task-specific Args (1 env)."""
    overrides = dict(HOVER_ARGS if task == "hover" else FIGURE8_ARGS)
    overrides.update(seed=seed, device=device)
    args = Args(**overrides)
    pos_start = 0
    if task == "hover":  # probe the hover pos slice offset
        probe = make_env("hover", 1, "cpu", args)
        pos_start = hover_pos_offset(probe)
        probe.close()
    return args.finalize(pos_start=pos_start)


def load_agent(args: Args, env: Any, params_path: Path) -> Agent:
    """Reconstruct the Agent and load the pickled actor/critic params."""
    obs_dim = env.single_observation_space.shape[0]
    act_dim = env.single_action_space.shape[0]
    agent = Agent.create(
        jax.random.PRNGKey(args.seed),
        obs_dim,
        act_dim,
        args.hidden_size,
        args.actor_lr,
        args.critic_lr,
        args.init_logstd,
        args.max_grad_norm,
    )
    with open(params_path, "rb") as f:
        params = pickle.load(f)
    actor_params = jax.tree_util.tree_map(jp.asarray, params["actor"])
    critic_params = jax.tree_util.tree_map(jp.asarray, params["critic"])
    return agent.replace(
        actor_states=agent.actor_states.replace(params=actor_params),
        critic_states=agent.critic_states.replace(params=critic_params),
    )


def render_live(env: Any, task: str) -> None:
    """Draw the per-task goal/trajectory markers and live-render world 0."""

    base = env.unwrapped
    sim = base.sim
    sim.data = base.data
    if sim.viewer is None:  # marker list only exists after the first render
        sim.render(world=0)
    if isinstance(base, ReachPosEnv):
        draw_points(
            sim, base.goal_pos[None, 0], rgba=jp.array([1.0, 0.0, 0.0, 1.0]), size=0.02
        )
    elif isinstance(base, FigureEightEnv):
        trajectories = np.array(base.trajectories)
        idx = base.steps + base.sample_offsets[None, ...] % base.trajectories.shape[1]
        next_traj = np.array(
            base.trajectories[jp.arange(base.trajectories.shape[0])[:, None], idx]
        )
        draw_line(
            sim,
            trajectories[0, 0:-1:8, :],
            rgba=jp.array([1, 1, 1, 0.4]),
            start_size=2.0,
            end_size=2.0,
        )
        draw_line(
            sim, next_traj[0], rgba=jp.array([1, 0, 0, 1]), start_size=3.0, end_size=3.0
        )
        draw_points(sim, next_traj[0], rgba=jp.array([1.0, 0, 0, 1]), size=0.01)
        cur = base.trajectories[0, base.steps[0] % base.trajectories.shape[1]]
        draw_points(sim, cur, rgba=jp.array([0.0, 1.0, 0.0, 0.4]), size=0.02)
    sim.render(world=0)


def run(task: str, params_path: Path, seed: int, device: str, episodes: int) -> float:
    """Roll out the deterministic policy, live-render every step, return the task metric."""
    args = build_eval_args(task, seed, device)
    env = make_env(task, 1, device, args)
    pos_start = hover_pos_offset(env) if task == "hover" else 0
    n_steps = int(env.unwrapped.max_episode_time * env.unwrapped.freq) + 1
    agent = load_agent(args, env, params_path)

    @jax.jit
    def policy_step(env: Any, obs: Array) -> tuple[Any, Array, Array, Array]:
        """One deterministic step; returns env, next obs, target distance, done flag."""
        action = agent.mean_action(agent.actor_states.params, obs)
        env, (next_obs, _, term, trunc, info) = env.step(env, action)
        done = term | trunc
        if task == "figure8":
            pos = env.unwrapped.data.states.pos[:, 0, :]
            dist = jp.linalg.norm(pos - env.unwrapped.target_pos, axis=-1)
        else:
            dist = jp.linalg.norm(
                info["final_obs"][:, pos_start : pos_start + 3], axis=-1
            )
        return env, next_obs, dist, done

    dists: list[float] = []
    final_dists: list[float] = []
    print(f"Rendering {episodes} episode(s) x {n_steps} steps for task '{task}'...")
    for ep in range(episodes):
        env, (obs, _) = env.reset(env, seed=seed + 12345 + ep)
        for _ in range(n_steps):
            render_live(env, task)
            env, obs, dist, done = policy_step(env, obs)
            d = float(np.asarray(dist)[0])
            dists.append(d)
            if bool(np.asarray(done)[0]):
                final_dists.append(d)
    env.close()

    if task == "figure8":
        return float(np.sqrt(np.mean(np.asarray(dists) ** 2)))  # RMSE over all steps
    return float(np.mean(final_dists)) if final_dists else float(dists[-1])


def main(
    task: str = "hover",
    params: str | None = None,
    device: str = "cpu",
    seed: int = 0,
    episodes: int = 1,
) -> None:
    """Live-render the trained policy and print the task metric."""
    root = Path(__file__).parent
    params_path = Path(params) if params else root / f"ppo_{task}_params.pkl"
    if not params_path.exists():
        raise FileNotFoundError(f"Params file not found: {params_path}")
    metric = run(task, params_path, seed, device, episodes)
    if task == "hover":
        print(f"Render metric: final distance to hover point = {metric:.4f} m")
    else:
        print(
            f"Render metric: figure-8 position RMSE = {metric * 1000:.3f} mm ({metric:.4f} m)"
        )


if __name__ == "__main__":
    fire.Fire(main)

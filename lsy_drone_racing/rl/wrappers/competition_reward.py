"""Competition reward wrapper for self-play multi-drone racing.

Ports the opponent-relative reward shaping (rank, segment lead, proximity, downwash, victory) from
the host-side ``OpponentWrapper`` into the JAX functional pipeline, so it compiles into
``lax.scan``/``jit`` along with the rest of the rollout.

Placement matters: this wrapper must sit **before** ``RelativeRacingObs`` in the chain. It needs
the raw, per-drone ``pos``/``target_gate``/``gates_pos`` straight out of ``race_core.obs`` --
exactly the fields ``RelativeRacingObs`` drops or transforms into body-frame features. Note that
``gates_pos`` is per-drone (``(n_envs, n_drones, n_gates, 3)``) despite the underlying gate layout
being shared across drones: ``race_core.obs()`` broadcasts the shared gate positions against each
drone's own ``gates_visited`` sensor mask. Any position in the chain before ``RelativeRacingObs``
(after ``LogRewardComponents``, before/after
``SpinUpRotors``/``ActionPenalty``/``NormalizeActions``/``ZeroYaw``) works, since none of those
touch ``pos``/``target_gate``/``gates_pos``.

Only the ego drone (index 0) receives the competition reward, matching ``OpponentWrapper``.
Currently supports exactly 2 drones (1 opponent) -- generalizing to >2 drones would need a
nearest-opponent rule analogous to ``_opponent_body_frame`` in ``observation.py``.

Note: unlike ``OpponentWrapper``, this wrapper has no ``opponent_active`` gate. Whether the
opponent-drone action is meaningful yet (vs. a placeholder before ``opponent_start_step``) is a
rollout/opponent-pool concern outside the env; if the rollout feeds a static/no-op action for drone
1 before activation, the competition reward will simply be near-zero (opponent sits still) rather
than exactly gated off. Flag this if you want hard gating -- it would need a step counter threaded
through as an extra field.
"""

from __future__ import annotations

from typing import Any, Callable

import flax.struct as struct
import jax
import jax.numpy as jnp
from jax import Array

from lsy_drone_racing.rl.wrappers.wrapper_base import Wrapper


def _build_competition_reward_fn(
    rank_coef: float,
    segment_lead_coef: float,
    proximity_coef: float,
    proximity_threshold: float,
    victory_coef: float,
    downwash_coef: float,
    downwash_base_radius: float,
    downwash_expansion: float,
    downwash_vertical_softening: float,
) -> Callable[[Array, Array, Array, Array, Array, Array], dict[str, Array]]:
    """Build the pure-JAX ego-vs-single-opponent reward function (identical math to OpponentWrapper)."""

    def _competition_reward(
        ego_pos: Array,
        ego_gate: Array,
        ego_gates_pos: Array,
        opp_pos: Array,
        opp_gate: Array,
        terminated: Array,
    ) -> dict[str, Array]:
        # --- rank reward: how many gates ahead of the opponent the ego is ---
        ego_gate_f = ego_gate.astype(jnp.float32)
        opp_gate_f = opp_gate.astype(jnp.float32)
        rank = jnp.clip(ego_gate_f - opp_gate_f, 0, 3)

        # --- proximity penalty: discourage flying too close to the opponent ---
        rel_pos = opp_pos - ego_pos
        dist = jnp.linalg.norm(rel_pos, axis=-1)
        proximity = -jnp.clip(proximity_threshold - dist, 0.0, proximity_threshold)

        # --- segment lead reward: closer to the shared next gate than the opponent is ---
        safe_gate = jnp.clip(ego_gate, 0, ego_gates_pos.shape[1] - 1)
        ego_next_gate_pos = ego_gates_pos[jnp.arange(ego_gates_pos.shape[0]), safe_gate]
        ego_dist = jnp.linalg.norm(ego_next_gate_pos - ego_pos, axis=-1)
        opp_dist = jnp.linalg.norm(ego_next_gate_pos - opp_pos, axis=-1)
        dist_advantage = jnp.clip(opp_dist - ego_dist, 0.0, 5.0)
        same_segment = (ego_gate == opp_gate).astype(jnp.float32)
        segment_lead = same_segment * dist_advantage

        # --- victory reward: ego finished, opponent didn't, episode ended ---
        ego_finished = ego_gate == -1
        ego_leading = opp_gate != -1
        victory = (ego_finished & ego_leading & terminated).astype(jnp.float32)

        # --- downwash penalty: penalize ego for flying directly below the opponent ---
        vertical_dist = rel_pos[..., 2]  # positive = opponent above ego
        horizontal_dist = jnp.linalg.norm(rel_pos[..., :2], axis=-1)
        cone_radius = downwash_base_radius + downwash_expansion * vertical_dist
        proximity_factor = jnp.clip(1.0 - horizontal_dist / cone_radius, 0.0, 1.0)
        vertical_strength = 1.0 / (vertical_dist + downwash_vertical_softening)
        raw_downwash = proximity_factor * vertical_strength
        in_cone = (vertical_dist > 0.0) & (horizontal_dist < cone_radius)
        downwash = -jnp.where(in_cone, raw_downwash, 0.0)

        return {
            "rank": rank_coef * rank,
            "segment_lead": segment_lead_coef * segment_lead,
            "proximity": proximity_coef * proximity,
            "victory": victory_coef * victory,
            "downwash": downwash_coef * downwash,
        }

    return _competition_reward


@struct.dataclass
class CompetitionReward(Wrapper):
    """Add the ego-vs-opponent competition reward to the ego drone's (index 0) reward.

    Must be inserted before ``RelativeRacingObs`` in the wrapper chain -- see module docstring.
    """

    base: struct.PyTreeNode = struct.field(pytree_node=True)
    step: Callable = struct.field(pytree_node=False)
    reset: Callable = struct.field(pytree_node=False)

    @classmethod
    def create(
        cls,
        base: struct.PyTreeNode,
        n_drones: int,
        rank_coef: float = 1.0,
        segment_lead_coef: float = 0.5,
        proximity_coef: float = 1.0,
        proximity_threshold: float = 0.1,
        victory_coef: float = 50.0,
        downwash_coef: float = 1.0,
        downwash_base_radius: float = 0.2,
        downwash_expansion: float = 0.5,
        downwash_vertical_softening: float = 0.5,
    ) -> CompetitionReward:
        """Create the competition reward wrapper around ``base``.

        Args:
            base: The wrapped env, upstream of ``RelativeRacingObs`` in the chain (its ``obs`` dict
                must still expose the raw, per-drone ``pos``/``target_gate``/``gates_pos``).
            n_drones: Number of drones in the track config. Must be exactly 2 (ego + 1 opponent);
                asserted at build time.
            rank_coef, segment_lead_coef, proximity_coef, proximity_threshold, victory_coef,
                downwash_coef, downwash_base_radius, downwash_expansion,
                downwash_vertical_softening: Reward-shaping coefficients, identical in meaning to
                ``OpponentWrapper``'s constructor args of the same names.
        """
        assert n_drones == 2, (
            f"CompetitionReward currently only supports 1v1 racing (n_drones == 2), got "
            f"n_drones={n_drones}. Generalizing to >2 drones needs a nearest-opponent rule (see "
            f"_opponent_body_frame in observation.py)."
        )
        competition_reward_fn = jax.jit(
            _build_competition_reward_fn(
                rank_coef,
                segment_lead_coef,
                proximity_coef,
                proximity_threshold,
                victory_coef,
                downwash_coef,
                downwash_base_radius,
                downwash_expansion,
                downwash_vertical_softening,
            )
        )

        def _apply(
            obs: dict, reward: Array, terminated: Array
        ) -> tuple[Array, dict[str, Array]]:
            # pos/target_gate: (n_envs, n_drones, ...) -> index the drone axis.
            ego_pos, opp_pos = obs["pos"][:, 0], obs["pos"][:, 1]
            ego_gate, opp_gate = obs["target_gate"][:, 0], obs["target_gate"][:, 1]
            # gates_pos: (n_envs, n_drones, n_gates, 3) -- race_core.obs() broadcasts the raw,
            # env-shared gate positions against the per-drone gates_visited sensor mask, so it
            # comes out per-drone despite the underlying gate layout being shared. Slice to ego's
            # view of the gates.
            ego_gates_pos = obs["gates_pos"][:, 0]
            ego_terminated = terminated[:, 0]
            components = competition_reward_fn(
                ego_pos, ego_gate, ego_gates_pos, opp_pos, opp_gate, ego_terminated
            )
            competition_reward = sum(components.values())
            # reward: (n_envs, n_drones) from the base env's reward_fn. Only the ego slot gets the
            # competition bonus; the opponent's reward (unused when it's a frozen snapshot) is
            # left untouched.
            reward = reward.at[:, 0].add(competition_reward)
            return reward, components

        def reset(
            env: CompetitionReward, *, seed: int | None = None, options: dict | None = None
        ) -> tuple[CompetitionReward, tuple[Any, Any]]:
            base_env, (obs, info) = env.base.reset(env.base, seed=seed, options=options)
            return env.replace(base=base_env), (obs, info)

        def step(
            env: CompetitionReward, action: Array
        ) -> tuple[CompetitionReward, tuple[Any, ...]]:
            base_env, (obs, reward, terminated, truncated, info) = env.base.step(env.base, action)
            reward, components = _apply(obs, reward, terminated)
            info = {**info, **{f"rew/comp_{name}": v for name, v in components.items()}}
            return env.replace(base=base_env), (obs, reward, terminated, truncated, info)

        return cls(base=base, step=step, reset=reset)
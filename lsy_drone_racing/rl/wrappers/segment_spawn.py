"""Functional wrapper that respawns drones in per-gate approach cones for curriculum training.

This drives the curriculum *without* modifying the (competition) racing env / ``race_core``: it
edits the env's ``EnvData`` from the wrapper step boundary, exactly like ``SpinUpRotors``. On every
(auto)reset it overrides the freshly reset drone's pose and target gate with a cone spawn (for a
``1 - p_start_position`` fraction of envs), so the policy practices every gate from varied,
recoverable poses. See :mod:`lsy_drone_racing.envs.segment_spawn` for the geometry and schedules.

Because the vectorized env uses NEXT_STEP autoreset, resets happen *inside* the jitted step: an env
done at step ``t`` is reset at the start of step ``t + 1``. We therefore remember which envs were
done last step (``done_last_step``, threaded through the scan carry) and, after that reset has run,
overwrite their spawn before the next action is applied. Unlike ``SpinUpRotors`` (which only seeds
the unobserved ``rotor_vel``), the drone pose is part of the observation, so we also recompute the
returned observation for the just-reset envs -- the policy must see the cone pose it will act from.

The wrapper stays *inactive* (``active = False``) until ``set_progress`` is called, which training
does once per iteration (it also advances ``tau``). Evaluation never calls it, so eval always starts
from the true race start. ``active``/``tau`` are traced carry fields (not Python flags), so the
spawn logic is always traced and merely masked off when inactive -- keeping the env scannable.

Place this wrapper *directly* over the innermost env-data layer (above ``LogRewardComponents``,
below ``SpinUpRotors``) so ``env.unwrapped.data`` is the raw ``EnvData`` it edits and the
recomputed observation is in base format.
"""

from __future__ import annotations

from typing import Any, Callable

import flax.struct as struct
import jax
import jax.numpy as jnp
from jax import Array

from lsy_drone_racing.envs.race_core import obs as race_obs
from lsy_drone_racing.envs.segment_spawn import SegmentSpawnConfig, apply_spawn, segment_spawn
from lsy_drone_racing.envs.utils import load_track
from lsy_drone_racing.rl.wrappers.wrapper_base import Wrapper


@struct.dataclass
class SegmentSpawn(Wrapper):
    """Override (auto)reset spawns with cone samples in a target gate's approach corridor."""

    base: struct.PyTreeNode = struct.field(pytree_node=True)
    # Recurrent curriculum state, threaded through the scan carry.
    key: Array = struct.field(pytree_node=True)  # spawn PRNG key (split each respawn)
    tau: Array = struct.field(pytree_node=True)  # training progress in [0, 1]
    active: Array = struct.field(pytree_node=True)  # scalar bool; False until set_progress called
    done_last_step: Array = struct.field(pytree_node=True)  # envs autoreset at start of this step
    true_start: Array = struct.field(pytree_node=True)  # per-env: did the episode start true?
    start_gate: Array = struct.field(pytree_node=True)  # per-env target gate at episode start

    step: Callable = struct.field(pytree_node=False)
    reset: Callable = struct.field(pytree_node=False)

    def set_progress(self, tau: Array) -> SegmentSpawn:
        """Set training progress ``tau in [0, 1]`` and activate the curriculum.

        Training calls this once per iteration (via the wrapper chain); evaluation never does, so
        eval keeps the true race start. ``tau`` drives the spawn schedules (kappa, p_start_pos).
        """
        tau = jnp.clip(jnp.asarray(tau, dtype=jnp.float32), 0.0, 1.0)
        return self.replace(tau=tau, active=jnp.bool_(True), base=self.base.set_progress(tau))

    @classmethod
    def create(
        cls,
        base: struct.PyTreeNode,
        track: Any,
        config: SegmentSpawnConfig | None = None,
        seed: int = 0,
    ) -> SegmentSpawn:
        """Create a SegmentSpawn curriculum wrapper around the (innermost-data) base environment.

        Args:
            base: The functional env to wrap; ``base.unwrapped.data`` must be the raw ``EnvData``.
            track: The track config (``config.env.track``), used to read the true race start pose.
            config: Spawn geometry + curriculum schedule. Defaults to ``SegmentSpawnConfig()``.
            seed: Seed for the spawn PRNG key.
        """
        cfg = config if config is not None else SegmentSpawnConfig()
        num_envs = base.num_envs
        _, _, drones = load_track(track)
        nominal_start = jnp.asarray(drones["pos"])[0]  # (3,), the true race start position

        def respawn(data: Any, mask: Array, key: Array, tau: Array) -> tuple[Any, Array]:
            spawn_pos, spawn_vel, target_gate, cone_mask = segment_spawn(
                key, data.gates_pos, data.gates_quat, data.obstacles_pos, nominal_start, tau, cfg
            )
            return apply_spawn(data, mask & cone_mask, spawn_pos, spawn_vel, target_gate), cone_mask

        def obs_fn(data: Any) -> dict:
            return {k: v[:, 0] for k, v in race_obs(data).items()}

        def spawn(env: SegmentSpawn, mask: Array) -> tuple[SegmentSpawn, dict]:
            """Override the masked (and curriculum-active) envs' spawns; return recomputed obs."""
            key, subkey = jax.random.split(env.key)
            eff_mask = mask & env.active  # inactive curriculum -> no overrides (eval: true start)
            data, cone_mask = respawn(env.unwrapped.data, eff_mask, subkey, env.tau)
            env = Wrapper.recursive_replace(env, data=data)
            # Masked envs that were cone-spawned no longer start from the true start; the rest do.
            true_start = jnp.where(eff_mask, ~cone_mask, env.true_start)
            # Record the start gate for (re)spawned envs (vs the terminal gate -> cone-pass metric).
            start_gate = jnp.where(eff_mask, data.target_gate[:, 0], env.start_gate)
            env = env.replace(key=key, true_start=true_start, start_gate=start_gate)
            return env, obs_fn(data)

        def reset(
            env: SegmentSpawn, *, seed: int | None = None, options: dict | None = None
        ) -> tuple[SegmentSpawn, tuple[Any, Any]]:
            base_env, (obs, info) = env.base.reset(env.base, seed=seed, options=options)
            env = env.replace(base=base_env, done_last_step=jnp.zeros(num_envs, dtype=bool))
            env, obs = spawn(env, jnp.ones(num_envs, dtype=bool))  # cone-spawn curriculum subset
            return env, (obs, info)

        def step(env: SegmentSpawn, action: Array) -> tuple[SegmentSpawn, tuple[Any, ...]]:
            base_env, (obs, reward, terminated, truncated, info) = env.base.step(env.base, action)
            env = env.replace(base=base_env)
            # Envs done last step were just autoreset by this one -> override their spawn now. The
            # mask makes "none done" (and inactive curriculum) a no-op; obs is recomputed for all
            # envs from the (possibly respawned) data and equals passthrough obs when unchanged.
            env, obs = spawn(env, env.done_last_step)
            info = {**info, "target_gate": env.unwrapped.data.target_gate[:, 0]}
            env = env.replace(done_last_step=jnp.logical_or(terminated, truncated))
            # Expose, per env, whether the episode the done flag refers to started from the true
            # start and the gate it was spawned in front of. Envs done *this* step aren't respawned
            # yet, so both still reflect the ending episode.
            info = {
                **info,
                "episode_true_start": env.true_start,
                "episode_start_gate": env.start_gate,
            }
            return env, (obs, reward, terminated, truncated, info)

        return cls(
            base=base,
            key=jax.random.key(seed),
            tau=jnp.asarray(0.0, dtype=jnp.float32),
            active=jnp.bool_(False),
            done_last_step=jnp.zeros(num_envs, dtype=bool),
            true_start=jnp.ones(num_envs, dtype=bool),
            start_gate=jnp.zeros(num_envs, dtype=jnp.int32),
            step=step,
            reset=reset,
        )

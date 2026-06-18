"""Wrapper that respawns drones in per-gate approach cones for curriculum training.

This drives the curriculum *without* modifying the (competition) racing env / ``race_core``: it
edits the env's data from the Python step boundary, exactly like ``SpinUpRotors``. On every
(auto)reset it overrides the freshly reset drone's pose and target gate with a cone spawn (for a
``1 - p_start_position`` fraction of envs), so the policy practices every gate from varied,
recoverable poses. See :mod:`lsy_drone_racing.envs.segment_spawn` for the geometry and schedules.

Because the vectorized env uses NEXT_STEP autoreset, resets happen *inside* the jitted step: an env
done at step ``t`` is reset at the start of step ``t + 1``. We therefore remember which envs were
done last step and, after that reset has run, overwrite their spawn before the next action is
applied. Unlike ``SpinUpRotors`` (which only seeds the unobserved ``rotor_vel``), the drone pose is
part of the observation, so we also recompute the returned observation for the just-reset envs --
the policy must see the cone pose it will actually act from.

The wrapper stays *inactive* until ``set_progress`` is called (training pushes progress each
iteration). Evaluation never calls it, so eval always starts from the true race start.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from gymnasium.vector import VectorEnv, VectorWrapper
from jax import Array

from lsy_drone_racing.envs.race_core import obs as race_obs
from lsy_drone_racing.envs.segment_spawn import SegmentSpawnConfig, apply_spawn, segment_spawn
from lsy_drone_racing.envs.utils import load_track

jp = jnp


class SegmentSpawn(VectorWrapper):
    """Override (auto)reset spawns with cone samples in a target gate's approach corridor."""

    def __init__(self, env: VectorEnv, config: SegmentSpawnConfig | None = None, seed: int = 0):
        """Initialize the wrapper.

        Args:
            env: The vectorized racing env (``unwrapped`` must be the ``VecDroneRaceEnv``). This
                wrapper should sit innermost (directly around the base env) so that it manages the
                base ``data`` and returns base-format observations to the wrappers above it.
            config: Spawn geometry + curriculum schedule. Defaults to ``SegmentSpawnConfig()``.
            seed: Seed for the spawn PRNG key.
        """
        super().__init__(env)
        self.cfg = config if config is not None else SegmentSpawnConfig()
        _, _, drones = load_track(env.unwrapped.track)
        nominal_start = jp.asarray(drones["pos"])[0]  # (3,), the true race start position
        self._key = jax.random.key(seed)
        self._tau = jp.asarray(0.0, dtype=jp.float32)
        self._active = False  # inactive until training pushes progress via set_progress
        self._done_last_step = jp.zeros(self.num_envs, dtype=bool)
        # Per-env flag: did the *current* episode start from the true race start (vs a cone spawn)?
        # Used to filter training metrics to genuine full-track attempts.
        self._true_start = jp.ones(self.num_envs, dtype=bool)
        # Per-env target gate at the start of the *current* episode. Paired with the terminal target
        # gate, this lets training measure whether a cone-spawned episode passed the gate it was
        # spawned in front of (cone-spawn gate-pass rate).
        self._start_gate = jp.zeros(self.num_envs, dtype=jp.int32)

        cfg = self.cfg

        @jax.jit
        def _respawn(data: Any, mask: Array, key: Array, tau: Array) -> tuple[Any, Array]:
            spawn_pos, spawn_vel, target_gate, cone_mask = segment_spawn(
                key, data.gates_pos, data.gates_quat, data.obstacles_pos, nominal_start, tau, cfg
            )
            return apply_spawn(data, mask & cone_mask, spawn_pos, spawn_vel, target_gate), cone_mask

        @jax.jit
        def _obs(data: Any) -> dict:
            return {k: v[:, 0] for k, v in race_obs(data).items()}

        self._respawn = _respawn
        self._obs = _obs

    @property
    def device(self) -> str:
        """Delegate to the base env so downstream wrappers (e.g. NormalizeActions) find it."""
        return self.env.unwrapped.device

    def set_progress(self, tau: float) -> None:
        """Set training progress ``tau in [0, 1]`` (= global_step / total_timesteps) and activate.

        Training calls this once per iteration; evaluation never does, so eval keeps the true start.
        """
        self._tau = jp.asarray(min(max(tau, 0.0), 1.0), dtype=jp.float32)
        self._active = True

    def _spawn(self, mask: Array) -> dict:
        """Override the masked envs' spawns in the base data and return the recomputed obs."""
        self._key, subkey = jax.random.split(self._key)
        base = self.env.unwrapped
        base.data, cone_mask = self._respawn(base.data, mask, subkey, self._tau)
        # Masked envs that were cone-spawned no longer start from the true start; the rest do.
        self._true_start = jp.where(mask, ~cone_mask, self._true_start)
        # Record the start gate for the (re)spawned envs so the cone-spawn pass-rate metric can
        # compare it against the terminal target gate at episode end.
        self._start_gate = jp.where(mask, base.data.target_gate[:, 0], self._start_gate)
        return self._obs(base.data)

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[dict, dict]:
        """Reset all envs, then cone-spawn the curriculum subset."""
        obs, info = self.env.reset(seed=seed, options=options)
        self._done_last_step = jp.zeros(self.num_envs, dtype=bool)
        if self._active:
            obs = self._spawn(jp.ones(self.num_envs, dtype=bool))
        return obs, info

    def step(self, action: Array) -> tuple[dict, Array, Array, Array, dict]:
        """Step, then cone-spawn any env that was autoreset during this step."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        if self._active and bool(self._done_last_step.any()):
            # Envs done last step were just autoreset by this one -> override their spawn now.
            obs = self._spawn(self._done_last_step)
            if "target_gate" in info:
                info = {**info, "target_gate": self.env.unwrapped.data.target_gate[:, 0]}
        self._done_last_step = terminated | truncated
        # Expose, per env, whether the episode the done flag refers to started from the true start
        # and the gate it was spawned in front of. Envs done *this* step haven't been respawned yet,
        # so both still reflect the ending episode.
        info = {
            **info,
            "episode_true_start": self._true_start,
            "episode_start_gate": self._start_gate,
        }
        return obs, reward, terminated, truncated, info

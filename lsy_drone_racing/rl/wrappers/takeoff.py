"""Functional wrapper that spins the rotors up after every (auto)reset to assist takeoff."""

from __future__ import annotations

from typing import Any, Callable

import flax.struct as struct
import jax.numpy as jnp
from crazyflow.utils import leaf_replace
from jax import Array

from lsy_drone_racing.rl.wrappers.wrapper_base import Wrapper

# Steady-state (hover) rotor speed for the cf21B_500 first_principles drone. Seeding rotor_vel to
# this on (auto)reset starts the freshly reset drone in equilibrium, so a neutral action holds
# altitude and any positive thrust lifts off immediately -- instead of sitting on the ground with
# cold rotors and never exploring takeoff.
HOVER_ROTOR_VEL = 16200.0


@struct.dataclass
class SpinUpRotors(Wrapper):
    """Seed the rotors at ~hover RPM whenever an env (auto)resets, to help the drone take off.

    The racing env starts the drone on the ground with cold rotors (``rotor_vel = 0``), so a
    neutral action keeps it grounded and the only safe behaviour the policy discovers is to sit
    still. This wrapper seeds ``rotor_vel`` at the seed value on every (auto)reset without modifying
    the (competition) racing env, by editing the innermost env's sim state.

    Because the vectorized env uses NEXT_STEP autoreset (an env done at step ``t`` is reset at the
    start of ``t + 1``), ``done_last_step`` is threaded through the scan carry: we remember which
    envs were done last step and warm their (now reset, still cold) rotors after that reset has run.
    ``rotor_vel`` is unobserved sim state, so warming it after the observation is computed does not
    change the returned observation.
    """

    base: struct.PyTreeNode = struct.field(pytree_node=True)
    done_last_step: Array = struct.field(pytree_node=True)

    step: Callable = struct.field(pytree_node=False)
    reset: Callable = struct.field(pytree_node=False)

    @classmethod
    def create(
        cls, base: struct.PyTreeNode, rotor_vel: float = HOVER_ROTOR_VEL, n_drones: int = 1
    ) -> SpinUpRotors:
        """Create a SpinUpRotors wrapper around the base environment.

        ``n_drones > 1`` (multi-agent racing): the base env reports per-drone ``(n_envs, n_drones)``
        done flags, but rotor warming is per-env (a world reset warms every drone in it), so the
        done flags are reduced with ``any`` over the drone axis into the ``(n_envs,)`` mask
        ``done_last_step`` and ``warm_rotors`` expect.
        """
        num_envs = base.num_envs
        multi = n_drones > 1

        def env_done(terminated: Array, truncated: Array) -> Array:
            """Per-env done mask (n_envs,): reduce over the drone axis in multi-agent."""
            done = terminated | truncated
            return done.any(axis=-1) if multi else done

        def warm_rotors(env: SpinUpRotors, mask: Array) -> SpinUpRotors:
            """Seed rotor_vel to the hover value for the masked envs in the innermost env's data."""
            data = env.unwrapped.data
            states = data.sim_data.states
            target = jnp.full_like(states.rotor_vel, rotor_vel)
            states = leaf_replace(states, mask, rotor_vel=target)
            new_data = data.replace(sim_data=data.sim_data.replace(states=states))
            return Wrapper.recursive_replace(env, data=new_data)

        def reset(
            env: SpinUpRotors, *, seed: int | None = None, options: dict | None = None
        ) -> tuple[SpinUpRotors, tuple[Any, Any]]:
            base_env, (obs, info) = env.base.reset(env.base, seed=seed, options=options)
            env = env.replace(base=base_env, done_last_step=jnp.zeros(num_envs, dtype=bool))
            env = warm_rotors(env, jnp.ones(num_envs, dtype=bool))  # warm every drone
            return env, (obs, info)

        def step(env: SpinUpRotors, action: Array) -> tuple[SpinUpRotors, tuple[Any, ...]]:
            base_env, (obs, reward, terminated, truncated, info) = env.base.step(env.base, action)
            env = env.replace(base=base_env)
            # Envs done on the previous step were just autoreset by this step -> warm them now,
            # before their next action is applied. The mask makes "none done" a no-op.
            env = warm_rotors(env, env.done_last_step)
            env = env.replace(done_last_step=env_done(terminated, truncated))
            return env, (obs, reward, terminated, truncated, info)

        return cls(
            base=base, done_last_step=jnp.zeros(num_envs, dtype=bool), step=step, reset=reset
        )

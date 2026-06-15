"""Wrapper that spins the rotors up after every (auto)reset to assist takeoff."""

import jax.numpy as jnp
from crazyflow.utils import leaf_replace
from gymnasium.vector import VectorEnv, VectorWrapper
from jax import Array

jp = jnp


class SpinUpRotors(VectorWrapper):
    """Seed the rotors at ~hover RPM whenever an env (auto)resets, to help the drone take off.

    The racing env starts the drone on the ground (z ~ 0.01) with cold rotors (rotor_vel = 0),
    so a neutral action keeps it grounded and the only safe behaviour the policy discovers is to
    sit still -- it never explores taking off. The hover/trajectory tasks sidestep this by seeding
    rotor_vel at reset; this wrapper replicates that *without* modifying the (competition) racing
    env, by editing the env's sim state from the Python step boundary.

    Because the vectorized env uses NEXT_STEP autoreset, resets happen *inside* the jitted step
    kernel: an env that returns ``done`` at step ``t`` is reset at the start of step ``t + 1``. We
    therefore remember which envs were done last step and, after that reset has run (i.e. once the
    next ``step`` returns), overwrite their ``rotor_vel`` before the next action is applied. The
    overwritten state is what the env reads on the following step, so the freshly reset drone
    starts with warm rotors.
    """

    def __init__(self, env: VectorEnv, rotor_vel: float = 16200.0):
        """Initialize the wrapper.

        Args:
            env: The vectorized racing environment (any wrapper chain whose ``unwrapped`` is the
                ``VecDroneRaceEnv``).
            rotor_vel: Rotor angular velocity to seed on reset. The default ~16200 is the
                steady-state (hover) rotor speed for the cf21B_500 ``first_principles`` drone, so
                the freshly reset drone starts in equilibrium: a neutral action holds altitude and
                any positive thrust lifts off immediately, with no spin-up dead time. (The
                hover/trajectory tasks use 10000, which is well below hover for this model and
                barely improves on cold rotors.)
        """
        super().__init__(env)
        self.rotor_vel = rotor_vel
        # Envs reset during the *next* step are those done now; nothing is pending at construction.
        self._done_last_step = jp.zeros(self.num_envs, dtype=bool)

    @property
    def device(self) -> str:
        """Delegate to the base env so downstream wrappers (e.g. NormalizeActions) find it."""
        return self.env.unwrapped.device

    def _warm_rotors(self, mask: Array) -> None:
        """Set rotor_vel to the seed value for the masked envs in the base env's sim state."""
        data = self.env.unwrapped.data
        states = data.sim_data.states
        target = jp.full_like(states.rotor_vel, self.rotor_vel)
        states = leaf_replace(states, mask, rotor_vel=target)
        self.env.unwrapped.data = data.replace(sim_data=data.sim_data.replace(states=states))

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[dict, dict]:
        """Reset all envs and warm every drone's rotors."""
        obs, info = self.env.reset(seed=seed, options=options)
        self._warm_rotors(jp.ones(self.num_envs, dtype=bool))
        self._done_last_step = jp.zeros(self.num_envs, dtype=bool)
        return obs, info

    def step(self, action: Array) -> tuple[dict, Array, Array, Array, dict]:
        """Step, then warm the rotors of any env that was autoreset during this step."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        # Envs done on the previous step have just been autoreset by this one -> warm them now,
        # before their next action is applied.
        if bool(self._done_last_step.any()):
            self._warm_rotors(self._done_last_step)
        self._done_last_step = terminated | truncated
        return obs, reward, terminated, truncated, info

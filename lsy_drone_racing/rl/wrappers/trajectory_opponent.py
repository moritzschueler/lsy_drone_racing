"""JAX-jittable, single-pass position->attitude PID trajectory follower for scripted opponents.

Ports the cascaded position/velocity PID + desired-attitude construction from
``lsy_drone_racing.control.attitude_controller.AttitudeController`` (a proven host-side controller
that already flies this exact waypoint spline under the same ``control_mode="attitude"`` action
convention) into pure, batched JAX. This lets a scripted opponent run inside the same
``jax.lax.scan``-compiled rollout as the self-play policy (see ``ippo.py``) with no host round-trip
after construction: the spline is fit once with ``scipy`` at build time and baked into fixed-size
coefficient arrays that a traced step function evaluates.

Every array here carries a leading ``(n_envs, n_opponents)`` (or a subset thereof) batch shape --
``TrajectoryPID.action`` is elementwise/broadcast-only (no explicit ``vmap`` needed) so it applies
uniformly whether there is one scripted opponent slot or several.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.scipy.spatial.transform import Rotation as R
from scipy.interpolate import CubicSpline

# Waypoints and PID gains identical to the reference host-side
# ``lsy_drone_racing.control.attitude_controller.AttitudeController``. ``start_pos`` (that
# controller's first waypoint, taken from ``obs["pos"]`` at runtime) is substituted at build time
# with the opponent drone's nominal track spawn position -- see ``build_trajectory_pid``.
DEFAULT_WAYPOINTS = np.array(
    [
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
DEFAULT_T_TOTAL = 18.0  # seconds, nominal (speed multiplier == 1.0) single-pass duration
DEFAULT_KP = (0.4, 0.4, 1.25)
DEFAULT_KI = (0.05, 0.05, 0.05)
DEFAULT_KD = (0.2, 0.2, 0.4)
DEFAULT_KI_RANGE = (2.0, 2.0, 0.4)


def _fit_position_spline(waypoints: np.ndarray, t_total: float) -> tuple[np.ndarray, np.ndarray]:
    """Fit a not-a-knot cubic position spline through ``waypoints`` (matches the reference PID).

    Returns ``(breakpoints, coeffs)``: ``breakpoints`` has shape ``(n_knots,)``, ``coeffs`` has
    shape ``(4, n_segments, 3)`` (``scipy.PPoly`` power-basis convention, highest degree first), so
    segment ``i``'s position at local time ``dt = t - breakpoints[i]`` is
    ``coeffs[0, i] * dt**3 + coeffs[1, i] * dt**2 + coeffs[2, i] * dt + coeffs[3, i]``.
    """
    t = np.linspace(0.0, t_total, len(waypoints))
    spline = CubicSpline(t, waypoints, axis=0)
    return spline.x, spline.c


@dataclass
class TrajectoryPID:
    """Static (host-built), pure-batched-JAX cascaded position PID for a scripted opponent.

    Every field is a compile-time constant baked into the jaxpr wherever :meth:`action` is traced
    from inside ``jax.lax.scan``/``jit`` (no host round-trip). Not a flax pytree -- it is built once
    outside the training scan and only ever closed over, never threaded through a carry.
    """

    breakpoints: Array  # (n_knots,)
    coeffs: Array  # (4, n_segments, 3), see _fit_position_spline
    t_total: float  # seconds; virtual time is clamped to this (single pass, then hold)
    drone_mass: float
    freq: float  # env step frequency (Hz); integrates the PID's integral error term
    kp: Array  # (3,)
    ki: Array  # (3,)
    kd: Array  # (3,)
    ki_range: Array  # (3,) integral-error clamp
    action_low: Array  # (4,) physical [roll, pitch, yaw, thrust] action-space bounds
    action_high: Array  # (4,)
    g: float = 9.81

    def _spline(self, t: Array) -> tuple[Array, Array]:
        """Desired (position, velocity) at ``t``, single-pass clamped to ``[0, t_total]``."""
        t = jnp.clip(t, 0.0, self.t_total)
        n_segments = self.coeffs.shape[1]
        idx = jnp.clip(jnp.searchsorted(self.breakpoints, t, side="right") - 1, 0, n_segments - 1)
        dt = (t - self.breakpoints[idx])[..., None]  # (..., 1), broadcasts against the xyz axis
        c = self.coeffs[:, idx]  # (4, ..., 3)
        pos = ((c[0] * dt + c[1]) * dt + c[2]) * dt + c[3]
        vel = (3.0 * c[0] * dt + 2.0 * c[1]) * dt + c[2]
        return pos, vel

    def action(
        self, pos: Array, vel: Array, quat: Array, virtual_t: Array, i_error: Array
    ) -> tuple[Array, Array]:
        """Cascaded position/velocity PID -> desired attitude, batched over any leading shape.

        Args:
            pos: World position, ``(..., 3)``.
            vel: World velocity, ``(..., 3)``.
            quat: World attitude, ``(..., 4)``.
            virtual_t: Per-slot elapsed trajectory time (already speed-scaled by the caller),
                ``(...,)``.
            i_error: Integral error carried across steps, ``(..., 3)``.

        Returns:
            ``(action_phys, new_i_error)``. ``action_phys`` is ``(..., 4)`` in the same physical
            units/order as ``race_core.build_action_space("attitude", ...)``:
            ``[roll, pitch, yaw, thrust]``.
        """
        des_pos, des_vel = self._spline(virtual_t)
        pos_error = des_pos - pos
        vel_error = des_vel - vel

        i_error = jnp.clip(i_error + pos_error / self.freq, -self.ki_range, self.ki_range)
        target_thrust = self.kp * pos_error + self.ki * i_error + self.kd * vel_error
        target_thrust = target_thrust.at[..., 2].add(self.drone_mass * self.g)

        z_axis = R.from_quat(quat).as_matrix()[..., :, 2]  # body z-axis in world frame
        thrust = jnp.sum(target_thrust * z_axis, axis=-1)

        z_des = target_thrust / jnp.linalg.norm(target_thrust, axis=-1, keepdims=True)
        x_c = jnp.zeros_like(z_des).at[..., 0].set(1.0)  # des_yaw == 0 -> x_c_des = [1, 0, 0]
        y_des = jnp.cross(z_des, x_c)
        y_des = y_des / jnp.linalg.norm(y_des, axis=-1, keepdims=True)
        x_des = jnp.cross(y_des, z_des)

        rot_des = jnp.stack([x_des, y_des, z_des], axis=-1)  # columns = desired body axes in world
        euler_des = R.from_matrix(rot_des).as_euler("xyz")  # (..., 3): roll, pitch, yaw

        action_phys = jnp.concatenate([euler_des, thrust[..., None]], axis=-1)
        return action_phys, i_error

    def normalize(self, action_phys: Array) -> Array:
        """Inverse of ``wrappers.reward.NormalizeActions``: physical units -> ``[-1, 1]``."""
        scale = (self.action_high - self.action_low) / 2.0
        mean = (self.action_high + self.action_low) / 2.0
        return jnp.clip((action_phys - mean) / scale, -1.0, 1.0)


def build_trajectory_pid(
    start_pos: np.ndarray,
    drone_mass: float,
    freq: float,
    control_mode: str,
    action_low: np.ndarray,
    action_high: np.ndarray,
    waypoints: np.ndarray = DEFAULT_WAYPOINTS,
    t_total: float = DEFAULT_T_TOTAL,
    kp: tuple[float, float, float] = DEFAULT_KP,
    ki: tuple[float, float, float] = DEFAULT_KI,
    kd: tuple[float, float, float] = DEFAULT_KD,
    ki_range: tuple[float, float, float] = DEFAULT_KI_RANGE,
) -> TrajectoryPID:
    """Fit the waypoint spline and package the batched-JAX PID config.

    Args:
        start_pos: The opponent drone's nominal track spawn position (3,), prepended to
            ``waypoints`` as the spline's first knot.
        drone_mass: Drone mass (kg), for gravity feed-forward.
        freq: Env step frequency (Hz) -- both the PID's integral-error timestep and the rate at
            which callers should advance ``virtual_t``.
        control_mode: Must be ``"attitude"`` -- this controller emits physical
            ``[roll, pitch, yaw, thrust]`` setpoints, meaningless under state control.
        action_low: The env's physical (pre-``NormalizeActions``) action lower bound, from
            ``race_core.build_action_space(control_mode, drone_model)``.
        action_high: The matching physical action upper bound.
        waypoints: Waypoints to fly through after ``start_pos``, defaulting to the reference
            ``AttitudeController``'s track.
        t_total: Nominal (speed multiplier 1.0) time to fly the spline once, in seconds.
        kp: Proportional gain, defaulting to the reference ``AttitudeController``'s value.
        ki: Integral gain, defaulting to the reference ``AttitudeController``'s value.
        kd: Derivative gain, defaulting to the reference ``AttitudeController``'s value.
        ki_range: Integral-error clamp, defaulting to the reference ``AttitudeController``'s value.
    """
    assert control_mode == "attitude", (
        "the scripted PID opponent emits physical roll/pitch/yaw/thrust setpoints and needs "
        f"control_mode='attitude', got '{control_mode}'."
    )
    full_waypoints = np.concatenate([np.asarray(start_pos, dtype=np.float64)[None], waypoints])
    breakpoints, coeffs = _fit_position_spline(full_waypoints, t_total)
    return TrajectoryPID(
        breakpoints=jnp.asarray(breakpoints, dtype=jnp.float32),
        coeffs=jnp.asarray(coeffs, dtype=jnp.float32),
        t_total=float(t_total),
        drone_mass=float(drone_mass),
        freq=float(freq),
        kp=jnp.asarray(kp, dtype=jnp.float32),
        ki=jnp.asarray(ki, dtype=jnp.float32),
        kd=jnp.asarray(kd, dtype=jnp.float32),
        ki_range=jnp.asarray(ki_range, dtype=jnp.float32),
        action_low=jnp.asarray(action_low, dtype=jnp.float32),
        action_high=jnp.asarray(action_high, dtype=jnp.float32),
    )

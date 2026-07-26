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
from typing import Any

import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.scipy.spatial.transform import Rotation as R
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation

from lsy_drone_racing.rl.wrappers.wrapper_base import Wrapper

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

# Random mid-track spawn support (see teleport_opponents). Both in seconds of nominal (speed
# multiplier == 1.0) trajectory time, the same axis as ``virtual_t``/``gate_times``.
GATE_TIME_EPS = 0.1  # Forward bias when mapping a spawn time to a target gate: spawning a hair
# past gate g's plane with n_gates_passed still targeting g would stall the gate counter forever
# (``gate_passed`` requires the *previous* position strictly before the plane), while targeting
# gate g + 1 a hair early is monotone and self-corrects at the next plane crossing.
SPAWN_TIME_MARGIN = 0.5  # Latest allowed spawn time is this far before the last gate's pass time,
# so a teleported opponent always has at least one gate left to fly (never spawns "finished").


def _fit_position_spline(waypoints: np.ndarray, t_total: float) -> CubicSpline:
    """Fit a not-a-knot cubic position spline through ``waypoints`` (matches the reference PID).

    The returned ``scipy`` spline exposes ``x`` (breakpoints, shape ``(n_knots,)``) and ``c``
    (coeffs, shape ``(4, n_segments, 3)``; ``scipy.PPoly`` power-basis convention, highest degree
    first), so segment ``i``'s position at local time ``dt = t - x[i]`` is
    ``c[0, i] * dt**3 + c[1, i] * dt**2 + c[2, i] * dt + c[3, i]``.
    """
    t = np.linspace(0.0, t_total, len(waypoints))
    return CubicSpline(t, waypoints, axis=0)


def _compute_gate_pass_times(
    spline: CubicSpline, gates: list[Any], t_total: float, n_samples: int = 4000
) -> np.ndarray:
    """Times at which the spline crosses each gate's plane, in nominal trajectory time.

    Gates are crossed along their local +x axis (see ``envs.utils.gate_passed``). For each gate the
    spline is sampled densely, transformed into the (nominal) gate frame, and the first local-x
    sign crossing after the previous gate's pass time that is loosely within the gate frame is
    refined with a linear sub-sample interpolation. Sequential search (rather than closest
    approach) is required because the track re-approaches earlier gates.

    Args:
        spline: The fitted position spline (float64, from ``_fit_position_spline``).
        gates: ``config.env.track.gates`` entries with ``"pos"`` and ``"rpy"`` fields.
        t_total: Nominal single-pass duration of the spline, in seconds.
        n_samples: Number of dense time samples over ``[0, t_total]``.

    Returns:
        Strictly increasing pass times, shape ``(n_gates,)``, all within ``(0, t_total)``.
    """
    ts = np.linspace(0.0, t_total, n_samples)
    pos = spline(ts)  # (n_samples, 3)
    t_prev, times = 0.0, []
    for gate in gates:
        rot = Rotation.from_euler("xyz", np.asarray(gate["rpy"], dtype=np.float64))
        local = rot.inv().apply(pos - np.asarray(gate["pos"], dtype=np.float64))
        x = local[:, 0]
        crossing = (x[:-1] < 0) & (x[1:] >= 0) & (ts[1:] > t_prev)
        # Loose in-plane check: reject far-field plane crossings (the plane is infinite, the
        # trajectory only flies through the ~0.4 m opening).
        crossing &= np.linalg.norm(local[1:, 1:], axis=-1) < 0.6
        assert crossing.any(), (
            f"opponent spline never crosses the plane of the gate at {gate['pos']} -- the gate "
            "pass times needed for random mid-track spawns cannot be computed for this track."
        )
        i = int(np.argmax(crossing))
        alpha = -x[i] / (x[i + 1] - x[i])
        t_prev = ts[i] + alpha * (ts[i + 1] - ts[i])
        times.append(t_prev)
    return np.asarray(times)


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
    # (n_gates,) nominal-tau gate-plane crossing times (same time axis as ``virtual_t``, so they
    # compare directly regardless of the per-episode speed multiplier). Only baked when ``gates``
    # is passed to ``build_trajectory_pid``; required by ``teleport_opponents``.
    gate_times: Array | None = None

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
        self, pos: Array, vel: Array, quat: Array, virtual_t: Array, speed: Array, i_error: Array
    ) -> tuple[Array, Array]:
        """Cascaded position/velocity PID -> desired attitude, batched over any leading shape.

        Args:
            pos: World position, ``(..., 3)``.
            vel: World velocity, ``(..., 3)``.
            quat: World attitude, ``(..., 4)``.
            virtual_t: Per-slot elapsed trajectory time (already speed-scaled by the caller),
                ``(...,)``.
            speed: The same per-slot time-scale multiplier used to advance ``virtual_t``,
                broadcastable against ``virtual_t``. ``_spline`` differentiates position with
                respect to its own internal (nominal, speed==1) time axis tau; by the chain rule
                the real-world (m/s) velocity target is ``d(pos)/dt = d(pos)/dtau * dtau/dt``, and
                ``dtau/dt`` is exactly ``speed``. Without this scaling the velocity feedforward
                would be wrong by a factor of ``speed`` at any speed other than 1.0, making the PID
                lag a target that is (correctly) moving faster or slower than the feedforward says.
            i_error: Integral error carried across steps, ``(..., 3)``.

        Returns:
            ``(action_phys, new_i_error)``. ``action_phys`` is ``(..., 4)`` in the same physical
            units/order as ``race_core.build_action_space("attitude", ...)``:
            ``[roll, pitch, yaw, thrust]``.
        """
        des_pos, des_vel_dtau = self._spline(virtual_t)
        des_vel = des_vel_dtau * jnp.asarray(speed)[..., None]
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
    gates: list[Any] | None = None,
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
        gates: Optional ``config.env.track.gates`` entries (``"pos"``/``"rpy"``). When given, the
            spline's nominal gate-plane crossing times are baked into ``gate_times`` (needed for
            random mid-track spawns via ``teleport_opponents``).
    """
    assert control_mode == "attitude", (
        "the scripted PID opponent emits physical roll/pitch/yaw/thrust setpoints and needs "
        f"control_mode='attitude', got '{control_mode}'."
    )
    full_waypoints = np.concatenate([np.asarray(start_pos, dtype=np.float64)[None], waypoints])
    spline = _fit_position_spline(full_waypoints, t_total)
    gate_times = None
    if gates is not None:
        gate_times = jnp.asarray(
            _compute_gate_pass_times(spline, gates, t_total), dtype=jnp.float32
        )
    return TrajectoryPID(
        breakpoints=jnp.asarray(spline.x, dtype=jnp.float32),
        coeffs=jnp.asarray(spline.c, dtype=jnp.float32),
        t_total=float(t_total),
        drone_mass=float(drone_mass),
        freq=float(freq),
        kp=jnp.asarray(kp, dtype=jnp.float32),
        ki=jnp.asarray(ki, dtype=jnp.float32),
        kd=jnp.asarray(kd, dtype=jnp.float32),
        ki_range=jnp.asarray(ki_range, dtype=jnp.float32),
        action_low=jnp.asarray(action_low, dtype=jnp.float32),
        action_high=jnp.asarray(action_high, dtype=jnp.float32),
        gate_times=gate_times,
    )


# Waypoint geometry and gains for the gate/obstacle-aware navigator opponent (ported from the
# reference host-side ``lsy_drone_racing.control.attitude_controller_variant.AttitudeController_1``,
# which flies this exact track -- entry/gate/exit points offset off each gate's crossing normal,
# plus a dip point near a nearby obstacle -- and clears ~50% of level2 runs end to end). Unlike
# ``DEFAULT_WAYPOINTS`` (a fixed path independent of the track config), these are derived from
# ``config.env.track.gates``/``obstacles`` at build time, so they stay correct for whichever track
# config is passed to :func:`build_navigator_pid`. The per-gate obstacle indices/offsets are tuned
# for -- and only valid on -- a 4-gate track laid out like level2/multi_level2.
_NAV_GATE_IN_OFFSET = 0.23  # m along the gate normal, entry point
_NAV_GATE_IN_OFFSET_PREV = 0.10  # m along the vector from the previous waypoint
_NAV_GATE_OUT_OFFSET = 0.23  # m along the gate normal, exit point
_NAV_GATE_OUT_OFFSET_NEXT = 0.10  # m along the vector toward the next gate
_NAV_DIP_DEGREE = 120  # deg; above this incidence angle the exit point dips the other way
_NAV_POINT_AT_OBSTACLE = (True, True, True, False)  # per-gate: add a dip waypoint near an obstacle
_NAV_OBSTACLE_IND = (1, 0, 3, 0)  # per-gate index into obstacles_pos for the dip waypoint
_NAV_OBSTACLE_OFFSET = np.array(
    [[0.17, -0.17, 0.1], [0.2, -0.2, -0.1], [0.1, 0.2, 0.2], [0.0, 0.0, 0.0]]
)
DEFAULT_NAV_T_TOTAL = 11.0  # seconds, nominal (speed multiplier == 1.0) single-pass duration
DEFAULT_NAV_KP = (0.7, 0.7, 2.7)
DEFAULT_NAV_KI = (0.0, 0.0, 0.0)
DEFAULT_NAV_KD = (0.4, 0.4, 0.8)
DEFAULT_NAV_KI_RANGE = (2.0, 2.0, 0.4)


def _build_navigator_waypoints(
    start_pos: np.ndarray, gates_pos: np.ndarray, gates_quat: np.ndarray, obstacles_pos: np.ndarray
) -> np.ndarray:
    """Build the entry/gate/exit(/obstacle-dip) waypoint sequence for the navigator opponent.

    Pure-numpy port of ``AttitudeController_1.create_waypoints``, taking the nominal track arrays
    directly instead of a live observation dict -- this only ever runs once, host-side, at build
    time (see :func:`build_navigator_pid`), exactly like ``_fit_position_spline`` above.
    """
    assert len(gates_pos) == 4, (
        "the navigator opponent's per-gate obstacle indices/offsets are tuned for a 4-gate track "
        f"like level2/multi_level2, got {len(gates_pos)} gates."
    )
    waypoints = [np.asarray(start_pos, dtype=np.float64)]
    takeoff = waypoints[0].copy()
    takeoff += [0.6, -0.1, 0.3]
    waypoints.append(takeoff)

    for i, (pos, quat) in enumerate(zip(gates_pos, gates_quat)):
        normal = Rotation.from_quat(quat).apply([1.0, 0.0, 0.0])
        vec_prev_to_gate = pos - waypoints[-1]
        vec_prev_to_gate_norm = vec_prev_to_gate / np.linalg.norm(vec_prev_to_gate)

        if i + 1 < len(gates_pos):
            vec_gate_to_next_gate = gates_pos[i + 1] - pos
            vec_gate_to_next_gate_norm = vec_gate_to_next_gate / np.linalg.norm(
                vec_gate_to_next_gate
            )
        else:
            vec_gate_to_next_gate_norm = vec_prev_to_gate_norm

        cos_theta = np.dot(normal, vec_gate_to_next_gate_norm)
        theta_dec = np.degrees(np.arccos(cos_theta))

        gate_in_dir_vec = _NAV_GATE_IN_OFFSET_PREV * vec_prev_to_gate_norm
        gate_out_dir_vec = _NAV_GATE_OUT_OFFSET_NEXT * vec_gate_to_next_gate_norm

        length_normal_in = np.dot(gate_in_dir_vec, normal)
        length_normal_out = np.dot(gate_out_dir_vec, normal)

        add_normal_in = length_normal_in - _NAV_GATE_IN_OFFSET
        add_normal_out = _NAV_GATE_OUT_OFFSET - length_normal_out

        entry = pos - gate_in_dir_vec + add_normal_in * normal
        if theta_dec < _NAV_DIP_DEGREE:
            exit_ = pos + gate_out_dir_vec + add_normal_out * normal
        else:
            # Distance is in the opposite direction when theta_dec > 90.
            add_normal_out = _NAV_GATE_OUT_OFFSET + length_normal_out
            exit_ = pos + gate_out_dir_vec - add_normal_out * normal

        waypoints.append(entry)
        waypoints.append(pos)
        waypoints.append(exit_)

        if _NAV_POINT_AT_OBSTACLE[i]:
            dip = obstacles_pos[_NAV_OBSTACLE_IND[i]].copy()
            dip[2] = exit_[2]  # fly the dip at the previous waypoint's height
            dip = dip + _NAV_OBSTACLE_OFFSET[i]
            waypoints.append(dip)

    return np.array(waypoints)


def _fit_distance_spline(waypoints: np.ndarray, t_total: float) -> CubicSpline:
    """Fit a not-a-knot cubic spline timed by cumulative waypoint distance, not equal spacing.

    Matches ``AttitudeController_1.create_spline``: waypoints from the navigator builder are
    unevenly spaced (a short obstacle-dip hop next to a long inter-gate leg), so distance-based
    timing keeps the feed-forward velocity roughly constant instead of racing through short legs.
    """
    distances = np.linalg.norm(np.diff(waypoints, axis=0), axis=1)
    cumulative_dist = np.insert(np.cumsum(distances), 0, 0)
    t = t_total * cumulative_dist / cumulative_dist[-1]
    return CubicSpline(t, waypoints, axis=0)


def build_navigator_pid(
    start_pos: np.ndarray,
    drone_mass: float,
    freq: float,
    control_mode: str,
    action_low: np.ndarray,
    action_high: np.ndarray,
    gates: list[Any],
    obstacles: list[Any],
    t_total: float = DEFAULT_NAV_T_TOTAL,
    kp: tuple[float, float, float] = DEFAULT_NAV_KP,
    ki: tuple[float, float, float] = DEFAULT_NAV_KI,
    kd: tuple[float, float, float] = DEFAULT_NAV_KD,
    ki_range: tuple[float, float, float] = DEFAULT_NAV_KI_RANGE,
) -> TrajectoryPID:
    """Build the gate/obstacle-aware navigator opponent for a specific track config.

    Unlike :func:`build_trajectory_pid`'s fixed ``DEFAULT_WAYPOINTS``, the waypoints here are
    derived from ``gates``/``obstacles`` themselves (see :func:`_build_navigator_waypoints`), so
    the resulting spline actually threads this track's gates and obstacles rather than a
    once-tuned, track-independent path. The result is a plain :class:`TrajectoryPID`, so it plugs
    into ``ippo.py``/``teleport_opponents`` exactly like :func:`build_trajectory_pid`'s output.

    Args:
        start_pos: The opponent drone's nominal track spawn position (3,).
        drone_mass: Drone mass (kg), for gravity feed-forward.
        freq: Env step frequency (Hz).
        control_mode: Must be ``"attitude"``.
        action_low: The env's physical (pre-``NormalizeActions``) action lower bound.
        action_high: The matching physical action upper bound.
        gates: ``config.env.track.gates`` entries (``"pos"``/``"rpy"``), raw list order.
        obstacles: ``config.env.track.obstacles`` entries (``"pos"``), raw list order.
        t_total: Nominal (speed multiplier 1.0) time to fly the spline once, in seconds.
        kp: Proportional gain, defaulting to the reference navigator's tuned value.
        ki: Integral gain, defaulting to the reference navigator's tuned value.
        kd: Derivative gain, defaulting to the reference navigator's tuned value.
        ki_range: Integral-error clamp, defaulting to the reference navigator's tuned value.
    """
    assert control_mode == "attitude", (
        "the scripted PID opponent emits physical roll/pitch/yaw/thrust setpoints and needs "
        f"control_mode='attitude', got '{control_mode}'."
    )
    gates_pos = np.array([g["pos"] for g in gates], dtype=np.float64)
    gates_quat = Rotation.from_euler(
        "xyz", np.array([g["rpy"] for g in gates], dtype=np.float64)
    ).as_quat()
    obstacles_pos = np.array([o["pos"] for o in obstacles], dtype=np.float64)

    waypoints = _build_navigator_waypoints(
        np.asarray(start_pos, dtype=np.float64), gates_pos, gates_quat, obstacles_pos
    )
    spline = _fit_distance_spline(waypoints, t_total)
    gate_times = jnp.asarray(_compute_gate_pass_times(spline, gates, t_total), dtype=jnp.float32)

    return TrajectoryPID(
        breakpoints=jnp.asarray(spline.x, dtype=jnp.float32),
        coeffs=jnp.asarray(spline.c, dtype=jnp.float32),
        t_total=float(t_total),
        drone_mass=float(drone_mass),
        freq=float(freq),
        kp=jnp.asarray(kp, dtype=jnp.float32),
        ki=jnp.asarray(ki, dtype=jnp.float32),
        kd=jnp.asarray(kd, dtype=jnp.float32),
        ki_range=jnp.asarray(ki_range, dtype=jnp.float32),
        action_low=jnp.asarray(action_low, dtype=jnp.float32),
        action_high=jnp.asarray(action_high, dtype=jnp.float32),
        gate_times=gate_times,
    )


def teleport_opponents(
    env: Wrapper, pid: TrajectoryPID, traj_t: Array, traj_speed: Array, mask: Array
) -> Wrapper:
    """Place opponent drones (slots 1..) of the masked ``(env, slot)`` pairs onto the PID spline.

    Rewrites the innermost env data (the same post-autoreset edit pattern as ``SpinUpRotors``):
    position and velocity are set to the spline state at ``traj_t`` (velocity scaled by the
    per-slot speed multiplier so it matches the PID's feedforward), ``last_drone_pos`` is set to
    the spawn position so ``_update_target_gates`` doesn't see a phantom pad->mid-track gate
    crossing on the next step, and ``n_gates_passed`` is advanced to match the gate the spline
    approaches at ``traj_t`` (forward-biased by ``GATE_TIME_EPS``, see there) so gate bookkeeping
    and the competition-reward rank/segment-lead/victory terms treat the opponent as genuinely
    ahead. The reset-randomized attitude and the pad ``takeoff_pos`` are deliberately kept.

    ``pid.gate_times`` has exactly ``n_gates`` entries (one crossing time per physical gate along
    the single-pass, non-looping spline) and is built from ``config.env.track.gates`` in raw list
    order -- so the searchsorted result below only lines up with ``n_gates_passed`` when
    ``track.gate_order`` is the plain identity ``[1, 2, ..., n_gates]`` (see the assertion at the
    ``build_trajectory_pid`` call site in ``ippo.py``). A permuted or repeating gate_order would
    desync this opponent's real (fixed) flight path from the bookkeeping's expected target gate.

    Call this right *after* the env step in which the NEXT_STEP autoreset fired (masked by the
    pre-step ``marked_for_reset`` flags): the reset has just restored pad spawn state, and this
    edit moves the opponent before its next action is applied. The observation returned by that
    reset step still shows the opponent on the pad for exactly one frame; everything is consistent
    from the following step. (Teleporting before the step is impossible -- the autoreset inside it
    would overwrite the edit -- and recomputing the observation would require re-running the whole
    obs wrapper chain.)

    Args:
        env: The (wrapped) vectorized multi-drone racing env pytree.
        pid: The scripted opponent controller; must be built with ``gates`` (``gate_times`` set).
        traj_t: Nominal virtual trajectory time per opponent slot, ``(n_envs, n_opponents)``.
        traj_speed: Per-slot speed multiplier (scales the spawn velocity), same shape.
        mask: Boolean ``(n_envs, n_opponents)`` mask of slots to teleport.
    """
    assert pid.gate_times is not None, (
        "teleport_opponents needs the PID's gate_times -- build the TrajectoryPID with gates=..."
    )
    data = env.unwrapped.data
    states = data.sim_data.states
    pos, vel_dtau = pid._spline(traj_t)  # (E, k, 3) each
    vel = vel_dtau * traj_speed[..., None]
    n_gates_passed = jnp.searchsorted(pid.gate_times, traj_t + GATE_TIME_EPS, side="right")
    m = mask[..., None]
    new_states = states.replace(
        pos=states.pos.at[:, 1:].set(jnp.where(m, pos, states.pos[:, 1:])),
        vel=states.vel.at[:, 1:].set(jnp.where(m, vel, states.vel[:, 1:])),
    )
    new_data = data.replace(
        sim_data=data.sim_data.replace(states=new_states),
        n_gates_passed=data.n_gates_passed.at[:, 1:].set(
            jnp.where(
                mask,
                n_gates_passed.astype(data.n_gates_passed.dtype),
                data.n_gates_passed[:, 1:],
            )
        ),
        last_drone_pos=data.last_drone_pos.at[:, 1:].set(
            jnp.where(m, pos, data.last_drone_pos[:, 1:])
        ),
    )
    return Wrapper.recursive_replace(env, data=new_data)

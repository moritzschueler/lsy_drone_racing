"""Host-side (numpy) twin of the scripted PID opponent used in multi-agent RL training.

``lsy_drone_racing.rl.wrappers.trajectory_opponent.TrajectoryPID`` flies the scripted opponent
inside the jitted training rollout: a cascaded position/velocity PID onto a cubic waypoint spline
that is traversed once at a per-episode speed multiplier, then held. This module runs that exact
control law as an ordinary ``Controller``, so the same opponent can be raced in ``sim.py`` /
``multi_sim.py`` / deployment, e.g.::

    pixi run python scripts/multi_sim.py --config rl_multi_level0.toml \
        --controllers "rl_multi_agent_racing_controller.py,pid_opponent_controller.py"

The waypoints, spline duration and PID gains are imported from the training module rather than
copied, so the two cannot drift apart. Differences from the training-time opponent, all forced by
running outside the batched env:

* The spline's first knot is this drone's actual reset position (``obs["pos"]``), where training
  bakes in the nominal ``config.env.track.drones[1]["pos"]``. Identical on an unrandomized track.
* Training samples the speed multiplier per episode from ``U(opponent_pid_speed_min,
  opponent_pid_speed_max)`` (0.6 to 1.6). A controller has no such episode sampler, so ``speed``
  defaults to the nominal 1.0 and is set per run via the config (see :data:`DEFAULT_SPEED`).
* Training's mid-track "headstart" *teleports* the opponent onto the spline at ``t0``. Nothing can
  teleport a real drone, so ``start_frac`` here only offsets the virtual clock and the drone flies
  from the pad to catch up. It is off by default; see :data:`DEFAULT_START_FRAC`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from drone_models.core import load_params
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control import Controller
from lsy_drone_racing.envs.race_core import build_action_space
from lsy_drone_racing.rl.wrappers.trajectory_opponent import (
    DEFAULT_KD,
    DEFAULT_KI,
    DEFAULT_KI_RANGE,
    DEFAULT_KP,
    DEFAULT_T_TOTAL,
    DEFAULT_WAYPOINTS,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

# Trajectory time-scale multiplier: 1.0 == the nominal DEFAULT_T_TOTAL-second single pass, the
# centre of the [0.6, 1.6] range training samples per episode. Override per run with
# ``[controller] pid_speed`` in the track config.
DEFAULT_SPEED = 1.0
# Virtual-clock headstart as a fraction of the (speed-scaled) trajectory duration; see the module
# docstring for why this is not the same thing as training's teleport. Override per run with
# ``[controller] pid_start_frac``.
DEFAULT_START_FRAC = 0.0

_G = 9.81


class PIDOpponentController(Controller):
    """Scripted spline-following PID opponent, matching the multi-agent training opponent."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """Fit the waypoint spline and prepare the PID.

        Args:
            obs: The initial observation. Batched (one row per drone) in multi-drone races, where
                ``info["rank"]`` selects this drone's row.
            info: Additional reset information. ``info["rank"]`` is this drone's index in a
                multi-drone race, and absent in a single-drone race.
            config: The race configuration; provides the step frequency, drone model (mass and
                action bounds) and the optional ``[controller]`` overrides.
        """
        super().__init__(obs, info, config)
        self.rank = int(info["rank"]) if "rank" in info else None
        obs0 = self._slice(obs)

        self._freq = config.env.freq
        self.drone_mass = load_params(config.sim.physics, config.sim.drone_model)["mass"]

        self.kp = np.asarray(DEFAULT_KP)
        self.ki = np.asarray(DEFAULT_KI)
        self.kd = np.asarray(DEFAULT_KD)
        self.ki_range = np.asarray(DEFAULT_KI_RANGE)
        self.i_error = np.zeros(3)

        controller_cfg = config.get("controller", {})
        self._speed = DEFAULT_SPEED
        self._start_frac = DEFAULT_START_FRAC
        assert self._speed > 0.0, f"pid_speed must be positive, got {self._speed}"
        assert 0.0 <= self._start_frac < 1.0, (
            f"pid_start_frac must be in [0, 1), got {self._start_frac}"
        )

        # Multi-drone configs (multi_sim.py) put control_mode under the per-drone env.kwargs[rank]
        # table instead of a single top-level env.control_mode.
        kwargs = config.env.get("kwargs") if hasattr(config.env, "get") else None
        if self.rank is not None and kwargs is not None:
            control_mode = kwargs[self.rank]["control_mode"]
        else:
            control_mode = config.env.control_mode
        assert control_mode == "attitude", (
            "the scripted PID opponent emits physical roll/pitch/yaw/thrust setpoints and needs "
            f"control_mode='attitude', got '{control_mode}'."
        )
        action_space = build_action_space(control_mode, config.sim.drone_model)
        self._action_low = np.asarray(action_space.low)
        self._action_high = np.asarray(action_space.high)

        # Nominal (speed == 1) trajectory time axis, exactly as in TrajectoryPID: the drone's own
        # spawn is the first knot, and virtual time advances at `speed` per second of wall clock.
        self._t_total = DEFAULT_T_TOTAL
        waypoints = np.concatenate([np.asarray(obs0["pos"], dtype=np.float64)[None], DEFAULT_WAYPOINTS])
        t = np.linspace(0, self._t_total, len(waypoints))
        self._des_pos_spline = CubicSpline(t, waypoints)
        self._des_vel_spline = self._des_pos_spline.derivative()

        self._t = self._start_frac * self._t_total
        self._finished = False

    def _slice(self, obs: dict[str, NDArray[np.floating]]) -> dict[str, NDArray[np.floating]]:
        """This drone's row of a batched multi-drone observation (a no-op when unbatched)."""
        if self.rank is None:
            return obs
        return {k: np.asarray(v)[self.rank] for k, v in obs.items()}

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Compute the next desired roll/pitch/yaw and collective thrust.

        Args:
            obs: The current observation, batched over drones in a multi-drone race.
            info: Optional additional information as a dictionary.

        Returns:
            [r_des, p_des, y_des, t_des] as a numpy array.
        """
        obs = self._slice(obs)
        t = min(self._t, self._t_total)
        if t >= self._t_total:  # Single pass done -- hold the final waypoint
            self._finished = True

        des_pos = self._des_pos_spline(t)
        # The spline differentiates w.r.t. its own nominal time axis tau; the real-world velocity
        # target is d(pos)/dtau * dtau/dt, and dtau/dt is exactly the speed multiplier.
        des_vel = self._des_vel_spline(t) * self._speed
        des_yaw = 0.0

        pos_error = des_pos - obs["pos"]
        vel_error = des_vel - obs["vel"]

        self.i_error = np.clip(
            self.i_error + pos_error * (1 / self._freq), -self.ki_range, self.ki_range
        )

        target_thrust = self.kp * pos_error + self.ki * self.i_error + self.kd * vel_error
        target_thrust[2] += self.drone_mass * _G

        z_axis = R.from_quat(obs["quat"]).as_matrix()[:, 2]
        thrust_desired = target_thrust.dot(z_axis)

        z_axis_desired = target_thrust / np.linalg.norm(target_thrust)
        x_c_des = np.array([np.cos(des_yaw), np.sin(des_yaw), 0.0])
        y_axis_desired = np.cross(z_axis_desired, x_c_des)
        y_axis_desired /= np.linalg.norm(y_axis_desired)
        x_axis_desired = np.cross(y_axis_desired, z_axis_desired)

        R_desired = np.vstack([x_axis_desired, y_axis_desired, z_axis_desired]).T
        euler_desired = R.from_matrix(R_desired).as_euler("xyz", degrees=False)

        action = np.concatenate([euler_desired, [thrust_desired]])
        # Training feeds this action through NormalizeActions' inverse and back, which saturates it
        # at the physical action-space bounds. Clip here so the flown command matches.
        return np.clip(action, self._action_low, self._action_high).astype(np.float32)

    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        """Advance the virtual trajectory clock.

        Returns:
            True if the controller is finished, False otherwise.
        """
        self._t += self._speed / self._freq
        return self._finished

    def episode_callback(self):
        """Reset the internal state."""
        self.i_error[:] = 0
        self._t = self._start_frac * self._t_total
        self._finished = False

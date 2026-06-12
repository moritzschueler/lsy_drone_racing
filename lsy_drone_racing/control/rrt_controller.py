"""RRT* based drone racing controller."""

import threading
from typing import Any

import numpy as np
from crazyflow.sim.visualize import draw_line, draw_points
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control import Controller
from lsy_drone_racing.planning.rrt_star_planner import plan_rrt_star_path


class PIDTracker:
    """Per-axis PID that outputs a correction in that axis."""

    def __init__(
        self, kp: float, ki: float, kd: float, max_integral: float = 2.0, max_output: float = 5.0
    ):
        """Initialize the PID tracker with gains and limits."""
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.max_integral = max_integral
        self.max_output = max_output
        self._integral = np.zeros(3)
        self._last_error = np.zeros(3)
        self._first = True

    def reset(self) -> None:
        """Reset integrator and derivative state."""
        self._integral = np.zeros(3)
        self._last_error = np.zeros(3)
        self._first = True

    def update(self, error: np.ndarray, dt: float) -> np.ndarray:
        """Compute PID output given the current error and timestep."""
        self._integral += error * dt
        self._integral = np.clip(self._integral, -self.max_integral, self.max_integral)
        if self._first:
            derivative = np.zeros(3)
            self._first = False
        else:
            derivative = (error - self._last_error) / max(dt, 1e-6)
        self._last_error = error.copy()
        output = self.kp * error + self.ki * self._integral + self.kd * derivative
        return np.clip(output, -self.max_output, self.max_output)


class RrtController(Controller):
    """Drone racing controller using RRT* path planning and a cubic spline trajectory."""

    def __init__(self, obs: dict, info: dict, config: Any):
        """Initialize the RRT* controller, build the initial spline trajectory."""
        super().__init__(obs, info, config)

        self._freq = config.env.freq
        self._tick = 0
        self._finished = False
        self.dt = 1.0 / config.env.freq

        self.cruise_speed = 0.4
        self.look_ahead_dist = 0.0

        self.current_s = 0.0
        self._arc_length = 1.0
        self._t_total = 1.0

        self.obstacle_radius = 0.15
        self.detour_margin = 0.35
        self.gate_offset = 0.35
        self.gate_half_opening = 0.22
        self.max_obstacle_dist = 2
        self.rrt_max_time = 1.0  # per segment; runs in background so keeps sim real-time

        self._gate_corners = []
        self._pre_gate_waypoints = np.array([])
        self._gate_center_waypoints = np.array([])
        self._post_gate_waypoints = np.array([])

        # PID controllers - copied from A* controller
        self.pid_pos = PIDTracker(0.5, 0.01, 0.40, max_integral=1.0, max_output=3.0)
        self.pid_vel = PIDTracker(0.25, 0.01, 0.1, max_integral=0.5, max_output=2.0)
        self.pid_acc = PIDTracker(0.01, 0.00, 0.0, max_integral=0.2, max_output=1.0)

        self._last_gates_pos = None
        self._last_gates_quat = None
        self._last_obstacles_pos = None
        self._last_target_gate = -2

        # Background planning state.  _next_plan is written by the bg thread and read by the main
        # thread; CPython's GIL makes a single attribute assignment atomic, so no explicit lock is
        # needed.  _plan_gen lets us discard results from superseded planning runs.
        self._next_plan: np.ndarray | None = None
        self._plan_gen: int = 0
        self._bg_thread: threading.Thread | None = None

        # Build an immediate straight-line fallback so the drone can start flying right away,
        # then kick off RRT* in the background.
        self._build_fallback_spline(obs)
        self._cache_state(obs)
        self._start_bg_plan(obs)

    def _build_arc_length_table(self, n_samples: int = 500) -> None:
        """Build a lookup table mapping arc length to spline parameter t."""
        t_samp = np.linspace(0, self._t_total, n_samples)
        pts = self.spline(t_samp)
        seg_lens = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        arc_s = np.concatenate([[0.0], np.cumsum(seg_lens)])

        self._arc_length = float(arc_s[-1])
        self._s_to_t = CubicSpline(arc_s, t_samp)

    def _s_to_spline(self, s: float) -> float:
        """Convert an arc-length value to a spline parameter t."""
        s = float(np.clip(s, 0.0, self._arc_length))
        return float(np.clip(float(self._s_to_t(s)), 0.0, self._t_total))

    def _update_milestones(self, remaining_pos: np.ndarray, remaining_quat: np.ndarray) -> None:
        """Recompute the pre/center/post gate visualization waypoints."""
        pre, center, post = [], [], []
        for i in range(len(remaining_pos)):
            normal = R.from_quat(remaining_quat[i]).apply([1, 0, 0])
            pre.append(remaining_pos[i] - normal * self.gate_offset)
            center.append(remaining_pos[i].copy())
            post.append(remaining_pos[i] + normal * self.gate_offset)
        self._pre_gate_waypoints = np.array(pre) if pre else np.array([])
        self._gate_center_waypoints = np.array(center) if center else np.array([])
        self._post_gate_waypoints = np.array(post) if post else np.array([])

    def _apply_waypoints(self, waypoints: np.ndarray, obs: dict) -> None:
        """Fit a cubic spline to waypoints and rebuild the arc-length table."""
        if len(waypoints) < 2:
            waypoints = np.vstack([waypoints, waypoints + [0, 0, 0.01]])
        t_steps = np.arange(len(waypoints))
        self._t_total = float(len(waypoints) - 1)
        vel = np.array(obs.get("vel", [0.0, 0.0, 0.0]))
        speed = np.linalg.norm(vel)
        if speed > 0.1:
            tangent = (vel / speed) * np.linalg.norm(waypoints[1] - waypoints[0])
            self.spline = CubicSpline(t_steps, waypoints, bc_type=((1, tangent), "natural"))
        else:
            self.spline = CubicSpline(t_steps, waypoints, bc_type="natural")
        self._build_arc_length_table()
        self._visual_trajectory = self.spline(np.linspace(0, self._t_total, 800))

    def _reanchor(self, pos: np.ndarray) -> None:
        """Set current_s to the closest point on the current spline to pos."""
        limit = min(self._arc_length, self.cruise_speed * 3.0)
        s_search = np.linspace(0.0, limit, 200)
        pts = self.spline(np.array([self._s_to_spline(s) for s in s_search]))
        self.current_s = float(s_search[np.linalg.norm(pts - pos, axis=1).argmin()])

    def _build_fallback_spline(self, obs: dict) -> None:
        """Build an immediate straight-line spline through remaining gates (no planning)."""
        start_pos = np.array(obs["pos"])
        gates_pos = np.array(obs["gates_pos"])
        gates_quat = np.array(obs["gates_quat"])
        target_gate = int(obs["target_gate"])
        remaining_pos = gates_pos[target_gate:]
        remaining_quat = gates_quat[target_gate:]
        self._update_milestones(remaining_pos, remaining_quat)
        if len(remaining_pos) == 0:
            waypoints = start_pos.reshape(1, 3)
        else:
            waypoints = np.vstack([start_pos, remaining_pos])
        self._apply_waypoints(waypoints, obs)

    def _start_bg_plan(self, obs: dict) -> None:
        """Snapshot current state and launch RRT* in a daemon thread."""
        self._plan_gen += 1
        gen = self._plan_gen
        obs_snap = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in obs.items()}

        def _run() -> None:
            try:
                start_pos = np.array(obs_snap["pos"])
                gates_pos = np.array(obs_snap["gates_pos"])
                gates_quat = np.array(obs_snap["gates_quat"])
                obstacles = np.array(obs_snap["obstacles_pos"])
                target_gate = int(obs_snap["target_gate"])
                remaining_pos = gates_pos[target_gate:]
                remaining_quat = gates_quat[target_gate:]
                if len(remaining_pos) == 0:
                    return
                waypoints = plan_rrt_star_path(
                    start_pos, remaining_pos, remaining_quat, obstacles, self
                )
                # Only stage if this is still the latest planning request (GIL makes this atomic).
                if gen == self._plan_gen:
                    self._next_plan = waypoints
            except Exception:  # noqa: BLE001
                pass

        self._bg_thread = threading.Thread(target=_run, daemon=True)
        self._bg_thread.start()

    def _state_changed(self, obs: dict) -> bool:
        """Return True if the relevant environment state has changed enough to warrant a replan."""
        if self._last_gates_pos is None:
            return True

        target_gate = int(obs["target_gate"])

        if target_gate >= 0:
            if not np.allclose(
                obs["gates_pos"][target_gate], self._last_gates_pos[target_gate], atol=1e-2
            ):
                return True

        if target_gate >= 0:
            target_gate_pos = obs["gates_pos"][target_gate]
            for i, obs_pos in enumerate(obs["obstacles_pos"]):
                if np.linalg.norm(target_gate_pos - obs_pos) > self.max_obstacle_dist:
                    continue  # not near the target gate — ignore
                if not np.allclose(obs_pos, self._last_obstacles_pos[i], atol=0.001):
                    return True

        return False

    def _cache_state(self, obs: dict) -> None:
        """Cache the current gate and obstacle state for change detection."""
        self._last_gates_pos = obs["gates_pos"].copy()
        self._last_gates_quat = obs["gates_quat"].copy()
        self._last_obstacles_pos = obs["obstacles_pos"].copy()
        self._last_target_gate = int(obs["target_gate"])

    def compute_control(self, obs: dict, info: dict | None = None) -> np.ndarray:
        """Compute the control action for the current timestep."""
        pos = np.array(obs["pos"])

        if self._state_changed(obs):
            self._cache_state(obs)
            self._build_fallback_spline(obs)
            self._reanchor(pos)
            self._start_bg_plan(obs)

        # Swap in the RRT* result as soon as the background thread finishes.
        pending = self._next_plan
        if pending is not None:
            self._next_plan = None
            self._apply_waypoints(pending, obs)
            self._reanchor(pos)

        target_s = min(self.current_s + self.look_ahead_dist, self._arc_length)
        t = self._s_to_spline(target_s)

        ref_pos = self.spline(t)
        spline_tangent = self.spline(t, 1)
        ds_dt = max(np.linalg.norm(spline_tangent), 1e-6)
        dt_ds = 1.0 / ds_dt

        ref_vel = spline_tangent * dt_ds * self.cruise_speed
        ref_acc = self.spline(t, 2) * dt_ds**2 * self.cruise_speed**2

        actual_vel = np.array(obs["vel"])

        pos_correction = self.pid_pos.update(ref_pos - pos, self.dt)
        vel_correction = self.pid_vel.update(ref_vel - actual_vel, self.dt)
        acc_correction = self.pid_acc.update(ref_acc, self.dt)

        return np.array(
            [
                *(ref_pos + pos_correction),
                *(ref_vel + vel_correction),
                *(ref_acc + acc_correction),
                0,
                0,
                0,
                0,
            ],
            dtype=np.float32,
        )

    def step_callback(
        self,
        action: np.ndarray,
        obs: dict,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        """Update the arc-length progress and check for episode completion."""
        drone_pos = np.array(obs["pos"])

        s_search = np.linspace(
            max(0.0, self.current_s - 0.2),
            min(self._arc_length, self.current_s + self.cruise_speed * 0.5),
            60,
        )
        t_search = np.array([self._s_to_spline(s) for s in s_search])
        pts = self.spline(t_search)
        dists = np.linalg.norm(pts - drone_pos, axis=1)
        best_s = float(s_search[np.argmin(dists)])

        time_advance = self.cruise_speed * self.dt
        self.current_s = float(
            np.clip(max(self.current_s + time_advance * 0.5, best_s), 0.0, self._arc_length)
        )

        if int(obs["target_gate"]) == -1:
            self._finished = True

        self._tick += 1
        return self._finished

    def episode_reset(self) -> None:
        """Reset controller state at the start of a new episode."""
        self._tick = 0
        self._finished = False
        self.current_s = 0.0
        self.pid_pos.reset()
        self.pid_vel.reset()
        self.pid_acc.reset()
        self._last_gates_pos = None
        self._last_gates_quat = None
        self._last_obstacles_pos = None
        self._last_target_gate = -2
        self._next_plan = None
        self._plan_gen += 1  # invalidate any in-flight background plan

    def render_callback(self, sim: Any) -> None:
        """Draw the planned trajectory and current setpoint in the visualizer."""
        draw_line(sim, self._visual_trajectory, rgba=(0.0, 0.5, 1.0, 1.0))

        # Disabled: causes apparent collisions with the drone (markers are visual-only but
        # visually interfere; re-enable for debugging by uncommenting the two lines below).
        # t_now = self._s_to_spline(min(self.current_s + self.look_ahead_dist, self._arc_length))
        # draw_points(sim, self.spline(t_now).reshape(1, -1), rgba=(0.5, 0.5, 1.0, 1.0), size=0.02)

        # Visualize waypoint milestones
        #if len(self._pre_gate_waypoints) > 0:
        #    draw_points(sim, self._pre_gate_waypoints, rgba=(0.0, 0.0, 1.0, 1.0), size=0.012)
        #    draw_points(sim, self._gate_center_waypoints, rgba=(0.0, 1.0, 1.0, 1.0), size=0.015)
        #    draw_points(sim, self._post_gate_waypoints, rgba=(1.0, 1.0, 0.0, 1.0), size=0.012)
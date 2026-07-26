"""This module implements an AttitudeController for quadrotor control.

It utilizes the collective thrust interface for drone control to compute control commands based on
current state observations and desired waypoints. The attitude control is handled by computing a
PID control law for position tracking, incorporating gravity compensation in thrust calculations.

The waypoints are generated using cubic spline interpolation from a set of predefined waypoints.
Note that the trajectory uses pre-defined waypoints instead of dynamically generating a good path.
"""

from __future__ import annotations  # Python 3.10 type hints

import math
from typing import TYPE_CHECKING

import numpy as np
from crazyflow.sim.visualize import draw_line, draw_points
from drone_models.core import load_params
from scipy.interpolate import CubicSpline, make_interp_spline
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control import Controller

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from crazyflow import Sim


class AttitudeController_1(Controller):
    """Trajectory following controller using the collective thrust and attitude interface."""

    def create_waypoints(self, obs: dict[str, NDArray[np.floating]]):
        """Generate waypoints for racing through gates.

        Args:
            obs: The current observation containing gate positions and orientations.
        """
        gate_in_offset = 0.23  # metres along gate normal for entry/exit points — TODO: tune
        gate_in_offset_prev = 0.10  # metres along vector from previous waypoint
        gate_out_offset = 0.23  # metres along gate normal for entry/exit points — TODO: tune
        gate_out_offset_next = 0.10  # metres along vector from previous waypoint

        dip_degree = 120

        point_at_obstacle = [True, True, True, False]
        obstacle_ind = np.array([1,0,3,0])
        offset_at_obstacle = np.array([[0.17,-0.17,0.1],[0.2,-0.2,-0.1],[0.1,0.2,0.2],[0,0,0]])

        waypoints = [self._start_pos]

        takeoff = self._start_pos.copy()
        takeoff += [0.6, -0.1, 0.3]  # lift to 0.5m before going anywhere
        waypoints.append(takeoff)

        gates_pos = obs["gates_pos"]    # shape (n_gates, 3)
        gates_quat = obs["gates_quat"]  # shape (n_gates, 4), format [x, y, z, w]

        for i, (pos, quat) in enumerate(zip(gates_pos, gates_quat)):
            normal = R.from_quat(quat).apply([1.0, 0.0, 0.0])
            vec_prev_to_gate = pos -waypoints[-1]
            vec_prev_to_gate_norm = vec_prev_to_gate / np.linalg.norm(vec_prev_to_gate)

            if i + 1 < len(gates_pos):
                vec_gate_to_next_gate = gates_pos[i+1] - pos
                vec_gate_to_next_gate_norm = vec_gate_to_next_gate / np.linalg.norm(vec_gate_to_next_gate)
            else:
                vec_gate_to_next_gate = vec_prev_to_gate    #continue in same direction as bevore
                vec_gate_to_next_gate_norm = vec_prev_to_gate_norm

            #When to perform a dip
            cos_theta = np.dot(normal,vec_gate_to_next_gate_norm) # bouth are already normed
            theta_dec = np.degrees(np.arccos(cos_theta))

            gate_in_dir_vec = gate_in_offset_prev * vec_prev_to_gate_norm
            gate_out_dir_vec = gate_out_offset_next * vec_gate_to_next_gate_norm

            length_normal_in = np.dot(gate_in_dir_vec, normal)
            length_normal_out = np.dot(gate_out_dir_vec, normal)

            add_normal_in = length_normal_in - gate_in_offset
            add_normal_out = gate_out_offset - length_normal_out

            entry = pos - gate_in_dir_vec + add_normal_in * normal
            if theta_dec < dip_degree:
                exit_ = pos + gate_out_dir_vec + add_normal_out * normal
            else:
                add_normal_out = gate_out_offset + length_normal_out        #alternative computation since distance is opposide direction if theta_dec>90
                exit_ = pos + gate_out_dir_vec - add_normal_out * normal

            waypoints.append(entry)
            waypoints.append(pos)
            waypoints.append(exit_)

            if point_at_obstacle[i]:
                obs_pos = obs["obstacles_pos"][obstacle_ind[i]].copy()
                obs_pos[2] = exit_[2]  # change z pos to previous waypoint

                obs_pos = obs_pos + offset_at_obstacle[i]

                waypoints.append(obs_pos)


        waypoints = np.array(waypoints)  # shape (1 + n_gates*3, 3)

        self._waypoints = waypoints

    def create_spline_old(self):
        """Create spline interpolation for waypoints."""
        self._t_total = 10  # s
        t = np.linspace(0, self._t_total, len(self._waypoints))

        cubic_spline = True
        if cubic_spline:
            self._des_pos_spline = CubicSpline(t, self._waypoints)
            self._des_vel_spline = self._des_pos_spline.derivative()
        else:
            # k=3 → cubic B-spline
            bspline = make_interp_spline(t, self._waypoints, k=3)
            self._des_pos_spline = bspline
            self._des_vel_spline = bspline.derivative(1)

    def create_spline(self):
        """Create spline interpolation for waypoints."""
        #self._t_total = 7.8  # s Used for submission
        self._t_total = 11

        # Distance-based timing
        distances = np.linalg.norm(np.diff(self._waypoints, axis=0), axis=1)
        cumulative_dist = np.insert(np.cumsum(distances), 0, 0)
        t = self._t_total * cumulative_dist / cumulative_dist[-1]

        cubic_spline = True
        if cubic_spline:
            # Spline with boundary conditions
            self._des_pos_spline = CubicSpline(t, self._waypoints) #, bc_type=((1, np.zeros(3)), (1, np.zeros(3))))
            self._des_vel_spline = self._des_pos_spline.derivative()
        else:
            # k=3 → cubic B-spline
            bspline = make_interp_spline(t, self._waypoints, k=3)
            self._des_pos_spline = bspline
            self._des_vel_spline = bspline.derivative(1)

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """Initialize the attitude controller.

        Args:
            obs: The initial observation of the environment's state. See the environment's
                observation space for details.
            info: Additional environment information from the reset.
            config: The configuration of the environment.
        """
        super().__init__(obs, info, config)
        self._freq = config.env.freq

        # For more info on the models, check out https://github.com/utiasDSL/drone-models
        drone_params = load_params(config.sim.physics, config.sim.drone_model)
        self.drone_mass = drone_params["mass"]

        PID_var = 2
        if PID_var == 0:
            # original
            self.kp = np.array([0.4, 0.4, 1.25])
            self.ki = np.array([0.05, 0.05, 0.05])
            self.kd = np.array([0.2, 0.2, 0.4])
            self.ki_range = np.array([2.0, 2.0, 0.4])
        elif PID_var == 1:
            # good
            self.kp = np.array([0.8, 0.8, 2.5])
            self.ki = np.array([0.05, 0.05, 0.05])
            self.kd = np.array([0.4, 0.4, 0.8])
            self.ki_range = np.array([2.0, 2.0, 0.4])
        else:
            # without I (best)
            self.kp = np.array([0.7, 0.7, 2.7])
            self.ki = np.array([0.00, 0.00, 0.00])
            self.kd = np.array([0.4, 0.4, 0.8])
            self.ki_range = np.array([2.0, 2.0, 0.4])

        self.i_error = np.zeros(3)
        self.g = 9.81

        self._start_pos = obs["pos"].copy()  # start at current drone position

        self.create_waypoints(obs)
        self.create_spline()


        self._tick = 0
        self._finished = False



    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Compute the next desired collective thrust and roll/pitch/yaw of the drone.

        Args:
            obs: The current observation of the environment. See the environment's observation space
                for details.
            info: Optional additional information as a dictionary.

        Returns:
            The orientation as roll, pitch, yaw angles, and the collective thrust
            [r_des, p_des, y_des, t_des] as a numpy array.
        """
        t = min(self._tick / self._freq, self._t_total)
        if t >= self._t_total:  # Maximum duration reached
            self._finished = True

        #Update splines with current observations
        self.create_waypoints(obs)
        self.create_spline()

        des_pos = self._des_pos_spline(t)
        des_vel = self._des_vel_spline(t)
        des_yaw = 0.0

        # Calculate the deviations from the desired trajectory
        pos_error = des_pos - obs["pos"]
        vel_error = des_vel - obs["vel"]

        # Update integral error
        self.i_error += pos_error * (1 / self._freq)
        self.i_error = np.clip(self.i_error, -self.ki_range, self.ki_range)

        # Compute target thrust
        target_thrust = np.zeros(3)
        target_thrust += self.kp * pos_error
        target_thrust += self.ki * self.i_error
        target_thrust += self.kd * vel_error
        target_thrust[2] += self.drone_mass * self.g

        # Update z_axis to the current orientation of the drone
        z_axis = R.from_quat(obs["quat"]).as_matrix()[:, 2]

        # update current thrust
        thrust_desired = target_thrust.dot(z_axis)

        # update z_axis_desired
        z_axis_desired = target_thrust / np.linalg.norm(target_thrust)
        x_c_des = np.array([math.cos(des_yaw), math.sin(des_yaw), 0.0])
        y_axis_desired = np.cross(z_axis_desired, x_c_des)
        y_axis_desired /= np.linalg.norm(y_axis_desired)
        x_axis_desired = np.cross(y_axis_desired, z_axis_desired)

        R_desired = np.vstack([x_axis_desired, y_axis_desired, z_axis_desired]).T
        euler_desired = R.from_matrix(R_desired).as_euler("xyz", degrees=False)

        action = np.concatenate([euler_desired, [thrust_desired]], dtype=np.float32)

        return action

    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        """Increment the tick counter.

        Returns:
            True if the controller is finished, False otherwise.
        """
        self._tick += 1
        return self._finished

    def episode_callback(self):
        """Reset the internal state."""
        self.i_error[:] = 0
        self._tick = 0

    def render_callback(self, sim: Sim):
        """Visualize the desired trajectory and the current setpoint."""
        setpoint = self._des_pos_spline(self._tick / self._freq).reshape(1, -1)
        draw_points(sim, setpoint, rgba=(1.0, 0.0, 0.0, 1.0), size=0.02)
        trajectory = self._des_pos_spline(np.linspace(0, self._t_total, 100))
        draw_line(sim, trajectory, rgba=(0.0, 1.0, 0.0, 1.0))

        draw_points(sim, self._waypoints, rgba=(0.0, 0.0, 1.0, 1.0), size=0.03)

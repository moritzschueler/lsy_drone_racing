"""A* based path planning for drone racing."""

from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control.astar import astar_3d


def plan_astar_path(
    start_pos: np.ndarray,
    gates_pos: np.ndarray,
    gates_quat: np.ndarray,
    obstacles: np.ndarray,
    config: Any,
) -> np.ndarray:
    """Plan a collision-free path through all gates using A* algorithm.
    
    Args:
        start_pos: Starting position (3,)
        gates_pos: Gate positions (N, 3)
        gates_quat: Gate quaternions (N, 4)
        obstacles: Obstacle positions (M, 3)
        config: Configuration object with planning parameters
        
    Returns:
        Waypoints array (K, 3) for trajectory planning
    """
    # Configuration parameters
    gate_offset = getattr(config, 'gate_offset', 0.35)
    detour_margin = getattr(config, 'detour_margin', 0.35)
    voxel_size = getattr(config, 'voxel_size', 0.1)
    
    # Generate virtual obstacle points around gate frames
    virtual_obs = _gate_frame_obstacles(gates_pos, gates_quat, gate_offset)
    
    # Sample obstacle positions (vertical columns)
    sampled_rods = []
    ROD_MAX_HEIGHT = 2.0
    ROD_STEP = 0.20
    for rod_pos in obstacles:
        zs = np.arange(0.0, ROD_MAX_HEIGHT + ROD_STEP, ROD_STEP)
        for z in zs:
            sampled_rods.append(np.array([rod_pos[0], rod_pos[1], z]))
    
    all_obstacles = virtual_obs + sampled_rods
    
    # Generate pre/post gate waypoints
    raw_pre_gate_waypoints = []
    raw_waypoints = []
    raw_post_gate_waypoints = []
    gate_normals = []
    
    for i in range(len(gates_pos)):
        r = R.from_quat(gates_quat[i])
        gate_normal = r.apply([1, 0, 0])
        
        pre_wp = gates_pos[i] - gate_normal * gate_offset
        post_wp = gates_pos[i] + gate_normal * gate_offset
        
        raw_pre_gate_waypoints.append(pre_wp)
        raw_waypoints.append(gates_pos[i].copy())
        raw_post_gate_waypoints.append(post_wp)
        gate_normals.append(gate_normal)
    
    # Plan path using A*
    final_waypoints = []
    
    # Path from start to first gate
    for j, point in enumerate(
        astar_3d(
            start=start_pos,
            goal=raw_pre_gate_waypoints[0],
            obstacles=all_obstacles,
            voxel_size=voxel_size,
            obstacle_clearance=detour_margin,
            gate_normal=(None, gate_normals[0]),
        )
    ):
        if j % 3 == 0:
            final_waypoints.append(point)
    final_waypoints.append(raw_waypoints[0])
    
    # Paths between gates
    for i in range(1, len(gates_pos)):
        start = raw_post_gate_waypoints[i - 1]
        end_pos = raw_pre_gate_waypoints[i]
        for k, point in enumerate(
            astar_3d(
                start=start,
                goal=end_pos,
                obstacles=all_obstacles,
                voxel_size=voxel_size,
                obstacle_clearance=detour_margin,
                gate_normal=(
                    (gate_normals[i - 1], None)
                    if i == len(gates_pos)
                    else (gate_normals[i - 1], gate_normals[i])
                ),
            )
        ):
            if k % 3 == 0:
                final_waypoints.append(point)
        final_waypoints.append(raw_waypoints[i])
    
    final_waypoints.append(raw_waypoints[-1])
    final_waypoints.append(raw_post_gate_waypoints[-1])
    
    return np.vstack(final_waypoints)


def _gate_frame_obstacles(
    gates_pos: np.ndarray, gates_quat: np.ndarray, gate_offset: float
) -> list:
    """Generate virtual obstacle points around gate frames.
    
    Args:
        gates_pos: Gate positions (N, 3)
        gates_quat: Gate quaternions (N, 4)
        gate_offset: Offset distance from gate center
        
    Returns:
        List of virtual obstacle points
    """
    virtual_obs = []
    GATE_INNER_HALF = 0.20
    GATE_OUTER_HALF = 0.4
    
    for i in range(len(gates_pos)):
        r = R.from_quat(gates_quat[i])
        lateral = r.apply([0, 1, 0])
        lateral = np.array([lateral[0], lateral[1], 0.0])
        lateral /= max(np.linalg.norm(lateral), 1e-6)
        
        for half in (GATE_INNER_HALF, GATE_OUTER_HALF):
            corners = []
            for lat_sign in (+1, -1):
                for z_sign in (+1, -1):
                    corner = (
                        gates_pos[i]
                        + lat_sign * half * lateral
                        + np.array([0, 0, z_sign * half])
                    )
                    if half == GATE_OUTER_HALF:
                        corners.append(corner)
                    virtual_obs.append(corner)
            
            if half == GATE_OUTER_HALF:
                edges = [
                    (corners[0], corners[2]),  # top
                    (corners[1], corners[3]),  # bottom
                    (corners[0], corners[1]),  # right
                    (corners[2], corners[3]),  # left
                ]
                for a, b in edges:
                    for t in (0.2, 0.4, 0.6, 0.8):
                        pt = a + t * (b - a)
                        virtual_obs.append(pt)
    
    return virtual_obs

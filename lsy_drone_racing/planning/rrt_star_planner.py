"""RRT* based path planning for drone racing using OMPL."""

from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as R

try:
    import ompl.base as ob
    import ompl.geometric as og
except ImportError:
    raise ImportError("OMPL library is required for RRT* planning. Install it via conda-forge.")


def plan_rrt_star_path(
    start_pos: np.ndarray,
    gates_pos: np.ndarray,
    gates_quat: np.ndarray,
    obstacles: np.ndarray,
    config: Any,
) -> np.ndarray:
    """Plan a collision-free path through all gates using RRT* algorithm.
    
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
    obstacle_radius = getattr(config, 'obstacle_radius', 0.15)
    max_planning_time = getattr(config, 'rrt_max_time', 5.0)
    
    # Sample obstacle columns
    sampled_rods = []
    ROD_MAX_HEIGHT = 2.0
    ROD_STEP = 0.20
    for rod_pos in obstacles:
        zs = np.arange(0.0, ROD_MAX_HEIGHT + ROD_STEP, ROD_STEP)
        for z in zs:
            sampled_rods.append(np.array([rod_pos[0], rod_pos[1], z]))
    
    # For approach/transition segments, exclude gate frame obstacles since we're planning to go
    # through the gates. Only use rod obstacles for transitions
    inter_gate_obstacles = sampled_rods  # Rod obstacles only
    
    # Generate pre/post gate waypoints and collect all targets
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
    
    # Plan full path through all gates
    final_waypoints = []
    
    # Path from start to first gate - use rod obstacles only (not gate frames)
    waypoints_segment = _plan_rrt_star_segment(
        start_pos,
        raw_pre_gate_waypoints[0],
        inter_gate_obstacles,
        obstacle_radius,
        max_planning_time,
    )
    final_waypoints.extend(waypoints_segment)
    
    # Add mandatory gate passage waypoints (no planning - these are fixed waypoints)
    final_waypoints.append(raw_pre_gate_waypoints[0])
    final_waypoints.append(raw_waypoints[0])
    final_waypoints.append(raw_post_gate_waypoints[0])
    
    # Paths between gates
    for i in range(1, len(gates_pos)):
        # Plan from previous post-gate to next pre-gate - use rod obstacles only
        waypoints_segment = _plan_rrt_star_segment(
            raw_post_gate_waypoints[i - 1],
            raw_pre_gate_waypoints[i],
            inter_gate_obstacles,
            obstacle_radius,
            max_planning_time,
        )
        final_waypoints.extend(waypoints_segment)
        
        # Add mandatory gate passage waypoints (no planning - these are fixed waypoints)
        final_waypoints.append(raw_pre_gate_waypoints[i])
        final_waypoints.append(raw_waypoints[i])
        final_waypoints.append(raw_post_gate_waypoints[i])
    
    return np.vstack(final_waypoints)


def _plan_rrt_star_segment(
    start: np.ndarray,
    goal: np.ndarray,
    obstacles: list,
    obstacle_radius: float,
    planning_time: float,
) -> list:
    """Plan a single segment using RRT*.
    
    Args:
        start: Start position (3,)
        goal: Goal position (3,)
        obstacles: List of obstacle positions
        obstacle_radius: Radius of obstacles
        planning_time: Maximum planning time in seconds
        
    Returns:
        List of waypoints (excluding start and goal)
    """
    space = ob.RealVectorStateSpace(3)

    bounds = ob.RealVectorBounds(3)
    all_points_arr = np.array([start, goal] + obstacles) if obstacles else np.vstack([start, goal])
    min_coords = all_points_arr.min(axis=0) - 5.0
    max_coords = all_points_arr.max(axis=0) + 5.0
    for i in range(3):
        bounds.setLow(i, float(min_coords[i]))
        bounds.setHigh(i, float(max_coords[i]))
    space.setBounds(bounds)

    si = ob.SpaceInformation(space)

    # Pre-build obstacle array and squared threshold once; isValid is called ~hundreds of thousands
    # of times per planning call, so avoiding a Python loop here matters a lot.
    obs_arr = np.array(obstacles, dtype=np.float64) if obstacles else np.empty((0, 3))
    min_dist_sq = float((obstacle_radius + 0.05) ** 2)

    class CollisionChecker(ob.StateValidityChecker):
        def isValid(self, state: Any) -> bool:
            pos = np.array([state[0], state[1], state[2]])
            if not obs_arr.size:
                return True
            d = obs_arr - pos
            return not bool(np.any((d * d).sum(axis=1) < min_dist_sq))
    
    # Keep a Python reference to the checker so it isn't GC'd while OMPL holds a C++ pointer to it.
    checker = CollisionChecker(si)
    si.setStateValidityChecker(checker)
    si.setup()

    # allocState() returns an owning Python object; do NOT call si.freeState() explicitly —
    # the binding's destructor will free the C++ memory automatically.
    start_state = si.allocState()
    for i in range(3):
        start_state[i] = start[i]

    goal_state = si.allocState()
    for i in range(3):
        goal_state[i] = goal[i]

    # Check if start and goal states are valid
    if not si.isValid(start_state):
        return [start + (goal - start) * t for t in np.linspace(0.1, 0.9, 5)]

    if not si.isValid(goal_state):
        return [start + (goal - start) * t for t in np.linspace(0.1, 0.9, 5)]

    pdef = ob.ProblemDefinition(si)
    pdef.setStartAndGoalStates(start_state, goal_state)

    # Create RRT* planner
    planner = og.RRTstar(si)
    planner.setProblemDefinition(pdef)
    planner.setup()

    # Solve
    solved = planner.solve(planning_time)

    if not solved:
        return [start + (goal - start) * t for t in np.linspace(0.1, 0.9, 5)]

    # Extract solution path
    solution_path = pdef.getSolutionPath()
    num_states = solution_path.getStateCount()

    waypoints = []
    for i in range(1, num_states - 1):
        state = solution_path.getState(i)
        pos = np.array([float(state[j]) for j in range(3)])
        waypoints.append(pos)

    return (
        waypoints
        if waypoints
        else [start + (goal - start) * t for t in np.linspace(0.1, 0.9, 5)]
    )


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

"""Unit tests for the champion-paper gate distance / progress reward and the speed hinge."""

import jax.numpy as jnp
import numpy as np
import pytest

from lsy_drone_racing.rl.tasks.single_agent_racing import (
    GATE_HALF_EXTENT,
    build_racing_reward,
    gate_opening_distance,
    quadratic_speed_penalty,
)
from lsy_drone_racing.rl.wrappers.reward import action_smoothness_penalty


def _dist(along: float, y: float = 0.0, z: float = 0.0) -> float:
    """Distance from a drone at gate-local offset (along, y, z) to the gate opening.

    The gate sits at the world origin with identity orientation, so the gate frame equals the world
    frame and ``along`` is the gate-local +x (traversal) coordinate: <0 entry side, >0 exit side.
    """
    drone_pos = jnp.array([[[along, y, z]]])  # (E=1, D=1, 3)
    gates_pos = jnp.zeros((1, 1, 3))
    gates_quat = jnp.array([[[0.0, 0.0, 0.0, 1.0]]])  # identity xyzw
    target_gate = jnp.array([[0]])
    d = gate_opening_distance(drone_pos, gates_pos, gates_quat, target_gate, GATE_HALF_EXTENT)
    return float(d[0, 0])


@pytest.mark.unit
def test_distance_is_zero_inside_the_opening():
    """Anywhere in the gate plane within the opening half-extent reads distance 0."""
    assert _dist(0.0) == pytest.approx(0.0, abs=1e-6)
    assert _dist(0.0, y=GATE_HALF_EXTENT, z=GATE_HALF_EXTENT) == pytest.approx(0.0, abs=1e-6)


@pytest.mark.unit
def test_entry_approach_distance_decreases_monotonically():
    """Flying forward toward the opening from the entry side lowers the distance (progress > 0)."""
    alongs = [-2.8, -2.0, -1.0, -0.5, -0.2, -0.05]  # approaching the plane from -x
    dists = [_dist(a) for a in alongs]
    assert np.all(np.diff(dists) < 0), f"distance not decreasing on approach: {dists}"


@pytest.mark.unit
def test_distance_is_unbounded_with_constant_gradient():
    """Unlike the old bounded potential, distance grows linearly with a non-vanishing gradient."""
    assert _dist(-2.8) == pytest.approx(2.8, abs=1e-6)  # straight-on distance == |along|
    # Equal steps in along give equal distance changes near and far from the gate (no vanishing
    # pull).
    near = _dist(-0.5) - _dist(-0.3)
    far = _dist(-2.5) - _dist(-2.3)
    assert near == pytest.approx(far, abs=1e-6)


@pytest.mark.unit
def test_distance_is_symmetric_entry_vs_exit():
    """No exit asymmetry (champion approach): equal |along| reads equal distance on either side."""
    assert _dist(-0.5) == pytest.approx(_dist(+0.5), abs=1e-6)
    assert _dist(-1.3) == pytest.approx(_dist(+1.3), abs=1e-6)


@pytest.mark.unit
def test_lateral_offset_increases_distance_but_is_clamped_in_opening():
    """Beside the frame a larger lateral offset is farther; within the half-extent it is free."""
    assert _dist(-0.5, y=1.2) > _dist(-0.5, y=0.0)
    assert _dist(-0.5, y=GATE_HALF_EXTENT) == pytest.approx(_dist(-0.5, y=0.0), abs=1e-6)


@pytest.mark.unit
def test_build_reward_constructs_for_any_positive_coefficients():
    """Any sensible coefficients build a reward fn.

    The old gate_bonus >= progress_coef coupling is gone (distance-reduction has no full-range
    crossing drop), so the constraint no longer applies.
    """
    assert callable(build_racing_reward(progress_coef=1.0, gate_bonus=10.0))
    assert callable(build_racing_reward(progress_coef=20.0, gate_bonus=2.0))


@pytest.mark.unit
def test_action_smoothness_is_zero_when_command_is_unchanged():
    """No change between consecutive commands -> no penalty."""
    a = jnp.array([[0.3, -0.2, 0.0, 0.5]])  # (n_envs=1, 4) -> per-env (1,) penalty
    assert float(action_smoothness_penalty(a, a, coef=0.01)[0]) == pytest.approx(0.0)


@pytest.mark.unit
def test_action_smoothness_penalizes_squared_change_over_all_dims():
    """Penalty is -coef * ||a - a_prev||^2 summed over the action dims."""
    a_prev = jnp.zeros((1, 4))
    a = jnp.array([[0.1, 0.2, 0.0, 0.0]])  # ||.||^2 = 0.01 + 0.04 = 0.05
    assert float(action_smoothness_penalty(a, a_prev, coef=0.5)[0]) == pytest.approx(-0.025)


@pytest.mark.unit
def test_action_smoothness_uses_the_bounded_action():
    """Out-of-range commands are clipped to [-1, 1] before differencing.

    So an unbounded high-variance policy cannot inflate the penalty -- it saturates at the bounded
    value.
    """
    a_prev = jnp.zeros((1, 4))
    huge = jnp.array([[10.0, 0.0, 0.0, 0.0]])
    clipped = jnp.array([[1.0, 0.0, 0.0, 0.0]])
    assert float(action_smoothness_penalty(huge, a_prev, coef=0.1)[0]) == pytest.approx(
        float(action_smoothness_penalty(clipped, a_prev, coef=0.1)[0])
    )


@pytest.mark.unit
def test_speed_hinge_is_zero_below_threshold_and_grows_quadratically_above():
    """The speed hinge is exactly 0 below the threshold and equals (speed - threshold)^2 above."""
    threshold = 4.0
    below = jnp.array([0.0, 1.0, 2.0, 3.0, 4.0])  # at/below the threshold -> a true free zone
    assert np.all(np.asarray(quadratic_speed_penalty(below, threshold)) == 0.0)
    # Above the threshold it is exactly the squared overshoot (convex, monotone increasing).
    above = jnp.array([5.0, 6.0, 8.0])
    pen = np.asarray(quadratic_speed_penalty(above, threshold))
    assert np.allclose(pen, (np.asarray(above) - threshold) ** 2)  # 1.0, 4.0, 16.0
    assert np.all(np.diff(pen) > 0)  # monotone increasing
    assert np.all(np.diff(np.diff(pen)) > 0)  # convex: steepening with overshoot (quadratic)

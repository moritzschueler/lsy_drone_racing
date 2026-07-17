"""Unit tests for the scripted PID opponent's gate pass times (random mid-track spawn support)."""

import jax.numpy as jnp
import numpy as np
import pytest

from lsy_drone_racing.rl.wrappers.trajectory_opponent import (
    DEFAULT_T_TOTAL,
    GATE_TIME_EPS,
    TrajectoryPID,
    build_trajectory_pid,
)

# The multi-agent track (config/rl_multi_level2_norand.toml): drone 1's pad spawn and the gates
# the default waypoint spline flies through.
OPPONENT_START_POS = np.array([-1.5, 0.95, 0.01])
GATES = [
    {"pos": [0.5, 0.25, 0.7], "rpy": [0.0, 0.0, -0.78]},
    {"pos": [1.05, 0.75, 1.2], "rpy": [0.0, 0.0, 2.35]},
    {"pos": [-1.0, -0.25, 0.7], "rpy": [0.0, 0.0, 3.14]},
    {"pos": [0.0, -0.75, 1.2], "rpy": [0.0, 0.0, 0.0]},
]


def _build(gates: list | None) -> TrajectoryPID:
    return build_trajectory_pid(
        start_pos=OPPONENT_START_POS,
        drone_mass=0.033,
        freq=50.0,
        control_mode="attitude",
        action_low=np.array([-0.5, -0.5, -3.14, 0.0]),
        action_high=np.array([0.5, 0.5, 3.14, 0.6]),
        gates=gates,
    )


@pytest.mark.unit
def test_gate_times_are_strictly_increasing_and_in_range():
    pid = _build(GATES)
    times = np.asarray(pid.gate_times)
    assert times.shape == (len(GATES),)
    assert np.all(np.diff(times) > 0), f"gate times not strictly increasing: {times}"
    assert np.all((times > 0.0) & (times < DEFAULT_T_TOTAL)), f"gate times out of range: {times}"


@pytest.mark.unit
def test_spline_is_near_each_gate_at_its_pass_time():
    pid = _build(GATES)
    for i, gate in enumerate(GATES):
        pos, _ = pid._spline(pid.gate_times[i])
        dist = float(jnp.linalg.norm(pos - jnp.asarray(gate["pos"])))
        assert dist < 0.3, f"spline is {dist:.2f} m from gate {i} at its pass time"


@pytest.mark.unit
def test_spawn_target_gate_mapping():
    """The forward-biased searchsorted maps spawn times to the gate the spline approaches next."""
    pid = _build(GATES)
    times = np.asarray(pid.gate_times)

    def target(t: float) -> int:
        return int(jnp.searchsorted(pid.gate_times, t + GATE_TIME_EPS, side="right"))

    assert target(0.0) == 0  # pad spawn targets the first gate
    for i, t_pass in enumerate(times):
        # Comfortably before gate i's plane (beyond the eps bias): still targets gate i.
        assert target(float(t_pass) - 2 * GATE_TIME_EPS) == i
        # On/just past the plane: biased forward to gate i + 1 (never stuck on a passed gate).
        assert target(float(t_pass)) == i + 1


@pytest.mark.unit
def test_builds_without_gates():
    """gates=None keeps the previous behavior: no gate times baked."""
    pid = _build(None)
    assert pid.gate_times is None

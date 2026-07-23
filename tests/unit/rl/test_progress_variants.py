"""Unit tests for the swappable progress-reward potentials (champion / asymmetric / fancy).

The per-step progress reward is ``coef * (Phi(curr) - Phi(prev))`` for the selected potential
``Phi``. These tests pin the two properties every variant must have to be a usable progress
potential -- it is a deterministic function of state, and approaching the opening from the entry side
raises it (progress > 0) -- plus the champion variant's exact equivalence to the old
distance-reduction reward and the registry/param plumbing.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from lsy_drone_racing.rl.tasks.progress_variants import (
    GATE_HALF_EXTENT,
    PROGRESS_VARIANTS,
    build_progress_potential,
    default_progress_params,
    gate_opening_distance,
)

# Gate at the world origin with identity orientation: the gate frame equals the world frame, so
# ``along`` is the gate-local +x (traversal) coordinate (<0 entry side, >0 exit side).
_GATES_POS = jnp.zeros((1, 1, 3))
_GATES_QUAT = jnp.array([[[0.0, 0.0, 0.0, 1.0]]])  # identity xyzw
_GATE_ID = jnp.array([[0]])


def _phi(name: str, along: float, y: float = 0.0, z: float = 0.0) -> float:
    """Potential of variant ``name`` at gate-local offset (along, y, z), default shape params."""
    potential = build_progress_potential(name, default_progress_params())
    pos = jnp.array([[[along, y, z]]])  # (E=1, D=1, 3)
    return float(potential(pos, _GATES_POS, _GATES_QUAT, _GATE_ID, GATE_HALF_EXTENT)[0, 0])


@pytest.mark.unit
def test_registry_holds_the_three_shipped_variants():
    """Champion / asymmetric / fancy are all registered and buildable."""
    assert set(PROGRESS_VARIANTS) == {"champion", "asymmetric", "fancy"}
    for name in PROGRESS_VARIANTS:
        assert callable(build_progress_potential(name, default_progress_params()))


@pytest.mark.unit
@pytest.mark.parametrize("name", ["champion", "asymmetric", "fancy"])
def test_entry_approach_raises_the_potential(name: str):
    """Flying forward toward the opening from the entry side increases Phi (progress > 0)."""
    alongs = [-2.8, -2.0, -1.0, -0.5, -0.2, -0.05]  # approaching the plane from -x
    phis = [_phi(name, a) for a in alongs]
    assert np.all(np.diff(phis) > 0), f"{name}: potential not increasing on approach: {phis}"


@pytest.mark.unit
@pytest.mark.parametrize("name", ["champion", "asymmetric", "fancy"])
def test_potential_is_a_deterministic_function_of_state(name: str):
    """Same state -> same potential (a true potential, so the per-step progress telescopes)."""
    assert _phi(name, -1.3, y=0.1) == _phi(name, -1.3, y=0.1)


@pytest.mark.unit
def test_champion_potential_equals_negative_opening_distance():
    """Champion is exactly Phi = -gate_opening_distance, so it reproduces the old progress reward."""
    for along, y in [(-2.8, 0.0), (-0.5, 0.0), (0.5, 1.2), (-1.3, GATE_HALF_EXTENT)]:
        pos = jnp.array([[[along, y, 0.0]]])
        d = float(
            gate_opening_distance(pos, _GATES_POS, _GATES_QUAT, _GATE_ID, GATE_HALF_EXTENT)[0, 0]
        )
        assert _phi("champion", along, y=y) == pytest.approx(-d, abs=1e-6)


@pytest.mark.unit
def test_asymmetric_exit_side_is_lower_than_entry_side():
    """The exit (already-passed) side decays faster, so it reads as lower potential than entry."""
    assert _phi("asymmetric", 0.5) < _phi("asymmetric", -0.5)


@pytest.mark.unit
def test_fancy_rewards_axis_alignment_over_off_axis():
    """At the same standoff, lined up on the gate axis beats an off-axis approach (alignment term)."""
    assert _phi("fancy", -1.0, y=0.0) > _phi("fancy", -1.0, y=1.0)


@pytest.mark.unit
def test_build_progress_potential_rejects_unknown_variant():
    """An unknown variant name fails loudly rather than silently picking a default."""
    with pytest.raises(ValueError, match="unknown progress variant"):
        build_progress_potential("nope", default_progress_params())


@pytest.mark.unit
def test_build_ignores_dormant_variant_params():
    """The full params dict (every variant's knobs) builds fine; only the active variant's are used."""
    params = default_progress_params()
    params["asymmetric"] = {"reach": 1.0, "sharpness": 0.2, "exit_scale": 5.0}
    # champion ignores all of asymmetric's params and still builds + matches -distance.
    pot = build_progress_potential("champion", params)
    pos = jnp.array([[[-0.5, 0.0, 0.0]]])
    assert float(
        pot(pos, _GATES_POS, _GATES_QUAT, _GATE_ID, GATE_HALF_EXTENT)[0, 0]
    ) == pytest.approx(-0.5, abs=1e-6)

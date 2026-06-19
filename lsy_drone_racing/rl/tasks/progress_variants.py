"""Swappable progress-reward potentials for the single-agent racing task.

The dense per-step *progress* reward is always the telescoping increase of a per-gate **potential**
``Phi`` measured against the gate that was the target at the *start* of the step::

    progress = progress_coef * (Phi(curr_pos) - Phi(prev_pos))      # both vs prev target gate

Because every variant is a deterministic function of state (a potential), this difference
telescopes over a path and cannot be farmed by looping -- that guarantee lives here, in the shared
``Phi(curr) - Phi(prev)`` machinery (see ``racing_reward_components``), not in any single variant.

A variant therefore only has to define ``Phi`` (higher == closer to the goal). The three shipped
here are:

* ``champion`` -- ``Phi = -gate_opening_distance`` (Kaufmann et al. 2023): the metres-closed reward,
  a constant gradient straight into the opening. No tunable shape params.
* ``asymmetric`` -- a non-negative through-gate funnel that peaks at the opening and decays faster on
  the already-passed (exit) side. Params: ``reach`` (long-range pull), ``sharpness`` (tight near-gate
  funnel), ``exit_scale`` (how much more the exit side costs).
* ``fancy`` -- a blended angle/distance potential adapted from an earlier brax reward: an exponential
  distance bump near the opening that hands off to a through-gate *alignment* term as the drone nears
  the plane. No tunable shape params (the blend/decay constants are baked in).

Selection is the ``Args.progress = (name, coef)`` tuple; per-variant shape params live in
``Args.progress_params`` (a dict keyed by variant name that always carries *every* variant's params,
so switching variants never requires deleting/re-adding knobs). ``build_progress_potential`` resolves
a name + that dict into a ready ``PotentialFn`` closure.
"""

from functools import partial
from typing import Callable

import jax.numpy as jnp
from jax import Array
from jax.scipy.spatial.transform import Rotation as R

# Half-extent (m) of the square gate opening, used by the distance-based potentials as the cuboid
# corridor through which *any* crossing point is equally good. Matches the gate_size passed to
# gate_passed in race_core's _update_target_gates ((0.45, 0.45) -> 0.225 half-extent); keep the two
# in sync so the dense reward's notion of "inside the opening" agrees with the env's pass detection.
GATE_HALF_EXTENT = 0.225

# A progress potential: drone position(s), gate geometry, the per-drone target gate index, and the
# opening half-extent -> potential Phi, shape (n_envs, n_drones). Higher == closer to the goal.
PotentialFn = Callable[[Array, Array, Array, Array, float], Array]


def _target_gate_frame(
    drone_pos: Array, gates_pos: Array, gates_quat: Array, target_gate: Array
) -> tuple[Array, Array, Array]:
    """Locate each drone's target gate and express the drone offset in that gate's frame.

    Args:
        drone_pos: Drone positions, shape (n_envs, n_drones, 3).
        gates_pos: Gate centers, shape (n_envs, n_gates, 3).
        gates_quat: Gate orientations (xyzw), shape (n_envs, n_gates, 4).
        target_gate: Current target gate index per drone, shape (n_envs, n_drones). -1 (finished)
            wraps to the last gate.

    Returns:
        ``(gate_pos, rot, local)``: the target gate center (n_envs, n_drones, 3), the gate's
        rotation matrix gate->world (n_envs, n_drones, 3, 3), and the drone position in the gate
        frame (n_envs, n_drones, 3) -- local x is the along-axis (traversal) coordinate, local y/z
        span the gate opening.
    """
    n_gates = gates_pos.shape[1]
    env_idx = jnp.arange(gates_pos.shape[0])[:, None]
    idx = target_gate % n_gates  # -1 (finished) indicates the last gate
    gate_pos = gates_pos[env_idx, idx]  # (E, D, 3)
    gate_quat = gates_quat[env_idx, idx]  # (E, D, 4)
    n_envs, n_drones = gate_pos.shape[:2]
    rot = R.from_quat(gate_quat.reshape(-1, 4)).as_matrix().reshape(n_envs, n_drones, 3, 3)
    # rot maps gate-frame vectors to world; its transpose maps the world offset into the gate frame.
    local = jnp.einsum("edji,edj->edi", rot, drone_pos - gate_pos)
    return gate_pos, rot, local


def gate_opening_distance(
    drone_pos: Array,
    gates_pos: Array,
    gates_quat: Array,
    target_gate: Array,
    half_extent: float,
) -> Array:
    """Euclidean distance (m) from each drone to its target gate's *opening* (lower == closer).

    The distance used by the champion-paper progress reward (Kaufmann et al., "Champion-level drone
    racing using deep RL", 2023): the per-step reward is the *reduction* of this distance, so closing
    on the gate banks reward proportional to the metres covered, with a constant gradient all the way
    into the opening. It is unbounded but a deterministic function of state (a potential
    ``Phi = -distance``), so progress telescopes and cannot be farmed by looping.

    Geometry (in the gate frame; the +x column of the gate rotation is the traversal normal):
    ``along`` is the gate-local x (traversal) coordinate; the lateral (y, z) offsets are clamped to
    the opening half-extent per axis, giving a cuboid corridor of equally-good crossing points that
    matches ``gate_passed``'s box test::

        oy = max(|y| - h, 0),  oz = max(|z| - h, 0)
        distance = sqrt(along**2 + oy**2 + oz**2)

    Returns distance, shape (n_envs, n_drones).
    """
    _, _, local = _target_gate_frame(drone_pos, gates_pos, gates_quat, target_gate)
    along = local[..., 0]
    oy = jnp.maximum(jnp.abs(local[..., 1]) - half_extent, 0.0)
    oz = jnp.maximum(jnp.abs(local[..., 2]) - half_extent, 0.0)
    return jnp.sqrt(along**2 + oy**2 + oz**2)


def _champion_potential(**_: float) -> PotentialFn:
    """Champion-paper potential ``Phi = -gate_opening_distance`` (no shape params)."""

    def phi(pos: Array, gates_pos: Array, gates_quat: Array, target_gate: Array, h: float) -> Array:
        return -gate_opening_distance(pos, gates_pos, gates_quat, target_gate, h)

    return phi


def _asymmetric_potential(
    reach: float = 2.0, sharpness: float = 0.3, exit_scale: float = 3.0, **_: float
) -> PotentialFn:
    """Directional through-gate funnel, ``Phi`` in (0, 1], peaking at the opening (see module doc).

    The entry/exit asymmetry is folded into the *distance*: the along coordinate is inflated by
    ``exit_scale`` on the exit side, so being past the gate counts as much farther away. ``Phi``
    blends two length scales, both maximal at the opening: ``reach`` (long-range pull) and
    ``sharpness`` (tight near-gate funnel)::

        along_eff = along (entry) | exit_scale*along (exit)
        distance  = sqrt(along_eff**2 + oy**2 + oz**2)
        Phi       = 0.5*exp(-distance/reach) + 0.5*exp(-distance/sharpness)
    """

    def phi(pos: Array, gates_pos: Array, gates_quat: Array, target_gate: Array, h: float) -> Array:
        _, _, local = _target_gate_frame(pos, gates_pos, gates_quat, target_gate)
        along = local[..., 0]
        oy = jnp.maximum(jnp.abs(local[..., 1]) - h, 0.0)
        oz = jnp.maximum(jnp.abs(local[..., 2]) - h, 0.0)
        along_eff = jnp.where(along > 0.0, along * exit_scale, along)
        distance = jnp.sqrt(along_eff**2 + oy**2 + oz**2)
        return 0.5 * jnp.exp(-distance / reach) + 0.5 * jnp.exp(-distance / sharpness)

    return phi


@partial(jnp.vectorize, signature="(n),(n),(m)->()")
def _fancy_progress(pos: Array, gate_pos: Array, gate_quat: Array) -> Array:
    """Blended angle/distance potential for one drone vs one gate (adapted from an earlier brax reward).

    Near the opening (small distance) the potential is the through-gate *alignment* (``angle_progress``,
    +1 lined up to cross, -1 on the wrong side); far away it is a pure exponential distance bump. The
    blend weight ``gamma_angle = exp(-2*distance)`` hands off from distance to alignment as the drone
    nears the plane.

    Adapted to this repo's gate convention: the original used the gate's *y* axis as the through-gate
    normal; here the traversal normal is the gate-rotation **+x** column (``gate_nx``), with the
    opening spanned by ``gate_ny``/``gate_nz`` -- matching :func:`gate_opening_distance` and
    ``gate_passed``.
    """
    gate_rot = R.from_quat(gate_quat).as_matrix()
    gate_nx = gate_rot[:3, 0]  # through-gate (traversal) normal in this repo's convention
    gate_ny = gate_rot[:3, 1]  # opening axes
    gate_nz = gate_rot[:3, 2]
    gate_width = 2 * GATE_HALF_EXTENT  # 0.45 m square opening
    op = pos - gate_pos  # vector from gate centre to the drone
    # Project op onto the normal for "height" (signed distance from the gate plane).
    dist_to_plane = jnp.dot(op, gate_nx)
    proj_on_plane = pos - dist_to_plane * gate_nx
    # Express the in-plane projection in the (y, z) opening axes and clamp to the square opening.
    local = proj_on_plane - gate_pos
    y = jnp.dot(local, gate_ny)
    z = jnp.dot(local, gate_nz)
    y_clamped = jnp.clip(y, -gate_width / 2, gate_width / 2)
    z_clamped = jnp.clip(z, -gate_width / 2, gate_width / 2)
    # Closest point on the square opening, and the unit direction to it.
    closest_pt = gate_pos + y_clamped * gate_ny + z_clamped * gate_nz
    closest_n = (p := pos - closest_pt) / jnp.linalg.norm(p + 1e-6)
    angle = jnp.arccos(jnp.clip(jnp.dot(closest_n, gate_nx), -1, 1))
    angle_progress = 2 * (angle / jnp.pi - 0.5)
    distance = jnp.linalg.norm(pos - closest_pt)
    distance_progress = jnp.exp(-2 * distance)
    gamma_angle = distance_progress
    gamma_distance = 1 - gamma_angle
    return gamma_angle * angle_progress + gamma_distance * distance_progress


def _fancy_potential(**_: float) -> PotentialFn:
    """Blended angle/distance potential (no shape params); see :func:`_fancy_progress`."""

    def phi(pos: Array, gates_pos: Array, gates_quat: Array, target_gate: Array, h: float) -> Array:
        n_gates = gates_pos.shape[1]
        env_idx = jnp.arange(gates_pos.shape[0])[:, None]
        idx = target_gate % n_gates  # -1 (finished) wraps to the last gate
        gate_pos = gates_pos[env_idx, idx]  # (E, D, 3)
        gate_quat = gates_quat[env_idx, idx]  # (E, D, 4)
        return _fancy_progress(pos, gate_pos, gate_quat)  # vectorized over (E, D)

    return phi


# name -> factory(**shape_params) -> PotentialFn. Each factory binds the variant's shape params
# (ignoring any it doesn't use) and returns a Phi(pos, gates_pos, gates_quat, target_gate, h) closure.
PROGRESS_VARIANTS: dict[str, Callable[..., PotentialFn]] = {
    "champion": _champion_potential,
    "asymmetric": _asymmetric_potential,
    "fancy": _fancy_potential,
}


def default_progress_params() -> dict[str, dict]:
    """Default ``progress_params`` dict: every variant's shape params, always present.

    Carrying all variants' knobs (even the dormant ones) means switching the active variant via the
    ``progress`` tuple never requires deleting/re-adding params. ``champion`` and ``fancy`` have no
    tunable shape params, so their entries are empty.
    """
    return {
        "champion": {},
        "asymmetric": {"reach": 2.0, "sharpness": 0.3, "exit_scale": 3.0},
        "fancy": {},
    }


def build_progress_potential(name: str, progress_params: dict[str, dict]) -> PotentialFn:
    """Resolve a variant ``name`` + the full ``progress_params`` dict into a ready ``PotentialFn``.

    Only ``progress_params[name]`` is consumed (the other variants' dormant params are ignored), so
    the dict can always carry every variant's knobs.
    """
    if name not in PROGRESS_VARIANTS:
        raise ValueError(
            f"unknown progress variant {name!r}; choose one of {sorted(PROGRESS_VARIANTS)}"
        )
    return PROGRESS_VARIANTS[name](**progress_params.get(name, {}))

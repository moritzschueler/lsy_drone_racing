"""Cone-based segment spawning for curriculum training.

Pure-JAX geometry for sampling drone start poses in the *approach corridor* of a chosen gate,
plus the cosine curriculum schedules that drive it. None of this touches the env reset pipeline
(``race_core``); it only produces poses + a target-gate index, which an integration layer (a
wrapper) writes into the environment data.

Conceptual model (see design discussion):

* A spawn for target gate ``i`` is drawn from a truncated cone (frustum) whose tip sits at the gate
  center and whose axis points along the gate's *entry* side (``-n_i``, the negative traversal
  normal). The cone radius grows with distance from the gate, ``r(s) = s * tan(theta)``, so close
  spawns are forced near the gate axis (well aligned with the opening) while far spawns may be
  off-axis. The frustum is truncated at ``d_min`` (always leave runway) and ``d_max`` (segment
  length, capped).
* Candidates are filtered for horizontal clearance to obstacles so spawns are *recoverable* (no
  predetermined crash): fixed-budget rejection picks the first candidate clearing ``margin``, else
  the best-clearance fallback.
* Two cosine-annealed knobs, on *separate* windows: ``kappa`` (cone size, ramps first -> discovery)
  and ``p_start_position`` (probability of using the true race start instead of a segment spawn,
  ramps later -> converge to the deployment distribution).
"""

from __future__ import annotations

from functools import partial
from typing import Any

import jax
import jax.numpy as jp
from flax.struct import dataclass as flax_dataclass
from jax import Array
from jax.scipy.spatial.transform import Rotation as R


@flax_dataclass
class SegmentSpawnConfig:
    """Static configuration for segment spawning (all fields are plain floats / ints).

    Curriculum windows are expressed in *training progress* ``tau in [0, 1]`` (= global_step /
    total_timesteps).
    """

    gate_offset: float = 0.1  # exit-point offset of the predecessor gate, defines segment length
    d_min: float = 0.25  # min standoff from the gate (always leave runway), meters
    d_max_cap: float = 1.5  # global cap on segment length, meters
    theta_max: float = 0.4  # cone half-angle at kappa=1, radians (~34 deg)
    margin: float = 0.30  # required horizontal clearance to any obstacle, meters
    z_min: float = 0.20  # spawn altitude floor, meters
    z_max: float = 2.0  # spawn altitude ceiling, meters
    n_candidates: int = 12  # fixed rejection budget per env
    # (a) cone-size schedule: kappa ramps kappa_min -> 1 over [a0, a1]
    a0: float = 0.05
    a1: float = 0.7
    kappa_min: float = 0.10
    # (b) true-start mixture schedule: p_start_position ramps p_start_min -> p_start_max over
    # [b0, b1]. p_start_min keeps a constant trickle of true-start episodes from the very start, so
    # the deployment distribution is always represented and gate-progress metrics are never empty.
    b0: float = 0.25
    b1: float = 0.85
    p_start_min: float = 0.05
    p_start_max: float = 0.80
    # (c) initial-speed schedule: cone spawns start with speed v0 along the gate normal (through the
    # opening), annealed v0_max -> 0 over [c0, c1]. Early momentum carries the drone through the gate
    # so it collects gate_bonus and gets a positive "through-the-gate" signal to bootstrap from;
    # removing it later forces the policy to generate its own forward speed. Only cone spawns get it
    # (true-start envs keep their grounded zero-velocity reset).
    c0: float = 0.0
    c1: float = 0.5
    v0_max: float = 0.5  # m/s along the gate normal at tau=0


def cosine_ramp(tau: Array, t0: float, t1: float, v0: float, v1: float) -> Array:
    """Cosine ease-in-out ramp from ``v0`` to ``v1`` over the progress window ``[t0, t1]``.

    Clamped to ``v0`` before ``t0`` and ``v1`` after ``t1``. ``tau`` is training progress in
    ``[0, 1]``.
    """
    u = jp.clip((tau - t0) / (t1 - t0), 0.0, 1.0)
    return v0 + (v1 - v0) * (1.0 - jp.cos(jp.pi * u)) / 2.0


def kappa_schedule(tau: Array, cfg: SegmentSpawnConfig) -> Array:
    """Cone-size fraction kappa(tau): kappa_min -> 1 over the (a) window. Ramps first."""
    return cosine_ramp(tau, cfg.a0, cfg.a1, cfg.kappa_min, 1.0)


def p_start_schedule(tau: Array, cfg: SegmentSpawnConfig) -> Array:
    """True-start probability p_start_position(tau): p_start_min -> p_start_max over the (b) window.

    The constant floor ``p_start_min`` applies even before the (b) window, so some fraction of
    episodes always starts from the true race start.
    """
    return cosine_ramp(tau, cfg.b0, cfg.b1, cfg.p_start_min, cfg.p_start_max)


def v0_schedule(tau: Array, cfg: SegmentSpawnConfig) -> Array:
    """Cone-spawn initial speed v0(tau): v0_max -> 0 over the (c) window. Ramps down early."""
    return cosine_ramp(tau, cfg.c0, cfg.c1, cfg.v0_max, 0.0)


def gate_normals(gates_quat: Array) -> Array:
    """Per-gate +x traversal axis in world frame. Shape (..., 4) -> (..., 3).

    Same convention as ``progress_target`` / ``gate_passed``: the gate's local +x is the direction
    of travel through the opening.
    """
    return R.from_quat(gates_quat).as_matrix()[..., :, 0]


def _perp_basis(n: Array) -> tuple[Array, Array]:
    """Two orthonormal vectors spanning the plane perpendicular to unit vector ``n`` (3,)."""
    # Pick a reference axis not (nearly) parallel to n to avoid a degenerate cross product.
    ref = jp.where(jp.abs(n[2]) < 0.9, jp.array([0.0, 0.0, 1.0]), jp.array([1.0, 0.0, 0.0]))
    e1 = jp.cross(n, ref)
    e1 = e1 / (jp.linalg.norm(e1) + 1e-9)
    e2 = jp.cross(n, e1)
    return e1, e2


def _sample_cone_spawn(
    key: Array,
    gate_pos: Array,
    gate_quat: Array,
    d_max: Array,
    obstacles_pos: Array,
    kappa: Array,
    *,
    d_min: float,
    theta_max: float,
    margin: float,
    z_min: float,
    z_max: float,
    n_candidates: int,
) -> Array:
    """Sample one recoverable spawn position in the approach cone of a single gate.

    Draws ``n_candidates`` points in the (kappa-scaled) frustum on the gate's entry side and returns
    the best one by clearance: the first candidate clearing ``margin`` with valid altitude, else the
    candidate with the largest clearance / least bound violation.

    Args:
        key: PRNG key.
        gate_pos: Target gate center, (3,).
        gate_quat: Target gate orientation (xyzw), (4,).
        d_max: Segment length cap for this gate (already min'd with the global cap), scalar.
        obstacles_pos: Obstacle centers, (n_obstacles, 3); only xy is used.
        kappa: Cone-size fraction in (0, 1], scalar (curriculum knob).
        d_min, theta_max, margin, z_min, z_max, n_candidates: see ``SegmentSpawnConfig``.

    Returns:
        Spawn position, (3,).
    """
    n = gate_normals(gate_quat)  # (3,) entry side is -n
    e1, e2 = _perp_basis(n)

    k_s, k_t, k_p = jax.random.split(key, 3)
    # Ensure a positive-width distance interval even for very short/degenerate segments.
    s_hi = jp.maximum(d_min + kappa * (d_max - d_min), d_min + 1e-3)
    s = jax.random.uniform(k_s, (n_candidates,), minval=d_min, maxval=s_hi)
    theta = jax.random.uniform(k_t, (n_candidates,), minval=0.0, maxval=kappa * theta_max)
    phi = jax.random.uniform(k_p, (n_candidates,), minval=0.0, maxval=2.0 * jp.pi)

    radial = s * jp.tan(theta)  # (n_candidates,)
    lateral = radial[:, None] * (jp.cos(phi)[:, None] * e1[None] + jp.sin(phi)[:, None] * e2[None])
    cand = gate_pos[None] - s[:, None] * n[None] + lateral  # (n_candidates, 3), entry side by -s*n

    # Horizontal clearance to the nearest obstacle for each candidate.
    dxy = cand[:, None, :2] - obstacles_pos[None, :, :2]  # (n_candidates, n_obstacles, 2)
    clearance = jp.min(jp.linalg.norm(dxy, axis=-1), axis=-1)  # (n_candidates,)
    z = cand[:, 2]
    z_pen = jp.maximum(z_min - z, 0.0) + jp.maximum(z - z_max, 0.0)  # >0 if out of altitude bounds

    valid = (clearance > margin) & (z_pen <= 0.0)
    # Prefer valid candidates (huge bonus); among invalid, prefer most clearance / least violation.
    score = jp.where(valid, 1e6 + clearance, clearance - 1e3 * z_pen)
    return cand[jp.argmax(score)]


def segment_spawn(
    key: Array,
    gates_pos: Array,
    gates_quat: Array,
    obstacles_pos: Array,
    nominal_start: Array,
    tau: Array,
    cfg: SegmentSpawnConfig,
) -> tuple[Array, Array, Array, Array]:
    """Sample per-env cone spawns and the curriculum mixture mask.

    For every env: pick a target gate uniformly, compute its segment length (distance to the
    predecessor gate's exit point, or to the true start for gate 0, capped at ``d_max_cap``), and
    sample a recoverable cone spawn. Independently, a fraction ``p_start_position(tau)`` of envs are
    flagged to keep the *true race start* instead (``cone_mask == False``); the integration layer
    blends accordingly.

    Args:
        key: PRNG key.
        gates_pos: (n_envs, n_gates, 3).
        gates_quat: (n_envs, n_gates, 4) xyzw.
        obstacles_pos: (n_envs, n_obstacles, 3).
        nominal_start: True drone start position, (3,) — the predecessor reference for gate 0.
        tau: Training progress scalar in [0, 1].
        cfg: Static spawn configuration.

    Returns:
        ``(spawn_pos, spawn_vel, target_gate, cone_mask)``:
          * ``spawn_pos`` (n_envs, 3): cone spawn position for each env (only meaningful where
            ``cone_mask`` is True; the integration layer ignores it otherwise).
          * ``spawn_vel`` (n_envs, 3): initial velocity along the gate normal (through the opening),
            magnitude ``v0_schedule(tau)``; only meaningful where ``cone_mask`` is True.
          * ``target_gate`` (n_envs,): chosen target gate index for the cone spawn.
          * ``cone_mask`` (n_envs,): True where the env should use the cone spawn, False where it
            should keep the true race start.
    """
    n_envs, n_gates = gates_pos.shape[0], gates_pos.shape[1]
    env_idx = jp.arange(n_envs)
    k_gate, k_mix, k_pose = jax.random.split(key, 3)

    # Uniform target-gate choice per env.
    target_gate = jax.random.randint(k_gate, (n_envs,), 0, n_gates)
    gp = gates_pos[env_idx, target_gate]  # (n_envs, 3)
    gq = gates_quat[env_idx, target_gate]  # (n_envs, 4)

    # Predecessor reference: previous gate's exit point, or the true start for gate 0.
    normals = gate_normals(gates_quat)  # (n_envs, n_gates, 3)
    prev_idx = (target_gate - 1) % n_gates
    prev_exit = gates_pos[env_idx, prev_idx] + cfg.gate_offset * normals[env_idx, prev_idx]
    prev_ref = jp.where((target_gate == 0)[:, None], nominal_start[None, :], prev_exit)
    chord = jp.linalg.norm(gp - prev_ref, axis=-1)  # (n_envs,)
    d_max = jp.minimum(chord, cfg.d_max_cap)

    kappa = kappa_schedule(tau, cfg)
    sampler = partial(
        _sample_cone_spawn,
        d_min=cfg.d_min,
        theta_max=cfg.theta_max,
        margin=cfg.margin,
        z_min=cfg.z_min,
        z_max=cfg.z_max,
        n_candidates=cfg.n_candidates,
    )
    keys = jax.random.split(k_pose, n_envs)
    spawn_pos = jax.vmap(sampler, in_axes=(0, 0, 0, 0, 0, None))(
        keys, gp, gq, d_max, obstacles_pos, kappa
    )

    # Initial velocity: speed v0(tau) along the target gate normal (+x, the through-gate direction).
    spawn_vel = v0_schedule(tau, cfg) * gate_normals(gq)  # (n_envs, 3)

    # Mixture: with prob p_start_position keep the true start (cone_mask False), else cone spawn.
    p_start = p_start_schedule(tau, cfg)
    cone_mask = jax.random.uniform(k_mix, (n_envs,)) >= p_start
    return spawn_pos, spawn_vel, target_gate, cone_mask


def apply_spawn(data: Any, mask: Array, spawn_pos: Array, spawn_vel: Array, target_gate: Array) -> Any:
    """Write cone spawns into the environment data for the masked envs.

    Pure function on ``EnvData`` (does not touch the reset pipeline). For each env where ``mask`` is
    True it overrides the drone pose (position from ``spawn_pos``, upright orientation, initial
    linear velocity ``spawn_vel`` along the gate normal, zero angular velocity), sets the target
    gate, resets ``takeoff_pos`` / ``last_drone_pos`` to the new position, and recomputes
    ``gates_visited`` / ``obstacles_visited`` from it (mirroring ``_reset_env_data`` so the in-range
    flags match the new spawn). Envs where ``mask`` is False are left exactly as-is.

    Args:
        data: The environment data (``EnvData``).
        mask: Per-env override mask, (n_envs,).
        spawn_pos: Cone spawn positions, (n_envs, 3).
        spawn_vel: Cone spawn initial velocities (along the gate normal), (n_envs, 3).
        target_gate: Chosen target gate per env, (n_envs,).

    Returns:
        Updated environment data.
    """
    md = mask[:, None, None]  # (n_envs, 1, 1) -> broadcasts over (n_envs, n_drones, *)
    states = data.sim_data.states
    spawn_d = spawn_pos[:, None, :]  # (n_envs, 1, 3)
    spawn_v = spawn_vel[:, None, :]  # (n_envs, 1, 3)
    upright = jp.array([0.0, 0.0, 0.0, 1.0])  # identity quat (xyzw)
    states = states.replace(
        pos=jp.where(md, spawn_d, states.pos),
        quat=jp.where(md, upright, states.quat),
        vel=jp.where(md, spawn_v, states.vel),
        ang_vel=jp.where(md, 0.0, states.ang_vel),
    )
    sim_data = data.sim_data.replace(states=states)
    new_pos = states.pos
    sr = data.sensor_range
    dg = new_pos[..., None, :2] - data.gates_pos[:, None, :, :2]
    gates_visited = jp.where(md, jp.linalg.norm(dg, axis=-1) < sr, data.gates_visited)
    do = new_pos[..., None, :2] - data.obstacles_pos[:, None, :, :2]
    obstacles_visited = jp.where(md, jp.linalg.norm(do, axis=-1) < sr, data.obstacles_visited)
    return data.replace(
        sim_data=sim_data,
        target_gate=jp.where(mask[:, None], target_gate[:, None], data.target_gate),
        takeoff_pos=jp.where(md, spawn_d, data.takeoff_pos),
        last_drone_pos=jp.where(md, spawn_d, data.last_drone_pos),
        gates_visited=gates_visited,
        obstacles_visited=obstacles_visited,
    )

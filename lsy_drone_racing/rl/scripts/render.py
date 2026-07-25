import pickle
import time
from pathlib import Path
from typing import Any

import fire
import imageio.v2 as imageio
import jax.numpy as jnp
import numpy as np
from crazyflow.sim import Sim
from crazyflow.sim.visualize import draw_line
from drone_models.core import load_params
from flax import nnx
from jax import Array
from scipy.spatial.transform import Rotation as R

import lsy_drone_racing.envs.race_core as race_core
from lsy_drone_racing.envs.race_core import build_action_space
from lsy_drone_racing.rl.agents.ppo_agent import Agent
from lsy_drone_racing.rl.tasks import get_task
from lsy_drone_racing.rl.wrappers.trajectory_opponent import (
    SPAWN_TIME_MARGIN,
    build_trajectory_pid,
    teleport_opponents,
)
from lsy_drone_racing.rl.wrappers.wrapper_base import Wrapper
from lsy_drone_racing.utils import env_param, load_config, strip_env_randomization

CHECKPOINT_DIR = Path(__file__).parents[1] / "checkpoints"

# Trajectory-trail colors: drone 0 (the trainable ego) is green, drones 1.. (opponents) are red --
# matching the green/red drone markers drawn in ``wrappers.racing_env``.
_EGO_TRAJ_RGBA = np.array([0.0, 1.0, 0.0, 1.0])
_OPPONENT_TRAJ_RGBA = np.array([1.0, 0.0, 0.0, 1.0])
# Cap on line segments drawn per drone. A full race is ~900 steps at 50 Hz; drawing every point for
# every drone each frame would blow past ``Sim.max_visual_geom`` and slow the viewer, so the trail
# is decimated to at most this many points (recent detail is preserved by keeping the last point).
_MAX_TRAJ_POINTS = 400

# --- Camera viewpoints for ``--view`` -------------------------------------------------------------
# All views drive the free camera (``camera=-1``) via a ``cam_config`` dict in the same
# {distance, azimuth, elevation, lookat} shape the env configs use (see ``[[sim.cam_config]]`` in
# config/*.toml). ``current`` reuses the config's own camera; the other two frame the track/gate
# with the same angles ``capture_track.py`` uses for its still shots.
VIEWS = ("current", "top", "gate_side")
_GATE_SIDE_ELEVATION = -12.0  # slightly-above-horizontal look at the gate.
_GATE_SIDE_AZIMUTH_OFFSET = 90.0  # +-90 from the gate's yaw -> edge-on side profile of the frame.
_GATE_SIDE_DISTANCE = 2.4  # further back than capture_track's still, to catch a passing drone.
_TRACK_TOP_ELEVATION = -89.0  # near-straight-down (exactly -90 makes azimuth degenerate).
_TRACK_TOP_AZIMUTH = 90.0  # +x runs across the image, +y up it.
_TRACK_FIT_MARGIN = 1.7  # >1 leaves a border around the track; larger zooms out.
_TRACK_PAD = 0.7  # meters of slack around object center points, for their own geometry.


def _view_cam_config(view: str, gate_idx: int, eval_env: Wrapper, aspect: float) -> dict | None:
    """Build the free-camera ``cam_config`` for a ``--view`` choice (``None`` == use the config's).

    Reads the reset track layout (gates/obstacles/drones) off the env to frame the shot, so it must
    be called *after* ``reset``. ``aspect`` is the render width/height, needed to fit the whole
    track into the (vertical-fovy) frame for the top-down view.
    """
    if view == "current":
        return None
    data = eval_env.unwrapped.data
    sim = eval_env.unwrapped.base_env.sim
    gates_pos = np.asarray(data.gates_pos[0])  # [n_gates, 3], world 0
    if view == "gate_side":
        if not 0 <= gate_idx < len(gates_pos):
            raise ValueError(
                f"--view_gate {gate_idx} out of range (track has {len(gates_pos)} gates)."
            )
        yaw = R.from_quat(np.asarray(data.gates_quat[0])[gate_idx]).as_euler("xyz")[2]  # radians
        return {
            # ``lookat`` MUST be an ndarray: gymnasium's ``_set_cam_config`` only applies array
            # camera fields (via ``cam.lookat[:] = value``) when the value is an ndarray -- a plain
            # list silently falls through ``setattr`` and is ignored, leaving the camera at origin.
            "lookat": np.asarray(gates_pos[gate_idx], dtype=float),
            "distance": _GATE_SIDE_DISTANCE,
            "azimuth": float(np.rad2deg(yaw) + _GATE_SIDE_AZIMUTH_OFFSET),
            "elevation": _GATE_SIDE_ELEVATION,
        }
    if view == "top":
        pts = np.concatenate(
            [gates_pos, np.asarray(data.obstacles_pos[0]), np.asarray(data.sim_data.states.pos[0])]
        )
        low, high = pts.min(axis=0) - _TRACK_PAD, pts.max(axis=0) + _TRACK_PAD
        half = (high - low) / 2.0
        fovy = sim.mj_model.vis.global_.fovy
        # Fit the wider of the two half-extents; fovy is vertical, so divide the x half-extent by
        # the aspect ratio before comparing (else a wide track over-zooms).
        distance = _TRACK_FIT_MARGIN * max(half[1], half[0] / aspect) / np.tan(np.deg2rad(fovy) / 2)
        return {
            # ndarray lookat -- see the note in the gate_side branch above.
            "lookat": np.array([(low[0] + high[0]) / 2, (low[1] + high[1]) / 2, 0.0]),
            "distance": float(distance),
            "azimuth": _TRACK_TOP_AZIMUTH,
            "elevation": _TRACK_TOP_ELEVATION,
        }
    raise ValueError(f"Unknown --view '{view}'. Choose one of {VIEWS}.")


def _latest_checkpoint(task: str) -> Path | None:
    """Return the most recently modified checkpoint for a task, or None if none exist."""
    candidates = list((CHECKPOINT_DIR / task).glob("*.ckpt"))
    return max(candidates, key=lambda p: p.stat().st_mtime) if candidates else None


def _record_positions(
    eval_env: Wrapper, alive: np.ndarray, trails: list[list[np.ndarray]]
) -> None:
    """Append world-0 positions to each still-alive drone's trail.

    ``alive[d]`` is False once drone ``d`` has crashed/finished, so its trail freezes at the crash
    point instead of tracing the drone tumbling or falling to the ground after it's done.
    """
    pos = np.asarray(eval_env.unwrapped.data.sim_data.states.pos[0])  # [n_drones, 3]
    for drone, trail in enumerate(trails):
        if alive[drone]:
            trail.append(pos[drone])


def _draw_trajectories(sim: Sim, trails: list[list[np.ndarray]]) -> None:
    """Draw each drone's flown trail into the viewer, green for ego / red for opponents.

    ``trails[d]`` is the list of ``[3]`` positions collected while drone ``d`` was airborne; it
    stops growing once the drone crashes/finishes, so the frozen trail stays visible without the
    jump to the drone's below-ground warp/reset pose. Markers don't persist in the viewer, so this
    must be called before every ``sim.render()``.
    """
    for drone, trail in enumerate(trails):
        if len(trail) < 2:  # Need at least two points to make a line segment.
            continue
        positions = np.asarray(trail)  # [T, 3]
        if len(positions) > _MAX_TRAJ_POINTS:
            idx = np.unique(np.linspace(0, len(positions) - 1, _MAX_TRAJ_POINTS).astype(int))
            positions = positions[idx]
        rgba = _EGO_TRAJ_RGBA if drone == 0 else _OPPONENT_TRAJ_RGBA
        draw_line(sim, positions, rgba=rgba)


def main(
    task: str = "single_agent_racing",
    config: str = "level0.toml",
    checkpoint: str | None = None,
    opponent: str = "self_play",
    opponent_pid_speed: float = 1.0,
    opponent_pid_start_frac: float = 0.0,
    ego_start_delay: float = 0.0,
    show_trajectory: bool = True,
    no_randomization: bool = False,
    seed: int | None = None,
    view: str = "current",
    view_gate: int = 1,
    video: str | None = None,
    video_fps: int | None = None,
    video_width: int = 1280,
    video_height: int = 720,
    disable_warp: bool = False,
    **kwargs: Any,
):
    """Render the simulation for a single trained PPO agent on the given task.

    Args:
        task: Task name (one of the keys in ``lsy_drone_racing.rl.tasks.TASKS``).
        checkpoint: Path to a specific checkpoint to evaluate.
        opponent: For multi-drone tasks, how drones ``1..`` are controlled. ``"self_play"``
            (default) mirrors the loaded ego checkpoint onto every drone, like training's
            self-play eval. ``"pid"`` instead flies them with the scripted waypoint-following
            trajectory PID (``wrappers.trajectory_opponent``), at a fixed (non-random) speed --
            needs ``control_mode == "attitude"``. Ignored for single-drone tasks.
        opponent_pid_speed: Speed multiplier for the PID opponent (1.0 == the nominal ~18s single
            pass). Only used when ``opponent == "pid"``.
        opponent_pid_start_frac: Deterministic fraction of the trajectory at which the PID
            opponent starts (0.0 == the pad, like training's random mid-track spawns but fixed for
            visual inspection; clamped to stay before the last gate). Only used when
            ``opponent == "pid"``.
        ego_start_delay: Seconds the ego drone (0) hovers on its pad before its policy takes over.
            Unlike ``opponent_pid_start_frac`` (which *teleports* the opponent ahead), the opponent
            here flies normally from the pad during the delay, building a genuine head start. The
            ego is held in place by the same attitude PID targeting its start pad, so this needs
            ``opponent == "pid"`` (attitude control).
        show_trajectory: If True (default), draw each drone's flown trail into the viewer -- the
            ego drone (0) in green and the opponent(s) (1..) in red. Set False to disable.
        no_randomization: If True, strip the config's ``[env.randomizations]`` /
            ``[env.disturbances]`` blocks for a clean, deterministic race -- needed so a scripted
            PID opponent (``opponent="pid"``) doesn't crash into randomized gates.
        seed: Reset seed, pinned so repeated runs are bit-identical (defaults to the task's own
            fixed seed, 42). Also overrides the config's ``[env] seed`` (shipped as ``-1`` ==
            "random") so nothing is left to chance -- use this when comparing two variants: with a
            fixed seed, drones are dynamically independent (no aerodynamic/downwash model between
            them), so a scripted PID opponent (``opponent="pid"``) flies the *same* trajectory
            regardless of what the ego does, unless the two actually collide (a real physical event
            that disables both) or you're comparing frames at different step indices because the
            two runs have different total lengths.
        view: Camera viewpoint. ``"current"`` (default) uses the config's camera (the angled whole-
            track view). ``"top"`` looks straight down at the whole track. ``"gate_side"`` frames a
            single gate (``--view_gate``) edge-on from the side. Applies to both the live viewer and
            ``--video``.
        view_gate: Which gate ``view="gate_side"`` looks at, as a 0-indexed gate number (default
            ``1`` == the second gate). Ignored for the other views.
        video: If set, render *offscreen* to this file instead of opening the interactive viewer --
            no window pops up. The same chase camera, drone markers and trajectory trails as the
            live view are captured. The format follows the extension: ``.mp4`` (H.264 -- small, no
            frame-coalescing; recommended) or ``.gif`` (256-color, larger). Both work out of the
            box.
        video_fps: Playback frame rate. Defaults to the env's control frequency (``freq``), which
            is required for real-time playback: exactly one captured frame is written per sim step
            (no resampling), so setting this to anything other than ``freq`` stretches or
            compresses the whole video's apparent speed by ``freq / video_fps`` -- e.g. control
            freq 50 with ``video_fps=30`` plays everything, including ``ego_start_delay``, ~1.67x
            slower than real sim time (a requested 18s hold would show up around 30s into the
            video).
        video_width: Offscreen frame width in pixels (only used when ``video`` is set).
        video_height: Offscreen frame height in pixels (only used when ``video`` is set).
        disable_warp: Diagnostic flag, not needed for normal use. Normally a crashed/disabled drone
            is teleported to ``(-1, -1, -1)`` every step (``race_core._warp_disabled_drones``) so it
            doesn't clutter the track -- but its velocity/quat/ang_vel are *not* reset, so it keeps
            "flying" from that fixed point. If True, this patches that warp into a no-op
            (monkeypatched onto the ``race_core`` module before the env is built, so it's picked up
            by ``jax.jit`` tracing): a disabled drone then keeps flying/falling from wherever it
            actually crashed instead of being snapped to ``-1``. This was used to help isolate a bug
            where one drone crashing corrupted a still-racing sibling's rotor speeds (fixed in
            ``wrappers.takeoff.SpinUpRotors``) -- it ruled out the warp's extreme coordinate as the
            cause. Kept for future debugging of similar cross-drone issues.
    """
    assert opponent in ("self_play", "pid"), f"Unknown opponent mode '{opponent}'."
    assert view in VIEWS, f"Unknown view '{view}'. Choose one of {VIEWS}."
    if disable_warp:
        race_core._warp_disabled_drones = lambda data: data
        print("Diagnostic: _warp_disabled_drones patched to a no-op (disabled drones keep flying).")
    task_spec = get_task(task)
    # Render a single env: the viewer only ever shows world 0, so building the training default
    # (num_envs=1024) would step 1024 mjx envs eagerly per frame (crippling on a laptop / CPU) and
    # make ``done`` fire as soon as *any* of the 1024 ends -- aborting the shown drone mid-track.
    kwargs.setdefault("num_envs", 1)
    if seed is not None:
        kwargs["seed"] = seed
    args = task_spec.args_cls.create(**kwargs)

    checkpoint_dir = CHECKPOINT_DIR / task

    if not checkpoint:
        latest_checkpoint = _latest_checkpoint(task)
        if not latest_checkpoint:
            raise FileNotFoundError(
                f"No checkpoints found for task '{task}' in {checkpoint_dir}, can't render. Aborting!"
            )
        else:
            print(f"No checkpoint specified, using latest checkpoint: {latest_checkpoint}")
            model_path = latest_checkpoint
    else:
        model_path = CHECKPOINT_DIR / task / checkpoint

    env_config = load_config(Path(__file__).parents[3] / "config" / config)
    # The shipped configs use ``seed = -1`` ("random"). The reset seed below already governs the
    # episode, but pin the config's too so nothing construction-time is left random.
    env_config.env.seed = args.seed
    if no_randomization:
        strip_env_randomization(env_config)
    print(f"Seed: {args.seed} (a PID opponent flies the same trajectory regardless of the ego).")

    eval_env: Wrapper = task_spec.make_env(args, env_config)
    action_dim = int(np.prod(eval_env.single_action_space.shape))
    obs_dim = int(np.prod(eval_env.single_observation_space.shape))

    agent = Agent(obs_dim, action_dim, rngs=nnx.Rngs(0))
    with open(model_path, "rb") as f:
        nnx.update(agent, pickle.load(f))

    # Single-agent envs (RacingEnv) expose no ``n_drones`` attribute -- they're implicitly 1 drone;
    # only the multi-drone MultiRacingEnv carries the drone count.
    n_drones = getattr(eval_env.unwrapped, "n_drones", 1)
    assert ego_start_delay == 0.0 or opponent == "pid", (
        "ego_start_delay holds the ego with the attitude PID and needs opponent='pid'."
    )
    traj_pid = None
    ego_pid = None  # attitude PID that hovers drone 0 on its pad during ``ego_start_delay``
    if opponent == "pid":
        assert n_drones > 1, "opponent='pid' needs a multi-drone task/config."
        assert env_param(env_config, "control_mode") == "attitude", (
            f"opponent='pid' needs an attitude-control track config, got control_mode="
            f"'{env_param(env_config, 'control_mode')}'."
        )
        drone_mass = load_params(env_config.sim.physics, env_config.sim.drone_model)["mass"]
        action_space = build_action_space(
            env_param(env_config, "control_mode"), env_config.sim.drone_model
        )
        traj_pid = build_trajectory_pid(
            start_pos=np.asarray(env_config.env.track.drones[1]["pos"]),
            drone_mass=drone_mass,
            freq=env_param(env_config, "freq"),
            control_mode=env_param(env_config, "control_mode"),
            action_low=np.asarray(action_space.low),
            action_high=np.asarray(action_space.high),
            gates=env_config.env.track.gates,
        )
        print(f"Opponent(s): scripted PID trajectory-follower, speed {opponent_pid_speed}x.")
        if ego_start_delay > 0.0:
            # Hold-only PID for the ego: same gains/attitude convention, but seeded at the ego's own
            # pad. Driven at virtual_t=0, speed=0 it targets that pad with zero velocity -> a hover
            # hold. Waypoints/gates are irrelevant here (never advanced past t=0), so gates=None.
            ego_pid = build_trajectory_pid(
                start_pos=np.asarray(env_config.env.track.drones[0]["pos"]),
                drone_mass=drone_mass,
                freq=env_param(env_config, "freq"),
                control_mode=env_param(env_config, "control_mode"),
                action_low=np.asarray(action_space.low),
                action_high=np.asarray(action_space.high),
            )

    @nnx.jit
    def policy_step(agent: Agent, env: Wrapper, obs: Array) -> tuple[Wrapper, Array, Array, Array]:
        """One deterministic (mean-action) env step, compiled so the full wrapper stack fuses.

        ``render()`` stays outside jit -- it mutates the host-side MuJoCo sim/viewer -- but the
        physics + wrapper chain run as a single compiled kernel instead of eager op dispatch.
        """
        mean, _, _ = agent(obs)
        env, (obs, _, terminated, truncated, _) = env.step(env, mean)
        return env, obs, terminated, truncated

    @nnx.jit
    def policy_step_pid(
        agent: Agent,
        env: Wrapper,
        obs: Array,
        traj_t: Array,
        i_error: Array,
        ego_hold: Array,
        ego_i_error: Array,
    ) -> tuple[Wrapper, Array, Array, Array, Array, Array, Array]:
        """Like ``policy_step``, but drones ``1..`` are driven by ``traj_pid`` instead of ``agent``.

        Reads the opponents' raw ``pos``/``vel``/``quat`` off the unwrapped sim state (bypassing
        the flattened/relative observation, which drops absolute position) -- the same pattern
        ``ippo.py``'s rollout uses to mix in scripted-PID opponents during training.

        When ``ego_pid`` is set and ``ego_hold`` is true, drone 0's policy action is replaced by a
        hover-hold on its pad (``ego_start_delay``), so the opponent builds a head start while the
        ego waits. ``ego_i_error`` carries that hold PID's integral term across steps.
        """
        mean, _, _ = agent(obs)
        states = env.unwrapped.data.sim_data.states
        opp_pos, opp_vel, opp_quat = states.pos[:, 1:], states.vel[:, 1:], states.quat[:, 1:]
        action_phys, i_error = traj_pid.action(
            opp_pos, opp_vel, opp_quat, traj_t, opponent_pid_speed, i_error
        )
        action = mean.at[:, 1:].set(traj_pid.normalize(action_phys))
        if ego_pid is not None:
            # Hold drone 0 on its pad (virtual_t=0, speed=0 -> position hold, zero velocity target)
            # while ``ego_hold``; once the delay elapses, fall through to the policy's mean action.
            ego_pos, ego_vel, ego_quat = states.pos[:, 0], states.vel[:, 0], states.quat[:, 0]
            hold_phys, ego_i_error = ego_pid.action(
                ego_pos, ego_vel, ego_quat, jnp.zeros(ego_pos.shape[0]), 0.0, ego_i_error
            )
            ego_action = jnp.where(ego_hold, ego_pid.normalize(hold_phys), mean[:, 0])
            action = action.at[:, 0].set(ego_action)
        env, (obs, _, terminated, truncated, _) = env.step(env, action)
        traj_t = traj_t + opponent_pid_speed / env_param(env_config, "freq")
        return env, obs, terminated, truncated, traj_t, i_error, ego_i_error

    eval_env, (obs, _) = eval_env.reset(eval_env, seed=args.seed)
    sim = eval_env.unwrapped.base_env.sim if show_trajectory else None
    trails: list[list[np.ndarray]] = [[] for _ in range(n_drones)]  # world-0 path per drone
    alive = np.ones(n_drones, dtype=bool)  # a drone's trail freezes once it crashes/finishes
    if sim is not None:
        # The viewer is built lazily with ``max_geom=sim.max_visual_geom`` on the first render, so
        # raise the budget now to fit the decimated trails (plus the per-drone markers) for the
        # whole race -- otherwise ``draw_line`` raises once the trail grows past the default 1000.
        sim.max_visual_geom = max(sim.max_visual_geom, _MAX_TRAJ_POINTS * n_drones + 100)
    traj_t = jnp.zeros((1, n_drones - 1))
    i_error = jnp.zeros((1, n_drones - 1, 3))
    ego_i_error = jnp.zeros((1, 3))  # hold-PID integral term for drone 0 (world 0)
    ego_hold_steps = round(ego_start_delay * env_param(env_config, "freq"))
    if ego_pid is not None:
        print(
            f"Ego waits {ego_start_delay:.1f}s ({ego_hold_steps} steps) on its pad before racing; "
            "the PID opponent flies from the pad meanwhile."
        )
    if traj_pid is not None and opponent_pid_start_frac > 0.0:
        # Place the PID opponent(s) mid-track at the requested trajectory fraction, mirroring
        # training's random mid-track spawns but deterministic for visual inspection.
        spawn_t_max = float(traj_pid.gate_times[-1]) - SPAWN_TIME_MARGIN
        traj_t = jnp.full_like(traj_t, min(opponent_pid_start_frac * traj_pid.t_total, spawn_t_max))
        eval_env = teleport_opponents(
            eval_env,
            traj_pid,
            traj_t,
            jnp.full_like(traj_t, opponent_pid_speed),
            jnp.ones_like(traj_t, dtype=bool),
        )
        print(f"PID opponent starts mid-track at t0 = {float(traj_t[0, 0]):.2f}s.")
    # When ``video`` is set, render offscreen (``mode="rgb_array"``) and stream frames straight into
    # the writer instead of opening the interactive window. Streaming (vs. collecting a list) keeps
    # a full ~900-frame race from holding gigabytes of pixels in RAM.
    # Pick the camera viewpoint (``--view``). ``current`` -> None -> env keeps its config camera;
    # ``top``/``gate_side`` -> a free-camera cam_config that overrides it, for both live and video.
    cam_config = _view_cam_config(view, view_gate, eval_env, video_width / video_height)
    render_kwargs: dict[str, Any] = {}
    if cam_config is not None:
        render_kwargs |= {"camera": -1, "cam_config": cam_config}
        gate_note = f" (gate {view_gate})" if view == "gate_side" else ""
        print(f"View: {view}{gate_note}.")

    record = video is not None
    writer = None
    fps = video_fps if video_fps is not None else int(env_param(env_config, "freq"))
    if record:
        writer = imageio.get_writer(video, fps=fps)
        render_kwargs |= {"mode": "rgb_array", "width": video_width, "height": video_height}
        print(
            f"Recording offscreen -> {Path(video).resolve()} "
            f"({video_width}x{video_height} @ {fps} fps)"
        )

    frames_written = 0

    def _show() -> None:
        """Render one frame: to the live viewer, or append it to the video file when recording."""
        nonlocal frames_written
        frame = eval_env.render(**render_kwargs)
        if writer is not None:
            writer.append_data(np.asarray(frame, dtype=np.uint8))
            frames_written += 1

    if sim is not None:
        _record_positions(eval_env, alive, trails)
        _draw_trajectories(sim, trails)
    _show()
    done = False
    step = 0
    # Pace the *live* viewer to wall-clock real time (recording skips this: frames are written 1:1
    # and played back at ``video_fps``, so they're already accurate regardless of compute time).
    # Without this, each iteration takes however long physics+render actually costs -- typically
    # more than 1/freq seconds -- so anything timed in steps (e.g. ``ego_start_delay``) visibly
    # drifts later, the more so the more steps it holds for. ``next_frame_at`` is an absolute
    # deadline (not "sleep 1/freq every step") so a single slow frame (e.g. first-call jit compile)
    # doesn't cause a catch-up burst of unthrottled frames afterwards.
    frame_period = 0.0 if record else 1.0 / env_param(env_config, "freq")
    next_frame_at = time.monotonic() + frame_period

    while not done:
        if traj_pid is not None:
            # ego_hold is a traced scalar (not a Python bool) so flipping it after the hold window
            # doesn't retrigger a jit recompile.
            ego_hold = jnp.asarray(step < ego_hold_steps)
            eval_env, obs, terminated, truncated, traj_t, i_error, ego_i_error = policy_step_pid(
                agent, eval_env, jnp.asarray(obs), traj_t, i_error, ego_hold, ego_i_error
            )
        else:
            eval_env, obs, terminated, truncated = policy_step(agent, eval_env, jnp.asarray(obs))
        if not record:
            now = time.monotonic()
            if now < next_frame_at:
                time.sleep(next_frame_at - now)
                now = next_frame_at
            next_frame_at = now + frame_period
        # Per-drone done flags for world 0 (a scalar for single-drone tasks, [n_drones] for multi).
        done_per_drone = np.atleast_1d(np.asarray(terminated | truncated)[0]).reshape(-1)
        # Freeze a drone's trail *before* recording this step: a crashed drone is disabled and
        # warped below ground (pos -> [-1, -1, -1]) on the very step it's marked done, so recording
        # it would draw a big jump from the crash point down to that warp/reset pose.
        alive &= ~done_per_drone
        if sim is not None:
            _record_positions(eval_env, alive, trails)
            _draw_trajectories(sim, trails)
        _show()
        # For multi-drone tasks, terminated/truncated are per-drone -- wait for the whole race
        # (every drone finished/crashed/timed out) instead of stopping as soon as any one drone
        # (e.g. the ego) is done, matching the env's own NEXT_STEP autoreset semantics: the world
        # doesn't actually reset until all drones are settled.
        done = bool(jnp.all(terminated | truncated))
        step += 1
    # Hold the final frame for a moment so the crash/finish (and trajectory trails) stays visible
    # instead of the window vanishing the instant the race ends. When recording, append ~1s of the
    # frozen final frame to the video for the same effect; otherwise keep the viewer responsive.
    if record:
        for _ in range(fps):
            if sim is not None:
                _draw_trajectories(sim, trails)
            _show()
        writer.close()
        print(f"Saved video ({frames_written} frames @ {fps} fps) to {Path(video).resolve()}")
    else:
        hold_end = time.monotonic() + 2.0
        while time.monotonic() < hold_end:
            if sim is not None:
                _draw_trajectories(sim, trails)
            eval_env.render()
    eval_env.close()


if __name__ == "__main__":
    fire.Fire(main)

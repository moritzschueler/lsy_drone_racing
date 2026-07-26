"""High-resolution offscreen capture of the race track and every gate.

Renders PNGs with MuJoCo's *offscreen* renderer (``mujoco.Renderer``) -- no interactive window, so
the output resolution is set purely by ``--width``/``--height`` and is **not** limited by your
monitor (unlike the press-``T`` screenshot in ``render.py``, which only captures the on-screen
window framebuffer).

For every gate it writes a top-down and an edge-on side view, plus one top-down and one side view
of the whole track. By default the scene is captured at the track's reset state (gates/obstacles
placed, drones on their start pads). ``render.py`` reuses ``capture_from_env`` to take the same
shots at the *end of a race* -- see the note there about warped drones.

Example:
    pixi run python -m lsy_drone_racing.rl.scripts.capture_track --config level0.toml \
        --width 3840 --height 2160 --out_dir track_shots
"""

from pathlib import Path
from typing import Any

import fire
import imageio.v2 as imageio
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.rl.tasks import get_task
from lsy_drone_racing.rl.wrappers.wrapper_base import Wrapper
from lsy_drone_racing.utils import load_config

# --- Camera framing knobs (degrees / meters). Tweak these to taste. --------------------------------
# Elevation is the vertical camera angle: -90 looks straight down, 0 is horizontal.
_GATE_TOP_ELEVATION = -89.0  # near-straight-down; exactly -90 makes azimuth degenerate.
_GATE_SIDE_ELEVATION = -12.0  # slightly-above-eye-level horizontal look at the gate.
_GATE_DISTANCE = 1.9  # camera-to-gate distance for the close-up gate shots.
# Azimuth offsets added to each gate's yaw, so the framing follows the gate's orientation.
# A gate is a *vertical square frame*, so the view angle matters:
#   offset 0    -> camera faces the gate head-on, looking through the opening (shows the square).
#   offset +-90 -> camera sees the frame edge-on, i.e. a thin profile from the side.
_GATE_TOP_AZIMUTH_OFFSET = 0.0
_GATE_SIDE_AZIMUTH_OFFSET = 90.0  # edge-on side profile of the gate.

_TRACK_TOP_ELEVATION = -89.0
_TRACK_SIDE_ELEVATION = -22.0
_TRACK_TOP_AZIMUTH = 90.0  # +x runs across the top-down image, +y up it.
_TRACK_SIDE_AZIMUTH = 90.0
# The whole-track fit is deliberately loose: gate frames extend ~0.6 m past their center points and
# the elevated gates (z up to 1.2 m) sit closer to an overhead camera, so they magnify and drift
# outward. The pad + margin leave enough room that nothing clips at the edges.
_TRACK_FIT_MARGIN = 1.6  # >1 leaves a border around the track; larger zooms out.
_TRACK_SIDE_ZOOM = 1.4  # side view sits further back (perspective depth needs extra room).
_TRACK_PAD = 0.7  # meters of slack around object center points, for their own geometry.


def _fit_distance(half_w: float, half_h: float, aspect: float, fovy_deg: float) -> float:
    """Camera distance that fits a ``2*half_w`` x ``2*half_h`` box in frame.

    MuJoCo's fovy is the *vertical* field of view, so the horizontal half-extent has to be divided
    by the aspect ratio before comparing -- fitting a single bounding radius instead over-zooms on
    a wide track shot like this one.
    """
    return _TRACK_FIT_MARGIN * max(half_h, half_w / aspect) / np.tan(np.deg2rad(fovy_deg) / 2.0)


def _render(
    renderer: mujoco.Renderer,
    mj_data: mujoco.MjData,
    lookat: np.ndarray,
    distance: float,
    azimuth: float,
    elevation: float,
) -> np.ndarray:
    """Render a single offscreen frame from a free camera and return it as an HxWx3 uint8 array."""
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = lookat
    cam.distance = float(distance)
    cam.azimuth = float(azimuth)
    cam.elevation = float(elevation)
    renderer.update_scene(mj_data, camera=cam)
    return renderer.render()


def capture_shots(
    sim: Any,
    gates_pos: np.ndarray,
    gates_quat: np.ndarray,
    obstacles_pos: np.ndarray,
    drones_pos: np.ndarray,
    width: int,
    height: int,
    out_dir: str,
) -> list[str]:
    """Render + save the top/side gate shots and the whole-track shots; return the file paths.

    Assumes ``sim.mj_data`` already reflects the scene to capture (see ``capture_from_env``, which
    does the sync). ``gates_quat`` is xyzw; positions are ``[N, 3]`` for the world being captured.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    # The offscreen framebuffer defaults to 640x480 in the model XML; the renderer refuses any
    # larger image. Raising it here is what lets us capture well above screen resolution.
    sim.mj_model.vis.global_.offwidth = width
    sim.mj_model.vis.global_.offheight = height
    gate_yaws = R.from_quat(gates_quat).as_euler("xyz")[:, 2]  # radians
    renderer = mujoco.Renderer(sim.mj_model, height=height, width=width)
    saved: list[str] = []

    try:
        # --- Per-gate top-down and edge-on side close-ups ----------------------------------------
        for i, (pos, yaw) in enumerate(zip(gates_pos, gate_yaws)):
            yaw_deg = float(np.rad2deg(yaw))
            top = _render(
                renderer, sim.mj_data, pos, _GATE_DISTANCE,
                yaw_deg + _GATE_TOP_AZIMUTH_OFFSET, _GATE_TOP_ELEVATION,
            )
            side = _render(
                renderer, sim.mj_data, pos, _GATE_DISTANCE,
                yaw_deg + _GATE_SIDE_AZIMUTH_OFFSET, _GATE_SIDE_ELEVATION,
            )
            for name, img in (("top", top), ("side", side)):
                fname = out_path / f"gate{i}_{name}.png"
                imageio.imwrite(fname, img)
                saved.append(str(fname))

        # --- Whole-track top-down and side -------------------------------------------------------
        # Frame the bounding box of everything on the track (gates, obstacles, drones), padded for
        # the objects' own size since these are only their center points.
        pts = np.concatenate([gates_pos, obstacles_pos, drones_pos], axis=0)
        low, high = pts.min(axis=0) - _TRACK_PAD, pts.max(axis=0) + _TRACK_PAD
        half = (high - low) / 2.0
        aspect = width / height
        fovy = sim.mj_model.vis.global_.fovy
        cx, cy = (low[0] + high[0]) / 2.0, (low[1] + high[1]) / 2.0

        # Top-down (azimuth 90): +x across the image, +y up it.
        track_top = _render(
            renderer, sim.mj_data, np.array([cx, cy, 0.0]),
            _fit_distance(half[0], half[1], aspect, fovy),
            _TRACK_TOP_AZIMUTH, _TRACK_TOP_ELEVATION,
        )
        # Side (azimuth 90, looking along +y): +x across the image, +z up it.
        cz = max((low[2] + high[2]) / 2.0, 0.0)
        track_side = _render(
            renderer, sim.mj_data, np.array([cx, cy, cz]),
            _fit_distance(half[0], half[2], aspect, fovy) * _TRACK_SIDE_ZOOM,
            _TRACK_SIDE_AZIMUTH, _TRACK_SIDE_ELEVATION,
        )
        for name, img in (("top", track_top), ("side", track_side)):
            fname = out_path / f"track_{name}.png"
            imageio.imwrite(fname, img)
            saved.append(str(fname))
    finally:
        renderer.close()

    return saved


def capture_from_env(
    eval_env: Wrapper, width: int, height: int, out_dir: str, world: int = 0
) -> list[str]:
    """Sync ``world``'s current state into MuJoCo and capture the track/gate shots from it.

    Works on any live env -- a freshly reset track (see ``main``) or the final frame of a race (see
    ``render.py``'s ``capture_dir`` option). Returns the saved file paths.
    """
    # The base gym env owns the loaded MuJoCo model + the compiled render-sync kernel that places the
    # gates/obstacles (mocap bodies) and drones into the sim data, exactly like RaceCoreEnv.render().
    base = eval_env.unwrapped.base_env
    sim = base.sim
    data = eval_env.unwrapped.data
    data, sim.mjx_data = base._render_sync(data, sim.mjx_data)
    sim.mj_data.qpos[:] = np.asarray(sim.mjx_data.qpos[world])
    sim.mj_data.mocap_pos[:] = np.asarray(sim.mjx_data.mocap_pos[world])
    sim.mj_data.mocap_quat[:] = np.asarray(sim.mjx_data.mocap_quat[world])
    mujoco.mj_forward(sim.mj_model, sim.mj_data)
    return capture_shots(
        sim,
        np.asarray(data.gates_pos[world]),
        np.asarray(data.gates_quat[world]),  # xyzw
        np.asarray(data.obstacles_pos[world]),
        np.asarray(data.sim_data.states.pos[world]),
        width,
        height,
        out_dir,
    )


def main(
    task: str = "single_agent_racing",
    config: str = "level0.toml",
    width: int = 1920,
    height: int = 1080,
    out_dir: str = "track_shots",
    seed: int | None = None,
    **kwargs: Any,
):
    """Capture high-res top/side PNGs of every gate and of the whole track (empty reset scene).

    Args:
        task: Task name (one of the keys in ``lsy_drone_racing.rl.tasks.TASKS``). Only used to build
            an env with the track loaded; the trained policy is not needed.
        config: Track config file under ``config/`` (its gates/obstacles define the scene).
        width: Output image width in pixels (not limited by your screen -- this is offscreen).
        height: Output image height in pixels.
        out_dir: Directory the PNGs are written to (created if missing).
        seed: Reset seed. With a randomized config this picks the track layout; ``None`` uses the
            task's default seed.
        **kwargs: Extra env args forwarded to the task's args factory (e.g. ``control_mode``).
    """
    task_spec = get_task(task)
    kwargs.setdefault("num_envs", 1)
    args = task_spec.args_cls.create(**kwargs)
    env_config = load_config(Path(__file__).parents[3] / "config" / config)

    eval_env = task_spec.make_env(args, env_config)
    eval_env, _ = eval_env.reset(eval_env, seed=seed if seed is not None else args.seed)

    saved = capture_from_env(eval_env, width, height, out_dir)
    print(f"Saved {len(saved)} images ({width}x{height}) to {Path(out_dir).resolve()}:")
    for s in saved:
        print(f"  {s}")


if __name__ == "__main__":
    fire.Fire(main)

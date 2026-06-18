"""Headless driver for the LSY drone racing simulation.

Runs one or more episodes of `scripts/sim.py`'s control loop with rendering
forced into MuJoCo's offscreen (rgb_array) path, so it works on a headless
box via EGL -- no X server, no `xvfb`, no GUI window. Saves PNG frames of the
drone flying the track and prints per-episode stats.

Run it through pixi with the EGL backend selected:

    MUJOCO_GL=egl pixi run python \
        .claude/skills/run-lsy-drone-racing/driver.py \
        --config level1.toml --frames 8 --out /tmp/drone_run

Flags:
    --config      config file in config/ (default level1.toml).
    --controller  controller file in lsy_drone_racing/control/ to override the
                  one named in the config (default: use the config's). Handy
                  because level0/level2 point at a controller file that may not
                  exist on this branch.
    --n_runs      number of episodes (default 1).
    --frames      number of evenly-spaced screenshots per episode (default 6).
    --out         output directory for PNGs (default /tmp/drone_run).
    --width/--height  render resolution (default 960x540).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import fire
import gymnasium
import numpy as np
from gymnasium.wrappers.jax_to_numpy import JaxToNumpy
from PIL import Image

from lsy_drone_racing.utils import load_config, load_controller

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[3]  # driver.py -> repo root


def run(
    config: str = "level1.toml",
    controller: str | None = None,
    n_runs: int = 1,
    frames: int = 6,
    out: str = "/tmp/drone_run",
    width: int = 960,
    height: int = 540,
    camera: int | str | None = None,
) -> list[float | None]:
    """Run the sim headless and dump screenshots. Returns per-episode times."""
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.pop("WAYLAND_DISPLAY", None)

    cfg = load_config(ROOT / "config" / config)
    cfg.sim.render = False  # we drive rendering ourselves via rgb_array
    control_path = ROOT / "lsy_drone_racing/control"
    controller_cls = load_controller(control_path / (controller or cfg.controller.file))

    env = gymnasium.make(
        cfg.env.id,
        freq=cfg.env.freq,
        sim_config=cfg.sim,
        sensor_range=cfg.env.sensor_range,
        control_mode=cfg.env.control_mode,
        track=cfg.env.track,
        disturbances=cfg.env.get("disturbances"),
        randomizations=cfg.env.get("randomizations"),
        seed=cfg.env.seed,
    )
    env = JaxToNumpy(env)
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)

    ep_times: list[float | None] = []
    for run_idx in range(n_runs):
        obs, info = env.reset()
        ctrl = controller_cls(obs, info, cfg)
        u = env.unwrapped
        cam = u.settings.camera if camera is None else camera
        cam_cfg = u.settings.cam_config if camera is None else None
        # Estimate a step budget so we can space screenshots across the episode.
        budget = int(cfg.env.freq * 15)
        shot_every = max(1, budget // max(1, frames))
        i = shot = 0
        curr_time = 0.0
        while True:
            curr_time = i / cfg.env.freq
            action = ctrl.compute_control(obs, info)
            obs, reward, terminated, truncated, info = env.step(action)
            finished = ctrl.step_callback(action, obs, reward, terminated, truncated, info)
            if i % shot_every == 0 and shot < frames:
                img = u.sim.render(
                    mode="rgb_array",
                    camera=cam,
                    cam_config=cam_cfg,
                    width=width,
                    height=height,
                )
                path = out_dir / f"run{run_idx}_frame{shot:02d}.png"
                Image.fromarray(np.asarray(img)).save(path)
                logger.info("saved %s (t=%.2fs, gate=%s)", path, curr_time, obs["target_gate"])
                shot += 1
            if terminated or truncated or finished:
                break
            i += 1
        ctrl.episode_callback()
        gates = obs["target_gate"]
        done = gates == -1
        gates = len(cfg.env.track.gates) if done else gates
        logger.info(
            "episode %d: time=%.2fs finished=%s gates_passed=%s", run_idx, curr_time, done, gates
        )
        ctrl.episode_reset()
        ep_times.append(curr_time if done else None)
    env.close()
    logger.info("screenshots in %s", out_dir)
    return ep_times


if __name__ == "__main__":
    logging.basicConfig()
    logging.getLogger("lsy_drone_racing").setLevel(logging.INFO)
    logger.setLevel(logging.INFO)
    fire.Fire(run, serialize=lambda _: None)

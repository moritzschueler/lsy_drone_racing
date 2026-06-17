---
name: run-lsy-drone-racing
description: Build, run, and screenshot the LSY autonomous drone racing simulation, and run RL training / the test suite. Use when asked to run, launch, simulate, render, screenshot, train, or test the drone racing project (sim.py, the racing env, or the PPO RL agent).
---

# Run LSY Drone Racing

Crazyflie drone-racing simulator (MuJoCo via `crazyflow`) plus a PPO RL stack.
The runnable surfaces are:

- **The simulation** — `scripts/sim.py` flies a controller through a gate track.
  Its built-in `render=true` opens a MuJoCo GUI window (needs a display). The
  agent path uses **`driver.py`**, which forces MuJoCo's offscreen `rgb_array`
  path (`MUJOCO_GL=egl`) and saves PNGs — **no X server / no `xvfb` needed**.
- **RL training** — `python -m lsy_drone_racing.rl.scripts.train`, runs on CPU.
- **Tests** — `pytest`.

Everything runs through **`pixi`** (the only supported runner here — not bare
`python`/venv/conda). Paths below are relative to the repo root
(`/home/philipp/development/lsy_drone_racing`). The driver lives at
`.claude/skills/run-lsy-drone-racing/driver.py`.

## Prerequisites

`pixi` (already at `~/.pixi/bin/pixi`) resolves all Python deps into `.pixi/envs/`.
No `apt-get` needed: headless rendering uses EGL via the bundled MuJoCo +
system `libEGL_mesa.so.0`. `xvfb` is **not** installed and **not** required.

## Run the sim + screenshots (agent path)

This is the path to use. It runs a full episode and drops evenly-spaced PNGs of
the drone flying the track:

```bash
MUJOCO_GL=egl pixi run python .claude/skills/run-lsy-drone-racing/driver.py \
  --config level1.toml --frames 8 --out /tmp/drone_run
```

Logs each saved frame with sim time + current target gate, then per-episode
`time / finished / gates_passed`. Frames land at `/tmp/drone_run/run0_frame*.png`.

The default world camera makes the drone small. For a drone-tracking view that
clearly shows the Crazyflie, pass a named camera:

```bash
MUJOCO_GL=egl pixi run python .claude/skills/run-lsy-drone-racing/driver.py \
  --config level1.toml --frames 6 --camera "track_cam:0" --out /tmp/drone_track
```

Then **open and look at** a mid-flight PNG (e.g. `/tmp/drone_track/run0_frame02.png`).
Useful flags: `--controller <file.py>` (override the config's controller),
`--n_runs N`, `--width/--height`, `--camera` (`-1` world, `0` drone FPV,
`"track_cam:0"`, `"fpv_cam:0"`).

### Headless sanity check (no rendering)

Fastest confirmation the control loop works end-to-end:

```bash
pixi run python scripts/sim.py --config level1.toml --render False
```

Prints `Flight time` / `Finished` / `Gates passed` (level1 + `state_controller.py`
finishes all 4 gates in ~16.5s).

## RL training

> **Do not launch training yourself.** The repo owner runs all RL training
> explicitly because they need the W&B charts and run history. Never start a
> training run (or any command) without an explicit instruction and approval.
> The commands below are reference only — show them, don't execute them.

Tiny CPU smoke (wandb off, 1 iteration — finishes in seconds, writes a checkpoint
to `lsy_drone_racing/rl/checkpoints/<task>/`; delete it after):

```bash
pixi run python -m lsy_drone_racing.rl.scripts.train \
  --task hover --wandb_enabled False --num_envs 64 --total_timesteps 512 \
  --num_eval_iterations 0
```

Tasks: `single_agent_racing` (default), `hover`, `random_trajectory_following`.
Real training drops the overrides (defaults: `num_envs=1024`,
`total_timesteps=1_500_000`, `jax_device=cpu`) and logs to W&B unless
`--wandb_enabled False`. `--train False` evaluates the latest checkpoint instead.

## Tests

```bash
pixi run -e tests pytest -q tests/unit
```

50 unit tests, ~110s. (`tests/integration/` also exists; it's slower.) The
`tests` env adds `pytest`; training and the sim run in the `default` env.

## Run the sim (human path)

`pixi run python scripts/sim.py --config level1.toml` with `render = true` in the
config opens a MuJoCo GUI window. **Useless headless** — it needs a display and
will fail/ hang without one. Use the driver instead.

## Gotchas

- **`level0.toml` / `level2.toml` reference a controller that doesn't exist on
  this branch** (`rl_controller_gates.py`) → sim aborts with
  `AssertionError: Controller file not found`. Use `level1.toml`/`level3.toml`
  (which use `state_controller.py`), or pass `--controller <existing>.py`.
  Available controllers live in `lsy_drone_racing/control/`.
- **`render=true` uses MuJoCo `"human"` mode = a GLFW window.** There's no
  display in this container. Don't try to make `sim.py --render True` work; the
  driver renders `rgb_array` offscreen instead. `MUJOCO_GL=egl` is what makes
  that work headless (the driver sets it as a default, but exporting it is
  belt-and-suspenders).
- The driver computes `ROOT` as `parents[3]` of its own path (repo root). If you
  move the skill directory, fix that line.
- `RuntimeWarning: overflow encountered in cast` from JAX prints on every run —
  it's benign, ignore it.
- The training smoke writes a real `.ckpt`; remove it so it doesn't pollute
  `lsy_drone_racing/rl/checkpoints/`.

## Troubleshooting

- **`Configuration file not found: .../config/levelX.toml`** — you ran the driver
  from outside the repo or moved it; `ROOT` (`parents[3]`) no longer points at
  the repo root.
- **Black / empty PNGs or an EGL error** — `MUJOCO_GL` wasn't `egl`. Prefix the
  command with `MUJOCO_GL=egl`.
- **`ModuleNotFoundError` for flax/optax/wandb** — you're outside `pixi`. Always
  launch via `pixi run ...`.

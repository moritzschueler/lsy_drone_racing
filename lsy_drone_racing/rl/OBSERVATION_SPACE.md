# Racing Observation Space

Overview of the observation fed to the policy for the **single-agent racing task**
([`tasks/single_agent_racing.py`](tasks/single_agent_racing.py)). The raw environment observation is
transformed by a chain of wrappers into a flat **52-dimensional float32 vector** per environment,
expressed entirely in the **drone body frame**.

## Wrapper pipeline

Built in [`make_env`](tasks/single_agent_racing.py) (inner → outer):

```
VecDroneRaceEnv          # raw dict obs: pos, quat, vel, ang_vel, n_gates_passed,
                         #               gate_sequence, gate_sequence_direction,
                         #               gates_pos, gates_quat, gates_visited,
                         #               obstacles_pos, obstacles_visited
  → SpinUpRotors         # takeoff (no obs change)
  → NormalizeActions     # action scaling (no obs change)
  → ActionSmoothnessPenalty # champion action-smoothness penalty; adds `last_action` (4) to the dict
  → ZeroYaw              # zeroes yaw command (no obs change)
  → RelativeRacingObs    # body-frame geometry, next-2 gates as corner positions, drops ang_vel
  → FlattenJaxObservation# concatenate to one float32 vector (alphabetical key order)
```

The body-frame recast lives in `RelativeRacingObs` /
[`_relative_racing_obs`](wrappers/observation.py).

## Components

All vector quantities are expressed in the **drone body frame** (world quantities rotated by
`Rᵀ`, where `R` is the drone's body→world rotation).

| Key | Shape | Dim | Description |
|---|---|---:|---|
| `gates_corners` | (2, 4, 3) | 24 | Next 2 gates, each as the 4 opening-corner positions relative to the drone, body frame: `Rᵀ(corner_world − drone_pos)`. Corners sit at gate-local `(0, ±0.225, ±0.225)` and jointly encode each gate's center, orientation and scale |
| `gates_visited` | (2,) | 2 | Whether each of the next 2 gates has been sensed (else its position is the nominal guess) |
| `grav_body` | (3,) | 3 | Gravity direction in the body frame (unit vector) — the drone's tilt relative to "down" |
| `last_action` | (4,) | 4 | Previous **bounded** action `[roll, pitch, yaw, thrust]` (clipped to [-1, 1]); yaw is always 0 |
| `obstacles_rel_pos` | (4, 3) | 12 | All obstacle positions relative to the drone, body frame: `Rᵀ(obstacle_pos − drone_pos)` |
| `obstacles_visited` | (4,) | 4 | Whether each obstacle has been sensed |
| `vel` | (3,) | 3 | Drone linear velocity, body frame |
| **Total** | | **52** | |

> Obstacle count (4) is track-dependent (`level0.toml`); the gate slice is fixed at the next 2.

## Flat vector layout

`FlattenJaxObservation` concatenates keys in **alphabetical order** (gymnasium's `spaces.Dict`
sorts keys), casting everything to `float32`. Column ranges:

| Range | Field |
|---|---|
| `0:24` | `gates_corners` |
| `24:26` | `gates_visited` |
| `26:29` | `grav_body` |
| `29:33` | `last_action` |
| `33:45` | `obstacles_rel_pos` |
| `45:49` | `obstacles_visited` |
| `49:52` | `vel` |

## Design notes

- **Fully body-frame**: every position, orientation, and velocity is expressed in the drone's
  own frame, and the absolute drone position is dropped. This makes the observation invariant to
  where the track sits in the world *and* to the drone's heading.
- **`grav_body` instead of the world attitude**: in a body-frame observation the full world
  rotation matrix would be redundant — except that a quadrotor thrusts along its body-z axis and
  must know its tilt relative to gravity to stay controlled. `grav_body` is the gravity direction
  in the body frame: the minimal world reference the observation keeps. Yaw-about-vertical is
  intentionally not observable (it is frozen and irrelevant once everything else is relative).
- **Next 2 gates only**: the current target gate plus the following one, selected by resolving the
  next 2 entries of the configured gate order (`n_gates_passed`/`gate_sequence`, neither of which
  is itself part of the observation) into physical gate ids. Sequence positions are clamped to the
  last entry once the track is finished. This keeps the observation size independent of track
  length, and generalizes to a `gate_order` that permutes or revisits gates.
- **Gates as opening corners**: each upcoming gate is given as the body-frame positions of its four
  opening corners (`(0, ±0.225, ±0.225)` in the gate frame) rather than a center + rotation-matrix
  pair. Four corner points encode the gate's center, orientation *and* scale jointly, with no
  quaternion double-cover discontinuity, and line up directly with the gate-opening geometry the
  progress reward uses (`GATE_HALF_EXTENT = 0.225`). Same dim count as the old 6+18 encoding (24).
- **No angular velocity**: the drone's body rates are *not* observed. Under `control_mode="attitude"`
  the 500 Hz onboard controller closes the rate loop, so the 50 Hz policy steers via attitude
  (`grav_body`) and lets the inner loop handle rates; this drops a noisy, fast-changing signal. If
  tight-corner control turns oscillatory, `ang_vel` (3 dims, body frame) is the first thing to re-add.
- **`gates_visited` / `obstacles_visited`**: with `sensor_range` sensing, an object's position
  is the *nominal* track value until the drone gets close enough to sense it, then the *true*
  value. These flags tell the policy which it is currently seeing.

### Known trade-offs / possible follow-ups

- **No absolute altitude (z)**: the drone has no direct ground/ceiling reference; height is only
  implicit via relative gate positions. Re-add a scalar world `z` if altitude drift appears.

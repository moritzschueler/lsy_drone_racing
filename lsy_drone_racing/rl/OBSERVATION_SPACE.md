# Racing Observation Space

Overview of the observation fed to the policy for the **single-agent racing task**
([`tasks/racing.py`](tasks/racing.py)). The raw environment observation is transformed by a
chain of wrappers into a flat **55-dimensional float32 vector** per environment, expressed
entirely in the **drone body frame**.

## Wrapper pipeline

Built in [`make_env`](tasks/racing.py) (inner → outer):

```
VecDroneRaceEnv          # raw dict obs: pos, quat, vel, ang_vel, target_gate,
                         #               gates_pos, gates_quat, gates_visited,
                         #               obstacles_pos, obstacles_visited
  → SpinUpRotors         # takeoff (no obs change)
  → NormalizeActions     # action scaling (no obs change)
  → ActionSmoothnessPenalty # champion action-smoothness penalty; adds `last_action` (4) to the dict
  → ZeroYaw              # zeroes yaw command (no obs change)
  → RelativeRacingObs    # recast to body-frame geometry, next-2 gates, rotation matrices
  → FlattenJaxObservation# concatenate to one float32 vector (alphabetical key order)
```

The body-frame recast lives in `RelativeRacingObs` /
[`_relative_racing_obs`](wrappers/observation.py).

## Components

All vector quantities are expressed in the **drone body frame** (world quantities rotated by
`Rᵀ`, where `R` is the drone's body→world rotation).

| Key | Shape | Dim | Description |
|---|---|---:|---|
| `ang_vel` | (3,) | 3 | Drone angular velocity, body frame |
| `gates_rel_pos` | (2, 3) | 6 | Next 2 gate centers relative to the drone, body frame: `Rᵀ(gate_pos − drone_pos)` |
| `gates_rot` | (2, 9) | 18 | Next 2 gate orientations relative to the drone, flattened 3×3 matrices: `Rᵀ R_gate` (first column = gate +x traversal axis) |
| `gates_visited` | (2,) | 2 | Whether each of the next 2 gates has been sensed (else its position is the nominal guess) |
| `grav_body` | (3,) | 3 | Gravity direction in the body frame (unit vector) — the drone's tilt relative to "down" |
| `last_action` | (4,) | 4 | Previous **bounded** action `[roll, pitch, yaw, thrust]` (clipped to [-1, 1]); yaw is always 0 |
| `obstacles_rel_pos` | (4, 3) | 12 | All obstacle positions relative to the drone, body frame: `Rᵀ(obstacle_pos − drone_pos)` |
| `obstacles_visited` | (4,) | 4 | Whether each obstacle has been sensed |
| `vel` | (3,) | 3 | Drone linear velocity, body frame |
| **Total** | | **55** | |

> Obstacle count (4) is track-dependent (`level0.toml`); the gate slice is fixed at the next 2.

## Flat vector layout

`FlattenJaxObservation` concatenates keys in **alphabetical order** (gymnasium's `spaces.Dict`
sorts keys), casting everything to `float32`. Column ranges:

| Range | Field |
|---|---|
| `0:3` | `ang_vel` |
| `3:9` | `gates_rel_pos` |
| `9:27` | `gates_rot` |
| `27:29` | `gates_visited` |
| `29:32` | `grav_body` |
| `32:36` | `last_action` |
| `36:48` | `obstacles_rel_pos` |
| `48:52` | `obstacles_visited` |
| `52:55` | `vel` |

## Design notes

- **Fully body-frame**: every position, orientation, and velocity is expressed in the drone's
  own frame, and the absolute drone position is dropped. This makes the observation invariant to
  where the track sits in the world *and* to the drone's heading.
- **`grav_body` instead of the world attitude**: in a body-frame observation the full world
  rotation matrix would be redundant — except that a quadrotor thrusts along its body-z axis and
  must know its tilt relative to gravity to stay controlled. `grav_body` is the gravity direction
  in the body frame: the minimal world reference the observation keeps. Yaw-about-vertical is
  intentionally not observable (it is frozen and irrelevant once everything else is relative).
- **Next 2 gates only**: the current target gate plus the following one (selected via the
  internal `target_gate` index, which is *not* part of the observation). Indices are clamped to
  the last gate; once the track is finished (`target_gate = -1`) the index clamps to 0. This
  keeps the observation size independent of track length.
- **Rotation matrices instead of quaternions**: avoids the quaternion double-cover
  discontinuity; the gate's +x traversal axis is directly readable as the first matrix column.
- **`gates_visited` / `obstacles_visited`**: with `sensor_range` sensing, an object's position
  is the *nominal* track value until the drone gets close enough to sense it, then the *true*
  value. These flags tell the policy which it is currently seeing.

### Known trade-offs / possible follow-ups

- **No absolute altitude (z)**: the drone has no direct ground/ceiling reference; height is only
  implicit via relative gate positions. Re-add a scalar world `z` if altitude drift appears.

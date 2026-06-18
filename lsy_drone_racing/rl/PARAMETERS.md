# Racing RL — parameter reference

What every knob in the single-agent racing pipeline does, and which way to turn it. Values shown
are the current task defaults; treat them as a starting point, not gospel.

**Where the parameters live**

| Source | Holds |
| --- | --- |
| [`config.py`](config.py) (`Args`) | All PPO / optimization hyperparameters + the reward-coefficient *defaults* (shared across tasks). |
| [`tasks/racing.py`](tasks/racing.py) (`RACING_CONFIG`, `GATE_HALF_EXTENT`) | Per-task overrides merged over `Args` before `Args.create`, plus the gate-opening half-extent used by the progress term. **This is what the racing runs actually use.** |
| [`../envs/segment_spawn.py`](../envs/segment_spawn.py) (`SegmentSpawnConfig`) | The curriculum (cone-spawn geometry + annealing schedules). |

**Environment constants** (not in the config, but you need them to reason about the rest):
`freq = 50 Hz`, `max_episode_steps = 1500` (= 30 s), `control_mode = "attitude"`,
`sensor_range = 0.7 m`. Gate opening ≈ 0.4 m wide, frame ≈ 0.72 m outer.

---

## 1. Reward shaping (the heart of it)

The per-step reward is the sum of several terms. The env-side terms (progress, bonuses, penalties)
are computed in `racing_reward_components`; the single wrapper term (`d_act`) is added by
`ActionSmoothnessPenalty`. **Every term is logged separately to wandb as `reward/<term>`** —
all in the same units (mean reward contribution per env-step), so they're directly comparable and
sum to the per-step total reward.

> **Reading the component charts.** Dense terms (`progress`, `d_act`, `speed`) fire every step,
> so the chart ≈ their typical per-step size. Sparse/event terms (`gate_bonus`, `finish`, `crash`,
> `timeout`) fire only on the rare step the event happens, so chart value = event value × event rate.
> A small `gate_bonus` chart (e.g. 0.05) with `gate_bonus=5` means passes happen on ~1 % of steps,
> not that a pass is worth 0.05. To recover the per-event size, divide by the event rate
> (e.g. `crash`: −0.010 ÷ (1 / 290 steps) ≈ −3 = `crash_penalty`).

### `progress_coef` (current **1.5**, champion paper λ₁ = 1.0)
Weight on the dense progress term: `progress_coef · (dₜ₋₁ − dₜ)`, the per-step **reduction in distance
to the target gate opening** `d` (see `gate_opening_distance`) — the champion-paper progress reward
(Kaufmann et al. 2023). This is the **workhorse** positive reward: clearly positive whenever the drone
closes on the gate, proportional to the metres covered, with a *constant* gradient right into the
opening (no saturation). The champion policy leans almost entirely on it.
- **Units matter:** `d` is in **metres**, so at cruise (~1–2 m/s, 50 Hz) the per-step reduction is
  ~0.02–0.04 m and `reward/progress ≈ progress_coef · 0.02–0.04`. This is a different scale from the
  old bounded 0–1 potential — hence the coef dropped from ~5 to ~1.5.
- **↑** Stronger, more continuous pull toward gates; the dominant driver. Too high → reckless (dive at
  gates, clip frames) and can drown out the value signal.
- **↓** Weaker forward pull; the policy has less reason to move and tends to loiter/hover.
- **No crossing-drop coupling.** Being distance-reduction (a potential `Φ = −d`), the gate-advance only
  shifts the reference distance by one step's displacement, not by the full coef — so crossing a gate is
  never net-penalized and there is **no** `gate_bonus ≥ progress_coef` requirement (that assertion was
  removed). It is still a potential, so progress can't be farmed by looping.
- **Watch:** `reward/progress` should be **clearly positive** while the drone is racing. Persistently
  ~0 or negative means it's loitering / drifting from its target — fix that before adding sparse
  incentives. (The old bounded potential sat at ~0 by construction; this term should not.)

### `gate_bonus` (current **5.0**)
One-off bonus each time `target_gate` advances (an actual pass). Provides the discrete "go *through*,
not just *near*" incentive that the telescoping progress term can't.
- **↑** Crossing becomes worth more risk. The crash-vs-cross break-even success probability is
  `crash / (gate_bonus + crash)` — at 5/3 that's `3/8 ≈ 0.38`; at 10 it's `3/13 ≈ 0.23`. Lower
  threshold ⇒ the policy attempts crossings even when it's only moderately likely to make it.
- **↓** Crossing has to "pay for itself" via progress alone; with imprecise control the policy backs
  off and the cone-pass rate collapses as exploration anneals.
- **No floor any more.** With the champion distance-reduction progress there is no full-range crossing
  drop to shadow, so the old `gate_bonus ≥ progress_coef` assertion was removed. Size `gate_bonus`
  purely by the crash/cross risk trade-off above.
- **Trap:** a large `gate_bonus` is only dangerous *in combination with* an asymmetric
  `timeout_penalty` (see that). With the directional progress potential (a non-gameable potential,
  see `GATE_HALF_EXTENT`) and symmetric failure penalties, raising it is safe.

### `finish_bonus` (current **10.0**)
Large one-off bonus when the final gate is passed (`target_gate → −1`).
- **↑** Stronger pull to complete the *whole* track vs. stopping after a few gates.
- **↓** Less incentive to finish; the policy may settle for partial laps.
- **Note:** keep it clearly above `gate_bonus` and the failure penalties so finishing always wins.

### `crash_penalty` (current **3.0**)
Penalty when a drone is disabled without finishing (`p_z < 0` floor-sink or gate-frame collision).
The crash also **ends the episode**, so its true cost is `crash_penalty` **plus the forfeited future
return** — usually much larger than 3.
- **↑** More crash-averse → safer but more timid; can suppress crossing attempts entirely.
- **↓** Bolder, but risks the drone flailing/out-of-bounds with no deterrent.

### `timeout_penalty` (current **3.0**)
Penalty when the episode truncates at `max_episode_steps` (30 s) without finishing.
- **Keep it ≈ `crash_penalty` (symmetric).** Both are "failed to finish"; making them equal means the
  policy is indifferent to *how* it fails and just maximizes progress/bonuses before the end.
- **Trap (suicide):** if `timeout_penalty ≫ crash_penalty` (e.g. 30 vs 5), surviving-without-finishing
  becomes worse than crashing, so once a drone exhausts the easy progress it **deliberately crashes**
  (−5) rather than loiter to timeout (−30). This is what made earlier runs "commit suicide."
- **Note:** in practice timeout rarely fires (drones crash at ~290 steps, far short of 1500), so this
  term is mostly latent — but a bad value still distorts the value function. Don't set it large.

### `d_act_coef` (current **0.001**; champion λ₅ ≈ 1e-4 relative to λ₁=1.0)
The **single** champion-style action-smoothness penalty (`ActionSmoothnessPenalty`):
`d_act_coef · ‖clip(aₜ) − clip(aₜ₋₁)‖²`, summed over all action dims (roll/pitch/thrust; yaw is
zeroed). It replaces the old four-term stack (`rpy_coef` attitude-magnitude, `act_coef` thrust-energy,
`d_act_xy_coef`/`d_act_th_coef` split smoothness) with one coefficient — matching the champion reward,
which penalizes only the *change* in command, not attitude or thrust level.
- **Computed on the bounded action** (clipped to [-1, 1], what `NormalizeActions` applies), so a
  high-variance policy can't inflate it and it no longer doubles as an entropy penalty (the old raw-
  output version shrank as exploration annealed, rewarding "fly calm" over "pass gates" — a driver of
  the cone-pass collapse).
- **↑** Smoother commands (good for sim-to-real), but too high and the policy optimizes "stop
  thrashing" instead of "pass gates". Keep it a **whisper** vs `progress` until passing is solid.
- **↓ / 0** Removes the smoothness pressure entirely; safe while you're still chasing gate-passing.
- **What it gave up:** per-channel weighting (thrust vs roll/pitch) and any absolute-attitude /
  thrust-energy regularizer. Reintroduce a split or a small `rpy`-style term *only* if a specific
  pathology appears (thrust bobbing, acrobatic attitudes, energy/thermal limits for sim-to-real).

> `rpy_coef` / `act_coef` / `d_act_th_coef` / `d_act_xy_coef` still exist in `Args` but are used only
> by the **hover / trajectory** tasks (their `AngleReward` + `ActionPenalty` wrappers). Racing ignores
> them and uses `d_act_coef` alone.

### `speed_coef` (current **0.00 — disabled**) / `max_speed` (current **3.0 m/s**) / `speed_penalty_slope` (current **0.15**)
Exponential **speed barrier**, env-side dense term (logged as `reward/speed`). With normalized speed
`u = ‖vel‖ / max_speed`:

```
reward_speed = −speed_coef · ( exp( speed_penalty_slope · u/(1−u) ) − 1 )
```

**Currently disabled** (`speed_coef = 0`): isolation runs showed it suppressed forward flight (a convex
barrier penalizes speed *variance*, not just its level, choking exploration toward higher cruise), so
it was turned off while the propulsion problem is being chased. Originally introduced because the
bounded progress potential let the policy race **too fast to learn the track**.
An earlier `(u)**power` ramp barely bit below the limit; this **diverges toward `max_speed`** so it is
an effective ceiling the drone cannot exceed (no discontinuous clip).
- **`max_speed`** is the asymptote: the penalty is 0 at rest, grows exponentially, and blows up at
  `‖vel‖ = max_speed`. (At `slope = 0.15` the unweighted barrier ≈ 0.16 at 0.5×, 0.42 at 0.7×, 2.9 at
  0.9×, then a wall.) The exponent is **clamped** (`_SPEED_ARG_CAP` in `racing.py`) so the penalty
  saturates at a large-but-finite value (`expm1(6) ≈ 402`, times `speed_coef`) instead of `inf` —
  required for float32 / advantage stability; `speed >= max_speed` is clipped onto the wall.
- **`speed_penalty_slope`** sets where the wall rises: **↑** → earlier/steeper (firmer, lower effective
  ceiling); **↓** → the drone can get closer to `max_speed` before the penalty bites.
- **`speed_coef`** trades against `progress_coef`. It's **dense** (every step), so even a small value
  competes with per-step progress (≈ `progress_coef · ΔΦ`, ~0.01–0.05/step at cruise on this 50 Hz
  track). Watch `reward/speed` vs `reward/progress` and the `diagnostics/vel_along` chart: if the drone
  creeps, lower `speed_coef` / `speed_penalty_slope` or raise `max_speed`; if it still dives through
  gates, raise them or lower `max_speed`. Set `speed_coef = 0` to disable entirely.
- Visualised in [`parameter_visualizations.ipynb`](parameter_visualizations.ipynb) §6 (drag the slope).

---

## 2. `GATE_HALF_EXTENT` (current **0.225 m**)

Half-extent of the square gate opening used by the progress reward (`gate_opening_distance`). The
champion-paper progress term rewards the per-step reduction of the distance to this *cuboid opening*
(not a single centre point):

```
distance = sqrt(along² + max(|y| − h, 0)² + max(|z| − h, 0)²)   # h = GATE_HALF_EXTENT
progress = progress_coef · (distanceₜ₋₁ − distanceₜ)
```

(gate frame: `along` = traversal-axis gap to the gate plane, `y`/`z` span the opening; lateral offsets
inside the opening clamp to 0, so any crossing point counts equally).

- **Unbounded, constant gradient.** `distance` grows ~linearly with separation and is 0 in the opening,
  so closing on the gate banks clearly-positive progress proportional to the metres covered, with the
  same per-metre reward far out and right at the opening (no near-gate saturation). This replaces the
  old bounded `exp` potential whose gain vanished near the gate and whose crossing drop pushed
  `reward/progress` to ~0.
- **No entry/exit asymmetry.** Crossing the gate the *right* way (−x → +x) is enforced by `gate_passed`
  / the gate-advance and the crash penalty — matching the paper — not by an `exit_scale` term (removed).
- **Potential ⇒ non-farmable.** `Φ = −distance` is a deterministic function of state, so progress
  telescopes and can't be farmed by looping. `gate_bonus` confirms the actual crossing.
- **Keep in sync** with the `gate_size` used by `gate_passed` in `race_core._update_target_gates`
  (currently `(0.45, 0.45)` → `h = 0.225`) so "inside the opening" matches the env's pass detection.

> **Removed knobs:** `progress_reach`, `progress_sharpness`, `exit_scale` belonged to the old bounded
> potential and no longer exist. The distance-reduction term has no length scales to tune — only
> `progress_coef` (its weight) and `GATE_HALF_EXTENT` (the opening size).

- **`GATE_HALF_EXTENT` (`h`)**: shrinking `h` tightens the corridor of equally-good crossing points
  (demands a more accurate lineup); enlarging it loosens it. Keep it matched to `gate_passed`'s box.

---

## 3. Curriculum — `SegmentSpawnConfig` ([`segment_spawn.py`](../envs/segment_spawn.py))

On (auto)reset, a fraction of drones are respawned in a chosen gate's approach cone so every gate is
practiced from varied, recoverable poses. Two cosine schedules anneal it over training progress
`τ = global_step / total_timesteps`.

| Param | Current | Meaning | ↑ effect / ↓ effect |
| --- | --- | --- | --- |
| `gate_offset` | 0.1 | **Segment length** def: predecessor gate's exit offset (⚠ *not* the reward `GATE_OFFSET`). | ↑ longer segments / ↓ shorter. |
| `d_min` | 0.25 | Min standoff from the gate (always leave runway), m. | ↑ farther minimum spawn / ↓ closer (harder, may spawn in the frame). |
| `d_max_cap` | 1.5 | Global cap on segment length, m. | ↑ allows farther spawns / ↓ keeps spawns near gates. |
| `theta_max` | 0.4 (~23°) | Cone half-angle at `kappa=1`. | ↑ more off-axis (harder) / ↓ more on-axis (easier). |
| `margin` | 0.30 | Required horizontal clearance to obstacles, m. | ↑ safer spawns / ↓ riskier, may spawn near obstacles. |
| `z_min`,`z_max` | 0.20, 2.0 | Spawn altitude floor/ceiling, m. | widen ↔ narrow the vertical spawn band. |
| `n_candidates` | 12 | Rejection-sampling budget per env for clearance. | ↑ better-cleared spawns, slower / ↓ faster, more rejects. |
| **`a0`, `a1`** | 0.05, 0.70 | τ-window over which **cone size `kappa`** ramps `kappa_min → 1`. | Earlier/wider window ⇒ difficulty ramps sooner. |
| `kappa_min` | 0.10 | Cone-size floor before `a0`. | ↑ starts harder / ↓ starts trivially easy (spawns hug the gate axis). |
| **`b0`, `b1`** | 0.25, 0.85 | τ-window over which **true-start probability `p_start`** ramps `p_start_min → p_start_max`. | Earlier window ⇒ converge to the real start distribution sooner. |
| `p_start_min` | 0.05 | Floor fraction of episodes starting from the *true race start*. | ↑ more full-track practice early (but harder) / ↓ almost all cone spawns early. |
| `p_start_max` | 0.80 | Final true-start fraction. | ↑ ends closer to deployment distribution. |
| **`c0`, `c1`** | 0.0, 0.50 | τ-window over which **cone-spawn initial speed `v0`** anneals `v0_max → 0` (momentum crutch through the gate; cone spawns only). | Later/wider window ⇒ crutch withdrawn more slowly, longer to learn self-propulsion. |
| `v0_max` | 0.5 | Through-gate spawn speed at `τ=0`, m/s. | ↑ stronger early momentum crutch (masks propulsion learning) / 0 ⇒ drone must self-propel from rest. |

> **Diagnostic tie-in:** `kappa` is *flat at `kappa_min`* until `τ = a0` (= 2.5M steps at 50M total),
> so during the first ~2.5M steps spawns are point-blank (0.25–0.38 m, ≤3.4° off-axis) and *static*.
> If `cone_gate_pass_rate` collapses in that window, it's genuine policy degradation, **not** the
> curriculum getting harder.

---

## 4. PPO / optimization

| Param | Current | Meaning | ↑ effect | ↓ effect |
| --- | --- | --- | --- | --- |
| `gamma` | 0.99 | Discount. Effective horizon ≈ `1/(1−γ)/freq` s. At 50 Hz: 0.99→2 s, 0.96→0.5 s. | Longer horizon; sparse bonuses propagate back (needed for goal-reaching). Too high ⇒ high-variance value targets. | Short-sighted; gate/finish bonuses get discounted into irrelevance on the approach. |
| `gae_lambda` | 0.97 | GAE bias/variance trade-off. | Lower bias, higher variance. | Higher bias, lower variance. |
| `learning_rate` | 3e-4 | Adam step size (annealed to 0 if `anneal_lr`). | Faster but less stable / can diverge. | Slower, steadier. |
| `anneal_lr` | true | Linearly decay LR to 0 over training. | — | constant LR. |
| `ent_coef` | 0.007 | Entropy bonus (exploration). | More exploration; delays premature convergence to risk-averse local optima (the collapse). Too high ⇒ noisy, imprecise control. | Sharper, more deterministic policy sooner — can lock in "don't cross" before the skill forms. |
| `anneal_ent_coef` | false | Linearly anneal `ent_coef` → 0 over training (like `anneal_lr`). | explores early, sharpens late. | constant entropy bonus throughout. |
| `clip_coef` (ε) | 0.2 | PPO ratio clip (champion: 0.2). | Bigger policy steps / less conservative. | Smaller, safer updates. |
| `clip_vloss` | true | Clip the value loss too. | — | unclipped value loss. |
| `vf_coef` | 0.7 | Value-loss weight. | Critic learns faster, may crowd out policy. | Weaker critic. |
| `max_grad_norm` | 1.5 | Gradient-norm clip. | Looser (bigger steps) / tighter (more stable). | — |
| `target_kl` | 0.03 | Early-stop the epoch loop if mean approx-KL exceeds this. | Allows bigger per-iter policy change. | More conservative, fewer effective updates. |
| `norm_adv` | true | Normalize advantages per batch. | — | raw advantages. |
| `update_epochs` | 4 | PPO passes over each rollout. | More reuse per batch (faster, risk of overfit/large KL). | Less reuse, more on-policy. |
| `num_minibatches` | 8 | Minibatches per epoch ⇒ `minibatch_size = num_envs·num_steps / 8`. | More, smaller updates. | Fewer, larger updates. |

## 5. Scale / rollout

| Param | Current | Meaning | Notes |
| --- | --- | --- | --- |
| `total_timesteps` | 50M | Training budget. Also sets `τ` ⇒ **drives the curriculum schedules.** Changing it rescales when `kappa`/`p_start` ramp. | Halving it makes the curriculum ramp twice as fast in wall-clock. |
| `num_envs` | 1024 | Parallel envs. | ↑ throughput + batch size, more memory. |
| `num_steps` | 128 | Rollout length per env per update ⇒ `batch_size = num_envs·num_steps`. | ↑ longer horizon per update, more on-policy, more memory. |

## 6. Stopping

There is no rule-based early stopping — the run trains for the full `total_timesteps`. The best
checkpoint (highest recent mean true-start **gates-passed**) is snapshotted throughout, and **Ctrl-C**
stops gracefully at the end of the current iteration, writing out that best checkpoint.

---

## 7. Diagnostics to watch (logged in [`ppo.py`](ppo.py))

- **`reward/<term>`** — per-component reward (§1). The single best view of *what the policy is
  actually optimizing*. If reward climbs while `gate_bonus`/`finish` fall, a dense term is being
  chased instead of crossing — read off which one.
- **`train/cone_gate_pass_rate`** — fraction of cone-spawned episodes that pass their spawn gate.
  Isolates curriculum skill from full-track skill. **Peak-then-collapse = a reward exploit or
  risk-averse convergence, not slow learning.**
- **`train/gates_passed` / `train/completed`** — true-start (deployment-like) episodes only; the real
  scoreboard. Often ~0 long after cone-passing works, because full-track is much harder.
- **`losses/entropy`** — proxy for action noise. The cone-pass collapse tends to track entropy
  annealing: exploration luck being smoothed away.

## Quick rules of thumb

- **Make `progress` the positive workhorse; keep it positive.** Bonuses are seasoning.
- **`gate_offset = 0`** unless you have a specific reason — positive values invite the fly-around exploit.
- **`crash_penalty ≈ timeout_penalty`** — symmetric failure, no suicide/park incentive.
- **Action-smoothness penalties stay tiny** (~1e-3) until passing is solid; they're for polish, not discovery.
- **A peak-then-collapse in `cone_gate_pass_rate` is a red flag** — instrument the `reward/*` components before touching coefficients.

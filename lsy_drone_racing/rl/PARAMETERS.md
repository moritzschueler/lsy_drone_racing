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
are computed in `racing_reward_components`; the wrapper terms (`rpy`, `act`, `d_act_*`) are added by
`AngleReward` / `ActionPenalty`. **Every term is logged separately to wandb as `reward/<term>`** —
all in the same units (mean reward contribution per env-step), so they're directly comparable and
sum to the per-step total reward.

> **Reading the component charts.** Dense terms (`progress`, `rpy`, `act`, `d_act_*`) fire every step,
> so the chart ≈ their typical per-step size. Sparse/event terms (`gate_bonus`, `finish`, `crash`,
> `timeout`) fire only on the rare step the event happens, so chart value = event value × event rate.
> A small `gate_bonus` chart (e.g. 0.05) with `gate_bonus=15` means passes happen on ~0.3 % of steps,
> not that a pass is worth 0.05. To recover the per-event size, divide by the event rate
> (e.g. `crash`: −0.017 ÷ (1 / 290 steps) ≈ −5 = `crash_penalty`).

### `progress_coef` (current **3.0**, champion paper λ₁ = 1.0)
Weight on the dense progress term: `progress_coef · (Φₜ − Φₜ₋₁)`, the per-step increase of the
**directional gate potential** `Φ` (see `gate_progress_potential`, and `progress_reach` /
`progress_sharpness`). This is meant to be the **workhorse** positive reward — the champion racing
policy leans almost entirely on it (no gate bonus at all).
- **↑** Stronger, more continuous pull toward gates; the dominant driver. Too high can make the drone
  reckless (dive at gates, clip frames) and can drown out the value signal.
- **↓** Weaker forward pull; the policy has less reason to move and tends to loiter/hover.
- **Coupling:** `Φ ∈ (−1, 1]`, so the one-step drop as the drone crosses the gate plane (entry side
  high → exit side low) is up to `2 · progress_coef`. `gate_bonus` must stay `≥ 2 · progress_coef`
  (asserted in `build_racing_reward`) or crossing a gate is net-penalized. **Re-check this whenever
  you retune `progress_coef`.**
- **Watch:** `reward/progress` no longer telescopes to net displacement; the crossing drop is real and
  shadowed by `gate_bonus`. A persistently **negative** `reward/progress` outside crossings still means
  the drone is drifting away from its target — fix that before adding sparse incentives.

### `gate_bonus` (current **15.0**)
One-off bonus each time `target_gate` advances (an actual pass). Provides the discrete "go *through*,
not just *near*" incentive that the telescoping progress term can't.
- **↑** Crossing becomes worth more risk. The crash-vs-cross break-even success probability is
  `crash / (gate_bonus + crash)` — at 15/5 that's `5/20 = 0.25`; at 30 it's `5/35 ≈ 0.14`. Lower
  threshold ⇒ the policy attempts crossings even when it's only moderately likely to make it.
- **↓** Crossing has to "pay for itself" via progress alone; with imprecise control the policy backs
  off and the cone-pass rate collapses as exploration anneals.
- **Floor:** `gate_bonus` must be `≥ 2 · progress_coef` (asserted in `build_racing_reward`) so it
  shadows the one-step potential drop at a crossing; at the current `progress_coef = 3` that floor is
  6, well below 15.
- **Trap:** a large `gate_bonus` is only dangerous *in combination with* an asymmetric
  `timeout_penalty` (see that). With the directional progress potential (a non-gameable potential,
  see `GATE_HALF_EXTENT`) and symmetric failure penalties, raising it is safe.

### `finish_bonus` (current **30.0**)
Large one-off bonus when the final gate is passed (`target_gate → −1`).
- **↑** Stronger pull to complete the *whole* track vs. stopping after a few gates.
- **↓** Less incentive to finish; the policy may settle for partial laps.
- **Note:** keep it clearly above `gate_bonus` and the failure penalties so finishing always wins.

### `crash_penalty` (current **5.0**, matches champion paper)
Penalty when a drone is disabled without finishing (`p_z < 0` floor-sink or gate-frame collision).
The crash also **ends the episode**, so its true cost is `crash_penalty` **plus the forfeited future
return** — usually much larger than 5.
- **↑** More crash-averse → safer but more timid; can suppress crossing attempts entirely.
- **↓** Bolder, but risks the drone flailing/out-of-bounds with no deterrent.

### `timeout_penalty` (current **5.0**)
Penalty when the episode truncates at `max_episode_steps` (30 s) without finishing.
- **Keep it ≈ `crash_penalty` (symmetric).** Both are "failed to finish"; making them equal means the
  policy is indifferent to *how* it fails and just maximizes progress/bonuses before the end.
- **Trap (suicide):** if `timeout_penalty ≫ crash_penalty` (e.g. 30 vs 5), surviving-without-finishing
  becomes worse than crashing, so once a drone exhausts the easy progress it **deliberately crashes**
  (−5) rather than loiter to timeout (−30). This is what made earlier runs "commit suicide."
- **Note:** in practice timeout rarely fires (drones crash at ~290 steps, far short of 1500), so this
  term is mostly latent — but a bad value still distorts the value function. Don't set it large.

### `rpy_coef` (current **0.001**)
Penalty on attitude magnitude: `rpy_coef · ‖rpy‖`. **No analogue in the champion reward** (their
attitude shaping is the perception term, which we don't have).
- **↑** Encourages level flight; but tilting *is* how a quad translates, so a high value fights the
  motion needed to reach gates (it was a quiet contributor to the early "won't move" behavior).
- **↓ / 0** Frees the drone to tilt and accelerate. Safe to keep tiny or zero for this task.

### `act_coef` (current **0.001**)
Energy penalty on the thrust command: `act_coef · thrust²`.
- **↑** Pushes thrust toward the normalized midpoint — which is **below hover**, so a high value makes
  the drone sink to the floor. This was part of the original altitude collapse; we zeroed it, now a
  whisper.
- **↓ / 0** No thrust bias. Prefer near-zero; if you want energy economy, penalize *deviation from
  hover*, not raw thrust².

### `d_act_xy_coef`, `d_act_th_coef` (current **0.001** each; champion λ₅ ≈ 1e-4 relative to λ₁=1.0)
Action-smoothness penalties: `d_act_xy · ‖Δ(roll,pitch)‖²` and `d_act_th · Δthrust²`. The champion
paper **does** use this (`λ₅‖aₜ − aₜ₋₁‖²`) — but as a whisper-level regularizer, ~4 orders of
magnitude below progress.
- **↑** Smoother commands (good for sim-to-real), but at 0.1 these *dominated* the reward (~−1.0/step,
  ~20× the gate signal) and the policy optimized "stop thrashing" instead of "pass gates" — the
  cone-pass collapse. They're computed on the **raw, unbounded** policy output (outside
  `NormalizeActions`), so they scale with policy variance and act like a second entropy penalty.
- **↓ / 0** Removes that distraction; needed for the task to be learnable. Reintroduce *small*
  (~1e-3 × progress contribution) only once gate-passing is solid, for deployment smoothness.
- **Structural fix (todo):** move `ActionPenalty` *inside* `NormalizeActions` so it penalizes the
  bounded, applied action instead of the raw output. Also changes what `last_action` means in the
  obs, so validate carefully.

### `speed_coef` (current **0.05**) / `max_speed` (current **3.0 m/s**) / `speed_penalty_slope` (current **0.3**)
Exponential **speed barrier**, env-side dense term (logged as `reward/speed`). With normalized speed
`u = ‖vel‖ / max_speed`:

```
reward_speed = −speed_coef · ( exp( speed_penalty_slope · u/(1−u) ) − 1 )
```

Introduced because the bounded progress potential lets the policy race **too fast to learn the track**.
An earlier `(u)**power` ramp barely bit below the limit; this **diverges toward `max_speed`** so it is
an effective ceiling the drone cannot exceed (no discontinuous clip).
- **`max_speed`** is the asymptote: the penalty is 0 at rest, grows exponentially, and blows up at
  `‖vel‖ = max_speed`. (At `slope = 0.3`: ~0.018·`speed_coef`-scale at 0.5×, ~0.05 at 0.7×, ~0.7 at
  0.9×, then a wall.) The exponent is **clamped** (`_SPEED_ARG_CAP` in `racing.py`) so the penalty
  saturates at a large-but-finite value (~`20` at `speed_coef = 0.05`) instead of `inf` — required for
  float32 / advantage stability; `speed >= max_speed` is clipped onto the wall.
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

Half-extent of the square gate opening used by the progress potential (`gate_progress_potential`).
The potential is built on the distance to this *cuboid opening* (not a single point):

```
distance = sqrt(along² + max(|y| - h, 0)² + max(|z| - h, 0)²)   # h = GATE_HALF_EXTENT
```

(gate frame: `along` = traversal-axis gap to the gate plane, `y`/`z` span the opening). The
entry/exit asymmetry is folded into the *distance* — the `along` coordinate is inflated by
`exit_scale` on the exit side — and the potential is a non-negative blend of two length scales:

```
along_eff = along            if along ≤ 0   (entry side)
along_eff = exit_scale·along if along > 0   (exit side — counts as farther away)
distance  = sqrt(along_eff² + max(|y| − h, 0)² + max(|z| − h, 0)²)   # h = GATE_HALF_EXTENT
Φ         = 0.5·exp(−distance / progress_reach) + 0.5·exp(−distance / progress_sharpness)
```

- **Peak at the gate plane.** `Φ ∈ (0, 1]` and is maximal *at* the opening (`distance = 0`) — which is
  exactly where the target gate advances. So a forward traversal climbs monotonically to the peak and
  banks **net-positive** progress over the whole segment, instead of the old directional ±1 potential
  whose peak sat on the entry side and trough on the exit side, making every successful crossing
  net-*negative* (the bug this replaced). See the design history in `parameter_visualizations.ipynb`.
- **Through-gate funnel.** The exit side decays `exit_scale`× faster, so skirting past the frame or
  sitting on the wrong side reads as low potential and the drone is pulled around to the entry side.
- **Two length scales** (`progress_reach`, `progress_sharpness`) decouple the far-field reach from the
  tight near-gate funnel — a single exponential could not do both.
- Being a deterministic function of state, `Φ` is a **potential**: progress cannot be farmed by
  looping. `gate_bonus` confirms the actual crossing (and shadows the crossing-step drop; see
  `progress_coef`).
- **Keep in sync** with the `gate_size` used by `gate_passed` in `race_core._update_target_gates`
  (currently `(0.45, 0.45)` → `h = 0.225`) so "inside the opening" matches the env's pass detection.

### `progress_reach` (current **2.0 m**) / `progress_sharpness` (current **0.3 m**)
- **`progress_reach`** sets how far the far-field still pulls. Size it to the **largest gate-to-gate
  gap** so there is no flat dead zone between gates. On the current track the worst nominal gap is
  G2→G3 ≈ 2.34 m, ~2.8 m with position randomization — `2.0 m` keeps `Φ ≈ 0.25` (clear gradient) even
  at 2.8 m. Too small → the drone has to "find" the next gate by chance; too large → over-flat, weak
  per-step signal.
- **`progress_sharpness`** sets how close to the gate the directional term takes over. Smaller →
  the entry-vs-exit funnel is tighter to the opening (less directional bias far out); larger → the
  drone is steered onto the correct approach line from further away, at the cost of penalizing
  off-axis positions sooner.
  Shrinking `h` tightens the corridor (demands more accurate lineup); enlarging it loosens it.

---

## 3. Curriculum — `SegmentSpawnConfig` ([`segment_spawn.py`](../envs/segment_spawn.py))

On (auto)reset, a fraction of drones are respawned in a chosen gate's approach cone so every gate is
practiced from varied, recoverable poses. Two cosine schedules anneal it over training progress
`τ = global_step / total_timesteps`.

| Param | Current | Meaning | ↑ effect / ↓ effect |
| --- | --- | --- | --- |
| `gate_offset` | 0.5 | **Segment length** def: predecessor gate's exit offset (⚠ *not* the reward `GATE_OFFSET`). | ↑ longer segments / ↓ shorter. |
| `d_min` | 0.25 | Min standoff from the gate (always leave runway), m. | ↑ farther minimum spawn / ↓ closer (harder, may spawn in the frame). |
| `d_max_cap` | 1.5 | Global cap on segment length, m. | ↑ allows farther spawns / ↓ keeps spawns near gates. |
| `theta_max` | 0.6 (~34°) | Cone half-angle at `kappa=1`. | ↑ more off-axis (harder) / ↓ more on-axis (easier). |
| `margin` | 0.30 | Required horizontal clearance to obstacles, m. | ↑ safer spawns / ↓ riskier, may spawn near obstacles. |
| `z_min`,`z_max` | 0.20, 2.0 | Spawn altitude floor/ceiling, m. | widen ↔ narrow the vertical spawn band. |
| `n_candidates` | 12 | Rejection-sampling budget per env for clearance. | ↑ better-cleared spawns, slower / ↓ faster, more rejects. |
| **`a0`, `a1`** | 0.05, 0.50 | τ-window over which **cone size `kappa`** ramps `kappa_min → 1`. | Earlier/wider window ⇒ difficulty ramps sooner. |
| `kappa_min` | 0.10 | Cone-size floor before `a0`. | ↑ starts harder / ↓ starts trivially easy (spawns hug the gate axis). |
| **`b0`, `b1`** | 0.40, 0.85 | τ-window over which **true-start probability `p_start`** ramps `p_start_min → p_start_max`. | Earlier window ⇒ converge to the real start distribution sooner. |
| `p_start_min` | 0.05 | Floor fraction of episodes starting from the *true race start*. | ↑ more full-track practice early (but harder) / ↓ almost all cone spawns early. |
| `p_start_max` | 0.75 | Final true-start fraction. | ↑ ends closer to deployment distribution. |

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
| `ent_coef` | 0.01 | Entropy bonus (exploration). | More exploration; delays premature convergence to risk-averse local optima (the collapse). Too high ⇒ noisy, imprecise control. | Sharper, more deterministic policy sooner — can lock in "don't cross" before the skill forms. |
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

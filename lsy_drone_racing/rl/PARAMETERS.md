# Racing RL — parameter reference

What every knob in the single-agent racing pipeline does, and which way to turn it. Values shown
are the current task defaults; treat them as a starting point, not gospel.

**Where the parameters live**

| Source | Holds |
| --- | --- |
| [`config.py`](config.py) (`Args`) | All PPO / optimization hyperparameters + the reward-coefficient *defaults* (shared across tasks). |
| [`tasks/single_agent_racing.py`](tasks/single_agent_racing.py) (`RacingArgs`, `GATE_HALF_EXTENT`) | The `RacingArgs(Args)` subclass whose field defaults override `Args` for racing, plus the gate-opening half-extent used by the progress term. **This is what the racing runs actually use** (CLI `--flag`s still layer on top). |

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

> **Reading the component charts.** Dense terms (`progress`, `d_act`, `speed`, `time`) fire every
> step, so the chart ≈ their typical per-step size. Sparse/event terms (`gate_bonus`, `finish`,
> `crash`, `timeout`) fire only on the rare step the event happens, so chart value = event value ×
> event rate. A small `gate_bonus` chart (e.g. 0.2) with `gate_bonus=20` means passes happen on ~1 %
> of steps, not that a pass is worth 0.2. To recover the per-event size, divide by the event rate
> (e.g. `crash`: −0.017 ÷ (1 / 290 steps) ≈ −5 = `crash_penalty`).

### `progress` (current **`("champion", 5.0)`**, champion paper λ₁ = 1.0)
The dense progress term is **swappable**: `progress = (variant, coef)` selects which per-gate potential
`Φ` is used, and `coef` weights it. The per-step reward is always the telescoping increase
`coef · (Φₜ − Φₜ₋₁)` (measured against the gate that was the target at the *start* of the step), so it
is a true potential — non-farmable by looping — for every variant. Variants live in
[`tasks/progress_variants.py`](tasks/progress_variants.py) (`PROGRESS_VARIANTS`); per-variant shape
params live in `progress_params` (a dict that **always carries every variant's knobs**, so switching the
active variant never drops a param). CLI: `--progress champion,5.0` (and e.g.
`--progress asymmetric,4.0`).

The three shipped variants:
- **`champion`** (default) — `Φ = −gate_opening_distance`: the per-step **reduction in distance to the
  target gate opening** `d` (the champion-paper reward, Kaufmann et al. 2023). The **workhorse**
  positive reward: clearly positive whenever the drone closes on the gate, proportional to the metres
  covered, with a *constant* gradient right into the opening (no saturation). `Φ ≤ 0`, unbounded.
- **`asymmetric`** — a non-negative through-gate funnel (`Φ ∈ (0, 1]`) peaking at the opening and
  decaying *faster* on the already-passed (exit) side (params `reach`, `sharpness`, `exit_scale` in
  `progress_params["asymmetric"]`, defaults `2.0 / 0.3 / 3.0`). The old "bounded potential".
- **`fancy`** — a blended angle/distance potential (exponential distance bump near the opening handing
  off to a through-gate alignment term, `gamma_angle = exp(−2·distance)`); no tunable shape params.

`coef` (the workhorse weight, currently **5.0** with `champion`):
- **Units matter:** `d` is in **metres**, so at cruise (~1–2 m/s, 50 Hz) the per-step reduction is
  ~0.02–0.04 m and `reward/progress ≈ progress_coef · 0.02–0.04` (≈ 0.1–0.2/step at the current 5.0).
  This is a different scale from the old bounded 0–1 potential, so the metres-based term is weighted
  heavily (the champion leans almost entirely on progress).
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

### `gate_bonus` (current **20.0**)
One-off bonus each time `target_gate` advances (an actual pass). Provides the discrete "go *through*,
not just *near*" incentive that the telescoping progress term can't.
- **↑** Crossing becomes worth more risk. The crash-vs-cross break-even success probability is
  `crash / (gate_bonus + crash)` — at 5/5 that's `5/25 = 0.20`; raising `gate_bonus` lowers it
  further. Lower threshold ⇒ the policy attempts crossings even when it's only moderately likely to
  make it.
- **↓** Crossing has to "pay for itself" via progress alone; with imprecise control the policy backs
  off and the gate-pass rate collapses as exploration anneals.
- **No floor any more.** With the champion distance-reduction progress there is no full-range crossing
  drop to shadow, so the old `gate_bonus ≥ progress_coef` assertion was removed. Size `gate_bonus`
  purely by the crash/cross risk trade-off above.
- **Trap:** a large `gate_bonus` is only dangerous *in combination with* an asymmetric
  `timeout_penalty` (see that). With the directional progress potential (a non-gameable potential,
  see `GATE_HALF_EXTENT`) and symmetric failure penalties, raising it is safe.

### `finish_bonus` (current **30.0**)
Large one-off bonus when the final gate is passed (`target_gate → −1`).
- **↑** Stronger pull to complete the *whole* track vs. stopping after a few gates.
- **↓** Less incentive to finish; the policy may settle for partial laps.
- **Note:** keep it clearly above `gate_bonus` and the failure penalties so finishing always wins.

### `crash_penalty` (current **5.0**)
Penalty when a drone is disabled without finishing (`p_z < 0` floor-sink or gate-frame collision).
The crash also **ends the episode**, so its true cost is `crash_penalty` **plus the forfeited future
return** — usually much larger than 5.
- **↑** More crash-averse → safer but more timid; can suppress crossing attempts entirely.
- **↓** Bolder, but risks the drone flailing/out-of-bounds with no deterrent.

### `timeout_penalty` (current **5.0**)
One-off penalty when the episode truncates at `max_episode_steps` (30 s) without finishing.
- **Terminal ⇒ little pace pressure.** Fired once at step 1500 and discounted over the whole horizon
  (`γ=0.99` ⇒ `γ¹⁰⁰⁰ ≈ 4e-5`), it has a near-zero gradient at the moment the policy picks its pace —
  raising it (a 20→5 walk-back happened after 20 did nothing) doesn't make the drone race. The dense
  `time_penalty` (below) now owns pace pressure; `timeout_penalty` just discourages the degenerate
  "idle out the clock" end-state.
- **Keep it ≈ `crash_penalty` (symmetric).** Both are "failed to finish"; equal cost means the policy
  is indifferent to *how* it fails and won't game one against the other.
- **Trap (suicide):** if `timeout_penalty ≫ crash_penalty` (e.g. 30 vs 5), surviving-without-finishing
  becomes worse than crashing, so once a drone exhausts the easy progress it **deliberately crashes**
  (−5) rather than loiter to timeout (−30). This is what made earlier runs "commit suicide."

### `time_penalty` (current **0.03**)
Dense per-step "living"/time cost charged **every step the drone is still actively racing** (not yet
finished); the finishing step and post-finish idle steps are exempt. This is the racing *pace*
pressure that the terminal `timeout_penalty` structurally cannot provide.
- **Why it's needed:** the progress term telescopes — creeping and sprinting to a gate bank the *same*
  cumulative progress — and a single penalty at the horizon is discounted to nothing at decision time.
  A constant per-step cost gives a non-vanishing gradient toward finishing *sooner*, so forward flight
  beats the safe-but-slow basin where the drone hovers/creeps and times out (the failure that
  motivated this term: `diagnostics/vel_mean` decaying to ~0.4 m/s, timeout firing nearly every
  episode).
- **Scale:** at 0.03 it's ≈ the ~0.04/step `progress` while creeping, so dwelling visibly costs reward.
- **↑** Stronger pace pressure (faster racing); too high and the policy may crash early just to escape
  the clock — watch `reward/crash` worsening. **↓ / 0** removes pace pressure; the creep returns.
- **Watch:** `diagnostics/vel_along` / `vel_mean` should *rise*; `reward/timeout` rate should fall
  toward zero as the drone starts finishing. `reward/time ≈ −0.03 × (fraction of steps still racing)`.

### `d_act_coef` (current **0.001**; champion λ₅ ≈ 1e-4 relative to λ₁=1.0)
The **single** champion-style action-smoothness penalty (`ActionSmoothnessPenalty`):
`d_act_coef · ‖clip(aₜ) − clip(aₜ₋₁)‖²`, summed over all action dims (roll/pitch/thrust; yaw is
zeroed). It replaces the old four-term stack (`rpy_coef` attitude-magnitude, `act_coef` thrust-energy,
`d_act_xy_coef`/`d_act_th_coef` split smoothness) with one coefficient — matching the champion reward,
which penalizes only the *change* in command, not attitude or thrust level.
- **Computed on the bounded action** (clipped to [-1, 1], what `NormalizeActions` applies), so a
  high-variance policy can't inflate it and it no longer doubles as an entropy penalty (the old raw-
  output version shrank as exploration annealed, rewarding "fly calm" over "pass gates" — a driver of
  the gate-pass collapse).
- **↑** Smoother commands (good for sim-to-real), but too high and the policy optimizes "stop
  thrashing" instead of "pass gates". Keep it a **whisper** vs `progress` until passing is solid.
- **↓ / 0** Removes the smoothness pressure entirely; safe while you're still chasing gate-passing.
- **What it gave up:** per-channel weighting (thrust vs roll/pitch) and any absolute-attitude /
  thrust-energy regularizer. Reintroduce a split or a small `rpy`-style term *only* if a specific
  pathology appears (thrust bobbing, acrobatic attitudes, energy/thermal limits for sim-to-real).

> `rpy_coef` / `act_coef` / `d_act_th_coef` / `d_act_xy_coef` still exist in `Args` but are used only
> by the **hover / trajectory** tasks (their `AngleReward` + `ActionPenalty` wrappers). Racing ignores
> them and uses `d_act_coef` alone.

### `speed_coef` (current **0.05**) / `speed_threshold` (current **4.0 m/s**)
**Quadratic speed hinge**, env-side dense term (logged as `reward/speed`). A one-sided penalty on speed
above `speed_threshold`:

```
reward_speed = −speed_coef · max(0, ‖vel‖ − speed_threshold)²
```

It is **exactly zero below the threshold** (a true free zone — no penalty, no gradient — so the drone
races freely up to it) and grows **quadratically** above it, so the marginal cost rises with the
overshoot: a little over is cheap, blowing well past it is expensive. Introduced because the policy
raced **too fast (~8 m/s) to learn to brake for the two U-turns** — the hinge caps top speed without
taxing the straights. It replaces the earlier exponential barrier (`max_speed` / `speed_penalty_slope`),
which had no free zone (a small penalty/gradient at *every* speed) and so suppressed forward flight.
- **`speed_threshold`** is where the penalty turns on: ↑ → the drone may cruise faster before it bites;
  ↓ → a lower effective speed cap. The hinge is C¹ at the knee (value *and* gradient are 0 at the
  threshold), so it adds no gradient discontinuity.
- **`speed_coef`** sets both the weight and, because the shape is fixed, the steepness of the wall — it
  trades against `progress_coef`. It's **dense** (every step) and **convex**, so the cost climbs fast
  with overshoot: at `speed_coef = 0.05`, `speed_threshold = 4` the per-step penalty is ~0.05 at 5 m/s,
  ~0.2 at 6 m/s, ~0.8 at 8 m/s. Watch `reward/speed` vs `reward/progress` and the `diagnostics/vel_along`
  / `diagnostics/vel_max` charts: if the drone creeps, lower `speed_coef` or raise `speed_threshold`; if
  it still blows through the U-turns, raise `speed_coef` or lower `speed_threshold`. Set `speed_coef = 0`
  to disable entirely.
- Visualised in [`parameter_visualizations.ipynb`](parameter_visualizations.ipynb) §6.

---

## 2. `GATE_HALF_EXTENT` (current **0.225 m**)

Half-extent of the square gate opening used by the progress reward (`gate_opening_distance`). The
champion-paper progress term rewards the per-step reduction of the distance to this *cuboid opening*
(not a single centre point):

```
distance = sqrt(along² + max(|y| − h, 0)² + max(|z| − h, 0)²)   # h = GATE_HALF_EXTENT
progress = coef · (distanceₜ₋₁ − distanceₜ)                     # champion variant; coef = progress[1]
```

`GATE_HALF_EXTENT` feeds the `champion` and `asymmetric` variants (both use this cuboid opening); the
`fancy` variant uses the same 0.45 m square internally. The formula block below is the `champion` case.


(gate frame: `along` = traversal-axis gap to the gate plane, `y`/`z` span the opening; lateral offsets
inside the opening clamp to 0, so any crossing point counts equally).

- **Unbounded, constant gradient.** `distance` grows ~linearly with separation and is 0 in the opening,
  so closing on the gate banks clearly-positive progress proportional to the metres covered, with the
  same per-metre reward far out and right at the opening (no near-gate saturation). This replaces the
  old bounded `exp` potential whose gain vanished near the gate and whose crossing drop pushed
  `reward/progress` to ~0.
- **No entry/exit asymmetry (champion).** Crossing the gate the *right* way (−x → +x) is enforced by
  `gate_passed` / the gate-advance and the crash penalty — matching the paper — not by an `exit_scale`
  term. (The `asymmetric` variant *does* fold an `exit_scale` asymmetry into its distance; champion does
  not.)
- **Potential ⇒ non-farmable.** `Φ = −distance` is a deterministic function of state, so progress
  telescopes and can't be farmed by looping. `gate_bonus` confirms the actual crossing.
- **Keep in sync** with the `gate_size` used by `gate_passed` in `race_core._update_target_gates`
  (currently `(0.45, 0.45)` → `h = 0.225`) so "inside the opening" matches the env's pass detection.

> **Variant-specific knobs:** `reach`, `sharpness`, `exit_scale` are the `asymmetric` variant's length
> scales — they live in `progress_params["asymmetric"]` (not as flat `Args` fields) and only bite when
> `progress = ("asymmetric", …)`. The default `champion` distance-reduction term has no length scales to
> tune — only its `coef` (`progress[1]`) and `GATE_HALF_EXTENT` (the opening size).

- **`GATE_HALF_EXTENT` (`h`)**: shrinking `h` tightens the corridor of equally-good crossing points
  (demands a more accurate lineup); enlarging it loosens it. Keep it matched to `gate_passed`'s box.

---

## 3. PPO / optimization

| Param | Current | Meaning | ↑ effect | ↓ effect |
| --- | --- | --- | --- | --- |
| `gamma` | 0.99 | Discount. Effective horizon ≈ `1/(1−γ)/freq` s. At 50 Hz: 0.99→2 s, 0.96→0.5 s. | Longer horizon; sparse bonuses propagate back (needed for goal-reaching). Too high ⇒ high-variance value targets. | Short-sighted; gate/finish bonuses get discounted into irrelevance on the approach. |
| `gae_lambda` | 0.97 | GAE bias/variance trade-off. | Lower bias, higher variance. | Higher bias, lower variance. |
| `learning_rate` | 3e-4 | Adam step size (annealed to 0 if `anneal_lr`). | Faster but less stable / can diverge. | Slower, steadier. |
| `anneal_lr` | true | Linearly decay LR to 0 over training. | — | constant LR. |
| `ent_coef` | 0.01 | Entropy bonus (exploration). | More exploration; delays premature convergence to risk-averse local optima (the collapse). Too high ⇒ noisy, imprecise control (and `losses/entropy` *rising* over training = the policy paid to stay diffuse — consider `anneal_ent_coef`). | Sharper, more deterministic policy sooner — can lock in "don't cross" before the skill forms. |
| `anneal_ent_coef` | false | Linearly anneal `ent_coef` → 0 over training (like `anneal_lr`). | explores early, sharpens late. | constant entropy bonus throughout. |
| `clip_coef` (ε) | 0.2 | PPO ratio clip (champion: 0.2). | Bigger policy steps / less conservative. | Smaller, safer updates. |
| `clip_vloss` | true | Clip the value loss too. | — | unclipped value loss. |
| `vf_coef` | 0.7 | Value-loss weight. | Critic learns faster, may crowd out policy. | Weaker critic. |
| `max_grad_norm` | 1.5 | Gradient-norm clip. | Looser (bigger steps) / tighter (more stable). | — |
| `target_kl` | 0.03 | Early-stop the epoch loop if mean approx-KL exceeds this. | Allows bigger per-iter policy change. | More conservative, fewer effective updates. |
| `norm_adv` | true | Normalize advantages per batch. | — | raw advantages. |
| `update_epochs` | 4 | PPO passes over each rollout. | More reuse per batch (faster, risk of overfit/large KL). | Less reuse, more on-policy. |
| `num_minibatches` | 8 | Minibatches per epoch ⇒ `minibatch_size = num_envs·num_steps / 8`. | More, smaller updates. | Fewer, larger updates. |

## 4. Scale / rollout

| Param | Current | Meaning | Notes |
| --- | --- | --- | --- |
| `total_timesteps` | 150M | Training budget (also sets the LR/entropy anneal horizon). | ↑ more training, later anneal / ↓ shorter run, faster anneal. |
| `num_envs` | 1024 | Parallel envs. | ↑ throughput + batch size, more memory. |
| `num_steps` | 128 | Rollout length per env per update ⇒ `batch_size = num_envs·num_steps`. | ↑ longer horizon per update, more on-policy, more memory. |

## 5. Stopping

There is no rule-based early stopping — the run trains for the full `total_timesteps`. The best
checkpoint (highest mean **gates-passed**, logged as `charts/gates_passed`) is tracked in the scan
carry and written out at the end of the run.

---

## 6. Diagnostics to watch (logged in [`ppo.py`](ppo.py))

- **`reward/<term>`** — per-component reward (§1). The single best view of *what the policy is
  actually optimizing*. If reward climbs while `gate_bonus`/`finish` fall, a dense term is being
  chased instead of crossing — read off which one.
- **`train/gates_passed` / `train/completed`** — mean gates passed and full-track completion rate over
  finished episodes; the real scoreboard. **Peak-then-collapse = a reward exploit or risk-averse
  convergence, not slow learning.**
- **`loss/entropy`** — proxy for action noise. A gate-pass collapse tends to track entropy
  annealing: exploration luck being smoothed away.

## Quick rules of thumb

- **Make `progress` the positive workhorse; keep it positive.** Bonuses are seasoning.
- **`gate_offset = 0`** unless you have a specific reason — positive values invite the fly-around exploit.
- **`crash_penalty ≈ timeout_penalty`** — symmetric failure, no suicide/park incentive.
- **Pace pressure is dense, not terminal** — `time_penalty` (per-step) makes the drone race; the
  terminal `timeout_penalty` is discounted to nothing at decision time and won't. If the drone creeps,
  raise `time_penalty`, don't raise `timeout_penalty`.
- **Action-smoothness penalties stay tiny** (~1e-3) until passing is solid; they're for polish, not discovery.
- **A peak-then-collapse in `train/gates_passed` is a red flag** — instrument the `reward/*` components before touching coefficients.

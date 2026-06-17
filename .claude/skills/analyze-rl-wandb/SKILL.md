---
name: analyze-rl-wandb
description: Analyze pasted W&B charts from a racing-PPO training run, connect what the charts show to the live repo state (reward ideas + parameter values), and produce a ranked diagnosis of what works, what doesn't, and what is most likely blocking a better policy. Use when the user pastes wandb screenshots / training curves and asks what they indicate, why training stalled/collapsed, or which knob to change.
---

# Analyze RL training W&B charts

The user pastes screenshots of W&B charts from a single-agent racing PPO run and
wants a diagnosis grounded in the actual code: **what the charts indicate, what's
working, what isn't, and the changes most likely to unblock a better policy.**

Your job is to bridge *charts* ↔ *repo*. The charts show behavior; the repo
(`lsy_drone_racing/rl/`) holds the reward design and the exact coefficients that
produced it. Don't analyze the charts in a vacuum and don't trust any static
list of params/metrics — **read the live config every time** (see step 1).

Paths below are relative to the repo root. **Do not run training or any sim** —
the user runs all training themselves (the only thing you execute here is the
read-only extractor).

## Step 1 — pull the live training context (always first)

```bash
python3 .claude/skills/analyze-rl-wandb/extract_context.py
```

This parses `config.py`, `tasks/racing.py`, and `envs/segment_spawn.py` *from
source* (no imports, nothing heavy) and prints:

- **Effective hyperparameters** = `Args` defaults merged with the task's
  `RACING_CONFIG` overrides (`[*]` = overridden, `[+]` = task-only). The newest
  run uses these (repo HEAD); CLI kwargs would override them.
- **Derived** batch size / iteration count, and **curriculum breakpoints in
  global steps** (e.g. `kappa` cone-size is *frozen* until `a0·total_timesteps`).
- **Live chart inventory** — every metric the code logs, grouped by prefix.
- **PARAMETERS.md coverage** — which knobs have written guidance and which are
  UNDOCUMENTED (analyze those from code + first principles, never from the doc).

It is deliberately non-hardcoded: knobs/metrics added or removed later show up
automatically. Don't maintain your own copy of the parameter list — read this.

## Step 2 — check the pasted charts against the inventory

Compare what the user pasted to the inventory from step 1. For diagnosing the
characteristic racing failure (learns to pass gates, then collapses) the
high-signal charts are:

- `reward/*` — **all** components (they're per-step-mean, comparable, and sum to
  `train/reward`); this is the single best view of *what the policy optimizes*.
- `train/cone_gate_pass_rate` — curriculum skill (isolates it from full-track).
- `train/gates_passed`, `train/completed` — the true-start (deployment) scoreboard.
- `losses/entropy` + `charts/ent_coef` — exploration / annealing.
- `diagnostics/vel_*`, `diagnostics/act_*` — what the drone is physically doing.

If high-signal charts are **missing**, list exactly which ones and ask the user
to paste them — **then proceed** with what's available, marking any conclusion
you couldn't verify. Also note that the multi-run overlays only let you attribute
exact param values to the **newest** line (= repo HEAD); older lines (lower
trailing number, different color) are read **qualitatively** — direction of
change across runs, not exact knob values, unless the user provides them.

## Step 3 — read the interpretive context

- **PARAMETERS.md** (`lsy_drone_racing/rl/PARAMETERS.md`) — what each knob does
  and which way to turn it (the ↑/↓ reasoning). Use it for *meaning*, not for
  current values (it drifts; step 1 has the truth).
- **Project memories** `racing-reward-park-exploit` and
  `racing-takeoff-warm-rotors` (in your memory index) — known failure
  signatures and what has already been ruled out. Don't re-propose dead ends.

### Chart-reading conventions (get these right or the read is wrong)

- **Dense terms** (`progress`, `rpy`, `act`, `d_act_xy`, `d_act_th`, `speed`)
  fire every step → the chart ≈ typical per-step size. **Sparse/event terms**
  (`gate_bonus`, `finish`, `crash`, `timeout`) fire only on the event step →
  chart value = event value × event rate. To recover per-event size, divide by
  the rate (e.g. `crash ≈ −0.017/step` at ~1 crash / 290 steps ⇒ ≈ −5 ≈
  `crash_penalty`). A small `gate_bonus` chart means *passes are rare*, not that
  a pass is cheap.
- Place chart events on the **curriculum timeline** from step 1. A
  `cone_gate_pass_rate` drop *while `kappa` is still frozen* (before
  `a0·total_timesteps`) is genuine policy degradation, **not** the task getting
  harder. After the ramp starts, a dip can just be rising difficulty.
- `reward/progress` going persistently negative outside gate crossings = drone
  drifting away from its target. `train/reward` climbing while
  `gate_bonus`/`finish` fall = a dense term is being farmed instead of crossing —
  read off *which* component grows.
- `losses/explained_variance` near 0/negative = critic not fitting returns;
  `approx_kl` ≫ `target_kl` or high `clipfrac` = updates too aggressive.

## Step 4 — write the diagnosis (output format)

Produce, grounded in the live values from step 1:

1. **What's working** — concrete, cite the chart + the value/trend.
2. **What's not** — the failure signature, with the curriculum-timeline read.
3. **Most likely blockers, ranked** — for each: the mechanism, the chart
   evidence, the responsible knob with its **current value and `file:line`**,
   and whether it matches a known memory signature.
4. **Ranked fixes** — knob → `file:line` → direction (↑/↓ or value) → expected
   chart effect → what to watch to confirm. One primary change at a time;
   respect the documented couplings (e.g. `gate_bonus ≥ 2·progress_coef`).
5. **Missing context** — charts or run configs that would sharpen the call.

Be honest about uncertainty: if two hypotheses fit the same charts, say so and
name the chart that would disambiguate.

## Gotchas

- **`RACING_CONFIG` is an *annotated* assignment** (`RACING_CONFIG: dict = {...}`).
  The merged values it produces (`progress_coef=5`, `gate_bonus=10`, …) differ
  from both `Args` defaults *and* from PARAMETERS.md. Always take values from the
  extractor, which handles the merge — eyeballing the two files by hand is the
  classic mistake (PARAMETERS.md itself is currently stale on ~10 coefficients).
- **The newest run = repo HEAD.** If the user says the pasted run is older or used
  CLI overrides, the extractor's values are wrong for that line — ask for its config.
- Reward components are logged as `reward/<term>` but stashed in code as
  `rew/<term>`; peak speed is `max/vel` → charts as `diagnostics/vel_max`. The
  extractor already normalizes these to the on-W&B names.
- `total_timesteps` drives the curriculum schedule, so changing it rescales every
  breakpoint in step 1's output. Re-run the extractor if the run used a different
  budget.

## Troubleshooting

- **Extractor shows no `[*]` tags / suspiciously default values** — the
  `RACING_CONFIG` parse failed (e.g. someone changed it to a non-literal). Open
  `tasks/racing.py` and read `RACING_CONFIG` directly.
- **A knob in the charts isn't in the extractor output** — it's a CLI-only
  override or brand-new; ask the user for the value.

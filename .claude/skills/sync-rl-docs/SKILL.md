---
name: sync-rl-docs
description: Sync the racing-RL reference docs (PARAMETERS.md, OBSERVATION_SPACE.md) to the live code so the documented "current" values, dims, and knob lists never drift from what training actually uses. Use after ANY change to a reward coefficient, PPO/optimization hyperparameter, curriculum (SegmentSpawnConfig) value, env constant, or the observation pipeline — and proactively whenever the user asks for such a change.
---

# Keep the racing-RL reference docs in sync with the code

Two hand-written docs describe the racing pipeline and **drift from the code constantly**:

- [`lsy_drone_racing/rl/PARAMETERS.md`](../../../lsy_drone_racing/rl/PARAMETERS.md) — every knob, what it
  does, which way to turn it, and its **current value**.
- [`lsy_drone_racing/rl/OBSERVATION_SPACE.md`](../../../lsy_drone_racing/rl/OBSERVATION_SPACE.md) — the
  observation dims, keys, flatten order, and wrapper pipeline.

Your job when this skill runs: **bring the docs back in line with the live code**, touching only what
actually changed, and flag anything structural that needs a human-written prose update.

## When to run it (the whole point)

Run this **every time** a change lands (or is requested) that affects either doc:

- a reward coefficient (`progress_coef`, `gate_bonus`, `finish_bonus`, `crash_penalty`,
  `timeout_penalty`, `d_act_coef`, `speed_coef`, `speed_threshold`, `speed_softness`, `GATE_HALF_EXTENT`),
- a PPO / optimization hyperparameter (`gamma`, `learning_rate`, `ent_coef`, `clip_coef`, `update_epochs`,
  `target_kl`, `num_steps`, `total_timesteps`, …),
- a curriculum value in `SegmentSpawnConfig` (`a0/a1`, `b0/b1`, `c0/c1`, `kappa_min`, `theta_max`,
  `p_start_*`, `v0_max`, `gate_offset`, `d_min`, `d_max_cap`, …),
- an env constant (`freq`, `max_episode_steps`, `control_mode`, `sensor_range`, gate size),
- the observation pipeline (`observation.py`, the wrapper chain in `make_env`, `N_NEXT_GATES`,
  obstacle/gate counts, new/removed obs keys).

When the user asks you to make such a change, **make the change, then immediately run this sync as part
of the same task** — don't wait to be asked twice.

## Source of truth (read these, never trust the docs' own numbers)

The docs are the thing being corrected, so never copy a value *from* a doc. Read live:

1. **Effective hyperparameters = task subclass merged over base `Args`.** The racing task defines
   [`RacingArgs`](../../../lsy_drone_racing/rl/tasks/racing.py) (a `@dataclass(Args)` subclass) whose field
   defaults override [`Args`](../../../lsy_drone_racing/rl/config.py); anything `RacingArgs` doesn't set
   inherits the `Args` default. The documented "current" value of any knob = **`RacingArgs` value if it
   sets one, else the `Args` default**.
   - ⚠ This used to be a `RACING_CONFIG: dict` merged before `Args.create`. It was refactored to the
     `RacingArgs` subclass. If you see code still using `RACING_CONFIG`, read that dict instead — but the
     merge rule (task-override-over-`Args`) is the same. The `analyze-rl-wandb` skill's
     `extract_context.py` parses the *old* `RACING_CONFIG`; treat its output as a hint only and verify
     against the actual `RacingArgs` source, updating that extractor if it has gone stale.
2. **Reward structure** — [`tasks/racing.py`](../../../lsy_drone_racing/rl/tasks/racing.py):
   `racing_reward_components` (which terms exist + their math), `GATE_HALF_EXTENT`, and whether progress is
   the champion distance-reduction (`gate_opening_distance`) or the bounded potential
   (`gate_progress_potential` with `reach`/`sharpness`/`exit_scale`). **These are not interchangeable** —
   the prose, the formula block, and the "removed knobs" note in PARAMETERS.md §1–§2 must match whichever
   one the code currently has.
3. **Curriculum** — [`SegmentSpawnConfig`](../../../lsy_drone_racing/envs/segment_spawn.py) field defaults
   and the `*_schedule` functions (which window drives which knob).
4. **Observation** — [`wrappers/observation.py`](../../../lsy_drone_racing/rl/wrappers/observation.py):
   `RelativeRacingObs.__init__`'s spec dict (keys + shapes), `N_NEXT_GATES`, `FlattenJaxObservation`
   (alphabetical key order), and the wrapper chain order in `make_env`.
5. **Env constants** — the loaded `config/level0.toml` (`freq`, `sensor_range`, `control_mode`) and the
   env's `max_episode_steps`; obstacle/gate counts come from the track.

## Procedure

1. **Diff doc vs code.** For each knob the doc lists a "current **X**", read the live value and compare.
   Build the list of mismatches before editing anything.
2. **Patch values surgically.** Update each stale "current **X**" in place. Do **not** rewrite the
   surrounding prose unless the *meaning* changed (see step 4). Match the file's existing format
   (`### name (current **value**)`, table cells, etc.).
3. **Recompute derived examples.** Some prose embeds numbers derived from coefficients — e.g. the
   crash-vs-cross break-even `crash/(gate_bonus+crash)` in the `gate_bonus` section, the
   "kappa flat until τ=a0 (= N M steps)" tie-in, the chart-reading example ratios. Update these to the
   new values too, or they silently lie.
4. **Handle structural changes (need judgment, not just a value swap):**
   - **New knob** → add a row/section in the right place, with a real ↑/↓ explanation (don't stub it).
   - **Removed knob** → delete its entry; if it was load-bearing, add a one-line note under "Removed knobs".
   - **Progress formula swapped** (champion ↔ bounded potential) → rewrite §1 `progress_coef`, §2
     `GATE_HALF_EXTENT` formula block, and the "removed/added knobs" note to match; these are the
     highest-risk drift points.
   - **Observation change** (new/removed key, shape change, reorder, `N_NEXT_GATES`, obstacle count) →
     update the components table, the total dim, the flat-vector column ranges (recompute every range —
     they cascade), and the wrapper-pipeline block.
5. **Verify the obs total.** After any OBSERVATION_SPACE.md edit, re-add the per-key dims and confirm the
   stated **Total** and the last column range agree (they must equal `FlattenJaxObservation`'s `flat_dim`).
6. **Report** the exact list of values changed and any structural edits made, so the user can sanity-check.

## Gotchas

- **Don't document the value you just set from memory — re-read it from the file you edited.** If the
  user said "set `progress_coef` to 3" but the code shows 2.8, the doc follows the *code*.
- **Working tree vs HEAD:** document the **working-tree** value (that's what the next run uses), even if
  uncommitted — this pairs with the `wandb-runs/<name>-<id>` provenance branches, which pin exactly that.
- **PARAMETERS.md prose is intentionally opinionated** (the ↑/↓ reasoning, "rules of thumb"). Preserve
  that voice; you're correcting facts, not flattening the guidance.
- **Two docs, one change can hit both** — an observation-pipeline edit can change both the obs doc *and*
  PARAMETERS.md (e.g. if a wrapper that adds a reward term moved). Check both every time.

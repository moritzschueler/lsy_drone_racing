# Kaufmann et al. 2023 — "Champion-level drone racing using deep reinforcement learning" (Swift)

> Structured extract of the RL-relevant content for quick reference. Source: Nature 620, 982–987
> (31 Aug 2023). DOI: [10.1038/s41586-023-06419-4](https://doi.org/10.1038/s41586-023-06419-4).
> Open-access full text: https://pmc.ncbi.nlm.nih.gov/articles/PMC10468397/ .
> Authors: Elia Kaufmann, Leonard Bauersfeld, Antonio Loquercio, Matthias Müller, Vladlen Koltun,
> Davide Scaramuzza (UZH Robotics & Perception Group + Intel Labs).
> This is a text extract, not the original binary PDF (the upload arrives as extracted text).

## TL;DR for our project
Swift = the "champion" reference this repo's reward design is modelled on. The progress term we run as
`progress = ("champion", coef)` (`Phi = -gate_opening_distance`, per-step reduction of distance to the
gate) is **exactly** Swift's `r_prog`. Our recent switch to encoding the next gate by its **4 corner
positions** also matches Swift's observation (R^12 = 4 corners). Key divergences: Swift trains on **one
fixed track** with random-gate resets (no curriculum, no dynamics randomization), uses a tiny **2×128**
MLP, and adds a **perception** reward (keep next gate in camera FoV) we don't have.

## Training budget (the headline comparison)
- **1 × 10^8 (100M) environment interactions** for the base policy → **~50 min** on one workstation
  (i9-12900K, RTX 3090, 32 GB DDR5).
- **+2 × 10^7 (20M)** environment interactions for real-world residual **fine-tuning** (→ 120M total).
- **100 agents in parallel**, episodes of **1,500 steps**.
- Framework: TensorFlow Agents. Optimizer: **Adam, lr = 3 × 10^-4** (both policy and value nets).
- (Our run: 50M timesteps, 1024 envs → ~50% of Swift's 100M base / ~42% of the full pipeline, and
  spread across a randomized curriculum rather than one fixed track.)

## Algorithm & networks
- **PPO** (Schulman 2017), actor-critic; only the policy net is deployed.
- Policy net **and** value net: **two-layer MLP, 128 units/layer, LeakyReLU (negative slope 0.2)**.
- Value (critic) net gets **privileged info** (exact pos/orientation/velocity) concatenated to inputs;
  policy net does not (asymmetric actor-critic).

## Observation (o_t ∈ R^31)
1. **Robot state estimate, R^15**: position, velocity, attitude **as a rotation matrix** (rotation
   matrix used over quaternion to avoid ambiguity).
2. **Next gate, R^12**: relative position of the **four gate corners** w.r.t. the vehicle.
3. **Previous action, R^4**.
All observations normalized before the network.

## Action (a_t ∈ R^4)
Mass-normalized **collective thrust + body rates (CTBR)** — the same modality human pilots use; processed
by an onboard Betaflight PID + ESC into motor commands.

## Reward (dense, shaped)
`r_t = r_prog + r_perc + r_cmd − r_crash`  (eq. 7)

- **Progress** (eq. 8): `r_prog = λ1 · (d_{t-1}^Gate − d_t^Gate)` — reward = the per-step *reduction*
  of the distance `d^Gate` from the vehicle CoM to the **center of the next gate**. (Cites Song et al.
  2021, IROS.) This is the "champion progress reward" our `("champion", …)` variant implements.
- **Perception** (eq. 8): `r_perc = λ2 · exp(λ3 · δ_cam^4)`, where `δ_cam` is the angle between the
  camera optical axis and the next gate center — rewards keeping the next gate in view (better pose est.).
- **Command smoothness** (eq. 9): `r_cmd = λ4 · ‖a_t^ω‖ + λ5 · ‖a_t − a_{t-1}‖^2` (body-rate magnitude +
  action-difference penalty).
- **Crash** (eq. 9): `r_crash = 5.0` if `p_z < 0` or gate collision, else `0`; triggering **ends the
  episode**.

### Reward / PPO hyperparameters (Extended Data Table 1a)
| symbol | value | meaning |
|---|---|---|
| γ | 0.99 | discount factor |
| ε | 0.2 | PPO importance-ratio clipping |
| λ1 | 1.0 | progress weight |
| λ2 | 0.02 | perception weight |
| λ3 | −10.0 | perception sharpness (inside exp) |
| λ4 | −2e-4 | body-rate penalty |
| λ5 | −1e-4 | action-smoothness penalty |
| lr | 3e-4 | Adam (policy + value) |

## Episode resets / curriculum
- At each reset, every agent is initialized at a **random gate** on the track, with **bounded
  perturbation around a state previously observed when passing that gate** (a form of state-based
  spawn — comparable in spirit to this repo's per-gate cone spawns, but on a single fixed track).
- **No dynamics randomization at training time** — robustness comes instead from real-world residual
  fine-tuning, not domain randomization.

## Sim-to-real (context, not RL-core)
- Empirical **residual models** identified from ~50 s (3 rollouts) of real flight: residual
  **observations** via Gaussian processes, residual **dynamics** via k-NN (k=5, 800–1000 samples).
  Policy is fine-tuned in the residual-augmented sim. Further fine-tuning iterations gave negligible gains.
- High-fidelity quadrotor sim: grey-box polynomial aerodynamics, Betaflight low-level controller model
  (<1% motor-command error), grey-box battery/ESC model.

## Track & hardware
- Track: **7 square gates**, volume 30 × 30 × 8 m, **75 m lap**; raced for **3 laps**. Designed by a
  pro FPV pilot; includes a Split-S (hardest segment).
- Drone: Agilicious-based, **870 g**, max static thrust ~35 N → **thrust-to-weight ≈ 4.1**. Onboard
  Jetson TX2; policy runs on CPU at 100 Hz (8 ms/inference). Intel RealSense T265 VIO at 100 Hz.

## Results (headline)
- Swift won **15/25** head-to-head races vs three champions (Vanover, Bitmatta, Schaepper) and set the
  fastest recorded lap. Best time-to-finish 17.465 s. In sim ablations, Swift completes **100%** of the
  track under domain shift while zero-shot / domain-randomization / time-optimal-MPC baselines collapse.

## Data / code
- Data + analysis + Swift pseudocode on Zenodo: https://doi.org/10.5281/zenodo.7955278 . Full source is
  intentionally not released.

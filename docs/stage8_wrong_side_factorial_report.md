# Stage 8 2×2 Factorial Wrong-Side Recovery Screening (`stage8_wrong_side_factorial_v1`)

## 1. Experimental Setup & Integrity

- **Experiment ID**: `stage8_wrong_side_factorial_v1`
- **Script**: [`scripts/run_stage8_wrong_side_factorial.py`](file:///c:/Users/inouy/source/sheep_dog/scripts/run_stage8_wrong_side_factorial.py)
- **Results Artifact**: [`artifacts/experiments/stage8_wrong_side_factorial_v1/factorial_results.json`](file:///c:/Users/inouy/source/sheep_dog/artifacts/experiments/stage8_wrong_side_factorial_v1/factorial_results.json)
- **Parent Checkpoint**: `artifacts/models/maskable-ppo-05065228.zip` (Ckpt 613, SHA-256 `da8ce02fcdcd9100601def6fb733762c5c9ce134d11d07058e57919763530e60`, verified `unchanged=true`).
- **Protected Artifacts**: `effective-training-config.json`, `training-state.json`, and `training-telemetry.sqlite` verified SHA-256 unchanged before and after execution.
- **Training Budget & Shared Settings**: `train_seed=7`, `327,680` timesteps (`10 segments × 32,768`), `ppo_env_workers=16`, `effective_learning_rate=0.00015`, identical rewards, neural architecture, and evaluation criteria.
- **2×2 Factorial Design** (50% wrong-side mode + 50% Treatment-v2 base `{randomized_flock: 0.245, nearby_stray: 0.082, split_flock: 0.082, coherent_wall_recovery: 0.054, fixed_easy: 0.037}`):
  - **Arm A**: `wrong_side_mild: 0.50`, `effective_entropy_coef = 0.005`
  - **Arm B**: `wrong_side_mild: 0.50`, `effective_entropy_coef = 0.020`
  - **Arm C**: `wrong_side_safe_radius: 0.50`, `effective_entropy_coef = 0.020`
  - **Arm D**: `wrong_side_safe_radius: 0.50`, `effective_entropy_coef = 0.005`
- **New Spawn Mode (`wrong_side_safe_radius`)**: Implemented in [`src/sheepdog/config.py`](file:///c:/Users/inouy/source/sheep_dog/src/sheepdog/config.py) and [`src/sheepdog/environment.py`](file:///c:/Users/inouy/source/sheep_dog/src/sheepdog/environment.py); samples flock center `>= 28` cells from `pen.center` when possible and places all 3 dogs on the pen/lateral-pen side at `min_dist > sheep_vision` (targeting `14.0 <= min_dist <= 18.0`), verified across 50 seeds in [`tests/test_wrong_side_safe_radius.py`](file:///c:/Users/inouy/source/sheep_dog/tests/test_wrong_side_safe_radius.py).
- **Total Wall-Clock**: `1342.4s` (~22.4 min total across baseline + 4 arms + all evaluations).

---

## 2. Segment-by-Segment Progression (`seg_2..seg_10`) & Baseline

| Arm | Ckpt | Dev-10 SR (Rew, NP) | Mild-5 SR (Rew, Flank, Disp, H0, NP) | Safe-5 SR (Rew, Flank, Disp, H0, NP) |
| :--- | :---: | :---: | :---: | :---: |
| **Baseline** | Ckpt 613 | 9/10 (412.1, 30.6) | 0/5 (-456.6, 80%, 27.65, 0.687, 260.0) | 1/5 (-269.6, 80%, 32.65, 0.929, 208.0) |
| **Arm A** (`mild, 0.005`) | `seg_2` | 9/10 (378.1, 33.0) | 0/5 (-502.0, 80%, 28.50, 0.798, 260.0) | 1/5 (-264.7, 80%, 32.90, 0.855, 208.2) |
| **Arm A** (`mild, 0.005`) | `seg_4` | 9/10 (381.1, 29.9) | 1/5 (-296.0, 80%, 26.90, 0.781, 209.6) | 2/5 (-178.7, 80%, 32.25, 0.801, 173.6) |
| **Arm A** (`mild, 0.005`) | `seg_6` | 8/10 (294.4, 53.7) | 1/5 (-218.6, 80%, 26.35, 0.775, 209.0) | 3/5 (+35.8, 80%, 31.00, 0.879, 130.2) |
| **Arm A** (`mild, 0.005`) | **`seg_8` (best)** | 9/10 (468.1, 28.0) | 1/5 (-247.6, 80%, 26.50, 0.811, 208.2) | **4/5 (+168.9, 80%, 30.70, 0.835, 60.8)** |
| **Arm A** (`mild, 0.005`) | `seg_10` | 8/10 (328.4, 56.2) | 1/5 (-232.4, 80%, 25.95, 0.797, 208.2) | 3/5 (+83.2, 80%, 29.15, 0.769, 119.0) |
| **Arm B** (`mild, 0.020`) | `seg_2` | 9/10 (385.8, 26.6) | 1/5 (-268.4, 80%, 27.30, 0.747, 208.0) | 1/5 (-246.4, 80%, 32.55, 0.908, 208.0) |
| **Arm B** (`mild, 0.020`) | `seg_4` | **10/10** (476.6, 7.5) | 0/5 (-469.7, 80%, 27.30, 0.744, 260.0) | 1/5 (-236.0, 80%, 33.40, 0.902, 208.0) |
| **Arm B** (`mild, 0.020`) | `seg_6` | **10/10** (476.1, 12.3) | 0/5 (-428.0, 80%, 25.05, 0.801, 260.0) | 2/5 (-159.1, 80%, 30.95, 0.969, 197.2) |
| **Arm B** (`mild, 0.020`) | `seg_8` | **10/10** (472.1, 12.0) | 0/5 (-429.5, 80%, 27.65, 0.819, 260.0) | 3/5 (+53.3, 80%, 30.55, 0.885, 105.4) |
| **Arm B** (`mild, 0.020`) | **`seg_10` (best)** | 9/10 (454.6, 31.5) | 1/5 (-226.2, 80%, 23.85, 0.881, 209.4) | 3/5 (+11.1, 80%, 29.85, 1.013, 143.6) |
| **Arm C** (`safe, 0.020`) | `seg_2` | 6/10 (233.8, 105.5) | 0/5 (-473.3, 60%, 24.55, 0.807, 260.0) | 1/5 (-285.9, 60%, 28.90, 1.085, 208.0) |
| **Arm C** (`safe, 0.020`) | **`seg_4` (best)** | **10/10** (483.1, 2.0) | 1/5 (-276.2, 80%, 25.05, 0.897, 208.0) | 2/5 (-140.0, 60%, 29.60, 1.184, 172.8) |
| **Arm C** (`safe, 0.020`) | `seg_6` | 9/10 (386.7, 27.9) | 1/5 (-247.6, 60%, 24.30, 0.971, 209.6) | 1/5 (-275.5, 40%, 27.70, 1.256, 208.2) |
| **Arm C** (`safe, 0.020`) | `seg_8` | 9/10 (381.4, 27.1) | 0/5 (-419.1, 40%, 23.75, 0.984, 260.0) | 1/5 (-292.6, 60%, 28.10, 1.301, 208.0) |
| **Arm C** (`safe, 0.020`) | `seg_10` | 8/10 (308.7, 53.4) | 0/5 (-442.5, 60%, 27.05, 0.947, 260.0) | 1/5 (-259.1, 60%, 29.55, 1.330, 208.0) |
| **Arm D** (`safe, 0.005`) | `seg_2` | 9/10 (360.5, 34.9) | 0/5 (-426.9, 80%, 26.60, 0.769, 260.0) | 1/5 (-244.4, 60%, 30.20, 1.024, 208.0) |
| **Arm D** (`safe, 0.005`) | `seg_4` | 8/10 (342.4, 54.7) | 1/5 (-274.9, 60%, 24.70, 0.768, 211.4) | 1/5 (-244.1, 60%, 29.75, 1.081, 208.0) |
| **Arm D** (`safe, 0.005`) | `seg_6` | 8/10 (342.6, 54.3) | 0/5 (-450.3, 60%, 23.75, 0.786, 260.0) | 1/5 (-280.2, 40%, 28.00, 1.094, 208.0) |
| **Arm D** (`safe, 0.005`) | **`seg_8` (best)** | 9/10 (364.0, 27.9) | 1/5 (-230.4, 60%, 26.10, 0.793, 214.0) | 1/5 (-265.4, 60%, 29.00, 1.058, 208.0) |
| **Arm D** (`safe, 0.005`) | `seg_10` | 9/10 (386.8, 28.0) | 0/5 (-359.2, 80%, 25.15, 0.787, 260.0) | 1/5 (-267.1, 60%, 29.40, 1.101, 208.0) |

*(Every failure across all diagnostic rollouts terminated via `no-progress` (`no_progress_window = 260`); 0 timeouts.)*

---

## 3. Best-Checkpoint Retention (`Diag-20`) & Unseen Confirmation (`Confirm-10`)

| Arm | Best Seg | Train Wall (s) | Dev-10 SR (Rew) | Diag-20 SR (Rew) | Mild-5 SR | Safe-5 SR | Confirm Mild-10 (`3001..3010`) | Confirm Safe-10 (`3011..3020`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline** | Ckpt 613 | — | 9/10 (412.1) | 16/20 (238.3) | 0/5 | 1/5 | — | — |
| **Arm A** (`mild, 0.005`) | `seg_8` | 183.2 | 9/10 (468.1) | 16/20 (236.9) | 1/5 | **4/5** | **2/10** (-263.3, NP 224.3) | **2/10** (-232.7, NP 214.7) |
| **Arm B** (`mild, 0.020`) | `seg_10` | 184.2 | 9/10 (454.6) | **17/20 (272.0)** | 1/5 | 3/5 | 1/10 (-342.1, NP 247.8) | **2/10** (-272.2, NP 227.0) |
| **Arm C** (`safe, 0.020`) | `seg_4` | 163.7 | **10/10 (483.1)** | **17/20 (264.2)** | 1/5 | 2/5 | 1/10 (-389.8, NP 234.1) | 1/10 (-350.5, NP 234.6) |
| **Arm D** (`safe, 0.005`) | `seg_8` | 163.5 | 9/10 (364.0) | **17/20 (253.4)** | 1/5 | 1/5 | *Skipped (< 2/5)* | *Skipped (< 2/5)* |

---

## 4. Answers to the 10 Factorial Questions

1. **Does `wrong_side_mild=0.50` vs `wrong_side_safe_radius=0.50` improve wrong-side recovery?**
   - Yes, **`wrong_side_mild=0.50` (Arms A & B) outperformed `wrong_side_safe_radius=0.50` (Arms C & D)** on `diag_wrong_side_safe_radius` (`4/5` and `3/5` vs `2/5` and `1/5`), with positive mean rewards (`+168.9` and `+53.3`). However, neither mode solved `diag_wrong_side_mild` reliably (`0/5`–`1/5`).
2. **Does `entropy_coef = 0.020` vs `0.005` improve wrong-side recovery or hurt retention?**
   - Under `wrong_side_mild=0.50`, `0.005` (Arm A) achieved higher peak `safe` recovery (`4/5` vs `3/5`) and Confirmation-10 `mild` (`2/10` vs `1/10`), while `0.020` (Arm B) achieved better `Dev-10` (`10/10` at `seg_4..seg_8`) and `Diag-20` retention (`17/20` at `272.0` vs `16/20` at `236.9`). Under `wrong_side_safe_radius=0.50`, `0.020` (Arm C) caused over-exploration (`H0` rose to `1.330`) and degraded flanking (`40%–60%`).
3. **Is there an interaction between spawn mode and entropy?**
   - Yes: `0.020` paired with `wrong_side_mild` stabilized ordinary herding (`10/10` Dev, `17/20` Diag-20) while still learning `3/5` `safe` recovery, whereas `0.020` paired with `wrong_side_safe_radius` inflated `H0` (`0.929 -> 1.330`) without a flanking/containment reward gradient.
4. **What do `flank_reached_pct` and `early_sheep_disp` reveal?**
   - Even Baseline (`Ckpt 613`) already has `flank_reached_pct = 80%` and `early_sheep_disp = 27.65` (mild) / `32.65` (safe). Dogs are **not** frozen or failing to cross behind the flock; they reach the far side in 80% of episodes, but early wrong-side contact scatters the sheep and the policy fails to re-gather them before `no_progress_window = 260`.
5. **What does `step0_entropy` (`H0`) reveal?**
   - Baseline `H0` is already moderate (`0.687` on mild `2024`, `0.929` on safe `2034`), so policy collapse is not the primary bottleneck. Raising `H0` to `1.330` (Arm C) reduced `flank_reached_pct` from `80%` to `40%–60%`.
6. **What are the failure modes (`no-progress` vs `timeout`)?**
   - **100%** of wrong-side failures across Baseline and all 4 arms terminated via `no-progress` (`260` steps without new best route distance / penned count); 0 timeouts occurred.
7. **Does high wrong-side exposure (50%) cause catastrophic forgetting on ordinary herding (`Dev-10`, `Diag-20`)?**
   - No. `Dev-10` stayed at `9/10`–`10/10` at best checkpoints, and `Diag-20` matched or beat Baseline (`16/20` in Arm A, `17/20` in Arms B/C/D vs Baseline `16/20`).
8. **Did any arm qualify for and pass the 10-seed Confirmation sets (`3001..3020`)?**
   - Arms A, B, and C qualified (`>= 2/5` on a diagnostic set), but on the unseen 10-seed Confirmation sets they only scored `2/10` mild / `2/10` safe (Arm A), `1/10` mild / `2/10` safe (Arm B), and `1/10` mild / `1/10` safe (Arm C).
9. **Were all constraints and protected artifacts preserved?**
   - Yes: Ckpt 613, `effective-training-config.json`, `training-state.json`, and `training-telemetry.sqlite` are bit-for-bit unchanged; zero scripted recovery or reward/architecture changes were introduced.
10. **Recommendation**:
    - **Do NOT promote any arm** to production `spawn_mix` or launch larger runs yet.
    - The diagnostic evidence (`80%` `flank_reached_pct` at baseline, `27.65–32.65` early sheep displacement, `100%` `no-progress` terminations, and `2/10` max on unseen 10-seed confirmation) proves that **curriculum spawn geometry + entropy tuning alone is not sufficient** for reliable wrong-side recovery (`wrong_side_mild` / `wrong_side_safe_radius`).
    - **Next concrete step**: Inspect replays of Arms A/B on `2021..2025` and `3011..3020` to see *why* dogs fail after initial flanking/scattering (e.g., `TeamStrategy` wrong-side role/collector target or `no_progress_window=260` cutting off wide re-gathering loops) before designing any targeted fix.

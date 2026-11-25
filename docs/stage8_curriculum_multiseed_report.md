# Stage 8 Curriculum Multi-Seed Controlled Comparison (`stage8_curriculum_multiseed_v1`)

## 1. Experimental Setup & Integrity

- **Experiment ID**: `stage8_curriculum_multiseed_v1`
- **Script**: [`scripts/run_stage8_curriculum_multiseed.py`](file:///c:/Users/inouy/source/sheep_dog/scripts/run_stage8_curriculum_multiseed.py)
- **Results Artifact**: [`artifacts/experiments/stage8_curriculum_multiseed_v1/multiseed_results.json`](file:///c:/Users/inouy/source/sheep_dog/artifacts/experiments/stage8_curriculum_multiseed_v1/multiseed_results.json)
- **Parent Checkpoint**: `artifacts/models/maskable-ppo-05065228.zip` (Ckpt 613, SHA-256 `da8ce02fcdcd9100601def6fb733762c5c9ce134d11d07058e57919763530e60`, verified unchanged `parent_unchanged=true`).
- **Training Budget**: 3 independent training seeds (`7, 101, 202`) × 2 arms × 327,680 timesteps (`10 segments × 32,768` timesteps) per run, in isolated directories `artifacts/experiments/stage8_curriculum_multiseed_v1/seed_<S>/<arm>/`.
- **Fixed Hyperparameters**: `effective_learning_rate=0.00015`, `effective_entropy_coef=0.005`, `ppo_env_workers=16`, unchanged rewards, architecture (`[256, 256, 128]`, `observation_size=208`, `action_size=9`), and evaluation criteria.
- **Spawn Mixes**:
  - **Control**: `{"fixed_easy": 0.30, "randomized_flock": 0.55, "nearby_stray": 0.15}`
  - **Treatment**: `{"randomized_flock": 0.35, "nearby_stray": 0.15, "split_flock": 0.15, "wall_recovery": 0.10, "fixed_easy": 0.10, "wrong_side_recovery": 0.15}`
- **Multi-Checkpoint Dev Validation**: Evaluated segments `2, 4, 6, 8, 10` on the 10-seed dev set (`[11, 23, 37, 41, 53, 59, 61, 67, 71, 73]`), selecting the best checkpoint per run ranked by `(success_rate, average_reward, -average_completion_steps)`.
- **Deterministic Evaluation Suites** (run on Baseline Ckpt 613 and each run's best dev-validated checkpoint):
  1. **Dev-10**: `[11, 23, 37, 41, 53, 59, 61, 67, 71, 73]`
  2. **Diagnostic-20**: `[101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197]`
  3. **Untouched Final Test-30 (seeds > 1000)**: `[1009, 1013, 1019, 1021, 1031, 1033, 1039, 1049, 1051, 1061, 1063, 1069, 1087, 1091, 1093, 1097, 1103, 1109, 1117, 1123, 1129, 1151, 1153, 1163, 1171, 1181, 1187, 1193, 1201, 1213]`
  4. **Category Stress Probe-15**: 5 `wrong_side_recovery` (`2001..2005`), 5 `split_flock` (`2006..2010`), 5 `wall_recovery` (`2011..2015`) using temporary `spawn_mix` overrides for the probe only.

---

## 2. Multi-Checkpoint Dev Selection & Trajectory

| Arm | Train Seed | Best Seg (Timesteps) | Seg 2 SR (Rew) | Seg 4 SR (Rew) | Seg 6 SR (Rew) | Seg 8 SR (Rew) | Seg 10 SR (Rew) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Control** | 7 | `seg_6` (196,608) | 0.80 (343.1) | 0.90 (382.2) | **0.90 (411.9)** | 0.80 (264.9) | 0.70 (244.0) |
| **Control** | 101 | `seg_8` (262,144) | 0.80 (303.6) | 0.90 (364.5) | 0.80 (358.3) | **0.90 (399.9)** | 0.80 (353.8) |
| **Control** | 202 | `seg_4` (131,072) | 0.80 (358.9) | **0.90 (382.8)** | 0.90 (382.2) | 0.90 (370.3) | 0.90 (378.2) |
| **Treatment** | 7 | `seg_2` (65,536) | **0.90 (404.7)** | 0.90 (378.7) | 0.80 (275.6) | 0.80 (280.8) | 0.80 (269.1) |
| **Treatment** | 101 | `seg_2` (65,536) | **0.90 (365.2)** | 0.80 (296.8) | 0.70 (170.0) | 0.70 (179.9) | 0.80 (241.8) |
| **Treatment** | 202 | `seg_6` (196,608) | 0.80 (312.6) | 0.90 (398.3) | **1.00 (467.9)** | 0.80 (321.4) | 0.70 (207.9) |

**Key observation**: Under **Treatment**, later segments (`seg_6`–`seg_10`) consistently degraded dev performance on seeds 7 and 101 (and `seg_8`–`seg_10` on seed 202), so dev-selection stopped early at `seg_2` (65,536 ts) for 2 of 3 seeds and `seg_6` (196,608 ts) for seed 202. Control peaked later on average (`seg_4`, `seg_6`, `seg_8`, mean 196,608 ts vs Treatment mean 109,227 ts).

---

## 3. Aggregated Results Across Training Seeds (Mean ± Std [Min, Max])

### 3.1 Dev-10, Diagnostic-20, Untouched Test-30, and Stress Probe-15

| Suite | Arm | Success Rate | Avg Penned | Avg Steps (All) | Avg Steps (Success) | Avg Reward | Avg No-Progress Steps | No-Progress Fails | Osc Fails (All) | Corner Stuck Fails |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dev-10** | Baseline | 0.900 (9/10) | 3.60 | 133.9 | 104.8 | 412.1 | 30.6 | 1 | 0 (1) | 0 |
| **Dev-10** | Control | 0.900 ± 0.000 [0.900, 0.900] | 3.60 ± 0.00 | 134.7 ± 10.7 | 106.3 ± 6.8 | 398.2 ± 14.6 | 31.5 ± 6.1 | 1.0 ± 0.0 | 1.0 ± 0.0 (2.3) | 0.0 ± 0.0 |
| **Dev-10** | Treatment | **0.933 ± 0.058** [0.900, 1.000] | **3.80 ± 0.20** | 138.7 ± 24.0 | 112.6 ± 5.9 | **412.6 ± 51.8** | **24.2 ± 20.4** | 0.7 ± 0.6 | 0.7 ± 0.6 (3.7) | 0.0 ± 0.0 |
| **Diag-20** | Baseline | 0.800 (16/20) | 3.20 | 154.9 | 84.4 | 238.3 | 53.6 | 4 | 4 (7) | 0 |
| **Diag-20** | Control | 0.850 ± 0.050 [0.800, 0.900] | 3.45 ± 0.28 | 142.7 ± 10.0 | 88.5 ± 7.2 | 279.9 ± 55.4 | 40.1 ± 12.1 | 3.0 ± 1.0 | 1.3 ± 0.6 (6.3) | 0.3 ± 0.6 |
| **Diag-20** | Treatment | **0.867 ± 0.029** [0.850, 0.900] | **3.47 ± 0.12** | 142.0 ± 5.5 | 93.7 ± 8.5 | **283.1 ± 25.6** | **39.6 ± 3.6** | 2.7 ± 0.6 | 2.3 ± 0.6 (5.7) | 0.0 ± 0.0 |
| **Test-30** | Baseline | 0.900 (27/30) | 3.70 | 129.9 | 93.4 | 359.6 | 29.0 | 3 | 2 (5) | 0 |
| **Test-30** | Control | 0.900 ± 0.033 [0.867, 0.933] | 3.73 ± 0.03 | 133.0 ± 8.3 | 96.1 ± 3.9 | 366.8 ± 13.3 | 31.7 ± 7.5 | 3.0 ± 1.0 | 3.0 ± 1.0 (8.0) | 0.3 ± 0.6 |
| **Test-30** | Treatment | **0.933 ± 0.000** [0.933, 0.933] | 3.73 ± 0.00 | **125.8 ± 5.4** | 96.1 ± 3.2 | **368.4 ± 7.6** | **20.9 ± 1.5** | **2.0 ± 0.0** | 1.7 ± 0.6 (6.3) | 0.0 ± 0.0 |
| **Probe-15** | Baseline | 0.200 (3/15) | 1.20 | 362.1 | 238.7 | -265.4 | 219.5 | 12 | 12 (14) | 2 |
| **Probe-15** | Control | 0.156 ± 0.038 [0.133, 0.200] | 1.31 ± 0.37 | 366.5 ± 6.7 | 206.2 ± 89.5 | -260.2 ± 50.2 | 227.5 ± 4.3 | 12.7 ± 0.6 | 9.0 ± 0.0 (10.0) | 0.0 ± 0.0 |
| **Probe-15** | Treatment | **0.289 ± 0.038** [0.267, 0.333] | **1.40 ± 0.18** | **336.8 ± 10.1** | **182.7 ± 15.2** | **-191.2 ± 30.4** | **194.4 ± 9.8** | **10.7 ± 0.6** | 10.0 ± 1.0 (10.7) | 1.3 ± 0.6 |

### 3.2 Per-Seed Breakdown

| Arm | Train Seed | Dev-10 SR (Rew) | Diag-20 SR (Rew) | Untouched Test-30 SR (Rew) | Stress Probe-15 SR (Rew) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline** | — | 0.900 (412.1) | 0.800 (238.3) | 0.900 (359.6) | 0.200 (-265.4) |
| **Control** | 7 | 0.900 (411.9) | 0.800 (230.7) | 0.900 (353.0) | 0.133 (-303.8) |
| **Control** | 101 | 0.900 (399.9) | 0.850 (268.9) | 0.933 (379.5) | 0.133 (-271.4) |
| **Control** | 202 | 0.900 (382.8) | 0.900 (340.0) | 0.867 (367.8) | 0.200 (-205.4) |
| **Treatment** | 7 | 0.900 (404.7) | 0.850 (269.6) | 0.933 (369.9) | 0.267 (-213.8) |
| **Treatment** | 101 | 0.900 (365.2) | 0.850 (267.1) | 0.933 (375.1) | 0.333 (-156.6) |
| **Treatment** | 202 | 1.000 (467.9) | 0.900 (312.5) | 0.933 (360.1) | 0.267 (-203.0) |

---

## 4. Per-Category Breakdown

### 4.1 Standard Spawn Categories (Untouched Test-30: 10 `fixed_easy`, 3 `nearby_stray`, 17 `randomized_flock`)

| Category | Baseline SR (Rew) | Control Mean ± Std [Min, Max] (Rew) | Treatment Mean ± Std [Min, Max] (Rew) |
| :--- | :---: | :---: | :---: |
| `fixed_easy` (10) | 1.000 (516.3) | 0.967 ± 0.058 [0.900, 1.000] (503.1) | **1.000 ± 0.000** [1.000, 1.000] (508.1) |
| `nearby_stray` (3) | 0.667 (63.6) | 0.667 ± 0.000 [0.667, 0.667] (75.2) | 0.667 ± 0.000 [0.667, 0.667] (48.5) |
| `randomized_flock` (17) | 0.882 (319.7) | 0.902 ± 0.034 [0.882, 0.941] (338.0) | **0.941 ± 0.000** [0.941, 0.941] (342.6) |

*(In Diagnostic-20: `fixed_easy` (3) improved from Baseline 0.667 to 1.000 in both Control and Treatment; `nearby_stray` (4) was 0.750 Baseline vs 0.833 ± 0.144 Control vs 0.750 ± 0.000 Treatment; `randomized_flock` (13) was 0.846 Baseline vs 0.821 ± 0.044 Control vs 0.872 ± 0.044 Treatment.)*

### 4.2 Stress Probe Categories (5 seeds each)

| Category | Baseline SR (Rew) | Control Mean ± Std [Min, Max] (Rew) | Treatment Mean ± Std [Min, Max] (Rew) |
| :--- | :---: | :---: | :---: |
| `split_flock` (5) | 0.600 (166.3) | 0.400 ± 0.000 [0.400, 0.400] (34.1) | **0.800 ± 0.000** [0.800, 0.800] (290.5) |
| `wall_recovery` (5) | 0.000 (-455.6) | 0.067 ± 0.115 [0.000, 0.200] (-336.3) | 0.067 ± 0.115 [0.000, 0.200] (-384.3) |
| `wrong_side_recovery` (5) | 0.000 (-506.9) | 0.000 ± 0.000 [0.000, 0.000] (-478.3) | 0.000 ± 0.000 [0.000, 0.000] (-479.7) |

**Key finding on Stress Probe**:
- Treatment lifted `split_flock` from **0.600 (Baseline) and 0.400 (Control)** to **0.800 (4/5 across all 3 seeds)**, reversing Control's forgetting on split flocks.
- However, **`wrong_side_recovery` remained 0/5 (0.000)** across every single run of both Control and Treatment, and **`wall_recovery` remained near-zero (0.067 mean, 1/5 on one seed only)** in both arms.

---

## 5. Catastrophic Forgetting & Regression Analysis

| Arm | Train Seed | Dev-10 Regressed (Gained) | Diag-20 Regressed (Gained) | Untouched Test-30 Regressed (Gained) | Net Delta (Dev / Diag / Test) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Control** | 7 | `[]` (`[]`) | `[181]` (`[109]`) | `[1103]` (`[1171]`) | 0 / 0 / 0 |
| **Control** | 101 | `[]` (`[]`) | `[]` (`[109]`) | `[]` (`[1171]`) | 0 / +1 / +1 |
| **Control** | 202 | `[]` (`[]`) | `[173]` (`[109, 131, 193]`) | `[1033, 1193]` (`[1171]`) | 0 / +2 / -1 |
| **Treatment** | 7 | `[61]` (`[37]`) | `[]` (`[109]`) | `[]` (`[1171]`) | 0 / +1 / +1 |
| **Treatment** | 101 | `[]` (`[]`) | `[]` (`[109]`) | `[]` (`[1171]`) | 0 / +1 / +1 |
| **Treatment** | 202 | `[]` (`[37]`) | `[]` (`[109, 131]`) | `[]` (`[1171]`) | +1 / +2 / +1 |

- **Zero forgetting on Diag-20 and Untouched Test-30 for Treatment**: Across all 3 seeds, Treatment had **0 regressions** on Diagnostic-20 and Untouched Test-30 (compared to Control's 1–2 regressions on seeds 7 and 202).
- **Dev-10 Seed 61**: Regressed only in Treatment seed 7 (`seg_2`), while Treatment seeds 101 and 202 both passed seed 61.

---

## 6. Seed 61 & Diagnostic Seeds `131, 137, 193` Failure Analysis

### Dev Seed 61 (`randomized_flock`)
- **Baseline (Ckpt 613)**: **SUCCESS** in 203 steps (`reward=700.49`, `4/4` penned, `no_progress_steps=8`, `oscillation_detected=False`, `corner_stuck_at_end=False`).
- **Control (all 3 seeds)**: **SUCCESS** in 182 / 194 / 219 steps (`reward=743.3 / 723.3 / 700.0`).
- **Treatment**:
  - **Seed 7 (`seg_2`)**: **FAILED** (`no-progress` at 548 steps, `2/4` penned, `reward=30.14`, `no_progress_steps=260`, `oscillation_detected=True`). In the earlier 98k-timestep pilot (`stage8_curriculum_pilot_v1`), Treatment also regressed on seed 61 at `seg_3` (`no-progress` at 464 steps, `3/4` penned, `oscillation_detected=True`, `corner_stuck_at_end=True`).
  - **Seeds 101 (`seg_2`) & 202 (`seg_6`)**: **SUCCESS** in 182 and 215 steps (`reward=729.56` and `646.60`, `no_progress_steps=0`, `oscillation_detected=False`).
  - **Diagnosis**: Seed 61 is sensitive to early policy perturbation when training with higher recovery-scenario density (seed 7 `seg_2` got stuck oscillating with 2 sheep remaining), but multi-checkpoint dev validation + seed diversity shows this regression is not systematic across seeds (2/3 Treatment seeds solved it cleanly).

### Diagnostic Seeds `131, 137, 193`
- **Seed 131 (`randomized_flock`)**:
  - Baseline failed (`no-progress` at 397 steps, `0/4` penned, `reward=-413.72`, `oscillation_detected=True`).
  - Solved by **Control seed 202** (`177` steps, `reward=538.56`) and **Treatment seed 202** (`176` steps, `reward=560.23`), while seeds 7 and 101 in both arms still terminated in `no-progress` with `0/4` penned.
- **Seed 137 (`randomized_flock`)**:
  - **0/7 solve rate** across Baseline and all 6 trained runs (`steps=558–586`, `0/4` penned, `reward=-739.3` to `-557.0`, `no-progress` with `oscillation_detected=True` in every run).
- **Seed 193 (`nearby_stray`)**:
  - Baseline failed (`no-progress` at 424 steps, `0/4` penned, `reward=-263.34`, `oscillation_detected=True`).
  - Solved only by **Control seed 202** (`155` steps, `reward=597.70`); failed in Control seeds 7 & 101 and all 3 Treatment seeds (`0/4` penned, `no-progress`, `oscillation_detected=True` in all Treatment runs).

---

## 7. Wall-Clock Cost

- **Training time per run (327,680 timesteps, 10 segments, 16 workers)**:
  - **Control**: `202.8s` (seed 7, mtime-estimated), `198.4s` (seed 101), `221.7s` (seed 202) — mean **207.6s (~3.5 min)** per run.
  - **Treatment**: `210.6s` (seed 7, mtime-estimated), `212.0s` (seed 101), `211.8s` (seed 202) — mean **211.5s (~3.5 min)** per run.
- **Total resumed run wall-clock** (`baseline_eval` + `seed_7` eval + `seed_101` & `seed_202` train+eval): **1,246.8s (~20.8 min)**. End-to-end compute including `seed_7` training (~413s) was **~1,660s (~27.7 min)**.

---

## 8. Smallest Evidence-Supported Recommendation

1. **Do NOT auto-promote** either curriculum yet, and **do NOT deploy** Treatment as a full replacement without addressing two clear gaps:
   - **Positive signal**: With multi-checkpoint dev selection, **Treatment** outperformed both Baseline and Control on **Untouched Test-30** (`0.933 ± 0.000` vs `0.900 ± 0.033` Control and `0.900` Baseline, with **0 regressions** across all 3 seeds), **Diagnostic-20** (`0.867 ± 0.029` vs `0.850 ± 0.050` Control and `0.800` Baseline), **Dev-10** (`0.933 ± 0.058` vs `0.900 ± 0.000`), and **Stress Probe `split_flock`** (`0.800 ± 0.000` vs `0.400 ± 0.000` Control and `0.600` Baseline).
   - **Two critical limitations**:
     1. **Zero transfer to `wrong_side_recovery` (0/5) and near-zero `wall_recovery` (0.067)** — mixing 15% `wrong_side_recovery` + 10% `wall_recovery` at `lr=1.5e-4` over 327k steps did not teach the policy to solve those hard recovery spawns.
     2. **Later-segment degradation under Treatment**: Dev performance peaked at `seg_2` (65k steps) for 2/3 Treatment seeds and degraded by `seg_6`–`seg_10`, indicating that prolonged training on the 6-mode Treatment mix at `lr=1.5e-4` destabilizes previously learned behaviors unless early-stopped by dev validation.
2. **Smallest next step**:
   - Keep **multi-checkpoint dev validation** mandatory for any curriculum continuation.
   - Before promoting a Stage 8 curriculum update, run a targeted follow-up testing either (a) a gentler Treatment mix / lower LR (`lr=7.5e-5`) to prevent late-segment dev drift, or (b) focused inspection of why `wrong_side_recovery` and `wall_recovery` remain 0/5 and 1/15 (observation/horizon vs curriculum weight) rather than assuming spawn-mix alone will solve them.

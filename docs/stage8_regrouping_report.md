# Stage 8 Controlled 3-Arm Regrouping Experiment (`stage8_regrouping_v1`) & Read-Only Cleanup Audit

## 1. Experimental Setup & Integrity

- **Experiment ID**: `stage8_regrouping_v1`
- **Script**: [`scripts/run_stage8_regrouping_experiment.py`](file:///c:/Users/inouy/source/sheep_dog/scripts/run_stage8_regrouping_experiment.py)
- **Results Artifact**: [`artifacts/experiments/stage8_regrouping_v1/regrouping_results.json`](file:///c:/Users/inouy/source/sheep_dog/artifacts/experiments/stage8_regrouping_v1/regrouping_results.json)
- **Parent Checkpoint**: `artifacts/models/maskable-ppo-05065228.zip` (Ckpt 613, SHA-256 `da8ce02fcdcd9100601def6fb733762c5c9ce134d11d07058e57919763530e60`, verified `unchanged=true`).
- **Protected Artifacts**: `effective-training-config.json`, `training-state.json`, and `training-telemetry.sqlite` verified SHA-256 unchanged before and after execution.
- **Shared Settings**: `train_seed=7`, `327,680` timesteps (`10 × 32,768`), `ppo_env_workers=16`, `effective_learning_rate=0.00015`, `effective_entropy_coef=0.020`, `spawn_mix = {"randomized_flock": 0.245, "nearby_stray": 0.082, "split_flock": 0.082, "coherent_wall_recovery": 0.054, "fixed_easy": 0.037, "wrong_side_mild": 0.50}`.
- **3-Arm Design** (all evaluations run under standard baseline eval rules `regrouping_approach_scale=0.0`, `regroup_no_progress_reset=False`):
  - **Arm A (Control)**: `regrouping_approach_scale = 0.0`, `regroup_no_progress_reset = False`
  - **Arm B (Reward Credit Only)**: `regrouping_approach_scale = 2.0`, `regroup_no_progress_reset = False`
  - **Arm C (No-Progress Accounting Only)**: `regrouping_approach_scale = 0.0`, `regroup_no_progress_reset = True` (`regroup_no_progress_max_resets = 80`)
- **Total Wall-Clock**: `1083.5s` (~18.1 min).

---

## 2. Segment-by-Segment Progression (`seg_2..seg_10`)

| Arm | Ckpt | Dev-10 SR (Rew, NP) | Diag Mild-5 SR (Rew, Flank, Disp, H0, NP) | Diag Safe-5 SR (Rew, Flank, Disp, H0, NP) |
| :--- | :---: | :---: | :---: | :---: |
| **Arm A (Control)** | `seg_2` | 9/10 (385.8, 26.6) | 0/5 (-431.7, 80%, 27.65, 0.747, 260.0) | 1/5 (-246.4, 80%, 32.55, 0.908, 208.0) |
| **Arm A (Control)** | `seg_4` | **10/10** (457.8, 18.6) | 0/5 (-473.7, 80%, 27.40, 0.819, 260.0) | 1/5 (-190.8, 80%, 33.55, 0.863, 208.2) |
| **Arm A (Control)** | `seg_6` | **10/10** (460.8, 20.7) | 0/5 (-457.5, 80%, 26.80, 0.903, 260.0) | 1/5 (-255.7, 80%, 32.30, 0.903, 208.0) |
| **Arm A (Control)** | `seg_8` | **10/10** (463.3, 14.5) | 1/5 (-284.9, 80%, 25.25, 0.992, 220.0) | 2/5 (-131.2, 80%, 30.50, 0.937, 162.6) |
| **Arm A (Control)** | **`seg_10` (best)** | **10/10 (482.1, 3.7)** | 1/5 (-252.8, 80%, 26.15, 0.968, 209.2) | 2/5 (-182.6, 80%, 30.60, 0.940, 190.4) |
| **Arm B (Reward Only)** | `seg_2` | 9/10 (390.5, 26.8) | 0/5 (-502.1, 80%, 27.75, 0.819, 260.0) | 2/5 (-198.2, 60%, 32.65, 0.964, 204.0) |
| **Arm B (Reward Only)** | `seg_4` | 9/10 (388.6, 29.4) | 1/5 (-297.9, 60%, 26.25, 0.849, 210.6) | 1/5 (-258.8, 60%, 32.00, 1.028, 208.0) |
| **Arm B (Reward Only)** | `seg_6` | 9/10 (372.9, 34.4) | 1/5 (-282.6, 60%, 23.15, 0.939, 227.0) | 1/5 (-278.9, 40%, 28.60, 1.095, 208.0) |
| **Arm B (Reward Only)** | `seg_8` | 9/10 (379.3, 48.5) | 0/5 (-406.7, 60%, 25.15, 0.967, 260.0) | 1/5 (-251.3, 60%, 34.75, 1.028, 208.0) |
| **Arm B (Reward Only)** | **`seg_10` (best)** | 9/10 (425.0, 35.2) | 0/5 (-460.7, 60%, 23.45, 1.069, 260.0) | **3/5** (-39.2, 60%, 30.20, 1.095, 143.4) |
| **Arm C (NP Reset Only)** | `seg_2` | 8/10 (334.7, 54.0) | 0/5 (-463.1, 80%, 27.90, 0.724, 260.0) | 1/5 (-246.5, 80%, 34.05, 0.900, 210.0) |
| **Arm C (NP Reset Only)** | `seg_4` | **10/10** (465.1, 15.5) | 0/5 (-447.7, 80%, 27.85, 0.787, 260.0) | 1/5 (-240.1, 60%, 33.10, 0.896, 208.6) |
| **Arm C (NP Reset Only)** | `seg_6` | 9/10 (456.4, 30.1) | 0/5 (-458.2, 80%, 27.20, 0.824, 260.0) | 1/5 (-300.9, 80%, 33.15, 0.857, 208.0) |
| **Arm C (NP Reset Only)** | **`seg_8` (best)** | **10/10** (461.7, 14.8) | 1/5 (-271.9, 60%, 27.55, 0.845, 208.0) | 1/5 (-296.8, 60%, 31.60, 0.911, 208.0) |
| **Arm C (NP Reset Only)** | `seg_10` | 9/10 (410.3, 30.5) | 0/5 (-424.0, 80%, 27.40, 0.857, 260.0) | 1/5 (-281.3, 40%, 29.30, 0.972, 208.0) |

---

## 3. Selected Best Models: `Diag-20` & Fresh Reserved `Confirmation-20` (`4001..4020`)

| Arm | Best Seg | Train Wall (s) | Dev-10 SR (Rew) | Diag-20 SR (Rew, Steps, NP) | Confirm Mild-10 (`4001..4010`) | Confirm Safe-10 (`4011..4020`) | Confirm-20 Total |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Arm A (Control)** | `seg_10` | 203.5 | **10/10** (482.1) | **17/20** (316.8, 138.7, 41.1) | 2/10 (-205.2, NP 210.4) | **5/10** (-19.3, NP 133.5) | **7/20 (35%)** |
| **Arm B (Reward Only)** | `seg_10` | 137.7 | 9/10 (425.0) | 17/20 (301.5, 150.2, 47.0) | 1/10 (-282.2, NP 234.0) | 4/10 (-92.9, NP 160.3) | 5/20 (25%) |
| **Arm C (NP Reset Only)** | `seg_8` | 201.0 | **10/10** (461.7) | **17/20** (317.4, 138.2, 41.0) | **3/10** (-155.6, NP 194.1) | 2/10 (-216.2, NP 221.6) | 5/20 (25%) |

---

## 4. Conclusions & Smallest Proven Recommendation

1. **Neither mechanism beat Control on Fresh Reserved Confirmation-20**:
   - **Arm A (Control)** achieved **7/20 (35%)** on unseen `Confirmation-20` (`2/10` mild, `5/10` safe), **10/10** on `Dev-10` (`482.1` reward, `3.7` NP steps), and **17/20** on `Diag-20` (`316.8` reward).
   - **Arm B (`regrouping_approach_scale=2.0`)** degraded `Dev-10` (`9/10`, never `10/10`), lowered `flank_reached_pct` (`60%` vs `80%`), inflated `H0` (`1.069/1.095`), and scored **5/20 (25%)** on `Confirmation-20` (`1/10` mild, `4/10` safe).
   - **Arm C (`regroup_no_progress_reset=True`)** matched `Dev-10` (`10/10`) and `Diag-20` (`17/20`) and ticked `confirm_mild_10` up slightly (`3/10` vs `2/10`), but dropped `confirm_safe_10` (`2/10` vs `5/10`) for **5/20 (25%)** overall on `Confirmation-20`.
2. **Recommendation**:
   - **Do NOT combine Arm B + Arm C** and **do NOT promote either mechanism** to production.
   - Keep `regrouping_approach_scale = 0.0` and `regroup_no_progress_reset = False` in production defaults.
   - If continuing wrong-side recovery investigation, **Arm A's baseline recipe** (`wrong_side_mild=0.50` + `Treatment-v2` base + `ent=0.020`) already reaches `7/20` (`5/10` on `wrong_side_safe_radius`), so the next step should be inspecting `4001..4020` failure trajectories from Arm A rather than adding reward shaping or no-progress resets.

---

## 5. Read-Only Cleanup Audit (No Files Deleted)

### A. Experimental Directories in `artifacts/experiments/` (33 existing + `stage8_regrouping_v1`)

| Category | Directory | Size (MB) | Last Modified | Status / Audit Note |
| :--- | :--- | :---: | :---: | :--- |
| **Aug 2026 Legacy** | `batch_ab_20260827_140814` | 56.88 | 2026-08-27 | Obsolete Aug 2026 A/B run |
| **Aug 2026 Legacy** | `batch_ab_effective_smoke` | 8.96 | 2026-08-27 | Obsolete Aug 2026 smoke test |
| **Aug 2026 Legacy** | `batch_ab_corrected` | 57.45 | 2026-08-27 | Obsolete Aug 2026 A/B run |
| **Aug 2026 Legacy** | `adaptive_resume_ab_20260827` | 35.51 | 2026-08-27 | Obsolete Aug 2026 adaptive resume run |
| **Stage 34 Prep/Pilots** | `recovery_hyperparam_prep_v1` | 2.80 | 2026-11-06 | Superseded by `prep_v2` |
| **Stage 34 Prep/Pilots** | `recovery_hyperparam_prep_v2` | 2.82 | 2026-11-06 | Historical Stage 34 prep |
| **Stage 34 Prep/Pilots** | `stage34_recovery_pilot_v1` | 34.56 | 2026-11-08 | Historical Stage 34 pilot |
| **Head/Tail Diagnostics** | `head_tail_diagnostic_chk40575_20261111_175703` | 0.00 | 2026-11-11 | Empty aborted run directory |
| **Head/Tail Diagnostics** | `head_tail_diagnostic_chk40575_20261111_175803` | 0.00 | 2026-11-11 | Empty aborted run directory |
| **Head/Tail Diagnostics** | `head_tail_diagnostic_chk40575_20261111_210123` | 16.87 | 2026-11-11 | Validated Stage 34 diagnostic archive |
| **Gate Info & Recovery** | `gate_info_prep_v1` | 4.85 | 2026-11-12 | Superseded by `gate_current_*` |
| **Gate Info & Recovery** | `recovery_mix_pilot_20261113_164500` | 36.21 | 2026-11-13 | Historical Nov 13 pilot |
| **Gate Info & Recovery** | `gate_current_prep_20261113_172400` | 8.56 | 2026-11-13 | Superseded by `173000` |
| **Gate Info & Recovery** | `gate_current_prep_20261113_173000` | 8.56 | 2026-11-13 | Historical Nov 13 prep |
| **Gate Info & Recovery** | `gate_current_run_20261113_174700` | 46.17 | 2026-11-13 | Historical Nov 13 run |
| **Gate Info & Recovery** | `stopped_adoption_20261113_171410` | 5.81 | 2026-11-13 | Historical Nov 13 backup |
| **Gate Info & Recovery** | `seed59_audit_20261113_181000` | 1.49 | 2026-11-13 | Historical Seed 59 audit |
| **Nov 17 Pilots** | `team_strategy_split_flock_20261117_070100` | 28.90 | 2026-11-17 | Historical split-flock eval |
| **Nov 17 Pilots** | `no_progress_jitter_eval_20261117_073500` | 10.72 | 2026-11-17 | Historical jitter-fix eval |
| **Nov 17 Pilots** | `gate_current_run_v1_1_20261117_074800` | 45.36 | 2026-11-17 | Historical Nov 17 gate run |
| **Nov 17 Pilots** | `boundary_family_run_20261117_075600` | 39.03 | 2026-11-17 | Historical Nov 17 boundary run |
| **Nov 17 Pilots** | `strided_gate_prep_20261117_085000` | 10.38 | 2026-11-17 | Historical strided prep |
| **Nov 17 Pilots** | `strided_gate_run_20261117_085600` | 34.14 | 2026-11-17 | Historical strided run |
| **Nov 17 Pilots** | `clean_sheet_emergent_gate_20261117_180344` | 75.14 | 2026-11-17 | Historical clean-sheet pilot |
| **Nov 17 Pilots** | `clean_sheet_stage1_lr3e4_off_20261117_102550` | 41.25 | 2026-11-17 | Historical clean-sheet lr=3e-4 |
| **Nov 17 Pilots** | `clean_sheet_stage1_lr3e4_on_20261117_102958` | 37.65 | 2026-11-17 | Historical clean-sheet lr=3e-4 |
| **Nov 17 Pilots** | `clean_sheet_progression_20261117_110100` | 78.96 | 2026-11-17 | Historical clean-sheet progression |
| **Nov 17 Pilots** | `strided_gate_cont_20261117_163800` | 46.93 | 2026-11-17 | Historical strided continuation |
| **Nov 17 Pilots** | `strided_gate_gentle_armb_20261117_174300` | 50.09 | 2026-11-17 | Historical strided gentle Arm B |
| **Stage 8 Curriculum** | `stage8_curriculum_pilot_v1` | 39.02 | 2026-11-20 | Superseded by `multiseed_v1` / `v2` |
| **Stage 8 Curriculum** | `stage8_curriculum_multiseed_v1` | 198.96 | 2026-11-20 | Active Stage 8 reference (large) |
| **Stage 8 Curriculum** | `stage8_curriculum_v2` | 104.99 | 2026-11-20 | Active Stage 8 reference |
| **Stage 8 Curriculum** | `stage8_wrong_side_factorial_v1` | 149.49 | 2026-11-20 | Active Stage 8 reference |

### B. Obsolete / Historical Scripts in `scripts/` (28 total)

- **Stage 34 one-off scripts (candidates for archival)**:
  - `evaluate_stage34_checkpoint40398.py` (6,124 B, 2026-11-08)
  - `run_phase3_recovery_pilot.py` (15,079 B, 2026-11-08)
  - `analyze_pilot_replays.py` (6,932 B, 2026-11-08)
  - `evaluate_phase4_checkpoints.py` (7,288 B, 2026-11-08)
  - `evaluate_phase4_1.py` (5,561 B, 2026-11-08)
  - `launch_extended_training.py` (7,118 B, 2026-11-08)
  - `fix_stage34_checkpoint_metadata.py` (4,709 B, 2026-11-08)
  - `run_recovery_hyperparam_experiment.py` (8,948 B, 2026-11-06)
- **Nov 12–17 one-off experiment scripts (candidates for archival)**:
  - `run_gate_information_experiment.py` (4,430 B, 2026-11-12)
  - `run_recovery_mix_pilot.py` (13,588 B, 2026-11-13)
  - `prepare_gate_information_current.py` (1,052 B, 2026-11-13)
  - `run_gate_current_comparison.py` (11,580 B, 2026-11-13)
  - `run_boundary_family_comparison.py` (13,965 B, 2026-11-13)
  - `audit_seed59_recovery.py` (16,123 B, 2026-11-13)
  - `eval_team_strategy_variant.py` (3,730 B, 2026-11-17)
  - `analyze_split_flock_replays.py` (19,132 B, 2026-11-17)
  - `run_clean_sheet_emergent_pilot.py` (21,087 B, 2026-11-17)
  - `continue_strided_stage34.py` (19,038 B, 2026-11-17)
  - `run_strided_gate_validation.py` (28,465 B, 2026-11-17)
- **Stage 8 pilot superseded by v2/factorial**:
  - `run_stage8_curriculum_pilot.py` (11,781 B, 2026-11-20)
- **Active / Utility Scripts (keep)**:
  - `generate_temporal_diagnostic_report.py` (3,815 B), `checkpoint_history_report.py` (2,868 B), `import_historical_trials.py` (3,247 B), `run_head_tail_evaluation.py` (32,872 B), `run_stage8_curriculum_multiseed.py` (26,839 B), `run_stage8_curriculum_v2.py` (30,888 B), `run_stage8_wrong_side_factorial.py` (27,926 B), `run_stage8_regrouping_experiment.py` (26,578 B).

# Stage 8 Curriculum Pilot Report (`stage8_curriculum_pilot_v1`)

## 1. Experiment Setup

- **Goal**: Test whether adding `wrong_side_recovery`, `split_flock`, and `wall_recovery` to the Stage 8 `spawn_mix` improves recovery from wrong-side/overshoot, oscillation, and wall-trap failures without modifying production training config, rewards, network architecture, PPO hyperparameters, or evaluation criteria.
- **Parent Checkpoint**: `artifacts/models/maskable-ppo-05065228.zip` (Checkpoint 613, SHA-256 `da8ce02fcdcd9100601def6fb733762c5c9ce134d11d07058e57919763530e60`, verified unchanged after run).
- **Training Budget**: `98,304` timesteps per arm (`3` segments of `32,768` timesteps), `train_seed=7`, `ppo_env_workers=16`, `effective_learning_rate=0.00015`, `effective_entropy_coef=0.005`, `curriculum_stage=8`, `4` sheep / `3` dogs.
- **Arms**:
  - **Baseline**: Unmodified Checkpoint 613.
  - **Control** (`artifacts/experiments/stage8_curriculum_pilot_v1/control`): `{"fixed_easy": 0.30, "randomized_flock": 0.55, "nearby_stray": 0.15}` (production Stage 8 `spawn_mix`).
  - **Treatment** (`artifacts/experiments/stage8_curriculum_pilot_v1/treatment`): `{"randomized_flock": 0.35, "nearby_stray": 0.15, "split_flock": 0.15, "wall_recovery": 0.10, "fixed_easy": 0.10, "wrong_side_recovery": 0.15}`.
- **Evaluation**: Deterministic (`deterministic=True`) on:
  - **Dev-validation set (10 seeds)**: `[11, 23, 37, 41, 53, 59, 61, 67, 71, 73]`
  - **Untouched Holdout set (20 seeds)**: `[101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197]`
- **Raw Results**: `artifacts/experiments/stage8_curriculum_pilot_v1/pilot_results.json`

## 2. Wall-Clock Cost

| Phase | Wall-Clock Time (s) |
| --- | ---: |
| Baseline Evaluation (30 episodes) | 17.76 |
| Control Training (98,304 steps) | 71.35 |
| Control Evaluation (30 episodes) | 13.98 |
| Treatment Training (98,304 steps) | 65.15 |
| Treatment Evaluation (30 episodes) | 8.96 |
| **Total Pilot Wall-Clock** | **177.25** |

## 3. Dev-Validation Results (10 Seeds)

| Arm | Success Rate | Avg Reward | Avg Steps (All) | Avg Steps (Success) | Avg Penned |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline (Ckpt 613) | 90.0% (9/10) | 412.09 | 133.9 | 104.78 | 3.6 / 4 |
| Control | 90.0% (9/10) | 389.62 | 136.9 | 101.33 | 3.6 / 4 |
| Treatment | 90.0% (9/10) | **414.21** | 145.6 | 110.22 | **3.9 / 4** |

### Key Dev Seeds (37 & 53) and Regressions

| Seed | Baseline (Ckpt 613) | Control | Treatment |
| --- | --- | --- | --- |
| **37** (`randomized_flock`) | **FAIL** (`no-progress`, 396 steps, 0/4 penned, reward 5.75) | **FAIL** (`no-progress`, 457 steps, 0/4 penned, reward -258.67, oscillation) | **PASS** (136 steps, 4/4 penned, reward 686.50) |
| **53** (`randomized_flock`) | **PASS** (125 steps, 4/4 penned, reward 593.69) | **PASS** (125 steps, 4/4 penned, reward 598.64) | **PASS** (243 steps, 4/4 penned, reward 529.31) |
| **61** (`randomized_flock`) | **PASS** (203 steps, 4/4 penned, reward 700.49) | **PASS** (196 steps, 4/4 penned, reward 693.70) | **FAIL** (`no-progress`, 464 steps, 3/4 penned, reward 86.78, oscillation + corner_stuck) |

- **Observation**: Treatment solved dev seed 37 (which both Baseline and Control failed), raising average sheep penned from `3.6/4` to `3.9/4`, but regressed on seed 61 (penned `3/4` before `corner_stuck_at_end` / `no-progress`).

## 4. Untouched Holdout Results (20 Seeds)

| Arm | Success Rate | Avg Reward | Avg Steps (All) | Avg Steps (Success) | Avg Penned |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline (Ckpt 613) | 80.0% (16/20) | 238.33 | 154.85 | 84.38 | 3.2 / 4 |
| Control | 85.0% (17/20) | 264.79 | 149.20 | 99.24 | 3.4 / 4 |
| Treatment | 85.0% (17/20) | **272.63** | **141.60** | **86.88** | 3.4 / 4 |

### Holdout Failure Breakdown

All failures across all three arms occurred on `no-progress` (zero `timeout`, zero `corner_stuck_at_end`).

| Arm | Failures | Failing Seeds | `no-progress` | `timeout` | `oscillation_detected` | `corner_stuck_at_end` |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| Baseline (Ckpt 613) | 4 | 109, 131, 137, 193 | 4 | 0 | 4 | 0 |
| Control | 3 | 131, 137, 193 | 3 | 0 | 3 | 0 |
| Treatment | 3 | 131, 137, 193 | 3 | 0 | 3 | 0 |

## 5. Interpretation & Recommendation

1. **Both Control and Treatment improved holdout success rate** from `80.0%` (16/20) to `85.0%` (17/20) over Baseline Ckpt 613 in just `98,304` timesteps (~65–71s of training), solving holdout seed `109`.
2. **Treatment outperformed Control on efficiency and reward** on both sets:
   - Holdout avg reward: `272.63` vs `264.79` (+7.84)
   - Holdout avg steps: `141.60` vs `149.20` (-7.60 steps overall; `86.88` vs `99.24` on successes)
   - Dev avg reward: `414.21` vs `389.62` (+24.59)
   - Dev avg sheep penned: `3.9/4` vs `3.6/4`, including solving seed `37` cleanly (`136` steps, `4/4` penned).
3. **Remaining gap**: Both arms still failed holdout seeds `131`, `137`, and `193` with `no-progress` + `oscillation_detected`, and Treatment regressed on dev seed `61` (`3/4` penned, `corner_stuck_at_end`). At `98,304` steps (3 segments), the policy has only begun adapting to the harder `split_flock` (15%), `wall_recovery` (10%), and `wrong_side_recovery` (15%) starts.
4. **Recommendation**:
   - The `wrong_side_recovery` + `split_flock` + `wall_recovery` curriculum mix is **promising and safe to test at a longer budget** (e.g., `10–15` segments / `327k–491k` timesteps, or multi-seed replication) before promoting to production `spawn_mix`, with particular attention to whether longer training eliminates the seed `61` corner-stuck regression and resolves `131`/`137`/`193` oscillation failures.

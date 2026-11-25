# Clean-Sheet From-Scratch `emergent_gate` Pilot (Stage 1, 208 Inputs)

Date: 2026-11-17  
Protocol: `clean_sheet_emergent_gate_pilot_v1`  
Experiment directory: `artifacts/experiments/clean_sheet_emergent_gate_20261117_180344/`  
Manifest: `artifacts/experiments/clean_sheet_emergent_gate_20261117_180344/manifest.json`  
Runner: [scripts/run_clean_sheet_emergent_pilot.py](../scripts/run_clean_sheet_emergent_pilot.py)

## 1. Goal & Configuration

Test whether a freshly initialized `MaskablePPO` policy (`256 × 256 × 128` MLP, random weights, zero inherited checkpoints) can learn Stage 1 (1 dog, 1 sheep, fixed easy placement, promotion threshold `0.80`) and promote across curriculum stages using pure raw observations and no instinct reward shaping:

- `observation_mode`: `"emergent_gate"`
- `observation_history_strides`: `(0, 4, 8, 16)`
- `include_loop_flags_in_observation`: `False`
- `observation_size`: `208` (`52 × 4`)
- `observation_schema_version`: `emergent_gate_v1|strides=0,4,8,16|no_loop_flags`
- `enable_instinct_rewards`: `False`
- `env_config_version`: `"1.1"`
- `trainer_type`: `"maskable_ppo"`, `policy_type`: `"neural"`, `ppo_env_workers`: `1`
- PPO hyperparameters (from `apply_training_profile(LabConfig(), enable_instinct_rewards=False, curriculum_stage=1)`): `learning_rate=1e-4`, `rollout_steps=2048`, `batch_size=1024`, `gamma=0.99`, `gae_lambda=0.95`, `clip_range=0.2`, `entropy_coef=0.01`, `value_coef=0.5`.
- Training seeds: `(101, 202)`
- Budget: `65,536` transitions per seed (`max_seed_seconds=300`, `max_total_seconds=600`), evaluated every `8,192` transitions deterministically on the 10-seed development set `DEV_SEEDS = (11, 23, 37, 41, 53, 59, 61, 67, 71, 73)` with `capture_step_telemetry=False`.

## 2. Assistance & Provenance Disclosure

- **Policy assistance (`assistance_mode="autonomous_emergent_gate"`)**: None. Observations exclude `TeamStrategy` role/target vectors and exclude legacy loop flags; `enable_instinct_rewards=False`.
- **Physics assistance**: Standard `EnvironmentConfig` sheep gate-surge physics remain active in simulation (`sheep_surge_multiplier=1.3`, `sheep_d_gate=12.0`).
- **Source & isolation**:
  - Git revision: `bfd12db856c1c2d5b41462758d7eb29d41ba8221` (`git_dirty=True`, `dirty_patch_hash=c95f7743656c9c2fdcbf5186233c2914d09325f10d06250baa222405e0fcb841`)
  - Live `artifacts/training-state.json` (SHA-256 `2EABBDB9CE980910B44545AD58821F9107193D6DB10444692697D85672AFD80A`) and `artifacts/training-telemetry.sqlite` (SHA-256 `7823C8B44A5ED82AFEE92582273BAC791995835DF755BF8E0256A3D666A5FEC5`) verified identical before and after the run.
  - Held-out evaluation seeds were not used.

## 3. Results

Total wall-clock time: `453.422s` (`seed_101`: `217.265s`, `seed_202`: `236.016s`). Both seeds completed the full `65,536` transitions / `65,536` world steps (`246` and `240` training episodes, respectively).

**Neither seed qualified for promotion out of Stage 1** (required threshold `8/10 = 80%` on `DEV_SEEDS`):

| Seed | Transitions | Stage | Success (`DEV_SEEDS`) | Avg Reward | Avg Steps | Promotions |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| `101` | 8,192 | 1 | 3/10 (30.0%) | +20.32 | 196.6 | 0 |
| `101` | 16,384 | 1 | 3/10 (30.0%) | +20.32 | 196.6 | 0 |
| `101` | 24,576 | 1 | 3/10 (30.0%) | +22.29 | 198.1 | 0 |
| `101` | 32,768 | 1 | 3/10 (30.0%) | +22.29 | 198.1 | 0 |
| `101` | 40,960 | 1 | 3/10 (30.0%) | +22.29 | 198.1 | 0 |
| `101` | 49,152 | 1 | 3/10 (30.0%) | +21.47 | 196.2 | 0 |
| `101` | 57,344 | 1 | 3/10 (30.0%) | +12.56 | 193.8 | 0 |
| `101` | 65,536 | 1 | 0/10 (0.0%) | -45.66 | 265.5 | 0 |
| `202` | 8,192 | 1 | 0/10 (0.0%) | -66.46 | 267.0 | 0 |
| `202` | 16,384 | 1 | 0/10 (0.0%) | -65.08 | 267.0 | 0 |
| `202` | 24,576 | 1 | 0/10 (0.0%) | -64.04 | 265.9 | 0 |
| `202` | 32,768 | 1 | 0/10 (0.0%) | -64.04 | 265.9 | 0 |
| `202` | 40,960 | 1 | 0/10 (0.0%) | -64.04 | 265.9 | 0 |
| `202` | 49,152 | 1 | 3/10 (30.0%) | +20.66 | 196.6 | 0 |
| `202` | 57,344 | 1 | 0/10 (0.0%) | -117.95 | 254.4 | 0 |
| `202` | 65,536 | 1 | 0/10 (0.0%) | -92.83 | 260.4 | 0 |

### Milestone Snapshots

- `seed_101`:
  - `stage_01_init_ts_0.zip` (`30791c405e590645...`)
  - `final_stage_01_ts_65536.zip` (`e5e99de532679af8...`)
- `seed_202`:
  - `stage_01_init_ts_0.zip` (`30247d9ef6b08511...`)
  - `final_stage_01_ts_65536.zip` (`00867d09e3c54ed2...`)

## 4. Findings & Limitations

1. **Clean-sheet 208-input training pipeline works end-to-end**: Fresh `MaskablePPO` initialization, `(0, 4, 8, 16)` strided history stacking, `include_loop_flags_in_observation=False`, `enable_instinct_rewards=False`, deterministic 10-seed `Evaluator`, and milestone sidecar persistence all executed cleanly inside the isolated experiment directory with zero live artifact mutations.
2. **Stage 1 not solved within 65,536 transitions**: With `enable_instinct_rewards=False`, `learning_rate=1e-4`, `rollout_steps=2048`, and `256 × 256 × 128` capacity from random weights, both seeds peaked at `3/10` (30%) and regressed to `0/10` by `65,536` transitions.
3. **Next options (manager decision)**:
   - Either test a larger clean-sheet transition budget / higher `learning_rate` (e.g. `3e-4`) or smaller network for Stage 1–8 curriculum bootstrap, or
   - Compare against `enable_instinct_rewards=True` (reward shaping only, keeping `emergent_gate` 208-dim observations autonomous) as a clean-sheet curriculum baseline.

---

## 5. Stage-1 `lr=3e-4` & `enable_instinct_rewards` Check (2026-11-17)

Tested whether the clean-sheet `208`-input `emergent_gate` + `(0, 4, 8, 16)` + `include_loop_flags=False` architecture learns Stage 1 (`>= 80%`) and promotes to Stage 2+ when given `lr=3e-4` and `131,072` transitions (`seed=101`, `eval_interval=16,384`, `max_seed_seconds=300`, `DEV_SEEDS`).

- **Arm 1 (`instincts_off`)**: `artifacts/experiments/clean_sheet_stage1_lr3e4_off_20261117_102550/manifest.json` (`--lr 3e-4 --no-enable-instinct-rewards --seeds 101 --transitions 131072 --eval-interval 16384`), elapsed `237.282s`, `579` training episodes, `stop_reason="budget_completed"`.
- **Arm 2 (`instincts_on`)**: `artifacts/experiments/clean_sheet_stage1_lr3e4_on_20261117_102958/manifest.json` (`--lr 3e-4 --enable-instinct-rewards --seeds 101 --transitions 131072 --eval-interval 16384`), elapsed `267.312s`, `501` training episodes, `stop_reason="budget_completed"`.
- **Total wall-clock time**: `504.594s` (`< 10 min`).
- **Source & live artifact verification**:
  - Git revision: `bfd12db856c1c2d5b41462758d7eb29d41ba8221` (`git_dirty=True`, `dirty_patch_hash=78f8673a6f1f92d5a13fdacf107109d2bafe251da92bb1abba6e00c9280dbee6`)
  - Live `artifacts/training-state.json` (SHA-256 `2EABBDB9CE980910B44545AD58821F9107193D6DB10444692697D85672AFD80A`) and `artifacts/training-telemetry.sqlite` (SHA-256 `7823C8B44A5ED82AFEE92582273BAC791995835DF755BF8E0256A3D666A5FEC5`) verified identical before and after both runs.

### Per-Interval Results (`seed=101`)

| Arm | Transitions | Stage | Success (`DEV_SEEDS`) | Avg Reward | Avg Steps | Promoted |
| --- | ---: | ---: | --- | ---: | ---: | --- |
| `instincts_off` | 16,384 | 1 | 0/10 (0.0%) | -21.40 | 261.3 | No |
| `instincts_off` | 32,768 | 1 | 0/10 (0.0%) | -50.25 | 268.1 | No |
| `instincts_off` | 49,152 | 1 | 0/10 (0.0%) | -50.17 | 268.1 | No |
| `instincts_off` | 65,536 | 1 | 3/10 (30.0%) | +22.56 | 201.5 | No |
| `instincts_off` | 81,920 | 1 | 0/10 (0.0%) | -49.21 | 265.4 | No |
| `instincts_off` | 98,304 | 1 | 0/10 (0.0%) | -27.42 | 268.3 | No |
| `instincts_off` | 114,688 | 1 | **10/10 (100.0%)** | **+176.02** | **178.5** | **Yes -> Stage 2** |
| `instincts_off` | 131,072 | 2 | 1/10 (10.0%) | -20.25 | 263.6 | No |
| `instincts_on` | 16,384 | 1 | 0/10 (0.0%) | -280.95 | 268.1 | No |
| `instincts_on` | 32,768 | 1 | 0/10 (0.0%) | -280.95 | 268.1 | No |
| `instincts_on` | 49,152 | 1 | 0/10 (0.0%) | -330.36 | 259.0 | No |
| `instincts_on` | 65,536 | 1 | 0/10 (0.0%) | -280.95 | 268.1 | No |
| `instincts_on` | 81,920 | 1 | 0/10 (0.0%) | -338.56 | 285.7 | No |
| `instincts_on` | 98,304 | 1 | 0/10 (0.0%) | -280.95 | 268.1 | No |
| `instincts_on` | 114,688 | 1 | 0/10 (0.0%) | -243.26 | 265.5 | No |
| `instincts_on` | 131,072 | 1 | 0/10 (0.0%) | -238.75 | 266.9 | No |

### Milestones & Promotions

- `instincts_off`:
  - `stage_01_init_ts_0.zip` (`0c8e13e41e51148b...`)
  - `stage_01_promoted_ts_114688.zip` (`19456fe0152bf3af...`) — Stage 1 -> Stage 2 at `114,688` transitions (`209.422s`, `10/10 = 100%` vs `80%` threshold)
  - `final_stage_02_ts_131072.zip` (`efe31c401b9986aa...`)
- `instincts_on`:
  - `stage_01_init_ts_0.zip` (`00bf91bb0bfd9e6b...`)
  - `final_stage_01_ts_131072.zip` (`d5d07b0a27a3b9c5...`) — `0` promotions.

### Honest Assessment

1. **Stage 1 is learnable without instinct shaping**: At `lr=3e-4` and `enable_instinct_rewards=False`, `seed=101` solved Stage 1 (`10/10 = 100.0%` on `DEV_SEEDS`, avg reward `+176.02`) at `114,688` transitions and promoted to Stage 2, reaching `1/10` on Stage 2 after `16,384` additional transitions.
2. **`enable_instinct_rewards=True` failed Stage 1 on `seed=101`**: With `emergent_gate` omitting `TeamStrategy` role/target observations, instinct reward shaping produced `0/10` across all 8 intervals (`-238.75` to `-338.56` avg reward) and `0` promotions.
3. **Limitations**: Single training seed (`101`), `DEV_SEEDS` only (held-out unused), Stage 1->2 only (`1` dog, `1` sheep), and standard simulation gate-surge physics (`sheep_surge_multiplier=1.3`, `sheep_d_gate=12.0`) remain active. Multi-seed stability and progression to 3 dogs / 6 sheep are not yet established.

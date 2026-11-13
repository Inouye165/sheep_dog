# Stage 34 Gate-Information Observation Ablation Experiment

## Purpose & Hypothesis

Diagnostic telemetry analysis of Stage 34 Checkpoint 40575 (`maskable-ppo-stage34-339740239.zip`, SHA-256 `bc6b9883ab5f9343255c15e59cbf0252e8268808731482fec2f5b417d3196a00`) identified closed-fence blindness as a candidate cause of late-episode stalls (notably development seeds 53 and 67): the 54-dimensional `"guided"` observation vector includes `pen_x`/`pen_y` (pen center) but no explicit gate-opening coordinates or outward gate normal.

This experiment tests a single-factor observation intervention from the pinned 54-input Stage 34 parent checkpoint:

- **Arm A (`arm_a_control_zero_gate`)**: `observation_mode="guided_gate_control"` — 58 inputs consisting of the exact 54 existing `"guided"` features followed by 4 constant-zero features `(0.0, 0.0, 0.0, 0.0)`.
- **Arm B (`arm_b_treatment_gate_info`)**: `observation_mode="guided_gate_treatment"` — 58 inputs consisting of the exact 54 existing `"guided"` features followed by 4 explicit gate features (`GATE_OBSERVATION_SCHEMA_VERSION = "guided_gate_v1"`):
  - `gate_center_dx = clamp((gate_center.x - dog.x) / max(1, width - 1), -1.0, 1.0)`
  - `gate_center_dy = clamp((gate_center.y - dog.y) / max(1, height - 1), -1.0, 1.0)`
  - `gate_normal_x`, `gate_normal_y`: outward unit normal `(-vx, -vy)` from `environment._gate_vector_into_pen()` (`bottom -> (0, 1)`, `top -> (0, -1)`, `left -> (-1, 0)`, `right -> (1, 0)`).

## Architecture & Weight Conversion (`54 -> 58`)

`expand_policy_observation_inputs(...)` in `src/sheepdog/policies/neural.py`:

- Expands `mlp_extractor.policy_net[0]` and `mlp_extractor.value_net[0]` (plus `shared_net` or `Linear` feature-extractor layers if present) from `in_features=54` to `in_features=58`.
- Preserves `[:, :54]` weights and all biases bit-for-bit; initializes `[:, 54:58]` to exact `0.0`.
- Updates `model.observation_space`, `model.policy.observation_space`, feature extractor `_features_dim`, and `model.rollout_buffer` to shape `(58,)`.
- **Fair optimizer strategy (approximate continuation)**: Rebuilds `model.policy.optimizer` with fresh empty Adam moment buffers bound to the new parameters in **both** Arm A and Arm B identically.

## Locked Protocol & Budget

- **Parent checkpoint**: `artifacts/checkpoints/stage34/maskable-ppo-stage34-339740239.zip` (SHA-256 `bc6b9883ab5f9343255c15e59cbf0252e8268808731482fec2f5b417d3196a00`)
- **Parent metadata**: `artifacts/checkpoints/checkpoint-chk_run_20260817_063453_91f4_pv_10369_ts_339740239.json`
- **Locked hyperparameters**: `net_arch=[256, 256, 128]`, `gamma=0.997`, `gae_lambda=0.98`, `ent_coef=0.0075`, `learning_rate=5e-5`, `n_steps=2048`, `batch_size=1024`, Stage 34, 3 dogs, 6 sheep, `max_steps=2320`, `no_progress_window=780`, `enable_adaptive_learning=False`, `auto_rollback_on_collapse=False`, auto-promotion disabled.
- **Paired training seeds**: `(101, 202, 303)`
- **Pilot budget**: `90,112` policy transitions per arm-seed (`44` PPO rollouts of `2,048` transitions = `30,037.33` world steps under `sequential_shared_dog`; `4` checkpoints of `22,528` transitions = `11` rollouts each). Wall-clock limits: `1,800s` per arm-seed, `10,800s` total.
- **Development evaluation manifest**: `(11, 23, 37, 41, 53, 59, 61, 67, 71, 73)` with head/tail diagnostic telemetry (`H=200, L=500`) enabled on selected diagnostic evaluations only (never during training).
- **Held-out evaluation**: `pending_manifest_audit`.

## Metrics & Decision Rules

- **Primary metrics**: Full-flock success rate (with explicit denominator `N=10` per training seed, `N=30` pooled), timeout rate, no-progress-stop rate, steps to success (failures reported separately), targeted failure seeds `{53, 67, 73}`, regressions on `{11, 23, 37, 41, 59, 61, 71}`, and `compute_fence_stall_duration(frames, pen_origin, pen_width, pen_height, pen_opening)` (world steps where at least one unpenned sheep is Chebyshev-adjacent to `Pen.fence_cells()` with no penning or route progress).
- **Continue / Reject**: Compare Arm B against equal-budget continued-training Arm A across all 3 seeds (`101, 202, 303`) with per-seed outcomes and Wilson 95% CI.

## CLI Usage

```powershell
# 1. Prepare isolated workspaces and convert 54 -> 58 weights (no training)
.venv/Scripts/python.exe scripts/run_gate_information_experiment.py prepare --experiment-root artifacts/experiments/gate_info_prep_v1

# 2. Verify workspace integrity, config parity, and 58-input models
.venv/Scripts/python.exe scripts/run_gate_information_experiment.py verify --experiment-root artifacts/experiments/gate_info_prep_v1

# 3. Run one isolated (arm, seed) pilot when approved
.venv/Scripts/python.exe scripts/run_gate_information_experiment.py run-arm --experiment-root artifacts/experiments/gate_info_prep_v1 --arm arm_a_control_zero_gate --seed 101
```

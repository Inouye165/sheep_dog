# `no_progress_steps` Best-Consolidation Gating Fix

## Problem
In [src/sheepdog/environment.py](../src/sheepdog/environment.py), `average_distance_delta` and `farthest_to_pen_delta` were already gated on `best_entry_progress` (`newly_penned > 0 or current_entry_distance < self._best_entry_distance - 1e-6`), preventing back-and-forth route oscillation from resetting `_no_progress_steps`. However, `flock_spread_delta` and `farthest_to_flock_center_delta` were computed strictly step-to-step ($t-1 \to t$). As a result, 1-cell sheep bounces decremented `_no_progress_steps` by 1 on alternating steps, keeping `no_progress_steps` near `0` during 2,320-step oscillations (seeds 59, 73) and below `780` during a triple-`wait` deadlock (seed 71).

## Fix Implemented
- Added `_best_flock_spread: float | None` and `_best_farthest_to_flock_center: float | None` to `SheepdogEnvironment` (`__init__`, cleared to `None` in `reset()` and `reset_from_scenario()`).
- In `step()`, initialized both to the pre-step values when `None`, gated `flock_spread_delta` and `farthest_to_flock_center_delta` on `newly_penned > 0 or current < best - 1e-6`, and passed the improvement over the best-so-far (`self._best_* - current_*`) to `_progress_made_for_no_progress(...)`.
- Re-anchored `_best_flock_spread` and `_best_farthest_to_flock_center` whenever `newly_penned > 0` because penning a sheep changes the remaining unpenned flock geometry.
- Retained `self._no_progress_steps = max(0, self._no_progress_steps - 1)` when progress is made.
- Bumped `ENV_CONFIG_VERSION` from `"1.0"` to `"1.1"`.
- No `TeamStrategy` heuristics were modified and no training was run.

## Compatibility Decision
`ENV_CONFIG_VERSION` is bumped `"1.0"` -> `"1.1"`. Gating consolidation progress on best-so-far changes both episode termination (`no-progress` stops at `no_progress_window = 780` when consolidation stalls) and the `no_progress_ratio` observation feature (plus `no_progress_penalty`). Existing `"1.0"` checkpoints (`pv_10486`, `pv_10483`) can be evaluated under `"1.1"` for comparison, but must not be silently treated as having identical observation or termination semantics without retraining/re-baseline.

## Isolated Evaluation Check (No Training, Guided / Assisted)
- **Experiment dir**: `artifacts/experiments/no_progress_jitter_eval_20261117_073500/`
- **Frozen config**: `config.json` copied from `artifacts/experiments/team_strategy_split_flock_20261117_070100/config.json` (Stage 34, `observation_mode: guided`, `no_progress_window: 780`, `max_steps: 2320`, `TeamStrategy` 1.1.0 options all OFF).
- **Models**: `pv_10486` (`maskable-ppo-stage34-343886090.zip`) and `pv_10483` (`maskable-ppo-stage34-343787786.zip`), deterministic evaluation on 10 dev seeds (`11, 23, 37, 41, 53, 59, 61, 67, 71, 73`), `capture_step_telemetry=False`, held-out seeds unused.
- **Live artifacts untouched (SHA-256 identical before and after)**:
  - `artifacts/training-state.json`: `2EABBDB9CE980910B44545AD58821F9107193D6DB10444692697D85672AFD80A`
  - `artifacts/checkpoints/stage34/maskable-ppo-stage34-343886090.zip`: `9A4453F1741CFD28564DF7EEBF89CD05B4983EDEECF703378515491C34915178`
  - `artifacts/checkpoints/stage34/maskable-ppo-stage34-343787786.zip`: `26AECD34C7D1723FEB4AE2F44A03FBDDC16C494B2855A6C26216B31703C54DD2`

### Per-Seed Comparison vs `"1.0"` Baseline (`team_strategy_split_flock_20261117_070100`)

#### `pv_10486` (Baseline: `7/10`, avg reward `-922.75`, avg steps `947.4` | `"1.1"`: `7/10`, avg reward `-432.18`, avg steps `564.0`)
| Seed | Baseline (`1.0`) Stop / Steps / Penned / Final `np` / Reward | Fixed (`1.1`) Stop / Steps / Penned / Final `np` / Reward |
|---:|---|---|
| 11 | `success` / 248 / 6/6 / `np=0` / 280.60 | `success` / 248 / 6/6 / `np=0` / 279.57 |
| 23 | `success` / 804 / 6/6 / `np=6` / -608.68 | `success` / 341 / 6/6 / `np=54` / 83.51 |
| 37 | `success` / 380 / 6/6 / `np=5` / 267.82 | `success` / 558 / 6/6 / `np=122` / -19.82 |
| 41 | `success` / 232 / 6/6 / `np=0` / 223.80 | `success` / 227 / 6/6 / `np=11` / 222.54 |
| 53 | `success` / 233 / 6/6 / `np=0` / 359.51 | `success` / 233 / 6/6 / `np=0` / 359.51 |
| 59 | `timeout` / 2320 / 0/6 / `np=0` / -3725.91 | `no-progress` / 1098 / 0/6 / `np=780` / -2118.46 |
| 61 | `success` / 356 / 6/6 / `np=2` / 201.11 | `success` / 482 / 6/6 / `np=146` / -24.27 |
| 67 | `success` / 261 / 6/6 / `np=0` / 275.70 | `success` / 466 / 6/6 / `np=239` / -122.20 |
| 71 | `timeout` / 2320 / 3/6 / `np=633` / -2351.57 | `no-progress` / 1116 / 3/6 / `np=780` / -1316.69 |
| 73 | `timeout` / 2320 / 0/6 / `np=0` / -4149.90 | `no-progress` / 871 / 0/6 / `np=780` / -1665.51 |

#### `pv_10483` (Baseline: `8/10`, avg reward `-1080.00`, avg steps `883.1` | `"1.1"`: `6/10`, avg reward `-628.05`, avg steps `629.1`)
| Seed | Baseline (`1.0`) Stop / Steps / Penned / Final `np` / Reward | Fixed (`1.1`) Stop / Steps / Penned / Final `np` / Reward |
|---:|---|---|
| 11 | `success` / 165 / 6/6 / `np=0` / 341.95 | `success` / 165 / 6/6 / `np=0` / 341.89 |
| 23 | `success` / 204 / 6/6 / `np=1` / 345.53 | `success` / 204 / 6/6 / `np=1` / 345.48 |
| 37 | `success` / 1071 / 6/6 / `np=127` / -946.55 | `success` / 512 / 6/6 / `np=87` / 179.98 |
| 41 | `success` / 962 / 6/6 / `np=754` / -1276.04 | `success` / 266 / 6/6 / `np=92` / 149.73 |
| 53 | `success` / 235 / 6/6 / `np=0` / 379.25 | `success` / 235 / 6/6 / `np=0` / 379.14 |
| 59 | `timeout` / 2320 / 0/6 / `np=0` / -4378.75 | `no-progress` / 857 / 0/6 / `np=780` / -1649.29 |
| 61 | `success` / 584 / 6/6 / `np=3` / -100.73 | `success` / 958 / 6/6 / `np=429` / -693.80 |
| 67 | `success` / 403 / 6/6 / `np=0` / -208.86 | `no-progress` / 1149 / 5/6 / `np=780` / -2102.70 |
| 71 | `success` / 567 / 6/6 / `np=6` / -96.90 | `no-progress` / 1075 / 0/6 / `np=780` / -1302.35 |
| 73 | `timeout` / 2320 / 0/6 / `np=1` / -4858.90 | `no-progress` / 870 / 0/6 / `np=780` / -1928.55 |

### Observations & Limitations
- All five previously timed-out episodes (`pv_10486` seeds 59, 71, 73; `pv_10483` seeds 59, 73) now terminate cleanly with `stop_reason="no-progress"` at `no_progress_steps=780` (steps 857–1116) instead of running to the 2,320-step timeout with `no_progress_steps` suppressed (`0`–`633`).
- On `pv_10486`, all 7 passing seeds continue to pass (`7/10`). On `pv_10483`, seeds 67 (`5/6` at step 1149) and 71 (`0/6` at step 1075) now stop at `no-progress` instead of succeeding, and several passing seeds on both models change step counts because `no_progress_ratio` is part of the policy observation (so un-suppressed `no_progress_steps` shifts the `"1.0"`-trained policy's trajectory).
- **Verification**:
  - `.venv\Scripts\python.exe -m pytest tests/test_no_progress_jitter.py tests/test_team_strategy_split_flock.py tests/test_evaluation_ground_truth_replays.py -q`
  - `.venv\Scripts\python.exe -m ruff check src/sheepdog/environment.py tests/test_no_progress_jitter.py`

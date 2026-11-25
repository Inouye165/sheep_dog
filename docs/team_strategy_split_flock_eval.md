# TeamStrategy split-flock / corner options — evaluation (ASSISTED / guided, no training)

## Hypotheses (to verify, not proven causes)
1. Rear/flank targets use the centroid of *all* unpenned sheep, so a stray drags them between groups (`main_group_targets`).
2. A stalled role holder never yields because the stagnation penalty floors the bonus at -5 (`collector_yield`: penalty up to 40, floor -30, applies to any role held >150 steps).
3. Dogs box a wall/corner stray at distance 1 (`corner_clearance`: collector approaches from the open side at 4 cells; other targets pushed ≥4 cells from a wall stray).

All options are scripted guidance (**assisted**), default OFF (`TeamStrategy.DEFAULT_*` class attrs / keyword ctor args), VERSION 1.1.0. Defaults reproduce 1.0.0 targets exactly (golden file `tests/golden_team_strategy_v1_0_0.json` generated from `git HEAD` code). No teleporting, no forced sheep movement, no reset-as-success. `environment.py` untouched.

## Setup
- Experiment dir: `artifacts/experiments/team_strategy_split_flock_20261117_070100/` (models copied with hashes, `config.json` = frozen `gate_current_run_20261113_174700/baseline_config.json`, guided mode, `dirty.patch`, `source_revision.txt`, per-run `result.json` + replays under `<model>__<variant>/`).
- Source revision `bfd12db856c1c2d5b41462758d7eb29d41ba8221`, **dirty tree** (see `dirty.patch`; includes unrelated pre-existing work).
- Models (SHA-256): pv_10486 = `9a4453f1741cfd28564dff...` (full: `9A4453F1741CFD28564DF7EEBF89CD05B4983EDEECF703378515491C34915178`, stage34-343886090); pv_10483 = `26AECD34C7D1723FEB4AE2F44A03FBDDC16C494B2855A6C26216B31703C54DD2` (stage34-343787786).
- Dev seeds 11,23,37,41,53,59,61,67,71,73; deterministic; 2320-step budget; no step telemetry; held-out unused. 10 evaluations of 10 episodes (≤ budget).
- Command: `python -m scripts.eval_team_strategy_variant --out <dir> --model <zip> --model-name <name> --variant {baseline,main_group,collector_yield,corner_clearance,all} --config <dir>/config.json` (PYTHONPATH=src;.).

## Results (S = success, F = fail/timeout; steps)
| Model/variant | pooled | 11 | 23 | 37 | 41 | 53 | 59 | 61 | 67 | 71 | 73 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 10486 baseline | 7/10 | S248 | S804 | S380 | S232 | S233 | F | S356 | S261 | F | F |
| 10486 main_group | 9/10 | S235 | S729 | S445 | S192 | S1037 | S413 | S187 | S512 | S410 | F |
| 10486 collector_yield | 8/10 | S265 | S804 | S380 | S232 | S233 | F | S356 | S261 | S491 | F |
| 10486 corner_clearance | 9/10 | S597 | S804 | S353 | S305 | S233 | S382 | S356 | S783 | S347 | F |
| 10486 all | 7/10 | S1001 | S656 | S751 | S192 | S530 | F | S187 | **F** | S446 | F |
| 10483 baseline | 8/10 | S165 | S204 | S1071 | S962 | S235 | F | S584 | S403 | S567 | F |
| 10483 main_group | 5/10 | S613 | S464 | S602 | **F** | **F** | F | S643 | **F** | S1001 | F |
| 10483 collector_yield | 7/10 | S165 | S204 | S460 | **F** | S235 | F | S943 | S423 | S710 | F |
| 10483 corner_clearance | 8/10 | S165 | S204 | S477 | S962 | S235 | F | S474 | S551 | S878 | F |
| 10483 all | 6/10 | S597 | S626 | S1396 | **F** | S1132 | F | S898 | **F** | S527 | F |

Bold = regression vs that model's baseline. Pooled over both models (20 episodes): baseline 15/20, main_group 14/20, collector_yield 15/20, corner_clearance 17/20, all 13/20. Seed 73 failed in every run. Per-episode failure types, penned counts and replays are in each `result.json` / replay dir.

## Reading
- Observed: the options help pv_10486 (main_group, corner_clearance each fixed 59 and 71) but main_group and all **hurt** pv_10483 (4 regressions); all-together is worse than the parts. Effects are model-specific.
- Inferred only: guided observations include TeamStrategy targets, so checkpoints trained on 1.0.0 targets see shifted inputs; part of the changes may be input shift, not better geometry. Not separated here.
- No variant is a robust fix; single-evaluation-per-cell, two correlated models, 10 development seeds: no uncertainty claims, not release evidence. Seed 73 (stray on bottom wall) unresolved.

## Compatibility decision
Evaluation-only. No live adoption, no config/default change, no checkpoint reinterpretation; any adoption needs retraining/re-evaluation with the new target semantics and held-out evaluation.

## Limitations
No step telemetry or per-failure causal replay analysis of the new variants performed; evaluator "pv" parity with the manager's earlier numbers: baseline pv_10486 reproduced 7/10 with failures 59/71/73. Live artifacts verified unchanged: `artifacts/training-state.json` SHA-256 `2EABBDB9…AFD80A`, both zips hashes identical before/after.

## Offline step-level replay analysis — 2026-11-17 (ASSISTED / guided, read-only)

- **Script**: [scripts/analyze_split_flock_replays.py](../scripts/analyze_split_flock_replays.py) (`PYTHONPATH=src;.`) using `analyze_episode_temporal_interactions`, `analyze_repetition_and_recovery`, and reconstructed `TeamStrategy` targets from recorded `initial_snapshot` and per-frame snapshots.
- **Provenance note**: All runs here are **assisted / guided** (`54`-input policy with `TeamStrategy` targets in the observation). Reconstructed targets are estimated from recorded pre-step entity positions and recorded post-step roles.

### 1. Why Seed 73 fails across all 5 variants on both `pv_10486` and `pv_10483`
- **Observed**: All 10 Seed 73 episodes time out at `2320` steps with `0/6` penned, `repeated_movement_fraction = 0.991..0.999`, and `1` unresolved stall (`stalls_count = 1..2`). In every run, **sheep `S2`** is the sole stray for all `500/500` final steps (`stray_counts={2: 500}`), pinned along the **bottom wall** (`y = 99..103`, grid height `104`):
  - `pv_10486` `baseline` / `collector_yield` / `corner_clearance`: `S2` at `x = 30, y = 101..103`, main group (`S0, S1, S3, S4, S5`) clustered at `x = 52..62, y = 52..56`. `D1` holds `rear_pressure` (`500/500` tail steps) at `(26..30, 75..79)`, while `D0` and `D2` swap `collector` and `left_flanker` (`289/211` in `baseline`, `286/214` in `corner_clearance`) at `(16..24, 93..97)`.
  - `pv_10483` `baseline` / `collector_yield` / `corner_clearance`: `S2` at `x = 9, y = 101..103`, main group at `x = 43..50, y = 55..60`. `D0` and `D2` oscillate at `(0..2, 91..98)` swapping `collector`/`left_flanker`; `D1` holds `rear_pressure` at `(2..6, 77..79)`.
  - `main_group` (`pv_10486` / `pv_10483`) and `all`: `S2` ends at `(0..1, 103)` (`pv_10486 main_group`), `(6, 102..103)` (`pv_10486 all` and `pv_10483 all`), or `(25, 99..102)` (`pv_10483 main_group`), with all three dogs locked in a single `2321`-step `oscillation_loop` (`0..2320`).
- **Observed mechanism gaps**:
  1. **`corner_clearance`**: Only shifts the `collector` target `1` cell inward (`+1` away from the bottom wall, e.g., `(28, 99) -> (28, 98)` on `pv_10486` and `(7, 98) -> (7, 97)` on `pv_10483`), and `_clear_of_stray` never alters `rear_pressure`/`left_flanker` targets because they are already `> 4` cells from `S2`. Because the collector dogs are oscillating `8..15` cells away from `S2`, a `1`-cell target shift does not break the policy's local action loop.
  2. **`collector_yield`**: Only deepens the stickiness penalty after `150` steps (`YIELD_FLOOR = -30.0`). On `baseline`, `D0`/`D2` already swap `collector`/`left_flanker` every ~200 steps, while `D1` remains ~20 cells closer to `rear_pressure` than `D0`/`D2`. Even when all 3 dogs churn roles in `all`, all 3 remain trapped near `x = 0..8, y = 75..99`.
  3. **`main_group`**: Moves `rear_pressure`/`left_flanker` targets onto the main group (`(39, 57)/(39, 65)` on `pv_10486`, `(49, 48)/(49, 56)` on `pv_10483`), removing split-centroid drag. However, the dogs remain stuck in local `2..4`-cell oscillation loops `20..40` cells away from those targets (`repeated_frac = 0.999`).
- **Inferred cause**: Once `S2` reaches the left/bottom wall (steps `40..100`) and the dogs settle into distant local oscillation loops, none of the three `TeamStrategy` target/role adjustments can force the `54`-input policy to traverse `10..30` cells of open field to approach `S2` or the main group.

### 2. Why `main_group` regressed `pv_10483` on seeds 41, 53, and 67
- **Seed 41 (`baseline` S@962 vs `main_group` F(no-progress, 3/6)@1955)**:
  - **Observed**: First target diff at step `87`; first action/state diff at step `92` (`stray=0` at `(26, 90)`). `main_group` shifts `D2` (`(30, 92) -> (31, 93)`) and `D1` (`(34, 99) -> (35, 99)`) by `(+1, +1)` / `(+1, 0)`. At step `92`, `D1` chooses `down` (`(34, 95)`) instead of `left` (`(33, 93)`). `baseline` pens `S3@173, S0@184, S1@198, S4@199, S2@200, S5@962`; `main_group` pens `S3@166, S4,S5@172`, then traps `S0` in the bottom-right corner `(142, 102..103)` with `S1,S2` at `(41, 91..94)` for steps `172..1955` (`repeated_frac = 0.758`, `1` unresolved stall).
- **Seed 53 (`baseline` S@235 vs `main_group` F(no-progress, 5/6)@1640)**:
  - **Observed**: First target diff at step `1`; first action/state diff at step `7`. At step `5` (`stray=3` at `(132, 93)`), `main_group` shifts `D1`'s target by `(-12, -12)` from `(67, 33)` to `(55, 21)` and `D0` from `(2, 101)` to `(2, 95)`. At step `7`, `D0` turns `left` instead of `down`. `main_group` later pens `5` sheep (`S2,S5@257, S0@266, S3@451, S1@913`) but stalls on `S4` at `(115, 88..97)` from step `917..1640` with `wrong_side_fence_overshoot` (all 3 dogs at `x = 105..114`, left of `S4`, while pen is on the right).
- **Seed 67 (`baseline` S@403 vs `main_group` F(timeout, 0/6)@2320)**:
  - **Observed**: First target diff at step `9`; first action/state diff at step `10`. At step `9` (`stray=2` at `(109, 29)`), `main_group` shifts `D2` by `(-11, +4)` from `(52, 44)` to `(41, 48)` and `D1` from `(21, 83)` to `(16, 81)`. At step `10`, `D2` moves `up` instead of `right`, disrupting the early gather; `S5` ends pinned at `(3, 102..103)` while `S0,S3` sit at `(141..142, 75..77)` and `S1,S2,S4` sit at `(83..86, 67..69)` (`repeated_frac = 0.922`).
- **Inferred cause**: In seeds `53` and `67`, `main_group` introduces large (`11..12` cell) early target shifts on steps `1..9` that push the `54`-input policy (trained on `1.0.0` all-sheep centroids) onto off-distribution trajectories; in seed `41`, even a `1`-cell target shift at step `92` flips a boundary action and leads to a corner trap.

### 3. How `corner_clearance` solved seeds 59 and 71 on `pv_10486` (and `pv_10483` parity)
- **`pv_10486` Seed 59 (`baseline` F(timeout, 0/6)@2320 -> `corner_clearance` S@382)**:
  - **Observed**: First target diff at step `55`; first action/state diff at step `61`. Stray `S5` is along the top wall at `(18..20, 3)`. `corner_clearance` shifts `D0`'s `collector` target `1` cell deeper into the open field (`(17, 6) -> (17, 7)` at step `61`), causing `D0` to move `down` (`(6, 4) -> (6, 6)`) instead of `right` (`(8, 4)`). This guides `S5` away from the top wall (`(23, 5)` at step `66` instead of `(22, 2)`), avoiding `baseline`'s `1934`-step bottom-left corner trap `(7, 101..102)`. All 6 sheep pen by step `382` (`S5@382`; `repeated_frac` drops from `0.897` to `0.402`).
- **`pv_10486` Seed 71 (`baseline` F(timeout, 3/6)@2320 -> `corner_clearance` S@347)**:
  - **Observed**: First target diff at step `20`; first action/state diff at step `26`. Stray `S2` is near the bottom-left wall at `(17, 101..103)`. `corner_clearance` shifts `D0`'s `collector` target `1` cell higher (`(15, 99) -> (15, 98)` at step `26`), causing `D0` to move `left` (`(12, 95) -> (10, 95)`) instead of `down` (`(12, 97)`). `S2` is kept clear of the `(0, 103)` corner trap (`baseline` stalls all 3 dogs at `(0..2, 102..103)` executing `wait` from step `560..2320`), penned at step `326`, and all 6 pen by step `347` (`repeated_frac` drops from `0.882` to `0.336`).
- **Why `pv_10483` stayed `8/10` with zero regressions**:
  - **Observed**: `corner_clearance` only alters targets when `wall_margin(stray) <= 3` (typically a `1`-cell `collector` offset), leaving open-field targets unchanged. On `pv_10483` Seed `59` (first diff step `76`, action diff `77`), `D0` is already pinned at `(0..2, 2..6)` and never approaches `S5` at `(10, 0..2)`, so Seed `59` still times out `0/6`. On Seed `71`, it slows completion (`567 -> 878` steps) but still solves `6/6`.
- **Latent limitation**: `corner_clearance` only helps when a collector dog is actively interacting with a wall/corner stray; it does **not** break distant oscillation loops (Seed `73` on both models, Seed `59` on `pv_10483`).

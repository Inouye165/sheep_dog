# Current-parent gate-information comparison results

The bounded comparison completed without interruption. Gate information is a promising candidate, but no model was promoted and the stage is not solved. The original 8/10 parent remains preserved.

## Identity and protocol

- Parent: Stage 34 checkpoint 40897 / policy version 10483 / timestep 343787786; SHA-256 `26aecd34c7d1723feb4ae2f44a03fbddc16c494b2855a6c26216b31703c54dd2`. The immutable preparation was copied into the run workspace; active app state and models were not mutated.
- Run manifest: `artifacts/experiments/gate_current_run_20261113_174700/manifest.json`; prepared input: `artifacts/experiments/gate_current_prep_20261113_173000/manifest.json`. The directory identifier predates actual start at 17:48:26 on 2026-11-13 (America/Los_Angeles). Source revision, tracked dirty patch, untracked hashes, dependency versions, full configs, models, replays, and 129 artifact checksums are preserved in the run.
- Hypothesis: explicit relative gate location and outward normal improve recovery versus four zero inputs, with all other settings equal. Both explicit modes have 58 inputs, preserved original parameters, zero new input columns initially, and identical fresh Adam resets. The parent had 54 inputs. This is approximate continuation, not exact resumption.
- Policy remains guided/assisted, with exactly three dogs and six sheep. Existing role guidance and physics assistance remain. No scripted recovery, new tactical mask, reward, physics, or termination change was introduced.
- Parent-effective settings were retained: hidden layers 256/256/128, gamma 0.997 on the sequential-dog transition timebase, GAE 0.98, entropy 0.015, learning rate 0.0001, rollout 2048, batch 1024, 10 optimizer epochs. Both arms used 25% failure-start probability and the same fixed equal class weights (gate-wall, high-pen, wall-stray). Adaptation, promotion and rollback were disabled.
- Fresh ordinary-start baseline: 8/10, reproducing saved parent outcomes and step counts. All six arms used paired continuation seeds 101, 202, 303 and the same ten development seeds (11, 23, 37, 41, 53, 59, 61, 67, 71, 73). Held-out cases were unused. Every arm evaluation episode audited its real environment as 58 inputs.
- All six training arms completed exactly 12,288 policy transitions / 4,096 measured world steps each: 73,728 transitions / 24,576 training world steps total. There were 70 full evaluation episodes / 72,549 evaluation world steps and two additional one-world-step wiring preflights. Main run elapsed 294.406 seconds; offline analysis added 24.297 seconds without extra policy episodes. Limits were 120 seconds per training arm, 900 seconds overall, and at most 70 full evaluations. No limit was hit.

## Completion and failures

| Continuation seed | Zero-gate control | Gate-information treatment | Paired success difference |
|---|---:|---:|---:|
| 101 | 6/10 | 7/10 | +1 |
| 202 | 7/10 | 7/10 | +0 |
| 303 | 4/10 | 8/10 | +4 |
| Total | 17/30 (56.7%) | 22/30 (73.3%) | +5 |

Control had 12 timeouts and one no-progress stop; treatment had seven timeouts and one no-progress stop. Success-only mean completion steps were 387.24 versus 500.00; those conditional populations differ and this does not establish a speed effect. Descriptive pooled Wilson 95% intervals are [39.2%, 72.6%] and [55.6%, 85.8%]. Shared parent and repeated development cases are correlated; these are not a paired significance test or held-out release evidence.

Each cell below is success/timeout/no-progress, final penned count, and actual world steps. Partial penning is never counted as success.

| Development seed | Parent | C101 | G101 | C202 | G202 | C303 | G303 |
|---|---|---|---|---|---|---|---|
| 11 | success, 6/6, 165 | success, 6/6, 243 | success, 6/6, 399 | success, 6/6, 791 | success, 6/6, 280 | success, 6/6, 301 | success, 6/6, 317 |
| 23 | success, 6/6, 204 | success, 6/6, 185 | success, 6/6, 194 | success, 6/6, 292 | success, 6/6, 237 | success, 6/6, 264 | success, 6/6, 147 |
| 37 | success, 6/6, 1071 | timeout, 0/6, 2320 | success, 6/6, 295 | success, 6/6, 310 | success, 6/6, 594 | success, 6/6, 345 | success, 6/6, 266 |
| 41 | success, 6/6, 962 | success, 6/6, 270 | success, 6/6, 395 | success, 6/6, 123 | success, 6/6, 131 | no-progress, 5/6, 969 | success, 6/6, 1002 |
| 53 | success, 6/6, 235 | success, 6/6, 248 | success, 6/6, 415 | success, 6/6, 239 | no-progress, 5/6, 1086 | timeout, 0/6, 2320 | success, 6/6, 269 |
| 59 | timeout, 0/6, 2320 | timeout, 0/6, 2320 | timeout, 0/6, 2320 | timeout, 0/6, 2320 | timeout, 0/6, 2320 | timeout, 0/6, 2320 | timeout, 0/6, 2320 |
| 61 | success, 6/6, 584 | success, 6/6, 1013 | success, 6/6, 371 | success, 6/6, 198 | success, 6/6, 342 | success, 6/6, 452 | success, 6/6, 426 |
| 67 | success, 6/6, 403 | timeout, 0/6, 2320 | timeout, 2/6, 2320 | timeout, 1/6, 2320 | timeout, 2/6, 2320 | timeout, 0/6, 2320 | success, 6/6, 714 |
| 71 | success, 6/6, 567 | success, 6/6, 655 | success, 6/6, 849 | success, 6/6, 654 | success, 6/6, 1817 | timeout, 0/6, 2320 | success, 6/6, 962 |
| 73 | timeout, 0/6, 2320 | timeout, 0/6, 2320 | timeout, 0/6, 2320 | timeout, 0/6, 2320 | success, 6/6, 578 | timeout, 0/6, 2320 | timeout, 0/6, 2320 |

## Actual practice and recovery evidence

Counts below are episode starts, including the final unfinished start; they are not completed episode counts. The probability was fixed at 25%, but observed exposure differed because this is a short run.

| Arm / seed | Normal | Gate-wall | High-pen | Wall-stray | Total starts |
|---|---:|---:|---:|---:|---:|
| C101 | 6 | 2 | 0 | 2 | 10 |
| G101 | 6 | 2 | 0 | 1 | 9 |
| C202 | 14 | 1 | 3 | 1 | 19 |
| G202 | 12 | 1 | 3 | 0 | 16 |
| C303 | 13 | 2 | 1 | 0 | 16 |
| G303 | 14 | 2 | 1 | 0 | 17 |

There were no high-pen starts in the 101 pair, and no wall-stray starts in either 303 arm or the 202 treatment. This limited exposure and scenario fidelity prevent treating a short negative case as proof that gate information cannot help.

Offline repetition/recovery analysis covered all 14 parent/control/treatment replays for the two pretraining failures, seeds 59 and 73. Results and replay hashes are in `offline_recovery_metrics.json`. Seed 59 had 2,310–2,320 repeated-movement steps out of 2,320 actual episode steps, zero recovered stalls, and an unresolved stall in every model. Treatment 202 solved seed 73 in 578 steps with 220 repeated-movement steps, two recovered stalls and zero unresolved stalls; its paired control timed out after 2,320 steps with 2,262 repeated steps and one unresolved stall. Analyzer totals include step zero, so the separately recorded actual episode-world-step counts are the proper episode budget denominator.

Parent seed 59 ends with five sheep near (46..50, 46..48), straggler (10, 2), dogs (0, 2), (10, 31), (2, 22), and pen origin (128, 3). Parent seed 73 ends with five sheep near (46..50, 56..58), straggler S2 at (9, 102), dogs (0, 96), (6, 77), (0, 91), and a left-opening pen at (128, 73). Both report zero gate-corridor-failure and controlled-stall steps despite long repeated movement. These observed geometries show failure before useful gate approach; wall/straggler recovery and team positioning are candidate explanations, not established causes.

## Decision and verification

The manager retained gate information as promising (+1, 0, +4 paired successes), without replacing the original parent or promoting a selected model. Treatment 202 provides an inspectable success on previously persistent seed 73, but regressed seed 53 to 5/6 and no-progress; all seven evaluated models still failed seed 59 with zero penned. Treatment pooled 22/30 does not establish improvement over the preserved parent baseline of 8/10. The short budget, optimizer reset, limited class exposure, and correlated development cases remain material limits. Next: bounded investigation of seed 59 wall/split-flock recovery and continuation stability; no further hours-long run is justified by this pilot alone.

A real evaluation wiring defect was fixed before this run: SheepdogEnvironment constructed the 54-input guided builder for gate modes. It now uses the same existing explicit observation-builder factory as the adapter. Existing guided/emergent feature outputs remain unchanged, invalid modes are rejected, and all new 58-input modes use their declared schema. Real Evaluator + real model tests cover guided54/control58/treatment58 and reject mismatches before action selection. After the isolated run, the manager authorized a graceful backend-only restart to adopt this constructor fix. Health/readiness passed with training stopped, Stage 34 pv10483 / checkpoint 40897 unchanged, and the viewer preserved. The original model hash stayed unchanged; verification is recorded in `artifacts/experiments/stopped_adoption_20261113_171410/factory_adoption_20261113_175450/verification.json`. No further restart or training resumption occurred.

Commands:
```powershell
.venv/Scripts/python.exe -m pytest tests/test_gate_current_comparison.py tests/test_gate_information_preparation.py tests/test_gate_information_experiment.py -q
.venv/Scripts/python.exe -m ruff check scripts/run_gate_current_comparison.py tests/test_gate_current_comparison.py src/sheepdog/environment.py
.venv/Scripts/python.exe -m scripts.run_gate_current_comparison --output-dir artifacts/experiments/gate_current_run_20261113_174700
```

Checks: 23 tests passed, one historical-parent test skipped; Ruff and diff whitespace checks passed. The manager independently verified all six budgets, 70 evaluations, all 60 arm input audits, the immutable parent, and all 129 run checksums including offline analysis. No viewer source changed, so viewer tests/build were not run. No held-out evaluation, live default change, original-model overwrite, or long-job resumption occurred.

# Recovery practice sampling pilot

This diagnostic compares 25% versus 60% recovery-start probability from the
same Stage 34 parent checkpoint, policy version 10481 / timestep 343721120.
The historical development evaluation scored 6/10 full-flock successes. A fresh
parent evaluation under the current source is required before any training.

Hypothesis: more frequent varied starts from wall-stray extraction, high-pen
standoff, and gate-wall flock classes improve full-flock completion while
retaining ordinary-start performance. Fixed equal weights for these three
classes are applied to both arms. This is a fixed development-directed pilot,
not adaptive closed-loop training. Ordinary evaluation starts use the original
environment reset; no failure scenario is injected into evaluation.

The policy remains **guided/assisted**, with three dogs and six sheep and the
checkpoint's environment and reward configuration. Existing physics assistance
is retained. Observations, actions, physics, reward, and termination are not
changed. Both arms independently restore the same parent's optimizer state.
Resume is approximate: fresh training RNG and environment state are intentional
and identical for each paired training seed; no claim of exact resumption.

Paired training seeds are 101, 202, and 303. Each of six runs may execute at most
12,288 sequential policy transitions, six 2,048-transition PPO rollouts, roughly
4,096 world steps. The runner counts actual completed world steps separately.
This is a small diagnostic exposure, not sufficient evidence to reject a
longer-term sampling benefit if results are inconclusive. Maximum training is
73,728 transitions, maximum evaluation 70 episodes, maximum training wall time
120 seconds per arm, and overall wall time 900 seconds. Limits are checked at
simulation steps; a single model update or file operation may finish after its
deadline. Partial runs remain explicitly incomplete and are not comparable as
equal-budget results.

Ten development seeds (11, 23, 37, 41, 53, 59, 61, 67, 71, 73) are reused for
parent and each arm. Held-out cases remain unused. Report full-flock completion
with denominator, timeout and no-progress rates, successful completion steps,
and existing gate-corridor/controlled-stall metrics. Wilson intervals describe
episode samples only: common evaluation seeds and a common parent induce
dependence. Preserve per-seed failures and successes through complete ordinary
replays; optional detailed decision telemetry is disabled to limit compute.
No pilot result automatically promotes or replaces an active checkpoint.

The runner copies the parent before loading, records source revision, dirty
patch and untracked hashes, dependency versions, full arm configs, checkpoint
identity and checksums in a new experiment directory. It does not touch the
active training settings or checkpoint. A separate Python process isolates
global capture settings and limits PyTorch to one CPU thread.

```powershell
# Fresh parent baseline only; choose a new directory each invocation.
.venv/Scripts/python.exe scripts/run_recovery_mix_pilot.py --output-dir artifacts/experiments/recovery_mix_pilot_<timestamp>

# Parent baseline followed by six bounded training/evaluation runs.
.venv/Scripts/python.exe scripts/run_recovery_mix_pilot.py --output-dir artifacts/experiments/recovery_mix_pilot_<timestamp> --train
```

Execution requires the repaired sampler/worker telemetry and legal training
scenario starts. Unit checks do not establish that learning improved. Actual
execution results and any remaining limits belong in the experiment manifest.

## Executed development diagnostic — 2026-11-13

Command executed:

```powershell
.venv/Scripts/python.exe scripts/run_recovery_mix_pilot.py --output-dir artifacts/experiments/recovery_mix_pilot_20261113_164500 --train
```

The directory suffix is an identifier; baseline execution began around 16:28
America/Los_Angeles. The run completed in 432.157 seconds. All six arms executed
exactly 12,288 transitions and 4,096 measured world steps: 73,728 training
transitions and 24,576 training world steps total. The parent plus six arms
completed all 70 evaluation episodes, totaling 59,397 evaluation world steps.
No deadline interruption occurred. These are three paired continuation seeds
from one shared parent, not independently trained-from-scratch models.

Parent SHA-256:
`1728b24e1663207bbb55b7542d5eb50b6cbd0b4c5b40b89fa91a2e5b2a607801`.
The original parent remained unchanged after execution. Full configuration,
dependency versions, revision and dirty patch identity, artifact checksums,
per-episode results, and all success/failure replays are in
`artifacts/experiments/recovery_mix_pilot_20261113_164500/manifest.json` and its
referenced files. Model optimization retained the serialized parent's
gamma .997, GAE .98, entropy .015, learning rate .000065, rollout size 2048,
batch size 1024, and 10 epochs; both arms restored the same optimizer state.
The live training process and active configuration were left intact. The source
sampling/legality fixes were used in these isolated runs.

Full-flock success means all six sheep inside the pen under the existing
completion rule by 2,320 world steps. Timeout and no-progress rates use all ten
development episodes as denominator. Successful completion time is averaged
only over that arm's successful episodes; it is not the evaluator's general
average episode length. Every failure below was a timeout; no-progress-stop
rate was zero in every evaluated group.

| Continuation seed | Configured ratio | Full-flock success | Timeout | Mean steps among successes | Mean gate-corridor failure steps/episode | Mean controlled-stall steps/episode |
| --- | --- | --- | --- | --- | --- | --- |
| Parent | — | 6/10 | 4/10 | 390.17 | 123.6 | 49.7 |
| 101 | 25% | 7/10 | 3/10 | 299.43 | 2.2 | 17.6 |
| 101 | 60% | 8/10 | 2/10 | 425.25 | 5.3 | 26.4 |
| 202 | 25% | 7/10 | 3/10 | 345.14 | 56.9 | 18.7 |
| 202 | 60% | 9/10 | 1/10 | 398.11 | 4.9 | 31.9 |
| 303 | 25% | 9/10 | 1/10 | 367.56 | 7.4 | 13.2 |
| 303 | 60% | 8/10 | 2/10 | 641.38 | 7.8 | 37.0 |

Gate-corridor failure and controlled-stall durations retain the environment's
existing diagnostic counter definitions. They are descriptive episode counters,
not a claim that gate geometry causally produced the failure. The pilot did not
separately validate recovery onset/time, oscillation duration, or every recovery
scenario class, so these metrics cannot satisfy release coverage requirements.

The configured ratio is a per-start probability. Actual counts include the
last episode start, which can still be unfinished when training stops:

| Seed | Ratio | Normal starts | Wall-stray | High-pen | Gate-wall | Targeted starts / all starts |
| --- | --- | --- | --- | --- | --- | --- |
| 101 | 25% | 12 | 2 | 1 | 3 | 6/18 (33.3%) |
| 101 | 60% | 7 | 7 | 8 | 6 | 21/28 (75.0%) |
| 202 | 25% | 13 | 0 | 3 | 1 | 4/17 (23.5%) |
| 202 | 60% | 11 | 4 | 3 | 3 | 10/21 (47.6%) |
| 303 | 25% | 12 | 0 | 1 | 2 | 3/15 (20.0%) |
| 303 | 60% | 7 | 1 | 4 | 3 | 8/15 (53.3%) |

For exact paired inspection, cells contain successful completion world steps,
or `F<n>` for timeout with `<n>` of six sheep penned:

| Development seed | Parent | 101 / 25% | 101 / 60% | 202 / 25% | 202 / 60% | 303 / 25% | 303 / 60% |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 11 | 213 | 158 | 175 | 534 | 366 | 190 | 243 |
| 23 | 475 | 309 | 310 | 176 | 222 | 233 | 261 |
| 37 | 348 | 235 | 280 | 287 | 305 | 249 | 228 |
| 41 | 449 | 534 | 817 | F4 | 567 | 216 | 152 |
| 53 | F0 | 307 | 536 | 417 | 387 | 331 | 1681 |
| 59 | F5 | F0 | F0 | F0 | 554 | 380 | 1641 |
| 61 | 627 | 180 | 557 | 425 | 478 | 451 | 681 |
| 67 | F0 | F1 | 522 | 206 | 261 | 671 | F0 |
| 71 | 229 | 373 | 205 | 371 | 443 | 587 | 244 |
| 73 | F0 | F0 | F0 | F0 | F0 | F0 | F0 |

Paired discordances: 60% gained seed 67 for continuation seed 101, gained seeds
41 and 59 for continuation seed 202, and lost seed 67 for continuation seed 303.
Paired success-count differences were +1, +2, and −1 out of ten.

Pooled development completion was 23/30 (76.7%) for 25% and 25/30 (83.3%) for
60%. Descriptive Wilson 95% intervals are [59.1%, 88.2%] and [66.4%, 92.7%].
These overlapping intervals are not a paired significance test; repeated
development cases and the shared parent invalidate an interpretation as 30
independent held-out trials. Successful-episode mean time pooled across arms
was 340.00 versus 484.64 world steps. Those conditional populations differ, so
the difference is not a causal estimate of slowing. In continuation seed 303,
however, both arms solved development seeds 53 and 59 and the 60% arm took
1,681/1,641 steps versus 331/380 in its paired control.

Decision: retain the 25% default pending stronger evidence; do not automatically
adopt the 60% profile or promote any selected pilot model. Seed 73 remained at
zero sheep penned in all seven evaluations, while one 60% continuation regressed
seed 67. Prioritize a controlled gate-information experiment and an audit of
failure classification/scenario fidelity before a longer sampling comparison.
The stage is not solved, the ratio benefit is not proven, and held-out cases
remain untouched.

Verification: `.venv/Scripts/python.exe -m pytest tests/test_recovery_mix_pilot.py -q`
passed 7 tests; `.venv/Scripts/python.exe -m ruff check scripts/run_recovery_mix_pilot.py tests/test_recovery_mix_pilot.py`
passed. The manager separately verified 26 focused sampling/integration tests
and changed-file Ruff. Viewer checks were not run because no viewer code changed.

Manager independent verification confirmed complete six-arm/70-episode budget accounting, all 113 artifact checksums, and unchanged source-parent hash.

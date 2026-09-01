# RL Evaluation and Experiment Roadmap

This document records changes considered for the MaskablePPO training path, the
evidence collected, decisions made, and the next experiments in priority order.
The goal is to change production training only when a controlled comparison
shows a reliable behavioral improvement.

## Current Production Decision

Keep PPO training at `batch_size=1024`.

Keep adaptive-state restoration. It is required for a resumed run to honor its
saved controller state, but the completed experiment did not show that stage-2
restoration improves average policy success. Treat the stage-2 values as a
retuning target, not as a proven performance improvement.

Do not promote any model produced by the batch-size experiments. They use
isolated artifact directories and exist only to compare training behavior.

The `batch_size=512` candidate improved the pooled result, but it did not improve
two of the three independently trained models. That seed sensitivity is too high
to justify changing production training.

## Model and Training Configuration Reviewed

- Algorithm: MaskablePPO with invalid-action masking
- Network: shared MLP with hidden layers `[128, 128, 128]`
- Observations: 54
- Actions: 9
- Learning rate: `1e-4`
- Final learning rate: `3e-5`
- Rollout steps per environment: `2048`
- Production batch size: `1024`
- Discount factor: `0.99`
- GAE lambda: `0.98`
- PPO clip range: `0.2`
- Entropy coefficient: `0.01`
- Value coefficient: `0.5`
- Target KL: `0.03`
- Current Windows runtime: one environment worker

With one worker, each rollout contains 2,048 samples. Batch size 1,024 therefore
creates two minibatches per PPO epoch, while batch size 512 creates four. The
experiment tested whether the additional optimizer updates improve learning, not
just whether they make training faster.

## Current Assessment: Can This Setup Learn Well?

Yes, the setup is capable of learning. MaskablePPO, action masking, the network
size, observation space, and the individual PPO hyperparameters are all
reasonable for this task. The trained policies reached 48% to 75% success across
the three batch-1,024 control seeds, and one experimental model reached 90%.
There is no current evidence that the network is too small or that PPO is the
wrong algorithm.

It is not yet learning as reliably or efficiently as it should. The strongest
structural concern is the Gym transition model. With three dogs, PPO records
three environment transitions for one world transition. The first two dog turns
return zero reward and do not advance the world; only the third advances all
dogs and sheep and returns the team reward. The pending actions selected for the
earlier dogs are not present in later dogs' observations, even though they help
determine the eventual transition and reward. PPO also applies discounting and
GAE per dog turn rather than per world step. The class named `JointActionRLEnv`
uses the same collect-three-actions pattern from SB3's perspective and does not
remove this issue.

Training is also highly seed-sensitive: the adaptive-resume treatment changed
success by -7, -26, and +22 points across seeds. The configured eight-worker
sampling design does not run on Windows by default; the runtime silently uses
one in-process environment. That mismatch may add rollout correlation, but it
ranks behind correcting what one PPO transition means.

The learning-rate configuration is also ambiguous. Adaptive learning replaces
the configured linear schedule with a constant stage value, so
`learning_rate_final=3e-5` does not control an adaptive run. That is configuration
debt, but it should be clarified only after measuring the stage-2 rate directly.

A post-hoc diagnostic across 600 final evaluation episodes found a second
target worth testing. The cumulative `stray_ignore_penalty` averaged -442.6 on
failures and -12.5 on successes; 233 episodes accumulated less than -100 from
that component. Success was 81.3% on `fixed_easy`, 54.6% on
`randomized_flock`, and 40.6% on `nearby_stray`. These are descriptive
associations, not proof that the penalty causes failures, so reward changes rank
behind the cleaner optimizer experiments.

PPO diagnostics do not show a consistently broken value function. Across the
completed adaptive runs, mean explained variance was generally about 0.82 to
0.90. Approximate KL was usually around 0.003 and clip fraction around 1% to 2%,
well below the configured `target_kl=0.03`. This makes a controlled increase in
optimization passes more defensible than changing several reward weights, while
still not guaranteeing that more epochs will help.

## Batch-Size Evaluation

### Invalid First Run

The first 1,024-versus-512 experiment was invalid. Stable-Baselines3 restored
`batch_size=1024` from the checkpoint ZIP after each candidate configuration was
created, so both labeled arms actually trained with batch size 1,024.

The identical outcomes across all paired arms exposed the problem. The reported
verdict from `artifacts/experiments/batch_ab_20260827_140814` must not be used.

The loader was corrected to reapply the configured batch size after loading a
checkpoint. The experiment runner now records and verifies the effective runtime
batch size and fails immediately if it differs from the requested value.

### Corrected Experiment Design

- One frozen baseline checkpoint for every arm
- Three independent training seeds: 7, 17, and 29
- Two arms per seed: batch sizes 1,024 and 512
- Equal budget: 512,000 timesteps per arm
- Three rollout-aligned training checkpoints per arm
- 100 identical deterministic evaluation seeds per arm
- Separate models, state, checkpoints, and evaluations for every arm
- Effective batch size verified after model loading

The generated report is available locally at
`artifacts/experiments/batch_ab_corrected/comparison.md`.

### Corrected Results

| Training seed | Success at 1,024 | Success at 512 | Difference | Timeout at 1,024 | Timeout at 512 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 7 | 75% | 71% | -4 points | 16% | 19% |
| 17 | 66% | 50% | -16 points | 26% | 29% |
| 29 | 48% | 90% | +42 points | 37% | 9% |
| **Mean** | **63.0%** | **70.3%** | **+7.3 points** | **26.3%** | **19.0%** |

Across the 300 paired evaluation episodes, the 512 candidate had 68 exclusive
successes and the 1,024 control had 46. The exact paired p-value was `0.048725`.
Median successful completion was four steps slower with 512.

### Interpretation and Decision

The pooled result makes batch size 512 look better, but the independent training
seed is the important unit for judging training reliability. Batch size 512 made
two trained models worse and one dramatically better. Three training seeds are
not enough to conclude that the average improvement will generalize.

The candidate also fell to 30% success at an intermediate 10-seed checkpoint,
crossing the experiment's conservative collapse guardrail before recovering.
This checkpoint alone is not proof that 512 is harmful, but it reinforces the
observed instability.

Batch size 512 produced larger, still-safe PPO updates. It can help the optimizer
escape a weak trajectory, but the current evidence says it does not improve
learning reliably. Production remains at 1,024.

## Adaptive-Resume Evaluation

**Status:** Implemented, regression-tested, and behaviorally evaluated

The frozen baseline had reached adaptive stage 2, using learning rate `8e-5` and
entropy coefficient `0.008`. Resumed training previously constructed a new
controller at stage 1, returning to learning rate `1e-4` and entropy `0.01`.

This is a correctness defect, not a speculative hyperparameter preference.
The trainer now persists and loads the adaptive stage, EMA success rate, and
consecutive-hit state. It restores them when the curriculum stage has not
changed and preserves the intentional stage-1 reset after a curriculum change.

Controller and real trainer resume tests pass. The trainer-level test observes
the settings at the PPO `learn()` boundary and confirms stage 2 applies learning
rate `8e-5`, optimizer learning rate `8e-5`, and entropy `0.008`. The combined
focused suite passes all 28 tests.

A controlled three-training-seed comparison then measured 512,000 timesteps per
arm with 100 paired evaluation seeds. Restoring stage 2 reduced mean success
from 63.0% to 59.3%, while reducing timeout rate from 26.3% to 24.0% and
eliminating the observed checkpoint collapse (one legacy collapse versus zero
restored collapses). Per-seed success differences were -7, -26, and +22 points;
the paired success result was not statistically significant (`p=0.347`).

**Decision:** retain adaptive-state persistence because it makes resume behavior
correct and reduced observed instability, but do not claim a learning-quality
improvement. The current stage-2 damping is a retuning target: test a milder
multiplier or a short resume warm-up before promoting it as the best-performing
training behavior. Full results are in
`artifacts/experiments/adaptive_resume_ab_20260827/comparison.md`.

## Three Next Experiment Choices

Run these in the listed order. Stop after each experiment, make its production
decision, and use that decision as the fixed baseline for the next experiment.

### Choice 1: Use One True Joint Team Transition

**Why first:** this corrects the learning problem's representation instead of
tuning around it. In the current three-dog adapter, two thirds of PPO transitions
have zero reward and no world movement. Later dog decisions cannot observe the
pending actions on which the next world state depends, and a 2,048-step rollout
contains only about 682 world transitions. This is a larger and more direct
learning constraint than worker count, network width, or a scalar reward weight.

**Control and treatment:** the control remains the current `Discrete(9)` dog-turn
adapter. The treatment exposes one team step as one Gym step, with a factorized
`MultiDiscrete([9, 9, 9])` action, masks for each dog, one centralized team
observation, and one immediate team reward. All three dog actions are chosen from
the same pre-movement state and the world advances exactly once. Do not add
shepherd commands, new rewards, recurrence, or attention in this experiment.

This is an architecture comparison, so the observation and action heads are
necessarily different and existing checkpoints are incompatible. Compare fresh
models initialized from the same training seeds. Match hidden-layer capacity as
closely as practical and report parameter counts. Measure budget in world
transitions and dog actions, not raw SB3 timesteps, because the control currently
counts three timesteps per world transition.

**Fixed variables:** environment dynamics, role assignment, reward function,
curriculum sequence, learning rate, entropy, PPO epochs, batch size per collected
world sample, training layouts, final evaluation seeds, and checkpoint selection
rules.

**Budget:** begin with a stage-8 learning smoke comparison to verify masks,
transition counts, and reward timing. Then run both formulations through the same
curriculum with seeds 7, 17, and 29 and equal world-transition budgets. Evaluate
the stage-8 models on the same untouched 100-seed final bank.

**Primary decision metrics:** per-training-seed stage-8 success, sample efficiency
as success versus world transitions, timeout rate, collapse count, nearby-stray
success, KL, clip fraction, and wall-clock cost.

**Accept:** the joint formulation improves at least two of three trained models,
mean success by at least 7 points, and success at matched world-transition
milestones, while timeouts and collapse count do not worsen. Otherwise retain the
current adapter and test the narrower fallback of adding pending-action features
and team-step-corrected discounting.

### Choice 2: Increase PPO Epochs From 10 to 15

**Why second:** the completed runs generally make small policy updates. Mean
approximate KL is usually near 0.003 against a 0.03 stopping target, and only
about 1% to 2% of samples are clipped. The current code relies on MaskablePPO's
implicit default of 10 epochs. Fifteen epochs is a controlled way to extract more
learning from each rollout without simultaneously reducing batch size or raising
the learning rate.

**Control and treatment:** explicitly configure `n_epochs=10` for the control and
`n_epochs=15` for the treatment. Keep target KL enabled so unexpectedly large
updates stop early. Do not change batch size, learning rate, entropy, worker
count, or rollout size in this experiment.

**Fixed variables:** the accepted formulation from choice 1, identical baseline
weights and controller state for each matched arm, rewards, curriculum, training
and evaluation seeds, collected world transitions, and checkpoint cadence.

**Budget:** seeds 7, 17, and 29 in both arms for the equivalent of 512,000 current
dog-action timesteps, followed by the same 100 paired deterministic evaluation
seeds per model.

**Primary decision metrics:** per-training-seed success, timeout, minimum
checkpoint success, collapse count, approximate KL, clip fraction, entropy,
explained variance, and the expected increase in wall-clock training time.

**Accept:** 15 epochs improves at least two of three trained models and mean
success by at least 5 points, with timeout within 3 points of control, no added
collapse, median KL below 0.015, and clip fraction below 15%. Otherwise keep 10;
low KL alone is not a reason to promote more optimization.

### Choice 3: Enable Failure-Directed Scenario Sampling

**Why third:** both scenario training and failure-directed training are disabled,
while held-out success is 81.3% on `fixed_easy`, 54.6% on `randomized_flock`,
and 40.6% on `nearby_stray`. The repository already has a feedback loop that
classifies evaluation failures and directs 25% of subsequent episodes toward
procedural hard cases, with decay to reduce forgetting. This targets demonstrated
weaknesses without changing what behavior the reward defines.

**Control and treatment:** compare `failure_directed_training_enabled=false`
against `true` with the existing target ratio `0.25`, decay `0.60`, and minimum
weight `0.05`. Do not also enable the static scenario mix or alter spawn weights.

Use a dedicated internal validation seed set to update failure weights. Those
seeds become training feedback and must not be reused for final claims. Keep the
100-seed final evaluation bank completely untouched until training ends.

**Fixed variables:** accepted choices 1 and 2, baseline weights and state,
environment spawn mix, reward function, curriculum, PPO settings, total world
transitions, checkpoint cadence, training seeds, and final evaluation seeds.

**Budget:** seeds 7, 17, and 29 in both arms for the equivalent of 512,000 current
dog-action timesteps, with matched internal validation budgets and a final
100-seed evaluation. Report scenario exposure as well as outcomes.

**Primary decision metrics:** per-training-seed overall success, nearby-stray and
randomized-flock success, timeout rate, sheep penned on failure, final farthest
distance, fixed-easy retention, and actual targeted-scenario percentage.

**Accept:** the treatment improves at least two of three trained models, mean
overall success by at least 5 points, and nearby-stray success by at least 8
points, without reducing fixed-easy success by more than 3 points or increasing
timeouts by more than 3 points. Otherwise leave failure-directed sampling off.

## Broader Options Considered but Not Ranked in the Top Three

- **Eight parallel workers:** still worth a later throughput and rollout-diversity
   experiment, but it does not correct hidden pending actions or dog-turn reward
   timing. On Windows it also adds process-lifecycle risk.
- **Stray-penalty reduction or normalization:** the cumulative value is large on
   failures, but that is partly because failures last longer. Explained variance
   is generally strong, PPO normalizes advantages, and the current evidence is
   correlational. Keep collecting component diagnostics before changing it.
- **Stage-2 learning rate or entropy:** the adaptive-resume result was mixed and
   changed both values together. These are smaller tuning opportunities after the
   transition model and update strength are settled.
- **Larger MLP, LSTM, or attention:** one existing MLP reached 90% success and no
   capacity diagnostic currently identifies network size as the bottleneck.
   Recurrence may help partial observability, but a Markov team-step formulation
   should be tested first.
- **Behavior cloning from `heuristic_expert`:** rejected as a current top option.
   On a stratified 20-seed stage-8 diagnostic, the scripted expert had 0% success,
   85% timeout, and averaged 0.6 sheep penned. It is not a suitable teacher at the
   current stage.
- **Enabling all instinct rewards:** seven simultaneous reward components would
   make attribution poor and can change the task objective. Diagnose and test one
   component only if the three higher-confidence experiments do not help.

## Experiment Rules

1. Change one training factor at a time.
2. Freeze the same baseline weights and state for every arm.
3. Record requested and effective runtime settings.
4. Use equal timestep budgets and identical held-out evaluation seeds.
5. Use multiple independent training seeds.
6. Keep experiment artifacts separate from production artifacts.
7. Reject changes that improve the mean through one exceptional seed while most
   independently trained models regress.
8. Do not promote experimental models automatically.

## Decision Log

| Date | Change evaluated | Decision | Reason |
| --- | --- | --- | --- |
| 2026-08-27 | Batch size 1,024 versus 512 | Keep 1,024 | 512 improved the pooled mean but regressed two of three independently trained models. |
| 2026-08-27 | Reapply configured batch size after loading PPO ZIP | Keep fix | Required for configuration correctness; verified with unit, real-model, and smoke tests. |
| 2026-08-27 | Promote models from batch-size experiment | Reject | Experimental models are isolated evidence and are not approved production checkpoints. |

# Architecture

## Environment Loop

The environment is a deterministic seeded grid simulation on a denser logical field than the original release. A reset call seeds the random generator, places dogs and sheep into unique open cells, and defines the pen location. Each step applies dog actions, resolves occupancy conflicts deterministically, then moves sheep based on nearby dog pressure, flock cohesion, wall avoidance, and small seed-driven tie-break jitter.

Episodes end when all sheep are penned, the step limit is reached, or the no-progress guard trips.

## Entity Model

- Dogs are shared-policy agents.
- Sheep are reactive agents that flee pressure and try to remain near the flock.
- Dogs, sheep, and other dogs do not share a logical cell.
- The pen is a rectangular goal region.
- The field is configurable so the scenario can scale beyond a toy map.

## Movement Resolution

- Dogs move one logical cell per simulation step.
- Sheep move one logical cell per simulation step.
- The field resolution is increased instead of relying on multi-cell jumps.
- Occupancy conflicts are deterministic: lower-index agents keep priority for a contested target cell, and blocked agents stay put.
- Sheep randomness is intentionally small and only used to break otherwise equivalent local options.
- Deadlock handling watches for wall-pinned sheep, repeated two-position dog loops, and stalled progress, then increases the value of lateral/flank positioning relative to straight-line pressure.

## Reward Design

Reward is split into named components so it can be audited and tested:

- progress_to_pen
- sheep_penned
- flock_cohesion
- scatter_penalty
- time_penalty
- no_progress_penalty
- wall_pressure_penalty
- wait_penalty
- terminal_success
- terminal_failure

The total reward is the sum of these components.

## Policy Modes

- `random_untrained` is a pure legal-action baseline with no flock strategy.
- `instinct_only` uses sheep-local spacing, circling, and straggler recovery without reading the pen target.
- `heuristic_expert` is a pen-aware scripted controller that computes pressure positions behind the flock relative to the pen.
- `trained_policy` is the shared role-aware linear policy learned through hill climbing.
- `neural_policy` is the shared role-aware neural policy used by the MaskablePPO experiment.

The trainable baseline uses a simple hill-climbing loop. Hill climbing is the optimization algorithm, not the model. The model itself is a shared linear policy with explicit role-aware weights. This is not PPO and not a neural network.

The experimental path uses a small MLP policy trained with MaskablePPO. MaskablePPO is the optimization algorithm, not the model. The role assignment is still scripted so the linear and neural policies consume comparable role-aware observations.

Dynamic roles are assigned by a scripted team strategy each step. Once a role is assigned, the shared linear policy scores legal actions with both global herding weights and role-specific linear weight groups for rear pressure, flankers, collectors, and blockers.

The architecture is now split into modular layers so trainer and model swaps stay local:

- environment: deterministic sheepdog simulation and action legality
- observation builder: role-aware flat features for one dog
- policy model: linear baseline or neural experiment
- trainer: hill climbing or MaskablePPO
- evaluation: fixed-seed checkpoint measurement
- benchmark harness: same-seed policy comparison across modes
- replay export: JSON artifacts consumed by the web viewer

The PPO adapter uses a pragmatic first implementation: one shared policy acts for one dog at a time through a sequential gym wrapper, and the environment only advances after every dog has supplied an action for the current team step. Invalid action masking is exposed from the underlying environment legality rules.

Reward shaping may still include target-progress terms during training. That signal is deliberately separated from no-training action selection so the default dog team does not inherit an expert pen controller for free.

## Checkpoint and Evaluation Flow

Training runs the configured trainer across scheduled checkpoints and evaluates each checkpoint on fixed comparison seeds. Hill-climb candidate mutations can be scored on a separate fixed candidate seed set so policy promotion is less sensitive to a single rollout. Evaluation writes JSON summaries, CSV rows, and replay files for every seed. Training also writes a compact checkpoint index for the UI.

Checkpoint metadata is backward-compatible with older linear checkpoints and now also carries trainer type, policy type, and optional neural policy state paths so playback and benchmarking can load either baseline or experiment artifacts.

## UI State Flow

The React app does not simulate the environment itself. A Python API server manages training control and replay generation, and the viewer reads checkpoint and replay JSON files from `web/public/generated/`, then plays them back in the browser. The viewer scales icon sizes relative to logical grid density so a denser field does not make the dogs and sheep visually tiny, and it applies short transitions to make one-cell motion read as smoother movement.

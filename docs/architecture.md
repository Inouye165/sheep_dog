# Architecture

## Environment Loop

The environment is a deterministic grid simulation. A reset call seeds the random generator, places dogs and sheep, and defines the pen location. Each step applies dog actions, then moves sheep based on nearby dog pressure, flock cohesion, wall avoidance, and deterministic tie-breaking.

Episodes end when all sheep are penned, the step limit is reached, or the no-progress guard trips.

## Entity Model

- Dogs are shared-policy agents.
- Sheep are reactive agents that flee pressure and try to remain near the flock.
- The pen is a rectangular goal region.
- The field is configurable so the scenario can scale beyond a toy map.

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

- Random baseline
- Heuristic baseline
- Shared trainable linear policy

The trainable policy uses a simple hill-climbing loop. That keeps the first version honest without pretending a larger RL stack exists yet.

## Checkpoint and Evaluation Flow

Training runs the policy across scheduled checkpoints and evaluates each checkpoint on fixed seeds. Evaluation writes JSON summaries, CSV rows, and replay files for every seed. Training also writes a compact checkpoint index for the UI.

## UI State Flow

The React app does not simulate the environment itself. It reads checkpoint and replay JSON files from `web/public/generated/`, then plays them back in the browser. The controls manage playback state and data selection, not live simulation state.

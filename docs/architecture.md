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

---

## Hierarchical Shepherd + Neural Dog Architecture

Branch: `feat/hierarchical-shepherd-neural-training`

This layer sits **above** the original sequential PPO baseline and does not
replace it.  All existing policies, trainers, and checkpoints are preserved.

### Two-level decision hierarchy

```
Shepherd  (high-level: "gather", "drive_to_pen", "apply_pressure", …)
    │
    └─► Dogs  (low-level: choose one of 9 grid moves per step)
```

| Layer | What is scripted | What is learned |
|-------|-----------------|-----------------|
| Sheep movement | reactive pressure model (always scripted) | — |
| Shepherd commands | Phase A: `ScriptedShepherd` rule-set | Phase B (future): replace with a learned command policy |
| Dog movement | — | neural MLP via MaskablePPO |

Dogs are **not** given scripted movement targets.  They receive only the
shepherd's current command as an additional observation feature, and must
discover good positions through reward.

### New modules

| Module | Purpose |
|--------|---------|
| `src/sheepdog/shepherd.py` | `ShepherdCommand` enum (8 values), `COMMAND_ORDER`, `ScriptedShepherd` |
| `src/sheepdog/observations.py` | `HierarchicalObservationBuilder` — base features + 8 cmd one-hot + 7 identity features |
| `src/sheepdog/rewards.py` | `HierarchicalRewardConfig`, `HierarchicalRewardComputer` |
| `src/sheepdog/training/joint_rl_env.py` | `JointActionRLEnv` — all dogs act per step, joint team step |
| `src/sheepdog/policies/hierarchical.py` | `ShepherdNeuralDogPolicy` — inference: scripted shepherd + MaskablePPO dogs |
| `src/sheepdog/training/hierarchical_trainer.py` | `HierarchicalMaskablePPOTrainer` |
| `src/sheepdog/evaluation/benchmark.py` | `run_herding_eval_report()` — proof-of-learning report |

### Observation space

Each dog observes:

1. **Base role-aware features** – same as the sequential PPO baseline
   (pen direction, flock centroid, nearest sheep, role assignment, etc.)
2. **Shepherd command** – 8-value one-hot (`shepherd_cmd_<name>`)
3. **Dog identity** – `dog_id_normalized`, `dog_count_normalized`,
   `dog_id_slot_0…4` (5-value one-hot)

Total extra features: **15**  (8 + 2 + 5)

### `JointActionRLEnv` vs `SheepdogRLAdapter`

Both look like a single-agent Gymnasium env from SB3's perspective.

`SheepdogRLAdapter` (baseline):
- dogs act sequentially; only the last dog's step fires the team update
- credit assignment across early dogs is diluted

`JointActionRLEnv` (hierarchical):
- cycles through dogs 0…N-1, collecting one action per dog per world step
- team update fires when all N actions are committed
- every dog receives a full shared reward after its round completes
- shepherd command is updated after each team step

### Training

```powershell
# Hierarchical neural dogs (Phase A shepherd, learned dog policy)
sheepdog train-hierarchical --total-timesteps 500000

# Proof-of-learning report comparing all policies
sheepdog herding-eval --hierarchical-model artifacts/models/hierarchical/model-000500.zip \
    --hierarchical-checkpoint 500000
```

Checkpoint files use the prefix `hierarchical_checkpoint-NNNNNN.json` to
coexist with baseline checkpoints in the same `artifacts/` directory.

### Proof-of-learning evaluation

```powershell
sheepdog herding-eval [--hierarchical-model PATH] [--hierarchical-checkpoint N]
```

This runs all four policies on the same seeds and writes:

- `reports/herding_eval_latest.json` — machine-readable results
- `reports/herding_eval_latest.md` — human-readable comparison table

Learning is demonstrated when `hierarchical_checkpoint_N` exceeds
`random_policy` and `instinct_only` on success rate and average reward, and
approaches `heuristic_expert`.

### Extending to Phase B (learned shepherd)

Subclass `ScriptedShepherd` and override `issue_command(environment)`.
Pass the instance to `JointActionRLEnv` and `ShepherdNeuralDogPolicy` via
their `shepherd=` constructor argument.  No other code changes are required.

### Artifact layout

```
artifacts/
  checkpoints/
    checkpoint-NNNNNN.json            ← baseline hill-climb / PPO
    hierarchical_checkpoint-NNNNNN.json  ← hierarchical neural
  models/
    hierarchical/
      model-NNNNNN.zip                ← MaskablePPO model weights
  hierarchical-training-state.json    ← hierarchical training resume state
reports/
  herding_eval_latest.json
  herding_eval_latest.md
```


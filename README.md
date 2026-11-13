# Sheepdog Herding Lab

**Can a team of dogs learn to gather a flock, recover strays, and guide every sheep through a pen entrance?**

Sheepdog Herding Lab is an experimental reinforcement-learning environment for exploring that question. A Python simulation models dogs, reactive sheep, fences, and pen geometry. A React interface lets you train policies, inspect learning, compare checkpoints, and watch recorded evaluation episodes.

The current saved experiment uses a **shared actor–critic neural policy trained with MaskablePPO**, with three hidden layers per branch: **256 → 256 → 128**. Dogs learn movement decisions from numerical observations and team rewards. Guided training supplies scripted tactical roles and targets; additional paths explore learning without those inputs and choosing the whole team's actions together.

[Quick start](#quick-start) · [The dogs' brain](#the-dogs-brain) · [Training modes](#training-modes) · [Evaluation](#evaluation-and-evidence) · [Project map](#project-map)

## What the experiment explores

Herding is an indirect control problem: dogs move themselves, and sheep react. A useful policy must learn when to approach, flank, sprint, hold position, collect a stray, or clear the entrance.

The lab brings together:

- **A reproducible simulation:** seeded layouts and sheep behavior, explicit occupancy rules, walls, fences, and sheep personalities.
- **Learning at several levels of guidance:** a linear baseline, guided and emergent neural policies, shepherd commands, and centralized team actions.
- **A curriculum:** progressively harder layouts, flock sizes, pen approaches, and recovery situations.
- **Behavioral evidence:** fixed-seed evaluation, per-component rewards, checkpoint comparisons, diagnostics, and recorded trajectories.
- **Persistent experiments:** resumable training, model checkpoints, backups, adaptive learning controls, and failure-directed practice.

Success means **every sheep is penned**. Partial completion is useful diagnostic information, but counts as a failed episode.

## Quick start

### Requirements

- Python **3.11 or 3.12**; the package declares Python below 3.13.
- Node.js and npm for the web interface.
- PowerShell for the supplied service launchers.

From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev,rl]"
.\start-app.ps1
```

Open the [web interface](http://127.0.0.1:5173). The backend runs at [http://127.0.0.1:8000](http://127.0.0.1:8000).

The launcher uses the local virtual environment when available, installs missing web dependencies, starts the services, and writes logs under `artifacts/startup/`. Restoring a large saved experiment can take several minutes.

```powershell
.\stop-app.ps1
.\restart-app.ps1
```

Stopping through the supplied script requests persistence before shutdown. Startup supports restoring saved training; see [startup details](STARTUP.md).

To run the services manually, use two terminals:

```powershell
# Terminal 1: backend
.\.venv\Scripts\python.exe -m sheepdog.server
```

```powershell
# Terminal 2: frontend
cd web
npm install
npm run dev -- --host 127.0.0.1 --port 5173
```

## The dogs' brain

### A precise description

> The dogs share a feedforward actor–critic policy trained with Maskable Proximal Policy Optimization. In guided mode, each dog receives 54 numerical features. The actor and critic are separate multilayer perceptrons, each with three fully connected hidden layers of 256, 256, and 128 neurons using Tanh activations. The actor produces scores for nine discrete actions; the critic predicts expected discounted return.

In everyday language: **the dogs share one learned brain that chooses movements, while a second branch helps it learn by estimating how promising a situation is.**

### Structure

```text
                         54-feature observation
                                  |
                         Flatten (no weights)
                                  |
                 +----------------+----------------+
                 |                                 |
             ACTOR BRANCH                     CRITIC BRANCH
          Linear: 54 → 256                  Linear: 54 → 256
                Tanh                              Tanh
          Linear: 256 → 256                 Linear: 256 → 256
                Tanh                              Tanh
          Linear: 256 → 128                 Linear: 256 → 128
                Tanh                              Tanh
          Linear: 128 → 9                   Linear: 128 → 1
                 |                                 |
          Legal-action mask                 Estimated return
                 |
       Categorical action distribution
                 |
             Dog's action
```

The actor's nine choices are:

| Movement | Actions |
|---|---|
| Walk | Up, down, left, right |
| Sprint | Up, down, left, right |
| Hold | Wait |

These are cardinal directions with two movement speeds. There are no diagonal actions.

The mask removes illegal choices before action selection. The critic is used during training; choosing a movement depends on the actor.

### How many layers?

The clearest answer is **three hidden layers per branch**.

| Counting convention | Count per branch |
|---|---:|
| Hidden layers only | 3 |
| Learned linear layers, including the output layer | 4 |
| Input, hidden layers, and output | 5 |

The full actor–critic model has **eight learned linear layers across its two parallel branches**. That does not make it an eight-layer sequential network.

For the 54-input, nine-action configuration, the actor has **113,929 parameters**, the critic has **112,897**, and the total is **226,826 trainable parameters**, including biases.

### What is shared?

All dogs use the **same actor weights** and contribute experience to the same training process. Different observations allow that shared policy to choose different actions for different dogs.

The actor and critic have **separate hidden-layer weights**. Their common flattening step has no learned parameters. Sharing a policy among dogs and sharing layers between actor and critic are different design choices.

### What does the model see?

The guided observation builder supplies 54 features:

| Feature group | Count |
|---|---:|
| Dog, pen, flock-center, and assigned-target coordinates | 8 |
| Distances, flock spread, wall proximity, stall/loop indicators, and stray flag | 14 |
| Tactical role one-hot encoding | 5 |
| Focus-sheep relative position and distance | 3 |
| Stray-sheep relative position | 2 |
| Six sheep slots: relative coordinates and penned flag | 18 |
| Two teammate slots: relative coordinates | 4 |
| **Total** | **54** |

Inputs are engineered numerical features. The policy is a feedforward MLP; it receives some history-derived indicators, but has no recurrent memory. Fixed entity slots keep the guided input shape stable across curriculum stages.

Implementation: [observations](src/sheepdog/observations.py), [neural policy](src/sheepdog/policies/neural.py), and [configuration](src/sheepdog/config.py).

## What is learned, and what is programmed?

**Learned:** the neural weights that turn observations into movement probabilities, and the critic's estimate of future return.

**Programmed:** sheep reactions, collision and fence rules, legal-action constraints, reward definitions, scenario generation, and curriculum logic. Guided mode also programs tactical role assignment and role targets. Hierarchical mode currently uses a scripted shepherd.

This distinction defines the research question. Guided training asks whether a policy can learn effective movement with tactical assistance. Emergent observation mode asks how much behavior can be learned when role labels and targets are removed.

The sheep are simulated agents with programmed behavior; they are not independently trained neural policies.

## Training modes

Observation mode and trainer type are separate settings.

| Path | Guidance and decision structure | Model / trainer |
|---|---|---|
| Linear baseline | Scripted tactical roles; learned action scoring | Shared linear policy / hill climbing |
| Guided neural | Role and target features supplied per dog | Shared MLP / MaskablePPO |
| Emergent neural | Removes scripted role and target inputs; includes dog identity and spatial features | Shared MLP / MaskablePPO |
| Hierarchical neural | Scripted shepherd supplies high-level commands to neural dogs | Command-conditioned MLP / hierarchical MaskablePPO trainer |
| Joint-team neural | Centralized observation; chooses actions for the dog slots in one team decision | Team MLP / joint MaskablePPO trainer |

Other policies provide random, instinct-only, and scripted expert comparisons.

Changing observation mode changes the input architecture. Checkpoints require compatible observation shapes, action spaces, and network widths. A change to geometry or reward semantics may preserve tensor shapes while still changing what a trained policy means.

### One training step versus one world step

In the standard shared-dog PPO adapter, actions are selected one dog at a time and buffered. The world advances after the final dog chooses, and that transition emits the team reward. Earlier dog selections return zero reward without advancing the world.

With three dogs, **three PPO transitions correspond to one simulation step**. This matters for interpreting timestep counts, discounting, and credit assignment. The joint-team path explicitly represents a simultaneous team decision.

See the [evaluation roadmap](docs/rl-evaluation-roadmap.md) for the motivation and experiment history.

## How learning works

1. Reset into a seeded layout or sampled training scenario.
2. Build each dog's observation and legal-action mask.
3. Select actions, advance the world, and let sheep react.
4. Compute the shared reward and episode outcome.
5. Collect rollouts and update the actor and critic with PPO.
6. Evaluate checkpoints, record evidence, and persist training state.

The reward combines task completion with intermediate signals such as route progress toward the pen, flock cohesion, gate alignment, and stray recovery. Penalties discourage scattering, stalling, crowding, and wasted time. Optional instinct rewards add further pressure-position and safe-distance shaping.

Pen progress accounts for fences and the open entrance using legal route distance. Penned sheep are retained. Episodes end on success, the step limit, or configured no-progress conditions.

Curriculum stages vary the environment and difficulty. Failure-directed sampling increases exposure to diagnosed problem classes. Adaptive controls can change learning settings and restore a checkpoint after a detected collapse.

## Run experiments

The plain CLI defaults to the linear hill-climbing baseline. Select the neural trainer explicitly for PPO.

```powershell
# Linear baseline
.\.venv\Scripts\python.exe -m sheepdog.cli train --trainer-type hill_climb --policy-type linear --output-dir artifacts/hill_climb

# Fresh guided neural experiment
.\.venv\Scripts\python.exe -m sheepdog.cli train --trainer-type maskable_ppo --policy-type neural --total-timesteps 20000 --output-dir artifacts/guided_trial

# Hierarchical experiment
.\.venv\Scripts\python.exe -m sheepdog.cli train-hierarchical --total-timesteps 20000 --output-dir artifacts/hierarchical_trial
```

Use a new output directory for a new experiment. Training in an existing output directory can restore its saved state.

For a configured run:

```powershell
.\.venv\Scripts\python.exe -m sheepdog.cli train --config path/to/config.json --output-dir artifacts/my_experiment
```

The JSON configuration groups settings under `environment`, `rewards`, `training`, and `policy`. Set `training.observation_mode` to `guided` or `emergent`. Select `joint_maskable_ppo` as the trainer type for the centralized team path.

### Saved experiment snapshot

The local saved files inspected on **October 31, 2026** describe this run:

| Setting | Saved value |
|---|---|
| Trainer / observation mode | MaskablePPO / guided |
| Hidden layers per branch | 256, 256, 128 |
| Input / action dimensions | 54 / 9 |
| Curriculum stage | 34 |
| Field / team / flock | 144 × 104 / 3 dogs / 6 sheep |
| Rollout length / minibatch size | 2,048 / 1,024 |
| Base learning rate / discount factor | 0.0001 / 0.99 |
| Optional instinct rewards | Disabled |
| Adaptive learning / failure-directed training | Enabled |
| Actual saved environment worker count | 1 |

The saved state reports approximately **314 million cumulative PPO timesteps** and a **best recorded evaluation success rate of 100%**. This is a historical checkpoint result on the configured evaluation suite, not a generalization guarantee or a fresh evaluation.

These values come from `artifacts/effective-training-config.json` and `artifacts/training-state.json`; local artifacts may change or be absent in another checkout. The configured eight workers are overridden to one by default on Windows.

## Evaluation and evidence

Checkpoint evaluations use fixed seeds so policies can be compared on the same starts. The default evaluation set contains ten seeds.

Assess behavior using several measures together:

- **Success rate:** fraction of episodes with every sheep penned.
- **Completion time and steps:** efficiency when comparing successful episodes.
- **Sheep penned and timeout rate:** the difference between partial progress and finishing.
- **Reward breakdown:** which incentives the policy is exploiting or satisfying.
- **Recorded trajectories:** whether behavior actually matches the intended skill.

Evaluation replays record the actions, snapshots, and rewards from the rollout. The web interface can replay those exact trajectories, including compressed `.json.gz` files. On-demand runs are separate simulations.

### Temporal interaction diagnostics

Open the **🔬 Temporal Diagnostics** workspace in either the Classic UI (`http://127.0.0.1:5173/?workspace=temporal&replay=checkpoint-039909-seed-000041`) or Studio v2 (`http://127.0.0.1:5173/v2?workspace=temporal&replay=checkpoint-039909-seed-000041`), or click **🔬 Temporal Evidence** while watching any recorded episode to inspect synchronized evidence across world steps:

- **Above-the-fold player and adjacent interval selector:** the animated field, sticky playback transport (`Play/Pause`, `-10/-1/+1/+10` step buttons, speed selector, world-step scrubber, and lead-in/interval/aftermath quick-seek buttons), and synchronized interval selector appear immediately in the initial viewport without scrolling through introductory prose. Introductory guides, provenance keys, and full interval route/reward tables are organized in collapsible `<details>` sections while essential evidence warnings (`Not recorded (starts at step 1)`, frame gaps, `guidance_certainty`, telemetry truncation) remain visible in the top summary bar.
- **Animated field with toggleable overlays:** view recorded dog/sheep trajectories alongside optional recent trails, estimated or recorded guidance targets, derived BFS legal-route waypoints, and descriptive dog–sheep–waypoint geometry.
- **Synchronized interval timeline with stable episode-wide IDs:** select detected interaction intervals (*Repeated movement*, *Stationary hold*, *Productive repositioning*, *Counterproductive pressure candidate*, or *Unresolved pattern*) to seek directly to the interval's lead-in and scrub through the active interval and aftermath by exact `world_step` numbers (including recordings that start at step 1 or contain frame gaps). Filtering intervals by unpenned-sheep count (`All`, `6..1 Sheep`) preserves each interval's episode-wide ID (e.g., `#11` remains `#11` when filtered to `2 Sheep`).
- **Synchronized progress, route distance, reward, and stall charts:** compare unpenned sheep count, aggregate and per-sheep BFS legal-route distances (`S0`–`S5`), reward components, and `no_progress_steps` (including spread-jitter resets).
- **Entity inspector and epistemic provenance:** inspect pre-step decisions (selected vs. executed action, replacement reason, legal-action mask, recomputed or recorded action probabilities, value estimate, and labeled 54/52/69-feature observations) separately from post-step positions and displacements, with explicit badges distinguishing *Recorded*, *Derived*, *Estimated*, *Recomputed*, *Truncated*, and *Not recorded* fields.
- **Read-only preview mode:** launch an isolated read-only diagnostic server without restoring or starting training using `python -m sheepdog.server --read-only --port 8015`.

Compare checkpoint families with the benchmark command:

```powershell
.\.venv\Scripts\python.exe -m sheepdog.cli benchmark --seeds 11 23 37 41 53 --linear-output-dir artifacts/hill_climb --neural-output-dir artifacts/guided_trial
```

For meaningful comparisons, keep the environment, seeds, reward semantics, and evaluation protocol consistent. Confirm improvements across independent training seeds and held-out situations; a best result on a repeatedly used suite alone does not establish robustness.

## Project map

| Location | Responsibility |
|---|---|
| `src/sheepdog/environment.py` | World state, physics, legal actions, and stepping |
| `src/sheepdog/observations.py` | Guided, emergent, and hierarchical feature builders |
| `src/sheepdog/team_strategy.py` | Scripted tactical role assignment |
| `src/sheepdog/shepherd.py` | High-level shepherd commands |
| `src/sheepdog/rewards.py` | Inspectable reward components |
| `src/sheepdog/curriculum.py` | Stage definitions and environment overrides |
| `src/sheepdog/policies/` | Linear, neural, hierarchical, joint-team, and comparison policies |
| `src/sheepdog/training/` | PPO adapters, trainers, sampling, telemetry, and recovery |
| `src/sheepdog/evaluation/` | Benchmarks, scenarios, and evaluation |
| `src/sheepdog/server.py` | Backend API and training orchestration |
| `web/src/` | React interface, field playback, charts, and diagnostics |
| `tests/` | Python regression tests |
| `artifacts/` | Local models, checkpoints, state, evaluations, and telemetry |

### Saved outputs

- `effective-training-config.json`: configuration associated with training.
- `training-state.json`: saved progress and model references.
- `checkpoints/` and `models/`: checkpoint metadata and model weights.
- `evaluations/` and `replays/`: evaluation summaries and recorded trajectories.
- `training-telemetry.sqlite`: persisted episode telemetry.
- `backups/`: saved recovery artifacts.
- `web/public/generated/`: exported assets for the viewer.

## Development checks

```powershell
# Backend tests
.\.venv\Scripts\python.exe -m pytest

# Frontend checks, from web/
npm test
npm run lint
npm run build
```

Long-running Python tests are marked `slow`; use `-m "not slow"` for a shorter test run.

## Further reading

- [Startup and restoration](STARTUP.md)
- [Architecture notes](docs/architecture.md)
- [Network guide](docs/sheep_dog_rl_network_guide.md)
- [Observation modes](docs/training-modes.md)
- [Evaluation roadmap and experiment decisions](docs/rl-evaluation-roadmap.md)
- [Model training history](docs/model_training_history.md)
- [Model changelog](MODEL_CHANGELOG.md)
- [Previous README, preserved for comparison](README.before-2026-10-31.md)

Some deeper documents describe earlier architectures and experiments. For the exact architecture of a run, inspect its checkpoint metadata, saved configuration, and the corresponding implementation.

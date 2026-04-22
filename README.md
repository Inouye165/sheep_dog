# Sheepdog Herding Lab

Sheepdog Herding Lab is a deterministic reinforcement-learning simulation where a team of dogs learns to herd multiple sheep into a pen.

The project is split into two parts:

- A Python simulation, training, and evaluation pipeline.
- A React + TypeScript replay viewer that reads exported checkpoint data.

The first version is intentionally honest about what it does. It trains a simple shared policy with hill climbing, measures real checkpoint results, and exports replays that the UI can inspect.

## Why This Exists

The goal is to make herd behavior visible, measurable, and comparable across checkpoints without hiding the actual results behind demo logic or hand-waved metrics.

## Quick Start

1. Create a Python environment and install the backend package in editable mode with dev tools.
2. Run the training pipeline to generate checkpoints, evaluations, and replay files.
3. Install the web dependencies and start the Vite dev server.
4. Open the UI and load the exported replay data.

Example commands:

```powershell
python -m pip install -e .[dev]
python -m sheepdog.server
python -m sheepdog.cli train
python -m sheepdog.cli export-demo
cd web
npm install
npm run dev
```

## Repo Structure

- `src/sheepdog/` Python simulation, policies, training, evaluation, checkpointing, and replay export.
- `tests/` Python regression tests.
- `web/` React + TypeScript UI for replay playback and checkpoint comparison.
- `docs/architecture.md` High-level architecture notes.
- `artifacts/` Generated training and evaluation output.

## Architecture Overview

The simulation is grid based and deterministic for a fixed seed. Dogs act first, sheep react to nearby pressure, and the environment tracks progress toward the pen. Rewards are decomposed into named pieces so the total can be audited.

The trainer uses a simple shared trainable policy with hill climbing. It does not pretend to be PPO or another full RL algorithm. Checkpoints are real evaluation snapshots, not fabricated milestones.

The current baseline now adds dynamic team roles on top of that same hill-climbing policy. Dogs still share one linear weight vector, but each step the environment assigns tactical jobs such as rear pressure, flanking, collecting strays, and blocking near the pen, then scores actions against those temporary responsibilities.

The web app is a viewer, not a second simulation engine. It loads exported checkpoint and replay JSON files from `web/public/generated/` and plays them back frame by frame.

## Simulation Concepts

- Dogs are AI-controlled and share one policy.
- Dogs can switch dynamic tactical roles each step while still using the shared linear hill-climbing policy.
- Sheep flee from nearby dogs, stay loosely flocked, and avoid walls.
- The pen is a goal area inside the field.
- Episodes succeed only when every sheep is penned.
- The environment stops on success, timeout, or no-progress conditions.

## Training

Run training with:

```powershell
python -m sheepdog.cli train
```

This writes checkpoint JSON files under `artifacts/checkpoints/`, evaluation JSON and CSV files under `artifacts/evaluations/`, and web assets under `web/public/generated/`.

## Checkpoints

Default checkpoint episodes:

`0, 5, 10, 25, 50, 100, 500, 1000`

Each checkpoint includes:

- checkpoint episode
- total training episodes so far
- policy name and weights
- environment configuration
- reward configuration
- success rate over evaluation seeds
- average completion steps and seconds
- timeout rate
- average sheep penned
- average reward
- replay path for the exported run

## Evaluation

Evaluation uses fixed seeds so checkpoints are compared fairly.

The evaluator writes:

- JSON summary files
- CSV per-seed records
- replay JSON files for each evaluated seed

Those files live under `artifacts/evaluations/` and are copied to the web public folder for playback.

## UI

Start the web viewer with:

```powershell
cd web
npm install
npm run dev
```

The UI shows:

- the field, dogs, sheep, and pen
- compact dog role labels during replay
- the selected checkpoint and seed
- policy mode
- run state
- sheep penned count
- elapsed simulated time and steps
- reward breakdown
- no-progress warning
- start, pause, resume, stop, reset, evaluation refresh, and export controls

## Interactive Training

Start the API server in one terminal:

```powershell
python -m sheepdog.server
```

Then open the UI, choose an episode count in the Train panel, toggle Fast mode if you want the replay to advance without visible delay, and press Train. The page will poll training status and refresh the latest checkpoint as each episode finishes.

## Tests

Python:

```powershell
python -m pip install -e .[dev]
python -m pytest
ruff check src tests
ruff format --check src tests
```

Web:

```powershell
cd web
npm install
npm run lint
npm run test
npm run build
```

## Configuration Example

You can provide a JSON config file to the CLI. Example:

```json
{
  "environment": {
    "width": 48,
    "height": 32,
    "dogs": 3,
    "sheep": 8,
    "max_steps": 360
  },
  "rewards": {
    "progress_scale": 2.0,
    "sheep_penned_reward": 8.0,
    "terminal_success_reward": 20.0
  },
  "training": {
    "episodes": 1000,
    "output_dir": "artifacts"
  }
}
```

## Known Limitations

- The first release uses a simple hill-climbing policy rather than PPO.
- Dynamic cooperation is role-aware, but it is still a shared linear baseline rather than a neural policy.
- The UI is a replay viewer for exported runs, not a live training dashboard.
- There is no backend API server yet; the browser reads generated files directly.

## Next Steps

1. Compare this role-aware hill-climbing baseline against PPO or MaskablePPO rather than replacing it blindly.
2. Add richer role analytics and per-dog trajectory overlays in the replay viewer.
3. Expand replay analytics with per-step trajectory charts.
4. Add a lightweight local API if live browser control becomes necessary.

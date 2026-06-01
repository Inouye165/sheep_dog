"""Evaluate checkpoints against saved named scenarios."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from sheepdog.config import LabConfig
from sheepdog.environment import EpisodeResult, SheepdogEnvironment
from sheepdog.evaluation.evaluator import _policy_metadata
from sheepdog.evaluation.scenarios import Scenario, ScenarioStore
from sheepdog.policies.base import Policy
from sheepdog.replay.store import ReplayStore

CheckpointMode = Literal["latest", "global_best", "scenario_best", "specific"]

EMPTY_INDEX: dict[str, Any] = {
    "scenarios": [],
    "runs": [],
    "best_by_scenario": {},
    "latest_checkpoint_episode": None,
    "latest_runs": [],
}


def is_strictly_better_scenario_result(
    candidate: dict[str, Any],
    current: dict[str, Any] | None,
) -> bool:
    """Return True when *candidate* strictly beats *current* on a saved scenario."""

    if current is None:
        return True
    if bool(candidate["success"]) and not bool(current["success"]):
        return True
    if bool(current["success"]) and not bool(candidate["success"]):
        return False
    if int(candidate["sheep_penned"]) > int(current["sheep_penned"]):
        return True
    if int(candidate["sheep_penned"]) < int(current["sheep_penned"]):
        return False
    if int(candidate["steps"]) < int(current["steps"]):
        return True
    if int(candidate["steps"]) > int(current["steps"]):
        return False
    return float(candidate["reward_total"]) > float(current["reward_total"])


@dataclass(frozen=True, slots=True)
class ScenarioRunResult:
    """Outcome of running one policy on one saved scenario."""

    scenario_id: str
    checkpoint_episode: int
    success: bool
    sheep_penned: int
    steps: int
    timeout: bool
    stopped: bool
    reward_total: float
    replay_path: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ScenarioResultsStore:
    """Track per-scenario checkpoint results in ``scenario-results.json``."""

    FILENAME = "scenario-results.json"
    WEB_FILENAME = "scenario-index.json"

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / self.FILENAME
        self.replay_store = ReplayStore(self.root / "replays")
        if not self.path.exists():
            self.path.write_text(json.dumps(EMPTY_INDEX, indent=2), encoding="utf-8")

    def _read(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"runs": [], "best_by_scenario": {}}
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {"runs": [], "best_by_scenario": {}}
        if not isinstance(payload, dict):
            return {"runs": [], "best_by_scenario": {}}
        payload.setdefault("runs", [])
        payload.setdefault("best_by_scenario", {})
        return payload

    def best_for_scenario(self, scenario_id: str) -> dict[str, Any] | None:
        best = self._read().get("best_by_scenario", {})
        if not isinstance(best, dict):
            return None
        entry = best.get(scenario_id)
        return entry if isinstance(entry, dict) else None

    def record_run(self, result: ScenarioRunResult) -> dict[str, Any]:
        """Append a run and update per-scenario best checkpoint when improved."""

        payload = self._read()
        runs = payload.setdefault("runs", [])
        if not isinstance(runs, list):
            runs = []
            payload["runs"] = runs
        entry = result.to_dict()
        runs.append(entry)
        best_by_scenario = payload.setdefault("best_by_scenario", {})
        if not isinstance(best_by_scenario, dict):
            best_by_scenario = {}
            payload["best_by_scenario"] = best_by_scenario
        current_best = best_by_scenario.get(result.scenario_id)
        if is_strictly_better_scenario_result(entry, current_best):
            best_by_scenario[result.scenario_id] = dict(entry)
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload

    def latest_by_checkpoint(self, checkpoint_episode: int) -> list[dict[str, Any]]:
        runs = self._read().get("runs", [])
        if not isinstance(runs, list):
            return []
        return [
            run
            for run in runs
            if isinstance(run, dict) and int(run.get("checkpoint_episode", -1)) == checkpoint_episode
        ]

    def export_index(
        self,
        *,
        scenarios: tuple[Scenario, ...],
        latest_checkpoint_episode: int | None,
        web_export_dir: Path,
    ) -> Path:
        """Write scenario definitions and results for static hosting."""

        payload = self._read()
        best_by_scenario = payload.get("best_by_scenario", {})
        if not isinstance(best_by_scenario, dict):
            best_by_scenario = {}
        runs = payload.get("runs", [])
        if not isinstance(runs, list):
            runs = []
        latest_runs: list[dict[str, Any]] = []
        if latest_checkpoint_episode is not None:
            latest_runs = self.latest_by_checkpoint(latest_checkpoint_episode)
        export_payload = {
            "scenarios": [scenario.to_dict() for scenario in scenarios],
            "best_by_scenario": best_by_scenario,
            "runs": runs,
            "latest_checkpoint_episode": latest_checkpoint_episode,
            "latest_runs": latest_runs,
        }
        web_export_dir.mkdir(parents=True, exist_ok=True)
        target = web_export_dir / self.WEB_FILENAME
        target.write_text(json.dumps(export_payload, indent=2), encoding="utf-8")
        return target

    def build_bundle(
        self,
        *,
        scenarios: tuple[Scenario, ...],
        latest_checkpoint_episode: int | None,
    ) -> dict[str, Any]:
        """Return the full scenario index payload for API responses."""

        payload = self._read()
        best_by_scenario = payload.get("best_by_scenario", {})
        if not isinstance(best_by_scenario, dict):
            best_by_scenario = {}
        runs = payload.get("runs", [])
        if not isinstance(runs, list):
            runs = []
        latest_runs: list[dict[str, Any]] = []
        if latest_checkpoint_episode is not None:
            latest_runs = self.latest_by_checkpoint(latest_checkpoint_episode)
        return {
            "scenarios": [scenario.to_dict() for scenario in scenarios],
            "best_by_scenario": best_by_scenario,
            "runs": runs,
            "latest_checkpoint_episode": latest_checkpoint_episode,
            "latest_runs": latest_runs,
        }


def config_for_scenario(base_config: LabConfig, scenario: Scenario) -> LabConfig:
    """Build a lab config sized for the saved scenario layout."""

    from dataclasses import replace

    environment = replace(
        base_config.environment,
        width=scenario.width,
        height=scenario.height,
        dogs=len(scenario.dogs),
        sheep=len(scenario.sheep),
        pen_width=scenario.pen.width,
        pen_height=scenario.pen.height,
        pen_opening=scenario.pen.opening,
        sheep_personality_strength=scenario.sheep_personality_strength,
        sheep_personality_seed_offset=scenario.sheep_personality_seed_offset,
        seed_offset=scenario.seed_offset,
    )
    return replace(base_config, environment=environment)


def run_scenario(
    policy: Policy,
    base_config: LabConfig,
    scenario: Scenario,
    *,
    capture_replay: bool = False,
) -> EpisodeResult:
    """Run *policy* on a fixed-layout scenario (uses ``reset_from_scenario``)."""

    config = config_for_scenario(base_config, scenario)
    environment = SheepdogEnvironment(config)
    return environment.run_policy_on_scenario(policy, scenario, capture_replay=capture_replay)


def _write_scenario_replay(
    results_store: ScenarioResultsStore,
    *,
    scenario: Scenario,
    checkpoint_episode: int,
    policy: Policy,
    episode: EpisodeResult,
    web_export_dir: Path,
    run_config: LabConfig,
) -> str:
    trainer_type, policy_type, replay_mode = _policy_metadata(
        episode.policy_name,
        trainer_type=getattr(policy, "trainer_type", None),
        policy_type=getattr(policy, "policy_type", None),
    )
    replay_name = f"scenario-{scenario.id}-checkpoint-{checkpoint_episode:06d}.json"
    artifact_path = results_store.replay_store.write(
        replay_name,
        {
            "seed": scenario.seed,
            "scenario_id": scenario.id,
            "scenario_name": scenario.name,
            "checkpoint_episode": checkpoint_episode,
            "policy_name": episode.policy_name,
            "trainer_type": trainer_type,
            "policy_type": policy_type,
            "replay_mode": replay_mode,
            "environment": {
                "dogs": run_config.environment.dogs,
                "sheep": run_config.environment.sheep,
                "width": run_config.environment.width,
                "height": run_config.environment.height,
                "sheep_personality_strength": run_config.environment.sheep_personality_strength,
                "curriculum_stage": run_config.rewards.instincts.curriculum_stage,
                "enable_instinct_rewards": run_config.rewards.instincts.enable_instinct_rewards,
            },
            "final_snapshot": episode.final_snapshot.to_dict(),
            "stats": asdict(episode.stats),
            "frames": [frame.to_dict() for frame in episode.replay],
        },
    )
    web_replays = web_export_dir / "replays"
    web_replays.mkdir(parents=True, exist_ok=True)
    web_target = web_replays / replay_name
    web_target.write_text(artifact_path.read_text(encoding="utf-8"), encoding="utf-8")
    return f"/generated/replays/{replay_name}"


def evaluate_scenario(
    base_config: LabConfig,
    policy: Policy,
    scenario: Scenario,
    checkpoint_episode: int,
    *,
    record_result: bool = True,
) -> ScenarioRunResult:
    """Run *policy* on *scenario*, persist replay, and optionally record metrics."""

    run_config = config_for_scenario(base_config, scenario)
    episode = run_scenario(policy, base_config, scenario, capture_replay=True)
    output_root = Path(base_config.training.output_dir)
    results_store = ScenarioResultsStore(output_root / "scenarios")
    web_export_dir = Path(base_config.training.web_export_dir)
    replay_path = _write_scenario_replay(
        results_store,
        scenario=scenario,
        checkpoint_episode=checkpoint_episode,
        policy=policy,
        episode=episode,
        web_export_dir=web_export_dir,
        run_config=run_config,
    )
    result = ScenarioRunResult(
        scenario_id=scenario.id,
        checkpoint_episode=checkpoint_episode,
        success=episode.stats.success,
        sheep_penned=episode.stats.sheep_penned,
        steps=episode.stats.steps,
        timeout=episode.stats.timeout,
        stopped=episode.stats.stopped,
        reward_total=episode.stats.reward_total,
        replay_path=replay_path,
    )
    if record_result:
        results_store.record_run(result)
        scenario_store = ScenarioStore(output_root / "scenarios")
        results_store.export_index(
            scenarios=scenario_store.list(),
            latest_checkpoint_episode=checkpoint_episode,
            web_export_dir=web_export_dir,
        )
    return result


def evaluate_checkpoint_on_scenarios(
    base_config: LabConfig,
    policy: Policy,
    checkpoint_episode: int,
    *,
    scenario_ids: tuple[str, ...] | None = None,
) -> list[ScenarioRunResult]:
    """Evaluate *policy* on all (or selected) saved scenarios."""

    output_root = Path(base_config.training.output_dir)
    scenario_store = ScenarioStore(output_root / "scenarios")
    scenarios = scenario_store.list()
    if scenario_ids is not None:
        allowed = set(scenario_ids)
        scenarios = tuple(scenario for scenario in scenarios if scenario.id in allowed)
    results: list[ScenarioRunResult] = []
    for scenario in scenarios:
        results.append(
            evaluate_scenario(
                base_config,
                policy,
                scenario,
                checkpoint_episode,
                record_result=True,
            )
        )
    return results


def resolve_latest_checkpoint_episode(output_root: Path) -> int | None:
    summary_path = output_root / "training-summary.json"
    if not summary_path.exists():
        return None
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    checkpoints = payload.get("checkpoints")
    if not isinstance(checkpoints, list) or not checkpoints:
        return None
    latest = checkpoints[-1]
    if isinstance(latest, dict) and latest.get("checkpoint_episode") is not None:
        return int(latest["checkpoint_episode"])
    return None


def resolve_global_best_checkpoint_episode(
    output_root: Path,
    *,
    web_export_dir: Path | None = None,
) -> int | None:
    state_path = output_root / "training-state.json"
    if state_path.exists():
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
            formal = state.get("best_formal_episode")
            if formal is not None:
                return int(formal)
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            pass
    candidates: list[Path] = []
    if web_export_dir is not None:
        candidates.append(Path(web_export_dir) / "checkpoint-index.json")
    default_web = Path(LabConfig().training.web_export_dir) / "checkpoint-index.json"
    if default_web not in candidates:
        candidates.append(default_web)
    for index_path in candidates:
        if not index_path.exists():
            continue
        try:
            payload = json.loads(index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        checkpoints = payload.get("checkpoints")
        if not isinstance(checkpoints, list) or not checkpoints:
            continue
        best = checkpoints[0]
        for entry in checkpoints[1:]:
            if not isinstance(entry, dict) or not isinstance(best, dict):
                continue
            if _is_strictly_better_checkpoint_entry(entry, best):
                best = entry
        if isinstance(best, dict) and best.get("checkpoint_episode") is not None:
            return int(best["checkpoint_episode"])
    return resolve_latest_checkpoint_episode(output_root)


def _is_strictly_better_checkpoint_entry(candidate: dict[str, Any], current: dict[str, Any]) -> bool:
    c_stage = (candidate.get("reward_config") or {}).get("instincts", {}).get("curriculum_stage", 0)
    cur_stage = (current.get("reward_config") or {}).get("instincts", {}).get("curriculum_stage", 0)
    if c_stage > cur_stage:
        return True
    if c_stage < cur_stage:
        return False
    c_rate = float(candidate.get("success_rate", 0.0))
    cur_rate = float(current.get("success_rate", 0.0))
    if c_rate > cur_rate:
        return True
    if c_rate < cur_rate:
        return False
    c_steps = float(candidate.get("average_completion_steps", float("inf")))
    cur_steps = float(current.get("average_completion_steps", float("inf")))
    return c_steps < cur_steps


def resolve_checkpoint_episode(
    mode: CheckpointMode,
    *,
    output_root: Path,
    scenario_id: str | None = None,
    explicit_episode: int | None = None,
    web_export_dir: Path | None = None,
) -> int:
    """Map a UI checkpoint mode to a concrete checkpoint episode number."""

    if mode == "specific":
        if explicit_episode is None:
            raise ValueError("checkpoint_episode is required when checkpoint_mode is 'specific'")
        return int(explicit_episode)
    if mode == "latest":
        episode = resolve_latest_checkpoint_episode(output_root)
        if episode is None:
            raise FileNotFoundError("No checkpoints found for latest mode")
        return episode
    if mode == "global_best":
        episode = resolve_global_best_checkpoint_episode(
            output_root,
            web_export_dir=web_export_dir,
        )
        if episode is None:
            raise FileNotFoundError("No global best checkpoint found")
        return episode
    if mode == "scenario_best":
        if not scenario_id:
            raise ValueError("scenario_id is required for scenario_best mode")
        best = ScenarioResultsStore(output_root / "scenarios").best_for_scenario(scenario_id)
        if best is None or best.get("checkpoint_episode") is None:
            raise FileNotFoundError(f"No best checkpoint recorded yet for scenario {scenario_id}")
        return int(best["checkpoint_episode"])
    raise ValueError(f"Unknown checkpoint_mode: {mode}")


def refresh_scenario_exports(base_config: LabConfig) -> dict[str, Any]:
    """Ensure JSON artifacts exist and return the scenario bundle."""

    output_root = Path(base_config.training.output_dir)
    scenario_store = ScenarioStore(output_root / "scenarios")
    results_store = ScenarioResultsStore(output_root / "scenarios")
    latest = resolve_latest_checkpoint_episode(output_root)
    bundle = results_store.build_bundle(
        scenarios=scenario_store.list(),
        latest_checkpoint_episode=latest,
    )
    results_store.export_index(
        scenarios=scenario_store.list(),
        latest_checkpoint_episode=latest,
        web_export_dir=Path(base_config.training.web_export_dir),
    )
    return bundle


def export_scenario_index(base_config: LabConfig, latest_checkpoint_episode: int | None) -> dict[str, Any]:
    """Refresh persisted scenario index files and return the bundle."""

    if latest_checkpoint_episode is None:
        latest_checkpoint_episode = resolve_latest_checkpoint_episode(
            Path(base_config.training.output_dir)
        )
    return refresh_scenario_exports(base_config)

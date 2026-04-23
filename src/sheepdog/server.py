"""Local HTTP API for interactive training control."""

from __future__ import annotations

import json
import shutil
import threading
from dataclasses import asdict, dataclass, replace
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from sheepdog.checkpoints.store import CheckpointMetadata
from sheepdog.config import (
    EnvironmentConfig,
    InstinctRewardConfig,
    LabConfig,
    RewardConfig,
    TrainingConfig,
)
from sheepdog.curriculum import apply_training_profile
from sheepdog.environment import SheepdogEnvironment
from sheepdog.policies.base import PolicyMode
from sheepdog.policies.factory import load_playable_policy
from sheepdog.policies.heuristic import InstinctOnlyPolicy
from sheepdog.training.factory import create_trainer
from sheepdog.training.trainer import Trainer


@dataclass(frozen=True, slots=True)
class ReplaySelection:
    """Resolved replay policy, config, and truthfulness metadata."""

    config: LabConfig
    checkpoint_episode: int | None
    trainer_type: str
    policy_type: str
    policy_mode: str
    replay_mode: str


def _policy_metadata(
    policy_mode: str,
    trainer_type: str | None = None,
    policy_type: str | None = None,
    *,
    trained: bool = False,
) -> tuple[str, str, str]:
    """Return normalized trainer, policy, and replay-mode labels."""

    normalized_mode = policy_mode or "instinct_only"
    normalized_trainer = trainer_type or "baseline"
    normalized_policy_type = policy_type or "instinct"
    replay_mode = "baseline"

    if normalized_mode == "neural_policy" or normalized_trainer == "maskable_ppo":
        normalized_trainer = "maskable_ppo"
        normalized_policy_type = "neural"
        replay_mode = "neural_ppo"
    elif normalized_mode == "trained_policy" and trained:
        normalized_trainer = "hill_climb"
        normalized_policy_type = "linear"
        replay_mode = "trained_linear"
    elif normalized_mode == "heuristic_expert":
        normalized_trainer = "baseline"
        normalized_policy_type = "heuristic"
    elif normalized_mode in {"random_untrained", "random_policy"}:
        normalized_trainer = "baseline"
        normalized_policy_type = "random"
    else:
        normalized_mode = "instinct_only"
        normalized_trainer = "baseline"
        normalized_policy_type = "instinct"

    return normalized_trainer, normalized_policy_type, replay_mode


def _reward_config_from_payload(payload: dict[str, Any]) -> RewardConfig:
    reward_payload = dict(payload)
    instincts_payload = reward_payload.pop("instincts", None)
    instincts = (
        InstinctRewardConfig(**instincts_payload)
        if isinstance(instincts_payload, dict)
        else InstinctRewardConfig()
    )
    return RewardConfig(instincts=instincts, **reward_payload)


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _load_checkpoint_payload(output_root: Path, checkpoint_episode: int) -> dict[str, Any]:
    checkpoint_path = output_root / "checkpoints" / f"checkpoint-{checkpoint_episode:06d}.json"
    payload = _load_json(checkpoint_path)
    if payload is None:
        raise FileNotFoundError(f"Checkpoint {checkpoint_episode} not found")
    return payload


def _load_latest_checkpoint_payload(output_root: Path) -> dict[str, Any] | None:
    summary_payload = _load_json(output_root / "training-summary.json")
    if not isinstance(summary_payload, dict):
        return None
    checkpoints = summary_payload.get("checkpoints")
    if not isinstance(checkpoints, list) or not checkpoints:
        return None
    latest = checkpoints[-1]
    checkpoint_name = latest.get("checkpoint")
    if isinstance(checkpoint_name, str):
        payload = _load_json(output_root / "checkpoints" / checkpoint_name)
        if isinstance(payload, dict):
            return payload
    checkpoint_episode = latest.get("checkpoint_episode")
    if checkpoint_episode is None:
        return None
    return _load_checkpoint_payload(output_root, int(checkpoint_episode))


def _config_from_checkpoint_payload(base_config: LabConfig, payload: dict[str, Any]) -> LabConfig:
    environment_payload = payload.get("environment_config")
    reward_payload = payload.get("reward_config")
    environment = (
        EnvironmentConfig(**environment_payload)
        if isinstance(environment_payload, dict)
        else base_config.environment
    )
    rewards = (
        _reward_config_from_payload(reward_payload)
        if isinstance(reward_payload, dict)
        else base_config.rewards
    )
    return replace(base_config, environment=environment, rewards=rewards)


def _resolve_replay_selection(
    base_config: LabConfig,
    *,
    checkpoint_episode: int | None = None,
    policy_mode: PolicyMode | None = None,
    effective_config: dict[str, Any] | None = None,
) -> ReplaySelection:
    """Resolve the truthful replay mode and effective config for one run."""

    output_root = Path(base_config.training.output_dir)
    requested_mode = policy_mode
    if checkpoint_episode is not None:
        checkpoint_payload = _load_checkpoint_payload(output_root, checkpoint_episode)
        resolved_mode = str(checkpoint_payload.get("policy_name") or requested_mode or "instinct_only")
        trained = int(checkpoint_payload.get("total_training_episodes", 0)) > 0
        trainer_type, policy_type, replay_mode = _policy_metadata(
            resolved_mode,
            str(checkpoint_payload.get("trainer_type") or ""),
            str(checkpoint_payload.get("policy_type") or ""),
            trained=trained,
        )
        replay_config = _config_from_checkpoint_payload(base_config, checkpoint_payload)
        replay_config = replace(
            replay_config,
            policy=replace(replay_config.policy, policy_mode=resolved_mode),
        )
        return ReplaySelection(
            config=replay_config,
            checkpoint_episode=checkpoint_episode,
            trainer_type=trainer_type,
            policy_type=policy_type,
            policy_mode=resolved_mode,
            replay_mode=replay_mode,
        )

    latest_checkpoint_payload = _load_latest_checkpoint_payload(output_root)
    if requested_mode in {None, "trained_policy", "neural_policy"} and latest_checkpoint_payload:
        latest_total = int(latest_checkpoint_payload.get("total_training_episodes", 0))
        latest_mode = str(latest_checkpoint_payload.get("policy_name") or requested_mode or "instinct_only")
        if latest_total > 0 and (requested_mode is None or requested_mode == latest_mode):
            latest_checkpoint_episode = int(latest_checkpoint_payload.get("checkpoint_episode", 0))
            trainer_type, policy_type, replay_mode = _policy_metadata(
                latest_mode,
                str(latest_checkpoint_payload.get("trainer_type") or ""),
                str(latest_checkpoint_payload.get("policy_type") or ""),
                trained=True,
            )
            replay_config = _config_from_checkpoint_payload(base_config, latest_checkpoint_payload)
            replay_config = replace(
                replay_config,
                policy=replace(replay_config.policy, policy_mode=latest_mode),
            )
            return ReplaySelection(
                config=replay_config,
                checkpoint_episode=latest_checkpoint_episode,
                trainer_type=trainer_type,
                policy_type=policy_type,
                policy_mode=latest_mode,
                replay_mode=replay_mode,
            )

    enable_instinct_rewards = None
    curriculum_stage = None
    debug_reward_breakdown = None
    if isinstance(effective_config, dict):
        enable_instinct_rewards = effective_config.get("enable_instinct_rewards")
        curriculum_stage = effective_config.get("curriculum_stage")
        debug_reward_breakdown = effective_config.get("debug_reward_breakdown")
    replay_config = apply_training_profile(
        base_config,
        enable_instinct_rewards=(
            None if enable_instinct_rewards is None else bool(enable_instinct_rewards)
        ),
        curriculum_stage=(None if curriculum_stage is None else int(curriculum_stage)),
        debug_reward_breakdown=(
            None if debug_reward_breakdown is None else bool(debug_reward_breakdown)
        ),
    )
    resolved_mode = requested_mode or "instinct_only"
    trainer_type, policy_type, replay_mode = _policy_metadata(resolved_mode, trained=False)
    replay_config = replace(
        replay_config,
        policy=replace(replay_config.policy, policy_mode=resolved_mode),
    )
    return ReplaySelection(
        config=replay_config,
        checkpoint_episode=None,
        trainer_type=trainer_type,
        policy_type=policy_type,
        policy_mode=resolved_mode,
        replay_mode=replay_mode,
    )


def _read_persisted_total() -> int:
    """Best-effort read of persisted total episodes for status display."""
    try:
        config = LabConfig()
        state_path = Path(config.training.output_dir) / Trainer.STATE_FILENAME
        if not state_path.exists():
            return 0
        payload = json.loads(state_path.read_text(encoding="utf-8"))
        return int(payload.get("total_episodes_trained", 0))
    except (OSError, json.JSONDecodeError, ValueError):
        return 0


def _build_training_job_config(
    requested_episodes: int,
    fast_mode: bool,
    *,
    enable_instinct_rewards: bool | None = None,
    curriculum_stage: int | None = None,
    debug_reward_breakdown: bool | None = None,
) -> LabConfig:
    """Build the effective training configuration for one requested job."""

    config = LabConfig()
    total_episodes = max(1, requested_episodes)
    training_episodes = max(0, total_episodes - 1)
    checkpoint_episodes = tuple(range(total_episodes))
    evaluation_seeds = (11,) if fast_mode else config.training.evaluation_seeds
    training_config = TrainingConfig(
        episodes=training_episodes,
        checkpoint_episodes=checkpoint_episodes,
        evaluation_seeds=evaluation_seeds,
        train_seed=config.training.train_seed,
        evaluation_seed=config.training.evaluation_seed,
        candidate_evaluation_seeds=config.training.candidate_evaluation_seeds,
        candidate_pool_size=config.training.candidate_pool_size,
        mutation_scale=config.training.mutation_scale,
        output_dir=config.training.output_dir,
        web_export_dir=config.training.web_export_dir,
    )
    job_config = replace(config, training=training_config)
    return apply_training_profile(
        job_config,
        enable_instinct_rewards=enable_instinct_rewards,
        curriculum_stage=curriculum_stage,
        debug_reward_breakdown=debug_reward_breakdown,
    )


def _load_playable_policy(
    config: LabConfig,
    *,
    checkpoint_episode: int | None = None,
    policy_mode: PolicyMode | None = None,
) -> object:
    """Return a runnable policy for replay requests."""

    return load_playable_policy(
        config,
        checkpoint_episode=checkpoint_episode,
        policy_mode=policy_mode,
    )


def _replay_payload(result: Any) -> dict[str, Any]:
    """Convert an environment run result to the web replay schema."""

    return {
        "seed": result.seed,
        "policy_name": result.policy_name,
        "final_snapshot": result.final_snapshot.to_dict(),
        "stats": asdict(result.stats),
        "frames": [frame.to_dict() for frame in result.replay],
    }


class _BaselineExportTrainer(Trainer):
    """Expose protected export helpers for the baseline flow."""

    def export_baseline_assets(
        self,
        config: LabConfig,
        checkpoint_payload: dict[str, Any],
        representative_replay_path: Path,
        checkpoint_path: Path,
        summary: Any,
    ) -> None:
        self._export_web_assets(
            Path(config.training.web_export_dir),
            [checkpoint_payload],
            summary,
            representative_replay_path,
            checkpoint_path,
        )


class TrainingManager:
    """Track one background training job at a time."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._status: dict[str, Any] = self._initial_status()

    def _initial_status(self) -> dict[str, Any]:
        config = LabConfig()
        instincts = config.rewards.instincts
        trainer_type, policy_type, replay_mode = _policy_metadata(config.policy.policy_mode)
        return {
            "running": False,
            "fast_mode": True,
            "trainer_type": trainer_type,
            "policy_type": policy_type,
            "enable_instinct_rewards": instincts.enable_instinct_rewards,
            "policy_mode": config.policy.policy_mode,
            "replay_mode": replay_mode,
            "allow_instinct_target_awareness": config.policy.allow_instinct_target_awareness,
            "handler_target_enabled": config.policy.handler_target_enabled,
            "debug_reward_breakdown": instincts.debug_reward_breakdown,
            "curriculum_stage": instincts.curriculum_stage,
            "requested_episodes": 0,
            "completed_episodes": 0,
            "batch_total_episodes": 0,
            "batch_completed_episodes": 0,
            "total_episodes_trained": _read_persisted_total(),
            "current_episode": None,
            "checkpoint_episode": None,
            "latest_checkpoint_episode": None,
            "latest_seed": None,
            "latest_replay_path": None,
            "best_score": None,
            "phase": "idle",
            "message": "Idle",
            "error": None,
        }

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._status)

    def start(
        self,
        requested_episodes: int,
        fast_mode: bool,
        *,
        enable_instinct_rewards: bool | None = None,
        curriculum_stage: int | None = None,
        debug_reward_breakdown: bool | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return dict(self._status)

            self._status = self._initial_status()
            self._status.update(
                {
                    "running": True,
                    "fast_mode": fast_mode,
                    "enable_instinct_rewards": (
                        self._status["enable_instinct_rewards"]
                        if enable_instinct_rewards is None
                        else enable_instinct_rewards
                    ),
                    "debug_reward_breakdown": (
                        self._status["debug_reward_breakdown"]
                        if debug_reward_breakdown is None
                        else debug_reward_breakdown
                    ),
                    "curriculum_stage": (
                        self._status["curriculum_stage"]
                        if curriculum_stage is None
                        else max(0, int(curriculum_stage))
                    ),
                    "requested_episodes": requested_episodes,
                    "message": "Queued training job",
                }
            )
            self._thread = threading.Thread(
                target=self._run_training,
                args=(requested_episodes, fast_mode),
                kwargs={
                    "enable_instinct_rewards": enable_instinct_rewards,
                    "curriculum_stage": curriculum_stage,
                    "debug_reward_breakdown": debug_reward_breakdown,
                },
                daemon=True,
            )
            self._thread.start()
            return dict(self._status)

    def clear(self) -> tuple[dict[str, Any], int]:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                payload = dict(self._status)
                payload["message"] = "Cannot clear training while a job is running"
                return payload, HTTPStatus.CONFLICT

        config = LabConfig()
        self._clear_training_outputs(config)
        self._export_untrained_baseline(config)

        with self._lock:
            self._thread = None
            self._status = self._initial_status()
            self._status["message"] = "Training cleared. Baseline replay restored"
            return dict(self._status), HTTPStatus.OK

    def _clear_training_outputs(self, config: LabConfig) -> None:
        output_root = Path(config.training.output_dir)
        generated_root = Path(config.training.web_export_dir)
        self._remove_path(output_root / "checkpoints")
        self._remove_path(output_root / "evaluations")
        self._remove_path(output_root / Trainer.STATE_FILENAME)
        self._remove_path(output_root / "training-summary.json")
        self._remove_path(generated_root / "replays")
        self._remove_path(generated_root / "latest-checkpoint.json")
        self._remove_path(generated_root / "latest-evaluation.json")
        self._remove_path(generated_root / "latest-replay.json")
        self._remove_path(generated_root / "checkpoint-index.json")

    def _export_untrained_baseline(self, config: LabConfig) -> None:
        trainer = _BaselineExportTrainer(config, config.training.output_dir)
        baseline_policy = InstinctOnlyPolicy()
        checkpoint_episode = 0
        summary, evaluation_json, _csv_path = trainer.evaluator.evaluate(
            baseline_policy,
            config.training.evaluation_seeds,
            checkpoint_episode=checkpoint_episode,
        )
        representative_replay_path = Path(summary.records[0].replay_path)
        metadata = CheckpointMetadata(
            checkpoint_episode=checkpoint_episode,
            total_training_episodes=0,
            policy_name=baseline_policy.name,
            trainer_type="baseline",
            policy_type="instinct",
            seed=config.training.train_seed,
            success_rate=summary.success_rate,
            average_completion_steps=summary.average_completion_steps,
            timeout_rate=summary.timeout_rate,
            average_sheep_penned=summary.average_sheep_penned,
            average_reward=summary.average_reward,
            environment_config=asdict(config.environment),
            reward_config=asdict(config.rewards),
            evaluation_replay_path=str(representative_replay_path),
        )
        checkpoint_path = trainer.checkpoint_store.write(metadata)
        checkpoint_payload = {
            "checkpoint_episode": checkpoint_episode,
            "checkpoint": checkpoint_path.name,
            "evaluation": evaluation_json.name,
            "replay": str(representative_replay_path),
            "policy_name": baseline_policy.name,
            "trainer_type": "baseline",
            "policy_type": "instinct",
            "policy_mode": baseline_policy.name,
            "replay_mode": "baseline",
            "total_training_episodes": 0,
            "success_rate": summary.success_rate,
            "timeout_rate": summary.timeout_rate,
            "average_completion_steps": summary.average_completion_steps,
            "average_completion_seconds": summary.average_completion_seconds,
            "average_sheep_penned": summary.average_sheep_penned,
            "average_reward": summary.average_reward,
            "average_distance_to_pen": summary.average_distance_to_pen,
            "average_flock_spread": summary.average_flock_spread,
            "environment_config": asdict(config.environment),
            "reward_config": asdict(config.rewards),
            "records": [record.to_dict() for record in summary.records],
        }
        trainer.export_baseline_assets(
            config,
            checkpoint_payload,
            representative_replay_path,
            checkpoint_path,
            summary,
        )
        training_summary_path = Path(config.training.output_dir) / "training-summary.json"
        training_summary_path.write_text(
            json.dumps(
                {
                    "checkpoints": [checkpoint_payload],
                    "trainer_type": "baseline",
                    "policy_type": "instinct",
                    "policy_mode": baseline_policy.name,
                    "replay_mode": "baseline",
                    "total_episodes_trained": 0,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        state_path = Path(config.training.output_dir) / Trainer.STATE_FILENAME
        state_path.write_text(
            json.dumps(
                {
                    "total_episodes_trained": 0,
                    "weights": None,
                    "best_score": None,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def _remove_path(self, path: Path) -> None:
        if not path.exists():
            return
        if path.is_dir():
            shutil.rmtree(path)
            return
        path.unlink()

    def _update_status(self, payload: dict[str, Any]) -> None:
        with self._lock:
            self._status.update(payload)

    def run_live_replay(
        self,
        seed: int,
        *,
        checkpoint_episode: int | None = None,
        policy_mode: PolicyMode | None = None,
        effective_config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        config = LabConfig()
        selection = _resolve_replay_selection(
            config,
            checkpoint_episode=checkpoint_episode,
            policy_mode=policy_mode,
            effective_config=effective_config,
        )
        policy = _load_playable_policy(
            selection.config,
            checkpoint_episode=selection.checkpoint_episode,
            policy_mode=selection.policy_mode,
        )
        result = SheepdogEnvironment(selection.config).run_policy(
            policy,
            seed=seed,
            capture_replay=True,
        )
        payload = _replay_payload(result)
        payload.update(
            {
                "trainer_type": selection.trainer_type,
                "policy_type": selection.policy_type,
                "policy_mode": selection.policy_mode,
                "replay_mode": selection.replay_mode,
                "checkpoint_episode": selection.checkpoint_episode,
                "environment": {
                    "dogs": selection.config.environment.dogs,
                    "sheep": selection.config.environment.sheep,
                    "width": selection.config.environment.width,
                    "height": selection.config.environment.height,
                    "curriculum_stage": selection.config.rewards.instincts.curriculum_stage,
                    "enable_instinct_rewards": (
                        selection.config.rewards.instincts.enable_instinct_rewards
                    ),
                },
            }
        )
        replay_path = Path(config.training.web_export_dir) / "latest-replay.json"
        replay_path.parent.mkdir(parents=True, exist_ok=True)
        replay_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload

    def _run_training(
        self,
        requested_episodes: int,
        fast_mode: bool,
        *,
        enable_instinct_rewards: bool | None = None,
        curriculum_stage: int | None = None,
        debug_reward_breakdown: bool | None = None,
    ) -> None:
        job_config = _build_training_job_config(
            requested_episodes,
            fast_mode,
            enable_instinct_rewards=enable_instinct_rewards,
            curriculum_stage=curriculum_stage,
            debug_reward_breakdown=debug_reward_breakdown,
        )
        total_episodes = max(1, requested_episodes)
        trainer = create_trainer(job_config, job_config.training.output_dir)

        def progress_callback(payload: dict[str, Any]) -> None:
            checkpoint_episode = payload.get("checkpoint_episode")
            summary = payload.get("summary")
            replay_path = payload.get("replay_path")
            latest_seed = None
            if summary and summary.get("records"):
                latest_seed = summary["records"][0].get("seed")
                if replay_path is None:
                    replay_path = summary["records"][0].get("replay_path")
            batch_completed = payload.get("batch_completed_episodes", 0)
            batch_total = payload.get("batch_total_episodes", total_episodes)
            total_trained = payload.get("total_episodes_trained")
            update: dict[str, Any] = {
                "running": payload.get("phase") != "complete",
                "phase": payload.get("phase", "running"),
                "requested_episodes": batch_total,
                "completed_episodes": batch_completed,
                "batch_total_episodes": batch_total,
                "batch_completed_episodes": batch_completed,
                "current_episode": payload.get("current_episode"),
                "checkpoint_episode": checkpoint_episode,
                "best_score": payload.get("best_score"),
                "message": payload.get("message", "Training"),
                "error": None,
            }
            if total_trained is not None:
                update["total_episodes_trained"] = total_trained
            if checkpoint_episode is not None:
                update["latest_checkpoint_episode"] = checkpoint_episode
                update["latest_seed"] = latest_seed
                update["latest_replay_path"] = replay_path
            self._update_status(update)

        try:
            self._update_status(
                {
                    "running": True,
                    "phase": "training",
                    "fast_mode": fast_mode,
                    "trainer_type": job_config.training.trainer_type,
                    "policy_type": job_config.training.policy_type,
                    "enable_instinct_rewards": job_config.rewards.instincts.enable_instinct_rewards,
                    "policy_mode": job_config.policy.policy_mode,
                    "replay_mode": "baseline",
                    "allow_instinct_target_awareness": (
                        job_config.policy.allow_instinct_target_awareness
                    ),
                    "handler_target_enabled": job_config.policy.handler_target_enabled,
                    "debug_reward_breakdown": job_config.rewards.instincts.debug_reward_breakdown,
                    "curriculum_stage": job_config.rewards.instincts.curriculum_stage,
                    "requested_episodes": total_episodes,
                    "completed_episodes": 0,
                    "batch_total_episodes": total_episodes,
                    "batch_completed_episodes": 0,
                    "current_episode": None,
                    "checkpoint_episode": None,
                    "latest_checkpoint_episode": None,
                    "latest_seed": None,
                    "latest_replay_path": None,
                    "message": "Training in progress",
                    "error": None,
                }
            )
            trainer.train(progress_callback=progress_callback)
            with self._lock:
                self._status["running"] = False
                self._status["phase"] = "complete"
                self._status["message"] = "Training complete"
        except (OSError, RuntimeError, ValueError) as exc:  # pragma: no cover
            with self._lock:
                self._status["running"] = False
                self._status["phase"] = "error"
                self._status["message"] = "Training failed"
                self._status["error"] = str(exc)


class TrainingRequestHandler(BaseHTTPRequestHandler):
    """HTTP endpoint for interactive training control."""

    manager = TrainingManager()

    def _json_response(self, payload: Any, status: int = HTTPStatus.OK) -> None:
        body = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict[str, Any]:
        content_length = int(self.headers.get("Content-Length", "0"))
        if content_length <= 0:
            return {}
        raw_body = self.rfile.read(content_length)
        return json.loads(raw_body.decode("utf-8"))

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/api/training/status":
            self._json_response(self.manager.snapshot())
            return
        if self.path == "/api/health":
            self._json_response({"ok": True})
            return
        self._json_response({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802
        if self.path in {"/api/training/clear", "/api/training/reset"}:
            payload, status = self.manager.clear()
            self._json_response(payload, status=status)
            return

        payload = self._read_json()
        if self.path == "/api/replay/run":
            seed = int(payload.get("seed", 11))
            checkpoint_episode = payload.get("checkpoint_episode")
            policy_mode = payload.get("policy_mode")
            effective_config = payload.get("effective_config")
            try:
                replay = self.manager.run_live_replay(
                    seed,
                    checkpoint_episode=(
                        None if checkpoint_episode is None else int(checkpoint_episode)
                    ),
                    policy_mode=policy_mode,
                    effective_config=(
                        effective_config if isinstance(effective_config, dict) else None
                    ),
                )
            except FileNotFoundError as exc:
                self._json_response({"error": str(exc)}, status=HTTPStatus.NOT_FOUND)
                return
            self._json_response(replay)
            return

        if self.path != "/api/training/start":
            self._json_response({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)
            return

        requested_episodes = max(1, int(payload.get("episodes", 1)))
        fast_mode = bool(payload.get("fast_mode", True))
        enable_instinct_rewards = payload.get("enable_instinct_rewards")
        curriculum_stage = payload.get("curriculum_stage")
        debug_reward_breakdown = payload.get("debug_reward_breakdown")
        self._json_response(
            self.manager.start(
                requested_episodes,
                fast_mode,
                enable_instinct_rewards=(
                    None
                    if enable_instinct_rewards is None
                    else bool(enable_instinct_rewards)
                ),
                curriculum_stage=(
                    None if curriculum_stage is None else int(curriculum_stage)
                ),
                debug_reward_breakdown=(
                    None
                    if debug_reward_breakdown is None
                    else bool(debug_reward_breakdown)
                ),
            )
        )

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()


def main() -> None:
    """Run the local training API server."""

    server = ThreadingHTTPServer(("127.0.0.1", 8000), TrainingRequestHandler)
    print("Sheepdog training API listening on http://127.0.0.1:8000")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

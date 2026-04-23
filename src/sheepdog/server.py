"""Local HTTP API for interactive training control."""

from __future__ import annotations

import json
import shutil
import threading
from dataclasses import asdict, replace
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from sheepdog.checkpoints.store import CheckpointMetadata
from sheepdog.config import LabConfig, PolicyConfig, TrainingConfig
from sheepdog.curriculum import apply_training_profile
from sheepdog.environment import SheepdogEnvironment
from sheepdog.policies.base import PolicyMode
from sheepdog.policies.heuristic import HeuristicExpertPolicy, InstinctOnlyPolicy
from sheepdog.policies.random_policy import RandomPolicy
from sheepdog.policies.trainable import PolicyWeights, TrainableLinearPolicy
from sheepdog.training.trainer import Trainer


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

    selected_mode = policy_mode or config.policy.policy_mode
    if selected_mode in {"random_untrained", "random_policy"}:
        return RandomPolicy()
    if selected_mode == "heuristic_expert":
        return HeuristicExpertPolicy()
    if selected_mode == "instinct_only":
        return InstinctOnlyPolicy()

    output_root = Path(config.training.output_dir)
    weights_payload: dict[str, float] | None = None
    if checkpoint_episode is not None:
        checkpoint_path = output_root / "checkpoints" / f"checkpoint-{checkpoint_episode:06d}.json"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint {checkpoint_episode} not found")
        payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        weights_payload = payload.get("policy_weights")
    else:
        state_path = output_root / Trainer.STATE_FILENAME
        if state_path.exists():
            payload = json.loads(state_path.read_text(encoding="utf-8"))
            weights_payload = payload.get("weights")
    return TrainableLinearPolicy(PolicyWeights.from_dict(weights_payload))


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
    """Expose the baseline export flow through one public helper."""

    def export_baseline_assets(
        self,
        config: LabConfig,
        checkpoint_payload: dict[str, Any],
        representative_replay_path: Path,
        checkpoint_path: Path,
        baseline_policy: TrainableLinearPolicy,
        summary: Any,
    ) -> None:
        self._export_web_assets(
            Path(config.training.web_export_dir),
            [checkpoint_payload],
            summary,
            representative_replay_path,
            checkpoint_path,
        )
        self._export_training_summary([checkpoint_payload], baseline_policy.weights, 0)
        self._save_state(
            0,
            baseline_policy.weights,
            self._evaluate_candidate(baseline_policy).score,
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
        return {
            "running": False,
            "fast_mode": True,
            "enable_instinct_rewards": instincts.enable_instinct_rewards,
            "policy_mode": config.policy.policy_mode,
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
        baseline_policy = TrainableLinearPolicy(PolicyWeights())
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
            seed=config.training.train_seed,
            success_rate=summary.success_rate,
            average_completion_steps=summary.average_completion_steps,
            timeout_rate=summary.timeout_rate,
            average_sheep_penned=summary.average_sheep_penned,
            average_reward=summary.average_reward,
            environment_config=asdict(config.environment),
            reward_config=asdict(config.rewards),
            policy_weights=asdict(baseline_policy.weights),
            evaluation_replay_path=str(representative_replay_path),
        )
        checkpoint_path = trainer.checkpoint_store.write(metadata)
        checkpoint_payload = {
            "checkpoint_episode": checkpoint_episode,
            "checkpoint": checkpoint_path.name,
            "evaluation": evaluation_json.name,
            "replay": str(representative_replay_path),
            "success_rate": summary.success_rate,
            "timeout_rate": summary.timeout_rate,
            "average_completion_steps": summary.average_completion_steps,
            "average_completion_seconds": summary.average_completion_seconds,
            "average_sheep_penned": summary.average_sheep_penned,
            "average_reward": summary.average_reward,
            "records": [record.to_dict() for record in summary.records],
        }
        trainer.export_baseline_assets(
            config,
            checkpoint_payload,
            representative_replay_path,
            checkpoint_path,
            baseline_policy,
            summary,
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
    ) -> dict[str, Any]:
        config = LabConfig()
        effective_policy = policy_mode or config.policy.policy_mode
        effective_config = replace(config, policy=PolicyConfig(policy_mode=effective_policy))
        policy = _load_playable_policy(
            effective_config,
            checkpoint_episode=checkpoint_episode,
            policy_mode=effective_policy,
        )
        result = SheepdogEnvironment(effective_config).run_policy(
            policy,
            seed=seed,
            capture_replay=True,
        )
        payload = _replay_payload(result)
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
        trainer = Trainer(job_config, job_config.training.output_dir)

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
                    "enable_instinct_rewards": job_config.rewards.instincts.enable_instinct_rewards,
                    "policy_mode": job_config.policy.policy_mode,
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
            try:
                replay = self.manager.run_live_replay(
                    seed,
                    checkpoint_episode=(
                        None if checkpoint_episode is None else int(checkpoint_episode)
                    ),
                    policy_mode=policy_mode,
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

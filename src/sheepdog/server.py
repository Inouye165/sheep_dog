"""Local HTTP API for interactive training control."""

from __future__ import annotations

import json
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from sheepdog.config import LabConfig, TrainingConfig
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


class TrainingManager:
    """Track one background training job at a time."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._status: dict[str, Any] = self._initial_status()

    def _initial_status(self) -> dict[str, Any]:
        return {
            "running": False,
            "fast_mode": True,
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

    def start(self, requested_episodes: int, fast_mode: bool) -> dict[str, Any]:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return dict(self._status)

            self._status = self._initial_status()
            self._status.update(
                {
                    "running": True,
                    "fast_mode": fast_mode,
                    "requested_episodes": requested_episodes,
                    "message": "Queued training job",
                }
            )
            self._thread = threading.Thread(
                target=self._run_training,
                args=(requested_episodes, fast_mode),
                daemon=True,
            )
            self._thread.start()
            return dict(self._status)

    def _update_status(self, payload: dict[str, Any]) -> None:
        with self._lock:
            self._status.update(payload)

    def _run_training(self, requested_episodes: int, fast_mode: bool) -> None:
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
            mutation_scale=config.training.mutation_scale,
            output_dir=config.training.output_dir,
            web_export_dir=config.training.web_export_dir,
        )
        job_config = LabConfig(
            environment=config.environment,
            rewards=config.rewards,
            training=training_config,
        )
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
        except Exception as exc:  # pragma: no cover - surfaced through UI
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
        if self.path != "/api/training/start":
            self._json_response({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)
            return

        payload = self._read_json()
        requested_episodes = int(payload.get("episodes", 1))
        fast_mode = bool(payload.get("fast_mode", True))
        if requested_episodes < 1:
            requested_episodes = 1
        self._json_response(self.manager.start(requested_episodes, fast_mode))

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        return


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

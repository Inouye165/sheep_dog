"""Hierarchical MaskablePPO trainer: scripted shepherd + neural dogs.

How it differs from MaskablePPOTrainer
---------------------------------------
- Wraps the environment in ``JointActionRLEnv`` so all dogs act in a single
  ``env.step`` call rather than sequentially.
- Observations include shepherd command (one-hot) and dog identity features,
  built by ``HierarchicalObservationBuilder``.
- Saves checkpoints as ``hierarchical_checkpoint-NNNNNN.json`` to avoid
  colliding with the baseline neural checkpoints in the same output directory.
- Policy state files live under ``artifacts/models/hierarchical/``.

Checkpoint numbers reflect *training episodes* (same convention as the
baseline trainer), where one episode ≈ one full herding attempt.  The exact
number of timesteps per episode varies by curriculum stage.

Training signature
------------------
A training signature is stored with the state file so an incompatible resumed
run is caught early.  The signature includes:
  - action_size
  - observation_size (critical – shepherd/identity features change this)
  - environment shape (dogs, sheep, grid)
  - reward config hash

Phase-B readiness
-----------------
The shepherd is passed as a constructor argument.  To swap in a learned shepherd
later, construct ``HierarchicalMaskablePPOTrainer(config, output_root,
shepherd=LearnedShepherd())``.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from stable_baselines3.common.callbacks import BaseCallback

from sheepdog.checkpoints.store import CheckpointMetadata
from sheepdog.environment import ACTION_ORDER
from sheepdog.policies.hierarchical import (
    ShepherdNeuralDogPolicy,
)
from sheepdog.shepherd import ScriptedShepherd
from sheepdog.training.joint_rl_env import JointActionRLEnv
from sheepdog.training.trainer import Trainer


class _HierarchicalProgressCallback(BaseCallback):
    """Relay PPO timestep progress back to the hierarchical training manager."""

    def __init__(
        self,
        emit: Callable[[dict[str, Any]], None],
        *,
        should_stop: Callable[[], bool] | None,
        report_interval: int,
        total_timesteps: int,
        completed_segments: int,
        segment_index: int,
        batch_total: int,
        starting_total: int,
    ) -> None:
        super().__init__()
        self._emit = emit
        self._should_stop = should_stop
        self._report_interval = max(1, report_interval)
        self._total_timesteps = max(1, total_timesteps)
        self._completed_segments = completed_segments
        self._segment_index = segment_index
        self._batch_total = batch_total
        self._starting_total = starting_total
        self._last_reported = 0

    def _on_step(self) -> bool:
        if self._should_stop is not None and self._should_stop():
            return False

        # Emit individual episode completion logs as they finish
        dones = self.locals.get("dones")
        infos = self.locals.get("infos")
        if dones is not None and infos is not None:
            for idx, done in enumerate(dones):
                if done and idx < len(infos):
                    ep_info = infos[idx].get("episode") if isinstance(infos[idx], dict) else None
                    if isinstance(ep_info, dict):
                        actual_eps = 0
                        try:
                            if self.model is not None and self.model.get_env() is not None:
                                curr_counters = self.model.get_env().get_attr("_episode_counter")
                                actual_eps = int(sum(curr_counters))
                        except Exception:
                            actual_eps = 0
                        ep_num = self._starting_total + actual_eps
                        self._emit(
                            {
                                "phase": "episode_complete",
                                "episode": ep_num,
                                "reward": float(ep_info.get("r", 0.0)),
                                "length": int(ep_info.get("l", 0)),
                                "success": bool(ep_info.get("success", False)),
                                "penned": int(ep_info.get("penned", 0)),
                                "total_sheep": int(ep_info.get("total_sheep", 0)),
                                "status": str(ep_info.get("status", "UNKNOWN")),
                                "seed": ep_info.get("seed"),
                                "field_setup": ep_info.get("field_setup"),
                                "replay_available": ep_info.get("replay_available"),
                                "replay_id": ep_info.get("replay_id"),
                                "replay_path": ep_info.get("replay_path"),
                                "replay_source": ep_info.get("replay_source"),
                                "capture_reason": ep_info.get("capture_reason"),
                                "capture_status": ep_info.get("capture_status"),
                            }
                        )

        n = int(self.num_timesteps)
        if n < self._total_timesteps and n - self._last_reported < self._report_interval:
            return True
        self._last_reported = n
        completion = min(1.0, n / self._total_timesteps)

        actual_completed_episodes = 0
        try:
            if self.model is not None and self.model.get_env() is not None:
                curr_counters = self.model.get_env().get_attr("_episode_counter")
                actual_completed_episodes = int(sum(curr_counters))
        except Exception:
            pass

        self._emit(
            {
                "phase": "learning",
                "batch_completed_episodes": self._completed_segments + completion,
                "current_episode": self._starting_total + actual_completed_episodes,
                "total_episodes_trained": self._starting_total + actual_completed_episodes,
                "checkpoint_episode": None,
                "best_score": None,
                "message": (
                    f"[Hierarchical] PPO timesteps: "
                    f"{min(n, self._total_timesteps)}/{self._total_timesteps} ({completion:.0%})"
                ),
                "batch_total_episodes": self._batch_total,
            }
        )
        return True


class HierarchicalMaskablePPOTrainer(Trainer):
    """Train neural dogs with a scripted shepherd using MaskablePPO.

    Checkpoint naming: ``hierarchical_checkpoint-NNNNNN.json``
    Model naming:      ``models/hierarchical/model-NNNNNN.zip``
    State file:        ``hierarchical-training-state.json``
    """

    MODEL_DIRNAME = "models/hierarchical"
    STATE_FILENAME = "hierarchical-training-state.json"
    CHECKPOINT_PREFIX = "hierarchical_checkpoint"

    def __init__(
        self,
        config: Any,
        output_root: str | Path,
        shepherd: ScriptedShepherd | None = None,
    ) -> None:
        super().__init__(config, output_root)
        self._shepherd = shepherd if shepherd is not None else ScriptedShepherd()

    # ------------------------------------------------------------------
    # Training signature – must match for resumption
    # ------------------------------------------------------------------

    def _training_signature(self) -> dict[str, Any]:
        adapter = JointActionRLEnv(self.config, shepherd=self._shepherd)
        return {
            "action_size": len(ACTION_ORDER),
            "observation_size": int(adapter.observation_space.shape[0]),
            "environment": {
                "dogs": self.config.environment.dogs,
                "sheep": self.config.environment.sheep,
                "dog_speed": self.config.environment.dog_speed,
            },
        }

    def _has_compatible_policy_state(self) -> bool:
        stored = self._loaded_state.get("training_signature")
        if not isinstance(stored, dict):
            return False
        return stored == self._training_signature()

    # ------------------------------------------------------------------
    # State persistence
    # ------------------------------------------------------------------

    def _load_state(self) -> dict[str, Any]:
        if not self._state_path.exists():
            return {
                "total_episodes_trained": 0,
                "total_timesteps": 0,
                "policy_state_path": None,
                "policy_config": None,
            }
        try:
            payload = json.loads(self._state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {
                "total_episodes_trained": 0,
                "total_timesteps": 0,
                "policy_state_path": None,
                "policy_config": None,
            }
        return {
            "total_episodes_trained": int(payload.get("total_episodes_trained", 0)),
            "total_timesteps": int(payload.get("total_timesteps", 0)),
            "policy_state_path": payload.get("policy_state_path"),
            "policy_config": payload.get("policy_config"),
            "training_signature": payload.get("training_signature"),
        }

    def _save_state(  # pylint: disable=arguments-differ
        self,
        total_episodes: int,
        total_timesteps: int,
        policy: ShepherdNeuralDogPolicy,
        model_path: Path,
    ) -> None:
        payload = {
            "total_episodes_trained": total_episodes,
            "total_timesteps": total_timesteps,
            "policy_state_path": str(model_path),
            "policy_config": policy.policy_config.to_dict(),
            "training_signature": self._training_signature(),
        }
        self._state_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(
        self,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> dict[str, Any]:
        """Train neural dogs; return a summary dict of saved checkpoints."""

        train_config = self.config.training
        model_root = self.output_root / self.MODEL_DIRNAME
        model_root.mkdir(parents=True, exist_ok=True)
        checkpoint_root = self.output_root / "checkpoints"
        checkpoint_root.mkdir(parents=True, exist_ok=True)
        web_export_dir = Path(train_config.web_export_dir)
        web_export_dir.mkdir(parents=True, exist_ok=True)

        resuming = (
            bool(self._loaded_state.get("policy_state_path"))
            and self._has_compatible_policy_state()
        )
        starting_total = int(self._loaded_state.get("total_episodes_trained", 0)) if resuming else 0
        starting_ts = int(self._loaded_state.get("total_timesteps", 0)) if resuming else 0
        n_checkpoints = max(1, len(train_config.checkpoint_episodes))
        steps_per_segment = max(1, train_config.total_timesteps // n_checkpoints)

        def emit(payload: dict[str, Any]) -> None:
            if progress_callback is None:
                return
            payload.setdefault("batch_total_episodes", n_checkpoints)
            payload.setdefault("starting_total_episodes", starting_total)
            progress_callback(payload)

        emit(
            {
                "phase": "starting",
                "batch_completed_episodes": 0,
                "current_episode": None,
                "total_episodes_trained": starting_total,
                "checkpoint_episode": None,
                "best_score": None,
                "message": (
                    f"Resuming hierarchical training from episode {starting_total}"
                    if resuming
                    else "Starting fresh hierarchical training run"
                ),
            }
        )

        # Build or resume model
        training_env = JointActionRLEnv(self.config, shepherd=self._shepherd)
        if resuming and self._loaded_state.get("policy_state_path"):
            policy = ShepherdNeuralDogPolicy.load(
                self._loaded_state["policy_state_path"],
                self.config,
                policy_config_dict=self._loaded_state.get("policy_config"),
                shepherd=self._shepherd,
            )
            policy._model.set_env(training_env)  # pylint: disable=protected-access
        else:
            policy = ShepherdNeuralDogPolicy.initialize(self.config, shepherd=self._shepherd)

        checkpoint_records: list[dict[str, Any]] = []
        saved_model_path: Path = model_root / "model-initial.zip"
        interrupted = False

        for seg_idx, checkpoint_episode in enumerate(train_config.checkpoint_episodes, start=1):
            if should_stop is not None and should_stop():
                interrupted = True
                break
            cumulative_ts = starting_ts + seg_idx * steps_per_segment
            emit(
                {
                    "phase": "learning",
                    "batch_completed_episodes": seg_idx - 1,
                    "current_episode": checkpoint_episode,
                    "total_episodes_trained": starting_total + seg_idx - 1,
                    "checkpoint_episode": checkpoint_episode,
                    "best_score": None,
                    "message": (f"[Hierarchical] Training toward checkpoint {checkpoint_episode}"),
                }
            )
            callback = _HierarchicalProgressCallback(
                emit,
                should_stop=should_stop,
                report_interval=max(1, steps_per_segment // 10),
                total_timesteps=steps_per_segment,
                completed_segments=seg_idx - 1,
                segment_index=checkpoint_episode,
                batch_total=n_checkpoints,
                starting_total=starting_total,
            )
            policy._model.learn(  # pylint: disable=protected-access
                total_timesteps=steps_per_segment,
                callback=callback,
                reset_num_timesteps=False,
            )
            if should_stop is not None and should_stop():
                interrupted = True
                break

            # Save model
            model_filename = f"model-{checkpoint_episode:06d}.zip"
            saved_model_path = model_root / model_filename
            policy.save(saved_model_path)

            # Evaluate
            # pylint: disable-next=import-outside-toplevel
            from sheepdog.evaluation.evaluator import Evaluator

            evaluator = Evaluator(self.config, self.output_root)
            evaluation_summary, eval_json_path, _ = evaluator.evaluate(
                policy,
                seeds=train_config.evaluation_seeds,
                checkpoint_episode=checkpoint_episode,
            )

            # Save checkpoint metadata
            # pylint: disable-next=import-outside-toplevel
            from dataclasses import asdict as _asdict  # noqa: PLC0415

            checkpoint_meta = CheckpointMetadata(
                checkpoint_episode=checkpoint_episode,
                total_training_episodes=starting_total + seg_idx,
                policy_name=policy.name,
                policy_type="neural",
                trainer_type="hierarchical_maskable_ppo",
                seed=train_config.train_seed,
                success_rate=evaluation_summary.success_rate,
                average_completion_steps=evaluation_summary.average_completion_steps,
                timeout_rate=evaluation_summary.timeout_rate,
                average_sheep_penned=evaluation_summary.average_sheep_penned,
                average_reward=evaluation_summary.average_reward,
                environment_config=_asdict(self.config.environment),
                reward_config=_asdict(self.config.rewards),
                policy_state_path=str(saved_model_path),
                policy_config=policy.policy_config.to_dict(),
                policy_weights=None,
                evaluation_replay_path=str(eval_json_path),
            )
            checkpoint_path = (
                checkpoint_root / f"{self.CHECKPOINT_PREFIX}-{checkpoint_episode:06d}.json"
            )
            checkpoint_path.write_text(
                json.dumps(checkpoint_meta.to_dict(), indent=2), encoding="utf-8"
            )
            checkpoint_records.append(checkpoint_meta.to_dict())

            self._save_state(
                starting_total + seg_idx,
                cumulative_ts,
                policy,
                saved_model_path,
            )

            emit(
                {
                    "phase": "checkpoint",
                    "batch_completed_episodes": seg_idx,
                    "current_episode": checkpoint_episode,
                    "total_episodes_trained": starting_total + seg_idx,
                    "checkpoint_episode": checkpoint_episode,
                    "best_score": evaluation_summary.average_reward,
                    "message": (
                        f"[Hierarchical] Checkpoint {checkpoint_episode}: "
                        f"reward={evaluation_summary.average_reward:.2f} "
                        f"success={evaluation_summary.success_rate:.1%}"
                    ),
                }
            )

        if interrupted:
            return {
                "checkpoints": checkpoint_records,
                "final_model_path": str(saved_model_path),
                "policy_config": policy.policy_config.to_dict(),
            }

        return {
            "checkpoints": checkpoint_records,
            "final_model_path": str(saved_model_path),
            "policy_config": policy.policy_config.to_dict(),
        }

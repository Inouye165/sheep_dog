"""Local HTTP API for interactive training control."""

# pylint: disable=too-many-lines
from __future__ import annotations

import datetime
import json
import shutil
import threading
import traceback
from dataclasses import asdict, dataclass, replace
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlsplit

from sheepdog.atomic_io import atomic_write_json, atomic_write_text
from sheepdog.checkpoints.store import CheckpointMetadata
from sheepdog.config import (
    EnvironmentConfig,
    InstinctRewardConfig,
    LabConfig,
    RewardConfig,
    TrainingConfig,
)
from sheepdog.curriculum import apply_training_profile
from sheepdog.curriculum import available_stages
from sheepdog.curriculum import validate_curriculum_stage
from sheepdog.environment import ACTION_ORDER, SheepdogEnvironment
from sheepdog.evaluation.scenario_evaluator import (
    config_for_scenario,
    evaluate_scenario,
    refresh_scenario_exports,
    resolve_checkpoint_episode,
)
from sheepdog.evaluation.scenarios import ScenarioStore, scenario_from_snapshot
from sheepdog.policies.base import Policy, PolicyMode
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
    total_training_episodes: int = 0


class _EarlyPromotionSignal(Exception):
    """Internal control-flow signal for immediate stage promotion."""

    def __init__(
        self,
        *,
        checkpoint_episode: int,
        best_success: float,
        qualified_streak: int,
        seed_gate_hits: int,
        full_success_hits: int,
    ) -> None:
        super().__init__("early-promotion")
        self.checkpoint_episode = checkpoint_episode
        self.best_success = best_success
        self.qualified_streak = qualified_streak
        self.seed_gate_hits = seed_gate_hits
        self.full_success_hits = full_success_hits


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
        resolved_mode = str(
            checkpoint_payload.get("policy_name") or requested_mode or "instinct_only"
        )
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
            total_training_episodes=int(checkpoint_payload.get("total_training_episodes", 0)),
        )

    latest_checkpoint_payload = _load_latest_checkpoint_payload(output_root)
    if requested_mode in {None, "trained_policy", "neural_policy"} and latest_checkpoint_payload:
        latest_total = int(latest_checkpoint_payload.get("total_training_episodes", 0))
        latest_mode = str(
            latest_checkpoint_payload.get("policy_name") or requested_mode or "instinct_only"
        )
        if latest_total > 0 and (requested_mode is None or requested_mode == latest_mode):
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
            latest_episode = latest_checkpoint_payload.get("checkpoint_episode")
            return ReplaySelection(
                config=replay_config,
                checkpoint_episode=int(latest_episode) if latest_episode is not None else None,
                trainer_type=trainer_type,
                policy_type=policy_type,
                policy_mode=latest_mode,
                replay_mode=replay_mode,
                total_training_episodes=latest_total,
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


STAGE_HISTORY_FILENAME = "stage-history.json"
TRAINING_SETTINGS_FILENAME = "training-settings.json"
HYPERPARAMS_FILENAME = "user-hyperparams.json"
AUTO_PROMOTE_SUCCESS_THRESHOLD = 0.9
AUTO_PROMOTE_MAX_TIMEOUT_RATE = 0.1
AUTO_PROMOTE_REWARD_TOLERANCE_RATIO = 0.05
AUTO_PROMOTE_MIN_QUALIFIED_STREAK = 3
AUTO_PROMOTE_MIN_SEED_GATE_HITS = 3
AUTO_PROMOTE_MIN_FULL_SUCCESS_HITS = 2
AUTO_PROMOTE_FULL_SUCCESS_RATE_THRESHOLD = 0.999
MAX_STAGE_MASTERY_QUALIFIED_STREAK = 5
MAX_STAGE_MASTERY_FULL_SUCCESS_HITS = 3
PLATEAU_STOP_MIN_CHECKPOINTS = 20
PLATEAU_STOP_NO_IMPROVEMENT_STREAK = 20


def _auto_promote_gate_defaults() -> dict[str, Any]:
    """Return the default auto-promotion diagnostics payload."""

    return {
        "decision": "pending",
        "reason": "Awaiting checkpoint evaluation",
        "seed_count": 0,
        "success_count": 0,
        "best_success": 0.0,
        "best_reward": None,
        "seed_gate_ok": False,
        "success_rate_ok": False,
        "timeout_ok": False,
        "reward_close_ok": False,
        "qualified_streak": 0,
        "min_qualified_streak": AUTO_PROMOTE_MIN_QUALIFIED_STREAK,
        "seed_gate_hits": 0,
        "min_seed_gate_hits": AUTO_PROMOTE_MIN_SEED_GATE_HITS,
        "seed_gate_target_met": False,
        "full_success_hits": 0,
        "min_full_success_hits": AUTO_PROMOTE_MIN_FULL_SUCCESS_HITS,
        "full_success_target_met": False,
        "full_success_rate_threshold": AUTO_PROMOTE_FULL_SUCCESS_RATE_THRESHOLD,
        "success_threshold": AUTO_PROMOTE_SUCCESS_THRESHOLD,
        "max_timeout_rate": AUTO_PROMOTE_MAX_TIMEOUT_RATE,
        "reward_tolerance_ratio": AUTO_PROMOTE_REWARD_TOLERANCE_RATIO,
    }


def _seed_success_gate(success_count: int, seed_count: int) -> bool:
    """Return whether a checkpoint satisfies the per-seed promotion gate."""

    if seed_count <= 0:
        return False
    if seed_count >= 5:
        return success_count >= 3
    if seed_count == 4:
        return success_count >= 3
    if seed_count == 3:
        return success_count >= 2
    return success_count >= seed_count


def _reward_within_tolerance(reward: float, best_reward: float) -> bool:
    """Return whether *reward* is within the accepted gap from *best_reward*."""

    if best_reward == float("-inf"):
        return True
    tolerance = max(15.0, abs(best_reward) * AUTO_PROMOTE_REWARD_TOLERANCE_RATIO)
    return reward >= (best_reward - tolerance)


def _curriculum_stage_metadata() -> tuple[list[int], int]:
    """Return available curriculum stages and the highest stage number."""

    stages = [int(stage) for stage in available_stages()]
    max_stage = max(stages) if stages else 0
    return stages, max_stage


def _default_user_hyperparams() -> dict[str, Any]:
    """Return the canonical default user-adjustable hyperparameter values."""
    config = LabConfig()
    return {
        "environment": {
            "sheep_personality_strength": config.environment.sheep_personality_strength,
            "sheep_speed": config.environment.sheep_speed,
            "sheep_vision": config.environment.sheep_vision,
            "flock_radius": config.environment.flock_radius,
            "dog_speed": config.environment.dog_speed,
            "dog_sprint_multiplier": config.environment.dog_sprint_multiplier,
            "dog_vision": config.environment.dog_vision,
        },
        "training": {
            "learning_rate": config.training.learning_rate,
            "learning_rate_final": config.training.learning_rate_final,
            "entropy_coef": config.training.entropy_coef,
            "gamma": config.training.gamma,
            "gae_lambda": config.training.gae_lambda,
            "clip_range": config.training.clip_range,
            "rollout_steps": config.training.rollout_steps,
            "batch_size": config.training.batch_size,
            "value_coef": config.training.value_coef,
        },
        "rewards": {
            "time_penalty": config.rewards.time_penalty,
            "progress_scale": config.rewards.progress_scale,
            "sheep_penned_reward": config.rewards.sheep_penned_reward,
            "wait_penalty": config.rewards.wait_penalty,
            "no_progress_penalty": config.rewards.no_progress_penalty,
            "terminal_success_reward": config.rewards.terminal_success_reward,
            "terminal_failure_penalty": config.rewards.terminal_failure_penalty,
            "flock_cohesion_scale": config.rewards.flock_cohesion_scale,
            "scatter_penalty_scale": config.rewards.scatter_penalty_scale,
            "sprint_cost_scale": config.rewards.sprint_cost_scale,
        },
    }


def _read_user_hyperparams(output_root: Path) -> dict[str, Any]:
    """Load persisted user hyperparameters, filling missing keys from defaults."""
    defaults = _default_user_hyperparams()
    path = output_root / HYPERPARAMS_FILENAME
    try:
        saved = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return defaults
    # Merge: saved values override defaults, but missing keys fall back.
    merged: dict[str, Any] = {}
    for section, section_defaults in defaults.items():
        saved_section = saved.get(section, {})
        merged[section] = {k: saved_section.get(k, v) for k, v in section_defaults.items()}
    return merged


def _write_user_hyperparams(output_root: Path, payload: dict[str, Any]) -> dict[str, Any]:
    """Persist user hyperparameters and return the merged result."""
    defaults = _default_user_hyperparams()
    merged: dict[str, Any] = {}
    for section, section_defaults in defaults.items():
        incoming = payload.get(section, {})
        merged[section] = {}
        for key, default_val in section_defaults.items():
            raw = incoming.get(key, default_val)
            # Coerce to the same type as the default.
            try:
                merged[section][key] = type(default_val)(raw)
            except (TypeError, ValueError):
                merged[section][key] = default_val
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / HYPERPARAMS_FILENAME).write_text(json.dumps(merged, indent=2), encoding="utf-8")
    return merged


def _apply_user_hyperparams(base_config: LabConfig, user_params: dict[str, Any]) -> LabConfig:
    """Return a LabConfig with user-adjustable fields overridden from *user_params*."""
    env_overrides = {
        k: v
        for k, v in user_params.get("environment", {}).items()
        if hasattr(base_config.environment, k)
    }
    training_overrides = {
        k: v for k, v in user_params.get("training", {}).items() if hasattr(base_config.training, k)
    }
    reward_overrides = {
        k: v
        for k, v in user_params.get("rewards", {}).items()
        if hasattr(base_config.rewards, k) and k != "instincts"
    }
    env = (
        replace(base_config.environment, **env_overrides)
        if env_overrides
        else base_config.environment
    )
    training = (
        replace(base_config.training, **training_overrides)
        if training_overrides
        else base_config.training
    )
    rewards = (
        replace(base_config.rewards, **reward_overrides)
        if reward_overrides
        else base_config.rewards
    )
    return replace(base_config, environment=env, training=training, rewards=rewards)


def _apply_environment_overrides(
    config: LabConfig, environment_overrides: dict[str, Any] | None
) -> LabConfig:
    """Merge per-run environment overrides without persisting to user hyperparams."""
    if not isinstance(environment_overrides, dict):
        return config
    env_overrides: dict[str, Any] = {}
    for key, value in environment_overrides.items():
        if not hasattr(config.environment, key):
            continue
        default_val = getattr(config.environment, key)
        try:
            env_overrides[key] = type(default_val)(value)
        except (TypeError, ValueError):
            continue
    if not env_overrides:
        return config
    return replace(config, environment=replace(config.environment, **env_overrides))


def _read_persisted_settings(output_root: Path) -> dict[str, Any]:
    """Best-effort read of persisted curriculum settings for initial status.

    Reads the explicit settings file only. Stage history is cumulative and is
    not a reliable source for the currently selected curriculum stage.
    """
    settings_path = output_root / TRAINING_SETTINGS_FILENAME
    result: dict[str, Any] = {}
    try:
        payload = json.loads(settings_path.read_text(encoding="utf-8"))
        if "curriculum_stage" in payload:
            result["curriculum_stage"] = int(payload["curriculum_stage"])
        if "enable_instinct_rewards" in payload:
            result["enable_instinct_rewards"] = bool(payload["enable_instinct_rewards"])
        if "debug_reward_breakdown" in payload:
            result["debug_reward_breakdown"] = bool(payload["debug_reward_breakdown"])
        if "auto_promote" in payload:
            result["auto_promote"] = bool(payload["auto_promote"])
    except (OSError, json.JSONDecodeError, ValueError, TypeError):
        pass
    return result


def _read_stage_history(output_root: Path) -> dict[str, int]:
    """Read cumulative per-stage episode counts; returns empty dict on any error."""
    path = output_root / STAGE_HISTORY_FILENAME
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return {str(k): int(v) for k, v in raw.items() if isinstance(v, (int, float))}
    except (OSError, json.JSONDecodeError, ValueError):
        return {}


def _update_stage_history(output_root: Path, stage: int, episodes_added: int) -> dict[str, int]:
    """Append *episodes_added* to the given *stage* bucket and persist."""
    history = _read_stage_history(output_root)
    key = str(stage)
    history[key] = history.get(key, 0) + episodes_added
    path = output_root / STAGE_HISTORY_FILENAME
    path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    return history


def _build_training_job_config(
    requested_episodes: int,
    fast_mode: bool,
    *,
    enable_instinct_rewards: bool | None = None,
    curriculum_stage: int | None = None,
    debug_reward_breakdown: bool | None = None,
) -> LabConfig:
    """Build the effective training configuration for one requested job."""

    output_root = Path(LabConfig().training.output_dir)
    user_params = _read_user_hyperparams(output_root)
    config = _apply_user_hyperparams(LabConfig(), user_params)
    total_episodes = max(1, requested_episodes)
    training_episodes = max(0, total_episodes - 1)
    checkpoint_episodes = tuple(range(total_episodes))
    # Scale total_timesteps with the number of checkpoints so each segment
    # receives a meaningful training budget.  Use at least the config default.
    # Late curriculum stages need more rollouts to avoid timeout-only drift.
    stage_for_budget = int(
        curriculum_stage
        if curriculum_stage is not None
        else config.rewards.instincts.curriculum_stage
    )
    if fast_mode:
        if stage_for_budget >= 25:
            steps_per_episode = 12_000
        elif stage_for_budget >= 21:
            steps_per_episode = 8_000
        else:
            steps_per_episode = 4_000
    else:
        steps_per_episode = 25_000
    # Fast mode keeps latency low while using a small fixed seed set to reduce
    # overfitting to a single deterministic scenario.
    evaluation_seeds = (11, 23, 37) if fast_mode else config.training.evaluation_seeds
    total_timesteps = max(config.training.total_timesteps, total_episodes * steps_per_episode)
    training_config = TrainingConfig(
        trainer_type="maskable_ppo",
        policy_type="neural",
        episodes=training_episodes,
        checkpoint_episodes=checkpoint_episodes,
        evaluation_seeds=evaluation_seeds,
        train_seed=config.training.train_seed,
        evaluation_seed=config.training.evaluation_seed,
        candidate_evaluation_seeds=config.training.candidate_evaluation_seeds,
        candidate_pool_size=config.training.candidate_pool_size,
        mutation_scale=config.training.mutation_scale,
        total_timesteps=total_timesteps,
        output_dir=config.training.output_dir,
        web_export_dir=config.training.web_export_dir,
        learning_rate=config.training.learning_rate,
        learning_rate_final=config.training.learning_rate_final,
        entropy_coef=config.training.entropy_coef,
        gamma=config.training.gamma,
        gae_lambda=config.training.gae_lambda,
        clip_range=config.training.clip_range,
        rollout_steps=config.training.rollout_steps,
        batch_size=config.training.batch_size,
        value_coef=config.training.value_coef,
    )
    job_config = replace(
        config,
        training=training_config,
        policy=replace(config.policy, policy_mode="neural_policy"),
    )
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
) -> Policy:
    """Return a runnable policy for replay requests."""

    return load_playable_policy(
        config,
        checkpoint_episode=checkpoint_episode,
        policy_mode=policy_mode,
    )


def _parse_scenario_action_path(path: str) -> tuple[str, str] | None:
    """Return ``(scenario_id, action)`` for ``/api/scenarios/{id}/evaluate|replay``."""

    prefix = "/api/scenarios/"
    if not path.startswith(prefix):
        return None
    rest = path[len(prefix) :].split("?")[0].strip("/")
    parts = rest.split("/")
    if len(parts) != 2:
        return None
    scenario_id, action = parts
    if action not in {"evaluate", "replay"}:
        return None
    return scenario_id, action


def _replay_payload(result: Any) -> dict[str, Any]:
    """Convert an environment run result to the web replay schema."""

    return {
        "seed": result.seed,
        "policy_name": result.policy_name,
        "final_snapshot": result.final_snapshot.to_dict(),
"stats": asdict(result.stats, dict_factory=dict),
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
        """Export web assets for the current checkpoint to the web export directory."""
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
        self._reconcile_web_exports()
        self._status: dict[str, Any] = self._initial_status()

    @staticmethod
    def _reconcile_web_exports() -> None:
        """Catch the UI's web export up to the trainer state on startup.

        An interrupted run (e.g. an overnight reboot) can leave the web
        ``checkpoint-index.json`` behind the trainer's ``training-summary.json``,
        which the UI shows as lost progress.  Rebuild the web assets from the
        summary when they are out of sync.  Startup must never fail on this.
        """
        try:
            config = LabConfig()
            trainer = Trainer(config, config.training.output_dir)
            trainer.reconcile_web_exports(config.training.web_export_dir)
        except Exception:  # noqa: BLE001 - reconciliation must not block startup
            pass

    def _initial_status(self) -> dict[str, Any]:
        config = LabConfig()
        instincts = config.rewards.instincts
        available_curriculum_stages, max_curriculum_stage = _curriculum_stage_metadata()
        trainer_type, policy_type, replay_mode = _policy_metadata(config.policy.policy_mode)
        output_root = Path(config.training.output_dir)
        stage_history = _read_stage_history(output_root)
        persisted = _read_persisted_settings(output_root)
        return {
            "running": False,
            "fast_mode": True,
            "trainer_type": trainer_type,
            "policy_type": policy_type,
            "enable_instinct_rewards": persisted.get(
                "enable_instinct_rewards", instincts.enable_instinct_rewards
            ),
            "policy_mode": config.policy.policy_mode,
            "replay_mode": replay_mode,
            "allow_instinct_target_awareness": config.policy.allow_instinct_target_awareness,
            "handler_target_enabled": config.policy.handler_target_enabled,
            "debug_reward_breakdown": persisted.get(
                "debug_reward_breakdown", instincts.debug_reward_breakdown
            ),
            "auto_promote": persisted.get("auto_promote", True),
            "auto_promote_threshold": AUTO_PROMOTE_SUCCESS_THRESHOLD,
            "auto_promote_stages_completed": 0,
            "auto_promote_gate": _auto_promote_gate_defaults(),
            "available_curriculum_stages": available_curriculum_stages,
            "max_curriculum_stage": max_curriculum_stage,
            "curriculum_stage": persisted.get("curriculum_stage", instincts.curriculum_stage),
            "requested_episodes": 0,
            "completed_episodes": 0,
            "batch_total_episodes": 0,
            "batch_completed_episodes": 0,
            "total_episodes_trained": _read_persisted_total(),
            "stage_history": stage_history,
            "grand_total_episodes": sum(stage_history.values()),
            "current_episode": None,
            "checkpoint_episode": None,
            "latest_checkpoint_episode": None,
            "latest_seed": None,
            "latest_replay_path": None,
            "best_score": None,
            "latest_success_rate": None,
            "latest_avg_sheep_penned": None,
            "latest_avg_reward": None,
            "latest_timeout_rate": None,
            "latest_stopped_rate": None,
            "latest_avg_no_progress_steps": None,
            "latest_avg_distance_to_pen": None,
            "latest_avg_flock_spread": None,
            "latest_avg_farthest_distance_to_pen": None,
            "latest_avg_farthest_distance_to_flock_center": None,
            "phase": "idle",
            "message": "Idle",
            "error": None,
            "error_type": None,
            "traceback": None,
            "seed_episode": None,
            "starting_episode": None,
        }

    def snapshot(self) -> dict[str, Any]:
        """Return a thread-safe copy of the current training status."""
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
        auto_promote: bool | None = None,
        promote_from_checkpoint_episode: int | None = None,
    ) -> dict[str, Any]:
        """Start a background training job and return the initial status."""
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
                    "auto_promote": (
                        self._status["auto_promote"]
                        if auto_promote is None
                        else bool(auto_promote)
                    ),
                    "auto_promote_threshold": AUTO_PROMOTE_SUCCESS_THRESHOLD,
                    "auto_promote_stages_completed": 0,
                    "auto_promote_gate": _auto_promote_gate_defaults(),
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
                    "auto_promote": auto_promote,
                    "promote_from_checkpoint_episode": promote_from_checkpoint_episode,
                },
                daemon=True,
            )
            self._thread.start()
            return dict(self._status)

    def clear(self) -> tuple[dict[str, Any], int]:
        """Stop any running job, clear outputs, and restore the baseline replay."""
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                payload = dict(self._status)
                payload["message"] = "Cannot clear training while a job is running"
                return payload, HTTPStatus.CONFLICT

        config = LabConfig()
        self._clear_training_outputs(config)
        self._remove_path(Path(config.training.output_dir) / "archive")

        with self._lock:
            self._thread = None
            self._status = self._initial_status()
            self._status["phase"] = "clearing"
            self._status["message"] = "Clearing... restoring baseline replay"

        def _restore_baseline() -> None:
            self._export_untrained_baseline(config)
            with self._lock:
                self._status["phase"] = "idle"
                self._status["message"] = "Training cleared. Baseline replay restored"

        threading.Thread(target=_restore_baseline, daemon=True).start()

        with self._lock:
            return dict(self._status), HTTPStatus.OK

    def reset_journey(self) -> tuple[dict[str, Any], int]:
        """Archive current artifacts, reset status to stage 1, and restore baseline."""
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                payload = dict(self._status)
                payload["message"] = "Cannot reset journey while a job is running"
                return payload, HTTPStatus.CONFLICT

        config = LabConfig()
        archive_dir = self._archive_training_outputs(config)
        self._clear_training_outputs(config)

        output_root = Path(config.training.output_dir)
        settings_path = output_root / TRAINING_SETTINGS_FILENAME
        settings_path.parent.mkdir(parents=True, exist_ok=True)
        settings_payload = {
            "curriculum_stage": 1,
            "enable_instinct_rewards": True,
            "debug_reward_breakdown": False,
            "auto_promote": True,
            "updated_at": datetime.datetime.now(datetime.UTC).isoformat(),
        }
        settings_path.write_text(json.dumps(settings_payload, indent=2), encoding="utf-8")

        with self._lock:
            self._thread = None
            self._status = self._initial_status()
            self._status["curriculum_stage"] = 1
            self._status["phase"] = "clearing"
            self._status["message"] = "Resetting journey... restoring baseline replay"

        def _restore_baseline() -> None:
            self._export_untrained_baseline(config)
            with self._lock:
                self._status["phase"] = "idle"
                if archive_dir is not None:
                    self._status["message"] = (
                        f"Journey reset to Stage 1. Archived previous run to {archive_dir}"
                    )
                else:
                    self._status["message"] = "Journey reset to Stage 1. No prior artifacts found"

        threading.Thread(target=_restore_baseline, daemon=True).start()

        with self._lock:
            return dict(self._status), HTTPStatus.OK

    def rewind_to_stage(self, stage: int) -> tuple[dict[str, Any], int]:
        """Drop artifacts from stages above *stage* and reset active stage to *stage*."""
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                payload = dict(self._status)
                payload["message"] = "Cannot rewind stages while a job is running"
                return payload, HTTPStatus.CONFLICT

        try:
            normalized_stage = validate_curriculum_stage(stage)
            if normalized_stage <= 0:
                normalized_stage = 1
        except ValueError as exc:
            return {"error": str(exc)}, HTTPStatus.BAD_REQUEST

        config = LabConfig()
        output_root = Path(config.training.output_dir)
        summary_path = output_root / "training-summary.json"
        summary_payload = _load_json(summary_path) or {}
        checkpoints = summary_payload.get("checkpoints", [])
        checkpoints = checkpoints if isinstance(checkpoints, list) else []

        kept_checkpoints = [
            entry for entry in checkpoints if self._checkpoint_stage(entry) <= normalized_stage
        ]
        kept_checkpoint_names = {
            str(entry.get("checkpoint"))
            for entry in kept_checkpoints
            if isinstance(entry.get("checkpoint"), str)
        }

        checkpoints_dir = output_root / "checkpoints"
        if checkpoints_dir.exists():
            for checkpoint_file in checkpoints_dir.glob("checkpoint-*.json"):
                if checkpoint_file.name not in kept_checkpoint_names:
                    self._remove_path(checkpoint_file)

        total_trained = 0
        latest_policy_state_path: str | None = None
        latest_checkpoint_payload: dict[str, Any] | None = None
        if kept_checkpoints:
            latest_checkpoint = max(
                kept_checkpoints,
                key=lambda entry: int(entry.get("checkpoint_episode", -1)),
            )
            total_trained = int(
                latest_checkpoint.get(
                    "total_training_episodes",
                    latest_checkpoint.get("checkpoint_episode", 0),
                )
            )
            latest_policy_state_path = latest_checkpoint.get("policy_state_path")
            latest_checkpoint_name = latest_checkpoint.get("checkpoint")
            if isinstance(latest_checkpoint_name, str):
                latest_checkpoint_payload = _load_json(checkpoints_dir / latest_checkpoint_name)

        new_summary_payload = dict(summary_payload)
        new_summary_payload["checkpoints"] = kept_checkpoints
        new_summary_payload["total_episodes_trained"] = total_trained
        if "final_model_path" in new_summary_payload:
            new_summary_payload["final_model_path"] = latest_policy_state_path or ""
        summary_path.write_text(json.dumps(new_summary_payload, indent=2), encoding="utf-8")

        stage_history = _read_stage_history(output_root)
        truncated_history = {
            str(key): int(value)
            for key, value in stage_history.items()
            if int(key) <= normalized_stage
        }
        (output_root / STAGE_HISTORY_FILENAME).write_text(
            json.dumps(truncated_history, indent=2),
            encoding="utf-8",
        )

        settings_path = output_root / TRAINING_SETTINGS_FILENAME
        persisted_settings = _read_persisted_settings(output_root)
        persisted_settings["curriculum_stage"] = normalized_stage
        persisted_settings["updated_at"] = datetime.datetime.now(datetime.UTC).isoformat()
        settings_path.parent.mkdir(parents=True, exist_ok=True)
        settings_path.write_text(json.dumps(persisted_settings, indent=2), encoding="utf-8")

        self._rewrite_state_for_kept_checkpoints(
            output_root,
            kept_checkpoints,
            latest_checkpoint_payload,
        )
        self._rewrite_web_exports_from_checkpoints(config, kept_checkpoints)

        with self._lock:
            self._thread = None
            self._status = self._initial_status()
            self._status["curriculum_stage"] = normalized_stage
            self._status["phase"] = "idle"
            self._status["message"] = (
                f"Rewound to Stage {normalized_stage}. Removed artifacts above this stage."
            )
            return dict(self._status), HTTPStatus.OK

    @staticmethod
    def _checkpoint_stage(payload: dict[str, Any]) -> int:
        """Extract curriculum stage from a checkpoint payload-like dict."""
        reward_stage = (
            payload.get("reward_config", {})
            .get("instincts", {})
            .get("curriculum_stage")
        )
        if isinstance(reward_stage, (int, float)):
            return int(reward_stage)
        env_stage = payload.get("environment_config", {}).get("curriculum_stage")
        if isinstance(env_stage, (int, float)):
            return int(env_stage)
        return 0

    def _rewrite_state_for_kept_checkpoints(
        self,
        output_root: Path,
        kept_checkpoints: list[dict[str, Any]],
        latest_checkpoint_payload: dict[str, Any] | None,
    ) -> None:
        """Rebuild trainer state so resumed training cannot see pruned higher-stage models."""
        state_path = output_root / Trainer.STATE_FILENAME
        existing_state = _load_json(state_path) or {}

        if not kept_checkpoints:
            self._remove_path(state_path)
            return

        latest_checkpoint = max(
            kept_checkpoints,
            key=lambda entry: int(entry.get("checkpoint_episode", -1)),
        )
        latest_stage = self._checkpoint_stage(latest_checkpoint)
        latest_total = int(
            latest_checkpoint.get(
                "total_training_episodes",
                latest_checkpoint.get("checkpoint_episode", 0),
            )
        )

        best_checkpoint = max(
            kept_checkpoints,
            key=lambda entry: (
                self._checkpoint_stage(entry),
                float(entry.get("success_rate", -1.0) or -1.0),
                float(entry.get("average_reward", float("-inf")) or float("-inf")),
                -float(entry.get("average_completion_steps", float("inf")) or float("inf")),
            ),
        )
        best_stage = self._checkpoint_stage(best_checkpoint)
        best_checkpoint_name = best_checkpoint.get("checkpoint")
        best_checkpoint_payload = (
            _load_json(output_root / "checkpoints" / best_checkpoint_name)
            if isinstance(best_checkpoint_name, str)
            else None
        )

        new_state = dict(existing_state)
        new_state["total_episodes_trained"] = latest_total
        new_state["incomplete_batch"] = None

        latest_policy_state = latest_checkpoint.get("policy_state_path")
        if isinstance(latest_policy_state, str):
            new_state["policy_state_path"] = latest_policy_state
        elif "policy_state_path" in new_state:
            new_state["policy_state_path"] = None

        if isinstance(latest_checkpoint_payload, dict) and latest_checkpoint_payload.get("policy_config") is not None:
            new_state["policy_config"] = latest_checkpoint_payload.get("policy_config")

        if isinstance(latest_checkpoint_payload, dict) and latest_checkpoint_payload.get("policy_weights") is not None:
            new_state["weights"] = latest_checkpoint_payload.get("policy_weights")

        best_policy_state = best_checkpoint.get("policy_state_path")
        if isinstance(best_policy_state, str):
            new_state["best_model_path"] = best_policy_state
        if best_policy_state is not None:
            new_state["best_model_curriculum_stage"] = best_stage
            new_state["best_success_rate"] = best_checkpoint.get("success_rate")
            new_state["best_average_reward"] = best_checkpoint.get("average_reward")
            new_state["best_completion_steps"] = best_checkpoint.get("average_completion_steps")

        if isinstance(best_checkpoint_payload, dict) and best_checkpoint_payload.get("policy_weights") is not None:
            new_state["best_formal_weights"] = best_checkpoint_payload.get("policy_weights")
            new_state["best_formal_episode"] = best_checkpoint.get("checkpoint_episode")
            new_state["best_formal_success_rate"] = best_checkpoint.get("success_rate")
            new_state["best_formal_avg_reward"] = best_checkpoint.get("average_reward")
            new_state["best_formal_avg_steps"] = best_checkpoint.get("average_completion_steps")
            new_state["best_formal_curriculum_stage"] = best_stage
            new_state["hill_climb_curriculum_stage"] = latest_stage

        state_path.write_text(json.dumps(new_state, indent=2), encoding="utf-8")

    def _rewrite_web_exports_from_checkpoints(
        self,
        config: LabConfig,
        checkpoints: list[dict[str, Any]],
    ) -> None:
        """Rewrite generated checkpoint index/replays from kept checkpoints only."""
        output_root = Path(config.training.output_dir)
        web_dir = Path(config.training.web_export_dir)
        web_dir.mkdir(parents=True, exist_ok=True)
        replay_output_dir = web_dir / "replays"
        self._remove_path(replay_output_dir)
        replay_output_dir.mkdir(parents=True, exist_ok=True)

        def resolve_source(path_str: str) -> Path:
            source = Path(path_str)
            if source.is_absolute():
                return source
            repo_root = output_root.parent
            return repo_root / source

        def export_record(record: dict[str, Any]) -> dict[str, Any]:
            exported_record = dict(record)
            source_path_str = exported_record.get("replay_path")
            if not isinstance(source_path_str, str):
                return exported_record
            source_path = resolve_source(source_path_str)
            target_path = replay_output_dir / source_path.name
            if source_path.exists():
                target_path.write_text(source_path.read_text(encoding="utf-8"), encoding="utf-8")
                exported_record["replay_path"] = f"/generated/replays/{target_path.name}"
            return exported_record

        exported_checkpoints: list[dict[str, Any]] = []
        for checkpoint in checkpoints:
            exported_checkpoint = dict(checkpoint)
            records = checkpoint.get("records", [])
            exported_checkpoint["records"] = [
                export_record(record)
                for record in records
                if isinstance(record, dict)
            ]
            exported_checkpoints.append(exported_checkpoint)

        latest_payload = exported_checkpoints[-1] if exported_checkpoints else None
        atomic_write_json(
            web_dir / "checkpoint-index.json",
            {"checkpoints": exported_checkpoints, "latest": latest_payload},
        )

        if latest_payload is not None:
            atomic_write_json(
                web_dir / "latest-checkpoint.json",
                {"checkpoint": latest_payload.get("checkpoint")},
            )
            atomic_write_json(web_dir / "latest-evaluation.json", latest_payload)
            latest_records = latest_payload.get("records") or []
            if latest_records:
                replay_rel = latest_records[0].get("replay_path", "")
                replay_src = web_dir / str(replay_rel).removeprefix("/generated/")
                if replay_src.exists():
                    atomic_write_text(
                        web_dir / "latest-replay.json",
                        replay_src.read_text(encoding="utf-8"),
                    )
                else:
                    self._remove_path(web_dir / "latest-replay.json")
            else:
                self._remove_path(web_dir / "latest-replay.json")
            return

        self._remove_path(web_dir / "latest-checkpoint.json")
        self._remove_path(web_dir / "latest-evaluation.json")
        self._remove_path(web_dir / "latest-replay.json")

    # ── Config / history helpers ─────────────────────────────────────────────

    def get_config(self) -> dict[str, Any]:
        """Return the most-recently saved effective training config."""
        output_root = Path(LabConfig().training.output_dir)
        return _load_json(output_root / "effective-training-config.json") or {}

    def get_config_history(self) -> dict[str, Any]:
        """Return the config revision history."""
        output_root = Path(LabConfig().training.output_dir)
        return _load_json(output_root / "config-history.json") or {"revisions": []}

    def get_hyperparams(self) -> dict[str, Any]:
        """Return persisted user hyperparameters (with defaults for any missing keys)."""
        output_root = Path(LabConfig().training.output_dir)
        return _read_user_hyperparams(output_root)

    def get_network_topology(self) -> dict[str, Any]:
        """Return read-only neural topology metadata for visualization."""
        config = LabConfig()
        output_root = Path(config.training.output_dir)
        effective_config = _load_json(output_root / "effective-training-config.json") or {}
        training_payload = (
            effective_config.get("training", {}) if isinstance(effective_config, dict) else {}
        )
        training_payload = training_payload if isinstance(training_payload, dict) else {}

        training_state = _load_json(output_root / Trainer.STATE_FILENAME) or {}
        policy_config = (
            training_state.get("policy_config", {}) if isinstance(training_state, dict) else {}
        )
        policy_config = policy_config if isinstance(policy_config, dict) else {}
        training_signature = (
            training_state.get("training_signature", {})
            if isinstance(training_state, dict)
            else {}
        )
        training_signature = training_signature if isinstance(training_signature, dict) else {}

        hidden_sizes = policy_config.get("hidden_sizes")
        if not isinstance(hidden_sizes, list):
            hidden_sizes = training_payload.get("neural_hidden_sizes")
        if not isinstance(hidden_sizes, list):
            hidden_sizes = list(config.training.neural_hidden_sizes)

        observation_size = policy_config.get("observation_size")
        if not isinstance(observation_size, int) or observation_size <= 0:
            observation_size = training_signature.get("observation_size")
        if not isinstance(observation_size, int) or observation_size <= 0:
            observation_size = 54

        action_size = policy_config.get("action_size")
        if not isinstance(action_size, int) or action_size <= 0:
            action_size = training_signature.get("action_size")
        if not isinstance(action_size, int) or action_size <= 0:
            action_size = len(ACTION_ORDER)

        observation_mode = training_payload.get("observation_mode")
        if not isinstance(observation_mode, str):
            observation_mode = training_signature.get("observation_mode")
        if not isinstance(observation_mode, str):
            observation_mode = config.training.observation_mode

        invalid_action_masking = training_payload.get("invalid_action_masking")
        if not isinstance(invalid_action_masking, bool):
            invalid_action_masking = config.training.invalid_action_masking

        return {
            "observation_mode": observation_mode,
            "hidden_layer_sizes": [int(v) for v in hidden_sizes if isinstance(v, int) and v > 0],
            "observation_size": int(observation_size),
            "action_size": int(action_size),
            "actor_head": {
                "type": "dense",
                "node_count": int(action_size),
                "output": "action_logits",
            },
            "critic_head": {
                "type": "dense",
                "node_count": 1,
                "output": "state_value",
            },
            "action_masking_enabled": bool(invalid_action_masking),
            "connectivity": "dense_fully_connected",
        }

    def save_hyperparams(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Persist user hyperparameter overrides and return the merged result."""
        output_root = Path(LabConfig().training.output_dir)
        return _write_user_hyperparams(output_root, payload)

    def save_config_revision(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Append *payload* as a new revision in config-history.json."""
        output_root = Path(LabConfig().training.output_dir)
        history_path = output_root / "config-history.json"
        history = _load_json(history_path) or {"revisions": []}
        revisions: list[dict[str, Any]] = history.get("revisions", [])
        entry = {
            "id": len(revisions) + 1,
            "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
            **payload,
        }
        revisions.append(entry)
        history["revisions"] = revisions
        history_path.parent.mkdir(parents=True, exist_ok=True)
        history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
        return history

    def _clear_training_outputs(self, config: LabConfig) -> None:
        output_root = Path(config.training.output_dir)
        generated_root = Path(config.training.web_export_dir)
        self._remove_path(output_root / "checkpoints")
        self._remove_path(output_root / "evaluations")
        self._remove_path(output_root / Trainer.STATE_FILENAME)
        self._remove_path(output_root / "training-summary.json")
        self._remove_path(output_root / STAGE_HISTORY_FILENAME)
        self._remove_path(output_root / TRAINING_SETTINGS_FILENAME)
        self._remove_path(generated_root / "replays")
        self._remove_path(generated_root / "latest-checkpoint.json")
        self._remove_path(generated_root / "latest-evaluation.json")
        self._remove_path(generated_root / "latest-replay.json")
        self._remove_path(generated_root / "checkpoint-index.json")

    def _archive_training_outputs(self, config: LabConfig) -> str | None:
        output_root = Path(config.training.output_dir)
        generated_root = Path(config.training.web_export_dir)
        timestamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%d-%H%M%S")
        archive_root = output_root / "archive" / f"journey-{timestamp}"
        mappings: list[tuple[Path, Path]] = [
            (output_root / "checkpoints", archive_root / "checkpoints"),
            (output_root / "evaluations", archive_root / "evaluations"),
            (output_root / Trainer.STATE_FILENAME, archive_root / Trainer.STATE_FILENAME),
            (output_root / "training-summary.json", archive_root / "training-summary.json"),
            (output_root / STAGE_HISTORY_FILENAME, archive_root / STAGE_HISTORY_FILENAME),
            (output_root / TRAINING_SETTINGS_FILENAME, archive_root / TRAINING_SETTINGS_FILENAME),
            (
                output_root / "effective-training-config.json",
                archive_root / "effective-training-config.json",
            ),
            (generated_root / "replays", archive_root / "web-replays"),
            (generated_root / "latest-checkpoint.json", archive_root / "latest-checkpoint.json"),
            (generated_root / "latest-evaluation.json", archive_root / "latest-evaluation.json"),
            (generated_root / "latest-replay.json", archive_root / "latest-replay.json"),
            (generated_root / "checkpoint-index.json", archive_root / "checkpoint-index.json"),
        ]

        moved_any = False
        for src, dst in mappings:
            if not src.exists():
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))
            moved_any = True
        return str(archive_root) if moved_any else None

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
        environment_overrides: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run one episode with the selected policy and return the replay dict."""
        output_root = Path(LabConfig().training.output_dir)
        user_params = _read_user_hyperparams(output_root)
        config = _apply_user_hyperparams(LabConfig(), user_params)
        selection = _resolve_replay_selection(
            config,
            checkpoint_episode=checkpoint_episode,
            policy_mode=policy_mode,
            effective_config=effective_config,
        )
        run_config = _apply_environment_overrides(selection.config, environment_overrides)
        policy = _load_playable_policy(
            run_config,
            checkpoint_episode=selection.checkpoint_episode,
            policy_mode=cast(PolicyMode, selection.policy_mode),
        )
        result = SheepdogEnvironment(run_config).run_policy(
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
                "total_training_episodes": selection.total_training_episodes,
                "environment": {
                    "dogs": run_config.environment.dogs,
                    "sheep": run_config.environment.sheep,
                    "width": run_config.environment.width,
                    "height": run_config.environment.height,
                    "sheep_personality_strength": run_config.environment.sheep_personality_strength,
                    "curriculum_stage": run_config.rewards.instincts.curriculum_stage,
                    "enable_instinct_rewards": (
                        run_config.rewards.instincts.enable_instinct_rewards
                    ),
                },
            }
        )
        replay_path = Path(config.training.web_export_dir) / "latest-replay.json"
        replay_path.parent.mkdir(parents=True, exist_ok=True)
        replay_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload

    def _lab_config(self) -> LabConfig:
        output_root = Path(LabConfig().training.output_dir)
        user_params = _read_user_hyperparams(output_root)
        return _apply_user_hyperparams(LabConfig(), user_params)

    def _resolve_scenario_checkpoint(
        self,
        config: LabConfig,
        payload: dict[str, Any],
        *,
        scenario_id: str | None = None,
    ) -> int:
        mode = str(payload.get("checkpoint_mode", "latest"))
        explicit = payload.get("checkpoint_episode")
        explicit_episode = None if explicit is None else int(explicit)
        if mode not in {"latest", "global_best", "scenario_best", "specific"}:
            mode = "latest"
        return resolve_checkpoint_episode(
            mode,  # type: ignore[arg-type]
            output_root=Path(config.training.output_dir),
            scenario_id=scenario_id,
            explicit_episode=explicit_episode,
            web_export_dir=Path(config.training.web_export_dir),
        )

    def list_scenarios(self) -> dict[str, Any]:
        """Return the current scenario index."""
        config = self._lab_config()
        return refresh_scenario_exports(config)

    def save_scenario(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Persist a new scenario from the web payload and return updated index."""
        config = self._lab_config()
        store = ScenarioStore(Path(config.training.output_dir) / "scenarios")
        name = str(payload.get("name", "Unnamed scenario")).strip() or "Unnamed scenario"
        seed = int(payload.get("seed", 11))
        snapshot = payload.get("snapshot")
        if not isinstance(snapshot, dict):
            raise ValueError("snapshot is required")
        description = str(payload.get("description", payload.get("notes", "")))
        scenario = scenario_from_snapshot(
            name=name,
            seed=seed,
            snapshot=snapshot,
            sheep_personality_strength=float(
                payload.get(
                    "sheep_personality_strength",
                    config.environment.sheep_personality_strength,
                )
            ),
            sheep_personality_seed_offset=int(
                payload.get(
                    "sheep_personality_seed_offset",
                    config.environment.sheep_personality_seed_offset,
                )
            ),
            seed_offset=int(payload.get("seed_offset", config.environment.seed_offset)),
            description=description,
        )
        store.save(scenario)
        refresh_scenario_exports(config)
        return scenario.to_dict()

    def evaluate_scenario_by_id(
        self,
        scenario_id: str,
        payload: dict[str, Any],
        *,
        policy_mode: PolicyMode | None = None,
        effective_config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Evaluate one checkpoint on one saved scenario and record metrics."""

        config = self._lab_config()
        store = ScenarioStore(Path(config.training.output_dir) / "scenarios")
        scenario = store.get(scenario_id)
        if scenario is None:
            raise FileNotFoundError(f"Scenario {scenario_id} not found")
        checkpoint_episode = self._resolve_scenario_checkpoint(
            config,
            payload,
            scenario_id=scenario_id,
        )
        selection = _resolve_replay_selection(
            config,
            checkpoint_episode=checkpoint_episode,
            policy_mode=policy_mode or payload.get("policy_mode"),
            effective_config=effective_config or payload.get("effective_config"),
        )
        policy = _load_playable_policy(
            selection.config,
            checkpoint_episode=checkpoint_episode,
            policy_mode=cast(PolicyMode, selection.policy_mode),
        )
        result = evaluate_scenario(
            selection.config,
            policy,
            scenario,
            checkpoint_episode,
            record_result=True,
        )
        bundle = refresh_scenario_exports(config)
        return {
            "checkpoint_episode": checkpoint_episode,
            "result": result.to_dict(),
            "index": bundle,
        }

    def replay_scenario_by_id(
        self,
        scenario_id: str,
        payload: dict[str, Any],
        *,
        policy_mode: PolicyMode | None = None,
        effective_config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run one checkpoint on a saved scenario and return replay JSON (no metric recording)."""

        config = self._lab_config()
        store = ScenarioStore(Path(config.training.output_dir) / "scenarios")
        scenario = store.get(scenario_id)
        if scenario is None:
            raise FileNotFoundError(f"Scenario {scenario_id} not found")
        checkpoint_episode = self._resolve_scenario_checkpoint(
            config,
            payload,
            scenario_id=scenario_id,
        )
        selection = _resolve_replay_selection(
            config,
            checkpoint_episode=checkpoint_episode,
            policy_mode=policy_mode or payload.get("policy_mode"),
            effective_config=effective_config or payload.get("effective_config"),
        )
        policy = _load_playable_policy(
            selection.config,
            checkpoint_episode=checkpoint_episode,
            policy_mode=cast(PolicyMode, selection.policy_mode),
        )
        run_config = config_for_scenario(selection.config, scenario)
        initial = run_config.environment
        result = SheepdogEnvironment(run_config).run_policy_on_scenario(
            policy,
            scenario,
            capture_replay=True,
        )
        replay_payload = _replay_payload(result)
        replay_payload.update(
            {
                "scenario_id": scenario.id,
                "scenario_name": scenario.name,
                "trainer_type": selection.trainer_type,
                "policy_type": selection.policy_type,
                "policy_mode": selection.policy_mode,
                "replay_mode": selection.replay_mode,
                "checkpoint_episode": checkpoint_episode,
                "total_training_episodes": selection.total_training_episodes,
                "environment": {
                    "dogs": initial.dogs,
                    "sheep": initial.sheep,
                    "width": initial.width,
                    "height": initial.height,
                    "sheep_personality_strength": initial.sheep_personality_strength,
                    "curriculum_stage": run_config.rewards.instincts.curriculum_stage,
                    "enable_instinct_rewards": run_config.rewards.instincts.enable_instinct_rewards,
                },
            }
        )
        return replay_payload

    def _run_training(
        self,
        requested_episodes: int,
        fast_mode: bool,
        *,
        enable_instinct_rewards: bool | None = None,
        curriculum_stage: int | None = None,
        debug_reward_breakdown: bool | None = None,
        auto_promote: bool | None = None,
        promote_from_checkpoint_episode: int | None = None,
    ) -> None:
        try:
            total_episodes = max(1, requested_episodes)
            # Respect the user-requested batch size so progress reflects the
            # configured run length (for example 75 episodes shows as 75).
            batch_episodes = total_episodes
            available_stage_numbers, max_stage = _curriculum_stage_metadata()
            auto_promote_enabled = True if auto_promote is None else bool(auto_promote)
            current_stage = max(1, int(curriculum_stage) if curriculum_stage is not None else 1)
            if available_stage_numbers and current_stage not in available_stage_numbers:
                current_stage = max(1, min(current_stage, max_stage))
            resume_checkpoint_episode = promote_from_checkpoint_episode
            promoted_stages = 0
            output_root = Path(LabConfig().training.output_dir)

            stage_best_checkpoint_episode: int | None = None
            stage_best_rank = (-1.0, float("-inf"), float("-inf"), float("-inf"))
            stage_best_reward = float("-inf")
            stage_qualified_streak = 0
            stage_seed_gate_hits = 0
            stage_full_success_hits = 0
            stage_seed_count = 0
            stage_checkpoints_seen = 0
            stage_no_improvement_streak = 0
            stage_batch_completed_episodes = 0

            while True:
                job_config = _build_training_job_config(
                    batch_episodes,
                    fast_mode,
                    enable_instinct_rewards=enable_instinct_rewards,
                    curriculum_stage=current_stage,
                    debug_reward_breakdown=debug_reward_breakdown,
                )
                trainer = create_trainer(job_config, job_config.training.output_dir)

                if resume_checkpoint_episode is not None:
                    try:
                        checkpoint_payload = _load_checkpoint_payload(
                            output_root, resume_checkpoint_episode
                        )
                        policy_weights = checkpoint_payload.get("policy_weights")
                        if policy_weights is not None:
                            trainer.override_start_weights(policy_weights)
                    except (FileNotFoundError, KeyError, TypeError):
                        pass

                def progress_callback(payload: dict[str, Any]) -> None:
                    nonlocal stage_best_checkpoint_episode, stage_best_rank
                    nonlocal stage_best_reward, stage_qualified_streak
                    nonlocal stage_seed_gate_hits, stage_full_success_hits, stage_seed_count
                    nonlocal stage_checkpoints_seen, stage_no_improvement_streak
                    nonlocal stage_batch_completed_episodes

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
                    if isinstance(batch_completed, (int, float)):
                        stage_batch_completed_episodes = max(
                            stage_batch_completed_episodes,
                            int(batch_completed),
                        )
                    total_trained = payload.get("total_episodes_trained")
                    update: dict[str, Any] = {
                        "running": True,
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
                        "error_type": None,
                        "traceback": None,
                    }
                    if total_trained is not None:
                        update["total_episodes_trained"] = total_trained
                    if checkpoint_episode is not None:
                        update["latest_checkpoint_episode"] = checkpoint_episode
                        update["latest_seed"] = latest_seed
                        update["latest_replay_path"] = replay_path
                    if isinstance(summary, dict) and payload.get("phase") == "checkpoint":
                        update["latest_success_rate"] = summary.get("success_rate")
                        update["latest_avg_sheep_penned"] = summary.get("average_sheep_penned")
                        update["latest_avg_reward"] = summary.get("average_reward")
                        update["latest_timeout_rate"] = summary.get("timeout_rate")
                        update["latest_stopped_rate"] = summary.get("stopped_rate")
                        update["latest_avg_no_progress_steps"] = summary.get(
                            "average_no_progress_steps"
                        )
                        update["latest_avg_distance_to_pen"] = summary.get(
                            "average_distance_to_pen"
                        )
                        update["latest_avg_flock_spread"] = summary.get("average_flock_spread")
                        update["latest_avg_farthest_distance_to_pen"] = summary.get(
                            "average_farthest_distance_to_pen"
                        )
                        update["latest_avg_farthest_distance_to_flock_center"] = summary.get(
                            "average_farthest_distance_to_flock_center"
                        )

                        success_rate = float(summary.get("success_rate", -1.0))
                        average_reward = float(summary.get("average_reward", float("-inf")))
                        timeout_rate = float(summary.get("timeout_rate", 1.0))
                        avg_penned = float(summary.get("average_sheep_penned", 0.0))
                        records_raw = summary.get("records", [])
                        success_count = 0
                        if isinstance(records_raw, (list, tuple)):
                            records = [record for record in records_raw if isinstance(record, dict)]
                            success_count = sum(
                                1 for record in records if bool(record.get("success"))
                            )
                            stage_seed_count = len(records)
                        seed_gate_ok = _seed_success_gate(success_count, stage_seed_count)
                        if stage_seed_count <= 0:
                            # Some checkpoints may not include per-seed records in the
                            # callback payload; fall back to aggregate quality signals so
                            # promotion does not get permanently blocked.
                            seed_gate_ok = (
                                success_rate >= AUTO_PROMOTE_SUCCESS_THRESHOLD
                                and timeout_rate <= AUTO_PROMOTE_MAX_TIMEOUT_RATE
                            )
                        reward_close_to_best = _reward_within_tolerance(
                            average_reward,
                            stage_best_reward,
                        )
                        qualified_for_promotion = (
                            seed_gate_ok
                            and success_rate >= AUTO_PROMOTE_SUCCESS_THRESHOLD
                            and timeout_rate <= AUTO_PROMOTE_MAX_TIMEOUT_RATE
                            and reward_close_to_best
                        )
                        stage_qualified_streak = (
                            stage_qualified_streak + 1 if qualified_for_promotion else 0
                        )
                        if seed_gate_ok:
                            stage_seed_gate_hits += 1
                        full_success_checkpoint = (
                            success_rate >= AUTO_PROMOTE_FULL_SUCCESS_RATE_THRESHOLD
                            and timeout_rate <= AUTO_PROMOTE_MAX_TIMEOUT_RATE
                            and (
                                seed_gate_ok
                                or stage_seed_count <= 0
                            )
                        )
                        if full_success_checkpoint:
                            stage_full_success_hits += 1

                        candidate_rank = (
                            success_rate,
                            average_reward,
                            -timeout_rate,
                            avg_penned,
                        )
                        if checkpoint_episode is not None and candidate_rank > stage_best_rank:
                            stage_best_rank = candidate_rank
                            stage_best_checkpoint_episode = int(checkpoint_episode)
                            stage_no_improvement_streak = 0
                        else:
                            stage_no_improvement_streak += 1
                        stage_checkpoints_seen += 1
                        stage_best_reward = max(stage_best_reward, average_reward)
                        update["auto_promote_gate"] = {
                            "decision": "pending",
                            "reason": (
                                "Checkpoint meets gate"
                                if qualified_for_promotion
                                                                else "Checkpoint below gate"
                            ),
                            "seed_count": stage_seed_count,
                            "success_count": success_count,
                            "best_success": max(0.0, stage_best_rank[0]),
                            "best_reward": (
                                None
                                if stage_best_reward == float("-inf")
                                else stage_best_reward
                            ),
                            "seed_gate_ok": seed_gate_ok,
                            "success_rate_ok": success_rate >= AUTO_PROMOTE_SUCCESS_THRESHOLD,
                            "timeout_ok": timeout_rate <= AUTO_PROMOTE_MAX_TIMEOUT_RATE,
                            "reward_close_ok": reward_close_to_best,
                            "qualified_streak": stage_qualified_streak,
                            "min_qualified_streak": AUTO_PROMOTE_MIN_QUALIFIED_STREAK,
                            "seed_gate_hits": stage_seed_gate_hits,
                            "min_seed_gate_hits": AUTO_PROMOTE_MIN_SEED_GATE_HITS,
                            "seed_gate_target_met": (
                                stage_seed_gate_hits >= AUTO_PROMOTE_MIN_SEED_GATE_HITS
                            ),
                            "full_success_hits": stage_full_success_hits,
                            "min_full_success_hits": AUTO_PROMOTE_MIN_FULL_SUCCESS_HITS,
                            "full_success_target_met": (
                                stage_full_success_hits >= AUTO_PROMOTE_MIN_FULL_SUCCESS_HITS
                            ),
                            "full_success_rate_threshold": (
                                AUTO_PROMOTE_FULL_SUCCESS_RATE_THRESHOLD
                            ),
                            "success_threshold": AUTO_PROMOTE_SUCCESS_THRESHOLD,
                            "max_timeout_rate": AUTO_PROMOTE_MAX_TIMEOUT_RATE,
                            "reward_tolerance_ratio": AUTO_PROMOTE_REWARD_TOLERANCE_RATIO,
                        }
                    if (
                        payload.get("phase") == "starting"
                        and payload.get("seed_episode") is not None
                    ):
                        update["seed_episode"] = payload.get("seed_episode")
                    if payload.get("phase") == "starting" and total_trained is not None:
                        update["starting_episode"] = total_trained
                    early_promotion_signal: _EarlyPromotionSignal | None = None
                    if isinstance(summary, dict) and payload.get("phase") == "checkpoint":
                        promotion_checkpoint_episode: int | None = None
                        if stage_best_checkpoint_episode is not None:
                            promotion_checkpoint_episode = int(stage_best_checkpoint_episode)
                        elif checkpoint_episode is not None:
                            promotion_checkpoint_episode = int(checkpoint_episode)

                        best_success_for_gate = max(stage_best_rank[0], success_rate)
                        seed_gate_target_met = (
                            stage_seed_gate_hits >= AUTO_PROMOTE_MIN_SEED_GATE_HITS
                        )
                        full_success_target_met = (
                            stage_full_success_hits >= AUTO_PROMOTE_MIN_FULL_SUCCESS_HITS
                        )
                        should_auto_promote_now = (
                            auto_promote_enabled
                            and stage < max_stage
                            and promotion_checkpoint_episode is not None
                            and best_success_for_gate >= AUTO_PROMOTE_SUCCESS_THRESHOLD
                            and seed_gate_target_met
                            and full_success_target_met
                            and stage_qualified_streak >= AUTO_PROMOTE_MIN_QUALIFIED_STREAK
                        )
                        update["auto_promote_gate_ready"] = bool(should_auto_promote_now)
                        if should_auto_promote_now:
                            update["auto_promote_gate"] = {
                                "decision": "promote_ready",
                                "reason": "Promotion criteria met mid-batch",
                                "seed_count": stage_seed_count,
                                "success_count": success_count,
                                "best_success": max(0.0, best_success_for_gate),
                                "best_reward": (
                                    None
                                    if stage_best_reward == float("-inf")
                                    else stage_best_reward
                                ),
                                "seed_gate_ok": seed_gate_target_met,
                                "success_rate_ok": (
                                    best_success_for_gate >= AUTO_PROMOTE_SUCCESS_THRESHOLD
                                ),
                                "timeout_ok": True,
                                "reward_close_ok": True,
                                "qualified_streak": stage_qualified_streak,
                                "min_qualified_streak": AUTO_PROMOTE_MIN_QUALIFIED_STREAK,
                                "seed_gate_hits": stage_seed_gate_hits,
                                "min_seed_gate_hits": AUTO_PROMOTE_MIN_SEED_GATE_HITS,
                                "seed_gate_target_met": seed_gate_target_met,
                                "full_success_hits": stage_full_success_hits,
                                "min_full_success_hits": AUTO_PROMOTE_MIN_FULL_SUCCESS_HITS,
                                "full_success_target_met": full_success_target_met,
                                "full_success_rate_threshold": (
                                    AUTO_PROMOTE_FULL_SUCCESS_RATE_THRESHOLD
                                ),
                                "success_threshold": AUTO_PROMOTE_SUCCESS_THRESHOLD,
                                "max_timeout_rate": AUTO_PROMOTE_MAX_TIMEOUT_RATE,
                                "reward_tolerance_ratio": AUTO_PROMOTE_REWARD_TOLERANCE_RATIO,
                            }
                            update["message"] = (
                                f"Stage {stage} mastered at checkpoint "
                                f"ep {promotion_checkpoint_episode}; promoting now"
                            )
                            early_promotion_signal = _EarlyPromotionSignal(
                                checkpoint_episode=int(promotion_checkpoint_episode),
                                best_success=max(0.0, best_success_for_gate),
                                qualified_streak=stage_qualified_streak,
                                seed_gate_hits=stage_seed_gate_hits,
                                full_success_hits=stage_full_success_hits,
                            )
                    self._update_status(update)
                    if early_promotion_signal is not None:
                        raise early_promotion_signal

                effective_config_path = (
                    Path(job_config.training.output_dir) / "effective-training-config.json"
                )
                effective_config_path.parent.mkdir(parents=True, exist_ok=True)
                config_dict = job_config.to_dict()
                effective_config_path.write_text(json.dumps(config_dict, indent=2),
                                                 encoding="utf-8")

                instincts_on = job_config.rewards.instincts.enable_instinct_rewards
                stage = job_config.rewards.instincts.curriculum_stage
                label = (
                    f"Stage {stage} \u00b7 "
                    f"{'instincts ON' if instincts_on else 'instincts OFF'} \u00b7 "
                    f"{'fast' if fast_mode else 'full'} mode \u00b7 "
                    f"{total_episodes} ep"
                )
                self.save_config_revision(
                    {
                        "source": "training_start",
                        "label": label,
                        "training_settings": {
                            "episodes": total_episodes,
                            "fast_mode": fast_mode,
                            "enable_instinct_rewards": instincts_on,
                            "curriculum_stage": stage,
                            "debug_reward_breakdown": (
                                job_config.rewards.instincts.debug_reward_breakdown
                            ),
                            "auto_promote": auto_promote_enabled,
                        },
                        "config": config_dict,
                    }
                )
                settings_path = Path(job_config.training.output_dir) / TRAINING_SETTINGS_FILENAME
                settings_path.write_text(
                    json.dumps(
                        {
                            "curriculum_stage": stage,
                            "enable_instinct_rewards": (
                                job_config.rewards.instincts.enable_instinct_rewards
                            ),
                            "debug_reward_breakdown": (
                                job_config.rewards.instincts.debug_reward_breakdown
                            ),
                            "auto_promote": auto_promote_enabled,
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )

                self._update_status(
                    {
                        "running": True,
                        "phase": "training",
                        "fast_mode": fast_mode,
                        "trainer_type": job_config.training.trainer_type,
                        "policy_type": job_config.training.policy_type,
                        "enable_instinct_rewards": (
                            job_config.rewards.instincts.enable_instinct_rewards
                        ),
                        "policy_mode": job_config.policy.policy_mode,
                        "replay_mode": "baseline",
                        "allow_instinct_target_awareness": (
                            job_config.policy.allow_instinct_target_awareness
                        ),
                        "handler_target_enabled": job_config.policy.handler_target_enabled,
                        "debug_reward_breakdown": (
                            job_config.rewards.instincts.debug_reward_breakdown
                        ),
                        "curriculum_stage": stage,
                        "auto_promote": auto_promote_enabled,
                        "auto_promote_threshold": AUTO_PROMOTE_SUCCESS_THRESHOLD,
                        "auto_promote_stages_completed": promoted_stages,
                        "auto_promote_gate": {
                            **_auto_promote_gate_defaults(),
                            "reason": "Collecting checkpoint evidence",
                        },
                        "requested_episodes": total_episodes,
                        "completed_episodes": 0,
                        "batch_total_episodes": batch_episodes,
                        "batch_completed_episodes": 0,
                        "current_episode": None,
                        "checkpoint_episode": None,
                        "latest_checkpoint_episode": None,
                        "latest_seed": None,
                        "latest_replay_path": None,
                        "message": "Training in progress",
                        "error": None,
                        "starting_episode": trainer.total_episodes_trained,
                    }
                )

                early_promotion: _EarlyPromotionSignal | None = None
                try:
                    trainer.train(progress_callback=progress_callback)
                except _EarlyPromotionSignal as signal:
                    early_promotion = signal
                    stage_best_checkpoint_episode = signal.checkpoint_episode

                completed_for_stage = (
                    stage_batch_completed_episodes
                    if early_promotion is not None
                    else batch_episodes
                )
                history = _update_stage_history(output_root, stage, completed_for_stage)
                self._update_status(
                    {
                        "stage_history": history,
                        "grand_total_episodes": sum(history.values()),
                    }
                )

                best_success = stage_best_rank[0]
                seed_gate_target_met = stage_seed_gate_hits >= AUTO_PROMOTE_MIN_SEED_GATE_HITS
                full_success_target_met = (
                    stage_full_success_hits >= AUTO_PROMOTE_MIN_FULL_SUCCESS_HITS
                )
                should_auto_promote = (
                    auto_promote_enabled
                    and stage < max_stage
                    and stage_best_checkpoint_episode is not None
                    and best_success >= AUTO_PROMOTE_SUCCESS_THRESHOLD
                    and seed_gate_target_met
                    and full_success_target_met
                    and stage_qualified_streak >= AUTO_PROMOTE_MIN_QUALIFIED_STREAK
                )
                if early_promotion is not None:
                    should_auto_promote = True
                    best_success = max(best_success, early_promotion.best_success)
                if not should_auto_promote:
                    stage_mastered = (
                        stage >= max_stage
                        and stage_qualified_streak >= MAX_STAGE_MASTERY_QUALIFIED_STREAK
                        and stage_full_success_hits >= MAX_STAGE_MASTERY_FULL_SUCCESS_HITS
                    )
                    plateaued = (
                        stage_checkpoints_seen >= PLATEAU_STOP_MIN_CHECKPOINTS
                        and stage_no_improvement_streak >= PLATEAU_STOP_NO_IMPROVEMENT_STREAK
                    )
                    if stage_mastered:
                        self._update_status(
                            {
                                "auto_promote_gate": {
                                    "decision": "hold",
                                    "reason": "Max stage mastered; stopping",
                                    "seed_count": stage_seed_count,
                                    "success_count": 0,
                                    "best_success": max(0.0, best_success),
                                    "best_reward": (
                                        None
                                        if stage_best_reward == float("-inf")
                                        else stage_best_reward
                                    ),
                                    "seed_gate_ok": seed_gate_target_met,
                                    "success_rate_ok": (
                                        best_success >= AUTO_PROMOTE_SUCCESS_THRESHOLD
                                    ),
                                    "timeout_ok": True,
                                    "reward_close_ok": True,
                                    "qualified_streak": stage_qualified_streak,
                                    "min_qualified_streak": AUTO_PROMOTE_MIN_QUALIFIED_STREAK,
                                    "seed_gate_hits": stage_seed_gate_hits,
                                    "min_seed_gate_hits": AUTO_PROMOTE_MIN_SEED_GATE_HITS,
                                    "seed_gate_target_met": seed_gate_target_met,
                                    "full_success_hits": stage_full_success_hits,
                                    "min_full_success_hits": AUTO_PROMOTE_MIN_FULL_SUCCESS_HITS,
                                    "full_success_target_met": full_success_target_met,
                                    "full_success_rate_threshold": (
                                        AUTO_PROMOTE_FULL_SUCCESS_RATE_THRESHOLD
                                    ),
                                    "success_threshold": AUTO_PROMOTE_SUCCESS_THRESHOLD,
                                    "max_timeout_rate": AUTO_PROMOTE_MAX_TIMEOUT_RATE,
                                    "reward_tolerance_ratio": AUTO_PROMOTE_REWARD_TOLERANCE_RATIO,
                                },
                                "message": (
                                    f"Stopped at max stage {stage}: sustained mastery "
                                    f"(qualified streak {stage_qualified_streak}, "
                                    f"full-success hits {stage_full_success_hits})."
                                ),
                            }
                        )
                        break
                    if stage >= max_stage and plateaued:
                        self._update_status(
                            {
                                "auto_promote_gate": {
                                    "decision": "hold",
                                    "reason": "Likely plateau; stopping",
                                    "seed_count": stage_seed_count,
                                    "success_count": 0,
                                    "best_success": max(0.0, best_success),
                                    "best_reward": (
                                        None
                                        if stage_best_reward == float("-inf")
                                        else stage_best_reward
                                    ),
                                    "seed_gate_ok": seed_gate_target_met,
                                    "success_rate_ok": (
                                        best_success >= AUTO_PROMOTE_SUCCESS_THRESHOLD
                                    ),
                                    "timeout_ok": True,
                                    "reward_close_ok": True,
                                    "qualified_streak": stage_qualified_streak,
                                    "min_qualified_streak": AUTO_PROMOTE_MIN_QUALIFIED_STREAK,
                                    "seed_gate_hits": stage_seed_gate_hits,
                                    "min_seed_gate_hits": AUTO_PROMOTE_MIN_SEED_GATE_HITS,
                                    "seed_gate_target_met": seed_gate_target_met,
                                    "full_success_hits": stage_full_success_hits,
                                    "min_full_success_hits": AUTO_PROMOTE_MIN_FULL_SUCCESS_HITS,
                                    "full_success_target_met": full_success_target_met,
                                    "full_success_rate_threshold": (
                                        AUTO_PROMOTE_FULL_SUCCESS_RATE_THRESHOLD
                                    ),
                                    "success_threshold": AUTO_PROMOTE_SUCCESS_THRESHOLD,
                                    "max_timeout_rate": AUTO_PROMOTE_MAX_TIMEOUT_RATE,
                                    "reward_tolerance_ratio": AUTO_PROMOTE_REWARD_TOLERANCE_RATIO,
                                },
                                "message": (
                                    f"Stopped at Stage {stage}: likely plateau after "
                                    f"{stage_checkpoints_seen} checkpoints with no new best in "
                                    f"the last {stage_no_improvement_streak}."
                                ),
                            }
                        )
                        break
                    # Batch complete: hold the stage and stop this run. The user can
                    # inspect diagnostics and start the next batch intentionally.
                    self._update_status(
                        {
                            "auto_promote_gate": {
                                "decision": "hold",
                                "reason": "Promotion criteria not met yet",
                                "seed_count": stage_seed_count,
                                "success_count": 0,
                                "best_success": max(0.0, best_success),
                                "best_reward": (
                                    None
                                    if stage_best_reward == float("-inf")
                                    else stage_best_reward
                                ),
                                "seed_gate_ok": seed_gate_target_met,
                                "success_rate_ok": best_success >= AUTO_PROMOTE_SUCCESS_THRESHOLD,
                                "timeout_ok": True,
                                "reward_close_ok": True,
                                "qualified_streak": stage_qualified_streak,
                                "min_qualified_streak": AUTO_PROMOTE_MIN_QUALIFIED_STREAK,
                                "seed_gate_hits": stage_seed_gate_hits,
                                "min_seed_gate_hits": AUTO_PROMOTE_MIN_SEED_GATE_HITS,
                                "seed_gate_target_met": seed_gate_target_met,
                                "full_success_hits": stage_full_success_hits,
                                "min_full_success_hits": AUTO_PROMOTE_MIN_FULL_SUCCESS_HITS,
                                "full_success_target_met": full_success_target_met,
                                "full_success_rate_threshold": (
                                    AUTO_PROMOTE_FULL_SUCCESS_RATE_THRESHOLD
                                ),
                                "success_threshold": AUTO_PROMOTE_SUCCESS_THRESHOLD,
                                "max_timeout_rate": AUTO_PROMOTE_MAX_TIMEOUT_RATE,
                                "reward_tolerance_ratio": AUTO_PROMOTE_REWARD_TOLERANCE_RATIO,
                            },
                            "message": (
                                f"Batch complete at Stage {stage}: best success {best_success:.0%}, "
                                f"qualified streak {stage_qualified_streak}, "
                                f"seed hits {stage_seed_gate_hits}, "
                                f"full-success hits {stage_full_success_hits}."
                            ),
                        }
                    )
                    resume_checkpoint_episode = stage_best_checkpoint_episode
                    break

                promoted_stages += 1
                current_stage = stage + 1
                resume_checkpoint_episode = stage_best_checkpoint_episode
                stage_best_checkpoint_episode = None
                stage_best_rank = (-1.0, float("-inf"), float("-inf"), float("-inf"))
                stage_best_reward = float("-inf")
                stage_qualified_streak = 0
                stage_seed_gate_hits = 0
                stage_full_success_hits = 0
                stage_seed_count = 0
                stage_checkpoints_seen = 0
                stage_no_improvement_streak = 0
                self._update_status(
                    {
                        "curriculum_stage": current_stage,
                        "auto_promote_stages_completed": promoted_stages,
                        "auto_promote_gate": {
                            "decision": "promote",
                            "reason": "Promotion criteria met",
                            "seed_count": stage_seed_count,
                            "success_count": 0,
                            "best_success": max(0.0, best_success),
                            "best_reward": (
                                None
                                if stage_best_reward == float("-inf")
                                else stage_best_reward
                            ),
                            "seed_gate_ok": True,
                            "success_rate_ok": True,
                            "timeout_ok": True,
                            "reward_close_ok": True,
                            "qualified_streak": stage_qualified_streak,
                            "min_qualified_streak": AUTO_PROMOTE_MIN_QUALIFIED_STREAK,
                            "seed_gate_hits": stage_seed_gate_hits,
                            "min_seed_gate_hits": AUTO_PROMOTE_MIN_SEED_GATE_HITS,
                            "seed_gate_target_met": seed_gate_target_met,
                            "full_success_hits": stage_full_success_hits,
                            "min_full_success_hits": AUTO_PROMOTE_MIN_FULL_SUCCESS_HITS,
                            "full_success_target_met": full_success_target_met,
                            "full_success_rate_threshold": (
                                AUTO_PROMOTE_FULL_SUCCESS_RATE_THRESHOLD
                            ),
                            "success_threshold": AUTO_PROMOTE_SUCCESS_THRESHOLD,
                            "max_timeout_rate": AUTO_PROMOTE_MAX_TIMEOUT_RATE,
                            "reward_tolerance_ratio": AUTO_PROMOTE_REWARD_TOLERANCE_RATIO,
                        },
                        "message": (
                            f"Auto-promoted to Stage {current_stage} "
                            f"from checkpoint ep {stage_best_checkpoint_episode}"
                        ),
                    }
                )

            with self._lock:
                self._status["running"] = False
                self._status["phase"] = "complete"
                if "Training complete" not in self._status.get("message", ""):
                    self._status["message"] = "Training complete"
        except Exception as exc:  # pragma: no cover  # pylint: disable=broad-exception-caught
            full_traceback = traceback.format_exc()
            # Print the full traceback to the server console so the failure is
            # always visible even when the UI only surfaces a short message.
            print("\n===== TRAINING FAILED =====", flush=True)
            print(full_traceback, flush=True)
            print("===========================\n", flush=True)
            with self._lock:
                self._status["running"] = False
                self._status["phase"] = "error"
                self._status["message"] = f"Training failed ({type(exc).__name__}): {exc}"
                self._status["error"] = str(exc)
                self._status["error_type"] = type(exc).__name__
                self._status["traceback"] = full_traceback


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

    def _file_response(self, file_path: Path) -> None:
        """Read *file_path* fully then send — avoids Content-Length races."""
        try:
            body = file_path.read_bytes()
        except FileNotFoundError:
            self.send_response(HTTPStatus.NOT_FOUND)
            self.send_header("Content-Length", "0")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            return
        suffix = file_path.suffix.lower()
        content_type = (
            "application/json; charset=utf-8" if suffix == ".json" else "application/octet-stream"
        )
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Access-Control-Allow-Origin", "*")
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

    def do_GET(self) -> None:  # noqa: N802  # pylint: disable=invalid-name
        """Handle HTTP GET requests."""
        request_path = urlsplit(self.path).path

        if request_path.startswith("/generated/"):
            rel = request_path[len("/generated/") :]
            if ".." in rel:
                self._json_response({"error": "Forbidden"}, status=HTTPStatus.FORBIDDEN)
                return
            web_export_dir = Path(LabConfig().training.web_export_dir).resolve()
            target = (web_export_dir / rel).resolve()
            if not str(target).startswith(str(web_export_dir)):
                self._json_response({"error": "Forbidden"}, status=HTTPStatus.FORBIDDEN)
                return
            self._file_response(target)
            return
        if request_path == "/api/training/status":
            self._json_response(self.manager.snapshot())
            return
        if request_path == "/api/health":
            self._json_response({"ok": True})
            return
        if request_path == "/api/config":
            self._json_response(self.manager.get_config())
            return
        if request_path == "/api/config/history":
            self._json_response(self.manager.get_config_history())
            return
        if request_path == "/api/config/hyperparams":
            self._json_response(self.manager.get_hyperparams())
            return
        if request_path in {"/api/network/topology", "/api/network/topology/"}:
            self._json_response(self.manager.get_network_topology())
            return
        if request_path == "/api/scenarios":
            self._json_response(self.manager.list_scenarios())
            return
        self._json_response({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802  # pylint: disable=invalid-name
        """Handle HTTP POST requests."""
        if self.path in {"/api/training/clear", "/api/training/reset"}:
            payload, status = self.manager.clear()
            self._json_response(payload, status=status)
            return
        if self.path == "/api/training/reset-journey":
            payload, status = self.manager.reset_journey()
            self._json_response(payload, status=status)
            return
        if self.path == "/api/training/rewind":
            body = self._read_json()
            target_stage = int(body.get("stage", 1))
            payload, status = self.manager.rewind_to_stage(target_stage)
            self._json_response(payload, status=status)
            return

        if self.path == "/api/config/history":
            body = self._read_json()
            history = self.manager.save_config_revision(body)
            self._json_response(history)
            return

        if self.path == "/api/config/hyperparams":
            body = self._read_json()
            merged = self.manager.save_hyperparams(body)
            self._json_response(merged)
            return

        payload = self._read_json()
        scenario_action = _parse_scenario_action_path(self.path)
        if scenario_action is not None:
            scenario_id, action = scenario_action
            policy_mode = payload.get("policy_mode")
            effective_config = payload.get("effective_config")
            try:
                if action == "evaluate":
                    result = self.manager.evaluate_scenario_by_id(
                        scenario_id,
                        payload,
                        policy_mode=policy_mode,
                        effective_config=(
                            effective_config if isinstance(effective_config, dict) else None
                        ),
                    )
                else:
                    result = self.manager.replay_scenario_by_id(
                        scenario_id,
                        payload,
                        policy_mode=policy_mode,
                        effective_config=(
                            effective_config if isinstance(effective_config, dict) else None
                        ),
                    )
            except FileNotFoundError as exc:
                self._json_response({"error": str(exc)}, status=HTTPStatus.NOT_FOUND)
                return
            except ValueError as exc:
                self._json_response({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                return
            self._json_response(result)
            return

        if self.path == "/api/scenarios":
            try:
                scenario = self.manager.save_scenario(payload)
            except (ValueError, TypeError) as exc:
                self._json_response({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                return
            self._json_response(scenario)
            return

        if self.path == "/api/replay/run":
            seed = int(payload.get("seed", 11))
            checkpoint_episode = payload.get("checkpoint_episode")
            policy_mode = payload.get("policy_mode")
            effective_config = payload.get("effective_config")
            environment_overrides = payload.get("environment_overrides")
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
                    environment_overrides=(
                        environment_overrides
                        if isinstance(environment_overrides, dict)
                        else None
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
        auto_promote = payload.get("auto_promote")
        promote_from_checkpoint_episode = payload.get("promote_from_checkpoint_episode")
        self._json_response(
            self.manager.start(
                requested_episodes,
                fast_mode,
                enable_instinct_rewards=(
                    None if enable_instinct_rewards is None else bool(enable_instinct_rewards)
                ),
                curriculum_stage=(None if curriculum_stage is None else int(curriculum_stage)),
                debug_reward_breakdown=(
                    None if debug_reward_breakdown is None else bool(debug_reward_breakdown)
                ),
                auto_promote=(None if auto_promote is None else bool(auto_promote)),
                promote_from_checkpoint_episode=(
                    None
                    if promote_from_checkpoint_episode is None
                    else int(promote_from_checkpoint_episode)
                ),
            )
        )

    def do_OPTIONS(self) -> None:  # noqa: N802  # pylint: disable=invalid-name
        """Handle HTTP OPTIONS (CORS preflight) requests."""
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

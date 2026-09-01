"""Tests for controlled adaptive-resume experiments."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from unittest.mock import patch

from sheepdog.adaptive_resume_experiment import _decide, run_adaptive_resume_experiment
from sheepdog.batch_size_experiment import BatchSizeArmResult
from sheepdog.config import LabConfig


def _write_baseline(tmp_path: Path) -> Path:
    """Create a frozen stage-2 baseline."""

    baseline_root = tmp_path / "baseline"
    model_path = baseline_root / "start-model.zip"
    baseline_root.mkdir()
    with zipfile.ZipFile(model_path, "w") as archive:
        archive.writestr("placeholder", "model")
    state = {
        "total_episodes_trained": 42,
        "total_timesteps": 86_016,
        "policy_state_path": str(model_path),
        "best_model_path": str(model_path),
        "adaptive_step_state": {
            "stage": 2,
            "curriculum_stage": 0,
            "ema_success_rate": 0.65,
            "consecutive_hits": 0,
        },
        "policy_config": {},
        "training_signature": {},
    }
    (baseline_root / "training-state.json").write_text(
        json.dumps(state), encoding="utf-8"
    )
    return baseline_root


def _write_legacy_comparison(tmp_path: Path, baseline_root: Path) -> Path:
    """Create one matching legacy stage-1 result."""

    evaluation_path = tmp_path / "legacy-evaluation.json"
    evaluation_path.write_text(
        json.dumps({"records": [{"seed": 101, "success": False}]}),
        encoding="utf-8",
    )
    comparison = {
        "control_batch_size": 1024,
        "timesteps_per_arm": 8_192,
        "train_seeds": [7],
        "evaluation_seeds": [101],
        "output_dir": str(baseline_root.parent),
        "arm_results": [
            {
                "train_seed": 7,
                "batch_size": 1024,
                "success_rate": 0.0,
                "timeout_rate": 1.0,
                "average_reward": 0.0,
                "median_success_steps": float("inf"),
                "approx_kl": 0.001,
                "clip_fraction": 0.0,
                "explained_variance": 0.5,
                "policy_gradient_loss": -0.001,
                "minimum_checkpoint_success_rate": 0.4,
                "evaluation_json": str(evaluation_path),
                "output_dir": str(tmp_path / "legacy"),
                "final_model_path": str(tmp_path / "legacy.zip"),
            }
        ],
    }
    comparison_path = tmp_path / "comparison.json"
    comparison_path.write_text(json.dumps(comparison), encoding="utf-8")
    return comparison_path


def test_experiment_runs_only_restored_arm_from_matching_frozen_baseline(
    tmp_path: Path,
) -> None:
    """The new arm should differ from the legacy result only by adaptive state."""

    baseline_root = _write_baseline(tmp_path)
    comparison_path = _write_legacy_comparison(tmp_path, baseline_root)
    output_root = tmp_path / "resume-ab"
    observed_states: list[dict[str, object]] = []

    def fake_run_arm(config, *, arm_root: Path, **_kwargs):
        observed_states.append(
            json.loads((arm_root / "training-state.json").read_text(encoding="utf-8"))
        )
        result = BatchSizeArmResult(
            train_seed=config.training.train_seed,
            batch_size=config.training.batch_size,
            effective_batch_size=config.training.batch_size,
            success_rate=1.0,
            timeout_rate=0.0,
            average_reward=10.0,
            median_success_steps=100.0,
            approx_kl=0.001,
            clip_fraction=0.0,
            explained_variance=0.6,
            policy_gradient_loss=-0.001,
            minimum_checkpoint_success_rate=0.5,
            candidate_only_successes=0,
            control_only_successes=0,
            output_dir=str(arm_root),
            evaluation_json=str(arm_root / "evaluation.json"),
            final_model_path=str(arm_root / "model.zip"),
        )
        return result, (True,)

    config = LabConfig()
    with patch(
        "sheepdog.adaptive_resume_experiment._run_arm", side_effect=fake_run_arm
    ):
        result = run_adaptive_resume_experiment(
            config,
            baseline_root=baseline_root,
            legacy_comparison_path=comparison_path,
            output_root=output_root,
            total_timesteps=8_192,
            train_seeds=(7,),
            evaluation_seeds=(101,),
        )

    assert len(observed_states) == 1
    assert observed_states[0]["adaptive_step_state"]["stage"] == 2
    assert result.restored_success_rate == 1.0
    assert result.legacy_success_rate == 0.0
    assert result.paired_restored_wins == 1
    assert (output_root / "comparison.json").exists()


def test_decision_is_mixed_when_success_and_robustness_disagree() -> None:
    """Lower success with fewer timeouts and collapses is a tradeoff, not a regression."""

    verdict, _reason = _decide(
        success_difference=-0.0367,
        timeout_difference=-0.0233,
        seed_differences=(-0.07, -0.26, 0.22),
        legacy_collapses=1,
        restored_collapses=0,
    )

    assert verdict == "MIXED"

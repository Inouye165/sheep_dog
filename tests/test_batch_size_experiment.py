"""Tests for controlled MaskablePPO batch-size experiments."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from unittest.mock import patch

from sheepdog.batch_size_experiment import (
    BatchSizeArmResult,
    _checkpoint_schedule,
    _decide,
    _exact_paired_p_value,
    _snapshot_baseline,
    run_batch_size_experiment,
)
from sheepdog.config import LabConfig


def _write_baseline(tmp_path: Path) -> Path:
    """Create a minimal valid baseline state and SB3-style zip artifact."""

    baseline_root = tmp_path / "artifacts"
    model_path = baseline_root / "models" / "best-model.zip"
    model_path.parent.mkdir(parents=True)
    with zipfile.ZipFile(model_path, "w") as archive:
        archive.writestr("placeholder", "model")
    (baseline_root / "training-state.json").write_text(
        json.dumps(
            {
                "total_episodes_trained": 42,
                "total_environment_episodes": 420,
                "total_timesteps": 86_016,
                "policy_state_path": str(model_path),
                "best_model_path": str(model_path),
                "best_success_rate": 0.9,
                "best_average_reward": 300.0,
                "best_completion_steps": 200.0,
                "policy_config": {
                    "hidden_sizes": [128, 128, 128],
                    "observation_size": 54,
                    "action_size": 9,
                    "env_workers": 1,
                },
                "training_signature": {},
                "incomplete_batch": {"batch_completed_segments": 5},
                "run_id": "source-run",
            }
        ),
        encoding="utf-8",
    )
    return baseline_root


def _arm_result(root: Path, batch_size: int, success_rate: float) -> BatchSizeArmResult:
    """Build a compact synthetic arm result for orchestration tests."""

    return BatchSizeArmResult(
        train_seed=7,
        batch_size=batch_size,
        effective_batch_size=batch_size,
        success_rate=success_rate,
        timeout_rate=1.0 - success_rate,
        average_reward=300.0,
        median_success_steps=200.0,
        approx_kl=0.003,
        clip_fraction=0.01,
        explained_variance=0.9,
        policy_gradient_loss=-0.001,
        minimum_checkpoint_success_rate=0.8,
        candidate_only_successes=0,
        control_only_successes=0,
        output_dir=str(root),
        evaluation_json=str(root / "evaluation.json"),
        final_model_path=str(root / "model.zip"),
    )


def test_checkpoint_schedule_is_rollout_aligned() -> None:
    """Experiment checkpoints should compare equal numbers of complete rollouts."""

    assert _checkpoint_schedule(512_000, 2_048) == (102_400, 256_000, 512_000)


def test_snapshot_freezes_model_and_discards_incomplete_batch(tmp_path: Path) -> None:
    """Both arms should use a stable model and not resume an active live batch."""

    baseline_root = _write_baseline(tmp_path)
    output_root = tmp_path / "experiment"
    output_root.mkdir()

    state = _snapshot_baseline(baseline_root, output_root)

    snapshot_model = Path(state["best_model_path"])
    assert snapshot_model == Path(state["policy_state_path"])
    assert snapshot_model.parent == output_root / "baseline"
    assert zipfile.is_zipfile(snapshot_model)
    assert state["incomplete_batch"] is None


def test_decision_requires_clear_improvement_and_rejects_regression() -> None:
    """Decision guardrails should favor safety when evidence is weak or negative."""

    significant_p = _exact_paired_p_value(candidate_wins=18, control_wins=2)
    verdict, _reason = _decide(
        success_difference=0.06,
        timeout_difference=-0.04,
        candidate_wins=18,
        control_wins=2,
        p_value=significant_p,
        candidate_minimum_success=0.7,
    )
    regressed, _reason = _decide(
        success_difference=-0.02,
        timeout_difference=0.02,
        candidate_wins=2,
        control_wins=8,
        p_value=0.1,
        candidate_minimum_success=0.7,
    )

    assert significant_p < 0.05
    assert verdict == "KEEP_512"
    assert regressed == "KEEP_1024"


def test_experiment_isolates_equal_budget_arms_and_writes_reports(tmp_path: Path) -> None:
    """The orchestrator should vary only batch size within each paired seed."""

    baseline_root = _write_baseline(tmp_path)
    output_root = tmp_path / "experiment"
    seen_configs: list[LabConfig] = []

    def fake_run_arm(config: LabConfig, *, arm_root: Path, **_kwargs):
        seen_configs.append(config)
        if config.training.batch_size == 512:
            return _arm_result(arm_root, 512, 1.0), (True, True, True, True)
        return _arm_result(arm_root, 1024, 0.5), (False, False, True, True)

    with patch("sheepdog.batch_size_experiment._run_arm", side_effect=fake_run_arm):
        result = run_batch_size_experiment(
            LabConfig(),
            baseline_root=baseline_root,
            output_root=output_root,
            total_timesteps=8_192,
            train_seeds=(7,),
            evaluation_seeds=(101, 102, 103, 104),
        )

    assert {config.training.batch_size for config in seen_configs} == {512, 1024}
    assert {config.training.total_timesteps for config in seen_configs} == {8_192}
    assert len({config.training.output_dir for config in seen_configs}) == 2
    assert all(config.training.backup_enabled is False for config in seen_configs)
    assert result.paired_candidate_wins == 2
    assert result.verdict == "INCONCLUSIVE"
    assert (output_root / "comparison.json").exists()
    assert (output_root / "arms.csv").exists()
    assert (output_root / "comparison.md").exists()

"""Unit and smoke tests for scripts/audit_heldout_50.py."""

from __future__ import annotations

import gzip
import json
from dataclasses import replace
from pathlib import Path

import pytest
from scripts.audit_heldout_50 import (
    ADVERSARIAL_SCENARIO_SEEDS,
    DEV_SEEDS,
    HELDOUT_50_SEEDS,
    _compute_repeated_movement_stats,
    _resolve_assistance_disclosure,
    _select_representative_successes,
    is_prime,
    main,
    run_audit,
    validate_heldout_seeds,
    wilson_ci_95,
)

from sheepdog.config import LabConfig
from sheepdog.curriculum import apply_training_profile
from sheepdog.evaluation.evaluator import EvaluationRecord
from sheepdog.policies.neural import NeuralPolicy


def test_heldout_50_seeds_are_unique_primes_disjoint_from_dev() -> None:
    assert len(HELDOUT_50_SEEDS) == 50
    assert len(set(HELDOUT_50_SEEDS)) == 50
    assert all(is_prime(seed) for seed in HELDOUT_50_SEEDS)
    assert all(seed >= 1009 for seed in HELDOUT_50_SEEDS)
    assert set(HELDOUT_50_SEEDS).isdisjoint(set(DEV_SEEDS))
    assert validate_heldout_seeds(HELDOUT_50_SEEDS) == HELDOUT_50_SEEDS

    # Confirm they are the exact first 50 primes starting at 1009.
    primes: list[int] = []
    candidate = 1009
    while len(primes) < 50:
        if is_prime(candidate):
            primes.append(candidate)
        candidate += 1
    assert tuple(primes) == HELDOUT_50_SEEDS


def test_wilson_ci_95_known_cases() -> None:
    assert wilson_ci_95(0, 0) == (None, None)

    low_zero, high_zero = wilson_ci_95(0, 10)
    assert low_zero == pytest.approx(0.0, abs=1e-9)
    assert high_zero is not None and 0.27 < high_zero < 0.30

    low_all, high_all = wilson_ci_95(10, 10)
    assert low_all is not None and 0.70 < low_all < 0.73
    assert high_all == pytest.approx(1.0, abs=1e-9)

    low_45, high_45 = wilson_ci_95(45, 50)
    assert low_45 is not None and high_45 is not None
    assert 0.78 < low_45 < 0.80
    assert 0.95 < high_45 < 0.965


def test_audit_rejects_non_empty_output_dir_without_overwrite(tmp_path: Path) -> None:
    tiny_cfg = _make_tiny_config(tmp_path)
    model_path = _save_tiny_model(tmp_path / "model.zip", tiny_cfg)
    out_dir = tmp_path / "existing_out"
    out_dir.mkdir()
    (out_dir / "sentinel.txt").write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError, match="non-empty"):
        run_audit(
            model_path=model_path,
            stage=38,
            output_dir=out_dir,
            overwrite=False,
            workers=1,
            seed_count=1,
            skip_scenarios=True,
            base_config=tiny_cfg,
        )

    assert main([
        "--model-path", str(model_path),
        "--output-dir", str(out_dir),
        "--workers", "1",
        "--seed-count", "1",
        "--skip-scenarios",
    ]) == 2


def _make_tiny_config(tmp_path: Path, *, observation_mode: str = "guided") -> LabConfig:
    cfg = apply_training_profile(LabConfig(), curriculum_stage=38)
    return replace(
        cfg,
        environment=replace(
            cfg.environment,
            max_steps=20,
            no_progress_window=10,
        ),
        training=replace(
            cfg.training,
            neural_hidden_sizes=(8,),
            rollout_steps=32,
            batch_size=16,
            ppo_env_workers=1,
            observation_mode=observation_mode,
            observation_history_strides=(0,),
            include_loop_flags_in_observation=True,
            output_dir=str(tmp_path / "train_out"),
        ),
        policy=replace(cfg.policy, policy_mode="neural_policy"),
    )


def _save_tiny_model(target: Path, cfg: LabConfig) -> Path:
    policy = NeuralPolicy.initialize(cfg)
    try:
        return policy.save(target)
    finally:
        policy.close()


def test_repeated_movement_and_assistance_disclosure_and_representative_selection() -> None:
    frames = [
        {"snapshot": {"dogs": [{"index": 0, "x": 5, "y": 5}, {"index": 1, "x": 10, "y": 10}]}},
        {"snapshot": {"dogs": [{"index": 0, "x": 6, "y": 5}, {"index": 1, "x": 11, "y": 10}]}},
        {"snapshot": {"dogs": [{"index": 0, "x": 5, "y": 5}, {"index": 1, "x": 12, "y": 10}]}},
        {"snapshot": {"dogs": [{"index": 0, "x": 6, "y": 5}, {"index": 1, "x": 13, "y": 10}]}},
        {"snapshot": {"dogs": [{"index": 0, "x": 7, "y": 5}, {"index": 1, "x": 14, "y": 10}]}},
    ]
    rep_steps, rep_frac = _compute_repeated_movement_stats(frames)
    assert rep_steps == 1
    assert rep_frac == pytest.approx(0.2)

    guided = _resolve_assistance_disclosure("guided")
    assert guided["assistance_mode"] == "guided"
    assert guided["is_guided"] is True
    assert guided["is_autonomous_claimed"] is False

    emergent_gate = _resolve_assistance_disclosure("emergent_gate")
    assert emergent_gate["assistance_mode"] == "emergent_gate"
    assert emergent_gate["is_guided"] is False
    assert emergent_gate["is_autonomous_claimed"] is False

    def _rec(seed: int, success: bool, steps: int) -> EvaluationRecord:
        return EvaluationRecord(
            seed=seed,
            success=success,
            timeout=not success,
            stopped=False,
            steps=steps,
            simulated_seconds=float(steps),
            sheep_penned=6 if success else 4,
            final_sheep_distance_to_pen=0.0 if success else 10.0,
            final_flock_spread=1.0,
            no_progress_steps=0,
            stop_reason="success" if success else "timeout",
            spawn_mode="scattered_sheep",
            reward_total=10.0 if success else -10.0,
            final_farthest_distance_to_pen=0.0 if success else 12.0,
            final_farthest_distance_to_flock_center=1.0,
            role_switches=0,
            collector_activations=0,
            blocker_activations=0,
            cumulative_gate_progress=0.0,
            controlled_stall_steps=0,
            left_flank_occupancy_steps=0,
            right_flank_occupancy_steps=0,
            gate_corridor_occupancy_peak=0.0,
            gate_corridor_failure_steps=0,
            dog_role_occupancy={},
            reward_breakdown={},
            replay_path=f"replay_{seed}.json.gz",
        )

    selected = _select_representative_successes([
        _rec(1009, True, 300),
        _rec(1013, True, 120),
        _rec(1019, False, 2440),
        _rec(1021, True, 900),
    ])
    assert [r.seed for r in selected] == [1013, 1021]


def test_audit_end_to_end_smoke_in_tmp_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Ensure nothing in the repository's artifacts/ is touched even if a path leaks.
    repo_artifacts_dir = Path("artifacts").resolve()
    before_artifacts = (
        {p.name: p.stat().st_mtime_ns for p in repo_artifacts_dir.iterdir()}
        if repo_artifacts_dir.exists()
        else {}
    )

    monkeypatch.chdir(tmp_path)
    cfg = _make_tiny_config(tmp_path, observation_mode="guided")
    model_path = _save_tiny_model(tmp_path / "tiny-stage38.zip", cfg)
    out_dir = tmp_path / "audit_out"

    report = run_audit(
        model_path=model_path,
        stage=38,
        output_dir=out_dir,
        overwrite=False,
        workers=1,
        seed_count=2,
        skip_scenarios=False,
        scenario_types=("scattered_sheep", "split_flock"),
        scenario_seeds=(ADVERSARIAL_SCENARIO_SEEDS[0],),
        base_config=cfg,
    )

    report_json = out_dir / "audit_report.json"
    report_md = out_dir / "audit_report.md"
    assert report_json.is_file()
    assert report_md.is_file()

    loaded = json.loads(report_json.read_text(encoding="utf-8"))
    for key in (
        "report_version",
        "generated_at",
        "stage",
        "model",
        "assistance",
        "observation_history_strides",
        "include_loop_flags_in_observation",
        "seed_disjointness",
        "heldout_50",
        "adversarial_scenarios",
        "failures",
        "failure_summary",
        "retained_replays",
    ):
        assert key in loaded

    assert loaded["assistance"]["observation_mode"] == "guided"
    assert loaded["assistance"]["assistance_mode"] == "guided"
    assert loaded["assistance"]["is_autonomous_claimed"] is False
    assert loaded["observation_history_strides"] == [0]
    assert loaded["heldout_50"]["total"] == 2
    assert loaded["adversarial_scenarios"]["total"] == 2
    assert loaded["seed_disjointness"]["disjoint_from_dev"] is True

    # Untrained tiny policy on max_steps=20 will produce failures with populated diagnostics.
    assert loaded["failure_summary"]["total_failures"] >= 1
    first_failure = loaded["failures"][0]
    assert first_failure["termination_reason"] in {"no_progress", "timeout", "other"}
    assert "unpenned_sheep" in first_failure and len(first_failure["unpenned_sheep"]) >= 1
    sample_sheep = first_failure["unpenned_sheep"][0]
    for sheep_key in ("index", "personality", "x", "y", "wall_margin", "near_wall_or_corner"):
        assert sheep_key in sample_sheep
    assert "repeated_movement_steps" in first_failure
    assert "repeated_movement_fraction" in first_failure

    # Retained failure replays must be valid gzipped JSON inside out_dir/replays/.
    replays_dir = out_dir / "replays"
    assert replays_dir.is_dir()
    replay_files = list(replays_dir.glob("*.json.gz"))
    assert len(replay_files) >= 1
    for rf in replay_files:
        assert rf.resolve().is_relative_to(out_dir.resolve())
        with gzip.open(rf, "rt", encoding="utf-8") as handle:
            bundle = json.load(handle)
        assert "frames" in bundle and "stats" in bundle and "final_snapshot" in bundle

    # Nothing written inside tmp_path outside expected model, train_out, and out_dir.
    assert set(p.name for p in tmp_path.iterdir()) <= {
        "tiny-stage38.zip",
        "train_out",
        "audit_out",
    }

    # Verify original artifacts/ directory was untouched.
    if repo_artifacts_dir.exists():
        after_artifacts = {p.name: p.stat().st_mtime_ns for p in repo_artifacts_dir.iterdir()}
        assert after_artifacts == before_artifacts

    assert report["model"]["sha256"] == loaded["model"]["sha256"]

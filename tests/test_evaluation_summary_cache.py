"""Focused tests for evaluation summary cache and promotion gate invariance."""

from __future__ import annotations

import json
from pathlib import Path

from sheepdog.evaluation.summary_cache import (
    CACHE_VERSION,
    default_eval_cache_path,
    load_evaluation_summaries,
)
from sheepdog.server import (
    _DIR_EVAL_CACHE,
    _EVAL_CACHE_LOCK,
    _load_all_persisted_evaluations,
    compute_promotion_gate_snapshot,
)


def _clear_server_dir_cache() -> None:
    with _EVAL_CACHE_LOCK:
        _DIR_EVAL_CACHE.clear()


def _write_eval_file(
    eval_dir: Path,
    filename: str,
    *,
    ep: int,
    stage: int = 1,
    mode: str = "confidence",
    success_rate: float = 0.9,
    avg_reward: float = 42.5,
    avg_steps: float = 80.0,
    seeds: tuple[int, ...] = tuple(range(101, 111)),
    failing_seeds: set[int] | None = None,
    run_id: str | None = None,
    observation_schema_hash: str | None = None,
    environment_config_hash: str | None = None,
    evaluation_seed_set_id: str | None = None,
    evaluation_id: str | None = None,
) -> Path:
    eval_dir.mkdir(parents=True, exist_ok=True)
    failing = failing_seeds or set()
    records = [
        {
            "seed": s,
            "success": s not in failing,
            "timeout": s in failing,
            "steps": 150 if s in failing else int(avg_steps),
            "extra_heavy_blob": [1, 2, 3] * 20,
        }
        for s in seeds
    ]
    payload: dict[str, object] = {
        "evaluation_id": evaluation_id or f"eval_ep_{ep}_{mode}_{run_id or 'legacy'}",
        "checkpoint_id": f"chk_ep_{ep}_{run_id or 'legacy'}",
        "checkpoint_episode": ep,
        "curriculum_stage": stage,
        "evaluation_mode": mode,
        "promotion_eligible": mode == "confidence",
        "success_rate": success_rate,
        "average_reward": avg_reward,
        "average_completion_steps": avg_steps,
        "records": records,
    }
    if run_id is not None:
        payload["run_id"] = run_id
    if observation_schema_hash is not None:
        payload["observation_schema_hash"] = observation_schema_hash
    if environment_config_hash is not None:
        payload["environment_config_hash"] = environment_config_hash
    if evaluation_seed_set_id is not None:
        payload["evaluation_seed_set_id"] = evaluation_seed_set_id
    path = eval_dir / filename
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_checkpoint_file(
    output_root: Path,
    ep: int,
    stage: int = 1,
    *,
    run_id: str | None = None,
    observation_schema_hash: str | None = None,
    environment_config_hash: str | None = None,
    evaluation_mode: str | None = None,
) -> Path:
    cp_dir = output_root / "checkpoints"
    cp_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        "checkpoint_episode": ep,
        "checkpoint_id": f"chk_ep_{ep}_{run_id or 'legacy'}",
        "curriculum_stage": stage,
        "evaluation_seeds": list(range(101, 111)),
        "evaluation_seed_count": 10,
        "success_rate": 0.9,
    }
    if run_id is not None:
        payload["run_id"] = run_id
    if observation_schema_hash is not None:
        payload["observation_schema_hash"] = observation_schema_hash
    if environment_config_hash is not None:
        payload["environment_config_hash"] = environment_config_hash
    if evaluation_mode is not None:
        payload["evaluation_mode"] = evaluation_mode
    path = cp_dir / f"checkpoint-{ep:06d}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_eval_cache_matches_uncached_promotion_snapshot(tmp_path: Path) -> None:
    _clear_server_dir_cache()
    eval_dir = tmp_path / "evaluations"
    for idx, ep in enumerate((10, 20, 30, 40, 50, 60)):
        _write_checkpoint_file(tmp_path, ep, stage=1)
        _write_eval_file(
            eval_dir,
            f"evaluation-checkpoint-{ep:06d}.json",
            ep=ep,
            stage=1,
            mode="confidence",
            success_rate=0.9,
            avg_reward=40.0 + idx,
            avg_steps=95.0 - idx * 2,
            failing_seeds={101},
        )
    # Also add a stage-2 eval and a quick eval on ep=60 to test stage/mode filtering
    _write_eval_file(
        eval_dir,
        "eval_stage2_ep70.json",
        ep=70,
        stage=2,
        mode="confidence",
        success_rate=1.0,
        failing_seeds=set(),
    )
    _write_eval_file(
        eval_dir,
        "eval_quick_ep60.json",
        ep=60,
        stage=1,
        mode="quick",
        success_rate=0.0,
        seeds=(101, 102, 103, 104, 105),
        failing_seeds={101, 102, 103, 104, 105},
    )
    # Ignored files
    (eval_dir / "eval_temp.inprogress.json").write_text("{}", encoding="utf-8")
    (eval_dir / "other.json").write_text("{}", encoding="utf-8")

    # Cold load (builds cache)
    cache_path = default_eval_cache_path(tmp_path)
    assert not cache_path.exists()
    cold_evals = _load_all_persisted_evaluations(tmp_path)
    assert cache_path.exists()
    cold_gate = compute_promotion_gate_snapshot(tmp_path, target_ep=60)

    # Warm load (from disk cache after clearing in-memory _DIR_EVAL_CACHE)
    _clear_server_dir_cache()
    warm_evals = _load_all_persisted_evaluations(tmp_path)
    warm_gate = compute_promotion_gate_snapshot(tmp_path, target_ep=60)

    assert len(cold_evals) == len(warm_evals) == 8
    assert cold_gate == warm_gate
    assert warm_gate["stage"] == 1
    assert warm_gate["formal_evaluations_available"] == 6


def test_eval_cache_incremental_addition_and_corrupt_fallback(tmp_path: Path) -> None:
    _clear_server_dir_cache()
    eval_dir = tmp_path / "evaluations"
    for ep in (10, 20, 30, 40, 50):
        _write_checkpoint_file(tmp_path, ep, stage=1)
        _write_eval_file(
            eval_dir,
            f"evaluation-checkpoint-{ep:06d}.json",
            ep=ep,
            stage=1,
            mode="confidence",
            success_rate=1.0,
            failing_seeds=set(),
        )

    gate_before = compute_promotion_gate_snapshot(tmp_path, target_ep=50)
    assert gate_before["formal_evaluations_available"] == 5
    assert gate_before["decision"] == "pending"

    # Incremental addition: 6th evaluation lands
    _write_checkpoint_file(tmp_path, 60, stage=1)
    _write_eval_file(
        eval_dir,
        "evaluation-checkpoint-000060.json",
        ep=60,
        stage=1,
        mode="confidence",
        success_rate=1.0,
        failing_seeds=set(),
    )
    _clear_server_dir_cache()
    gate_after = compute_promotion_gate_snapshot(tmp_path, target_ep=60)
    assert gate_after["formal_evaluations_available"] == 6
    assert gate_after["decision"] == "promote_ready"
    assert gate_after["ready"] is True

    # Corrupt cache fallback
    cache_path = default_eval_cache_path(tmp_path)
    cache_path.write_text("{corrupt json", encoding="utf-8")
    _clear_server_dir_cache()
    gate_corrupt = compute_promotion_gate_snapshot(tmp_path, target_ep=60)
    assert gate_corrupt == gate_after

    # Version mismatch fallback
    cache_path.write_text(
        json.dumps({"version": CACHE_VERSION + 99, "fields": [], "record_fields": [], "entries": {}}),
        encoding="utf-8",
    )
    rebuilt = load_evaluation_summaries(eval_dir, cache_path)
    assert len(rebuilt) == 6
    rebuilt_json = json.loads(cache_path.read_text(encoding="utf-8"))
    assert rebuilt_json["version"] == CACHE_VERSION


def test_promotion_gate_excludes_stage_transition_baseline(tmp_path: Path) -> None:
    _clear_server_dir_cache()
    eval_dir = tmp_path / "evaluations"
    run_id = "run_stage_baseline_test"
    obs_hash = "obs_hash_v1"
    env_hash = "env_hash_v1"

    # 1 stage_transition_baseline evaluation at 100% + 5 confidence evaluations at 100%
    _write_checkpoint_file(
        tmp_path,
        5,
        stage=2,
        run_id=run_id,
        observation_schema_hash=obs_hash,
        environment_config_hash=env_hash,
        evaluation_mode="stage_transition_baseline",
    )
    _write_eval_file(
        eval_dir,
        "evaluation-checkpoint-000005.json",
        ep=5,
        stage=2,
        mode="stage_transition_baseline",
        success_rate=1.0,
        failing_seeds=set(),
        run_id=run_id,
        observation_schema_hash=obs_hash,
        environment_config_hash=env_hash,
    )
    for ep in (10, 20, 30, 40, 50):
        _write_checkpoint_file(
            tmp_path,
            ep,
            stage=2,
            run_id=run_id,
            observation_schema_hash=obs_hash,
            environment_config_hash=env_hash,
        )
        _write_eval_file(
            eval_dir,
            f"evaluation-checkpoint-{ep:06d}.json",
            ep=ep,
            stage=2,
            mode="confidence",
            success_rate=1.0,
            failing_seeds=set(),
            run_id=run_id,
            observation_schema_hash=obs_hash,
            environment_config_hash=env_hash,
        )

    gate = compute_promotion_gate_snapshot(tmp_path, target_ep=50)
    assert gate["formal_evaluations_available"] == 5
    assert gate["decision"] == "pending"
    assert gate["ready"] is False

    # Querying the baseline checkpoint itself must also exclude it (including fallback path)
    baseline_gate = compute_promotion_gate_snapshot(tmp_path, target_ep=5)
    assert baseline_gate["formal_evaluations_available"] == 0
    assert baseline_gate["decision"] == "pending"
    assert baseline_gate["ready"] is False


def test_promotion_gate_isolates_runs_with_repeated_checkpoint_episodes(tmp_path: Path) -> None:
    _clear_server_dir_cache()
    eval_dir = tmp_path / "evaluations"
    obs_hash = "obs_hash_shared"
    env_hash = "env_hash_shared"

    # Run A has 6 formal 100% evaluations at eps 10..60
    for ep in (10, 20, 30, 40, 50, 60):
        _write_eval_file(
            eval_dir,
            f"eval_run_a_ep{ep:06d}.json",
            ep=ep,
            stage=1,
            mode="confidence",
            success_rate=1.0,
            failing_seeds=set(),
            run_id="run_a",
            observation_schema_hash=obs_hash,
            environment_config_hash=env_hash,
        )

    # Run B only has 3 formal 100% evaluations at eps 10..30
    for ep in (10, 20, 30):
        _write_checkpoint_file(
            tmp_path,
            ep,
            stage=1,
            run_id="run_b",
            observation_schema_hash=obs_hash,
            environment_config_hash=env_hash,
        )
        _write_eval_file(
            eval_dir,
            f"eval_run_b_ep{ep:06d}.json",
            ep=ep,
            stage=1,
            mode="confidence",
            success_rate=1.0,
            failing_seeds=set(),
            run_id="run_b",
            observation_schema_hash=obs_hash,
            environment_config_hash=env_hash,
        )
    # Also write a Run B checkpoint at ep=60 (without a Run B eval at ep=60) and a Run A
    # canonical evaluation-checkpoint-000060.json to verify neither all_evals nor fallback leaks.
    _write_checkpoint_file(
        tmp_path,
        60,
        stage=1,
        run_id="run_b",
        observation_schema_hash=obs_hash,
        environment_config_hash=env_hash,
    )
    _write_eval_file(
        eval_dir,
        "evaluation-checkpoint-000060.json",
        ep=60,
        stage=1,
        mode="confidence",
        success_rate=1.0,
        failing_seeds=set(),
        run_id="run_a",
        observation_schema_hash=obs_hash,
        environment_config_hash=env_hash,
        evaluation_id="eval_ep_60_canonical_run_a",
    )

    gate_b_30 = compute_promotion_gate_snapshot(tmp_path, target_ep=30)
    assert gate_b_30["formal_evaluations_available"] == 3
    assert gate_b_30["decision"] == "pending"
    assert gate_b_30["ready"] is False

    gate_b_60 = compute_promotion_gate_snapshot(tmp_path, target_ep=60)
    assert gate_b_60["formal_evaluations_available"] == 3
    assert gate_b_60["decision"] == "pending"
    assert gate_b_60["ready"] is False


def test_promotion_gate_isolates_observation_schema_and_env_config_hashes(tmp_path: Path) -> None:
    _clear_server_dir_cache()
    eval_dir = tmp_path / "evaluations"
    run_id = "run_same_id"

    # 3 evaluations matching target schema + env hashes
    for ep in (10, 20, 30):
        _write_eval_file(
            eval_dir,
            f"eval_match_ep{ep:06d}.json",
            ep=ep,
            stage=2,
            mode="confidence",
            success_rate=1.0,
            failing_seeds=set(),
            run_id=run_id,
            observation_schema_hash="obs_v2",
            environment_config_hash="env_v2",
        )
    # 2 evaluations with different observation_schema_hash
    for ep in (40, 50):
        _write_eval_file(
            eval_dir,
            f"eval_diff_obs_ep{ep:06d}.json",
            ep=ep,
            stage=2,
            mode="confidence",
            success_rate=1.0,
            failing_seeds=set(),
            run_id=run_id,
            observation_schema_hash="obs_v1",
            environment_config_hash="env_v2",
        )
    # 2 evaluations with different environment_config_hash
    for ep in (60, 70):
        _write_eval_file(
            eval_dir,
            f"eval_diff_env_ep{ep:06d}.json",
            ep=ep,
            stage=2,
            mode="confidence",
            success_rate=1.0,
            failing_seeds=set(),
            run_id=run_id,
            observation_schema_hash="obs_v2",
            environment_config_hash="env_v1",
        )

    _write_checkpoint_file(
        tmp_path,
        70,
        stage=2,
        run_id=run_id,
        observation_schema_hash="obs_v2",
        environment_config_hash="env_v2",
    )

    gate = compute_promotion_gate_snapshot(tmp_path, target_ep=70)
    assert gate["formal_evaluations_available"] == 3
    assert gate["decision"] == "pending"
    assert gate["ready"] is False


def test_promotion_gate_legacy_untagged_and_tagged_runs_never_mix(tmp_path: Path) -> None:
    _clear_server_dir_cache()
    eval_dir = tmp_path / "evaluations"

    # 6 legacy untagged evaluations (run_id/obs_hash/env_hash omitted -> None)
    for ep in (10, 20, 30, 40, 50, 60):
        _write_checkpoint_file(tmp_path, ep, stage=1)
        _write_eval_file(
            eval_dir,
            f"eval_legacy_ep{ep:06d}.json",
            ep=ep,
            stage=1,
            mode="confidence",
            success_rate=1.0,
            failing_seeds=set(),
        )

    # 2 modern tagged evaluations on the same stage
    for ep in (70, 80):
        _write_checkpoint_file(
            tmp_path,
            ep,
            stage=1,
            run_id="run_modern",
            observation_schema_hash="obs_modern",
            environment_config_hash="env_modern",
        )
        _write_eval_file(
            eval_dir,
            f"eval_modern_ep{ep:06d}.json",
            ep=ep,
            stage=1,
            mode="confidence",
            success_rate=1.0,
            failing_seeds=set(),
            run_id="run_modern",
            observation_schema_hash="obs_modern",
            environment_config_hash="env_modern",
        )

    # All 8 evaluations (6 legacy + 2 modern) remain visible for historical reporting
    all_loaded = _load_all_persisted_evaluations(tmp_path)
    assert len(all_loaded) == 8

    # Legacy checkpoint sees only the 6 legacy evaluations -> promote_ready
    legacy_gate = compute_promotion_gate_snapshot(tmp_path, target_ep=60)
    assert legacy_gate["formal_evaluations_available"] == 6
    assert legacy_gate["decision"] == "promote_ready"
    assert legacy_gate["ready"] is True

    # Modern tagged checkpoint sees only its 2 tagged evaluations -> pending (never mixes legacy)
    modern_gate = compute_promotion_gate_snapshot(tmp_path, target_ep=80)
    assert modern_gate["formal_evaluations_available"] == 2
    assert modern_gate["decision"] == "pending"
    assert modern_gate["ready"] is False

    # Partially-tagged checkpoint (run_id set, obs/env hashes None) still excludes all 6 legacy records
    _write_checkpoint_file(
        tmp_path,
        90,
        stage=1,
        run_id="run_modern",
        observation_schema_hash=None,
        environment_config_hash=None,
    )
    partial_gate = compute_promotion_gate_snapshot(tmp_path, target_ep=90)
    assert partial_gate["formal_evaluations_available"] == 0
    assert partial_gate["decision"] == "pending"
    assert partial_gate["ready"] is False


def test_stale_v1_eval_cache_is_invalidated_and_rebuilt_with_isolation_fields(tmp_path: Path) -> None:
    _clear_server_dir_cache()
    eval_dir = tmp_path / "evaluations"
    eval_path = _write_eval_file(
        eval_dir,
        "evaluation-checkpoint-000010.json",
        ep=10,
        stage=3,
        mode="confidence",
        success_rate=1.0,
        failing_seeds=set(),
        run_id="run_v2_rebuild",
        observation_schema_hash="obs_hash_abc",
        environment_config_hash="env_hash_def",
        evaluation_seed_set_id="seedset_123",
    )
    stat = eval_path.stat()
    cache_path = default_eval_cache_path(tmp_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    # Persist a stale v1 cache matching file size/mtime_ns but missing the new isolation fields
    v1_cache = {
        "version": 1,
        "fields": [
            "average_completion_steps",
            "average_reward",
            "checkpoint_episode",
            "checkpoint_id",
            "curriculum_stage",
            "evaluation_id",
            "evaluation_mode",
            "promotion_eligible",
            "success_rate",
        ],
        "record_fields": ["seed", "steps", "success", "timeout"],
        "entries": {
            eval_path.name: {
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "summary": {
                    "checkpoint_episode": 10,
                    "curriculum_stage": 3,
                    "evaluation_mode": "confidence",
                    "success_rate": 1.0,
                },
            }
        },
    }
    cache_path.write_text(json.dumps(v1_cache), encoding="utf-8")

    summaries = load_evaluation_summaries(eval_dir, cache_path)
    assert len(summaries) == 1
    assert summaries[0]["run_id"] == "run_v2_rebuild"
    assert summaries[0]["observation_schema_hash"] == "obs_hash_abc"
    assert summaries[0]["environment_config_hash"] == "env_hash_def"
    assert summaries[0]["evaluation_seed_set_id"] == "seedset_123"

    rebuilt_cache = json.loads(cache_path.read_text(encoding="utf-8"))
    assert rebuilt_cache["version"] == CACHE_VERSION == 2
    assert "run_id" in rebuilt_cache["fields"]
    assert "observation_schema_hash" in rebuilt_cache["fields"]
    assert "environment_config_hash" in rebuilt_cache["fields"]
    assert "evaluation_seed_set_id" in rebuilt_cache["fields"]

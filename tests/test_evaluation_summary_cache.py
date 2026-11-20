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
    payload = {
        "evaluation_id": f"eval_ep_{ep}_{mode}",
        "checkpoint_id": f"chk_ep_{ep}",
        "checkpoint_episode": ep,
        "curriculum_stage": stage,
        "evaluation_mode": mode,
        "promotion_eligible": mode == "confidence",
        "success_rate": success_rate,
        "average_reward": avg_reward,
        "average_completion_steps": avg_steps,
        "records": records,
    }
    path = eval_dir / filename
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_checkpoint_file(output_root: Path, ep: int, stage: int = 1) -> Path:
    cp_dir = output_root / "checkpoints"
    cp_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint_episode": ep,
        "checkpoint_id": f"chk_ep_{ep}",
        "curriculum_stage": stage,
        "evaluation_seeds": list(range(101, 111)),
        "evaluation_seed_count": 10,
        "success_rate": 0.9,
    }
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

import json
import pytest
from pathlib import Path
from sheepdog.training.stage_health import compute_stage_health_summary, _build_empty_summary


def test_empty_stage_health():
    empty = _build_empty_summary(3)
    d = empty.to_dict()
    assert d["stage"] == 3
    assert d["total_stage_checkpoints"] == 0
    assert d["status"] == "yellow"
    assert d["all_time_stage_success_rate"] == 0.0
    assert isinstance(d["prescriptive_recommendations"], list)


def test_compute_stage_health_synthetic(tmp_path: Path):
    evals_dir = tmp_path / "evaluations"
    evals_dir.mkdir(parents=True)

    # Create 5 synthetic evaluations for stage 4
    for i in range(1, 6):
        eval_payload = {
            "evaluation_id": f"eval_test_{i}",
            "policy_version": 100 + i,
            "curriculum_stage": 4,
            "checkpoint_episode": i * 10,
            "success_rate": 0.8 if i < 5 else 1.0,
            "average_completion_steps": 250.0 - i * 10,
            "average_reward": 150.0 + i * 20,
            "evaluation_mode": "confidence",
            "records": [
                {
                    "seed": 11,
                    "success": True,
                    "steps": 200,
                    "sheep_penned": 2,
                    "total_sheep": 2,
                },
                {
                    "seed": 23,
                    "success": i >= 3,
                    "steps": 220,
                    "sheep_penned": 2 if i >= 3 else 1,
                    "total_sheep": 2,
                },
            ],
        }
        with open(evals_dir / f"eval_pv_{100+i}_quick.json", "w", encoding="utf-8") as f:
            json.dump(eval_payload, f)

    res = compute_stage_health_summary(output_dir=tmp_path, target_stage=4, force_refresh=True)

    assert res["stage"] == 4
    assert res["total_stage_checkpoints"] == 5
    assert res["peak_stage_success_rate"] == 1.0
    assert res["status"] in ("green", "yellow")
    assert len(res["seed_matrix"]) == 2
    assert len(res["recent_trajectory"]) == 5
    assert res["failure_progress"]["total_failures"] == 2
    assert len(res["prescriptive_recommendations"]) > 0

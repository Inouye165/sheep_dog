import json
from pathlib import Path
from sheepdog.server import compute_promotion_gate_snapshot

def write_test_evaluation(
    output_root: Path,
    episode: int,
    stage: int = 1,
    policy_version: int = 1,
    checkpoint_id: str = "chk_test",
    seed_set_id: str = "seeds_1",
    seeds: list[int] = None,
    successes: list[bool] = None,
    timeouts: list[bool] = None,
    success_rate: float = 0.9,
    timeout_rate: float = 0.0,
    average_reward: float = 100.0,
    observation_schema_hash: str = "obs_h1",
    action_space_hash: str = "act_h1",
    environment_config_hash: str = "env_h1",
):
    if seeds is None:
        seeds = [11, 23, 37, 41, 53, 67, 71, 79, 83, 97]
    if successes is None:
        successes = [True] * len(seeds)
    if timeouts is None:
        timeouts = [False] * len(seeds)
        
    records = []
    for seed, success, timeout in zip(seeds, successes, timeouts):
        records.append({
            "seed": seed,
            "success": success,
            "timeout": timeout,
            "steps": 100,
            "stop_reason": "success" if success else "timeout" if timeout else "failed",
        })
        
    eval_summary = {
        "checkpoint_episode": episode,
        "checkpoint_id": checkpoint_id,
        "policy_version": policy_version,
        "curriculum_stage": stage,
        "evaluation_seed_set_id": seed_set_id,
        "success_rate": success_rate,
        "timeout_rate": timeout_rate,
        "average_reward": average_reward,
        "observation_schema_hash": observation_schema_hash,
        "action_space_hash": action_space_hash,
        "environment_config_hash": environment_config_hash,
        "records": records,
    }
    
    eval_dir = output_root / "evaluations"
    eval_dir.mkdir(parents=True, exist_ok=True)
    p = eval_dir / f"evaluation-checkpoint-{episode:06d}.json"
    with p.open("w", encoding="utf-8") as f:
        json.dump(eval_summary, f)
        
    checkpoint_payload = {
        "checkpoint_episode": episode,
        "checkpoint_id": checkpoint_id,
        "policy_version": policy_version,
        "curriculum_stage": stage,
        "evaluation_seed_set_id": seed_set_id,
        "evaluation_seeds": seeds,
        "evaluation_seed_count": len(seeds),
        "observation_schema_hash": observation_schema_hash,
        "action_space_hash": action_space_hash,
        "environment_config_hash": environment_config_hash,
    }
    
    chk_dir = output_root / "checkpoints"
    chk_dir.mkdir(parents=True, exist_ok=True)
    chk_p = chk_dir / f"checkpoint-{episode:06d}.json"
    with chk_p.open("w", encoding="utf-8") as f:
        json.dump(checkpoint_payload, f)


def test_seed_consistency_failure(tmp_path: Path) -> None:
    # Ep 10: 9/10, seed 11 fails
    write_test_evaluation(tmp_path, 10, successes=[False] + [True]*9)
    # Ep 20: 9/10, seed 11 fails
    write_test_evaluation(tmp_path, 20, successes=[False] + [True]*9)
    # Ep 30: 9/10, seed 11 fails
    write_test_evaluation(tmp_path, 30, successes=[False] + [True]*9)
    
    gate = compute_promotion_gate_snapshot(tmp_path, 30)
    assert gate["decision"] == "blocked"
    assert "Seed 11 passed only 0 of the last 3 required evaluations." in gate["reason"]


def test_seed_consistency_success(tmp_path: Path) -> None:
    # Ep 10: 10/10 (full success)
    write_test_evaluation(tmp_path, 10, successes=[True]*10, success_rate=1.0)
    # Ep 20: 10/10 (full success)
    write_test_evaluation(tmp_path, 20, successes=[True]*10, success_rate=1.0)
    # Ep 30: 9/10, seed 11 fails
    write_test_evaluation(tmp_path, 30, successes=[False] + [True]*9, success_rate=0.9)
    
    gate = compute_promotion_gate_snapshot(tmp_path, 30)
    assert gate["seed_gate_ok"] is True
    assert gate["decision"] == "promote_ready"


def test_incomplete_evaluation_ignored(tmp_path: Path) -> None:
    # Ep 10: 10/10
    write_test_evaluation(tmp_path, 10, successes=[True]*10, success_rate=1.0)
    # Ep 20: 9/10 (fewer seeds evaluated than checkpoint expects)
    write_test_evaluation(tmp_path, 20, successes=[True]*9, success_rate=1.0)
    
    # Gate at ep 20 should say Incomplete seed results
    gate_20 = compute_promotion_gate_snapshot(tmp_path, 20)
    assert gate_20["decision"] == "pending"
    assert gate_20["reason"] == "Incomplete seed results"
    
    # Ep 30: 10/10
    write_test_evaluation(tmp_path, 30, successes=[True]*10, success_rate=1.0)
    
    # Check that history at ep 30 ignores ep 20 and only sees ep 10 and ep 30 (length 2)
    gate_30 = compute_promotion_gate_snapshot(tmp_path, 30)
    assert gate_30["decision"] == "pending"
    assert all(len(hist) == 2 for hist in gate_30["rolling_history_results"].values())


def test_diagnostics_7_10_evaluation_matches(tmp_path: Path) -> None:
    write_test_evaluation(tmp_path, 10, successes=[False, False, False] + [True]*7, success_rate=0.7)
    
    gate = compute_promotion_gate_snapshot(tmp_path, 10)
    assert gate["seed_count"] == 10
    assert gate["success_count"] == 7
    assert gate["decision"] == "blocked"
    assert "Success rate 70% below threshold" in gate["reason"]


def test_mismatches_rejected(tmp_path: Path) -> None:
    # Write normal
    write_test_evaluation(tmp_path, 10)
    
    # 1. Checkpoint mismatch (different episode in evaluation than checkpoint)
    # Let's manually overwrite evaluation to mismatched episode
    eval_path = tmp_path / "evaluations" / "evaluation-checkpoint-000010.json"
    with eval_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    data["checkpoint_episode"] = 999
    with eval_path.open("w", encoding="utf-8") as f:
        json.dump(data, f)
        
    gate = compute_promotion_gate_snapshot(tmp_path, 10)
    assert gate["decision"] == "pending"
    assert gate["reason"] == "Checkpoint mismatch"
    
    # Restore evaluation episode and test policy version mismatch
    data["checkpoint_episode"] = 10
    data["policy_version"] = 999
    with eval_path.open("w", encoding="utf-8") as f:
        json.dump(data, f)
        
    gate = compute_promotion_gate_snapshot(tmp_path, 10)
    assert gate["decision"] == "pending"
    assert gate["reason"] == "Policy-version mismatch"


def test_stage_or_seed_set_change_resets_history(tmp_path: Path) -> None:
    # Ep 10: Stage 1, seeds_1, 10/10
    write_test_evaluation(tmp_path, 10, stage=1, seed_set_id="seeds_1", successes=[True]*10, success_rate=1.0)
    # Ep 20: Stage 2, seeds_1, 10/10
    write_test_evaluation(tmp_path, 20, stage=2, seed_set_id="seeds_1", successes=[True]*10, success_rate=1.0)
    
    gate = compute_promotion_gate_snapshot(tmp_path, 20)
    # At ep 20 (Stage 2), rolling history has length 1 (only ep 20).
    assert gate["decision"] == "pending"
    assert all(len(hist) == 1 for hist in gate["rolling_history_results"].values())


def test_overall_thresholds_work(tmp_path: Path) -> None:
    # 1. Success rate too low
    write_test_evaluation(tmp_path, 10, success_rate=0.8)
    gate = compute_promotion_gate_snapshot(tmp_path, 10)
    assert gate["decision"] == "blocked"
    assert "Success rate" in gate["reason"]
    
    # 2. Timeout rate too high
    write_test_evaluation(tmp_path, 20, timeout_rate=0.2)
    gate = compute_promotion_gate_snapshot(tmp_path, 20)
    assert gate["decision"] == "blocked"
    assert "Timeout rate" in gate["reason"]

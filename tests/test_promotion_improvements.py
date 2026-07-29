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
    seeds: list[int] | None = None,
    successes: list[bool] | None = None,
    timeouts: list[bool] | None = None,
    success_rate: float | None = None,
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
    if success_rate is None:
        success_rate = sum(1 for s in successes if s) / len(seeds)
        
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


def test_example_a_rolling_promotion_ready(tmp_path: Path) -> None:
    # 10/10, 9/10 (seed 97 fails), 8/10 (seeds 83, 71 fail) -> Combined 27/30 (90%)
    write_test_evaluation(tmp_path, 10, successes=[True]*10)
    write_test_evaluation(tmp_path, 20, successes=[True]*8 + [False] + [True])
    write_test_evaluation(tmp_path, 30, successes=[True]*6 + [False, False] + [True]*2)
    
    gate = compute_promotion_gate_snapshot(tmp_path, 30)
    assert gate["decision"] == "promote_ready"
    assert gate["aggregate_success_rate"] == 0.9
    assert gate["latest_floor_passed"] is True
    assert gate["recent_qualifying_checkpoints"] == 2


def test_example_b_latest_floor_blocked(tmp_path: Path) -> None:
    # 10/10, 10/10, 7/10 -> Combined 27/30 (90%), but latest is 70% < 80%
    write_test_evaluation(tmp_path, 10, successes=[True]*10)
    write_test_evaluation(tmp_path, 20, successes=[True]*10)
    write_test_evaluation(tmp_path, 30, successes=[True]*7 + [False]*3)
    
    gate = compute_promotion_gate_snapshot(tmp_path, 30)
    assert gate["decision"] == "blocked"
    assert gate["latest_floor_passed"] is False
    assert any("Latest evaluation is 70%" in r for r in gate["blocking_reasons"])


def test_example_c_all_90_promotion_ready(tmp_path: Path) -> None:
    # 9/10 (seed 97), 9/10 (seed 83), 9/10 (seed 71) -> Combined 27/30 (90%)
    write_test_evaluation(tmp_path, 10, successes=[True]*9 + [False])
    write_test_evaluation(tmp_path, 20, successes=[True]*8 + [False] + [True])
    write_test_evaluation(tmp_path, 30, successes=[True]*7 + [False] + [True]*2)
    
    gate = compute_promotion_gate_snapshot(tmp_path, 30)
    assert gate["decision"] == "promote_ready"
    assert gate["aggregate_success_rate"] == 0.9


def test_example_d_varied_success_promotion_ready(tmp_path: Path) -> None:
    # 10/10, 8/10 (seeds 83, 71 fail), 9/10 (seed 97 fails) -> Combined 27/30 (90%)
    write_test_evaluation(tmp_path, 10, successes=[True]*10)
    write_test_evaluation(tmp_path, 20, successes=[True]*6 + [False, False] + [True]*2)
    write_test_evaluation(tmp_path, 30, successes=[True]*9 + [False])
    
    gate = compute_promotion_gate_snapshot(tmp_path, 30)
    assert gate["decision"] == "promote_ready"
    assert gate["aggregate_success_rate"] == 0.9


def test_example_e_aggregate_below_90_blocked(tmp_path: Path) -> None:
    # 10/10, 10/10, 8/10, 8/10, 8/10 -> Combined 44/50 (88%)
    write_test_evaluation(tmp_path, 10, successes=[True]*10)
    write_test_evaluation(tmp_path, 20, successes=[True]*10)
    write_test_evaluation(tmp_path, 30, successes=[True]*6 + [False, False] + [True]*2)
    write_test_evaluation(tmp_path, 40, successes=[True]*4 + [False, False] + [True]*4)
    write_test_evaluation(tmp_path, 50, successes=[False, False] + [True]*8)
    
    gate = compute_promotion_gate_snapshot(tmp_path, 50)
    assert gate["decision"] == "blocked"
    assert gate["aggregate_success_rate"] == 0.88
    assert any("Aggregate success is 88%" in r for r in gate["blocking_reasons"])


def test_example_f_repeated_seed_failure_blocked(tmp_path: Path) -> None:
    # Seed index 3 (41) fails in 3 of the last 5 evaluations
    seeds = [11, 23, 37, 41, 53, 67, 71, 79, 83, 97]
    def make_succ(fail_index):
        s = [True]*10
        if fail_index is not None:
            s[fail_index] = False
        return s

    write_test_evaluation(tmp_path, 10, seeds=seeds, successes=make_succ(3))
    write_test_evaluation(tmp_path, 20, seeds=seeds, successes=make_succ(3))
    write_test_evaluation(tmp_path, 30, seeds=seeds, successes=make_succ(3))
    write_test_evaluation(tmp_path, 40, seeds=seeds, successes=make_succ(None))
    write_test_evaluation(tmp_path, 50, seeds=seeds, successes=make_succ(None))
    
    gate = compute_promotion_gate_snapshot(tmp_path, 50)
    assert gate["decision"] == "blocked"
    assert 41 in gate["blocking_seeds"]
    assert any("Seed 41 failed in 3 of the last 5" in r for r in gate["blocking_reasons"])


def test_incomplete_evaluation_ignored(tmp_path: Path) -> None:
    write_test_evaluation(tmp_path, 10, successes=[True]*10, success_rate=1.0)
    write_test_evaluation(tmp_path, 20, successes=[True]*9, success_rate=1.0)
    
    gate_20 = compute_promotion_gate_snapshot(tmp_path, 20)
    assert gate_20["decision"] == "pending"
    assert any("Incomplete seed results" in r for r in gate_20["blocking_reasons"])


def test_mismatches_rejected(tmp_path: Path) -> None:
    write_test_evaluation(tmp_path, 10)
    
    eval_path = tmp_path / "evaluations" / "evaluation-checkpoint-000010.json"
    with eval_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    data["checkpoint_episode"] = 999
    with eval_path.open("w", encoding="utf-8") as f:
        json.dump(data, f)
        
    gate = compute_promotion_gate_snapshot(tmp_path, 10)
    assert gate["decision"] == "pending"
    assert any("Checkpoint mismatch" in r for r in gate["blocking_reasons"])

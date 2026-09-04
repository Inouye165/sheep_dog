"""Unit and integration tests for Evaluation Replay Retention Manager."""

from pathlib import Path
import json
from unittest.mock import MagicMock

import pytest

from sheepdog.config import LabConfig
from sheepdog.evaluation.retention import (
    EvaluationReplayRetentionManager,
    EvaluationRetentionPolicy,
)
from sheepdog.evaluation.evaluator import Evaluator
from sheepdog.policies.base import Policy


class DummyPolicy(Policy):
    @property
    def name(self) -> str:
        return "dummy_policy"

    def select_actions(self, environment, deterministic: bool = True) -> tuple[str, ...]:
        return tuple("wait" for _ in range(environment.dog_count))


def test_retention_policy_rules(tmp_path: Path) -> None:
    eval_dir = tmp_path / "evaluations"
    eval_dir.mkdir(parents=True)
    replays_dir = eval_dir / "replays"
    replays_dir.mkdir(parents=True)

    mgr = EvaluationReplayRetentionManager(
        eval_dir,
        policy=EvaluationRetentionPolicy(save_first=True, milestone_interval=25, keep_latest=True),
    )

    # Evaluation #1: Should be retained as "first"
    p1 = replays_dir / "chk-01.json"
    p1.write_text("{}", encoding="utf-8")
    res1 = mgr.register_and_prune("eval-1", 1, 100, [p1])
    assert res1["status"] == "first"
    assert p1.exists()

    # Evaluation #2: Should be retained as "latest"
    p2 = replays_dir / "chk-02.json"
    p2.write_text("{}", encoding="utf-8")
    res2 = mgr.register_and_prune("eval-2", 2, 200, [p2])
    assert res2["status"] == "latest"
    assert p1.exists()  # #1 still kept
    assert p2.exists()

    # Evaluation #3: Now #3 is "latest", #2 should be pruned, #1 kept!
    p3 = replays_dir / "chk-03.json"
    p3.write_text("{}", encoding="utf-8")
    res3 = mgr.register_and_prune("eval-3", 3, 300, [p3])
    assert res3["status"] == "latest"
    assert p1.exists()  # #1 kept
    assert not p2.exists()  # #2 pruned!
    assert p3.exists()  # #3 kept

    # Pin evaluation #3
    assert mgr.pin_evaluation("eval-3", pinned=True)
    assert mgr.is_pinned("eval-3")

    # Evaluation #4: #4 becomes latest, #3 was pinned so it must NOT be pruned!
    p4 = replays_dir / "chk-04.json"
    p4.write_text("{}", encoding="utf-8")
    res4 = mgr.register_and_prune("eval-4", 4, 400, [p4])
    assert res4["status"] == "latest"
    assert p1.exists()  # #1 kept
    assert p3.exists()  # #3 kept because it was pinned!
    assert p4.exists()  # #4 kept (latest)

    # Evaluation #25: Milestone
    p25 = replays_dir / "chk-25.json"
    p25.write_text("{}", encoding="utf-8")
    res25 = mgr.register_and_prune("eval-25", 25, 2500, [p25])
    assert res25["status"] == "milestone"
    assert p25.exists()

    # Evaluation #26: #25 milestone must be retained, #4 unpinned was pruned
    p26 = replays_dir / "chk-26.json"
    p26.write_text("{}", encoding="utf-8")
    res26 = mgr.register_and_prune("eval-26", 26, 2600, [p26])
    assert res26["status"] == "latest"
    assert p1.exists()
    assert p3.exists()  # pinned
    assert not p4.exists()  # pruned
    assert p25.exists()  # milestone
    assert p26.exists()  # latest


def test_evaluator_saves_replays_and_attaches_retention(tmp_path: Path) -> None:
    config = LabConfig()
    eval_dir = tmp_path / "evaluations"
    evaluator = Evaluator(config, eval_dir)

    summary, json_path, csv_path = evaluator.evaluate(
        DummyPolicy(),
        (11, 12),
        checkpoint_episode=100,
        capture_replays=True,
        evaluation_index=1,
    )

    assert summary.evaluation_index == 1
    assert summary.retention_status == "first"
    assert len(summary.records) == 2
    assert summary.records[0].replay_path != ""
    assert Path(summary.records[0].replay_path).exists()

    # Check that saved JSON includes retention fields
    saved_data = json.loads(json_path.read_text(encoding="utf-8"))
    assert saved_data["evaluation_index"] == 1
    assert saved_data["retention_status"] == "first"


def test_server_replay_and_pin_endpoints(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    evals_dir = artifacts / "evaluations"
    replays_dir = evals_dir / "replays"
    replays_dir.mkdir(parents=True)

    test_replay = replays_dir / "checkpoint-000100-seed-000011.json"
    test_replay.write_text(json.dumps({"seed": 11, "frames": [{"test": 1}]}), encoding="utf-8")

    eval_json = evals_dir / "eval_test_01.json"
    eval_json.write_text(
        json.dumps({
            "evaluation_id": "eval_test_01",
            "evaluation_index": 25,
            "checkpoint_episode": 100,
            "records": [{"seed": 11, "replay_path": str(test_replay)}],
        }),
        encoding="utf-8",
    )

    from sheepdog.server import TrainingManager
    from sheepdog.config import TrainingConfig
    from unittest.mock import patch

    config = LabConfig(
        training=TrainingConfig(
            output_dir=str(artifacts),
            web_export_dir=str(tmp_path / "web"),
        )
    )
    with patch("sheepdog.server.LabConfig", return_value=config):
        manager = TrainingManager()
        evals = manager.get_recent_evaluations()
        assert len(evals) == 1
        assert evals[0]["is_milestone"] is True
        assert evals[0]["pinned"] is False

        # Pin the evaluation
        from sheepdog.evaluation.retention import EvaluationReplayRetentionManager
        ret_mgr = EvaluationReplayRetentionManager(evals_dir)
        assert ret_mgr.pin_evaluation("eval_test_01", pinned=True)

        evals_after = manager.get_recent_evaluations()
        assert evals_after[0]["pinned"] is True

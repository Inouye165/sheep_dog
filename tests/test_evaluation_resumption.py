"""Tests for evaluation interruption and resumption."""

from pathlib import Path
from unittest.mock import patch

import pytest

from sheepdog.config import LabConfig
from sheepdog.evaluation.evaluator import EvaluationInterruptedError, Evaluator
from sheepdog.policies.heuristic import InstinctOnlyPolicy


def test_evaluator_interruption_saves_progress_and_resumes(tmp_path: Path) -> None:
    """Verify that an interrupted evaluation saves progress and resumes seamlessly."""
    config = LabConfig()
    eval_dir = tmp_path / "evaluations"
    evaluator = Evaluator(config, eval_dir)

    policy = InstinctOnlyPolicy()
    seeds = (101, 102, 103)

    # 1. First run: stop after seed 1 completes
    call_count = 0

    def stop_after_seed_1() -> bool:
        nonlocal call_count
        call_count += 1
        return call_count > 1

    with pytest.raises(EvaluationInterruptedError) as exc_info:
        evaluator.evaluate(
            policy,
            seeds,
            checkpoint_episode=10,
            evaluation_mode="test_interrupt",
            checkpoint_id="chk_test_1",
            should_stop=stop_after_seed_1,
        )

    # Verify inprogress file exists
    assert exc_info.value.inprogress_path is not None
    inprogress_path = exc_info.value.inprogress_path
    assert inprogress_path.exists()

    # 2. Second run: resume and complete remaining seeds
    # Track environment runs to ensure seed 1 is NOT re-run
    executed_seeds = []

    from sheepdog.environment import SheepdogEnvironment

    original_env_run = SheepdogEnvironment.run_policy

    def tracking_run_policy(self, pol, seed, *args, **kwargs):
        executed_seeds.append(seed)
        return original_env_run(self, pol, seed, *args, **kwargs)

    with patch.object(SheepdogEnvironment, "run_policy", side_effect=tracking_run_policy, autospec=True):
        summary, json_path, csv_path = evaluator.evaluate(
            policy,
            seeds,
            checkpoint_episode=10,
            evaluation_mode="test_interrupt",
            checkpoint_id="chk_test_1",
            should_stop=lambda: False,
        )

    # Seed 1 was already completed, so only seeds 102 and 103 should have executed!
    assert executed_seeds == [102, 103]

    # Summary should contain all 3 seeds
    assert len(summary.records) == 3
    assert [r.seed for r in summary.records] == [101, 102, 103]

    # inprogress file should be cleaned up on completion
    assert not inprogress_path.exists()
    assert json_path.exists()
    assert csv_path.exists()

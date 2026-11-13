"""Focused unit test for the head/tail evaluation runner script."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from scripts.run_head_tail_evaluation import (
    compute_file_sha256,
    get_dependency_versions,
    get_git_info,
    load_target_configuration,
    run_head_tail_evaluation,
    serialize_replay_to_disk,
    verify_phase2_invariance,
)

from sheepdog.config import LabConfig
from sheepdog.entities import EpisodeStats
from sheepdog.environment import EpisodeResult


def test_compute_file_sha256(tmp_path: Path) -> None:
    test_file = tmp_path / "sample.bin"
    test_file.write_bytes(b"hello sheepdog")
    digest = compute_file_sha256(test_file)
    assert len(digest) == 64
    assert isinstance(digest, str)


def test_get_git_info() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    info = get_git_info(repo_root)
    assert "source_revision" in info
    assert "is_dirty" in info
    assert "status_lines_count" in info
    assert "untracked_count" in info
    assert "modified_count" in info
    assert info["source_revision"] is not None


def test_get_dependency_versions() -> None:
    deps = get_dependency_versions()
    assert "python" in deps
    assert "torch" in deps
    assert "numpy" in deps
    assert "stable_baselines3" in deps
    assert "sb3_contrib" in deps


def test_load_target_configuration() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    chk_json = (
        repo_root
        / "artifacts"
        / "checkpoints"
        / "checkpoint-chk_run_20260817_063453_91f4_pv_10369_ts_339740239.json"
    )
    if chk_json.exists():
        cfg, meta = load_target_configuration(chk_json)
        assert cfg.environment.width == 144
        assert cfg.environment.height == 104
        assert cfg.environment.dogs == 3
        assert cfg.environment.sheep == 6
        assert meta["checkpoint_id"] == "chk_run_20260817_063453_91f4_pv_10369_ts_339740239"


def test_verify_phase2_invariance_mismatch_raises() -> None:
    res_off = MagicMock(spec=EpisodeResult)
    res_off.replay = []
    res_on = MagicMock(spec=EpisodeResult)
    res_on.replay = [MagicMock()]

    rng_off = {
        "terminated": True,
        "timeout": False,
        "stopped": False,
        "success": True,
        "stop_reason": "success",
    }
    rng_on = {
        "terminated": True,
        "timeout": False,
        "stopped": False,
        "success": True,
        "stop_reason": "success",
    }

    with pytest.raises(AssertionError, match="Episode length mismatch"):
        verify_phase2_invariance(res_off, rng_off, res_on, rng_on, H=2, L=2)


def test_discount_provenance_serialization(tmp_path: Path) -> None:
    """Verify discount provenance, evaluation_gamma, checkpoint_gamma, and gamma_mismatch flag."""
    res = MagicMock(spec=EpisodeResult)
    res.replay = []
    res.initial_snapshot = None
    res.final_snapshot = None
    res.stats = None

    config = LabConfig()
    assert config.training.gamma == 0.99

    out_file = tmp_path / "replay_test.json.gz"
    size = serialize_replay_to_disk(
        res,
        config,
        out_file,
        seed=42,
        checkpoint_id="chk_test",
        run_id="run_test",
        checkpoint_gamma=0.997,
        discount_timebase="sequential_shared_dog",
    )
    assert size > 0
    assert out_file.exists()

    with gzip.open(out_file, "rt", encoding="utf-8") as gz:
        data = json.load(gz)

    assert data["discount_timebase"] == "sequential_shared_dog"
    assert data["checkpoint_gamma"] == 0.997
    assert data["evaluation_gamma"] == 0.99
    assert data["gamma_mismatch"] is True

    # Case 2: Matching gamma
    cfg_dict = config.to_dict()
    cfg_dict["training"]["gamma"] = 0.997
    config2 = LabConfig.from_dict(cfg_dict)

    out_file2 = tmp_path / "replay_test2.json.gz"
    serialize_replay_to_disk(
        res,
        config2,
        out_file2,
        seed=42,
        checkpoint_id="chk_test",
        run_id="run_test",
        checkpoint_gamma=0.997,
        discount_timebase="sequential_shared_dog",
    )
    with gzip.open(out_file2, "rt", encoding="utf-8") as gz:
        data2 = json.load(gz)
    assert data2["gamma_mismatch"] is False


def test_verify_phase2_invariance_numpy_5_element() -> None:
    """Verify that all 5 elements of the NumPy RNG state tuple are checked for invariance."""
    res_off = MagicMock(spec=EpisodeResult)
    res_off.replay = []
    res_on = MagicMock(spec=EpisodeResult)
    res_on.replay = []

    base_np_state = ("MT19937", np.zeros(624, dtype=np.uint32), 624, 0, 0.0)
    base_py_state = (3, (1, 2, 3), None)
    base_torch_state = torch.tensor([1, 2, 3], dtype=torch.uint8)

    base_rng = {
        "terminated": True,
        "timeout": False,
        "stopped": False,
        "success": True,
        "stop_reason": "success",
        "py_random": base_py_state,
        "np_random": base_np_state,
        "torch_random": base_torch_state,
        "env_rng": (3, (1, 2, 3), None),
    }

    # Identical: should pass
    rng_off = dict(base_rng)
    rng_on = dict(base_rng)
    res = verify_phase2_invariance(res_off, rng_off, res_on, rng_on, H=2, L=2)
    assert res["status"] == "passed"

    # Mismatch in element 2 (pos)
    rng_on_pos = dict(base_rng)
    rng_on_pos["np_random"] = ("MT19937", np.zeros(624, dtype=np.uint32), 100, 0, 0.0)
    with pytest.raises(AssertionError, match="NumPy random state diverged"):
        verify_phase2_invariance(res_off, rng_off, res_on, rng_on_pos, H=2, L=2)

    # Mismatch in element 3 (has_gauss)
    rng_on_gauss = dict(base_rng)
    rng_on_gauss["np_random"] = ("MT19937", np.zeros(624, dtype=np.uint32), 624, 1, 0.0)
    with pytest.raises(AssertionError, match="NumPy random state diverged"):
        verify_phase2_invariance(res_off, rng_off, res_on, rng_on_gauss, H=2, L=2)

    # Mismatch in element 4 (cached_gaussian)
    rng_on_val = dict(base_rng)
    rng_on_val["np_random"] = ("MT19937", np.zeros(624, dtype=np.uint32), 624, 0, 1.23)
    with pytest.raises(AssertionError, match="NumPy random state diverged"):
        verify_phase2_invariance(res_off, rng_off, res_on, rng_on_val, H=2, L=2)


def test_directory_collision_rejection(tmp_path: Path) -> None:
    """Verify that existing non-empty directory is rejected unless allowed."""
    repo_root = Path(__file__).resolve().parent.parent
    colliding_dir = tmp_path / "existing_exp"
    colliding_dir.mkdir()
    (colliding_dir / "marker.txt").write_text("existing content")

    # Should raise FileExistsError without flags
    with pytest.raises(FileExistsError, match="exists and is not empty"):
        run_head_tail_evaluation(
            colliding_dir,
            repo_root,
            allow_existing_dir=False,
            reuse_existing_replays=False,
        )


def test_limits_enforcement(tmp_path: Path) -> None:
    """Verify that episode, step, and time limits raise appropriate errors."""
    repo_root = Path(__file__).resolve().parent.parent

    # Mock policy loading so limit testing runs fast
    with patch("scripts.run_head_tail_evaluation.NeuralPolicy.load") as mock_load, patch(
        "scripts.run_head_tail_evaluation.run_single_episode"
    ) as mock_run, patch(
        "scripts.run_head_tail_evaluation.verify_phase2_invariance"
    ):
        mock_policy = MagicMock()
        mock_policy.checkpoint_gamma = 0.997
        mock_policy.config.hidden_sizes = [256, 256, 128]
        mock_policy.config.observation_size = 54
        mock_policy.config.action_size = 9
        mock_load.return_value = mock_policy

        mock_stats = EpisodeStats(
            success=False,
            timeout=False,
            stopped=True,
            stop_reason="no-progress",
            sheep_penned=5,
            reward_total=0.0,
            steps=1,
        )
        dummy_frame = MagicMock()
        dummy_frame.to_dict.return_value = {"step": 1, "actions": ["wait", "wait", "wait"]}

        mock_result = MagicMock()
        mock_result.replay = [dummy_frame]
        mock_result.stats = mock_stats
        mock_result.initial_snapshot = None
        mock_result.final_snapshot = None
        mock_rng = {
            "terminated": True, "timeout": False, "stopped": True, "success": False, "stop_reason": "no-progress",
            "elapsed_seconds": 0.1,
            "py_random": (3, (1, 2), None),
            "np_random": ("MT19937", np.zeros(624, dtype=np.uint32), 624, 0, 0.0),
            "torch_random": torch.tensor([1], dtype=torch.uint8),
            "env_rng": (3, (1, 2), None),
        }
        mock_run.return_value = (mock_result, mock_rng)

        # Test episode budget limit (fails at start of phase 3)
        out_dir_ep = tmp_path / "limits_ep"
        with pytest.raises(ValueError, match="Episode budget exceeded"):
            run_head_tail_evaluation(
                out_dir_ep,
                repo_root,
                max_episodes=1,  # Control (1) + Treatment (2) exceeds 1
            )

        # Test world steps budget limit
        out_dir_steps = tmp_path / "limits_steps"
        with pytest.raises(ValueError, match="World step budget exceeded"):
            run_head_tail_evaluation(
                out_dir_steps,
                repo_root,
                max_world_steps=1,  # Total steps is 2 (1 control + 1 treatment) > 1
            )

        # Test wallclock timeout limit
        out_dir_time = tmp_path / "limits_time"
        with pytest.raises(TimeoutError, match="Wall-clock time limit exceeded"):
            run_head_tail_evaluation(
                out_dir_time,
                repo_root,
                max_wallclock_seconds=-1.0,
            )


def test_reuse_accounting(tmp_path: Path) -> None:
    """Verify that reused replays are clearly marked with reused=True and unknown elapsed time."""
    repo_root = Path(__file__).resolve().parent.parent
    chk_json = (
        repo_root
        / "artifacts"
        / "checkpoints"
        / "checkpoint-chk_run_20260817_063453_91f4_pv_10369_ts_339740239.json"
    )
    if not chk_json.exists():
        pytest.skip("Checkpoint JSON not found")

    out_dir = tmp_path / "reuse_test_exp"
    replays_dir = out_dir / "replays"
    replays_dir.mkdir(parents=True, exist_ok=True)

    # Create dummy replay for seed 11
    dummy_replay_data = {
        "seed": 11,
        "stats": {"success": True, "timeout": False, "stopped": False, "sheep_penned": 6, "reward_total": 100.0},
        "frames": [{"step": 1, "actions": ["wait", "wait", "wait"]}],
    }
    dummy_file = replays_dir / "checkpoint-040575-seed-000011.json.gz"
    with gzip.open(dummy_file, "wt", encoding="utf-8") as gf:
        json.dump(dummy_replay_data, gf)

    # Mock NeuralPolicy and SheepdogEnvironment so we don't do real RL rollout
    with patch("scripts.run_head_tail_evaluation.NeuralPolicy.load") as mock_load, patch(
        "scripts.run_head_tail_evaluation.run_single_episode"
    ) as mock_run, patch(
        "scripts.run_head_tail_evaluation.verify_phase2_invariance"
    ) as mock_inv, patch(
        "scripts.run_head_tail_evaluation.analyze_episode_temporal_interactions"
    ) as mock_analyze:
        mock_policy = MagicMock()
        mock_policy.checkpoint_gamma = 0.997
        mock_policy.config.hidden_sizes = [256, 256, 128]
        mock_policy.config.observation_size = 54
        mock_policy.config.action_size = 9
        mock_load.return_value = mock_policy

        mock_stats = EpisodeStats(
            success=False,
            timeout=False,
            stopped=True,
            stop_reason="no-progress",
            sheep_penned=5,
            reward_total=0.0,
            steps=1,
        )
        dummy_frame = MagicMock()
        dummy_frame.to_dict.return_value = {"step": 1, "actions": ["wait", "wait", "wait"]}

        mock_result = MagicMock()
        mock_result.replay = [dummy_frame]
        mock_result.stats = mock_stats
        mock_result.initial_snapshot = None
        mock_result.final_snapshot = None
        mock_rng = {
            "terminated": True, "timeout": False, "stopped": True, "success": False, "stop_reason": "no-progress",
            "elapsed_seconds": 1.23,
            "py_random": (3, (1, 2), None),
            "np_random": ("MT19937", np.zeros(624, dtype=np.uint32), 624, 0, 0.0),
            "torch_random": torch.tensor([1], dtype=torch.uint8),
            "env_rng": (3, (1, 2), None),
        }
        mock_run.return_value = (mock_result, mock_rng)
        mock_inv.return_value = {
            "status": "passed", "episode_length": 1, "success": False, "timeout": False,
            "stopped": True, "stop_reason": "no-progress", "head_frames_retained": 1,
            "tail_frames_retained": 1, "gap_marker_found": False, "gap_marker_details": {},
            "invariance_verified": True,
        }

        mock_analysis = MagicMock()
        mock_analysis.intervals = []
        mock_analysis.evidence_audit = MagicMock(has_recorded_step_telemetry=True)
        mock_analyze.return_value = mock_analysis

        manifest = run_head_tail_evaluation(
            out_dir,
            repo_root,
            reuse_existing_replays=True,
            allow_existing_dir=True,
        )

        episodes = manifest["phase3_evaluation"]["episodes"]
        ep11 = next(ep for ep in episodes if ep["seed"] == 11)
        assert ep11["reused"] is True
        assert ep11["elapsed_seconds"] is None
        assert ep11["elapsed_seconds_status"] == "unknown"

        ep53 = next(ep for ep in episodes if ep["seed"] == 53)
        assert ep53["reused"] is False
        assert ep53["elapsed_seconds"] is not None
        assert ep53["elapsed_seconds_status"] == "measured"


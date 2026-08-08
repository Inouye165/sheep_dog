import tempfile
from pathlib import Path
import pytest
from unittest.mock import MagicMock

from sheepdog.training.episode_store import EpisodeStore
from sheepdog.training.telemetry import CurriculumTelemetryManager
from sheepdog.training.maskable_ppo import _TrainingProgressCallback


def test_authentic_global_timestep_recording():
    """Test that newly completed episodes save authentic non-null global_timestep from trainer counter."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        db_path = Path(tmpdir) / "test-telemetry.sqlite"
        store = EpisodeStore(db_path)
        
        # Insert an episode with explicit trainer global_timestep = 45123 (distinct from episode number 5)
        store.add_episode({
            "global_environment_episode": 5,
            "stage": 1,
            "reward": 15.5,
            "penned": 1,
            "total_sheep": 1,
            "success": True,
            "status": "SUCCESS",
            "global_timestep": 45123,
            "policy_version": 2,
        })
        store.flush()
        
        res = store.get_episodes(stage=1)
        episodes = res.get("episodes", []) if isinstance(res, dict) else res
        assert len(episodes) == 1
        ep = episodes[0]
        assert ep["global_environment_episode"] == 5
        assert ep["global_timestep"] == 45123
        assert ep["global_timestep"] != ep["global_environment_episode"]
        store.close()


def test_vector_env_multi_completion_handling():
    """Test that multiple episode completions from vector environments share the trainer timestep."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        db_path = Path(tmpdir) / "test-telemetry.sqlite"
        store = EpisodeStore(db_path)
        
        trainer_ts = 90240
        # Two episodes finishing on the exact same step in vector env
        for ep_num in [10, 11]:
            store.add_episode({
                "global_environment_episode": ep_num,
                "stage": 1,
                "reward": 10.0,
                "penned": 1,
                "total_sheep": 1,
                "success": True,
                "status": "SUCCESS",
                "global_timestep": trainer_ts,
                "policy_version": 3,
            })
        store.flush()
            
        res = store.get_episodes(stage=1)
        episodes = res.get("episodes", []) if isinstance(res, dict) else res
        assert len(episodes) == 2
        assert episodes[0]["global_timestep"] == 90240
        assert episodes[1]["global_timestep"] == 90240
        store.close()


def test_legacy_null_values_supported():
    """Test that legacy SQLite rows with global_timestep = null remain supported without errors."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        db_path = Path(tmpdir) / "test-telemetry.sqlite"
        store = EpisodeStore(db_path)
        
        # Legacy row with global_timestep omitted / None
        store.add_episode({
            "global_environment_episode": 1,
            "stage": 1,
            "reward": -50.0,
            "penned": 0,
            "total_sheep": 1,
            "success": False,
            "status": "STOPPED",
            "global_timestep": None,
        })
        store.flush()
        
        res = store.get_episodes(stage=1)
        episodes = res.get("episodes", []) if isinstance(res, dict) else res
        assert len(episodes) == 1
        assert episodes[0]["global_timestep"] is None
        store.close()


def test_progress_callback_emits_authentic_timestep():
    """Test that _TrainingProgressCallback emits current_global_ts in episode_complete events."""
    emitted = []
    def mock_emit(payload):
        emitted.append(payload)
        
    cb = _TrainingProgressCallback(
        mock_emit,
        should_stop=None,
        report_interval=100,
        total_timesteps=10000,
        starting_total_episodes=0,
        batch_total_episodes=10,
        batch_total_timesteps=10000,
        completed_timesteps=45000,
        completed_segments=1,
        segment_index=1,
        policy_version=2,
        starting_total=10,
    )
    cb.num_timesteps = 2048
    
    cb.model = MagicMock()
    cb.model._n_updates = 0
    mock_env = MagicMock()
    mock_env.get_attr.return_value = [5]
    cb.model.get_env.return_value = mock_env
    
    cb.locals = {
        "dones": [True],
        "infos": [{"episode": {"r": 12.0, "l": 200, "success": True, "penned": 1, "total_sheep": 1, "status": "SUCCESS"}}],
    }
    cb._on_step()
    
    ep_events = [e for e in emitted if e.get("phase") == "episode_complete"]
    assert len(ep_events) == 1
    assert ep_events[0]["global_timestep"] == 47048
    assert ep_events[0]["episode"] == 15

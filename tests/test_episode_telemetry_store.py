"""Unit tests for durable SQLite episode telemetry store, queue saturation, error safety, and 50k benchmark."""

import os
import tempfile
import time
from pathlib import Path

import pytest
from sheepdog.training.episode_store import EpisodeStore


@pytest.fixture
def temp_store():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        db_path = Path(tmpdir) / "test-telemetry.sqlite"
        store = EpisodeStore(db_path=db_path)
        try:
            yield store
        finally:
            store.close()


def test_1_episode_complete_creates_sqlite_record(temp_store):
    record = {
        "event_key": "run1_ep_1_env_0_seed_42",
        "run_id": "run1",
        "global_environment_episode": 1,
        "episode_in_stage": 1,
        "curriculum_stage": 8,
        "global_timestep": 1000,
        "policy_version": 5,
        "reward": 450.5,
        "result": "SUCCESS",
        "success": True,
        "penned": 4,
        "total_sheep": 4,
        "steps": 120,
        "seed": 42,
    }
    temp_store.add_episode(record)
    temp_store.flush()

    res = temp_store.get_episodes(limit=10)
    assert res["total_matching"] == 1
    assert len(res["episodes"]) == 1
    ep = res["episodes"][0]
    assert ep["event_key"] == "run1_ep_1_env_0_seed_42"
    assert ep["run_id"] == "run1"
    assert ep["reward"] == 450.5
    assert ep["success"] is True
    assert ep["sheep_penned"] == 4


def test_2_duplicate_processing_creates_no_duplicate_record(temp_store):
    record = {
        "event_key": "dup_event_key_1",
        "run_id": "run1",
        "global_environment_episode": 2,
        "curriculum_stage": 8,
        "reward": 200.0,
        "success": True,
    }
    temp_store.add_episode(record)
    temp_store.add_episode(record)
    temp_store.flush()

    res = temp_store.get_episodes(limit=10)
    assert res["total_matching"] == 1
    assert len(res["episodes"]) == 1


def test_3_all_required_fields_stored_correctly(temp_store):
    record = {
        "event_key": "event_all_fields",
        "run_id": "run_test_all",
        "session_id": "sess_123",
        "global_environment_episode": 10,
        "episode_in_stage": 5,
        "curriculum_stage": 3,
        "global_timestep": 50000,
        "policy_version": 12,
        "completed_at": "2026-08-03T12:00:00Z",
        "active_runtime_seconds_total": 123.45,
        "reward": 512.75,
        "status": "SUCCESS",
        "success": True,
        "penned": 4,
        "total_sheep": 4,
        "length": 88,
        "seed": 999,
        "checkpoint_id": "chk_10",
    }
    temp_store.add_episode(record)
    temp_store.flush()

    res = temp_store.get_episodes(limit=10)
    ep = res["episodes"][0]
    assert ep["session_id"] == "sess_123"
    assert ep["global_environment_episode"] == 10
    assert ep["curriculum_stage"] == 3
    assert ep["global_timestep"] == 50000
    assert ep["policy_version"] == 12
    assert ep["completed_at"] == "2026-08-03T12:00:00Z"
    assert ep["active_runtime_seconds_total"] == 123.45
    assert ep["reward"] == 512.75
    assert ep["result"] == "SUCCESS"
    assert ep["success"] is True
    assert ep["steps"] == 88
    assert ep["seed"] == 999
    assert ep["checkpoint_id"] == "chk_10"


def test_4_records_survive_store_restart(temp_store):
    record = {
        "event_key": "persistent_ep_1",
        "run_id": "run_persist",
        "global_environment_episode": 100,
        "reward": 300.0,
    }
    temp_store.add_episode(record)
    temp_store.flush()
    db_path = temp_store.db_path
    temp_store.close()

    new_store = EpisodeStore(db_path=db_path)
    res = new_store.get_episodes(limit=10)
    assert res["total_matching"] == 1
    assert res["episodes"][0]["event_key"] == "persistent_ep_1"
    new_store.close()


def test_5_store_works_with_sqlite_wal_enabled(temp_store):
    with temp_store._get_connection() as conn:
        cursor = conn.execute("PRAGMA journal_mode;")
        mode = cursor.fetchone()[0]
        assert mode.lower() == "wal"


def test_6_api_default_pagination_and_after_id(temp_store):
    for i in range(1, 15):
        temp_store.add_episode({
            "event_key": f"paged_ep_{i}",
            "run_id": "run_page",
            "global_environment_episode": i,
            "curriculum_stage": 8,
        })
    temp_store.flush()

    res1 = temp_store.get_episodes(limit=5)
    assert len(res1["episodes"]) == 5
    assert res1["has_more"] is True
    first_latest = res1["latest_id"]

    res2 = temp_store.get_episodes(after_id=first_latest, limit=5)
    assert len(res2["episodes"]) == 5
    assert res2["episodes"][0]["id"] > first_latest


def test_7_stage_and_run_filtering(temp_store):
    temp_store.add_episode({"event_key": "st1_ep1", "run_id": "r1", "curriculum_stage": 1, "global_environment_episode": 1})
    temp_store.add_episode({"event_key": "st8_ep2", "run_id": "r1", "curriculum_stage": 8, "global_environment_episode": 2})
    temp_store.add_episode({"event_key": "st8_ep3", "run_id": "r2", "curriculum_stage": 8, "global_environment_episode": 3})
    temp_store.flush()

    res_st8 = temp_store.get_episodes(stage=8)
    assert res_st8["total_matching"] == 2

    res_r2 = temp_store.get_episodes(run_id="r2")
    assert res_r2["total_matching"] == 1
    assert res_r2["episodes"][0]["event_key"] == "st8_ep3"


def test_8_flushing_pending_writes(temp_store):
    temp_store.add_episode({"event_key": "flush_test_1", "run_id": "r_flush", "global_environment_episode": 1})
    temp_store.flush()
    res = temp_store.get_episodes()
    assert res["total_matching"] == 1


def test_9_queue_saturation_drop_and_warning():
    """Verify that when queue maxsize is reached, add_episode drops records without throwing/blocking."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        db_path = Path(tmpdir) / "sat-test.sqlite"
        store = EpisodeStore(db_path=db_path, max_queue_size=5)
        # Pause worker processing by taking lock if worker is running
        for i in range(10):
            store.add_episode({"event_key": f"sat_ep_{i}", "global_environment_episode": i})

        assert store.dropped_count >= 1
        store.flush()
        res = store.get_episodes(limit=20)
        assert res["total_matching"] <= 10
        store.close()


def test_10_context_manager_close():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        db_path = Path(tmpdir) / "cm-test.sqlite"
        with EpisodeStore(db_path=db_path) as store:
            store.add_episode({"event_key": "cm_ep_1", "global_environment_episode": 1})
        # After exiting context, store is flushed and closed
        new_store = EpisodeStore(db_path=db_path)
        res = new_store.get_episodes()
        assert res["total_matching"] == 1
        new_store.close()


def test_11_performance_benchmark_50k_dataset():
    """Synthetic benchmark with 50,000 records measuring insertion, after_id, stage filter, and file size."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        db_path = Path(tmpdir) / "50k-telemetry.sqlite"
        store = EpisodeStore(db_path=db_path)

        records = [
            {
                "event_key": f"batch_50k_ep_{i}",
                "run_id": "run_50k",
                "global_environment_episode": i,
                "curriculum_stage": (i % 8) + 1,
                "global_timestep": i * 500,
                "reward": 100.0 + (i % 50),
                "success": bool(i % 2 == 0),
            }
            for i in range(1, 50001)
        ]

        t0 = time.perf_counter()
        with store._get_connection() as conn:
            store._insert_batch(conn, records)
        t_insert = time.perf_counter() - t0

        t1 = time.perf_counter()
        res_after = store.get_episodes(after_id=25000, limit=500)
        t_query_after = time.perf_counter() - t1

        t2 = time.perf_counter()
        res_stage = store.get_episodes(stage=8, limit=500)
        t_query_stage = time.perf_counter() - t2

        db_size_mb = db_path.stat().st_size / (1024 * 1024)

        assert res_after["total_matching"] == 25000
        assert len(res_after["episodes"]) == 500
        assert t_query_after < 0.1  # Fast indexed query under 100ms
        assert t_query_stage < 0.1  # Fast stage filter under 100ms
        assert db_size_mb > 0

        store.close()

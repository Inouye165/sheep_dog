"""Backend tests for learning curve telemetry data semantics, SQLite persistence, and API behavior."""

import pytest

from sheepdog.training.episode_store import EpisodeStore


@pytest.fixture
def temp_db(tmp_path):
    db_file = tmp_path / "test_telemetry.sqlite"
    store = EpisodeStore(db_path=db_file)
    yield store
    store.close()


def test_episode_store_reward_breakdown_schema_and_persistence(temp_db):
    """Verify that reward_breakdown JSON dict is persisted and retrieved correctly."""
    store = temp_db

    ep1_breakdown = {
        "progress_to_pen": 45.2,
        "sheep_penned": 100.0,
        "time_penalty": -12.5,
        "no_progress_penalty": -5.0,
    }

    store.add_episode({
        "global_environment_episode": 1,
        "episode_in_stage": 1,
        "curriculum_stage": 3,
        "reward": 127.7,
        "penned": 3,
        "total_sheep": 3,
        "success": True,
        "status": "SUCCESS",
        "steps": 140,
        "reward_breakdown": ep1_breakdown,
    })

    # Historical episode without reward breakdown
    store.add_episode({
        "global_environment_episode": 2,
        "episode_in_stage": 2,
        "curriculum_stage": 3,
        "reward": -50.0,
        "penned": 0,
        "total_sheep": 3,
        "success": False,
        "status": "TIMEOUT",
        "steps": 600,
        # No reward_breakdown key provided
    })

    store.flush()

    res = store.get_episodes(stage=3)
    episodes = res["episodes"]
    assert len(episodes) == 2

    # Episode 1 has breakdown
    ep1 = next(e for e in episodes if e["global_environment_episode"] == 1)
    assert ep1["success"] is True
    assert ep1["steps"] == 140
    assert ep1["reward_breakdown"] == ep1_breakdown

    # Episode 2 has no breakdown -> must be None, NOT empty dict or false zeroes
    ep2 = next(e for e in episodes if e["global_environment_episode"] == 2)
    assert ep2["success"] is False
    assert ep2["steps"] == 600
    assert ep2["reward_breakdown"] is None


def test_canonical_episode_query_filtering(temp_db):
    """Verify querying episodes by stage returns exact stage episodes without mixing stages."""
    store = temp_db

    # Stage 1 episodes
    for ep in range(1, 6):
        store.add_episode({
            "global_environment_episode": ep,
            "episode_in_stage": ep,
            "curriculum_stage": 1,
            "reward": float(ep * 10),
            "penned": 1,
            "total_sheep": 1,
            "success": True,
            "status": "SUCCESS",
            "steps": 100 + ep,
        })

    # Stage 3 episodes
    for ep in range(1, 6):
        store.add_episode({
            "global_environment_episode": 5 + ep,
            "episode_in_stage": ep,
            "curriculum_stage": 3,
            "reward": float(ep * 20),
            "penned": ep % 2,
            "total_sheep": 3,
            "success": bool(ep % 2 == 1),
            "status": "SUCCESS" if ep % 2 == 1 else "TIMEOUT",
            "steps": 200 + ep,
        })

    store.flush()

    stage3_res = store.get_episodes(stage=3)
    episodes3 = stage3_res["episodes"]
    assert len(episodes3) == 5
    assert all(e["curriculum_stage"] == 3 for e in episodes3)
    assert [e["episode_in_stage"] for e in episodes3] == [1, 2, 3, 4, 5]

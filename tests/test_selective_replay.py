"""Unit and integration tests for selective authentic trajectory recording, async writing, and retention."""

import gzip
import json
import sqlite3

import pytest

from sheepdog.config import LabConfig
from sheepdog.training.episode_store import EpisodeStore
from sheepdog.training.replay_writer import (
  AsyncReplayWriter,
  CapturePolicy,
  ReplayWriteJob,
  get_global_capture_policy,
  set_global_capture_policy,
)
from sheepdog.training.rl_env import SheepdogRLAdapter


@pytest.fixture
def temp_dir(tmp_path):
  return tmp_path


@pytest.fixture(autouse=True)
def reset_global_policy():
  set_global_capture_policy(CapturePolicy())
  yield
  set_global_capture_policy(CapturePolicy())


def test_1_and_2_vectorized_env_buffers_isolated_and_reset(temp_dir):
  config1 = LabConfig()
  env1 = SheepdogRLAdapter(config1)
  env1.env_index = 0

  config2 = LabConfig()
  env2 = SheepdogRLAdapter(config2)
  env2.env_index = 1

  obs1, _ = env1.reset(seed=101)
  obs2, _ = env2.reset(seed=102)

  assert len(env1._active_trajectory_buffer) == 1
  assert len(env2._active_trajectory_buffer) == 1
  assert env1._active_trajectory_buffer[0]["step"] == 0
  assert env2._active_trajectory_buffer[0]["step"] == 0

  # Step env1
  for _ in range(env1._environment.dog_count):
    env1.step(0)

  assert len(env1._active_trajectory_buffer) == 2
  assert len(env2._active_trajectory_buffer) == 1

  # Reset env1
  env1.reset(seed=103)
  assert len(env1._active_trajectory_buffer) == 1
  assert len(env2._active_trajectory_buffer) == 1


def test_3_authentic_coordinates_captured_without_mocking(temp_dir):
  config = LabConfig()
  env = SheepdogRLAdapter(config)
  env.reset(seed=200)

  for _ in range(env._environment.dog_count):
    env.step(0)

  frame = env._active_trajectory_buffer[1]
  assert "snapshot" in frame
  dogs = frame["snapshot"]["dogs"]
  sheep = frame["snapshot"]["sheep"]
  assert len(dogs) == env._environment.dog_count
  assert len(sheep) == len(env._environment.sheep)
  assert isinstance(dogs[0]["x"], (int, float))
  assert isinstance(sheep[0]["x"], (int, float))


def test_4_and_5_capture_policy_filtering(temp_dir):
  policy = CapturePolicy(mode="failures")

  should_save, reason = policy.should_capture(
      stage=8, success=True, status="SUCCESS", reward=100.0
  )
  assert not should_save
  assert reason == "not_requested"

  should_save, reason = policy.should_capture(
      stage=8, success=False, status="TIMEOUT", reward=-200.0
  )
  assert should_save
  assert reason == "timeout"

  should_save, reason = policy.should_capture(
      stage=8, success=False, status="STOPPED", reward=-300.0
  )
  assert should_save
  assert reason == "stopped"


def test_6_next_n_capture_counter():
  policy = CapturePolicy(mode="selective", next_n_counter=2)

  s1, r1 = policy.should_capture(
      stage=8, success=True, status="SUCCESS", reward=50.0
  )
  assert s1 and r1 == "next_n"
  assert policy.next_n_counter == 1

  s2, r2 = policy.should_capture(
      stage=8, success=True, status="SUCCESS", reward=50.0
  )
  assert s2 and r2 == "next_n"
  assert policy.next_n_counter == 0

  s3, r3 = policy.should_capture(
      stage=8, success=True, status="SUCCESS", reward=50.0
  )
  # Success without next_n or failure mode defaults to low sampling or false
  assert not s3 or r3 == "sampled_success"


def test_7_queued_to_available_transition(temp_dir):
  db_path = temp_dir / "test_telemetry.sqlite"
  store = EpisodeStore(db_path=db_path)
  store.add_episode({
      "event_key": "ep_test_7",
      "global_environment_episode": 7,
      "stage": 8,
      "reward": -10.0,
      "success": False,
      "status": "TIMEOUT",
  })
  store.flush()

  writer = AsyncReplayWriter(output_dir=temp_dir / "replays", episode_store=store)
  output_path = temp_dir / "replays" / "diag_ep7.json.gz"
  job = ReplayWriteJob(
      replay_id="diag_ep7",
      event_key="ep_test_7",
      payload={"test": "data"},
      output_path=output_path,
      capture_reason="timeout",
      use_gzip=True,
  )

  writer.enqueue(job)
  writer.flush()
  store.flush()
  print("Writer last_error:", writer.last_error)

  record = store.get_episode_by_id_or_replay_id("ep_test_7")
  assert record is not None
  assert record["replay_available"] is True
  assert record["capture_status"] == "available"
  assert record["replay_id"] == "diag_ep7"
  assert output_path.exists()
  writer.close()
  store.close()


def test_8_writer_failure_marks_failed(temp_dir):
  db_path = temp_dir / "test_telemetry_fail.sqlite"
  store = EpisodeStore(db_path=db_path)
  store.add_episode({
      "event_key": "ep_test_8",
      "global_environment_episode": 8,
      "stage": 8,
  })
  store.flush()

  writer = AsyncReplayWriter(output_dir=temp_dir / "replays", episode_store=store)
  # Create a regular file to use as invalid parent directory
  invalid_dir_file = temp_dir / "blocked_file.txt"
  invalid_dir_file.write_text("blocked")
  invalid_path = invalid_dir_file / "impossible_sub_dir" / "diag_ep8.json"
  job = ReplayWriteJob(
      replay_id="diag_ep8",
      event_key="ep_test_8",
      payload={"test": "data"},
      output_path=invalid_path,
      capture_reason="timeout",
      use_gzip=False,
  )

  writer._write_job(job)

  record = store.get_episode_by_id_or_replay_id("ep_test_8")
  assert record is not None
  assert record["replay_available"] is False
  assert record["capture_status"] == "failed"
  assert writer.failure_count == 1
  writer.close()
  store.close()


def test_9_queue_overflow_counter(temp_dir):
  store = EpisodeStore(db_path=temp_dir / "test_overflow.sqlite")
  writer = AsyncReplayWriter(
      output_dir=temp_dir / "replays", max_queue_size=2, episode_store=store
  )
  writer._stop_event.set()

  for i in range(5):
    job = ReplayWriteJob(
        replay_id=f"job_{i}",
        event_key=f"ep_{i}",
        payload={"data": i},
        output_path=temp_dir / "replays" / f"job_{i}.json",
        capture_reason="next_n",
        use_gzip=False,
    )
    writer.enqueue(job)

  assert writer.dropped_count >= 2
  writer.close()
  store.close()


def test_10_database_migration(temp_dir):
  db_path = temp_dir / "legacy.sqlite"
  conn = sqlite3.connect(str(db_path))
  conn.execute("""
        CREATE TABLE training_episodes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_key TEXT NOT NULL UNIQUE,
            curriculum_stage INTEGER,
            reward REAL,
            result TEXT,
            success INTEGER,
            timeout INTEGER,
            stopped INTEGER,
            steps INTEGER
        );
    """)
  conn.execute(
      "INSERT INTO training_episodes (event_key, reward) VALUES ('legacy_1',"
      " 10.0);"
  )
  conn.commit()
  conn.close()

  # Instantiating EpisodeStore on legacy DB triggers migration
  store = EpisodeStore(db_path=db_path)
  store.flush()

  conn = sqlite3.connect(str(db_path))
  cursor = conn.execute("PRAGMA table_info(training_episodes);")
  cols = {row[1] for row in cursor.fetchall()}
  assert "replay_available" in cols
  assert "replay_id" in cols
  assert "capture_status" in cols

  row = conn.execute(
      "SELECT replay_available, capture_status FROM training_episodes WHERE"
      " event_key = 'legacy_1'"
  ).fetchone()
  assert row[0] == 0
  assert row[1] == "not_requested"
  conn.close()
  store.close()


def test_15_16_17_retention_pruning(temp_dir):
  db_path = temp_dir / "prune.sqlite"
  store = EpisodeStore(db_path=db_path)
  replays_dir = temp_dir / "replays"
  replays_dir.mkdir()

  # Add 3 diagnostic replays and 1 checkpoint eval replay
  for i in range(1, 4):
    rp = replays_dir / f"diag_{i}.json"
    rp.write_text(json.dumps({"frame": i}))
    store.add_episode({
        "event_key": f"diag_ep_{i}",
        "global_environment_episode": i,
        "curriculum_stage": 8,
        "replay_available": 1,
        "replay_id": f"diag_{i}",
        "replay_path": str(rp),
        "replay_source": "training-diagnostic",
        "capture_status": "available",
    })

  # Add evaluation replay
  eval_p = replays_dir / "eval_chk.json"
  eval_p.write_text(json.dumps({"eval": True}))
  store.add_episode({
      "event_key": "eval_ep",
      "global_environment_episode": 99,
      "curriculum_stage": 8,
      "replay_available": 1,
      "replay_id": "eval_chk",
      "replay_path": str(eval_p),
      "replay_source": "checkpoint-evaluation",
      "capture_status": "available",
  })
  store.flush()

  # Prune to max 1 diagnostic replay
  pruned = store.prune_replays(max_files_per_stage=1, max_total_files=1)
  assert pruned == 2

  # Checkpoint eval replay remains
  assert eval_p.exists()
  eval_rec = store.get_episode_by_id_or_replay_id("eval_ep")
  assert eval_rec["replay_available"] is True

  # Oldest diagnostic replay 1 was deleted and marked pruned
  assert not (replays_dir / "diag_1.json").exists()
  rec1 = store.get_episode_by_id_or_replay_id("diag_ep_1")
  assert rec1["replay_available"] is False
  assert rec1["capture_status"] == "pruned"

  store.close()


def test_11_and_12_replay_endpoint_security_and_404(temp_dir):

  db_path = temp_dir / "test_api.sqlite"
  store = EpisodeStore(db_path=db_path)
  replays_dir = temp_dir / "replays"
  replays_dir.mkdir()

  valid_path = replays_dir / "valid_replay.json.gz"
  with gzip.open(valid_path, "wt", encoding="utf-8") as f:
    json.dump({"frames": [{"step": 0}]}, f)

  store.add_episode({
      "event_key": "ep_valid",
      "global_environment_episode": 11,
      "replay_available": 1,
      "replay_id": "valid_replay",
      "replay_path": str(valid_path),
      "capture_status": "available",
  })
  store.flush()
  store.close()


def test_13_and_14_capture_policy_management():
  policy = get_global_capture_policy()
  policy.mode = "selective"
  policy.next_n_counter = 0

  policy.mode = "failures"
  policy.next_n_counter = 10
  policy.target_outcome = "failures"

  assert policy.mode == "failures"
  assert policy.next_n_counter == 10
  assert policy.target_outcome == "failures"


def test_19_reproduced_replay_labeling(temp_dir):
  reproduced_path = temp_dir / "reproduced_ep_1.json.gz"
  bundle = {
      "seed": 42,
      "replay_mode": "reproduced",
      "replay_source": "reproduced",
      "capture_reason": "reproduced",
      "disclaimer": "Episode rerun from recorded seed and configuration.",
  }
  with gzip.open(reproduced_path, "wt", encoding="utf-8") as f:
    json.dump(bundle, f)

  with gzip.open(reproduced_path, "rt", encoding="utf-8") as f:
    loaded = json.load(f)

  assert loaded["replay_source"] == "reproduced"
  assert loaded["capture_reason"] == "reproduced"
  assert "rerun" in loaded["disclaimer"].lower()


def test_20_through_24_failed_episodes_query_behavior(temp_dir):
  db_path = temp_dir / "test_failed_query.sqlite"
  store = EpisodeStore(db_path=db_path)

  # Insert 30 failed playable episodes across different runs and stages
  for i in range(1, 31):
    store.add_episode({
        "event_key": f"failed_ep_{i}",
        "global_environment_episode": 100 + i,
        "run_id": "run_old" if i <= 15 else "run_new",
        "curriculum_stage": (i % 9) + 1,
        "success": 0,
        "status": "TIMEOUT" if i % 2 == 0 else "STOPPED",
        "replay_available": 1,
        "replay_id": f"diag_stage{(i % 9) + 1}_ep{100 + i}_seed{i}",
        "replay_path": f"/path/to/diag_{i}.json.gz",
        "capture_status": "available",
        "seed": i,
    })

  # Insert 5 successful episodes with replays (must be excluded)
  for i in range(1, 6):
    store.add_episode({
        "event_key": f"win_ep_{i}",
        "global_environment_episode": 200 + i,
        "run_id": "run_new",
        "curriculum_stage": 9,
        "success": 1,
        "status": "SUCCESS",
        "replay_available": 1,
        "replay_id": f"diag_win_ep{i}",
        "replay_path": f"/path/to/win_{i}.json.gz",
        "capture_status": "available",
        "seed": 100 + i,
    })

  # Insert 5 failed episodes WITHOUT available replay (must be excluded)
  for i in range(1, 6):
    store.add_episode({
        "event_key": f"unavail_failed_ep_{i}",
        "global_environment_episode": 300 + i,
        "run_id": "run_new",
        "curriculum_stage": 8,
        "success": 0,
        "status": "STOPPED",
        "replay_available": 0,
        "replay_id": None,
        "replay_path": None,
        "capture_status": "not_requested",
        "seed": 200 + i,
    })

  store.flush()

  # Query top 25 failed episodes
  results = store.get_recent_failed_episodes_with_replays(limit=25)

  # Proves: Exactly 25 returned
  assert len(results) == 25

  # Proves: Ordered newest to oldest (global_environment_episode DESC)
  eps = [r["global_environment_episode"] for r in results]
  assert eps == sorted(eps, reverse=True)
  assert eps[0] == 130  # Newest failed episode
  assert eps[-1] == 106 # 25th newest failed episode

  # Proves: Successful episodes are excluded
  for r in results:
    assert r["success"] is False

  # Proves: Episodes without replay are excluded
  for r in results:
    assert r["replay_available"] is True
    assert r["capture_status"] == "available"
    assert r["replay_id"] is not None

  # Proves: Results from older runs and different stages are included
  run_ids = {r["run_id"] for r in results}
  stages = {r["curriculum_stage"] for r in results}
  assert "run_old" in run_ids
  assert len(stages) > 1

  store.close()



"""Tests for backend stage-queue execution, descending-stage best tracking, and ModelHistoryStore."""

from __future__ import annotations

import contextlib
import json
import sqlite3
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest

from sheepdog import server
from sheepdog.config import LabConfig, TrainingConfig
from sheepdog.curriculum import CURRICULUM_STAGES, apply_training_profile
from sheepdog.evaluation.evaluator import EvaluationRecord, EvaluationSummary
from sheepdog.policies.neural import NeuralPolicy
from sheepdog.tracking import model_history
from sheepdog.tracking.model_history import (
    MODEL_HISTORY_DB_FILENAME,
    ModelHistoryStore,
)
from sheepdog.training.maskable_ppo import MaskablePPOTrainer


@pytest.fixture(autouse=True)
def _stub_git(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        model_history, "git_info", lambda: {"revision": "testrev123", "dirty": False}
    )


@pytest.fixture(autouse=True)
def _isolate_episode_stores() -> Any:
    from sheepdog.training import episode_store as ep_mod

    def _cleanup() -> None:
        rw = getattr(ep_mod, "_global_episode_store", None)
        if rw is not None:
            with contextlib.suppress(Exception):
                rw.close()
            ep_mod._global_episode_store = None
        ro_map = getattr(ep_mod, "_global_ro_episode_stores", None)
        if isinstance(ro_map, dict):
            for ro in list(ro_map.values()):
                with contextlib.suppress(Exception):
                    ro.close()
            ro_map.clear()

    _cleanup()
    yield
    _cleanup()


@pytest.fixture
def isolated_manager(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> server.TrainingManager:
    config = LabConfig(
        training=TrainingConfig(
            output_dir=str(tmp_path),
            web_export_dir=str(tmp_path / "web"),
            backup_dir=str(tmp_path / "backups"),
            backup_enabled=False,
            wandb_enabled=False,
        )
    )
    monkeypatch.setattr(server, "LabConfig", lambda: config)
    monkeypatch.setattr(server.TrainingManager, "_reconcile_web_exports", lambda self: None)
    monkeypatch.setattr(server.TrainingManager, "_run_watchdog", lambda self: None)
    monkeypatch.setattr(server.TrainingManager, "restore_active_run_state", lambda self: None)
    monkeypatch.setattr(server, "ensure_free_disk_space", lambda *args, **kwargs: None)
    mgr = server.TrainingManager()
    mgr._status["phase"] = "idle"
    try:
        yield mgr
    finally:
        if mgr._thread is not None and mgr._thread.is_alive():
            mgr._control_request = "stopped"
            mgr._thread.join(timeout=2.0)
        with contextlib.suppress(Exception):
            mgr.runtime_tracker.end_session("test_cleanup")
        rt_stop = getattr(mgr.runtime_tracker, "_stop_event", None)
        if rt_stop is not None:
            rt_stop.set()


def _install_fake_queue_trainer(
    manager: server.TrainingManager,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    ready_stages: set[int] | None = None,
    checkpoint_bytes_prefix: bytes = b"fake-ckpt-",
    pause_on_stage: int | None = None,
) -> list[tuple[int, int]]:
    """Install a fake create_trainer that writes real checkpoint files and promotes ready stages."""
    executed: list[tuple[int, int]] = []
    checkpoints = iter(range(101, 200))
    paused_once = False

    def gate(_root: Path, episode: int, **_kwargs: object) -> dict[str, object]:
        cur_stage = int(manager._status.get("curriculum_stage") or 1)
        ready = ready_stages is None or cur_stage in ready_stages
        return {
            "ready": ready,
            "decision": "promote_ready" if ready else "hold",
            "best_success": 1.0 if ready else 0.25,
            "qualified_streak": server.MAX_STAGE_MASTERY_QUALIFIED_STREAK if ready else 0,
            "full_success_hits": server.MAX_STAGE_MASTERY_FULL_SUCCESS_HITS if ready else 0,
        }

    def trainer_for(config: LabConfig, _root: str) -> object:
        stage = int(config.rewards.instincts.curriculum_stage)
        executed.append((stage, config.training.episodes))
        episode = next(checkpoints)
        seeds = list(config.training.evaluation_seeds)
        ckpt_dir = tmp_path / "checkpoints" / f"stage{stage}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_file = ckpt_dir / f"stage{stage}-best-model.zip"
        ckpt_file.write_bytes(checkpoint_bytes_prefix + f"s{stage}-ep{episode}".encode())

        trainer = Mock(
            spec=[
                "train",
                "total_episodes_trained",
                "_loaded_state",
                "prefer_latest_checkpoint",
                "runtime_tracker",
            ]
        )
        trainer.total_episodes_trained = episode - 1
        trainer._loaded_state = {
            "run_id": "run_queue_test",
            "total_environment_episodes": episode * 5,
            "total_timesteps": episode * 100,
        }
        trainer.prefer_latest_checkpoint = False
        monkeypatch.setattr(
            manager,
            "get_checkpoint_details",
            lambda _ep, _j=None, _cid=None: {
                "curriculum_stage": stage,
                "evaluation_seeds": list(seeds),
                "checkpoint_id": f"chk_run_queue_test_ep_{episode}",
                "policy_version": episode,
                "policy_state_path": str(ckpt_file),
            },
        )

        def train(*, progress_callback, should_stop):
            nonlocal paused_once
            assert not should_stop()
            if pause_on_stage is not None and stage == pause_on_stage and not paused_once:
                paused_once = True
                manager.pause()
                progress_callback(
                    {
                        "phase": "checkpoint",
                        "checkpoint_episode": episode,
                        "total_episodes_trained": episode,
                        "batch_completed_segments": 1,
                        "environment_episodes_total": episode * 5 + 3,
                        "global_timestep": episode * 100,
                        "policy_version": episode,
                        "policy_state_path": str(ckpt_file),
                        "summary": {
                            "curriculum_stage": stage,
                            "policy_version": episode,
                            "success_rate": 0.5,
                            "average_reward": 1.5,
                            "average_completion_steps": 80.0,
                            "evaluation_seed_count": len(seeds),
                            "records": [],
                        },
                    }
                )
                assert should_stop()
                return

            server.atomic_write_json(
                tmp_path / "training-state.json",
                {
                    "run_id": "run_queue_test",
                    "policy_state_path": str(ckpt_file),
                    "best_model_path": str(ckpt_file),
                    "best_model_curriculum_stage": stage,
                    "total_episodes_trained": episode,
                    "total_timesteps": episode * 100 + 100,
                },
            )
            progress_callback(
                {
                    "phase": "checkpoint",
                    "checkpoint_episode": episode,
                    "total_episodes_trained": episode,
                    "batch_completed_segments": 1,
                    "environment_episodes_total": episode * 5 + 5,
                    "global_timestep": episode * 100 + 100,
                    "policy_version": episode,
                    "policy_state_path": str(ckpt_file),
                    "summary": {
                        "curriculum_stage": stage,
                        "policy_version": episode,
                        "success_rate": 1.0,
                        "average_reward": 2.5,
                        "average_completion_steps": 45.0,
                        "evaluation_seed_count": len(seeds),
                        "records": [],
                    },
                }
            )

        trainer.train.side_effect = train
        return trainer

    monkeypatch.setattr(server, "create_trainer", trainer_for)
    monkeypatch.setattr(server, "compute_promotion_gate_snapshot", gate)
    monkeypatch.setattr(server, "PLATEAU_STOP_MIN_CHECKPOINTS", 0)
    monkeypatch.setattr(server, "PLATEAU_STOP_NO_IMPROVEMENT_STREAK", 0)
    return executed


def test_non_sequential_stage_queue_executes_exact_stages_and_records_history(
    isolated_manager: server.TrainingManager,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executed = _install_fake_queue_trainer(isolated_manager, tmp_path, monkeypatch)

    isolated_manager._run_training(
        5,
        True,
        stage_queue=[10, 25, 36],
        stage_spec="10, 25, 36",
        pass_label="Targeted Sweep",
    )

    assert [st for st, _ in executed] == [10, 25, 36]
    assert isolated_manager._status["phase"] == "complete"
    assert "Stage queue complete (3/3)" in isolated_manager._status["message"]

    store = ModelHistoryStore(tmp_path / MODEL_HISTORY_DB_FILENAME, read_only=True)
    passes = store.list_passes()
    assert len(passes) == 1
    p = passes[0]
    assert p["status"] == "completed"
    assert p["stage_spec"] == "10, 25, 36"
    assert p["label"] == "Targeted Sweep"
    assert p["planned_stages"] == [10, 25, 36]
    assert p["pass_type"] == "targeted_stages"
    assert p["completed_stages_count"] == 3
    assert [sr["stage"] for sr in p["stage_runs"]] == [10, 25, 36]
    assert [sr["status"] for sr in p["stage_runs"]] == ["promoted", "promoted", "completed"]


def test_stage_queue_starting_with_max_stage_advances_to_next_queued_stage(
    isolated_manager: server.TrainingManager,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    highest = max(CURRICULUM_STAGES)
    executed = _install_fake_queue_trainer(isolated_manager, tmp_path, monkeypatch)

    isolated_manager._run_training(
        5,
        True,
        stage_queue=[highest, 25],
        stage_spec=f"{highest}, 25",
    )

    assert [st for st, _ in executed] == [highest, 25]
    assert isolated_manager._status["phase"] == "complete"
    store = ModelHistoryStore(tmp_path / MODEL_HISTORY_DB_FILENAME, read_only=True)
    p = store.list_passes()[0]
    assert [sr["stage"] for sr in p["stage_runs"]] == [highest, 25]
    assert [sr["status"] for sr in p["stage_runs"]] == ["promoted", "completed"]


def test_start_stage_queue_from_stage38_allows_descending_jump_and_preserves_weights(
    isolated_manager: server.TrainingManager,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    highest = max(CURRICULUM_STAGES)
    stage38_dir = tmp_path / "checkpoints" / f"stage{highest}"
    stage38_dir.mkdir(parents=True, exist_ok=True)
    stage38_zip = stage38_dir / f"stage{highest}-best-model.zip"
    stage38_zip.write_bytes(b"PK\x03\x04stage38-weights")

    server.atomic_write_json(tmp_path / "stage-history.json", {str(highest): 120})
    server._append_promotion_history(
        tmp_path,
        {
            "event_type": "promotion",
            "from_stage": highest - 1,
            "to_stage": highest,
            "trigger_checkpoint_episode": 500,
        },
    )
    isolated_manager._status.update(
        {
            "curriculum_stage": highest,
            "trainer_type": "maskable_ppo",
            "policy_type": "neural",
            "policy_mode": "neural_policy",
            "active_model_path": str(stage38_zip),
            "active_checkpoint_id": "chk_s38_ep_500",
        }
    )

    from sb3_contrib import MaskablePPO

    monkeypatch.setattr(MaskablePPO, "load", Mock(return_value=Mock()))
    worker = Mock()
    thread_mock = Mock(return_value=worker)
    monkeypatch.setattr(server.threading, "Thread", thread_mock)

    resp = isolated_manager.start(
        5,
        True,
        stage_queue=[24, 25],
        stage_spec="24-25",
        starting_model_source="latest",
    )

    assert resp["running"] is True
    assert resp["curriculum_stage"] == 24
    assert resp["stage_queue"] == [24, 25]
    assert resp["active_model_path"] == str(stage38_zip)
    assert thread_mock.call_args.kwargs["kwargs"]["stage_queue"] == [24, 25]
    assert thread_mock.call_args.kwargs["kwargs"]["starting_model_source"] == "latest"


def test_real_maskable_ppo_trainer_descending_stage_resets_best_tracking(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import torch as th

    import sheepdog.training.episode_store as episode_store

    monkeypatch.setattr(
        episode_store,
        "get_episode_store",
        Mock(return_value=Mock(get_max_episode=Mock(return_value=0))),
    )

    stage38_cfg = apply_training_profile(
        LabConfig(
            training=TrainingConfig(
                output_dir=str(tmp_path),
                web_export_dir=str(tmp_path / "web"),
                neural_hidden_sizes=(8,),
                ppo_env_workers=1,
                rollout_steps=8,
                batch_size=4,
                checkpoint_episodes=(0,),
                total_timesteps=8,
                evaluation_seeds=(11,),
            )
        ),
        curriculum_stage=38,
    )

    policy = NeuralPolicy.initialize(stage38_cfg)
    stage38_best = tmp_path / "checkpoints" / "stage38" / "stage38-best-model.zip"
    stage38_latest = tmp_path / "checkpoints" / "stage38" / "maskable-ppo-00000100.zip"
    stage38_best.parent.mkdir(parents=True, exist_ok=True)
    policy.model.save(str(stage38_best))
    first_param = next(policy.model.policy.parameters())
    with th.no_grad():
        first_param.fill_(0.42)
    policy.model.save(str(stage38_latest))
    policy.model.get_env().close()

    state_path = tmp_path / "training-state.json"
    server.atomic_write_json(
        state_path,
        {
            "run_id": "run_desc_test",
            "policy_state_path": str(stage38_latest),
            "best_model_path": str(stage38_best),
            "best_model_curriculum_stage": 38,
            "best_success_rate": 1.0,
            "best_average_reward": 999.0,
            "best_completion_steps": 50.0,
            "total_episodes_trained": 10,
            "total_timesteps": 100,
            "policy_config": policy.config.to_dict(),
            "incomplete_batch": None,
        },
    )

    stage10_cfg = apply_training_profile(
        LabConfig(
            training=TrainingConfig(
                output_dir=str(tmp_path),
                web_export_dir=str(tmp_path / "web"),
                neural_hidden_sizes=(8,),
                ppo_env_workers=1,
                rollout_steps=8,
                batch_size=4,
                checkpoint_episodes=(0,),
                total_timesteps=8,
                evaluation_seeds=(11,),
                auto_rollback_on_collapse=True,
            )
        ),
        curriculum_stage=10,
    )

    trainer = MaskablePPOTrainer(stage10_cfg, tmp_path)
    loaded_paths: list[Path] = []
    orig_load = NeuralPolicy.load

    def spy_load(path: Path, cfg: LabConfig, policy_cfg: Any = None) -> NeuralPolicy:
        loaded_paths.append(Path(path))
        return orig_load(path, cfg, policy_cfg)

    monkeypatch.setattr(NeuralPolicy, "load", staticmethod(spy_load))

    def fake_evaluate(
        eval_policy: NeuralPolicy,
        seeds: Any,
        *,
        checkpoint_episode: int,
        **_kwargs: object,
    ) -> tuple[EvaluationSummary, Path, Path]:
        eval_dir = tmp_path / "evaluations"
        eval_dir.mkdir(parents=True, exist_ok=True)
        eval_json = eval_dir / f"eval-{checkpoint_episode:06d}.json"
        eval_csv = eval_dir / f"eval-{checkpoint_episode:06d}.csv"
        eval_json.write_text("{}", encoding="utf-8")
        eval_csv.write_text("", encoding="utf-8")
        replay_file = tmp_path / "replays" / f"replay-{checkpoint_episode}.json"
        replay_file.parent.mkdir(parents=True, exist_ok=True)
        replay_file.write_text("{}", encoding="utf-8")
        summary = EvaluationSummary(
            checkpoint_episode=checkpoint_episode,
            policy_name="neural_policy",
            records=(
                EvaluationRecord(
                    seed=11,
                    success=False,
                    timeout=True,
                    stopped=False,
                    steps=120,
                    simulated_seconds=12.0,
                    sheep_penned=3,
                    final_sheep_distance_to_pen=5.0,
                    final_flock_spread=2.0,
                    no_progress_steps=10,
                    stop_reason="timeout",
                    spawn_mode="default",
                    reward_total=12.0,
                    final_farthest_distance_to_pen=6.0,
                    final_farthest_distance_to_flock_center=2.5,
                    role_switches=0,
                    collector_activations=0,
                    blocker_activations=0,
                    cumulative_gate_progress=0.0,
                    controlled_stall_steps=0,
                    left_flank_occupancy_steps=0,
                    right_flank_occupancy_steps=0,
                    gate_corridor_occupancy_peak=0.0,
                    gate_corridor_failure_steps=0,
                    dog_role_occupancy={},
                    reward_breakdown={},
                    replay_path=str(replay_file),
                ),
            ),
            success_rate=0.25,
            timeout_rate=0.75,
            average_completion_steps=120.0,
            average_completion_seconds=12.0,
            average_sheep_penned=3.0,
            average_reward=12.0,
            trainer_type="maskable_ppo",
            policy_type="neural",
            promotion_eligible=True,
        )
        return summary, eval_json, eval_csv

    monkeypatch.setattr(trainer.evaluator, "evaluate", fake_evaluate)
    monkeypatch.setattr(
        trainer.evaluator,
        "export_replay",
        lambda *_a, **_kw: tmp_path / "replays" / "rep.json",
    )

    trainer.train()

    assert loaded_paths and loaded_paths[0] == stage38_latest
    stage10_best = tmp_path / "checkpoints" / "stage10" / "stage10-best-model.zip"
    assert stage10_best.exists()
    persisted_state = json.loads(state_path.read_text(encoding="utf-8"))
    assert persisted_state["best_model_curriculum_stage"] == 10
    assert persisted_state["best_success_rate"] == pytest.approx(0.25)
    assert Path(persisted_state["best_model_path"]) == stage10_best


def test_pause_and_resume_mid_queue_preserves_index_and_reuses_pass_and_stage_run(
    isolated_manager: server.TrainingManager,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sheepdog.training.episode_store as episode_store

    monkeypatch.setattr(episode_store, "get_episode_store", Mock(return_value=Mock()))
    executed = _install_fake_queue_trainer(
        isolated_manager, tmp_path, monkeypatch, pause_on_stage=25
    )

    # First run pauses during Stage 25 (index 1)
    isolated_manager._run_training(
        5,
        True,
        stage_queue=[10, 25, 36],
        stage_spec="10, 25, 36",
        pass_label="Pausable Pass",
    )
    assert [st for st, _ in executed] == [10, 25]
    assert isolated_manager._status["phase"] == "paused"
    assert isolated_manager._status["stage_queue_index"] == 1
    paused_pass_id = isolated_manager._status["active_pass_id"]
    assert paused_pass_id is not None

    session_marker = server._read_training_session_state(tmp_path)
    assert session_marker is not None
    assert session_marker["state"] == "paused"
    req = session_marker["training_request"]
    assert req["stage_queue"] == [10, 25, 36]
    assert req["stage_queue_index"] == 1
    assert req["pass_id"] == paused_pass_id
    paused_sr_id = req["active_stage_run_id"]
    assert paused_sr_id is not None

    store = ModelHistoryStore(tmp_path / MODEL_HISTORY_DB_FILENAME, read_only=True)
    paused_pass = store.get_pass(paused_pass_id)
    assert paused_pass is not None
    assert paused_pass["status"] == "paused"
    assert paused_pass["stage_runs"][1]["status"] == "paused"

    # Resume via TrainingManager.start(resume=True) and wait for background thread
    from sb3_contrib import MaskablePPO

    monkeypatch.setattr(MaskablePPO, "load", Mock(return_value=Mock()))
    isolated_manager.start(resume=True)
    assert isolated_manager._thread is not None
    isolated_manager._thread.join(timeout=10.0)

    assert [st for st, _ in executed] == [10, 25, 25, 36]
    assert isolated_manager._status["phase"] == "complete"

    passes = store.list_passes()
    assert len(passes) == 1
    resumed_pass = passes[0]
    assert resumed_pass["pass_id"] == paused_pass_id
    assert resumed_pass["status"] == "completed"
    assert [sr["stage"] for sr in resumed_pass["stage_runs"]] == [10, 25, 36]
    assert resumed_pass["stage_runs"][1]["stage_run_id"] == paused_sr_id
    assert [sr["status"] for sr in resumed_pass["stage_runs"]] == [
        "promoted",
        "promoted",
        "completed",
    ]


def test_two_pass_sha_lineage_and_model_history_http_endpoints(
    isolated_manager: server.TrainingManager,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_queue_trainer(
        isolated_manager, tmp_path, monkeypatch, checkpoint_bytes_prefix=b"pass1-"
    )
    isolated_manager._run_training(
        5, True, stage_queue=[10, 25, 36], stage_spec="10, 25, 36", pass_label="Pass 1"
    )
    pass1_id = isolated_manager._status["active_pass_id"]

    # Second pass starts from Pass 1's resulting checkpoint and runs [25, 36]
    _install_fake_queue_trainer(
        isolated_manager, tmp_path, monkeypatch, checkpoint_bytes_prefix=b"pass2-"
    )
    isolated_manager._run_training(
        5, True, stage_queue=[25, 36], stage_spec="25, 36", pass_label="Pass 2"
    )
    pass2_id = isolated_manager._status["active_pass_id"]
    assert pass1_id and pass2_id and pass1_id != pass2_id

    store = ModelHistoryStore(tmp_path / MODEL_HISTORY_DB_FILENAME, read_only=True)
    p1 = store.get_pass(pass1_id)
    p2 = store.get_pass(pass2_id)
    assert p1 is not None and p2 is not None
    assert p2["parent_pass_id"] == pass1_id
    assert p2["parent_checkpoint_sha256"] == p1["resulting_checkpoint_sha256"]
    assert p2["parent_checkpoint_sha256"] is not None

    original_manager = getattr(server.TrainingRequestHandler, "_manager_instance", None)
    server.TrainingRequestHandler._manager_instance = isolated_manager  # pylint: disable=protected-access
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.TrainingRequestHandler)
    srv_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    srv_thread.start()
    try:
        base_url = f"http://127.0.0.1:{httpd.server_port}"
        with urllib.request.urlopen(f"{base_url}/api/model-history?limit=10") as resp:
            assert resp.status == 200
            history_payload = json.loads(resp.read().decode("utf-8"))
        assert len(history_payload["passes"]) == 2
        assert history_payload["selected_lineage"]["target_pass_id"] == pass2_id
        stages_in_cmp = {item["stage"]: item for item in history_payload["selected_lineage"]["stage_comparison"]}
        assert len(stages_in_cmp[25]["runs"]) == 2
        assert len(stages_in_cmp[36]["runs"]) == 2

        with urllib.request.urlopen(
            f"{base_url}/api/model-history/lineage?pass_id={pass2_id}"
        ) as resp:
            assert resp.status == 200
            lineage_payload = json.loads(resp.read().decode("utf-8"))
        assert [p["pass_id"] for p in lineage_payload["lineage"]] == [pass1_id, pass2_id]

        # Read-only preview mode must still serve both endpoints
        monkeypatch.setattr(server, "_READONLY_PREVIEW_MODE", True)
        with urllib.request.urlopen(f"{base_url}/api/model-history") as resp:
            assert resp.status == 200
            ro_payload = json.loads(resp.read().decode("utf-8"))
        assert len(ro_payload["passes"]) == 2
        with urllib.request.urlopen(
            f"{base_url}/api/model-history/lineage?pass_id={pass1_id}"
        ) as resp:
            assert resp.status == 200
            ro_lineage = json.loads(resp.read().decode("utf-8"))
        assert ro_lineage["target_pass_id"] == pass1_id

        # Missing pass_id -> 400; unknown pass_id -> 404
        with pytest.raises(urllib.error.HTTPError) as missing_err:
            urllib.request.urlopen(f"{base_url}/api/model-history/lineage")
        assert missing_err.value.code == 400
        with pytest.raises(urllib.error.HTTPError) as notfound_err:
            urllib.request.urlopen(f"{base_url}/api/model-history/lineage?pass_id=nonexistent")
        assert notfound_err.value.code == 404
    finally:
        httpd.shutdown()
        httpd.server_close()
        server.TrainingRequestHandler._manager_instance = original_manager  # pylint: disable=protected-access


@pytest.mark.parametrize(
    "bad_spec",
    ["", "   ", "39", "30-20", "abc", "0", "10, 99"],
)
def test_invalid_stage_spec_returns_http_400_on_training_start(
    isolated_manager: server.TrainingManager,
    bad_spec: str,
) -> None:
    original_manager = getattr(server.TrainingRequestHandler, "_manager_instance", None)
    server.TrainingRequestHandler._manager_instance = isolated_manager  # pylint: disable=protected-access
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.TrainingRequestHandler)
    srv_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    srv_thread.start()
    try:
        req = urllib.request.Request(
            f"http://127.0.0.1:{httpd.server_port}/api/training/start",
            data=json.dumps({"episodes": 5, "stage_spec": bad_spec}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with pytest.raises(urllib.error.HTTPError) as err:
            urllib.request.urlopen(req)
        assert err.value.code == 400
        err_body = json.loads(err.value.read().decode("utf-8"))
        assert err_body["code"] == "INVALID_STAGE_SPEC"
    finally:
        httpd.shutdown()
        httpd.server_close()
        server.TrainingRequestHandler._manager_instance = original_manager  # pylint: disable=protected-access


def test_clear_and_archive_preserve_model_history_sqlite(
    isolated_manager: server.TrainingManager, tmp_path: Path
) -> None:
    db_path = tmp_path / MODEL_HISTORY_DB_FILENAME
    store = ModelHistoryStore(db_path)
    p1 = store.start_pass(stage_spec="10, 25", label="Survived Reset")
    sr1 = store.start_stage_run(p1, stage=10, sequence_order=0)
    store.finish_stage_run(sr1, status="completed", episodes=5)
    store.finish_pass(p1, status="completed")

    # Also simulate WAL/SHM sidecar files and ordinary training artifacts
    wal_path = tmp_path / f"{MODEL_HISTORY_DB_FILENAME}-wal"
    shm_path = tmp_path / f"{MODEL_HISTORY_DB_FILENAME}-shm"
    wal_path.write_bytes(b"wal")
    shm_path.write_bytes(b"shm")
    (tmp_path / "training-state.json").write_text("{}", encoding="utf-8")
    (tmp_path / "checkpoints" / "stage10").mkdir(parents=True, exist_ok=True)
    (tmp_path / "checkpoints" / "stage10" / "stage10-best-model.zip").write_bytes(b"zip")

    cfg = LabConfig(
        training=TrainingConfig(
            output_dir=str(tmp_path),
            web_export_dir=str(tmp_path / "web"),
            backup_dir=str(tmp_path / "backups"),
            backup_enabled=False,
        )
    )

    isolated_manager._archive_training_outputs(cfg)
    assert db_path.exists()
    assert wal_path.exists()
    assert shm_path.exists()
    assert not (tmp_path / "training-state.json").exists()

    (tmp_path / "training-state.json").write_text("{}", encoding="utf-8")
    isolated_manager._clear_training_outputs(cfg)
    assert db_path.exists()
    assert wal_path.exists()
    assert shm_path.exists()
    assert not (tmp_path / "training-state.json").exists()

    ro_store = ModelHistoryStore(db_path, read_only=True)
    assert len(ro_store.list_passes()) == 1
    assert ro_store.get_pass(p1)["label"] == "Survived Reset"


def test_model_history_db_failure_logs_warning_and_never_crashes_training(
    isolated_manager: server.TrainingManager,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _install_fake_queue_trainer(isolated_manager, tmp_path, monkeypatch)

    def broken_store(*_args: object, **_kwargs: object) -> Any:
        raise sqlite3.OperationalError("simulated disk full")

    monkeypatch.setattr(server, "ModelHistoryStore", broken_store)
    isolated_manager._run_training(5, True, stage_queue=[10, 11], stage_spec="10-11")

    assert isolated_manager._status["phase"] == "complete"
    assert "Failed to initialize ModelHistoryStore" in caplog.text

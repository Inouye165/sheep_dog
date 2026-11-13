"""HTTP integration test for server early warnings API endpoints (Reduced Scope)."""

from __future__ import annotations

from pathlib import Path

from sheepdog.server import _readonly_preview_status_payload
from sheepdog.training.early_warning import EarlyWarningMonitor, WarningStatus


def test_readonly_preview_status_payload_contains_early_warnings():
    """Verify that read-only preview status includes early warnings summary."""
    status = _readonly_preview_status_payload()
    assert "early_warnings_summary" in status
    assert status["early_warnings_summary"]["monitoring_status"] == "read_only_preview"
    assert status["early_warnings_summary"]["total_active"] >= 1


def test_early_warning_monitor_acknowledgement_and_persistence(tmp_path: Path):
    """Verify disk persistence of early-warnings.json and acknowledge operation."""
    eval_dir = tmp_path / "evaluations"
    eval_dir.mkdir(parents=True, exist_ok=True)
    diag_dir = eval_dir / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)

    # Create dummy evaluation and diagnostic summary
    import json
    eval_file = eval_dir / "eval_test_01.json"
    with eval_file.open("w", encoding="utf-8") as f:
        json.dump({
            "evaluation_id": "eval_test_01",
            "checkpoint_episode": 100,
            "curriculum_stage": 1,
            "success_rate": 0.5,
            "records": [{"seed": 1}],
        }, f)

    diag_file = diag_dir / "eval_test_01.diagnostics.json"
    with diag_file.open("w", encoding="utf-8") as f:
        json.dump({
            "metrics_version": "2.0",
            "evaluation_id": "eval_test_01",
            "repeated_movement_fraction": 0.65,
            "repeated_movement_steps": 650,
            "total_world_steps": 1000,
            "stalls_count": 2,
            "recovery_rate": 0.5,
        }, f)

    monitor = EarlyWarningMonitor(tmp_path)
    data = monitor.get_warnings_summary(current_stage=1, selected_eval_path=eval_file)
    assert len(data["active_warnings"]) == 1
    w_id = data["active_warnings"][0]["id"]

    # Acknowledge
    ok = monitor.acknowledge_warning(w_id, note="Tested acknowledgment")
    assert ok is True

    # Check that store file exists on disk
    store_file = tmp_path / "early-warnings.json"
    assert store_file.exists()

    # Re-read
    data_after = monitor.get_warnings_summary(current_stage=1, selected_eval_path=eval_file, force_refresh=True)
    assert len(data_after["active_warnings"]) == 0
    assert len(data_after["acknowledged_warnings"]) == 1
    assert data_after["acknowledged_warnings"][0]["status"] == WarningStatus.ACKNOWLEDGED.value

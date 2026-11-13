"""Unit tests for training session events, parameter diffs, and model change detection."""

from __future__ import annotations

import json
from pathlib import Path

from sheepdog.training.session_events import (
    diff_configs,
    extract_changelog_events,
    get_session_events,
    parse_changelog_date,
    parse_iso_timestamp,
)


def test_parse_iso_timestamp():
    dt = parse_iso_timestamp("2026-09-29T14:15:00.000Z")
    assert dt is not None
    assert dt.year == 2026
    assert dt.month == 9
    assert dt.day == 29
    assert dt.hour == 14

    assert parse_iso_timestamp(None) is None
    assert parse_iso_timestamp("invalid-date") is None


def test_parse_changelog_date():
    raw = "2026-09-29 07:15:00 PDT / 14:15:00 UTC"
    iso = parse_changelog_date(raw)
    assert iso == "2026-09-29T14:15:00+00:00"

    raw2 = "2026-09-27 13:58:37 PDT / 20:58:37 UTC"
    iso2 = parse_changelog_date(raw2)
    assert iso2 == "2026-09-27T20:58:37+00:00"


def test_diff_configs():
    c1 = {
        "environment": {"max_steps": 1820, "width": 60},
        "training": {"entropy_coef": 0.01, "learning_rate": 0.0001},
    }
    c2 = {
        "environment": {"max_steps": 2200, "width": 60},
        "training": {"entropy_coef": 0.02, "learning_rate": 0.0001},
    }
    diffs = diff_configs(c1, c2)
    assert len(diffs) == 2
    params = {d["param"]: (d["from_val"], d["to_val"]) for d in diffs}
    assert params["max_steps"] == ("1820", "2200")
    assert params["entropy_coef"] == ("0.01", "0.02")


def test_extract_changelog_events(tmp_path: Path):
    md_content = """# Changelog
### [2026-09-29 07:15:00 PDT / 14:15:00 UTC] - Stage 26 Plateau Remediation
* **Problem**: Too many timeouts.
* **Changes**:
  1. `max_steps`: Increased from `1820` to `2200`.
  2. `entropy_coef`: Added stage override `0.020`.
* **Rationale**: Grants extra steps.

### [2026-09-27 13:58:37 PDT / 20:58:37 UTC] - Net2WiderNet Model Architecture Upgrade
* **Problem**: Capacity ceiling.
* **Changes**:
  1. Widen hidden layers from `[128, 128, 128]` to `[256, 256, 128]`.
"""
    cl_file = tmp_path / "MODEL_CHANGELOG.md"
    cl_file.write_text(md_content, encoding="utf-8")

    events = extract_changelog_events(cl_file)
    assert len(events) == 2
    assert events[0]["event_type"] == "param_change"
    assert events[0]["title"] == "Stage 26 Plateau Remediation"
    assert events[0]["rationale"] == "Grants extra steps."

    assert events[1]["event_type"] == "model_change"
    assert "Net2WiderNet" in events[1]["title"]
    assert any(c["param"] == "neural_hidden_sizes" for c in events[1]["changes"])


def test_get_session_events(tmp_path: Path):
    # Setup mock artifacts
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()

    runtime_data = {
        "schema_version": 1,
        "sessions": [
            {
                "session_id": "session_alpha",
                "run_id": "run_1",
                "started_at": "2026-09-29T10:00:00+00:00",
                "ended_at": "2026-09-29T11:00:00+00:00",
                "status": "ended",
                "end_reason": "completed",
            },
            {
                "session_id": "session_beta",
                "run_id": "run_1",
                "started_at": "2026-09-29T12:00:00+00:00",
                "status": "running",
            },
        ],
    }
    (artifacts / "training-runtime.json").write_text(json.dumps(runtime_data), encoding="utf-8")

    summary_data = {
        "checkpoints": [
            {
                "checkpoint_episode": 100,
                "global_timestep": 50000,
                "session_id": "session_alpha",
                "success_rate": 0.5,
                "average_reward": 10.0,
                "average_completion_steps": 200,
            },
            {
                "checkpoint_episode": 200,
                "global_timestep": 100000,
                "session_id": "session_beta",
                "success_rate": 0.8,
                "average_reward": 50.0,
                "average_completion_steps": 150,
            },
        ],
    }
    (artifacts / "training-summary.json").write_text(json.dumps(summary_data), encoding="utf-8")

    cfg_history = {
        "revisions": [
            {
                "id": 1,
                "timestamp": "2026-09-29T10:00:00+00:00",
                "config": {"environment": {"max_steps": 1000}},
            },
            {
                "id": 2,
                "timestamp": "2026-09-29T12:00:00+00:00",
                "config": {"environment": {"max_steps": 2000}},
            },
        ],
    }
    (artifacts / "config-history.json").write_text(json.dumps(cfg_history), encoding="utf-8")

    res = get_session_events(artifacts, changelog_path=tmp_path / "MODEL_CHANGELOG.md")
    assert res["total_sessions"] == 2
    assert res["current_session_id"] == "session_beta"
    assert len(res["events"]) == 2

    e1 = res["events"][0]
    assert e1["session_id"] == "session_alpha"
    assert e1["session_stats"]["first_episode"] == 100

    e2 = res["events"][1]
    assert e2["session_id"] == "session_beta"
    assert e2["is_current"] is True
    assert e2["event_type"] == "param_change"
    assert any(c["param"] == "max_steps" and c["to_val"] == "2000" for c in e2["changes"])

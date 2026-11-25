"""Per-stage training-time records, backfill, and run-to-run comparison.

Storage is an append-only JSON-lines file (``stage-timing.jsonl``) under the
artifact output root.  One record is written per stage completion (promotion).

Definitions (all durations are seconds):

* ``wall_seconds``: ``ended_at - started_at``.  Includes pauses, stops,
  evaluation and checkpointing between those two instants.
* ``training_seconds``: training-loop seconds for that stage taken from the
  runtime ledger (``training-runtime.json``), which already excludes pauses
  and evaluation.  ``null`` when the ledger has no figure (derived records).
* ``policy_timesteps``: PPO policy timesteps consumed in the stage, from the
  difference of the cumulative ``_ts_<n>`` counter in trigger checkpoint ids.
* ``world_steps``: not recorded by the trainer today, so it is ``null``
  unless the promotion event supplies it (see ``world_steps_note``).
* ``steps_per_second``: ``policy_timesteps / training_seconds`` when training
  seconds exist, else ``policy_timesteps / wall_seconds``
  (``steps_per_second_basis`` says which).

``source`` is ``recorded`` for records written at promotion time and
``derived`` for records reconstructed from ``promotion-history.json``.
"""

from __future__ import annotations

import contextlib
import datetime
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import threading
from pathlib import Path
from typing import Any

STAGE_TIMING_FILENAME = "stage-timing.jsonl"
STAGE_TIMING_STATE_FILENAME = "stage-timing-state.json"
RUNTIME_LEDGER_FILENAME = "training-runtime.json"
EFFECTIVE_CONFIG_FILENAME = "effective-training-config.json"
RUN_LABEL_ENV_VAR = "SHEEPDOG_RUN_LABEL"

_WRITE_LOCK = threading.RLock()
_RUN_ID_TIME = re.compile(r"run_(\d{8})_(\d{6})")
_TIMESTEP_IN_ID = re.compile(r"_ts_(\d+)")

_TRAINING_CONFIG_KEYS = (
    "trainer_type",
    "policy_type",
    "observation_mode",
    "ppo_env_workers",
    "rollout_steps",
    "batch_size",
    "learning_rate",
    "gamma",
    "gae_lambda",
    "entropy_coef",
    "train_seed",
    "invalid_action_masking",
    "scenario_training_enabled",
    "failure_directed_training_enabled",
    "neural_hidden_sizes",
    "torch_num_threads",
)
_ENV_CONFIG_KEYS = ("dogs", "sheep", "max_steps", "width", "height")

_COMPARE_METRICS = ("wall_seconds", "training_seconds", "policy_timesteps", "world_steps", "steps_per_second")


def _parse_time(value: Any) -> datetime.datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=datetime.UTC)
    return parsed


def _run_id_start(run_id: str | None) -> datetime.datetime | None:
    """Parse ``run_YYYYMMDD_HHMMSS_xxxx``; the id encodes local wall-clock time."""

    match = _RUN_ID_TIME.match(run_id or "")
    if not match:
        return None
    try:
        naive = datetime.datetime.strptime("".join(match.groups()), "%Y%m%d%H%M%S")
    except ValueError:
        return None
    return naive.astimezone().astimezone(datetime.UTC)


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None


def read_records(output_root: Path) -> list[dict[str, Any]]:
    """Read all stage timing records, skipping unreadable or torn lines.

    Defensively deduplicates by ``promotion_event_id`` (preferring ``recorded``
    over ``derived`` if both appear, otherwise keeping the first occurrence).
    """

    path = Path(output_root) / STAGE_TIMING_FILENAME
    records: list[dict[str, Any]] = []
    index_by_event_id: dict[Any, int] = {}
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return records
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except ValueError:
            continue
        if not isinstance(obj, dict):
            continue
        event_id = obj.get("promotion_event_id")
        if not event_id:
            records.append(obj)
            continue
        existing_idx = index_by_event_id.get(event_id)
        if existing_idx is None:
            index_by_event_id[event_id] = len(records)
            records.append(obj)
        elif records[existing_idx].get("source") != "recorded" and obj.get("source") == "recorded":
            records[existing_idx] = obj
    return records


def _append_record(output_root: Path, record: dict[str, Any]) -> None:
    path = Path(output_root) / STAGE_TIMING_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
    with _WRITE_LOCK:
        # Guard against a previous torn write lacking a trailing newline.
        prefix = ""
        try:
            if path.exists() and path.stat().st_size > 0:
                with path.open("rb") as handle:
                    handle.seek(-1, os.SEEK_END)
                    if handle.read(1) != b"\n":
                        prefix = "\n"
        except OSError:
            prefix = ""
        with path.open("ab") as handle:
            handle.write((prefix + line).encode("utf-8"))
            handle.flush()
            os.fsync(handle.fileno())


# ---------------------------------------------------------------- environment

_GIT_CACHE: dict[str, Any] = {}


def git_info() -> dict[str, Any]:
    """Return ``{"revision": str|None, "dirty": bool|None}`` (cached per process)."""

    if _GIT_CACHE:
        return dict(_GIT_CACHE)
    repo = Path(__file__).resolve().parents[3]
    info: dict[str, Any] = {"revision": None, "dirty": None}
    try:
        rev = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, timeout=5, check=False
        )
        if rev.returncode == 0:
            info["revision"] = rev.stdout.strip() or None
            status = subprocess.run(
                ["git", "status", "--porcelain", "--untracked-files=no"],
                cwd=repo, capture_output=True, text=True, timeout=10, check=False,
            )
            if status.returncode == 0:
                info["dirty"] = bool(status.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        pass
    _GIT_CACHE.update(info)
    return dict(info)


def _version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def software_info() -> dict[str, Any]:
    """Return interpreter and ML library versions."""
    return {
        "python": platform.python_version(),
        "torch": _version("torch"),
        "stable_baselines3": _version("stable-baselines3"),
        "sb3_contrib": _version("sb3-contrib"),
    }


def hardware_info() -> dict[str, Any]:
    """Hostname-agnostic hardware summary."""

    device = "cpu"
    try:
        import torch  # noqa: PLC0415

        if torch.cuda.is_available():
            device = "cuda"
    except Exception:  # pragma: no cover - torch optional
        pass
    return {
        "cpu_count": os.cpu_count(),
        "device": device,
        "os": platform.system(),
        "machine": platform.machine(),
    }


def config_snapshot(output_root: Path) -> dict[str, Any]:
    """Compact effective-config snapshot with a stable hash and assistance mode."""

    raw = _read_json(Path(output_root) / EFFECTIVE_CONFIG_FILENAME)
    snapshot: dict[str, Any] = {}
    if isinstance(raw, dict):
        training = raw.get("training") if isinstance(raw.get("training"), dict) else {}
        env = raw.get("environment") if isinstance(raw.get("environment"), dict) else {}
        snapshot = {
            "training": {k: training[k] for k in _TRAINING_CONFIG_KEYS if k in training},
            "environment": {k: env[k] for k in _ENV_CONFIG_KEYS if k in env},
        }
    if not snapshot:
        return {"snapshot": None, "hash": None, "assistance_mode": None}
    digest = hashlib.sha256(json.dumps(snapshot, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    mode = snapshot["training"].get("observation_mode")
    return {"snapshot": snapshot, "hash": digest, "assistance_mode": mode}


def _runtime_stage_seconds(output_root: Path, run_id: str | None, stage: int) -> dict[str, float] | None:
    """Sum ledger seconds for (run, stage) across all sessions; None if absent."""

    ledger = _read_json(Path(output_root) / RUNTIME_LEDGER_FILENAME)
    if not isinstance(ledger, dict) or not run_id:
        return None
    totals: dict[str, float] = {}
    found = False
    for session in ledger.get("sessions") or []:
        if not isinstance(session, dict) or session.get("run_id") != run_id:
            continue
        stage_block = (session.get("stage_seconds") or {}).get(str(stage))
        if not isinstance(stage_block, dict):
            continue
        for key in ("training_seconds", "evaluation_seconds"):
            value = stage_block.get(key)
            if isinstance(value, int | float):
                totals[key] = totals.get(key, 0.0) + float(value)
                found = True
    return totals if found else None


# ------------------------------------------------------------ record building


def _timesteps_from_event(event: dict[str, Any]) -> int | None:
    for key in ("trigger_checkpoint_id", "checkpoint_id"):
        match = _TIMESTEP_IN_ID.search(str(event.get(key) or ""))
        if match:
            return int(match.group(1))
    return None


def _is_stage_completion(event: dict[str, Any]) -> bool:
    return (
        event.get("event_type") == "promotion"
        and event.get("from_stage") is not None
        and bool(event.get("promotion_event_id"))
    )


def _number(value: Any) -> float | None:
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else None


def _base_record(event: dict[str, Any], source: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "promotion_event_id": event.get("promotion_event_id"),
        "run_id": event.get("run_id"),
        "stage": event.get("from_stage"),
        "to_stage": event.get("to_stage"),
        "source": source,
        "ended_at": event.get("promoted_at") or event.get("timestamp"),
        "trigger_checkpoint_id": event.get("trigger_checkpoint_id") or event.get("checkpoint_id"),
        "trigger_checkpoint_episode": event.get("trigger_checkpoint_episode"),
        "trigger_policy_version": event.get("trigger_policy_version") or event.get("policy_version"),
        "success_rate": event.get("success_rate"),
        "success_count": event.get("success_count"),
        "evaluation_seed_count": event.get("evaluation_seed_count"),
        "environment_config_hash": event.get("environment_config_hash"),
        "world_steps": _number(event.get("world_steps")),
        "world_steps_note": (
            None if event.get("world_steps") is not None
            else "world steps are not recorded by the trainer; use policy_timesteps"
        ),
        "run_label": os.environ.get(RUN_LABEL_ENV_VAR) or None,
    }


def _finish_times(record: dict[str, Any]) -> None:
    start = _parse_time(record.get("started_at"))
    end = _parse_time(record.get("ended_at"))
    record["wall_seconds"] = round((end - start).total_seconds(), 3) if start and end and end >= start else None


def _finish_rates(record: dict[str, Any]) -> None:
    steps = record.get("policy_timesteps")
    for key, basis in (("training_seconds", "training_seconds"), ("wall_seconds", "wall_seconds")):
        seconds = record.get(key)
        if steps and seconds and seconds > 0:
            record["steps_per_second"] = round(steps / seconds, 3)
            record["steps_per_second_basis"] = basis
            return
    record["steps_per_second"] = None
    record["steps_per_second_basis"] = None


def _load_state(output_root: Path) -> dict[str, Any]:
    state = _read_json(Path(output_root) / STAGE_TIMING_STATE_FILENAME)
    return state if isinstance(state, dict) else {}


def note_stage_start(output_root: Path, run_id: str | None, stage: int, started_at: str, timesteps: int | None) -> None:
    """Persist when *stage* of *run_id* began so restarts do not lose it."""

    from sheepdog.atomic_io import atomic_write_json  # noqa: PLC0415

    state = _load_state(output_root)
    state["run_id"] = run_id
    state["stage"] = stage
    state["started_at"] = started_at
    state["start_timesteps"] = timesteps
    atomic_write_json(Path(output_root) / STAGE_TIMING_STATE_FILENAME, state)


def record_promotion(output_root: Path, event: dict[str, Any]) -> dict[str, Any] | None:
    """Write a ``recorded`` stage record for a promotion event; idempotent by event id."""

    if not _is_stage_completion(event):
        return None
    output_root = Path(output_root)
    event_id = event["promotion_event_id"]
    with _WRITE_LOCK:
        existing = read_records(output_root)
        if any(rec.get("promotion_event_id") == event_id for rec in existing):
            return None
        run_id = event.get("run_id")
        stage = int(event["from_stage"])
        record = _base_record(event, "recorded")
        end_ts = _timesteps_from_event(event)

        state = _load_state(output_root)
        started_at: str | None = None
        start_ts: int | None = None
        confidence = "medium"
        if state.get("run_id") == run_id and state.get("stage") == stage and state.get("started_at"):
            started_at = state["started_at"]
            start_ts = state.get("start_timesteps")
        else:
            for rec in reversed(existing):
                if rec.get("run_id") == run_id and rec.get("to_stage") == stage:
                    started_at = rec.get("ended_at")
                    start_ts = rec.get("trigger_timesteps")
                    break
        if started_at is None:
            run_start = _run_id_start(run_id)
            if run_start is not None and stage == 1:
                started_at = run_start.isoformat()
                start_ts = 0
                confidence = "low"
        record["started_at"] = started_at
        record["timing_confidence"] = confidence if started_at else "none"
        _finish_times(record)

        record["trigger_timesteps"] = end_ts
        steps = end_ts - int(start_ts) if end_ts is not None and start_ts is not None and end_ts >= int(start_ts) else None
        record["policy_timesteps"] = steps
        record["episodes"] = (_read_json(output_root / "stage-history.json") or {}).get(str(stage))

        seconds = _runtime_stage_seconds(output_root, run_id, stage)
        prior_train = sum(
            rec.get("_ledger_training_seconds") or 0.0
            for rec in existing
            if rec.get("run_id") == run_id and rec.get("stage") == stage
        )
        prior_eval = sum(
            rec.get("_ledger_evaluation_seconds") or 0.0
            for rec in existing
            if rec.get("run_id") == run_id and rec.get("stage") == stage
        )
        if seconds and "training_seconds" in seconds:
            record["_ledger_training_seconds"] = seconds["training_seconds"]
            record["training_seconds"] = round(max(0.0, seconds["training_seconds"] - prior_train), 3)
            record["training_seconds_note"] = "runtime ledger; excludes pauses and evaluation"
        else:
            record["training_seconds"] = None
            record["training_seconds_note"] = "runtime ledger had no figure for this stage"
        if seconds and "evaluation_seconds" in seconds:
            record["_ledger_evaluation_seconds"] = seconds["evaluation_seconds"]
            record["evaluation_seconds"] = round(max(0.0, seconds["evaluation_seconds"] - prior_eval), 3)
        else:
            record["evaluation_seconds"] = None
        _finish_rates(record)

        cfg = config_snapshot(output_root)
        record["config_hash"] = cfg["hash"]
        record["config_snapshot"] = cfg["snapshot"]
        record["assistance_mode"] = cfg["assistance_mode"]
        git = git_info()
        record["git_revision"] = git["revision"]
        record["git_dirty"] = git["dirty"]
        record["software"] = software_info()
        record["hardware"] = hardware_info()
        _append_record(output_root, record)

        try:
            ended = event.get("promoted_at") or event.get("timestamp")
            if ended and event.get("to_stage") is not None:
                note_stage_start(output_root, run_id, int(event["to_stage"]), ended, end_ts)
        except (OSError, ValueError, TypeError):
            pass
        return record


def backfill_from_promotion_history(output_root: Path, events: list[dict[str, Any]]) -> int:
    """Append ``derived`` records for promotion events lacking one; returns count added."""

    output_root = Path(output_root)
    with _WRITE_LOCK:
        existing = read_records(output_root)
        seen = {rec.get("promotion_event_id") for rec in existing}
        ordered = sorted(
            (e for e in events if isinstance(e, dict) and _is_stage_completion(e)),
            key=lambda e: str(e.get("promoted_at") or e.get("timestamp") or ""),
        )
        timeline = list(existing)
        added = 0
        for event in ordered:
            if event["promotion_event_id"] in seen:
                # Make sure its end time is available as a start for the next stage.
                continue
            run_id = event.get("run_id")
            stage = int(event["from_stage"])
            record = _base_record(event, "derived")
            end_ts = _timesteps_from_event(event)
            started_at, start_ts = None, None
            for rec in reversed(timeline):
                if rec.get("run_id") == run_id and rec.get("to_stage") == stage:
                    started_at, start_ts = rec.get("ended_at"), rec.get("trigger_timesteps")
                    break
            confidence = "medium"
            if started_at is None:
                run_start = _run_id_start(run_id)
                if run_start is not None and stage == 1:
                    started_at, start_ts, confidence = run_start.isoformat(), 0, "low"
            record["started_at"] = started_at
            record["timing_confidence"] = confidence if started_at else "none"
            _finish_times(record)
            record["trigger_timesteps"] = end_ts
            record["policy_timesteps"] = (
                end_ts - int(start_ts) if end_ts is not None and start_ts is not None and end_ts >= int(start_ts) else None
            )
            record["episodes"] = None
            record["training_seconds"] = None
            record["training_seconds_note"] = (
                "derived from promotion timestamps; pauses, stops and evaluation are included in wall_seconds"
            )
            record["evaluation_seconds"] = None
            _finish_rates(record)
            record["config_hash"] = None
            record["config_snapshot"] = None
            record["assistance_mode"] = None
            record["git_revision"] = None
            record["git_dirty"] = None
            record["software"] = None
            record["hardware"] = None
            record["run_label"] = None
            _append_record(output_root, record)
            timeline.append(record)
            seen.add(event["promotion_event_id"])
            added += 1
        return added


# ---------------------------------------------------------------- comparison


def _pct_delta(new: float | None, old: float | None) -> float | None:
    if new is None or old is None or old == 0:
        return None
    return round((new - old) / abs(old) * 100.0, 2)


def _compare_pair(latest: dict[str, Any], ref: dict[str, Any]) -> dict[str, Any]:
    deltas: dict[str, Any] = {}
    for metric in _COMPARE_METRICS:
        a, b = _number(latest.get(metric)), _number(ref.get(metric))
        deltas[metric] = {"latest": a, "reference": b, "pct_delta": _pct_delta(a, b)}
    cfg_a, cfg_b = latest.get("config_hash"), ref.get("config_hash")
    mode_a, mode_b = latest.get("assistance_mode"), ref.get("assistance_mode")
    config_differs = cfg_a != cfg_b
    assistance_differs = mode_a != mode_b
    notes: list[str] = []
    if assistance_differs:
        notes.append(f"assistance mode differs ({mode_b or 'unknown'} vs {mode_a or 'unknown'})")
    if config_differs:
        notes.append("config hash differs or is unknown on one side")
    if latest.get("source") != ref.get("source"):
        notes.append("mixed recorded/derived timing; wall seconds are not equally precise")
    return {
        "run_id": ref.get("run_id"),
        "deltas": deltas,
        "config_hash_differs": config_differs,
        "assistance_differs": assistance_differs,
        "comparable": not (config_differs or assistance_differs),
        "notes": notes,
    }


def compare_stages(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per-stage comparison of latest run vs previous run and vs best (fastest wall time)."""

    by_stage: dict[int, list[dict[str, Any]]] = {}
    for rec in records:
        if isinstance(rec.get("stage"), int):
            by_stage.setdefault(rec["stage"], []).append(rec)
    out: list[dict[str, Any]] = []
    for stage in sorted(by_stage):
        per_run: dict[str, dict[str, Any]] = {}
        for rec in sorted(by_stage[stage], key=lambda r: str(r.get("ended_at") or "")):
            per_run[str(rec.get("run_id"))] = rec  # latest completion per run wins
        runs = sorted(per_run.values(), key=lambda r: str(r.get("ended_at") or ""))
        entry: dict[str, Any] = {"stage": stage, "run_count": len(runs), "latest_run_id": runs[-1].get("run_id")}
        entry["vs_previous"] = _compare_pair(runs[-1], runs[-2]) if len(runs) >= 2 else None
        timed = [r for r in runs if _number(r.get("wall_seconds")) is not None]
        best = min(timed, key=lambda r: r["wall_seconds"]) if timed else None
        entry["best_run_id"] = best.get("run_id") if best else None
        entry["vs_best"] = (
            _compare_pair(runs[-1], best) if best is not None and best is not runs[-1] else None
        )
        out.append(entry)
    return out


def _public(record: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in record.items() if not k.startswith("_")}


def build_payload(output_root: Path, promotion_events: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Backfill (lazily, idempotent) then return records plus per-stage comparison."""

    if promotion_events:
        with contextlib.suppress(OSError):
            backfill_from_promotion_history(output_root, promotion_events)
    records = read_records(output_root)
    return {
        "schema_version": 1,
        "records": [_public(r) for r in records],
        "comparison": compare_stages(records),
        "definitions": {
            "wall_seconds": "ended_at minus started_at; includes pauses, evaluation and checkpointing",
            "training_seconds": "training-loop seconds from the runtime ledger; null when unavailable",
            "policy_timesteps": "PPO policy timesteps consumed in the stage",
            "steps_per_second": "policy_timesteps divided by training_seconds (else wall_seconds)",
        },
    }

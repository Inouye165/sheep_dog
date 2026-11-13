"""Extraction and synthesis of training sessions, parameter diffs, and model changes for UI visual cues."""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def parse_iso_timestamp(ts_str: str | None) -> datetime | None:
    """Safely parse an ISO-8601 timestamp string into a timezone-aware datetime."""
    if not ts_str:
        return None
    try:
        clean_str = ts_str.replace("Z", "+00:00")
        dt = datetime.fromisoformat(clean_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return dt
    except Exception:
        return None


def parse_changelog_date(raw_time: str) -> str | None:
    """Parse changelog header times like '2026-09-29 07:15:00 PDT / 14:15:00 UTC'."""
    date_match = re.search(r"(\d{4}-\d{2}-\d{2})", raw_time)
    if not date_match:
        return None
    d_str = date_match.group(1)
    utc_time_match = re.search(r"(\d{2}:\d{2}:\d{2})\s*UTC", raw_time)
    if utc_time_match:
        t_str = utc_time_match.group(1)
        return f"{d_str}T{t_str}+00:00"
    first_time_match = re.search(r"(\d{2}:\d{2}:\d{2})", raw_time)
    if first_time_match:
        t_str = first_time_match.group(1)
        return f"{d_str}T{t_str}+00:00"
    return f"{d_str}T00:00:00+00:00"


def extract_changelog_events(changelog_path: Path) -> list[dict[str, Any]]:
    """Parse structured markdown entries from MODEL_CHANGELOG.md."""
    if not changelog_path.exists():
        return []

    try:
        text = changelog_path.read_text(encoding="utf-8")
    except Exception:
        return []

    entries = []
    pattern = re.compile(
        r"###\s*\[([^\]]+)\]\s*-\s*([^\n]+)\n(.*?)(?=\n###|\Z)",
        re.DOTALL,
    )

    for match in pattern.finditer(text):
        time_part = match.group(1).strip()
        title = match.group(2).strip()
        body = match.group(3).strip()

        utc_ts = parse_changelog_date(time_part)

        is_model = any(
            kw in title.lower()
            for kw in ("architecture", "wider", "weights", "model", "net2widernet")
        )
        event_type = "model_change" if is_model else "param_change"

        changes: list[dict[str, Any]] = []
        for line in body.split("\n"):
            line = line.strip()
            diff_match = re.search(
                r"[`*]*([\w_]+)[`*]*:\s*(?:Increased|Decreased|Adjusted|Set to|Added|Widen)?\s*(?:from\s*[`*]*([^`*]+)[`*]*\s*to\s*[`*]*([^`*]+)[`*]*|to\s*[`*]*([^`*]+)[`*]*)",
                line,
                re.IGNORECASE,
            )
            if diff_match:
                p_name = diff_match.group(1)
                from_val = diff_match.group(2) if diff_match.group(2) is not None else "previous"
                to_val = diff_match.group(3) if diff_match.group(3) is not None else diff_match.group(4)
                changes.append({
                    "param": p_name,
                    "from_val": from_val.strip() if from_val else "",
                    "to_val": to_val.strip() if to_val else "",
                    "category": "architecture" if is_model else "tuning",
                })

        # Add explicit changes if none extracted by regex
        if is_model and not changes and "wider" in title.lower():
            changes.append({
                "param": "neural_hidden_sizes",
                "from_val": "[128, 128, 128] (41K params)",
                "to_val": "[256, 256, 128] (115K params, ~2.8x capacity)",
                "category": "architecture",
            })

        rationale = ""
        rat_match = re.search(r"\*{1,2}\s*Rationale\s*\*{1,2}:\s*([^\n]+)", body, re.IGNORECASE)
        if rat_match:
            rationale = rat_match.group(1).strip()
        elif "problem" in body.lower():
            prob_match = re.search(r"\*{1,2}\s*Problem\s*\*{1,2}:\s*([^\n]+)", body, re.IGNORECASE)
            if prob_match:
                rationale = prob_match.group(1).strip()

        entries.append({
            "raw_time": time_part,
            "timestamp": utc_ts,
            "title": title,
            "event_type": event_type,
            "changes": changes,
            "rationale": rationale,
        })
    return entries


def _flatten_config(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten a nested configuration dictionary."""
    items: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            items.update(_flatten_config(v, key))
        elif not isinstance(v, list) or len(v) <= 10:  # Skip long step/episode lists
            items[key] = v
    return items


def diff_configs(c1: dict[str, Any], c2: dict[str, Any]) -> list[dict[str, Any]]:
    """Compare two configuration dictionaries and return a list of human-readable diffs."""
    f1 = _flatten_config(c1)
    f2 = _flatten_config(c2)
    all_keys = set(f1.keys()) | set(f2.keys())
    diffs = []

    # Significant keys to prioritize
    significance_priority = [
        "environment.max_steps",
        "environment.no_progress_window",
        "environment.curriculum_stage",
        "training.entropy_coef",
        "training.learning_rate",
        "training.neural_hidden_sizes",
        "policy.policy_mode",
        "rewards.wait_penalty",
        "rewards.gate_corridor_progress_scale",
        "rewards.gate_alignment_scale",
        "training.auto_promote_threshold",
    ]

    for k in sorted(all_keys, key=lambda x: (x not in significance_priority, x)):
        # Ignore noisy metadata
        if any(ign in k for ign in ("checkpoint_episodes", "checkpoint_timesteps", "timestamp", "id", "label")):
            continue
        v1 = f1.get(k)
        v2 = f2.get(k)
        if v1 != v2:
            category = "environment" if k.startswith("environment.") else (
                "rewards" if k.startswith("rewards.") else (
                    "architecture" if "hidden" in k or "policy" in k else "training"
                )
            )
            short_name = k.split(".")[-1]
            diffs.append({
                "param": short_name,
                "full_path": k,
                "from_val": str(v1) if v1 is not None else "None",
                "to_val": str(v2) if v2 is not None else "None",
                "category": category,
            })
    return diffs


def get_session_events(
    output_root: Path,
    changelog_path: Path | None = None,
    runtime_tracker: Any = None,
) -> dict[str, Any]:
    """Compile and return chronological session boundaries, model changes, and parameter adjustments."""
    if changelog_path is None:
        changelog_path = output_root.parent / "MODEL_CHANGELOG.md"

    # 1. Gather all runtime sessions
    runtime_path = output_root / "training-runtime.json"
    sessions: list[dict[str, Any]] = []
    if runtime_path.exists():
        try:
            runtime_data = json.loads(runtime_path.read_text(encoding="utf-8"))
            if isinstance(runtime_data, dict) and isinstance(runtime_data.get("sessions"), list):
                sessions = [dict(s) for s in runtime_data["sessions"]]
        except Exception:
            pass

    if runtime_tracker is not None:
        try:
            snap = runtime_tracker.snapshot()
            live_sessions = snap.get("sessions", [])
            existing_sids = {s.get("session_id") for s in sessions if s.get("session_id")}
            for ls in live_sessions:
                if ls.get("session_id") and ls.get("session_id") not in existing_sids:
                    sessions.append(dict(ls))
                elif ls.get("session_id") in existing_sids:
                    # Update live status
                    for s in sessions:
                        if s.get("session_id") == ls.get("session_id"):
                            s.update(ls)
        except Exception:
            pass

    # Sort sessions chronologically
    sessions.sort(key=lambda s: s.get("started_at") or "")

    # 2. Gather checkpoints to map metrics and coordinates to sessions
    summary_path = output_root / "training-summary.json"
    checkpoints: list[dict[str, Any]] = []
    if summary_path.exists():
        try:
            summary_data = json.loads(summary_path.read_text(encoding="utf-8"))
            if isinstance(summary_data, dict):
                checkpoints = summary_data.get("checkpoints", [])
            elif isinstance(summary_data, list):
                checkpoints = summary_data
        except Exception:
            pass

    session_chks: dict[str, list[dict[str, Any]]] = {}
    for c in checkpoints:
        sid = c.get("session_id")
        if sid:
            session_chks.setdefault(sid, []).append(c)

    # 3. Gather config revisions
    cfg_history_path = output_root / "config-history.json"
    revisions: list[dict[str, Any]] = []
    if cfg_history_path.exists():
        try:
            cfg_data = json.loads(cfg_history_path.read_text(encoding="utf-8"))
            if isinstance(cfg_data, dict):
                revisions = cfg_data.get("revisions", [])
        except Exception:
            pass

    # 4. Gather changelog explicit events
    changelog_events = extract_changelog_events(changelog_path)

    # 5. Gather promotions
    promo_path = output_root / "promotion-history.json"
    promotions: list[dict[str, Any]] = []
    if promo_path.exists():
        try:
            promo_data = json.loads(promo_path.read_text(encoding="utf-8"))
            if isinstance(promo_data, list):
                promotions = promo_data
        except Exception:
            pass

    events: list[dict[str, Any]] = []
    current_session_id = sessions[-1].get("session_id") if sessions else None

    # Track prior revision config to calculate diffs across sessions
    prev_config: dict[str, Any] | None = None

    for idx, s in enumerate(sessions):
        sid = s.get("session_id")
        if not sid:
            continue

        started_at = s.get("started_at")
        ended_at = s.get("ended_at") or s.get("last_heartbeat_at")
        start_dt = parse_iso_timestamp(started_at)
        start_ms = int(start_dt.timestamp() * 1000) if start_dt else 0

        end_dt = parse_iso_timestamp(ended_at)
        end_ms = int(end_dt.timestamp() * 1000) if end_dt else None

        c_list = session_chks.get(sid, [])
        first_ep = c_list[0].get("checkpoint_episode") if c_list else None
        last_ep = c_list[-1].get("checkpoint_episode") if c_list else None
        first_ts = c_list[0].get("global_timestep") if c_list else None
        last_ts = c_list[-1].get("global_timestep") if c_list else None

        # Performance summary for this session
        avg_success = (
            sum(float(c.get("success_rate", 0)) for c in c_list) / len(c_list)
            if c_list else None
        )
        avg_reward = (
            sum(float(c.get("average_reward", 0)) for c in c_list) / len(c_list)
            if c_list else None
        )
        steps_list = [
            float(c.get("average_completion_steps", 0))
            for c in c_list
            if c.get("average_completion_steps") is not None
        ]
        avg_steps = (sum(steps_list) / len(steps_list)) if steps_list else None

        session_stats = {
            "checkpoint_count": len(c_list),
            "first_episode": first_ep,
            "last_episode": last_ep,
            "first_timestep": first_ts,
            "last_timestep": last_ts,
            "avg_success_rate": round(avg_success * 100, 1) if avg_success is not None else None,
            "avg_reward": round(avg_reward, 1) if avg_reward is not None else None,
            "avg_steps": round(avg_steps, 0) if avg_steps is not None else None,
            "training_seconds": s.get("training_seconds"),
            "active_seconds_total": s.get("active_seconds_total"),
            "end_reason": s.get("end_reason"),
            "status": s.get("status"),
        }

        # Check for correlated config revision
        matched_rev: dict[str, Any] | None = None
        if start_dt and revisions:
            # Find closest revision within 300 seconds
            best_diff = 300.0
            for r in revisions:
                r_dt = parse_iso_timestamp(r.get("timestamp"))
                if r_dt:
                    diff = abs((r_dt - start_dt).total_seconds())
                    if diff < best_diff:
                        best_diff = diff
                        matched_rev = r

        changes: list[dict[str, Any]] = []
        if matched_rev and prev_config:
            rev_cfg = matched_rev.get("config", {})
            changes = diff_configs(prev_config, rev_cfg)

        if matched_rev:
            prev_config = matched_rev.get("config", {})

        # Check for changelog events matching this session's timeframe
        matched_cl: dict[str, Any] | None = None
        if start_dt:
            for cle in changelog_events:
                cl_dt = parse_iso_timestamp(cle.get("timestamp"))
                if cl_dt:
                    diff = abs((cl_dt - start_dt).total_seconds())
                    # Within 1 hour of session start
                    if diff < 3600.0:
                        matched_cl = cle
                        break

        # Check for stage promotions in this session
        matched_promo: dict[str, Any] | None = None
        if start_dt:
            for p in promotions:
                p_dt = parse_iso_timestamp(p.get("promoted_at") or p.get("timestamp"))
                if p_dt:
                    diff = (p_dt - start_dt).total_seconds()
                    # Just before or during this session
                    if -60.0 <= diff <= 120.0:
                        matched_promo = p
                        break

        # Determine event type and title
        event_type = "session_start"
        title = f"Session #{idx + 1} Started"
        summary = f"Training session {sid[:12]}… initiated"
        badge_icon = "🔄"
        badge_color = "#38bdf8"
        rationale = None

        if matched_cl:
            event_type = matched_cl["event_type"]
            title = matched_cl["title"]
            badge_icon = "🧠" if event_type == "model_change" else "⚙️"
            badge_color = "#c084fc" if event_type == "model_change" else "#f59e0b"
            rationale = matched_cl.get("rationale")
            # Merge changelog specific changes if not captured by config diff
            if matched_cl.get("changes"):
                existing_params = {c["param"] for c in changes}
                for clc in matched_cl["changes"]:
                    if clc["param"] not in existing_params:
                        changes.append(clc)
        elif matched_promo:
            event_type = "promotion"
            title = f"Stage Promotion: Stage {matched_promo.get('from_stage')} → {matched_promo.get('to_stage')}"
            summary = f"Model achieved qualifying streak and promoted to Stage {matched_promo.get('to_stage')}"
            badge_icon = "🏆"
            badge_color = "#34d399"
        elif changes:
            event_type = "param_change"
            title = f"Session #{idx + 1}: Parameter Update"
            summary = f"{len(changes)} configuration parameters updated"
            badge_icon = "⚙️"
            badge_color = "#f59e0b"

        events.append({
            "id": f"evt_{sid}",
            "session_id": sid,
            "session_index": idx + 1,
            "run_id": s.get("run_id"),
            "timestamp": started_at,
            "timestamp_ms": start_ms,
            "ended_at": ended_at,
            "ended_ms": end_ms,
            "checkpoint_episode": first_ep,
            "global_timestep": first_ts,
            "event_type": event_type,
            "title": title,
            "summary": summary,
            "changes": changes,
            "rationale": rationale,
            "badge_icon": badge_icon,
            "badge_color": badge_color,
            "session_stats": session_stats,
            "is_current": sid == current_session_id,
        })

    return {
        "events": events,
        "current_session_id": current_session_id,
        "total_sessions": len(sessions),
    }

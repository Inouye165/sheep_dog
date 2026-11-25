"""Read-only 50-seed unseen held-out and adversarial recovery audit script.

Evaluates a trained checkpoint on 50 disjoint prime held-out seeds (>= 1009)
plus adversarial scenario families from ``list_scenario_types()``, computes
Wilson 95% confidence intervals and per-failure diagnostics, retains compressed
replays only for failures and up to 2 representative successes, and writes
``audit_report.json`` and ``audit_report.md`` to an isolated output directory.
"""

from __future__ import annotations

import argparse
import contextlib
import gzip
import json
import math
import os
import re
import sys
import zipfile
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sheepdog.atomic_io import atomic_write_json, atomic_write_text
from sheepdog.checkpoints.sidecar import compute_file_sha256, load_and_verify_sidecar
from sheepdog.config import LabConfig, validate_observation_history_strides
from sheepdog.curriculum import apply_training_profile
from sheepdog.environment import EpisodeResult, SheepdogEnvironment
from sheepdog.evaluation.evaluator import EvaluationRecord, Evaluator, _policy_metadata
from sheepdog.policies.neural import NeuralPolicy
from sheepdog.replay.store import ReplayStore
from sheepdog.tracking.registry import classify_assistance_mode
from sheepdog.training.diagnostics.repetition_recovery import analyze_repetition_and_recovery
from sheepdog.training.training_scenarios import get_scenario_builder, list_scenario_types

DEV_SEEDS: tuple[int, ...] = (11, 23, 37, 41, 53, 59, 61, 67, 71, 73)

HELDOUT_50_SEEDS: tuple[int, ...] = (
    1009, 1013, 1019, 1021, 1031, 1033, 1039, 1049, 1051, 1061,
    1063, 1069, 1087, 1091, 1093, 1097, 1103, 1109, 1117, 1123,
    1129, 1151, 1153, 1163, 1171, 1181, 1187, 1193, 1201, 1213,
    1217, 1223, 1229, 1231, 1237, 1249, 1259, 1277, 1279, 1283,
    1289, 1291, 1297, 1301, 1303, 1307, 1319, 1321, 1327, 1361,
)

ADVERSARIAL_SCENARIO_SEEDS: tuple[int, ...] = (2003, 2011)

Z_95 = 1.959963984540054

_FORBIDDEN_OUTPUT_TARGETS: tuple[Path, ...] = (
    Path("artifacts/checkpoints"),
    Path("artifacts/training-state.json"),
    Path("artifacts/training-telemetry.sqlite"),
    Path("artifacts/model-history.sqlite"),
)


def is_prime(n: int) -> bool:
    """Return True if *n* is a prime integer >= 2."""
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    limit = math.isqrt(n)
    return all(n % factor != 0 for factor in range(3, limit + 1, 2))


def validate_heldout_seeds(
    seeds: Sequence[int] = HELDOUT_50_SEEDS,
    *,
    dev_seeds: Sequence[int] = DEV_SEEDS,
    expected_count: int = 50,
    min_value: int = 1009,
) -> tuple[int, ...]:
    """Validate that held-out seeds are prime, unique, >= min_value, and disjoint from dev seeds."""
    seed_tuple = tuple(int(s) for s in seeds)
    if len(seed_tuple) != expected_count:
        raise ValueError(f"Expected {expected_count} held-out seeds, got {len(seed_tuple)}")
    if len(set(seed_tuple)) != len(seed_tuple):
        raise ValueError("Held-out seeds must be unique")
    non_primes = [s for s in seed_tuple if not is_prime(s)]
    if non_primes:
        raise ValueError(f"Held-out seeds contain non-primes: {non_primes}")
    too_small = [s for s in seed_tuple if s < min_value]
    if too_small:
        raise ValueError(f"Held-out seeds must be >= {min_value}, found {too_small}")
    overlap = set(seed_tuple) & set(int(s) for s in dev_seeds)
    if overlap:
        raise ValueError(f"Held-out seeds overlap dev seeds: {sorted(overlap)}")
    return seed_tuple


validate_heldout_seeds(HELDOUT_50_SEEDS)


def wilson_ci_95(successes: int, total: int) -> tuple[float | None, float | None]:
    """Compute the exact 95% Wilson score confidence interval for a binomial proportion."""
    if total <= 0:
        return (None, None)
    if successes < 0 or successes > total:
        raise ValueError(f"successes ({successes}) must be in [0, {total}]")
    p = successes / total
    z2 = Z_95 * Z_95
    denom = 1.0 + z2 / total
    center = (p + z2 / (2.0 * total)) / denom
    half_width = (Z_95 * math.sqrt((p * (1.0 - p) + z2 / (4.0 * total)) / total)) / denom
    low = max(0.0, center - half_width)
    high = min(1.0, center + half_width)
    return (low, high)


def _validate_output_dir(output_dir: Path, *, overwrite: bool) -> Path:
    """Validate and prepare an isolated output directory without touching active state."""
    resolved = output_dir.resolve()
    for forbidden in _FORBIDDEN_OUTPUT_TARGETS:
        forbidden_resolved = forbidden.resolve()
        if resolved == forbidden_resolved or (
            forbidden.name == "checkpoints" and forbidden_resolved in resolved.parents
        ):
            raise ValueError(f"Refusing to write into protected path: {output_dir}")

    if output_dir.exists():
        if output_dir.is_file():
            raise ValueError(f"Output path exists and is a file: {output_dir}")
        if any(output_dir.iterdir()) and not overwrite:
            raise FileExistsError(
                f"Output directory {output_dir} is non-empty; pass --overwrite to reuse it"
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _inspect_model_zip(model_path: Path) -> dict[str, Any]:
    """Read observation space shape, net_arch, and gamma from a MaskablePPO zip archive."""
    info: dict[str, Any] = {}
    try:
        with zipfile.ZipFile(model_path, "r") as zf:
            if "data" not in zf.namelist():
                return info
            data = json.loads(zf.read("data").decode("utf-8"))
    except (OSError, zipfile.BadZipFile, json.JSONDecodeError, UnicodeDecodeError):
        return info

    obs_space = data.get("observation_space")
    if isinstance(obs_space, dict):
        shape = obs_space.get("_shape")
        if isinstance(shape, list) and len(shape) == 1:
            info["observation_size"] = int(shape[0])
    policy_kwargs = data.get("policy_kwargs")
    if isinstance(policy_kwargs, dict) and isinstance(policy_kwargs.get("net_arch"), list):
        info["hidden_sizes"] = tuple(int(v) for v in policy_kwargs["net_arch"])
    if isinstance(data.get("gamma"), (int, float)):
        info["gamma"] = float(data["gamma"])
    return info


def _read_json_if_exists(path: Path) -> dict[str, Any] | None:
    """Safely read a JSON file without modifying it."""
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def _resolve_assistance_disclosure(obs_mode: str | None) -> dict[str, Any]:
    """Classify and explicitly disclose assistance mode without claiming autonomy for guided runs."""
    classified = classify_assistance_mode(obs_mode)
    if obs_mode in {"guided", "guided_gate_control", "guided_gate_treatment"}:
        effective_mode = "guided"
    elif obs_mode == "emergent_gate":
        effective_mode = "emergent_gate"
    elif classified in {"guided", "emergent"}:
        effective_mode = classified
    else:
        effective_mode = classified

    return {
        "observation_mode": obs_mode,
        "assistance_mode": effective_mode,
        "registry_classified_assistance_mode": classified,
        "is_guided": effective_mode == "guided",
        "is_autonomous_claimed": False,
        "disclosure": (
            "Guided observations include scripted TeamStrategy roles/targets and must NOT be "
            "labeled autonomous."
            if effective_mode == "guided"
            else (
                "Emergent gate observations omit TeamStrategy roles/targets and expose explicit "
                "gate geometry features; autonomy requires dedicated audit verification."
                if effective_mode == "emergent_gate"
                else "Observation mode is non-guided or unknown; autonomy is not claimed without audit."
            )
        ),
    }


def _resolve_audit_config(
    model_path: Path,
    *,
    stage: int,
    output_dir: Path,
    base_config: LabConfig | None = None,
) -> tuple[LabConfig, dict[str, Any] | None, dict[str, Any] | None, dict[str, Any]]:
    """Build evaluation LabConfig matching checkpoint metadata without mutating artifacts."""
    zip_info = _inspect_model_zip(model_path)
    sidecar = load_and_verify_sidecar(model_path)
    policy_config_hint: dict[str, Any] | None = None
    matched_source = "curriculum_profile"

    if base_config is not None:
        config = base_config
        matched_source = "explicit_config"
    else:
        config = apply_training_profile(LabConfig(), curriculum_stage=stage)
        artifacts_dir = Path("artifacts")
        training_state = _read_json_if_exists(artifacts_dir / "training-state.json")
        effective_cfg = _read_json_if_exists(artifacts_dir / "effective-training-config.json")

        resolved_model = model_path.resolve()
        state_matches_model = False
        if isinstance(training_state, dict):
            for key in ("best_model_path", "policy_state_path"):
                raw_path = training_state.get(key)
                if isinstance(raw_path, str) and raw_path:
                    with contextlib.suppress(OSError):
                        if Path(raw_path).resolve() == resolved_model:
                            state_matches_model = True
                            break
            if not state_matches_model and training_state.get("best_model_curriculum_stage") == stage:
                expected_stage_dir = (artifacts_dir / "checkpoints" / f"stage{stage}").resolve()
                if resolved_model.parent == expected_stage_dir:
                    state_matches_model = True

        if state_matches_model and isinstance(effective_cfg, dict):
            try:
                candidate = LabConfig.from_dict(effective_cfg)
                if candidate.environment.curriculum_stage == stage:
                    config = candidate
                    matched_source = "effective-training-config.json"
            except Exception:
                pass

        if state_matches_model and isinstance(training_state, dict):
            raw_pc = training_state.get("policy_config")
            if isinstance(raw_pc, dict):
                zip_obs = zip_info.get("observation_size")
                zip_arch = zip_info.get("hidden_sizes")
                pc_obs = raw_pc.get("observation_size")
                pc_arch = tuple(int(v) for v in raw_pc.get("hidden_sizes", ())) if raw_pc.get("hidden_sizes") else None
                if (zip_obs is None or pc_obs == zip_obs) and (zip_arch is None or pc_arch == zip_arch):
                    policy_config_hint = dict(raw_pc)

        if "hidden_sizes" in zip_info and tuple(config.training.neural_hidden_sizes) != zip_info["hidden_sizes"]:
            config = replace(
                config,
                training=replace(config.training, neural_hidden_sizes=zip_info["hidden_sizes"]),
            )

        if policy_config_hint is not None:
            raw_obs_mode = policy_config_hint.get("observation_mode")
            raw_strides = policy_config_hint.get("observation_history_strides")
            raw_loop_flags = policy_config_hint.get("include_loop_flags_in_observation")
            training_updates: dict[str, Any] = {}
            if isinstance(raw_obs_mode, str) and raw_obs_mode:
                training_updates["observation_mode"] = raw_obs_mode
            if isinstance(raw_strides, list) and raw_strides:
                training_updates["observation_history_strides"] = validate_observation_history_strides(
                    [int(s) for s in raw_strides]
                )
            if raw_loop_flags is not None:
                training_updates["include_loop_flags_in_observation"] = bool(raw_loop_flags)
            if training_updates:
                config = replace(config, training=replace(config.training, **training_updates))
        elif "observation_size" in zip_info:
            zip_obs = zip_info["observation_size"]
            # Adjust observation_mode when no metadata hint exists and single-frame size matches emergent (50) vs guided (54).
            if config.training.observation_history_strides == (0,):
                if zip_obs == 50 and config.training.observation_mode == "guided":
                    config = replace(
                        config,
                        training=replace(config.training, observation_mode="emergent"),
                    )
                elif zip_obs == 58 and config.training.observation_mode in {"guided", "emergent"}:
                    config = replace(
                        config,
                        training=replace(config.training, observation_mode="guided_gate_treatment"),
                    )

        if config.policy.policy_mode != "neural_policy":
            config = replace(config, policy=replace(config.policy, policy_mode="neural_policy"))

    # Keep NeuralPolicy.load isolated from artifacts/tb_logs and multi-process training VecEnvs.
    load_config = replace(
        config,
        training=replace(
            config.training,
            output_dir=str(output_dir),
            ppo_env_workers=1,
        ),
    )
    eval_config = replace(
        config,
        training=replace(
            config.training,
            output_dir=str(output_dir),
        ),
    )
    return eval_config, policy_config_hint, sidecar, {
        "config_source": matched_source,
        "zip_metadata": zip_info,
        "load_config": load_config,
    }


def _infer_checkpoint_episode(model_path: Path) -> int:
    """Extract numeric checkpoint suffix if present, otherwise return 0."""
    match = re.search(r"(\d+)$", model_path.stem)
    if match:
        return int(match.group(1))
    return 0


def _load_replay_payload(replay_path: Path) -> dict[str, Any]:
    """Read a plain JSON or gzip-compressed JSON replay file."""
    if replay_path.suffix == ".gz":
        with gzip.open(replay_path, "rt", encoding="utf-8") as handle:
            return json.load(handle)
    return json.loads(replay_path.read_text(encoding="utf-8"))


def _compute_repeated_movement_stats(frames: Sequence[dict[str, Any]]) -> tuple[int, float]:
    """Count world steps where at least one dog is in a 4-step 2-position loop."""
    total_steps = len(frames)
    if total_steps == 0:
        return 0, 0.0

    dog_histories: dict[int, list[tuple[int, int]]] = {}
    repeated_steps = 0

    for frame in frames:
        snap = frame.get("snapshot", frame)
        dogs = snap.get("dogs", []) if isinstance(snap, dict) else []
        step_repeated = False
        for idx, dog in enumerate(dogs):
            if isinstance(dog, dict):
                pos = (int(dog["x"]), int(dog["y"]))
                dog_id = int(dog.get("index", idx))
            else:
                pos = (int(getattr(dog, "x", 0)), int(getattr(dog, "y", 0)))
                dog_id = int(getattr(dog, "index", idx))
            hist = dog_histories.setdefault(dog_id, [])
            hist.append(pos)
            if len(hist) >= 4:
                p0, p1, p2, p3 = hist[-4], hist[-3], hist[-2], hist[-1]
                if p3 == p1 and p2 == p0 and len({p0, p1, p2, p3}) <= 2:
                    step_repeated = True
        if step_repeated:
            repeated_steps += 1

    return repeated_steps, repeated_steps / total_steps


def _termination_reason_from_stats(stats: dict[str, Any] | Any) -> str:
    """Classify termination reason cleanly as no_progress, timeout, or other."""
    if isinstance(stats, dict):
        stopped_no_progress = bool(stats.get("stopped_no_progress", False))
        timeout = bool(stats.get("timeout", False))
        stopped = bool(stats.get("stopped", False))
        stop_reason = str(stats.get("stop_reason", "") or "")
    else:
        stopped_no_progress = bool(getattr(stats, "stopped_no_progress", False))
        timeout = bool(getattr(stats, "timeout", False))
        stopped = bool(getattr(stats, "stopped", False))
        stop_reason = str(getattr(stats, "stop_reason", "") or "")

    if stopped_no_progress or stop_reason == "no-progress" or (stopped and not timeout):
        return "no_progress"
    if timeout or stop_reason == "timeout":
        return "timeout"
    return "other"


def _extract_unpenned_sheep_diagnostics(final_snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    """Compute wall/corner margins and personalities for all unpenned sheep in final snapshot."""
    width = int(final_snapshot.get("grid_width") or final_snapshot.get("field_width") or 0)
    height = int(final_snapshot.get("grid_height") or final_snapshot.get("field_height") or 0)
    unpenned: list[dict[str, Any]] = []
    for sheep in final_snapshot.get("sheep", []):
        if bool(sheep.get("penned", False)):
            continue
        x = int(sheep["x"])
        y = int(sheep["y"])
        wall_margin = min(x, max(0, width - 1 - x), y, max(0, height - 1 - y))
        unpenned.append({
            "index": int(sheep.get("index", len(unpenned))),
            "personality": str(sheep.get("personality") or "unknown"),
            "x": x,
            "y": y,
            "wall_margin": wall_margin,
            "near_wall_or_corner": wall_margin <= 3,
        })
    return unpenned


def _build_failure_diagnostic_from_bundle(
    *,
    bundle: dict[str, Any],
    replay_path: Path | None,
    label: str,
    seed: int,
    scenario_type: str | None = None,
) -> dict[str, Any]:
    """Compute per-failure diagnostics from a replay bundle."""
    stats = bundle.get("stats", {})
    final_snapshot = bundle.get("final_snapshot", {})
    frames = bundle.get("frames", [])
    sheep_list = final_snapshot.get("sheep", [])
    total_sheep = len(sheep_list)
    sheep_penned = int(stats.get("sheep_penned", sum(1 for s in sheep_list if s.get("penned"))))
    unpenned_sheep = _extract_unpenned_sheep_diagnostics(final_snapshot)
    rep_steps, rep_fraction = _compute_repeated_movement_stats(frames)

    rep_recovery_summary: dict[str, Any] | None = None
    with contextlib.suppress(Exception):
        rr = analyze_repetition_and_recovery(bundle, dog_count=int(bundle.get("dog_count", 3)))
        rep_recovery_summary = {
            "active_repetition_steps": rr.repeated_movement_steps,
            "active_repetition_fraction": (
                round(rr.repeated_movement_fraction, 4)
                if rr.repeated_movement_fraction is not None
                else None
            ),
            "stalls_count": rr.stalls_count,
            "recovery_rate": round(rr.recovery_rate, 4) if rr.recovery_rate is not None else None,
        }

    return {
        "label": label,
        "seed": int(seed),
        "scenario_type": scenario_type,
        "termination_reason": _termination_reason_from_stats(stats),
        "steps": int(stats.get("steps", len(frames))),
        "sheep_penned": sheep_penned,
        "total_sheep": total_sheep,
        "reward_total": round(float(stats.get("reward_total", 0.0)), 4),
        "no_progress_steps": int(stats.get("no_progress_steps", 0)),
        "unpenned_sheep": unpenned_sheep,
        "has_near_wall_or_corner_sheep": any(s["near_wall_or_corner"] for s in unpenned_sheep),
        "repeated_movement_steps": rep_steps,
        "repeated_movement_fraction": round(rep_fraction, 6),
        "repetition_recovery_summary": rep_recovery_summary,
        "replay_path": str(replay_path) if replay_path is not None else None,
    }


def _episode_result_to_bundle(
    result: EpisodeResult,
    policy: NeuralPolicy,
    config: LabConfig,
    *,
    scenario_type: str | None = None,
) -> dict[str, Any]:
    """Serialize an EpisodeResult into a standard compressed replay bundle dict."""
    trainer_type, policy_type, replay_mode = _policy_metadata(
        result.policy_name,
        trainer_type=getattr(policy, "trainer_type", None),
        policy_type=getattr(policy, "policy_type", None),
    )
    return {
        "seed": result.seed,
        "scenario_type": scenario_type,
        "policy_name": result.policy_name,
        "trainer_type": trainer_type,
        "policy_type": policy_type,
        "policy_mode": result.policy_name,
        "replay_mode": replay_mode,
        "environment": {
            **asdict(config.environment),
            "curriculum_stage": config.rewards.instincts.curriculum_stage,
            "enable_instinct_rewards": config.rewards.instincts.enable_instinct_rewards,
        },
        "dog_count": int(config.environment.dogs),
        "initial_snapshot": (
            result.initial_snapshot.to_dict() if result.initial_snapshot is not None else None
        ),
        "final_snapshot": result.final_snapshot.to_dict(),
        "stats": asdict(result.stats),
        "frames": [frame.to_dict() for frame in result.replay],
    }


def _aggregate_failure_summary(failures: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate termination reasons, near-misses, wall/corner traps, loops, and personalities."""
    term_counts: Counter[str] = Counter({"no_progress": 0, "timeout": 0, "other": 0})
    personality_counts: Counter[str] = Counter()
    near_miss_5_of_6 = 0
    partial_0_to_4 = 0
    near_wall_or_corner_failures = 0
    high_repetition_failures = 0

    for item in failures:
        reason = str(item.get("termination_reason", "other"))
        term_counts[reason] += 1
        penned = int(item.get("sheep_penned", 0))
        total = int(item.get("total_sheep", 6))
        if total >= 6 and penned == total - 1:
            near_miss_5_of_6 += 1
        elif penned < total:
            partial_0_to_4 += 1
        if item.get("has_near_wall_or_corner_sheep"):
            near_wall_or_corner_failures += 1
        if float(item.get("repeated_movement_fraction", 0.0)) >= 0.25:
            high_repetition_failures += 1
        for sheep in item.get("unpenned_sheep", []):
            personality_counts[str(sheep.get("personality", "unknown"))] += 1

    return {
        "total_failures": len(failures),
        "by_termination_reason": {
            "no_progress": int(term_counts["no_progress"]),
            "timeout": int(term_counts["timeout"]),
            "other": int(term_counts["other"]),
        },
        "near_miss_5_of_6_count": near_miss_5_of_6,
        "partial_0_to_4_count": partial_0_to_4,
        "near_wall_or_corner_failures": near_wall_or_corner_failures,
        "high_repetition_failures": high_repetition_failures,
        "unpenned_personality_counts": dict(sorted(personality_counts.items())),
    }


def _select_representative_successes(
    records: Sequence[EvaluationRecord],
) -> tuple[EvaluationRecord, ...]:
    """Select up to 2 representative success records (fastest and slowest)."""
    successes = [r for r in records if r.success]
    if not successes:
        return ()
    fastest = min(successes, key=lambda r: (r.steps, r.seed))
    slowest = max(successes, key=lambda r: (r.steps, -r.seed))
    if fastest.seed == slowest.seed:
        return (fastest,)
    return (fastest, slowest)


def _format_ci_pct(ci: tuple[float | None, float | None]) -> str:
    """Format a Wilson CI tuple as a percentage range string."""
    if ci[0] is None or ci[1] is None:
        return "N/A"
    return f"[{ci[0] * 100:.1f}%, {ci[1] * 100:.1f}%]"


def _render_markdown_report(report: dict[str, Any]) -> str:
    """Render a concise human-readable Markdown audit report."""
    model_info = report["model"]
    heldout = report["heldout_50"]
    adversarial = report["adversarial_scenarios"]
    failure_summary = report["failure_summary"]
    assistance = report["assistance"]

    lines = [
        "# Held-Out 50-Seed & Adversarial Recovery Audit Report",
        "",
        f"- **Generated (UTC)**: `{report['generated_at']}`",
        f"- **Model Path**: `{model_info['path']}`",
        f"- **Model SHA-256**: `{model_info['sha256']}`",
        f"- **Curriculum Stage**: `{report['stage']}`",
        f"- **Observation Mode**: `{assistance['observation_mode']}`",
        f"- **Assistance Mode**: `{assistance['assistance_mode']}` (is_autonomous_claimed=`{assistance['is_autonomous_claimed']}`)",
        f"- **Observation History Strides**: `{report['observation_history_strides']}`",
        f"- **Include Loop Flags**: `{report['include_loop_flags_in_observation']}`",
        f"- **Disclosure**: {assistance['disclosure']}",
        "",
        "## Held-Out Prime Seeds Summary",
        "",
        f"- **Seeds Evaluated**: `{heldout['total']}` (disjoint from dev seeds `{list(DEV_SEEDS)}`)",
        f"- **Successes**: `{heldout['successes']}/{heldout['total']}` (`{heldout['success_rate'] * 100:.1f}%`)",
        f"- **Wilson 95% CI**: `{_format_ci_pct((heldout['wilson_ci_95']['low'], heldout['wilson_ci_95']['high']))}`",
        f"- **Average Steps**: `{heldout['average_steps']:.1f}`",
        f"- **Average Sheep Penned**: `{heldout['average_sheep_penned']:.2f}`",
        f"- **Average Reward**: `{heldout['average_reward']:.2f}`",
        "",
        "## Adversarial Scenario Summary",
        "",
    ]
    if adversarial["skipped"]:
        lines.append("- Adversarial scenario suite was skipped (`--skip-scenarios`).")
    else:
        adv_ci = (adversarial["wilson_ci_95"]["low"], adversarial["wilson_ci_95"]["high"])
        lines.extend([
            f"- **Runs Evaluated**: `{adversarial['total']}` (`{len(adversarial['scenario_types'])}` types x `{len(adversarial['seeds'])}` seeds)",
            f"- **Successes**: `{adversarial['successes']}/{adversarial['total']}` (`{(adversarial['success_rate'] or 0.0) * 100:.1f}%`)",
            f"- **Wilson 95% CI**: `{_format_ci_pct(adv_ci)}`",
            "",
            "| Scenario Type | Successes | Total | Success Rate |",
            "| --- | ---: | ---: | ---: |",
        ])
        for stype, sdata in adversarial["by_scenario_type"].items():
            rate = (sdata["success_rate"] or 0.0) * 100.0
            lines.append(f"| `{stype}` | {sdata['successes']} | {sdata['total']} | {rate:.1f}% |")

    lines.extend([
        "",
        "## Failure Diagnostics Summary",
        "",
        f"- **Total Failures (Held-Out + Adversarial)**: `{failure_summary['total_failures']}`",
        f"- **Termination Reasons**: `no_progress={failure_summary['by_termination_reason']['no_progress']}`, `timeout={failure_summary['by_termination_reason']['timeout']}`, `other={failure_summary['by_termination_reason']['other']}`",
        f"- **Near-Miss (5/6 penned)**: `{failure_summary['near_miss_5_of_6_count']}`",
        f"- **Other Partial (0-4/6 penned)**: `{failure_summary['partial_0_to_4_count']}`",
        f"- **Near Wall/Corner Failures (`wall_margin <= 3`)**: `{failure_summary['near_wall_or_corner_failures']}`",
        f"- **High Repetition Failures (`fraction >= 0.25`)**: `{failure_summary['high_repetition_failures']}`",
        f"- **Unpenned Sheep Personalities**: `{failure_summary['unpenned_personality_counts']}`",
        "",
        "## Retained Replays",
        "",
    ])
    for rep in report["retained_replays"]["representative_successes"]:
        lines.append(f"- **Representative Success ({rep['role']}, seed {rep['seed']}, {rep['steps']} steps)**: `{rep['replay_path']}`")
    for fail in report["failures"]:
        lines.append(
            f"- **Failure ({fail['label']}, {fail['termination_reason']}, {fail['sheep_penned']}/{fail['total_sheep']} penned, loop_frac={fail['repeated_movement_fraction']:.3f})**: `{fail['replay_path']}`"
        )
    lines.append("")
    return "\n".join(lines)


def run_audit(
    *,
    model_path: str | Path = Path("artifacts/checkpoints/stage38/stage38-best-model.zip"),
    stage: int = 38,
    output_dir: str | Path | None = None,
    overwrite: bool = False,
    workers: int | None = None,
    seed_count: int = 50,
    skip_scenarios: bool = False,
    scenario_types: Sequence[str] | None = None,
    scenario_seeds: Sequence[int] = ADVERSARIAL_SCENARIO_SEEDS,
    base_config: LabConfig | None = None,
) -> dict[str, Any]:
    """Execute the read-only held-out and adversarial recovery audit."""
    if not 1 <= int(seed_count) <= len(HELDOUT_50_SEEDS):
        raise ValueError(f"seed_count must be between 1 and {len(HELDOUT_50_SEEDS)}, got {seed_count}")

    resolved_model_path = Path(model_path)
    if not resolved_model_path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {resolved_model_path}")

    validate_heldout_seeds(HELDOUT_50_SEEDS)
    selected_seeds = HELDOUT_50_SEEDS[: int(seed_count)]

    if output_dir is None:
        stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        resolved_output_dir = Path(f"artifacts/experiments/heldout_50_audit_{stamp}")
    else:
        resolved_output_dir = Path(output_dir)

    _validate_output_dir(resolved_output_dir, overwrite=overwrite)

    effective_workers = (
        max(1, int(workers))
        if workers is not None
        else min(8, max(1, (os.cpu_count() or 2) // 2))
    )

    eval_config, policy_config_hint, sidecar, cfg_meta = _resolve_audit_config(
        resolved_model_path,
        stage=stage,
        output_dir=resolved_output_dir,
        base_config=base_config,
    )
    load_config = cfg_meta["load_config"]

    model_sha256 = compute_file_sha256(resolved_model_path)
    checkpoint_episode = _infer_checkpoint_episode(resolved_model_path)

    policy = NeuralPolicy.load(
        resolved_model_path,
        config=load_config,
        policy_config=policy_config_hint,
    )
    evaluator = Evaluator(eval_config, resolved_output_dir)
    try:
        summary, eval_json_path, eval_csv_path = evaluator.evaluate(
            policy=policy,
            seeds=selected_seeds,
            checkpoint_episode=checkpoint_episode,
            deterministic=True,
            capture_replays=True,
            capture_step_telemetry=False,
            evaluation_mode="heldout_50_audit",
            curriculum_stage=stage,
            record_in_registry=False,
            max_workers=effective_workers,
        )

        representative_successes = _select_representative_successes(summary.records)
        rep_success_seeds = {rec.seed for rec in representative_successes}
        rep_success_roles: dict[int, str] = {}
        if representative_successes:
            rep_success_roles[representative_successes[0].seed] = "fastest_success"
            if len(representative_successes) > 1:
                rep_success_roles[representative_successes[1].seed] = "slowest_success"

        heldout_records_payload: list[dict[str, Any]] = []
        heldout_failures: list[dict[str, Any]] = []
        retained_success_replays: list[dict[str, Any]] = []
        retained_failure_replays: list[str] = []

        for rec in summary.records:
            replay_path = Path(rec.replay_path) if rec.replay_path else None
            kept_replay: str | None = None

            if not rec.success:
                if replay_path is not None and replay_path.is_file():
                    bundle = _load_replay_payload(replay_path)
                    kept_replay = str(replay_path)
                    retained_failure_replays.append(kept_replay)
                else:
                    bundle = {
                        "stats": {
                            "stopped": rec.stopped,
                            "timeout": rec.timeout,
                            "stop_reason": rec.stop_reason,
                            "sheep_penned": rec.sheep_penned,
                            "steps": rec.steps,
                            "reward_total": rec.reward_total,
                            "no_progress_steps": rec.no_progress_steps,
                        },
                        "final_snapshot": {
                            "grid_width": eval_config.environment.width,
                            "grid_height": eval_config.environment.height,
                            "sheep": [
                                {
                                    "index": idx,
                                    "x": int(pos[0]),
                                    "y": int(pos[1]),
                                    "penned": False,
                                    "personality": "unknown",
                                }
                                for idx, pos in enumerate(rec.final_sheep_positions or ())
                            ],
                        },
                        "frames": [],
                    }
                fail_diag = _build_failure_diagnostic_from_bundle(
                    bundle=bundle,
                    replay_path=replay_path if kept_replay else None,
                    label=f"heldout_seed_{rec.seed}",
                    seed=rec.seed,
                    scenario_type=None,
                )
                heldout_failures.append(fail_diag)
            else:
                if rec.seed in rep_success_seeds and replay_path is not None and replay_path.is_file():
                    kept_replay = str(replay_path)
                    retained_success_replays.append({
                        "seed": rec.seed,
                        "role": rep_success_roles[rec.seed],
                        "steps": rec.steps,
                        "replay_path": kept_replay,
                    })
                elif replay_path is not None and replay_path.is_file():
                    with contextlib.suppress(OSError):
                        replay_path.unlink()

            heldout_records_payload.append({
                "seed": rec.seed,
                "success": rec.success,
                "timeout": rec.timeout,
                "stopped": rec.stopped,
                "stop_reason": rec.stop_reason,
                "steps": rec.steps,
                "sheep_penned": rec.sheep_penned,
                "reward_total": round(float(rec.reward_total), 4),
                "retained_replay_path": kept_replay,
            })

        heldout_successes = sum(1 for r in summary.records if r.success)
        heldout_total = len(summary.records)
        ho_low, ho_high = wilson_ci_95(heldout_successes, heldout_total)

        adversarial_records: list[dict[str, Any]] = []
        adversarial_failures: list[dict[str, Any]] = []
        by_scenario_type: dict[str, dict[str, Any]] = {}
        active_scenario_types: list[str] = []

        if not skip_scenarios:
            available_types = (
                list(scenario_types)
                if scenario_types is not None
                else [st for st in list_scenario_types() if st != "normal_random"]
            )
            active_scenario_types = [st for st in available_types if st != "normal_random"]
            replay_store = ReplayStore(resolved_output_dir / "replays")
            scenario_env = SheepdogEnvironment(eval_config)

            for stype in active_scenario_types:
                builder = get_scenario_builder(stype)
                stype_total = 0
                stype_successes = 0
                for s_seed in scenario_seeds:
                    scenario = builder(seed=int(s_seed), config=eval_config.environment)
                    result = scenario_env.run_policy_on_scenario(
                        policy,
                        scenario,
                        capture_replay=True,
                        deterministic=True,
                    )
                    stype_total += 1
                    is_success = bool(result.stats.success)
                    if is_success:
                        stype_successes += 1
                        saved_adv_replay: Path | None = None
                    else:
                        bundle = _episode_result_to_bundle(
                            result,
                            policy,
                            eval_config,
                            scenario_type=stype,
                        )
                        saved_adv_replay = replay_store.write(
                            f"adversarial-{stype}-seed-{int(s_seed):06d}.json",
                            bundle,
                            compress=True,
                        )
                        retained_failure_replays.append(str(saved_adv_replay))
                        adversarial_failures.append(
                            _build_failure_diagnostic_from_bundle(
                                bundle=bundle,
                                replay_path=saved_adv_replay,
                                label=f"adversarial_{stype}_seed_{int(s_seed)}",
                                seed=int(s_seed),
                                scenario_type=stype,
                            )
                        )

                    adversarial_records.append({
                        "scenario_type": stype,
                        "seed": int(s_seed),
                        "success": is_success,
                        "timeout": bool(result.stats.timeout),
                        "stopped": bool(result.stats.stopped),
                        "stop_reason": result.stats.stop_reason,
                        "steps": int(result.stats.steps),
                        "sheep_penned": int(result.stats.sheep_penned),
                        "reward_total": round(float(result.stats.reward_total), 4),
                        "retained_replay_path": str(saved_adv_replay) if saved_adv_replay else None,
                    })

                s_low, s_high = wilson_ci_95(stype_successes, stype_total)
                by_scenario_type[stype] = {
                    "successes": stype_successes,
                    "total": stype_total,
                    "success_rate": (stype_successes / stype_total) if stype_total > 0 else None,
                    "wilson_ci_95": {"low": s_low, "high": s_high},
                }

        adv_total = len(adversarial_records)
        adv_successes = sum(1 for r in adversarial_records if r["success"])
        adv_low, adv_high = wilson_ci_95(adv_successes, adv_total)

        all_failures = heldout_failures + adversarial_failures
        failure_summary = _aggregate_failure_summary(all_failures)
        assistance_info = _resolve_assistance_disclosure(eval_config.training.observation_mode)

        report: dict[str, Any] = {
            "report_version": "1.0",
            "generated_at": datetime.now(UTC).isoformat(),
            "stage": int(stage),
            "model": {
                "path": str(resolved_model_path),
                "sha256": model_sha256,
                "checkpoint_episode": checkpoint_episode,
                "hidden_sizes": list(policy.config.hidden_sizes),
                "observation_size": int(policy.config.observation_size),
                "sidecar_verified": sidecar is not None,
            },
            "assistance": assistance_info,
            "observation_history_strides": list(eval_config.training.observation_history_strides),
            "include_loop_flags_in_observation": bool(
                eval_config.training.include_loop_flags_in_observation
            ),
            "config_resolution": {
                "source": cfg_meta["config_source"],
                "zip_metadata": cfg_meta["zip_metadata"],
                "workers": effective_workers,
            },
            "seed_disjointness": {
                "dev_seeds": list(DEV_SEEDS),
                "heldout_seeds": list(selected_seeds),
                "disjoint_from_dev": len(set(selected_seeds) & set(DEV_SEEDS)) == 0,
            },
            "heldout_50": {
                "total": heldout_total,
                "successes": heldout_successes,
                "success_rate": summary.success_rate,
                "wilson_ci_95": {"low": ho_low, "high": ho_high},
                "timeout_rate": summary.timeout_rate,
                "stopped_rate": summary.stopped_rate,
                "average_steps": summary.average_completion_steps,
                "average_sheep_penned": summary.average_sheep_penned,
                "average_reward": summary.average_reward,
                "evaluator_summary_json": str(eval_json_path),
                "evaluator_summary_csv": str(eval_csv_path),
                "records": heldout_records_payload,
            },
            "adversarial_scenarios": {
                "skipped": bool(skip_scenarios),
                "scenario_types": active_scenario_types,
                "seeds": [int(s) for s in scenario_seeds],
                "total": adv_total,
                "successes": adv_successes,
                "success_rate": (adv_successes / adv_total) if adv_total > 0 else None,
                "wilson_ci_95": {"low": adv_low, "high": adv_high},
                "by_scenario_type": by_scenario_type,
                "records": adversarial_records,
            },
            "failures": all_failures,
            "failure_summary": failure_summary,
            "retained_replays": {
                "replays_dir": str(resolved_output_dir / "replays"),
                "failure_replays": retained_failure_replays,
                "representative_successes": retained_success_replays,
            },
        }

        report_json_path = resolved_output_dir / "audit_report.json"
        report_md_path = resolved_output_dir / "audit_report.md"
        atomic_write_json(report_json_path, report)
        atomic_write_text(report_md_path, _render_markdown_report(report))

        return report
    finally:
        with contextlib.suppress(Exception):
            evaluator.close()
        with contextlib.suppress(Exception):
            policy.close()


def _build_arg_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser for the held-out 50-seed audit."""
    default_workers = min(8, max(1, (os.cpu_count() or 2) // 2))
    parser = argparse.ArgumentParser(
        description="Read-only 50-seed held-out and adversarial recovery audit."
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("artifacts/checkpoints/stage38/stage38-best-model.zip"),
        help="Path to the MaskablePPO checkpoint .zip file.",
    )
    parser.add_argument(
        "--stage",
        type=int,
        default=38,
        help="Curriculum stage profile to evaluate against (default: 38).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Isolated output directory (default: artifacts/experiments/heldout_50_audit_<YYYYMMDD_HHMMSS>).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into an existing non-empty --output-dir.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=default_workers,
        help=f"Evaluation worker processes (default: {default_workers}).",
    )
    parser.add_argument(
        "--seed-count",
        type=int,
        default=50,
        help="Number of held-out prime seeds to evaluate (1..50, default: 50).",
    )
    parser.add_argument(
        "--skip-scenarios",
        action="store_true",
        help="Skip the adversarial scenario family suite.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    try:
        report = run_audit(
            model_path=args.model_path,
            stage=args.stage,
            output_dir=args.output_dir,
            overwrite=args.overwrite,
            workers=args.workers,
            seed_count=args.seed_count,
            skip_scenarios=args.skip_scenarios,
        )
    except (FileExistsError, FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    heldout = report["heldout_50"]
    adv = report["adversarial_scenarios"]
    ho_ci = _format_ci_pct((heldout["wilson_ci_95"]["low"], heldout["wilson_ci_95"]["high"]))
    print("\n=== HELD-OUT 50 AUDIT SUMMARY ===")
    print(f"Model: {report['model']['path']} (sha256={report['model']['sha256'][:12]}...)")
    print(
        f"Assistance: {report['assistance']['assistance_mode']} "
        f"(obs_mode={report['assistance']['observation_mode']}, "
        f"strides={report['observation_history_strides']})"
    )
    print(
        f"Held-out: {heldout['successes']}/{heldout['total']} "
        f"({heldout['success_rate'] * 100:.1f}%) Wilson 95% CI {ho_ci}"
    )
    if not adv["skipped"]:
        adv_ci = _format_ci_pct((adv["wilson_ci_95"]["low"], adv["wilson_ci_95"]["high"]))
        print(
            f"Adversarial: {adv['successes']}/{adv['total']} "
            f"({(adv['success_rate'] or 0.0) * 100:.1f}%) Wilson 95% CI {adv_ci}"
        )
    print(f"Failures diagnosed: {report['failure_summary']['total_failures']}")
    print(f"Output directory: {Path(report['retained_replays']['replays_dir']).parent}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

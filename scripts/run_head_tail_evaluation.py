"""Head-tail telemetry evaluation runner for historical Checkpoint 40575.

Executes a bounded, isolated development evaluation:
1. Phase 1: Identity and configuration verification.
2. Phase 2: One-seed capture invariance gate on seed 53 (Capture OFF vs ON).
3. Phase 3: Bounded 10-seed capture (head=200, tail=500) on seeds 11, 23, 37, 41, 53, 59, 61, 67, 71, 73.
4. Outputs all replays, per-episode analysis, and manifest.json to an isolated experiment directory.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import gzip
import hashlib
import json
import logging
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from sheepdog.config import LabConfig
from sheepdog.environment import EpisodeResult, SheepdogEnvironment
from sheepdog.policies.neural import NeuralPolicy
from sheepdog.training.diagnostics.temporal_interaction import (
    analyze_episode_temporal_interactions,
)

logger = logging.getLogger(__name__)

TARGET_CHECKPOINT_LABEL = 40575
TARGET_CHECKPOINT_ID = "chk_run_20260817_063453_91f4_pv_10369_ts_339740239"
TARGET_MODEL_RELPATH = "artifacts/checkpoints/stage34/maskable-ppo-stage34-339740239.zip"
TARGET_CHECKPOINT_JSON = (
    "artifacts/checkpoints/checkpoint-chk_run_20260817_063453_91f4_pv_10369_ts_339740239.json"
)
EXPECTED_SHA256 = "bc6b9883ab5f9343255c15e59cbf0252e8268808731482fec2f5b417d3196a00"
DEV_SEEDS = [11, 23, 37, 41, 53, 59, 61, 67, 71, 73]

# Safety limits for evaluation runs
MAX_EPISODE_BUDGET = 11  # 10 development seeds + 1 control episode
MAX_WORLD_STEPS_BUDGET = 25520  # 11 episodes * max 2,320 steps
MAX_WALLCLOCK_SECONDS = 3600.0  # 60 minutes


def compute_file_sha256(path: Path) -> str:
    """Compute the SHA-256 digest of a file on disk."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def get_git_info(repo_root: Path) -> dict[str, Any]:
    """Capture current git commit hash, porcelain status, and dirty status."""
    try:
        rev = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=repo_root, text=True
        )
        is_dirty = bool(status.strip())
        status_lines = status.splitlines() if status.strip() else []
        untracked = [line for line in status_lines if line.startswith("??")]
        modified = [line for line in status_lines if not line.startswith("??")]
        return {
            "source_revision": rev,
            "is_dirty": is_dirty,
            "status_lines_count": len(status_lines),
            "untracked_count": len(untracked),
            "modified_count": len(modified),
        }
    except Exception as exc:
        return {"source_revision": None, "is_dirty": None, "error": str(exc)}


def get_dependency_versions() -> dict[str, str]:
    """Capture installed runtime dependency versions."""
    import sb3_contrib
    import stable_baselines3

    return {
        "python": sys.version,
        "torch": str(torch.__version__),
        "numpy": str(np.__version__),
        "stable_baselines3": str(stable_baselines3.__version__),
        "sb3_contrib": str(sb3_contrib.__version__),
    }


def load_target_configuration(checkpoint_json_path: Path) -> tuple[LabConfig, dict[str, Any]]:
    """Load and parse environment and reward configuration from checkpoint JSON."""
    with open(checkpoint_json_path, encoding="utf-8") as f:
        chk_data = json.load(f)

    base_dict = LabConfig().to_dict()
    base_dict["environment"] = chk_data["environment_config"]
    base_dict["rewards"] = chk_data["reward_config"]
    config = LabConfig.from_dict(base_dict)
    return config, chk_data


def run_single_episode(
    config: LabConfig,
    policy: NeuralPolicy,
    seed: int,
    *,
    capture_step_telemetry: bool,
    telemetry_mode: str = "prefix",
    telemetry_head_steps: int = 200,
    telemetry_tail_steps: int = 500,
    deterministic: bool = True,
    init_rng_seed: int = 53,
) -> tuple[EpisodeResult, dict[str, Any]]:
    """Run an episode with explicit initial RNG state setup and capture ending RNG states."""
    random.seed(init_rng_seed)
    np.random.seed(init_rng_seed)
    torch.manual_seed(init_rng_seed)

    env = SheepdogEnvironment(
        config,
        capture_step_telemetry=capture_step_telemetry,
        telemetry_mode=telemetry_mode,
        telemetry_head_steps=telemetry_head_steps,
        telemetry_tail_steps=telemetry_tail_steps,
    )

    t0 = time.perf_counter()
    result = env.run_policy(
        policy,
        seed=seed,
        capture_replay=True,
        deterministic=deterministic,
        capture_step_telemetry=capture_step_telemetry,
        telemetry_mode=telemetry_mode,
        telemetry_head_steps=telemetry_head_steps,
        telemetry_tail_steps=telemetry_tail_steps,
    )
    elapsed = time.perf_counter() - t0

    rng_states = {
        "py_random": random.getstate(),
        "np_random": np.random.get_state(),
        "torch_random": torch.get_rng_state(),
        "env_rng": env._rng.getstate(),
        "terminated": env._terminated,
        "timeout": env._timeout,
        "stopped": env._stopped,
        "success": env._success,
        "stop_reason": env._stop_reason,
        "elapsed_seconds": elapsed,
    }
    return result, rng_states


def verify_phase2_invariance(
    res_off: EpisodeResult,
    rng_off: dict[str, Any],
    res_on: EpisodeResult,
    rng_on: dict[str, Any],
    H: int,
    L: int,
) -> dict[str, Any]:
    """Verify bit-for-bit invariance between capture OFF and capture ON."""
    replay_off = res_off.replay or []
    replay_on = res_on.replay or []

    T_off = len(replay_off)
    T_on = len(replay_on)
    if T_off != T_on:
        raise AssertionError(f"Episode length mismatch: OFF has {T_off}, ON has {T_on}")

    T = T_off

    # 1. Action sequences invariance
    for step_idx in range(T):
        f_off = replay_off[step_idx]
        f_on = replay_on[step_idx]
        if f_off.actions != f_on.actions:
            raise AssertionError(
                f"Action mismatch at step {step_idx + 1}: OFF={f_off.actions}, ON={f_on.actions}"
            )

        # 2. Reward breakdown invariance (all 28 components)
        rew_off = dataclasses.asdict(f_off.reward)
        rew_on = dataclasses.asdict(f_on.reward)
        if rew_off != rew_on:
            raise AssertionError(f"Reward breakdown mismatch at step {step_idx + 1}")

        # 3. Snapshot invariance
        snap_off = f_off.snapshot.to_dict()
        snap_on = f_on.snapshot.to_dict()
        if snap_off != snap_on:
            raise AssertionError(f"Snapshot mismatch at step {step_idx + 1}")

    # 4. Termination flags invariance
    for key in ["terminated", "timeout", "stopped", "success", "stop_reason"]:
        if rng_off[key] != rng_on[key]:
            raise AssertionError(
                f"Termination flag mismatch for {key}: OFF={rng_off[key]}, ON={rng_on[key]}"
            )

    # 5. Exact RNG state invariance
    if rng_off["py_random"] != rng_on["py_random"]:
        raise AssertionError("Python random state diverged!")
    np_off = rng_off["np_random"]
    np_on = rng_on["np_random"]
    if not (
        isinstance(np_off, tuple)
        and isinstance(np_on, tuple)
        and len(np_off) == 5
        and len(np_on) == 5
    ):
        raise AssertionError("NumPy random state is not a 5-element tuple!")
    if (
        np_off[0] != np_on[0]
        or not np.array_equal(np_off[1], np_on[1])
        or np_off[2] != np_on[2]
        or np_off[3] != np_on[3]
        or np_off[4] != np_on[4]
    ):
        raise AssertionError("NumPy random state diverged across 5-tuple state elements!")
    if not torch.equal(rng_off["torch_random"], rng_on["torch_random"]):
        raise AssertionError("PyTorch RNG state diverged!")
    if rng_off["env_rng"] != rng_on["env_rng"]:
        raise AssertionError("Environment _rng state diverged!")

    # 6. Verify retained telemetry structure on run ON
    head_frames_with_telemetry = 0
    tail_frames_with_telemetry = 0
    omitted_frames = 0
    gap_marker_found = False
    gap_marker_details: dict[str, Any] = {}

    expected_head_end = min(H, T)
    expected_tail_start = max(1, T - L + 1)

    for step_idx in range(T):
        f = replay_on[step_idx]
        step_num = step_idx + 1
        st = f.step_telemetry
        if st is not None:
            if isinstance(st, dict) and st.get("telemetry_omitted"):
                gap_marker_found = True
                gap_marker_details = st
                omitted_frames += 1
            else:
                if step_num <= expected_head_end:
                    head_frames_with_telemetry += 1
                if step_num >= expected_tail_start:
                    tail_frames_with_telemetry += 1
        else:
            omitted_frames += 1

    if T > H + L:
        if not gap_marker_found:
            raise AssertionError(f"Expected gap marker at frame {H + 1} for T={T} > H+L={H+L}")
        omitted_marker_step = H + 1
        marker = replay_on[omitted_marker_step - 1].step_telemetry
        assert isinstance(marker, dict) and marker.get("telemetry_omitted") is True
        assert marker.get("omitted_range") == f"steps_{H + 1}_to_{expected_tail_start - 1}"
        assert marker.get("omitted_steps_count") == expected_tail_start - 1 - H

    return {
        "status": "passed",
        "episode_length": T,
        "success": rng_on["success"],
        "timeout": rng_on["timeout"],
        "stopped": rng_on["stopped"],
        "stop_reason": rng_on["stop_reason"],
        "head_frames_retained": head_frames_with_telemetry,
        "tail_frames_retained": tail_frames_with_telemetry,
        "gap_marker_found": gap_marker_found,
        "gap_marker_details": gap_marker_details,
        "invariance_verified": True,
    }


def serialize_replay_to_disk(
    result: EpisodeResult,
    config: LabConfig,
    out_path: Path,
    *,
    seed: int,
    checkpoint_id: str,
    run_id: str,
    policy_name: str = "neural_policy",
    trainer_type: str = "maskable_ppo",
    policy_type: str = "neural",
    telemetry_mode: str = "head_tail",
    telemetry_head_steps: int = 200,
    telemetry_tail_steps: int = 500,
    capture_step_telemetry: bool = True,
    checkpoint_gamma: float = 0.997,
    discount_timebase: str = "sequential_shared_dog",
) -> int:
    """Serialize episode replay bundle to compressed JSON (.json.gz)."""
    replay_frames = [f.to_dict() for f in (result.replay or [])]
    initial_snapshot_dict = (
        result.initial_snapshot.to_dict() if result.initial_snapshot is not None else None
    )
    final_snapshot_dict = (
        result.final_snapshot.to_dict() if result.final_snapshot is not None else {}
    )
    if result.stats is not None:
        stats_dict = (
            dataclasses.asdict(result.stats)
            if dataclasses.is_dataclass(result.stats)
            else (result.stats if isinstance(result.stats, dict) else vars(result.stats))
        )
    else:
        stats_dict = {}

    T_steps = len(replay_frames)
    H_steps = telemetry_head_steps
    L_steps = telemetry_tail_steps

    if telemetry_mode == "head_tail":
        tail_start = max(1, T_steps - L_steps + 1)
        has_gap = T_steps > H_steps + L_steps
        count_detailed = (
            min(H_steps, T_steps) + max(0, min(L_steps, T_steps - H_steps))
            if capture_step_telemetry
            else 0
        )
        count_omitted = (T_steps - count_detailed) if capture_step_telemetry else 0
        telemetry_coverage = {
            "mode": telemetry_mode,
            "total_steps": T_steps,
            "captured_detailed_steps": count_detailed,
            "omitted_steps_count": count_omitted,
            "retained_head_range": (
                [1, min(H_steps, T_steps)] if (H_steps > 0 and capture_step_telemetry) else None
            ),
            "retained_tail_range": (
                [tail_start, T_steps]
                if (L_steps > 0 and tail_start <= T_steps and capture_step_telemetry)
                else None
            ),
            "omitted_range": [H_steps + 1, tail_start - 1] if has_gap else None,
            "telemetry_truncated": has_gap,
            "complete_decision_coverage": (tail_start <= H_steps + 1) and capture_step_telemetry,
        }
    else:
        telemetry_coverage = {
            "mode": telemetry_mode,
            "total_steps": T_steps,
            "captured_detailed_steps": 0,
            "omitted_steps_count": T_steps,
            "retained_head_range": None,
            "retained_tail_range": None,
            "omitted_range": None,
            "telemetry_truncated": False,
            "complete_decision_coverage": False,
        }

    evaluation_gamma = float(config.training.gamma)
    chk_gamma = float(checkpoint_gamma)
    gamma_mismatch = bool(evaluation_gamma != chk_gamma)

    bundle: dict[str, Any] = {
        "checkpoint_id": checkpoint_id,
        "run_id": run_id,
        "seed": seed,
        "policy_name": policy_name,
        "trainer_type": trainer_type,
        "policy_type": policy_type,
        "policy_mode": policy_name,
        "replay_mode": "neural_ppo",
        "arena": {
            "width": config.environment.width,
            "height": config.environment.height,
        },
        "environment": {
            **dataclasses.asdict(config.environment),
            "curriculum_stage": config.rewards.instincts.curriculum_stage,
            "enable_instinct_rewards": config.rewards.instincts.enable_instinct_rewards,
        },
        "strategy_version": "1.0.0",
        "discount_timebase": discount_timebase,
        "checkpoint_gamma": chk_gamma,
        "evaluation_gamma": evaluation_gamma,
        "gamma_mismatch": gamma_mismatch,
        "telemetry_mode": telemetry_mode,
        "telemetry_head_steps": telemetry_head_steps if telemetry_mode == "head_tail" else None,
        "telemetry_tail_steps": telemetry_tail_steps if telemetry_mode == "head_tail" else None,
        "telemetry_truncated": telemetry_coverage.get("telemetry_truncated", False),
        "telemetry_coverage": telemetry_coverage,
        "config": config.to_dict(),
        "stats": stats_dict,
        "initial_snapshot": initial_snapshot_dict,
        "final_snapshot": final_snapshot_dict,
        "frames": replay_frames,
    }

    raw_bytes = json.dumps(bundle, separators=(",", ":")).encode("utf-8")
    with gzip.open(out_path, "wb", compresslevel=6) as gz:
        gz.write(raw_bytes)

    return out_path.stat().st_size


def run_head_tail_evaluation(
    out_dir: Path,
    repo_root: Path,
    *,
    checkpoint_json: Path | None = None,
    model_path: Path | None = None,
    head_steps: int = 200,
    tail_steps: int = 500,
    allow_existing_dir: bool = False,
    reuse_existing_replays: bool = False,
    max_episodes: int = MAX_EPISODE_BUDGET,
    max_world_steps: int = MAX_WORLD_STEPS_BUDGET,
    max_wallclock_seconds: float = MAX_WALLCLOCK_SECONDS,
) -> dict[str, Any]:
    """Execute complete head/tail evaluation on checkpoint 40575 across 10 dev seeds."""
    if out_dir.exists() and any(out_dir.iterdir()) and not (
        allow_existing_dir or reuse_existing_replays
    ):
        raise FileExistsError(
            f"Output directory '{out_dir}' exists and is not empty. "
            "Pass --allow-existing-dir or --reuse-replays to proceed."
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    replays_dir = out_dir / "replays"
    replays_dir.mkdir(parents=True, exist_ok=True)

    git_info = get_git_info(repo_root)
    deps_info = get_dependency_versions()

    chk_json_path = (
        checkpoint_json
        if checkpoint_json is not None
        else repo_root / TARGET_CHECKPOINT_JSON
    )
    weights_path = (
        model_path
        if model_path is not None
        else repo_root / TARGET_MODEL_RELPATH
    )

    # Phase 1: Verify weights SHA256
    if not weights_path.exists():
        raise FileNotFoundError(f"Model weights not found at {weights_path}")
    actual_sha256 = compute_file_sha256(weights_path)
    if actual_sha256 != EXPECTED_SHA256:
        raise ValueError(
            f"SHA256 mismatch for {weights_path}!\nExpected: {EXPECTED_SHA256}\nActual:   {actual_sha256}"
        )

    config, chk_meta = load_target_configuration(chk_json_path)

    # Load neural policy
    policy_config = chk_meta.get("policy_config")
    policy = NeuralPolicy.load(weights_path, config, policy_config=policy_config)

    print("=================================================================")
    print("PHASE 1: Verified Checkpoint 40575 Identity and Configuration")
    print(f"  Weights SHA256: {actual_sha256}")
    print(f"  Curriculum Stage: {config.rewards.instincts.curriculum_stage}")
    print(f"  Arena: {config.environment.width}x{config.environment.height}")
    print(f"  Dogs: {config.environment.dogs}, Sheep: {config.environment.sheep}")
    print(f"  Max Steps: {config.environment.max_steps}")
    print(f"  Checkpoint Gamma: {policy.checkpoint_gamma}")
    print(f"  Evaluation Gamma: {config.training.gamma}")
    print("  Discount Timebase: sequential_shared_dog")
    print("=================================================================")

    t_gate_start = time.perf_counter()

    def check_limits(episodes_count: int, world_steps: int) -> None:
        if episodes_count > max_episodes:
            raise ValueError(
                f"Episode budget exceeded: {episodes_count} > {max_episodes}"
            )
        if world_steps > max_world_steps:
            raise ValueError(
                f"World step budget exceeded: {world_steps} > {max_world_steps}"
            )
        elapsed = time.perf_counter() - t_gate_start
        if elapsed > max_wallclock_seconds:
            raise TimeoutError(
                f"Wall-clock time limit exceeded: {elapsed:.2f}s > {max_wallclock_seconds:.2f}s"
            )

    # Phase 2: Invariance Gate on Seed 53
    print("\nPHASE 2: Running One-Seed Capture Invariance Gate on Seed 53...")

    # Run A: Control (Capture OFF)
    print("  Running Control: Seed 53 with capture_step_telemetry=False...")
    res_off, rng_off = run_single_episode(
        config,
        policy,
        seed=53,
        capture_step_telemetry=False,
        telemetry_mode="prefix",
        telemetry_head_steps=head_steps,
        telemetry_tail_steps=tail_steps,
        deterministic=True,
        init_rng_seed=53,
    )
    control_size = serialize_replay_to_disk(
        res_off,
        config,
        replays_dir / "checkpoint-040575-seed-000053_control_off.json.gz",
        seed=53,
        checkpoint_id=TARGET_CHECKPOINT_ID,
        run_id="run_20260817_063453_91f4",
        capture_step_telemetry=False,
        checkpoint_gamma=policy.checkpoint_gamma,
        discount_timebase="sequential_shared_dog",
    )

    # Run B: Treatment (Capture ON: head=200, tail=500)
    print(f"  Running Treatment: Seed 53 with Head={head_steps}, Tail={tail_steps}...")
    res_on, rng_on = run_single_episode(
        config,
        policy,
        seed=53,
        capture_step_telemetry=True,
        telemetry_mode="head_tail",
        telemetry_head_steps=head_steps,
        telemetry_tail_steps=tail_steps,
        deterministic=True,
        init_rng_seed=53,
    )
    treatment_seed53_path = replays_dir / "checkpoint-040575-seed-000053.json.gz"
    treatment_size = serialize_replay_to_disk(
        res_on,
        config,
        treatment_seed53_path,
        seed=53,
        checkpoint_id=TARGET_CHECKPOINT_ID,
        run_id="run_20260817_063453_91f4",
        telemetry_mode="head_tail",
        telemetry_head_steps=head_steps,
        telemetry_tail_steps=tail_steps,
        capture_step_telemetry=True,
        checkpoint_gamma=policy.checkpoint_gamma,
        discount_timebase="sequential_shared_dog",
    )

    invariance_proof = verify_phase2_invariance(
        res_off, rng_off, res_on, rng_on, H=head_steps, L=tail_steps
    )
    t_gate_elapsed = time.perf_counter() - t_gate_start
    print(f"  Invariance Gate PASSED in {t_gate_elapsed:.2f}s!")
    print(f"    Episode Length: {invariance_proof['episode_length']} steps")
    print(f"    Outcome: {invariance_proof['stop_reason']}")
    print(f"    Control Replay Size: {control_size} bytes")
    print(f"    Treatment Replay Size: {treatment_size} bytes")

    # Phase 3: Bounded Ten-Seed Capture
    print("\nPHASE 3: Running Bounded Development Seeds Evaluation...")
    episodes_data: list[dict[str, Any]] = []

    # Store seed 53 treatment into episodes_data
    episodes_data.append({
        "seed": 53,
        "is_control": False,
        "reused": False,
        "steps": len(res_on.replay or []),
        "success": bool(res_on.stats.success),
        "timeout": bool(res_on.stats.timeout),
        "stopped": bool(res_on.stats.stopped),
        "stop_reason": res_on.stats.stop_reason or (
            "success"
            if res_on.stats.success
            else ("timeout" if res_on.stats.timeout else "unknown")
        ),
        "sheep_penned": int(res_on.stats.sheep_penned),
        "reward_total": float(res_on.stats.reward_total),
        "replay_path": str(treatment_seed53_path.relative_to(out_dir)),
        "replay_size_bytes": treatment_size,
        "elapsed_seconds": rng_on["elapsed_seconds"],
        "elapsed_seconds_status": "measured",
    })

    remaining_seeds = [s for s in DEV_SEEDS if s != 53]
    total_world_steps = len(res_on.replay or []) + len(res_off.replay or [])
    check_limits(episodes_count=2, world_steps=total_world_steps)

    for idx, seed in enumerate(remaining_seeds, start=1):
        replay_path = replays_dir / f"checkpoint-040575-seed-{seed:06d}.json.gz"
        if reuse_existing_replays and replay_path.exists():
            print(f"  Reusing existing replay for Seed {seed} ({idx}/{len(remaining_seeds)})...")
            with gzip.open(replay_path, "rt", encoding="utf-8") as rf:
                b = json.load(rf)
                st = b.get("stats", {})
                steps = len(b.get("frames", []))
                total_world_steps += steps
                stop_r = st.get("stop_reason") or (
                    "success"
                    if st.get("success")
                    else ("timeout" if st.get("timeout") else "unknown")
                )
                episodes_data.append({
                    "seed": seed,
                    "is_control": False,
                    "reused": True,
                    "steps": steps,
                    "success": bool(st.get("success")),
                    "timeout": bool(st.get("timeout")),
                    "stopped": bool(st.get("stopped")),
                    "stop_reason": stop_r,
                    "sheep_penned": int(st.get("sheep_penned", 0)),
                    "reward_total": float(st.get("reward_total", 0.0)),
                    "replay_path": str(replay_path.relative_to(out_dir)),
                    "replay_size_bytes": replay_path.stat().st_size,
                    "elapsed_seconds": None,
                    "elapsed_seconds_status": "unknown",
                })
            check_limits(episodes_count=len(episodes_data) + 1, world_steps=total_world_steps)
            continue

        print(
            f"  Evaluating Seed {seed} ({idx}/{len(remaining_seeds)}) with Head/Tail Telemetry..."
        )
        t_seed_start = time.perf_counter()
        res, rng = run_single_episode(
            config,
            policy,
            seed=seed,
            capture_step_telemetry=True,
            telemetry_mode="head_tail",
            telemetry_head_steps=head_steps,
            telemetry_tail_steps=tail_steps,
            deterministic=True,
            init_rng_seed=seed,
        )
        elapsed_seed = time.perf_counter() - t_seed_start
        steps = len(res.replay or [])
        total_world_steps += steps

        rep_size = serialize_replay_to_disk(
            res,
            config,
            replay_path,
            seed=seed,
            checkpoint_id=TARGET_CHECKPOINT_ID,
            run_id="run_20260817_063453_91f4",
            telemetry_mode="head_tail",
            telemetry_head_steps=head_steps,
            telemetry_tail_steps=tail_steps,
            capture_step_telemetry=True,
            checkpoint_gamma=policy.checkpoint_gamma,
            discount_timebase="sequential_shared_dog",
        )

        stop_r = res.stats.stop_reason or (
            "success"
            if res.stats.success
            else ("timeout" if res.stats.timeout else "unknown")
        )
        print(
            f"    Done Seed {seed}: steps={steps}, penned={res.stats.sheep_penned}/6, "
            f"outcome={stop_r}, size={rep_size}B in {elapsed_seed:.2f}s"
        )

        episodes_data.append({
            "seed": seed,
            "is_control": False,
            "reused": False,
            "steps": steps,
            "success": bool(res.stats.success),
            "timeout": bool(res.stats.timeout),
            "stopped": bool(res.stats.stopped),
            "stop_reason": stop_r,
            "sheep_penned": int(res.stats.sheep_penned),
            "reward_total": float(res.stats.reward_total),
            "replay_path": str(replay_path.relative_to(out_dir)),
            "replay_size_bytes": rep_size,
            "elapsed_seconds": elapsed_seed,
            "elapsed_seconds_status": "measured",
        })
        check_limits(episodes_count=len(episodes_data) + 1, world_steps=total_world_steps)

    # Sort episodes_data by seed
    episodes_data.sort(key=lambda x: x["seed"])

    # Phase 4: Analyze Evidence using offline temporal interaction analyzer
    print("\nPHASE 4: Analyzing Temporal Interaction Evidence across Replays...")
    analysis_summaries: list[dict[str, Any]] = []
    for ep in episodes_data:
        rep_file = out_dir / ep["replay_path"]
        analysis = analyze_episode_temporal_interactions(rep_file)

        # Extract late tail telemetry decision coverage
        late_decisions_retained = 0
        head_decisions_retained = 0
        tail_start_step = max(1, ep["steps"] - tail_steps + 1)

        # Inspect frames in replay
        with gzip.open(rep_file, "rt", encoding="utf-8") as f_gz:
            bundle = json.load(f_gz)
            for f in bundle.get("frames", []):
                st = f.get("step_telemetry")
                if st and not (isinstance(st, dict) and st.get("telemetry_omitted")):
                    step_num = f.get("step")
                    if step_num is not None:
                        if step_num <= head_steps:
                            head_decisions_retained += 1
                        if step_num >= tail_start_step:
                            late_decisions_retained += 1

        audit_dict = (
            dataclasses.asdict(analysis.evidence_audit)
            if dataclasses.is_dataclass(analysis.evidence_audit)
            else (
                analysis.evidence_audit
                if isinstance(analysis.evidence_audit, dict)
                else {}
            )
        )
        ep_analysis = {
            "seed": ep["seed"],
            "steps": ep["steps"],
            "sheep_penned": ep["sheep_penned"],
            "outcome": ep["stop_reason"],
            "interval_count": len(analysis.intervals),
            "head_decisions_retained": head_decisions_retained,
            "late_decisions_retained": late_decisions_retained,
            "has_recorded_step_telemetry": analysis.evidence_audit.has_recorded_step_telemetry,
            "evidence_audit": audit_dict,
            "intervals": [it.to_dict() for it in analysis.intervals],
        }
        analysis_summaries.append(ep_analysis)

    manifest: dict[str, Any] = {
        "manifest_version": "1.0.0",
        "created_timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
        "experiment_name": "head_tail_diagnostic_chk40575",
        "git_provenance": git_info,
        "dependency_versions": deps_info,
        "target_checkpoint": {
            "label": TARGET_CHECKPOINT_LABEL,
            "checkpoint_id": TARGET_CHECKPOINT_ID,
            "weights_path": str(weights_path),
            "weights_sha256": actual_sha256,
            "policy_mode": "neural_policy",
            "observation_mode": "guided",
            "discount_timebase": "sequential_shared_dog",
            "checkpoint_gamma": policy.checkpoint_gamma,
            "evaluation_gamma": float(config.training.gamma),
            "gamma_mismatch": bool(config.training.gamma != policy.checkpoint_gamma),
            "hidden_sizes": list(policy.config.hidden_sizes),
            "observation_size": policy.config.observation_size,
            "action_size": policy.config.action_size,
            "arena": {
                "width": config.environment.width,
                "height": config.environment.height,
            },
            "dogs": config.environment.dogs,
            "sheep": config.environment.sheep,
            "max_steps": config.environment.max_steps,
            "curriculum_stage": config.rewards.instincts.curriculum_stage,
            "sqlite_registry_status": "unregistered_historical_checkpoint",
        },
        "phase2_invariance_gate": {
            "seed": 53,
            "invariance_proof": invariance_proof,
            "control_replay_path": "replays/checkpoint-040575-seed-000053_control_off.json.gz",
            "control_replay_size_bytes": control_size,
            "treatment_replay_path": "replays/checkpoint-040575-seed-000053.json.gz",
            "treatment_replay_size_bytes": treatment_size,
        },
        "phase3_evaluation": {
            "seeds": DEV_SEEDS,
            "telemetry_mode": "head_tail",
            "telemetry_head_steps": head_steps,
            "telemetry_tail_steps": tail_steps,
            "total_episodes_evaluated": len(episodes_data) + 1,  # 10 dev seeds + 1 control
            "total_world_steps": total_world_steps,
            "elapsed_seconds_total": time.perf_counter() - t_gate_start,
            "episodes": episodes_data,
        },
        "phase4_analysis": analysis_summaries,
    }

    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nEvaluation Complete! Manifest and replays saved to: {out_dir}")
    return manifest


def main() -> None:
    """CLI entry point for running head/tail evaluation on checkpoint 40575."""
    parser = argparse.ArgumentParser(
        description="Run isolated head/tail evaluation on checkpoint 40575"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Target output directory (defaults to artifacts/experiments/head_tail_diagnostic_chk40575_<timestamp>)",
    )
    parser.add_argument(
        "--checkpoint-json",
        type=str,
        default=None,
        help=f"Path to checkpoint JSON metadata (defaults to {TARGET_CHECKPOINT_JSON})",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help=f"Path to model weights zip file (defaults to {TARGET_MODEL_RELPATH})",
    )
    parser.add_argument("--head-steps", type=int, default=200, help="Head steps to retain")
    parser.add_argument("--tail-steps", type=int, default=500, help="Tail steps to retain")
    parser.add_argument(
        "--allow-existing-dir",
        action="store_true",
        help="Allow writing into an existing non-empty output directory",
    )
    parser.add_argument(
        "--reuse-replays",
        action="store_true",
        help="Reuse already written replays in output directory if present",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = (
            repo_root
            / "artifacts"
            / "experiments"
            / f"head_tail_diagnostic_chk40575_{timestamp_str}"
        )

    chk_json = Path(args.checkpoint_json) if args.checkpoint_json else None
    model_p = Path(args.model_path) if args.model_path else None

    run_head_tail_evaluation(
        out_dir,
        repo_root,
        checkpoint_json=chk_json,
        model_path=model_p,
        head_steps=args.head_steps,
        tail_steps=args.tail_steps,
        allow_existing_dir=args.allow_existing_dir,
        reuse_existing_replays=args.reuse_replays,
    )


if __name__ == "__main__":
    main()

"""Offline step-level replay analysis for team_strategy_split_flock_20261117_070100.

Reads saved evaluation result.json and .json.gz replay bundles (read-only),
runs temporal_interaction and repetition_recovery diagnostics, and reconstructs
TeamStrategy role targets across variants to explain:
1. Seed 73 persistent failures across all variants on pv_10486 and pv_10483.
2. pv_10483 main_group regressions on seeds 41, 53, and 67.
3. pv_10486 corner_clearance wins on seeds 59 and 71.
"""

from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sheepdog.config import EnvironmentConfig
from sheepdog.entities import DogRole, DogState, Pen, Point, SheepState
from sheepdog.team_strategy import TeamStrategy
from sheepdog.training.diagnostics.repetition_recovery import analyze_repetition_and_recovery
from sheepdog.training.diagnostics.temporal_interaction import (
    analyze_episode_temporal_interactions,
)

EXPERIMENT_DIR = Path("artifacts/experiments/team_strategy_split_flock_20261117_070100")

VARIANT_FLAGS: dict[str, dict[str, bool]] = {
    "baseline": {
        "main_group_targets": False,
        "collector_yield": False,
        "corner_clearance": False,
    },
    "main_group": {
        "main_group_targets": True,
        "collector_yield": False,
        "corner_clearance": False,
    },
    "collector_yield": {
        "main_group_targets": False,
        "collector_yield": True,
        "corner_clearance": False,
    },
    "corner_clearance": {
        "main_group_targets": False,
        "collector_yield": False,
        "corner_clearance": True,
    },
    "all": {
        "main_group_targets": True,
        "collector_yield": True,
        "corner_clearance": True,
    },
}


@dataclass
class StepTrace:
    """Per-step recorded state paired with reconstructed TeamStrategy targets."""

    step: int
    penned_count: int
    unpenned_indices: list[int]
    unpenned_positions: dict[int, tuple[int, int]]
    dog_positions: list[tuple[int, int]]
    dog_roles: list[str]
    actions: list[str]
    no_progress_steps: int
    stray_idx: int | None
    targets: dict[int, tuple[int, int]]
    target_roles: dict[int, str]
    alt_targets: dict[int, tuple[int, int]] | None = None
    alt_stray_idx: int | None = None


def load_bundle(path: Path) -> dict[str, Any]:
    """Load a gzipped JSON replay bundle."""
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def replay_path_for(model: str, variant: str, seed: int) -> Path:
    """Return the replay path for a model, variant, and seed."""
    return (
        EXPERIMENT_DIR
        / f"{model}__{variant}"
        / "replays"
        / f"checkpoint-000000-seed-{seed:06d}.json.gz"
    )


def result_path_for(model: str, variant: str) -> Path:
    """Return the result.json path for a model and variant."""
    return EXPERIMENT_DIR / f"{model}__{variant}" / "result.json"


def _build_env_config(env_dict: dict[str, Any]) -> EnvironmentConfig:
    valid_fields = set(EnvironmentConfig.__dataclass_fields__.keys())
    filtered = {k: v for k, v in env_dict.items() if k in valid_fields}
    if "pen_size" in filtered and isinstance(filtered["pen_size"], list):
        filtered["pen_size"] = tuple(filtered["pen_size"])
    return EnvironmentConfig(**filtered)


def _pen_from_snapshot(snap: dict[str, Any]) -> Pen:
    pen_dict = snap["pen"]
    origin = pen_dict["origin"]
    return Pen(
        origin=Point(int(origin["x"]), int(origin["y"])),
        width=int(pen_dict["width"]),
        height=int(pen_dict["height"]),
        opening=str(pen_dict["opening"]),
    )


def _snapshot_to_entities(
    snap: dict[str, Any],
    prev_roles: list[DogRole | None],
    steps_in_role: list[int],
) -> tuple[list[DogState], list[SheepState]]:
    dogs: list[DogState] = []
    for d in snap["dogs"]:
        idx = int(d["index"])
        role = prev_roles[idx]
        dogs.append(
            DogState(
                index=idx,
                position=Point(int(d["x"]), int(d["y"])),
                current_role=role,
                steps_in_role=steps_in_role[idx],
            )
        )
    sheep: list[SheepState] = []
    for s in snap["sheep"]:
        sheep.append(
            SheepState(
                index=int(s["index"]),
                position=Point(int(s["x"]), int(s["y"])),
                penned=bool(s["penned"]),
                personality=str(s.get("personality") or "calm"),
            )
        )
    return dogs, sheep


def reconstruct_trace(
    bundle: dict[str, Any],
    variant: str,
    compare_variant: str | None = None,
) -> list[StepTrace]:
    """Reconstruct per-step TeamStrategy targets alongside recorded frames."""
    cfg = _build_env_config(bundle["environment"])
    flags = VARIANT_FLAGS[variant]
    strategy = TeamStrategy(cfg.width, cfg.height, cfg, **flags)
    strategy.reset()

    alt_strategy: TeamStrategy | None = None
    if compare_variant is not None:
        alt_strategy = TeamStrategy(
            cfg.width, cfg.height, cfg, **VARIANT_FLAGS[compare_variant]
        )
        alt_strategy.reset()

    frames = bundle["frames"]
    pre_snap = bundle["initial_snapshot"]
    dog_count = len(pre_snap["dogs"])
    prev_roles: list[DogRole | None] = [None] * dog_count
    steps_in_role = [0] * dog_count
    alt_prev_roles: list[DogRole | None] = [None] * dog_count
    alt_steps_in_role = [0] * dog_count

    traces: list[StepTrace] = []
    for frame in frames:
        step_num = int(frame["step"])
        post_snap = frame["snapshot"]
        pen = _pen_from_snapshot(pre_snap)

        dogs, sheep = _snapshot_to_entities(pre_snap, prev_roles, steps_in_role)
        assignments, snap_strat = strategy.assign_roles(dogs, sheep, pen)

        targets = {idx: (a.target.x, a.target.y) for idx, a in assignments.items()}
        target_roles = {idx: a.role.value for idx, a in assignments.items()}

        alt_targets: dict[int, tuple[int, int]] | None = None
        alt_stray_idx: int | None = None
        if alt_strategy is not None:
            alt_dogs, alt_sheep = _snapshot_to_entities(pre_snap, alt_prev_roles, alt_steps_in_role)
            alt_assign, alt_snap_strat = alt_strategy.assign_roles(alt_dogs, alt_sheep, pen)
            alt_targets = {idx: (a.target.x, a.target.y) for idx, a in alt_assign.items()}
            alt_stray_idx = alt_snap_strat.stray_sheep_index
            for idx, a in alt_assign.items():
                if alt_prev_roles[idx] == a.role:
                    alt_steps_in_role[idx] += 1
                else:
                    alt_prev_roles[idx] = a.role
                    alt_steps_in_role[idx] = 1

        # Update role hold tracking from actual recorded post-step roles.
        recorded_roles = [str(d["role"]) for d in post_snap["dogs"]]
        for idx, role_str in enumerate(recorded_roles):
            role_enum = DogRole(role_str)
            if prev_roles[idx] == role_enum:
                steps_in_role[idx] += 1
            else:
                prev_roles[idx] = role_enum
                steps_in_role[idx] = 1

        unpenned_pos = {
            int(s["index"]): (int(s["x"]), int(s["y"]))
            for s in post_snap["sheep"]
            if not s["penned"]
        }
        traces.append(
            StepTrace(
                step=step_num,
                penned_count=int(post_snap["penned_count"]),
                unpenned_indices=sorted(unpenned_pos.keys()),
                unpenned_positions=unpenned_pos,
                dog_positions=[(int(d["x"]), int(d["y"])) for d in post_snap["dogs"]],
                dog_roles=recorded_roles,
                actions=[str(a) for a in frame["actions"]],
                no_progress_steps=int(post_snap["no_progress_steps"]),
                stray_idx=snap_strat.stray_sheep_index,
                targets=targets,
                target_roles=target_roles,
                alt_targets=alt_targets,
                alt_stray_idx=alt_stray_idx,
            )
        )
        pre_snap = post_snap
    return traces


def summarize_diagnostics(path: Path) -> dict[str, Any]:
    """Run temporal_interaction and repetition_recovery diagnostics on a replay."""
    ti = analyze_episode_temporal_interactions(path)
    rr = analyze_repetition_and_recovery(path)
    return {
        "steps": ti.total_steps,
        "penned": ti.final_penned_count,
        "success": ti.outcome_status == "success",
        "stop_reason": ti.stop_reason,
        "ti_intervals": [
            {
                "phase": iv.unpenned_sheep_count,
                "start": iv.start_step,
                "end": iv.end_step,
                "len": iv.duration_steps,
                "kind": iv.classification,
                "dogs": iv.affected_dog_indices,
                "sheep": iv.affected_sheep_indices,
                "mechanisms": iv.candidate_mechanisms,
            }
            for iv in ti.intervals
        ],
        "rr_summary": {
            "repeated_steps": rr.repeated_movement_steps,
            "repeated_frac": round(rr.repeated_movement_fraction, 3),
            "stalls_count": rr.stalls_count,
            "recovered": rr.recovered_stalls_count,
            "relapsed": rr.relapsed_stalls_count,
            "unresolved": rr.unresolved_stalls_count,
        },
    }


def print_result_matrix() -> None:
    """Print the 2-model x 5-variant x 10-seed outcome matrix."""
    print("=== 1. RESULT MATRIX (ALL 10 RUNS) ===")
    for model in ("pv_10486", "pv_10483"):
        for variant in ("baseline", "main_group", "collector_yield", "corner_clearance", "all"):
            r = json.loads(result_path_for(model, variant).read_text(encoding="utf-8"))
            recs = {int(rec["seed"]): rec for rec in r["records"]}
            cells = []
            for seed in (11, 23, 37, 41, 53, 59, 61, 67, 71, 73):
                rec = recs[seed]
                tag = "S" if rec["success"] else f"F({rec['stop_reason'][:4]},{rec['sheep_penned']}/6)"
                cells.append(f"{seed}:{tag}@{rec['steps']}")
            print(f"{model:8s} {variant:16s} | {int(round(r['success_rate']*10)):2d}/10 | " + " ".join(cells))


def analyze_seed_73() -> None:
    """Analyze seed 73 across all models and variants."""
    print("\n=== 2. SEED 73 ACROSS ALL VARIANTS ===")
    for model in ("pv_10486", "pv_10483"):
        for variant in ("baseline", "main_group", "collector_yield", "corner_clearance", "all"):
            p = replay_path_for(model, variant, 73)
            bundle = load_bundle(p)
            diag = summarize_diagnostics(p)
            traces = reconstruct_trace(bundle, variant)

            # Find penning steps
            pen_events: list[str] = []
            prev_unpenned = set(range(6))
            for tr in traces:
                cur_unpenned = set(tr.unpenned_indices)
                newly_penned = sorted(prev_unpenned - cur_unpenned)
                if newly_penned:
                    pen_events.append(f"s{newly_penned}@step{tr.step}")
                prev_unpenned = cur_unpenned

            # Summarize last 500 steps: unpenned positions, stray counts, role counts, dog action loops
            tail = traces[-500:]
            stray_counts = Counter(tr.stray_idx for tr in tail)
            role_counts = {
                d: Counter(tr.dog_roles[d] for tr in tail).most_common(3) for d in range(3)
            }
            dog_pos_bounds = {
                d: (
                    min(tr.dog_positions[d][0] for tr in tail),
                    max(tr.dog_positions[d][0] for tr in tail),
                    min(tr.dog_positions[d][1] for tr in tail),
                    max(tr.dog_positions[d][1] for tr in tail),
                )
                for d in range(3)
            }
            sheep_pos_bounds = {}
            for s_idx in traces[-1].unpenned_indices:
                xs = [tr.unpenned_positions[s_idx][0] for tr in tail if s_idx in tr.unpenned_positions]
                ys = [tr.unpenned_positions[s_idx][1] for tr in tail if s_idx in tr.unpenned_positions]
                sheep_pos_bounds[s_idx] = (min(xs), max(xs), min(ys), max(ys))

            pen_dict = bundle["frames"][0]["snapshot"]["pen"]
            print(
                f"\n[{model} | {variant} | seed 73] "
                f"pen=({pen_dict['origin']['x']},{pen_dict['origin']['y']},"
                f"{pen_dict['width']}x{pen_dict['height']},opening={pen_dict['opening']})"
            )
            print(f"  outcome: {diag['stop_reason']} steps={diag['steps']} penned={diag['penned']}/6 | penned_events={pen_events}")
            print(f"  rr: {diag['rr_summary']}")
            print(f"  ti_intervals ({len(diag['ti_intervals'])}): {diag['ti_intervals']}")
            print(f"  tail500 stray_counts={dict(stray_counts)} unpenned_bounds={sheep_pos_bounds}")
            print(f"  tail500 dog_bounds={dog_pos_bounds} roles={role_counts}")
            # Sample 3 frames near end
            for sample_step in (traces[-1].step - 10, traces[-1].step - 1, traces[-1].step):
                if 1 <= sample_step <= len(traces):
                    tr = traces[sample_step - 1]
                    print(
                        f"    step {tr.step}: dogs={tr.dog_positions} roles={tr.dog_roles} "
                        f"targets={tr.targets} stray={tr.stray_idx} sheep={tr.unpenned_positions} "
                        f"actions={tr.actions} noprog={tr.no_progress_steps}"
                    )


def compare_pair(model: str, base_var: str, test_var: str, seed: int, label: str) -> None:
    """Compare two variants step-by-step on a single model and seed."""
    p_base = replay_path_for(model, base_var, seed)
    p_test = replay_path_for(model, test_var, seed)
    b_base = load_bundle(p_base)
    b_test = load_bundle(p_test)
    d_base = summarize_diagnostics(p_base)
    d_test = summarize_diagnostics(p_test)

    # Reconstruct base trace with alt targets from test_var to see exact step where test_var would change targets
    t_base = reconstruct_trace(b_base, base_var, compare_variant=test_var)
    t_test = reconstruct_trace(b_test, test_var, compare_variant=base_var)

    first_target_diff: int | None = None
    first_action_diff: int | None = None
    first_state_diff: int | None = None
    min_len = min(len(t_base), len(t_test))
    for i in range(min_len):
        tb, tt = t_base[i], t_test[i]
        if first_target_diff is None and tb.targets != tb.alt_targets:
            first_target_diff = tb.step
        if first_action_diff is None and tb.actions != tt.actions:
            first_action_diff = tb.step
        if first_state_diff is None and (
            tb.dog_positions != tt.dog_positions or tb.unpenned_positions != tt.unpenned_positions
        ):
            first_state_diff = tb.step

    print(f"\n=== {label}: {model} seed {seed} ({base_var} vs {test_var}) ===")
    print(f"  {base_var}: {d_base['stop_reason']} steps={d_base['steps']} penned={d_base['penned']}/6 rr={d_base['rr_summary']}")
    print(f"  {test_var}: {d_test['stop_reason']} steps={d_test['steps']} penned={d_test['penned']}/6 rr={d_test['rr_summary']}")
    print(f"  first_target_diff_on_base={first_target_diff}, first_action_diff={first_action_diff}, first_state_diff={first_state_diff}")

    if first_action_diff is not None:
        start = max(1, first_action_diff - 2)
        end = min(min_len, first_action_diff + 5)
        print(f"  --- Divergence window steps {start}..{end} ---")
        for s in range(start, end + 1):
            tb, tt = t_base[s - 1], t_test[s - 1]
            print(
                f"    step {s} [{base_var}]: stray={tb.stray_idx} targets={tb.targets} alt={tb.alt_targets} "
                f"dogs={tb.dog_positions} actions={tb.actions} sheep={tb.unpenned_positions}"
            )
            print(
                f"    step {s} [{test_var}]: stray={tt.stray_idx} targets={tt.targets} "
                f"dogs={tt.dog_positions} actions={tt.actions} sheep={tt.unpenned_positions}"
            )

    print(f"  {base_var} ti_intervals: {d_base['ti_intervals']}")
    print(f"  {test_var} ti_intervals: {d_test['ti_intervals']}")

    # Print penning timelines
    for name, tr_list in ((base_var, t_base), (test_var, t_test)):
        events = []
        prev = set(range(6))
        for tr in tr_list:
            cur = set(tr.unpenned_indices)
            diff = sorted(prev - cur)
            if diff:
                events.append(f"s{diff}@{tr.step}")
            prev = cur
        print(f"  {name} penning: {events}; final_unpenned={tr_list[-1].unpenned_positions}")

    # For failing run, sample tail
    for name, tr_list, diag in ((base_var, t_base, d_base), (test_var, t_test, d_test)):
        if not diag["success"]:
            tail = tr_list[-min(300, len(tr_list)):]
            s_bounds = {}
            for s_idx in tr_list[-1].unpenned_indices:
                xs = [tr.unpenned_positions[s_idx][0] for tr in tail if s_idx in tr.unpenned_positions]
                ys = [tr.unpenned_positions[s_idx][1] for tr in tail if s_idx in tr.unpenned_positions]
                s_bounds[s_idx] = (min(xs), max(xs), min(ys), max(ys))
            d_bounds = {
                d: (
                    min(tr.dog_positions[d][0] for tr in tail),
                    max(tr.dog_positions[d][0] for tr in tail),
                    min(tr.dog_positions[d][1] for tr in tail),
                    max(tr.dog_positions[d][1] for tr in tail),
                )
                for d in range(3)
            }
            print(f"  {name} fail tail300: unpenned_bounds={s_bounds} dog_bounds={d_bounds}")
            for sample_step in (tr_list[-1].step - 5, tr_list[-1].step):
                tr = tr_list[sample_step - 1]
                print(
                    f"    step {tr.step}: dogs={tr.dog_positions} roles={tr.dog_roles} "
                    f"targets={tr.targets} stray={tr.stray_idx} sheep={tr.unpenned_positions} actions={tr.actions}"
                )


def main() -> None:
    """CLI entrypoint for offline split-flock replay analysis."""
    parser = argparse.ArgumentParser(description="Analyze split-flock evaluation replays.")
    parser.add_argument(
        "--section",
        choices=("all", "matrix", "seed73", "regressions", "wins"),
        default="all",
    )
    args = parser.parse_args()

    if args.section in ("all", "matrix"):
        print_result_matrix()
    if args.section in ("all", "seed73"):
        analyze_seed_73()
    if args.section in ("all", "regressions"):
        for seed in (41, 53, 67):
            compare_pair("pv_10483", "baseline", "main_group", seed, "REGRESSION")
    if args.section in ("all", "wins"):
        for seed in (59, 71):
            compare_pair("pv_10486", "baseline", "corner_clearance", seed, "CORNER_CLEARANCE WIN")
        for seed in (59, 71):
            compare_pair("pv_10483", "baseline", "corner_clearance", seed, "PV_10483 CORNER_CLEARANCE CHECK")


if __name__ == "__main__":
    main()

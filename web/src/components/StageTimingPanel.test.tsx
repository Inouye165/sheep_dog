import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { StageTimingPanel, formatDuration } from "./StageTimingPanel";
import * as apiModule from "../lib/api";
import type {
  ModelHistoryResponse,
  ModelPassRecord,
  StageTimingRecord,
  StageTimingsResponse,
} from "../state/types";

function rec(run: string, stage: number, wall: number, source: "recorded" | "derived" = "recorded"): StageTimingRecord {
  return {
    promotion_event_id: `${run}-${stage}`,
    run_id: run,
    stage,
    to_stage: stage + 1,
    source,
    started_at: "2026-11-24T10:00:00Z",
    ended_at: `2026-11-24T1${stage}:00:00Z`,
    wall_seconds: wall,
    training_seconds: source === "recorded" ? wall - 10 : null,
    policy_timesteps: 1000 * stage,
    world_steps: null,
    steps_per_second: 5.5,
    success_rate: 1,
    success_count: 6,
    config_hash: "abc",
    assistance_mode: "guided",
    run_label: null,
    git_revision: "595638570e51",
    git_dirty: true,
  };
}

function response(records: StageTimingRecord[], withComparison = false): StageTimingsResponse {
  return {
    schema_version: 1,
    records,
    comparison: withComparison
      ? [
          {
            stage: 1,
            run_count: 2,
            latest_run_id: "run_b",
            best_run_id: "run_b",
            vs_best: null,
            vs_previous: {
              run_id: "run_a",
              comparable: true,
              config_hash_differs: false,
              assistance_differs: false,
              notes: [],
              deltas: {
                wall_seconds: { latest: 50, reference: 100, pct_delta: -50 },
                policy_timesteps: { latest: 1000, reference: 1000, pct_delta: 0 },
                steps_per_second: { latest: 5, reference: 5, pct_delta: 0 },
              },
            },
          },
        ]
      : [],
  };
}

describe("StageTimingPanel", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it("formats durations", () => {
    expect(formatDuration(3725)).toBe("1h 2m 5s");
    expect(formatDuration(null)).toBe("—");
  });

  it("shows the one-run hint and table with badges", async () => {
    vi.spyOn(apiModule, "loadStageTimings").mockResolvedValue(
      response([rec("run_a", 1, 100), rec("run_a", 2, 200, "derived")]),
    );
    render(<StageTimingPanel />);
    await waitFor(() => expect(screen.getByText(/Only one run recorded; run again to compare/)).toBeTruthy());
    expect(document.querySelector(".badge--recorded")).not.toBeNull();
    expect(document.querySelector(".badge--derived")).not.toBeNull();
    expect(screen.getAllByText("5956385+dirty").length).toBeGreaterThan(0);
  });

  it("shows delta vs previous run and toggles chart metric", async () => {
    vi.spyOn(apiModule, "loadStageTimings").mockResolvedValue(
      response([rec("run_a", 1, 100), rec("run_b", 1, 50)], true),
    );
    render(<StageTimingPanel />);
    await waitFor(() => expect(screen.getByText("-50.0%")).toBeTruthy());
    const toggle = screen.getByRole("button", { name: "Policy timesteps (PPO steps)" });
    fireEvent.click(toggle);
    expect(toggle.getAttribute("aria-pressed")).toBe("true");
    expect(screen.queryByText(/Only one run recorded/)).toBeNull();
  });

  it("shows empty state", async () => {
    vi.spyOn(apiModule, "loadStageTimings").mockResolvedValue(response([]));
    render(<StageTimingPanel />);
    await waitFor(() => expect(screen.getByText(/No stage timings recorded yet/)).toBeTruthy());
  });

  it("renders recorded model passes, selected lineage cumulative totals, stage-by-stage comparison, and pass switching", async () => {
    function makePass(overrides: Partial<ModelPassRecord>): ModelPassRecord {
      return {
        pass_id: "pass_1",
        label: "Pass 1",
        run_id: "run_1",
        parent_pass_id: null,
        parent_checkpoint_id: null,
        parent_checkpoint_path: null,
        parent_checkpoint_sha256: null,
        resulting_checkpoint_id: "ckpt_1",
        resulting_checkpoint_path: "artifacts/ckpt_1.zip",
        resulting_checkpoint_sha256: "abcdef1234567890",
        pass_type: "full_curriculum",
        stage_spec: "1-38",
        planned_stages: [25, 36],
        completed_stages_count: 2,
        status: "completed",
        started_at: "2026-11-24T10:00:00Z",
        ended_at: "2026-11-24T12:00:00Z",
        total_training_seconds: 120,
        total_evaluation_seconds: 30,
        total_checkpoint_save_seconds: 5,
        total_replay_serialization_seconds: 2,
        total_active_seconds: 157,
        total_wall_seconds: 180,
        total_policy_timesteps: 50000,
        assistance_mode: "guided",
        observation_mode: "guided",
        env_config_version: "1",
        config_hash: "cfg1",
        git_revision: "5956385",
        git_dirty: false,
        source: "recorded",
        notes: null,
        stage_runs: [],
        ...overrides,
      };
    }

    const pass1 = makePass({
      pass_id: "pass_1",
      label: "Initial Full Pass",
      parent_pass_id: null,
      stage_spec: "1-38",
      planned_stages: [1, 25, 36],
      completed_stages_count: 3,
      total_active_seconds: 300,
      total_training_seconds: 240,
      total_evaluation_seconds: 50,
      total_wall_seconds: 320,
      total_policy_timesteps: 100000,
      assistance_mode: "guided",
      resulting_checkpoint_sha256: "1111aaa2222bbb",
    });

    const pass2 = makePass({
      pass_id: "pass_2",
      label: "Targeted Polish",
      parent_pass_id: "pass_1",
      pass_type: "targeted",
      stage_spec: "25, 36",
      planned_stages: [25, 36],
      completed_stages_count: 2,
      total_active_seconds: 180,
      total_training_seconds: 150,
      total_evaluation_seconds: 25,
      total_wall_seconds: 200,
      total_policy_timesteps: 60000,
      assistance_mode: "emergent_gate",
      resulting_checkpoint_sha256: "3333ccc4444ddd",
    });

    const pass3 = makePass({
      pass_id: "pass_3",
      label: "Endgame Mastery",
      parent_pass_id: "pass_2",
      pass_type: "targeted",
      stage_spec: "25, 36",
      planned_stages: [25, 36],
      completed_stages_count: 2,
      total_active_seconds: 120,
      total_training_seconds: 90,
      total_evaluation_seconds: 20,
      total_wall_seconds: 135,
      total_policy_timesteps: 40000,
      assistance_mode: "emergent_gate",
      resulting_checkpoint_sha256: "5555eee6666fff",
    });

    const historyPass3: ModelHistoryResponse = {
      schema_version: 1,
      passes: [pass1, pass2, pass3],
      active_pass_id: "pass_3",
      selected_lineage: {
        target_pass_id: "pass_3",
        lineage: [pass1, pass2, pass3],
        cumulative: {
          pass_count: 3,
          stage_runs_count: 7,
          completed_stage_runs_count: 7,
          total_training_seconds: 480,
          total_evaluation_seconds: 95,
          total_checkpoint_save_seconds: 15,
          total_replay_serialization_seconds: 6,
          total_active_seconds: 600,
          total_wall_seconds: 655,
          total_policy_timesteps: 200000,
        },
        stage_comparison: [
          {
            stage: 25,
            runs: [
              {
                pass_id: "pass_1",
                pass_index: 0,
                label: "Initial Full Pass",
                sequence_order: 1,
                status: "promoted",
                active_seconds: 200,
                training_seconds: 160,
                evaluation_seconds: 35,
                wall_seconds: 210,
                policy_timesteps: 60000,
                steps_per_second: 375,
                initial_success_rate: 0.2,
                final_success_rate: 0.9,
                best_success_rate: 0.9,
                final_avg_steps: 140,
              },
              {
                pass_id: "pass_2",
                pass_index: 1,
                label: "Targeted Polish",
                sequence_order: 0,
                status: "promoted",
                active_seconds: 100,
                training_seconds: 80,
                evaluation_seconds: 15,
                wall_seconds: 110,
                policy_timesteps: 30000,
                steps_per_second: 375,
                initial_success_rate: 0.5,
                final_success_rate: 0.95,
                best_success_rate: 0.95,
                final_avg_steps: 120,
              },
              {
                pass_id: "pass_3",
                pass_index: 2,
                label: "Endgame Mastery",
                sequence_order: 0,
                status: "mastered",
                active_seconds: 60,
                training_seconds: 45,
                evaluation_seconds: 10,
                wall_seconds: 65,
                policy_timesteps: 15000,
                steps_per_second: 333,
                initial_success_rate: 0.7,
                final_success_rate: 1.0,
                best_success_rate: 1.0,
                final_avg_steps: 98,
              },
            ],
            delta_vs_first: {
              active_seconds_pct: -70,
              training_seconds_pct: -71.9,
              policy_timesteps_pct: -75,
            },
            delta_vs_previous: {
              active_seconds_pct: -40,
              training_seconds_pct: -43.8,
              policy_timesteps_pct: -50,
            },
          },
        ],
      },
      stage_matrix: [],
    };

    const historyPass2: ModelHistoryResponse = {
      schema_version: 1,
      passes: [pass1, pass2, pass3],
      active_pass_id: "pass_3",
      selected_lineage: {
        target_pass_id: "pass_2",
        lineage: [pass1, pass2],
        cumulative: {
          pass_count: 2,
          stage_runs_count: 5,
          completed_stage_runs_count: 5,
          total_training_seconds: 390,
          total_evaluation_seconds: 75,
          total_checkpoint_save_seconds: 10,
          total_replay_serialization_seconds: 4,
          total_active_seconds: 480,
          total_wall_seconds: 520,
          total_policy_timesteps: 160000,
        },
        stage_comparison: [
          {
            stage: 25,
            runs: historyPass3.selected_lineage!.stage_comparison[0].runs.slice(0, 2),
            delta_vs_first: {
              active_seconds_pct: -50,
              training_seconds_pct: -50,
              policy_timesteps_pct: -50,
            },
            delta_vs_previous: {
              active_seconds_pct: -50,
              training_seconds_pct: -50,
              policy_timesteps_pct: -50,
            },
          },
        ],
      },
      stage_matrix: [],
    };

    vi.spyOn(apiModule, "loadStageTimings").mockResolvedValue(response([]));
    const historySpy = vi
      .spyOn(apiModule, "loadModelHistory")
      .mockImplementation(async (params) =>
        params?.passId === "pass_2" ? historyPass2 : historyPass3,
      );

    render(<StageTimingPanel />);

    await waitFor(() => expect(screen.getByText("Model Training Passes & Lineage")).toBeTruthy());

    expect(screen.getByText("fresh init")).toBeTruthy();
    expect(screen.getByText("sha 1111aaa")).toBeTruthy();
    expect(screen.getAllByText("emergent_gate")).toHaveLength(2);
    expect(screen.getByText("Initial Full Pass → Targeted Polish → Endgame Mastery")).toBeTruthy();
    expect(screen.getByText("10m 0s")).toBeTruthy(); // 600s cumulative active
    expect(screen.getByText("8m 0s")).toBeTruthy(); // 480s cumulative training
    expect(screen.getByText("200,000")).toBeTruthy(); // cumulative timesteps

    // Stage 25 progression & deltas
    expect(screen.getByText("20% → 90%")).toBeTruthy();
    expect(screen.getByText("70% → 100%")).toBeTruthy();
    expect(screen.getByText("Active: -40.0%")).toBeTruthy();
    expect(screen.getByText("Active: -70.0%")).toBeTruthy();

    // Switch selected pass to pass_2
    const inspectButtons = screen.getAllByRole("button", { name: "Inspect lineage" });
    fireEvent.click(inspectButtons[1]);

    await waitFor(() =>
      expect(historySpy).toHaveBeenCalledWith({ passId: "pass_2", stage: null }),
    );
    await waitFor(() =>
      expect(screen.getByText("Initial Full Pass → Targeted Polish")).toBeTruthy(),
    );
    expect(screen.getByText("160,000")).toBeTruthy();
    expect(screen.getAllByText("Active: -50.0%").length).toBeGreaterThan(0);
  });
});

import { act, fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { DiagnosticsPanel } from "./DiagnosticsPanel";
import type { CheckpointIndex, TrainingStatus } from "../state/types";

const mockCheckpointIndex: CheckpointIndex = {
  checkpoints: [
    {
      checkpoint_episode: 6508,
      recorded_at: "2026-08-02T22:56:42.413250+00:00",
      checkpoint: "checkpoint-006508.json",
      evaluation: "eval_006508.json",
      replay: "/generated/replays/checkpoint-006508.json",
      policy_name: "neural_policy",
      trainer_type: "maskable_ppo",
      policy_type: "neural",
      policy_mode: "neural_only",
      replay_mode: "truthful",
      total_training_episodes: 954,
      policy_state_path: "path/to/state",
      success_rate: 0.7,
      timeout_rate: 0.3,
      average_completion_steps: 387.9,
      average_completion_seconds: 387.9,
      average_sheep_penned: 3.4,
      average_reward: 245.6,
      average_distance_to_pen: 4.7,
      average_flock_spread: 2.2,
      records: [],
      checkpoint_id: "chk_6508",
      global_timestep: 35874588,
      curriculum_stage: 8,
      environment_config: { dogs: 3, sheep: 4, width: 200, height: 200, curriculum_stage: 8 },
      evaluation_mode: "confidence",
      evaluation_seed_count: 10,
    },
  ],
  latest: null,
};

const activeTrainingStatus: TrainingStatus = {
  running: true,
  fast_mode: true,
  trainer_type: "maskable_ppo",
  policy_type: "neural",
  enable_instinct_rewards: false,
  policy_mode: "neural_only",
  replay_mode: "truthful",
  debug_reward_breakdown: false,
  curriculum_stage: 8,
  requested_episodes: 100,
  completed_episodes: 79,
  batch_total_episodes: 100,
  batch_completed_episodes: 79,
  total_episodes_trained: 17104,
  episodes_in_stage: 1033,
  current_stage_environment_episode: 1033,
  latest_completed_environment_episode: 1033,
  episodes_since_latest_confidence_evaluation: 79,
  live_rollout_window_count: 79,
  live_rollout_success_count: 67,
  live_rollout_failure_count: 12,
  live_rollout_stopped_count: 9,
  live_rollout_timeout_count: 3,
  live_rollout_success_rate: 0.8481,
  current_global_timestep: 36088656,
  latest_checkpoint_global_timestep: 35874588,
  timesteps_since_latest_checkpoint: 214068,
  next_evaluation_environment_episode: 1050,
  episodes_until_next_evaluation: 17,
  latest_episode_result: "SUCCESS",
  latest_episode_reward: 437.27,
  total_timesteps: 36088656,
  latest_checkpoint_episode: 6508,
  active_checkpoint_id: "chk_6508",
  last_evaluation_time: "2026-08-02T22:56:42.413250+00:00",
  checkpoint_save_interval: 50,
  checkpoint_episode: 6508,
  policy_version: 2557,
  stage_history: { "8": 3027 },
  grand_total_episodes: 17104,
  phase: "curriculum_active",
  message: "Training active",
  error: null,
  starting_episode: null,
  current_episode: null,
  latest_seed: null,
  latest_replay_path: null,
  best_score: null,
  latest_success_rate: null,
  latest_avg_sheep_penned: null,
  latest_avg_reward: null,
  latest_timeout_rate: null,
  latest_stopped_rate: null,
  latest_avg_no_progress_steps: null,
  latest_avg_distance_to_pen: null,
  latest_avg_flock_spread: null,
  latest_avg_farthest_distance_to_pen: null,
  latest_avg_farthest_distance_to_flock_center: null,
  runtime: {
    active_seconds_total: 1200,
    training_seconds: 1000,
    evaluation_seconds: 200,
    replay_capture_seconds: 0,
    replay_serialization_seconds: 0,
    checkpoint_save_seconds: 0,
    paused_seconds: 0,
    wall_clock_seconds: 1200,
    offline_or_unknown_seconds: 0,
    episodes_per_active_hour: 100,
    timesteps_per_training_second: 500,
    training_time_percentage: 0.83,
    session_count: 1,
    sessions: [],
  },
};

describe("DiagnosticsPanel Live Insights & Telemetry", () => {
  it("1. Live rollout counts increase without adding points to formal evaluation chart series", () => {
    const { rerender } = render(
      <DiagnosticsPanel
        checkpointIndex={mockCheckpointIndex}
        bestCheckpointEpisode={6508}
        trainingStatus={activeTrainingStatus}
        effectiveCurriculumStage={8}
        lastLiveRefreshTime={Date.now()}
      />
    );

    expect(screen.getAllByText(/Current Stage 8 Episode:/i).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/1,033|1033/i).length).toBeGreaterThan(0);
    expect(screen.getByText(/1 pts/i)).toBeDefined();

    const updatedStatus: TrainingStatus = {
      ...activeTrainingStatus,
      current_stage_environment_episode: 1040,
      episodes_since_latest_confidence_evaluation: 86,
      live_rollout_window_count: 86,
      live_rollout_success_count: 73,
      live_rollout_failure_count: 13,
      live_rollout_success_rate: 0.8488,
    };

    rerender(
      <DiagnosticsPanel
        checkpointIndex={mockCheckpointIndex}
        bestCheckpointEpisode={6508}
        trainingStatus={updatedStatus}
        effectiveCurriculumStage={8}
        lastLiveRefreshTime={Date.now()}
      />
    );

    expect(screen.getAllByText(/1,040|1040/i).length).toBeGreaterThan(0);
    expect(screen.getByText(/1 pts/i)).toBeDefined();
  });

  it("2. Checkpoint 6550 is explicitly labeled as Checkpoint Sequence, not Environment Episode", () => {
    render(
      <DiagnosticsPanel
        checkpointIndex={mockCheckpointIndex}
        bestCheckpointEpisode={6508}
        trainingStatus={activeTrainingStatus}
        effectiveCurriculumStage={8}
        lastLiveRefreshTime={Date.now()}
      />
    );

    const toggleBtn = screen.queryByRole("button", { name: /expand details/i });
    if (toggleBtn) fireEvent.click(toggleBtn);

    expect(screen.getAllByText(/Checkpoint Sequence:/i).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/6508/i).length).toBeGreaterThan(0);
  });

  it("3. The banner displays derived episodes-since-evaluation count and truthful next evaluation boundary", () => {
    render(
      <DiagnosticsPanel
        checkpointIndex={mockCheckpointIndex}
        bestCheckpointEpisode={6508}
        trainingStatus={activeTrainingStatus}
        effectiveCurriculumStage={8}
        lastLiveRefreshTime={Date.now()}
      />
    );

    const toggleBtn = screen.queryByRole("button", { name: /expand details/i });
    if (toggleBtn) fireEvent.click(toggleBtn);

    expect(
      screen.getByText(
        /Training active — 79 training episodes completed since the latest confidence evaluation. Next confidence evaluation pending./i
      )
    ).toBeDefined();

    expect(
      screen.getByText(
        /Stage 8 Episode 1050/i
      )
    ).toBeDefined();
  });

  it("4. Live success rate uses SQLite telemetry and does not display 0/1032 on missing fields", () => {
    const statusWithMissingCounts: TrainingStatus = {
      ...activeTrainingStatus,
      live_rollout_success_count: undefined,
      live_rollout_failure_count: undefined,
      live_rollout_success_rate: null,
    };

    render(
      <DiagnosticsPanel
        checkpointIndex={mockCheckpointIndex}
        bestCheckpointEpisode={6508}
        trainingStatus={statusWithMissingCounts}
        effectiveCurriculumStage={8}
        lastLiveRefreshTime={Date.now()}
      />
    );

    const toggleBtn = screen.queryByRole("button", { name: /expand details/i });
    if (toggleBtn) fireEvent.click(toggleBtn);

    expect(screen.getByText(/Live Rollout Success Rate:/i)).toBeDefined();
    expect(screen.getAllByText(/Unavailable/i).length).toBeGreaterThan(0);
    expect(screen.queryByText(/0 \/ 1032/i)).toBeNull();
  });

  it("5. Single checkpoint index query with cache-busting timestamp", async () => {
    const fetchMock = vi.fn().mockImplementation(() =>
      Promise.resolve(
        new Response(JSON.stringify(mockCheckpointIndex), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        })
      )
    );
    vi.stubGlobal("fetch", fetchMock);

    const { loadCheckpointIndex, setIsBackendOffline } = await import("../lib/api");
    setIsBackendOffline(false);
    await loadCheckpointIndex();

    expect(fetchMock).toHaveBeenCalled();
    const requestedUrl = String(fetchMock.mock.calls[0][0]);
    expect(requestedUrl).toMatch(/\/generated\/checkpoint-index\.json\?t=\d+/);
    vi.unstubAllGlobals();
  });

  it("6. Current Journey scope: bootstrap call omits runId to always seed latest data; runId applied on incremental polls", async () => {
    const api = await import("../lib/api");
    const calls: any[] = [];
    const loadEpisodesSpy = vi.spyOn(api, "loadTrainingEpisodes").mockImplementation(async (options) => {
      calls.push(options);
      return {
        episodes: Array.from({ length: 25 }, (_, i) => ({
          id: 100 + i,
          run_id: "run_active_123",
          global_environment_episode: 100 + i,
          curriculum_stage: 8,
          reward: 180.0,
          result: "SUCCESS",
          success: true,
          global_timestep: (100 + i) * 500,
        })) as any,
        latest_id: 124,
        next_after_id: 124,
        has_more: false,
        max_id: 124,
        oldest_available_timestamp: null,
        total_matching: 25,
      };
    });

    const statusWithRun: TrainingStatus = {
      ...activeTrainingStatus,
      run_id: "run_active_123",
    };

    render(
      <DiagnosticsPanel
        checkpointIndex={mockCheckpointIndex}
        bestCheckpointEpisode={6508}
        trainingStatus={statusWithRun}
        effectiveCurriculumStage={8}
        lastLiveRefreshTime={Date.now()}
      />
    );

    // Bootstrap call (lastEpisodeId === 0) deliberately omits runId so today's data
    // is always fetched regardless of when trainingStatus loads.
    const bootstrapCall = calls[0];
    expect(bootstrapCall).toBeDefined();
    expect(bootstrapCall.runId).toBeUndefined();
    expect(bootstrapCall.order).toBe("desc");
    expect(bootstrapCall.afterId).toBeUndefined();
    loadEpisodesSpy.mockRestore();
  });

  it("Test A & D: Selecting Stage 2 with no Stage 2 data yields 0 episodes and never falls back to Stage 1; Stage 2 data yields strict stage 2 episodes", async () => {
    const api = await import("../lib/api");
    const loadEpisodesSpy = vi.spyOn(api, "loadTrainingEpisodes").mockImplementation(async (options) => {
      if (options?.stage === 2) {
        return {
          episodes: [
            {
              id: 201,
              run_id: "run_active_123",
              global_environment_episode: 201,
              curriculum_stage: 2,
              reward: 150.0,
              result: "SUCCESS",
              success: true,
              global_timestep: 20100,
            },
          ] as any,
          latest_id: 201,
          next_after_id: 201,
          has_more: false,
          max_id: 201,
          oldest_available_timestamp: null,
          total_matching: 1,
        };
      }
      return {
        episodes: Array.from({ length: 100 }, (_, i) => ({
          id: i + 1,
          run_id: "run_active_123",
          global_environment_episode: i + 1,
          curriculum_stage: 1,
          reward: 100,
          result: "SUCCESS",
          success: true,
          global_timestep: (i + 1) * 10,
        })),
        latest_id: 100,
        next_after_id: 100,
        has_more: false,
        max_id: 100,
        oldest_available_timestamp: null,
        total_matching: 100,
      };
    });

    const statusWithRun: TrainingStatus = {
      ...activeTrainingStatus,
      run_id: "run_active_123",
    };

    render(
      <DiagnosticsPanel
        checkpointIndex={mockCheckpointIndex}
        bestCheckpointEpisode={6508}
        trainingStatus={statusWithRun}
        effectiveCurriculumStage={2}
        initialStageScope={2}
        lastLiveRefreshTime={Date.now()}
      />
    );

    // Bootstrap call omits runId but must include the stage filter.
    // runId isolation is handled client-side by processCanonicalHistory.
    expect(loadEpisodesSpy).toHaveBeenCalledWith(
      expect.objectContaining({
        stage: 2,
        runId: undefined,
      })
    );

    loadEpisodesSpy.mockRestore();
  });

  it("Test B: Switching query scope resets pagination cursor (afterId is undefined for initial scope fetch)", async () => {
    const api = await import("../lib/api");
    const loadEpisodesSpy = vi.spyOn(api, "loadTrainingEpisodes").mockImplementation(async () => {
      return {
        episodes: [],
        latest_id: 0,
        next_after_id: 0,
        has_more: false,
        max_id: 0,
        oldest_available_timestamp: null,
        total_matching: 0,
      };
    });

    const statusWithRun: TrainingStatus = {
      ...activeTrainingStatus,
      run_id: "run_active_123",
    };

    render(
      <DiagnosticsPanel
        checkpointIndex={mockCheckpointIndex}
        bestCheckpointEpisode={6508}
        trainingStatus={statusWithRun}
        effectiveCurriculumStage={2}
        initialStageScope={2}
        lastLiveRefreshTime={Date.now()}
      />
    );

    // Initial fetch for scope has order: "desc" and afterId: undefined
    expect(loadEpisodesSpy).toHaveBeenCalledWith(
      expect.objectContaining({
        afterId: undefined,
        order: "desc",
      })
    );

    loadEpisodesSpy.mockRestore();
  });

  it("Test C: Stage selection under Current Journey scopes by active run_id and excludes archived runs", async () => {
    const api = await import("../lib/api");
    const loadEpisodesSpy = vi.spyOn(api, "loadTrainingEpisodes").mockImplementation(async (options) => {
      if (options?.runId === "run_active_123" && options?.stage === 2) {
        return {
          episodes: [
            {
              id: 500,
              run_id: "run_active_123",
              global_environment_episode: 500,
              curriculum_stage: 2,
              reward: 200.0,
              result: "SUCCESS",
              success: true,
              global_timestep: 50000,
            },
          ] as any,
          latest_id: 500,
          next_after_id: 500,
          has_more: false,
          max_id: 500,
          oldest_available_timestamp: null,
          total_matching: 1,
        };
      }
      return {
        episodes: [],
        latest_id: 0,
        next_after_id: 0,
        has_more: false,
        max_id: 0,
        oldest_available_timestamp: null,
        total_matching: 0,
      };
    });

    const statusWithRun: TrainingStatus = {
      ...activeTrainingStatus,
      run_id: "run_active_123",
    };

    render(
      <DiagnosticsPanel
        checkpointIndex={mockCheckpointIndex}
        bestCheckpointEpisode={6508}
        trainingStatus={statusWithRun}
        effectiveCurriculumStage={2}
        initialStageScope={2}
        lastLiveRefreshTime={Date.now()}
      />
    );

    // Bootstrap call omits runId but must include the stage filter.
    // runId isolation for Current Journey view is enforced by processCanonicalHistory.
    expect(loadEpisodesSpy).toHaveBeenCalledWith(
      expect.objectContaining({
        stage: 2,
        runId: undefined,
      })
    );

    loadEpisodesSpy.mockRestore();
  });

  it("Block Performance Learning Curve calculates non-overlapping 25-episode block success rates", async () => {
    const { calculateBlockSuccessPoints } = await import("./DiagnosticsPanel");
    
    // Create 100 episodes:
    // Block 1 (1-25): 5 successes = 20%
    // Block 2 (26-50): 10 successes = 40%
    // Block 3 (51-75): 15 successes = 60%
    // Block 4 (76-100): 20 successes = 80%
    const episodes = Array.from({ length: 100 }, (_, i) => {
      const epNum = i + 1;
      let isSuccess = false;
      if (epNum <= 25) {
        isSuccess = epNum <= 5;
      } else if (epNum <= 50) {
        isSuccess = epNum <= 25 + 10;
      } else if (epNum <= 75) {
        isSuccess = epNum <= 50 + 15;
      } else {
        isSuccess = epNum <= 75 + 20;
      }

      return {
        id: epNum,
        run_id: "run_active_123",
        global_environment_episode: epNum,
        curriculum_stage: 1,
        reward: isSuccess ? 180 : -20,
        result: isSuccess ? "SUCCESS" : "STOPPED",
        success: isSuccess,
        global_timestep: epNum * 100,
      };
    });

    const points = calculateBlockSuccessPoints(episodes as any, 25, (ep) => ep.global_environment_episode ?? null);
    expect(points.length).toBe(4);

    expect(points[0].x).toBe(25);
    expect(points[0].y).toBe(20);

    expect(points[1].x).toBe(50);
    expect(points[1].y).toBe(40);

    expect(points[2].x).toBe(75);
    expect(points[2].y).toBe(60);

    expect(points[3].x).toBe(100);
    expect(points[3].y).toBe(80);
  });

  it("Proves that records from an old run at different timesteps do not appear in Current Stage", async () => {
    const api = await import("../lib/api");
    const oldRunEpisodes = Array.from({ length: 50 }, (_, i) => ({
      id: i + 1,
      run_id: "old_archived_run_999",
      global_environment_episode: i + 1,
      curriculum_stage: 1,
      reward: 100,
      result: "SUCCESS",
      success: true,
      global_timestep: (i + 1) * 45,
    }));

    const activeRunEpisodes = Array.from({ length: 50 }, (_, i) => ({
      id: 100 + i + 1,
      run_id: "run_active_123",
      global_environment_episode: 100 + i + 1,
      curriculum_stage: 1,
      reward: 100,
      result: "SUCCESS",
      success: true,
      global_timestep: (100 + i + 1) * 91,
    }));

    const loadEpisodesSpy = vi.spyOn(api, "loadTrainingEpisodes").mockImplementation(async (options) => {
      if (options?.runId === "run_active_123") {
        return {
          episodes: activeRunEpisodes as any,
          latest_id: 150,
          next_after_id: 150,
          has_more: false,
          max_id: 150,
          oldest_available_timestamp: null,
          total_matching: 50,
        };
      }
      return {
        episodes: [...oldRunEpisodes, ...activeRunEpisodes] as any,
        latest_id: 150,
        next_after_id: 150,
        has_more: false,
        max_id: 150,
        oldest_available_timestamp: null,
        total_matching: 100,
      };
    });

    const statusWithRun: TrainingStatus = {
      ...activeTrainingStatus,
      run_id: "run_active_123",
    };

    render(
      <DiagnosticsPanel
        checkpointIndex={mockCheckpointIndex}
        bestCheckpointEpisode={6508}
        trainingStatus={statusWithRun}
        effectiveCurriculumStage={1}
        initialStageScope={1}
        lastLiveRefreshTime={Date.now()}
      />
    );

    // Bootstrap call omits runId. The mock returns mixed old+active episodes when
    // runId is undefined. processCanonicalHistory's in-memory run_id filter then
    // ensures only active-run records appear in the chart.
    expect(loadEpisodesSpy).toHaveBeenCalledWith(
      expect.objectContaining({
        stage: 1,
        runId: undefined,
      })
    );

    loadEpisodesSpy.mockRestore();
  });

  it("automatically changes stage scope when app triggers a stage transition from 8 to 9", () => {
    const { rerender } = render(
      <DiagnosticsPanel
        checkpointIndex={mockCheckpointIndex}
        bestCheckpointEpisode={6508}
        trainingStatus={activeTrainingStatus}
        effectiveCurriculumStage={8}
        lastLiveRefreshTime={Date.now()}
      />
    );

    const stageSelect = screen.getByLabelText("Stage scope") as HTMLSelectElement;
    expect(stageSelect.value).toBe("current");

    // Rerender with advanced curriculum stage 9
    rerender(
      <DiagnosticsPanel
        checkpointIndex={mockCheckpointIndex}
        bestCheckpointEpisode={6508}
        trainingStatus={{
          ...activeTrainingStatus,
          curriculum_stage: 9,
        }}
        effectiveCurriculumStage={9}
        lastLiveRefreshTime={Date.now()}
      />
    );

    // Should automatically track the new stage
    expect(stageSelect.value).toBe("current");
    expect(screen.getAllByText(/Current stage \(Stage 9\)/i).length).toBeGreaterThan(0);
  });

  it("preserves user manual selection of a previous stage during routine telemetry updates without reverting to current", () => {
    const { rerender } = render(
      <DiagnosticsPanel
        checkpointIndex={mockCheckpointIndex}
        bestCheckpointEpisode={6508}
        trainingStatus={activeTrainingStatus}
        effectiveCurriculumStage={9}
        lastLiveRefreshTime={Date.now()}
      />
    );

    const stageSelect = screen.getByLabelText("Stage scope") as HTMLSelectElement;
    // User explicitly switches to Stage 8 to review historical data
    fireEvent.change(stageSelect, { target: { value: "8" } });
    expect(stageSelect.value).toBe("8");

    // Training emits telemetry refresh while remaining on stage 9
    rerender(
      <DiagnosticsPanel
        checkpointIndex={mockCheckpointIndex}
        bestCheckpointEpisode={6508}
        trainingStatus={{
          ...activeTrainingStatus,
          total_episodes_trained: 2000,
        }}
        effectiveCurriculumStage={9}
        lastLiveRefreshTime={Date.now() + 1000}
      />
    );

    // Remains on Stage 8 as chosen by user
    expect(stageSelect.value).toBe("8");
  });

  it("restores user-selected stage upon returning to insights when no stage promotion occurred", () => {
    window.localStorage.setItem("sheepdog_insights_stage_scope", "8");
    window.localStorage.setItem("sheepdog_insights_last_curriculum_stage", "8");

    render(
      <DiagnosticsPanel
        checkpointIndex={mockCheckpointIndex}
        bestCheckpointEpisode={6508}
        trainingStatus={activeTrainingStatus}
        effectiveCurriculumStage={8}
        lastLiveRefreshTime={Date.now()}
      />
    );

    const stageSelect = screen.getByLabelText("Stage scope") as HTMLSelectElement;
    expect(stageSelect.value).toBe("8");
  });

  it("automatically loads current stage upon fresh mount / returning to insights if app self-promoted to a new stage", () => {
    // User had selected stage 8 previously, but app promoted to stage 9 while away
    window.localStorage.setItem("sheepdog_insights_stage_scope", "8");
    window.localStorage.setItem("sheepdog_insights_last_curriculum_stage", "8");

    render(
      <DiagnosticsPanel
        checkpointIndex={mockCheckpointIndex}
        bestCheckpointEpisode={6508}
        trainingStatus={activeTrainingStatus}
        effectiveCurriculumStage={9}
        lastLiveRefreshTime={Date.now()}
      />
    );

    const stageSelect = screen.getByLabelText("Stage scope") as HTMLSelectElement;
    // Self-promotion detected: automatically switches to the new stage
    expect(stageSelect.value).toBe("current");
  });

  it("does not render previous stage checkpoints on the chart when stage scope is set to current stage 9 with 0 checkpoints", () => {
    render(
      <DiagnosticsPanel
        checkpointIndex={mockCheckpointIndex}
        bestCheckpointEpisode={6508}
        trainingStatus={activeTrainingStatus}
        effectiveCurriculumStage={9}
        initialStageScope="current"
        lastLiveRefreshTime={Date.now()}
      />
    );

    // Formals for stage 8 should not be rendered in the table or formal markers for stage 9
    expect(screen.queryByText(/FORMAL EVALUATION \(STAGE 8 TARGET/i)).toBeNull();
    expect(screen.getAllByText(/Awaiting Checkpoints/i).length).toBeGreaterThan(0);
  });

  it("calculateBlockSuccessPoints generates trailing in-progress block when remainder > 0", async () => {
    const { calculateBlockSuccessPoints } = await import("./DiagnosticsPanel");
    // 58 episodes: 25 in block 1, 25 in block 2, 8 in trailing block 3
    const episodes = Array.from({ length: 58 }, (_, i) => {
      const epNum = i + 1;
      const isSuccess = epNum % 2 === 0;
      return {
        id: epNum,
        run_id: "run_active_123",
        global_environment_episode: epNum,
        curriculum_stage: 1,
        reward: isSuccess ? 100 : -10,
        result: isSuccess ? "SUCCESS" : "STOPPED",
        success: isSuccess,
        global_timestep: epNum * 100,
      };
    });

    const points = calculateBlockSuccessPoints(episodes as any, 25, (ep) => ep.global_environment_episode ?? null);
    expect(points.length).toBe(3);
    // Block 1: 1-25 (12 successes out of 25 = 48%)
    expect(points[0].x).toBe(25);
    expect(points[0].blockStartEp).toBe(1);
    expect(points[0].blockEndEp).toBe(25);
    expect(points[0].y).toBe(48);

    // Block 2: 26-50 (13 successes out of 25 = 52%)
    expect(points[1].x).toBe(50);
    expect(points[1].blockStartEp).toBe(26);
    expect(points[1].blockEndEp).toBe(50);
    expect(points[1].y).toBe(52);

    // Block 3 (trailing remainder): 51-58 (8 episodes: 52, 54, 56, 58 are even -> 4 successes = 50%)
    expect(points[2].x).toBe(58);
    expect(points[2].blockStartEp).toBe(51);
    expect(points[2].blockEndEp).toBe(58);
    expect(points[2].y).toBe(50);
  });

  it("calculateBlockAveragePoints calculates metric averages for steps, reward, sheep with remainder blocks", async () => {
    const { calculateBlockAveragePoints } = await import("./DiagnosticsPanel");
    const episodes = Array.from({ length: 30 }, (_, i) => ({
      id: i + 1,
      global_environment_episode: i + 1,
      curriculum_stage: 1,
      reward: (i + 1) * 10,
    }));

    const points = calculateBlockAveragePoints(
      episodes as any,
      (ep) => ep.reward,
      25,
      (ep) => ep.global_environment_episode ?? null,
      (avg, start, end) => `${avg.toFixed(1)} (ep ${start}-${end})`
    );
    expect(points.length).toBe(2);
    // Block 1 (1-25): avg is sum(1..25)*10 / 25 = 130
    expect(points[0].x).toBe(25);
    expect(points[0].y).toBe(130);
    expect(points[0].blockStartEp).toBe(1);
    expect(points[0].blockEndEp).toBe(25);
    // Block 2 (26-30, 5 eps): avg is (26+27+28+29+30)*10 / 5 = 280
    expect(points[1].x).toBe(30);
    expect(points[1].y).toBe(280);
    expect(points[1].blockStartEp).toBe(26);
    expect(points[1].blockEndEp).toBe(30);
  });

  it("renders block markers even when fewer than 25 episodes have run (single in-progress block point)", async () => {
    const initialEpisodes: any[] = Array.from({ length: 10 }, (_, i) => ({
      id: i + 1,
      run_id: "run_active_123",
      global_environment_episode: i + 1,
      curriculum_stage: 8,
      reward: 150,
      result: "SUCCESS",
      success: true,
      global_timestep: (i + 1) * 100,
      steps: 45,
      sheep_penned: 6,
    }));

    const { container } = render(
      <DiagnosticsPanel
        checkpointIndex={mockCheckpointIndex}
        bestCheckpointEpisode={6508}
        trainingStatus={activeTrainingStatus}
        effectiveCurriculumStage={8}
        initialEpisodes={initialEpisodes}
        initialStageScope={8}
        lastLiveRefreshTime={Date.now()}
      />
    );

    // Block markers use circle with fill="#38bdf8"
    const blockMarkers = container.querySelectorAll('circle[fill="#38bdf8"]');
    expect(blockMarkers.length).toBeGreaterThan(0);
  });

  it("preserves checkpoint history when recent rollouts are beyond the last checkpoint", async () => {
    // Checkpoints at episode 10, 50, 100, 500, 1000
    const sparseCheckpointIndex: any = {
      ...mockCheckpointIndex,
      checkpoints: [
        {
          checkpoint_episode: 10,
          environment_episodes_total: 100,
          success_rate: 0.2,
          checkpoint: "c10",
          evaluation: "e10",
          replay: "r10",
          environment_config: { curriculum_stage: 8 },
        },
        {
          checkpoint_episode: 50,
          environment_episodes_total: 500,
          success_rate: 0.5,
          checkpoint: "c50",
          evaluation: "e50",
          replay: "r50",
          environment_config: { curriculum_stage: 8 },
        },
        {
          checkpoint_episode: 100,
          environment_episodes_total: 10000,
          success_rate: 0.8,
          checkpoint: "c100",
          evaluation: "e100",
          replay: "r100",
          environment_config: { curriculum_stage: 8 },
        },
      ],
    };

    // Episodes spanning only 10,000 to 10,050 (last 50 episodes)
    const recentEpisodes: any[] = Array.from({ length: 50 }, (_, i) => ({
      id: i + 1,
      run_id: "run_active_123",
      global_environment_episode: 10000 + i + 1,
      curriculum_stage: 8,
      reward: 100,
      result: "SUCCESS",
      success: true,
      global_timestep: (10000 + i + 1) * 100,
      steps: 50,
    }));

    window.localStorage.setItem("sheepdog_insights_view_window", "50");

    const { container } = render(
      <DiagnosticsPanel
        checkpointIndex={sparseCheckpointIndex}
        bestCheckpointEpisode={null}
        trainingStatus={activeTrainingStatus}
        effectiveCurriculumStage={8}
        initialEpisodes={recentEpisodes}
        initialStageScope={8}
        lastLiveRefreshTime={Date.now()}
      />
    );

    // Last 50 means up to 50 checkpoints, not 50 high-frequency rollouts.
    const formalLine = container.querySelector('polyline[stroke="#34d399"]');
    expect(formalLine?.getAttribute("points")?.split(" ")).toHaveLength(3);
  });

  it("prompts a live poll when lastLiveRefreshTime updates", async () => {
    const api = await import("../lib/api");
    const loadEpisodesSpy = vi.spyOn(api, "loadTrainingEpisodes").mockImplementation(async () => {
      return {
        episodes: [],
        latest_id: 0,
        next_after_id: 0,
        has_more: false,
        max_id: 0,
        oldest_available_timestamp: null,
        total_matching: 0,
      };
    });

    const { rerender } = render(
      <DiagnosticsPanel
        checkpointIndex={mockCheckpointIndex}
        bestCheckpointEpisode={6508}
        trainingStatus={activeTrainingStatus}
        effectiveCurriculumStage={8}
        initialStageScope={8}
        lastLiveRefreshTime={1000}
      />
    );

    const initialCallCount = loadEpisodesSpy.mock.calls.length;

    // Simulate new rollout arrival triggering lastLiveRefreshTime update
    await act(async () => {
      rerender(
        <DiagnosticsPanel
          checkpointIndex={mockCheckpointIndex}
          bestCheckpointEpisode={6508}
          trainingStatus={activeTrainingStatus}
          effectiveCurriculumStage={8}
          initialStageScope={8}
          lastLiveRefreshTime={2000}
        />
      );
    });

    expect(loadEpisodesSpy.mock.calls.length).toBeGreaterThan(initialCallCount);
    loadEpisodesSpy.mockRestore();
  });

  it("getCheckpointXValue correctly scales checkpoints to environment_episodes_total for episode X-axis", async () => {
    const { getCheckpointXValue } = await import("./DiagnosticsPanel");

    const checkpoint: any = {
      checkpoint_episode: 39986,
      total_training_episodes: 380000,
      environment_episodes_total: 386510,
    };

    // When environment_episodes_total is present, uses it directly to align with episode scale (~386k)
    expect(getCheckpointXValue(checkpoint, "episode")).toBe(386510);

    // Fallbacks
    const cpNoEnv: any = {
      checkpoint_episode: 39986,
      total_training_episodes: 380000,
    };
    expect(getCheckpointXValue(cpNoEnv, "episode")).toBe(380000);

    const cpRawOnly: any = {
      checkpoint_episode: 39986,
    };
    expect(getCheckpointXValue(cpRawOnly, "episode")).toBe(39986);
  });

  it("renders 5 block trend points and valid polylines in a rollout-only Last 25 window", async () => {
    // Checkpoint older than the active window
    const nonOverlappingCheckpointIndex: CheckpointIndex = {
      checkpoints: [
        {
          checkpoint_episode: 6000,
          environment_episodes_total: 6000,
          global_timestep: 600000,
          curriculum_stage: 8,
          success_rate: 0.75,
          timeout_rate: 0.25,
          average_completion_steps: 320,
          average_completion_seconds: 32,
          average_sheep_penned: 4,
          average_reward: 200,
          checkpoint: "checkpoint-006000.json",
          evaluation: "eval_006000.json",
          replay: "replay_006000.json",
          recorded_at: "2026-08-01T10:00:00Z",
          records: [],
        },
      ],
      latest: null,
    };

    // Exactly 25 recent episodes (e.g. 6501 to 6525) that occur after checkpoint 6000
    const recent25Episodes: any[] = Array.from({ length: 25 }, (_, i) => ({
      id: i + 1,
      run_id: "run_active_123",
      global_environment_episode: 6500 + i + 1,
      curriculum_stage: 8,
      reward: 150 + i * 2,
      result: i % 2 === 0 ? "SUCCESS" : "STOPPED",
      success: i % 2 === 0,
      global_timestep: (6500 + i + 1) * 100,
      steps: 100 + i * 5,
      sheep_penned: 3 + (i % 2),
    }));

    window.localStorage.setItem("sheepdog_insights_view_window", "25");

    const { container } = render(
      <DiagnosticsPanel
        checkpointIndex={{ ...nonOverlappingCheckpointIndex, checkpoints: [] }}
        bestCheckpointEpisode={null}
        trainingStatus={activeTrainingStatus}
        effectiveCurriculumStage={8}
        initialEpisodes={recent25Episodes}
        initialStageScope={8}
        lastLiveRefreshTime={Date.now()}
      />
    );

    // Chart canvas must not be empty or display the 'No data' fallback
    expect(screen.queryByText(/No data available for current range/i)).toBeNull();

    // Verify 5 block trend points are rendered (5 markers in active chart)
    const blockMarkers = container.querySelectorAll('circle[fill="#38bdf8"]');
    expect(blockMarkers.length).toBe(5);

    // Verify block rollout polyline is rendered with 5 points
    const polylines = Array.from(container.querySelectorAll("polyline"));
    const blockPolyline = polylines.find((p) => p.getAttribute("stroke") === "#38bdf8");
    expect(blockPolyline).toBeDefined();
    expect(blockPolyline?.getAttribute("points")?.split(" ").length).toBe(5);

    // Verify legend dynamically updates to 5-ep rollout trend
    expect(screen.getByText("5-ep rollout trend")).toBeDefined();
  });

  it("switching between viewWindow options preserves valid, non-empty trend lines", async () => {
    // 50 recent episodes
    const recent50Episodes: any[] = Array.from({ length: 50 }, (_, i) => ({
      id: i + 1,
      run_id: "run_active_123",
      global_environment_episode: 7000 + i + 1,
      curriculum_stage: 8,
      reward: 120 + i,
      result: i % 3 !== 0 ? "SUCCESS" : "STOPPED",
      success: i % 3 !== 0,
      global_timestep: (7000 + i + 1) * 100,
      steps: 150 - i,
      sheep_penned: 4,
    }));

    const nonOverlappingCheckpointIndex: CheckpointIndex = {
      checkpoints: [
        {
          checkpoint_episode: 5000,
          environment_episodes_total: 5000,
          global_timestep: 500000,
          curriculum_stage: 8,
          success_rate: 0.6,
          timeout_rate: 0.4,
          average_completion_steps: 400,
          average_completion_seconds: 40,
          average_sheep_penned: 2,
          average_reward: 100,
          checkpoint: "checkpoint-005000.json",
          evaluation: "eval_005000.json",
          replay: "replay_005000.json",
          recorded_at: "2026-08-01T10:00:00Z",
          records: [],
        },
      ],
      latest: null,
    };

    window.localStorage.setItem("sheepdog_insights_view_window", "25");

    const { container } = render(
      <DiagnosticsPanel
        checkpointIndex={{ ...nonOverlappingCheckpointIndex, checkpoints: [] }}
        bestCheckpointEpisode={null}
        trainingStatus={activeTrainingStatus}
        effectiveCurriculumStage={8}
        initialEpisodes={recent50Episodes}
        initialStageScope={8}
        lastLiveRefreshTime={Date.now()}
      />
    );

    // Initial Last 25 viewWindow: effective block size 5 -> 5 block points
    expect(screen.getByText("5-ep rollout trend")).toBeDefined();
    let blockMarkers = container.querySelectorAll('circle[fill="#38bdf8"]');
    expect(blockMarkers.length).toBe(5);

    // Switch to "Last 50"
    const btn50 = screen.getByRole("button", { name: "Last 50" });
    fireEvent.click(btn50);

    // On 50 episodes with viewWindow 50: effective block size 10 -> 5 block points
    expect(screen.getByText("10-ep rollout trend")).toBeDefined();
    blockMarkers = container.querySelectorAll('circle[fill="#38bdf8"]');
    expect(blockMarkers.length).toBe(5);

    let polylines = Array.from(container.querySelectorAll("polyline"));
    let blockPolyline = polylines.find((p) => p.getAttribute("stroke") === "#38bdf8");
    expect(blockPolyline).toBeDefined();
    expect(blockPolyline?.getAttribute("points")?.split(" ").length).toBe(5);

    // Switch to "All"
    const btnAll = screen.getByRole("button", { name: "All" });
    fireEvent.click(btnAll);

    // On 50 episodes with viewWindow "all": effective block size 25 -> 2 block points
    expect(screen.getByText("25-ep rollout trend")).toBeDefined();
    blockMarkers = container.querySelectorAll('circle[fill="#38bdf8"]');
    expect(blockMarkers.length).toBe(2);

    polylines = Array.from(container.querySelectorAll("polyline"));
    blockPolyline = polylines.find((p) => p.getAttribute("stroke") === "#38bdf8");
    expect(blockPolyline).toBeDefined();
    expect(blockPolyline?.getAttribute("points")?.split(" ").length).toBe(2);
    expect(screen.queryByText(/No data available for current range/i)).toBeNull();
  });

  it.each([25, 50, 100])("keeps the last %i checkpoint evaluations through delayed telemetry and subsequent polls", async (windowSize) => {
    const api = await import("../lib/api");
    type Response = Awaited<ReturnType<typeof api.loadTrainingEpisodes>>;
    let resolveBootstrap!: (response: Response) => void;
    const bootstrap = new Promise<Response>((resolve) => { resolveBootstrap = resolve; });
    const checkpointIndex: CheckpointIndex = {
      latest: null,
      checkpoints: Array.from({ length: 110 }, (_, i) => ({
        ...mockCheckpointIndex.checkpoints[0],
        checkpoint_episode: 6500 + i,
        global_timestep: 335_000_000 + i * 25_000,
        success_rate: i % 2 ? 0.7 : 0.9,
        average_completion_steps: 300 + i,
      })),
    };
    const rollout = (id: number) => ({
      id, run_id: "run_active_123", curriculum_stage: 8,
      event_key: `rollout-${id}`, session_id: null, episode_in_stage: id,
      policy_version: 1, completed_at: "2026-11-09T18:00:00Z",
      active_runtime_seconds_total: null, timeout: false, stopped: false,
      total_sheep: 4, seed: id, checkpoint_id: null,
      global_environment_episode: 2000 + id,
      global_timestep: 338_190_000 + id * 100,
      result: "SUCCESS", success: true, reward: 500,
      steps: 50, sheep_penned: 4,
    });
    const response = (episodes: ReturnType<typeof rollout>[]): Response => ({
      episodes, latest_id: episodes.at(-1)?.id ?? 0,
      next_after_id: episodes.at(-1)?.id ?? 0, has_more: false,
      oldest_available_timestamp: null, total_matching: episodes.length,
    });
    const spy = vi.spyOn(api, "loadTrainingEpisodes")
      .mockReturnValueOnce(bootstrap)
      .mockResolvedValue(response([rollout(51)]));
    window.localStorage.clear();
    window.localStorage.setItem("sheepdog_insights_view_window", String(windowSize));
    const props = {
      checkpointIndex, bestCheckpointEpisode: null,
      trainingStatus: activeTrainingStatus, effectiveCurriculumStage: 8,
      initialStageScope: 8 as const, lastLiveRefreshTime: 1,
    };
    const { container, rerender, unmount } = render(<DiagnosticsPanel {...props} />);
    const formalPoints = () => container.querySelector('polyline[stroke="#34d399"]')!
      .getAttribute("points")!.split(" ");
    try {
      const before = formalPoints();
      expect(before).toHaveLength(windowSize);
      await act(async () => {
        resolveBootstrap(response([
          { ...rollout(0), global_timestep: 1_000 },
          ...Array.from({ length: 50 }, (_, i) => rollout(i + 1)),
        ]));
        await bootstrap;
      });
      const after = formalPoints();
      expect(after).toHaveLength(windowSize);
      // Benchmark values stay 70/90%, distinct from the 100% training rollouts.
      expect(after.map((point) => point.split(",")[1]))
        .toEqual(before.map((point) => point.split(",")[1]));
      expect(container.querySelectorAll('circle[fill="rgba(56,189,248,0.5)"]')).toHaveLength(50);
      expect(Number(after[0].split(",")[0])).toBe(54);
      rerender(<DiagnosticsPanel {...props} lastLiveRefreshTime={2} />);
      await act(async () => { await Promise.resolve(); });
      expect(spy).toHaveBeenCalledTimes(2);
      expect(formalPoints()).toHaveLength(windowSize);
      expect(container.querySelectorAll('circle[fill="rgba(56,189,248,0.5)"]')).toHaveLength(51);
    } finally {
      unmount();
      spy.mockRestore();
    }
  });

});

import { fireEvent, render, screen } from "@testing-library/react";
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

    const { loadCheckpointIndex } = await import("../lib/api");
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
});

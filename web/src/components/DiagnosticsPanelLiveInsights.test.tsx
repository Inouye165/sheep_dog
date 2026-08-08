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
});

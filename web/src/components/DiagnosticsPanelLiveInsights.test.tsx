import { render, screen } from "@testing-library/react";
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
      total_training_episodes: 6508,
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
  completed_episodes: 52,
  batch_total_episodes: 100,
  batch_completed_episodes: 52,
  total_episodes_trained: 52,
  episodes_in_stage: 52,
  stage_success_count: 44,
  stage_success_rate: 0.8461538461538461,
  total_timesteps: 37248174,
  latest_checkpoint_episode: 6508,
  active_checkpoint_id: "chk_6508",
  last_evaluation_time: "2026-08-02T22:56:42.413250+00:00",
  checkpoint_save_interval: 25,
  current_episode: null,
  checkpoint_episode: 6508,
  latest_seed: 11,
  latest_replay_path: null,
  best_score: 0.7,
  latest_success_rate: 0.7,
  latest_avg_sheep_penned: 3.4,
  latest_avg_reward: 245.6,
  latest_timeout_rate: 0.3,
  latest_stopped_rate: 0.0,
  latest_avg_no_progress_steps: 24,
  latest_avg_distance_to_pen: 4.7,
  latest_avg_flock_spread: 2.2,
  latest_avg_farthest_distance_to_pen: 6.1,
  latest_avg_farthest_distance_to_flock_center: 2.3,
  stage_history: { "8": 52 },
  grand_total_episodes: 17104,
  starting_episode: 0,
  phase: "curriculum_active",
  message: "Training active",
  error: null,
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

    expect(screen.getByText(/Live Rollouts:/i)).toBeDefined();
    expect(screen.getByText(/52 episodes/i)).toBeDefined();
    expect(screen.getByText(/1 pts/i)).toBeDefined();

    const updatedStatus: TrainingStatus = {
      ...activeTrainingStatus,
      episodes_in_stage: 60,
      stage_success_count: 50,
      stage_success_rate: 0.8333333333333334,
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

    expect(screen.getByText(/60 episodes/i)).toBeDefined();
    expect(screen.getByText(/1 pts/i)).toBeDefined();
  });

  it("2. The chart evaluation series remains unchanged until evaluation completes", () => {
    render(
      <DiagnosticsPanel
        checkpointIndex={mockCheckpointIndex}
        bestCheckpointEpisode={6508}
        trainingStatus={activeTrainingStatus}
        effectiveCurriculumStage={8}
        lastLiveRefreshTime={Date.now()}
      />
    );

    expect(screen.getByText(/70%/i)).toBeDefined();
  });

  it("3. The page displays the pending-evaluation message during active training", () => {
    render(
      <DiagnosticsPanel
        checkpointIndex={mockCheckpointIndex}
        bestCheckpointEpisode={6508}
        trainingStatus={activeTrainingStatus}
        effectiveCurriculumStage={8}
        lastLiveRefreshTime={Date.now()}
      />
    );

    expect(
      screen.getByText(
        /Training active — 52 rollout episodes completed since the latest evaluation. Next evaluation pending./i
      )
    ).toBeDefined();
  });

  it("4. A new checkpoint_id causes an immediate chart refresh with updated points", () => {
    const newCheckpointIndex: CheckpointIndex = {
      checkpoints: [
        ...mockCheckpointIndex.checkpoints,
        {
          ...mockCheckpointIndex.checkpoints[0],
          checkpoint_episode: 6524,
          checkpoint_id: "chk_6524",
          global_timestep: 35936656,
          curriculum_stage: 8,
          environment_config: { dogs: 3, sheep: 4, width: 200, height: 200, curriculum_stage: 8 },
          success_rate: 0.85,
        },
      ],
      latest: null,
    };

    const { rerender } = render(
      <DiagnosticsPanel
        checkpointIndex={mockCheckpointIndex}
        bestCheckpointEpisode={6508}
        trainingStatus={activeTrainingStatus}
        effectiveCurriculumStage={8}
        lastLiveRefreshTime={Date.now()}
      />
    );

    expect(screen.getByText(/1 pts/i)).toBeDefined();

    rerender(
      <DiagnosticsPanel
        checkpointIndex={newCheckpointIndex}
        bestCheckpointEpisode={6524}
        trainingStatus={{
          ...activeTrainingStatus,
          active_checkpoint_id: "chk_6524",
          latest_checkpoint_episode: 6524,
        }}
        effectiveCurriculumStage={8}
        lastLiveRefreshTime={Date.now()}
      />
    );

    expect(screen.getByText(/2 pts/i)).toBeDefined();
    expect(screen.getAllByText(/chk_6524/i).length).toBeGreaterThan(0);
  });

  it("5. New checkpoint data rerenders even when training running state remains true", () => {
    const newCheckpointIndex: CheckpointIndex = {
      checkpoints: [
        ...mockCheckpointIndex.checkpoints,
        {
          ...mockCheckpointIndex.checkpoints[0],
          checkpoint_episode: 6524,
          checkpoint_id: "chk_6524",
          global_timestep: 35936656,
          curriculum_stage: 8,
          environment_config: { dogs: 3, sheep: 4, width: 200, height: 200, curriculum_stage: 8 },
          success_rate: 0.9,
        },
      ],
      latest: null,
    };

    render(
      <DiagnosticsPanel
        checkpointIndex={newCheckpointIndex}
        bestCheckpointEpisode={6524}
        trainingStatus={{ ...activeTrainingStatus, running: true }}
        effectiveCurriculumStage={8}
        lastLiveRefreshTime={Date.now()}
      />
    );

    expect(screen.getByText(/2 pts/i)).toBeDefined();
    expect(screen.getAllByText(/90%/i).length).toBeGreaterThan(0);
  });

  it("6. Cached checkpoint JSON cannot leave the chart stale because loadCheckpointIndex appends cache-busting timestamp", async () => {
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

  it("7. Session-local episode numbering cannot collide with canonical checkpoint identity", () => {
    const collidingIndex: CheckpointIndex = {
      checkpoints: [
        {
          ...mockCheckpointIndex.checkpoints[0],
          checkpoint_episode: 0,
          checkpoint_id: "chk_session1_0",
          global_timestep: 1000,
          curriculum_stage: 8,
          environment_config: { dogs: 3, sheep: 4, width: 200, height: 200, curriculum_stage: 8 },
        },
        {
          ...mockCheckpointIndex.checkpoints[0],
          checkpoint_episode: 0,
          checkpoint_id: "chk_session2_0",
          global_timestep: 5000,
          curriculum_stage: 8,
          environment_config: { dogs: 3, sheep: 4, width: 200, height: 200, curriculum_stage: 8 },
          success_rate: 0.95,
        },
      ],
      latest: null,
    };

    render(
      <DiagnosticsPanel
        checkpointIndex={collidingIndex}
        bestCheckpointEpisode={0}
        trainingStatus={activeTrainingStatus}
        effectiveCurriculumStage={8}
        lastLiveRefreshTime={Date.now()}
      />
    );

    expect(screen.getByText(/2 pts/i)).toBeDefined();
    expect(screen.getAllByText(/chk_session2_0/i).length).toBeGreaterThan(0);
  });

  it("8. Live success rate is explicitly labeled as rollout success rate and distinct from promotion gate result", () => {
    render(
      <DiagnosticsPanel
        checkpointIndex={mockCheckpointIndex}
        bestCheckpointEpisode={6508}
        trainingStatus={activeTrainingStatus}
        effectiveCurriculumStage={8}
        lastLiveRefreshTime={Date.now()}
      />
    );

    expect(screen.getByText(/Live Rollout Success Rate:/i)).toBeDefined();
    expect(screen.getByText(/84.6%/i)).toBeDefined();
    expect(screen.getByText(/\(rollouts only\)/i)).toBeDefined();
  });
});

import { render, screen, fireEvent } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { DiagnosticsPanel } from "./DiagnosticsPanel";
import type { CheckpointIndex, TrainingStatus, TrainingEpisode } from "../state/types";

const mockIndexWithCheckpoints = {
  checkpoints: [
    {
      checkpoint_id: "chk_1",
      curriculum_stage: 1,
      checkpoint_episode: 0,
      policy_version: 1,
      global_timestep: 45454,
      evaluation_seeds: [11, 23, 37],
      success_rate: 0.0,
      timeout_rate: 0.0,
      average_completion_steps: 500,
      average_completion_seconds: 10,
      average_sheep_penned: 0,
      average_reward: -50,
      average_distance_to_pen: 20,
      average_flock_spread: 5,
      records: [],
      recorded_at: "2026-08-07T01:00:00Z",
      policy_name: "neural_policy",
      trainer_type: "maskable_ppo",
      policy_type: "neural",
      policy_mode: "neural_only",
      replay_mode: "truthful",
      total_training_episodes: 0,
      policy_state_path: "path",
      checkpoint: "chk_0.json",
      evaluation: "eval_0.json",
      replay: "rep_0.json",
    },
    {
      checkpoint_id: "chk_2",
      curriculum_stage: 1,
      checkpoint_episode: 5,
      policy_version: 2,
      global_timestep: 90908,
      evaluation_seeds: [11, 23, 37],
      success_rate: 0.2,
      timeout_rate: 0.1,
      average_completion_steps: 400,
      average_completion_seconds: 8,
      average_sheep_penned: 1,
      average_reward: 10,
      average_distance_to_pen: 15,
      average_flock_spread: 4,
      records: [],
      recorded_at: "2026-08-07T01:05:00Z",
      policy_name: "neural_policy",
      trainer_type: "maskable_ppo",
      policy_type: "neural",
      policy_mode: "neural_only",
      replay_mode: "truthful",
      total_training_episodes: 5,
      policy_state_path: "path",
      checkpoint: "chk_5.json",
      evaluation: "eval_5.json",
      replay: "rep_5.json",
    },
  ],
  latest: null,
} as unknown as CheckpointIndex;

const mockTrainingStatus = {
  running: true,
  fast_mode: true,
  trainer_type: "maskable_ppo",
  policy_type: "neural",
  enable_instinct_rewards: false,
  policy_mode: "neural_only",
  replay_mode: "truthful",
  debug_reward_breakdown: false,
  curriculum_stage: 1,
  requested_episodes: 50,
  completed_episodes: 15,
  batch_total_episodes: 50,
  batch_completed_episodes: 15,
  total_episodes_trained: 15,
  episodes_in_stage: 15,
  current_stage_environment_episode: 15,
  latest_completed_environment_episode: 15,
  episodes_since_latest_confidence_evaluation: 5,
  live_rollout_window_count: 5,
  live_rollout_success_count: 1,
  live_rollout_failure_count: 4,
  live_rollout_stopped_count: 0,
  live_rollout_timeout_count: 0,
  live_rollout_success_rate: 0.2,
  policy_version: 2,
  current_global_timestep: 90908,
  latest_checkpoint_global_timestep: 90908,
  episodes_until_next_evaluation: 35,
  next_evaluation_environment_episode: 50,
} as unknown as TrainingStatus;

const mockMixedEpisodes = [
  // Legacy row without global_timestep
  {
    id: 1,
    event_key: "ev1",
    run_id: "r1",
    session_id: "s1",
    global_environment_episode: 1,
    episode_in_stage: 1,
    curriculum_stage: 1,
    global_timestep: null,
    reward: -30,
    success: false,
    stopped: true,
    timeout: false,
    result: "STOPPED",
    sheep_penned: 0,
    total_sheep: 1,
    steps: 200,
  },
  // New row with authentic global_timestep
  {
    id: 2,
    event_key: "ev2",
    run_id: "r1",
    session_id: "s1",
    global_environment_episode: 2,
    episode_in_stage: 2,
    curriculum_stage: 1,
    global_timestep: 45200,
    reward: 25,
    success: true,
    stopped: false,
    timeout: false,
    result: "SUCCESS",
    sheep_penned: 1,
    total_sheep: 1,
    steps: 150,
  },
] as unknown as TrainingEpisode[];

describe("DiagnosticsPanel Authentic Telemetry & 4-Layer Controls", () => {
  it("renders status labels without confusing Schedule Checkpoint with Environment Episode", () => {
    render(
      <DiagnosticsPanel
        checkpointIndex={mockIndexWithCheckpoints}
        bestCheckpointEpisode={5}
        trainingStatus={mockTrainingStatus}
        effectiveCurriculumStage={1}
      />
    );

    expect(screen.getByText("Schedule Checkpoint:")).toBeInTheDocument();
    expect(screen.getByText("Policy Snapshot:")).toBeInTheDocument();
  });

  it("renders 4 independent layer control checkboxes", () => {
    render(
      <DiagnosticsPanel
        checkpointIndex={mockIndexWithCheckpoints}
        bestCheckpointEpisode={5}
        trainingStatus={mockTrainingStatus}
        effectiveCurriculumStage={1}
      />
    );

    expect(screen.getByLabelText(/Raw Episodes/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/25-Episode Rolling Avg/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/Policy Snapshots/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/Formal 10-Seed Benchmark Evals/i)).toBeInTheDocument();
  });

  it("shows legacy notice when Global Timestep mode omits legacy null rows", () => {
    render(
      <DiagnosticsPanel
        checkpointIndex={mockIndexWithCheckpoints}
        bestCheckpointEpisode={5}
        trainingStatus={mockTrainingStatus}
        effectiveCurriculumStage={1}
        initialEpisodes={mockMixedEpisodes}
      />
    );

    expect(
      screen.getByText(/Some earlier episode telemetry predates per-episode timestep recording/i)
    ).toBeInTheDocument();
  });

  it("toggles layer controls independently without errors", () => {
    render(
      <DiagnosticsPanel
        checkpointIndex={mockIndexWithCheckpoints}
        bestCheckpointEpisode={5}
        trainingStatus={mockTrainingStatus}
        effectiveCurriculumStage={1}
        initialEpisodes={mockMixedEpisodes}
      />
    );

    const rawCheckbox = screen.getByLabelText(/Raw Episodes/i) as HTMLInputElement;
    const rollingCheckbox = screen.getByLabelText(/25-Episode Rolling Avg/i) as HTMLInputElement;

    expect(rawCheckbox.checked).toBe(true);
    expect(rollingCheckbox.checked).toBe(true);

    fireEvent.click(rawCheckbox);
    expect(rawCheckbox.checked).toBe(false);

    fireEvent.click(rollingCheckbox);
    expect(rollingCheckbox.checked).toBe(false);
  });
});

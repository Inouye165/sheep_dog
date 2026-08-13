import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import {
  DiagnosticsPanel,
  calculateRawSuccessY,
  calculateRollingSuccess,
} from "./DiagnosticsPanel";
import type { CheckpointIndex, TrainingEpisode, TrainingStatus } from "../state/types";

const mockIndexWithCheckpoint: CheckpointIndex = {
  checkpoints: [
    {
      checkpoint_episode: 100,
      recorded_at: "2026-08-06T20:00:00Z",
      checkpoint: "chk_100.pt",
      evaluation: "eval_100.json",
      replay: "replay_100.json",
      success_rate: 1.0,
      timeout_rate: 0.0,
      average_completion_steps: 45,
      average_completion_seconds: 12.3,
      average_sheep_penned: 3,
      average_reward: 250.0,
      records: [],
      checkpoint_id: "chk_100",
      global_timestep: 10000,
      curriculum_stage: 1,
    },
  ],
  latest: null,
};

const mockTrainingStatus: any = {
  running: true,
  fast_mode: true,
  trainer_type: "maskable_ppo",
  policy_type: "neural",
  enable_instinct_rewards: false,
  policy_mode: "neural_only",
  replay_mode: "truthful",
  debug_reward_breakdown: false,
  curriculum_stage: 1,
  requested_episodes: 100,
  completed_episodes: 30,
  batch_total_episodes: 100,
  batch_completed_episodes: 30,
  total_episodes_trained: 30,
  episodes_in_stage: 30,
  current_stage_environment_episode: 30,
  latest_completed_environment_episode: 30,
  episodes_since_latest_confidence_evaluation: 30,
  live_rollout_window_count: 30,
};

describe("Insights Success Rate Chart Data Pipeline", () => {
  it("calculates 100% for 30 consecutive SUCCESS training episodes", () => {
    // Episodes 11895–11924 (30 consecutive SUCCESS results)
    const consecutiveSuccessEpisodes: Partial<TrainingEpisode>[] = Array.from({ length: 30 }, (_, i) => ({
      id: 11895 + i,
      result: "SUCCESS",
      success: true,
      reward: 250.0,
      steps: 45,
    }));

    // Verify each raw point equals 100
    for (const ep of consecutiveSuccessEpisodes) {
      expect(calculateRawSuccessY(ep)).toBe(100);
    }

    // Verify 25-episode rolling average equals exactly 100.0%
    const rollingAvg = calculateRollingSuccess(consecutiveSuccessEpisodes, 25);
    expect(rollingAvg).toBe(100);
  });

  it("handles mixed success/failure and missing fields cleanly", () => {
    const mixedEpisodes: Partial<TrainingEpisode>[] = [
      ...Array.from({ length: 10 }, (_, i) => ({
        id: 100 + i,
        result: "SUCCESS",
        success: true,
      })),
      ...Array.from({ length: 15 }, (_, i) => ({
        id: 110 + i,
        result: "TIMEOUT",
        success: false,
      })),
    ];

    // 10 successes, 15 failures -> 10 * 100 / 25 = 40%
    const rollingAvg = calculateRollingSuccess(mixedEpisodes, 25);
    expect(rollingAvg).toBe(40);

    // Missing/undefined fields return null instead of converting to zero
    expect(calculateRawSuccessY({})).toBeNull();
    expect(calculateRawSuccessY({ result: "" })).toBeNull();
  });

  it("renders DiagnosticsPanel component without crash", () => {
    const episodes: TrainingEpisode[] = Array.from({ length: 30 }, (_, i) => ({
      id: 100 + i,
      event_key: `ep_${100 + i}`,
      run_id: "run_test",
      session_id: "sess_test",
      global_environment_episode: 100 + i,
      episode_in_stage: 100 + i,
      curriculum_stage: 1,
      global_timestep: (100 + i) * 100,
      policy_version: 1,
      completed_at: "2026-08-06T20:00:00Z",
      active_runtime_seconds_total: 100,
      reward: 250.0,
      result: "SUCCESS",
      success: true,
      timeout: false,
      stopped: false,
      sheep_penned: 3,
      total_sheep: 3,
      steps: 45,
      seed: 42,
      checkpoint_id: "chk_1",
    }));

    render(
      <DiagnosticsPanel
        checkpointIndex={mockIndexWithCheckpoint}
        bestCheckpointEpisode={100}
        trainingStatus={mockTrainingStatus}
        effectiveCurriculumStage={1}
        initialEpisodes={episodes}
      />
    );

    expect(screen.getByText("Learning Curve")).toBeInTheDocument();
  });
});

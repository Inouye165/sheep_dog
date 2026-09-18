import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { StageProgressTelemetry, formatTimeSpan } from "./StageProgressTelemetry";
import type { TrainingStatus, TrainingEpisode } from "../state/types";

describe("StageProgressTelemetry", () => {
  it("formats durations accurately with formatTimeSpan", () => {
    expect(formatTimeSpan(0)).toBe("0s");
    expect(formatTimeSpan(45)).toBe("45s");
    expect(formatTimeSpan(125)).toBe("2m 05s");
    expect(formatTimeSpan(3600)).toBe("1h 00m");
    expect(formatTimeSpan(3665)).toBe("1h 01m");
    expect(formatTimeSpan(null)).toBe("—");
    expect(formatTimeSpan(-5)).toBe("—");
  });

  it("renders stage onboarding discovery milestone when awaiting checkpoints", () => {
    render(
      <StageProgressTelemetry
        isLiveTraining={true}
        curriculumStage={24}
        stageScopedCheckpointsCount={0}
        episodesSinceEvaluation={13}
        episodesUntilNextEvaluation={37}
        nextEvaluationBoundary={50}
        currentStageEp={13}
      />
    );

    expect(screen.getByRole("region", { name: /Stage Cadence Progress/i })).toBeInTheDocument();
    expect(
      screen.getByText(/Stage 24 Baseline Discovery: Progress to Checkpoint 1 \(13\/50 episodes\)/i)
    ).toBeInTheDocument();
    expect(screen.getByText("26%")).toBeInTheDocument();
    expect(screen.getByText(/● LIVE PACE/i)).toBeInTheDocument();

    // Progress bar ARIA
    const progressBar = screen.getByRole("progressbar");
    expect(progressBar).toHaveAttribute("aria-valuenow", "26");
  });

  it("calculates time completed, ETA, and estimated total based on recent episode duration", () => {
    const mockEpisodes: TrainingEpisode[] = [
      {
        id: 1,
        event_key: "ep-1",
        run_id: "run-1",
        session_id: "s-1",
        global_environment_episode: 100,
        episode_in_stage: 1,
        curriculum_stage: 24,
        global_timestep: 500,
        policy_version: 1,
        completed_at: "2026-09-14T12:00:00.000Z",
        active_runtime_seconds_total: 10,
        reward: 10,
        result: "SUCCESS",
        success: true,
        timeout: false,
        stopped: false,
        sheep_penned: 6,
        total_sheep: 6,
        steps: 100,
        seed: 42,
        checkpoint_id: null,
      },
      {
        id: 2,
        event_key: "ep-2",
        run_id: "run-1",
        session_id: "s-1",
        global_environment_episode: 101,
        episode_in_stage: 2,
        curriculum_stage: 24,
        global_timestep: 1000,
        policy_version: 1,
        // 20 seconds later for 1 episode step -> 20s/ep
        completed_at: "2026-09-14T12:00:20.000Z",
        active_runtime_seconds_total: 30,
        reward: 12,
        result: "SUCCESS",
        success: true,
        timeout: false,
        stopped: false,
        sheep_penned: 6,
        total_sheep: 6,
        steps: 110,
        seed: 43,
        checkpoint_id: null,
      },
    ];

    render(
      <StageProgressTelemetry
        isLiveTraining={true}
        curriculumStage={24}
        stageScopedCheckpointsCount={0}
        episodesSinceEvaluation={10}
        episodesUntilNextEvaluation={40}
        nextEvaluationBoundary={50}
        currentStageEp={10}
        recentEpisodes={mockEpisodes}
      />
    );

    // 10 done * 20s = 200s = 3m 20s
    expect(screen.getByTestId("time-completed-val")).toHaveTextContent("3m 20s");
    // 40 remaining * 20s = 800s = 13m 20s
    expect(screen.getByTestId("time-left-val")).toHaveTextContent("~13m 20s");
    // total = 1000s = 16m 40s
    expect(screen.getByTestId("estimated-total-val")).toHaveTextContent("16m 40s");
  });

  it("handles active formal benchmark evaluation in progress", () => {
    const status: TrainingStatus = {
      running: true,
      fast_mode: true,
      enable_instinct_rewards: true,
      debug_reward_breakdown: false,
      curriculum_stage: 24,
      requested_episodes: 50000,
      completed_episodes: 1200,
      batch_total_episodes: 50,
      batch_completed_episodes: 12,
      total_episodes_trained: 1200,
      stage_history: { "24": 1200 },
      grand_total_episodes: 1200,
      current_episode: 1200,
      checkpoint_episode: 1150,
      latest_checkpoint_episode: 1150,
      latest_seed: 107,
      latest_replay_path: null,
      best_score: 500,
      latest_success_rate: 0.9,
      latest_avg_sheep_penned: 6,
      latest_avg_reward: 500,
      latest_timeout_rate: 0,
      latest_stopped_rate: 0,
      latest_avg_no_progress_steps: 0,
      latest_avg_distance_to_pen: 0,
      latest_avg_flock_spread: 2,
      latest_avg_farthest_distance_to_pen: 5,
      latest_avg_farthest_distance_to_flock_center: 3,
      phase: "evaluation",
      message: "Evaluating seed 107 (3/10)",
      error: null,
      starting_episode: 1,
      evaluation_in_progress: true,
      evaluation_seed: 107,
      evaluation_seed_index: 3,
      evaluation_total_seeds: 10,
      evaluation_completed_seeds: 2,
    };

    render(
      <StageProgressTelemetry
        trainingStatus={status}
        isLiveTraining={true}
        curriculumStage={24}
        stageScopedCheckpointsCount={2}
      />
    );

    expect(
      screen.getByText(/Formal Benchmark In Progress: Seed 107 \(2\/10 seeds\)/i)
    ).toBeInTheDocument();
    expect(screen.getByText("20%")).toBeInTheDocument();
    expect(screen.getByText("2 seeds done")).toBeInTheDocument();
    expect(screen.getByText("8 seeds left")).toBeInTheDocument();
  });
});

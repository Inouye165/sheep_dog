import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { EvaluationBanner } from "./EvaluationBanner";
import type { TrainingStatus } from "../state/types";

describe("EvaluationBanner", () => {
  it("renders null when there is no evaluation in progress or recent results", () => {
    const status: TrainingStatus = {
      running: true,
      fast_mode: true,
      enable_instinct_rewards: false,
      debug_reward_breakdown: false,
      curriculum_stage: 1,
      requested_episodes: 100,
      completed_episodes: 10,
      batch_total_episodes: 100,
      batch_completed_episodes: 10,
      total_episodes_trained: 10,
      stage_history: {},
      grand_total_episodes: 10,
      current_episode: 10,
      checkpoint_episode: null,
      latest_checkpoint_episode: null,
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
      phase: "running",
      message: "Training",
      error: null,
      starting_episode: null,
    };

    const { container } = render(<EvaluationBanner status={status} />);
    expect(container.firstChild).toBeNull();
  });

  it("renders live evaluation set status, current seed, message, and seed chips", () => {
    const status: TrainingStatus = {
      running: true,
      fast_mode: true,
      enable_instinct_rewards: false,
      debug_reward_breakdown: false,
      curriculum_stage: 3,
      requested_episodes: 100,
      completed_episodes: 25,
      batch_total_episodes: 100,
      batch_completed_episodes: 25,
      total_episodes_trained: 25,
      stage_history: {},
      grand_total_episodes: 25,
      current_episode: 25,
      checkpoint_episode: 25,
      latest_checkpoint_episode: null,
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
      phase: "evaluation",
      message: "Evaluating seed 107 (3 of 10) in progress...",
      error: null,
      starting_episode: null,
      evaluation_in_progress: true,
      evaluation_mode: "confidence",
      evaluation_seed: 107,
      evaluation_seed_index: 3,
      evaluation_total_seeds: 10,
      evaluation_completed_seeds: 2,
      evaluation_success_count: 2,
      evaluation_message: "Evaluation [confidence] in progress: seed 107 (3/10)",
      evaluation_status: {
        in_progress: true,
        mode: "confidence",
        current_seed: 107,
        seed_index: 3,
        total_seeds: 10,
        completed_seeds: 2,
        success_count: 2,
        timeout_count: 0,
        success_rate: 1.0,
        seeds: [101, 103, 107, 109, 113],
        message: "Evaluation [confidence] in progress: seed 107 (3/10)",
        latest_result: {
          seed: 103,
          success: true,
          timeout: false,
          status: "SUCCESS",
          reward: 22.5,
          penned: 6,
          total_sheep: 6,
          steps: 140,
        },
        recent_results: [
          {
            seed: 101,
            success: true,
            timeout: false,
            status: "SUCCESS",
            reward: 20.0,
            penned: 6,
            total_sheep: 6,
            steps: 155,
          },
          {
            seed: 103,
            success: true,
            timeout: false,
            status: "SUCCESS",
            reward: 22.5,
            penned: 6,
            total_sheep: 6,
            steps: 140,
          },
        ],
      },
    };

    render(<EvaluationBanner status={status} />);

    // 1. Should show banner
    expect(screen.getByTestId("evaluation-banner")).toBeInTheDocument();

    // 2. Should show mode
    expect(screen.getByText(/Confidence Evaluation Running/i)).toBeInTheDocument();

    // 3. Should show current seed on test
    const currentSeedBadge = screen.getByTestId("evaluation-current-seed-badge");
    expect(currentSeedBadge).toHaveTextContent("107");
    expect(currentSeedBadge).toHaveTextContent("3/10");

    // 4. Should show in-progress message
    const msg = screen.getByTestId("evaluation-in-progress-message");
    expect(msg).toHaveTextContent("Evaluation [confidence] in progress: seed 107 (3/10)");

    // 5. Should show score
    expect(screen.getByText(/2 \/ 2 passed/i)).toBeInTheDocument();

    // 6. Should show seed track
    const track = screen.getByTestId("evaluation-seeds-track");
    expect(track).toHaveTextContent("101");
    expect(track).toHaveTextContent("103");
    expect(track).toHaveTextContent("107");
    expect(track).toHaveTextContent("109");

    // 7. Should show latest result
    const latest = screen.getByTestId("evaluation-latest-result-summary");
    expect(latest).toHaveTextContent("Seed 103");
    expect(latest).toHaveTextContent("SUCCESS");
    expect(latest).toHaveTextContent("6/6");
  });
});

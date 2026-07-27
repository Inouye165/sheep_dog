import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { describe, expect, it, vi, beforeEach } from "vitest";
import { CopyAgentDataButton } from "./CopyAgentDataButton";
import { loadHyperparams, loadTrainingDiagnostics } from "../lib/api";

// Mock API functions
vi.mock("../lib/api", () => ({
  loadHyperparams: vi.fn(),
  loadTrainingDiagnostics: vi.fn(),
}));

describe("CopyAgentDataButton", () => {
  const mockWriteText = vi.fn().mockResolvedValue(undefined);

  beforeEach(() => {
    vi.clearAllMocks();
    // Mock navigator.clipboard
    Object.assign(navigator, {
      clipboard: {
        writeText: mockWriteText,
      },
    });
  });

  it("renders correctly with default text", () => {
    render(
      <CopyAgentDataButton
        trainingStatus={null}
        checkpointIndex={null}
        curriculumStage={1}
      />
    );

    expect(screen.getByText("Copy Agent Data")).toBeInTheDocument();
  });

  it("calls loadHyperparams, loadTrainingDiagnostics, and navigator.clipboard.writeText when clicked", async () => {
    const mockHyperparams = {
      training: { learning_rate: 0.0003 },
      rewards: { terminal_success_reward: 10 },
      environment: { sheep_speed: 1.0 },
    };
    vi.mocked(loadHyperparams).mockResolvedValue(mockHyperparams as any);
    
    const mockDiagnostics = {
      diagnosticsAvailable: true,
      snapshot: {
        snapshot: {
          snapshot_timestamp: "2026-07-10T12:00:00Z",
          active_run_id: "run_test_123",
          active_checkpoint_id: "chk_test_ep_50",
          loaded_model_id: "test_model",
          policy_version: 3,
          ppo_update_count: 15,
          global_timestep: 5000,
          current_rollout_progress: "0.5",
          current_curriculum_stage: 1,
          config_hash: "hash123",
          observation_schema_hash: "obs123",
          action_space_hash: "act123",
          reward_schema_version: "v1",
          evaluation_timestamp: "2026-07-10T12:05:00Z",
          evaluation_policy_version: 3,
          evaluation_checkpoint_id: "chk_test_ep_50",
          latest_current_stage_evaluation: {
            checkpoint_id: "chk_test_ep_50",
            checkpoint_episode: 50,
            policy_version: 3,
            evaluation_timestamp: "2026-07-10T12:05:00Z",
            success_rate: 0.6,
            average_reward: 12.0,
            timeout_rate: 0.2,
            stopped_rate: 0.0,
            average_no_progress_steps: 0.0,
            average_sheep_penned: 3.0,
            average_distance_to_pen: 5.0,
            average_flock_spread: 2.0,
            average_farthest_distance_to_pen: 10.0,
            average_farthest_distance_to_flock_center: 4.0
          }
        },
        completeness: {
          table: [],
          readiness: "READY",
          reasons: []
        },
        config_snapshot: {
          "training.learning_rate": { default: 0.0003, ui: null, stage: null, checkpoint: 0.0003, active: 0.0003, source: "checkpoint" }
        },
        config_anomalies: [],
        environment_mismatches: [],
        scenario_coverage: {
          unique_seeds_count: 5,
          unique_configs_count: 5,
          sheep_to_pen_distance: { min: 1, max: 10, avg: 5 },
          dog_to_sheep_distance: { min: 2, max: 8, avg: 4 },
          resemblance_counts: {},
          resemblance_successes: {}
        },
        neural_architecture: {
          status: "COMPLETE",
          algorithm: "PPO",
          policy_class: "ActorCriticPolicy",
          feed_forward_or_recurrent: "feed-forward",
          observation_space_shape: [39],
          observation_data_type: "float32",
          observation_feature_count: 39,
          feature_extractor_class: "FlattenExtractor",
          feature_extractor_output_dimension: 64,
          actor_hidden_layers: [64, 64],
          critic_hidden_layers: [64, 64],
          shared_layers: [],
          activation_function: "tanh",
          action_space_type: "discrete",
          action_count: 5,
          ordered_action_mapping: [],
          distribution_type: "categorical",
          orthogonal_initialization_setting: true,
          normalization_settings: "none",
          total_trainable_parameter_count: 10000,
          device: "cpu",
          configured_architecture: "default",
          loaded_architecture: "default",
          compatibility_status: "COMPATIBLE"
        },
        ppo_metrics: [],
        evaluation_records: [
          {
            seed: 11,
            success: true,
            steps: 45,
            stop_reason: "success",
            initial_sheep_distance_to_pen: 5.0,
            min_sheep_distance_to_pen: 0.0,
            final_dog_to_sheep_distance: 1.5,
            num_waits: 2,
            num_sprints: 5,
            num_invalid_actions: 0,
            most_frequent_action: "move_up",
            oscillation_detected: false
          }
        ],
        failed_seed_trajectories: {},
        observation_diagnostics: null,
        counter_reconciliation: {
          rows: [],
          warnings: []
        },
        health_warnings: [],
        training_status: {}
      },
      error: null
    };
    vi.mocked(loadTrainingDiagnostics).mockResolvedValue(mockDiagnostics as any);

    render(
      <CopyAgentDataButton
        trainingStatus={{
          running: false,
          fast_mode: true,
          enable_instinct_rewards: false,
          curriculum_stage: 1,
          requested_episodes: 100,
          completed_episodes: 50,
          batch_total_episodes: 100,
          batch_completed_episodes: 50,
          total_episodes_trained: 200,
          stage_history: {},
          grand_total_episodes: 200,
          current_episode: null,
          checkpoint_episode: null,
          latest_checkpoint_episode: null,
          latest_seed: null,
          latest_replay_path: null,
          best_score: null,
          latest_success_rate: 0.6,
          latest_avg_sheep_penned: 3,
          latest_avg_reward: 5.2,
          phase: "idle",
          message: "Idle",
          error: null,
          starting_episode: null,
        } as any}
        checkpointIndex={{
          checkpoints: [
            {
              checkpoint_episode: 50,
              checkpoint: "cp50",
              evaluation: "eval50",
              replay: "rep50",
              success_rate: 0.6,
              timeout_rate: 0.1,
              average_completion_steps: 120,
              average_completion_seconds: 12,
              average_sheep_penned: 3.5,
              average_reward: 4.8,
              records: [],
            },
          ],
          latest: null,
        }}
        curriculumStage={1}
      />
    );

    const button = screen.getByRole("button", { name: "Copy agent data to clipboard" });
    fireEvent.click(button);

    // Click confirm copy in modal
    const confirmBtn = screen.getByRole("button", { name: "Copy to Clipboard" });
    fireEvent.click(confirmBtn);

    await waitFor(() => {
      expect(loadHyperparams).toHaveBeenCalled();
      expect(loadTrainingDiagnostics).toHaveBeenCalled();
    });

    await waitFor(() => {
      expect(mockWriteText).toHaveBeenCalled();
    });

    const copiedText = mockWriteText.mock.calls[0][0];
    expect(copiedText).toContain("# SHEEPDOG AGENT DIAGNOSTICS REPORT");
    expect(copiedText).toContain("Stage 1 (1 dog · 1 sheep · fixed easy penning)");
    expect(copiedText).toContain("**Success Rate**: 60%");
    expect(copiedText).toContain("**Learning Rate**: 0.0003");

    expect(screen.getByText("Agent Data Copied!")).toBeInTheDocument();
  });

  it("handles diagnostics api failure and copies report with caution block", async () => {
    vi.mocked(loadHyperparams).mockResolvedValue(null as any);
    vi.mocked(loadTrainingDiagnostics).mockResolvedValue({
      diagnosticsAvailable: false,
      snapshot: null,
      error: {
        code: "TEST_ERROR_CODE",
        message: "Simulated endpoint error message",
        exceptionType: "ValueError",
        endpoint: "/api/training/diagnostics"
      }
    });

    render(
      <CopyAgentDataButton
        trainingStatus={null}
        checkpointIndex={null}
        curriculumStage={1}
      />
    );

    const button = screen.getByRole("button", { name: "Copy agent data to clipboard" });
    fireEvent.click(button);

    // Click confirm copy in modal
    const confirmBtn = screen.getByRole("button", { name: "Copy to Clipboard" });
    fireEvent.click(confirmBtn);

    await waitFor(() => {
      expect(loadTrainingDiagnostics).toHaveBeenCalled();
    });

    await waitFor(() => {
      expect(mockWriteText).toHaveBeenCalled();
    });

    const copiedText = mockWriteText.mock.calls[0][0];
    expect(copiedText).toContain("DIAGNOSTICS API FAILURE");
    expect(copiedText).toContain("Simulated endpoint error message");
    expect(copiedText).toContain("TEST_ERROR_CODE");
    expect(copiedText).toContain("ValueError");

    // Button text changes to "Diagnostics unavailable"
    expect(screen.getByText("Diagnostics unavailable")).toBeInTheDocument();
  });

  it("allows selecting a single stage and formats the stage summary", async () => {
    const mockHyperparams = {
      training: { learning_rate: 0.0003 },
      rewards: { terminal_success_reward: 10 },
      environment: { sheep_speed: 1.0 },
    };
    vi.mocked(loadHyperparams).mockResolvedValue(mockHyperparams as any);
    vi.mocked(loadTrainingDiagnostics).mockResolvedValue({
      diagnosticsAvailable: true,
      snapshot: {
        snapshot: {
          active_curriculum_stage: 1,
          ppo_update_count: 5
        },
        completeness: { table: [], readiness: "READY", reasons: [] }
      }
    } as any);

    render(
      <CopyAgentDataButton
        trainingStatus={null}
        checkpointIndex={{
          checkpoints: [
            {
              checkpoint_episode: 50,
              success_rate: 0.8,
              timeout_rate: 0.0,
              average_completion_steps: 100,
              average_completion_seconds: 10,
              average_sheep_penned: 4.0,
              average_reward: 15.0,
              checkpoint: "",
              evaluation: "",
              replay: "",
              records: [
                { seed: 42, success: true, timeout: false, stopped: false, steps: 10, simulated_seconds: 5, sheep_penned: 4, final_sheep_distance_to_pen: 0, no_progress_steps: 0, reward_total: 15, reward_breakdown: {} as any, replay_path: "" }
              ],
              reward_config: { instincts: { curriculum_stage: 1 } }
            }
          ],
          latest: null
        }}
        curriculumStage={1}
      />
    );

    const button = screen.getByRole("button", { name: "Copy agent data to clipboard" });
    fireEvent.click(button);

    // Verify modal is open
    expect(screen.getByText("Copy Agent Data - Select Stage")).toBeInTheDocument();

    // Select "Current Stage" and click "Copy to Clipboard"
    const confirmBtn = screen.getByRole("button", { name: "Copy to Clipboard" });
    fireEvent.click(confirmBtn);

    await waitFor(() => {
      expect(mockWriteText).toHaveBeenCalled();
    });

    const copiedText = mockWriteText.mock.calls[0][0];
    expect(copiedText).toContain("CURRENT STAGE SUMMARY");
    expect(copiedText).toContain("Stage: 1");
    expect(copiedText).toContain("- Successes: 1/1");
    expect(copiedText).toContain("- Average reward: 15.0");
  });
});

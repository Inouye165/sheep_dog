import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { StudioApp } from "./StudioApp";

vi.mock("../lib/api", () => ({
  checkBackendHealth: vi.fn().mockResolvedValue({
    ok: true,
    status: "healthy",
    readOnlyPreview: false,
    trainingRunning: true,
  }),
  loadEarlyWarnings: vi.fn().mockResolvedValue({
    enabled: true,
    monitoring_status: "current",
    is_read_only_preview: false,
    mode_description: "Warning-Only",
    active_warnings: [],
    acknowledged_warnings: [],
    resolved_warnings: [],
    summary: { critical_count: 0, warning_count: 0, info_count: 0, total_active: 0, total_resolved: 0 },
    status_explanation: "Normal",
  }),
  acknowledgeEarlyWarning: vi.fn().mockResolvedValue({ success: true, warning_id: "test" }),
  isBackendStartingError: () => false,
  loadTrainingStatus: vi.fn().mockResolvedValue({
    running: true,
    phase: "training",
    curriculum_stage: 5,
    requested_episodes: 500,
    completed_episodes: 240,
    batch_total_episodes: 500,
    batch_completed_episodes: 240,
    checkpoint_save_interval: 50,
    episodes_since_latest_confidence_evaluation: 40,
    episodes_until_next_evaluation: 10,
    latest_success_rate: 0.8,
    best_score: 0.9,
    latest_avg_reward: 112.4,
    latest_avg_sheep_penned: 2.8,
    latest_timeout_rate: 0.2,
    latest_stopped_rate: 0.0,
    trainer_type: "ppo",
    auto_promote_gate: {
      ready: false,
      decision: "hold",
      reason: "Building streak",
      formal_evaluations_available: 3,
      formal_evaluations_required: 3,
      qualified_evaluations: 1,
      qualified_evaluations_required: 2,
      recent_average_success: 0.8,
      success_threshold: 0.8,
      persistent_seed_failure: false,
      blocking_seeds: [],
      step_efficiency_improving: false,
    },
  }),
  loadCheckpointIndex: vi.fn().mockResolvedValue({
    checkpoints: [
      {
        checkpoint_episode: 200,
        curriculum_stage: 5,
        success_rate: 0.8,
        average_reward: 112.4,
        average_sheep_penned: 2.8,
        average_completion_steps: 94,
        checkpoint: "generated/checkpoints/ep200.pt",
        evaluation: "generated/evaluations/ep200.json",
        replay: "generated/replays/ep200_seed101.json",
        records: [101, 103, 107, 109, 113, 127, 131, 137, 139, 149].map((seed, idx) => ({
          seed,
          success: idx < 8,
          timeout: idx >= 8,
          stopped: false,
          steps: idx < 8 ? 85 + idx * 3 : 200,
          simulated_seconds: 45,
          sheep_penned: idx < 8 ? 3 : 1,
          reward_total: idx < 8 ? 120 : -15,
          replay_path: `generated/replays/ep200_seed${seed}.json`,
        })),
      },
    ],
  }),
  loadTrainingEpisodes: vi.fn().mockResolvedValue({
    episodes: [
      {
        id: 1,
        event_key: "ep-1",
        run_id: "run-1",
        session_id: "sess-1",
        global_environment_episode: 199,
        episode_in_stage: 49,
        curriculum_stage: 5,
        global_timestep: 19900,
        policy_version: 4,
        completed_at: "2026-10-14T08:00:00Z",
        active_runtime_seconds_total: 120,
        reward: 115,
        result: "SUCCESS",
        success: true,
        timeout: false,
        stopped: false,
        sheep_penned: 3,
        total_sheep: 3,
        steps: 90,
        seed: 101,
        checkpoint_id: null,
      },
      {
        id: 2,
        event_key: "ep-2",
        run_id: "run-1",
        session_id: "sess-1",
        global_environment_episode: 200,
        episode_in_stage: 50,
        curriculum_stage: 5,
        global_timestep: 20000,
        policy_version: 4,
        completed_at: "2026-10-14T08:01:00Z",
        active_runtime_seconds_total: 125,
        reward: 122,
        result: "SUCCESS",
        success: true,
        timeout: false,
        stopped: false,
        sheep_penned: 3,
        total_sheep: 3,
        steps: 86,
        seed: 103,
        checkpoint_id: null,
      },
    ],
    latest_id: 2,
    next_after_id: 3,
    has_more: false,
    oldest_available_timestamp: null,
    total_matching: 2,
  }),
  loadRecentEvaluations: vi.fn().mockResolvedValue([]),
  loadSessionEvents: vi.fn().mockResolvedValue({ events: [], total_sessions: 0 }),
  loadStageDiagnostics: vi.fn().mockResolvedValue(null),
  loadEffectiveConfig: vi.fn().mockResolvedValue({}),
  loadNetworkTopology: vi.fn().mockResolvedValue(null),
  loadReplay: vi.fn().mockResolvedValue({
    seed: 101,
    policy_name: "neural_policy",
    final_snapshot: {
      step: 85,
      simulated_seconds: 42.5,
      grid_width: 40,
      grid_height: 30,
      dogs: [{ index: 0, x: 15, y: 12, role: "rear_pressure" }],
      sheep: [{ index: 0, x: 31, y: 14, penned: true, personality: "obedient" }],
      pen: { origin: { x: 30, y: 10 }, width: 8, height: 10, opening: "left" },
      penned_count: 1,
      average_distance_to_pen: 0,
      flock_spread: 0,
      no_progress_steps: 0,
      terminated: true,
      timeout: false,
      stopped: false,
      success: true,
      status: "all_sheep_penned",
    },
    stats: {
      steps: 85,
      simulated_seconds: 42.5,
      sheep_penned: 3,
      timeout: false,
      terminated: true,
      success: true,
      stopped: false,
      stop_reason: "all_sheep_penned",
      reward_total: 120,
      no_progress_steps: 0,
      final_avg_distance_to_pen: 0,
      final_flock_spread: 0,
      role_distribution: {},
      role_switches: 0,
      collector_activations: 0,
      blocker_activations: 0,
      sheep_split_events: 0,
      final_reward_breakdown: {
        progress_to_pen: 10,
        sheep_penned: 100,
        flock_cohesion: 5,
        scatter_penalty: 0,
        time_penalty: -5,
        no_progress_penalty: 0,
        wall_pressure_penalty: 0,
        wait_penalty: 0,
        terminal_success: 10,
        terminal_failure: 0,
        total: 120,
      },
    },
    frames: [
      {
        step: 0,
        actions: ["right"],
        reward: {
          progress_to_pen: 1,
          sheep_penned: 0,
          flock_cohesion: 1,
          scatter_penalty: 0,
          time_penalty: -0.1,
          no_progress_penalty: 0,
          wall_pressure_penalty: 0,
          wait_penalty: 0,
          terminal_success: 0,
          terminal_failure: 0,
          total: 1.9,
        },
        snapshot: {
          step: 0,
          simulated_seconds: 0,
          grid_width: 40,
          grid_height: 30,
          dogs: [{ index: 0, x: 10, y: 12, role: "rear_pressure" }],
          sheep: [{ index: 0, x: 20, y: 14, penned: false, personality: "obedient" }],
          pen: { origin: { x: 30, y: 10 }, width: 8, height: 10, opening: "left" },
          penned_count: 0,
          average_distance_to_pen: 10,
          flock_spread: 1,
          no_progress_steps: 0,
          terminated: false,
          timeout: false,
          stopped: false,
          success: false,
          status: "running",
        },
      },
    ],
  }),
  fetchReplayById: vi.fn().mockResolvedValue(null),
  runReplay: vi.fn(),
  startTraining: vi.fn(),
  pauseTraining: vi.fn(),
  stopTraining: vi.fn(),
  clearTraining: vi.fn(),
  resetJourneyTraining: vi.fn(),
  crossStageFork: vi.fn(),
  startRemediationFork: vi.fn(),
  shutdownApp: vi.fn(),
}));

describe("StudioApp v2", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("renders the Command Center, KPI ribbon, 10-seed strip, and switches to the 10-Seed Evaluation Arena", async () => {
    render(<StudioApp />);

    expect(screen.getByText("Sheepdog Studio")).toBeInTheDocument();
    expect(screen.getByText("Classic UI ↗")).toHaveAttribute("href", "/");

    await waitFor(() => {
      expect(screen.getByText(/TRAINING · Ep 240\/500/i)).toBeInTheDocument();
    });

    // Verify all 10 benchmark seeds appear in the quick strip
    expect(screen.getByText("#101")).toBeInTheDocument();
    expect(screen.getByText("#149")).toBeInTheDocument();

    // Switch to the 10-Seed Evaluation Arena
    const arenaTab = screen.getByRole("button", { name: /10-Seed Evaluation Arena/i });
    fireEvent.click(arenaTab);

    await waitFor(() => {
      expect(screen.getByText("Seed #101")).toBeInTheDocument();
      expect(screen.getByText("Seed #149")).toBeInTheDocument();
    });

    // Switch to the 10-Seed Matrix view inside the Arena
    const matrixBtn = screen.getByRole("button", { name: /10-Seed Matrix/i });
    fireEvent.click(matrixBtn);

    expect(
      screen.getByText(/Cross-Evaluation 10-Seed Reliability Matrix/i)
    ).toBeInTheDocument();
  });

  it("allows picking a seed and loads the selected seed replay", async () => {
    const api = await import("../lib/api");
    render(<StudioApp />);

    await waitFor(() => {
      expect(screen.getByLabelText("Select replay seed")).toBeInTheDocument();
    });

    const seedSelect = screen.getByLabelText("Select replay seed") as HTMLSelectElement;
    expect(seedSelect.value).toBe("101");

    // Pick seed #137 via the HUD dropdown
    fireEvent.change(seedSelect, { target: { value: "137" } });
    await waitFor(() => {
      expect(seedSelect.value).toBe("137");
      expect(api.loadReplay).toHaveBeenCalledWith("generated/replays/ep200_seed137.json");
    });

    // Pick next seed (#139) via the Next seed button
    const nextBtn = screen.getByLabelText("Next seed");
    fireEvent.click(nextBtn);
    await waitFor(() => {
      expect(seedSelect.value).toBe("139");
      expect(api.loadReplay).toHaveBeenCalledWith("generated/replays/ep200_seed139.json");
    });
  });
});

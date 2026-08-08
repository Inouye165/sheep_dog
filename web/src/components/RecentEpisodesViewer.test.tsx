import { describe, it, expect, beforeEach, vi } from "vitest";
import { render, screen, fireEvent, act } from "@testing-library/react";
import React from "react";
import {
  episodeHistoryStore,
  formatEpisodeLabel,
  convertTrainingEpisodeToRecord,
  createPreseededEpisodesForTesting,
} from "../lib/episodeHistoryStore";
import { RecentEpisodesViewer } from "./RecentEpisodesViewer";
import { loadTrainingEpisodes, loadFailedEpisodes, fetchReplayById } from "../lib/api";
import type { EpisodeRecord, TrainingEpisode } from "../state/types";

vi.mock("../lib/api", () => ({
  loadTrainingEpisodes: vi.fn(),
  loadFailedEpisodes: vi.fn(),
  fetchReplayById: vi.fn().mockResolvedValue(null),
  fetchCapturePolicy: vi.fn().mockResolvedValue({
    mode: "selective",
    next_n_counter: 0,
    success_sample_rate: 0.05,
    target_stage: null,
    target_outcome: "all",
    queued_writes: 0,
    written_count: 0,
    dropped_count: 0,
    failure_count: 0,
  }),
  updateCapturePolicy: vi.fn().mockResolvedValue({
    mode: "selective",
    next_n_counter: 10,
    success_sample_rate: 0.05,
    target_stage: null,
    target_outcome: "failures",
    queued_writes: 0,
    written_count: 0,
    dropped_count: 0,
    failure_count: 0,
  }),
  reproduceEpisode: vi.fn().mockResolvedValue(null),
}));

const mockLoadTrainingEpisodes = vi.mocked(loadTrainingEpisodes);
const mockLoadFailedEpisodes = vi.mocked(loadFailedEpisodes);
const mockFetchReplayById = vi.mocked(fetchReplayById);

describe("EpisodeHistoryStore Unit Tests", () => {
  beforeEach(() => {
    episodeHistoryStore.clear();
    vi.clearAllMocks();
  });

  it("1. Store initialization contains no preseeded production records", () => {
    const storeEpisodes = episodeHistoryStore.getEpisodes();
    expect(storeEpisodes.length).toBe(0);
  });

  it("2. Empty API response leaves the episode list empty", () => {
    episodeHistoryStore.syncWithApiEpisodes([]);
    expect(episodeHistoryStore.getEpisodes().length).toBe(0);
  });

  it("5. Real and mock/preseeded records are never mixed", () => {
    const fakeRecords = createPreseededEpisodesForTesting();
    expect(fakeRecords.length).toBe(50);
    // Explicit test fixture records exist separately
    expect(fakeRecords[0].episode_id).toBe(1400);

    // Production store syncs ONLY real API episodes
    const realApiEpisode: TrainingEpisode = {
      id: 2009,
      event_key: "ep_2009",
      run_id: "run_stage8",
      session_id: null,
      global_environment_episode: 1720,
      episode_in_stage: 1720,
      curriculum_stage: 8,
      global_timestep: 100000,
      policy_version: 1,
      completed_at: "2026-08-05T15:00:00Z",
      active_runtime_seconds_total: 10.0,
      reward: -100,
      result: "TIMEOUT",
      success: false,
      timeout: true,
      stopped: false,
      sheep_penned: 1,
      total_sheep: 4,
      steps: 1040,
      seed: 123,
      checkpoint_id: "chk_100",
    };

    episodeHistoryStore.syncWithApiEpisodes([realApiEpisode]);
    const episodes = episodeHistoryStore.getEpisodes();

    expect(episodes.length).toBe(1);
    expect(episodes[0].episode_id).toBe(1720);
    expect(episodes.some((e) => Number(e.episode_id) <= 1400)).toBe(false);
  });

  it("6. A real Stage 8 timeout with steps: 1040 displays 1,040 steps in record conversion", () => {
    const ep: TrainingEpisode = {
      id: 2009,
      event_key: "ep_2009",
      run_id: "run1",
      session_id: null,
      global_environment_episode: 1720,
      episode_in_stage: 1720,
      curriculum_stage: 8,
      global_timestep: null,
      policy_version: 1,
      completed_at: "2026-08-05T15:00:00Z",
      active_runtime_seconds_total: null,
      reward: -219.0,
      result: "TIMEOUT",
      success: false,
      timeout: true,
      stopped: false,
      sheep_penned: 1,
      total_sheep: 4,
      steps: 1040,
      seed: 42,
      checkpoint_id: "chk_123",
    };

    const record = convertTrainingEpisodeToRecord(ep);
    expect(record.total_moves).toBe(1040);
    expect(record.stage).toBe(8);
    expect(record.outcome).toBe("timeout");
  });

  it("7. A real stopped episode with steps: 938 displays 938 steps in record conversion", () => {
    const ep: TrainingEpisode = {
      id: 1993,
      event_key: "ep_1993",
      run_id: "run1",
      session_id: null,
      global_environment_episode: 1704,
      episode_in_stage: 1704,
      curriculum_stage: 8,
      global_timestep: null,
      policy_version: 1,
      completed_at: "2026-08-05T15:25:00Z",
      active_runtime_seconds_total: null,
      reward: -340.15,
      result: "STOPPED",
      success: false,
      timeout: false,
      stopped: true,
      sheep_penned: 2,
      total_sheep: 4,
      steps: 938,
      seed: 260,
      checkpoint_id: "chk_6784",
    };

    const record = convertTrainingEpisodeToRecord(ep);
    expect(record.total_moves).toBe(938);
    expect(record.stage).toBe(8);
    expect(record.outcome).toBe("loss");
  });

  it("8. Scalar training episodes contain no synthetic frames", () => {
    const ep: TrainingEpisode = {
      id: 1,
      event_key: "ep_1",
      run_id: "run1",
      session_id: null,
      global_environment_episode: 1,
      episode_in_stage: 1,
      curriculum_stage: 8,
      global_timestep: null,
      policy_version: 1,
      completed_at: "2026-08-05T15:00:00Z",
      active_runtime_seconds_total: null,
      reward: 10.0,
      result: "SUCCESS",
      success: true,
      timeout: false,
      stopped: false,
      sheep_penned: 4,
      total_sheep: 4,
      steps: 120,
      seed: 100,
      checkpoint_id: null,
    };

    const record = convertTrainingEpisodeToRecord(ep);
    expect(record.move_history).toBeUndefined();
    expect(record.initial_state).toBeUndefined();
    expect(record.replayAvailable).toBe(false);
  });

  it("11. Missing curriculum stage displays unknown rather than defaulting to Stage 8", () => {
    const ep: TrainingEpisode = {
      id: 1,
      event_key: "ep_1",
      run_id: "run1",
      session_id: null,
      global_environment_episode: 1,
      episode_in_stage: 1,
      curriculum_stage: (null as unknown) as number,
      global_timestep: null,
      policy_version: 1,
      completed_at: "2026-08-05T15:00:00Z",
      active_runtime_seconds_total: null,
      reward: 0,
      result: "STOPPED",
      success: false,
      timeout: false,
      stopped: true,
      sheep_penned: 0,
      total_sheep: 4,
      steps: 50,
      seed: 100,
      checkpoint_id: null,
    };

    const record = convertTrainingEpisodeToRecord(ep);
    expect(record.stage).toBeNull();
  });
});

describe("RecentEpisodesViewer UI Integration Tests", () => {
  beforeEach(() => {
    episodeHistoryStore.clear();
    mockLoadTrainingEpisodes.mockReset();
    vi.clearAllMocks();
  });

  it("2. Empty API response leaves the UI in empty state", async () => {
    mockLoadFailedEpisodes.mockResolvedValue([]);

    await act(async () => {
      render(<RecentEpisodesViewer />);
    });

    expect(screen.getByTestId("empty-state")).toBeInTheDocument();
    expect(screen.getByText("No recorded training episodes are available.")).toBeInTheDocument();
  });

  it("3. API failure displays an error and does not display fake episodes", async () => {
    mockLoadFailedEpisodes.mockResolvedValue(null);

    await act(async () => {
      render(<RecentEpisodesViewer />);
    });

    expect(screen.getByTestId("error-state")).toBeInTheDocument();
    expect(screen.getByText("Failed to load recent training episodes.")).toBeInTheDocument();
    expect(screen.queryByText(/Episode #1351/i)).not.toBeInTheDocument();
  });

  it("4. An API response with seven episodes displays exactly seven episodes", async () => {
    const mockEpisodes: TrainingEpisode[] = Array.from({ length: 7 }, (_, i) => ({
      id: 100 + i,
      event_key: `ep_${100 + i}`,
      run_id: "run1",
      session_id: null,
      global_environment_episode: 100 + i,
      episode_in_stage: i + 1,
      curriculum_stage: 8,
      global_timestep: null,
      policy_version: 1,
      completed_at: `2026-08-05T15:00:${String(i).padStart(2, "0")}Z`,
      active_runtime_seconds_total: null,
      reward: i * 10,
      result: "STOPPED",
      success: false,
      timeout: false,
      stopped: true,
      sheep_penned: 2,
      total_sheep: 4,
      steps: 100 + i * 50,
      seed: 500 + i,
      checkpoint_id: null,
      replay_available: true,
      replay_id: `diag_ep${100 + i}`,
      capture_status: "available",
    }));

    mockLoadFailedEpisodes.mockResolvedValue(mockEpisodes);

    await act(async () => {
      render(<RecentEpisodesViewer />);
    });

    const dropdown = screen.getByLabelText("Recent Episodes Selector") as HTMLSelectElement;
    expect(dropdown.options.length).toBe(7);
  });

  it("6 & 7 & 12. Displays real step counts (1040 & 938) and scalar details without fabricating grid coordinates", async () => {
    const mockEpisodes: TrainingEpisode[] = [
      {
        id: 2009,
        event_key: "ep_2009",
        run_id: "run1",
        session_id: null,
        global_environment_episode: 1720,
        episode_in_stage: 1720,
        curriculum_stage: 8,
        global_timestep: null,
        policy_version: 1,
        completed_at: "2026-08-05T15:00:00Z",
        active_runtime_seconds_total: null,
        reward: -219.0,
        result: "TIMEOUT",
        success: false,
        timeout: true,
        stopped: false,
        sheep_penned: 1,
        total_sheep: 4,
        steps: 1040,
        seed: 42,
        checkpoint_id: "chk_stage8_1040",
        replay_available: true,
        replay_id: "diag_1720",
        capture_status: "available",
      },
      {
        id: 1993,
        event_key: "ep_1993",
        run_id: "run1",
        session_id: null,
        global_environment_episode: 1704,
        episode_in_stage: 1704,
        curriculum_stage: 8,
        global_timestep: null,
        policy_version: 1,
        completed_at: "2026-08-05T15:25:00Z",
        active_runtime_seconds_total: null,
        reward: -340.15,
        result: "STOPPED",
        success: false,
        timeout: false,
        stopped: true,
        sheep_penned: 2,
        total_sheep: 4,
        steps: 938,
        seed: 260,
        checkpoint_id: "chk_stage8_938",
        replay_available: true,
        replay_id: "diag_1704",
        capture_status: "available",
      },
    ];

    mockLoadFailedEpisodes.mockResolvedValue(mockEpisodes);

    await act(async () => {
      render(<RecentEpisodesViewer />);
    });

    // Most recent episode (1720 with 1040 steps) is selected
    expect(screen.getByText("Episode #1720")).toBeInTheDocument();
    expect(screen.getByText("1,040")).toBeInTheDocument();

    // Select second episode (1704 with 938 steps)
    const dropdown = screen.getByLabelText("Recent Episodes Selector") as HTMLSelectElement;
    fireEvent.change(dropdown, { target: { value: "1704" } });

    expect(screen.getByText("Episode #1704")).toBeInTheDocument();
    expect(screen.getByText("938")).toBeInTheDocument();
  });

  it("9. Replay controls are disabled or absent without authentic replay data", async () => {
    const mockEpisodes: TrainingEpisode[] = [
      {
        id: 100,
        event_key: "ep_100",
        run_id: "run1",
        session_id: null,
        global_environment_episode: 100,
        episode_in_stage: 100,
        curriculum_stage: 8,
        global_timestep: null,
        policy_version: 1,
        completed_at: "2026-08-05T15:00:00Z",
        active_runtime_seconds_total: null,
        reward: -50,
        result: "STOPPED",
        success: false,
        timeout: false,
        stopped: true,
        sheep_penned: 1,
        total_sheep: 4,
        steps: 400,
        seed: 12,
        checkpoint_id: null,
        replay_available: false,
        replay_id: null,
        capture_status: "not_requested",
      },
    ];

    mockLoadFailedEpisodes.mockResolvedValue(mockEpisodes);

    await act(async () => {
      render(<RecentEpisodesViewer />);
    });

    expect(screen.getByTestId("no-replay-banner")).toBeInTheDocument();
    expect(screen.getByText("Replay not recorded")).toBeInTheDocument();

    const disabledPlayBtn = screen.getByText("▶ Play Replay (Disabled)") as HTMLButtonElement;
    expect(disabledPlayBtn).toBeDisabled();
  });

  it("10. Authentic checkpoint replay records still open and display real StepRecord trajectory data", async () => {
    const authenticRecord: EpisodeRecord = {
      episode_id: 5000,
      timestamp: "16:00:00",
      stage: 8,
      outcome: "loss",
      outcome_label: "STOPPED",
      total_moves: 10,
      sheep_penned: 2,
      total_sheep: 4,
      replayAvailable: true,
      replaySource: "checkpoint-evaluation",
      initial_state: {
        step: 0,
        simulated_seconds: 0,
        grid_width: 108,
        grid_height: 78,
        dogs: [{ index: 0, x: 10, y: 10 }],
        sheep: [{ index: 0, x: 20, y: 20, penned: false }],
        pen: { origin: { x: 94, y: 1 }, width: 14, height: 14 },
        penned_count: 0,
        average_distance_to_pen: 50,
        flock_spread: 2,
        no_progress_steps: 0,
        terminated: false,
        timeout: false,
        stopped: false,
        success: false,
        status: "RUNNING",
      },
      move_history: Array.from({ length: 11 }, (_, s) => ({
        step: s,
        actions: ["ADVANCE"],
        snapshot: {
          step: s,
          simulated_seconds: s,
          grid_width: 108,
          grid_height: 78,
          dogs: [{ index: 0, x: 10 + s, y: 10 }],
          sheep: [{ index: 0, x: 20 + s, y: 20, penned: false }],
          pen: { origin: { x: 94, y: 1 }, width: 14, height: 14 },
          penned_count: 0,
          average_distance_to_pen: 50 - s,
          flock_spread: 2,
          no_progress_steps: 0,
          terminated: s === 10,
          timeout: false,
          stopped: s === 10,
          success: false,
          status: s === 10 ? "STOPPED" : "RUNNING",
        },
        reward: {
          progress_to_pen: 1.0,
          sheep_penned: 0,
          flock_cohesion: 0,
          scatter_penalty: 0,
          time_penalty: -0.01,
          no_progress_penalty: 0,
          wall_pressure_penalty: 0,
          wait_penalty: 0,
          terminal_success: 0,
          terminal_failure: 0,
          total: 1.0,
        },
      })),
    };

    // Pre-populate store
    episodeHistoryStore.setEpisodes([authenticRecord]);

    // Mock API returning null so store is not cleared by empty API sync
    mockLoadFailedEpisodes.mockImplementation(async () => null);

    await act(async () => {
      render(<RecentEpisodesViewer />);
    });

    expect(screen.getByText("Authentic Evaluation Replay (checkpoint-evaluation)")).toBeInTheDocument();
    expect(screen.getByText("▶ Play")).toBeInTheDocument();
  });

  it("11. The React dropdown renders all 25 API results from loadFailedEpisodes", async () => {
    const mock25FailedEpisodes: TrainingEpisode[] = Array.from({ length: 25 }, (_, i) => ({
      id: 2000 + i,
      event_key: `ep_${2000 + i}`,
      run_id: "run1",
      session_id: null,
      global_environment_episode: 2400 + i,
      episode_in_stage: i + 1,
      curriculum_stage: (i % 3) + 7,
      global_timestep: null,
      policy_version: 1,
      completed_at: `2026-08-05T15:${String(i % 60).padStart(2, "0")}:00Z`,
      active_runtime_seconds_total: null,
      reward: -100 - i,
      result: i % 2 === 0 ? "TIMEOUT" : "STOPPED",
      success: false,
      timeout: i % 2 === 0,
      stopped: i % 2 !== 0,
      sheep_penned: i % 4,
      total_sheep: 4,
      steps: 500 + i,
      seed: 10 + i,
      checkpoint_id: null,
      replay_available: true,
      replay_id: `diag_stage${(i % 3) + 7}_ep${2400 + i}_seed${10 + i}`,
      replay_path: `/artifacts/replays/diag_stage${(i % 3) + 7}_ep${2400 + i}_seed${10 + i}.json.gz`,
      capture_status: "available",
      capture_reason: i % 2 === 0 ? "timeout" : "stopped",
    })).reverse();

    mockLoadFailedEpisodes.mockResolvedValue(mock25FailedEpisodes);

    await act(async () => {
      render(<RecentEpisodesViewer />);
    });

    const dropdown = screen.getByLabelText("Recent Episodes Selector") as HTMLSelectElement;
    expect(dropdown.options.length).toBe(25);
    expect(dropdown.options[0].text).toContain("Episode 2424 — Stage 7 — Seed 34 — Failed");
  });

  it("12. Selecting an item in the dropdown requests the correct replay_id", async () => {
    const mockEpisodes: TrainingEpisode[] = [
      {
        id: 3001,
        event_key: "ep_3001",
        run_id: "run1",
        session_id: null,
        global_environment_episode: 2474,
        episode_in_stage: 74,
        curriculum_stage: 9,
        global_timestep: null,
        policy_version: 1,
        completed_at: "2026-08-05T15:00:00Z",
        active_runtime_seconds_total: null,
        reward: -200,
        result: "TIMEOUT",
        success: false,
        timeout: true,
        stopped: false,
        sheep_penned: 1,
        total_sheep: 4,
        steps: 800,
        seed: 87,
        checkpoint_id: null,
        replay_available: true,
        replay_id: "diag_stage9_ep2474_seed87",
        replay_path: "/artifacts/replays/diag_stage9_ep2474_seed87.json.gz",
        capture_status: "available",
        capture_reason: "timeout",
      },
      {
        id: 3000,
        event_key: "ep_3000",
        run_id: "run1",
        session_id: null,
        global_environment_episode: 2473,
        episode_in_stage: 73,
        curriculum_stage: 8,
        global_timestep: null,
        policy_version: 1,
        completed_at: "2026-08-05T14:59:00Z",
        active_runtime_seconds_total: null,
        reward: -300,
        result: "STOPPED",
        success: false,
        timeout: false,
        stopped: true,
        sheep_penned: 2,
        total_sheep: 4,
        steps: 600,
        seed: 86,
        checkpoint_id: null,
        replay_available: true,
        replay_id: "diag_stage8_ep2473_seed86",
        replay_path: "/artifacts/replays/diag_stage8_ep2473_seed86.json.gz",
        capture_status: "available",
        capture_reason: "stopped",
      },
    ];

    mockLoadFailedEpisodes.mockResolvedValue(mockEpisodes);

    await act(async () => {
      render(<RecentEpisodesViewer />);
    });

    const dropdown = screen.getByLabelText("Recent Episodes Selector") as HTMLSelectElement;
    expect(dropdown.value).toBe("2474");

    // Select second item (2473 with replay_id diag_stage8_ep2473_seed86)
    await act(async () => {
      fireEvent.change(dropdown, { target: { value: "2473" } });
    });

    expect(mockFetchReplayById).toHaveBeenCalledWith("diag_stage8_ep2473_seed86");
  });
});

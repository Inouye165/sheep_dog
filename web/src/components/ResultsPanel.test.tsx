import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { describe, expect, it, vi, beforeEach } from "vitest";
import { ResultsPanel } from "./ResultsPanel";
import { loadReplay } from "../lib/api";
import { exportResultsVideo } from "../lib/resultsVideo";
import type { CheckpointIndex, ReplayBundle } from "../state/types";

// Mock loadReplay from API
vi.mock("../lib/api", () => ({
  loadReplay: vi.fn(),
}));

// Mock resultsVideo module
vi.mock("../lib/resultsVideo", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../lib/resultsVideo")>();
  return {
    ...actual,
    exportResultsVideo: vi.fn(),
    makeResultsVideoFileName: vi.fn().mockReturnValue("sheepdog-results-video-20260610-111415.webm"),
  };
});

describe("ResultsPanel", () => {
  const mockCheckpointIndex: CheckpointIndex = {
    checkpoints: [
      {
        checkpoint_episode: 100,
        recorded_at: "2026-06-10",
        checkpoint: "cp1",
        evaluation: "eval1",
        replay: "rep1",
        success_rate: 0.8,
        timeout_rate: 0.2,
        average_completion_steps: 10,
        average_completion_seconds: 5,
        average_sheep_penned: 3,
        average_reward: 12.5,
        records: [{ seed: 42, success: true, timeout: false, stopped: false, steps: 10, simulated_seconds: 5, sheep_penned: 3, final_sheep_distance_to_pen: 0, no_progress_steps: 0, reward_total: 10, reward_breakdown: {} as any, replay_path: "/replays/1.json" }],
        reward_config: { instincts: { curriculum_stage: 1, enable_instinct_rewards: true } },
      },
    ],
    latest: null,
  };

  const createMockReplayBundle = (framesCount = 1): ReplayBundle => ({
    seed: 42,
    policy_name: "test",
    final_snapshot: {
      step: 0,
      simulated_seconds: 0,
      dogs: [],
      sheep: [],
      pen: { origin: { x: 0, y: 0 }, width: 10, height: 10 },
      penned_count: 0,
      average_distance_to_pen: 0,
      flock_spread: 0,
      no_progress_steps: 0,
      terminated: false,
      timeout: false,
      stopped: false,
      success: false,
      status: "done",
    },
    stats: {} as any,
    frames: Array.from({ length: framesCount }, (_, i) => ({
      step: i,
      actions: [],
      snapshot: {
        step: i,
        simulated_seconds: i,
        dogs: [],
        sheep: [],
        pen: { origin: { x: 0, y: 0 }, width: 10, height: 10 },
        penned_count: 0,
        average_distance_to_pen: 0,
        flock_spread: 0,
        no_progress_steps: 0,
        terminated: false,
        timeout: false,
        stopped: false,
        success: false,
        status: "running",
      },
      reward: {} as any,
    })),
  });

  beforeEach(() => {
    vi.stubGlobal("URL", {
      createObjectURL: vi.fn().mockReturnValue("blob:http://localhost/mock-blob"),
      revokeObjectURL: vi.fn(),
    });
    vi.clearAllMocks();
  });

  it("does not show Export Video button when no checkpoints exist", () => {
    render(<ResultsPanel checkpointIndex={{ checkpoints: [], latest: null }} />);
    expect(screen.queryByRole("button", { name: /Export Video/i })).not.toBeInTheDocument();
  });

  it("shows Export Video button when checkpoints exist", async () => {
    vi.mocked(loadReplay).mockResolvedValue(createMockReplayBundle());

    render(<ResultsPanel checkpointIndex={mockCheckpointIndex} />);

    // Button should be in the document
    const button = screen.getByRole("button", { name: /Export Video/i });
    expect(button).toBeInTheDocument();

    await waitFor(() => {
      expect(button).toBeEnabled();
    });
  });

  it("Export Video button is disabled while replay bundles are loading, then becomes enabled", async () => {
    let resolveReplay: (value: ReplayBundle) => void = () => {};
    const loadingPromise = new Promise<ReplayBundle>((resolve) => {
      resolveReplay = resolve;
    });

    vi.mocked(loadReplay).mockReturnValue(loadingPromise);

    render(<ResultsPanel checkpointIndex={mockCheckpointIndex} />);

    // Button should exist and be disabled initially
    const button = screen.getByRole("button", { name: /Export Video/i });
    expect(button).toBeDisabled();

    // Resolve loading
    resolveReplay(createMockReplayBundle());

    await waitFor(() => {
      expect(button).toBeEnabled();
    });
  });

  it("clicking Export Video uses the loaded replay bundles and triggers download", async () => {
    vi.mocked(loadReplay).mockResolvedValue(createMockReplayBundle());
    vi.mocked(exportResultsVideo).mockResolvedValue(new Blob(["mock-webm"], { type: "video/webm" }));

    // Mock anchor tag download trigger
    const clickSpy = vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(() => {});

    render(<ResultsPanel checkpointIndex={mockCheckpointIndex} />);

    const button = screen.getByRole("button", { name: /Export Video/i });

    // Wait for the load to finish and button to enable
    await waitFor(() => {
      expect(button).toBeEnabled();
    });

    fireEvent.click(button);

    // Button should show "Exporting..." state
    expect(button).toHaveTextContent(/Exporting/);

    // Wait for export helper to be called
    await waitFor(() => {
      expect(exportResultsVideo).toHaveBeenCalled();
      expect(clickSpy).toHaveBeenCalled();
    });

    // Check success status message
    expect(screen.getByText("Video export started")).toBeInTheDocument();

    // Button label should restore
    expect(button).toHaveTextContent("Export Video");
  });

  it("shows a clean error message if browser API is unsupported", async () => {
    vi.mocked(loadReplay).mockResolvedValue(createMockReplayBundle());
    vi.mocked(exportResultsVideo).mockRejectedValue(new Error("Video export is not supported in this browser"));

    render(<ResultsPanel checkpointIndex={mockCheckpointIndex} />);

    const button = screen.getByRole("button", { name: /Export Video/i });

    await waitFor(() => {
      expect(button).toBeEnabled();
    });

    fireEvent.click(button);

    await waitFor(() => {
      expect(screen.getByText("Video export is not supported in this browser")).toBeInTheDocument();
    });
  });

  it("preserves Play, Pause, Reset, and speed controls functionality", async () => {
    vi.mocked(loadReplay).mockResolvedValue(createMockReplayBundle(5));

    render(<ResultsPanel checkpointIndex={mockCheckpointIndex} />);

    // Wait for loading to finish so playing becomes possible
    await waitFor(() => {
      expect(screen.getByRole("button", { name: "Play" })).toBeEnabled();
    });

    const playBtn = screen.getByRole("button", { name: "Play" });
    const resetBtn = screen.getByRole("button", { name: "Reset" });
    const speedSelect = screen.getByRole("combobox");

    expect(playBtn).toBeEnabled();
    expect(resetBtn).toBeEnabled(); // Reset is enabled when not playing

    // Clicking Play starts playing
    fireEvent.click(playBtn);
    expect(playBtn).toHaveTextContent("Pause");
    expect(resetBtn).toBeDisabled(); // Reset is disabled when playing


    // Clicking Pause pauses playing
    fireEvent.click(playBtn);
    expect(playBtn).toHaveTextContent("Play");

    // Select speed control works
    fireEvent.change(speedSelect, { target: { value: "fast" } });
    expect(speedSelect).toHaveValue("fast");
  });
});

import { describe, expect, it, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { ChartHoverPortal } from "./DiagnosticsPanel";
import type { CheckpointEntry } from "../state/types";

const mockCheckpoint: CheckpointEntry = {
  checkpoint: "chk_100.pt",
  evaluation: "eval_100.json",
  replay: "replay_100.json",
  records: [],
  checkpoint_id: "cp_100",
  checkpoint_episode: 100,
  curriculum_stage: 1,
  success_rate: 0.9,
  timeout_rate: 0.05,
  average_reward: 120.5,
  average_completion_steps: 45,
  average_completion_seconds: 12.3,
  average_sheep_penned: 3,
  recorded_at: "2026-07-29T12:00:00Z",
  active_runtime_seconds_total: 600,
  wall_clock_elapsed_seconds: 720,
  session_id: "sess_123",
  promotion_gate: {
    decision: "blocked",
    reason: "Aggregate success is 88%; 90% required. Seed 41 failed in 3 of the last 5 evaluations.",
    window_size: 5,
    minimum_required_evaluations: 3,
    total_seed_trials: 50,
    total_successes: 44,
    aggregate_success_rate: 0.88,
    aggregate_timeout_rate: 0.04,
    latest_success_rate: 0.9,
    recent_qualifying_checkpoints: 2,
    recent_checkpoints_considered: 3,
    latest_floor_passed: true,
    reward_guard_passed: true,
    seed_consistency_passed: false,
    blocking_seeds: [41],
    blocking_reasons: [
      "Aggregate success is 88%; 90% required.",
      "Seed 41 failed in 3 of the last 5 evaluations.",
    ],
  },
};

describe("ChartHoverPortal Tooltip Positioning & Content", () => {
  it("positions correctly near center of viewport", () => {
    Object.defineProperty(window, "innerWidth", { writable: true, configurable: true, value: 1000 });
    Object.defineProperty(window, "innerHeight", { writable: true, configurable: true, value: 800 });

    const targetRect = {
      left: 450,
      right: 470,
      top: 400,
      bottom: 420,
      width: 20,
      height: 20,
    } as DOMRect;

    render(<ChartHoverPortal hoveredPoint={{ x: 100, y: 0.9, stage: 1, checkpoint: mockCheckpoint }} targetRect={targetRect} />);

    const tooltip = screen.getByTestId("chart-tooltip");
    expect(tooltip).toBeInTheDocument();
    expect(tooltip.style.position).toBe("fixed");
    expect(tooltip.style.maxWidth).toBe("calc(100vw - 24px)");
    expect(tooltip.style.maxHeight).toBe("calc(100vh - 24px)");
    expect(tooltip.style.overflowWrap).toBe("anywhere");
  });

  it("clamps horizontally near left edge", () => {
    Object.defineProperty(window, "innerWidth", { writable: true, configurable: true, value: 1000 });
    Object.defineProperty(window, "innerHeight", { writable: true, configurable: true, value: 800 });

    const targetRect = {
      left: 2,
      right: 22,
      top: 400,
      bottom: 420,
      width: 20,
      height: 20,
    } as DOMRect;

    render(<ChartHoverPortal hoveredPoint={{ x: 100, y: 0.9, stage: 1, checkpoint: mockCheckpoint }} targetRect={targetRect} />);

    const tooltip = screen.getByTestId("chart-tooltip");
    const leftVal = parseFloat(tooltip.style.left);
    expect(leftVal).toBeGreaterThanOrEqual(12);
  });

  it("clamps horizontally near right edge", () => {
    Object.defineProperty(window, "innerWidth", { writable: true, configurable: true, value: 1000 });
    Object.defineProperty(window, "innerHeight", { writable: true, configurable: true, value: 800 });

    const targetRect = {
      left: 980,
      right: 1000,
      top: 400,
      bottom: 420,
      width: 20,
      height: 20,
    } as DOMRect;

    render(<ChartHoverPortal hoveredPoint={{ x: 100, y: 0.9, stage: 1, checkpoint: mockCheckpoint }} targetRect={targetRect} />);

    const tooltip = screen.getByTestId("chart-tooltip");
    const leftVal = parseFloat(tooltip.style.left);
    expect(leftVal + 300).toBeLessThanOrEqual(1000 - 12 + 300);
  });

  it("flips vertical placement below point when near top edge", () => {
    Object.defineProperty(window, "innerWidth", { writable: true, configurable: true, value: 1000 });
    Object.defineProperty(window, "innerHeight", { writable: true, configurable: true, value: 800 });

    const targetRect = {
      left: 500,
      right: 520,
      top: 10,
      bottom: 30,
      width: 20,
      height: 20,
    } as DOMRect;

    render(<ChartHoverPortal hoveredPoint={{ x: 100, y: 0.9, stage: 1, checkpoint: mockCheckpoint }} targetRect={targetRect} />);

    const tooltip = screen.getByTestId("chart-tooltip");
    const topVal = parseFloat(tooltip.style.top);
    expect(topVal).toBeGreaterThanOrEqual(30);
  });

  it("renders detailed rolling promotion details and long blocking reasons", () => {
    const targetRect = {
      left: 500,
      right: 520,
      top: 400,
      bottom: 420,
      width: 20,
      height: 20,
    } as DOMRect;

    render(<ChartHoverPortal hoveredPoint={{ x: 100, y: 0.9, stage: 1, checkpoint: mockCheckpoint }} targetRect={targetRect} />);

    expect(screen.getByText(/🔴 NOT READY/i)).toBeInTheDocument();
    expect(screen.getByText(/Aggregate success is 88%; 90% required./i)).toBeInTheDocument();
    expect(screen.getByText(/Seed 41 failed in 3 of the last 5 evaluations./i)).toBeInTheDocument();
    expect(screen.getAllByText(/41/i).length).toBeGreaterThan(0);
  });

  it("handles viewport resize events gracefully", () => {
    const targetRect = {
      left: 500,
      right: 520,
      top: 400,
      bottom: 420,
      width: 20,
      height: 20,
    } as DOMRect;

    render(<ChartHoverPortal hoveredPoint={{ x: 100, y: 0.9, stage: 1, checkpoint: mockCheckpoint }} targetRect={targetRect} />);

    expect(screen.getByTestId("chart-tooltip")).toBeInTheDocument();

    Object.defineProperty(window, "innerWidth", { writable: true, configurable: true, value: 600 });
    fireEvent(window, new Event("resize"));

    const tooltip = screen.getByTestId("chart-tooltip");
    expect(tooltip).toBeInTheDocument();
  });

  it("renders redesigned bucket tooltip with TARGET MET banner when rate >= target", () => {
    const targetRect = { left: 500, right: 520, top: 400, bottom: 420, width: 20, height: 20 } as DOMRect;
    const mockBucket = {
      firstEpisode: 3364,
      lastEpisode: 3364,
      startTimestep: 1,
      endTimestep: 1,
      startTimestampMs: 1,
      endTimestampMs: 1,
      failureCount: 0,
      episodeCount: 1,
      successCount: 1,
      successRate: 90.0,
      avgSuccessfulSteps: 115,
      avgAllReward: 194.2,
      avgSheepPenned: 1.8,
      startTimestamp: "2026-08-12T14:45:51Z",
      endTimestamp: "2026-08-12T14:45:51Z",
      episodes: [{ curriculum_stage: 1 }] as any,
    };

    render(<ChartHoverPortal hoveredPoint={{ x: 3364, y: 90.0, stage: 1, bucket: mockBucket }} targetRect={targetRect} />);

    expect(screen.getByText(/Episode 3364/i)).toBeInTheDocument();
    expect(screen.getByText(/TARGET MET/i)).toBeInTheDocument();
    expect(screen.getByText(/Bucket performance met stage target/i)).toBeInTheDocument();
    expect(screen.getByRole("progressbar")).toBeInTheDocument();
  });

  it("renders STAGE IN PROGRESS banner when target is not yet met and no stagnation", () => {
    const targetRect = { left: 500, right: 520, top: 400, bottom: 420, width: 20, height: 20 } as DOMRect;
    const mockBucket = {
      firstEpisode: 100,
      lastEpisode: 105,
      startTimestep: 1,
      endTimestep: 1,
      startTimestampMs: 1,
      endTimestampMs: 1,
      failureCount: 3,
      episodeCount: 5,
      successCount: 2,
      successRate: 40.0,
      avgSuccessfulSteps: 80,
      avgAllReward: 85.0,
      avgSheepPenned: 1.2,
      startTimestamp: "2026-08-12T14:45:51Z",
      endTimestamp: "2026-08-12T14:45:51Z",
      episodes: [{ curriculum_stage: 1 }] as any,
    };

    render(<ChartHoverPortal hoveredPoint={{ x: 105, y: 40.0, stage: 1, bucket: mockBucket }} targetRect={targetRect} />);

    expect(screen.getByText(/STAGE IN PROGRESS/i)).toBeInTheDocument();
    expect(screen.getByText(/Model is actively training at Stage 1/i)).toBeInTheDocument();
  });

  it("renders STAGNATION DETECTED banner ONLY when multi-factor gauntlet is satisfied", () => {
    const targetRect = { left: 500, right: 520, top: 400, bottom: 420, width: 20, height: 20 } as DOMRect;
    const mockBucket = {
      firstEpisode: 1,
      lastEpisode: 120,
      startTimestep: 1,
      endTimestep: 1,
      startTimestampMs: 1,
      endTimestampMs: 1,
      failureCount: 100,
      episodeCount: 120,
      successCount: 20,
      successRate: 16.6,
      avgSuccessfulSteps: 50,
      avgAllReward: 20.0,
      avgSheepPenned: 0.3,
      startTimestamp: "2026-08-12T14:45:51Z",
      endTimestamp: "2026-08-12T14:45:51Z",
      episodes: [{ curriculum_stage: 1 }] as any,
    };

    render(<ChartHoverPortal hoveredPoint={{ x: 120, y: 16.6, stage: 1, bucket: mockBucket }} targetRect={targetRect} />);

    expect(screen.getByText(/STAGNATION DETECTED/i)).toBeInTheDocument();
    expect(screen.getByText(/Full stage telemetry confirms zero performance growth over 120 episodes/i)).toBeInTheDocument();
  });
});

import { render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi, beforeEach } from "vitest";
import { StageBottlenecksPanel } from "./StageBottlenecksPanel";
import * as apiModule from "../lib/api";
import type { StageBottleneckReport } from "../state/types";

const mockReport: StageBottleneckReport = {
  curriculum_stage: 3,
  total_episodes: 50,
  success_count: 20,
  timeout_count: 25,
  stopped_count: 5,
  corner_stuck_count: 18,
  overall_success_rate: 0.40,
  avg_steps: 450.2,
  avg_corner_time_pct: 0.55,
  avg_wall_time_pct: 0.25,
  earliest_timestamp: "2026-08-21T05:00:00Z",
  latest_timestamp: "2026-08-21T06:00:00Z",
  zone_stats: {
    top_left: {
      zone: "top_left",
      total: 15,
      wins: 1,
      win_rate: 0.0667,
      timeouts: 12,
      stopped: 2,
      trapped_at_end: 11,
      avg_steps: 580,
      avg_corner_pct: 0.75,
      avg_wall_pct: 0.15,
      is_corner: true,
      is_wall: false,
    },
    center: {
      zone: "center",
      total: 20,
      wins: 16,
      win_rate: 0.80,
      timeouts: 4,
      stopped: 0,
      trapped_at_end: 0,
      avg_steps: 220,
      avg_corner_pct: 0.05,
      avg_wall_pct: 0.10,
      is_corner: false,
      is_wall: false,
    },
  },
  pen_stats: {
    top_right: {
      placement: "top_right",
      total: 50,
      wins: 20,
      win_rate: 0.40,
      timeouts: 25,
      avg_steps: 450,
    },
  },
  setup_stats: {
    fixed_easy: {
      setup: "fixed_easy",
      total: 25,
      wins: 18,
      win_rate: 0.72,
      timeouts: 7,
      avg_steps: 320,
    },
    corner_cluster: {
      setup: "corner_cluster",
      total: 25,
      wins: 2,
      win_rate: 0.08,
      timeouts: 18,
      avg_steps: 580,
    },
  },
  terminal_failure_heatmap: {
    top_left: 14,
    bottom_left: 6,
  },
  insights: [
    {
      severity: "critical",
      type: "corner_entrapment",
      title: "General Corner Entrapment Bottleneck",
      message: "Episodes starting in corners fail 85% of the time.",
      metric: "6.7% vs 80.0%",
    },
  ],
};

describe("StageBottlenecksPanel", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it("renders stage summary statistics and bottleneck insights", async () => {
    vi.spyOn(apiModule, "loadStageDiagnostics").mockResolvedValue(mockReport);

    render(<StageBottlenecksPanel currentStage={3} />);

    expect(screen.getByText(/Stage Learning Bottlenecks & Spatial Heatmap/i)).toBeInTheDocument();

    await waitFor(() => {
      expect(screen.getAllByText("50").length).toBeGreaterThan(0); // total episodes
      expect(screen.getAllByText(/40%/).length).toBeGreaterThan(0); // overall win rate
      expect(screen.getAllByText("18").length).toBeGreaterThan(0); // corner stuck count
    });

    // Check insight card
    expect(screen.getByText(/General Corner Entrapment Bottleneck/i)).toBeInTheDocument();
    expect(screen.getByText(/Episodes starting in corners fail 85% of the time/i)).toBeInTheDocument();

    // Check zone grid item
    expect(screen.getByText("Top-Left Corner")).toBeInTheDocument();
    expect(screen.getByText("1 / 15 wins")).toBeInTheDocument();
  });
});

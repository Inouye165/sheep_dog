import { render, screen, waitFor, fireEvent } from "@testing-library/react";
import { describe, expect, it, vi, beforeEach } from "vitest";
import { StageHealthBanner } from "./StageHealthBanner";
import * as apiModule from "../lib/api";
import type { StageHealthSummary } from "../state/types";

const mockSummary: StageHealthSummary = {
  stage: 8,
  stage_title: "Stage 8 (3 Dogs, 4 Sheep, 108x78)",
  total_stage_checkpoints: 208,
  all_time_stage_success_rate: 0.711,
  recent_success_rate: 0.85,
  peak_stage_success_rate: 1.0,
  recent_avg_steps: 215.4,
  recent_avg_reward: 345.2,
  status: "green",
  status_label: "Healthy Learning · Surging",
  status_explanation: "The policy is progressing well on Stage 8 and hitting 90% peak benchmarks.",
  promotion_ready: false,
  promotion_status_text: "Promotion Candidate",
  failure_progress: {
    total_failures: 602,
    avg_penned_on_fail: 2.1,
    three_penned_pct: 0.377,
    two_penned_pct: 0.231,
    one_penned_pct: 0.211,
    zero_penned_pct: 0.181,
    closeness_score: 0.558,
  },
  seed_matrix: [
    {
      seed: 11,
      win_rate: 0.894,
      wins: 186,
      fails: 22,
      total: 208,
      status: "green",
      current_consecutive_fails: 0,
    },
    {
      seed: 67,
      win_rate: 0.312,
      wins: 65,
      fails: 143,
      total: 208,
      status: "red",
      current_consecutive_fails: 2,
    },
  ],
  recent_trajectory: [
    {
      pv: 2638,
      episode: 13825,
      success_rate: 0.9,
      steps: 209.8,
      reward: 373.5,
      mode: "quick",
      timestamp: "2026-08-31T07:18:00Z",
    },
  ],
  hyperparameter_audit: [
    {
      parameter: "farthest_sheep_progress_scale",
      current_value: 0.42,
      recommended_value: 0.55,
      status: "warn",
      note: "Recommended >= 0.55 for 4-sheep recovery.",
    },
  ],
  prescriptive_recommendations: [
    {
      type: "reward_tweak",
      title: "Boost Lone Straggler Approach Scale",
      description: "37%+ of failures leave exactly 1 sheep unpenned until timeout.",
      suggested_action: "Update CURRICULUM_REWARD_OVERRIDES[8]['farthest_sheep_progress_scale'] = 0.55",
      priority: "medium",
    },
  ],
};

describe("StageHealthBanner", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it("renders real-time health indicator, KPIs, and seed matrix", async () => {
    vi.spyOn(apiModule, "loadStageHealth").mockResolvedValue(mockSummary);

    render(<StageHealthBanner curriculumStage={8} isLiveTraining={true} />);

    await waitFor(() => {
      expect(screen.getByText("Healthy Learning · Surging")).toBeInTheDocument();
    });

    expect(screen.getByText("71.1%")).toBeInTheDocument();
    expect(screen.getByText("85.0%")).toBeInTheDocument();
    expect(screen.getByText("100%")).toBeInTheDocument();
    expect(screen.getByText("38%")).toBeInTheDocument();
    expect(screen.getByText("Seed Health:")).toBeInTheDocument();
    expect(screen.getByText("11")).toBeInTheDocument();
    expect(screen.getByText("67")).toBeInTheDocument();
  });

  it("opens diagnostic summary modal with recommendations when button is clicked", async () => {
    vi.spyOn(apiModule, "loadStageHealth").mockResolvedValue(mockSummary);

    render(<StageHealthBanner curriculumStage={8} isLiveTraining={true} />);

    await waitFor(() => {
      expect(screen.getByRole("button", { name: "Diagnose & Summary" })).toBeInTheDocument();
    });

    fireEvent.click(screen.getByRole("button", { name: "Diagnose & Summary" }));

    expect(
      screen.getByText("Stage 8 Diagnostic Audit & Recommendations")
    ).toBeInTheDocument();
    expect(screen.getByText("Boost Lone Straggler Approach Scale")).toBeInTheDocument();
    expect(
      screen.getByText("Update CURRICULUM_REWARD_OVERRIDES[8]['farthest_sheep_progress_scale'] = 0.55")
    ).toBeInTheDocument();
    expect(screen.getByText("farthest_sheep_progress_scale")).toBeInTheDocument();
  });
});

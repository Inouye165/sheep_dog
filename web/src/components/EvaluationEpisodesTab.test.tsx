import React from "react";
import { describe, it, expect, beforeEach, vi } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import "@testing-library/jest-dom";
import { EvaluationEpisodesTab } from "./EvaluationEpisodesTab";
import * as api from "../lib/api";
import type { EvaluationSummaryPayload } from "../state/types";

vi.mock("../lib/api", () => ({
  loadRecentEvaluations: vi.fn(),
  runLiveReplay: vi.fn(),
  fetchReplayById: vi.fn().mockResolvedValue(null),
  loadReplay: vi.fn().mockResolvedValue(null),
  pinEvaluation: vi.fn().mockResolvedValue({ success: true, evaluation_id: "eval-5000", pinned: true }),
}));

const mockEvaluations: EvaluationSummaryPayload[] = [
  {
    checkpoint_episode: 5000,
    policy_name: "neural_policy",
    curriculum_stage: 7,
    success_rate: 0.7,
    timeout_rate: 0.3,
    average_completion_steps: 250,
    average_sheep_penned: 3.2,
    average_reward: 150.5,
    evaluation_id: "eval-5000",
    records: [
      { seed: 11, success: true, steps: 180, sheep_penned: 4, stop_reason: "success", reward_total: 210 },
      { seed: 23, success: true, steps: 220, sheep_penned: 4, stop_reason: "success", reward_total: 195 },
      { seed: 37, success: false, steps: 980, sheep_penned: 2, stop_reason: "timeout", reward_total: -320 },
      { seed: 41, success: true, steps: 190, sheep_penned: 4, stop_reason: "success", reward_total: 205 },
      { seed: 53, success: false, steps: 980, sheep_penned: 1, stop_reason: "timeout", reward_total: -410, corner_time_pct: 0.45 },
      { seed: 59, success: true, steps: 240, sheep_penned: 4, stop_reason: "success", reward_total: 180 },
      { seed: 61, success: true, steps: 210, sheep_penned: 4, stop_reason: "success", reward_total: 200 },
      { seed: 67, success: false, steps: 980, sheep_penned: 3, stop_reason: "timeout", reward_total: -150 },
      { seed: 71, success: true, steps: 170, sheep_penned: 4, stop_reason: "success", reward_total: 225 },
      { seed: 73, success: true, steps: 195, sheep_penned: 4, stop_reason: "success", reward_total: 215 },
    ],
  },
  {
    checkpoint_episode: 4950,
    policy_name: "neural_policy",
    curriculum_stage: 7,
    success_rate: 0.6,
    timeout_rate: 0.4,
    average_completion_steps: 310,
    average_sheep_penned: 2.8,
    average_reward: 95.0,
    evaluation_id: "eval-4950",
    records: [
      { seed: 11, success: true, steps: 200, sheep_penned: 4, stop_reason: "success", reward_total: 190 },
      { seed: 23, success: false, steps: 980, sheep_penned: 2, stop_reason: "timeout", reward_total: -280 },
      { seed: 37, success: false, steps: 980, sheep_penned: 1, stop_reason: "timeout", reward_total: -400 },
      { seed: 41, success: true, steps: 210, sheep_penned: 4, stop_reason: "success", reward_total: 185 },
      { seed: 53, success: false, steps: 980, sheep_penned: 2, stop_reason: "timeout", reward_total: -350 },
      { seed: 59, success: true, steps: 230, sheep_penned: 4, stop_reason: "success", reward_total: 175 },
      { seed: 61, success: true, steps: 195, sheep_penned: 4, stop_reason: "success", reward_total: 200 },
      { seed: 67, success: false, steps: 980, sheep_penned: 2, stop_reason: "timeout", reward_total: -290 },
      { seed: 71, success: true, steps: 185, sheep_penned: 4, stop_reason: "success", reward_total: 210 },
      { seed: 73, success: true, steps: 205, sheep_penned: 4, stop_reason: "success", reward_total: 195 },
    ],
  },
];

describe("EvaluationEpisodesTab", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(api.loadRecentEvaluations).mockResolvedValue(mockEvaluations);
    vi.mocked(api.runLiveReplay).mockResolvedValue({
      frames: [
        {
          snapshot: {
            field_width: 96,
            field_height: 72,
            dogs: [{ position: { x: 10, y: 10 } }],
            sheep: [{ position: { x: 20, y: 20 }, penned: false }],
            pen: { origin: { x: 80, y: 10 }, width: 12, height: 12, opening: "bottom" },
          },
        },
        {
          snapshot: {
            field_width: 96,
            field_height: 72,
            dogs: [{ position: { x: 12, y: 12 } }],
            sheep: [{ position: { x: 22, y: 22 }, penned: false }],
            pen: { origin: { x: 80, y: 10 }, width: 12, height: 12, opening: "bottom" },
          },
        },
      ],
    } as any);
  });

  it("renders the benchmark inspector with evaluation selector", async () => {
    render(<EvaluationEpisodesTab currentStage={7} />);

    await waitFor(() => {
      expect(screen.getByText("Checkpoint #5000")).toBeInTheDocument();
      expect(screen.getByText("Checkpoint #4950")).toBeInTheDocument();
    });

    expect(screen.getByText("70% Pass")).toBeInTheDocument();
    expect(screen.getByText("60% Pass")).toBeInTheDocument();
  });

  it("renders all 10 pass/fail seed episode records", async () => {
    render(<EvaluationEpisodesTab currentStage={7} />);

    await waitFor(() => {
      expect(screen.getByText("Seed 11")).toBeInTheDocument();
      expect(screen.getByText("Seed 53")).toBeInTheDocument();
      expect(screen.getByText("Seed 73")).toBeInTheDocument();
    });

    // Check pass and fail badges
    const passBadges = screen.getAllByText("✓ PASS");
    const failBadges = screen.getAllByText("✗ FAIL");
    expect(passBadges.length).toBe(7);
    expect(failBadges.length).toBe(3);
  });

  it("filters episode records using outcome filter chips", async () => {
    render(<EvaluationEpisodesTab currentStage={7} />);

    await waitFor(() => {
      expect(screen.getByText("Seed 11")).toBeInTheDocument();
    });

    // Click Fail filter chip
    const failFilterChip = screen.getByRole("button", { name: /Fail \(3\)/i });
    fireEvent.click(failFilterChip);

    expect(screen.queryByText("Seed 11")).not.toBeInTheDocument();
    expect(screen.getByText("Seed 37")).toBeInTheDocument();
    expect(screen.getByText("Seed 53")).toBeInTheDocument();
    expect(screen.getByText("Seed 67")).toBeInTheDocument();

    // Click Pass filter chip
    const passFilterChip = screen.getByRole("button", { name: /Pass \(7\)/i });
    fireEvent.click(passFilterChip);

    expect(screen.getByText("Seed 11")).toBeInTheDocument();
    expect(screen.queryByText("Seed 53")).not.toBeInTheDocument();
  });

  it("switches active evaluation when clicking an evaluation card", async () => {
    render(<EvaluationEpisodesTab currentStage={7} />);

    await waitFor(() => {
      expect(screen.getByText("Checkpoint #5000")).toBeInTheDocument();
    });

    // Click on Checkpoint #4950
    const card4950 = screen.getByText("Checkpoint #4950").closest("button");
    expect(card4950).toBeTruthy();
    fireEvent.click(card4950!);

    await waitFor(() => {
      expect(screen.getByText("Episode Replay: Checkpoint #4950 · Seed 11")).toBeInTheDocument();
    });
  });

  it("allows selecting a seed episode to load its replay and scrub timeline", async () => {
    render(<EvaluationEpisodesTab currentStage={7} />);

    await waitFor(() => {
      expect(screen.getByText("Seed 53")).toBeInTheDocument();
    });

    // Click on Seed 53
    const seed53Card = screen.getByText("Seed 53").closest('[role="button"]');
    expect(seed53Card).toBeTruthy();
    fireEvent.click(seed53Card!);

    await waitFor(() => {
      expect(screen.getByText("Episode Replay: Checkpoint #5000 · Seed 53")).toBeInTheDocument();
    });

    // Check scrubber and play controls
    const playBtn = screen.getByRole("button", { name: /▶ Play/i });
    expect(playBtn).toBeInTheDocument();
  });

  it("renders all 10 seeds simultaneously in the 3x3+1 grid with context and master controller cards in row 4", async () => {
    render(<EvaluationEpisodesTab currentStage={7} />);

    await waitFor(() => {
      // All 10 seeds should be rendered as seed cards
      expect(screen.getByRole("button", { name: /Seed 11 Replay/i })).toBeInTheDocument();
      expect(screen.getByRole("button", { name: /Seed 23 Replay/i })).toBeInTheDocument();
      expect(screen.getByRole("button", { name: /Seed 37 Replay/i })).toBeInTheDocument();
      expect(screen.getByRole("button", { name: /Seed 41 Replay/i })).toBeInTheDocument();
      expect(screen.getByRole("button", { name: /Seed 53 Replay/i })).toBeInTheDocument();
      expect(screen.getByRole("button", { name: /Seed 59 Replay/i })).toBeInTheDocument();
      expect(screen.getByRole("button", { name: /Seed 61 Replay/i })).toBeInTheDocument();
      expect(screen.getByRole("button", { name: /Seed 67 Replay/i })).toBeInTheDocument();
      expect(screen.getByRole("button", { name: /Seed 71 Replay/i })).toBeInTheDocument();
      expect(screen.getByRole("button", { name: /Seed 73 Replay/i })).toBeInTheDocument();
    });

    // Verify row 4 empty areas: Context Card and Master Controller Card
    expect(screen.getByTestId("eval-context-card")).toBeInTheDocument();
    expect(screen.getByTestId("eval-master-card")).toBeInTheDocument();

    // Verify context card metrics
    expect(screen.getByText(/Evaluation Benchmark Context/i)).toBeInTheDocument();
    expect(screen.getByText("70% Pass Rate")).toBeInTheDocument();

    // Verify master controller controls
    expect(screen.getByText(/Master Playback \(All 10\)/i)).toBeInTheDocument();
    const masterScrubber = screen.getByLabelText(/Master Replay timeline scrubber/i);
    expect(masterScrubber).toBeInTheDocument();
  });

  it("synchronizes playback controls and master scrubber across all seeds", async () => {
    render(<EvaluationEpisodesTab currentStage={7} />);

    await waitFor(() => {
      expect(screen.getByTestId("eval-master-card")).toBeInTheDocument();
    });

    // Find master play button
    const masterPlayBtn = screen.getByRole("button", { name: /▶ Play/i });
    expect(masterPlayBtn).toBeInTheDocument();

    // Toggle play
    fireEvent.click(masterPlayBtn);
    expect(screen.getByRole("button", { name: /⏸ Pause/i })).toBeInTheDocument();

    // Toggle pause
    fireEvent.click(screen.getByRole("button", { name: /⏸ Pause/i }));
    expect(screen.getByRole("button", { name: /▶ Play/i })).toBeInTheDocument();

    // Master Scrubber can be adjusted
    const masterScrubber = screen.getByLabelText(/Master Replay timeline scrubber/i);
    fireEvent.change(masterScrubber, { target: { value: "1" } });
    expect(masterScrubber).toHaveValue("1");

    // Speed buttons
    const speed2xBtn = screen.getByRole("button", { name: "2x" });
    fireEvent.click(speed2xBtn);
    expect(speed2xBtn).toHaveClass("eval-speed-btn--active");
  });

  it("toggles between 10-seed grid view and single seed focus view", async () => {
    render(<EvaluationEpisodesTab currentStage={7} />);

    await waitFor(() => {
      expect(screen.getByRole("button", { name: /⊞ All 10 Seeds Grid/i })).toBeInTheDocument();
    });

    // Switch to Single Seed Focus
    const singleFocusBtn = screen.getByRole("button", { name: /⧉ Single Seed Focus/i });
    fireEvent.click(singleFocusBtn);

    // Should now show single seed visualizer
    expect(screen.getByText(/Formal Evaluation Benchmark Inspector/i)).toBeInTheDocument();
    expect(screen.queryByTestId("eval-master-card")).not.toBeInTheDocument();

    // Switch back to 10-seed grid
    const gridBtn = screen.getByRole("button", { name: /⊞ All 10 Seeds Grid/i });
    fireEvent.click(gridBtn);

    expect(screen.getByTestId("eval-master-card")).toBeInTheDocument();
    expect(screen.getByTestId("eval-context-card")).toBeInTheDocument();
  });

  it("allows pinning and unpinning evaluation replays", async () => {
    render(<EvaluationEpisodesTab currentStage={7} />);

    await waitFor(() => {
      expect(screen.getByRole("button", { name: /📌 Pin Replays/i })).toBeInTheDocument();
    });

    const pinBtn = screen.getByRole("button", { name: /📌 Pin Replays/i });
    fireEvent.click(pinBtn);

    await waitFor(() => {
      expect(api.pinEvaluation).toHaveBeenCalledWith("eval-5000", true);
      expect(screen.getByRole("button", { name: /📌 Replays Pinned/i })).toBeInTheDocument();
    });
  });
});

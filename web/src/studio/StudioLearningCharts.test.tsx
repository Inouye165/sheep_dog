import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { StudioLearningCharts } from "./StudioLearningCharts";
import type { CheckpointEntry } from "../state/types";

function makeEntry(
  episode: number,
  stage: number,
  journey: string,
  runId: string,
  successRate: number,
): CheckpointEntry {
  return {
    checkpoint_episode: episode,
    curriculum_stage: stage,
    journey,
    run_id: runId,
    success_rate: successRate,
    timeout_rate: 1 - successRate,
    average_completion_steps: 150,
    average_completion_seconds: 15,
    average_sheep_penned: successRate * 6,
    average_reward: 100,
    checkpoint: `chk_${episode}.zip`,
    evaluation: `eval_${episode}.json`,
    replay: `rep_${episode}.json`,
    records: [],
  };
}

describe("StudioLearningCharts — Current journey isolation", () => {
  it("shows only current journey Stage 24 evals (ep 1019-1029, 40%) and never archived ep 2940-2950 (100%)", () => {
    const archived = Array.from({ length: 11 }, (_, i) =>
      makeEntry(
        2940 + i,
        24,
        "journey-20261124-172927",
        "run_stage9_from_stage8_20261119_063250_40a1",
        1.0,
      ),
    );
    const current = Array.from({ length: 11 }, (_, i) =>
      makeEntry(1019 + i, 24, "current", "run_20261124_093542_33c5", 0.4),
    );

    render(
      <StudioLearningCharts
        rawEpisodes={[]}
        checkpoints={[...archived, ...current]}
        effectiveStage={24}
        activeRunId="run_20261124_093542_33c5"
      />,
    );

    expect(screen.getByText(/Showing Last 11 of 11 Evals/)).toBeTruthy();
    expect(screen.getByText(/Latest Ep 1,029: 40% \(4\/10\)/)).toBeTruthy();
    expect(screen.queryByText(/2,950/)).toBeNull();
    expect(screen.queryByText(/2,940/)).toBeNull();
    // Recent 5 Avg and Peak reflect only the current 40% evals, not archived 100% evals
    const summary = screen.getByText(/Recent 5 Avg:/).closest("span");
    expect(summary?.textContent).toContain("40%");
    expect(summary?.textContent).not.toContain("100%");
  });
});

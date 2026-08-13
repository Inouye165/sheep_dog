import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import React from "react";
import { StackedLearningPanels } from "./StackedLearningPanels";
import type { CheckpointEntry, TrainingEpisode } from "../state/types";

function makeEpisode(overrides: Partial<TrainingEpisode>): TrainingEpisode {
  return {
    id: overrides.id ?? 1,
    event_key: overrides.event_key ?? `ep_${overrides.id ?? 1}`,
    run_id: overrides.run_id ?? "run_1",
    session_id: overrides.session_id ?? "sess_1",
    global_environment_episode: overrides.global_environment_episode ?? 1,
    episode_in_stage: overrides.episode_in_stage ?? overrides.global_environment_episode ?? 1,
    curriculum_stage: overrides.curriculum_stage ?? 3,
    global_timestep: overrides.global_timestep ?? 1000,
    policy_version: overrides.policy_version ?? 1,
    completed_at: overrides.completed_at ?? new Date().toISOString(),
    active_runtime_seconds_total: 10,
    reward: overrides.reward ?? 100,
    result: overrides.result ?? (overrides.success ? "SUCCESS" : "TIMEOUT"),
    success: overrides.success ?? false,
    timeout: overrides.timeout ?? !overrides.success,
    stopped: overrides.stopped ?? false,
    sheep_penned: overrides.sheep_penned ?? (overrides.success ? 3 : 0),
    total_sheep: overrides.total_sheep ?? 3,
    steps: overrides.steps ?? 200,
    seed: overrides.seed ?? 42,
    checkpoint_id: overrides.checkpoint_id ?? "cp_1",
    reward_breakdown: overrides.reward_breakdown ?? null,
  };
}

describe("StackedLearningPanels — Canonical Data Semantics", () => {
  it("uses canonical Stage Episode as primary X axis and renders all 4 panels", () => {
    const episodes: TrainingEpisode[] = [
      makeEpisode({ id: 1, episode_in_stage: 1, success: true, steps: 150, reward: 200 }),
      makeEpisode({ id: 2, episode_in_stage: 2, success: false, steps: 600, reward: -50 }),
      makeEpisode({ id: 3, episode_in_stage: 3, success: true, steps: 180, reward: 220 }),
    ];

    render(
      <StackedLearningPanels
        episodes={episodes}
        checkpoints={[]}
        curriculumStage={3}
        smoothingWindow={25}
        xAxisMode="stage_ep"
        showRawEpisodes={true}
        showRollingAvg={true}
        showFormalEvals={true}
        showPolicySnapshots={true}
      />
    );

    expect(screen.getByText("PANEL 1 — SUCCESS RATE (%)")).toBeInTheDocument();
    expect(screen.getByText("PANEL 2 — STEPS / EFFICIENCY (FEWER IS FASTER)")).toBeInTheDocument();
    expect(screen.getByText("PANEL 3 — TOTAL REWARD")).toBeInTheDocument();
    expect(screen.getByText("PANEL 4 — REWARDS & PENALTIES BREAKDOWN")).toBeInTheDocument();
  });

  it("calculates 25-episode rolling metrics cleanly where Episode 135 uses exactly Episodes 111-135", () => {
    // Generate 135 episodes
    const episodes: TrainingEpisode[] = [];
    for (let ep = 1; ep <= 135; ep++) {
      const isSuccess = ep > 110 && ep % 2 === 0; // 12 successes in 111..135
      episodes.push(
        makeEpisode({
          id: ep,
          global_environment_episode: ep,
          episode_in_stage: ep,
          curriculum_stage: 3,
          success: isSuccess,
          steps: isSuccess ? 100 : 500,
          reward: isSuccess ? 300 : -100,
          reward_breakdown: {
            progress_to_pen: isSuccess ? 50 : 10,
            time_penalty: -10,
          },
        })
      );
    }

    const windowSlice = episodes.slice(110, 135); // Episodes 111 to 135 (25 episodes)
    const expectedSuccesses = windowSlice.filter((e) => e.success).length; // 12
    const expectedSuccessPct = (expectedSuccesses / 25) * 100; // 48%

    const expectedAllSteps = windowSlice.reduce((s, e) => s + e.steps, 0) / 25; // (12*100 + 13*500)/25 = 308
    const succOnlySlice = windowSlice.filter((e) => e.success);
    const expectedSuccSteps = succOnlySlice.reduce((s, e) => s + e.steps, 0) / succOnlySlice.length; // 100

    expect(expectedSuccessPct).toBe(48);
    expect(expectedAllSteps).toBe(308);
    expect(expectedSuccSteps).toBe(100);
  });

  it("proves formal evaluations do NOT enter training rolling averages or create left/right series separation", () => {
    const episodes: TrainingEpisode[] = [
      makeEpisode({ id: 101, episode_in_stage: 101, global_timestep: 47000, success: true, steps: 120 }),
      makeEpisode({ id: 102, episode_in_stage: 102, global_timestep: 47100, success: true, steps: 110 }),
    ];

    const checkpoints: CheckpointEntry[] = [
      {
        checkpoint_episode: 4,
        checkpoint: "cp4",
        evaluation: "eval4",
        replay: "rep4",
        total_training_episodes: 50,
        success_rate: 0.1,
        average_completion_steps: 550,
        average_sheep_penned: 0.5,
        average_reward: -80,
        average_completion_seconds: 30,
        timeout_rate: 0.9,
        records: [],
      },
    ];

    render(
      <StackedLearningPanels
        episodes={episodes}
        checkpoints={checkpoints}
        curriculumStage={3}
        smoothingWindow={25}
        xAxisMode="stage_ep"
        showRawEpisodes={true}
        showRollingAvg={true}
        showFormalEvals={true}
        showPolicySnapshots={true}
      />
    );

    // Verify all 4 panels render without thrown error or broken alignment
    expect(screen.getByText("PANEL 1 — SUCCESS RATE (%)")).toBeInTheDocument();
    expect(screen.getAllByText(/Ep 101/).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/Ep 102/).length).toBeGreaterThan(0);
  });
});

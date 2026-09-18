import { render, screen, fireEvent } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { DiagnosticsPanel } from "./DiagnosticsPanel";
import { TrainingPanel } from "./TrainingPanel";
import type { CheckpointIndex, TrainingStatus } from "../state/types";

describe("DiagnosticsPanel & TrainingPanel Promotion Readiness Integrity", () => {
  it("does NOT recommend promotion when Stage 25 is at 60% success (below 90% target)", () => {
    const stage25Checkpoints = Array.from({ length: 10 }, (_, i) => ({
      checkpoint_id: `chk_25_${i + 1}`,
      curriculum_stage: 25,
      checkpoint_episode: (i + 1) * 50,
      policy_version: i + 1,
      global_timestep: 105000000 + i * 100000,
      evaluation_seeds: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
      success_rate: i < 5 ? 0.1 : 0.6,
      timeout_rate: i < 5 ? 0.5 : 0.2,
      average_completion_steps: i < 5 ? 1800 : 1200,
      average_completion_seconds: 25,
      average_sheep_penned: i < 5 ? 1 : 3,
      average_reward: i < 5 ? 50 : 200,
      average_distance_to_pen: 5,
      average_flock_spread: 3,
      records: [],
      recorded_at: "2026-09-17T06:00:00Z",
      policy_name: "neural_policy",
      trainer_type: "maskable_ppo",
      policy_type: "neural",
      policy_mode: "neural_only",
      replay_mode: "truthful",
      total_training_episodes: (i + 1) * 50,
      policy_state_path: "path",
      checkpoint: `chk_${(i + 1) * 50}.json`,
      evaluation: `eval_${(i + 1) * 50}.json`,
      replay: `rep_${(i + 1) * 50}.json`,
    }));

    const mockIndex: CheckpointIndex = {
      checkpoints: stage25Checkpoints,
    };

    const mockStatus: TrainingStatus = {
      running: true,
      current_stage_curriculum_stage: 25,
      curriculum_stage: 25,
      total_episodes_trained: 500,
      total_timesteps: 106000000,
      current_global_timestep: 106000000,
      auto_promote_gate: {
        ready: false,
        decision: "hold",
        status_text: "COLLECTING EVIDENCE",
        reason: "Success rate (60%) is below 90% promotion target (0/6 qualifying evaluations)",
        blocking_reasons: ["Success rate below 90% gate"],
        stage: 25,
        success_threshold: 0.9,
        window_size: 8,
        formal_evaluations_available: 2,
        formal_evaluations_required: 6,
        qualified_evaluations: 0,
        qualified_evaluations_required: 6,
        recent_average_success: 0.35,
        persistent_seed_failure: false,
        blocking_seed: null,
        blocking_seed_consecutive_failures: 0,
        blocking_seeds: [],
        minimum_required_evaluations: 6,
        minimum_seed_trials: 60,
        total_seed_trials: 20,
        total_successes: 7,
        aggregate_success_rate: 0.35,
        aggregate_timeout_rate: 0.35,
        latest_success_rate: 0.6,
        recent_qualifying_checkpoints: 0,
        recent_checkpoints_considered: 2,
        latest_floor_passed: false,
        reward_guard_passed: true,
        seed_consistency_passed: false,
        seed_count: 10,
        success_count: 6,
        best_success: 0.6,
        best_reward: 200,
        seed_gate_ok: false,
        success_rate_ok: false,
        timeout_ok: true,
        reward_close_ok: true,
        qualified_streak: 0,
        min_qualified_streak: 6,
        seed_gate_hits: 0,
        min_seed_gate_hits: 6,
        seed_gate_target_met: false,
        full_success_hits: 0,
        min_full_success_hits: 0,
        full_success_target_met: true,
        full_success_rate_threshold: 0.999,
        max_timeout_rate: 0.1,
        reward_tolerance_ratio: 0.05,
        step_efficiency_improving: false,
        step_efficiency_delta_pct: null,
        recent_avg_steps: 1200,
      },
    };

    render(
      <DiagnosticsPanel
        checkpointIndex={mockIndex}
        trainingStatus={mockStatus}
        effectiveCurriculumStage={25}
        bestCheckpointEpisode={500}
        initialEpisodes={[]}
        initialStageScope="current"
      />,
    );

    // Switch to Health tab
    const healthTab = screen.getByRole("tab", { name: /health/i });
    fireEvent.click(healthTab);

    // Verify it does NOT claim "Promote to the next stage" or "Stage 25 ready"
    expect(screen.queryByText("Promote to the next stage")).toBeNull();
    expect(screen.queryByText("Stage 25 ready")).toBeNull();

    // Verify it reports performance is improving towards the 90% bar
    expect(screen.getByText("Performance improving")).toBeInTheDocument();
    expect(screen.getByText(/towards the 90% promotion bar/i)).toBeInTheDocument();

    // Verify Latest Success card shows promo bar 90% (not 50%)
    expect(screen.getByText(/promo bar 90%/i)).toBeInTheDocument();
  });

  it("reports 'Accumulating qualified streak' when at 90% but required 6 evals not yet reached", () => {
    const stage25Checkpoints = Array.from({ length: 10 }, (_, i) => ({
      checkpoint_id: `chk_25_${i + 1}`,
      curriculum_stage: 25,
      checkpoint_episode: (i + 1) * 50,
      policy_version: i + 1,
      global_timestep: 105000000 + i * 100000,
      evaluation_seeds: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
      success_rate: i === 9 ? 0.9 : 0.6,
      timeout_rate: 0.1,
      average_completion_steps: 1000,
      average_completion_seconds: 20,
      average_sheep_penned: 3,
      average_reward: 250,
      average_distance_to_pen: 5,
      average_flock_spread: 3,
      records: [],
      recorded_at: "2026-09-17T06:00:00Z",
      policy_name: "neural_policy",
      trainer_type: "maskable_ppo",
      policy_type: "neural",
      policy_mode: "neural_only",
      replay_mode: "truthful",
      total_training_episodes: (i + 1) * 50,
      policy_state_path: "path",
      checkpoint: `chk_${(i + 1) * 50}.json`,
      evaluation: `eval_${(i + 1) * 50}.json`,
      replay: `rep_${(i + 1) * 50}.json`,
    }));

    const mockIndex: CheckpointIndex = {
      checkpoints: stage25Checkpoints,
    };

    const mockStatus: TrainingStatus = {
      running: true,
      current_stage_curriculum_stage: 25,
      curriculum_stage: 25,
      total_episodes_trained: 500,
      total_timesteps: 106000000,
      current_global_timestep: 106000000,
      auto_promote_gate: {
        ready: false,
        decision: "hold",
        status_text: "ACCUMULATING STREAK",
        reason: "Accumulating qualified evaluations (1/6)",
        blocking_reasons: ["Need 5 more qualified evaluations"],
        stage: 25,
        success_threshold: 0.9,
        window_size: 8,
        formal_evaluations_available: 10,
        formal_evaluations_required: 6,
        qualified_evaluations: 1,
        qualified_evaluations_required: 6,
        recent_average_success: 0.9,
        persistent_seed_failure: false,
        blocking_seed: null,
        blocking_seed_consecutive_failures: 0,
        blocking_seeds: [],
        minimum_required_evaluations: 6,
        minimum_seed_trials: 60,
        total_seed_trials: 100,
        total_successes: 63,
        aggregate_success_rate: 0.63,
        aggregate_timeout_rate: 0.1,
        latest_success_rate: 0.9,
        recent_qualifying_checkpoints: 1,
        recent_checkpoints_considered: 10,
        latest_floor_passed: true,
        reward_guard_passed: true,
        seed_consistency_passed: true,
        seed_count: 10,
        success_count: 9,
        best_success: 0.9,
        best_reward: 250,
        seed_gate_ok: true,
        success_rate_ok: true,
        timeout_ok: true,
        reward_close_ok: true,
        qualified_streak: 1,
        min_qualified_streak: 6,
        seed_gate_hits: 1,
        min_seed_gate_hits: 6,
        seed_gate_target_met: false,
        full_success_hits: 0,
        min_full_success_hits: 0,
        full_success_target_met: true,
        full_success_rate_threshold: 0.999,
        max_timeout_rate: 0.1,
        reward_tolerance_ratio: 0.05,
        step_efficiency_improving: false,
        step_efficiency_delta_pct: null,
        recent_avg_steps: 1000,
      },
    };

    render(
      <DiagnosticsPanel
        checkpointIndex={mockIndex}
        trainingStatus={mockStatus}
        effectiveCurriculumStage={25}
        bestCheckpointEpisode={500}
        initialEpisodes={[]}
        initialStageScope="current"
      />,
    );

    const healthTab = screen.getByRole("tab", { name: /health/i });
    fireEvent.click(healthTab);

    // Should NOT say promote yet
    expect(screen.queryByText("Promote to the next stage")).toBeNull();
    expect(screen.queryByText("Stage 25 ready")).toBeNull();

    // Should report accumulating streak
    expect(screen.getByText(/Accumulating qualified streak \(1\/6\)/i)).toBeInTheDocument();
  });

  it("accurately reflects Stage 25 90% target in TrainingPanel promotion locks", () => {
    render(
      <TrainingPanel
        episodes={100}
        fastMode
        enableInstincts={false}
        curriculumStage={25}
        maxCurriculumStage={32}
        debugRewardBreakdown={false}
        autoPromote
        autoPromoteThreshold={0.9}
        autoPromoteStagesCompleted={24}
        running={false}
        clearing={false}
        batchCompletedEpisodes={0}
        batchTotalEpisodes={100}
        currentEpisode={null}
        totalEpisodesTrained={500}
        stageHistory={{}}
        grandTotalEpisodes={500}
        phase="idle"
        message="Idle"
        error={null}
        successRate={0.6}
        onEpisodesChange={vi.fn()}
        onFastModeChange={vi.fn()}
        onEnableInstinctsChange={vi.fn()}
        onCurriculumStageChange={vi.fn()}
        onDebugRewardBreakdownChange={vi.fn()}
        onAutoPromoteChange={vi.fn()}
        onStartTraining={vi.fn()}
        onPauseTraining={vi.fn()}
        onStopTraining={vi.fn()}
        onResumeTraining={vi.fn()}
        onClearTraining={vi.fn()}
        onResetJourney={vi.fn()}
        onPromote={vi.fn()}
      />,
    );

    // Must lock promotion based on 90% (not 50%)
    expect(screen.getByText(/Promotion locked until Stage 25 reaches ≥ 90% success/i)).toBeInTheDocument();
    expect(screen.getByText(/60% success — target ≥ 90% to promote/i)).toBeInTheDocument();
    expect(screen.queryByText(/ready to promote/i)).toBeNull();
  });
});

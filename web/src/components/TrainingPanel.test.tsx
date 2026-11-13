import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { TrainingPanel } from "./TrainingPanel";

describe("TrainingPanel", () => {
  it("shows redesigned stage labels and uses maxCurriculumStage for manual input", () => {
    const { rerender } = render(
      <TrainingPanel
        episodes={100}
        fastMode
        enableInstincts={false}
        curriculumStage={6}
        maxCurriculumStage={8}
        debugRewardBreakdown={false}
        autoPromote
        autoPromoteThreshold={0.5}
        autoPromoteStagesCompleted={0}
        running={false}
        clearing={false}
        batchCompletedEpisodes={0}
        batchTotalEpisodes={100}
        currentEpisode={null}
        totalEpisodesTrained={0}
        stageHistory={{}}
        grandTotalEpisodes={0}
        phase="idle"
        message="Idle"
        error={null}
        successRate={null}
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

    expect(screen.getByText("2 dogs · 3 sheep · tiny nearby stray starts")).toBeInTheDocument();

    rerender(
      <TrainingPanel
        episodes={100}
        fastMode
        enableInstincts={false}
        curriculumStage={7}
        maxCurriculumStage={8}
        debugRewardBreakdown={false}
        autoPromote
        autoPromoteThreshold={0.5}
        autoPromoteStagesCompleted={0}
        running={false}
        clearing={false}
        batchCompletedEpisodes={0}
        batchTotalEpisodes={100}
        currentEpisode={null}
        totalEpisodesTrained={0}
        stageHistory={{}}
        grandTotalEpisodes={0}
        phase="idle"
        message="Idle"
        error={null}
        successRate={null}
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

    expect(screen.getByText("2 dogs · 4 sheep · early nearby stray collection")).toBeInTheDocument();

    rerender(
      <TrainingPanel
        episodes={100}
        fastMode
        enableInstincts={false}
        curriculumStage={8}
        maxCurriculumStage={9}
        debugRewardBreakdown={false}
        autoPromote
        autoPromoteThreshold={0.5}
        autoPromoteStagesCompleted={0}
        running={false}
        clearing={false}
        batchCompletedEpisodes={0}
        batchTotalEpisodes={100}
        currentEpisode={null}
        totalEpisodesTrained={0}
        stageHistory={{}}
        grandTotalEpisodes={0}
        phase="idle"
        message="Idle"
        error={null}
        successRate={null}
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

    expect(screen.getByText("3 dogs · 4 sheep · nearby stray emphasis")).toBeInTheDocument();
    const stageInput = screen.getByLabelText("Stage (manual)");
    expect(stageInput).toHaveAttribute("max", "9");
  });

  it("renders disabled button and starting text when isStartingTraining is true", () => {
    render(
      <TrainingPanel
        episodes={100}
        fastMode
        enableInstincts={false}
        curriculumStage={1}
        maxCurriculumStage={8}
        debugRewardBreakdown={false}
        autoPromote
        autoPromoteThreshold={0.5}
        autoPromoteStagesCompleted={0}
        running={false}
        clearing={false}
        isStartingTraining={true}
        batchCompletedEpisodes={0}
        batchTotalEpisodes={100}
        currentEpisode={null}
        totalEpisodesTrained={0}
        stageHistory={{}}
        grandTotalEpisodes={0}
        phase="idle"
        message="Idle"
        error={null}
        successRate={null}
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

    const button = screen.getByRole("button", { name: /Starting training\.\.\./i });
    expect(button).toBeInTheDocument();
    expect(button).toBeDisabled();
    expect(screen.getAllByText("Starting training...").length).toBeGreaterThanOrEqual(1);
  });

  it("renders clear episode count progress labels and checkpoint counts", () => {
    render(
      <TrainingPanel
        episodes={225}
        fastMode
        enableInstincts={false}
        curriculumStage={8}
        maxCurriculumStage={32}
        debugRewardBreakdown={false}
        autoPromote
        autoPromoteThreshold={0.5}
        autoPromoteStagesCompleted={0}
        running={true}
        clearing={false}
        batchCompletedEpisodes={16}
        batchTotalEpisodes={225}
        batchCompletedSegments={2}
        batchTotalSegments={29}
        currentEpisode={5110}
        totalEpisodesTrained={5110}
        stageHistory={{}}
        grandTotalEpisodes={5110}
        phase="running"
        message="Learning neural policy..."
        error={null}
        successRate={null}
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

    expect(screen.getByLabelText("Batch Size (Episodes to Train)")).toHaveValue(225);
    expect(screen.getByText("16 / 225 Episodes (7%)")).toBeInTheDocument();
    expect(screen.getByText(/Checkpoint 3 of 29/i)).toBeInTheDocument();
  });

  it("renders auto-promote criteria popover on hover with stage-appropriate thresholds", () => {
    const { rerender } = render(
      <TrainingPanel
        episodes={100}
        fastMode
        enableInstincts={false}
        curriculumStage={1}
        maxCurriculumStage={5}
        debugRewardBreakdown={false}
        autoPromote
        autoPromoteStagesCompleted={0}
        autoPromoteGate={{
          decision: "pending",
          reason: "Collecting evidence",
          success_threshold: 0.80,
          formal_evaluations_available: 3,
          formal_evaluations_required: 6,
          qualified_evaluations: 2,
          qualified_evaluations_required: 5,
          recent_average_success: 0.75,
          persistent_seed_failure: false,
        }}
        running={false}
        clearing={false}
        batchCompletedEpisodes={0}
        batchTotalEpisodes={100}
        currentEpisode={null}
        totalEpisodesTrained={0}
        stageHistory={{}}
        grandTotalEpisodes={0}
        phase="idle"
        message="Idle"
        error={null}
        successRate={null}
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

    const trigger = screen.getByText("Auto-promote stages").closest(".auto-promote-popover-anchor")!;
    expect(trigger).toBeInTheDocument();

    // Trigger hover
    fireEvent.mouseEnter(trigger);

    expect(screen.getByTestId("auto-promote-popover")).toBeInTheDocument();
    expect(screen.getByText("Auto Promotion Criteria")).toBeInTheDocument();
    expect(screen.getByText("Stage target: 80% success")).toBeInTheDocument();
    expect(screen.getByText(/At least 75% of recent evaluations must meet 80%/i)).toBeInTheDocument();
    expect(screen.getByText("Status: COLLECTING EVIDENCE")).toBeInTheDocument();
    expect(screen.getByText(/3 formal evaluations available; minimum 6 required/i)).toBeInTheDocument();

    // Test Stage 2+ with ready gate
    rerender(
      <TrainingPanel
        episodes={100}
        fastMode
        enableInstincts={false}
        curriculumStage={2}
        maxCurriculumStage={5}
        debugRewardBreakdown={false}
        autoPromote
        autoPromoteStagesCompleted={0}
        autoPromoteGate={{
          ready: true,
          decision: "promote_ready",
          reason: "Promotion criteria met",
          success_threshold: 0.90,
          formal_evaluations_available: 6,
          formal_evaluations_required: 6,
          qualified_evaluations: 5,
          qualified_evaluations_required: 5,
          recent_average_success: 0.92,
          persistent_seed_failure: false,
        }}
        running={false}
        clearing={false}
        batchCompletedEpisodes={0}
        batchTotalEpisodes={100}
        currentEpisode={null}
        totalEpisodesTrained={0}
        stageHistory={{}}
        grandTotalEpisodes={0}
        phase="idle"
        message="Idle"
        error={null}
        successRate={null}
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

    expect(screen.getByText("Stage target: 90% success")).toBeInTheDocument();
    expect(screen.getByText("Status: READY TO PROMOTE")).toBeInTheDocument();
    expect(screen.getByText("All formal evaluation criteria met.")).toBeInTheDocument();

    // Test persistent seed failure
    rerender(
      <TrainingPanel
        episodes={100}
        fastMode
        enableInstincts={false}
        curriculumStage={2}
        maxCurriculumStage={5}
        debugRewardBreakdown={false}
        autoPromote
        autoPromoteStagesCompleted={0}
        autoPromoteGate={{
          ready: false,
          decision: "hold",
          reason: "Seed 7 failed in 3 consecutive evaluations",
          success_threshold: 0.90,
          formal_evaluations_available: 8,
          formal_evaluations_required: 6,
          qualified_evaluations: 5,
          qualified_evaluations_required: 6,
          recent_average_success: 0.86,
          persistent_seed_failure: true,
          blocking_seed: 7,
          blocking_seed_consecutive_failures: 3,
        }}
        running={false}
        clearing={false}
        batchCompletedEpisodes={0}
        batchTotalEpisodes={100}
        currentEpisode={null}
        totalEpisodesTrained={0}
        stageHistory={{}}
        grandTotalEpisodes={0}
        phase="idle"
        message="Idle"
        error={null}
        successRate={null}
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

    expect(screen.getByText("Seed 7 — 3 consecutive failures")).toBeInTheDocument();
    expect(screen.getByText("Status: NOT READY")).toBeInTheDocument();

    // Trigger unhover
    fireEvent.mouseLeave(trigger);
    expect(screen.queryByTestId("auto-promote-popover")).not.toBeInTheDocument();
  });

  it("renders progress bar with percentage and without spinner when training is active", () => {
    render(
      <TrainingPanel
        episodes={500}
        fastMode={false}
        enableInstincts={true}
        curriculumStage={25}
        maxCurriculumStage={32}
        debugRewardBreakdown={false}
        autoPromote={false}
        running={true}
        clearing={false}
        batchCompletedEpisodes={326}
        batchTotalEpisodes={500}
        currentEpisode={1326}
        totalEpisodesTrained={1326}
        stageHistory={{}}
        grandTotalEpisodes={1326}
        phase="running"
        message="Training active"
        error={null}
        successRate={null}
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

    const progressBar = screen.getByRole("progressbar", { name: "Training batch progress" });
    expect(progressBar).toBeInTheDocument();
    expect(progressBar).toHaveAttribute("aria-valuenow", "65");
    expect(screen.getByText("Training batch in progress...")).toBeInTheDocument();
    expect(screen.getByText("326 / 500 Episodes (65%)")).toBeInTheDocument();

    // Verify there is no spinner inside the progress bar
    expect(progressBar.querySelector(".spinner")).toBeNull();
  });

  it("computes and displays throughput rate and ETA from training start time", () => {
    // 326 episodes over 65 seconds = ~5.0 ep/s, remaining 174 episodes => ~35s ETA
    const startTime = new Date(Date.now() - 65_000).toISOString();
    render(
      <TrainingPanel
        episodes={500}
        fastMode={false}
        enableInstincts={true}
        curriculumStage={25}
        maxCurriculumStage={32}
        debugRewardBreakdown={false}
        autoPromote={false}
        running={true}
        clearing={false}
        batchCompletedEpisodes={326}
        batchTotalEpisodes={500}
        currentEpisode={1326}
        totalEpisodesTrained={1326}
        stageHistory={{}}
        grandTotalEpisodes={1326}
        phase="running"
        message="Training active"
        error={null}
        successRate={null}
        trainingStatus={{
          running: true,
          fast_mode: false,
          enable_instinct_rewards: true,
          debug_reward_breakdown: false,
          curriculum_stage: 25,
          requested_episodes: 500,
          completed_episodes: 326,
          batch_total_episodes: 500,
          batch_completed_episodes: 326,
          total_episodes_trained: 1326,
          stage_history: {},
          grand_total_episodes: 1326,
          current_episode: 1326,
          checkpoint_episode: 1300,
          latest_checkpoint_episode: 1300,
          latest_seed: 1,
          latest_replay_path: null,
          best_score: null,
          latest_success_rate: null,
          latest_avg_sheep_penned: null,
          latest_avg_reward: null,
          latest_timeout_rate: null,
          latest_stopped_rate: null,
          latest_avg_no_progress_steps: null,
          latest_avg_distance_to_pen: null,
          latest_avg_flock_spread: null,
          latest_avg_farthest_distance_to_pen: null,
          latest_avg_farthest_distance_to_flock_center: null,
          phase: "running",
          message: "Training active",
          error: null,
          starting_episode: 1000,
          training_start_time: startTime,
        }}
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

    const progressBar = screen.getByRole("progressbar", { name: "Training batch progress" });
    expect(progressBar).toBeInTheDocument();
    expect(screen.getByText(/ep\/s/)).toBeInTheDocument();
    expect(screen.getByText(/ETA/)).toBeInTheDocument();
  });
});



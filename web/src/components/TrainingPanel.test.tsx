import { render, screen } from "@testing-library/react";
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
});



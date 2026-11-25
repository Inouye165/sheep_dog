import { fireEvent, render, screen, within } from "@testing-library/react";
import { describe, it, expect, vi } from "vitest";
import { ProgressView } from "./ProgressView";
import type { CheckpointEntry, SegmentTiming, StageTrainingReport, TrainingStatus } from "../state/types";
import { loadStageTrainingReport } from "../lib/api";

vi.mock("../lib/api", () => ({
  loadStageTrainingReport: vi.fn(),
  loadStageTimings: vi.fn().mockResolvedValue({ schema_version: 1, records: [], comparison: [] }),
}));

vi.mock("../components/EarlyWarningsPanel", () => ({
  EarlyWarningsPanel: () => <div data-testid="early-warnings-panel" />,
}));

vi.mock("../components/ResultsPanel", () => ({
  ResultsPanel: () => <div data-testid="results-panel" />,
}));

const studioLearningChartsSpy = vi.fn();

vi.mock("../studio/StudioLearningCharts", () => ({
  StudioLearningCharts: (props: unknown) => {
    studioLearningChartsSpy(props);
    return <div data-testid="studio-learning-charts" />;
  },
}));

function makeCheckpoint(
  episode: number,
  stage = 3,
  segmentTiming?: SegmentTiming | null,
): CheckpointEntry {
  return {
    checkpoint_episode: episode,
    checkpoint: `chk_${episode}.pt`,
    evaluation: `eval_${episode}.json`,
    replay: `replay_${episode}.json`,
    curriculum_stage: stage,
    success_rate: 0.75,
    timeout_rate: 0.1,
    average_completion_steps: 120,
    average_completion_seconds: 12,
    average_sheep_penned: 5.2,
    average_reward: 42.5,
    records: [],
    segment_timing: segmentTiming,
  };
}

function makeBaseProps(overrides: Partial<React.ComponentProps<typeof ProgressView>> = {}) {
  return {
    checkpoints: [],
    checkpointIndex: null,
    trainingEpisodes: [],
    sessionEvents: [],
    trainingStatus: null,
    effectiveCurriculumStage: 3,
    bestCheckpointEpisode: null,
    selectedCheckpointEpisode: null,
    isReadOnlyPreview: false,
    onInspectTroubleSpots: vi.fn(),
    onSelectCheckpointForReplay: vi.fn(),
    onInspectCheckpointInTroubleSpots: vi.fn(),
    onOpenReplayInInspector: vi.fn(),
    ...overrides,
  };
}

describe("ProgressView — Training Speed & Eval Overhead", () => {
  it("renders empty state when no segment_timing is present", () => {
    const checkpoints = [makeCheckpoint(100, 3, null), makeCheckpoint(200, 3, undefined)];
    render(<ProgressView {...makeBaseProps({ checkpoints })} />);

    fireEvent.click(screen.getByRole("tab", { name: "Speed & Timing" }));

    const section = screen.getByRole("region", {
      name: /training speed and evaluation overhead/i,
    });
    expect(
      within(section).getByText(
        /No segment timing recorded yet\. Run a training segment to capture a throughput baseline\./i,
      ),
    ).toBeTruthy();
    expect(section.textContent).not.toContain("NaN");
  });

  it("renders KPI summary and chronological SVG stacked bars when segment_timing is populated", () => {
    const olderTiming: SegmentTiming = {
      segment_steps: 8192,
      learn_seconds: 10.0,
      transitions_per_sec: 819.2,
      quick_eval_seconds: 2.0,
      confidence_eval_seconds: 0,
      save_and_export_seconds: 0.5,
      segment_total_seconds: 12.5,
      eval_overhead_pct: 16.0,
      ppo_env_workers: 4,
    };

    const newerTiming: SegmentTiming = {
      segment_steps: 16384,
      learn_seconds: 12.4,
      transitions_per_sec: 1321.3,
      quick_eval_seconds: 3.1,
      confidence_eval_seconds: 5.5,
      save_and_export_seconds: 0.45,
      segment_total_seconds: 21.45,
      eval_overhead_pct: 40.1,
      ppo_env_workers: 8,
    };

    // Provide newest first or unsorted; ProgressView sorts filteredCheckpoints newest-first internally
    const checkpoints = [
      makeCheckpoint(200, 3, newerTiming),
      makeCheckpoint(100, 3, olderTiming),
    ];

    render(<ProgressView {...makeBaseProps({ checkpoints })} />);

    fireEvent.click(screen.getByRole("tab", { name: "Speed & Timing" }));

    const section = screen.getByRole("region", {
      name: /training speed and evaluation overhead/i,
    });

    // Latest KPI summary (from Ep 200)
    expect(within(section).getByText("1321 steps/s")).toBeTruthy();
    expect(within(section).getByText(/12\.4s · 8 workers/)).toBeTruthy();
    expect(within(section).getByText("3.1s")).toBeTruthy();
    expect(within(section).getByText("5.5s")).toBeTruthy();
    expect(within(section).getByText("40.1%")).toBeTruthy();
    expect(within(section).getByText(/21\.4s total · 0\.45s save/)).toBeTruthy();

    // SVG stacked-bar chart in chronological order (oldest left, newest right)
    const chart = within(section).getByRole("img", {
      name: /segment timing stacked bar chart/i,
    });
    const barGroups = chart.querySelectorAll("g");
    expect(barGroups).toHaveLength(2);
    expect(barGroups[0].getAttribute("aria-label")).toContain("Ep 100");
    expect(barGroups[1].getAttribute("aria-label")).toContain("Ep 200");
    expect(section.textContent).not.toContain("NaN");
  });

  it("falls back to trainingStatus.last_segment_timing when checkpoints lack segment_timing", () => {
    const fallbackTiming: SegmentTiming = {
      segment_steps: 4096,
      learn_seconds: 5.0,
      transitions_per_sec: null,
      quick_eval_seconds: 1.5,
      confidence_eval_seconds: 0,
      save_and_export_seconds: 0.25,
      segment_total_seconds: 6.75,
      eval_overhead_pct: 22.2,
      ppo_env_workers: 2,
    };

    const trainingStatus = {
      last_segment_timing: fallbackTiming,
    } as TrainingStatus;

    render(<ProgressView {...makeBaseProps({ checkpoints: [], trainingStatus })} />);

    fireEvent.click(screen.getByRole("tab", { name: "Speed & Timing" }));

    const section = screen.getByRole("region", {
      name: /training speed and evaluation overhead/i,
    });
    expect(within(section).getByText("—")).toBeTruthy();
    expect(within(section).getByText(/5\.0s · 2 workers/)).toBeTruthy();
    expect(within(section).getByText("0.0s (skipped)")).toBeTruthy();
    expect(within(section).getByText("22.2%")).toBeTruthy();
    expect(section.textContent).not.toContain("NaN");
  });
});

describe("ProgressView — Prominent Stage Indicator", () => {
  it("renders current stage pill without maxCurriculumStage and includes 'of N' when provided", () => {
    const { rerender } = render(<ProgressView {...makeBaseProps({ effectiveCurriculumStage: 3 })} />);
    expect(screen.getByTestId("progress-stage-pill").textContent).toBe("Stage 3");

    rerender(<ProgressView {...makeBaseProps({ effectiveCurriculumStage: 3, maxCurriculumStage: 32 })} />);
    expect(screen.getByTestId("progress-stage-pill").textContent).toBe("Stage 3 of 32");
  });
});

it("loads stage seed results on demand when Seeds & Outcomes tab opens and distinguishes training from held-out outcomes", async () => {
  const loadReport = vi.mocked(loadStageTrainingReport);
  loadReport.mockClear();
  const timing = {
    run_id: "run_1",
    curriculum_stage: 3,
    training_seconds: 126,
    evaluation_seconds: 15,
    active_seconds: 141,
  };
  const seedOutcome = { termination_reason: "timeout", sheep_penned: 5, total_sheep: 6, replay_id: "seed-11" };
  const report: StageTrainingReport = {
    run_id: "run_1",
    curriculum_stage: 3,
    timing,
    training: {
      sample_limit: 250,
      sampled_episodes: 3,
      seeds: [{ seed: 11, attempts: 3, failures: 2, latest: { ...seedOutcome, checkpoint_id: null } }],
    },
    evaluations: [{
      evaluation_id: "evaluation_195",
      checkpoint_id: "checkpoint_195",
      checkpoint_episode: 195,
      seed_set_id: "release",
      seed_set_role: "held_out",
      evaluation_mode: "confidence",
      environment_config_hash: "environment_hash",
      seeds: [{ ...seedOutcome, seed: 11, success: false }],
    }],
  };
  loadReport.mockResolvedValue(report);
  const onOpenReplayInInspector = vi.fn();
  render(<ProgressView {...makeBaseProps({
    trainingStatus: { run_id: "run_1", runtime: { stage_totals: [timing] } } as TrainingStatus,
    onOpenReplayInInspector,
  })} />);

  expect(loadReport).not.toHaveBeenCalled();
  fireEvent.click(screen.getByRole("tab", { name: "Seeds & Outcomes" }));

  const training = await screen.findByRole("table", { name: "Training seed outcomes" });
  expect(loadReport).toHaveBeenCalledWith(3, "run_1", expect.any(AbortSignal), undefined);
  expect(within(training).getByText("2 / 3")).toBeInTheDocument();
  expect(screen.getByText("2 m 21 s")).toBeInTheDocument();

  fireEvent.change(screen.getByLabelText("Seed results"), { target: { value: "evaluation_195" } });
  const outcomes = screen.getByRole("table", { name: "Evaluation seed outcomes" });
  expect(within(outcomes).getByText("1 / 1")).toBeInTheDocument();
  expect(screen.getByRole("option", { name: /Held-out/ })).toBeInTheDocument();
  fireEvent.click(within(outcomes).getByRole("button", { name: "Open replay for seed 11" }));
  expect(onOpenReplayInInspector).toHaveBeenCalledWith("seed-11");
});

describe("ProgressView stage timings", () => {
  it("shows the stage timings section inside Speed & Timing tab", async () => {
    render(<ProgressView {...makeBaseProps({ checkpoints: [] })} />);
    fireEvent.click(screen.getByRole("tab", { name: "Speed & Timing" }));
    expect(
      screen.getByRole("heading", { name: "Stage timings: how long each stage took to train" }),
    ).toBeTruthy();
    expect(await screen.findByText(/No stage timings recorded yet/)).toBeTruthy();
    expect(screen.getByText("artifacts/stage-timing.jsonl")).toBeTruthy();
  });
});

describe("ProgressView — Current journey isolation", () => {
  it("excludes archived journey checkpoints from charts, speed KPIs, and table while keeping them in Stage Report selector", () => {
    studioLearningChartsSpy.mockClear();
    const archivedTiming: SegmentTiming = {
      segment_steps: 16384,
      learn_seconds: 5.0,
      transitions_per_sec: 9999,
      quick_eval_seconds: 1.0,
      confidence_eval_seconds: 0,
      save_and_export_seconds: 0.2,
      segment_total_seconds: 6.2,
      eval_overhead_pct: 16.1,
      ppo_env_workers: 16,
    };
    const currentTiming: SegmentTiming = {
      segment_steps: 16384,
      learn_seconds: 12.0,
      transitions_per_sec: 1450,
      quick_eval_seconds: 2.0,
      confidence_eval_seconds: 0,
      save_and_export_seconds: 0.3,
      segment_total_seconds: 14.3,
      eval_overhead_pct: 14.0,
      ppo_env_workers: 8,
    };

    const archived: CheckpointEntry[] = Array.from({ length: 11 }, (_, i) => ({
      ...makeCheckpoint(2940 + i, 24, archivedTiming),
      journey: "journey-20261124-172927",
      run_id: "run_stage9_from_stage8_20261119_063250_40a1",
      success_rate: 1.0,
    }));
    const current: CheckpointEntry[] = Array.from({ length: 11 }, (_, i) => ({
      ...makeCheckpoint(1019 + i, 24, currentTiming),
      journey: "current",
      run_id: "run_20261124_093542_33c5",
      success_rate: 0.4,
    }));

    render(
      <ProgressView
        {...makeBaseProps({
          checkpoints: [...archived, ...current],
          effectiveCurriculumStage: 24,
          trainingStatus: { run_id: "run_20261124_093542_33c5" } as TrainingStatus,
        })}
      />,
    );

    // 1. StudioLearningCharts (default Learning Curve tab) receives ONLY current journey checkpoints
    expect(studioLearningChartsSpy).toHaveBeenCalled();
    const passedProps = studioLearningChartsSpy.mock.calls.at(-1)?.[0] as {
      checkpoints: CheckpointEntry[];
    };
    expect(passedProps.checkpoints.map((c) => c.checkpoint_episode)).toEqual(
      Array.from({ length: 11 }, (_, i) => 1019 + i),
    );
    expect(passedProps.checkpoints.some((c) => c.checkpoint_episode === 2950)).toBe(false);

    // 2. Recent Checkpoints table shows ONLY ep 1019-1029 (40%), NEVER ep 2950 (100%)
    fireEvent.click(screen.getByRole("tab", { name: /Checkpoints/ }));
    const tableSection = screen.getByRole("region", { name: /recent checkpoints/i });
    expect(within(tableSection).getByText("Ep 1029")).toBeTruthy();
    expect(within(tableSection).queryByText("Ep 2950")).toBeNull();
    expect(within(tableSection).queryByText("100%")).toBeNull();
    expect(within(tableSection).getAllByText("40%").length).toBe(11);

    // 3. Training Speed & Eval Overhead uses current journey's timing (1450 steps/s, not archived 9999)
    fireEvent.click(screen.getByRole("tab", { name: "Speed & Timing" }));
    const speedSection = screen.getByRole("region", {
      name: /training speed and evaluation overhead/i,
    });
    expect(within(speedSection).getByText("1450 steps/s")).toBeTruthy();
    expect(within(speedSection).queryByText("9999 steps/s")).toBeNull();

    // 4. Archived journey remains selectable in the Stage Report dropdown
    fireEvent.click(screen.getByRole("tab", { name: "Seeds & Outcomes" }));
    expect(screen.getByRole("option", { name: "20261124-172927" })).toBeTruthy();
  });
});

describe("ProgressView — Tabbed Dashboard, Guide, ExplainTip & Alerts", () => {
  it("defaults to Learning Curve tab with hero KPIs and supports switching + keyboard navigation across all 5 tabs", () => {
    vi.mocked(loadStageTrainingReport).mockResolvedValue({
      run_id: "run_1",
      curriculum_stage: 3,
      timing: null,
      training: { sample_limit: 250, sampled_episodes: 0, seeds: [] },
      evaluations: [],
    });

    const trainingStatus = {
      run_id: "run_1",
      auto_promote_gate: {
        ready: false,
        decision: "collecting",
        formal_evaluations_available: 4,
        formal_evaluations_required: 6,
        qualified_evaluations: 1,
        qualified_evaluations_required: 2,
        success_threshold: 0.9,
        recent_average_success: 0.85,
      },
      early_warnings_summary: { total_active: 0 },
    } as unknown as TrainingStatus;

    render(
      <ProgressView
        {...makeBaseProps({
          checkpoints: [makeCheckpoint(100, 3)],
          trainingStatus,
        })}
      />,
    );

    // Stage Overview Hero KPIs & Alerts All Clear
    const hero = screen.getByRole("region", { name: /promotion gate and stage health/i });
    expect(within(hero).getByText("85%")).toBeInTheDocument();
    expect(within(hero).getByText("1 / 2")).toBeInTheDocument();
    expect(within(hero).getByText("4 / 6")).toBeInTheDocument();
    expect(within(hero).getByText("90%")).toBeInTheDocument();
    expect(screen.getByTestId("progress-alerts-toggle").textContent).toContain("Alerts: All Clear");

    // Default tab is Learning Curve
    const learningTab = screen.getByRole("tab", { name: "Learning Curve" });
    expect(learningTab.getAttribute("aria-selected")).toBe("true");
    expect(screen.getByTestId("studio-learning-charts")).toBeInTheDocument();
    expect(screen.queryByTestId("results-panel")).toBeNull();

    // Keyboard navigation: ArrowRight -> Checkpoints -> Seeds & Outcomes -> Speed & Timing -> Milestones & Video -> Home -> Learning Curve
    fireEvent.keyDown(learningTab, { key: "ArrowRight" });
    const checkpointsTab = screen.getByRole("tab", { name: /Checkpoints/ });
    expect(checkpointsTab.getAttribute("aria-selected")).toBe("true");
    expect(screen.getByRole("region", { name: /recent checkpoints/i })).toBeInTheDocument();
    expect(screen.queryByTestId("studio-learning-charts")).toBeNull();

    fireEvent.keyDown(checkpointsTab, { key: "ArrowRight" });
    const seedsTab = screen.getByRole("tab", { name: "Seeds & Outcomes" });
    expect(seedsTab.getAttribute("aria-selected")).toBe("true");
    expect(screen.getByRole("region", { name: /stage timing and seed outcomes/i })).toBeInTheDocument();

    fireEvent.keyDown(seedsTab, { key: "ArrowRight" });
    const timingTab = screen.getByRole("tab", { name: "Speed & Timing" });
    expect(timingTab.getAttribute("aria-selected")).toBe("true");
    expect(
      screen.getByRole("region", { name: /training speed and evaluation overhead/i }),
    ).toBeInTheDocument();

    fireEvent.keyDown(timingTab, { key: "ArrowRight" });
    const milestonesTab = screen.getByRole("tab", { name: "Milestones & Video" });
    expect(milestonesTab.getAttribute("aria-selected")).toBe("true");
    expect(screen.getByTestId("results-panel")).toBeInTheDocument();

    fireEvent.keyDown(milestonesTab, { key: "ArrowLeft" });
    expect(timingTab.getAttribute("aria-selected")).toBe("true");

    fireEvent.keyDown(timingTab, { key: "Home" });
    expect(learningTab.getAttribute("aria-selected")).toBe("true");

    fireEvent.keyDown(learningTab, { key: "End" });
    expect(milestonesTab.getAttribute("aria-selected")).toBe("true");
  });

  it("opens and closes Page Guide modal and ExplainTip popovers", () => {
    render(<ProgressView {...makeBaseProps({ effectiveCurriculumStage: 5 })} />);

    fireEvent.click(screen.getByTestId("progress-guide-toggle"));
    expect(screen.getByRole("dialog", { name: "How to read the Progress page" })).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Close page guide" }));
    expect(screen.queryByRole("dialog", { name: "How to read the Progress page" })).toBeNull();

    fireEvent.click(screen.getByRole("button", { name: "Explain Recent Avg Success" }));
    expect(screen.getByRole("dialog", { name: "Recent Avg Success" })).toBeInTheDocument();
    fireEvent.keyDown(document, { key: "Escape" });
    expect(screen.queryByRole("dialog", { name: "Recent Avg Success" })).toBeNull();
  });

  it("toggles EarlyWarningsPanel when warnings are 0 and auto-opens when warnings > 0", () => {
    const zeroWarningsStatus = {
      early_warnings_summary: { total_active: 0 },
    } as unknown as TrainingStatus;

    const { rerender } = render(
      <ProgressView {...makeBaseProps({ trainingStatus: zeroWarningsStatus })} />,
    );

    const alertsBtn = screen.getByTestId("progress-alerts-toggle");
    expect(alertsBtn.getAttribute("aria-expanded")).toBe("false");
    expect(alertsBtn.textContent).toContain("Alerts: All Clear");
    expect(screen.queryByTestId("early-warnings-panel")).toBeNull();

    fireEvent.click(alertsBtn);
    expect(alertsBtn.getAttribute("aria-expanded")).toBe("true");
    expect(screen.getByTestId("early-warnings-panel")).toBeInTheDocument();

    fireEvent.click(alertsBtn);
    expect(alertsBtn.getAttribute("aria-expanded")).toBe("false");
    expect(screen.queryByTestId("early-warnings-panel")).toBeNull();

    // Fresh mount with active warnings > 0 auto-opens EarlyWarningsPanel
    const activeWarningsStatus = {
      early_warnings_summary: { total_active: 2 },
    } as unknown as TrainingStatus;
    rerender(<ProgressView key="active" {...makeBaseProps({ trainingStatus: activeWarningsStatus })} />);

    const activeAlertsBtn = screen.getByTestId("progress-alerts-toggle");
    expect(activeAlertsBtn.textContent).toContain("Alerts: 2 Active");
    expect(activeAlertsBtn.getAttribute("aria-expanded")).toBe("true");
    expect(screen.getByTestId("early-warnings-panel")).toBeInTheDocument();

    fireEvent.click(activeAlertsBtn);
    expect(activeAlertsBtn.getAttribute("aria-expanded")).toBe("false");
    expect(screen.queryByTestId("early-warnings-panel")).toBeNull();
  });

  it("displays rolling-window Qualified (Last 8) count and stability-tail explanation without consecutive-reset claims", () => {
    const trainingStatus = {
      auto_promote_gate: {
        ready: false,
        decision: "hold",
        reason: "Waiting for 6 of last 8 evaluations and 2-eval stability tail",
        window_size: 8,
        qualified_evaluations: 5,
        qualified_evaluations_required: 6,
        qualifying_tail_passed: false,
        qualifying_tail_required: 2,
        success_threshold: 0.9,
      },
    } as unknown as TrainingStatus;

    render(<ProgressView {...makeBaseProps({ effectiveCurriculumStage: 34, trainingStatus })} />);

    expect(screen.getByText("Qualified (Last 8)")).toBeInTheDocument();
    expect(screen.getByText("5 / 6")).toBeInTheDocument();
    expect(screen.queryByText("Qualified Streak")).toBeNull();

    fireEvent.click(screen.getByRole("button", { name: "Explain Qualified Evals (Last 8)" }));
    const dialog = screen.getByRole("dialog", { name: "Qualified Evals (Last 8)" });
    expect(dialog).toBeInTheDocument();
    expect(dialog.textContent).toContain(
      "How many of the last 8 formal evaluations met the stage target bar (not required to be consecutive).",
    );
    expect(dialog.textContent).toContain(
      "Requires 6 of the last 8 formal evaluations meeting ≥ 90% success.",
    );
    expect(dialog.textContent).toContain(
      "Separate stability tail: the last 2 formal evaluations must both meet ≥ 90% (not yet met).",
    );
    expect(dialog.textContent).toContain(
      "A single evaluation below 90% does not reset this count to 0; older evaluations roll out of the 8-evaluation window.",
    );
    expect(dialog.textContent).not.toMatch(/resets the streak to 0/i);
  });
});
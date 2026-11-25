import { useEffect, useMemo, useRef, useState } from "react";
import { EarlyWarningsPanel } from "../components/EarlyWarningsPanel";
import { ExplainTip, ProgressPageGuideModal } from "../components/ExplainTip";
import { ResultsPanel } from "../components/ResultsPanel";
import { StageTimingPanel } from "../components/StageTimingPanel";
import { loadStageTrainingReport } from "../lib/api";
import { filterCurrentJourneyCheckpoints, getCheckpointStage } from "../lib/chartPipeline";
import { StudioLearningCharts } from "../studio/StudioLearningCharts";
import type {
  CheckpointEntry,
  CheckpointIndex,
  SegmentTiming,
  SessionChangeEvent,
  StageEvaluationReport,
  StageRuntimeTotals,
  StageTrainingReport,
  TrainingEpisode,
  TrainingStatus,
} from "../state/types";

function toPercent(value: number | null | undefined): number | null {
  if (value == null || !Number.isFinite(value)) return null;
  return value <= 1 ? Math.round(value * 100) : Math.round(value);
}

function resolveCheckpointStageNumber(entry: CheckpointEntry): number {
  if (entry.curriculum_stage != null && Number.isFinite(entry.curriculum_stage)) {
    return entry.curriculum_stage;
  }
  if (entry.reward_config?.instincts?.curriculum_stage != null) {
    return entry.reward_config.instincts.curriculum_stage;
  }
  if (entry.environment_config?.curriculum_stage != null) {
    return entry.environment_config.curriculum_stage;
  }
  return getCheckpointStage(entry);
}

export interface ProgressViewProps {
  checkpoints: CheckpointEntry[];
  checkpointIndex: CheckpointIndex | null;
  trainingEpisodes: TrainingEpisode[];
  sessionEvents: SessionChangeEvent[];
  trainingStatus: TrainingStatus | null;
  effectiveCurriculumStage: number;
  maxCurriculumStage?: number;
  bestCheckpointEpisode: number | null;
  selectedCheckpointEpisode: number | null;
  isReadOnlyPreview?: boolean;
  onInspectTroubleSpots: () => void;
  onSelectCheckpointForReplay: (checkpointEpisode: number) => void;
  onInspectCheckpointInTroubleSpots: (checkpointEpisode: number) => void;
  onOpenReplayInInspector: (replayPath: string, step?: number) => void;
}

const PROGRESS_STAGE_FILTER_STORAGE_KEY = "sheepdog_progress_stage_filter";
const SEED_ROWS_PER_PAGE = 25;

type ProgressTabId = "learning" | "checkpoints" | "seeds" | "timing" | "milestones";

const PROGRESS_TABS: Array<{ id: ProgressTabId; label: string }> = [
  { id: "learning", label: "Learning Curve" },
  { id: "checkpoints", label: "Checkpoints" },
  { id: "seeds", label: "Seeds & Outcomes" },
  { id: "timing", label: "Speed & Timing" },
  { id: "milestones", label: "Milestones & Video" },
];

const SEED_SET_LABELS: Record<StageEvaluationReport["seed_set_role"], string> = {
  development: "Development",
  held_out: "Held-out",
  unknown: "Seed set role unknown",
};

function formatStageDuration(value: number): string {
  const seconds = Math.floor(value);
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor(seconds / 60) % 60;
  const remainder = seconds % 60;
  return [hours > 0 && `${hours} h`, (hours > 0 || minutes > 0) && `${minutes} m`, `${remainder} s`]
    .filter(Boolean)
    .join(" ");
}

function StageTimingTable({ totals }: { totals: StageRuntimeTotals[] }) {
  if (totals.length === 0) {
    return <p>Stage timers will appear once measured training activity is available.</p>;
  }
  return (
    <div className="ui-table-wrap">
      <table className="ui-table" aria-label="Active time by curriculum stage">
        <thead>
          <tr><th>Stage</th><th>Training</th><th>Evaluation</th><th>Total active</th></tr>
        </thead>
        <tbody>
          {totals.map((timing) => (
            <tr key={timing.curriculum_stage}>
              <th scope="row">{timing.curriculum_stage}</th>
              <td>{formatStageDuration(timing.training_seconds)}</td>
              <td>{formatStageDuration(timing.evaluation_seconds)}</td>
              <td>{formatStageDuration(timing.active_seconds)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function evaluationSourceKey(evaluation: StageEvaluationReport, index: number): string {
  return evaluation.evaluation_id ?? `evaluation-${index}`;
}

function StageSeedReport({
  stage,
  runId,
  journey,
  onOpenReplay,
  onInspectCheckpoint,
}: {
  stage: number;
  runId?: string;
  journey?: string;
  onOpenReplay?: ProgressViewProps["onOpenReplayInInspector"];
  onInspectCheckpoint?: ProgressViewProps["onInspectCheckpointInTroubleSpots"];
}) {
  const [report, setReport] = useState<StageTrainingReport | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [revision, setRevision] = useState(0);
  const [source, setSource] = useState("training");
  const [page, setPage] = useState(0);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const abort = new AbortController();
    void loadStageTrainingReport(stage, runId, abort.signal, journey)
      .then((result) => {
        if (!abort.signal.aborted) {
          setReport(result);
          setSource((previous) =>
            previous === "training" || result.evaluations.some((item, index) => evaluationSourceKey(item, index) === previous)
              ? previous
              : "training",
          );
        }
      })
      .catch((failure: unknown) => {
        if (!abort.signal.aborted) {
          setError(failure instanceof Error ? failure.message : "Unable to load stage data.");
        }
      })
      .finally(() => {
        if (!abort.signal.aborted) setLoading(false);
      });
    return () => abort.abort();
  }, [stage, runId, journey, revision]);

  const evaluation = report?.evaluations.find((item, index) => evaluationSourceKey(item, index) === source);
  const provenance = evaluation ? [
    ["Checkpoint ID", evaluation.checkpoint_id],
    ["Evaluated at", evaluation.evaluation_timestamp],
    ["Seed manifest", evaluation.seed_set_id],
    ["Assistance", evaluation.assistance_mode],
    ["Environment hash", evaluation.environment_config_hash],
    ["Observation schema", evaluation.observation_schema_hash],
    ["Action schema", evaluation.action_space_hash],
    ["Policy version", evaluation.policy_version],
    ["Experiment ID", evaluation.experiment_id],
    ["Activation ID", evaluation.activation_id],
    ["Change ID", evaluation.change_id],
    ["Verified source revision", evaluation.verified_source_revision],
    ["Model SHA-256", evaluation.model_sha256],
  ].filter(([, value]) => value != null) : [];
  const rows = useMemo(() => {
    if (!report) return [];
    if (evaluation) {
      return [...evaluation.seeds].sort((a, b) => Number(a.success) - Number(b.success) || a.seed - b.seed)
        .map((seed) => ({
          seed: seed.seed,
          attempts: 1,
          failures: Number(!seed.success),
          latest: { ...seed, checkpoint_id: evaluation.checkpoint_id },
        }));
    }
    return [...report.training.seeds].sort((a, b) => b.failures - a.failures || a.seed - b.seed);
  }, [report, evaluation]);
  const lastPage = Math.max(0, Math.ceil(rows.length / SEED_ROWS_PER_PAGE) - 1);
  const currentPage = Math.min(page, lastPage);
  const firstRowIndex = currentPage * SEED_ROWS_PER_PAGE;

  return (
    <>
      <div className="ui-control-row">
        <label htmlFor="stage-seed-report-source">Seed results</label>
        <select
          id="stage-seed-report-source"
          value={source}
          onChange={(event) => {
            setSource(event.target.value);
            setPage(0);
          }}
          disabled={!report}
        >
          <option value="training">Recent training episodes</option>
          {report?.evaluations.map((item, index) => (
            <option key={evaluationSourceKey(item, index)} value={evaluationSourceKey(item, index)}>
              {SEED_SET_LABELS[item.seed_set_role]} · {item.evaluation_mode ?? "Evaluation"}
              {item.checkpoint_episode != null ? ` · Checkpoint ${item.checkpoint_episode}` : ""}
            </option>
          ))}
        </select>
        <button type="button" disabled={loading} onClick={() => {
          setError(null);
          setLoading(true);
          setRevision((previous) => previous + 1);
        }}>
          Refresh seed results
        </button>
      </div>
      {loading ? <p role="status">Loading Stage {stage} seed results…</p> : null}
      {error ? <p role="alert">{error}</p> : null}
      {report ? (
        <>
          <p className="ui-meta-strip__note">
            Run {report.run_id ?? "unknown"} · Stage {report.curriculum_stage}
          </p>
          {journey ? <StageTimingTable totals={report.timing ? [report.timing] : []} /> : null}
          {evaluation ? (
            <>
              <p className="ui-meta-strip__note">
                {evaluation.seeds.filter((seed) => !seed.success).length} failures / {evaluation.seeds.length} evaluated seeds
              </p>
              {evaluation.checkpoint_episode != null && onInspectCheckpoint ? (
                <button type="button" onClick={() => {
                  if (evaluation.checkpoint_episode != null) onInspectCheckpoint(evaluation.checkpoint_episode);
                }}>
                  Inspect checkpoint {evaluation.checkpoint_episode}
                </button>
              ) : null}
              {provenance.length > 0 ? (
                <details className="ui-disclosure">
                  <summary>Evaluation and change details</summary>
                  <dl className="ui-disclosure__body">
                    {provenance.map(([label, value]) => (
                      <div key={label}>
                        <dt>{label}</dt>
                        <dd style={{ overflowWrap: "anywhere" }}>{value}</dd>
                      </div>
                    ))}
                  </dl>
                </details>
              ) : null}
            </>
          ) : (
            <p className="ui-meta-strip__note">
              {report.training.sampled_episodes} recent recorded training episodes, up to {report.training.sample_limit}.
            </p>
          )}
          {rows.length > 0 ? (
            <div className="ui-table-wrap">
              <table className="ui-table" aria-label={evaluation ? "Evaluation seed outcomes" : "Training seed outcomes"}>
                <thead>
                  <tr><th>Seed</th><th>Failures / attempts</th><th>Latest termination</th><th>Sheep penned</th><th>Checkpoint</th><th>Replay</th></tr>
                </thead>
                <tbody>
                  {rows.slice(firstRowIndex, firstRowIndex + SEED_ROWS_PER_PAGE).map(({ seed, attempts, failures, latest }) => {
                    const replaySource = latest.replay_id ?? latest.replay_path;
                    return (
                      <tr key={seed}>
                        <th scope="row">{seed}</th>
                        <td>{failures} / {attempts}</td>
                        <td>{latest.termination_reason?.replace(/_/g, " ") ?? "—"}</td>
                        <td>{latest.sheep_penned ?? "—"}{latest.total_sheep != null ? ` / ${latest.total_sheep}` : ""}</td>
                        <td style={{ overflowWrap: "anywhere" }}>{latest.checkpoint_id ?? "—"}</td>
                        <td>
                          {replaySource && onOpenReplay ? (
                            <button type="button" aria-label={`Open replay for seed ${seed}`} onClick={() => onOpenReplay(replaySource)}>
                              Open replay
                            </button>
                          ) : "—"}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          ) : <p>No seed records are available for this selection.</p>}
          {lastPage > 0 ? (
            <div className="ui-control-row">
              <button type="button" disabled={currentPage === 0} onClick={() => setPage(currentPage - 1)}>
                Previous seeds
              </button>
              <span>Page {currentPage + 1} of {lastPage + 1}</span>
              <button type="button" disabled={currentPage >= lastPage} onClick={() => setPage(currentPage + 1)}>
                Next seeds
              </button>
            </div>
          ) : null}
        </>
      ) : null}
    </>
  );
}

function loadSavedStageFilter(): string {
  try {
    const saved = window.localStorage.getItem(PROGRESS_STAGE_FILTER_STORAGE_KEY);
    if (saved === "current" || saved === "all" || (saved && /^\d+$/.test(saved))) {
      return saved;
    }
  } catch {
    // Ignore storage access failures.
  }
  return "current";
}

export function ProgressView({
  checkpoints,
  checkpointIndex,
  trainingEpisodes,
  sessionEvents,
  trainingStatus,
  effectiveCurriculumStage,
  maxCurriculumStage,
  bestCheckpointEpisode,
  selectedCheckpointEpisode,
  isReadOnlyPreview = false,
  onInspectTroubleSpots,
  onSelectCheckpointForReplay,
  onInspectCheckpointInTroubleSpots,
  onOpenReplayInInspector,
}: ProgressViewProps) {
  const activeWarningsCount = trainingStatus?.early_warnings_summary?.total_active ?? 0;
  const userToggledWarningsRef = useRef(false);
  const [warningsOpen, setWarningsOpen] = useState<boolean>(() => activeWarningsCount > 0);
  const [activeTab, setActiveTab] = useState<ProgressTabId>("learning");
  const [guideOpen, setGuideOpen] = useState<boolean>(false);
  const tabButtonRefs = useRef<Partial<Record<ProgressTabId, HTMLButtonElement | null>>>({});
  const [stageFilter, setStageFilter] = useState<string>(loadSavedStageFilter);
  const [showAllCheckpoints, setShowAllCheckpoints] = useState<boolean>(false);
  const [selectedReportStage, setSelectedReportStage] = useState<number | null>(null);
  const [selectedReportRun, setSelectedReportRun] = useState("");
  const [selectedReportJourney, setSelectedReportJourney] = useState("");
  const reportStage = selectedReportStage ?? effectiveCurriculumStage;
  const activeRunId = trainingStatus?.run_id;
  const seedsTabOpen = activeTab === "seeds";
  const reportJourneys = useMemo(() => {
    if (!seedsTabOpen) return [];
    const journeys = new Set<string>();
    for (const checkpoint of checkpoints) {
      if (checkpoint.journey && /^journey-\d{8}-\d{6}$/.test(checkpoint.journey)) {
        journeys.add(checkpoint.journey.slice("journey-".length));
      }
    }
    return Array.from(journeys).sort().reverse();
  }, [checkpoints, seedsTabOpen]);
  const reportJourney = reportJourneys.includes(selectedReportJourney) ? selectedReportJourney : undefined;
  const alternateReportRuns = useMemo(() => {
    if (!seedsTabOpen) return [];
    const runIds = new Set<string>();
    for (const checkpoint of checkpoints) {
      const matchesJourney = reportJourney
        ? checkpoint.journey === `journey-${reportJourney}`
        : !checkpoint.journey || checkpoint.journey === "current";
      if (matchesJourney && checkpoint.run_id && (reportJourney || checkpoint.run_id !== activeRunId)) {
        runIds.add(checkpoint.run_id);
      }
    }
    if (!reportJourney) {
      for (const timing of trainingStatus?.runtime?.stage_totals ?? []) {
        if (timing.run_id !== activeRunId) runIds.add(timing.run_id);
      }
    }
    return Array.from(runIds).sort();
  }, [activeRunId, checkpoints, reportJourney, seedsTabOpen, trainingStatus?.runtime?.stage_totals]);
  const reportRunId = alternateReportRuns.includes(selectedReportRun) ? selectedReportRun : undefined;
  const runId = reportRunId ?? (reportJourney ? undefined : activeRunId);
  const stageRuntimeTotals = (trainingStatus?.runtime?.stage_totals ?? [])
    .filter((timing) => timing.run_id === runId)
    .sort((a, b) => a.curriculum_stage - b.curriculum_stage);

  useEffect(() => {
    if (!userToggledWarningsRef.current && activeWarningsCount > 0) {
      setWarningsOpen(true);
    }
  }, [activeWarningsCount]);

  const gate = trainingStatus?.auto_promote_gate ?? null;
  const formalEvalsAvailable =
    gate?.formal_evaluations_available ?? gate?.recent_checkpoints_considered ?? 0;
  const formalEvalsRequired =
    gate?.formal_evaluations_required ?? gate?.minimum_required_evaluations ?? 6;
  const windowSize = gate?.window_size ?? 8;
  const tailRequired = gate?.qualifying_tail_required ?? 2;
  const qualifiedCount =
    gate?.qualified_evaluations ?? gate?.recent_qualifying_checkpoints ?? 0;
  const qualifiedRequired = gate?.qualified_evaluations_required ?? 2;
  const promotionTargetPct =
    toPercent(
      gate?.success_threshold ??
        trainingStatus?.auto_promote_threshold ??
        (effectiveCurriculumStage <= 1 ? 0.8 : 0.9),
    ) ?? 90;
  const recentAvgPct = toPercent(
    gate?.recent_average_success ?? gate?.aggregate_success_rate ?? null,
  );
  const blockingSeeds: number[] = useMemo(() => {
    if (gate?.blocking_seeds && gate.blocking_seeds.length > 0) {
      return gate.blocking_seeds;
    }
    if (gate?.blocking_seed != null) {
      return [gate.blocking_seed];
    }
    return [];
  }, [gate]);

  const gateStatusTone = gate?.ready
    ? "ui-badge--pass"
    : gate?.decision === "blocked" || gate?.decision === "hold" || blockingSeeds.length > 0
      ? "ui-badge--warn"
      : "";
  const gateStatusLabel = gate?.ready
    ? "Ready to promote"
    : gate?.status_text || (gate?.decision ? gate.decision.replace(/_/g, " ") : "Collecting evidence");

  const currentJourneyCheckpoints = useMemo(
    () => filterCurrentJourneyCheckpoints(checkpoints, activeRunId),
    [checkpoints, activeRunId],
  );

  const availableStages = useMemo(() => {
    const stages = new Set<number>();
    for (const cp of currentJourneyCheckpoints) {
      stages.add(resolveCheckpointStageNumber(cp));
    }
    return Array.from(stages).sort((a, b) => a - b);
  }, [currentJourneyCheckpoints]);

  const filteredCheckpoints = useMemo(() => {
    let list = currentJourneyCheckpoints;
    if (stageFilter === "current") {
      list = currentJourneyCheckpoints.filter(
        (c) => resolveCheckpointStageNumber(c) === effectiveCurriculumStage,
      );
    } else if (stageFilter !== "all") {
      const targetStage = Number(stageFilter);
      if (Number.isFinite(targetStage)) {
        list = currentJourneyCheckpoints.filter(
          (c) => resolveCheckpointStageNumber(c) === targetStage,
        );
      }
    }
    return [...list].sort((a, b) => b.checkpoint_episode - a.checkpoint_episode);
  }, [currentJourneyCheckpoints, stageFilter, effectiveCurriculumStage]);

  const visibleCheckpoints = showAllCheckpoints
    ? filteredCheckpoints
    : filteredCheckpoints.slice(0, 25);

  const timedCheckpointsDesc = useMemo(
    () =>
      filteredCheckpoints.filter(
        (cp): cp is CheckpointEntry & { segment_timing: SegmentTiming } =>
          cp.segment_timing != null,
      ),
    [filteredCheckpoints],
  );

  const latestTiming: SegmentTiming | null =
    timedCheckpointsDesc[0]?.segment_timing ?? trainingStatus?.last_segment_timing ?? null;

  const chartCheckpoints = useMemo(
    () => timedCheckpointsDesc.slice(0, 12).reverse(),
    [timedCheckpointsDesc],
  );

  const maxChartTotalSeconds = useMemo(() => {
    let maxVal = 0;
    for (const cp of chartCheckpoints) {
      const t = cp.segment_timing;
      const sum =
        Math.max(0, t.learn_seconds) +
        Math.max(0, t.quick_eval_seconds) +
        Math.max(0, t.confidence_eval_seconds) +
        Math.max(0, t.save_and_export_seconds);
      const total = Math.max(sum, t.segment_total_seconds ?? 0);
      if (Number.isFinite(total) && total > maxVal) {
        maxVal = total;
      }
    }
    return maxVal > 0 ? maxVal : 1;
  }, [chartCheckpoints]);

  const timingStageProp =
    stageFilter === "all"
      ? null
      : stageFilter === "current"
        ? effectiveCurriculumStage
        : Number.isFinite(Number(stageFilter))
          ? Number(stageFilter)
          : null;

  const recentAvgMeterTone =
    recentAvgPct !== null && recentAvgPct >= promotionTargetPct ? "pass" : "warn";
  const formalEvalFillPct = Math.min(
    100,
    Math.round((formalEvalsAvailable / Math.max(formalEvalsRequired, 1)) * 100),
  );
  const qualifiedPipCount = Math.min(Math.max(qualifiedRequired, 1), 10);
  const filledQualifiedPips = Math.min(qualifiedCount, qualifiedPipCount);

  const handleTabKeyDown = (event: React.KeyboardEvent<HTMLButtonElement>, currentIndex: number) => {
    let nextIndex: number | null = null;
    if (event.key === "ArrowRight") {
      nextIndex = (currentIndex + 1) % PROGRESS_TABS.length;
    } else if (event.key === "ArrowLeft") {
      nextIndex = (currentIndex - 1 + PROGRESS_TABS.length) % PROGRESS_TABS.length;
    } else if (event.key === "Home") {
      nextIndex = 0;
    } else if (event.key === "End") {
      nextIndex = PROGRESS_TABS.length - 1;
    }
    if (nextIndex === null) return;
    event.preventDefault();
    const nextTab = PROGRESS_TABS[nextIndex];
    setActiveTab(nextTab.id);
    tabButtonRefs.current[nextTab.id]?.focus();
  };

  const isFitLearning = activeTab === "learning" && !warningsOpen;

  return (
    <div className={`ui-view-stack ui-progress-view${isFitLearning ? " ui-progress-view--fit" : ""}`}>
      {/* 1. Compact Executive Stage Overview Hero */}
      <section className="ui-progress-hero" aria-label="Promotion gate and stage health">
        <div className="ui-progress-hero__top">
          <div className="ui-progress-hero__title-group">
            <span className="ui-stage-pill ui-stage-pill--compact" data-testid="progress-stage-pill">
              Stage {effectiveCurriculumStage}
              {maxCurriculumStage != null && maxCurriculumStage > 0
                ? ` of ${maxCurriculumStage}`
                : ""}
            </span>
            <h2 className="ui-panel__title">Stage {effectiveCurriculumStage} Progress</h2>
            <ExplainTip
              title="Promotion Gate & Stage Health"
              plain="Tracks whether the dogs are passing Stage tests consistently enough to unlock the next, harder stage."
              technical={[
                ...(gate?.reason ? [gate.reason] : []),
                "Promotion requires meeting the formal evaluation count, rolling-window qualifying evaluations, 2-eval stability tail, and stage pass-rate threshold.",
                "Partial sheep penning does not count as success — all 6 sheep must be penned within the episode step budget.",
              ]}
            />
            <span className={`ui-badge ${gateStatusTone}`} title={gate?.reason ?? undefined}>
              {gateStatusLabel}
            </span>
          </div>

          <div className="ui-progress-hero__actions">
            {blockingSeeds.length > 0 ? (
              <button
                type="button"
                className="ui-blocker-pill"
                onClick={onInspectTroubleSpots}
                title="Inspect blocking seeds"
              >
                Blocker: {blockingSeeds.map((s) => `#${s}`).join(", ")} · Inspect →
              </button>
            ) : null}

            <button
              type="button"
              data-testid="progress-alerts-toggle"
              aria-expanded={warningsOpen}
              className={`ui-alerts-pill ${
                activeWarningsCount === 0
                  ? "ui-alerts-pill--clear"
                  : "ui-alerts-pill--active"
              }`}
              onClick={() => {
                userToggledWarningsRef.current = true;
                setWarningsOpen((prev) => !prev);
              }}
            >
              <span className="ui-alerts-pill__dot" aria-hidden="true" />
              <span>
                {activeWarningsCount === 0
                  ? "Alerts: All Clear"
                  : `Alerts: ${activeWarningsCount} Active`}
              </span>
            </button>

            <button
              type="button"
              data-testid="progress-guide-toggle"
              className="ui-guide-btn"
              onClick={() => setGuideOpen(true)}
            >
              <span className="ui-guide-btn__badge" aria-hidden="true">
                ?
              </span>
              <span>Page Guide</span>
            </button>
          </div>
        </div>

        <div className="ui-metric-ribbon">
          <div className="ui-metric-ribbon__cell">
            <div className="ui-metric-ribbon__top">
              <span className="ui-metric-ribbon__label">Recent Avg Success</span>
              <ExplainTip
                title="Recent Avg Success"
                plain="Average share of test scenarios where all 6 sheep were penned in the latest formal evaluation window."
                technical={[
                  "Computed over the active stage's recent formal 10-seed evaluation window.",
                  `Needs at least ${promotionTargetPct}% to clear the stage bar.`,
                ]}
              />
            </div>
            <div className="ui-metric-ribbon__bottom">
              <span className="ui-metric-ribbon__value">
                {recentAvgPct !== null ? `${recentAvgPct}%` : "—"}
              </span>
              <div className="ui-kpi-meter" aria-hidden="true">
                <div
                  className={`ui-kpi-meter__fill ui-kpi-meter__fill--${recentAvgMeterTone}`}
                  style={{ width: `${Math.max(0, Math.min(100, recentAvgPct ?? 0))}%` }}
                />
                <span
                  className="ui-kpi-meter__target"
                  style={{ left: `${Math.max(0, Math.min(100, promotionTargetPct))}%` }}
                />
              </div>
            </div>
          </div>

          <div className="ui-metric-ribbon__cell">
            <div className="ui-metric-ribbon__top">
              <span className="ui-metric-ribbon__label">Qualified (Last {windowSize})</span>
              <ExplainTip
                title={`Qualified Evals (Last ${windowSize})`}
                plain={`How many of the last ${windowSize} formal evaluations met the stage target bar (not required to be consecutive).`}
                technical={[
                  `Requires ${qualifiedRequired} of the last ${windowSize} formal evaluations meeting ≥ ${promotionTargetPct}% success.`,
                  `Separate stability tail: the last ${tailRequired} formal evaluations must both meet ≥ ${promotionTargetPct}%${gate?.qualifying_tail_passed === false ? " (not yet met)" : gate?.qualifying_tail_passed === true ? " (met)" : ""}.`,
                  `A single evaluation below ${promotionTargetPct}% does not reset this count to 0; older evaluations roll out of the ${windowSize}-evaluation window.`,
                ]}
              />
            </div>
            <div className="ui-metric-ribbon__bottom">
              <span className="ui-metric-ribbon__value">
                {qualifiedCount} / {qualifiedRequired}
              </span>
              <div className="ui-kpi-pips" aria-hidden="true">
                {Array.from({ length: qualifiedPipCount }, (_, idx) => (
                  <span
                    key={idx}
                    className={`ui-kpi-pip${idx < filledQualifiedPips ? " ui-kpi-pip--filled" : ""}`}
                  />
                ))}
              </div>
            </div>
          </div>

          <div className="ui-metric-ribbon__cell">
            <div className="ui-metric-ribbon__top">
              <span className="ui-metric-ribbon__label">Formal Evals</span>
              <ExplainTip
                title="Formal Evals"
                plain="Number of full 10-seed test runs completed on the current curriculum stage."
                technical={[
                  `Minimum ${formalEvalsRequired} formal evaluations on the current stage before auto-promotion can trigger.`,
                  "Prevents premature promotion from a lucky early checkpoint.",
                ]}
              />
            </div>
            <div className="ui-metric-ribbon__bottom">
              <span className="ui-metric-ribbon__value">
                {formalEvalsAvailable} / {formalEvalsRequired}
              </span>
              <div className="ui-kpi-meter" aria-hidden="true">
                <div
                  className="ui-kpi-meter__fill ui-kpi-meter__fill--accent"
                  style={{ width: `${formalEvalFillPct}%` }}
                />
              </div>
            </div>
          </div>

          <div className="ui-metric-ribbon__cell">
            <div className="ui-metric-ribbon__top">
              <span className="ui-metric-ribbon__label">Pass Threshold</span>
              <ExplainTip
                title="Pass Threshold"
                align="right"
                plain="The minimum pass percentage required on a formal evaluation to qualify toward promotion."
                technical={[
                  `Stage ${effectiveCurriculumStage} promotion bar is ${promotionTargetPct}%.`,
                  "Each formal evaluation tests 10 fixed seeds with deterministic policy actions.",
                ]}
              />
            </div>
            <div className="ui-metric-ribbon__bottom">
              <span className="ui-metric-ribbon__value">{promotionTargetPct}%</span>
              <span className="ui-metric-ribbon__hint">Stage bar</span>
            </div>
          </div>
        </div>
      </section>

      {/* 2. Collapsible Advisory Warnings Drawer */}
      {warningsOpen ? (
        <EarlyWarningsPanel
          curriculumStage={effectiveCurriculumStage}
          isOpen={true}
          onToggle={() => {
            userToggledWarningsRef.current = true;
            setWarningsOpen(false);
          }}
          isReadOnlyPreview={isReadOnlyPreview}
          onOpenReplay={onOpenReplayInInspector}
        />
      ) : null}

      {/* 3. 5-Tab Navigation Bar */}
      <div className="ui-progress-tabs" role="tablist" aria-label="Progress sections">
        {PROGRESS_TABS.map((tab, idx) => {
          const selected = activeTab === tab.id;
          return (
            <button
              key={tab.id}
              ref={(el) => {
                tabButtonRefs.current[tab.id] = el;
              }}
              id={`progress-tab-${tab.id}`}
              type="button"
              role="tab"
              aria-selected={selected}
              aria-controls={`progress-tabpanel-${tab.id}`}
              tabIndex={selected ? 0 : -1}
              className={`ui-progress-tab${selected ? " ui-progress-tab--active" : ""}`}
              onClick={() => setActiveTab(tab.id)}
              onKeyDown={(e) => handleTabKeyDown(e, idx)}
            >
              <span>{tab.label}</span>
              {tab.id === "checkpoints" ? (
                <span className="ui-progress-tab__count">{filteredCheckpoints.length}</span>
              ) : null}
            </button>
          );
        })}
      </div>

      {/* 4. Single Active Tab Panel */}
      <div
        id={`progress-tabpanel-${activeTab}`}
        className="ui-progress-tabpanel"
        role="tabpanel"
        aria-labelledby={`progress-tab-${activeTab}`}
      >
        {activeTab === "learning" ? (
          <StudioLearningCharts
            rawEpisodes={trainingEpisodes}
            checkpoints={currentJourneyCheckpoints}
            sessionEvents={sessionEvents}
            effectiveStage={effectiveCurriculumStage}
            activeRunId={trainingStatus?.run_id}
            bestCheckpointEpisode={bestCheckpointEpisode}
            selectedCheckpointEpisode={selectedCheckpointEpisode}
            promotionTargetPct={promotionTargetPct}
            defaultMode="combined"
            onSelectCheckpoint={onSelectCheckpointForReplay}
            onInspectInArena={onInspectCheckpointInTroubleSpots}
          />
        ) : null}

        {activeTab === "checkpoints" ? (
          <section className="ui-panel" aria-label="Recent checkpoints">
            <div className="ui-panel__header">
              <div>
                <div style={{ display: "flex", alignItems: "center", gap: "0.45rem" }}>
                  <h2 className="ui-panel__title">Recent Checkpoints</h2>
                  <ExplainTip
                    title="Recent Checkpoints"
                    plain="Saved snapshots of the dogs' neural network during training. Each row summarizes how that snapshot performed across test scenarios."
                    technical={[
                      "Filtered to the current journey so archived runs never mix into active history.",
                      "Pass Rate = % of evaluated seeds where all 6 sheep were penned before timeout.",
                    ]}
                  />
                </div>
                <p className="ui-panel__subtitle">
                  Showing {visibleCheckpoints.length} of {filteredCheckpoints.length} checkpoints
                </p>
              </div>

              <div className="ui-control-row">
                <select
                  aria-label="Checkpoint history filter"
                  value={stageFilter}
                  onChange={(e) => {
                    const next = e.target.value;
                    setStageFilter(next);
                    setShowAllCheckpoints(false);
                    try {
                      window.localStorage.setItem(PROGRESS_STAGE_FILTER_STORAGE_KEY, next);
                    } catch {
                      // Ignore storage access failures.
                    }
                  }}
                >
                  <option value="current">Stage {effectiveCurriculumStage} (Current)</option>
                  <option value="all">All Stages ({currentJourneyCheckpoints.length})</option>
                  {availableStages.map((stage) => (
                    <option key={stage} value={String(stage)}>
                      Stage {stage}
                    </option>
                  ))}
                  {stageFilter !== "current" &&
                  stageFilter !== "all" &&
                  !availableStages.includes(Number(stageFilter)) ? (
                    <option value={stageFilter}>Stage {stageFilter}</option>
                  ) : null}
                </select>

                {filteredCheckpoints.length > 25 ? (
                  <button
                    type="button"
                    className="ui-btn--sm"
                    onClick={() => setShowAllCheckpoints((prev) => !prev)}
                  >
                    {showAllCheckpoints ? "Show recent 25" : `Show all (${filteredCheckpoints.length})`}
                  </button>
                ) : null}
              </div>
            </div>

            <div className="ui-table-wrap" style={{ border: "none", borderRadius: 0 }}>
              <table className="ui-table">
                <thead>
                  <tr>
                    <th>Episode</th>
                    <th>Stage</th>
                    <th>Pass Rate</th>
                    <th>Avg Penned</th>
                    <th>Avg Steps</th>
                    <th>Avg Reward</th>
                    <th>Timeouts</th>
                    <th>Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {visibleCheckpoints.length === 0 ? (
                    <tr>
                      <td colSpan={8} className="ui-table__empty">
                        No checkpoints found for the selected filter.
                      </td>
                    </tr>
                  ) : (
                    visibleCheckpoints.map((cp, idx) => {
                      const stage = resolveCheckpointStageNumber(cp);
                      const passPct = toPercent(cp.success_rate) ?? 0;
                      const timeoutPct = toPercent(cp.timeout_rate);
                      const isBest = cp.checkpoint_episode === bestCheckpointEpisode;
                      const sheepTotal = cp.environment_config?.sheep;
                      const passTone =
                        passPct >= promotionTargetPct
                          ? "ui-badge--pass"
                          : passPct >= 50
                            ? "ui-badge--warn"
                            : "ui-badge--fail";
                      const rowKey = cp.checkpoint_id
                        ? `${cp.checkpoint_id}-${idx}`
                        : `${cp.journey ?? "current"}-${cp.checkpoint_episode}-${idx}`;

                      return (
                        <tr key={rowKey}>
                          <td className="ui-mono">
                            <strong>Ep {cp.checkpoint_episode}</strong>{" "}
                            {isBest ? <span className="ui-badge ui-badge--accent">Best</span> : null}
                          </td>
                          <td className="ui-mono">S{stage}</td>
                          <td>
                            <span className={`ui-badge ${passTone}`}>{passPct}%</span>
                          </td>
                          <td className="ui-mono">
                            {cp.average_sheep_penned != null
                              ? `${cp.average_sheep_penned.toFixed(1)}${sheepTotal != null ? `/${sheepTotal}` : ""}`
                              : "—"}
                          </td>
                          <td className="ui-mono">
                            {cp.average_completion_steps != null
                              ? Math.round(cp.average_completion_steps)
                              : "—"}
                          </td>
                          <td className="ui-mono">
                            {cp.average_reward != null ? cp.average_reward.toFixed(1) : "—"}
                          </td>
                          <td className="ui-mono">{timeoutPct !== null ? `${timeoutPct}%` : "—"}</td>
                          <td>
                            <div className="ui-table-actions">
                              <button
                                type="button"
                                className="ui-btn--sm"
                                onClick={() => onSelectCheckpointForReplay(cp.checkpoint_episode)}
                              >
                                Watch Replay
                              </button>
                              <button
                                type="button"
                                className="ui-btn--sm ui-btn--ghost"
                                onClick={() => onInspectCheckpointInTroubleSpots(cp.checkpoint_episode)}
                              >
                                Inspect Seeds
                              </button>
                            </div>
                          </td>
                        </tr>
                      );
                    })
                  )}
                </tbody>
              </table>
            </div>
          </section>
        ) : null}

        {activeTab === "seeds" ? (
          <section className="ui-panel" aria-label="Stage timing and seed outcomes">
            <div className="ui-panel__header">
              <div>
                <div style={{ display: "flex", alignItems: "center", gap: "0.45rem" }}>
                  <h2 className="ui-panel__title">Stage Timing &amp; Seed Outcomes</h2>
                  <ExplainTip
                    title="Stage Timing & Seed Outcomes"
                    plain="Shows active training vs. evaluation time and which specific test seeds passed or failed on a stage."
                    technical={[
                      "Active time excludes paused and offline time.",
                      "Compare evaluations from the same stage with matched seeds, configuration, and episode budgets.",
                      "Distinguishes training rollouts, development evaluations, and held-out evaluations.",
                    ]}
                  />
                </div>
                <p className="ui-panel__subtitle">
                  Active runtime breakdown and per-seed pass/fail outcomes across journeys and runs
                </p>
              </div>
            </div>

            <div className="ui-panel__body ui-control-stack">
              {reportJourneys.length > 0 ? (
                <div className="ui-control-row">
                  <label htmlFor="stage-seed-report-journey">Report journey</label>
                  <select
                    id="stage-seed-report-journey"
                    value={reportJourney ?? ""}
                    onChange={(event) => {
                      setSelectedReportJourney(event.target.value);
                      setSelectedReportRun("");
                    }}
                  >
                    <option value="">Current journey</option>
                    {reportJourneys.map((id) => (
                      <option key={id} value={id}>
                        {id}
                      </option>
                    ))}
                  </select>
                </div>
              ) : null}
              {alternateReportRuns.length > 0 ? (
                <div className="ui-control-row">
                  <label htmlFor="stage-seed-report-run">Report run</label>
                  <select
                    id="stage-seed-report-run"
                    value={reportRunId ?? ""}
                    onChange={(event) => setSelectedReportRun(event.target.value)}
                  >
                    <option value="">{reportJourney ? "Last run in journey" : "Active run"}</option>
                    {alternateReportRuns.map((id) => (
                      <option key={id} value={id}>
                        {id}
                      </option>
                    ))}
                  </select>
                </div>
              ) : null}
              {reportJourney ? null : <StageTimingTable totals={stageRuntimeTotals} />}
              {isReadOnlyPreview ? (
                <p className="ui-inline-note">Seed reports require a connection to the training backend.</p>
              ) : (
                <>
                  <div className="ui-control-row">
                    <label htmlFor="stage-seed-report-stage">Report stage</label>
                    <select
                      id="stage-seed-report-stage"
                      value={reportStage}
                      onChange={(event) => setSelectedReportStage(Number(event.target.value))}
                    >
                      {Array.from(
                        { length: Math.max(maxCurriculumStage ?? 0, effectiveCurriculumStage) + 1 },
                        (_, stage) => (
                          <option key={stage} value={stage}>
                            Stage {stage}
                          </option>
                        ),
                      )}
                    </select>
                  </div>
                  <StageSeedReport
                    key={`${reportJourney ?? "current"}:${runId ?? "active"}:${reportStage}`}
                    stage={reportStage}
                    runId={runId}
                    journey={reportJourney}
                    onOpenReplay={reportJourney ? undefined : onOpenReplayInInspector}
                    onInspectCheckpoint={
                      reportJourney || reportRunId ? undefined : onInspectCheckpointInTroubleSpots
                    }
                  />
                </>
              )}
            </div>
          </section>
        ) : null}

        {activeTab === "timing" ? (
          <div className="ui-view-stack">
            <section className="ui-panel" aria-label="Training speed and evaluation overhead">
              <div className="ui-panel__header">
                <div>
                  <div style={{ display: "flex", alignItems: "center", gap: "0.45rem" }}>
                    <h2 className="ui-panel__title">Training Speed &amp; Eval Overhead</h2>
                    <ExplainTip
                      title="Training Speed & Eval Overhead"
                      plain="Shows how fast the computer is learning (steps per second) and how much time is spent testing versus learning."
                      technical={[
                        "Learning Throughput = policy transitions / learn_seconds across active PPO workers.",
                        "Eval Overhead % = (quick_eval_seconds + confidence_eval_seconds) / segment_total_seconds.",
                        "Bars show up to the last 12 timed checkpoints in chronological order.",
                      ]}
                    />
                  </div>
                  <p className="ui-panel__subtitle">
                    Per-segment learning throughput and evaluation breakdown
                    {chartCheckpoints.length > 0
                      ? ` (last ${chartCheckpoints.length} timed checkpoints)`
                      : ""}
                  </p>
                </div>
              </div>

              <div className="ui-panel__body">
                {!latestTiming ? (
                  <p
                    style={{
                      margin: 0,
                      fontSize: "12.5px",
                      color: "var(--ui-text-secondary, #94a3b8)",
                    }}
                  >
                    No segment timing recorded yet. Run a training segment to capture a throughput
                    baseline.
                  </p>
                ) : (
                  <div style={{ display: "flex", flexDirection: "column", gap: "0.85rem" }}>
                    <div className="ui-kpi-grid ui-kpi-grid--4">
                      <div className="ui-kpi-card">
                        <span className="ui-kpi-card__label">Learning Throughput</span>
                        <span className="ui-kpi-card__value">
                          {latestTiming.transitions_per_sec != null &&
                          Number.isFinite(latestTiming.transitions_per_sec)
                            ? `${Math.round(latestTiming.transitions_per_sec)} steps/s`
                            : "—"}
                        </span>
                        <span className="ui-kpi-card__sub">
                          {Number.isFinite(latestTiming.learn_seconds)
                            ? latestTiming.learn_seconds.toFixed(1)
                            : "0.0"}
                          s ·{" "}
                          {Number.isFinite(latestTiming.ppo_env_workers)
                            ? latestTiming.ppo_env_workers
                            : 0}{" "}
                          workers
                        </span>
                      </div>

                      <div className="ui-kpi-card">
                        <span className="ui-kpi-card__label">Quick Eval</span>
                        <span className="ui-kpi-card__value">
                          {Number.isFinite(latestTiming.quick_eval_seconds)
                            ? `${latestTiming.quick_eval_seconds.toFixed(1)}s`
                            : "0.0s"}
                        </span>
                        <span className="ui-kpi-card__sub">Per-checkpoint quick seeds</span>
                      </div>

                      <div className="ui-kpi-card">
                        <span className="ui-kpi-card__label">Confidence Eval</span>
                        <span className="ui-kpi-card__value">
                          {Number.isFinite(latestTiming.confidence_eval_seconds) &&
                          latestTiming.confidence_eval_seconds > 0
                            ? `${latestTiming.confidence_eval_seconds.toFixed(1)}s`
                            : "0.0s (skipped)"}
                        </span>
                        <span className="ui-kpi-card__sub">Full evaluation pass</span>
                      </div>

                      <div className="ui-kpi-card">
                        <span className="ui-kpi-card__label">Eval Overhead</span>
                        <span className="ui-kpi-card__value">
                          {Number.isFinite(latestTiming.eval_overhead_pct)
                            ? `${latestTiming.eval_overhead_pct.toFixed(1)}%`
                            : "0.0%"}
                        </span>
                        <span className="ui-kpi-card__sub">
                          {Number.isFinite(latestTiming.segment_total_seconds)
                            ? latestTiming.segment_total_seconds.toFixed(1)
                            : "0.0"}
                          s total ·{" "}
                          {Number.isFinite(latestTiming.save_and_export_seconds)
                            ? latestTiming.save_and_export_seconds.toFixed(2)
                            : "0.00"}
                          s save
                        </span>
                      </div>
                    </div>

                    {chartCheckpoints.length > 0 ? (
                      <div style={{ display: "flex", flexDirection: "column", gap: "0.5rem" }}>
                        <div
                          style={{
                            display: "flex",
                            flexWrap: "wrap",
                            alignItems: "center",
                            gap: "0.75rem",
                            fontSize: "11.5px",
                            color: "var(--ui-text-secondary, #94a3b8)",
                          }}
                        >
                          <span style={{ display: "inline-flex", alignItems: "center", gap: "0.35rem" }}>
                            <span
                              style={{
                                width: 10,
                                height: 10,
                                borderRadius: 2,
                                background: "var(--ui-accent, #3b82f6)",
                                display: "inline-block",
                              }}
                            />
                            Learn
                          </span>
                          <span style={{ display: "inline-flex", alignItems: "center", gap: "0.35rem" }}>
                            <span
                              style={{
                                width: 10,
                                height: 10,
                                borderRadius: 2,
                                background: "#38bdf8",
                                display: "inline-block",
                              }}
                            />
                            Quick Eval
                          </span>
                          <span style={{ display: "inline-flex", alignItems: "center", gap: "0.35rem" }}>
                            <span
                              style={{
                                width: 10,
                                height: 10,
                                borderRadius: 2,
                                background: "var(--ui-warn, #eab308)",
                                display: "inline-block",
                              }}
                            />
                            Confidence Eval
                          </span>
                          <span style={{ display: "inline-flex", alignItems: "center", gap: "0.35rem" }}>
                            <span
                              style={{
                                width: 10,
                                height: 10,
                                borderRadius: 2,
                                background: "#a855f7",
                                display: "inline-block",
                              }}
                            />
                            Save &amp; Export
                          </span>
                        </div>

                        {(() => {
                          const count = chartCheckpoints.length;
                          const slotWidth = 60;
                          const barWidth = 34;
                          const chartHeight = 110;
                          const topPad = 16;
                          const bottomPad = 36;
                          const svgWidth = Math.max(count * slotWidth, 240);
                          const svgHeight = topPad + chartHeight + bottomPad;

                          return (
                            <svg
                              viewBox={`0 0 ${svgWidth} ${svgHeight}`}
                              role="img"
                              aria-label="Segment timing stacked bar chart"
                              style={{ width: "100%", height: "auto", maxHeight: "190px", display: "block" }}
                            >
                              {chartCheckpoints.map((cp, index) => {
                                const t = cp.segment_timing;
                                const learn = Number.isFinite(t.learn_seconds)
                                  ? Math.max(0, t.learn_seconds)
                                  : 0;
                                const quick = Number.isFinite(t.quick_eval_seconds)
                                  ? Math.max(0, t.quick_eval_seconds)
                                  : 0;
                                const conf = Number.isFinite(t.confidence_eval_seconds)
                                  ? Math.max(0, t.confidence_eval_seconds)
                                  : 0;
                                const save = Number.isFinite(t.save_and_export_seconds)
                                  ? Math.max(0, t.save_and_export_seconds)
                                  : 0;
                                const totalSec =
                                  Number.isFinite(t.segment_total_seconds) &&
                                  t.segment_total_seconds > 0
                                    ? t.segment_total_seconds
                                    : learn + quick + conf + save;
                                const scale = chartHeight / maxChartTotalSeconds;
                                const learnH = learn * scale;
                                const quickH = quick * scale;
                                const confH = conf * scale;
                                const saveH = save * scale;

                                const baseY = topPad + chartHeight;
                                const learnY = baseY - learnH;
                                const quickY = learnY - quickH;
                                const confY = quickY - confH;
                                const saveY = confY - saveH;

                                const x = index * slotWidth + (slotWidth - barWidth) / 2;
                                const centerX = x + barWidth / 2;
                                const tpsLabel =
                                  t.transitions_per_sec != null &&
                                  Number.isFinite(t.transitions_per_sec)
                                    ? `${Math.round(t.transitions_per_sec)}/s`
                                    : "—";
                                const barSummary = `Ep ${cp.checkpoint_episode}: ${totalSec.toFixed(1)}s total (${tpsLabel} steps/s) — Learn ${learn.toFixed(1)}s, Quick Eval ${quick.toFixed(1)}s, Confidence Eval ${conf.toFixed(1)}s, Save ${save.toFixed(2)}s`;

                                return (
                                  <g
                                    key={cp.checkpoint_id ?? `${cp.checkpoint_episode}-${index}`}
                                    aria-label={barSummary}
                                  >
                                    <title>{barSummary}</title>
                                    {learnH > 0 ? (
                                      <rect
                                        x={x}
                                        y={learnY}
                                        width={barWidth}
                                        height={learnH}
                                        fill="var(--ui-accent, #3b82f6)"
                                      />
                                    ) : null}
                                    {quickH > 0 ? (
                                      <rect
                                        x={x}
                                        y={quickY}
                                        width={barWidth}
                                        height={quickH}
                                        fill="#38bdf8"
                                      />
                                    ) : null}
                                    {confH > 0 ? (
                                      <rect
                                        x={x}
                                        y={confY}
                                        width={barWidth}
                                        height={confH}
                                        fill="var(--ui-warn, #eab308)"
                                      />
                                    ) : null}
                                    {saveH > 0 ? (
                                      <rect
                                        x={x}
                                        y={saveY}
                                        width={barWidth}
                                        height={saveH}
                                        fill="#a855f7"
                                      />
                                    ) : null}
                                    <text
                                      x={centerX}
                                      y={Math.max(10, saveY - 4)}
                                      textAnchor="middle"
                                      fontSize="9.5"
                                      fill="var(--ui-text-secondary, #94a3b8)"
                                    >
                                      {totalSec.toFixed(1)}s
                                    </text>
                                    <text
                                      x={centerX}
                                      y={baseY + 13}
                                      textAnchor="middle"
                                      fontSize="9.5"
                                      fill="var(--ui-text, #f1f5f9)"
                                    >
                                      Ep {cp.checkpoint_episode}
                                    </text>
                                    <text
                                      x={centerX}
                                      y={baseY + 25}
                                      textAnchor="middle"
                                      fontSize="9"
                                      fill="var(--ui-text-muted, #64748b)"
                                    >
                                      {tpsLabel}
                                    </text>
                                  </g>
                                );
                              })}
                            </svg>
                          );
                        })()}
                      </div>
                    ) : null}
                  </div>
                )}
              </div>
            </section>

            <StageTimingPanel stage={timingStageProp} />
          </div>
        ) : null}

        {activeTab === "milestones" ? (
          <section className="ui-panel" aria-label="Stage milestone replays and video export">
            <div className="ui-panel__header">
              <div>
                <div style={{ display: "flex", alignItems: "center", gap: "0.45rem" }}>
                  <h2 className="ui-panel__title">Stage Milestone Replays &amp; Video Export</h2>
                  <ExplainTip
                    title="Stage Milestone Replays & Video Export"
                    plain="Browse milestone replays from completed stages and export shareable video clips."
                    technical={[
                      "Replays and metrics are tied to recorded stage milestone checkpoints.",
                      "Mounted on demand to avoid retaining extra replay buffers when viewing other tabs.",
                    ]}
                  />
                </div>
                <p className="ui-panel__subtitle">
                  Inspect recorded stage breakthroughs and export rendered herding videos
                </p>
              </div>
            </div>
            <div className="ui-panel__body">
              <ResultsPanel checkpointIndex={checkpointIndex} />
            </div>
          </section>
        ) : null}
      </div>

      <ProgressPageGuideModal
        isOpen={guideOpen}
        onClose={() => setGuideOpen(false)}
        stage={effectiveCurriculumStage}
        promotionTargetPct={promotionTargetPct}
      />
    </div>
  );
}

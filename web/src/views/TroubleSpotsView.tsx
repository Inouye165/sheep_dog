import { useMemo, useState } from "react";
import { FieldView } from "../components/FieldView";
import { SavedScenariosPanel } from "../components/SavedScenariosPanel";
import { ScenarioPanel } from "../components/ScenarioPanel";
import { StageBottlenecksPanel } from "../components/StageBottlenecksPanel";
import { TemporalDiagnosticsPanel } from "../components/TemporalDiagnosticsPanel";
import type { CheckpointMode } from "../lib/api";
import { analyzePerSeedReliability, getCheckpointStage } from "../lib/chartPipeline";
import type {
  CheckpointEntry,
  ReplayBundle,
  ReplaySnapshot,
  ScenarioIndex,
  TrainingStatus,
} from "../state/types";
import { StudioEvalArena } from "../studio/StudioEvalArena";
import type { AnyReplayPayload } from "../studio/StudioFieldCanvas";

export type TroubleSpotsMode = "failures" | "inspector" | "sandbox";

export interface TroubleSpotsViewProps {
  checkpoints: CheckpointEntry[];
  trainingStatus: TrainingStatus | null;
  effectiveCurriculumStage: number;
  bestCheckpointEpisode: number | null;
  selectedCheckpointEpisode: number | null;
  mode?: TroubleSpotsMode;
  defaultMode?: TroubleSpotsMode;
  onModeChange?: (mode: TroubleSpotsMode) => void;
  // Inspector props
  temporalReplayId: string | null;
  activeReplay: ReplayBundle | null;
  backendOffline: boolean;
  isReadOnlyPreview: boolean;
  backendStarting?: boolean;
  onSyncReplayToMainViewer: (replay: ReplayBundle, worldStep: number) => void;
  // Failure actions
  onWatchSeedReplay: (checkpointEpisode: number, seed: number) => void;
  onInspectSeedSteps: (replayId: string | null, checkpointEpisode: number, seed: number) => void;
  onSelectArenaReplay: (
    replay: AnyReplayPayload,
    checkpointEpisode: number,
    seed: number,
  ) => void;
  // Sandbox props
  scenarioSnapshot: ReplaySnapshot | null;
  scenarioReplay: ReplayBundle | null;
  scenarioIndex: ScenarioIndex | null;
  latestCheckpointEpisode: number | null;
  scenarioPersonalityStrength: number;
  scenarioSeed: number;
  scenarioFastMode: boolean;
  runningScenario: boolean;
  evaluatingScenarios: boolean;
  canEndScenarioEpisode: boolean;
  scenarioStatusLabel: string;
  scenarioPolicyLabel: string;
  scenarioCheckpointMode: CheckpointMode;
  specificScenarioCheckpointEpisode: number | null;
  selectedScenarioId: string | null;
  saveSnapshotSource: "initial" | "final";
  disabled?: boolean;
  onScenarioPersonalityStrengthChange: (value: number) => void;
  onScenarioSeedChange: (seed: number) => void;
  onScenarioFastModeChange: (enabled: boolean) => void;
  onRunScenario: () => void;
  onRestartScenarioPlayback: () => void;
  onEndScenarioEpisode: () => void;
  onScenarioCheckpointModeChange: (mode: CheckpointMode) => void;
  onSpecificScenarioCheckpointChange: (episode: number | null) => void;
  onSelectScenario: (scenarioId: string | null) => void;
  onSaveSnapshotSourceChange: (source: "initial" | "final") => void;
  onSaveScenarioFromReplay: () => void;
  onEvaluateScenarios: () => void;
  onRunSavedScenario: () => void;
}

function resolveStageNumber(entry: CheckpointEntry): number {
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

function extractReplayIdFromPath(replayPath?: string | null): string | null {
  if (!replayPath) return null;
  const base = replayPath.split(/[\\/]/).pop();
  if (!base) return null;
  return base.replace(/\.json(\.gz)?$/i, "");
}

export function TroubleSpotsView({
  checkpoints,
  trainingStatus,
  effectiveCurriculumStage,
  bestCheckpointEpisode,
  selectedCheckpointEpisode,
  mode: controlledMode,
  defaultMode = "failures",
  onModeChange,
  temporalReplayId,
  activeReplay,
  backendOffline,
  isReadOnlyPreview,
  backendStarting = false,
  onSyncReplayToMainViewer,
  onWatchSeedReplay,
  onInspectSeedSteps,
  onSelectArenaReplay,
  scenarioSnapshot,
  scenarioReplay,
  scenarioIndex,
  latestCheckpointEpisode,
  scenarioPersonalityStrength,
  scenarioSeed,
  scenarioFastMode,
  runningScenario,
  evaluatingScenarios,
  canEndScenarioEpisode,
  scenarioStatusLabel,
  scenarioPolicyLabel,
  scenarioCheckpointMode,
  specificScenarioCheckpointEpisode,
  selectedScenarioId,
  saveSnapshotSource,
  disabled = false,
  onScenarioPersonalityStrengthChange,
  onScenarioSeedChange,
  onScenarioFastModeChange,
  onRunScenario,
  onRestartScenarioPlayback,
  onEndScenarioEpisode,
  onScenarioCheckpointModeChange,
  onSpecificScenarioCheckpointChange,
  onSelectScenario,
  onSaveSnapshotSourceChange,
  onSaveScenarioFromReplay,
  onEvaluateScenarios,
  onRunSavedScenario,
}: TroubleSpotsViewProps) {
  const [internalMode, setInternalMode] = useState<TroubleSpotsMode>(defaultMode);
  const mode = controlledMode ?? internalMode;

  const setMode = (next: TroubleSpotsMode) => {
    if (controlledMode === undefined) {
      setInternalMode(next);
    }
    onModeChange?.(next);
  };

  // Normalize stage checkpoints ascending by episode so analyzePerSeedReliability's `.slice(-10)` takes the latest 10
  const stageCheckpointsAsc = useMemo(() => {
    return checkpoints
      .filter((c) => resolveStageNumber(c) === effectiveCurriculumStage)
      .sort((a, b) => a.checkpoint_episode - b.checkpoint_episode);
  }, [checkpoints, effectiveCurriculumStage]);

  const stageCheckpointsDesc = useMemo(() => {
    return [...stageCheckpointsAsc].reverse();
  }, [stageCheckpointsAsc]);

  const seedReliability = useMemo(
    () => analyzePerSeedReliability(stageCheckpointsAsc, effectiveCurriculumStage, 10),
    [stageCheckpointsAsc, effectiveCurriculumStage],
  );

  const failureBreakdown = useMemo(() => {
    const recent = stageCheckpointsAsc.slice(-10);
    let totalTrials = 0;
    let totalLosses = 0;
    let stragglerTimeouts = 0;
    let gateStalls = 0;
    let zeroPenBreakdowns = 0;

    for (const cp of recent) {
      const expectedSheep = cp.environment_config?.sheep ?? 6;
      const halfSheep = Math.ceil(expectedSheep * 0.5);
      for (const rec of cp.records ?? []) {
        totalTrials += 1;
        if (!rec.success) {
          totalLosses += 1;
          const penned = rec.sheep_penned ?? 0;
          if (penned >= halfSheep) {
            stragglerTimeouts += 1;
          } else if (penned > 0) {
            gateStalls += 1;
          } else {
            zeroPenBreakdowns += 1;
          }
        }
      }
    }

    return {
      evalCount: recent.length,
      totalTrials,
      totalLosses,
      stragglerTimeouts,
      gateStalls,
      zeroPenBreakdowns,
    };
  }, [stageCheckpointsAsc]);

  const hardestSeeds = useMemo(() => {
    const sorted = [...seedReliability.seeds].sort(
      (a, b) =>
        a.recentSuccessRate - b.recentSuccessRate ||
        b.consecutiveFailures - a.consecutiveFailures ||
        a.seed - b.seed,
    );

    return sorted.map((summary) => {
      // Prefer the newest checkpoint where this seed failed; otherwise newest with this seed
      const failingCp = stageCheckpointsDesc.find((cp) =>
        (cp.records ?? []).some((r) => r.seed === summary.seed && !r.success),
      );
      const anyCp = stageCheckpointsDesc.find((cp) =>
        (cp.records ?? []).some((r) => r.seed === summary.seed),
      );
      const targetCp = failingCp ?? anyCp ?? stageCheckpointsDesc[0] ?? null;
      const targetRec = targetCp?.records?.find((r) => r.seed === summary.seed) as
        | ((typeof targetCp.records)[number] & { replay_id?: string | null })
        | undefined;
      const checkpointEpisode = targetCp?.checkpoint_episode ?? selectedCheckpointEpisode ?? 0;
      const replayId =
        targetRec?.replay_id ??
        extractReplayIdFromPath(targetRec?.replay_path) ??
        (targetCp
          ? `checkpoint-${String(checkpointEpisode).padStart(6, "0")}-seed-${String(summary.seed).padStart(6, "0")}`
          : null);

      return {
        summary,
        checkpointEpisode,
        replayId,
      };
    });
  }, [seedReliability.seeds, stageCheckpointsDesc, selectedCheckpointEpisode]);

  const scenarioPenned = scenarioSnapshot?.penned_count ?? 0;
  const scenarioSheepCount =
    scenarioSnapshot?.sheep?.length ?? scenarioReplay?.final_snapshot?.sheep?.length ?? 0;

  return (
    <div className="ui-view-stack ui-view-stack--full">
      {/* Segmented sub-nav */}
      <div>
        <div className="ui-subnav" role="tablist" aria-label="Trouble spots view">
          <button
            type="button"
            role="tab"
            aria-selected={mode === "failures"}
            className={`ui-subnav__item${mode === "failures" ? " ui-subnav__item--active" : ""}`}
            onClick={() => setMode("failures")}
          >
            Scenario &amp; Seed Failures
          </button>
          <button
            type="button"
            role="tab"
            aria-selected={mode === "inspector"}
            className={`ui-subnav__item${mode === "inspector" ? " ui-subnav__item--active" : ""}`}
            onClick={() => setMode("inspector")}
          >
            Step-by-Step Inspector
          </button>
          <button
            type="button"
            role="tab"
            aria-selected={mode === "sandbox"}
            className={`ui-subnav__item${mode === "sandbox" ? " ui-subnav__item--active" : ""}`}
            onClick={() => setMode("sandbox")}
          >
            Scenario Sandbox
          </button>
        </div>
      </div>

      {/* Mode 1: Scenario & Seed Failures */}
      {mode === "failures" ? (
        <div className="ui-view-stack">
          <section className="ui-panel" aria-label="Failure breakdown and hardest seeds">
            <div className="ui-panel__header">
              <div>
                <h2 className="ui-panel__title">Failure Breakdown &amp; Hardest Seeds</h2>
                <p className="ui-panel__subtitle">
                  Stage {effectiveCurriculumStage} · Aggregated across last{" "}
                  {failureBreakdown.evalCount} formal evaluations ({failureBreakdown.totalLosses}{" "}
                  losses / {failureBreakdown.totalTrials} trials)
                </p>
              </div>
              {seedReliability.overallMedianSuccessfulSteps != null ? (
                <span className="ui-badge">
                  Median pass {seedReliability.overallMedianSuccessfulSteps}s
                </span>
              ) : null}
            </div>

            <div className="ui-panel__body ui-view-stack">
              <div className="ui-failure-grid">
                <div className="ui-failure-stat">
                  <span className="ui-failure-stat__label">
                    Straggler timeouts (&gt;=50% penned)
                  </span>
                  <span className="ui-failure-stat__value">
                    {failureBreakdown.stragglerTimeouts}
                  </span>
                  <span className="ui-failure-stat__hint">
                    Main flock penned; ran out of budget on final sheep
                  </span>
                </div>

                <div className="ui-failure-stat">
                  <span className="ui-failure-stat__label">
                    Gate / fence stalls (1-49% penned)
                  </span>
                  <span className="ui-failure-stat__value">{failureBreakdown.gateStalls}</span>
                  <span className="ui-failure-stat__hint">
                    Partial progress stalled near gate or outer pen wall
                  </span>
                </div>

                <div className="ui-failure-stat">
                  <span className="ui-failure-stat__label">Zero-pen breakdowns (0 penned)</span>
                  <span className="ui-failure-stat__value">
                    {failureBreakdown.zeroPenBreakdowns}
                  </span>
                  <span className="ui-failure-stat__hint">
                    Initial collection, corner extraction, or scatter failure
                  </span>
                </div>
              </div>

              <div className="ui-table-wrap">
                <table className="ui-table">
                  <thead>
                    <tr>
                      <th>Seed</th>
                      <th>Win Rate</th>
                      <th>Median Pass Steps</th>
                      <th>Status</th>
                      <th>Actions</th>
                    </tr>
                  </thead>
                  <tbody>
                    {hardestSeeds.length === 0 ? (
                      <tr>
                        <td colSpan={5} className="ui-table__empty">
                          No formal evaluation records for Stage {effectiveCurriculumStage} yet.
                        </td>
                      </tr>
                    ) : (
                      hardestSeeds.map(({ summary, checkpointEpisode, replayId }) => {
                        const statusTone = summary.isBlindSpot
                          ? "ui-badge--fail"
                          : summary.isInefficient
                            ? "ui-badge--warn"
                            : summary.failureCount === 0
                              ? "ui-badge--pass"
                              : "";
                        return (
                          <tr key={summary.seed}>
                            <td className="ui-mono">
                              <strong>Seed #{summary.seed}</strong>
                            </td>
                            <td className="ui-mono">
                              {summary.recentSuccessRate}% ({summary.successCount}/
                              {summary.totalTrials})
                            </td>
                            <td className="ui-mono">
                              {summary.typicalSuccessfulSteps != null
                                ? `${summary.typicalSuccessfulSteps}s`
                                : "—"}
                            </td>
                            <td>
                              <span className={`ui-badge ${statusTone}`}>{summary.statusText}</span>
                            </td>
                            <td>
                              <div className="ui-table-actions">
                                <button
                                  type="button"
                                  className="ui-btn--sm"
                                  onClick={() =>
                                    onWatchSeedReplay(checkpointEpisode, summary.seed)
                                  }
                                >
                                  Watch Replay
                                </button>
                                <button
                                  type="button"
                                  className="ui-btn--sm ui-btn--ghost"
                                  onClick={() =>
                                    onInspectSeedSteps(replayId, checkpointEpisode, summary.seed)
                                  }
                                >
                                  Inspect Steps
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
            </div>
          </section>

          <StageBottlenecksPanel
            currentStage={effectiveCurriculumStage}
            runId={trainingStatus?.run_id}
          />

          <StudioEvalArena
            checkpoints={checkpoints}
            effectiveStage={effectiveCurriculumStage}
            bestCheckpointEpisode={bestCheckpointEpisode}
            initialCheckpointEpisode={selectedCheckpointEpisode}
            onSelectSeedForMainPlayer={onSelectArenaReplay}
          />
        </div>
      ) : null}

      {/* Mode 2: Step-by-Step Inspector */}
      {mode === "inspector" ? (
        <div className="ui-inspector-wrap">
          <TemporalDiagnosticsPanel
            initialReplayId={temporalReplayId}
            activeReplay={activeReplay}
            activeReplayId={temporalReplayId}
            backendOffline={backendOffline}
            isReadOnlyPreview={isReadOnlyPreview}
            backendStarting={backendStarting}
            onSyncReplayToMainViewer={onSyncReplayToMainViewer}
          />
        </div>
      ) : null}

      {/* Mode 3: Scenario Sandbox */}
      {mode === "sandbox" ? (
        <div className="ui-train-split">
          <div className="ui-train-main">
            <div className="ui-panel ui-train-field-panel">
              <FieldView snapshot={scenarioSnapshot} />
            </div>

            <section className="ui-meta-strip" aria-label="Scenario status">
              <div className="ui-meta-strip__row">
                <div className="ui-meta-strip__stats">
                  <span className="ui-badge">{scenarioStatusLabel}</span>
                  <span className="ui-badge">{scenarioPolicyLabel}</span>
                  <span className="ui-meta-strip__stat">
                    Step <strong>{scenarioSnapshot?.step ?? 0}</strong>
                  </span>
                  <span className="ui-meta-strip__stat">
                    Seed <strong>{scenarioSeed}</strong>
                  </span>
                  <span className="ui-meta-strip__stat">
                    Penned{" "}
                    <strong>
                      {scenarioPenned}/{scenarioSheepCount}
                    </strong>
                  </span>
                </div>
              </div>
            </section>
          </div>

          <aside className="ui-train-rail" aria-label="Scenario sandbox controls">
            <ScenarioPanel
              personalityStrength={scenarioPersonalityStrength}
              seed={scenarioSeed}
              fastMode={scenarioFastMode}
              running={runningScenario}
              canEndEpisode={canEndScenarioEpisode}
              hasReplay={Boolean(scenarioReplay)}
              policyLabel={scenarioPolicyLabel}
              onPersonalityStrengthChange={onScenarioPersonalityStrengthChange}
              onSeedChange={onScenarioSeedChange}
              onFastModeChange={onScenarioFastModeChange}
              onRun={onRunScenario}
              onRestart={onRestartScenarioPlayback}
              onEndEpisode={onEndScenarioEpisode}
              disabled={disabled}
            />

            <SavedScenariosPanel
              scenarioIndex={scenarioIndex}
              checkpoints={checkpoints}
              latestCheckpointEpisode={latestCheckpointEpisode}
              bestCheckpointEpisode={bestCheckpointEpisode}
              checkpointMode={scenarioCheckpointMode}
              specificCheckpointEpisode={specificScenarioCheckpointEpisode}
              selectedScenarioId={selectedScenarioId}
              scenarioReplay={scenarioReplay}
              saveSnapshotSource={saveSnapshotSource}
              running={runningScenario || evaluatingScenarios}
              disabled={disabled}
              onCheckpointModeChange={onScenarioCheckpointModeChange}
              onSpecificCheckpointChange={onSpecificScenarioCheckpointChange}
              onSelectScenario={onSelectScenario}
              onSaveSnapshotSourceChange={onSaveSnapshotSourceChange}
              onSaveFromReplay={onSaveScenarioFromReplay}
              onEvaluate={onEvaluateScenarios}
              onRunScenario={onRunSavedScenario}
              canSaveFromReplay={Boolean(scenarioSnapshot)}
            />
          </aside>
        </div>
      ) : null}
    </div>
  );
}

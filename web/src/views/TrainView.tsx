import { useMemo, type ReactNode } from "react";
import { FieldView } from "../components/FieldView";
import { getCheckpointStage } from "../lib/chartPipeline";
import { formatStageSpec, parseStageSpec } from "../lib/stageSpec";
import type {
  AutoPromoteGateDiagnostics,
  CheckpointEntry,
  ReplayBundle,
  ReplaySnapshot,
  TrainingStatus,
} from "../state/types";

const STAGE_DESCRIPTIONS: Record<number, string> = {
  0: "Full problem — 3 dogs, 6 sheep, 80×60 grid",
  1: "1 dog · 1 sheep · fixed easy penning",
  2: "1 dog · 1 sheep · mild start randomization",
  3: "1 dog · 2 sheep · fixed mini-flock",
  4: "1 dog · 2 sheep · randomized mini-flock",
  5: "2 dogs · 3 sheep · fixed teamwork",
  6: "2 dogs · 3 sheep · tiny nearby stray starts",
  7: "2 dogs · 4 sheep · early nearby stray collection",
  8: "3 dogs · 4 sheep · nearby stray emphasis",
  9: "3 dogs · 4 sheep · stronger nearby stray recovery",
  10: "3 dogs · 5 sheep · nearby + first farther strays",
  11: "3 dogs · 5 sheep · farther stray recovery",
  12: "3 dogs · 6 sheep · group + one stray",
  13: "3 dogs · 6 sheep · two nearby strays",
  14: "3 dogs · 6 sheep · split flock (3+3)",
  15: "3 dogs · 6 sheep · partially scattered",
  16: "3 dogs · 6 sheep · scattered sheep",
  17: "3 dogs · 6 sheep · moving pen same wall",
  18: "3 dogs · 6 sheep · any-wall pen",
  19: "3 dogs · 6 sheep · wall pen away from corners",
  20: "3 dogs · 6 sheep · interior pen",
  21: "3 dogs · 6 sheep · random pen + random sheep",
  22: "3 dogs · 6 sheep · wider split/stray recovery",
  23: "3 dogs · 6 sheep · heavy scattered recovery",
  24: "3 dogs · 6 sheep · all-corners starts",
  25: "3 dogs · 6 sheep · corner-heavy starts before hard random mix",
  26: "3 dogs · 6 sheep · hard spawn mix (no personality bias)",
  27: "3 dogs · 6 sheep · mild personality variation",
  28: "3 dogs · 6 sheep · moderate personality variation",
  29: "3 dogs · 6 sheep · weaker cohesion bridge",
  30: "3 dogs · 6 sheep · disable no-pressure cohesion",
  31: "3 dogs · 6 sheep · reduce cohesion + stronger personalities",
  32: "3 dogs · 6 sheep · lowest cohesion + strongest personalities",
  33: "3 dogs · 6 sheep · mild personality variation (144×104)",
  34: "3 dogs · 6 sheep · moderate personality variation + wide scatter",
  35: "3 dogs · 6 sheep · weaker cohesion (pressure-coupled bridge)",
  36: "3 dogs · 6 sheep · disable no-pressure cohesion",
  37: "3 dogs · 6 sheep · reduced cohesion + stronger personalities",
  38: "3 dogs · 6 sheep · mastery benchmark (lowest cohesion + max personalities)",
};

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

function formatPolicyModeLabel(policyKey: string | undefined): string {
  switch (policyKey) {
    case "random_untrained":
      return "Random untrained";
    case "instinct_only":
      return "Instinct only";
    case "heuristic_expert":
      return "Heuristic expert";
    case "trained_policy":
      return "Trained policy";
    case "neural_policy":
      return "Neural PPO";
    default:
      return policyKey && policyKey !== "unloaded" ? policyKey : "No replay loaded";
  }
}

function resolvePolicyNote(policyKey: string | undefined): string | null {
  switch (policyKey) {
    case "random_untrained":
      return "Random untrained dogs have no herding strategy and no knowledge of the pen.";
    case "instinct_only":
      return "Instinct-only dogs can chase, circle, avoid diving into the flock, and recover nearby sheep, but they do not know where the pen is.";
    case "heuristic_expert":
      return "This expert heuristic is pen-aware and uses scripted pressure positioning toward the target.";
    case "trained_policy":
      return "Pen-directed behavior here comes from learned training weights rather than default instinct.";
    case "neural_policy":
      return "This replay is using the learned neural PPO policy rather than the instinct-only baseline.";
    default:
      return null;
  }
}

interface LiveProgressSnapshot {
  segStepsDone: number | null;
  segStepsTotal: number | null;
  segPct: number | null;
  batchPct: number;
  displayPct: number;
  timestepsSinceCheckpoint: number | null;
  totalSegments: number | null;
  completedSegments: number;
  liveEnvEpisode: number | null;
  latestResult: string | null;
  latestReward: number | null;
  rolloutWindowCount: number | null;
  rolloutWinPct: number | null;
}

function deriveLiveProgress(
  trainingStatus: TrainingStatus | null,
  batchCompleted: number,
  batchTotal: number,
  _stageCheckpointsLength: number,
): LiveProgressSnapshot {
  const message = trainingStatus?.message ?? "";
  const tsMatch = message.match(/(\d+)\s*\/\s*(\d+)\s*timesteps(?:\s*\((\d+)%\))?/i);
  const segMatch = message.match(/segment\s+(\d+)\s*\/\s*(\d+)/i);

  let segStepsDone = tsMatch ? Number(tsMatch[1]) : null;
  let segStepsTotal = tsMatch ? Number(tsMatch[2]) : null;
  const msgPct = tsMatch?.[3] != null ? Number(tsMatch[3]) : null;
  const msgSegIndex = segMatch ? Number(segMatch[1]) : null;
  const msgSegTotal = segMatch ? Number(segMatch[2]) : null;

  const timestepsSinceCheckpoint =
    trainingStatus?.timesteps_since_latest_checkpoint != null &&
    trainingStatus.timesteps_since_latest_checkpoint > 0
      ? trainingStatus.timesteps_since_latest_checkpoint
      : null;

  if (segStepsDone == null && timestepsSinceCheckpoint != null) {
    const batchTotalTs = trainingStatus?.batch_total_timesteps ?? 0;
    const totalSegsHint = trainingStatus?.batch_total_segments ?? msgSegTotal ?? 0;
    if (batchTotalTs > 0 && totalSegsHint > 0) {
      const approxSegTotal = Math.max(1, Math.round(batchTotalTs / totalSegsHint));
      if (timestepsSinceCheckpoint <= approxSegTotal) {
        segStepsDone = timestepsSinceCheckpoint;
        segStepsTotal = approxSegTotal;
      }
    }
  }

  let segPct: number | null = null;
  if (msgPct != null && Number.isFinite(msgPct)) {
    segPct = Math.min(100, Math.max(0, Math.round(msgPct)));
  } else if (segStepsDone != null && segStepsTotal != null && segStepsTotal > 0) {
    segPct = Math.min(100, Math.max(0, Math.round((segStepsDone / segStepsTotal) * 100)));
  } else if ((trainingStatus?.batch_completed_segments ?? 0) % 1 > 0) {
    segPct = Math.round(((trainingStatus?.batch_completed_segments ?? 0) % 1) * 100);
  }

  const batchPct =
    batchTotal > 0 ? Math.min(100, Math.round((batchCompleted / batchTotal) * 100)) : 0;
  const displayPct = segPct ?? batchPct;

  const totalSegments = trainingStatus?.batch_total_segments ?? msgSegTotal ?? null;
  const baseCompletedSegs = Math.floor(trainingStatus?.batch_completed_segments ?? 0);
  const completedSegments =
    msgSegIndex != null
      ? Math.max(baseCompletedSegs, Math.max(0, msgSegIndex - 1))
      : baseCompletedSegs;

  const liveEnvEpisode =
    trainingStatus?.latest_completed_environment_episode ??
    trainingStatus?.current_episode ??
    null;
  const latestResult = trainingStatus?.latest_episode_result ?? null;
  const latestReward = trainingStatus?.latest_episode_reward ?? null;
  const rolloutWindowCount = trainingStatus?.live_rollout_window_count ?? null;
  const rolloutWinPct = toPercent(trainingStatus?.live_rollout_success_rate ?? null);

  return {
    segStepsDone,
    segStepsTotal,
    segPct,
    batchPct,
    displayPct,
    timestepsSinceCheckpoint,
    totalSegments,
    completedSegments,
    liveEnvEpisode,
    latestResult,
    latestReward,
    rolloutWindowCount,
    rolloutWinPct,
  };
}

function rolloutResultBadgeTone(result: string | null): string {
  if (!result) return "";
  const upper = result.toUpperCase();
  if (upper === "SUCCESS" || upper === "WIN") return "ui-badge--pass";
  if (
    upper.includes("TIMEOUT") ||
    upper.includes("STOPPED") ||
    upper.includes("NO_PROGRESS") ||
    upper.includes("NO-PROGRESS")
  ) {
    return "ui-badge--warn";
  }
  return "ui-badge--fail";
}

interface UseTrainViewDerivedInput {
  checkpoints: CheckpointEntry[];
  selectedCheckpoint: CheckpointEntry | null;
  selectedCheckpointEpisode: number | null;
  bestCheckpointEpisode: number | null;
  curriculumStage: number;
  seedOptions: number[];
  selectedSeed: number | null;
  latestSuccessRate: number | null;
  autoPromoteGateProp?: AutoPromoteGateDiagnostics | null;
  trainingStatus: TrainingStatus | null;
  replay: ReplayBundle | null;
  snapshot: ReplaySnapshot | null;
}

/**
 * Derive scalar summaries and lightweight option/chip view-models outside
 * `TrainView` so V8 closures attached to JSX event handlers never share a
 * lexical `Context` with `checkpoints`, `selectedCheckpoint`, or `replay`.
 */
function useTrainViewDerived({
  checkpoints,
  selectedCheckpoint,
  selectedCheckpointEpisode,
  bestCheckpointEpisode,
  curriculumStage,
  seedOptions,
  selectedSeed,
  latestSuccessRate,
  autoPromoteGateProp,
  trainingStatus,
  replay,
  snapshot,
}: UseTrainViewDerivedInput) {
  const totalFrames = replay?.frames?.length ?? 0;

  const activeCheckpoint = useMemo(() => {
    if (selectedCheckpoint) return selectedCheckpoint;
    if (selectedCheckpointEpisode != null) {
      const byEp = checkpoints.find((c) => c.checkpoint_episode === selectedCheckpointEpisode);
      if (byEp) return byEp;
    }
    if (bestCheckpointEpisode != null) {
      const best = checkpoints.find((c) => c.checkpoint_episode === bestCheckpointEpisode);
      if (best) return best;
    }
    return checkpoints[checkpoints.length - 1] ?? null;
  }, [checkpoints, selectedCheckpoint, selectedCheckpointEpisode, bestCheckpointEpisode]);

  const stageCheckpoints = useMemo(() => {
    return checkpoints
      .filter((c) => resolveCheckpointStageNumber(c) === curriculumStage)
      .sort((a, b) => a.checkpoint_episode - b.checkpoint_episode);
  }, [checkpoints, curriculumStage]);

  const latestStageCheckpoint =
    stageCheckpoints.length > 0 ? stageCheckpoints[stageCheckpoints.length - 1] : null;

  const bestStageCheckpoint = useMemo(() => {
    if (stageCheckpoints.length === 0) return null;
    return stageCheckpoints.reduce((best, entry) => {
      if (!best) return entry;
      if (entry.success_rate > best.success_rate) return entry;
      if (
        entry.success_rate === best.success_rate &&
        (entry.average_completion_steps ?? Infinity) < (best.average_completion_steps ?? Infinity)
      ) {
        return entry;
      }
      return best;
    }, stageCheckpoints[0]);
  }, [stageCheckpoints]);

  const gate = autoPromoteGateProp ?? trainingStatus?.auto_promote_gate ?? null;

  const latestStageRecords = latestStageCheckpoint?.records ?? [];
  const latestFailCount = latestStageRecords.filter((r) => !r.success).length;
  const latestTotalSeeds = latestStageRecords.length;

  const evalPassPct = toPercent(
    latestSuccessRate ?? latestStageCheckpoint?.success_rate ?? null,
  );
  const peakPassPct = toPercent(bestStageCheckpoint?.success_rate ?? null);
  const avgStepsRaw =
    latestStageCheckpoint?.average_completion_steps ?? gate?.recent_avg_steps ?? null;
  const avgStepsDisplay =
    avgStepsRaw != null && Number.isFinite(avgStepsRaw) ? String(Math.round(avgStepsRaw)) : "—";

  const avgPennedRaw =
    trainingStatus?.latest_avg_sheep_penned ?? latestStageCheckpoint?.average_sheep_penned ?? null;
  const expectedSheep =
    latestStageCheckpoint?.environment_config?.sheep ??
    activeCheckpoint?.environment_config?.sheep ??
    replay?.environment?.sheep ??
    snapshot?.sheep?.length ??
    6;
  const avgPennedDisplay =
    avgPennedRaw != null && Number.isFinite(avgPennedRaw)
      ? `${Number.isInteger(avgPennedRaw) ? avgPennedRaw : avgPennedRaw.toFixed(1)}/${expectedSheep}`
      : "—";

  const qualifiedCount = gate?.qualified_evaluations ?? gate?.recent_qualifying_checkpoints ?? 0;
  const qualifiedRequired = gate?.qualified_evaluations_required ?? 2;
  const thresholdPct =
    toPercent(
      gate?.success_threshold ??
        trainingStatus?.auto_promote_threshold ??
        (curriculumStage <= 1 ? 0.8 : 0.9),
    ) ?? 90;

  const stageCheckpointsCount = stageCheckpoints.length;

  const decisionSummary = useMemo(() => {
    if (gate?.ready) {
      return {
        headline: `Ready to promote to Stage ${curriculumStage + 1}`,
        rationale: gate.reason || "Formal evaluation criteria satisfied for promotion.",
        action: "promote" as const,
      };
    }
    if (latestFailCount > 0 && latestTotalSeeds > 0) {
      return {
        headline: `${latestFailCount} of ${latestTotalSeeds} seeds failing — inspect trouble spots`,
        rationale:
          gate?.reason || "Inspect failing benchmark seeds and spatial bottlenecks before promoting.",
        action: "trouble" as const,
      };
    }
    if (gate && (gate.qualified_evaluations ?? 0) > 0) {
      return {
        headline: `Building qualifying evals (${gate.qualified_evaluations}/${gate.qualified_evaluations_required ?? 2})`,
        rationale:
          gate.reason ||
          `Collecting qualifying formal evaluations within the rolling ${gate.window_size ?? 8}-eval window to satisfy the promotion gate.`,
        action: "none" as const,
      };
    }
    return {
      headline: `Stage ${curriculumStage} training status`,
      rationale:
        gate?.reason ||
        (stageCheckpointsCount === 0
          ? "No formal evaluations recorded for this stage yet."
          : STAGE_DESCRIPTIONS[curriculumStage] ?? `Evaluating performance on Stage ${curriculumStage}.`),
      action: "none" as const,
    };
  }, [gate, curriculumStage, latestFailCount, latestTotalSeeds, stageCheckpointsCount]);

  // Run status & policy summary
  const policyKey =
    replay?.policy_name ??
    selectedCheckpoint?.policy_name ??
    replay?.policy_mode ??
    selectedCheckpoint?.policy_mode ??
    "unloaded";
  const policyModeLabel = formatPolicyModeLabel(policyKey);
  const policyNote = resolvePolicyNote(policyKey);
  const sheepPenned = snapshot?.penned_count ?? 0;
  const totalSheepInReplay =
    snapshot?.sheep?.length ?? replay?.final_snapshot?.sheep?.length ?? 0;
  const activeRoles = snapshot?.dogs?.map((dog) => dog.role).filter(Boolean) ?? [];
  const activeRoleSummary = activeRoles.length > 0 ? activeRoles.join(", ") : "—";
  const displaySeed = selectedSeed ?? replay?.seed ?? null;
  const noProgressGuardActive =
    (replay?.stats?.no_progress_steps ?? 0) > 0 || snapshot?.status === "no-progress";

  // Active checkpoint seed strip data
  const checkpointRecords = activeCheckpoint?.records ?? [];
  const checkpointRecordsCount = checkpointRecords.length;
  const checkpointPassCount = checkpointRecords.filter((r) => r.success).length;
  const checkpointAvgReward =
    activeCheckpoint?.average_reward != null && Number.isFinite(activeCheckpoint.average_reward)
      ? activeCheckpoint.average_reward.toFixed(1)
      : checkpointRecords.length > 0
        ? (
            checkpointRecords.reduce((sum, r) => sum + (r.reward_total ?? 0), 0) /
            checkpointRecords.length
          ).toFixed(1)
        : null;

  const seedOptionItems = useMemo(() => {
    const records = activeCheckpoint?.records;
    const totalSheep = activeCheckpoint?.environment_config?.sheep ?? expectedSheep;
    return seedOptions.map((seed) => {
      const record = records?.find((r) => r.seed === seed);
      let label = `Seed ${seed}`;
      if (record) {
        const outcome = record.success
          ? "success"
          : record.timeout
            ? "timeout"
            : record.stopped
              ? "stopped"
              : "fail";
        label = `Seed ${seed} — ${record.sheep_penned}/${totalSheep} · ${record.steps}s · ${outcome}`;
      }
      return { seed, label };
    });
  }, [activeCheckpoint, expectedSheep, seedOptions]);

  const stripSeedItems = useMemo(() => {
    const records = activeCheckpoint?.records ?? [];
    const seeds = records.length > 0 ? records.map((r) => r.seed) : seedOptions;
    return seeds.map((seed) => {
      const rec = records.find((r) => r.seed === seed);
      const chipTone = rec ? (rec.success ? "ui-seed-chip--pass" : "ui-seed-chip--fail") : "";
      const statusText = rec ? (rec.success ? "Pass" : "Fail") : "";
      return { seed, chipTone, statusText };
    });
  }, [activeCheckpoint, seedOptions]);

  const dropdownCheckpoints = useMemo(() => {
    if (checkpoints.length <= 200) {
      return checkpoints;
    }
    const startIndex = checkpoints.length - 200;
    const tail = checkpoints.slice(startIndex);
    const tailEpisodes = new Set(tail.map((c) => c.checkpoint_episode));
    const extras: CheckpointEntry[] = [];
    const seenExtraEpisodes = new Set<number>();
    for (let i = 0; i < startIndex; i += 1) {
      const entry = checkpoints[i];
      const ep = entry.checkpoint_episode;
      if (
        (ep === selectedCheckpointEpisode || ep === bestCheckpointEpisode) &&
        !tailEpisodes.has(ep) &&
        !seenExtraEpisodes.has(ep)
      ) {
        seenExtraEpisodes.add(ep);
        extras.push(entry);
      }
    }
    return extras.length > 0 ? [...extras, ...tail] : tail;
  }, [checkpoints, selectedCheckpointEpisode, bestCheckpointEpisode]);

  const checkpointSelectOptions: ReactNode = useMemo(() => {
    return dropdownCheckpoints.map((entry, idx) => {
      const ep = entry.checkpoint_episode;
      const stage = resolveCheckpointStageNumber(entry);
      const ratePct = toPercent(entry.success_rate) ?? 0;
      const steps = Math.round(entry.average_completion_steps ?? 0);
      const isBest = ep === bestCheckpointEpisode;
      const label = `S${stage} · Ep ${ep} · ${ratePct}% · ${steps}s${isBest ? " (Best)" : ""}`;
      const key = entry.checkpoint_id
        ? `${entry.checkpoint_id}-${idx}`
        : `${entry.run_id ?? "cp"}-ep-${ep}-${idx}`;
      return (
        <option key={key} value={ep}>
          {label}
        </option>
      );
    });
  }, [dropdownCheckpoints, bestCheckpointEpisode]);

  return {
    hasCheckpoints: checkpoints.length > 0,
    checkpointSelectOptions,
    seedOptionItems,
    stripSeedItems,
    activeCheckpointEpisode: activeCheckpoint?.checkpoint_episode ?? null,
    checkpointRecordsCount,
    checkpointPassCount,
    checkpointAvgReward,
    stageCheckpointsCount,
    evalPassPct,
    peakPassPct,
    avgStepsDisplay,
    avgPennedDisplay,
    expectedSheep,
    gate,
    latestFailCount,
    qualifiedCount,
    qualifiedRequired,
    thresholdPct,
    decisionSummary,
    policyModeLabel,
    policyNote,
    totalFrames,
    sheepPenned,
    totalSheepInReplay,
    activeRoleSummary,
    displaySeed,
    noProgressGuardActive,
  };
}

export interface TrainViewProps {
  checkpoints: CheckpointEntry[];
  selectedCheckpoint: CheckpointEntry | null;
  selectedCheckpointEpisode: number | null;
  bestCheckpointEpisode: number | null;
  seedOptions: number[];
  selectedSeed: number | null;
  replay: ReplayBundle | null;
  snapshot: ReplaySnapshot | null;
  frameIndex: number;
  isPlaying: boolean;
  statusLabel: string;
  canEndEpisode: boolean;
  runningCurrent: boolean;
  loadingSelectedReplay: boolean;
  playbackFastMode: boolean;
  trainingStatus: TrainingStatus | null;
  curriculumStage: number;
  maxCurriculumStage: number;
  episodes: number;
  fastMode: boolean;
  enableInstincts: boolean;
  autoPromote: boolean;
  continuous?: boolean;
  debugRewardBreakdown: boolean;
  startingModelSource: string;
  stageSpec?: string;
  onStageSpecChange?: (value: string) => void;
  passLabel?: string;
  onPassLabelChange?: (value: string) => void;
  running: boolean;
  clearing: boolean;
  isStartingTraining: boolean;
  lifecycleAction: "pause" | "stop" | null;
  latestSuccessRate: number | null;
  trainingError?: string | null;
  autoPromoteGate?: AutoPromoteGateDiagnostics | null;
  resumeAvailable?: boolean;
  resumeRemainingEpisodes?: number | null;
  isReadOnlyPreview?: boolean;
  disabled?: boolean;
  onSelectCheckpointEpisode: (episode: number) => void;
  onSelectSeed: (seed: number) => void;
  onSelectSeedAndReplay: (seed: number) => void;
  onRunCurrentReplay: () => void;
  onReplaySelected: () => void;
  onEndEpisode: () => void;
  onTogglePlay: () => void;
  onStepFrame: (delta: number) => void;
  onScrubFrame: (frameIndex: number) => void;
  onPlaybackFastModeChange: (enabled: boolean) => void;
  onOpenInspector: () => void;
  onOpenTroubleSpots: () => void;
  onOpenProgress: () => void;
  onPromote: () => void;
  onCurriculumStageChange: (stage: number) => void;
  onEpisodesChange: (episodes: number) => void;
  onStartTraining: () => void;
  onResumeTraining: () => void;
  onPauseTraining: () => void;
  onStopTraining: () => void;
  onEnableInstinctsChange: (enabled: boolean) => void;
  onAutoPromoteChange: (enabled: boolean) => void;
  onContinuousChange?: (enabled: boolean) => void;
  onStartingModelSourceChange: (source: string) => void;
  onFastModeChange: (enabled: boolean) => void;
  onDebugRewardBreakdownChange: (enabled: boolean) => void;
  onStartRemediation: () => void;
  onClearTraining: () => void;
}

export function TrainView({
  checkpoints,
  selectedCheckpoint,
  selectedCheckpointEpisode,
  bestCheckpointEpisode,
  seedOptions,
  selectedSeed,
  replay,
  snapshot,
  frameIndex,
  isPlaying,
  statusLabel,
  canEndEpisode,
  runningCurrent,
  loadingSelectedReplay,
  playbackFastMode,
  trainingStatus,
  curriculumStage,
  maxCurriculumStage,
  episodes,
  fastMode,
  enableInstincts,
  autoPromote,
  continuous = false,
  debugRewardBreakdown,
  startingModelSource,
  stageSpec = "",
  onStageSpecChange,
  passLabel = "",
  onPassLabelChange,
  running,
  clearing,
  isStartingTraining,
  lifecycleAction,
  latestSuccessRate,
  trainingError = null,
  autoPromoteGate: autoPromoteGateProp,
  resumeAvailable: resumeAvailableProp,
  resumeRemainingEpisodes: resumeRemainingProp,
  isReadOnlyPreview = false,
  disabled = false,
  onSelectCheckpointEpisode,
  onSelectSeed,
  onSelectSeedAndReplay,
  onRunCurrentReplay,
  onReplaySelected,
  onEndEpisode,
  onTogglePlay,
  onStepFrame,
  onScrubFrame,
  onPlaybackFastModeChange,
  onOpenInspector,
  onOpenTroubleSpots,
  onOpenProgress,
  onPromote,
  onCurriculumStageChange,
  onEpisodesChange,
  onStartTraining,
  onResumeTraining,
  onPauseTraining,
  onStopTraining,
  onEnableInstinctsChange,
  onAutoPromoteChange,
  onContinuousChange,
  onStartingModelSourceChange,
  onFastModeChange,
  onDebugRewardBreakdownChange,
  onStartRemediation,
  onClearTraining,
}: TrainViewProps) {
  const {
    hasCheckpoints,
    checkpointSelectOptions,
    seedOptionItems,
    stripSeedItems,
    activeCheckpointEpisode,
    checkpointRecordsCount,
    checkpointPassCount,
    checkpointAvgReward,
    stageCheckpointsCount,
    evalPassPct,
    peakPassPct,
    avgStepsDisplay,
    avgPennedDisplay,
    gate,
    latestFailCount,
    qualifiedCount,
    qualifiedRequired,
    thresholdPct,
    decisionSummary,
    policyModeLabel,
    policyNote,
    totalFrames,
    sheepPenned,
    totalSheepInReplay,
    activeRoleSummary,
    displaySeed,
    noProgressGuardActive,
  } = useTrainViewDerived({
    checkpoints,
    selectedCheckpoint,
    selectedCheckpointEpisode,
    bestCheckpointEpisode,
    curriculumStage,
    seedOptions,
    selectedSeed,
    latestSuccessRate,
    autoPromoteGateProp,
    trainingStatus,
    replay,
    snapshot,
  });

  const clampedFrameIndex = totalFrames > 0 ? Math.min(Math.max(0, frameIndex), totalFrames - 1) : 0;
  const activeWarningsCount = trainingStatus?.early_warnings_summary?.total_active ?? 0;

  const statusToneClass =
    statusLabel === "success"
      ? "ui-badge--pass"
      : statusLabel === "running"
        ? "ui-badge--accent"
        : statusLabel === "timeout" || statusLabel === "stopped"
          ? "ui-badge--warn"
          : "";

  // Live evaluation progress check
  const evalStatus = trainingStatus?.evaluation_status;
  const evalInProgress = Boolean(
    trainingStatus?.evaluation_in_progress ||
      evalStatus?.in_progress ||
      trainingStatus?.phase === "evaluation",
  );
  const evalMode = evalStatus?.mode || trainingStatus?.evaluation_mode || "evaluation";
  const evalCompleted = evalStatus?.completed_seeds ?? trainingStatus?.evaluation_completed_seeds ?? 0;
  const evalTotal =
    evalStatus?.total_seeds ??
    trainingStatus?.evaluation_total_seeds ??
    (evalStatus?.seeds?.length || 0);
  const evalCurrentSeed = evalStatus?.current_seed ?? trainingStatus?.evaluation_seed ?? null;
  const evalRecentResults = evalStatus?.recent_results || trainingStatus?.evaluation_recent_results || [];
  const evalAllSeeds =
    evalStatus?.seeds && evalStatus.seeds.length > 0
      ? evalStatus.seeds
      : Array.from(
          new Set([
            ...evalRecentResults.map((r) => r.seed),
            ...(evalCurrentSeed !== null ? [evalCurrentSeed] : []),
          ]),
        );
  const evalProgressPct =
    evalTotal > 0 ? Math.min(100, Math.round((evalCompleted / evalTotal) * 100)) : 0;

  // Training controls state
  const phase = trainingStatus?.phase ?? "idle";
  const busy =
    running ||
    clearing ||
    phase === "restoring" ||
    phase === "restore_failed" ||
    isStartingTraining;
  const batchCompleted =
    trainingStatus?.batch_completed_episodes ?? trainingStatus?.completed_episodes ?? 0;
  const batchTotal =
    trainingStatus?.batch_total_episodes ?? trainingStatus?.requested_episodes ?? episodes;
  const liveProgress = deriveLiveProgress(
    trainingStatus,
    batchCompleted,
    batchTotal,
    stageCheckpointsCount,
  );
  const hasSegmentProgress =
    (liveProgress.segStepsDone != null && liveProgress.segStepsTotal != null) ||
    liveProgress.segPct != null;
  const resumeAvailable = resumeAvailableProp ?? trainingStatus?.resume_available ?? false;
  const resumeRemainingEpisodes =
    resumeRemainingProp ?? trainingStatus?.resume_remaining_episodes ?? null;
  const continuousResume = trainingStatus?.resume_request?.continuous === true;
  const stageQueueResume = (trainingStatus?.resume_request?.stage_queue?.length ?? 0) > 0;
  const trimmedStageSpec = stageSpec.trim();
  const parsedStageSpec = useMemo(
    () => parseStageSpec(trimmedStageSpec, 1, Math.max(1, maxCurriculumStage)),
    [trimmedStageSpec, maxCurriculumStage],
  );
  const hasStageSpec = trimmedStageSpec.length > 0;
  const hasValidStageSpec =
    hasStageSpec && parsedStageSpec.error === null && parsedStageSpec.stages.length > 0;
  const stageSpecInvalid = hasStageSpec && parsedStageSpec.error !== null;
  const preparingStage =
    trainingStatus?.curriculum_stage != null && trainingStatus.curriculum_stage !== curriculumStage;
  const stageDesc = STAGE_DESCRIPTIONS[curriculumStage] ?? `Stage ${curriculumStage}`;
  const activeError = trainingError || trainingStatus?.error || null;

  const runBestLabel = runningCurrent
    ? "Running best model…"
    : bestCheckpointEpisode !== null
      ? `Run best model (ep ${bestCheckpointEpisode})`
      : "Run best model";

  return (
    <div className="ui-train-split">
      {/* LEFT COLUMN: Replay Toolbar + Field + Transport + Status & 10-Seed Strip */}
      <div className="ui-train-main">
        <section className="ui-toolbar" aria-label="Playback controls">
          <div className="ui-toolbar__field">
            <label htmlFor="train-checkpoint-select" className="ui-toolbar__label">
              Checkpoint
            </label>
            <select
              id="train-checkpoint-select"
              aria-label="Checkpoint"
              className="ui-toolbar__select"
              value={selectedCheckpointEpisode ?? ""}
              onChange={(e) => {
                if (e.target.value !== "") {
                  onSelectCheckpointEpisode(Number(e.target.value));
                }
              }}
              disabled={disabled || !hasCheckpoints}
            >
              {!hasCheckpoints ? (
                <option value="">No checkpoints exported</option>
              ) : null}
              {hasCheckpoints && selectedCheckpointEpisode == null ? (
                <option value="">Best model</option>
              ) : null}
              {checkpointSelectOptions}
            </select>
          </div>

          <div className="ui-toolbar__field">
            <label htmlFor="train-seed-select" className="ui-toolbar__label">
              Seed
            </label>
            <select
              id="train-seed-select"
              aria-label="Seed"
              className="ui-toolbar__select"
              value={selectedSeed ?? ""}
              onChange={(e) => {
                if (e.target.value !== "") {
                  onSelectSeed(Number(e.target.value));
                }
              }}
              disabled={disabled || seedOptions.length === 0}
            >
              {seedOptions.length === 0 ? <option value="">No seeds available</option> : null}
              {seedOptionItems.map((item) => (
                <option key={item.seed} value={item.seed}>
                  {item.label}
                </option>
              ))}
            </select>
          </div>

          <div className="ui-toolbar__actions">
            <button
              type="button"
              className="ui-btn--primary"
              onClick={onRunCurrentReplay}
              disabled={disabled || runningCurrent || isReadOnlyPreview}
            >
              {runBestLabel}
            </button>
            <button
              type="button"
              onClick={onReplaySelected}
              disabled={disabled || !selectedSeed || loadingSelectedReplay}
            >
              {loadingSelectedReplay ? "Loading…" : "Replay selected"}
            </button>
            <button
              type="button"
              onClick={onEndEpisode}
              disabled={disabled || !canEndEpisode}
            >
              End episode
            </button>
          </div>
        </section>

        {/* Herding Field + Direct Playback Transport */}
        <div className="ui-panel ui-train-field-panel">
          <FieldView snapshot={snapshot} />

          <div className="ui-transport-bar" aria-label="Playback transport">
            <div className="ui-transport-bar__group">
              <button
                type="button"
                className="ui-btn--sm ui-btn--primary"
                onClick={onTogglePlay}
                disabled={totalFrames === 0}
              >
                {isPlaying ? "Pause" : "Play"}
              </button>
              <button
                type="button"
                className="ui-btn--sm"
                onClick={() => onStepFrame(-1)}
                disabled={totalFrames === 0 || clampedFrameIndex <= 0}
                aria-label="Step back 1 frame"
              >
                -1
              </button>
              <button
                type="button"
                className="ui-btn--sm"
                onClick={() => onStepFrame(1)}
                disabled={totalFrames === 0 || clampedFrameIndex >= totalFrames - 1}
                aria-label="Step forward 1 frame"
              >
                +1
              </button>
            </div>

            <input
              type="range"
              className="ui-transport-bar__scrubber"
              aria-label="Episode frame scrubber"
              min={0}
              max={Math.max(0, totalFrames - 1)}
              value={clampedFrameIndex}
              onChange={(e) => onScrubFrame(Number(e.target.value))}
              disabled={totalFrames <= 1}
            />

            <span className="ui-transport-bar__time">
              Frame {totalFrames > 0 ? clampedFrameIndex + 1 : 0} / {totalFrames}
            </span>

            <div className="ui-transport-bar__group">
              <label className="ui-transport-bar__toggle">
                <input
                  type="checkbox"
                  checked={playbackFastMode}
                  onChange={(e) => onPlaybackFastModeChange(e.target.checked)}
                  disabled={disabled}
                />
                <span>Fast playback</span>
              </label>
              <button
                type="button"
                className="ui-btn--sm ui-btn--ghost"
                onClick={onOpenInspector}
              >
                Inspect Step-by-Step
              </button>
            </div>
          </div>
        </div>

        {/* Single Compact Run Status & 10-Seed Strip */}
        <section className="ui-meta-strip" aria-label="Run status">
          <div className="ui-meta-strip__row">
            <div className="ui-meta-strip__stats">
              <span className={`ui-badge ${statusToneClass}`}>{statusLabel}</span>
              <span className="ui-badge">{policyModeLabel}</span>
              <span className="ui-meta-strip__stat">
                Step <strong>{snapshot?.step ?? 0}</strong>
              </span>
              <span className="ui-meta-strip__stat">
                Seed <strong>{displaySeed ?? "—"}</strong>
              </span>
              <span className="ui-meta-strip__stat">
                Penned{" "}
                <strong>
                  {sheepPenned}/{totalSheepInReplay}
                </strong>
              </span>
              <span className="ui-meta-strip__stat">
                Roles <strong>{activeRoleSummary}</strong>
              </span>
            </div>
          </div>

          {running ? (
            <p className="ui-meta-strip__note">
              Field shows last saved checkpoint replay — background workers export a new replay at each checkpoint.
            </p>
          ) : null}

          {policyNote ? <p className="ui-meta-strip__note">{policyNote}</p> : null}

          {noProgressGuardActive ? (
            <p className="ui-meta-strip__note" role="status">
              No-progress guard is active. The episode should stop if progress stalls.
            </p>
          ) : null}

          {evalInProgress ? (
            <div key="eval-live" className="ui-meta-strip__eval">
              <div className="ui-meta-strip__eval-head">
                <span>
                  Evaluating [{evalMode}] · {evalCompleted}/{evalTotal} seeds
                  {evalCurrentSeed !== null ? ` · Current seed ${evalCurrentSeed}` : ""}
                </span>
                <span className="ui-mono">{evalProgressPct}%</span>
              </div>
              <div className="ui-progress" aria-label="Evaluation progress">
                <div
                  className="ui-progress__bar"
                  style={{ width: `${evalProgressPct}%` }}
                />
              </div>
              {evalAllSeeds.length > 0 ? (
                <div className="ui-seed-strip">
                  {evalAllSeeds.map((seed, idx) => {
                    const res = evalRecentResults.find((r) => r.seed === seed);
                    const isCurrent = seed === evalCurrentSeed;
                    const chipTone = res
                      ? res.success
                        ? "ui-seed-chip--pass"
                        : "ui-seed-chip--fail"
                      : isCurrent
                        ? "ui-seed-chip--active"
                        : "";
                    return (
                      <span
                        key={`eval-chip-${seed}-${idx}`}
                        className={`ui-seed-chip ${chipTone}`}
                      >
                        #{seed} {res ? (res.success ? "Pass" : "Fail") : isCurrent ? "Running" : "Queued"}
                      </span>
                    );
                  })}
                </div>
              ) : null}
            </div>
          ) : activeCheckpointEpisode !== null || stripSeedItems.length > 0 ? (
            <div key="eval-checkpoint" className="ui-meta-strip__eval">
              <div className="ui-meta-strip__eval-head">
                <span>
                  {activeCheckpointEpisode !== null
                    ? `Checkpoint Ep ${activeCheckpointEpisode} seeds`
                    : "Evaluation seeds"}
                </span>
                {checkpointRecordsCount > 0 ? (
                  <span className="ui-mono">
                    {checkpointPassCount}/{checkpointRecordsCount} passed
                    {checkpointAvgReward != null ? ` · Avg reward ${checkpointAvgReward}` : ""}
                  </span>
                ) : null}
              </div>
              <div className="ui-seed-strip">
                {stripSeedItems.map((item) => {
                  const isActive = selectedSeed === item.seed;
                  return (
                    <button
                      key={item.seed}
                      type="button"
                      className={`ui-seed-chip ${item.chipTone}${isActive ? " ui-seed-chip--active" : ""}`}
                      onClick={() => onSelectSeedAndReplay(item.seed)}
                    >
                      #{item.seed} {item.statusText}
                    </button>
                  );
                })}
              </div>
            </div>
          ) : null}
        </section>
      </div>

      {/* RIGHT RAIL: Decision Summary + Training Controls + Collapsible Advanced Options */}
      <aside className="ui-train-rail" aria-label="Training workspace rail">
        <section className="ui-section" aria-label="Decision summary">
          <div className="ui-section__header">
            <h2 className="ui-section__title">Decision Summary</h2>
            <span
              className={`ui-badge ${
                gate?.ready
                  ? "ui-badge--pass"
                  : latestFailCount > 0
                    ? "ui-badge--warn"
                    : ""
              }`}
            >
              Stage {curriculumStage}
            </span>
          </div>

          <div className="ui-decision">
            <h3 className="ui-decision__headline">{decisionSummary.headline}</h3>
            <p className="ui-decision__rationale">{decisionSummary.rationale}</p>

            {decisionSummary.action === "promote" ? (
              <div>
                <button type="button" onClick={onOpenProgress}>
                  View in Progress
                </button>
              </div>
            ) : null}

            {decisionSummary.action === "trouble" ? (
              <div>
                <button type="button" onClick={onOpenTroubleSpots}>
                  Open Trouble Spots
                </button>
              </div>
            ) : null}
          </div>

          <div className="ui-kpi-grid ui-kpi-grid--2">
            <div className="ui-kpi-card">
              <span className="ui-kpi-card__label">Eval Pass Rate</span>
              <span className="ui-kpi-card__value">
                {evalPassPct !== null ? `${evalPassPct}%` : "—"}
              </span>
              <span className="ui-kpi-card__sub">
                Peak {peakPassPct !== null ? `${peakPassPct}%` : "—"}
              </span>
            </div>

            <div className="ui-kpi-card">
              <span className="ui-kpi-card__label">Avg Steps</span>
              <span className="ui-kpi-card__value">{avgStepsDisplay}</span>
              <span className="ui-kpi-card__sub">Stage {curriculumStage} eval</span>
            </div>

            <div className="ui-kpi-card">
              <span className="ui-kpi-card__label">Avg Penned</span>
              <span className="ui-kpi-card__value">{avgPennedDisplay}</span>
              <span className="ui-kpi-card__sub">Per episode</span>
            </div>

            <div className="ui-kpi-card">
              <span className="ui-kpi-card__label">Promotion Gate</span>
              <span className="ui-kpi-card__value">
                {qualifiedCount}/{qualifiedRequired} qualified
              </span>
              <span className="ui-kpi-card__sub">target ≥{thresholdPct}%</span>
            </div>
          </div>

          {activeWarningsCount > 0 ? (
            <button
              type="button"
              className="ui-decision__alert-link"
              onClick={onOpenProgress}
            >
              {activeWarningsCount} advisory warning{activeWarningsCount === 1 ? "" : "s"} — view in Progress
            </button>
          ) : null}
        </section>

        <section className="ui-section" aria-label="Training controls">
          <div className="ui-section__header">
            <h2 className="ui-section__title">Training Controls</h2>
            <span className={`ui-badge ${running ? "ui-badge--pass" : ""}`}>
              {running ? "Running" : phase}
            </span>
          </div>

          <div className="ui-stage-header">
            <div className="ui-stage-header__top">
              <div className="ui-stage-callout" data-testid="train-stage-callout">
                <span className="stage-chip__label">Stage {curriculumStage}</span>
                {maxCurriculumStage > 0 ? (
                  <span className="ui-stage-callout__total">of {maxCurriculumStage}</span>
                ) : null}
              </div>
              <span className="ui-stage-header__desc">{stageDesc}</span>
            </div>
            <div className="ui-stage-header__controls">
              <select
                aria-label="Curriculum stage"
                value={curriculumStage}
                onChange={(e) => onCurriculumStageChange(Number(e.target.value))}
                disabled={busy}
              >
                {Array.from({ length: maxCurriculumStage + 1 }, (_, i) => i).map((s) => (
                  <option key={s} value={s}>
                    Stage {s}
                  </option>
                ))}
              </select>
              <button
                type="button"
                className="button-row__promote"
                onClick={onPromote}
                disabled={running || curriculumStage >= maxCurriculumStage || busy}
              >
                Promote → Stage {curriculumStage + 1}
              </button>
            </div>
          </div>

          <div className="ui-control-stack">
            {running ? (
              <>
                {trainingStatus?.stage_queue && trainingStatus.stage_queue.length > 0 ? (
                  <p className="ui-meta-strip__note" data-testid="stage-queue-progress">
                    {`Stage queue: ${(trainingStatus.stage_queue_index ?? 0) + 1} / ${trainingStatus.stage_queue.length} · Stage ${curriculumStage} (${trainingStatus.stage_spec ?? formatStageSpec(trainingStatus.stage_queue)})`}
                  </p>
                ) : continuous ? (
                  <p className="ui-meta-strip__note">Continuous training · Stage {curriculumStage}</p>
                ) : null}
                {hasSegmentProgress ? (
                  <div className="ui-progress-header">
                    <span className="ui-progress-header__label">Next checkpoint</span>
                    <span className="ui-mono">
                      {liveProgress.segStepsDone != null && liveProgress.segStepsTotal != null
                        ? `${liveProgress.segStepsDone.toLocaleString()} / ${liveProgress.segStepsTotal.toLocaleString()} timesteps (${liveProgress.displayPct}%)`
                        : `${liveProgress.displayPct}% of current segment`}
                    </span>
                  </div>
                ) : null}
                <div
                  className="ui-progress"
                  role="progressbar"
                  aria-label="Training batch progress"
                  aria-valuemin={0}
                  aria-valuemax={100}
                  aria-valuenow={liveProgress.displayPct}
                >
                  <div
                    className="ui-progress__bar"
                    style={{ width: `${liveProgress.displayPct}%` }}
                  />
                </div>
                <div className="ui-meta-strip__row">
                  <span className={hasSegmentProgress ? "ui-mono ui-progress-sub" : "ui-mono"}>
                    {hasSegmentProgress
                      ? `Batch checkpoints: ${liveProgress.completedSegments}${
                          liveProgress.totalSegments ? ` / ${liveProgress.totalSegments}` : ""
                        } saved · ${Math.floor(batchCompleted)} / ${batchTotal} episodes`
                      : `${Math.floor(batchCompleted)} / ${batchTotal} episodes (${liveProgress.batchPct}%)`}
                  </span>
                </div>
                {!hasSegmentProgress && liveProgress.timestepsSinceCheckpoint != null ? (
                  <p className="ui-meta-strip__note">
                    {liveProgress.timestepsSinceCheckpoint.toLocaleString()} timesteps since last checkpoint
                  </p>
                ) : null}
                {liveProgress.liveEnvEpisode != null ||
                (liveProgress.rolloutWindowCount ?? 0) > 0 ? (
                  <div className="ui-rollout-feed" data-testid="live-rollout-telemetry">
                    <div className="ui-rollout-feed__row">
                      <span className="ui-rollout-feed__label">Latest rollout:</span>
                      {liveProgress.liveEnvEpisode != null ? (
                        <strong className="ui-mono">
                          Ep #{liveProgress.liveEnvEpisode.toLocaleString()}
                        </strong>
                      ) : null}
                      {liveProgress.latestResult ? (
                        <span
                          className={`ui-badge ${rolloutResultBadgeTone(liveProgress.latestResult)}`}
                        >
                          {liveProgress.latestResult}
                        </span>
                      ) : null}
                      {liveProgress.latestReward != null &&
                      Number.isFinite(liveProgress.latestReward) ? (
                        <span className="ui-mono ui-rollout-feed__reward">
                          {liveProgress.latestReward > 0
                            ? `+${liveProgress.latestReward.toFixed(1)}`
                            : liveProgress.latestReward.toFixed(1)}
                        </span>
                      ) : null}
                    </div>
                    {liveProgress.rolloutWindowCount != null ? (
                      <div className="ui-rollout-feed__sub">
                        {liveProgress.rolloutWindowCount}{" "}
                        {liveProgress.rolloutWindowCount === 1 ? "rollout" : "rollouts"} since last checkpoint
                        {liveProgress.rolloutWinPct !== null
                          ? ` · ${liveProgress.rolloutWinPct}% rollout win rate`
                          : ""}
                      </div>
                    ) : null}
                  </div>
                ) : null}
                {trainingStatus?.message ? (
                  <p className="ui-meta-strip__note">{trainingStatus.message}</p>
                ) : null}
                <div className="ui-control-row">
                  <button
                    type="button"
                    onClick={onPauseTraining}
                    disabled={lifecycleAction !== null || clearing}
                  >
                    {lifecycleAction === "pause" ? "Pausing…" : "Pause after checkpoint"}
                  </button>
                  <button
                    type="button"
                    className="ui-btn--danger"
                    onClick={onStopTraining}
                    disabled={lifecycleAction !== null || clearing}
                  >
                    {lifecycleAction === "stop" ? "Stopping…" : "Stop after checkpoint"}
                  </button>
                </div>
              </>
            ) : (
              <>
                {onStageSpecChange ? (
                  <div className="ui-control-stack">
                    <div className="ui-field">
                      <label htmlFor="train-stage-spec-input" className="ui-field__label">
                        Stages to run (optional)
                      </label>
                      <div className="ui-control-row">
                        <input
                          id="train-stage-spec-input"
                          type="text"
                          value={stageSpec}
                          onChange={(e) => onStageSpecChange(e.target.value)}
                          placeholder="e.g. 1-38 or 10-12, 15-20, 25, 33-38"
                          disabled={busy}
                          aria-invalid={stageSpecInvalid || undefined}
                          style={{ flex: 1, minWidth: 0 }}
                        />
                        {hasStageSpec ? (
                          <button
                            type="button"
                            className="ui-btn--sm"
                            onClick={() => onStageSpecChange("")}
                            disabled={busy}
                          >
                            Clear
                          </button>
                        ) : null}
                      </div>
                    </div>
                    {hasValidStageSpec ? (
                      <p className="ui-meta-strip__note" data-testid="stage-spec-preview">
                        {`Queued ${parsedStageSpec.stages.length} stage${parsedStageSpec.stages.length === 1 ? "" : "s"} (${formatStageSpec(parsedStageSpec.stages)}) · ${startingModelSource === "fresh" ? "starts fresh model" : "uses current trained model"}`}
                      </p>
                    ) : null}
                    {stageSpecInvalid ? (
                      <div className="ui-inline-error" role="alert">
                        {parsedStageSpec.error}
                      </div>
                    ) : null}
                    {hasStageSpec && onPassLabelChange ? (
                      <div className="ui-field">
                        <label htmlFor="train-pass-label-input" className="ui-field__label">
                          Pass label
                        </label>
                        <input
                          id="train-pass-label-input"
                          type="text"
                          value={passLabel}
                          onChange={(e) => onPassLabelChange(e.target.value)}
                          placeholder="Optional pass name (e.g. 5 longest stages)"
                          disabled={busy}
                        />
                      </div>
                    ) : null}
                  </div>
                ) : null}
                <div className="ui-control-row">
                  <div className="ui-field ui-field--inline">
                    <label htmlFor="train-episodes-input" className="ui-field__label">
                      {continuous ? "Episode target per batch" : "Episodes"}
                    </label>
                    <input
                      id="train-episodes-input"
                      type="number"
                      min={1}
                      max={5000}
                      value={episodes}
                      onChange={(e) => onEpisodesChange(Math.max(1, Number(e.target.value) || 1))}
                      disabled={busy}
                      style={{ width: "86px" }}
                    />
                  </div>
                  <button
                    type="button"
                    className="ui-btn--primary"
                    onClick={onStartTraining}
                    disabled={isStartingTraining || clearing || isReadOnlyPreview || busy || stageSpecInvalid}
                  >
                    {isStartingTraining
                      ? "Starting…"
                      : hasValidStageSpec
                        ? parsedStageSpec.stages.length === 1
                          ? `Run Stage ${parsedStageSpec.stages[0]}`
                          : `Run ${parsedStageSpec.stages.length} stages`
                        : continuous
                          ? preparingStage ? `Prepare Stage ${curriculumStage}` : "Start continuous training"
                          : `Train ${episodes} more`}
                  </button>
                  {resumeAvailable && (continuousResume || (resumeRemainingEpisodes ?? 0) > 0 || stageQueueResume) ? (
                    <button
                      type="button"
                      className="button-row__promote"
                      onClick={onResumeTraining}
                      disabled={isStartingTraining || clearing || isReadOnlyPreview || busy}
                    >
                      {stageQueueResume
                        ? `Resume stage queue (${trainingStatus?.resume_request?.stage_spec ?? formatStageSpec(trainingStatus!.resume_request!.stage_queue!)})`
                        : continuousResume
                          ? "Resume continuous training"
                          : `Resume ${resumeRemainingEpisodes} remaining`}
                    </button>
                  ) : null}
                </div>
              </>
            )}

            <div className="ui-toggles-row">
              {onContinuousChange ? <label className="ui-toggle">
                <input
                  type="checkbox"
                  checked={continuous}
                  onChange={(e) => onContinuousChange(e.target.checked)}
                  disabled={busy || isReadOnlyPreview}
                />
                <span>Continuous training</span>
              </label> : null}
              <label className="ui-toggle">
                <input
                  type="checkbox"
                  checked={enableInstincts}
                  onChange={(e) => onEnableInstinctsChange(e.target.checked)}
                  disabled={busy}
                />
                <span>Enable instincts</span>
              </label>
              <label className="ui-toggle">
                <input
                  type="checkbox"
                  checked={continuous || autoPromote}
                  onChange={(e) => onAutoPromoteChange(e.target.checked)}
                  disabled={busy || continuous}
                />
                <span>Auto-promote stages</span>
              </label>
            </div>
            {continuous && !running ? (
              <p className="ui-meta-strip__note">
                Repeat batches until Stage {maxCurriculumStage} meets its mastery criteria.
                PPO uses the episode target to estimate a training budget.
              </p>
            ) : null}

            {!running &&
            trainingStatus?.message &&
            trainingStatus.message !== "Idle" &&
            !activeError ? (
              <p className="ui-meta-strip__note" role="status">
                {trainingStatus.message}
              </p>
            ) : null}

            {activeError ? (
              <div className="ui-inline-error" role="alert">
                {activeError}
              </div>
            ) : null}
          </div>
        </section>

        <section className="ui-section">
          <details className="ui-disclosure">
            <summary>Advanced options</summary>
            <div className="ui-disclosure__body ui-control-stack">
              <div className="ui-field">
                <label htmlFor="train-starting-model" className="ui-field__label">
                  Starting model
                </label>
                <select
                  id="train-starting-model"
                  aria-label="Starting model"
                  value={startingModelSource}
                  onChange={(e) => onStartingModelSourceChange(e.target.value)}
                  disabled={busy}
                >
                  <option value="latest">Latest checkpoint</option>
                  <option value="fresh">Fresh model (from scratch)</option>
                </select>
              </div>

              <div className="ui-toggles-row">
                <label className="ui-toggle">
                  <input
                    type="checkbox"
                    checked={fastMode}
                    onChange={(e) => onFastModeChange(e.target.checked)}
                    disabled={busy}
                  />
                  <span>Fast mode</span>
                </label>
                <label className="ui-toggle">
                  <input
                    type="checkbox"
                    checked={debugRewardBreakdown}
                    onChange={(e) => onDebugRewardBreakdownChange(e.target.checked)}
                    disabled={busy}
                  />
                  <span>Debug reward breakdown</span>
                </label>
              </div>

              <div className="ui-control-row">
                <button
                  type="button"
                  onClick={onStartRemediation}
                  disabled={busy || isReadOnlyPreview}
                >
                  Stage 9 remediation
                </button>
                <button
                  type="button"
                  className="ui-btn--danger"
                  onClick={onClearTraining}
                  disabled={busy || isReadOnlyPreview}
                >
                  {clearing ? "Clearing…" : "Clear"}
                </button>
              </div>
            </div>
          </details>
        </section>
      </aside>
    </div>
  );
}

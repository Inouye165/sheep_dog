import type { CheckpointEntry, TrainingEpisode } from "../state/types";

export type ViewWindow = "all" | 25 | 50 | 100;
export type SmoothingWindow = 25 | 50 | 100;
export type StageScope = "all" | "current" | "current-journey" | number;
export type XAxisMode = "timesteps" | "episode" | "runtime" | "calendar";

export interface CanonicalEpisodeRecord {
  id: number;
  run_id?: string;
  session_id?: string;
  curriculum_stage: number;
  global_environment_episode: number;
  episode_in_stage: number;
  global_timestep: number;
  completed_at: string;
  timestamp_ms: number;
  steps: number;
  success: boolean;
  result: string;
  reward: number;
  sheep_penned: number;
  total_sheep: number;
}

export interface RollingTrainingPoint extends CanonicalEpisodeRecord {
  rollingSuccessRate: number; // 0 to 100
  rollingSuccessfulSteps: number | null; // average steps among successes in window
  rollingAllSteps: number;
  rollingReward: number;
  rollingSheepPenned: number;
  windowCount: number;
}

export interface CheckpointEfficiencyMetrics {
  successRatePct: number;
  successCount: number;
  totalSeeds: number;
  medianSuccessfulSteps: number | null;
  meanSuccessfulSteps: number | null;
  worstSuccessfulSteps: number | null;
  failedSeeds: number[];
}

export interface FormalEvalMarker {
  checkpoint: CheckpointEntry;
  xVal: number;
  successRatePct: number;
  successCount: number;
  totalSeeds: number;
  medianSuccessfulSteps: number | null;
  meanSuccessfulSteps: number | null;
  worstSuccessfulSteps: number | null;
  failedSeeds: number[];
  avgReward: number | null;
  avgSheepPenned: number | null;
  stage: number;
  isBest: boolean;
  label: string;
}

export type EfficiencyTrendStatus = "improving" | "stable" | "regressing" | "collecting_evidence";

export interface EfficiencyTrendAnalysis {
  status: EfficiencyTrendStatus;
  statusLabel: string;
  recentMedian: number | null;
  priorMedian: number | null;
  percentageImprovement: number | null; // positive = faster/improved (fewer steps)
  evaluationsCount: number;
  evaluationsRequired: number;
  summaryText: string;
}

export type SeedReliabilityStatus = "reliable" | "normal_variance" | "blind_spot" | "inefficient";

export interface SeedReliabilitySummary {
  seed: number;
  totalTrials: number;
  successCount: number;
  failureCount: number;
  recentSuccessRate: number; // 0 to 100
  consecutiveFailures: number;
  typicalSuccessfulSteps: number | null;
  worstSuccessfulSteps: number | null;
  status: SeedReliabilityStatus;
  statusText: string;
  isBlindSpot: boolean;
  isInefficient: boolean;
}

export interface StagePerSeedAnalysis {
  seeds: SeedReliabilitySummary[];
  overallMedianSuccessfulSteps: number | null;
  blindSpotCount: number;
  inefficientCount: number;
  blindSpotSeeds: number[];
  inefficientSeeds: number[];
}

export interface EpisodeBucket {
  firstEpisode: number;
  lastEpisode: number;
  firstTimestep?: number;
  startTimestep?: number;
  endTimestep: number;
  startTimestampMs: number;
  endTimestampMs: number;
  startTimestamp?: string;
  endTimestamp?: string;
  episodeCount: number;
  successCount: number;
  failureCount: number;
  successRate: number; // 0-100
  avgSuccessfulSteps: number | null;
  avgAllSteps?: number;
  avgAllReward: number;
  avgSheepPenned: number;
  episodes: CanonicalEpisodeRecord[];
}

export function buildEpisodeBuckets(
  episodes: CanonicalEpisodeRecord[],
  bucketSize: number = 25
): EpisodeBucket[] {
  if (!episodes || episodes.length === 0) return [];
  const buckets: EpisodeBucket[] = [];
  const safeBucketSize = Math.max(1, bucketSize);

  for (let i = 0; i < episodes.length; i += safeBucketSize) {
    const chunk = episodes.slice(i, i + safeBucketSize);
    if (chunk.length === 0) continue;

    const first = chunk[0];
    const last = chunk[chunk.length - 1];

    const successes = chunk.filter((e) => e.success);
    const failures = chunk.filter((e) => !e.success);

    const successCount = successes.length;
    const failureCount = failures.length;
    const successRate = (successCount / chunk.length) * 100;

    const avgSuccessfulSteps =
      successCount > 0
        ? successes.reduce((sum, e) => sum + e.steps, 0) / successCount
        : null;

    const avgAllSteps = chunk.reduce((sum, e) => sum + e.steps, 0) / chunk.length;
    const avgAllReward = chunk.reduce((sum, e) => sum + e.reward, 0) / chunk.length;
    const avgSheepPenned = chunk.reduce((sum, e) => sum + e.sheep_penned, 0) / chunk.length;

    buckets.push({
      firstEpisode: first.episode_in_stage ?? first.global_environment_episode,
      lastEpisode: last.episode_in_stage ?? last.global_environment_episode,
      firstTimestep: first.global_timestep,
      startTimestep: first.global_timestep,
      endTimestep: last.global_timestep,
      startTimestampMs: first.timestamp_ms,
      endTimestampMs: last.timestamp_ms,
      startTimestamp: first.completed_at,
      endTimestamp: last.completed_at,
      episodeCount: chunk.length,
      successCount,
      failureCount,
      successRate,
      avgSuccessfulSteps,
      avgAllSteps,
      avgAllReward,
      avgSheepPenned,
      episodes: chunk,
    });
  }

  return buckets;
}

/**
 * Safely normalizes raw API/SQLite training episode records into a canonical structure.
 */
export function normalizeEpisodeRecord(raw: Partial<TrainingEpisode>, indexFallback: number = 0): CanonicalEpisodeRecord {
  const id = typeof raw.id === "number" && !isNaN(raw.id) ? raw.id : indexFallback;
  const stage = typeof raw.curriculum_stage === "number" && !isNaN(raw.curriculum_stage) ? raw.curriculum_stage : 1;
  const globalEp = typeof raw.global_environment_episode === "number" && !isNaN(raw.global_environment_episode)
    ? raw.global_environment_episode
    : typeof raw.episode_in_stage === "number" && !isNaN(raw.episode_in_stage)
    ? raw.episode_in_stage
    : id;
  const stageEp = typeof raw.episode_in_stage === "number" && !isNaN(raw.episode_in_stage)
    ? raw.episode_in_stage
    : globalEp;
  const globalTs = typeof raw.global_timestep === "number" && !isNaN(raw.global_timestep)
    ? raw.global_timestep
    : globalEp * 500;

  let success = false;
  if (raw.success === true || (raw.success as unknown) === 1) {
    success = true;
  } else if (typeof raw.result === "string" && raw.result.trim().toUpperCase() === "SUCCESS") {
    success = true;
  }

  const result = typeof raw.result === "string" && raw.result.trim().length > 0
    ? raw.result.trim().toUpperCase()
    : success
    ? "SUCCESS"
    : "TIMEOUT";

  const completedAt = raw.completed_at || (raw as any).created_at || new Date(1700000000000 + id * 1000).toISOString();
  let tsMs = new Date(completedAt).getTime();
  if (isNaN(tsMs)) {
    tsMs = 1700000000000 + id * 1000;
  }

  return {
    id,
    run_id: raw.run_id ?? undefined,
    session_id: raw.session_id ?? undefined,
    curriculum_stage: stage,
    global_environment_episode: globalEp,
    episode_in_stage: stageEp,
    global_timestep: globalTs,
    completed_at: completedAt,
    timestamp_ms: tsMs,
    steps: typeof raw.steps === "number" && !isNaN(raw.steps) ? raw.steps : 0,
    success,
    result,
    reward: typeof raw.reward === "number" && !isNaN(raw.reward) ? raw.reward : 0,
    sheep_penned: typeof raw.sheep_penned === "number" && !isNaN(raw.sheep_penned) ? raw.sheep_penned : 0,
    total_sheep: typeof raw.total_sheep === "number" && !isNaN(raw.total_sheep) ? raw.total_sheep : 3,
  };
}

export function getCheckpointStage(c: Partial<CheckpointEntry>): number {
  if ((c as any).stage != null && typeof (c as any).stage === "number" && !isNaN((c as any).stage)) return (c as any).stage;
  if (c.curriculum_stage != null && typeof c.curriculum_stage === "number" && !isNaN(c.curriculum_stage)) return c.curriculum_stage;
  if (c.reward_config?.instincts?.curriculum_stage != null && !isNaN(c.reward_config.instincts.curriculum_stage)) {
    return c.reward_config.instincts.curriculum_stage;
  }
  if (c.environment_config?.curriculum_stage != null && !isNaN(c.environment_config.curriculum_stage)) {
    return c.environment_config.curriculum_stage;
  }
  return 1;
}

/**
 * Strict, immutable canonical ordering comparator.
 * Primary key: global_timestep ASC
 * Secondary key: global_environment_episode ASC
 * Tertiary key: timestamp_ms ASC
 * Quaternary key: id ASC
 */
export function compareCanonicalEpisodes(a: CanonicalEpisodeRecord, b: CanonicalEpisodeRecord): number {
  if (a.global_timestep !== b.global_timestep) {
    return a.global_timestep - b.global_timestep;
  }
  if (a.global_environment_episode !== b.global_environment_episode) {
    return a.global_environment_episode - b.global_environment_episode;
  }
  if (a.timestamp_ms !== b.timestamp_ms) {
    return a.timestamp_ms - b.timestamp_ms;
  }
  return a.id - b.id;
}

/**
 * Single source of truth pipeline step 1-4:
 * 1. Normalize actual raw rollout records
 * 2. Deduplicate by unique id
 * 3. Filter by stage and run_id
 * 4. Sort strictly chronologically via compareCanonicalEpisodes
 * 
 * IMPORTANT INVARIANT: Zero raw rollout episodes returns empty array ([]).
 * Checkpoints are NEVER converted into fake pseudo training episodes.
 */
export function processCanonicalHistory(
  rawEpisodes: Partial<TrainingEpisode>[],
  stageScope: StageScope = "current-journey",
  activeRunId?: string,
  effectiveStage: number = 1
): CanonicalEpisodeRecord[] {
  if (!rawEpisodes || rawEpisodes.length === 0) {
    return [];
  }

  const normalized = rawEpisodes.map((raw, idx) => normalizeEpisodeRecord(raw, idx + 1));
  const map = new Map<number, CanonicalEpisodeRecord>();
  for (const ep of normalized) {
    map.set(ep.id, ep);
  }
  const allDeduped = Array.from(map.values());

  const applyFilters = (records: CanonicalEpisodeRecord[]): CanonicalEpisodeRecord[] => {
    if (stageScope === "current-journey") {
      let filtered = records.filter((e) => !e.run_id || e.run_id === "current" || !e.run_id.startsWith("journey-"));
      if (activeRunId) {
        filtered = filtered.filter((e) => e.run_id === activeRunId || e.run_id === "current" || !e.run_id?.startsWith("journey-"));
      }
      return filtered;
    }
    if (stageScope === "all") {
      return records;
    }
    if (stageScope === "current") {
      let filtered = records.filter((e) => e.curriculum_stage === effectiveStage);
      filtered = filtered.filter((e) => !e.run_id || e.run_id === "current" || !e.run_id.startsWith("journey-") || e.run_id === activeRunId);
      if (activeRunId && filtered.some((e) => e.run_id === activeRunId)) {
        filtered = filtered.filter((e) => e.run_id === activeRunId);
      }
      return filtered;
    }
    const targetStage = Number(stageScope);
    if (!isNaN(targetStage)) {
      let filtered = records.filter((e) => e.curriculum_stage === targetStage);
      const currentRunRecords = filtered.filter((e) => !e.run_id || e.run_id === "current" || !e.run_id.startsWith("journey-") || e.run_id === activeRunId);
      if (currentRunRecords.length > 0) {
        filtered = currentRunRecords;
      }
      if (activeRunId && filtered.some((e) => e.run_id === activeRunId)) {
        filtered = filtered.filter((e) => e.run_id === activeRunId);
      }
      return filtered;
    }
    return records;
  };

  const filtered = applyFilters(allDeduped);
  if (filtered.length > 0) {
    return filtered.sort(compareCanonicalEpisodes);
  }

  // If filtered by run_id resulted in 0, fallback to stage-only filter so recent episodes are visible
  const stageOnlyFiltered = ((): CanonicalEpisodeRecord[] => {
    if (stageScope === "all" || stageScope === "current-journey") return allDeduped;
    const targetStage = stageScope === "current" ? effectiveStage : Number(stageScope);
    if (!isNaN(targetStage)) return allDeduped.filter((e) => e.curriculum_stage === targetStage);
    return allDeduped;
  })();

  return stageOnlyFiltered.sort(compareCanonicalEpisodes);
}

/**
 * Computes rolling training statistics across the FULL canonical stage history FIRST.
 * Slicing for display window must only occur AFTER this computation.
 */
export function computeRollingTrainingSeries(
  fullCanonicalHistory: CanonicalEpisodeRecord[],
  windowSize: number = 50
): RollingTrainingPoint[] {
  if (fullCanonicalHistory.length === 0) return [];

  const w = Math.max(1, windowSize);
  return fullCanonicalHistory.map((ep, idx) => {
    const start = Math.max(0, idx - w + 1);
    const slice = fullCanonicalHistory.slice(start, idx + 1);
    const sliceLen = slice.length;

    const successes = slice.filter((e) => e.success);
    const rollingSuccessRate = (successes.length / sliceLen) * 100;

    let rollingSuccessfulSteps: number | null = null;
    if (successes.length > 0) {
      const sumSteps = successes.reduce((acc, e) => acc + e.steps, 0);
      rollingSuccessfulSteps = sumSteps / successes.length;
    }

    const rollingAllSteps = slice.reduce((acc, e) => acc + e.steps, 0) / sliceLen;
    const rollingReward = slice.reduce((acc, e) => acc + e.reward, 0) / sliceLen;
    const rollingSheepPenned = slice.reduce((acc, e) => acc + e.sheep_penned, 0) / sliceLen;

    return {
      id: ep.id,
      run_id: ep.run_id,
      session_id: ep.session_id,
      global_timestep: ep.global_timestep,
      global_environment_episode: ep.global_environment_episode,
      episode_in_stage: ep.episode_in_stage,
      completed_at: ep.completed_at,
      timestamp_ms: ep.timestamp_ms,
      curriculum_stage: ep.curriculum_stage,
      success: ep.success,
      result: ep.result,
      steps: ep.steps,
      reward: ep.reward,
      sheep_penned: ep.sheep_penned,
      total_sheep: ep.total_sheep,
      rollingSuccessRate,
      rollingSuccessfulSteps,
      rollingAllSteps,
      rollingReward,
      rollingSheepPenned,
      windowCount: sliceLen,
    };
  });
}

/**
 * Single source of truth pipeline: Select requested view window slice.
 * GUARANTEES tail-subset invariant: Last 25 === All.slice(-25), Last 50 === All.slice(-50), etc.
 */
export function selectWindowSlice<T>(
  canonicalHistory: T[],
  window: ViewWindow
): T[] {
  if (window === "all") return canonicalHistory;
  return canonicalHistory.slice(-window);
}

/**
 * Computes exact formal evaluation metrics from CheckpointEntry per-seed records.
 * Failed seeds/timeouts are strictly excluded from successful completion step statistics.
 */
export function computeCheckpointEfficiency(c: CheckpointEntry): CheckpointEfficiencyMetrics {
  const records = c.records ?? [];
  const rawRate = c.success_rate;
  let successRatePct = 0;
  if (rawRate != null && typeof rawRate === "number" && !isNaN(rawRate)) {
    successRatePct = rawRate <= 1.0 ? rawRate * 100 : rawRate;
  }

  if (records.length > 0) {
    const succ = records.filter((r) => r.success);
    const fail = records.filter((r) => !r.success);
    const totalSeeds = records.length;
    const successCount = succ.length;
    const rateFromRecords = (successCount / totalSeeds) * 100;
    const finalRate = !isNaN(rateFromRecords) ? rateFromRecords : successRatePct;

    if (succ.length > 0) {
      const sortedSteps = succ
        .map((r) => r.steps)
        .filter((s) => typeof s === "number" && !isNaN(s))
        .sort((a, b) => a - b);
      let medianSteps: number | null = null;
      if (sortedSteps.length > 0) {
        const mid = Math.floor(sortedSteps.length / 2);
        medianSteps = sortedSteps.length % 2 !== 0
          ? sortedSteps[mid]
          : (sortedSteps[mid - 1] + sortedSteps[mid]) / 2;
      }
      const meanSteps = sortedSteps.length > 0 ? sortedSteps.reduce((a, b) => a + b, 0) / sortedSteps.length : null;
      const worstSteps = sortedSteps.length > 0 ? sortedSteps[sortedSteps.length - 1] : null;
      return {
        successRatePct: finalRate,
        successCount,
        totalSeeds,
        medianSuccessfulSteps: medianSteps != null ? Math.round(medianSteps * 10) / 10 : null,
        meanSuccessfulSteps: meanSteps != null ? Math.round(meanSteps * 10) / 10 : null,
        worstSuccessfulSteps: worstSteps,
        failedSeeds: fail.map((r) => r.seed),
      };
    }

    return {
      successRatePct: finalRate,
      successCount: 0,
      totalSeeds,
      medianSuccessfulSteps: null,
      meanSuccessfulSteps: null,
      worstSuccessfulSteps: null,
      failedSeeds: fail.map((r) => r.seed),
    };
  }

  // Fallback for checkpoints without per-seed records (e.g. pruned older checkpoints)
  const totalSeeds = c.evaluation_seed_count ?? (c.evaluation_seeds ? c.evaluation_seeds.length : 10);
  const successCount = Math.round((successRatePct / 100) * totalSeeds);
  const avgSteps = c.average_completion_steps ?? null;

  return {
    successRatePct,
    successCount,
    totalSeeds,
    medianSuccessfulSteps: avgSteps,
    meanSuccessfulSteps: avgSteps,
    worstSuccessfulSteps: null,
    failedSeeds: [],
  };
}

/**
 * Builds landmark formal evaluation markers from checkpoint index.
 * Positioned strictly at their exact X coordinate (global_timestep, episode, or timestamp).
 */
export function buildFormalEvalMarkers(
  checkpoints: CheckpointEntry[],
  stageScope: StageScope,
  xAxisMode: XAxisMode,
  bestCheckpointEpisode?: number | null,
  effectiveStage: number = 1
): FormalEvalMarker[] {
  if (!checkpoints || checkpoints.length === 0) return [];

  const targetStage = stageScope === "current" || stageScope === "current-journey"
    ? effectiveStage
    : typeof stageScope === "number"
    ? stageScope
    : null;

  let filtered = checkpoints;
  if (targetStage !== null && !isNaN(targetStage)) {
    filtered = checkpoints.filter((c) => getCheckpointStage(c) === targetStage);
  }

  const markers: FormalEvalMarker[] = [];
  for (const c of filtered) {
    let xVal: number | null = null;
    if (xAxisMode === "timesteps") {
      xVal = c.global_timesteps ?? c.global_timestep ?? (c.checkpoint_episode != null ? c.checkpoint_episode * 500 : null);
    } else if (xAxisMode === "episode") {
      xVal = c.total_training_episodes ?? c.cumulative_environment_episodes ?? c.checkpoint_episode ?? null;
    } else if (xAxisMode === "runtime") {
      xVal = c.active_runtime_seconds_total ?? c.checkpoint_episode ?? null;
    } else if (xAxisMode === "calendar") {
      const ts = c.recorded_at ?? c.evaluation_timestamp ?? c.created_timestamp;
      xVal = ts ? new Date(ts).getTime() : c.checkpoint_episode ?? null;
    }

    if (xVal == null || isNaN(xVal)) continue;

    const eff = computeCheckpointEfficiency(c);

    markers.push({
      checkpoint: c,
      xVal,
      successRatePct: eff.successRatePct,
      successCount: eff.successCount,
      totalSeeds: eff.totalSeeds,
      medianSuccessfulSteps: eff.medianSuccessfulSteps,
      meanSuccessfulSteps: eff.meanSuccessfulSteps,
      worstSuccessfulSteps: eff.worstSuccessfulSteps,
      failedSeeds: eff.failedSeeds,
      avgReward: c.average_reward ?? null,
      avgSheepPenned: c.average_sheep_penned ?? null,
      stage: getCheckpointStage(c),
      isBest: c.checkpoint_episode === bestCheckpointEpisode,
      label: `Formal ${eff.totalSeeds}-Seed Evaluation: ${Math.round(eff.successRatePct)}% (${eff.successCount}/${eff.totalSeeds} seeds at ep ${c.checkpoint_episode})`,
    });
  }

  return markers.sort((a, b) => a.xVal - b.xVal);
}

/**
 * Computes an observational efficiency trend comparing recent vs prior formal evaluation medians.
 * Lower steps = positive improvement percentage.
 */
export function calculateEfficiencyTrend(
  checkpoints: CheckpointEntry[],
  targetStage?: number
): EfficiencyTrendAnalysis {
  const filtered = targetStage != null
    ? checkpoints.filter((c) => getCheckpointStage(c) === targetStage)
    : checkpoints;

  const validEvals: Array<{ checkpoint_episode: number; medianSteps: number }> = [];
  for (const c of filtered) {
    const eff = computeCheckpointEfficiency(c);
    if (eff.medianSuccessfulSteps != null && eff.medianSuccessfulSteps > 0 && eff.successCount > 0) {
      validEvals.push({
        checkpoint_episode: c.checkpoint_episode,
        medianSteps: eff.medianSuccessfulSteps,
      });
    }
  }

  const count = validEvals.length;
  if (count < 2) {
    return {
      status: "collecting_evidence",
      statusLabel: "COLLECTING EVIDENCE",
      recentMedian: count === 1 ? validEvals[0].medianSteps : null,
      priorMedian: null,
      percentageImprovement: null,
      evaluationsCount: count,
      evaluationsRequired: 6,
      summaryText: `Collecting evidence (${count}/6 formal evaluations with successes).`,
    };
  }

  const medianOf = (vals: number[]): number => {
    const sorted = [...vals].sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);
    return sorted.length % 2 !== 0 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
  };

  let recentMedian: number;
  let priorMedian: number;

  if (count >= 6) {
    const recent3 = validEvals.slice(-3).map((e) => e.medianSteps);
    const prior3 = validEvals.slice(-6, -3).map((e) => e.medianSteps);
    recentMedian = medianOf(recent3);
    priorMedian = medianOf(prior3);
  } else {
    const half = Math.floor(count / 2);
    const priorSlice = validEvals.slice(0, half).map((e) => e.medianSteps);
    const recentSlice = validEvals.slice(half).map((e) => e.medianSteps);
    recentMedian = medianOf(recentSlice);
    priorMedian = medianOf(priorSlice);
  }

  const pctImprovement = priorMedian > 0
    ? ((priorMedian - recentMedian) / priorMedian) * 100
    : 0;

  let status: EfficiencyTrendStatus;
  let statusLabel: string;
  if (pctImprovement >= 4.0) {
    status = "improving";
    statusLabel = `IMPROVING (+${pctImprovement.toFixed(1)}% faster)`;
  } else if (pctImprovement <= -4.0) {
    status = "regressing";
    statusLabel = `REGRESSING (${pctImprovement.toFixed(1)}% slower)`;
  } else {
    status = "stable";
    statusLabel = `STABLE (${pctImprovement >= 0 ? "+" : ""}${pctImprovement.toFixed(1)}%)`;
  }

  const roundedRecent = Math.round(recentMedian);
  const roundedPrior = Math.round(priorMedian);

  const summaryText = count >= 6
    ? `Recent 3 evals median (${roundedRecent} steps) vs prior 3 evals median (${roundedPrior} steps): ${statusLabel}`
    : `Recent evals median (${roundedRecent} steps) vs earlier evals (${roundedPrior} steps): ${statusLabel} (${count}/6 evals)`;

  return {
    status,
    statusLabel,
    recentMedian: Math.round(recentMedian * 10) / 10,
    priorMedian: Math.round(priorMedian * 10) / 10,
    percentageImprovement: Math.round(pctImprovement * 10) / 10,
    evaluationsCount: count,
    evaluationsRequired: 6,
    summaryText,
  };
}

/**
 * Aggregates deterministic benchmark seeds across recent formal evaluations.
 * Detects persistent blind spots (repeated failures) and inefficient scenarios (slow but successful).
 */
export function analyzePerSeedReliability(
  checkpoints: CheckpointEntry[],
  targetStage?: number,
  recentEvaluationsWindow: number = 10
): StagePerSeedAnalysis {
  const filtered = targetStage != null
    ? checkpoints.filter((c) => getCheckpointStage(c) === targetStage)
    : checkpoints;

  const recentWithRecords = filtered.filter((c) => c.records && c.records.length > 0).slice(-recentEvaluationsWindow);

  if (recentWithRecords.length === 0) {
    return {
      seeds: [],
      overallMedianSuccessfulSteps: null,
      blindSpotCount: 0,
      inefficientCount: 0,
      blindSpotSeeds: [],
      inefficientSeeds: [],
    };
  }

  const allSuccessfulSteps: number[] = [];
  const seedTrialMap = new Map<number, Array<{ success: boolean; steps: number; stop_reason?: string }>>();

  for (const c of recentWithRecords) {
    for (const record of c.records!) {
      if (!seedTrialMap.has(record.seed)) {
        seedTrialMap.set(record.seed, []);
      }
      seedTrialMap.get(record.seed)!.push({
        success: record.success,
        steps: record.steps,
        stop_reason: record.stop_reason,
      });
      if (record.success && record.steps > 0) {
        allSuccessfulSteps.push(record.steps);
      }
    }
  }

  let overallMedian: number | null = null;
  if (allSuccessfulSteps.length > 0) {
    allSuccessfulSteps.sort((a, b) => a - b);
    const mid = Math.floor(allSuccessfulSteps.length / 2);
    overallMedian = allSuccessfulSteps.length % 2 !== 0
      ? allSuccessfulSteps[mid]
      : (allSuccessfulSteps[mid - 1] + allSuccessfulSteps[mid]) / 2;
  }

  const summaries: SeedReliabilitySummary[] = [];

  for (const [seed, trials] of seedTrialMap.entries()) {
    const totalTrials = trials.length;
    const successes = trials.filter((t) => t.success);
    const successCount = successes.length;
    const failureCount = totalTrials - successCount;
    const recentSuccessRate = (successCount / totalTrials) * 100;

    let consecutiveFailures = 0;
    for (let i = trials.length - 1; i >= 0; i--) {
      if (!trials[i].success) {
        consecutiveFailures++;
      } else {
        break;
      }
    }

    const succSteps = successes.map((t) => t.steps).sort((a, b) => a - b);
    let typicalSteps: number | null = null;
    let worstSteps: number | null = null;
    if (succSteps.length > 0) {
      const mid = Math.floor(succSteps.length / 2);
      typicalSteps = succSteps.length % 2 !== 0
        ? succSteps[mid]
        : (succSteps[mid - 1] + succSteps[mid]) / 2;
      worstSteps = succSteps[succSteps.length - 1];
    }

    const isBlindSpot = consecutiveFailures >= 2 || (totalTrials >= 3 && recentSuccessRate <= 40);
    const isInefficient = !isBlindSpot &&
      successCount >= 2 &&
      overallMedian != null &&
      typicalSteps != null &&
      typicalSteps >= overallMedian * 1.4 &&
      (typicalSteps - overallMedian) >= 30;

    let status: SeedReliabilityStatus = "reliable";
    let statusText = "Reliable";
    if (isBlindSpot) {
      status = "blind_spot";
      statusText = consecutiveFailures >= 2
        ? `Blind Spot (${consecutiveFailures} consecutive fails)`
        : `Blind Spot (${Math.round(recentSuccessRate)}% success)`;
    } else if (isInefficient) {
      status = "inefficient";
      statusText = `Inefficient (~${Math.round(typicalSteps!)} steps vs ~${Math.round(overallMedian!)} median)`;
    } else if (failureCount > 0) {
      status = "normal_variance";
      statusText = `Occasional Failure (${successCount}/${totalTrials})`;
    }

    summaries.push({
      seed,
      totalTrials,
      successCount,
      failureCount,
      recentSuccessRate: Math.round(recentSuccessRate),
      consecutiveFailures,
      typicalSuccessfulSteps: typicalSteps != null ? Math.round(typicalSteps) : null,
      worstSuccessfulSteps: worstSteps != null ? Math.round(worstSteps) : null,
      status,
      statusText,
      isBlindSpot,
      isInefficient,
    });
  }

  summaries.sort((a, b) => a.seed - b.seed);

  const blindSpotSeeds = summaries.filter((s) => s.isBlindSpot).map((s) => s.seed);
  const inefficientSeeds = summaries.filter((s) => s.isInefficient).map((s) => s.seed);

  return {
    seeds: summaries,
    overallMedianSuccessfulSteps: overallMedian != null ? Math.round(overallMedian) : null,
    blindSpotCount: blindSpotSeeds.length,
    inefficientCount: inefficientSeeds.length,
    blindSpotSeeds,
    inefficientSeeds,
  };
}

/**
 * Asserts programmatically that X coordinates in a rendered series are strictly non-decreasing.
 */
export function assertMonotonicX(points: Array<{ x: number }>, seriesLabel: string = "Series"): void {
  for (let i = 0; i < points.length - 1; i++) {
    if (points[i].x > points[i + 1].x) {
      const msg = `[Chart Chronology Violation] ${seriesLabel} point index ${i} (x=${points[i].x}) > index ${i + 1} (x=${points[i + 1].x})`;
      console.error(msg);
      throw new Error(msg);
    }
  }
}

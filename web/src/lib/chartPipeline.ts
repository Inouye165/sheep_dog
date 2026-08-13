import type { CheckpointEntry, TrainingEpisode } from "../state/types";

export type ViewWindow = "all" | 25 | 50 | 100;
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
  isCheckpointFallback?: boolean;
  checkpointSuccessRate?: number;
}

export interface EpisodeBucket {
  bucketIndex?: number;
  firstEpisode: number;
  lastEpisode: number;
  startTimestep: number;
  endTimestep: number;
  startTimestamp: string;
  endTimestamp: string;
  startTimestampMs: number;
  endTimestampMs: number;
  episodeCount: number;
  successCount: number;
  failureCount: number;
  successRate: number; // 0 to 100
  avgSuccessfulSteps: number | null; // Average steps among successes, or null if 0 successes
  avgAllReward: number;
  avgSheepPenned: number;
  episodes?: CanonicalEpisodeRecord[];
  isCheckpointFallback?: boolean;
}

export interface FormalEvalMarker {
  checkpoint: CheckpointEntry;
  xVal: number;
  successRatePct: number;
  avgSteps: number | null;
  avgReward: number | null;
  avgSheepPenned: number | null;
  stage: number;
  isBest: boolean;
  label: string;
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
    : globalEp * 500; // fallback deterministic scale

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

  const completedAt = raw.completed_at || raw.created_at || new Date(1700000000000 + id * 1000).toISOString();
  let tsMs = new Date(completedAt).getTime();
  if (isNaN(tsMs)) {
    tsMs = 1700000000000 + id * 1000;
  }

  return {
    id,
    run_id: raw.run_id,
    session_id: raw.session_id,
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
  if (c.stage != null && typeof c.stage === "number" && !isNaN(c.stage)) return c.stage;
  if (c.curriculum_stage != null && typeof c.curriculum_stage === "number" && !isNaN(c.curriculum_stage)) return c.curriculum_stage;
  return 1;
}

export function normalizeCheckpointToCanonical(
  c: CheckpointEntry,
  indexFallback: number
): CanonicalEpisodeRecord {
  const rawRate = c.success_rate;
  let successRate = 0;
  if (rawRate != null && typeof rawRate === "number" && !isNaN(rawRate)) {
    successRate = rawRate <= 1.0 ? rawRate * 100 : rawRate;
  }

  const steps = c.average_completion_steps ?? c.average_completion_seconds ?? 600;
  const global_timestep = c.global_timestep ?? c.global_timesteps ?? c.checkpoint_episode ?? (indexFallback * 1000);
  const stage = getCheckpointStage(c);

  let timestamp_ms = 1700000000000 + indexFallback * 1000;
  const rawTs = c.recorded_at ?? c.created_timestamp ?? c.evaluation_timestamp;
  if (rawTs) {
    const parsed = new Date(rawTs).getTime();
    if (!isNaN(parsed)) timestamp_ms = parsed;
  }

  const episodeNum = c.checkpoint_episode ?? indexFallback;

  return {
    id: episodeNum,
    run_id: c.session_id || c.journey || "current",
    session_id: c.session_id,
    curriculum_stage: stage,
    global_environment_episode: episodeNum,
    episode_in_stage: episodeNum,
    global_timestep,
    completed_at: rawTs ?? new Date(timestamp_ms).toISOString(),
    timestamp_ms,
    steps,
    success: successRate >= 50,
    result: successRate >= 50 ? "SUCCESS" : "TIMEOUT",
    reward: c.average_reward ?? 0,
    sheep_penned: c.average_sheep_penned ?? 0,
    total_sheep: 10,
    isCheckpointFallback: true,
    checkpointSuccessRate: successRate,
  };
}

/**
 * Strict, immutable canonical ordering comparator.
 * Primary key: global_timestep ASC
 * Secondary key: timestamp_ms ASC
 * Tertiary key: id ASC
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
 * 1. Normalize records
 * 2. Deduplicate by unique id
 * 3. Filter by stage and run_id
 * 4. Sort strictly chronologically via compareCanonicalEpisodes
 * 
 * If rawEpisodes is empty or yields no matches, falls back seamlessly to checkpoints.
 */
export function processCanonicalHistory(
  rawEpisodes: Partial<TrainingEpisode>[],
  arg2?: CheckpointEntry[] | StageScope,
  arg3?: StageScope | string,
  arg4?: string | number,
  arg5?: number
): CanonicalEpisodeRecord[] {
  let checkpoints: CheckpointEntry[] = [];
  let stageScope: StageScope = "current-journey";
  let activeRunId: string | undefined = undefined;
  let effectiveStage: number = 1;

  if (Array.isArray(arg2)) {
    checkpoints = arg2;
    stageScope = (arg3 as StageScope) ?? "current-journey";
    activeRunId = arg4 as string | undefined;
    effectiveStage = (arg5 as number) ?? 1;
  } else {
    checkpoints = [];
    stageScope = (arg2 as StageScope) ?? "current-journey";
    activeRunId = arg3 as string | undefined;
    effectiveStage = (arg4 as number) ?? 1;
  }

  // Helper to filter canonical records by stage scope
  const applyFilters = (records: CanonicalEpisodeRecord[]): CanonicalEpisodeRecord[] => {
    if (stageScope === "current-journey") {
      if (activeRunId) {
        return records.filter((e) => e.run_id === activeRunId);
      }
      return records;
    }
    if (stageScope === "all") {
      return records;
    }
    const targetStage = stageScope === "current" ? effectiveStage : Number(stageScope);
    if (!isNaN(targetStage)) {
      let filtered = records.filter((e) => e.curriculum_stage === targetStage);
      if (activeRunId && filtered.some((e) => e.run_id === activeRunId)) {
        filtered = filtered.filter((e) => e.run_id === activeRunId);
      }
      return filtered;
    }
    return records;
  };

  // 1. Process raw rollout episodes if present
  if (rawEpisodes && rawEpisodes.length > 0) {
    const normalized = rawEpisodes.map((raw, idx) => normalizeEpisodeRecord(raw, idx + 1));
    const map = new Map<number, CanonicalEpisodeRecord>();
    for (const ep of normalized) {
      map.set(ep.id, ep);
    }
    const filtered = applyFilters(Array.from(map.values()));
    if (filtered.length > 0) {
      return filtered.sort(compareCanonicalEpisodes);
    }
  }

  // 2. Fallback to checkpoints if rawEpisodes is empty or returned 0 items
  if (checkpoints && checkpoints.length > 0) {
    const normalizedCkpts = checkpoints.map((c, idx) => normalizeCheckpointToCanonical(c, idx + 1));
    const filteredCkpts = applyFilters(normalizedCkpts);
    return filteredCkpts.sort(compareCanonicalEpisodes);
  }

  return [];
}

/**
 * Single source of truth pipeline step 5:
 * Select requested view window slice.
 * GUARANTEES tail-subset invariant: Last 25 === All.slice(-25), Last 50 === All.slice(-50), etc.
 */
export function selectWindowSlice(
  canonicalHistory: CanonicalEpisodeRecord[],
  window: ViewWindow
): CanonicalEpisodeRecord[] {
  if (window === "all") return canonicalHistory;
  return canonicalHistory.slice(-window);
}

/**
 * Single source of truth pipeline step 6 & 7:
 * Group active episode sequence into explicit, non-overlapping contiguous buckets.
 * Calculate success rate and average steps TOGETHER from the exact same bucket episodes.
 */
export function buildEpisodeBuckets(
  episodes: CanonicalEpisodeRecord[],
  targetBucketSize: number = 25
): EpisodeBucket[] {
  if (episodes.length === 0) return [];

  // Checkpoint fallback mode: each checkpoint represents a pre-aggregated evaluation point
  if (episodes[0].isCheckpointFallback) {
    return episodes.map((ep) => ({
      firstEpisode: ep.episode_in_stage,
      lastEpisode: ep.episode_in_stage,
      startTimestep: ep.global_timestep,
      endTimestep: ep.global_timestep,
      startTimestamp: ep.completed_at,
      endTimestamp: ep.completed_at,
      startTimestampMs: ep.timestamp_ms,
      endTimestampMs: ep.timestamp_ms,
      episodeCount: 1,
      successCount: ep.checkpointSuccessRate != null && ep.checkpointSuccessRate > 0 ? 1 : 0,
      failureCount: ep.checkpointSuccessRate != null && ep.checkpointSuccessRate > 0 ? 0 : 1,
      successRate: ep.checkpointSuccessRate ?? (ep.success ? 100 : 0),
      avgSuccessfulSteps: ep.steps > 0 ? ep.steps : null,
      avgAllReward: ep.reward,
      avgSheepPenned: ep.sheep_penned,
      isCheckpointFallback: true,
    }));
  }

  const bucketSize = Math.max(1, targetBucketSize);
  const buckets: EpisodeBucket[] = [];
  const totalBuckets = Math.ceil(episodes.length / bucketSize);

  for (let b = 0; b < totalBuckets; b++) {
    const slice = episodes.slice(b * bucketSize, (b + 1) * bucketSize);
    if (slice.length === 0) continue;

    const firstEp = slice[0];
    const lastEp = slice[slice.length - 1];

    const successes = slice.filter((e) => e.success);
    const failureCount = slice.length - successes.length;
    const successRate = (successes.length / slice.length) * 100;

    let avgSuccessfulSteps: number | null = null;
    if (successes.length > 0) {
      const totalSuccSteps = successes.reduce((sum, e) => sum + e.steps, 0);
      avgSuccessfulSteps = totalSuccSteps / successes.length;
    }

    const avgAllReward = slice.reduce((sum, e) => sum + e.reward, 0) / slice.length;
    const avgSheepPenned = slice.reduce((sum, e) => sum + e.sheep_penned, 0) / slice.length;

    buckets.push({
      bucketIndex: b + 1,
      firstEpisode: firstEp.episode_in_stage,
      lastEpisode: lastEp.episode_in_stage,
      startTimestep: firstEp.global_timestep,
      endTimestep: lastEp.global_timestep,
      startTimestamp: firstEp.completed_at,
      endTimestamp: lastEp.completed_at,
      startTimestampMs: firstEp.timestamp_ms,
      endTimestampMs: lastEp.timestamp_ms,
      episodeCount: slice.length,
      successCount: successes.length,
      failureCount,
      successRate,
      avgSuccessfulSteps,
      avgAllReward,
      avgSheepPenned,
      episodes: slice,
    });
  }

  return buckets;
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

/**
 * Builds landmark formal 10-seed evaluation markers from checkpoint index.
 * Positioned strictly at their exact X coordinate (global_timestep or timestamp).
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
    filtered = checkpoints.filter((c) => {
      const cStage = c.reward_config?.instincts?.curriculum_stage ?? c.environment_config?.curriculum_stage ?? c.curriculum_stage ?? -1;
      return cStage === targetStage;
    });
  }

  return filtered
    .map((c) => {
      let xVal: number | null = null;
      if (xAxisMode === "timesteps") {
        xVal = c.global_timesteps ?? c.checkpoint_episode ?? null;
      } else if (xAxisMode === "episode") {
        xVal = c.total_training_episodes ?? c.cumulative_environment_episodes ?? c.checkpoint_episode ?? null;
      } else if (xAxisMode === "runtime") {
        xVal = c.checkpoint_episode ?? null;
      } else if (xAxisMode === "calendar") {
        const ts = c.recorded_at ?? c.evaluation_timestamp ?? c.created_timestamp;
        xVal = ts ? new Date(ts).getTime() : c.checkpoint_episode ?? null;
      }

      if (xVal == null || isNaN(xVal)) return null;

      const rawRate = c.success_rate;
      const ratePct = rawRate <= 1.0 ? rawRate * 100 : rawRate;

      return {
        checkpoint: c,
        xVal,
        successRatePct: ratePct,
        avgSteps: c.average_completion_steps ?? null,
        avgReward: c.average_reward ?? null,
        avgSheepPenned: c.average_sheep_penned ?? null,
        stage: c.curriculum_stage ?? 1,
        isBest: c.checkpoint_episode === bestCheckpointEpisode,
        label: `Formal Benchmark: ${Math.round(ratePct)}% (${c.evaluation_seed_count ?? 10}-seed test at ep ${c.checkpoint_episode})`,
      };
    })
    .filter((m): m is FormalEvalMarker => m !== null)
    .sort((a, b) => a.xVal - b.xVal);
}

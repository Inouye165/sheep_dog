import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  type Dispatch,
  type MutableRefObject,
  type SetStateAction,
} from "react";
import {
  CHECKPOINT_INDEX_SAFETY_REFRESH_MS,
  checkBackendHealth,
  isBackendStartingError,
  loadCheckpointIndexConditional,
  loadEffectiveConfig,
  loadHyperparams,
  loadReplay,
  loadSessionEvents,
  loadTrainingEpisodes,
  loadTrainingStatus,
  runReplay,
} from "../lib/api";
import type {
  CheckpointEntry,
  CheckpointIndex,
  ReplayBundle,
  ReplaySnapshot,
  SessionChangeEvent,
  TrainingEpisode,
  TrainingStatus,
} from "../state/types";
import { isDocumentVisible } from "./useVisibility";

export const DEFAULT_EVAL_SEEDS = [101, 103, 107, 109, 113, 127, 131, 137, 139, 149];

export function getCheckpointStage(c: CheckpointEntry): number {
  if (
    c.reward_config?.instincts?.curriculum_stage !== undefined &&
    c.reward_config?.instincts?.curriculum_stage !== null
  ) {
    return c.reward_config.instincts.curriculum_stage;
  }
  if (
    c.environment_config?.curriculum_stage !== undefined &&
    c.environment_config?.curriculum_stage !== null
  ) {
    return c.environment_config.curriculum_stage;
  }
  return -1;
}

export function extractReplayId(replayPath?: string | null): string | null {
  if (!replayPath) return null;
  const base = replayPath.split(/[\\/]/).pop();
  return base ? base.replace(/\.json(\.gz)?$/i, "") : null;
}

function isStrictlyBetterCheckpoint(
  candidate: CheckpointEntry,
  current: CheckpointEntry,
): boolean {
  const cStage = getCheckpointStage(candidate);
  const curStage = getCheckpointStage(current);
  if (cStage > curStage) return true;
  if (cStage < curStage) return false;
  if (candidate.success_rate > current.success_rate) return true;
  if (candidate.success_rate < current.success_rate) return false;
  return (
    (candidate.average_completion_steps ?? Infinity) <
    (current.average_completion_steps ?? Infinity)
  );
}

const CLEAR_TRAINING_MESSAGE = "Training cleared. Baseline replay restored";
const TIMESTEP_MESSAGE_RE = /(\d+)\s*\/\s*(\d+)\s*timesteps(?:\s*\((\d+)%\))?/i;
const SEGMENT_MESSAGE_RE = /segment\s+(\d+)\s*\/\s*(\d+)/i;
const CHECKPOINT_FETCH_COOLDOWN_MS = 10_000;

export function mergeTrainingStatus(
  previous: TrainingStatus | null,
  next: TrainingStatus,
): TrainingStatus {
  let merged = next;
  if (
    next.phase === "idle" &&
    next.message === "Idle" &&
    previous?.message === CLEAR_TRAINING_MESSAGE
  ) {
    merged = { ...next, message: previous.message };
  }
  // Prevent batch progress and intra-segment telemetry from oscillating backwards while a batch is in flight.
  if (previous && previous.running && merged.running) {
    const sameBatch =
      previous.batch_total_episodes === merged.batch_total_episodes &&
      previous.batch_total_episodes > 0;

    if (sameBatch) {
      if (merged.batch_completed_episodes < previous.batch_completed_episodes) {
        merged = {
          ...merged,
          batch_completed_episodes: Math.max(
            previous.batch_completed_episodes,
            merged.batch_completed_episodes,
          ),
          completed_episodes: Math.max(previous.completed_episodes, merged.completed_episodes),
        };
      }
      if (merged.batch_total_segments == null && previous.batch_total_segments != null) {
        merged = {
          ...merged,
          batch_total_segments: previous.batch_total_segments,
        };
      }
      const sameSegmentCount =
        merged.batch_total_segments == null ||
        previous.batch_total_segments == null ||
        merged.batch_total_segments === previous.batch_total_segments;
      if (
        sameSegmentCount &&
        (merged.batch_completed_segments ?? 0) < (previous.batch_completed_segments ?? 0)
      ) {
        merged = {
          ...merged,
          batch_completed_segments: previous.batch_completed_segments,
        };
      }
      if (merged.batch_total_timesteps == null && previous.batch_total_timesteps != null) {
        merged = {
          ...merged,
          batch_total_timesteps: previous.batch_total_timesteps,
        };
      }

      const prevTsMatch = previous.message?.match(TIMESTEP_MESSAGE_RE);
      const nextTsMatch = next.message?.match(TIMESTEP_MESSAGE_RE);
      if (prevTsMatch && nextTsMatch) {
        const prevDone = Number(prevTsMatch[1]);
        const prevTotal = Number(prevTsMatch[2]);
        const nextDone = Number(nextTsMatch[1]);
        const nextTotal = Number(nextTsMatch[2]);
        if (prevTotal === nextTotal && nextDone < prevDone) {
          const checkpointAdvanced =
            next.latest_checkpoint_episode != null &&
            previous.latest_checkpoint_episode != null &&
            next.latest_checkpoint_episode > previous.latest_checkpoint_episode;
          const stageAdvanced = next.curriculum_stage !== previous.curriculum_stage;
          const prevSegRaw = previous.batch_completed_segments ?? 0;
          const nextSegRaw = next.batch_completed_segments ?? 0;
          const prevSegFromMsg = previous.message?.match(SEGMENT_MESSAGE_RE);
          const nextSegFromMsg = next.message?.match(SEGMENT_MESSAGE_RE);
          const prevSegNum = prevSegFromMsg
            ? Number(prevSegFromMsg[1])
            : Math.floor(prevSegRaw) + (prevDone < prevTotal ? 1 : 0);
          const nextSegNum = nextSegFromMsg ? Number(nextSegFromMsg[1]) : null;
          const segmentAdvanced =
            checkpointAdvanced ||
            stageAdvanced ||
            (nextSegNum !== null && nextSegNum > prevSegNum) ||
            (nextSegRaw > prevSegRaw && prevDone >= prevTotal) ||
            Math.floor(nextSegRaw) > Math.floor(prevSegRaw);
          if (!segmentAdvanced) {
            merged = {
              ...merged,
              message: previous.message,
            };
          }
        }
      }
    }

    if ((previous.current_global_timestep ?? 0) > (merged.current_global_timestep ?? 0)) {
      merged = {
        ...merged,
        current_global_timestep: previous.current_global_timestep,
      };
    }
  }
  return merged;
}

export interface UseTrainingStatusPollingOptions {
  activePollDelayMs: number;
  statusMutationSeqRef: MutableRefObject<number>;
  manualStageOverrideRef: MutableRefObject<boolean>;
  onStatusReceived: Dispatch<SetStateAction<TrainingStatus | null>>;
  syncTrainingStageFromStatus: (status: TrainingStatus | null) => void;
}

export function useTrainingStatusPolling({
  activePollDelayMs,
  statusMutationSeqRef,
  manualStageOverrideRef,
  onStatusReceived,
  syncTrainingStageFromStatus,
}: UseTrainingStatusPollingOptions): void {
  const pollDelayRef = useRef<number>(activePollDelayMs);
  const onStatusReceivedRef = useRef(onStatusReceived);
  const syncStageRef = useRef(syncTrainingStageFromStatus);
  onStatusReceivedRef.current = onStatusReceived;
  syncStageRef.current = syncTrainingStageFromStatus;

  useEffect(() => {
    let active = true;
    let timerId: ReturnType<typeof setTimeout> | null = null;
    let pollInFlight = false;
    let isInitialFetch = true;

    const clearPendingTimer = () => {
      if (timerId !== null) {
        clearTimeout(timerId);
        timerId = null;
      }
    };

    const scheduleNext = () => {
      if (!active || !isDocumentVisible()) {
        return;
      }
      clearPendingTimer();
      timerId = setTimeout(() => {
        timerId = null;
        void poll();
      }, pollDelayRef.current);
    };

    const poll = async () => {
      if (!active || pollInFlight || !isDocumentVisible()) {
        return;
      }
      pollInFlight = true;
      const firstFetch = isInitialFetch;
      isInitialFetch = false;
      const seq = statusMutationSeqRef.current;
      try {
        const status = await loadTrainingStatus();
        if (!active || seq !== statusMutationSeqRef.current) return;
        if (pollDelayRef.current !== activePollDelayMs) {
          console.log(
            `[Polling] Server back online — resetting poll delay to ${activePollDelayMs}ms.`,
          );
          pollDelayRef.current = activePollDelayMs;
        }
        onStatusReceivedRef.current((previous) => mergeTrainingStatus(previous, status));
        if (status.running) {
          syncStageRef.current(status);
        } else if (firstFetch) {
          const localStage = parseInt(localStorage.getItem("sheepdog_curriculum_stage") ?? "0", 10);
          const serverStage = status.curriculum_stage;
          const maxPromoteStage = (status.max_curriculum_stage ?? serverStage ?? 1) + 1;
          if (
            !manualStageOverrideRef.current &&
            serverStage != null &&
            (serverStage > localStage || localStage > maxPromoteStage)
          ) {
            syncStageRef.current(status);
          }
        }
      } catch (err) {
        if (!active || seq !== statusMutationSeqRef.current) return;
        const backendStarting = isBackendStartingError(err);
        const retryDelay = backendStarting ? 2000 : 5000;
        if (pollDelayRef.current !== retryDelay) {
          if (backendStarting) {
            console.info("[Polling] Backend is restoring training state; retrying every 2000ms.");
          } else {
            console.error("[Polling] Connection failed, switching to backoff delay of 5000ms.");
          }
          pollDelayRef.current = retryDelay;
        }
        const restoringMessage = backendStarting
          ? (err as Error).message || "Restoring training state"
          : null;
        onStatusReceivedRef.current((prev) => {
          const base = prev ?? {
            running: false,
            fast_mode: false,
            enable_instinct_rewards: false,
            debug_reward_breakdown: false,
            curriculum_stage: 0,
            requested_episodes: 0,
            completed_episodes: 0,
            batch_total_episodes: 0,
            batch_completed_episodes: 0,
            total_episodes_trained: 0,
            stage_history: {},
            grand_total_episodes: 0,
            current_episode: null,
            checkpoint_episode: null,
            latest_checkpoint_episode: null,
            latest_seed: null,
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
            phase: "offline",
            message: "Backend disconnected",
            error: "Backend disconnected",
            error_type: "Connection Error",
            starting_episode: null,
          };
          if (restoringMessage !== null) {
            return {
              ...base,
              running: false,
              phase: "restoring",
              message: restoringMessage,
              error: null,
              error_type: null,
            };
          }
          return {
            ...base,
            running: false,
            phase: "offline",
            message: "Backend disconnected",
            error: "Backend disconnected",
            error_type: "Connection Error",
          };
        });
      } finally {
        pollInFlight = false;
        scheduleNext();
      }
    };

    const handleVisibilityOrFocus = () => {
      if (!active) return;
      if (!isDocumentVisible()) {
        clearPendingTimer();
        return;
      }
      clearPendingTimer();
      void poll();
    };

    console.log("[Polling] Status polling started.");
    if (isDocumentVisible()) {
      void poll();
    }

    document.addEventListener("visibilitychange", handleVisibilityOrFocus);
    window.addEventListener("focus", handleVisibilityOrFocus);

    return () => {
      active = false;
      clearPendingTimer();
      document.removeEventListener("visibilitychange", handleVisibilityOrFocus);
      window.removeEventListener("focus", handleVisibilityOrFocus);
      console.log("[Polling] Status polling stopped.");
    };
  }, [activePollDelayMs, manualStageOverrideRef, statusMutationSeqRef]);
}

export interface CheckpointIndexMeta {
  hasLoaded: boolean;
  length: number;
  maxEp: number;
  lastId: string | number | null;
  etag: string | null;
}

function extractCheckpointIndexMeta(
  index: CheckpointIndex | null,
  etag: string | null,
): CheckpointIndexMeta {
  const checkpoints = index?.checkpoints ?? [];
  const lastCp = checkpoints.length > 0 ? checkpoints[checkpoints.length - 1] : null;
  return {
    hasLoaded: true,
    length: checkpoints.length,
    maxEp: lastCp ? lastCp.checkpoint_episode : -1,
    lastId: lastCp
      ? lastCp.checkpoint_id ?? lastCp.global_timestep ?? lastCp.checkpoint_episode ?? null
      : null,
    etag,
  };
}

export interface UseCheckpointIndexSyncOptions {
  hasTrainingStatus: boolean;
  statusRunning: boolean;
  statusPhase: string | null;
  statusError: string | null;
  statusLatestCpEp: number | null;
  statusActiveCpId: string | number | null;
  onIndexApplied: (index: CheckpointIndex | null) => void;
  onRefreshScenarios: () => Promise<void> | void;
  onClearError?: () => void;
}

export function useCheckpointIndexSync({
  hasTrainingStatus,
  statusRunning,
  statusPhase,
  statusError,
  statusLatestCpEp,
  statusActiveCpId,
  onIndexApplied,
  onRefreshScenarios,
  onClearError,
}: UseCheckpointIndexSyncOptions): {
  syncIndexMeta: (index: CheckpointIndex | null) => void;
  markIndexCleared: () => void;
} {
  const lastMetaRef = useRef<CheckpointIndexMeta>({
    hasLoaded: false,
    length: 0,
    maxEp: -1,
    lastId: null,
    etag: null,
  });
  const lastFetchedAtRef = useRef<number>(0);
  const lastFetchDidAdvanceRef = useRef<boolean>(true);
  const inFlightRef = useRef<boolean>(false);
  const lastObservedStatusEpRef = useRef<number | null>(null);
  const lastObservedActiveCpIdRef = useRef<string | number | null>(null);

  const onIndexAppliedRef = useRef(onIndexApplied);
  const onRefreshScenariosRef = useRef(onRefreshScenarios);
  const onClearErrorRef = useRef(onClearError);
  onIndexAppliedRef.current = onIndexApplied;
  onRefreshScenariosRef.current = onRefreshScenarios;
  onClearErrorRef.current = onClearError;

  const syncIndexMeta = useCallback((index: CheckpointIndex | null) => {
    lastMetaRef.current = extractCheckpointIndexMeta(index, null);
    lastFetchDidAdvanceRef.current = true;
  }, []);

  const markIndexCleared = useCallback(() => {
    lastMetaRef.current = {
      hasLoaded: false,
      length: 0,
      maxEp: -1,
      lastId: null,
      etag: null,
    };
    lastFetchedAtRef.current = 0;
    lastFetchDidAdvanceRef.current = true;
    lastObservedStatusEpRef.current = null;
    lastObservedActiveCpIdRef.current = null;
  }, []);

  useEffect(() => {
    if (
      !hasTrainingStatus ||
      statusPhase === "offline" ||
      statusPhase === "restoring" ||
      statusError
    ) {
      return;
    }

    let active = true;

    const performFetch = async () => {
      if (!active || inFlightRef.current || !isDocumentVisible()) {
        return;
      }
      inFlightRef.current = true;
      lastFetchedAtRef.current = Date.now();
      try {
        console.debug("[Sheepdog API] Server online; updating checkpoint index...");
        const res = await loadCheckpointIndexConditional(lastMetaRef.current.etag);
        if (!active) return;

        if (res.status === "not_modified") {
          lastMetaRef.current = {
            ...lastMetaRef.current,
            etag: res.etag ?? lastMetaRef.current.etag,
          };
          lastFetchDidAdvanceRef.current = false;
          return;
        }

        if (res.status === "empty") {
          lastFetchDidAdvanceRef.current = false;
          if (!lastMetaRef.current.hasLoaded) {
            lastMetaRef.current = {
              hasLoaded: true,
              length: 0,
              maxEp: -1,
              lastId: null,
              etag: null,
            };
          }
          return;
        }

        const prevMeta = lastMetaRef.current;
        const nextMeta = extractCheckpointIndexMeta(res.index, res.etag);
        const changed =
          !prevMeta.hasLoaded ||
          prevMeta.length !== nextMeta.length ||
          prevMeta.lastId !== nextMeta.lastId;
        const advanced =
          !prevMeta.hasLoaded ||
          nextMeta.maxEp > prevMeta.maxEp ||
          nextMeta.length > prevMeta.length ||
          nextMeta.lastId !== prevMeta.lastId;

        lastMetaRef.current = nextMeta;
        lastFetchDidAdvanceRef.current = advanced;

        if (changed) {
          onIndexAppliedRef.current(res.index);
          void onRefreshScenariosRef.current();
        }
        onClearErrorRef.current?.();
      } catch (err) {
        console.debug("[Sheepdog API] Checkpoint index update deferred:", err);
      } finally {
        inFlightRef.current = false;
      }
    };

    const now = Date.now();
    const withinCooldown = now - lastFetchedAtRef.current < CHECKPOINT_FETCH_COOLDOWN_MS;
    const needsInitialLoad = !lastMetaRef.current.hasLoaded;
    const statusEpIncreased =
      statusLatestCpEp !== null && statusLatestCpEp > lastMetaRef.current.maxEp;
    const statusEpChanged = statusLatestCpEp !== lastObservedStatusEpRef.current;
    const activeIdChanged =
      statusActiveCpId != null &&
      String(statusActiveCpId) !== String(lastObservedActiveCpIdRef.current ?? "");
    const activeIdMatchesLoaded =
      statusActiveCpId != null &&
      lastMetaRef.current.lastId != null &&
      String(statusActiveCpId) === String(lastMetaRef.current.lastId);

    lastObservedStatusEpRef.current = statusLatestCpEp;
    lastObservedActiveCpIdRef.current = statusActiveCpId;

    const shouldFetchFromStatusEp =
      statusEpIncreased &&
      ((statusEpChanged && lastFetchDidAdvanceRef.current) || !withinCooldown);
    const shouldFetchFromActiveId =
      activeIdChanged && !activeIdMatchesLoaded && !withinCooldown;

    if (
      isDocumentVisible() &&
      (needsInitialLoad || shouldFetchFromStatusEp || shouldFetchFromActiveId)
    ) {
      void performFetch();
    }

    const handleVisibilityChange = () => {
      if (!active || !isDocumentVisible()) {
        return;
      }
      if (statusRunning || !lastMetaRef.current.hasLoaded) {
        void performFetch();
      }
    };

    document.addEventListener("visibilitychange", handleVisibilityChange);

    let safetyIntervalId: ReturnType<typeof setInterval> | null = null;
    if (statusRunning) {
      safetyIntervalId = setInterval(() => {
        if (isDocumentVisible()) {
          void performFetch();
        }
      }, CHECKPOINT_INDEX_SAFETY_REFRESH_MS);
    }

    return () => {
      active = false;
      document.removeEventListener("visibilitychange", handleVisibilityChange);
      if (safetyIntervalId !== null) {
        clearInterval(safetyIntervalId);
      }
    };
  }, [
    hasTrainingStatus,
    statusRunning,
    statusPhase,
    statusError,
    statusLatestCpEp,
    statusActiveCpId,
  ]);

  return { syncIndexMeta, markIndexCleared };
}

export interface UseProgressInsightsPollingOptions {
  active: boolean;
  statusRunning: boolean;
  statusLatestCpEp: number | null;
  statusActiveCpId: string | number | null;
  onEpisodesReceived: Dispatch<SetStateAction<TrainingEpisode[]>>;
  onSessionEventsReceived: Dispatch<SetStateAction<SessionChangeEvent[]>>;
}

export function useProgressInsightsPolling({
  active: isEnabled,
  statusRunning,
  statusLatestCpEp,
  statusActiveCpId,
  onEpisodesReceived,
  onSessionEventsReceived,
}: UseProgressInsightsPollingOptions): void {
  const onEpisodesRef = useRef(onEpisodesReceived);
  const onEventsRef = useRef(onSessionEventsReceived);
  onEpisodesRef.current = onEpisodesReceived;
  onEventsRef.current = onSessionEventsReceived;

  useEffect(() => {
    if (!isEnabled) {
      return;
    }

    let active = true;
    let inFlight = false;

    const fetchInsights = async () => {
      if (!active || inFlight || !isDocumentVisible()) {
        return;
      }
      inFlight = true;
      try {
        const [epRes, evRes] = await Promise.allSettled([
          loadTrainingEpisodes({ limit: 500 }),
          loadSessionEvents(),
        ]);
        if (!active) return;
        if (epRes.status === "fulfilled" && epRes.value?.episodes) {
          onEpisodesRef.current(epRes.value.episodes);
        }
        if (evRes.status === "fulfilled" && evRes.value?.events) {
          onEventsRef.current(evRes.value.events);
        }
      } finally {
        inFlight = false;
      }
    };

    if (isDocumentVisible()) {
      void fetchInsights();
    }

    const handleVisibilityChange = () => {
      if (active && isDocumentVisible()) {
        void fetchInsights();
      }
    };
    document.addEventListener("visibilitychange", handleVisibilityChange);

    let timerId: ReturnType<typeof setInterval> | null = null;
    if (statusRunning) {
      timerId = setInterval(() => {
        if (isDocumentVisible()) {
          void fetchInsights();
        }
      }, 8000);
    }

    return () => {
      active = false;
      document.removeEventListener("visibilitychange", handleVisibilityChange);
      if (timerId !== null) {
        clearInterval(timerId);
      }
    };
  }, [isEnabled, statusRunning, statusLatestCpEp, statusActiveCpId]);
}

export interface UseDerivedCheckpointStateOptions {
  checkpointIndex: CheckpointIndex | null;
  selectedCheckpointEpisode: number | null;
  effectiveCurriculumStage: number;
  totalTrainingEpisodes: number;
  statusLatestCheckpointEpisode: number | null;
}

export function useDerivedCheckpointState({
  checkpointIndex,
  selectedCheckpointEpisode,
  effectiveCurriculumStage,
  totalTrainingEpisodes,
  statusLatestCheckpointEpisode,
}: UseDerivedCheckpointStateOptions) {
  const selectedCheckpoint = useMemo(() => {
    return (
      checkpointIndex?.checkpoints.find(
        (entry) => entry.checkpoint_episode === selectedCheckpointEpisode,
      ) ?? null
    );
  }, [checkpointIndex, selectedCheckpointEpisode]);

  const seedOptions = useMemo(() => {
    if (selectedCheckpoint?.records && selectedCheckpoint.records.length > 0) {
      return selectedCheckpoint.records.map((record) => record.seed);
    }
    if (selectedCheckpoint) {
      return DEFAULT_EVAL_SEEDS;
    }
    return [];
  }, [selectedCheckpoint]);

  const latestSuccessRate = useMemo(() => {
    const checkpoints = checkpointIndex?.checkpoints;
    if (!checkpoints?.length) return null;
    const stageCheckpoints = checkpoints.filter(
      (c) => getCheckpointStage(c) === effectiveCurriculumStage,
    );
    if (!stageCheckpoints.length) return null;
    return stageCheckpoints[stageCheckpoints.length - 1]?.success_rate ?? null;
  }, [checkpointIndex, effectiveCurriculumStage]);

  const bestStageFormalEpisode = useMemo(() => {
    const stageCheckpoints =
      checkpointIndex?.checkpoints.filter(
        (c) => getCheckpointStage(c) === effectiveCurriculumStage,
      ) ?? [];
    if (!stageCheckpoints.length) return null;
    const bestEntry = stageCheckpoints.reduce((best, entry) => {
      if (!best) return entry;
      if (entry.success_rate > best.success_rate) return entry;
      if (
        entry.success_rate === best.success_rate &&
        (entry.average_completion_steps ?? Infinity) < (best.average_completion_steps ?? Infinity)
      ) {
        return entry;
      }
      return best;
    }, null as CheckpointEntry | null);
    return bestEntry?.checkpoint_episode ?? null;
  }, [checkpointIndex, effectiveCurriculumStage]);

  const currentReplaySelection = useMemo(() => {
    const latestCheckpoint =
      checkpointIndex?.checkpoints[checkpointIndex.checkpoints.length - 1] ?? null;
    const resolvedTotalEpisodes =
      latestCheckpoint?.total_training_episodes ?? totalTrainingEpisodes;
    const latestPolicyMode = latestCheckpoint?.policy_mode ?? latestCheckpoint?.policy_name ?? null;

    if (latestCheckpoint && resolvedTotalEpisodes > 0) {
      const resolvedPolicyMode =
        latestPolicyMode === "neural_policy" ? "neural_policy" : "trained_policy";
      return {
        checkpointEpisode: null,
        trainerType:
          latestCheckpoint.trainer_type ??
          (resolvedPolicyMode === "neural_policy" ? "maskable_ppo" : "hill_climb"),
        policyType:
          latestCheckpoint.policy_type ??
          (resolvedPolicyMode === "neural_policy" ? "neural" : "linear"),
        policyMode: resolvedPolicyMode,
      };
    }

    return {
      checkpointEpisode: null,
      trainerType: "baseline",
      policyType: "instinct",
      policyMode: "instinct_only",
    };
  }, [checkpointIndex, totalTrainingEpisodes]);

  const latestCheckpointEpisode = useMemo(() => {
    const checkpoints = checkpointIndex?.checkpoints;
    if (!checkpoints?.length) {
      return statusLatestCheckpointEpisode;
    }
    return checkpoints[checkpoints.length - 1]?.checkpoint_episode ?? null;
  }, [checkpointIndex, statusLatestCheckpointEpisode]);

  const bestCheckpointEpisode = useMemo(() => {
    if (!checkpointIndex?.checkpoints.length) return null;
    const best = checkpointIndex.checkpoints.reduce((acc, entry) => {
      if (!acc) return entry;
      return isStrictlyBetterCheckpoint(entry, acc) ? entry : acc;
    }, null as CheckpointEntry | null);
    return best?.checkpoint_episode ?? null;
  }, [checkpointIndex]);

  return {
    selectedCheckpoint,
    seedOptions,
    latestSuccessRate,
    bestStageFormalEpisode,
    currentReplaySelection,
    latestCheckpointEpisode,
    bestCheckpointEpisode,
  };
}

export interface UseBackendHealthProbeOptions {
  onHealthReadOnlyPreview: (readOnly: boolean) => void;
  onHealthCheckDone: (done: boolean) => void;
}

export function useBackendHealthProbe({
  onHealthReadOnlyPreview,
  onHealthCheckDone,
}: UseBackendHealthProbeOptions): void {
  const onReadOnlyRef = useRef(onHealthReadOnlyPreview);
  const onDoneRef = useRef(onHealthCheckDone);
  onReadOnlyRef.current = onHealthReadOnlyPreview;
  onDoneRef.current = onHealthCheckDone;

  useEffect(() => {
    let active = true;
    void (async () => {
      try {
        const health = await checkBackendHealth();
        if (active) {
          onReadOnlyRef.current(Boolean(health.readOnlyPreview));
        }
      } catch {
        // Ignore health probe failures; status polling tracks offline state.
      } finally {
        if (active) {
          onDoneRef.current(true);
        }
      }
    })();
    return () => {
      active = false;
    };
  }, []);
}

export interface UseBackendConfigLoaderOptions {
  hasTrainingStatus: boolean;
  statusPhase: string | null;
  onPersonalityStrength: (strength: number) => void;
  onEffectiveConfig: (config: Record<string, unknown> | null) => void;
}

export function useBackendConfigLoader({
  hasTrainingStatus,
  statusPhase,
  onPersonalityStrength,
  onEffectiveConfig,
}: UseBackendConfigLoaderOptions): void {
  const onPersonalityRef = useRef(onPersonalityStrength);
  const onConfigRef = useRef(onEffectiveConfig);
  onPersonalityRef.current = onPersonalityStrength;
  onConfigRef.current = onEffectiveConfig;

  useEffect(() => {
    if (!hasTrainingStatus || statusPhase === "offline" || statusPhase === "restoring") {
      return;
    }
    let active = true;
    void (async () => {
      try {
        const hyperparams = await loadHyperparams();
        if (active && hyperparams?.environment) {
          onPersonalityRef.current(hyperparams.environment.sheep_personality_strength);
        }
      } catch {
        // Keep default when hyperparams are unavailable.
      }
    })();
    void (async () => {
      try {
        const config = await loadEffectiveConfig();
        if (active) {
          onConfigRef.current(config);
        }
      } catch {
        if (active) {
          onConfigRef.current(null);
        }
      }
    })();
    return () => {
      active = false;
    };
  }, [hasTrainingStatus, statusPhase]);
}

export interface UseSelectedReplaySyncOptions {
  isPageVisible: boolean;
  selectedCheckpoint: CheckpointEntry | null;
  selectedSeed: number | null;
  replayReloadNonce: number;
  latestReplayPath: string | null;
  statusRunning: boolean;
  effectiveEnableInstincts: boolean;
  effectiveCurriculumStage: number;
  effectiveDebugRewardBreakdown: boolean;
  statusReadOnlyPreview: boolean;
  healthCheckDone: boolean;
  replayRef: MutableRefObject<ReplayBundle | null>;
  lastLoadedSelectionKeyRef: MutableRefObject<string | null>;
  lastLoadedReplayIdRef: MutableRefObject<string | null>;
  lastHandledLatestReplayPathRef: MutableRefObject<string | null>;
  autoPlaySelectionKeyRef: MutableRefObject<string | null>;
  onReplayLoaded: (bundle: ReplayBundle, nextRunState: "running" | "idle") => void;
  onError: (message: string) => void;
}

export function useSelectedReplaySync({
  isPageVisible,
  selectedCheckpoint,
  selectedSeed,
  replayReloadNonce,
  latestReplayPath,
  statusRunning,
  effectiveEnableInstincts,
  effectiveCurriculumStage,
  effectiveDebugRewardBreakdown,
  statusReadOnlyPreview,
  healthCheckDone,
  replayRef,
  lastLoadedSelectionKeyRef,
  lastLoadedReplayIdRef,
  lastHandledLatestReplayPathRef,
  autoPlaySelectionKeyRef,
  onReplayLoaded,
  onError,
}: UseSelectedReplaySyncOptions): void {
  const onReplayLoadedRef = useRef(onReplayLoaded);
  const onErrorRef = useRef(onError);
  onReplayLoadedRef.current = onReplayLoaded;
  onErrorRef.current = onError;

  useEffect(() => {
    if (!isPageVisible) {
      return;
    }

    const latestReplayId = extractReplayId(latestReplayPath);

    if (selectedCheckpoint) {
      const seed = selectedSeed ?? selectedCheckpoint.records?.[0]?.seed ?? DEFAULT_EVAL_SEEDS[0];
      if (seed === null) {
        return;
      }
      const selectionKey = `${selectedCheckpoint.checkpoint_episode}:${seed}`;
      if (lastLoadedSelectionKeyRef.current === selectionKey) {
        if (
          latestReplayPath &&
          latestReplayPath !== lastHandledLatestReplayPathRef.current &&
          latestReplayId !== lastLoadedReplayIdRef.current
        ) {
          // Fall through to load the newer live replay path below.
        } else {
          if (latestReplayPath) {
            lastHandledLatestReplayPathRef.current = latestReplayPath;
          }
          return;
        }
      } else {
        const record = selectedCheckpoint.records?.find((entry) => entry.seed === seed);
        const recordReplayId = extractReplayId(record?.replay_path);

        if (recordReplayId && lastLoadedReplayIdRef.current === recordReplayId && replayRef.current) {
          lastLoadedSelectionKeyRef.current = selectionKey;
          if (latestReplayPath && latestReplayId === recordReplayId) {
            lastHandledLatestReplayPathRef.current = latestReplayPath;
          }
          return;
        }

        if (!record?.replay_path && (statusReadOnlyPreview || !healthCheckDone)) {
          return;
        }

        let cancelled = false;
        void (async () => {
          try {
            let bundle: ReplayBundle;
            if (record?.replay_path) {
              bundle = await loadReplay(record.replay_path);
            } else {
              const checkpointStage = getCheckpointStage(selectedCheckpoint);
              bundle = await runReplay({
                seed,
                checkpoint_episode: selectedCheckpoint.checkpoint_episode,
                trainer_type: selectedCheckpoint.trainer_type,
                policy_type: selectedCheckpoint.policy_type,
                policy_mode: selectedCheckpoint.policy_mode || selectedCheckpoint.policy_name,
                effective_config: {
                  enable_instinct_rewards: effectiveEnableInstincts,
                  curriculum_stage:
                    checkpointStage >= 0 ? checkpointStage : effectiveCurriculumStage,
                  debug_reward_breakdown: effectiveDebugRewardBreakdown,
                },
              });
            }
            if (cancelled) {
              return;
            }
            lastLoadedSelectionKeyRef.current = selectionKey;
            lastLoadedReplayIdRef.current = recordReplayId;
            if (latestReplayPath && latestReplayId && latestReplayId === recordReplayId) {
              lastHandledLatestReplayPathRef.current = latestReplayPath;
            }
            const shouldAutoPlay = autoPlaySelectionKeyRef.current === selectionKey;
            if (shouldAutoPlay) {
              autoPlaySelectionKeyRef.current = null;
            }
            const nextRunState =
              (shouldAutoPlay || statusRunning) && (bundle.frames?.length ?? 0) > 0
                ? "running"
                : "idle";
            onReplayLoadedRef.current(bundle, nextRunState);
          } catch (loadError) {
            if (!cancelled) {
              onErrorRef.current(
                loadError instanceof Error ? loadError.message : "Unable to load replay.",
              );
            }
          }
        })();
        return () => {
          cancelled = true;
        };
      }
    }

    if (
      !latestReplayPath ||
      latestReplayPath === lastHandledLatestReplayPathRef.current ||
      (latestReplayId && latestReplayId === lastLoadedReplayIdRef.current && replayRef.current)
    ) {
      if (latestReplayPath && latestReplayId === lastLoadedReplayIdRef.current && replayRef.current) {
        lastHandledLatestReplayPathRef.current = latestReplayPath;
      }
      return;
    }

    let cancelled = false;
    void (async () => {
      try {
        const bundle = await loadReplay(latestReplayPath);
        if (!cancelled && bundle) {
          lastHandledLatestReplayPathRef.current = latestReplayPath;
          lastLoadedReplayIdRef.current = latestReplayId;
          onReplayLoadedRef.current(bundle, "running");
        }
      } catch {
        // Fallback silently if replay is still being written
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [
    isPageVisible,
    selectedCheckpoint,
    selectedSeed,
    replayReloadNonce,
    latestReplayPath,
    statusRunning,
    effectiveEnableInstincts,
    effectiveCurriculumStage,
    effectiveDebugRewardBreakdown,
    statusReadOnlyPreview,
    healthCheckDone,
    replayRef,
    lastLoadedSelectionKeyRef,
    lastLoadedReplayIdRef,
    lastHandledLatestReplayPathRef,
    autoPlaySelectionKeyRef,
  ]);
}

export interface UseReplayPlaybackOptions {
  active: boolean;
  framesLength: number;
  delayMs: number;
  replayRef: MutableRefObject<ReplayBundle | null>;
  onStep: Dispatch<SetStateAction<number>>;
  onComplete: (finalSnapshot: ReplaySnapshot | null) => void;
}

export function useReplayPlayback({
  active,
  framesLength,
  delayMs,
  replayRef,
  onStep,
  onComplete,
}: UseReplayPlaybackOptions): void {
  const onStepRef = useRef(onStep);
  const onCompleteRef = useRef(onComplete);
  onStepRef.current = onStep;
  onCompleteRef.current = onComplete;

  useEffect(() => {
    if (!active || framesLength === 0) {
      return;
    }
    const timer = window.setInterval(() => {
      if (!isDocumentVisible()) {
        return;
      }
      const currentReplay = replayRef.current;
      const len = currentReplay?.frames?.length ?? 0;
      if (!currentReplay || len === 0) {
        return;
      }
      onStepRef.current((currentIndex) => {
        if (currentIndex >= len - 1) {
          window.clearInterval(timer);
          onCompleteRef.current(currentReplay.final_snapshot);
          return currentIndex;
        }
        return currentIndex + 1;
      });
    }, delayMs);
    return () => window.clearInterval(timer);
  }, [active, framesLength, delayMs, replayRef]);
}

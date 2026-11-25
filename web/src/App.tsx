import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { CopyAgentDataButton } from "./components/CopyAgentDataButton";
import { EvaluationEpisodesTab } from "./components/EvaluationEpisodesTab";
import {
  clearTraining,
  crossStageFork,
  evaluateScenario,
  loadCheckpointIndex,
  loadNetworkTopology,
  loadReplay,
  loadScenarioIndex,
  loadTrainingStatus,
  pauseTraining,
  replayScenario,
  resetJourneyTraining,
  rewindTraining,
  runReplay,
  saveScenario,
  shutdownApp,
  startRemediationFork,
  startTraining,
  stopTraining,
} from "./lib/api";
import type { CheckpointMode } from "./lib/api";
import {
  DEFAULT_EVAL_SEEDS,
  extractReplayId,
  getCheckpointStage,
  useBackendConfigLoader,
  useBackendHealthProbe,
  useCheckpointIndexSync,
  useDerivedCheckpointState,
  useProgressInsightsPolling,
  useReplayPlayback,
  useSelectedReplaySync,
  useTrainingStatusPolling,
} from "./hooks/useTrainingSync";
import { usePageVisible } from "./hooks/useVisibility";
import { parseStageSpec } from "./lib/stageSpec";
import { recommendedEpisodesForStage } from "./lib/trainingConfig";
import type {
  CheckpointIndex,
  NetworkTopologyInfo,
  ReplayBundle,
  ReplaySnapshot,
  ScenarioIndex,
  SessionChangeEvent,
  TrainingEpisode,
  TrainingStatus,
} from "./state/types";
import { ProgressView } from "./views/ProgressView";
import { SettingsView } from "./views/SettingsView";
import { TrainView } from "./views/TrainView";
import { TroubleSpotsView, type TroubleSpotsMode } from "./views/TroubleSpotsView";

type RunState = "idle" | "running" | "paused" | "success" | "timeout" | "stopped";
type ActiveTab = "train" | "progress" | "trouble" | "settings";

const APP_TABS: { id: ActiveTab; label: string }[] = [
  { id: "train", label: "Train" },
  { id: "progress", label: "Progress" },
  { id: "trouble", label: "Trouble Spots & Replays" },
  { id: "settings", label: "Settings" },
];

const DEFAULT_MAX_CURRICULUM_STAGE = 32;
const ACTIVE_POLL_DELAY_MS = import.meta.env.MODE === "test" ? 500 : 1500;
const TIMESTEP_MESSAGE_RE = /(\d+)\s*\/\s*(\d+)\s*timesteps(?:\s*\((\d+)%\))?/i;
const SEGMENT_MESSAGE_RE = /segment\s+(\d+)\s*\/\s*(\d+)/i;

function resolveInitialNavigation(): {
  activeTab: ActiveTab;
  troubleMode: TroubleSpotsMode;
  replayId: string | null;
} {
  if (typeof window === "undefined") {
    return { activeTab: "train", troubleMode: "failures", replayId: null };
  }
  const params = new URLSearchParams(window.location.search);
  const wsParam = (params.get("workspace") ?? "").toLowerCase();
  const tabParam = (params.get("tab") ?? params.get("mode") ?? "").toLowerCase();
  const replayId = params.get("replay")?.trim() || null;

  if (
    wsParam === "temporal" ||
    wsParam === "temporal-diagnostics" ||
    tabParam === "temporal" ||
    tabParam === "temporal-diagnostics" ||
    Boolean(replayId)
  ) {
    return { activeTab: "trouble", troubleMode: "inspector", replayId };
  }
  if (tabParam === "progress" || wsParam === "analytics") {
    return { activeTab: "progress", troubleMode: "failures", replayId: null };
  }
  if (tabParam === "trouble") {
    return { activeTab: "trouble", troubleMode: "failures", replayId: null };
  }
  if (tabParam === "sandbox") {
    return { activeTab: "trouble", troubleMode: "sandbox", replayId: null };
  }
  if (tabParam === "settings" || tabParam === "config") {
    return { activeTab: "settings", troubleMode: "failures", replayId: null };
  }
  return { activeTab: "train", troubleMode: "failures", replayId: null };
}

function resolveRunState(snapshot: ReplaySnapshot | null, currentState: RunState): RunState {
  if (!snapshot) {
    return currentState;
  }
  if (snapshot.success) {
    return "success";
  }
  if (snapshot.timeout) {
    return "timeout";
  }
  if (snapshot.stopped) {
    return "stopped";
  }
  return currentState;
}

export function App() {
  const initialNav = useMemo(() => resolveInitialNavigation(), []);

  const [isAppClosed, setIsAppClosed] = useState(false);
  const [checkpointIndex, setCheckpointIndex] = useState<CheckpointIndex | null>(null);
  const [replay, setReplay] = useState<ReplayBundle | null>(null);
  const [selectedCheckpointEpisode, setSelectedCheckpointEpisode] = useState<number | null>(null);
  const [selectedSeed, setSelectedSeed] = useState<number | null>(null);
  const [runState, setRunState] = useState<RunState>("idle");
  const [frameIndex, setFrameIndex] = useState(0);
  const [replayReloadNonce, setReplayReloadNonce] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [trainingError, setTrainingError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [healthCheckDone, setHealthCheckDone] = useState(false);
  const [healthReadOnlyPreview, setHealthReadOnlyPreview] = useState(false);
  const [effectiveConfig, setEffectiveConfig] = useState<Record<string, unknown> | null>(null);
  const [networkTopology, setNetworkTopology] = useState<NetworkTopologyInfo | null>(null);
  const [trainingEpisodeLogs, setTrainingEpisodeLogs] = useState<TrainingEpisode[]>([]);
  const [sessionEvents, setSessionEvents] = useState<SessionChangeEvent[]>([]);

  const [trainingEpisodes, setTrainingEpisodes] = useState(50);
  const [trainingFastMode, setTrainingFastMode] = useState(true);
  const [trainingEnableInstincts, setTrainingEnableInstincts] = useState(false);
  const [trainingAutoPromote, setTrainingAutoPromote] = useState(true);
  const [trainingContinuous, setTrainingContinuous] = useState(false);
  const [trainingCurriculumStage, setTrainingCurriculumStage] = useState(() => {
    const saved = localStorage.getItem("sheepdog_curriculum_stage");
    const parsed = saved !== null ? parseInt(saved, 10) : NaN;
    return !isNaN(parsed) && parsed >= 0 ? parsed : 1;
  });
  const [startingModelSource, setStartingModelSource] = useState("latest");
  const [trainingStageSpec, setTrainingStageSpec] = useState("");
  const [trainingPassLabel, setTrainingPassLabel] = useState("");
  const [trainingDebugRewardBreakdown, setTrainingDebugRewardBreakdown] = useState(false);
  const [playbackFastMode, setPlaybackFastMode] = useState(false);
  const [trainingStatus, setTrainingStatus] = useState<TrainingStatus | null>(null);
  const [clearingTraining, setClearingTraining] = useState(false);
  const [isStartingTraining, setIsStartingTraining] = useState(false);
  const [lifecycleAction, setLifecycleAction] = useState<"pause" | "stop" | null>(null);
  const [runningCurrentReplay, setRunningCurrentReplay] = useState(false);
  const [loadingSelectedReplay, setLoadingSelectedReplay] = useState(false);

  const [activeTab, setActiveTab] = useState<ActiveTab>(initialNav.activeTab);
  const [troubleMode, setTroubleMode] = useState<TroubleSpotsMode>(initialNav.troubleMode);
  const [temporalReplayId, setTemporalReplayId] = useState<string | null>(initialNav.replayId);

  const [scenarioReplay, setScenarioReplay] = useState<ReplayBundle | null>(null);
  const [scenarioFrameIndex, setScenarioFrameIndex] = useState(0);
  const [scenarioRunState, setScenarioRunState] = useState<RunState>("idle");
  const [scenarioPersonalityStrength, setScenarioPersonalityStrength] = useState(0);
  const [scenarioSeed, setScenarioSeed] = useState(11);
  const [scenarioFastMode, setScenarioFastMode] = useState(false);
  const [runningScenario, setRunningScenario] = useState(false);
  const [scenarioIndex, setScenarioIndex] = useState<ScenarioIndex | null>(null);
  const [scenarioCheckpointMode, setScenarioCheckpointMode] = useState<CheckpointMode>("latest");
  const [specificScenarioCheckpointEpisode, setSpecificScenarioCheckpointEpisode] = useState<number | null>(null);
  const [selectedScenarioId, setSelectedScenarioId] = useState<string | null>(null);
  const [saveSnapshotSource, setSaveSnapshotSource] = useState<"initial" | "final">("final");
  const [evaluatingScenarios, setEvaluatingScenarios] = useState(false);
  const [promoteFromEpisode, setPromoteFromEpisode] = useState<number | null>(null);

  const prevTrainingRunning = useRef(false);
  const manualStageOverrideRef = useRef(false);
  const statusMutationSeqRef = useRef(0);
  const lastLoadedSelectionKeyRef = useRef<string | null>(null);
  const lastLoadedReplayIdRef = useRef<string | null>(null);
  const lastHandledLatestReplayPathRef = useRef<string | null>(null);
  const autoPlaySelectionKeyRef = useRef<string | null>(null);

  const isPageVisible = usePageVisible();

  const checkpointIndexRef = useRef<CheckpointIndex | null>(null);
  checkpointIndexRef.current = checkpointIndex;
  const replayRef = useRef<ReplayBundle | null>(null);
  replayRef.current = replay;
  const scenarioReplayRef = useRef<ReplayBundle | null>(null);
  scenarioReplayRef.current = scenarioReplay;
  const scenarioFrameIndexRef = useRef(0);
  scenarioFrameIndexRef.current = scenarioFrameIndex;
  const selectedCheckpointEpisodeRef = useRef<number | null>(null);
  selectedCheckpointEpisodeRef.current = selectedCheckpointEpisode;
  const selectedSeedRef = useRef<number | null>(null);
  selectedSeedRef.current = selectedSeed;
  const statusPhaseRef = useRef<string | null>(null);
  const bestCheckpointEpisodeRef = useRef<number | null>(null);

  const syncTrainingStageFromStatus = useCallback((status: TrainingStatus | null): void => {
    const reportedStage = status?.curriculum_stage;
    if (reportedStage == null || reportedStage < 1) {
      return;
    }
    manualStageOverrideRef.current = false;
    setTrainingCurriculumStage(reportedStage);
    localStorage.setItem("sheepdog_curriculum_stage", String(reportedStage));
    if (status?.running) {
      setTrainingEpisodes(status.requested_episodes ?? recommendedEpisodesForStage(reportedStage));
    } else {
      setTrainingEpisodes(recommendedEpisodesForStage(reportedStage));
    }
  }, []);

  const playbackDelay = playbackFastMode ? 24 : 220;
  const scenarioPlaybackDelay = scenarioFastMode ? 24 : 220;
  const trainingRunning = trainingStatus?.running ?? false;
  const maxCurriculumStage = Math.max(
    trainingStatus?.max_curriculum_stage ?? 0,
    trainingCurriculumStage,
    DEFAULT_MAX_CURRICULUM_STAGE,
  );

  useEffect(() => {
    if (
      prevTrainingRunning.current &&
      !trainingRunning &&
      trainingStatus?.curriculum_stage != null
    ) {
      syncTrainingStageFromStatus(trainingStatus);
    }
    prevTrainingRunning.current = trainingRunning;
  }, [syncTrainingStageFromStatus, trainingRunning, trainingStatus]);

  const effectiveEnableInstincts = trainingRunning
    ? trainingStatus?.enable_instinct_rewards ?? trainingEnableInstincts
    : trainingEnableInstincts;
  const effectiveCurriculumStage = trainingRunning
    ? trainingStatus?.active_curriculum_stage ??
      trainingStatus?.curriculum_stage ??
      trainingCurriculumStage
    : trainingCurriculumStage;
  const effectiveDebugRewardBreakdown = trainingRunning
    ? trainingStatus?.debug_reward_breakdown ?? trainingDebugRewardBreakdown
    : trainingDebugRewardBreakdown;
  const effectiveAutoPromote = trainingRunning
    ? trainingStatus?.auto_promote ?? trainingAutoPromote
    : trainingAutoPromote;

  const {
    selectedCheckpoint,
    seedOptions,
    latestSuccessRate,
    bestStageFormalEpisode,
    currentReplaySelection,
    latestCheckpointEpisode,
    bestCheckpointEpisode,
  } = useDerivedCheckpointState({
    checkpointIndex,
    selectedCheckpointEpisode,
    effectiveCurriculumStage,
    totalTrainingEpisodes: trainingStatus?.total_episodes_trained ?? 0,
    statusLatestCheckpointEpisode: trainingStatus?.latest_checkpoint_episode ?? null,
  });
  bestCheckpointEpisodeRef.current = bestCheckpointEpisode;

  const currentFrame =
    replay?.frames?.[Math.min(frameIndex, Math.max((replay?.frames.length ?? 0) - 1, 0))] ?? null;
  const snapshot = currentFrame?.snapshot ?? replay?.final_snapshot ?? null;

  const scenarioCurrentFrame =
    scenarioReplay?.frames?.[
      Math.min(scenarioFrameIndex, Math.max((scenarioReplay?.frames.length ?? 0) - 1, 0))
    ] ?? null;
  const scenarioSnapshot =
    scenarioCurrentFrame?.snapshot ?? scenarioReplay?.final_snapshot ?? null;

  const latestCheckpointEpisodeRef = useRef<number | null>(null);
  latestCheckpointEpisodeRef.current = latestCheckpointEpisode;

  const refreshScenarioIndex = useCallback(async () => {
    try {
      const index = await loadScenarioIndex();
      setScenarioIndex(index);
    } catch {
      setScenarioIndex({
        scenarios: [],
        runs: [],
        best_by_scenario: {},
        latest_checkpoint_episode: latestCheckpointEpisodeRef.current,
        latest_runs: [],
      });
    }
  }, []);

  const applyCheckpointIndex = useCallback((index: CheckpointIndex | null) => {
    setCheckpointIndex((prev) => {
      if (!prev && !index) return null;
      if (prev && index && prev.checkpoints.length === index.checkpoints.length) {
        const prevLast = prev.checkpoints[prev.checkpoints.length - 1];
        const indexLast = index.checkpoints[index.checkpoints.length - 1];
        const prevLastId =
          prevLast?.checkpoint_id ?? prevLast?.global_timestep ?? prevLast?.checkpoint_episode;
        const indexLastId =
          indexLast?.checkpoint_id ?? indexLast?.global_timestep ?? indexLast?.checkpoint_episode;
        if (prevLastId === indexLastId) {
          return prev;
        }
      }
      return index;
    });
    if (!index) {
      lastLoadedSelectionKeyRef.current = null;
      lastLoadedReplayIdRef.current = null;
      lastHandledLatestReplayPathRef.current = null;
      autoPlaySelectionKeyRef.current = null;
      setReplay(null);
      setSelectedCheckpointEpisode(null);
      setSelectedSeed(11);
      setRunState("idle");
      setFrameIndex(0);
      return;
    }
    const latestCheckpoint = index.checkpoints[index.checkpoints.length - 1] ?? null;
    const checkpointEpisode =
      latestCheckpoint?.checkpoint_episode ?? index.latest?.checkpoint_episode ?? null;
    setSelectedCheckpointEpisode(checkpointEpisode);
    const seed =
      latestCheckpoint?.records?.[0]?.seed ??
      index.latest?.records?.[0]?.seed ??
      DEFAULT_EVAL_SEEDS[0];
    setSelectedSeed(seed);
  }, []);

  const hasTrainingStatus = trainingStatus !== null;
  const statusRunning = trainingStatus?.running ?? false;
  const statusPhase = trainingStatus?.phase ?? null;
  statusPhaseRef.current = statusPhase;
  const statusError = trainingStatus?.error ?? null;
  const statusLatestCpEp = trainingStatus?.latest_checkpoint_episode ?? null;
  const statusActiveCpId =
    trainingStatus?.active_checkpoint_id ?? trainingStatus?.active_policy_identity ?? null;
  const latestReplayPath = trainingStatus?.latest_replay_path ?? null;
  const statusReadOnlyPreview = Boolean(
    healthReadOnlyPreview ||
      trainingStatus?.read_only_preview ||
      trainingStatus?.mode === "read_only_preview" ||
      trainingStatus?.phase === "read_only_preview",
  );

  useBackendHealthProbe({
    onHealthReadOnlyPreview: setHealthReadOnlyPreview,
    onHealthCheckDone: setHealthCheckDone,
  });

  useEffect(() => {
    if (hasTrainingStatus && loading) {
      setLoading(false);
    }
  }, [hasTrainingStatus, loading]);

  const clearError = useCallback(() => setError(null), []);

  const { syncIndexMeta, markIndexCleared } = useCheckpointIndexSync({
    hasTrainingStatus,
    statusRunning,
    statusPhase,
    statusError,
    statusLatestCpEp,
    statusActiveCpId,
    onIndexApplied: applyCheckpointIndex,
    onRefreshScenarios: refreshScenarioIndex,
    onClearError: clearError,
  });

  useProgressInsightsPolling({
    active:
      activeTab === "progress" &&
      hasTrainingStatus &&
      statusPhase !== "offline" &&
      statusPhase !== "restoring",
    statusRunning,
    statusLatestCpEp,
    statusActiveCpId,
    onEpisodesReceived: setTrainingEpisodeLogs,
    onSessionEventsReceived: setSessionEvents,
  });

  const handleExpandNetwork = useCallback(async () => {
    if (statusPhaseRef.current === "offline" || statusPhaseRef.current === "restoring") {
      return;
    }
    try {
      const topology = await loadNetworkTopology();
      setNetworkTopology(topology);
    } catch {
      setNetworkTopology(null);
    }
  }, []);

  useBackendConfigLoader({
    hasTrainingStatus,
    statusPhase,
    onPersonalityStrength: setScenarioPersonalityStrength,
    onEffectiveConfig: setEffectiveConfig,
  });

  useTrainingStatusPolling({
    activePollDelayMs: ACTIVE_POLL_DELAY_MS,
    statusMutationSeqRef,
    manualStageOverrideRef,
    onStatusReceived: setTrainingStatus,
    syncTrainingStageFromStatus,
  });

  const handleReplayLoaded = useCallback((bundle: ReplayBundle, nextRunState: RunState) => {
    setReplay(bundle);
    setFrameIndex(0);
    setRunState(nextRunState);
  }, []);

  const handleReplayError = useCallback((message: string) => {
    setError(message);
  }, []);

  useSelectedReplaySync({
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
    onReplayLoaded: handleReplayLoaded,
    onError: handleReplayError,
  });

  const handleMainReplayComplete = useCallback((finalSnapshot: ReplaySnapshot | null) => {
    setRunState(resolveRunState(finalSnapshot, "idle"));
  }, []);

  const handleScenarioReplayComplete = useCallback((finalSnapshot: ReplaySnapshot | null) => {
    setScenarioRunState(resolveRunState(finalSnapshot, "idle"));
  }, []);

  useReplayPlayback({
    active: runState === "running",
    framesLength: replay?.frames?.length ?? 0,
    delayMs: playbackDelay,
    replayRef,
    onStep: setFrameIndex,
    onComplete: handleMainReplayComplete,
  });

  useReplayPlayback({
    active: scenarioRunState === "running",
    framesLength: scenarioReplay?.frames?.length ?? 0,
    delayMs: scenarioPlaybackDelay,
    replayRef: scenarioReplayRef,
    onStep: setScenarioFrameIndex,
    onComplete: handleScenarioReplayComplete,
  });

  async function handleCurriculumStageChange(nextStage: number) {
    const normalized = Number.isFinite(nextStage) ? Math.floor(nextStage) : 0;
    const clamped = Math.max(0, Math.min(maxCurriculumStage, normalized));

    if (clamped < effectiveCurriculumStage && !trainingRunning) {
      const serverActiveStage = trainingStatus?.curriculum_stage ?? 1;
      if (clamped < serverActiveStage) {
        if (statusReadOnlyPreview) {
          return;
        }
        const shouldRewind = window.confirm(
          `Restart from Stage ${clamped} and remove saved progress from higher stages? ` +
            "This hides those stages from the UI and prevents training from resuming from them.",
        );

        if (shouldRewind) {
          statusMutationSeqRef.current += 1;
          setTrainingError(null);
          setError(null);
          try {
            const status = await rewindTraining(clamped);
            statusMutationSeqRef.current += 1;
            setTrainingStatus(status);
            setPromoteFromEpisode(null);
            manualStageOverrideRef.current = true;
            setTrainingCurriculumStage(clamped);
            localStorage.setItem("sheepdog_curriculum_stage", String(clamped));
            setTrainingEpisodes(recommendedEpisodesForStage(clamped));
            lastLoadedSelectionKeyRef.current = null;
            lastLoadedReplayIdRef.current = null;
            lastHandledLatestReplayPathRef.current = null;
            markIndexCleared();
            const index = await loadCheckpointIndex();
            applyCheckpointIndex(index);
            syncIndexMeta(index);
            await refreshScenarioIndex();
            return;
          } catch (rewindError) {
            statusMutationSeqRef.current += 1;
            setTrainingError(
              rewindError instanceof Error
                ? rewindError.message
                : "Unable to rewind training state.",
            );
            return;
          }
        }
      }
    }

    statusMutationSeqRef.current += 1;
    manualStageOverrideRef.current = true;
    setTrainingCurriculumStage(clamped);
    localStorage.setItem("sheepdog_curriculum_stage", String(clamped));
    setTrainingEpisodes(recommendedEpisodesForStage(clamped));
  }

  function handleCheckpointChange(episode: number) {
    autoPlaySelectionKeyRef.current = null;
    lastLoadedSelectionKeyRef.current = null;
    lastLoadedReplayIdRef.current = null;
    if (trainingStatus?.latest_replay_path) {
      lastHandledLatestReplayPathRef.current = trainingStatus.latest_replay_path;
    }
    setSelectedCheckpointEpisode(episode);
    const checkpoint = checkpointIndexRef.current?.checkpoints.find(
      (entry) => entry.checkpoint_episode === episode,
    );
    const seed = checkpoint?.records?.[0]?.seed ?? DEFAULT_EVAL_SEEDS[0];
    setSelectedSeed(seed);
  }

  function handleSeedChange(seed: number) {
    autoPlaySelectionKeyRef.current = null;
    lastLoadedSelectionKeyRef.current = null;
    lastLoadedReplayIdRef.current = null;
    if (trainingStatus?.latest_replay_path) {
      lastHandledLatestReplayPathRef.current = trainingStatus.latest_replay_path;
    }
    setSelectedSeed(seed);
  }

  function handleSelectSeedAndReplay(seed: number) {
    const currentCheckpoints = checkpointIndexRef.current?.checkpoints ?? [];
    const currentBestEpisode = bestCheckpointEpisodeRef.current;
    const currentSelectedEpisode = selectedCheckpointEpisodeRef.current;
    const currentSelectedSeed = selectedSeedRef.current;
    const currentReplay = replayRef.current;

    const fallbackBest =
      currentBestEpisode != null
        ? currentCheckpoints.find((entry) => entry.checkpoint_episode === currentBestEpisode) ??
          null
        : null;
    const selectedCp =
      currentSelectedEpisode != null
        ? currentCheckpoints.find((entry) => entry.checkpoint_episode === currentSelectedEpisode) ??
          null
        : null;
    const targetCheckpoint =
      selectedCp ??
      fallbackBest ??
      currentCheckpoints[currentCheckpoints.length - 1] ??
      null;

    if (!targetCheckpoint) {
      setSelectedSeed(seed);
      return;
    }

    const targetEpisode = targetCheckpoint.checkpoint_episode;
    const targetKey = `${targetEpisode}:${seed}`;

    if (
      currentSelectedEpisode === targetEpisode &&
      currentSelectedSeed === seed &&
      currentReplay &&
      lastLoadedSelectionKeyRef.current === targetKey
    ) {
      setFrameIndex(0);
      setRunState(
        (currentReplay.frames?.length ?? 0) > 0
          ? "running"
          : resolveRunState(currentReplay.final_snapshot, "idle"),
      );
      return;
    }

    autoPlaySelectionKeyRef.current = targetKey;
    lastLoadedSelectionKeyRef.current = null;
    lastLoadedReplayIdRef.current = null;
    if (trainingStatus?.latest_replay_path) {
      lastHandledLatestReplayPathRef.current = trainingStatus.latest_replay_path;
    }
    if (currentSelectedEpisode !== targetEpisode) {
      setSelectedCheckpointEpisode(targetEpisode);
    }
    if (currentSelectedSeed !== seed) {
      setSelectedSeed(seed);
    } else {
      setReplayReloadNonce((n) => n + 1);
    }
  }

  function handlePromote() {
    statusMutationSeqRef.current += 1;
    setPromoteFromEpisode(bestStageFormalEpisode);
    manualStageOverrideRef.current = true;
    setTrainingCurriculumStage((prev) => {
      const next = Math.min(prev + 1, maxCurriculumStage);
      if (next === prev) {
        return prev;
      }
      localStorage.setItem("sheepdog_curriculum_stage", String(next));
      setTrainingEpisodes(recommendedEpisodesForStage(next));
      return next;
    });
  }

  async function handleStartTraining() {
    if (statusReadOnlyPreview) return;
    const trimmedStageSpec = trainingStageSpec.trim();
    const trimmedPassLabel = trainingPassLabel.trim();
    if (trimmedStageSpec.length > 0) {
      const parsed = parseStageSpec(trimmedStageSpec, 1, Math.max(1, maxCurriculumStage));
      if (parsed.error) {
        setTrainingError(parsed.error);
        return;
      }
    }
    statusMutationSeqRef.current += 1;
    setTrainingError(null);
    setError(null);
    setIsStartingTraining(true);
    try {
      const activeStage = trainingStatus?.curriculum_stage;
      let status: TrainingStatus;
      if (trimmedStageSpec.length > 0) {
        status = await startTraining({
          episodes: trainingEpisodes,
          fast_mode: trainingFastMode,
          enable_instinct_rewards: trainingEnableInstincts,
          debug_reward_breakdown: trainingDebugRewardBreakdown,
          stage_spec: trimmedStageSpec,
          ...(trimmedPassLabel ? { pass_label: trimmedPassLabel } : {}),
          starting_model_source: startingModelSource,
          continuous: true,
          auto_promote: true,
        });
      } else if (activeStage != null && trainingCurriculumStage !== activeStage) {
        status = await crossStageFork({
          target_stage: trainingCurriculumStage,
          starting_model_source: startingModelSource,
        });
      } else {
        status = await startTraining({
          episodes: trainingEpisodes,
          fast_mode: trainingFastMode,
          enable_instinct_rewards: trainingEnableInstincts,
          curriculum_stage: trainingCurriculumStage,
          debug_reward_breakdown: trainingDebugRewardBreakdown,
          auto_promote: trainingContinuous || trainingAutoPromote,
          ...(trainingContinuous ? { continuous: true } : {}),
          promote_from_checkpoint_episode: promoteFromEpisode ?? undefined,
        });
      }
      statusMutationSeqRef.current += 1;
      setPromoteFromEpisode(null);
      setTrainingStatus(status);
      syncTrainingStageFromStatus(status);
    } catch (startError) {
      statusMutationSeqRef.current += 1;
      setTrainingError(
        startError instanceof Error ? startError.message : "Unable to start training.",
      );
    } finally {
      setIsStartingTraining(false);
    }
  }

  async function handlePauseTraining() {
    if (lifecycleAction !== null || statusReadOnlyPreview) return;
    statusMutationSeqRef.current += 1;
    setTrainingError(null);
    setError(null);
    setLifecycleAction("pause");
    try {
      const status = await pauseTraining();
      statusMutationSeqRef.current += 1;
      setTrainingStatus(status);
    } catch (pauseError) {
      statusMutationSeqRef.current += 1;
      setTrainingError(
        pauseError instanceof Error ? pauseError.message : "Unable to pause training.",
      );
    } finally {
      setLifecycleAction(null);
    }
  }

  async function handleStopTraining() {
    if (lifecycleAction !== null || statusReadOnlyPreview) return;
    statusMutationSeqRef.current += 1;
    setTrainingError(null);
    setError(null);
    setLifecycleAction("stop");
    try {
      const status = await stopTraining();
      statusMutationSeqRef.current += 1;
      setTrainingStatus(status);
    } catch (stopError) {
      statusMutationSeqRef.current += 1;
      setTrainingError(
        stopError instanceof Error ? stopError.message : "Unable to stop training.",
      );
    } finally {
      setLifecycleAction(null);
    }
  }

  async function handleCloseApp() {
    if (statusReadOnlyPreview) return;
    if (
      !window.confirm(
        "Gracefully stop active training (if running) and shutdown the application server?",
      )
    ) {
      return;
    }
    setTrainingError(null);
    setError(null);
    try {
      await shutdownApp();
      setIsAppClosed(true);
    } catch {
      setIsAppClosed(true);
    }
  }

  async function handleResumeTraining() {
    if (statusReadOnlyPreview) return;
    const request = trainingStatus?.resume_request;
    const remainingEpisodes = trainingStatus?.resume_remaining_episodes ?? 0;
    const hasStageQueue = (request?.stage_queue?.length ?? 0) > 0;
    if (!request || (!request.continuous && !hasStageQueue && remainingEpisodes <= 0)) {
      setTrainingError("No resumable training session is available.");
      return;
    }

    statusMutationSeqRef.current += 1;
    setTrainingError(null);
    setError(null);
    setIsStartingTraining(true);
    try {
      const status = await startTraining(
        request.continuous || hasStageQueue
          ? { ...request, resume: true }
          : { ...request, episodes: remainingEpisodes },
      );
      statusMutationSeqRef.current += 1;
      setPromoteFromEpisode(null);
      setTrainingStatus(status);
    } catch (resumeError) {
      statusMutationSeqRef.current += 1;
      setTrainingError(
        resumeError instanceof Error ? resumeError.message : "Unable to resume training.",
      );
    } finally {
      setIsStartingTraining(false);
    }
  }

  async function handleStartRemediation() {
    if (statusReadOnlyPreview) return;
    statusMutationSeqRef.current += 1;
    setTrainingError(null);
    setError(null);
    setIsStartingTraining(true);
    try {
      const status = await startRemediationFork(9, 20);
      statusMutationSeqRef.current += 1;
      setTrainingStatus(status);
      syncTrainingStageFromStatus(status);
    } catch (remediationError) {
      statusMutationSeqRef.current += 1;
      setTrainingError(
        remediationError instanceof Error
          ? remediationError.message
          : "Unable to create Stage 9 remediation run.",
      );
    } finally {
      setIsStartingTraining(false);
    }
  }

  async function handleRefreshAllData() {
    try {
      console.debug("[Sheepdog API] Refreshing all data...");
      const status = await loadTrainingStatus();
      setTrainingStatus(status);
      if (status && status.phase !== "offline") {
        const index = await loadCheckpointIndex();
        applyCheckpointIndex(index);
        syncIndexMeta(index);
      }
    } catch (err) {
      console.error("Failed to refresh training data:", err);
    }
  }

  async function handleClearTraining() {
    if (statusReadOnlyPreview) return;
    if (
      !window.confirm(
        "Permanently delete all checkpoints, evaluations, training state, AND ALL ARCHIVED JOURNEY HISTORY? This action is destructive and cannot be undone.",
      )
    ) {
      return;
    }

    statusMutationSeqRef.current += 1;
    setClearingTraining(true);
    setTrainingError(null);
    setError(null);
    try {
      const status = await clearTraining();
      statusMutationSeqRef.current += 1;
      lastLoadedSelectionKeyRef.current = null;
      lastLoadedReplayIdRef.current = null;
      lastHandledLatestReplayPathRef.current = null;
      autoPlaySelectionKeyRef.current = null;
      markIndexCleared();
      const index = await loadCheckpointIndex();
      setTrainingStatus(status);
      setTrainingCurriculumStage(1);
      setStartingModelSource("fresh");
      localStorage.setItem("sheepdog_curriculum_stage", "1");
      setTrainingEpisodes(recommendedEpisodesForStage(1));
      setPromoteFromEpisode(null);
      applyCheckpointIndex(index);
      syncIndexMeta(index);
      const latestCheckpoint = index?.checkpoints[index.checkpoints.length - 1] ?? null;
      const seed = latestCheckpoint?.records?.[0]?.seed ?? null;
      if (latestCheckpoint && seed !== null) {
        const record =
          latestCheckpoint.records?.find((entry) => entry.seed === seed) ??
          latestCheckpoint.records?.[0];
        if (record && record.replay_path) {
          const bundle = await loadReplay(record.replay_path);
          lastLoadedSelectionKeyRef.current = `${latestCheckpoint.checkpoint_episode}:${seed}`;
          lastLoadedReplayIdRef.current = extractReplayId(record.replay_path);
          setReplay(bundle);
        }
      } else {
        setReplay(null);
      }
      setFrameIndex(0);
      setRunState("idle");
    } catch (clearError) {
      statusMutationSeqRef.current += 1;
      setTrainingError(
        clearError instanceof Error ? clearError.message : "Unable to clear training.",
      );
    } finally {
      setClearingTraining(false);
    }
  }

  async function handleResetJourney() {
    if (statusReadOnlyPreview) return;
    if (
      !window.confirm(
        "Archive your current training run/journey to the history log and start a fresh training journey from Stage 1? (Your current progress will be preserved in the archive).",
      )
    ) {
      return;
    }

    statusMutationSeqRef.current += 1;
    setClearingTraining(true);
    setTrainingError(null);
    setError(null);
    try {
      const status = await resetJourneyTraining();
      statusMutationSeqRef.current += 1;
      lastLoadedSelectionKeyRef.current = null;
      lastLoadedReplayIdRef.current = null;
      lastHandledLatestReplayPathRef.current = null;
      autoPlaySelectionKeyRef.current = null;
      markIndexCleared();
      const index = await loadCheckpointIndex();
      setTrainingStatus(status);
      setTrainingCurriculumStage(1);
      setStartingModelSource("fresh");
      localStorage.setItem("sheepdog_curriculum_stage", "1");
      setTrainingEpisodes(recommendedEpisodesForStage(1));
      setPromoteFromEpisode(null);
      applyCheckpointIndex(index);
      syncIndexMeta(index);
      setFrameIndex(0);
      setRunState("idle");
    } catch (resetError) {
      statusMutationSeqRef.current += 1;
      setTrainingError(
        resetError instanceof Error ? resetError.message : "Unable to reset training journey.",
      );
    } finally {
      setClearingTraining(false);
    }
  }

  const handleWatchReplay = useCallback((episode: number, preferredSeed?: number) => {
    const checkpoint = checkpointIndexRef.current?.checkpoints.find(
      (entry) => entry.checkpoint_episode === episode,
    );
    const seed =
      preferredSeed ?? checkpoint?.records?.[0]?.seed ?? DEFAULT_EVAL_SEEDS[0];
    const targetKey = `${episode}:${seed}`;
    const currentReplay = replayRef.current;
    if (
      selectedCheckpointEpisodeRef.current === episode &&
      selectedSeedRef.current === seed &&
      currentReplay &&
      lastLoadedSelectionKeyRef.current === targetKey
    ) {
      setFrameIndex(0);
      setRunState((currentReplay.frames?.length ?? 0) > 0 ? "running" : "idle");
    } else {
      autoPlaySelectionKeyRef.current = targetKey;
      lastLoadedSelectionKeyRef.current = null;
      lastLoadedReplayIdRef.current = null;
      setSelectedCheckpointEpisode(episode);
      setSelectedSeed(seed);
    }
    setActiveTab("train");
  }, []);

  function policySelectionForCheckpoint(episode: number | null) {
    const entry =
      checkpointIndexRef.current?.checkpoints.find(
        (checkpoint) => checkpoint.checkpoint_episode === episode,
      ) ?? null;
    if (!entry) {
      return currentReplaySelection;
    }
    const policyMode =
      entry.policy_mode === "neural_policy" || entry.policy_name === "neural_policy"
        ? "neural_policy"
        : entry.policy_mode === "trained_policy" || entry.policy_name === "trained_policy"
          ? "trained_policy"
          : currentReplaySelection.policyMode;
    return {
      checkpointEpisode: entry.checkpoint_episode,
      trainerType: entry.trainer_type ?? currentReplaySelection.trainerType,
      policyType: entry.policy_type ?? currentReplaySelection.policyType,
      policyMode,
    };
  }

  function buildScenarioCheckpointRequest() {
    const targetEpisode =
      scenarioCheckpointMode === "specific"
        ? specificScenarioCheckpointEpisode
        : latestCheckpointEpisode;
    const selection = policySelectionForCheckpoint(targetEpisode);
    return {
      checkpoint_mode: scenarioCheckpointMode,
      checkpoint_episode:
        scenarioCheckpointMode === "specific"
          ? specificScenarioCheckpointEpisode ?? undefined
          : undefined,
      policy_mode: selection.policyMode,
      trainer_type: selection.trainerType,
      policy_type: selection.policyType,
      effective_config: {
        enable_instinct_rewards: effectiveEnableInstincts,
        curriculum_stage: effectiveCurriculumStage,
        debug_reward_breakdown: effectiveDebugRewardBreakdown,
      },
    };
  }

  async function handleSaveScenario() {
    if (statusReadOnlyPreview) return;
    const currentScenarioReplay = scenarioReplayRef.current;
    const frames = currentScenarioReplay?.frames ?? [];
    const frameSnap =
      frames[Math.min(scenarioFrameIndexRef.current, Math.max(frames.length - 1, 0))]?.snapshot ??
      null;
    const layoutSnapshot =
      saveSnapshotSource === "initial"
        ? frames[0]?.snapshot ?? frameSnap ?? currentScenarioReplay?.final_snapshot ?? null
        : frameSnap ?? currentScenarioReplay?.final_snapshot ?? null;
    if (!layoutSnapshot) {
      setError("Run a live or saved scenario first, then save its layout.");
      return;
    }
    const name = window.prompt("Scenario name", `Hard case seed ${scenarioSeed}`);
    if (!name?.trim()) {
      return;
    }
    const description = window.prompt("Description (optional)", "") ?? "";
    setError(null);
    try {
      await saveScenario({
        name: name.trim(),
        seed: scenarioSeed,
        snapshot: layoutSnapshot,
        sheep_personality_strength: scenarioPersonalityStrength,
        description: description.trim(),
        snapshot_source: saveSnapshotSource,
      });
      await refreshScenarioIndex();
    } catch (saveError) {
      setError(saveError instanceof Error ? saveError.message : "Unable to save scenario.");
    }
  }

  async function handleEvaluateScenarios() {
    if (statusReadOnlyPreview) return;
    if (!selectedScenarioId) {
      setError("Select a saved scenario to evaluate.");
      return;
    }
    setEvaluatingScenarios(true);
    setError(null);
    try {
      const response = await evaluateScenario(selectedScenarioId, buildScenarioCheckpointRequest());
      setScenarioIndex(response.index);
      if (response.result.replay_path) {
        const bundle = await loadReplay(response.result.replay_path);
        setScenarioReplay(bundle);
        setScenarioFrameIndex(0);
        setScenarioRunState("running");
      }
    } catch (evalError) {
      setError(evalError instanceof Error ? evalError.message : "Unable to evaluate scenario.");
    } finally {
      setEvaluatingScenarios(false);
    }
  }

  async function handleRunSavedScenario() {
    if (statusReadOnlyPreview) return;
    if (!selectedScenarioId) {
      setError("Select a saved scenario.");
      return;
    }
    setRunningScenario(true);
    setError(null);
    try {
      const bundle = await replayScenario(selectedScenarioId, buildScenarioCheckpointRequest());
      setScenarioReplay(bundle);
      setScenarioFrameIndex(0);
      setScenarioRunState("running");
    } catch (runError) {
      setError(
        runError instanceof Error ? runError.message : "Unable to replay the saved scenario.",
      );
    } finally {
      setRunningScenario(false);
    }
  }

  async function handleRunScenario() {
    if (statusReadOnlyPreview) return;
    setRunningScenario(true);
    setError(null);
    setTrainingError(null);
    try {
      const bundle = await runReplay({
        seed: scenarioSeed,
        checkpoint_episode: currentReplaySelection.checkpointEpisode,
        trainer_type: currentReplaySelection.trainerType,
        policy_type: currentReplaySelection.policyType,
        policy_mode: currentReplaySelection.policyMode,
        effective_config: {
          enable_instinct_rewards: effectiveEnableInstincts,
          curriculum_stage: effectiveCurriculumStage,
          debug_reward_breakdown: effectiveDebugRewardBreakdown,
        },
        environment_overrides: {
          sheep_personality_strength: scenarioPersonalityStrength,
        },
      });
      setScenarioReplay(bundle);
      setScenarioSeed(bundle.seed);
      setScenarioFrameIndex(0);
      setScenarioRunState("running");
    } catch (runError) {
      setError(runError instanceof Error ? runError.message : "Unable to run the scenario.");
    } finally {
      setRunningScenario(false);
    }
  }

  function handleRestartScenarioPlayback() {
    if (!scenarioReplayRef.current) {
      return;
    }
    setScenarioFrameIndex(0);
    setScenarioRunState("running");
  }

  function handleEndScenarioEpisode() {
    const currentScenarioReplay = scenarioReplayRef.current;
    if (!currentScenarioReplay) {
      return;
    }
    setScenarioFrameIndex(Math.max(currentScenarioReplay.frames.length - 1, 0));
    setScenarioRunState(resolveRunState(currentScenarioReplay.final_snapshot, "idle"));
  }

  async function handleRunCurrentReplay() {
    if (statusReadOnlyPreview) return;
    setRunningCurrentReplay(true);
    setError(null);
    setTrainingError(null);
    try {
      const bundle = await runReplay({
        seed: selectedSeedRef.current ?? 11,
        checkpoint_episode: currentReplaySelection.checkpointEpisode,
        trainer_type: currentReplaySelection.trainerType,
        policy_type: currentReplaySelection.policyType,
        policy_mode: currentReplaySelection.policyMode,
        effective_config: {
          enable_instinct_rewards: effectiveEnableInstincts,
          curriculum_stage: effectiveCurriculumStage,
          debug_reward_breakdown: effectiveDebugRewardBreakdown,
        },
      });
      lastLoadedSelectionKeyRef.current = null;
      lastLoadedReplayIdRef.current = null;
      if (trainingStatus?.latest_replay_path) {
        lastHandledLatestReplayPathRef.current = trainingStatus.latest_replay_path;
      }
      autoPlaySelectionKeyRef.current = null;
      setReplay(bundle);
      setSelectedCheckpointEpisode(null);
      setSelectedSeed(bundle.seed);
      setFrameIndex(0);
      setRunState("running");
    } catch (runError) {
      setError(
        runError instanceof Error ? runError.message : "Unable to run the current dog team.",
      );
    } finally {
      setRunningCurrentReplay(false);
    }
  }

  async function handleReplaySelected() {
    const currentEp = selectedCheckpointEpisodeRef.current;
    const checkpoint = checkpointIndexRef.current?.checkpoints.find(
      (entry) => entry.checkpoint_episode === currentEp,
    );
    const targetSeed =
      selectedSeedRef.current ?? checkpoint?.records?.[0]?.seed ?? DEFAULT_EVAL_SEEDS[0];
    const record = checkpoint?.records?.find((r) => r.seed === targetSeed);

    setLoadingSelectedReplay(true);
    setError(null);
    try {
      let bundle: ReplayBundle;
      if (record?.replay_path) {
        bundle = await loadReplay(record.replay_path);
      } else if (checkpoint && targetSeed !== null && !statusReadOnlyPreview) {
        const checkpointStage = getCheckpointStage(checkpoint);
        bundle = await runReplay({
          seed: targetSeed,
          checkpoint_episode: checkpoint.checkpoint_episode,
          trainer_type: checkpoint.trainer_type,
          policy_type: checkpoint.policy_type,
          policy_mode: checkpoint.policy_mode || checkpoint.policy_name,
          effective_config: {
            enable_instinct_rewards: effectiveEnableInstincts,
            curriculum_stage: checkpointStage >= 0 ? checkpointStage : effectiveCurriculumStage,
            debug_reward_breakdown: effectiveDebugRewardBreakdown,
          },
        });
      } else {
        setFrameIndex(0);
        setRunState("running");
        return;
      }
      if (checkpoint && targetSeed !== null) {
        lastLoadedSelectionKeyRef.current = `${checkpoint.checkpoint_episode}:${targetSeed}`;
        lastLoadedReplayIdRef.current = extractReplayId(record?.replay_path);
      }
      if (trainingStatus?.latest_replay_path) {
        lastHandledLatestReplayPathRef.current = trainingStatus.latest_replay_path;
      }
      setReplay(bundle);
      setSelectedSeed(targetSeed);
      setFrameIndex(0);
      setRunState("running");
    } catch (loadError) {
      setError(loadError instanceof Error ? loadError.message : "Unable to load replay.");
    } finally {
      setLoadingSelectedReplay(false);
    }
  }

  function handleEndEpisode() {
    const currentReplay = replayRef.current;
    if (!currentReplay) {
      return;
    }
    setFrameIndex(Math.max(currentReplay.frames.length - 1, 0));
    setRunState(resolveRunState(currentReplay.final_snapshot, "idle"));
  }

  function handleTogglePlay() {
    const currentReplay = replayRef.current;
    if (!currentReplay?.frames?.length) {
      return;
    }
    if (runState === "running") {
      setRunState("paused");
      return;
    }
    if (frameIndex >= currentReplay.frames.length - 1) {
      setFrameIndex(0);
    }
    setRunState("running");
  }

  function handleStepFrame(delta: number) {
    const currentReplay = replayRef.current;
    if (!currentReplay?.frames?.length) {
      return;
    }
    setRunState("paused");
    setFrameIndex((current) =>
      Math.max(0, Math.min(currentReplay.frames.length - 1, current + delta)),
    );
  }

  function handleScrubFrame(nextIndex: number) {
    const currentReplay = replayRef.current;
    if (!currentReplay?.frames?.length) {
      return;
    }
    setRunState("paused");
    setFrameIndex(Math.max(0, Math.min(currentReplay.frames.length - 1, nextIndex)));
  }

  function handleOpenCurrentReplayInInspector() {
    const currentEp = selectedCheckpointEpisodeRef.current;
    const currentSeed = selectedSeedRef.current;
    const cp = checkpointIndexRef.current?.checkpoints.find(
      (entry) => entry.checkpoint_episode === currentEp,
    );
    const record = cp?.records?.find((r) => r.seed === currentSeed) ?? cp?.records?.[0];
    const fromPath = extractReplayId(record?.replay_path);
    if (fromPath) {
      setTemporalReplayId(fromPath);
    } else if (currentEp != null && currentSeed != null) {
      setTemporalReplayId(
        `checkpoint-${String(currentEp).padStart(6, "0")}-seed-${String(currentSeed).padStart(6, "0")}`,
      );
    }
    setTroubleMode("inspector");
    setActiveTab("trouble");
  }

  const statusLabel = resolveRunState(snapshot, runState);
  const canEndEpisode =
    Boolean(replay) && (runState === "running" || frameIndex < (replay?.frames.length ?? 0) - 1);
  const scenarioStatusLabel = resolveRunState(
    scenarioCurrentFrame?.snapshot ?? scenarioReplay?.final_snapshot ?? null,
    scenarioRunState,
  );
  const canEndScenarioEpisode =
    Boolean(scenarioReplay) &&
    (scenarioRunState === "running" ||
      scenarioFrameIndex < (scenarioReplay?.frames.length ?? 0) - 1);
  const scenarioPolicyLabel =
    scenarioReplay?.policy_name === "neural_policy"
      ? "Neural PPO"
      : scenarioReplay?.policy_name === "trained_policy"
        ? "Trained linear"
        : currentReplaySelection.policyMode === "neural_policy"
          ? "Neural PPO (best)"
          : currentReplaySelection.policyMode === "trained_policy"
            ? "Trained linear (best)"
            : "Instinct only";

  const cfgEnvironment = (effectiveConfig?.environment as Record<string, unknown> | undefined) ?? undefined;
  const cfgDogs = typeof cfgEnvironment?.dogs === "number" ? cfgEnvironment.dogs : null;
  const cfgSheep = typeof cfgEnvironment?.sheep === "number" ? cfgEnvironment.sheep : null;
  const activeEnvDogs =
    snapshot?.dogs?.length ??
    replay?.environment?.dogs ??
    selectedCheckpoint?.environment_config?.dogs ??
    cfgDogs;
  const activeEnvSheep =
    snapshot?.sheep?.length ??
    replay?.environment?.sheep ??
    selectedCheckpoint?.environment_config?.sheep ??
    cfgSheep;
  const topbarStageLabel =
    maxCurriculumStage > 0
      ? `Stage ${effectiveCurriculumStage} / ${maxCurriculumStage}`
      : `Stage ${effectiveCurriculumStage}`;
  const topbarEnvLabel =
    activeEnvDogs != null && activeEnvSheep != null && activeEnvDogs > 0 && activeEnvSheep > 0
      ? `· ${activeEnvDogs}d / ${activeEnvSheep}s`
      : null;

  const isEvaluating = Boolean(
    trainingStatus?.evaluation_in_progress ||
      trainingStatus?.evaluation_status?.in_progress ||
      statusPhase === "evaluation",
  );
  const isPaused =
    statusPhase === "paused" ||
    Boolean(trainingStatus?.resume_available && (trainingStatus?.resume_remaining_episodes ?? 0) > 0);
  const batchCompleted =
    trainingStatus?.batch_completed_episodes ?? trainingStatus?.completed_episodes ?? 0;
  const batchTotal =
    trainingStatus?.batch_total_episodes ?? trainingStatus?.requested_episodes ?? trainingEpisodes;

  let topbarStatusLabel = "Ready";
  let topbarStatusTone = "ui-badge--pass";
  if (statusPhase === "offline") {
    topbarStatusLabel = "Backend disconnected";
    topbarStatusTone = "ui-badge--fail";
  } else if (statusPhase === "restoring") {
    topbarStatusLabel = "Restoring state";
    topbarStatusTone = "ui-badge--warn";
  } else if (statusReadOnlyPreview) {
    topbarStatusLabel = "Read-only preview";
    topbarStatusTone = "ui-badge--warn";
  } else if (isEvaluating) {
    topbarStatusLabel = "Evaluating";
    topbarStatusTone = "ui-badge--accent";
  } else if (statusRunning) {
    const statusMsg = trainingStatus?.message ?? "";
    const tsMatch = statusMsg.match(TIMESTEP_MESSAGE_RE);
    const segMsgMatch = statusMsg.match(SEGMENT_MESSAGE_RE);
    let segPct: number | null = null;
    if (tsMatch?.[3] != null) {
      segPct = Math.min(100, Math.max(0, Number(tsMatch[3])));
    } else if (tsMatch && Number(tsMatch[2]) > 0) {
      segPct = Math.min(100, Math.max(0, Math.round((Number(tsMatch[1]) / Number(tsMatch[2])) * 100)));
    } else if ((trainingStatus?.batch_completed_segments ?? 0) % 1 > 0) {
      segPct = Math.round(((trainingStatus?.batch_completed_segments ?? 0) % 1) * 100);
    }
    const liveEp =
      trainingStatus?.latest_completed_environment_episode ??
      trainingStatus?.current_episode ??
      null;
    const totalSegs =
      trainingStatus?.batch_total_segments ?? (segMsgMatch ? Number(segMsgMatch[2]) : 0);

    if (segPct != null && totalSegs > 1) {
      const segDone = Math.max(
        Math.floor(trainingStatus?.batch_completed_segments ?? 0),
        segMsgMatch ? Math.max(0, Number(segMsgMatch[1]) - 1) : 0,
      );
      topbarStatusLabel = `Training · Ckpt ${segDone}/${totalSegs} (${segPct}%)${
        liveEp != null ? ` · Ep #${liveEp.toLocaleString()}` : ""
      }`;
    } else if (liveEp != null) {
      topbarStatusLabel = `Training · ${Math.floor(batchCompleted)}/${batchTotal} ep · Ep #${liveEp.toLocaleString()}`;
    } else {
      topbarStatusLabel = `Training · ${Math.floor(batchCompleted)}/${batchTotal} ep`;
    }
    topbarStatusTone = "ui-badge--pass";
  } else if (isPaused) {
    topbarStatusLabel = "Paused";
    topbarStatusTone = "ui-badge--warn";
  }

  const stageScopedLatestSuccess =
    latestSuccessRate ??
    (effectiveCurriculumStage === trainingStatus?.curriculum_stage
      ? trainingStatus?.latest_success_rate ?? null
      : null);

  if (isAppClosed) {
    return (
      <div className="ui-closed-screen">
        <div className="ui-closed-card">
          <h2>Application Closed</h2>
          <p>
            The training session has been stopped gracefully, and the local backend server has shut down.
          </p>
          <div className="ui-closed-card__hint">
            You may now safely close this browser window or tab.
          </div>
        </div>
      </div>
    );
  }

  const isStandaloneEval10 =
    typeof window !== "undefined" &&
    (new URLSearchParams(window.location.search).get("view") === "eval10" ||
      window.location.hash === "#eval10");

  if (isStandaloneEval10) {
    return (
      <div className="eval-standalone-container" data-testid="eval-standalone-window">
        <EvaluationEpisodesTab
          currentStage={effectiveCurriculumStage}
          runId={trainingStatus?.run_id}
          popupOnly={true}
          isOpen={true}
          standaloneWindow={true}
          onClose={() => window.close()}
        />
      </div>
    );
  }

  const useFixedWorkspace =
    activeTab === "train" || (activeTab === "trouble" && troubleMode === "sandbox");

  return (
    <div className="ui-shell">
      <header className="ui-topbar">
        <div className="ui-topbar__brand">
          <h1 className="ui-topbar__title">Sheepdog Lab</h1>
          <span className="ui-topbar__stage-pill" data-testid="topbar-stage-pill">
            <span>{topbarStageLabel}</span>
            {topbarEnvLabel ? (
              <span className="ui-topbar__stage-env">{topbarEnvLabel}</span>
            ) : null}
          </span>
        </div>

        <nav className="ui-topbar__nav" role="tablist" aria-label="Workspace sections">
          {APP_TABS.map(({ id, label }) => (
            <button
              key={id}
              type="button"
              role="tab"
              aria-selected={activeTab === id}
              className={`ui-topbar__tab${activeTab === id ? " ui-topbar__tab--active" : ""}`}
              onClick={() => setActiveTab(id)}
            >
              {label}
            </button>
          ))}
        </nav>

        <div className="ui-topbar__actions">
          <span className={`ui-badge ${topbarStatusTone}`}>{topbarStatusLabel}</span>

          {statusRunning ? (
            <>
              <button
                type="button"
                className="ui-btn--sm"
                aria-label="Pause training"
                title="Pause after current checkpoint"
                onClick={() => void handlePauseTraining()}
                disabled={lifecycleAction !== null || clearingTraining || statusReadOnlyPreview}
              >
                {lifecycleAction === "pause" ? "Pausing…" : "Pause"}
              </button>
              <button
                type="button"
                className="ui-btn--sm ui-btn--danger"
                aria-label="Stop training"
                title="Stop after current checkpoint"
                onClick={() => void handleStopTraining()}
                disabled={lifecycleAction !== null || clearingTraining || statusReadOnlyPreview}
              >
                {lifecycleAction === "stop" ? "Stopping…" : "Stop"}
              </button>
            </>
          ) : trainingStatus?.resume_available &&
            (trainingStatus.resume_request?.continuous ||
              (trainingStatus.resume_remaining_episodes ?? 0) > 0 ||
              (trainingStatus.resume_request?.stage_queue?.length ?? 0) > 0) ? (
            <button
              type="button"
              className="ui-btn--sm ui-btn--primary"
              aria-label="Resume training"
              onClick={() => void handleResumeTraining()}
              disabled={isStartingTraining || clearingTraining || statusReadOnlyPreview}
            >
              Resume
            </button>
          ) : null}

          <CopyAgentDataButton
            trainingStatus={trainingStatus}
            checkpointIndex={checkpointIndex}
            curriculumStage={effectiveCurriculumStage}
          />
        </div>
      </header>

      <main
        className={`ui-workspace${useFixedWorkspace ? " ui-workspace--fixed" : ""}`}
      >
        {statusPhase === "offline" ? (
          <div className="ui-banner ui-banner--error" role="alert">
            <div className="ui-banner__detail">
              <strong>Backend disconnected</strong>
              <span>Unable to reach the local training server. Retrying automatically.</span>
            </div>
          </div>
        ) : null}

        {statusPhase === "restoring" ? (
          <div className="ui-banner ui-banner--warn" role="status">
            <div className="ui-banner__detail">
              <strong>Restoring state</strong>
              <span>{trainingStatus?.message || "Restoring saved training state…"}</span>
            </div>
          </div>
        ) : null}

        {statusReadOnlyPreview ? (
          <div className="ui-banner ui-banner--warn" role="status">
            <div className="ui-banner__detail">
              <strong>Read-only preview</strong>
              <span>Training mutations and live simulation runs are disabled.</span>
            </div>
          </div>
        ) : null}

        {error && error !== "Backend disconnected" ? (
          <div className="ui-banner ui-banner--error" role="alert">
            <span>{error}</span>
          </div>
        ) : null}

        {trainingError &&
        activeTab !== "train" &&
        trainingError !== error &&
        trainingError !== "Backend disconnected" ? (
          <div className="ui-banner ui-banner--error" role="alert">
            <span>{trainingError}</span>
          </div>
        ) : null}

        {activeTab === "train" ? (
          <TrainView
            checkpoints={checkpointIndex?.checkpoints ?? []}
            selectedCheckpoint={selectedCheckpoint}
            selectedCheckpointEpisode={selectedCheckpointEpisode}
            bestCheckpointEpisode={bestCheckpointEpisode}
            seedOptions={seedOptions}
            selectedSeed={selectedSeed}
            replay={replay}
            snapshot={snapshot}
            frameIndex={frameIndex}
            isPlaying={runState === "running"}
            statusLabel={statusLabel}
            canEndEpisode={canEndEpisode}
            runningCurrent={runningCurrentReplay}
            loadingSelectedReplay={loadingSelectedReplay}
            playbackFastMode={playbackFastMode}
            trainingStatus={trainingStatus}
            curriculumStage={effectiveCurriculumStage}
            maxCurriculumStage={maxCurriculumStage}
            episodes={trainingEpisodes}
            fastMode={trainingFastMode}
            enableInstincts={effectiveEnableInstincts}
            autoPromote={effectiveAutoPromote}
            continuous={trainingRunning ? trainingStatus?.continuous ?? trainingContinuous : trainingContinuous}
            debugRewardBreakdown={effectiveDebugRewardBreakdown}
            startingModelSource={startingModelSource}
            stageSpec={trainingStageSpec}
            onStageSpecChange={setTrainingStageSpec}
            passLabel={trainingPassLabel}
            onPassLabelChange={setTrainingPassLabel}
            running={trainingRunning}
            clearing={clearingTraining}
            isStartingTraining={isStartingTraining}
            lifecycleAction={lifecycleAction}
            latestSuccessRate={stageScopedLatestSuccess}
            trainingError={trainingError}
            autoPromoteGate={trainingStatus?.auto_promote_gate ?? null}
            resumeAvailable={trainingStatus?.resume_available ?? false}
            resumeRemainingEpisodes={trainingStatus?.resume_remaining_episodes ?? null}
            isReadOnlyPreview={statusReadOnlyPreview}
            disabled={loading}
            onSelectCheckpointEpisode={handleCheckpointChange}
            onSelectSeed={handleSeedChange}
            onSelectSeedAndReplay={handleSelectSeedAndReplay}
            onRunCurrentReplay={() => void handleRunCurrentReplay()}
            onReplaySelected={() => void handleReplaySelected()}
            onEndEpisode={handleEndEpisode}
            onTogglePlay={handleTogglePlay}
            onStepFrame={handleStepFrame}
            onScrubFrame={handleScrubFrame}
            onPlaybackFastModeChange={setPlaybackFastMode}
            onOpenInspector={handleOpenCurrentReplayInInspector}
            onOpenTroubleSpots={() => {
              setTroubleMode("failures");
              setActiveTab("trouble");
            }}
            onOpenProgress={() => setActiveTab("progress")}
            onPromote={handlePromote}
            onCurriculumStageChange={(stage) => void handleCurriculumStageChange(stage)}
            onEpisodesChange={setTrainingEpisodes}
            onStartTraining={() => void handleStartTraining()}
            onResumeTraining={() => void handleResumeTraining()}
            onPauseTraining={() => void handlePauseTraining()}
            onStopTraining={() => void handleStopTraining()}
            onEnableInstinctsChange={setTrainingEnableInstincts}
            onAutoPromoteChange={setTrainingAutoPromote}
            onContinuousChange={setTrainingContinuous}
            onStartingModelSourceChange={setStartingModelSource}
            onFastModeChange={setTrainingFastMode}
            onDebugRewardBreakdownChange={setTrainingDebugRewardBreakdown}
            onStartRemediation={() => void handleStartRemediation()}
            onClearTraining={() => void handleClearTraining()}
          />
        ) : null}

        {activeTab === "progress" ? (
          <ProgressView
            checkpoints={checkpointIndex?.checkpoints ?? []}
            checkpointIndex={checkpointIndex}
            trainingEpisodes={trainingEpisodeLogs}
            sessionEvents={sessionEvents}
            trainingStatus={trainingStatus}
            effectiveCurriculumStage={effectiveCurriculumStage}
            maxCurriculumStage={maxCurriculumStage}
            bestCheckpointEpisode={bestCheckpointEpisode}
            selectedCheckpointEpisode={selectedCheckpointEpisode}
            isReadOnlyPreview={statusReadOnlyPreview}
            onInspectTroubleSpots={() => {
              setTroubleMode("failures");
              setActiveTab("trouble");
            }}
            onSelectCheckpointForReplay={handleWatchReplay}
            onInspectCheckpointInTroubleSpots={(episode) => {
              setSelectedCheckpointEpisode(episode);
              setTroubleMode("failures");
              setActiveTab("trouble");
            }}
            onOpenReplayInInspector={(replayPath) => {
              const id = extractReplayId(replayPath);
              if (id) {
                setTemporalReplayId(id);
              }
              setTroubleMode("inspector");
              setActiveTab("trouble");
            }}
          />
        ) : null}

        {activeTab === "trouble" ? (
          <TroubleSpotsView
            checkpoints={checkpointIndex?.checkpoints ?? []}
            trainingStatus={trainingStatus}
            effectiveCurriculumStage={effectiveCurriculumStage}
            bestCheckpointEpisode={bestCheckpointEpisode}
            selectedCheckpointEpisode={selectedCheckpointEpisode}
            mode={troubleMode}
            onModeChange={setTroubleMode}
            temporalReplayId={temporalReplayId}
            activeReplay={replay}
            backendOffline={statusPhase === "offline"}
            isReadOnlyPreview={statusReadOnlyPreview}
            backendStarting={statusPhase === "restoring"}
            onSyncReplayToMainViewer={(syncedReplay, worldStep) => {
              lastLoadedSelectionKeyRef.current =
                syncedReplay.checkpoint_episode != null
                  ? `${syncedReplay.checkpoint_episode}:${syncedReplay.seed}`
                  : null;
              lastLoadedReplayIdRef.current = null;
              setReplay(syncedReplay);
              const idx = syncedReplay.frames.findIndex(
                (fr) => (fr.step ?? fr.snapshot?.step) === worldStep,
              );
              setFrameIndex(idx >= 0 ? idx : 0);
              setRunState("paused");
            }}
            onWatchSeedReplay={handleWatchReplay}
            onInspectSeedSteps={(replayId, checkpointEpisode, seed) => {
              const resolvedId =
                replayId ||
                `checkpoint-${String(checkpointEpisode).padStart(6, "0")}-seed-${String(seed).padStart(6, "0")}`;
              setTemporalReplayId(resolvedId);
              setTroubleMode("inspector");
              setActiveTab("trouble");
            }}
            onSelectArenaReplay={(arenaReplay, checkpointEpisode, seed) => {
              lastLoadedSelectionKeyRef.current = `${checkpointEpisode}:${seed}`;
              lastLoadedReplayIdRef.current = null;
              setSelectedCheckpointEpisode(checkpointEpisode);
              setSelectedSeed(seed);
              setReplay(arenaReplay);
              setFrameIndex(0);
              setRunState((arenaReplay.frames?.length ?? 0) > 0 ? "running" : "idle");
              setActiveTab("train");
            }}
            scenarioSnapshot={scenarioSnapshot}
            scenarioReplay={scenarioReplay}
            scenarioIndex={scenarioIndex}
            latestCheckpointEpisode={latestCheckpointEpisode}
            scenarioPersonalityStrength={scenarioPersonalityStrength}
            scenarioSeed={scenarioSeed}
            scenarioFastMode={scenarioFastMode}
            runningScenario={runningScenario}
            evaluatingScenarios={evaluatingScenarios}
            canEndScenarioEpisode={canEndScenarioEpisode}
            scenarioStatusLabel={scenarioStatusLabel}
            scenarioPolicyLabel={scenarioPolicyLabel}
            scenarioCheckpointMode={scenarioCheckpointMode}
            specificScenarioCheckpointEpisode={specificScenarioCheckpointEpisode}
            selectedScenarioId={selectedScenarioId}
            saveSnapshotSource={saveSnapshotSource}
            disabled={loading || statusReadOnlyPreview}
            onScenarioPersonalityStrengthChange={setScenarioPersonalityStrength}
            onScenarioSeedChange={setScenarioSeed}
            onScenarioFastModeChange={setScenarioFastMode}
            onRunScenario={() => void handleRunScenario()}
            onRestartScenarioPlayback={handleRestartScenarioPlayback}
            onEndScenarioEpisode={handleEndScenarioEpisode}
            onScenarioCheckpointModeChange={setScenarioCheckpointMode}
            onSpecificScenarioCheckpointChange={setSpecificScenarioCheckpointEpisode}
            onSelectScenario={setSelectedScenarioId}
            onSaveSnapshotSourceChange={setSaveSnapshotSource}
            onSaveScenarioFromReplay={() => void handleSaveScenario()}
            onEvaluateScenarios={() => void handleEvaluateScenarios()}
            onRunSavedScenario={() => void handleRunSavedScenario()}
          />
        ) : null}

        {activeTab === "settings" ? (
          <SettingsView
            checkpointIndex={checkpointIndex}
            trainingStatus={trainingStatus}
            effectiveConfig={effectiveConfig}
            topologyInfo={networkTopology}
            running={trainingRunning}
            clearing={clearingTraining}
            isReadOnlyPreview={statusReadOnlyPreview}
            onResetJourney={() => void handleResetJourney()}
            onClearTraining={() => void handleClearTraining()}
            onCloseApp={() => void handleCloseApp()}
            onRefreshData={() => void handleRefreshAllData()}
            onWatchReplay={handleWatchReplay}
            onExpandNetwork={handleExpandNetwork}
          />
        ) : null}
      </main>
    </div>
  );
}

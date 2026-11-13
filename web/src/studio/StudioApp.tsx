import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import "./studio.css";
import {
  checkBackendHealth,
  clearTraining,
  crossStageFork,
  fetchReplayById,
  isBackendStartingError,
  loadCheckpointIndex,
  loadEffectiveConfig,
  loadNetworkTopology,
  loadReplay,
  loadSessionEvents,
  loadStageDiagnostics,
  loadTrainingEpisodes,
  loadTrainingStatus,
  pauseTraining,
  resetJourneyTraining,
  runReplay,
  shutdownApp,
  startRemediationFork,
  startTraining,
  stopTraining,
} from "../lib/api";
import {
  analyzePerSeedReliability,
  calculateEfficiencyTrend,
  getCheckpointStage,
} from "../lib/chartPipeline";
import { RECOMMENDED_EPISODES_BY_STAGE } from "../lib/trainingConfig";
import type {
  CheckpointEntry,
  CheckpointIndex,
  SessionChangeEvent,
  StageBottleneckReport,
  TrainingEpisode,
  TrainingStatus,
} from "../state/types";
import { dogColor } from "../components/dogPalette";
import { CopyAgentDataButton } from "../components/CopyAgentDataButton";
import { StageBottlenecksPanel } from "../components/StageBottlenecksPanel";
import { ResultsPanel } from "../components/ResultsPanel";
import { ConfigPanel } from "../components/ConfigPanel";
import { NetworkTab } from "../components/NetworkTab";
import { TemporalDiagnosticsPanel } from "../components/TemporalDiagnosticsPanel";
import { EarlyWarningsPanel } from "../components/EarlyWarningsPanel";
import {
  ROLE_LABELS,
  StudioFieldCanvas,
  snapshotFromReplay,
  type AnyReplayPayload,
} from "./StudioFieldCanvas";
import { StudioLearningCharts } from "./StudioLearningCharts";
import { StudioEvalArena } from "./StudioEvalArena";

// The checkpoint index is tens of MB on long runs and is served by the training
// process, so only re-download it when the status reports a new checkpoint (or as a
// slow safety net) instead of on every 3 s core-data poll.
const CHECKPOINT_INDEX_SAFETY_REFRESH_MS = 60_000;

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
  25: "3 dogs · 6 sheep · bridge: corner-heavy starts",
  26: "3 dogs · 6 sheep · hard spawn mix",
  27: "3 dogs · 6 sheep · mild personality variation",
  28: "3 dogs · 6 sheep · moderate personality variation",
  29: "3 dogs · 6 sheep · weaker cohesion bridge",
  30: "3 dogs · 6 sheep · disable no-pressure cohesion",
  31: "3 dogs · 6 sheep · reduce cohesion + stronger personalities",
  32: "3 dogs · 6 sheep · lowest cohesion + strongest personalities",
  33: "3 dogs · 6 sheep · heterogeneous sheep speeds (144×104)",
  34: "3 dogs · 6 sheep · extended arena + speed variance",
  35: "3 dogs · 6 sheep · high scatter + speed variance",
  36: "3 dogs · 6 sheep · zero-pressure cohesion disabled",
  37: "3 dogs · 6 sheep · independent sheep + dynamic pen",
  38: "3 dogs · 6 sheep · mastery benchmark (max complexity)",
};

const DEFAULT_BENCHMARK_SEEDS = [11, 23, 37, 41, 53, 59, 61, 67, 71, 73];

type StudioWorkspace = "command" | "arena" | "temporal" | "analytics";
type BottomIntelTab = "gate" | "seeds" | "controls";
type AnalyticsSubTab = "curves" | "bottlenecks" | "milestones" | "config";

function resolveInitialStudioNav(): {
  workspace: StudioWorkspace;
  replayId: string | null;
} {
  if (typeof window === "undefined") {
    return { workspace: "command", replayId: null };
  }
  const params = new URLSearchParams(window.location.search);
  const wsParam = (
    params.get("workspace") ??
    params.get("mode") ??
    params.get("tab") ??
    ""
  ).toLowerCase();
  const replayParam = params.get("replay")?.trim() || null;
  if (
    wsParam === "temporal" ||
    wsParam === "temporal-diagnostics" ||
    Boolean(replayParam)
  ) {
    return { workspace: "temporal", replayId: replayParam };
  }
  if (wsParam === "arena") {
    return { workspace: "arena", replayId: replayParam };
  }
  if (wsParam === "analytics") {
    return { workspace: "analytics", replayId: replayParam };
  }
  return { workspace: "command", replayId: replayParam };
}

function formatDuration(sec?: number | null): string {
  if (sec == null || !Number.isFinite(sec) || sec <= 0) return "—";
  const s = Math.round(sec);
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  const rem = s % 60;
  if (m < 60) return `${m}m ${rem}s`;
  const h = Math.floor(m / 60);
  return `${h}h ${m % 60}m`;
}

export function StudioApp() {
  // Ensure Studio v2 favicon is active in document.head
  useEffect(() => {
    const existing = document.querySelector("link[rel*='icon']") as HTMLLinkElement | null;
    if (existing) {
      existing.type = "image/svg+xml";
      existing.href = "/favicon-v2.svg";
    } else {
      const link = document.createElement("link");
      link.rel = "icon";
      link.type = "image/svg+xml";
      link.href = "/favicon-v2.svg";
      document.head.appendChild(link);
    }
  }, []);

  // Navigation state
  const initialNav = useMemo(() => resolveInitialStudioNav(), []);
  const [workspace, setWorkspace] = useState<StudioWorkspace>(initialNav.workspace);
  const [temporalReplayId, setTemporalReplayId] = useState<string | null>(initialNav.replayId);
  const [bottomTab, setBottomTab] = useState<BottomIntelTab>("gate");
  const [analyticsSubTab, setAnalyticsSubTab] = useState<AnalyticsSubTab>("curves");
  const [arenaTargetCheckpoint, setArenaTargetCheckpoint] = useState<number | null>(null);
  const [earlyWarningsOpen, setEarlyWarningsOpen] = useState<boolean>(() => {
    if (typeof window === "undefined") return false;
    const params = new URLSearchParams(window.location.search);
    return params.get("warnings") === "open" || params.get("warnings") === "1";
  });

  // Core data state
  const [trainingStatus, setTrainingStatus] = useState<TrainingStatus | null>(null);
  const [checkpointIndex, setCheckpointIndex] = useState<CheckpointIndex | null>(null);
  const [trainingEpisodes, setTrainingEpisodes] = useState<TrainingEpisode[]>([]);
  const [sessionEvents, setSessionEvents] = useState<SessionChangeEvent[]>([]);
  const [stageDiagnostics, setStageDiagnostics] = useState<StageBottleneckReport | null>(null);
  const [effectiveConfig, setEffectiveConfig] = useState<Record<string, unknown> | null>(null);
  const [topologyInfo, setTopologyInfo] = useState<{
    observation_mode: string;
    hidden_layer_sizes: number[];
    observation_size: number;
    action_size: number;
    action_masking_enabled: boolean;
  } | null>(null);

  const [backendStarting, setBackendStarting] = useState<boolean>(false);
  const [backendOffline, setBackendOffline] = useState<boolean>(false);
  const [isReadOnlyPreview, setIsReadOnlyPreview] = useState<boolean>(false);
  const [temporalCachedAvailable, setTemporalCachedAvailable] = useState<boolean>(false);
  const [actionMessage, setActionMessage] = useState<string | null>(null);
  const [actionBusy, setActionBusy] = useState<boolean>(false);

  // Training launch config state
  const [selectedStage, setSelectedStage] = useState<number>(1);
  const [batchEpisodes, setBatchEpisodes] = useState<number>(200);
  const [enableInstincts, setEnableInstincts] = useState<boolean>(true);
  const [autoPromote, setAutoPromote] = useState<boolean>(true);
  const [modelSource, setModelSource] = useState<"latest_stage" | "scratch">("latest_stage");
  const userTouchedStageRef = useRef<boolean>(false);

  // Command Center Episode Player state
  const [selectedCheckpointEp, setSelectedCheckpointEp] = useState<number | null>(null);
  const [selectedSeed, setSelectedSeed] = useState<number>(11);
  const [activeReplay, setActiveReplay] = useState<AnyReplayPayload | null>(null);
  const [loadedReplayKey, setLoadedReplayKey] = useState<string | null>(null);
  const [replayLoading, setReplayLoading] = useState<boolean>(false);
  const [frameIndex, setFrameIndex] = useState<number>(0);
  const [isPlaying, setIsPlaying] = useState<boolean>(true);
  const [playbackSpeed, setPlaybackSpeed] = useState<number>(2);
  const [rotateSeeds, setRotateSeeds] = useState<boolean>(true);
  const [loopReplay, setLoopReplay] = useState<boolean>(true);
  const [autoFollowLive, setAutoFollowLive] = useState<boolean>(true);
  const [lockUntilFinish, setLockUntilFinish] = useState<boolean>(true);
  const [episodeViewFilter, setEpisodeViewFilter] = useState<"all" | "losses" | "wins">("all");
  const [showRoles, setShowRoles] = useState<boolean>(true);
  const [showVectors, setShowVectors] = useState<boolean>(false);
  const loadedReplayKeyRef = useRef<string | null>(null);
  const replayCacheRef = useRef<Map<string, AnyReplayPayload>>(new Map());

  const cacheStudioReplay = useCallback((key: string, payload: AnyReplayPayload) => {
    const map = replayCacheRef.current;
    if (map.has(key)) {
      map.delete(key);
    } else if (map.size >= 40) {
      const oldestKey = map.keys().next().value;
      if (oldestKey !== undefined) {
        map.delete(oldestKey);
      }
    }
    map.set(key, payload);
  }, []);

  // Derived stage & checkpoints
  const effectiveStage = useMemo(() => {
    if (trainingStatus?.curriculum_stage != null) return trainingStatus.curriculum_stage;
    return selectedStage || 1;
  }, [trainingStatus?.curriculum_stage, selectedStage]);

  // Sync stage selector when backend stage advances
  useEffect(() => {
    if (trainingStatus?.curriculum_stage != null && !userTouchedStageRef.current) {
      const st = trainingStatus.curriculum_stage;
      setSelectedStage(st);
      const rec = RECOMMENDED_EPISODES_BY_STAGE[st];
      if (rec) setBatchEpisodes(rec);
    }
  }, [trainingStatus?.curriculum_stage]);

  // Poll training status, checkpoints, and episodes without recreating unchanged checkpointIndex references
  const checkpointIndexKeyRef = useRef<string | null>(null);
  const checkpointIndexFetchedAtRef = useRef(0);
  const mountedRef = useRef<boolean>(true);
  const pollInFlightRef = useRef<boolean>(false);
  const pendingExplicitRefreshRef = useRef<boolean>(false);
  const mutationEpochRef = useRef<number>(0);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
    };
  }, []);

  const refreshCoreData = useCallback(async (explicit = false) => {
    if (!mountedRef.current) return;
    if (pollInFlightRef.current) {
      if (explicit) {
        pendingExplicitRefreshRef.current = true;
      }
      return;
    }
    pollInFlightRef.current = true;
    const epochAtStart = mutationEpochRef.current;
    try {
      const health = await checkBackendHealth();
      if (!mountedRef.current || epochAtStart !== mutationEpochRef.current) return;
      if (health.ok && health.readOnlyPreview) {
        setIsReadOnlyPreview(true);
        setBackendStarting(false);
        setBackendOffline(false);
        setTrainingStatus(null);
        setCheckpointIndex(null);
        setTrainingEpisodes([]);
        return;
      }
      if (!health.ok && health.error !== "unimplemented_health_mock") {
        if (health.starting) {
          setBackendStarting(true);
          setBackendOffline(false);
        } else {
          setBackendStarting(false);
          setBackendOffline(true);
        }
        return;
      }

      let currentSt = effectiveStage;
      let statusKey: string | null = null;
      try {
        const status = await loadTrainingStatus();
        if (!mountedRef.current || epochAtStart !== mutationEpochRef.current) return;
        if (
          status?.read_only_preview === true ||
          status?.mode === "read_only_preview" ||
          status?.phase === "read_only_preview"
        ) {
          setIsReadOnlyPreview(true);
          setBackendStarting(false);
          setBackendOffline(false);
          setTrainingStatus(null);
          setCheckpointIndex(null);
          setTrainingEpisodes([]);
          return;
        }
        setIsReadOnlyPreview(false);
        setTrainingStatus(status);
        if (status?.curriculum_stage != null) {
          currentSt = status.curriculum_stage;
        }
        statusKey = [
          status?.latest_checkpoint_episode ?? "",
          status?.active_checkpoint_id ?? status?.active_policy_identity ?? "",
          status?.curriculum_stage ?? "",
          status?.run_id ?? "",
        ].join("|");
        setBackendStarting(false);
        setBackendOffline(false);
      } catch (err) {
        if (!mountedRef.current || epochAtStart !== mutationEpochRef.current) return;
        if (isBackendStartingError(err)) {
          setBackendStarting(true);
          setBackendOffline(false);
        } else {
          setBackendOffline(true);
        }
        return;
      }

      const indexStale =
        checkpointIndexKeyRef.current === null ||
        (statusKey !== null && statusKey !== checkpointIndexKeyRef.current) ||
        Date.now() - checkpointIndexFetchedAtRef.current >= CHECKPOINT_INDEX_SAFETY_REFRESH_MS;

      if (indexStale) {
        try {
          const cpIdx = await loadCheckpointIndex();
          if (!mountedRef.current || epochAtStart !== mutationEpochRef.current) return;
          if (cpIdx) {
            checkpointIndexKeyRef.current = statusKey ?? checkpointIndexKeyRef.current ?? "";
            checkpointIndexFetchedAtRef.current = Date.now();
            setCheckpointIndex((prev) => {
              if (!prev) return cpIdx;
              const prevList = prev.checkpoints ?? [];
              const nextList = cpIdx.checkpoints ?? [];
              const prevLast = prevList[prevList.length - 1];
              const nextLast = nextList[nextList.length - 1];
              if (
                prevList.length === nextList.length &&
                prevLast?.checkpoint_episode === nextLast?.checkpoint_episode &&
                (prev as any).best_checkpoint_episode === (cpIdx as any).best_checkpoint_episode
              ) {
                return prev;
              }
              return cpIdx;
            });
          }
        } catch {
          // Ignore transient checkpoint-index read error
        }
      }

      try {
        const epRes = await loadTrainingEpisodes({
          stage: currentSt,
          limit: 1200,
          order: "desc",
        });
        if (!mountedRef.current || epochAtStart !== mutationEpochRef.current) return;
        if (epRes?.episodes) {
          setTrainingEpisodes(epRes.episodes);
        }
      } catch {
        // Ignore
      }
    } finally {
      pollInFlightRef.current = false;
      if (pendingExplicitRefreshRef.current && mountedRef.current) {
        pendingExplicitRefreshRef.current = false;
        void refreshCoreData(false);
      }
    }
  }, [effectiveStage]);

  useEffect(() => {
    void refreshCoreData(false);
    const timer = window.setInterval(() => {
      void refreshCoreData(false);
    }, 3000);
    return () => window.clearInterval(timer);
  }, [refreshCoreData]);

  // Secondary diagnostics & metadata polling
  useEffect(() => {
    if (isReadOnlyPreview || backendOffline) return;
    let active = true;
    const fetchSecondary = async () => {
      try {
        const [evRes, diagRes, cfgRes, topoRes] = await Promise.allSettled([
          loadSessionEvents(),
          loadStageDiagnostics(effectiveStage, trainingStatus?.run_id ?? undefined),
          loadEffectiveConfig(),
          loadNetworkTopology(),
        ]);
        if (!active) return;
        if (evRes.status === "fulfilled" && evRes.value?.events) {
          setSessionEvents(evRes.value.events);
        }
        if (diagRes.status === "fulfilled" && diagRes.value) setStageDiagnostics(diagRes.value);
        if (cfgRes.status === "fulfilled" && cfgRes.value) setEffectiveConfig(cfgRes.value);
        if (topoRes.status === "fulfilled" && topoRes.value) setTopologyInfo(topoRes.value as any);
      } catch {
        // Ignore
      }
    };
    fetchSecondary();
    const timer = window.setInterval(fetchSecondary, 10000);
    return () => {
      active = false;
      window.clearInterval(timer);
    };
  }, [effectiveStage, trainingStatus?.run_id, isReadOnlyPreview, backendOffline]);

  const allCheckpoints = useMemo(
    () =>
      [...(checkpointIndex?.checkpoints ?? [])].sort(
        (a, b) => b.checkpoint_episode - a.checkpoint_episode
      ),
    [checkpointIndex]
  );

  const stageCheckpoints = useMemo(() => {
    const filtered = allCheckpoints.filter((c) => getCheckpointStage(c) === effectiveStage);
    const source = filtered.length > 0 ? filtered : allCheckpoints;
    const seen = new Set<number>();
    return source.filter((cp) => {
      if (seen.has(cp.checkpoint_episode)) return false;
      seen.add(cp.checkpoint_episode);
      return true;
    });
  }, [allCheckpoints, effectiveStage]);

  const activeCheckpoint = useMemo<CheckpointEntry | null>(() => {
    if (selectedCheckpointEp != null) {
      const foundInStage = stageCheckpoints.find(
        (c) => c.checkpoint_episode === selectedCheckpointEp
      );
      if (foundInStage) return foundInStage;
      const foundAny = allCheckpoints.find((c) => c.checkpoint_episode === selectedCheckpointEp);
      if (foundAny) return foundAny;
    }
    return stageCheckpoints[0] ?? allCheckpoints[0] ?? null;
  }, [allCheckpoints, stageCheckpoints, selectedCheckpointEp]);

  // Dynamic 10 benchmark seeds for the active checkpoint / stage
  const activeSeeds = useMemo<number[]>(() => {
    if (activeCheckpoint?.records && activeCheckpoint.records.length > 0) {
      return activeCheckpoint.records.map((r) => r.seed);
    }
    return DEFAULT_BENCHMARK_SEEDS;
  }, [activeCheckpoint]);

  // Seeds matching the current outcome filter ("all" | "losses" | "wins")
  const filteredSeedsForCheckpoint = useMemo<number[]>(() => {
    const records = activeCheckpoint?.records ?? [];
    if (records.length === 0 || episodeViewFilter === "all") return activeSeeds;
    const matching = records
      .filter((r) => (episodeViewFilter === "losses" ? !r.success : Boolean(r.success)))
      .map((r) => r.seed);
    return matching.length > 0 ? matching : activeSeeds;
  }, [activeCheckpoint, activeSeeds, episodeViewFilter]);

  const lossSeedCount = useMemo(() => {
    return (activeCheckpoint?.records ?? []).filter((r) => !r.success).length;
  }, [activeCheckpoint]);

  // User-initiated seed pick: switches filter to "all" if the picked seed is outside the active filter
  const handlePickSeed = useCallback(
    (seed: number) => {
      if (episodeViewFilter !== "all" && !filteredSeedsForCheckpoint.includes(seed)) {
        setEpisodeViewFilter("all");
      }
      setSelectedSeed(seed);
      setFrameIndex(0);
      setIsPlaying(true);
    },
    [episodeViewFilter, filteredSeedsForCheckpoint]
  );

  // Keep selectedSeed synced with activeSeeds & episodeViewFilter when filter or checkpoint changes
  useEffect(() => {
    if (
      filteredSeedsForCheckpoint.length > 0 &&
      !filteredSeedsForCheckpoint.includes(selectedSeed)
    ) {
      setSelectedSeed(filteredSeedsForCheckpoint[0]);
    }
  }, [filteredSeedsForCheckpoint, selectedSeed]);

  const totalFrames = activeReplay?.frames?.length ?? 0;

  // Auto-follow newest checkpoint:
  // If lockUntilFinish is ON and an episode is currently playing mid-stream, DO NOT interrupt it!
  useEffect(() => {
    if (!autoFollowLive || stageCheckpoints.length === 0) return;
    const latestEp = stageCheckpoints[0].checkpoint_episode;
    if (selectedCheckpointEp === latestEp) return;

    const isMidEpisode =
      lockUntilFinish && isPlaying && totalFrames > 1 && frameIndex < totalFrames - 1;
    if (!isMidEpisode || selectedCheckpointEp == null) {
      setSelectedCheckpointEp(latestEp);
    }
  }, [
    autoFollowLive,
    lockUntilFinish,
    isPlaying,
    totalFrames,
    frameIndex,
    stageCheckpoints,
    selectedCheckpointEp,
  ]);

  const activeCheckpointEp = activeCheckpoint?.checkpoint_episode ?? null;
  const currentReplayKey =
    activeCheckpoint && activeCheckpointEp != null
      ? `${getCheckpointStage(activeCheckpoint)}:${activeCheckpointEp}:${selectedSeed}`
      : null;

  // Load replay ONLY when (activeCheckpointEp, selectedSeed) actually changes
  useEffect(() => {
    if (!activeCheckpoint || activeCheckpointEp == null || !currentReplayKey) return;
    if (loadedReplayKeyRef.current === currentReplayKey && activeReplay !== null) {
      if (loadedReplayKey !== currentReplayKey) {
        setLoadedReplayKey(currentReplayKey);
      }
      return;
    }

    const cached = replayCacheRef.current.get(currentReplayKey);
    if (cached) {
      loadedReplayKeyRef.current = currentReplayKey;
      setActiveReplay(cached);
      setLoadedReplayKey(currentReplayKey);
      setFrameIndex(0);
      return;
    }

    let cancelled = false;

    const loadActiveSeedReplay = async () => {
      setReplayLoading(true);
      try {
        const records = activeCheckpoint.records ?? [];
        const rec = records.find((r) => r.seed === selectedSeed) as any;
        const targetSeed =
          selectedSeed ?? rec?.seed ?? records[0]?.seed ?? DEFAULT_BENCHMARK_SEEDS[0];
        let payload: AnyReplayPayload | null = null;

        if (rec?.replay_id) {
          try {
            payload = await fetchReplayById(String(rec.replay_id));
          } catch {
            payload = null;
          }
        }
        if (!payload && rec?.replay_path) {
          try {
            payload = await loadReplay(String(rec.replay_path));
          } catch {
            payload = null;
          }
        }
        // Only use checkpoint sample_replay_path / replay when loading the primary representative seed
        const isPrimarySeed = records.length === 0 || targetSeed === records[0]?.seed;
        const samplePath = (activeCheckpoint as any).sample_replay_path ?? activeCheckpoint.replay;
        if (!payload && isPrimarySeed && samplePath) {
          try {
            const candidate = await loadReplay(String(samplePath));
            if (candidate && (candidate.seed == null || Number(candidate.seed) === Number(targetSeed))) {
              payload = candidate;
            }
          } catch {
            payload = null;
          }
        }
        // Fallback: simulate episode on-demand for the requested seed
        if (!payload && !cancelled) {
          try {
            const cpStage = getCheckpointStage(activeCheckpoint);
            payload = await runReplay({
              seed: targetSeed,
              checkpoint_episode: activeCheckpoint.checkpoint_episode,
              trainer_type: activeCheckpoint.trainer_type,
              policy_type: activeCheckpoint.policy_type,
              policy_mode: activeCheckpoint.policy_mode || activeCheckpoint.policy_name,
              effective_config: {
                enable_instinct_rewards: enableInstincts,
                curriculum_stage: cpStage > 0 ? cpStage : effectiveStage,
                debug_reward_breakdown: true,
              },
            });
          } catch {
            payload = null;
          }
        }
        if (!cancelled && payload) {
          cacheStudioReplay(currentReplayKey, payload);
          loadedReplayKeyRef.current = currentReplayKey;
          setActiveReplay(payload);
          setLoadedReplayKey(currentReplayKey);
          setFrameIndex(0);
        }
      } finally {
        if (!cancelled) setReplayLoading(false);
      }
    };

    loadActiveSeedReplay();
    return () => {
      cancelled = true;
    };
  }, [
    activeCheckpoint,
    activeCheckpointEp,
    selectedSeed,
    currentReplayKey,
    loadedReplayKey,
    effectiveStage,
    enableInstincts,
    activeReplay,
    cacheStudioReplay,
  ]);

  // Frame advance loop for Command Center player while mid-episode
  useEffect(() => {
    if (
      !isPlaying ||
      workspace !== "command" ||
      replayLoading ||
      totalFrames <= 1 ||
      !currentReplayKey ||
      loadedReplayKey !== currentReplayKey ||
      frameIndex >= totalFrames - 1
    ) {
      return;
    }
    const intervalMs = Math.max(20, Math.round(115 / playbackSpeed));
    const timer = window.setInterval(() => {
      setFrameIndex((prev) => Math.min(totalFrames - 1, prev + 1));
    }, intervalMs);
    return () => window.clearInterval(timer);
  }, [
    isPlaying,
    workspace,
    replayLoading,
    totalFrames,
    currentReplayKey,
    loadedReplayKey,
    frameIndex,
    playbackSpeed,
  ]);

  // End-of-episode transition: rotate to next seed (or next live checkpoint) once final frame is reached
  useEffect(() => {
    if (
      !isPlaying ||
      workspace !== "command" ||
      replayLoading ||
      totalFrames <= 1 ||
      !currentReplayKey ||
      loadedReplayKey !== currentReplayKey ||
      frameIndex < totalFrames - 1
    ) {
      return;
    }
    const holdMs = Math.max(90, Math.round(180 / playbackSpeed));
    const timer = window.setTimeout(() => {
      const latestEp = stageCheckpoints[0]?.checkpoint_episode;
      const hasNewerCheckpoint =
        autoFollowLive && latestEp != null && latestEp !== activeCheckpointEp;

      if (rotateSeeds && filteredSeedsForCheckpoint.length > 1) {
        const curIdx = filteredSeedsForCheckpoint.indexOf(selectedSeed);
        const nextIdx = (curIdx + 1) % filteredSeedsForCheckpoint.length;
        const isWrapping = curIdx >= 0 && nextIdx === 0;
        const nextSeed = filteredSeedsForCheckpoint[nextIdx];

        if (hasNewerCheckpoint) {
          setSelectedCheckpointEp(latestEp);
        }
        if (!loopReplay && isWrapping && !hasNewerCheckpoint) {
          setIsPlaying(false);
          return;
        }
        if (nextSeed != null && nextSeed !== selectedSeed) {
          setSelectedSeed(nextSeed);
          return;
        }
      }

      if (hasNewerCheckpoint) {
        setSelectedCheckpointEp(latestEp);
        setFrameIndex(0);
        return;
      }

      if (loopReplay) {
        setFrameIndex(0);
      } else {
        setIsPlaying(false);
      }
    }, holdMs);
    return () => window.clearTimeout(timer);
  }, [
    isPlaying,
    workspace,
    replayLoading,
    totalFrames,
    currentReplayKey,
    loadedReplayKey,
    frameIndex,
    playbackSpeed,
    stageCheckpoints,
    autoFollowLive,
    activeCheckpointEp,
    rotateSeeds,
    filteredSeedsForCheckpoint,
    selectedSeed,
    loopReplay,
  ]);

  const currentSnapshot = useMemo(
    () => snapshotFromReplay(activeReplay, frameIndex),
    [activeReplay, frameIndex]
  );

  // Run an on-demand live episode in the Command Center player
  const handleRunLiveEpisode = useCallback(async () => {
    setReplayLoading(true);
    try {
      const payload = await runReplay({
        seed: selectedSeed,
        checkpoint_episode: activeCheckpoint?.checkpoint_episode,
        effective_config: {
          enable_instinct_rewards: enableInstincts,
          curriculum_stage: effectiveStage,
          debug_reward_breakdown: true,
        },
      });
      if (currentReplayKey) {
        cacheStudioReplay(currentReplayKey, payload);
        loadedReplayKeyRef.current = currentReplayKey;
        setLoadedReplayKey(currentReplayKey);
      }
      setActiveReplay(payload);
      setFrameIndex(0);
      setIsPlaying(true);
    } catch (err) {
      setActionMessage(err instanceof Error ? err.message : "Failed to run replay");
    } finally {
      setReplayLoading(false);
    }
  }, [selectedSeed, activeCheckpoint, effectiveStage, enableInstincts, currentReplayKey, cacheStudioReplay]);

  const handleTemporalCachedAvailableChange = useCallback((avail: boolean) => {
    setTemporalCachedAvailable(avail);
  }, []);

  const handleTemporalRequestReconnect = useCallback(() => {
    void refreshCoreData(true);
  }, [refreshCoreData]);

  // Training lifecycle handlers
  const handleStartTraining = async () => {
    mutationEpochRef.current += 1;
    setActionBusy(true);
    setActionMessage(null);
    try {
      const activeStage = trainingStatus?.curriculum_stage;
      let status: TrainingStatus;
      if (activeStage != null && selectedStage !== activeStage) {
        status = await crossStageFork({
          target_stage: selectedStage,
          starting_model_source: modelSource,
        });
      } else {
        status = await startTraining({
          episodes: batchEpisodes,
          fast_mode: true,
          enable_instinct_rewards: enableInstincts,
          curriculum_stage: selectedStage,
          debug_reward_breakdown: true,
          auto_promote: autoPromote,
        });
      }
      mutationEpochRef.current += 1;
      if (!mountedRef.current) return;
      setTrainingStatus(status);
      userTouchedStageRef.current = false;
      setActionMessage(`Started Stage ${selectedStage} batch (${batchEpisodes} episodes).`);
    } catch (err) {
      mutationEpochRef.current += 1;
      if (!mountedRef.current) return;
      setActionMessage(err instanceof Error ? err.message : "Failed to start training");
    } finally {
      if (mountedRef.current) setActionBusy(false);
    }
  };

  const handlePauseTraining = async () => {
    mutationEpochRef.current += 1;
    setActionBusy(true);
    try {
      const status = await pauseTraining();
      mutationEpochRef.current += 1;
      if (!mountedRef.current) return;
      setTrainingStatus(status);
      setActionMessage("Pause requested after current checkpoint.");
    } catch (err) {
      mutationEpochRef.current += 1;
      if (!mountedRef.current) return;
      setActionMessage(err instanceof Error ? err.message : "Failed to pause");
    } finally {
      if (mountedRef.current) setActionBusy(false);
    }
  };

  const handleStopTraining = async () => {
    mutationEpochRef.current += 1;
    setActionBusy(true);
    try {
      const status = await stopTraining();
      mutationEpochRef.current += 1;
      if (!mountedRef.current) return;
      setTrainingStatus(status);
      setActionMessage("Stop requested after current checkpoint.");
    } catch (err) {
      mutationEpochRef.current += 1;
      if (!mountedRef.current) return;
      setActionMessage(err instanceof Error ? err.message : "Failed to stop");
    } finally {
      if (mountedRef.current) setActionBusy(false);
    }
  };

  const handleResumeTraining = async () => {
    mutationEpochRef.current += 1;
    setActionBusy(true);
    try {
      const resumeReq = trainingStatus?.resume_request;
      const status = await startTraining({
        episodes: trainingStatus?.resume_remaining_episodes ?? batchEpisodes,
        fast_mode: resumeReq?.fast_mode ?? true,
        enable_instinct_rewards: resumeReq?.enable_instinct_rewards ?? enableInstincts,
        curriculum_stage: resumeReq?.curriculum_stage ?? effectiveStage,
        debug_reward_breakdown: resumeReq?.debug_reward_breakdown ?? true,
        auto_promote: resumeReq?.auto_promote ?? autoPromote,
        resume: true,
      });
      mutationEpochRef.current += 1;
      if (!mountedRef.current) return;
      setTrainingStatus(status);
      setActionMessage("Resumed remaining episodes in batch.");
    } catch (err) {
      mutationEpochRef.current += 1;
      if (!mountedRef.current) return;
      setActionMessage(err instanceof Error ? err.message : "Failed to resume");
    } finally {
      if (mountedRef.current) setActionBusy(false);
    }
  };

  const handlePromoteStage = async (targetSt: number) => {
    mutationEpochRef.current += 1;
    setActionBusy(true);
    try {
      const status = await crossStageFork({
        target_stage: targetSt,
        starting_model_source: "latest_stage",
      });
      mutationEpochRef.current += 1;
      if (!mountedRef.current) return;
      setTrainingStatus(status);
      setSelectedStage(targetSt);
      const rec = RECOMMENDED_EPISODES_BY_STAGE[targetSt];
      if (rec) setBatchEpisodes(rec);
      setActionMessage(`Promoted curriculum to Stage ${targetSt}.`);
    } catch (err) {
      mutationEpochRef.current += 1;
      if (!mountedRef.current) return;
      setActionMessage(err instanceof Error ? err.message : "Failed to promote stage");
    } finally {
      if (mountedRef.current) setActionBusy(false);
    }
  };

  // Derived KPI & Promotion Gate Diagnostics
  const isTraining = Boolean(trainingStatus?.running);
  const isEvaluating =
    trainingStatus?.phase === "evaluating" ||
    Boolean(trainingStatus?.evaluation_in_progress) ||
    Boolean(trainingStatus?.evaluation_status?.in_progress);
  const remainingResumeEps = trainingStatus?.resume_remaining_episodes ?? 0;
  const canResume =
    !isTraining && Boolean(trainingStatus?.resume_available) && remainingResumeEps > 0;

  const gateDiag =
    trainingStatus?.auto_promote_gate ??
    (trainingStatus as any)?.auto_promote_gate_diagnostics ??
    null;
  const promotionReady = Boolean(
    gateDiag?.ready ||
      gateDiag?.decision === "promote" ||
      gateDiag?.decision === "promote_ready"
  );

  const gateChecks = useMemo(() => {
    if (!gateDiag) return [];
    const reqEvals =
      gateDiag.formal_evaluations_required ??
      gateDiag.minimum_required_evaluations ??
      gateDiag.required_evaluations ??
      3;
    const curEvals =
      gateDiag.formal_evaluations_available ?? gateDiag.evaluations_in_stage ?? 0;
    const reqStreak =
      gateDiag.qualified_evaluations_required ?? gateDiag.required_consecutive ?? 2;
    const curStreak =
      gateDiag.qualified_evaluations ?? gateDiag.consecutive_qualified_streak ?? 0;
    const reqAvg = Math.round(
      ((gateDiag.success_threshold ?? gateDiag.required_recent_average ?? 0.8) <= 1
        ? (gateDiag.success_threshold ?? gateDiag.required_recent_average ?? 0.8) * 100
        : gateDiag.success_threshold ?? 80)
    );
    const rawCurAvg =
      gateDiag.recent_average_success ??
      gateDiag.aggregate_success_rate ??
      gateDiag.recent_average_success_rate ??
      0;
    const curAvg = Math.round(rawCurAvg <= 1 ? rawCurAvg * 100 : rawCurAvg);

    const blockingSeeds: number[] =
      gateDiag.blocking_seeds ?? gateDiag.persistent_failing_seeds ?? [];

    return [
      {
        label: "Minimum Stage Evaluations",
        passed: curEvals >= reqEvals,
        value: `${curEvals} / ${reqEvals} evals`,
      },
      {
        label: "Consecutive Qualified Evals",
        passed: curStreak >= reqStreak,
        value: `${curStreak} / ${reqStreak} qualified`,
      },
      {
        label: "Recent Rolling Eval Average",
        passed: Boolean(gateDiag.success_rate_ok ?? curAvg >= reqAvg),
        value: `${curAvg}% (req ≥${reqAvg}%)`,
      },
      {
        label: "Stability Tail Floor",
        passed: Boolean(gateDiag.qualifying_tail_passed ?? gateDiag.latest_floor_passed ?? true),
        value: gateDiag.qualifying_tail_passed === false ? "Tail below floor" : "Stable tail",
      },
      {
        label: "Persistent Seed Blind-Spot Guard",
        passed: !gateDiag.persistent_seed_failure && blockingSeeds.length === 0,
        value:
          blockingSeeds.length > 0
            ? `Failing: #${blockingSeeds.join(", #")}`
            : gateDiag.blocking_seed
            ? `Failing: #${gateDiag.blocking_seed}`
            : "0 persistent seed fails",
      },
      {
        label: "Step Efficiency Optimization",
        passed: !gateDiag.step_efficiency_improving,
        value: gateDiag.step_efficiency_improving
          ? "Optimizing steps (hold active)"
          : "Efficiency plateaued / ready",
      },
    ];
  }, [gateDiag]);

  const passedGateCheckCount = gateChecks.filter((c) => c.passed).length;

  const efficiencyTrend = useMemo(
    () => calculateEfficiencyTrend(allCheckpoints, effectiveStage),
    [allCheckpoints, effectiveStage]
  );

  const seedReliability = useMemo(
    () => analyzePerSeedReliability(allCheckpoints, effectiveStage, 10),
    [allCheckpoints, effectiveStage]
  );

  // Live Frame & Cumulative Reward Anatomy for the active replay
  const liveRewardSummary = useMemo(() => {
    const frames = activeReplay?.frames ?? [];
    if (frames.length === 0) return null;
    const clampedIdx = Math.max(0, Math.min(frameIndex, frames.length - 1));
    const currentStepReward = frames[clampedIdx]?.reward;

    let cumProgress = 0;
    let cumPenned = 0;
    let cumCohesion = 0;
    let cumScatter = 0;
    let cumWall = 0;
    let cumStall = 0;
    let cumTotal = 0;

    for (let i = 0; i <= clampedIdx; i++) {
      const r = frames[i]?.reward;
      if (!r) continue;
      cumProgress += r.progress_to_pen ?? 0;
      cumPenned += (r.sheep_penned ?? 0) + (r.terminal_success ?? 0);
      cumCohesion += r.flock_cohesion ?? 0;
      cumScatter += r.scatter_penalty ?? 0;
      cumWall += r.wall_pressure_penalty ?? 0;
      cumStall +=
        (r.no_progress_penalty ?? 0) +
        (r.wait_penalty ?? 0) +
        (r.time_penalty ?? 0) +
        (r.terminal_failure ?? 0);
      cumTotal += r.total ?? 0;
    }

    return {
      stepTotal: currentStepReward?.total ?? 0,
      stepProgress: currentStepReward?.progress_to_pen ?? 0,
      stepPenned:
        (currentStepReward?.sheep_penned ?? 0) +
        (currentStepReward?.terminal_success ?? 0),
      stepCohesion: currentStepReward?.flock_cohesion ?? 0,
      stepScatter: currentStepReward?.scatter_penalty ?? 0,
      stepWall: currentStepReward?.wall_pressure_penalty ?? 0,
      stepStall:
        (currentStepReward?.no_progress_penalty ?? 0) +
        (currentStepReward?.wait_penalty ?? 0) +
        (currentStepReward?.time_penalty ?? 0) +
        (currentStepReward?.terminal_failure ?? 0),
      cumProgress,
      cumPenned,
      cumCohesion,
      cumScatter,
      cumWall,
      cumStall,
      cumTotal,
    };
  }, [activeReplay, frameIndex]);

  // Failure Mode Anatomy across the recent 10 evaluations in the active stage
  const failureAnatomy = useMemo(() => {
    const recentEvals = stageCheckpoints.slice(0, 10);
    const expectedSheep = currentSnapshot?.sheep?.length || 6;
    let totalTrials = 0;
    let totalLosses = 0;
    let stragglerLosses = 0; // >= 50% sheep penned, ran out of time on last 1-2 sheep
    let midHerdingStalls = 0; // 1..49% sheep penned
    let zeroPenBreakdowns = 0; // 0 sheep penned
    let timeoutLosses = 0;
    let stallLosses = 0;

    for (const cp of recentEvals) {
      for (const rec of cp.records ?? []) {
        totalTrials++;
        if (!rec.success) {
          totalLosses++;
          if (rec.stopped) stallLosses++;
          else if (rec.timeout) timeoutLosses++;
          const penned = rec.sheep_penned ?? (rec as any).penned ?? 0;
          if (penned >= Math.ceil(expectedSheep * 0.5)) {
            stragglerLosses++;
          } else if (penned > 0) {
            midHerdingStalls++;
          } else {
            zeroPenBreakdowns++;
          }
        }
      }
    }

    const worstSeeds = [...seedReliability.seeds]
      .sort((a, b) => a.recentSuccessRate - b.recentSuccessRate)
      .slice(0, 3);

    return {
      evalCount: recentEvals.length,
      totalTrials,
      totalLosses,
      stragglerLosses,
      midHerdingStalls,
      zeroPenBreakdowns,
      timeoutLosses,
      stallLosses,
      stragglerPct:
        totalLosses > 0 ? Math.round((stragglerLosses / totalLosses) * 100) : 0,
      midStallPct:
        totalLosses > 0 ? Math.round((midHerdingStalls / totalLosses) * 100) : 0,
      zeroPenPct:
        totalLosses > 0 ? Math.round((zeroPenBreakdowns / totalLosses) * 100) : 0,
      worstSeeds,
    };
  }, [stageCheckpoints, currentSnapshot?.sheep?.length, seedReliability.seeds]);

  // Batch & Eval Cadence metrics
  const rawCompletedEps =
    trainingStatus?.batch_completed_episodes ?? trainingStatus?.completed_episodes ?? 0;
  const rawTotalBatchEps =
    trainingStatus?.batch_total_episodes ??
    trainingStatus?.requested_episodes ??
    batchEpisodes;
  const stageTotalEps =
    trainingStatus?.current_stage_environment_episode ??
    trainingStatus?.total_episodes_trained ??
    activeCheckpoint?.checkpoint_episode ??
    0;

  const completedEps =
    rawTotalBatchEps <= 1 && stageTotalEps > 1
      ? Math.round(stageTotalEps)
      : Math.round(rawCompletedEps);
  const totalBatchEps =
    rawTotalBatchEps <= 1 && stageTotalEps > 1
      ? Math.round(stageTotalEps)
      : Math.max(1, Math.round(rawTotalBatchEps));
  const batchPct =
    rawTotalBatchEps > 1
      ? Math.min(100, Math.round((rawCompletedEps / rawTotalBatchEps) * 100))
      : isTraining
      ? 100
      : 0;

  const evalInterval = trainingStatus?.checkpoint_save_interval ?? 50;
  const epsSinceEval = trainingStatus?.episodes_since_latest_confidence_evaluation ?? 0;
  const epsUntilEval =
    trainingStatus?.episodes_until_next_evaluation ??
    Math.max(0, evalInterval - epsSinceEval);

  const rawSuccessRate =
    trainingStatus?.latest_success_rate ?? activeCheckpoint?.success_rate ?? 0;
  const latestSuccessPct = Math.round(
    rawSuccessRate <= 1.0 ? rawSuccessRate * 100 : rawSuccessRate
  );
  const rawBestRate =
    trainingStatus?.best_score ?? activeCheckpoint?.success_rate ?? 0;
  const bestSuccessPct = Math.round(rawBestRate <= 1.0 ? rawBestRate * 100 : rawBestRate);

  const rolloutWinPct =
    trainingStatus?.live_rollout_success_rate != null
      ? Math.round(
          trainingStatus.live_rollout_success_rate <= 1
            ? trainingStatus.live_rollout_success_rate * 100
            : trainingStatus.live_rollout_success_rate
        )
      : null;

  const medianPassSteps =
    efficiencyTrend.recentMedian != null
      ? Math.round(efficiencyTrend.recentMedian)
      : activeCheckpoint?.average_completion_steps != null
      ? Math.round(activeCheckpoint.average_completion_steps)
      : null;

  const stageSelectorOptions = useMemo(() => {
    const map = new Map<number, string>();
    for (const [k, v] of Object.entries(STAGE_DESCRIPTIONS)) {
      map.set(Number(k), v);
    }
    if (!map.has(effectiveStage)) {
      map.set(effectiveStage, `3 dogs · 6 sheep · Stage ${effectiveStage}`);
    }
    if (!map.has(selectedStage)) {
      map.set(selectedStage, `3 dogs · 6 sheep · Stage ${selectedStage}`);
    }
    return Array.from(map.entries()).sort((a, b) => a[0] - b[0]);
  }, [effectiveStage, selectedStage]);

  return (
    <div className="sd-studio-root">
      {/* 1. TOP COMMAND HEADER */}
      <header className="sd-header">
        <div className="sd-header__brand">
          <div className="sd-header__logo" aria-hidden="true" style={{ padding: 2 }}>
            <img
              src="/favicon-v2.svg"
              alt=""
              style={{ width: 28, height: 28, display: "block", borderRadius: 6 }}
            />
          </div>
          <div className="sd-header__title">Sheepdog Studio</div>

          <div className="sd-header__divider" />

          {/* Curriculum Stage Selector */}
          <div style={{ display: "flex", alignItems: "center", gap: "0.45rem" }}>
            {isReadOnlyPreview ? (
              <select
                className="sd-select"
                value="read_only_preview"
                disabled
                aria-label="Curriculum stage selector"
                title="Training and curriculum stage selection are disabled in read-only preview mode"
                style={{ maxWidth: 285, fontWeight: 600, opacity: 0.8 }}
              >
                <option value="read_only_preview">
                  Read-Only Preview — No Active Training Stage
                </option>
              </select>
            ) : (
              <select
                className="sd-select"
                value={selectedStage}
                onChange={(e) => {
                  const st = Number(e.target.value);
                  mutationEpochRef.current += 1;
                  userTouchedStageRef.current = true;
                  setSelectedStage(st);
                  const rec = RECOMMENDED_EPISODES_BY_STAGE[st];
                  if (rec) setBatchEpisodes(rec);
                }}
                aria-label="Curriculum stage selector"
                style={{ maxWidth: 285, fontWeight: 600 }}
              >
                {stageSelectorOptions.map(([st, desc]) => (
                  <option key={st} value={st}>
                    Stage {st}: {desc}
                  </option>
                ))}
              </select>
            )}

            {!isReadOnlyPreview && effectiveStage < 38 && (
              <button
                type="button"
                className={`sd-btn sd-btn--sm ${
                  promotionReady ? "sd-btn--emerald" : "sd-btn--ghost"
                }`}
                disabled={isTraining || actionBusy}
                onClick={() => handlePromoteStage(effectiveStage + 1)}
                title={
                  promotionReady
                    ? `Stage ${effectiveStage} passed promotion gate! Click to advance to Stage ${
                        effectiveStage + 1
                      }`
                    : `Manually promote to Stage ${effectiveStage + 1}`
                }
              >
                {promotionReady
                  ? `★ Promote → Stage ${effectiveStage + 1}`
                  : `Promote → ${effectiveStage + 1}`}
              </button>
            )}
          </div>
        </div>

        {/* Center 3-Workspace Switcher */}
        <nav className="sd-workspace-nav" aria-label="Studio workspace views">
          <button
            type="button"
            className={`sd-workspace-tab ${
              workspace === "command" ? "sd-workspace-tab--active" : ""
            }`}
            onClick={() => setWorkspace("command")}
          >
            <span>🎛️ Command Center</span>
          </button>
          <button
            type="button"
            className={`sd-workspace-tab ${
              workspace === "arena" ? "sd-workspace-tab--active" : ""
            }`}
            onClick={() => setWorkspace("arena")}
          >
            <span>🏟️ 10-Seed Evaluation Arena</span>
            {!isReadOnlyPreview && checkpointIndex !== null && (
              <span className="sd-workspace-tab__count">{latestSuccessPct}%</span>
            )}
          </button>
          <button
            type="button"
            className={`sd-workspace-tab ${
              workspace === "temporal" ? "sd-workspace-tab--active" : ""
            }`}
            onClick={() => setWorkspace("temporal")}
            title="Temporal Interaction Diagnostics"
          >
            <span>🔬 Temporal</span>
          </button>
          <button
            type="button"
            className={`sd-workspace-tab ${
              workspace === "analytics" ? "sd-workspace-tab--active" : ""
            }`}
            onClick={() => setWorkspace("analytics")}
          >
            <span>📊 Analytics & Progression</span>
          </button>
        </nav>

        {/* Right Live Status & Primary Action Controls */}
        <div className="sd-header__actions">
          {/* Live Status Indicator */}
          {backendOffline ? (
            activeReplay !== null || temporalCachedAvailable ? (
              <div
                className="sd-status-pill sd-status-pill--evaluating"
                data-testid="studio-connection-status"
                data-connection-state="offline_cached_replay"
                title="Backend is currently unreachable; previously loaded replay remains available in client memory"
              >
                <span className="sd-status-dot" />
                <span>BACKEND OFFLINE · CACHED REPLAY AVAILABLE</span>
              </div>
            ) : (
              <div
                className="sd-status-pill sd-status-pill--offline"
                data-testid="studio-connection-status"
                data-connection-state="backend_offline"
              >
                <span className="sd-status-dot" />
                <span>BACKEND UNREACHABLE</span>
              </div>
            )
          ) : backendStarting ? (
            <div
              className="sd-status-pill sd-status-pill--evaluating"
              data-testid="studio-connection-status"
              data-connection-state="backend_starting"
            >
              <span className="sd-status-dot" />
              <span>STARTING SERVER...</span>
            </div>
          ) : isReadOnlyPreview ? (
            <div
              className="sd-status-pill sd-status-pill--idle"
              data-testid="studio-connection-status"
              data-connection-state="read_only_preview"
              style={{
                background: "rgba(14, 165, 233, 0.16)",
                borderColor: "rgba(56, 189, 248, 0.45)",
                color: "#7dd3fc",
              }}
              title="Connected to read-only diagnostic preview backend (TrainingManager and runtime mutations disabled)"
            >
              <span className="sd-status-dot" style={{ background: "#38bdf8" }} />
              <span>CONNECTED · READ-ONLY PREVIEW</span>
            </div>
          ) : isEvaluating ? (
            <div
              className="sd-status-pill sd-status-pill--evaluating"
              data-testid="studio-connection-status"
              data-connection-state="connected_normal"
            >
              <span className="sd-status-dot" />
              <span>
                EVALUATING 10 SEEDS (
                {trainingStatus?.evaluation_completed_seeds ??
                  trainingStatus?.evaluation_status?.completed_seeds ??
                  0}
                /
                {trainingStatus?.evaluation_total_seeds ??
                  trainingStatus?.evaluation_status?.total_seeds ??
                  10}
                )
              </span>
            </div>
          ) : isTraining ? (
            <div
              className="sd-status-pill sd-status-pill--running"
              data-testid="studio-connection-status"
              data-connection-state="connected_normal"
            >
              <span className="sd-status-dot" />
              <span>
                TRAINING · Ep {completedEps}/{totalBatchEps}
              </span>
            </div>
          ) : (
            <div
              className="sd-status-pill sd-status-pill--idle"
              data-testid="studio-connection-status"
              data-connection-state="connected_normal"
            >
              <span className="sd-status-dot" />
              <span>{canResume ? "CONNECTED · PAUSED" : "CONNECTED · READY"}</span>
            </div>
          )}

          {/* Primary Lifecycle Buttons */}
          {isReadOnlyPreview ? (
            <button
              type="button"
              data-testid="studio-train-btn"
              className="sd-btn sd-btn--ghost"
              disabled
              title="Training and runtime mutations are disabled in read-only preview mode to protect active runs and checkpoints."
              style={{ opacity: 0.65, cursor: "not-allowed" }}
            >
              🔒 Train Disabled (Preview Mode)
            </button>
          ) : !isTraining ? (
            <>
              <button
                type="button"
                data-testid="studio-train-btn"
                className="sd-btn sd-btn--primary"
                disabled={actionBusy || backendOffline}
                onClick={handleStartTraining}
              >
                ▶ Train ({batchEpisodes} ep)
              </button>
              {canResume && (
                <button
                  type="button"
                  className="sd-btn sd-btn--emerald"
                  disabled={actionBusy}
                  onClick={handleResumeTraining}
                >
                  ▶ Resume ({remainingResumeEps} ep)
                </button>
              )}
            </>
          ) : (
            <>
              <button
                type="button"
                className="sd-btn sd-btn--amber"
                disabled={actionBusy}
                onClick={handlePauseTraining}
                title="Pause cleanly after the next evaluation checkpoint"
              >
                ⏸ Pause
              </button>
              <button
                type="button"
                className="sd-btn sd-btn--danger"
                disabled={actionBusy}
                onClick={handleStopTraining}
                title="Stop cleanly after the next evaluation checkpoint"
              >
                ⏹ Stop
              </button>
            </>
          )}

          <CopyAgentDataButton
            trainingStatus={trainingStatus}
            checkpointIndex={checkpointIndex}
            curriculumStage={effectiveStage}
          />

          <a
            href="/"
            target="_blank"
            rel="noopener noreferrer"
            className="sd-btn sd-btn--sm sd-btn--ghost"
            style={{ textDecoration: "none" }}
            title="Open the original Classic UI in a separate tab (both run simultaneously)"
          >
            Classic UI ↗
          </a>
        </div>
      </header>

      {/* 2. FOCUSED 3-CARD KPI & PROMOTION GATE RIBBON (OR READ-ONLY PREVIEW STRIP) */}
      {workspace === "temporal" ? (
        isReadOnlyPreview ? (
          <section
            aria-label="Read-only preview status"
            data-testid="studio-readonly-preview-strip"
            style={{
              display: "flex",
              flexWrap: "wrap",
              alignItems: "center",
              justifyContent: "space-between",
              gap: "0.5rem",
              padding: "0.22rem 1.1rem",
              background: "rgba(9, 14, 24, 0.82)",
              borderBottom: "1px solid var(--sd-border-subtle)",
              fontSize: "0.72rem",
              color: "var(--sd-text-secondary)",
              flexShrink: 0,
            }}
          >
            <span>
              <strong style={{ color: "var(--sd-text-primary)" }}>
                {backendOffline
                  ? "Backend Disconnected (Cached Replay Active)"
                  : "Connected to Read-Only Diagnostic Preview"}
              </strong>{" "}
              — Training &amp; runtime mutations disabled (HTTP 403); default &amp; stale training KPIs suppressed.
            </span>
            <span className="sd-mono" style={{ fontSize: "0.69rem", color: "var(--sd-cyan)" }}>
              GET /api/health → read_only_preview
            </span>
          </section>
        ) : null
      ) : isReadOnlyPreview ? (
        <section
          className="sd-kpi-ribbon"
          aria-label="Read-only preview status"
          data-testid="studio-readonly-preview-strip"
        >
          <div className="sd-kpi-card">
            <div className="sd-kpi-card__header">
              <span className="sd-kpi-card__label">Preview Connection Status</span>
              <span
                className={`sd-badge ${
                  backendOffline ? "sd-badge--amber" : "sd-badge--cyan"
                }`}
              >
                {backendOffline
                  ? "Offline · Cached Replay"
                  : "Healthy · Read-Only"}
              </span>
            </div>
            <div className="sd-kpi-card__value-row">
              <span className="sd-kpi-card__value" style={{ fontSize: "1.02rem" }}>
                {backendOffline
                  ? "Backend Disconnected (Cached Replay Active)"
                  : "Connected to Read-Only Diagnostic Preview"}
              </span>
            </div>
            <div className="sd-kpi-card__sub">
              <span>
                No <code>TrainingManager</code> or checkpoint restore required
              </span>
              <span className="sd-mono">GET /api/health → read_only_preview</span>
            </div>
          </div>

          <div className="sd-kpi-card">
            <div className="sd-kpi-card__header">
              <span className="sd-kpi-card__label">Training &amp; Runtime Mutations</span>
              <span className="sd-badge sd-badge--indigo">Protected</span>
            </div>
            <div className="sd-kpi-card__value-row">
              <span className="sd-kpi-card__value" style={{ fontSize: "1.02rem" }}>
                Training Controls Disabled in Preview
              </span>
            </div>
            <div className="sd-kpi-card__sub">
              <span>All POST mutations return HTTP 403</span>
              <span>Active runs &amp; checkpoints untouched</span>
            </div>
          </div>

          <div className="sd-kpi-card">
            <div className="sd-kpi-card__header">
              <span className="sd-kpi-card__label">Live Training Telemetry</span>
              <span className="sd-badge sd-badge--emerald">Epistemic Guard Active</span>
            </div>
            <div className="sd-kpi-card__value-row">
              <span className="sd-kpi-card__value" style={{ fontSize: "1.02rem" }}>
                No Active Training Session in Preview
              </span>
            </div>
            <div className="sd-kpi-card__sub">
              <span>Default &amp; stale training KPIs suppressed</span>
              <span>Showing verified recorded episode evidence only</span>
            </div>
          </div>
        </section>
      ) : (
        <section className="sd-kpi-ribbon" aria-label="Key training metrics">
        {/* Card 1: Batch & Next Eval Cadence */}
        <div className="sd-kpi-card">
          <div className="sd-kpi-card__header">
            <span className="sd-kpi-card__label">Training Batch & Eval Cadence</span>
            <span className="sd-badge sd-badge--indigo">
              {trainingStatus?.runtime?.episodes_per_active_hour
                ? `${(trainingStatus.runtime.episodes_per_active_hour / 60).toFixed(1)} ep/m`
                : isTraining
                ? "Active"
                : "Idle"}
            </span>
          </div>
          <div className="sd-kpi-card__value-row">
            <span className="sd-kpi-card__value">
              {completedEps.toLocaleString()}{" "}
              <span style={{ fontSize: "0.78rem", color: "var(--sd-text-muted)" }}>
                / {totalBatchEps.toLocaleString()} eps
              </span>
            </span>
            <span className="sd-kpi-card__delta" style={{ color: "var(--sd-text-secondary)" }}>
              Runtime {formatDuration(trainingStatus?.runtime?.active_seconds_total)}
            </span>
          </div>
          <div className="sd-progress-track" title={`Batch progress: ${batchPct}%`}>
            <div
              className="sd-progress-fill sd-progress-fill--indigo"
              style={{ width: `${batchPct}%` }}
            />
          </div>
          <div className="sd-kpi-card__sub">
            <span>Next 10-Seed Eval in {epsUntilEval} eps</span>
            <span className="sd-mono">{allCheckpoints.length} ckpts saved</span>
          </div>
        </div>

        {/* Card 2: Combined 10-Seed Eval Pass % + Median Completion Steps */}
        <div
          className="sd-kpi-card"
          style={{ cursor: "pointer" }}
          onClick={() => setWorkspace("arena")}
          title="Click to inspect all 10 benchmark seeds side-by-side in the Evaluation Arena"
        >
          <div className="sd-kpi-card__header">
            <span className="sd-kpi-card__label">10-Seed Eval & Step Efficiency</span>
            <div style={{ display: "flex", alignItems: "center", gap: "0.3rem" }}>
              <span
                className={`sd-badge ${
                  latestSuccessPct >= 80
                    ? "sd-badge--emerald"
                    : latestSuccessPct >= 50
                    ? "sd-badge--amber"
                    : "sd-badge--rose"
                }`}
              >
                Peak {bestSuccessPct}%
              </span>
              {efficiencyTrend.status === "improving" && (
                <span className="sd-badge sd-badge--cyan">
                  ↓ {efficiencyTrend.percentageImprovement}% FASTER
                </span>
              )}
              {efficiencyTrend.status === "regressing" && (
                <span className="sd-badge sd-badge--rose">↑ SLOWER</span>
              )}
            </div>
          </div>
          <div className="sd-kpi-card__value-row">
            <span
              className="sd-kpi-card__value"
              style={{
                color:
                  latestSuccessPct >= 80
                    ? "var(--sd-emerald-bright)"
                    : latestSuccessPct >= 50
                    ? "var(--sd-amber-bright)"
                    : "var(--sd-text-primary)",
              }}
            >
              {latestSuccessPct}%{" "}
              <span style={{ fontSize: "0.78rem", color: "var(--sd-text-secondary)" }}>
                ({Math.round(latestSuccessPct / 10)}/10)
              </span>
            </span>
            <span className="sd-kpi-card__delta sd-mono" style={{ color: "var(--sd-cyan)" }}>
              {medianPassSteps != null ? `${medianPassSteps} median steps` : "— steps"}
            </span>
          </div>
          <div className="sd-progress-track">
            <div
              className="sd-progress-fill sd-progress-fill--emerald"
              style={{ width: `${Math.min(100, latestSuccessPct)}%` }}
            />
          </div>
          <div className="sd-kpi-card__sub">
            <span>
              {rolloutWinPct != null
                ? `Stochastic Rollout Win: ${rolloutWinPct}%`
                : `Prior Median: ${
                    efficiencyTrend.priorMedian != null
                      ? `${Math.round(efficiencyTrend.priorMedian)}s`
                      : "—"
                  }`}
            </span>
            <span style={{ color: "var(--sd-accent-bright)" }}>Open 10-Seed Arena →</span>
          </div>
        </div>

        {/* Card 3: Stage Promotion Gate (Single Source of Truth + Hover Checklist) */}
        <div className="sd-kpi-card sd-gate-popover-trigger" tabIndex={0}>
          <div className="sd-kpi-card__header">
            <span className="sd-kpi-card__label">Stage {effectiveStage} Promotion Gate</span>
            <span
              className={`sd-badge ${
                promotionReady
                  ? "sd-badge--emerald"
                  : gateDiag?.step_efficiency_improving
                  ? "sd-badge--cyan"
                  : "sd-badge--amber"
              }`}
            >
              {promotionReady
                ? "✓ READY TO PROMOTE"
                : gateDiag?.step_efficiency_improving
                ? "⚡ OPTIMIZING STEPS"
                : "BUILDING EVIDENCE"}
            </span>
          </div>
          <div className="sd-kpi-card__value-row">
            <span className="sd-kpi-card__value">
              {gateChecks.length > 0 ? `${passedGateCheckCount}/${gateChecks.length}` : "—"}
            </span>
            <span className="sd-kpi-card__delta" style={{ color: "var(--sd-text-secondary)" }}>
              gate checks passed ▾
            </span>
          </div>
          <div className="sd-progress-track">
            <div
              className={`sd-progress-fill ${
                promotionReady ? "sd-progress-fill--emerald" : "sd-progress-fill--amber"
              }`}
              style={{
                width: `${
                  gateChecks.length > 0
                    ? Math.round((passedGateCheckCount / gateChecks.length) * 100)
                    : 0
                }%`,
              }}
            />
          </div>
          <div className="sd-kpi-card__sub">
            <span>
              Blind spots: {seedReliability.blindSpotCount} seed
              {seedReliability.blindSpotCount === 1 ? "" : "s"}
            </span>
            <span>
              Qualified: {gateDiag?.qualified_evaluations ?? 0}/
              {gateDiag?.qualified_evaluations_required ?? 2} (Hover)
            </span>
          </div>

          {/* Hover Popover with Full Promotion Gate Checklist */}
          <div className="sd-gate-popover" role="tooltip">
            <div
              style={{
                fontSize: "0.75rem",
                fontWeight: 700,
                color: "var(--sd-text-primary)",
                marginBottom: "0.45rem",
                display: "flex",
                justifyContent: "space-between",
              }}
            >
              <span>Stage {effectiveStage} Promotion Gate</span>
              <span className="sd-mono">
                Target ≥
                {Math.round(
                  (gateDiag?.success_threshold ?? 0.8) <= 1
                    ? (gateDiag?.success_threshold ?? 0.8) * 100
                    : gateDiag?.success_threshold ?? 80
                )}
                %
              </span>
            </div>
            <div className="sd-gate-checklist">
              {gateChecks.length === 0 ? (
                <div style={{ fontSize: "0.72rem", color: "var(--sd-text-muted)" }}>
                  Waiting for first formal 10-seed evaluation in Stage {effectiveStage}.
                </div>
              ) : (
                gateChecks.map((chk) => (
                  <div key={chk.label} className="sd-gate-check-item">
                    <span className="sd-gate-check-item__label">
                      <span
                        style={{
                          color: chk.passed
                            ? "var(--sd-emerald-bright)"
                            : "var(--sd-amber-bright)",
                        }}
                      >
                        {chk.passed ? "✓" : "○"}
                      </span>
                      <span>{chk.label}</span>
                    </span>
                    <span className="sd-gate-check-item__val">{chk.value}</span>
                  </div>
                ))
              )}
            </div>
          </div>
        </div>

        {/* Card 4: Early Warnings (Warning-Only, No Auto-Stop) */}
        <div
          className="sd-kpi-card"
          tabIndex={0}
          style={{ cursor: "pointer" }}
          onClick={() => setEarlyWarningsOpen((prev) => !prev)}
          onKeyDown={(e) => {
            if (e.key === "Enter" || e.key === " ") {
              e.preventDefault();
              setEarlyWarningsOpen((prev) => !prev);
            }
          }}
          data-testid="studio-early-warnings-kpi-card"
        >
          <div className="sd-kpi-card__header">
            <span className="sd-kpi-card__label">Early Warnings</span>
            <span
              className={`sd-badge ${
                (trainingStatus?.early_warnings_summary?.critical_count ?? 0) > 0
                  ? "sd-badge--rose"
                  : (trainingStatus?.early_warnings_summary?.warning_count ?? 0) > 0
                  ? "sd-badge--amber"
                  : "sd-badge--emerald"
              }`}
            >
              {(trainingStatus?.early_warnings_summary?.total_active ?? 0) === 0
                ? "✓ Normal"
                : `${trainingStatus?.early_warnings_summary?.total_active} Active`}
            </span>
          </div>
          <div className="sd-kpi-card__value-row">
            <span className="sd-kpi-card__value">
              {(trainingStatus?.early_warnings_summary?.total_active ?? 0) === 0
                ? "All Clear"
                : `${trainingStatus?.early_warnings_summary?.total_active} Warning${
                    (trainingStatus?.early_warnings_summary?.total_active ?? 0) === 1 ? "" : "s"
                  }`}
            </span>
            <span className="sd-kpi-card__delta" style={{ color: "var(--sd-cyan)" }}>
              {earlyWarningsOpen ? "Hide ▲" : "Inspect ▼"}
            </span>
          </div>
          <div className="sd-kpi-card__sub">
            <span>Warning-Only · No Auto-Stop</span>
            <span style={{ color: "var(--sd-accent-bright)" }}>
              {trainingStatus?.early_warnings_monitoring_status === "read_only_preview" || isReadOnlyPreview
                ? "Preview Mode"
                : "Live Monitor"}
            </span>
          </div>
        </div>
      </section>
      )}

      {/* Optional Transient Action Feedback Banner */}
      {actionMessage && (
        <div
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            padding: "0.35rem 1.25rem",
            background: "rgba(99, 102, 241, 0.14)",
            borderBottom: "1px solid rgba(99, 102, 241, 0.3)",
            fontSize: "0.75rem",
            color: "var(--sd-text-primary)",
          }}
        >
          <span>{actionMessage}</span>
          <button
            type="button"
            className="sd-btn sd-btn--sm sd-btn--ghost"
            onClick={() => setActionMessage(null)}
          >
            ✕
          </button>
        </div>
      )}

      {/* EARLY WARNINGS PANEL */}
      <div style={{ padding: "0 1.25rem 0.5rem 1.25rem" }}>
        <EarlyWarningsPanel
          curriculumStage={effectiveStage}
          isOpen={earlyWarningsOpen}
          onToggle={() => setEarlyWarningsOpen((prev) => !prev)}
          isReadOnlyPreview={isReadOnlyPreview}
          onOpenReplay={(replayPath) => {
            const m = replayPath.match(/(?:checkpoint-\d+-seed-\d+|diag_[^./]+)/);
            if (m) {
              setTemporalReplayId(m[0]);
              setWorkspace("temporal");
            }
          }}
        />
      </div>

      {/* 3. MAIN WORKSPACE AREA */}
      <main className="sd-main">
        {/* ============================================================
            WORKSPACE 1: COMMAND CENTER (Unified Split View)
           ============================================================ */}
        {workspace === "command" && (
          <div className="sd-command-center">
            {/* LEFT PANE: Interactive Herding Field Player & Unified 10-Seed Strip */}
            <section className="sd-field-panel" aria-label="Interactive episode replay player">
              <div
                className="sd-panel-header"
                style={{ flexWrap: "nowrap", gap: "0.4rem", overflowX: "auto" }}
              >
                <div className="sd-panel-title" style={{ flexShrink: 0 }}>
                  <span className="sd-panel-title__icon">🎬</span>
                  <span style={{ whiteSpace: "nowrap" }}>Field Player</span>
                  <span
                    className="sd-badge sd-badge--indigo sd-mono"
                    style={{
                      minWidth: 104,
                      justifyContent: "center",
                      textAlign: "center",
                    }}
                    title="Active checkpoint episode (select any point on the chart or recent evals strip to switch)"
                  >
                    {activeCheckpoint
                      ? `Ep ${activeCheckpoint.checkpoint_episode}${
                          activeCheckpoint.checkpoint_episode ===
                          (checkpointIndex as any)?.best_checkpoint_episode
                            ? " ★"
                            : ""
                        }`
                      : "Ep —"}
                  </span>
                </div>

                <div
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: "0.32rem",
                    flexWrap: "nowrap",
                    flexShrink: 0,
                  }}
                >
                  {/* Outcome Filter: All | Losses Only | Wins Only */}
                  <div
                    className="sd-pill-group"
                    role="group"
                    aria-label="Filter episodes shown in player"
                    style={{ flexShrink: 0 }}
                  >
                    <button
                      type="button"
                      className={`sd-pill-btn ${
                        episodeViewFilter === "all" ? "sd-pill-btn--active" : ""
                      }`}
                      style={{ width: 34, textAlign: "center", padding: "0.18rem 0" }}
                      onClick={() => setEpisodeViewFilter("all")}
                      title="Show all evaluation seeds"
                    >
                      All
                    </button>
                    <button
                      type="button"
                      className={`sd-pill-btn ${
                        episodeViewFilter === "losses" ? "sd-pill-btn--active" : ""
                      }`}
                      style={{
                        width: 68,
                        textAlign: "center",
                        padding: "0.18rem 0",
                        color:
                          episodeViewFilter === "losses"
                            ? "#ffffff"
                            : "var(--sd-rose-bright)",
                      }}
                      onClick={() => setEpisodeViewFilter("losses")}
                      title={`Only play failed seeds (${lossSeedCount} failed in this evaluation)`}
                    >
                      ✗ Losses
                    </button>
                    <button
                      type="button"
                      className={`sd-pill-btn ${
                        episodeViewFilter === "wins" ? "sd-pill-btn--active" : ""
                      }`}
                      style={{
                        width: 56,
                        textAlign: "center",
                        padding: "0.18rem 0",
                        color:
                          episodeViewFilter === "wins"
                            ? "#ffffff"
                            : "var(--sd-emerald-bright)",
                      }}
                      onClick={() => setEpisodeViewFilter("wins")}
                      title="Only play passing seeds"
                    >
                      ✓ Wins
                    </button>
                  </div>

                  {/* Lock Episode to Finish Toggle (Fixed Width) */}
                  <button
                    type="button"
                    className={`sd-btn sd-btn--sm ${
                      lockUntilFinish ? "sd-btn--indigo" : "sd-btn--ghost"
                    }`}
                    style={{
                      width: 84,
                      minWidth: 84,
                      justifyContent: "center",
                      padding: "0.22rem 0.35rem",
                    }}
                    onClick={() => setLockUntilFinish((l) => !l)}
                    title="Lock current episode to play all the way to its final step before switching to a newer checkpoint"
                  >
                    🔒 Finish Ep
                  </button>

                  {/* Live Follow Toggle (Fixed Width & Constant Label) */}
                  <button
                    type="button"
                    className={`sd-btn sd-btn--sm ${
                      autoFollowLive ? "sd-btn--emerald" : "sd-btn--ghost"
                    }`}
                    style={{
                      width: 78,
                      minWidth: 78,
                      justifyContent: "center",
                      padding: "0.22rem 0.35rem",
                    }}
                    onClick={() => setAutoFollowLive((f) => !f)}
                    title="Automatically follow newest checkpoint after current episode finishes"
                  >
                    📡 Follow
                  </button>

                  {/* Run Live Episode (Fixed Width & Constant Label) */}
                  <button
                    type="button"
                    className="sd-btn sd-btn--sm sd-btn--primary"
                    style={{
                      width: 88,
                      minWidth: 88,
                      justifyContent: "center",
                      padding: "0.22rem 0.35rem",
                    }}
                    disabled={replayLoading}
                    onClick={handleRunLiveEpisode}
                    title="Simulate a fresh episode on-demand with current dogs"
                  >
                    ⚡ Run Live
                  </button>

                  {/* Open Temporal Diagnostics for Active Episode */}
                  <button
                    type="button"
                    className="sd-btn sd-btn--sm sd-btn--ghost"
                    style={{
                      minWidth: 118,
                      justifyContent: "center",
                      padding: "0.22rem 0.45rem",
                      borderColor: "rgba(56, 189, 248, 0.45)",
                      color: "#bae6fd",
                    }}
                    onClick={() => {
                      const rec = (activeCheckpoint?.records ?? []).find(
                        (r) => r.seed === selectedSeed
                      ) as any;
                      let candidateId: string | null = null;
                      if (rec?.replay_id) {
                        candidateId = String(rec.replay_id);
                      } else if (rec?.replay_path) {
                        const base = String(rec.replay_path)
                          .split(/[\\/]/)
                          .pop();
                        if (base) candidateId = base.replace(/\.json$/i, "");
                      } else if (activeCheckpoint?.checkpoint_episode != null && selectedSeed != null) {
                        candidateId = `checkpoint-${String(
                          activeCheckpoint.checkpoint_episode
                        ).padStart(6, "0")}-seed-${String(selectedSeed).padStart(6, "0")}`;
                      }
                      if (candidateId) {
                        setTemporalReplayId(candidateId);
                      }
                      setWorkspace("temporal");
                    }}
                    title="Open synchronized Temporal Interaction Diagnostics for this episode or curated examples"
                  >
                    🔬 Temporal Evidence
                  </button>
                </div>
              </div>

              {/* SVG Field Viewport with Streamlined Fixed-Width HUD */}
              <div className="sd-field-viewport">
                {currentSnapshot && (
                  <>
                    <div className="sd-field-hud-top-left">
                      <div
                        className="sd-hud-pill sd-mono"
                        style={{
                          display: "inline-flex",
                          alignItems: "center",
                          gap: "0.22rem",
                          padding: "0.14rem 0.35rem",
                          pointerEvents: "auto",
                        }}
                      >
                        <button
                          type="button"
                          onClick={() => {
                            const list =
                              filteredSeedsForCheckpoint.length > 0
                                ? filteredSeedsForCheckpoint
                                : activeSeeds;
                            if (list.length === 0) return;
                            const idx = list.indexOf(selectedSeed);
                            const prevSeed =
                              list[(idx > 0 ? idx - 1 : list.length - 1)];
                            handlePickSeed(prevSeed);
                          }}
                          style={{
                            background: "transparent",
                            border: "none",
                            color: "var(--sd-text-secondary)",
                            cursor: "pointer",
                            padding: "0 0.18rem",
                            fontSize: "0.8rem",
                            lineHeight: 1,
                          }}
                          title="Previous evaluation seed"
                          aria-label="Previous seed"
                        >
                          ‹
                        </button>
                        <select
                          aria-label="Select replay seed"
                          value={selectedSeed}
                          onChange={(e) => handlePickSeed(Number(e.target.value))}
                          style={{
                            background: "transparent",
                            border: "none",
                            color: "var(--sd-text-primary)",
                            fontFamily: "var(--sd-font-mono)",
                            fontWeight: 700,
                            fontSize: "0.72rem",
                            cursor: "pointer",
                            outline: "none",
                            padding: "0 0.1rem",
                          }}
                          title="Pick evaluation seed to replay"
                        >
                          {activeSeeds.map((seed) => {
                            const rec = (activeCheckpoint?.records ?? []).find(
                              (r) => r.seed === seed
                            );
                            const mark = rec
                              ? rec.success
                                ? " ✓"
                                : " ✗"
                              : "";
                            return (
                              <option
                                key={seed}
                                value={seed}
                                style={{ background: "#0f172a", color: "#f8fafc" }}
                              >
                                Seed #{seed}
                                {mark}
                              </option>
                            );
                          })}
                        </select>
                        <button
                          type="button"
                          onClick={() => {
                            const list =
                              filteredSeedsForCheckpoint.length > 0
                                ? filteredSeedsForCheckpoint
                                : activeSeeds;
                            if (list.length === 0) return;
                            const idx = list.indexOf(selectedSeed);
                            const nextSeed = list[(idx + 1) % list.length];
                            handlePickSeed(nextSeed);
                          }}
                          style={{
                            background: "transparent",
                            border: "none",
                            color: "var(--sd-text-secondary)",
                            cursor: "pointer",
                            padding: "0 0.18rem",
                            fontSize: "0.8rem",
                            lineHeight: 1,
                          }}
                          title="Next evaluation seed"
                          aria-label="Next seed"
                        >
                          ›
                        </button>
                      </div>
                      <span
                        className="sd-hud-pill sd-mono"
                        style={{ minWidth: 112, textAlign: "center" }}
                      >
                        Step <strong>{currentSnapshot.step}</strong> /{" "}
                        {Math.max(0, totalFrames - 1)}
                      </span>
                      {replayLoading && (
                        <span
                          className="sd-badge sd-badge--indigo sd-mono"
                          style={{ minWidth: 88, justifyContent: "center" }}
                        >
                          Loading #{selectedSeed}…
                        </span>
                      )}
                      {episodeViewFilter === "losses" && (
                        <span
                          className="sd-badge sd-badge--rose"
                          style={{ minWidth: 92, justifyContent: "center" }}
                        >
                          {lossSeedCount > 0
                            ? `✗ LOSSES (${lossSeedCount})`
                            : "★ 0 LOSSES"}
                        </span>
                      )}
                    </div>

                    <div className="sd-field-hud-top-right">
                      <span
                        className="sd-hud-pill sd-mono"
                        style={{ minWidth: 104, textAlign: "center" }}
                      >
                        🐑 Penned{" "}
                        <strong>
                          {currentSnapshot.penned_count ?? 0} /{" "}
                          {currentSnapshot.sheep?.length ?? 0}
                        </strong>
                      </span>
                      {(activeReplay?.stats || activeReplay?.summary) && (
                        <span
                          className={`sd-badge ${
                            (activeReplay.stats?.success ?? activeReplay.summary?.success)
                              ? "sd-badge--emerald"
                              : "sd-badge--rose"
                          }`}
                          style={{ minWidth: 108, justifyContent: "center" }}
                        >
                          {(activeReplay.stats?.success ?? activeReplay.summary?.success)
                            ? "✓ EPISODE PASS"
                            : "✗ EPISODE FAIL"}
                        </span>
                      )}
                    </div>
                  </>
                )}

                <StudioFieldCanvas
                  snapshot={currentSnapshot}
                  compact={false}
                  showRoles={showRoles}
                  showVectors={showVectors}
                  uniqueId="command-main"
                />
              </div>

              {/* Video Scrubber + Live Frame Reward Breakdown + Unified 10-Seed Strip */}
              <div className="sd-transport-bar">
                <div className="sd-transport-scrubber-row">
                  <button
                    type="button"
                    className="sd-btn sd-btn--sm sd-btn--icon"
                    onClick={() => setFrameIndex(0)}
                    title="Jump to start"
                  >
                    ⏮
                  </button>
                  <button
                    type="button"
                    className="sd-btn sd-btn--sm sd-btn--icon"
                    onClick={() => {
                      setIsPlaying(false);
                      setFrameIndex((f) => Math.max(0, f - 1));
                    }}
                    title="Step back 1 frame"
                  >
                    ◀
                  </button>
                  <button
                    type="button"
                    className="sd-btn sd-btn--sm sd-btn--primary"
                    onClick={() => setIsPlaying((p) => !p)}
                    style={{ width: 68, minWidth: 68, justifyContent: "center" }}
                  >
                    {isPlaying ? "⏸ Pause" : "▶ Play"}
                  </button>
                  <button
                    type="button"
                    className="sd-btn sd-btn--sm sd-btn--icon"
                    onClick={() => {
                      setIsPlaying(false);
                      setFrameIndex((f) => Math.min(Math.max(0, totalFrames - 1), f + 1));
                    }}
                    title="Step forward 1 frame"
                  >
                    ▶
                  </button>

                  <input
                    type="range"
                    className="sd-scrubber-slider"
                    min={0}
                    max={Math.max(0, totalFrames - 1)}
                    value={Math.min(frameIndex, Math.max(0, totalFrames - 1))}
                    onChange={(e) => {
                      setIsPlaying(false);
                      setFrameIndex(Number(e.target.value));
                    }}
                    aria-label="Episode frame scrubber"
                  />

                  <div className="sd-pill-group">
                    {[0.5, 1, 2, 4].map((spd) => (
                      <button
                        key={spd}
                        type="button"
                        className={`sd-pill-btn ${
                          playbackSpeed === spd ? "sd-pill-btn--active" : ""
                        }`}
                        onClick={() => setPlaybackSpeed(spd)}
                      >
                        {spd}x
                      </button>
                    ))}
                  </div>

                  <button
                    type="button"
                    className={`sd-btn sd-btn--sm ${
                      rotateSeeds ? "sd-btn--indigo" : "sd-btn--ghost"
                    }`}
                    style={{ minWidth: 74, justifyContent: "center", padding: "0.22rem 0.4rem" }}
                    onClick={() => setRotateSeeds((r) => !r)}
                    title={
                      rotateSeeds
                        ? "Auto-rotating through seeds after each episode finishes (click to hold current seed)"
                        : "Holding current seed (click to auto-rotate through all seeds)"
                    }
                  >
                    {rotateSeeds ? "🔀 Rotate" : "🔂 Hold"}
                  </button>

                  <button
                    type="button"
                    className={`sd-btn sd-btn--sm ${
                      loopReplay ? "sd-btn--emerald" : "sd-btn--ghost"
                    }`}
                    onClick={() => setLoopReplay((l) => !l)}
                    title="Loop episode playback"
                  >
                    🔁
                  </button>
                </div>

                {/* Live Step & Cumulative Reward Anatomy Bar (Explains WHY the agent gets rewarded/penalized at each step) */}
                {liveRewardSummary && (
                  <div
                    className="sd-mono"
                    style={{
                      display: "grid",
                      gridTemplateColumns: "repeat(7, minmax(0, 1fr))",
                      gap: "0.28rem",
                      padding: "0.3rem 0.45rem",
                      background: "rgba(9, 13, 22, 0.78)",
                      border: "1px solid var(--sd-border-subtle)",
                      borderRadius: "var(--sd-radius-sm)",
                      fontSize: "0.66rem",
                    }}
                    title="Live cumulative reward breakdown up to current step (with instant step delta)"
                  >
                    {[
                      {
                        label: "Drive",
                        cum: liveRewardSummary.cumProgress,
                        step: liveRewardSummary.stepProgress,
                        tip: "Progress driving flock toward pen gate",
                      },
                      {
                        label: "Penned",
                        cum: liveRewardSummary.cumPenned,
                        step: liveRewardSummary.stepPenned,
                        tip: "Sheep penned & terminal completion bonus",
                      },
                      {
                        label: "Cohesion",
                        cum: liveRewardSummary.cumCohesion,
                        step: liveRewardSummary.stepCohesion,
                        tip: "Flock grouping / cohesion reward",
                      },
                      {
                        label: "Scatter",
                        cum: liveRewardSummary.cumScatter,
                        step: liveRewardSummary.stepScatter,
                        tip: "Flock split / scatter penalty",
                      },
                      {
                        label: "Wall Pin",
                        cum: liveRewardSummary.cumWall,
                        step: liveRewardSummary.stepWall,
                        tip: "Penalty for pinning sheep against outer walls",
                      },
                      {
                        label: "Stall/Time",
                        cum: liveRewardSummary.cumStall,
                        step: liveRewardSummary.stepStall,
                        tip: "No-progress, waiting, and step time penalties",
                      },
                      {
                        label: "Net Return",
                        cum: liveRewardSummary.cumTotal,
                        step: liveRewardSummary.stepTotal,
                        tip: "Cumulative episode return up to this step",
                        isTotal: true,
                      },
                    ].map((item) => {
                      const valColor =
                        item.cum > 0.05
                          ? "var(--sd-emerald-bright)"
                          : item.cum < -0.05
                          ? "var(--sd-rose-bright)"
                          : "var(--sd-text-secondary)";
                      const stepActive = Math.abs(item.step) >= 0.01;
                      return (
                        <div
                          key={item.label}
                          title={`${item.tip} | Step Δ: ${
                            item.step >= 0 ? "+" : ""
                          }${item.step.toFixed(2)}`}
                          style={{
                            display: "flex",
                            flexDirection: "column",
                            alignItems: "center",
                            padding: "0.14rem 0.2rem",
                            borderRadius: 4,
                            background: item.isTotal
                              ? "rgba(99, 102, 241, 0.14)"
                              : stepActive
                              ? "rgba(255, 255, 255, 0.04)"
                              : "transparent",
                            border: item.isTotal
                              ? "1px solid rgba(99, 102, 241, 0.35)"
                              : "1px solid transparent",
                          }}
                        >
                          <span
                            style={{
                              fontSize: "0.58rem",
                              color: "var(--sd-text-muted)",
                              textTransform: "uppercase",
                              letterSpacing: "0.03em",
                            }}
                          >
                            {item.label}
                          </span>
                          <strong style={{ color: valColor }}>
                            {item.cum >= 0 ? "+" : ""}
                            {item.cum.toFixed(1)}
                          </strong>
                        </div>
                      );
                    })}
                  </div>
                )}

                {/* Unified 10-Seed Strip: Current Checkpoint Pass/Fail + Recent 10-Eval Win Rate % + Median/Worst Step Tooltips */}
                <div className="sd-seed-strip" aria-label="10-seed evaluation quick selector">
                  <span
                    style={{
                      fontSize: "0.66rem",
                      fontWeight: 700,
                      color: "var(--sd-text-muted)",
                      textTransform: "uppercase",
                      marginRight: "0.15rem",
                    }}
                  >
                    10 Seeds (Now · 10-Eval %):
                  </span>
                  {activeSeeds.map((seed) => {
                    const rec = (activeCheckpoint?.records ?? []).find(
                      (r) => r.seed === seed
                    ) as any;
                    const passed = rec ? Boolean(rec.success) : null;
                    const rew = rec ? (rec.reward_total ?? rec.total_reward ?? 0) : 0;
                    const rel = seedReliability.seeds.find((s) => s.seed === seed);
                    const winPct = rel?.recentSuccessRate;
                    const medSteps = rel?.typicalSuccessfulSteps;
                    const worstSteps = rel?.worstSuccessfulSteps;
                    return (
                      <button
                        key={seed}
                        type="button"
                        className={`sd-seed-chip ${
                          passed === true
                            ? "sd-seed-chip--pass"
                            : passed === false
                            ? "sd-seed-chip--fail"
                            : ""
                        } ${selectedSeed === seed ? "sd-seed-chip--active" : ""}`}
                        onClick={() => handlePickSeed(seed)}
                        title={
                          rec
                            ? `Seed #${seed}: ${
                                rec.success ? "PASS" : "FAIL"
                              } in ${rec.steps} steps (Return ${Number(rew).toFixed(1)})${
                                winPct != null
                                  ? ` | 10-Eval Win Rate: ${winPct}% (${rel?.successCount}/${rel?.totalTrials})`
                                  : ""
                              }${
                                medSteps != null
                                  ? ` | Median Pass: ${medSteps}s (Worst: ${worstSteps ?? "—"}s)`
                                  : ""
                              }`
                            : `Load Seed #${seed}`
                        }
                      >
                        <span>{passed === true ? "✓" : passed === false ? "✗" : "•"}</span>
                        <span>#{seed}</span>
                        {winPct != null ? (
                          <span style={{ opacity: 0.85, fontSize: "0.64rem" }}>
                            {winPct}%
                          </span>
                        ) : rec?.steps != null ? (
                          <span style={{ opacity: 0.75 }}>{rec.steps}s</span>
                        ) : null}
                      </button>
                    );
                  })}

                  <button
                    type="button"
                    className="sd-btn sd-btn--sm sd-btn--ghost"
                    style={{ marginLeft: "auto", fontSize: "0.66rem", padding: "0.16rem 0.45rem" }}
                    onClick={() => {
                      if (activeCheckpoint) {
                        setArenaTargetCheckpoint(activeCheckpoint.checkpoint_episode);
                      }
                      setWorkspace("arena");
                    }}
                  >
                    10-Seed Arena →
                  </button>
                </div>

                {/* Restored Compact Entity Key & Set-Once Overlay Controls */}
                {currentSnapshot && (
                  <div
                    style={{
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "space-between",
                      flexWrap: "wrap",
                      gap: "0.45rem",
                      fontSize: "0.66rem",
                      color: "var(--sd-text-secondary)",
                      paddingTop: "0.22rem",
                      borderTop: "1px solid var(--sd-border-subtle)",
                    }}
                  >
                    <div
                      style={{
                        display: "flex",
                        alignItems: "center",
                        gap: "0.6rem",
                        flexWrap: "wrap",
                      }}
                    >
                      {(currentSnapshot.dogs ?? []).map((dog) => (
                        <span
                          key={dog.index}
                          style={{
                            display: "inline-flex",
                            alignItems: "center",
                            gap: "0.22rem",
                          }}
                        >
                          <span
                            style={{
                              width: 8,
                              height: 8,
                              borderRadius: "50%",
                              background: dogColor(dog.index),
                              display: "inline-block",
                            }}
                          />
                          <span>
                            D{dog.index + 1}
                            {dog.role ? ` (${ROLE_LABELS[dog.role] ?? dog.role})` : ""}
                          </span>
                        </span>
                      ))}
                      <span style={{ color: "var(--sd-text-muted)" }}>|</span>
                      <span>🐑 White Sheep (🟢 Ring = Penned)</span>
                      <span>🟡 Pen Gate</span>
                    </div>

                    <div style={{ display: "flex", alignItems: "center", gap: "0.55rem" }}>
                      <label
                        style={{
                          display: "inline-flex",
                          alignItems: "center",
                          gap: "0.22rem",
                          cursor: "pointer",
                        }}
                        title="Show tactical role badges beneath dogs"
                      >
                        <input
                          type="checkbox"
                          checked={showRoles}
                          onChange={(e) => setShowRoles(e.target.checked)}
                        />
                        <span>Role Tags</span>
                      </label>
                      <label
                        style={{
                          display: "inline-flex",
                          alignItems: "center",
                          gap: "0.22rem",
                          cursor: "pointer",
                        }}
                        title="Show flock-center to pen-gate herding axis line"
                      >
                        <input
                          type="checkbox"
                          checked={showVectors}
                          onChange={(e) => setShowVectors(e.target.checked)}
                        />
                        <span>Herding Axis</span>
                      </label>
                    </div>
                  </div>
                )}
              </div>
            </section>

            {/* RIGHT PANE: Live Learning Curves + Bottom Learning Diagnostics Drawer */}
            <div className="sd-intel-pane">
              {/* Top: Interactive Learning Trajectory Charts */}
              <StudioLearningCharts
                rawEpisodes={trainingEpisodes}
                checkpoints={allCheckpoints}
                sessionEvents={sessionEvents}
                effectiveStage={effectiveStage}
                activeRunId={trainingStatus?.run_id ?? undefined}
                bestCheckpointEpisode={(checkpointIndex as any)?.best_checkpoint_episode}
                selectedCheckpointEpisode={activeCheckpoint?.checkpoint_episode}
                promotionTargetPct={Math.round(
                  ((gateDiag?.success_threshold ?? 0.9) <= 1
                    ? (gateDiag?.success_threshold ?? 0.9) * 100
                    : gateDiag?.success_threshold ?? 90)
                )}
                defaultMode="combined"
                onSelectCheckpoint={(ep) => {
                  setAutoFollowLive(false);
                  setSelectedCheckpointEp(ep);
                }}
                onInspectInArena={(ep) => {
                  setArenaTargetCheckpoint(ep);
                  setWorkspace("arena");
                }}
              />

              {/* Bottom: Learning Diagnostics, Failure Anatomy & Training Config */}
              <section
                className="sd-panel"
                style={{ flex: 1, display: "flex", flexDirection: "column" }}
              >
                {/* Contextualized Policy & Exploration Pulse Strip (No Duplicate Gate) */}
                <div
                  className="sd-mono"
                  style={{
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "space-between",
                    gap: "0.5rem",
                    flexWrap: "wrap",
                    padding: "0.36rem 0.85rem",
                    background: "rgba(9, 13, 22, 0.75)",
                    borderBottom: "1px solid var(--sd-border-subtle)",
                    fontSize: "0.67rem",
                    color: "var(--sd-text-secondary)",
                  }}
                >
                  <div
                    style={{
                      display: "flex",
                      alignItems: "center",
                      gap: "0.85rem",
                      flexWrap: "wrap",
                    }}
                  >
                    <span title="Current effective learning rate and adaptive schedule stage">
                      ⚡ LR:{" "}
                      <strong style={{ color: "var(--sd-text-primary)" }}>
                        {trainingStatus?.effective_learning_rate != null
                          ? trainingStatus.effective_learning_rate.toExponential(2)
                          : "3.00e-4"}
                      </strong>
                      {trainingStatus?.adaptive_lr_stage_label && (
                        <span style={{ color: "var(--sd-cyan)", marginLeft: 4 }}>
                          ({trainingStatus.adaptive_lr_stage_label})
                        </span>
                      )}
                    </span>

                    {rolloutWinPct != null && (
                      <span title="Comparison between deterministic 10-seed evaluation pass rate and noisy stochastic training rollouts">
                        Exploration Gap:{" "}
                        <strong style={{ color: "var(--sd-emerald-bright)" }}>
                          Eval {latestSuccessPct}%
                        </strong>{" "}
                        vs{" "}
                        <strong style={{ color: "var(--sd-accent-bright)" }}>
                          Rollout {rolloutWinPct}%
                        </strong>
                      </span>
                    )}

                    {trainingStatus?.explained_variance != null && (
                      <span title="Critic value network accuracy (≥60% Strong, 30–59% Learning, <30% Weak)">
                        Critic Fit:{" "}
                        <strong
                          style={{
                            color:
                              trainingStatus.explained_variance >= 0.6
                                ? "var(--sd-emerald-bright)"
                                : trainingStatus.explained_variance >= 0.3
                                ? "var(--sd-amber-bright)"
                                : "var(--sd-rose-bright)",
                          }}
                        >
                          {(trainingStatus.explained_variance * 100).toFixed(0)}% (
                          {trainingStatus.explained_variance >= 0.6
                            ? "Strong"
                            : trainingStatus.explained_variance >= 0.3
                            ? "Learning"
                            : "Weak"}
                          )
                        </strong>
                      </span>
                    )}
                  </div>

                  {failureAnatomy.totalLosses > 0 && (
                    <span
                      title="Breakdown of how recent evaluation losses terminated (Max-step Timeout vs Early No-Progress Stall)"
                      style={{ color: "var(--sd-text-muted)" }}
                    >
                      Loss Stops:{" "}
                      <strong style={{ color: "var(--sd-amber-bright)" }}>
                        {failureAnatomy.timeoutLosses} Timeouts
                      </strong>{" "}
                      ·{" "}
                      <strong style={{ color: "var(--sd-rose-bright)" }}>
                        {failureAnatomy.stallLosses} Stalls
                      </strong>
                    </span>
                  )}
                </div>

                <div className="sd-panel-header">
                  <div
                    className="sd-pill-group"
                    role="tablist"
                    aria-label="Intelligence drawer tabs"
                  >
                    <button
                      type="button"
                      role="tab"
                      aria-selected={bottomTab !== "controls"}
                      className={`sd-pill-btn ${
                        bottomTab !== "controls" ? "sd-pill-btn--active" : ""
                      }`}
                      onClick={() => setBottomTab("gate")}
                    >
                      🧠 Failure Anatomy & Spatial Zones
                    </button>
                    <button
                      type="button"
                      role="tab"
                      aria-selected={bottomTab === "controls"}
                      className={`sd-pill-btn ${
                        bottomTab === "controls" ? "sd-pill-btn--active" : ""
                      }`}
                      onClick={() => setBottomTab("controls")}
                    >
                      ⚙️ Training Config & Admin
                    </button>
                  </div>
                </div>

                {/* Drawer Tab 1: Failure Mode Anatomy & 9-Zone Spatial Spawn Health */}
                {bottomTab !== "controls" && (
                  <div
                    className="sd-panel-body"
                    style={{
                      display: "grid",
                      gridTemplateColumns: "1.08fr 0.92fr",
                      gap: "0.85rem",
                      alignItems: "start",
                    }}
                  >
                    {/* Left Column: Failure Mode Anatomy & Bottleneck Seeds */}
                    <div>
                      <div
                        style={{
                          fontSize: "0.73rem",
                          fontWeight: 700,
                          color: "var(--sd-text-secondary)",
                          marginBottom: "0.45rem",
                          display: "flex",
                          justifyContent: "space-between",
                        }}
                      >
                        <span>
                          Why Episodes Fail (Last {failureAnatomy.evalCount} Evals)
                        </span>
                        <span
                          className="sd-mono"
                          style={{
                            color:
                              failureAnatomy.totalLosses === 0
                                ? "var(--sd-emerald-bright)"
                                : "var(--sd-rose-bright)",
                          }}
                        >
                          {failureAnatomy.totalLosses} Losses / {failureAnatomy.totalTrials} Trials
                        </span>
                      </div>

                      {failureAnatomy.totalTrials === 0 ? (
                        <div
                          style={{
                            padding: "0.75rem",
                            borderRadius: "var(--sd-radius-sm)",
                            background: "rgba(9, 13, 22, 0.5)",
                            fontSize: "0.73rem",
                            color: "var(--sd-text-muted)",
                          }}
                        >
                          Run training on Stage {effectiveStage} to analyze failure modes across 10-seed evaluations.
                        </div>
                      ) : failureAnatomy.totalLosses === 0 ? (
                        <div
                          style={{
                            padding: "0.65rem 0.75rem",
                            borderRadius: "var(--sd-radius-sm)",
                            background: "rgba(16, 185, 129, 0.1)",
                            border: "1px solid rgba(16, 185, 129, 0.3)",
                            fontSize: "0.73rem",
                            color: "var(--sd-emerald-bright)",
                          }}
                        >
                          ★ 100% Pass Rate across recent evaluations—zero losses recorded!
                        </div>
                      ) : (
                        <div className="sd-gate-checklist" style={{ gap: "0.35rem" }}>
                          {[
                            {
                              label: "Straggler Timeout (≥50% sheep penned)",
                              count: failureAnatomy.stragglerLosses,
                              pct: failureAnatomy.stragglerPct,
                              color: "var(--sd-amber-bright)",
                              hint: "Main flock penned, ran out of steps collecting last 1–2 sheep",
                            },
                            {
                              label: "Mid-Flock / Gate Stall (1–49% penned)",
                              count: failureAnatomy.midHerdingStalls,
                              pct: failureAnatomy.midStallPct,
                              color: "var(--sd-cyan)",
                              hint: "Some sheep penned, remainder pinned on outer pen wall or gate",
                            },
                            {
                              label: "Zero-Pen Breakdown (0 sheep penned)",
                              count: failureAnatomy.zeroPenBreakdowns,
                              pct: failureAnatomy.zeroPenPct,
                              color: "var(--sd-rose-bright)",
                              hint: "Failed initial collection, corner extraction, or scattered flock",
                            },
                          ].map((mode) => (
                            <div
                              key={mode.label}
                              className="sd-gate-check-item"
                              title={mode.hint}
                              style={{ padding: "0.35rem 0.55rem" }}
                            >
                              <span
                                className="sd-gate-check-item__label"
                                style={{ fontSize: "0.71rem" }}
                              >
                                <span style={{ color: mode.color, fontWeight: 800 }}>●</span>
                                <span>{mode.label}</span>
                              </span>
                              <span className="sd-gate-check-item__val">
                                {mode.count} ({mode.pct}%)
                              </span>
                            </div>
                          ))}
                        </div>
                      )}

                      {/* Top Bottleneck Seeds with 1-Click Watch Loss */}
                      {failureAnatomy.worstSeeds.length > 0 && (
                        <div
                          style={{
                            marginTop: "0.5rem",
                            display: "flex",
                            alignItems: "center",
                            gap: "0.35rem",
                            flexWrap: "wrap",
                            fontSize: "0.68rem",
                          }}
                        >
                          <span style={{ color: "var(--sd-text-muted)", fontWeight: 700 }}>
                            Hardest Seeds:
                          </span>
                          {failureAnatomy.worstSeeds.map((ws) => (
                            <button
                              key={ws.seed}
                              type="button"
                              className="sd-btn sd-btn--sm sd-btn--ghost sd-mono"
                              style={{
                                padding: "0.14rem 0.42rem",
                                fontSize: "0.66rem",
                                borderColor:
                                  ws.recentSuccessRate < 60
                                    ? "rgba(244, 63, 94, 0.45)"
                                    : "var(--sd-border-subtle)",
                              }}
                              onClick={() => {
                                const failingCp = stageCheckpoints.find((cp) =>
                                  (cp.records ?? []).some(
                                    (r) => r.seed === ws.seed && !r.success
                                  )
                                );
                                if (failingCp) {
                                  setAutoFollowLive(false);
                                  setSelectedCheckpointEp(failingCp.checkpoint_episode);
                                }
                                handlePickSeed(ws.seed);
                              }}
                              title={`Click to jump to a failed replay of Seed #${ws.seed} (${ws.recentSuccessRate}% recent win rate · Median ${ws.typicalSuccessfulSteps ?? "—"}s)`}
                            >
                              ▶ #{ws.seed} ({ws.recentSuccessRate}%)
                            </button>
                          ))}
                        </div>
                      )}
                    </div>

                    {/* Right Column: 9-Zone Spatial Spawn Heatmap Summary */}
                    <div>
                      <div
                        style={{
                          fontSize: "0.73rem",
                          fontWeight: 700,
                          color: "var(--sd-text-secondary)",
                          marginBottom: "0.45rem",
                          display: "flex",
                          justifyContent: "space-between",
                        }}
                      >
                        <span>9-Zone Spatial Spawn Reliability</span>
                        <button
                          type="button"
                          className="sd-btn sd-btn--sm sd-btn--ghost"
                          style={{ padding: "0.08rem 0.4rem", fontSize: "0.66rem" }}
                          onClick={() => {
                            setAnalyticsSubTab("bottlenecks");
                            setWorkspace("analytics");
                          }}
                        >
                          Full Diagnostics →
                        </button>
                      </div>

                      {stageDiagnostics?.zone_stats &&
                      Object.keys(stageDiagnostics.zone_stats).length > 0 ? (
                        <div className="sd-zone-grid-3x3">
                          {[
                            "top_left",
                            "top_wall",
                            "top_right",
                            "left_wall",
                            "center",
                            "right_wall",
                            "bottom_left",
                            "bottom_wall",
                            "bottom_right",
                          ].map((zoneId) => {
                            const z = stageDiagnostics.zone_stats[zoneId];
                            const total = z?.total ?? 0;
                            const rawWinRate = z?.win_rate ?? 0;
                            const ratePct = Math.round(
                              rawWinRate <= 1 ? rawWinRate * 100 : rawWinRate
                            );
                            const color =
                              total === 0
                                ? "var(--sd-text-muted)"
                                : ratePct >= 80
                                ? "var(--sd-emerald-bright)"
                                : ratePct >= 50
                                ? "var(--sd-amber-bright)"
                                : "var(--sd-rose-bright)";
                            return (
                              <div key={zoneId} className="sd-zone-cell">
                                <div className="sd-zone-cell__name">
                                  {zoneId.replace(/_/g, " ")}
                                </div>
                                <div className="sd-zone-cell__rate" style={{ color }}>
                                  {total > 0 ? `${ratePct}%` : "—"}
                                </div>
                                <div className="sd-zone-cell__meta">{total} eps</div>
                              </div>
                            );
                          })}
                        </div>
                      ) : (
                        <div
                          style={{
                            padding: "0.75rem",
                            borderRadius: "var(--sd-radius-sm)",
                            background: "rgba(9, 13, 22, 0.5)",
                            fontSize: "0.73rem",
                            color: "var(--sd-text-muted)",
                          }}
                        >
                          Spatial zone diagnostics populate automatically as episodes complete in Stage {effectiveStage}.
                        </div>
                      )}
                    </div>
                  </div>
                )}

                {/* Drawer Tab 3: Training Configuration & System Admin */}
                {bottomTab === "controls" && (
                  <div className="sd-panel-body">
                    <div className="sd-config-grid">
                      <div className="sd-form-group">
                        <label className="sd-form-label" htmlFor="sd-batch-eps">
                          Episodes This Batch
                        </label>
                        <input
                          id="sd-batch-eps"
                          type="number"
                          className="sd-input"
                          min={10}
                          max={50000}
                          step={50}
                          value={batchEpisodes}
                          onChange={(e) => setBatchEpisodes(Math.max(1, Number(e.target.value)))}
                        />
                      </div>

                      <div className="sd-form-group">
                        <label className="sd-form-label" htmlFor="sd-model-src">
                          Starting Model Source
                        </label>
                        <select
                          id="sd-model-src"
                          className="sd-select"
                          value={modelSource}
                          onChange={(e) =>
                            setModelSource(e.target.value as "latest_stage" | "scratch")
                          }
                        >
                          <option value="latest_stage">Continue Latest Stage Checkpoint</option>
                          <option value="scratch">Train Fresh Policy From Scratch</option>
                        </select>
                      </div>

                      <div className="sd-form-group">
                        <label className="sd-form-label">Reward Shaping & Automation</label>
                        <div style={{ display: "flex", gap: "0.75rem", paddingTop: "0.25rem" }}>
                          <label
                            style={{
                              display: "inline-flex",
                              alignItems: "center",
                              gap: "0.35rem",
                              fontSize: "0.76rem",
                              cursor: "pointer",
                            }}
                          >
                            <input
                              type="checkbox"
                              checked={enableInstincts}
                              onChange={(e) => setEnableInstincts(e.target.checked)}
                            />
                            <span>Enable Instincts</span>
                          </label>

                          <label
                            style={{
                              display: "inline-flex",
                              alignItems: "center",
                              gap: "0.35rem",
                              fontSize: "0.76rem",
                              cursor: "pointer",
                            }}
                          >
                            <input
                              type="checkbox"
                              checked={autoPromote}
                              onChange={(e) => setAutoPromote(e.target.checked)}
                            />
                            <span>Auto-Promote Stages</span>
                          </label>
                        </div>
                      </div>
                    </div>

                    {/* Administrative Actions */}
                    <div
                      style={{
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "space-between",
                        flexWrap: "wrap",
                        gap: "0.5rem",
                        marginTop: "0.9rem",
                        paddingTop: "0.75rem",
                        borderTop: "1px solid var(--sd-border-subtle)",
                      }}
                    >
                      <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
                        <button
                          type="button"
                          className="sd-btn sd-btn--sm"
                          disabled={isTraining || actionBusy}
                          onClick={async () => {
                            mutationEpochRef.current += 1;
                            setActionBusy(true);
                            try {
                              const status = await startRemediationFork(9, 20);
                              mutationEpochRef.current += 1;
                              if (!mountedRef.current) return;
                              setTrainingStatus(status);
                              setActionMessage("Started Stage 9 stray recovery remediation pass.");
                            } catch (err) {
                              mutationEpochRef.current += 1;
                              if (!mountedRef.current) return;
                              setActionMessage(
                                err instanceof Error ? err.message : "Remediation failed"
                              );
                            } finally {
                              if (mountedRef.current) setActionBusy(false);
                            }
                          }}
                        >
                          🛠️ Stage 9 Stray Remediation
                        </button>

                        <button
                          type="button"
                          className="sd-btn sd-btn--sm"
                          disabled={isTraining || actionBusy}
                          onClick={async () => {
                            if (
                              !window.confirm(
                                "Reset current journey back to Stage 1 while archiving historical runs?"
                              )
                            )
                              return;
                            mutationEpochRef.current += 1;
                            setActionBusy(true);
                            try {
                              await resetJourneyTraining();
                              mutationEpochRef.current += 1;
                              await refreshCoreData(true);
                              if (!mountedRef.current) return;
                              setActionMessage("Reset current curriculum journey to Stage 1.");
                            } finally {
                              if (mountedRef.current) setActionBusy(false);
                            }
                          }}
                        >
                          ↺ Reset Journey
                        </button>
                      </div>

                      <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
                        <button
                          type="button"
                          className="sd-btn sd-btn--sm sd-btn--danger"
                          disabled={isTraining || actionBusy}
                          onClick={async () => {
                            if (
                              !window.confirm(
                                "Clear all generated checkpoints and replays? This cannot be undone."
                              )
                            )
                              return;
                            mutationEpochRef.current += 1;
                            setActionBusy(true);
                            try {
                              await clearTraining();
                              mutationEpochRef.current += 1;
                              await refreshCoreData(true);
                              if (!mountedRef.current) return;
                              setActionMessage("Cleared training data.");
                            } finally {
                              if (mountedRef.current) setActionBusy(false);
                            }
                          }}
                        >
                          🗑️ Clear Training Data
                        </button>

                        <button
                          type="button"
                          className="sd-btn sd-btn--sm sd-btn--danger"
                          onClick={async () => {
                            if (!window.confirm("Shut down the backend server?")) return;
                            await shutdownApp();
                            setActionMessage("Server shutdown requested.");
                          }}
                        >
                          ⏻ Shutdown Server
                        </button>
                      </div>
                    </div>
                  </div>
                )}
              </section>
            </div>
          </div>
        )}

        {/* ============================================================
            WORKSPACE 2: 10-SEED EVALUATION ARENA
           ============================================================ */}
        {workspace === "arena" && (
          <StudioEvalArena
            checkpoints={allCheckpoints}
            effectiveStage={effectiveStage}
            bestCheckpointEpisode={(checkpointIndex as any)?.best_checkpoint_episode}
            initialCheckpointEpisode={arenaTargetCheckpoint ?? activeCheckpoint?.checkpoint_episode}
            onSelectSeedForMainPlayer={(replay, ckptEp, seed) => {
              setAutoFollowLive(false);
              setSelectedCheckpointEp(ckptEp);
              handlePickSeed(seed);
              const matchedCp = allCheckpoints.find((c) => c.checkpoint_episode === ckptEp);
              const st = matchedCp ? getCheckpointStage(matchedCp) : effectiveStage;
              const key = `${st}:${ckptEp}:${seed}`;
              cacheStudioReplay(key, replay);
              loadedReplayKeyRef.current = key;
              setLoadedReplayKey(key);
              setActiveReplay(replay);
              setFrameIndex(0);
            }}
          />
        )}

        {/* ============================================================
            WORKSPACE 2.5: TEMPORAL INTERACTION DIAGNOSTICS
           ============================================================ */}
        {workspace === "temporal" && (
          <div
            style={{
              flex: 1,
              minHeight: 0,
              display: "flex",
              flexDirection: "column",
              overflow: "hidden",
            }}
          >
            <TemporalDiagnosticsPanel
              initialReplayId={temporalReplayId}
              activeReplay={activeReplay as any}
              activeReplayId={temporalReplayId}
              backendOffline={backendOffline}
              isReadOnlyPreview={isReadOnlyPreview}
              backendStarting={backendStarting}
              onCachedReplayAvailableChange={handleTemporalCachedAvailableChange}
              onRequestReconnect={handleTemporalRequestReconnect}
            />
          </div>
        )}

        {/* ============================================================
            WORKSPACE 3: ANALYTICS & PROGRESSION
           ============================================================ */}
        {workspace === "analytics" && (
          <div className="sd-analytics-view">
            <div
              style={{
                display: "flex",
                alignItems: "center",
                justifyContent: "space-between",
                background: "var(--sd-bg-panel)",
                padding: "0.6rem 0.9rem",
                borderRadius: "var(--sd-radius-md)",
                border: "1px solid var(--sd-border-subtle)",
              }}
            >
              <div className="sd-pill-group" role="tablist" aria-label="Analytics sub-views">
                <button
                  type="button"
                  role="tab"
                  aria-selected={analyticsSubTab === "curves"}
                  className={`sd-pill-btn ${
                    analyticsSubTab === "curves" ? "sd-pill-btn--active" : ""
                  }`}
                  onClick={() => setAnalyticsSubTab("curves")}
                >
                  📈 3-Panel Learning Curves
                </button>
                <button
                  type="button"
                  role="tab"
                  aria-selected={analyticsSubTab === "bottlenecks"}
                  className={`sd-pill-btn ${
                    analyticsSubTab === "bottlenecks" ? "sd-pill-btn--active" : ""
                  }`}
                  onClick={() => setAnalyticsSubTab("bottlenecks")}
                >
                  🗺️ 9-Zone Spatial Bottlenecks
                </button>
                <button
                  type="button"
                  role="tab"
                  aria-selected={analyticsSubTab === "milestones"}
                  className={`sd-pill-btn ${
                    analyticsSubTab === "milestones" ? "sd-pill-btn--active" : ""
                  }`}
                  onClick={() => setAnalyticsSubTab("milestones")}
                >
                  🏆 Stage Milestone Gallery
                </button>
                <button
                  type="button"
                  role="tab"
                  aria-selected={analyticsSubTab === "config"}
                  className={`sd-pill-btn ${
                    analyticsSubTab === "config" ? "sd-pill-btn--active" : ""
                  }`}
                  onClick={() => setAnalyticsSubTab("config")}
                >
                  🧠 Hyperparameters & Network Topology
                </button>
              </div>
            </div>

            {analyticsSubTab === "curves" && (
              <StudioLearningCharts
                rawEpisodes={trainingEpisodes}
                checkpoints={allCheckpoints}
                sessionEvents={sessionEvents}
                effectiveStage={effectiveStage}
                activeRunId={trainingStatus?.run_id ?? undefined}
                bestCheckpointEpisode={(checkpointIndex as any)?.best_checkpoint_episode}
                selectedCheckpointEpisode={activeCheckpoint?.checkpoint_episode}
                defaultMode="stacked"
                onSelectCheckpoint={(ep) => {
                  setAutoFollowLive(false);
                  setSelectedCheckpointEp(ep);
                }}
                onInspectInArena={(ep) => {
                  setArenaTargetCheckpoint(ep);
                  setWorkspace("arena");
                }}
              />
            )}

            {analyticsSubTab === "bottlenecks" && (
              <div className="sd-panel" style={{ padding: "1rem" }}>
                <StageBottlenecksPanel
                  currentStage={effectiveStage}
                  runId={trainingStatus?.run_id}
                />
              </div>
            )}

            {analyticsSubTab === "milestones" && (
              <div className="sd-panel" style={{ padding: "1rem" }}>
                <ResultsPanel checkpointIndex={checkpointIndex} />
              </div>
            )}

            {analyticsSubTab === "config" && (
              <div
                style={{
                  display: "grid",
                  gridTemplateColumns: "1fr 1fr",
                  gap: "0.85rem",
                  alignItems: "start",
                }}
              >
                <div className="sd-panel" style={{ padding: "1rem" }}>
                  <ConfigPanel />
                </div>
                <div className="sd-panel" style={{ padding: "1rem" }}>
                  <NetworkTab
                    checkpointIndex={checkpointIndex}
                    trainingStatus={trainingStatus}
                    effectiveConfig={effectiveConfig}
                    topologyInfo={topologyInfo}
                  />
                </div>
              </div>
            )}
          </div>
        )}
      </main>
    </div>
  );
}

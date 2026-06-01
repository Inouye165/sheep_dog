import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ControlBar } from "./components/ControlBar";
import { ConfigPanel } from "./components/ConfigPanel";
import { DiagnosticsPanel } from "./components/DiagnosticsPanel";
import { FieldView } from "./components/FieldView";
import { ScenarioPanel } from "./components/ScenarioPanel";
import { SavedScenariosPanel } from "./components/SavedScenariosPanel";
import { TrainingPanel } from "./components/TrainingPanel";
import { StatusPanel } from "./components/StatusPanel";
import { ResultsPanel } from "./components/ResultsPanel";
import {
  clearTraining,
  evaluateScenario,
  loadCheckpointIndex,
  loadHyperparams,
  loadReplay,
  loadScenarioIndex,
  loadTrainingStatus,
  replayScenario,
  runReplay,
  saveScenario,
  startTraining,
} from "./lib/api";
import type { CheckpointMode } from "./lib/api";
import type {
  CheckpointEntry,
  CheckpointIndex,
  ReplayBundle,
  ReplaySnapshot,
  ScenarioIndex,
  TrainingStatus,
} from "./state/types";

type RunState = "idle" | "running" | "paused" | "success" | "timeout" | "stopped";
type ActiveTab = "train" | "watch" | "test" | "insights" | "results" | "config";

const APP_TABS: { id: ActiveTab; label: string }[] = [
  { id: "train", label: "Train" },
  { id: "watch", label: "Watch" },
  { id: "test", label: "Scenarios" },
  { id: "insights", label: "Insights" },
  { id: "results", label: "Results" },
  { id: "config", label: "Config" },
];

const CLEAR_TRAINING_MESSAGE = "Training cleared. Baseline replay restored";

/** Mirrors RECOMMENDED_EPISODES in TrainingPanel — update both together. */
const RECOMMENDED_EPISODES_BY_STAGE: Record<number, number> = {
  0: 50,
  1: 50,
  2: 100,
  3: 150,
  4: 200,
  5: 300,
};
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

function mergeTrainingStatus(previous: TrainingStatus | null, next: TrainingStatus): TrainingStatus {
  let merged = next;
  if (next.phase === "idle" && next.message === "Idle" && previous?.message === CLEAR_TRAINING_MESSAGE) {
    merged = { ...next, message: previous.message };
  }
  // Prevent batch progress from oscillating backwards while a batch is in flight.
  if (
    previous &&
    previous.running &&
    merged.running &&
    previous.batch_total_episodes === merged.batch_total_episodes &&
    previous.batch_total_episodes > 0 &&
    merged.batch_completed_episodes < previous.batch_completed_episodes
  ) {
    merged = {
      ...merged,
      batch_completed_episodes: previous.batch_completed_episodes,
      completed_episodes: Math.max(previous.completed_episodes, merged.completed_episodes),
    };
  }
  return merged;
}

export function App() {
  const [checkpointIndex, setCheckpointIndex] = useState<CheckpointIndex | null>(null);
  const [replay, setReplay] = useState<ReplayBundle | null>(null);
  const [selectedCheckpointEpisode, setSelectedCheckpointEpisode] = useState<number | null>(null);
  const [selectedSeed, setSelectedSeed] = useState<number | null>(null);
  const [runState, setRunState] = useState<RunState>("idle");
  const [frameIndex, setFrameIndex] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [trainingError, setTrainingError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [trainingEpisodes, setTrainingEpisodes] = useState(50);
  const [trainingFastMode, setTrainingFastMode] = useState(true);
  const [trainingEnableInstincts, setTrainingEnableInstincts] = useState(false);
  const [trainingCurriculumStage, setTrainingCurriculumStage] = useState(() => {
    const saved = localStorage.getItem("sheepdog_curriculum_stage");
    const parsed = saved !== null ? parseInt(saved, 10) : NaN;
    return !isNaN(parsed) && parsed >= 1 ? parsed : 1;
  });
  const [trainingDebugRewardBreakdown, setTrainingDebugRewardBreakdown] = useState(false);
  const [playbackFastMode, setPlaybackFastMode] = useState(false);
  const [trainingStatus, setTrainingStatus] = useState<TrainingStatus | null>(null);
  const [clearingTraining, setClearingTraining] = useState(false);
  const [runningCurrentReplay, setRunningCurrentReplay] = useState(false);
  const [loadingSelectedReplay, setLoadingSelectedReplay] = useState(false);
  const [activeTab, setActiveTab] = useState<ActiveTab>("train");
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
  // Episode of the best-formal checkpoint for the current stage captured at
  // promote time, so the next training start can seed from the correct policy.
  const [promoteFromEpisode, setPromoteFromEpisode] = useState<number | null>(null);
  // Track previous running state so we can detect the running→idle transition.
  const prevTrainingRunning = useRef(false);

  const syncTrainingStageFromStatus = useCallback((status: TrainingStatus | null): void => {
    const reportedStage = status?.curriculum_stage;
    if (reportedStage == null || reportedStage < 1) {
      return;
    }
    setTrainingCurriculumStage(reportedStage);
    localStorage.setItem("sheepdog_curriculum_stage", String(reportedStage));
    setTrainingEpisodes(RECOMMENDED_EPISODES_BY_STAGE[reportedStage] ?? 50);
  }, []);

  const selectedCheckpoint = useMemo(() => {
    return checkpointIndex?.checkpoints.find((entry) => entry.checkpoint_episode === selectedCheckpointEpisode) ?? null;
  }, [checkpointIndex, selectedCheckpointEpisode]);

  const seedOptions = useMemo(
    () => (selectedCheckpoint ? selectedCheckpoint.records.map((record) => record.seed) : []),
    [selectedCheckpoint],
  );

  const playbackDelay = playbackFastMode ? 24 : 220;
  const scenarioPlaybackDelay = scenarioFastMode ? 24 : 220;
  const trainingRunning = trainingStatus?.running ?? false;

  // When training completes, lock the stage chip to the stage that was actually
  // trained and preset episodes to the recommended count for that stage.  Only
  // fire on the running→idle transition so polling doesn't clobber a user's
  // pending promote selection while training is idle.
  useEffect(() => {
    if (prevTrainingRunning.current && !trainingRunning && trainingStatus?.curriculum_stage != null) {
      syncTrainingStageFromStatus(trainingStatus);
    }
    prevTrainingRunning.current = trainingRunning;
  }, [syncTrainingStageFromStatus, trainingRunning, trainingStatus]);
  const effectiveEnableInstincts = trainingRunning
    ? trainingStatus?.enable_instinct_rewards ?? trainingEnableInstincts
    : trainingEnableInstincts;
  const effectiveCurriculumStage = trainingRunning
    ? trainingStatus?.curriculum_stage ?? trainingCurriculumStage
    : trainingCurriculumStage;
  const effectiveDebugRewardBreakdown = trainingRunning
    ? trainingStatus?.debug_reward_breakdown ?? trainingDebugRewardBreakdown
    : trainingDebugRewardBreakdown;

  const latestSuccessRate = useMemo(() => {
    const checkpoints = checkpointIndex?.checkpoints;
    if (!checkpoints?.length) return null;
    // Only consider checkpoints trained at the current curriculum stage so that
    // promoting immediately after a 100% run doesn't falsely show "ready to promote"
    // for the new stage before any training has been done.
    const stageCheckpoints = checkpoints.filter(
      (c) => c.reward_config?.instincts?.curriculum_stage === effectiveCurriculumStage,
    );
    if (!stageCheckpoints.length) return null;
    return stageCheckpoints[stageCheckpoints.length - 1]?.success_rate ?? null;
  }, [checkpointIndex, effectiveCurriculumStage]);

  // Best formally-evaluated checkpoint for the current stage, using the same
  // stage-aware ordering as isStrictlyBetterCheckpoint (success_rate primary,
  // average_reward tie-breaker).  This drives the promote-readiness badge so
  // the user sees the peak performance, not just the most recent run.
  const bestStageFormalEntry = useMemo(() => {
    const stageCheckpoints = checkpointIndex?.checkpoints.filter(
      (c) => c.reward_config?.instincts?.curriculum_stage === effectiveCurriculumStage,
    ) ?? [];
    if (!stageCheckpoints.length) return null;
    return stageCheckpoints.reduce((best, entry) => {
      if (!best) return entry;
      if (entry.success_rate > best.success_rate) return entry;
      if (
        entry.success_rate === best.success_rate &&
        (entry.average_completion_steps ?? Infinity) < (best.average_completion_steps ?? Infinity)
      )
        return entry;
      return best;
    }, null as CheckpointEntry | null);
  }, [checkpointIndex, effectiveCurriculumStage]);

  const currentReplaySelection = useMemo(() => {
    const latestCheckpoint = checkpointIndex?.checkpoints[checkpointIndex.checkpoints.length - 1] ?? null;
    const totalTrainingEpisodes =
      latestCheckpoint?.total_training_episodes ?? trainingStatus?.total_episodes_trained ?? 0;
    const latestPolicyMode = latestCheckpoint?.policy_mode ?? latestCheckpoint?.policy_name ?? null;

    if (latestCheckpoint && totalTrainingEpisodes > 0) {
      const resolvedPolicyMode = latestPolicyMode === "neural_policy" ? "neural_policy" : "trained_policy";
      return {
        // Pass null so the server loads from training-state.json → best_model_path
        // rather than the specific checkpoint model file (which may be stale or
        // from a lower curriculum stage).
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
  }, [checkpointIndex, trainingStatus?.total_episodes_trained]);

  const currentFrame =
    replay?.frames?.[Math.min(frameIndex, Math.max((replay?.frames.length ?? 0) - 1, 0))] ?? null;

  // Identify the highest-quality checkpoint across all entries using the same
  // stage-aware ordering the trainer uses: stage > success_rate > average_reward.
  // Returns true only when candidate is STRICTLY better than current (ties keep current).
  function isStrictlyBetterCheckpoint(candidate: CheckpointEntry, current: CheckpointEntry): boolean {
    const cStage = candidate.reward_config?.instincts?.curriculum_stage ?? 0;
    const curStage = current.reward_config?.instincts?.curriculum_stage ?? 0;
    if (cStage > curStage) return true;
    if (cStage < curStage) return false;
    if (candidate.success_rate > current.success_rate) return true;
    if (candidate.success_rate < current.success_rate) return false;
    // Fewer steps = faster completion = strictly better (reward not a tiebreaker).
    return (candidate.average_completion_steps ?? Infinity) < (current.average_completion_steps ?? Infinity);
  }

  const latestCheckpointEpisode = useMemo(() => {
    const checkpoints = checkpointIndex?.checkpoints;
    if (!checkpoints?.length) {
      return trainingStatus?.latest_checkpoint_episode ?? null;
    }
    return checkpoints[checkpoints.length - 1]?.checkpoint_episode ?? null;
  }, [checkpointIndex, trainingStatus?.latest_checkpoint_episode]);

  const bestCheckpointEpisode = useMemo(() => {
    if (!checkpointIndex?.checkpoints.length) return null;
    const best = checkpointIndex.checkpoints.reduce((acc, entry) => {
      if (!acc) return entry;
      return isStrictlyBetterCheckpoint(entry, acc) ? entry : acc;
    }, null as CheckpointEntry | null);
    return best?.checkpoint_episode ?? null;
  }, [checkpointIndex]);

  // Checkpoints that were once the running best but have since been surpassed.
  const pastBestEpisodes = useMemo(() => {
    const checkpoints = checkpointIndex?.checkpoints;
    if (!checkpoints?.length) return new Set<number>();
    const sorted = [...checkpoints].sort((a, b) => a.checkpoint_episode - b.checkpoint_episode);
    const past = new Set<number>();
    let runningBest: CheckpointEntry | null = null;
    for (const entry of sorted) {
      if (!runningBest) { runningBest = entry; continue; }
      if (isStrictlyBetterCheckpoint(entry, runningBest)) {
        past.add(runningBest.checkpoint_episode);
        runningBest = entry;
      }
    }
    return past;
  }, [checkpointIndex]);

  // Show every checkpoint that was ever the running best — this gives the full
  // learning-curve progression (first best, second best, …, current best).
  // If no checkpoint was ever better than the first, all are shown as a fallback.
  const visibleCheckpoints = useMemo(() => {
    const checkpoints = checkpointIndex?.checkpoints ?? [];
    if (!checkpoints.length) return [];
    const everBest = new Set<number>(pastBestEpisodes);
    if (bestCheckpointEpisode !== null) everBest.add(bestCheckpointEpisode);
    return everBest.size > 0 ? checkpoints.filter((c) => everBest.has(c.checkpoint_episode)) : checkpoints;
  }, [checkpointIndex, pastBestEpisodes, bestCheckpointEpisode]);

  const currentBestEntry = useMemo(() => {
    if (bestCheckpointEpisode === null || !checkpointIndex) return null;
    return checkpointIndex.checkpoints.find((c) => c.checkpoint_episode === bestCheckpointEpisode) ?? null;
  }, [checkpointIndex, bestCheckpointEpisode]);

  const previousBestEntry = useMemo(() => {
    if (!checkpointIndex || pastBestEpisodes.size === 0) return null;
    const prevEp = Math.max(...pastBestEpisodes);
    return checkpointIndex.checkpoints.find((c) => c.checkpoint_episode === prevEp) ?? null;
  }, [checkpointIndex, pastBestEpisodes]);

  const snapshot = currentFrame?.snapshot ?? replay?.final_snapshot ?? null;

  const scenarioCurrentFrame =
    scenarioReplay?.frames?.[
      Math.min(scenarioFrameIndex, Math.max((scenarioReplay?.frames.length ?? 0) - 1, 0))
    ] ?? null;
  const scenarioSnapshot =
    scenarioCurrentFrame?.snapshot ?? scenarioReplay?.final_snapshot ?? null;
  const fieldSnapshot = activeTab === "test" ? scenarioSnapshot : snapshot;

  async function refreshScenarioIndex() {
    try {
      const index = await loadScenarioIndex();
      setScenarioIndex(index);
    } catch {
      setScenarioIndex({
        scenarios: [],
        runs: [],
        best_by_scenario: {},
        latest_checkpoint_episode: latestCheckpointEpisode,
        latest_runs: [],
      });
    }
  }

  function buildScenarioCheckpointRequest() {
    return {
      checkpoint_mode: scenarioCheckpointMode,
      checkpoint_episode:
        scenarioCheckpointMode === "specific" ? specificScenarioCheckpointEpisode ?? undefined : undefined,
      policy_mode: policySelectionForCheckpoint(
        scenarioCheckpointMode === "specific" ? specificScenarioCheckpointEpisode : latestCheckpointEpisode,
      ).policyMode,
      trainer_type: policySelectionForCheckpoint(
        scenarioCheckpointMode === "specific" ? specificScenarioCheckpointEpisode : latestCheckpointEpisode,
      ).trainerType,
      policy_type: policySelectionForCheckpoint(
        scenarioCheckpointMode === "specific" ? specificScenarioCheckpointEpisode : latestCheckpointEpisode,
      ).policyType,
      effective_config: {
        enable_instinct_rewards: effectiveEnableInstincts,
        curriculum_stage: effectiveCurriculumStage,
        debug_reward_breakdown: effectiveDebugRewardBreakdown,
      },
    };
  }

  function applyCheckpointIndex(index: CheckpointIndex | null) {
    setCheckpointIndex(index);
    if (!index) {
      setReplay(null);
      setSelectedCheckpointEpisode(null);
      setSelectedSeed(11);
      setRunState("idle");
      setFrameIndex(0);
      return;
    }
    const latestCheckpoint = index.checkpoints[index.checkpoints.length - 1] ?? null;
    const checkpointEpisode = latestCheckpoint?.checkpoint_episode ?? index.latest?.checkpoint_episode ?? null;
    setSelectedCheckpointEpisode(checkpointEpisode);
    const seed = latestCheckpoint?.records[0]?.seed ?? index.latest?.records[0]?.seed ?? null;
    setSelectedSeed(seed);
  }

  useEffect(() => {
    let active = true;
    void (async () => {
      try {
        const index = await loadCheckpointIndex();
        if (!active) {
          return;
        }
        applyCheckpointIndex(index);
        await refreshScenarioIndex();
      } catch (loadError) {
        setError(loadError instanceof Error ? loadError.message : "Unable to load exported checkpoint data.");
      } finally {
        if (active) {
          setLoading(false);
        }
      }
    })();
    return () => {
      active = false;
    };
  }, []);

  useEffect(() => {
    let active = true;
    void (async () => {
      try {
        const hyperparams = await loadHyperparams();
        if (active) {
          setScenarioPersonalityStrength(hyperparams.environment.sheep_personality_strength);
        }
      } catch {
        // Keep default when hyperparams are unavailable.
      }
    })();
    return () => {
      active = false;
    };
  }, []);

  useEffect(() => {
    let active = true;

    void (async () => {
      try {
        const status = await loadTrainingStatus();
        if (active) {
          setTrainingStatus((previous) => mergeTrainingStatus(previous, status));
          // Adopt the server's stage on initial load when training is running,
          // or when the server reports a HIGHER stage than localStorage (the
          // user trained further on another session).  Never overwrite a local
          // stage that is >= the server's, so a pending Promote click is not
          // immediately reverted by polling.
          const localStage = parseInt(
            localStorage.getItem("sheepdog_curriculum_stage") ?? "0",
            10,
          );
          if (
            status.running ||
            (status.curriculum_stage != null && status.curriculum_stage > localStage)
          ) {
            syncTrainingStageFromStatus(status);
          }
        }
      } catch {
        if (active) {
          setTrainingStatus(null);
        }
      }
    })();

    const timer = window.setInterval(() => {
      void (async () => {
        try {
          const status = await loadTrainingStatus();
          if (active) {
            setTrainingStatus((previous) => mergeTrainingStatus(previous, status));
            // Guard: only overwrite the UI stage while a batch is in flight.
            // The running→idle useEffect handles locking the stage once done.
            if (status.running) {
              syncTrainingStageFromStatus(status);
            }
          }
        } catch {
          if (active) {
            setTrainingStatus(null);
          }
        }
      })();
    }, 500);

    const refreshNow = () => {
      if (document.visibilityState !== "visible") {
        return;
      }
      void (async () => {
        try {
          const status = await loadTrainingStatus();
          if (active) {
            setTrainingStatus((previous) => mergeTrainingStatus(previous, status));
            if (status.running) {
              syncTrainingStageFromStatus(status);
            }
          }
        } catch {
          if (active) {
            setTrainingStatus(null);
          }
        }
      })();
    };
    document.addEventListener("visibilitychange", refreshNow);
    window.addEventListener("focus", refreshNow);

    return () => {
      active = false;
      window.clearInterval(timer);
      document.removeEventListener("visibilitychange", refreshNow);
      window.removeEventListener("focus", refreshNow);
    };
  }, [syncTrainingStageFromStatus]);

  useEffect(() => {
    if (!trainingStatus?.running || trainingStatus.latest_checkpoint_episode === null) {
      return;
    }

    let cancelled = false;
    void (async () => {
      try {
        const index = await loadCheckpointIndex();
        if (cancelled || !index) {
          return;
        }
        setCheckpointIndex(index);
        void refreshScenarioIndex();
        const checkpoint =
          index.checkpoints.find((entry) => entry.checkpoint_episode === trainingStatus.latest_checkpoint_episode) ??
          index.checkpoints[index.checkpoints.length - 1] ??
          null;
        if (checkpoint) {
          setSelectedCheckpointEpisode(checkpoint.checkpoint_episode);
          const seed = checkpoint.records[0]?.seed ?? null;
          setSelectedSeed(seed);
          const record = checkpoint.records.find((entry) => entry.seed === seed) ?? checkpoint.records[0];
          if (record) {
            const bundle = await loadReplay(record.replay_path);
            if (cancelled) {
              return;
            }
            setReplay(bundle);
            setFrameIndex(0);
          }
        }
      } catch (loadError) {
        if (!cancelled) {
          setError(loadError instanceof Error ? loadError.message : "Unable to refresh training data.");
        }
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [trainingStatus?.latest_checkpoint_episode, trainingStatus?.running]);

  useEffect(() => {
    if (!selectedCheckpoint) {
      return;
    }
    const seed = selectedSeed ?? selectedCheckpoint.records[0]?.seed ?? null;
    if (seed === null) {
      return;
    }
    const record = selectedCheckpoint.records.find((entry) => entry.seed === seed) ?? selectedCheckpoint.records[0];
    if (!record) {
      return;
    }

    let cancelled = false;
    void (async () => {
      try {
        const bundle = await loadReplay(record.replay_path);
        if (cancelled) {
          return;
        }
        setReplay(bundle);
        setFrameIndex(0);
        setRunState("idle");
      } catch (loadError) {
        setError(loadError instanceof Error ? loadError.message : "Unable to load replay.");
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [selectedCheckpoint, selectedSeed]);

  useEffect(() => {
    if (runState !== "running" || !replay) {
      return;
    }
    const timer = window.setInterval(() => {
      setFrameIndex((currentIndex) => {
        if (currentIndex >= replay.frames.length - 1) {
          window.clearInterval(timer);
          setRunState(resolveRunState(replay.final_snapshot, "idle"));
          return currentIndex;
        }
        return currentIndex + 1;
      });
    }, playbackDelay);
    return () => window.clearInterval(timer);
  }, [playbackDelay, replay, runState]);

  useEffect(() => {
    if (scenarioRunState !== "running" || !scenarioReplay) {
      return;
    }
    const timer = window.setInterval(() => {
      setScenarioFrameIndex((currentIndex) => {
        if (currentIndex >= scenarioReplay.frames.length - 1) {
          window.clearInterval(timer);
          setScenarioRunState(resolveRunState(scenarioReplay.final_snapshot, "idle"));
          return currentIndex;
        }
        return currentIndex + 1;
      });
    }, scenarioPlaybackDelay);
    return () => window.clearInterval(timer);
  }, [scenarioPlaybackDelay, scenarioReplay, scenarioRunState]);

  function handleCheckpointChange(episode: number) {
    setSelectedCheckpointEpisode(episode);
    const checkpoint = checkpointIndex?.checkpoints.find((entry) => entry.checkpoint_episode === episode);
    setSelectedSeed(checkpoint?.records[0]?.seed ?? null);
  }

  function handleSeedChange(seed: number) {
    setSelectedSeed(seed);
  }

  function handlePromote() {
    // Capture the best formal checkpoint for this stage before bumping the
    // stage counter.  The episode ID is sent with the next training request so
    // the backend seeds Stage N+1 from the peak policy, not the hill-climbing
    // tail which may have lower formal success rate.
    setPromoteFromEpisode(bestStageFormalEntry?.checkpoint_episode ?? null);
    setTrainingCurriculumStage((prev) => {
      const next = prev >= 5 ? 1 : prev + 1;
      localStorage.setItem("sheepdog_curriculum_stage", String(next));
      setTrainingEpisodes(RECOMMENDED_EPISODES_BY_STAGE[next] ?? 100);
      return next;
    });
  }

  async function handleStartTraining() {
    setTrainingError(null);
    setError(null);
    try {
      const status = await startTraining({
        episodes: trainingEpisodes,
        fast_mode: trainingFastMode,
        enable_instinct_rewards: trainingEnableInstincts,
        curriculum_stage: trainingCurriculumStage,
        debug_reward_breakdown: trainingDebugRewardBreakdown,
        promote_from_checkpoint_episode: promoteFromEpisode ?? undefined,
      });
      // Clear the promote hint after it has been consumed by the training request.
      setPromoteFromEpisode(null);
      setTrainingStatus(status);
    } catch (startError) {
      setTrainingError(startError instanceof Error ? startError.message : "Unable to start training.");
    }
  }

  async function handleClearTraining() {
    if (!window.confirm("Clear all saved checkpoints, evaluations, and training state?")) {
      return;
    }

    setClearingTraining(true);
    setTrainingError(null);
    setError(null);
    try {
      const status = await clearTraining();
      const index = await loadCheckpointIndex();
      setTrainingStatus(status);
      setTrainingCurriculumStage(1);
      localStorage.setItem("sheepdog_curriculum_stage", "1");
      setTrainingEpisodes(RECOMMENDED_EPISODES_BY_STAGE[1] ?? 50);      setPromoteFromEpisode(null);      applyCheckpointIndex(index);
      const latestCheckpoint = index?.checkpoints[index.checkpoints.length - 1] ?? null;
      const seed = latestCheckpoint?.records[0]?.seed ?? null;
      if (latestCheckpoint && seed !== null) {
        const record = latestCheckpoint.records.find((entry) => entry.seed === seed) ?? latestCheckpoint.records[0];
        if (record) {
          const bundle = await loadReplay(record.replay_path);
          setReplay(bundle);
        }
      } else {
        setReplay(null);
      }
      setFrameIndex(0);
      setRunState("idle");
    } catch (clearError) {
      setTrainingError(clearError instanceof Error ? clearError.message : "Unable to clear training.");
    } finally {
      setClearingTraining(false);
    }
  }

  function policySelectionForCheckpoint(episode: number | null) {
    const entry =
      checkpointIndex?.checkpoints.find((checkpoint) => checkpoint.checkpoint_episode === episode) ?? null;
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

  async function handleSaveScenario() {
    const layoutSnapshot =
      saveSnapshotSource === "initial"
        ? scenarioReplay?.frames?.[0]?.snapshot ?? scenarioSnapshot
        : scenarioCurrentFrame?.snapshot ?? scenarioReplay?.final_snapshot ?? scenarioSnapshot;
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
      setError(runError instanceof Error ? runError.message : "Unable to replay the saved scenario.");
    } finally {
      setRunningScenario(false);
    }
  }

  async function handleRunScenario() {
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
    if (!scenarioReplay) {
      return;
    }
    setScenarioFrameIndex(0);
    setScenarioRunState("running");
  }

  function handleEndScenarioEpisode() {
    if (!scenarioReplay) {
      return;
    }
    setScenarioFrameIndex(Math.max(scenarioReplay.frames.length - 1, 0));
    setScenarioRunState(resolveRunState(scenarioReplay.final_snapshot, "idle"));
  }

  async function handleRunCurrentReplay() {
    setRunningCurrentReplay(true);
    setError(null);
    setTrainingError(null);
    try {
      const bundle = await runReplay({
        seed: selectedSeed ?? 11,
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
      setReplay(bundle);
      setSelectedCheckpointEpisode(null);
      setSelectedSeed(bundle.seed);
      setFrameIndex(0);
      setRunState("running");
    } catch (runError) {
      setError(runError instanceof Error ? runError.message : "Unable to run the current dog team.");
    } finally {
      setRunningCurrentReplay(false);
    }
  }

  async function handleReplaySelected() {
    const checkpoint = checkpointIndex?.checkpoints.find(
      (entry) => entry.checkpoint_episode === selectedCheckpointEpisode,
    );
    const record =
      checkpoint?.records.find((r) => r.seed === selectedSeed) ??
      checkpoint?.records[0];
    if (!record) {
      // No specific record found — just restart the current replay from the top.
      setFrameIndex(0);
      setRunState("running");
      return;
    }
    setLoadingSelectedReplay(true);
    setError(null);
    try {
      const bundle = await loadReplay(record.replay_path);
      setReplay(bundle);
      setSelectedSeed(record.seed);
      setFrameIndex(0);
      setRunState("running");
    } catch (loadError) {
      setError(loadError instanceof Error ? loadError.message : "Unable to load replay.");
    } finally {
      setLoadingSelectedReplay(false);
    }
  }

  function handleEndEpisode() {
    if (!replay) {
      return;
    }
    setFrameIndex(Math.max(replay.frames.length - 1, 0));
    setRunState(resolveRunState(replay.final_snapshot, "idle"));
  }

  const statusLabel = resolveRunState(snapshot, runState);
  const statusMessage = trainingStatus?.message ?? "Idle";
  const canEndEpisode = Boolean(replay) && (runState === "running" || frameIndex < (replay?.frames.length ?? 0) - 1);
  const scenarioStatusLabel = resolveRunState(
    scenarioCurrentFrame?.snapshot ?? scenarioReplay?.final_snapshot ?? null,
    scenarioRunState,
  );
  const canEndScenarioEpisode =
    Boolean(scenarioReplay) &&
    (scenarioRunState === "running" || scenarioFrameIndex < (scenarioReplay?.frames.length ?? 0) - 1);
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
  const tabButtons = APP_TABS.map(({ id, label }) => (
    <button
      key={id}
      role="tab"
      aria-selected={activeTab === id}
      className={`tab${activeTab === id ? " tab--active" : ""}`}
      onClick={() => setActiveTab(id)}
    >
      {label}
    </button>
  ));

  return (
    <main className="app-shell">
      {error ? <div className="warning-box warning-box--error">{error}</div> : null}
      {trainingError ? <div className="warning-box warning-box--error">{trainingError}</div> : null}

      {/* Unified tab bar - always in same position */}
      <div className="app-tab-bar" role="tablist">
        {tabButtons}
      </div>

      {activeTab === "insights" || activeTab === "results" || activeTab === "config" ? (
        <div className="insights-fullscreen">
          {activeTab === "insights" ? (
            <DiagnosticsPanel
              checkpointIndex={checkpointIndex}
              bestCheckpointEpisode={bestCheckpointEpisode}
              trainingStatus={trainingStatus}
              effectiveCurriculumStage={effectiveCurriculumStage}
            />
          ) : activeTab === "results" ? (
            <ResultsPanel checkpointIndex={checkpointIndex} />
          ) : (
            <ConfigPanel />
          )}
        </div>
      ) : (
        <div className="layout-grid">
          <FieldView snapshot={fieldSnapshot} />
          <aside className="side-column">
            {activeTab === "train" ? (
              <TrainingPanel
                episodes={trainingEpisodes}
                fastMode={trainingFastMode}
                enableInstincts={effectiveEnableInstincts}
                curriculumStage={effectiveCurriculumStage}
                debugRewardBreakdown={effectiveDebugRewardBreakdown}
                running={trainingStatus?.running ?? false}
                clearing={clearingTraining}
                batchCompletedEpisodes={trainingStatus?.batch_completed_episodes ?? trainingStatus?.completed_episodes ?? 0}
                batchTotalEpisodes={trainingStatus?.batch_total_episodes ?? trainingStatus?.requested_episodes ?? 0}
                currentEpisode={trainingStatus?.current_episode ?? null}
                totalEpisodesTrained={trainingStatus?.total_episodes_trained ?? 0}
                startingEpisode={trainingStatus?.starting_episode ?? null}
                stageHistory={trainingStatus?.stage_history ?? {}}
                grandTotalEpisodes={trainingStatus?.grand_total_episodes ?? 0}
                phase={trainingStatus?.phase ?? "idle"}
                message={statusMessage}
                error={trainingStatus?.error ?? null}
                successRate={bestStageFormalEntry?.success_rate ?? null}
                activeTrainerType={trainingStatus?.trainer_type ?? null}
                activePolicyType={trainingStatus?.policy_type ?? null}
                activeInstincts={trainingStatus?.enable_instinct_rewards ?? null}
                activeCurriculumStage={trainingStatus?.curriculum_stage ?? null}
                latestSuccessRate={trainingStatus?.latest_success_rate ?? latestSuccessRate}
                latestAvgSheepPenned={trainingStatus?.latest_avg_sheep_penned ?? null}
                latestAvgReward={trainingStatus?.latest_avg_reward ?? null}
                latestTimeoutRate={trainingStatus?.latest_timeout_rate ?? null}
                latestAvgDistanceToPen={trainingStatus?.latest_avg_distance_to_pen ?? null}
                latestCheckpointEpisode={trainingStatus?.latest_checkpoint_episode ?? null}
                onEpisodesChange={setTrainingEpisodes}
                onFastModeChange={setTrainingFastMode}
                onEnableInstinctsChange={setTrainingEnableInstincts}
                onCurriculumStageChange={setTrainingCurriculumStage}
                onDebugRewardBreakdownChange={setTrainingDebugRewardBreakdown}
                onStartTraining={handleStartTraining}
                onClearTraining={handleClearTraining}
                onPromote={handlePromote}
                currentBestEntry={currentBestEntry}
                previousBestEntry={previousBestEntry}
                seedEpisode={trainingStatus?.seed_episode ?? null}
              />
            ) : activeTab === "test" ? (
              <>
                <StatusPanel
                  snapshot={scenarioSnapshot}
                  replay={scenarioReplay}
                  selectedCheckpoint={null}
                  selectedCheckpointEpisode={null}
                  bestCheckpointEpisode={bestCheckpointEpisode}
                  selectedSeed={scenarioSeed}
                  runState={scenarioStatusLabel}
                />
                <ScenarioPanel
                  personalityStrength={scenarioPersonalityStrength}
                  seed={scenarioSeed}
                  fastMode={scenarioFastMode}
                  running={runningScenario}
                  canEndEpisode={canEndScenarioEpisode}
                  hasReplay={Boolean(scenarioReplay)}
                  policyLabel={scenarioPolicyLabel}
                  onPersonalityStrengthChange={setScenarioPersonalityStrength}
                  onSeedChange={setScenarioSeed}
                  onFastModeChange={setScenarioFastMode}
                  onRun={handleRunScenario}
                  onRestart={handleRestartScenarioPlayback}
                  onEndEpisode={handleEndScenarioEpisode}
                  disabled={loading}
                />
                <SavedScenariosPanel
                  scenarioIndex={scenarioIndex}
                  checkpoints={checkpointIndex?.checkpoints ?? []}
                  latestCheckpointEpisode={latestCheckpointEpisode}
                  bestCheckpointEpisode={bestCheckpointEpisode}
                  checkpointMode={scenarioCheckpointMode}
                  specificCheckpointEpisode={specificScenarioCheckpointEpisode}
                  selectedScenarioId={selectedScenarioId}
                  scenarioReplay={scenarioReplay}
                  saveSnapshotSource={saveSnapshotSource}
                  running={runningScenario || evaluatingScenarios}
                  disabled={loading}
                  onCheckpointModeChange={setScenarioCheckpointMode}
                  onSpecificCheckpointChange={setSpecificScenarioCheckpointEpisode}
                  onSelectScenario={setSelectedScenarioId}
                  onSaveSnapshotSourceChange={setSaveSnapshotSource}
                  onSaveFromReplay={handleSaveScenario}
                  onEvaluate={handleEvaluateScenarios}
                  onRunScenario={handleRunSavedScenario}
                  canSaveFromReplay={Boolean(scenarioSnapshot)}
                />
              </>
            ) : (
              <>
                <StatusPanel
                  snapshot={snapshot}
                  replay={replay}
                  selectedCheckpoint={selectedCheckpoint as CheckpointEntry | null}
                  selectedCheckpointEpisode={selectedCheckpointEpisode}
                  bestCheckpointEpisode={bestCheckpointEpisode}
                  selectedSeed={selectedSeed}
                  runState={statusLabel}
                />
                <ControlBar
                  checkpoints={visibleCheckpoints}
                  selectedCheckpointEpisode={selectedCheckpointEpisode}
                  bestCheckpointEpisode={bestCheckpointEpisode}
                  seedOptions={seedOptions}
                  selectedSeed={selectedSeed}
                  runningCurrent={runningCurrentReplay}
                  canEndEpisode={canEndEpisode}
                  onSelectCheckpointEpisode={handleCheckpointChange}
                  onSelectSeed={handleSeedChange}
                  onStart={handleReplaySelected}
                  runningSelected={loadingSelectedReplay}
                  onEndEpisode={handleEndEpisode}
                  onRunCurrent={handleRunCurrentReplay}
                  disabled={loading}
                  fastMode={playbackFastMode}
                  onFastModeChange={setPlaybackFastMode}
                />
              </>
            )}
          </aside>
        </div>
      )}
    </main>
  );
}

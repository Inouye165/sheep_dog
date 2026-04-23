import { useEffect, useMemo, useState } from "react";
import { ControlBar } from "./components/ControlBar";
import { FieldView } from "./components/FieldView";
import { TrainingPanel } from "./components/TrainingPanel";
import { StatusPanel } from "./components/StatusPanel";
import { clearTraining, loadCheckpointIndex, loadReplay, loadTrainingStatus, runReplay, startTraining } from "./lib/api";
import type { CheckpointIndex, ReplayBundle, ReplaySnapshot, TrainingStatus } from "./state/types";

type RunState = "idle" | "running" | "paused" | "success" | "timeout" | "stopped";

const CLEAR_TRAINING_MESSAGE = "Training cleared. Baseline replay restored";
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
  if (next.phase === "idle" && next.message === "Idle" && previous?.message === CLEAR_TRAINING_MESSAGE) {
    return { ...next, message: previous.message };
  }
  return next;
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
  const [trainingEpisodes, setTrainingEpisodes] = useState(5);
  const [trainingFastMode, setTrainingFastMode] = useState(true);
  const [trainingEnableInstincts, setTrainingEnableInstincts] = useState(true);
  const [trainingCurriculumStage, setTrainingCurriculumStage] = useState(1);
  const [trainingDebugRewardBreakdown, setTrainingDebugRewardBreakdown] = useState(false);
  const [playbackFastMode, setPlaybackFastMode] = useState(false);
  const [trainingStatus, setTrainingStatus] = useState<TrainingStatus | null>(null);
  const [clearingTraining, setClearingTraining] = useState(false);
  const [runningCurrentReplay, setRunningCurrentReplay] = useState(false);

  const selectedCheckpoint = useMemo(() => {
    return checkpointIndex?.checkpoints.find((entry) => entry.checkpoint_episode === selectedCheckpointEpisode) ?? null;
  }, [checkpointIndex, selectedCheckpointEpisode]);

  const seedOptions = useMemo(
    () => (selectedCheckpoint ? selectedCheckpoint.records.map((record) => record.seed) : []),
    [selectedCheckpoint],
  );

  const playbackDelay = playbackFastMode ? 24 : 220;

  const currentFrame =
    replay?.frames?.[Math.min(frameIndex, Math.max((replay?.frames.length ?? 0) - 1, 0))] ?? null;
  const snapshot = currentFrame?.snapshot ?? replay?.final_snapshot ?? null;

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
        const status = await loadTrainingStatus();
        if (active) {
          setTrainingStatus((previous) => mergeTrainingStatus(previous, status));
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
          }
        } catch {
          if (active) {
            setTrainingStatus(null);
          }
        }
      })();
    }, 500);

    return () => {
      active = false;
      window.clearInterval(timer);
    };
  }, []);

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

  function handleCheckpointChange(episode: number) {
    setSelectedCheckpointEpisode(episode);
    const checkpoint = checkpointIndex?.checkpoints.find((entry) => entry.checkpoint_episode === episode);
    setSelectedSeed(checkpoint?.records[0]?.seed ?? null);
  }

  function handleSeedChange(seed: number) {
    setSelectedSeed(seed);
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
      });
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
      applyCheckpointIndex(index);
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

  async function handleRunCurrentReplay() {
    setRunningCurrentReplay(true);
    setError(null);
    setTrainingError(null);
    try {
      const bundle = await runReplay({ seed: selectedSeed ?? 11 });
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

  function handleStart() {
    if (!replay) {
      return;
    }
    setFrameIndex(0);
    setRunState("running");
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

  return (
    <main className="app-shell">
      <header className="hero">
        <div>
          <p className="eyebrow">Sheepdog Herding Lab</p>
          <h1>Train, clear, and watch the dog team.</h1>
          <p className="hero__copy">Only the training controls and the live run details you need right now.</p>
        </div>
      </header>

      {error ? <div className="warning-box warning-box--error">{error}</div> : null}
      {trainingError ? <div className="warning-box warning-box--error">{trainingError}</div> : null}

      <div className="layout-grid">
        <FieldView snapshot={snapshot} />
        <aside className="side-column">
          <TrainingPanel
            episodes={trainingEpisodes}
            fastMode={trainingFastMode}
            enableInstincts={trainingStatus?.enable_instinct_rewards ?? trainingEnableInstincts}
            curriculumStage={trainingStatus?.curriculum_stage ?? trainingCurriculumStage}
            debugRewardBreakdown={trainingStatus?.debug_reward_breakdown ?? trainingDebugRewardBreakdown}
            running={trainingStatus?.running ?? false}
            clearing={clearingTraining}
            batchCompletedEpisodes={trainingStatus?.batch_completed_episodes ?? trainingStatus?.completed_episodes ?? 0}
            batchTotalEpisodes={trainingStatus?.batch_total_episodes ?? trainingStatus?.requested_episodes ?? 0}
            totalEpisodesTrained={trainingStatus?.total_episodes_trained ?? 0}
            phase={trainingStatus?.phase ?? "idle"}
            message={statusMessage}
            error={trainingStatus?.error ?? null}
            onEpisodesChange={setTrainingEpisodes}
            onFastModeChange={setTrainingFastMode}
            onEnableInstinctsChange={setTrainingEnableInstincts}
            onCurriculumStageChange={setTrainingCurriculumStage}
            onDebugRewardBreakdownChange={setTrainingDebugRewardBreakdown}
            onStartTraining={handleStartTraining}
            onClearTraining={handleClearTraining}
          />

          <StatusPanel
            snapshot={snapshot}
            replay={replay}
            selectedCheckpointEpisode={selectedCheckpointEpisode}
            selectedSeed={selectedSeed}
            runState={statusLabel}
          />

          <ControlBar
            checkpointEpisodes={checkpointIndex?.checkpoints.map((entry) => entry.checkpoint_episode) ?? []}
            selectedCheckpointEpisode={selectedCheckpointEpisode}
            seedOptions={seedOptions}
            selectedSeed={selectedSeed}
            runningCurrent={runningCurrentReplay}
            canEndEpisode={canEndEpisode}
            onSelectCheckpointEpisode={handleCheckpointChange}
            onSelectSeed={handleSeedChange}
            onStart={handleStart}
            onEndEpisode={handleEndEpisode}
            onRunCurrent={handleRunCurrentReplay}
            disabled={loading}
            fastMode={playbackFastMode}
            onFastModeChange={setPlaybackFastMode}
          />
        </aside>
      </div>
    </main>
  );
}

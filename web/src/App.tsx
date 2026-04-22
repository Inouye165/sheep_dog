import { useEffect, useMemo, useState } from "react";
import { ControlBar } from "./components/ControlBar";
import { FieldView } from "./components/FieldView";
import { TrainingPanel } from "./components/TrainingPanel";
import { StatusPanel } from "./components/StatusPanel";
import { clearTraining, loadCheckpointIndex, loadReplay, loadTrainingStatus, startTraining } from "./lib/api";
import type {
  CheckpointEntry,
  CheckpointIndex,
  EvaluationSummary,
  ReplayBundle,
  ReplaySnapshot,
  TrainingStatus,
} from "./state/types";

type RunState = "idle" | "running" | "paused" | "success" | "timeout" | "stopped";

const CLEAR_TRAINING_MESSAGE = "Training cleared. Baseline replay restored";

function checkpointToSummary(checkpoint: CheckpointEntry | null): EvaluationSummary | null {
  if (!checkpoint) {
    return null;
  }
  return {
    checkpoint_episode: checkpoint.checkpoint_episode,
    policy_name: "trained-checkpoint",
    records: checkpoint.records,
    success_rate: checkpoint.success_rate,
    timeout_rate: checkpoint.timeout_rate,
    average_completion_steps: checkpoint.average_completion_steps,
    average_completion_seconds: checkpoint.average_completion_seconds,
    average_sheep_penned: checkpoint.average_sheep_penned,
    average_reward: checkpoint.average_reward,
  };
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

function mergeTrainingStatus(previous: TrainingStatus | null, next: TrainingStatus): TrainingStatus {
  if (next.phase === "idle" && next.message === "Idle" && previous?.message === CLEAR_TRAINING_MESSAGE) {
    return { ...next, message: previous.message };
  }
  return next;
}

export function App() {
  const [checkpointIndex, setCheckpointIndex] = useState<CheckpointIndex | null>(null);
  const [evaluation, setEvaluation] = useState<EvaluationSummary | null>(null);
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
  const [playbackFastMode, setPlaybackFastMode] = useState(false);
  const [trainingStatus, setTrainingStatus] = useState<TrainingStatus | null>(null);

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

  useEffect(() => {
    let active = true;
    void (async () => {
      try {
        const index = await loadCheckpointIndex();
        if (!active) {
          return;
        }
        setCheckpointIndex(index);
        setEvaluation(index.latest);
        const latestCheckpoint = index.checkpoints[index.checkpoints.length - 1] ?? null;
        const checkpointEpisode = latestCheckpoint?.checkpoint_episode ?? index.latest?.checkpoint_episode ?? null;
        setSelectedCheckpointEpisode(checkpointEpisode);
        const seed = latestCheckpoint?.records[0]?.seed ?? index.latest?.records[0]?.seed ?? null;
        setSelectedSeed(seed);
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
        if (cancelled) {
          return;
        }
        setCheckpointIndex(index);
        setEvaluation(index.latest);
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
            setEvaluation(checkpointToSummary(checkpoint));
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
        setEvaluation(checkpointToSummary(selectedCheckpoint));
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
      const status = await startTraining({ episodes: trainingEpisodes, fast_mode: trainingFastMode });
      setTrainingStatus(status);
    } catch (startError) {
      setTrainingError(startError instanceof Error ? startError.message : "Unable to start training.");
    }
  }

  async function handleClearTraining() {
    setTrainingError(null);
    setError(null);
    try {
      const status = await clearTraining();
      const index = await loadCheckpointIndex();
      setTrainingStatus(status);
      setCheckpointIndex(index);
      setEvaluation(index.latest);
      const latestCheckpoint = index.checkpoints[index.checkpoints.length - 1] ?? null;
      const checkpointEpisode = latestCheckpoint?.checkpoint_episode ?? index.latest?.checkpoint_episode ?? null;
      const seed = latestCheckpoint?.records[0]?.seed ?? index.latest?.records[0]?.seed ?? null;
      setSelectedCheckpointEpisode(checkpointEpisode);
      setSelectedSeed(seed);
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
    }
  }

  function handleStart() {
    if (!replay) {
      return;
    }
    setFrameIndex(0);
    setRunState("running");
  }

  const statusLabel = resolveRunState(snapshot, runState);
  const replayMetrics = replay?.final_snapshot ?? null;
  const currentReward = currentFrame?.reward ?? replay?.stats?.final_reward_breakdown ?? null;
  const statusMessage = trainingStatus?.message ?? "Idle";

  return (
    <main className="app-shell">
      <header className="hero">
        <div>
          <p className="eyebrow">Sheepdog Herding Lab</p>
          <h1>Watch a dog team herd sheep into the pen.</h1>
          <p className="hero__copy">
            Deterministic simulation, checkpoint replay, and a shared linear hill-climbing trainer that stays visible while it runs.
          </p>
        </div>
        <div className="hero__facts">
          <span>Backend: Python</span>
          <span>UI: React + TypeScript</span>
          <span>Replay path: public assets</span>
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
            running={trainingStatus?.running ?? false}
            batchCompletedEpisodes={trainingStatus?.batch_completed_episodes ?? trainingStatus?.completed_episodes ?? 0}
            batchTotalEpisodes={trainingStatus?.batch_total_episodes ?? trainingStatus?.requested_episodes ?? 0}
            totalEpisodesTrained={trainingStatus?.total_episodes_trained ?? 0}
            phase={trainingStatus?.phase ?? "idle"}
            message={statusMessage}
            bestScore={trainingStatus?.best_score ?? null}
            error={trainingStatus?.error ?? null}
            onEpisodesChange={setTrainingEpisodes}
            onFastModeChange={setTrainingFastMode}
            onStartTraining={handleStartTraining}
            onClearTraining={handleClearTraining}
          />

          <StatusPanel
            snapshot={snapshot}
            replay={replay}
            evaluation={evaluation}
            rewardBreakdown={currentReward}
            episodeOutcome={replay?.final_snapshot?.status ?? "unloaded"}
            selectedCheckpointEpisode={selectedCheckpointEpisode}
            selectedSeed={selectedSeed}
            runState={statusLabel}
            trainingStatus={trainingStatus}
          />

          <ControlBar
            checkpointEpisodes={checkpointIndex?.checkpoints.map((entry) => entry.checkpoint_episode) ?? []}
            selectedCheckpointEpisode={selectedCheckpointEpisode}
            seedOptions={seedOptions}
            selectedSeed={selectedSeed}
            policyMode={replay?.policy_name ?? "unloaded"}
            runState={statusLabel}
            onSelectCheckpointEpisode={handleCheckpointChange}
            onSelectSeed={handleSeedChange}
            onStart={handleStart}
            disabled={loading}
            fastMode={playbackFastMode}
            onFastModeChange={setPlaybackFastMode}
          />

          <section className="reward-card" aria-label="Reward summary">
            <div className="reward-card__header">
              <div>
                <p className="eyebrow">Reward breakdown</p>
                <h2>Current frame</h2>
              </div>
              <span className="pill pill--muted">{currentReward ? currentReward.total.toFixed(2) : "0.00"}</span>
            </div>
            <dl className="reward-grid">
              <div>
                <dt>Progress to pen</dt>
                <dd>{currentReward ? currentReward.progress_to_pen.toFixed(2) : "0.00"}</dd>
              </div>
              <div>
                <dt>Sheep penned</dt>
                <dd>{currentReward ? currentReward.sheep_penned.toFixed(2) : "0.00"}</dd>
              </div>
              <div>
                <dt>Flock cohesion</dt>
                <dd>{currentReward ? currentReward.flock_cohesion.toFixed(2) : "0.00"}</dd>
              </div>
              <div>
                <dt>Scatter penalty</dt>
                <dd>{currentReward ? currentReward.scatter_penalty.toFixed(2) : "0.00"}</dd>
              </div>
              <div>
                <dt>Time penalty</dt>
                <dd>{currentReward ? currentReward.time_penalty.toFixed(2) : "0.00"}</dd>
              </div>
              <div>
                <dt>No-progress penalty</dt>
                <dd>{currentReward ? currentReward.no_progress_penalty.toFixed(2) : "0.00"}</dd>
              </div>
            </dl>
          </section>

          <section className="checkpoint-card" aria-label="Checkpoint summary">
            <div className="reward-card__header">
              <div>
                <p className="eyebrow">Selected checkpoint</p>
                <h2>Episode {selectedCheckpointEpisode ?? "-"}</h2>
              </div>
              <span className="pill">{replayMetrics?.status ?? "idle"}</span>
            </div>
            <dl className="checkpoint-grid">
              <div>
                <dt>Success rate</dt>
                <dd>{evaluation ? `${Math.round(evaluation.success_rate * 100)}%` : "-"}</dd>
              </div>
              <div>
                <dt>Timeout rate</dt>
                <dd>{evaluation ? `${Math.round(evaluation.timeout_rate * 100)}%` : "-"}</dd>
              </div>
              <div>
                <dt>Completion steps</dt>
                <dd>{evaluation ? evaluation.average_completion_steps.toFixed(1) : "-"}</dd>
              </div>
              <div>
                <dt>Average reward</dt>
                <dd>{evaluation ? evaluation.average_reward.toFixed(2) : "-"}</dd>
              </div>
            </dl>
          </section>
        </aside>
      </div>
    </main>
  );
}

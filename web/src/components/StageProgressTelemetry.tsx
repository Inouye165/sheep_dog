import React, { useMemo } from "react";
import type { TrainingStatus, TrainingEpisode } from "../state/types";

export interface StageProgressTelemetryProps {
  trainingStatus?: TrainingStatus | null;
  isLiveTraining?: boolean;
  curriculumStage?: number;
  stageScopedCheckpointsCount?: number;
  episodesSinceEvaluation?: number;
  episodesUntilNextEvaluation?: number;
  nextEvaluationBoundary?: number;
  currentStageEp?: number;
  recentEpisodes?: TrainingEpisode[];
  className?: string;
}

/**
 * Format a duration in seconds into a friendly human-readable string.
 * Examples: 42s, 4m 12s, 1h 25m
 */
export function formatTimeSpan(seconds: number | null | undefined): string {
  if (seconds == null || !Number.isFinite(seconds) || seconds < 0) return "—";
  const sec = Math.round(seconds);
  if (sec < 60) {
    return `${sec}s`;
  }
  const m = Math.floor(sec / 60);
  const s = sec % 60;
  if (m < 60) {
    return `${m}m ${s.toString().padStart(2, "0")}s`;
  }
  const h = Math.floor(m / 60);
  const remM = m % 60;
  return `${h}h ${remM.toString().padStart(2, "0")}m`;
}

export function StageProgressTelemetry({
  trainingStatus,
  isLiveTraining = false,
  curriculumStage,
  stageScopedCheckpointsCount = 0,
  episodesSinceEvaluation = 0,
  episodesUntilNextEvaluation,
  nextEvaluationBoundary,
  currentStageEp = 0,
  recentEpisodes = [],
  className = "",
}: StageProgressTelemetryProps) {
  const stage = curriculumStage ?? trainingStatus?.curriculum_stage ?? 1;
  const isAwaitingCheckpoints = stageScopedCheckpointsCount === 0;

  // Check if evaluation benchmark is actively executing
  const isEvalInProgress = Boolean(
    trainingStatus?.evaluation_in_progress ||
    trainingStatus?.evaluation_status?.in_progress ||
    trainingStatus?.phase === "evaluation"
  );

  // Compute cycle target counts
  const { episodesDone, episodesRemaining, totalCycleEpisodes, progressPct } = useMemo(() => {
    if (isEvalInProgress) {
      const totalSeeds =
        trainingStatus?.evaluation_total_seeds ??
        trainingStatus?.evaluation_status?.total_seeds ??
        10;
      const completedSeeds =
        trainingStatus?.evaluation_completed_seeds ??
        trainingStatus?.evaluation_status?.completed_seeds ??
        0;
      const pct = totalSeeds > 0 ? Math.min(100, Math.round((completedSeeds / totalSeeds) * 100)) : 0;
      return {
        episodesDone: completedSeeds,
        episodesRemaining: Math.max(0, totalSeeds - completedSeeds),
        totalCycleEpisodes: totalSeeds,
        progressPct: pct,
      };
    }

    const done = Math.max(0, episodesSinceEvaluation);
    const boundary = nextEvaluationBoundary ?? (Math.ceil((currentStageEp + 1) / 50) * 50);
    const remaining = episodesUntilNextEvaluation != null
      ? Math.max(0, episodesUntilNextEvaluation)
      : Math.max(1, boundary - currentStageEp);
    const total = Math.max(1, done + remaining);
    const pct = Math.min(100, Math.max(0, Math.round((done / total) * 100)));

    return {
      episodesDone: done,
      episodesRemaining: remaining,
      totalCycleEpisodes: total,
      progressPct: pct,
    };
  }, [
    isEvalInProgress,
    trainingStatus,
    episodesSinceEvaluation,
    episodesUntilNextEvaluation,
    nextEvaluationBoundary,
    currentStageEp,
  ]);

  // Compute pacing / throughput (seconds per episode)
  const secPerEpisode = useMemo<number | null>(() => {
    // 1. Calculate from timestamps of recent episodes
    if (recentEpisodes && recentEpisodes.length >= 2) {
      const timestamps = recentEpisodes
        .slice(-20)
        .map((e) => (e.completed_at ? new Date(e.completed_at).getTime() : null))
        .filter((t): t is number => t !== null && !isNaN(t));

      if (timestamps.length >= 2) {
        const spanMs = Math.max(...timestamps) - Math.min(...timestamps);
        const count = timestamps.length - 1;
        if (spanMs > 0 && count > 0) {
          const sec = spanMs / 1000 / count;
          if (sec > 0.05 && sec < 120) {
            return sec;
          }
        }
      }
    }

    // 2. Fallback to runtime episodes_per_active_hour
    const epsPerHour = trainingStatus?.runtime?.episodes_per_active_hour;
    if (epsPerHour && epsPerHour > 0) {
      return 3600 / epsPerHour;
    }

    // 3. Fallback to training_seconds / total_episodes
    const trainingSec = trainingStatus?.runtime?.training_seconds;
    const totalEps = trainingStatus?.grand_total_episodes ?? trainingStatus?.total_episodes_trained;
    if (trainingSec && totalEps && totalEps > 0 && trainingSec > 0) {
      const rate = trainingSec / totalEps;
      if (rate > 0.05 && rate < 120) {
        return rate;
      }
    }

    return null;
  }, [recentEpisodes, trainingStatus]);

  // Calculate elapsed time, time left (ETA), and total estimated duration
  const { elapsedFormatted, timeLeftFormatted, totalEstimatedFormatted, throughputFormatted } = useMemo(() => {
    if (!secPerEpisode) {
      const defaultPace = 2.5; // reasonable baseline heuristic
      const estElapsed = episodesDone * defaultPace;
      const estLeft = episodesRemaining * defaultPace;
      return {
        elapsedFormatted: episodesDone > 0 ? formatTimeSpan(estElapsed) : "0s",
        timeLeftFormatted: episodesDone >= 2 ? `~${formatTimeSpan(estLeft)}` : "Calibrating...",
        totalEstimatedFormatted: episodesDone >= 2 ? `~${formatTimeSpan(estElapsed + estLeft)}` : "—",
        throughputFormatted: null,
      };
    }

    const elapsedSec = episodesDone * secPerEpisode;
    const leftSec = episodesRemaining * secPerEpisode;
    const totalSec = elapsedSec + leftSec;
    const epsPerMin = 60 / secPerEpisode;

    return {
      elapsedFormatted: formatTimeSpan(elapsedSec),
      timeLeftFormatted: episodesDone === 0 && episodesRemaining > 0
        ? `~${formatTimeSpan(leftSec)}`
        : `~${formatTimeSpan(leftSec)}`,
      totalEstimatedFormatted: formatTimeSpan(totalSec),
      throughputFormatted: epsPerMin >= 10
        ? `${Math.round(epsPerMin)} eps/min`
        : `${epsPerMin.toFixed(1)} eps/min`,
    };
  }, [secPerEpisode, episodesDone, episodesRemaining]);

  // Context title based on stage status
  const milestoneTitle = useMemo(() => {
    if (isEvalInProgress) {
      const seed = trainingStatus?.evaluation_seed ?? trainingStatus?.evaluation_status?.current_seed ?? "?";
      return `Formal Benchmark In Progress: Seed ${seed} (${episodesDone}/${totalCycleEpisodes} seeds)`;
    }
    if (isAwaitingCheckpoints) {
      return `Stage ${stage} Baseline Discovery: Progress to Checkpoint 1 (${episodesDone}/${totalCycleEpisodes} episodes)`;
    }
    return `Confidence Evaluation Cadence: ${episodesDone}/${totalCycleEpisodes} episodes to next checkpoint`;
  }, [isEvalInProgress, isAwaitingCheckpoints, stage, episodesDone, totalCycleEpisodes, trainingStatus]);

  return (
    <div
      className={`stage-progress-telemetry ${className}`}
      data-testid="stage-progress-telemetry"
      role="region"
      aria-label="Stage Cadence Progress"
    >
      {/* Header / Subtitle row */}
      <div className="stage-progress-telemetry__header">
        <div className="stage-progress-telemetry__title-wrap">
          <span className="stage-progress-telemetry__title">{milestoneTitle}</span>
          {isLiveTraining && (
            <span className="stage-progress-telemetry__live-badge">● LIVE PACE</span>
          )}
        </div>
        <div className="stage-progress-telemetry__pct-badge">
          {progressPct}%
        </div>
      </div>

      {/* Progress Track Bar */}
      <div
        className="stage-progress-telemetry__track"
        role="progressbar"
        aria-valuenow={progressPct}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-label="Evaluation cycle progress"
      >
        <div
          className={`stage-progress-telemetry__bar ${isLiveTraining ? "stage-progress-telemetry__bar--animated" : ""}`}
          style={{ width: `${Math.max(3, progressPct)}%` }}
        />
      </div>

      {/* 3-Column Time Metrics Strip */}
      <div className="stage-progress-telemetry__metrics">
        {/* Time Completed (Elapsed) */}
        <div className="stage-progress-telemetry__metric-item">
          <span className="stage-progress-telemetry__metric-label">Time Completed</span>
          <span className="stage-progress-telemetry__metric-value" data-testid="time-completed-val">
            {elapsedFormatted}
          </span>
          <span className="stage-progress-telemetry__metric-sub">
            {episodesDone} {isEvalInProgress ? "seeds done" : "episodes done"}
          </span>
        </div>

        {/* Time Left (ETA) */}
        <div className="stage-progress-telemetry__metric-item">
          <span className="stage-progress-telemetry__metric-label">Time Left (ETA)</span>
          <span
            className="stage-progress-telemetry__metric-value stage-progress-telemetry__metric-value--eta"
            data-testid="time-left-val"
          >
            {timeLeftFormatted}
          </span>
          <span className="stage-progress-telemetry__metric-sub">
            {episodesRemaining} {isEvalInProgress ? "seeds left" : "eps remaining"}
          </span>
        </div>

        {/* Total Estimated Time */}
        <div className="stage-progress-telemetry__metric-item">
          <span className="stage-progress-telemetry__metric-label">Estimated Total</span>
          <span className="stage-progress-telemetry__metric-value" data-testid="estimated-total-val">
            {totalEstimatedFormatted}
          </span>
          <span className="stage-progress-telemetry__metric-sub">
            {throughputFormatted ? `Throughput: ${throughputFormatted}` : `${totalCycleEpisodes} total cycle target`}
          </span>
        </div>
      </div>
    </div>
  );
}

import type {
  EvaluationSummary,
  ReplayBundle,
  ReplaySnapshot,
  RewardBreakdown,
  TrainingStatus,
} from "../state/types";

interface StatusPanelProps {
  snapshot: ReplaySnapshot | null;
  replay: ReplayBundle | null;
  evaluation: EvaluationSummary | null;
  rewardBreakdown: RewardBreakdown | null;
  episodeOutcome: string;
  selectedCheckpointEpisode: number | null;
  selectedSeed: number | null;
  runState: string;
  trainingStatus: TrainingStatus | null;
}

function formatPercent(value: number): string {
  return `${Math.round(value * 100)}%`;
}

export function StatusPanel({
  snapshot,
  replay,
  evaluation,
  rewardBreakdown,
  episodeOutcome,
  selectedCheckpointEpisode,
  selectedSeed,
  runState,
  trainingStatus,
}: StatusPanelProps) {
  const sheepPenned = snapshot?.penned_count ?? 0;
  const totalSheep = snapshot?.sheep?.length ?? replay?.final_snapshot?.sheep?.length ?? 0;
  const completion = totalSheep === 0 ? 0 : sheepPenned / totalSheep;

  return (
    <section className="status-card" aria-label="Run status">
      <div className="status-card__header">
        <div>
          <p className="eyebrow">Status</p>
          <h2>{runState}</h2>
        </div>
        <div className="status-card__badges">
          <span className="pill">Checkpoint {selectedCheckpointEpisode ?? "-"}</span>
          <span className="pill pill--muted">Seed {selectedSeed ?? "-"}</span>
        </div>
      </div>

      <div className="status-grid">
        <div>
          <span>Sheep penned</span>
          <strong>
            {sheepPenned}/{totalSheep}
          </strong>
        </div>
        <div>
          <span>Simulated time</span>
          <strong>{snapshot ? `${snapshot.simulated_seconds.toFixed(0)}s` : "-"}</strong>
        </div>
        <div>
          <span>Steps</span>
          <strong>{snapshot?.step ?? 0}</strong>
        </div>
        <div>
          <span>Reward</span>
          <strong>{rewardBreakdown ? rewardBreakdown.total.toFixed(2) : "-"}</strong>
        </div>
      </div>

      <div className="progress-shell" aria-label="Completion progress">
        <div className="progress-shell__bar" style={{ width: `${completion * 100}%` }} />
      </div>

      <div className="status-card__summary">
        <div>
          <span>Checkpoint success rate</span>
          <strong>{evaluation ? formatPercent(evaluation.success_rate) : "-"}</strong>
        </div>
        <div>
          <span>Average completion time</span>
          <strong>{evaluation ? `${evaluation.average_completion_seconds.toFixed(1)}s` : "-"}</strong>
        </div>
        <div>
          <span>No-progress steps</span>
          <strong>{snapshot?.no_progress_steps ?? 0}</strong>
        </div>
        <div>
          <span>Policy mode</span>
          <strong>{replay?.policy_name ?? "unloaded"}</strong>
        </div>
        <div>
          <span>Episode outcome</span>
          <strong>{episodeOutcome}</strong>
        </div>
        <div>
          <span>Training job</span>
          <strong>
            {trainingStatus
              ? `${(trainingStatus.total_episodes_trained ?? 0).toLocaleString()} eps trained${
                  trainingStatus.running
                    ? ` · batch ${trainingStatus.batch_completed_episodes ?? 0}/${trainingStatus.batch_total_episodes ?? trainingStatus.requested_episodes ?? 0}`
                    : ""
                }`
              : "idle"}
          </strong>
        </div>
      </div>

      {(replay?.stats?.no_progress_steps ?? 0) > 0 || snapshot?.status === "no-progress" ? (
        <div className="warning-box" role="status">
          No-progress guard is active. The episode should stop if progress stalls.
        </div>
      ) : null}
    </section>
  );
}

import type { ReplayBundle, ReplaySnapshot } from "../state/types";

interface StatusPanelProps {
  snapshot: ReplaySnapshot | null;
  replay: ReplayBundle | null;
  selectedCheckpointEpisode: number | null;
  selectedSeed: number | null;
  runState: string;
}

export function StatusPanel({
  snapshot,
  replay,
  selectedCheckpointEpisode,
  selectedSeed,
  runState,
}: StatusPanelProps) {
  const sheepPenned = snapshot?.penned_count ?? 0;
  const totalSheep = snapshot?.sheep?.length ?? replay?.final_snapshot?.sheep?.length ?? 0;
  const completion = totalSheep === 0 ? 0 : sheepPenned / totalSheep;

  return (
    <section className="status-card" aria-label="Run status">
      <div className="status-card__header">
        <div>
          <p className="eyebrow">Watch now</p>
          <h2>{snapshot ? "Live run" : "Waiting for replay"}</h2>
        </div>
        <div className="status-card__badges">
          <span className="pill">{selectedCheckpointEpisode === null ? "Current dogs" : `Checkpoint ${selectedCheckpointEpisode}`}</span>
          <span className="pill pill--muted">{runState}</span>
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
          <span>Seed</span>
          <strong>{selectedSeed ?? "-"}</strong>
        </div>
        <div>
          <span>Policy</span>
          <strong>{replay?.policy_name ?? "-"}</strong>
        </div>
      </div>

      <div className="progress-shell" aria-label="Completion progress">
        <div className="progress-shell__bar" style={{ width: `${completion * 100}%` }} />
      </div>

      {(replay?.stats?.no_progress_steps ?? 0) > 0 || snapshot?.status === "no-progress" ? (
        <div className="warning-box" role="status">
          No-progress guard is active. The episode should stop if progress stalls.
        </div>
      ) : null}
    </section>
  );
}

import type { ReplayBundle, ReplaySnapshot } from "../state/types";

function formatRoleDistribution(roleDistribution: Record<string, number> | undefined): string {
  if (!roleDistribution) {
    return "-";
  }

  const entries = Object.entries(roleDistribution).filter(([, count]) => count > 0);
  if (entries.length === 0) {
    return "-";
  }

  return entries.map(([role, count]) => `${role}: ${count}`).join(" | ");
}

function policyLabel(policyName: string | undefined): string {
  switch (policyName) {
    case "random_untrained":
      return "Random untrained";
    case "instinct_only":
      return "Instinct only";
    case "heuristic_expert":
      return "Heuristic expert";
    case "trained_policy":
      return "Trained policy";
    default:
      return policyName ?? "-";
  }
}

function policyExplanation(policyName: string | undefined): string | null {
  switch (policyName) {
    case "random_untrained":
      return "Random untrained dogs have no herding strategy and no knowledge of the pen.";
    case "instinct_only":
      return "Instinct-only dogs can chase, circle, avoid diving into the flock, and recover nearby sheep, but they do not know where the pen is.";
    case "heuristic_expert":
      return "This expert heuristic is pen-aware and uses scripted pressure positioning toward the target.";
    case "trained_policy":
      return "Pen-directed behavior here comes from learned training weights rather than default instinct.";
    default:
      return null;
  }
}

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
  const activeRoles = snapshot?.dogs?.map((dog) => dog.role).filter(Boolean) ?? [];
  const activeRoleSummary = activeRoles.length > 0 ? activeRoles.join(", ") : "-";
  const roleDistributionSummary = formatRoleDistribution(replay?.stats.role_distribution);
  const explanation = policyExplanation(replay?.policy_name);
  const averageDistanceToPen = snapshot?.average_distance_to_pen ?? replay?.stats.final_avg_distance_to_pen ?? 0;
  const flockSpread = snapshot?.flock_spread ?? replay?.stats.final_flock_spread ?? 0;
  const gridWidth =
    snapshot?.grid_width ??
    snapshot?.field_width ??
    replay?.final_snapshot?.grid_width ??
    replay?.final_snapshot?.field_width ??
    40;
  const gridHeight =
    snapshot?.grid_height ??
    snapshot?.field_height ??
    replay?.final_snapshot?.grid_height ??
    replay?.final_snapshot?.field_height ??
    30;

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
          <strong>{policyLabel(replay?.policy_name)}</strong>
        </div>
        <div>
          <span>Grid size</span>
          <strong>{`${gridWidth} x ${gridHeight}`}</strong>
        </div>
      </div>

      <div className="progress-shell" aria-label="Completion progress">
        <div className="progress-shell__bar" style={{ width: `${completion * 100}%` }} />
      </div>

      <div className="status-card__summary">
        <div>
          <span>No-progress steps</span>
          <strong>{snapshot?.no_progress_steps ?? replay?.stats.no_progress_steps ?? 0}</strong>
        </div>
        <div>
          <span>Policy mode</span>
          <strong>{replay?.policy_name ?? "unloaded"}</strong>
        </div>
        <div>
          <span>Active roles</span>
          <strong>{activeRoleSummary}</strong>
        </div>
        <div>
          <span>Episode outcome</span>
          <strong>{snapshot?.status ?? replay?.final_snapshot.status ?? "-"}</strong>
        </div>
        <div>
          <span>Role switches</span>
          <strong>{replay?.stats.role_switches ?? 0}</strong>
        </div>
        <div>
          <span>Role distribution</span>
          <strong>{roleDistributionSummary}</strong>
        </div>
        <div>
          <span>Collector activations</span>
          <strong>{replay?.stats.collector_activations ?? 0}</strong>
        </div>
        <div>
          <span>Blocker activations</span>
          <strong>{replay?.stats.blocker_activations ?? 0}</strong>
        </div>
        <div>
          <span>Sheep split events</span>
          <strong>{replay?.stats.sheep_split_events ?? 0}</strong>
        </div>
        <div>
          <span>Avg distance to pen</span>
          <strong>{averageDistanceToPen.toFixed(1)}</strong>
        </div>
        <div>
          <span>Flock spread</span>
          <strong>{flockSpread.toFixed(1)}</strong>
        </div>
      </div>
      {(replay?.stats?.no_progress_steps ?? 0) > 0 || snapshot?.status === "no-progress" ? (
        <div className="warning-box" role="status">
          No-progress guard is active. The episode should stop if progress stalls.
        </div>
      ) : null}

      {explanation ? (
        <div className="warning-box" role="note">
          {explanation}
        </div>
      ) : null}
    </section>
  );
}

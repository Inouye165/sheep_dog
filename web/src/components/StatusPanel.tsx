import type { CheckpointEntry, ReplayBundle, ReplaySnapshot } from "../state/types";

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
    case "neural_policy":
      return "Neural PPO policy";
    default:
      return policyName ?? "-";
  }
}

function replayModeLabel(replayMode: string | undefined): string {
  switch (replayMode) {
    case "trained_linear":
      return "Trained linear";
    case "neural_ppo":
      return "Neural PPO";
    case "baseline":
      return "Baseline";
    default:
      return replayMode ?? "-";
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
    case "neural_policy":
      return "This replay is using the learned neural PPO policy rather than the instinct-only baseline.";
    default:
      return null;
  }
}

interface StatusPanelProps {
  snapshot: ReplaySnapshot | null;
  replay: ReplayBundle | null;
  selectedCheckpoint: CheckpointEntry | null;
  selectedCheckpointEpisode: number | null;
  selectedSeed: number | null;
  runState: string;
}

export function StatusPanel({
  snapshot,
  replay,
  selectedCheckpoint,
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
  const trainerType = replay?.trainer_type ?? selectedCheckpoint?.trainer_type ?? "-";
  const policyType = replay?.policy_type ?? selectedCheckpoint?.policy_type ?? "-";
  const policyMode = replay?.policy_mode ?? selectedCheckpoint?.policy_mode ?? replay?.policy_name ?? "unloaded";
  const replayMode = replay?.replay_mode ?? selectedCheckpoint?.replay_mode ?? "-";
  const replayDogs = replay?.environment?.dogs ?? selectedCheckpoint?.environment_config?.dogs ?? snapshot?.dogs.length ?? replay?.final_snapshot?.dogs.length ?? 0;
  const replaySheep = replay?.environment?.sheep ?? selectedCheckpoint?.environment_config?.sheep ?? snapshot?.sheep.length ?? replay?.final_snapshot?.sheep.length ?? 0;
  const replayCurriculumStage =
    replay?.environment?.curriculum_stage ??
    selectedCheckpoint?.reward_config?.instincts?.curriculum_stage ??
    snapshot?.debug?.curriculum_stage ??
    0;
  const replayUsesInstinctRewards =
    replay?.environment?.enable_instinct_rewards ??
    selectedCheckpoint?.reward_config?.instincts?.enable_instinct_rewards ??
    false;
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
          <strong>{policyLabel(policyMode)}</strong>
        </div>
        <div>
          <span>Grid size</span>
          <strong>{`${gridWidth} x ${gridHeight}`}</strong>
        </div>
        <div>
          <span>Replay kind</span>
          <strong>{replayModeLabel(replayMode)}</strong>
        </div>
        <div>
          <span>Dogs / sheep</span>
          <strong>{`${replayDogs} / ${replaySheep}`}</strong>
        </div>
        <div>
          <span>Curriculum stage</span>
          <strong>{replayCurriculumStage}</strong>
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
          <span>Trainer type</span>
          <strong>{trainerType}</strong>
        </div>
        <div>
          <span>Policy type</span>
          <strong>{policyType}</strong>
        </div>
        <div>
          <span>Policy mode</span>
          <strong>{policyMode}</strong>
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
        <div>
          <span>Instinct rewards</span>
          <strong>{replayUsesInstinctRewards ? "enabled" : "disabled"}</strong>
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

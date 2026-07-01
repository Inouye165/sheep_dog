import type { CheckpointEntry } from "../state/types";

interface ControlBarProps {
  checkpoints: CheckpointEntry[];
  selectedCheckpointEpisode: number | null;
  bestCheckpointEpisode: number | null;
  seedOptions: number[];
  selectedSeed: number | null;
  runningCurrent: boolean;
  canEndEpisode: boolean;
  onSelectCheckpointEpisode: (episode: number) => void;
  onSelectSeed: (seed: number) => void;
  onStart: () => void;
  onEndEpisode: () => void;
  onRunCurrent: () => void;
  runningSelected?: boolean;
  disabled?: boolean;
  fastMode: boolean;
  onFastModeChange: (enabled: boolean) => void;
}

export function ControlBar({
  checkpoints,
  selectedCheckpointEpisode,
  bestCheckpointEpisode,
  seedOptions,
  selectedSeed,
  runningCurrent,
  canEndEpisode,
  onSelectCheckpointEpisode,
  onSelectSeed,
  onStart,
  onEndEpisode,
  onRunCurrent,
  runningSelected,
  disabled,
  fastMode,
  onFastModeChange,
}: ControlBarProps) {
  const runBestLabel = runningCurrent
    ? "Running best model…"
    : bestCheckpointEpisode !== null
      ? `Run best model (ep ${bestCheckpointEpisode})`
      : "Run best model";

  return (
    <section className="controls-card" aria-label="Playback controls">
      <div className="controls-grid">
        <label>
          <span>Checkpoint</span>
          <select
            value={selectedCheckpointEpisode ?? ""}
            onChange={(event) => onSelectCheckpointEpisode(Number(event.target.value))}
            disabled={disabled || checkpoints.length === 0}
          >
            {checkpoints.length === 0 ? <option value="">No checkpoints exported</option> : null}
            {checkpoints.map((entry) => {
              const episode = entry.checkpoint_episode;
              const isBest = episode === bestCheckpointEpisode;
              const stage = entry.reward_config?.instincts?.curriculum_stage;
              const stageStr = stage != null ? `S${stage} · ` : "";
              const totalSeeds = entry.records?.length ?? 0;
              const successCount = totalSeeds > 0 ? Math.round(entry.success_rate * totalSeeds) : Math.round(entry.success_rate * 100);
              const successStr = totalSeeds > 0 ? `${successCount}/${totalSeeds} seeds` : `${Math.round(entry.success_rate * 100)}%`;
              const steps = Math.round(entry.average_completion_steps);
              const reward = entry.average_reward;
              const rewardStr = (reward >= 0 ? "+" : "") + reward.toFixed(1);
              let label = `${stageStr}Ep ${episode} · ${successStr} · ${steps} steps avg · ${rewardStr} R avg`;
              if (isBest) label += " — ★ Best";
              return (
                <option key={episode} value={episode}>
                  {label}
                </option>
              );
            })}
          </select>
        </label>

        <label>
          <span>Seed</span>
          <select
            value={selectedSeed ?? ""}
            onChange={(event) => onSelectSeed(Number(event.target.value))}
            disabled={disabled || seedOptions.length === 0}
          >
            {seedOptions.length === 0 ? <option value="">No seeds available</option> : null}
            {(() => {
              const selectedEntry = checkpoints.find((c) => c.checkpoint_episode === selectedCheckpointEpisode) ?? null;
              const totalSheep = selectedEntry?.environment_config?.sheep ?? null;
              return seedOptions.map((seed) => {
                const record = selectedEntry?.records?.find((r) => r.seed === seed);
                let seedLabel = `Seed ${seed}`;
                if (record) {
                  const outcome = record.success ? "success" : record.timeout ? "timeout" : "stopped";
                  const sheepStr = totalSheep != null ? `${record.sheep_penned}/${totalSheep}` : `${record.sheep_penned} sheep`;
                  seedLabel += ` — ${sheepStr} · ${record.steps} steps · ${outcome}`;
                }
                return (
                  <option key={seed} value={seed}>
                    {seedLabel}
                  </option>
                );
              });
            })()}
          </select>
        </label>

        <label className="training-toggle training-toggle--inline">
          <input
            type="checkbox"
            checked={fastMode}
            onChange={(event) => onFastModeChange(event.target.checked)}
            disabled={disabled}
          />
          <span>Fast playback</span>
        </label>
      </div>

      <div className="button-row">
        <button type="button" className="button-row__primary" onClick={onRunCurrent} disabled={runningCurrent}>
          {runBestLabel}
        </button>
        <button type="button" className="button-row__secondary" onClick={onStart} disabled={disabled || !selectedSeed || (runningSelected ?? false)}>
          {runningSelected ? "Loading…" : "Replay selected"}
        </button>
        <button
          type="button"
          className="button-row__secondary"
          onClick={onEndEpisode}
          disabled={disabled || !canEndEpisode}
        >
          End episode
        </button>
      </div>
    </section>
  );
}

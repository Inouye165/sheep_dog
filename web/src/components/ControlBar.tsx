interface ControlBarProps {
  checkpointEpisodes: number[];
  selectedCheckpointEpisode: number | null;
  seedOptions: number[];
  selectedSeed: number | null;
  runningCurrent: boolean;
  onSelectCheckpointEpisode: (episode: number) => void;
  onSelectSeed: (seed: number) => void;
  onStart: () => void;
  onRunCurrent: () => void;
  disabled?: boolean;
  fastMode: boolean;
  onFastModeChange: (enabled: boolean) => void;
}

export function ControlBar({
  checkpointEpisodes,
  selectedCheckpointEpisode,
  seedOptions,
  selectedSeed,
  runningCurrent,
  onSelectCheckpointEpisode,
  onSelectSeed,
  onStart,
  onRunCurrent,
  disabled,
  fastMode,
  onFastModeChange,
}: ControlBarProps) {
  return (
    <section className="controls-card" aria-label="Playback controls">
      <div className="controls-card__header">
        <div>
          <p className="eyebrow">Controls</p>
          <h2>Replay</h2>
        </div>
      </div>

      <div className="controls-grid">
        <label>
          <span>Checkpoint</span>
          <select
            value={selectedCheckpointEpisode ?? ""}
            onChange={(event) => onSelectCheckpointEpisode(Number(event.target.value))}
            disabled={disabled || checkpointEpisodes.length === 0}
          >
            {checkpointEpisodes.length === 0 ? <option value="">No checkpoints exported</option> : null}
            {checkpointEpisodes.map((episode) => (
              <option key={episode} value={episode}>
                Episode {episode}
              </option>
            ))}
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
            {seedOptions.map((seed) => (
              <option key={seed} value={seed}>
                Seed {seed}
              </option>
            ))}
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
          {runningCurrent ? "Running current dogs..." : "Run current dogs"}
        </button>
        <button type="button" className="button-row__secondary" onClick={onStart} disabled={disabled || !selectedSeed}>
          Start replay
        </button>
      </div>
    </section>
  );
}

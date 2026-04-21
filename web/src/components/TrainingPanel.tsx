interface TrainingPanelProps {
  episodes: number;
  fastMode: boolean;
  running: boolean;
  clearing: boolean;
  batchCompletedEpisodes: number;
  batchTotalEpisodes: number;
  totalEpisodesTrained: number;
  phase: string;
  message: string;
  error: string | null;
  onEpisodesChange: (episodes: number) => void;
  onFastModeChange: (enabled: boolean) => void;
  onStartTraining: () => void;
  onClearTraining: () => void;
}

export function TrainingPanel({
  episodes,
  fastMode,
  running,
  clearing,
  batchCompletedEpisodes,
  batchTotalEpisodes,
  totalEpisodesTrained,
  phase,
  message,
  error,
  onEpisodesChange,
  onFastModeChange,
  onStartTraining,
  onClearTraining,
}: TrainingPanelProps) {
  const denominator = batchTotalEpisodes || episodes;
  const progress = denominator === 0 ? 0 : Math.min(1, batchCompletedEpisodes / denominator);
  const safeTotal = Number.isFinite(totalEpisodesTrained) ? totalEpisodesTrained : 0;
  const busy = running || clearing;

  return (
    <section className="training-card" aria-label="Training controls">
      <div className="training-card__header">
        <div>
          <p className="eyebrow">Train</p>
          <h2>Training</h2>
        </div>
        <span className={`pill ${running ? "pill--live" : "pill--muted"}`}>{phase}</span>
      </div>

      <div className="training-grid">
        <label>
          <span>Episodes this batch</span>
          <input
            type="number"
            min={1}
            max={1000}
            value={episodes}
            onChange={(event) => onEpisodesChange(Number(event.target.value) || 1)}
            disabled={busy}
          />
        </label>

        <label className="training-toggle">
          <input
            type="checkbox"
            checked={fastMode}
            onChange={(event) => onFastModeChange(event.target.checked)}
            disabled={busy}
          />
          <span>Fast mode</span>
        </label>
      </div>

      <div className="training-summary">
        <div>
          <span>Total episodes trained</span>
          <strong>{safeTotal.toLocaleString()}</strong>
        </div>
        <div>
          <span>Batch progress</span>
          <strong>
            {batchCompletedEpisodes}/{denominator || "-"}
          </strong>
        </div>
        <div>
          <span>Status</span>
          <strong>{message}</strong>
        </div>
      </div>

      <div className="progress-shell" aria-label="Current batch progress">
        <div className="progress-shell__bar" style={{ width: `${progress * 100}%` }} />
      </div>

      {error ? <div className="warning-box warning-box--error">{error}</div> : null}

      <div className="button-row">
        <button type="button" className="button-row__primary" onClick={onStartTraining} disabled={busy}>
          {running ? "Training..." : `Train ${episodes} more`}
        </button>
        <button type="button" className="button-row__danger" onClick={onClearTraining} disabled={busy}>
          {clearing ? "Clearing..." : "Clear training data"}
        </button>
      </div>
    </section>
  );
}

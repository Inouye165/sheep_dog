const MAX_STAGE = 5;

const STAGE_DESCRIPTIONS: Record<number, string> = {
  0: "Full problem — 3 dogs, 6 sheep, 80×60 grid",
  1: "1 dog · 1 sheep · 60×45 grid",
  2: "1 dog · 3 sheep · 72×54 grid",
  3: "1 dog · 3 sheep · 120×84 grid",
  4: "2 dogs · 5 sheep · 132×90 grid",
  5: "3 dogs · 6 sheep · 144×96 grid",
};

interface TrainingPanelProps {
  episodes: number;
  fastMode: boolean;
  enableInstincts: boolean;
  curriculumStage: number;
  debugRewardBreakdown: boolean;
  running: boolean;
  clearing: boolean;
  batchCompletedEpisodes: number;
  batchTotalEpisodes: number;
  currentEpisode: number | null;
  totalEpisodesTrained: number;
  phase: string;
  message: string;
  error: string | null;
  successRate: number | null;
  onEpisodesChange: (episodes: number) => void;
  onFastModeChange: (enabled: boolean) => void;
  onEnableInstinctsChange: (enabled: boolean) => void;
  onCurriculumStageChange: (stage: number) => void;
  onDebugRewardBreakdownChange: (enabled: boolean) => void;
  onStartTraining: () => void;
  onClearTraining: () => void;
  onPromote: () => void;
}

export function TrainingPanel({
  episodes,
  fastMode,
  enableInstincts,
  curriculumStage,
  debugRewardBreakdown,
  running,
  clearing,
  batchCompletedEpisodes,
  batchTotalEpisodes,
  currentEpisode,
  totalEpisodesTrained,
  phase,
  message,
  error,
  successRate,
  onEpisodesChange,
  onFastModeChange,
  onEnableInstinctsChange,
  onCurriculumStageChange,
  onDebugRewardBreakdownChange,
  onStartTraining,
  onClearTraining,
  onPromote,
}: TrainingPanelProps) {
  const denominator = batchTotalEpisodes || episodes;
  const progress = denominator === 0 ? 0 : Math.min(1, batchCompletedEpisodes / denominator);
  const progressPct = Math.round(progress * 100);
  const completedDisplay = Math.floor(batchCompletedEpisodes);
  const safeTotal = Number.isFinite(totalEpisodesTrained) ? totalEpisodesTrained : 0;
  const busy = running || clearing;
  const canPromote = curriculumStage < MAX_STAGE && !busy;
  const stageDesc = STAGE_DESCRIPTIONS[curriculumStage] ?? `Stage ${curriculumStage}`;
  const successPct = successRate !== null ? `${Math.round(successRate * 100)}%` : "—";
  const successGood = successRate !== null && successRate >= 0.5;

  return (
    <section className="training-card" aria-label="Training controls">
      <div className="training-card__header">
        <div>
          <p className="eyebrow">Train</p>
          <h2>Training</h2>
        </div>
        <span className={`pill ${running ? "pill--live" : "pill--muted"}`}>{phase}</span>
      </div>

      {/* Stage row */}
      <div className="stage-row">
        <div className="stage-chip">
          <span className="stage-chip__label">Stage {curriculumStage}</span>
          <span className="stage-chip__desc">{stageDesc}</span>
        </div>
        {canPromote ? (
          <button type="button" className="button-row__promote" onClick={onPromote}>
            Promote → Stage {curriculumStage + 1}
          </button>
        ) : curriculumStage >= MAX_STAGE ? (
          <span className="pill pill--live">Max stage</span>
        ) : null}
      </div>

      {curriculumStage === 0 ? (
        <div className="warning-box warning-box--error" role="status">
          Stage 0 is the full problem — dogs rarely discover the pen from scratch.
          Promote to <strong>Stage 1</strong> to start simple.
        </div>
      ) : null}

      {/* Episodes + train */}
      <div className="training-primary">
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
      </div>

      <div
        className="progress-shell"
        role="progressbar"
        aria-label="Current batch progress"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={progressPct}
      >
        <div className="progress-shell__bar" style={{ width: `${progress * 100}%` }} />
        <span className="progress-shell__label">
          {running && currentEpisode !== null
            ? `Episode ${currentEpisode + 1} of ${denominator || "—"} · ${progressPct}%`
            : `${completedDisplay}/${denominator || "—"} · ${progressPct}%`}
        </span>
      </div>

      <div className="training-summary">
        <div>
          <span>Success rate</span>
          <strong style={{ color: successGood ? "var(--good)" : undefined }}>{successPct}</strong>
        </div>
        <div>
          <span>Total trained</span>
          <strong>{safeTotal.toLocaleString()}</strong>
        </div>
        <div>
          <span>Batch</span>
          <strong>
            {completedDisplay}/{denominator || "—"}
          </strong>
        </div>
        <div>
          <span>Status</span>
          <strong>{message}</strong>
        </div>
      </div>

      {/* Advanced settings — collapsed by default */}
      <details className="training-advanced">
        <summary>Advanced settings</summary>
        <div className="training-grid">
          <label className="training-toggle">
            <input
              type="checkbox"
              checked={fastMode}
              onChange={(event) => onFastModeChange(event.target.checked)}
              disabled={busy}
            />
            <span>Fast mode</span>
          </label>

          <label className="training-toggle">
            <input
              type="checkbox"
              checked={enableInstincts}
              onChange={(event) => onEnableInstinctsChange(event.target.checked)}
              disabled={busy}
            />
            <span>Enable instincts</span>
          </label>

          <label className="training-toggle">
            <input
              type="checkbox"
              checked={debugRewardBreakdown}
              onChange={(event) => onDebugRewardBreakdownChange(event.target.checked)}
              disabled={busy}
            />
            <span>Debug rewards</span>
          </label>

          <label>
            <span>Stage (manual)</span>
            <input
              type="number"
              min={0}
              max={5}
              value={curriculumStage}
              onChange={(event) => onCurriculumStageChange(Number(event.target.value) || 0)}
              disabled={busy}
            />
          </label>
        </div>

        <div className="warning-box">
          Old weights trained without instinct rewards may not transfer cleanly. Clear training data before switching to a new instincts curriculum run.
        </div>
      </details>

      {running ? (
        <div className="warning-box" role="status" aria-live="polite">
          Training runs server-side — you can switch tabs and the backend keeps going.
        </div>
      ) : null}

      {error ? <div className="warning-box warning-box--error">{error}</div> : null}

      <div className="button-row">
        <button type="button" className="button-row__primary" onClick={onStartTraining} disabled={busy}>
          {running ? "Training..." : `Train ${episodes} more`}
        </button>
        <button type="button" className="button-row__danger" onClick={onClearTraining} disabled={busy}>
          {clearing ? "Clearing..." : "Clear"}
        </button>
      </div>
    </section>
  );
}

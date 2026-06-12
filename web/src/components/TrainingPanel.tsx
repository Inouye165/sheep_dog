import type { CheckpointEntry } from "../state/types";

const STAGE_DESCRIPTIONS: Record<number, string> = {
  0: "Full problem — 3 dogs, 6 sheep, 80×60 grid",
  1: "1 dog · 1 sheep · 60×45 grid",
  2: "1 dog · 3 sheep · 72×54 grid",
  3: "1 dog · 3 sheep · 120×84 grid",
  4: "2 dogs · 5 sheep · 132×90 grid",
  5: "3 dogs · 6 sheep · 144×96 grid",
  6: "3 dogs · 6 sheep · nearby stray recovery",
  7: "3 dogs · 6 sheep · farther stray recovery",
  8: "3 dogs · 6 sheep · split/scattered recovery",
};

/** Ideal episode count to run per curriculum stage before evaluating. */
const RECOMMENDED_EPISODES: Record<number, number> = {
  0: 50,
  1: 50,
  2: 100,
  3: 150,
  4: 200,
  5: 300,
  6: 400,
  7: 500,
  8: 600,
};

/** Success rate threshold above which promoting to the next stage is recommended. */
const PROMOTE_THRESHOLD = 0.5;

interface TrainingPanelProps {
  episodes: number;
  fastMode: boolean;
  enableInstincts: boolean;
  curriculumStage: number;
  maxCurriculumStage: number;
  debugRewardBreakdown: boolean;
  running: boolean;
  clearing: boolean;
  batchCompletedEpisodes: number;
  batchTotalEpisodes: number;
  currentEpisode: number | null;
  totalEpisodesTrained: number;
  stageHistory: Record<string, number>;
  grandTotalEpisodes: number;
  phase: string;
  message: string;
  error: string | null;
  successRate: number | null;
  activeTrainerType?: string | null;
  activePolicyType?: string | null;
  activeInstincts?: boolean | null;
  activeCurriculumStage?: number | null;
  latestSuccessRate?: number | null;
  latestAvgSheepPenned?: number | null;
  latestAvgReward?: number | null;
  latestTimeoutRate?: number | null;
  latestAvgDistanceToPen?: number | null;
  latestCheckpointEpisode?: number | null;
  onEpisodesChange: (episodes: number) => void;
  onFastModeChange: (enabled: boolean) => void;
  onEnableInstinctsChange: (enabled: boolean) => void;
  onCurriculumStageChange: (stage: number) => void;
  onDebugRewardBreakdownChange: (enabled: boolean) => void;
  onStartTraining: () => void;
  onClearTraining: () => void;
  onPromote: () => void;
  currentBestEntry?: CheckpointEntry | null;
  previousBestEntry?: CheckpointEntry | null;
  seedEpisode?: number | null;
  startingEpisode?: number | null;
}

export function TrainingPanel({
  episodes,
  fastMode,
  enableInstincts,
  curriculumStage,
  maxCurriculumStage,
  debugRewardBreakdown,
  running,
  clearing,
  batchCompletedEpisodes,
  batchTotalEpisodes,
  currentEpisode,
  totalEpisodesTrained,
  stageHistory,
  grandTotalEpisodes,
  phase,
  message,
  error,
  successRate,
  activeTrainerType,
  activePolicyType,
  activeInstincts,
  activeCurriculumStage,
  latestSuccessRate,
  latestAvgSheepPenned,
  latestAvgReward,
  latestTimeoutRate,
  latestAvgDistanceToPen,
  latestCheckpointEpisode,
  onEpisodesChange,
  onFastModeChange,
  onEnableInstinctsChange,
  onCurriculumStageChange,
  onDebugRewardBreakdownChange,
  onStartTraining,
  onClearTraining,
  onPromote,
  currentBestEntry,
  previousBestEntry,
  seedEpisode,
  startingEpisode,
}: TrainingPanelProps) {
  const denominator = batchTotalEpisodes || episodes;
  const progress = denominator === 0 ? 0 : Math.min(1, batchCompletedEpisodes / denominator);
  const progressPct = Math.round(progress * 100);
  const completedDisplay = Math.floor(batchCompletedEpisodes);
  const safeTotal = Number.isFinite(totalEpisodesTrained) ? totalEpisodesTrained : 0;
  const safeGrand = Number.isFinite(grandTotalEpisodes) ? grandTotalEpisodes : 0;
  const stageHistoryEntries = Object.entries(stageHistory)
    .filter(([, v]) => v > 0)
    .sort(([a], [b]) => Number(a) - Number(b));
  const busy = running || clearing;
  const canPromote = curriculumStage < maxCurriculumStage && !busy;
  const stageDesc = STAGE_DESCRIPTIONS[curriculumStage] ?? `Stage ${curriculumStage}`;
  const successPct = successRate !== null ? `${Math.round(successRate * 100)}%` : "—";
  const successGood = successRate !== null && successRate >= 0.5;
  const recommendedEpisodes = RECOMMENDED_EPISODES[curriculumStage] ?? 100;
  const readyToPromote = successRate !== null && successRate >= PROMOTE_THRESHOLD && canPromote;

  function formatTimestamp(iso: string | null | undefined): string | null {
    if (!iso) return null;
    const d = new Date(iso);
    if (isNaN(d.getTime())) return null;
    const diffMin = Math.floor((Date.now() - d.getTime()) / 60_000);
    const timeStr = d.toLocaleTimeString(undefined, { hour: "numeric", minute: "2-digit" });
    let ago: string;
    if (diffMin < 1) ago = "just now";
    else if (diffMin < 60) ago = `${diffMin}m ago`;
    else {
      const h = Math.floor(diffMin / 60);
      const m = diffMin % 60;
      ago = m > 0 ? `${h}h ${m}m ago` : `${h}h ago`;
    }
    return `${timeStr} · ${ago}`;
  }

  function bestEntryLabel(entry: CheckpointEntry): string {
    const stage = entry.reward_config?.instincts?.curriculum_stage;
    const stageStr = stage != null ? `S${stage} · ` : "";
    return `${stageStr}Ep ${entry.checkpoint_episode}`;
  }

  // Active config — what the backend is actually using (or queued to use)
  const hasActiveConfig =
    activeTrainerType != null || activePolicyType != null || activeCurriculumStage != null;
  const activeStageLabel =
    activeCurriculumStage != null ? `Stage ${activeCurriculumStage}` : null;
  const activeInstinctsLabel =
    activeInstincts === true ? "instincts ON" : activeInstincts === false ? "instincts OFF" : null;

  // Latest checkpoint metrics
  const hasLatestMetrics =
    latestSuccessRate != null ||
    latestAvgSheepPenned != null ||
    latestAvgReward != null ||
    latestTimeoutRate != null ||
    latestAvgDistanceToPen != null;
  const fmtPct = (v: number | null | undefined) =>
    v != null ? `${Math.round(v * 100)}%` : "—";
  const fmtNum = (v: number | null | undefined, decimals = 2) =>
    v != null ? v.toFixed(decimals) : "—";

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
        ) : curriculumStage >= maxCurriculumStage ? (
          <span className="pill pill--live">Max stage</span>
        ) : null}
      </div>

      {readyToPromote ? (
        <div className="warning-box warning-box--success" role="status">
          ✓ {Math.round(successRate! * 100)}% success — ready to promote to Stage {curriculumStage + 1}
        </div>
      ) : successRate !== null && successRate < PROMOTE_THRESHOLD && !running && curriculumStage < maxCurriculumStage ? (
        <div className="warning-box" role="status">
          {Math.round(successRate * 100)}% success — train more before promoting
          (target ≥ {Math.round(PROMOTE_THRESHOLD * 100)}%)
        </div>
      ) : null}

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
        <div className="episodes-hint">
          Suggested: {recommendedEpisodes} for Stage {curriculumStage}
          {episodes !== recommendedEpisodes && !busy ? (
            <button
              type="button"
              className="episodes-hint__use"
              onClick={() => onEpisodesChange(recommendedEpisodes)}
            >
              Use suggested
            </button>
          ) : null}
        </div>
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
          <strong>{(safeGrand || safeTotal).toLocaleString()}</strong>
        </div>
        {startingEpisode != null ? (
          <div>
            <span>Starts from</span>
            <strong>{startingEpisode.toLocaleString()}</strong>
          </div>
        ) : null}
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
        {seedEpisode != null ? (
          <div>
            <span>Seed ep</span>
            <strong>{seedEpisode}</strong>
          </div>
        ) : null}
      </div>

      {(currentBestEntry || previousBestEntry) ? (
        <div className="best-perf">
          <div className={`best-perf__col${currentBestEntry ? " best-perf__col--current" : ""}`}>
            {currentBestEntry ? (
              <>
                <span className="best-perf__label">★ Best so far</span>
                <span className="best-perf__ep">{bestEntryLabel(currentBestEntry)}</span>
                <span
                  className="best-perf__rate"
                  style={{ color: currentBestEntry.success_rate >= 0.5 ? "var(--good)" : undefined }}
                >
                  {Math.round(currentBestEntry.success_rate * 100)}%
                </span>
                <span className="best-perf__reward">
                  {currentBestEntry.average_reward.toFixed(1)} R
                  {currentBestEntry.average_completion_steps != null
                    ? ` · ${Math.round(currentBestEntry.average_completion_steps)} steps`
                    : ""}
                </span>
                {formatTimestamp(currentBestEntry.recorded_at) ? (
                  <span className="best-perf__time">{formatTimestamp(currentBestEntry.recorded_at)}</span>
                ) : null}
              </>
            ) : null}
          </div>
          <div className="best-perf__col">
            {previousBestEntry ? (
              <>
                <span className="best-perf__label">Previous best</span>
                <span className="best-perf__ep">{bestEntryLabel(previousBestEntry)}</span>
                <span
                  className="best-perf__rate"
                  style={{ color: previousBestEntry.success_rate >= 0.5 ? "var(--good)" : "var(--muted)" }}
                >
                  {Math.round(previousBestEntry.success_rate * 100)}%
                </span>
                <span className="best-perf__reward">
                  {previousBestEntry.average_reward.toFixed(1)} R
                  {previousBestEntry.average_completion_steps != null
                    ? ` · ${Math.round(previousBestEntry.average_completion_steps)} steps`
                    : ""}
                </span>
                {formatTimestamp(previousBestEntry.recorded_at) ? (
                  <span className="best-perf__time">{formatTimestamp(previousBestEntry.recorded_at)}</span>
                ) : null}
              </>
            ) : null}
          </div>
        </div>
      ) : null}

      {stageHistoryEntries.length > 0 ? (
        <div className="stage-history">
          <span className="stage-history__label">Per-stage history</span>
          <div className="stage-history__entries">
            {stageHistoryEntries.map(([stage, eps]) => (
              <span key={stage} className="stage-history__entry">
                S{stage}: {eps.toLocaleString()}
              </span>
            ))}
            {stageHistoryEntries.length > 1 ? (
              <span className="stage-history__entry stage-history__entry--total">
                Total: {stageHistoryEntries.reduce((s, [, v]) => s + v, 0).toLocaleString()}
              </span>
            ) : null}
          </div>
        </div>
      ) : null}

      {hasLatestMetrics ? (
        <details className="training-advanced">
          <summary>Latest checkpoint metrics{latestCheckpointEpisode != null ? ` (ep ${latestCheckpointEpisode})` : ""}</summary>
          <div className="training-summary">
            <div>
              <span>Success</span>
              <strong style={{ color: latestSuccessRate != null && latestSuccessRate >= 0.5 ? "var(--good)" : undefined }}>
                {fmtPct(latestSuccessRate)}
              </strong>
            </div>
            <div>
              <span>Avg penned</span>
              <strong>{fmtNum(latestAvgSheepPenned)}</strong>
            </div>
            <div>
              <span>Avg reward</span>
              <strong>{fmtNum(latestAvgReward, 1)}</strong>
            </div>
            <div>
              <span>Timeout</span>
              <strong>{fmtPct(latestTimeoutRate)}</strong>
            </div>
            <div>
              <span>Dist-to-pen</span>
              <strong>{fmtNum(latestAvgDistanceToPen, 1)}</strong>
            </div>
          </div>
        </details>
      ) : null}

      {hasActiveConfig ? (
        <div className="active-config-banner" role="status">
          <span className="active-config-banner__label">Active:</span>
          {activeTrainerType ? <span>{activeTrainerType}</span> : null}
          {activePolicyType ? <span>{activePolicyType}</span> : null}
          {activeStageLabel ? <span>{activeStageLabel}</span> : null}
          {activeInstinctsLabel ? (
            <span style={{ color: activeInstincts ? "var(--good)" : "var(--muted)" }}>
              {activeInstinctsLabel}
            </span>
          ) : null}
        </div>
      ) : null}

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
              max={maxCurriculumStage}
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

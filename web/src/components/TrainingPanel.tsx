import type { AutoPromoteGateDiagnostics, CheckpointEntry } from "../state/types";

const STAGE_DESCRIPTIONS: Record<number, string> = {
  0: "Full problem — 3 dogs, 6 sheep, 80×60 grid",
  1: "1 dog · 1 sheep · fixed easy penning",
  2: "1 dog · 1 sheep · mild start randomization",
  3: "1 dog · 2 sheep · fixed mini-flock",
  4: "1 dog · 2 sheep · randomized mini-flock",
  5: "2 dogs · 3 sheep · fixed teamwork",
  6: "2 dogs · 3 sheep · tiny nearby stray starts",
  7: "2 dogs · 4 sheep · early nearby stray collection",
  8: "3 dogs · 4 sheep · nearby stray emphasis",
  9: "3 dogs · 4 sheep · stronger nearby stray recovery",
  10: "3 dogs · 5 sheep · nearby + first farther strays",
  11: "3 dogs · 5 sheep · farther stray recovery",
  12: "3 dogs · 6 sheep · group + one stray",
  13: "3 dogs · 6 sheep · two nearby strays",
  14: "3 dogs · 6 sheep · split flock (3+3)",
  15: "3 dogs · 6 sheep · partially scattered",
  16: "3 dogs · 6 sheep · scattered sheep",
  17: "3 dogs · 6 sheep · moving pen same wall",
  18: "3 dogs · 6 sheep · any-wall pen",
  19: "3 dogs · 6 sheep · wall pen away from corners",
  20: "3 dogs · 6 sheep · interior pen",
  21: "3 dogs · 6 sheep · random pen + random sheep",
  22: "3 dogs · 6 sheep · wider split/stray recovery",
  23: "3 dogs · 6 sheep · heavy scattered recovery",
  24: "3 dogs · 6 sheep · all-corners starts",
  25: "3 dogs · 6 sheep · bridge: corner-heavy starts before hard random mix",
  26: "3 dogs · 6 sheep · hard spawn mix (no personality bias)",
  27: "3 dogs · 6 sheep · add mild personality variation",
  28: "3 dogs · 6 sheep · moderate personality variation",
  29: "3 dogs · 6 sheep · bridge: weaker cohesion but still pressure-coupled",
  30: "3 dogs · 6 sheep · disable no-pressure cohesion",
  31: "3 dogs · 6 sheep · reduce cohesion + stronger personalities",
  32: "3 dogs · 6 sheep · lowest cohesion + strongest personalities",
};

/** Ideal episode count to run per curriculum stage before evaluating. */
const RECOMMENDED_EPISODES: Record<number, number> = {
  0: 50,
  1: 50,
  2: 75,
  3: 100,
  4: 125,
  5: 150,
  6: 175,
  7: 200,
  8: 225,
  9: 250,
  10: 275,
  11: 300,
  12: 325,
  13: 350,
  14: 375,
  15: 400,
  16: 450,
  17: 500,
  18: 550,
  19: 600,
  20: 650,
  21: 700,
  22: 750,
  23: 800,
  24: 850,
  25: 900,
  26: 950,
  27: 1000,
  28: 1050,
  29: 1100,
  30: 1200,
  31: 1300,
  32: 1400,
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
  autoPromote: boolean;
  autoPromoteThreshold?: number | null;
  autoPromoteStagesCompleted?: number;
  autoPromoteGate?: AutoPromoteGateDiagnostics | null;
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
  errorType?: string | null;
  traceback?: string | null;
  successRate: number | null;
  activeTrainerType?: string | null;
  activePolicyType?: string | null;
  activeInstincts?: boolean | null;
  activeCurriculumStage?: number | null;
  latestSuccessRate?: number | null;
  latestAvgSheepPenned?: number | null;
  latestAvgReward?: number | null;
  latestTimeoutRate?: number | null;
  latestStoppedRate?: number | null;
  latestAvgNoProgressSteps?: number | null;
  latestAvgDistanceToPen?: number | null;
  latestAvgFlockSpread?: number | null;
  latestAvgFarthestDistanceToPen?: number | null;
  latestAvgFarthestDistanceToFlockCenter?: number | null;
  latestCheckpointEpisode?: number | null;
  onEpisodesChange: (episodes: number) => void;
  onFastModeChange: (enabled: boolean) => void;
  onEnableInstinctsChange: (enabled: boolean) => void;
  onCurriculumStageChange: (stage: number) => void;
  onDebugRewardBreakdownChange: (enabled: boolean) => void;
  onAutoPromoteChange: (enabled: boolean) => void;
  onStartTraining: () => void;
  onPauseTraining: () => void;
  onStopTraining: () => void;
  onResumeTraining: () => void;
  onClearTraining: () => void;
  onResetJourney: () => void;
  onPromote: () => void;
  currentBestEntry?: CheckpointEntry | null;
  previousBestEntry?: CheckpointEntry | null;
  seedEpisode?: number | null;
  startingEpisode?: number | null;
  resumeAvailable?: boolean;
  resumeRemainingEpisodes?: number | null;
}

export function TrainingPanel({
  episodes,
  fastMode,
  enableInstincts,
  curriculumStage,
  maxCurriculumStage,
  debugRewardBreakdown,
  autoPromote,
  autoPromoteThreshold,
  autoPromoteStagesCompleted,
  autoPromoteGate,
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
  errorType,
  traceback,
  successRate,
  activeTrainerType,
  activePolicyType,
  activeInstincts,
  activeCurriculumStage,
  latestSuccessRate,
  latestAvgSheepPenned,
  latestAvgReward,
  latestTimeoutRate,
  latestStoppedRate,
  latestAvgNoProgressSteps,
  latestAvgDistanceToPen,
  latestAvgFlockSpread,
  latestAvgFarthestDistanceToPen,
  latestAvgFarthestDistanceToFlockCenter,
  latestCheckpointEpisode,
  onEpisodesChange,
  onFastModeChange,
  onEnableInstinctsChange,
  onCurriculumStageChange,
  onDebugRewardBreakdownChange,
  onAutoPromoteChange,
  onStartTraining,
  onPauseTraining,
  onStopTraining,
  onResumeTraining,
  onClearTraining,
  onResetJourney,
  onPromote,
  currentBestEntry,
  previousBestEntry,
  seedEpisode,
  startingEpisode,
  resumeAvailable = false,
  resumeRemainingEpisodes = null,
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
  const stageDesc = STAGE_DESCRIPTIONS[curriculumStage] ?? `Stage ${curriculumStage}`;
  const successPct = successRate !== null ? `${Math.round(successRate * 100)}%` : "—";
  const successGood = successRate !== null && successRate >= 0.5;
  const recommendedEpisodes = RECOMMENDED_EPISODES[curriculumStage] ?? 100;
  const hasPromotionHeadroom = curriculumStage < maxCurriculumStage;
  const canPromote = hasPromotionHeadroom && !busy;
  const readyToPromote = canPromote && successRate !== null && successRate >= PROMOTE_THRESHOLD;
  const canResume = !busy && resumeAvailable && (resumeRemainingEpisodes ?? 0) > 0;
  const effectiveAutoPromoteThreshold = autoPromoteThreshold ?? PROMOTE_THRESHOLD;
  const hasAutoPromoteGate = autoPromoteGate != null;
  const decisionToneClass =
    autoPromoteGate?.decision === "promote"
      ? "gate-pill gate-pill--pass"
      : autoPromoteGate?.decision === "hold"
        ? "gate-pill gate-pill--fail"
        : "gate-pill gate-pill--pending";
  const gateToneClass = (ok: boolean): string =>
    ok ? "gate-pill gate-pill--pass" : "gate-pill gate-pill--fail";

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
    latestStoppedRate != null ||
    latestAvgNoProgressSteps != null ||
    latestAvgDistanceToPen != null ||
    latestAvgFlockSpread != null ||
    latestAvgFarthestDistanceToPen != null ||
    latestAvgFarthestDistanceToFlockCenter != null;
  const fmtPct = (v: number | null | undefined) =>
    v != null ? `${Math.round(v * 100)}%` : "—";
  const fmtNum = (v: number | null | undefined, decimals = 2) =>
    v != null ? v.toFixed(decimals) : "—";

  return (
    <section className="training-card" aria-label="Training controls">
      <div className="training-card__header">
        <div>
          <p className="eyebrow">Curriculum learning</p>
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
          <button type="button" className="button-row__promote" onClick={onPromote} disabled={!readyToPromote}>
            Promote → Stage {curriculumStage + 1}
          </button>
        ) : curriculumStage >= maxCurriculumStage ? (
          <span className="pill pill--live">Max stage</span>
        ) : null}
      </div>

      {canPromote && !readyToPromote ? (
        <div className="warning-box" role="status">
          Promotion locked until Stage {curriculumStage} reaches ≥ {Math.round(PROMOTE_THRESHOLD * 100)}% success.
        </div>
      ) : null}

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

      <label className="training-toggle training-toggle--inline">
        <input
          type="checkbox"
          checked={enableInstincts}
          onChange={(event) => onEnableInstinctsChange(event.target.checked)}
          disabled={busy}
        />
        <span>Enable instincts</span>
      </label>

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

      <div className="button-row">
        <button type="button" className="button-row__primary" onClick={onStartTraining} disabled={busy}>
          {running ? "Training..." : `Train ${episodes} more`}
        </button>
        {running ? (
          <>
            <button type="button" onClick={onPauseTraining} disabled={clearing}>
              Pause after checkpoint
            </button>
            <button type="button" className="button-row__danger" onClick={onStopTraining} disabled={clearing}>
              Stop after checkpoint
            </button>
          </>
        ) : canResume ? (
          <button type="button" onClick={onResumeTraining} disabled={clearing}>
            Resume {resumeRemainingEpisodes} remaining
          </button>
        ) : null}
        <button type="button" className="button-row__danger" onClick={onClearTraining} disabled={busy}>
          {clearing ? "Clearing..." : "Clear"}
        </button>
        <button type="button" onClick={onResetJourney} disabled={busy}>
          Reset Journey
        </button>
      </div>

      <div style={{ marginTop: "0.75rem", fontSize: "0.75rem", color: "var(--muted)", display: "flex", flexDirection: "column", gap: "0.35rem", borderTop: "1px solid var(--panel-border)", paddingTop: "0.75rem", lineHeight: "1.4" }}>
        <div>
          <strong style={{ color: "var(--accent)" }}>Clear:</strong> Deletes current checkpoints, evaluations, and permanently deletes all archived journey history.
        </div>
        <div>
          <strong style={{ color: "var(--text)", opacity: 0.9 }}>Reset Journey:</strong> Archives current training progress to the history log and starts a fresh journey from Stage 1.
        </div>
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
        <div>
          <span>Auto-promotion</span>
          <strong>{autoPromote ? "ON" : "OFF"}</strong>
        </div>
        <div>
          <span>Auto threshold</span>
          <strong>{Math.round(effectiveAutoPromoteThreshold * 100)}%</strong>
        </div>
        <div>
          <span>Auto-promoted</span>
          <strong>{(autoPromoteStagesCompleted ?? 0).toLocaleString()} stages</strong>
        </div>
        {seedEpisode != null ? (
          <div>
            <span>Seed ep</span>
            <strong>{seedEpisode}</strong>
          </div>
        ) : null}
      </div>

      {hasAutoPromoteGate ? (
        <details className="training-advanced" open={autoPromoteGate.decision === "hold"}>
          <summary>Auto-promotion gate diagnostics</summary>
          <div className="training-summary">
            <div>
              <span>Decision</span>
              <strong className={decisionToneClass}>{autoPromoteGate.decision.toUpperCase()}</strong>
            </div>
            <div>
              <span>Reason</span>
              <strong>{autoPromoteGate.reason}</strong>
            </div>
            <div>
              <span>Seed gate</span>
              <strong className={gateToneClass(autoPromoteGate.seed_gate_target_met)}>
                {autoPromoteGate.seed_gate_hits}/{autoPromoteGate.min_seed_gate_hits}
                {autoPromoteGate.seed_gate_target_met ? " ✓" : ""}
              </strong>
            </div>
            <div>
              <span>Streak</span>
              <strong
                className={gateToneClass(
                  autoPromoteGate.qualified_streak >= autoPromoteGate.min_qualified_streak,
                )}
              >
                {autoPromoteGate.qualified_streak}/{autoPromoteGate.min_qualified_streak}
              </strong>
            </div>
            <div>
              <span>Best success</span>
              <strong>{Math.round(autoPromoteGate.best_success * 100)}%</strong>
            </div>
            <div>
              <span>Seeds in eval</span>
              <strong>{autoPromoteGate.seed_count}</strong>
            </div>
            <div>
              <span>Success gate</span>
              <strong className={gateToneClass(autoPromoteGate.success_rate_ok)}>
                {autoPromoteGate.success_rate_ok ? "pass" : "fail"}
              </strong>
            </div>
            <div>
              <span>Timeout gate</span>
              <strong className={gateToneClass(autoPromoteGate.timeout_ok)}>
                {autoPromoteGate.timeout_ok ? "pass" : "fail"}
              </strong>
            </div>
            <div>
              <span>Reward gate</span>
              <strong className={gateToneClass(autoPromoteGate.reward_close_ok)}>
                {autoPromoteGate.reward_close_ok ? "pass" : "fail"}
              </strong>
            </div>
            <div>
              <span>Full-success hits</span>
              <strong className={gateToneClass(autoPromoteGate.full_success_target_met)}>
                {autoPromoteGate.full_success_hits}/{autoPromoteGate.min_full_success_hits}
                {autoPromoteGate.full_success_target_met ? " ✓" : ""}
              </strong>
            </div>
          </div>
        </details>
      ) : null}

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
          <div className="stage-history__bars">
            {(() => {
              const maxEps = Math.max(...stageHistoryEntries.map(([, v]) => v));
              return stageHistoryEntries.map(([stage, eps]) => (
                <div key={stage} className="stage-history__bar-row">
                  <span className="stage-history__bar-key">S{stage}</span>
                  <span className="stage-history__bar-track">
                    <span
                      className="stage-history__bar-fill"
                      style={{ width: `${maxEps > 0 ? (eps / maxEps) * 100 : 0}%` }}
                    />
                  </span>
                  <span className="stage-history__bar-val">{eps.toLocaleString()}</span>
                </div>
              ));
            })()}
            {stageHistoryEntries.length > 1 ? (
              <div className="stage-history__bar-row stage-history__bar-row--total">
                <span className="stage-history__bar-key">Total</span>
                <span className="stage-history__bar-track" />
                <span className="stage-history__bar-val">
                  {stageHistoryEntries.reduce((s, [, v]) => s + v, 0).toLocaleString()}
                </span>
              </div>
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
              <span>No-progress stop</span>
              <strong>{fmtPct(latestStoppedRate)}</strong>
            </div>
            <div>
              <span>Dist-to-pen</span>
              <strong>{fmtNum(latestAvgDistanceToPen, 1)}</strong>
            </div>
            <div>
              <span>Flock spread</span>
              <strong>{fmtNum(latestAvgFlockSpread, 1)}</strong>
            </div>
            <div>
              <span>Farthest-to-pen</span>
              <strong>{fmtNum(latestAvgFarthestDistanceToPen, 1)}</strong>
            </div>
            <div>
              <span>Farthest-to-flock</span>
              <strong>{fmtNum(latestAvgFarthestDistanceToFlockCenter, 1)}</strong>
            </div>
            <div>
              <span>No-progress steps</span>
              <strong>{fmtNum(latestAvgNoProgressSteps, 1)}</strong>
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
              checked={debugRewardBreakdown}
              onChange={(event) => onDebugRewardBreakdownChange(event.target.checked)}
              disabled={busy}
            />
            <span>Debug rewards</span>
          </label>

          <label className="training-toggle">
            <input
              type="checkbox"
              checked={autoPromote}
              onChange={(event) => onAutoPromoteChange(event.target.checked)}
              disabled={busy}
            />
            <span>Auto-promote stages</span>
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

      {error ? (
        <div className="warning-box warning-box--error" role="alert">
          <strong>{errorType ? `${errorType}: ` : ""}</strong>
          {error}
          {traceback ? (
            <details className="training-advanced" style={{ marginTop: "0.5rem" }}>
              <summary>Technical traceback</summary>
              <pre style={{ whiteSpace: "pre-wrap", margin: "0.5rem 0 0" }}>{traceback}</pre>
            </details>
          ) : null}
        </div>
      ) : null}
    </section>
  );
}

import { useState } from "react";
import { CopyAgentDataButton } from "./CopyAgentDataButton";
import type { AutoPromoteGateDiagnostics, CheckpointEntry, CheckpointIndex, TrainingStatus } from "../state/types";

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

interface GlossaryItem {
  term: string;
  keyword: string; // Used for exact triggers
  category: string;
  definition: string;
  relevance: string;
}

const GLOSSARY_ITEMS: GlossaryItem[] = [
  {
    term: "Curriculum Learning",
    keyword: "curriculum",
    category: "Training Strategy",
    definition: "An optimization method where the agent starts learning very simple tasks and gradually advances to the full, complex task. Here, it scales from 1 dog herding 1 sheep to 3 dogs coordinating to herd 6 scattered sheep.",
    relevance: "Governs the training stage progression from Stage 1 up to Stage 32."
  },
  {
    term: "Episodes (Batch)",
    keyword: "episodes",
    category: "Reinforcement Learning",
    definition: "A single trial of herding from start to completion (either penning all sheep or reaching the timeout). Batches are groups of episodes run sequentially before policy evaluations and checkpoints are generated.",
    relevance: "Determined by the 'Episodes this batch' input value."
  },
  {
    term: "Instinct Rewards",
    keyword: "instincts",
    category: "Reward Shaping",
    definition: "Pre-programmed helper rewards based on expert heuristics (e.g. keeping dogs behind the flock, spatial spacing, stray recovery pressure). This guides the neural network before it discovers sparse success rewards.",
    relevance: "Controlled by the 'Enable instincts' toggle. Highly recommended for early curriculum stages."
  },
  {
    term: "Auto-Promotion",
    keyword: "auto-promote",
    category: "Curriculum Strategy",
    definition: "An automation checkpoint logic that promotes the network to the next curriculum stage when a set of criteria (minimum success rate, timeout threshold, low stray counts) are consistently met across multiple seeds.",
    relevance: "Managed by the 'Auto-promote stages' toggle and monitored via the diagnostics gate."
  },
  {
    term: "Success Rate",
    keyword: "success rate",
    category: "Evaluation Metric",
    definition: "The percentage of episodes within an evaluation batch where all sheep were successfully penned before timing out.",
    relevance: "The primary gate requirement for promoting to subsequent stages (target is >= 50%)."
  },
  {
    term: "No-progress Stop",
    keyword: "no-progress",
    category: "Simulation Controls",
    definition: "An early-termination check that stops an episode if the sheepdog flock hasn't advanced closer to the pen for too many steps, preventing wasted CPU cycles on frozen or deadlocked policies.",
    relevance: "Reflected in the 'No-progress stop' rate under metrics."
  },
  {
    term: "Timeout Rate",
    keyword: "timeout",
    category: "Evaluation Metric",
    definition: "The percentage of herding runs that reached the maximum step limit (e.g. 500 steps) before successfully penning all sheep.",
    relevance: "Ideally should be minimized (< 20%) to qualify for promotion."
  },
  {
    term: "Checkpoints",
    keyword: "checkpoints",
    category: "Model Lifecycle",
    definition: "Periodic snapshots of the policy's neural network weights. They allow the trainer to evaluate performance, rewind to previous stages, or resume interrupted training batches.",
    relevance: "Saved automatically at the end of each batch run."
  }
];


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
  antiCollapseWarning?: {
    triggered: boolean;
    message: string;
    recommendation?: string;
  } | null;
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
  onCloseApp?: () => void;
  trainingStatus?: TrainingStatus | null;
  checkpointIndex?: CheckpointIndex | null;
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
  antiCollapseWarning = null,
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
  onCloseApp = () => {},
  trainingStatus = null,
  checkpointIndex = null,
}: TrainingPanelProps) {
  const denominator = batchTotalEpisodes || episodes;
  const progress = denominator === 0 ? 0 : Math.min(1, batchCompletedEpisodes / denominator);
  const progressPct = Math.round(progress * 100);
  const completedDisplay = Math.floor(batchCompletedEpisodes);
  const activeEpisode = Math.min(denominator, Math.floor(batchCompletedEpisodes) + 1);
  const displayEpisodeIndex =
    currentEpisode !== null && startingEpisode != null && currentEpisode >= startingEpisode
      ? currentEpisode - startingEpisode
      : (currentEpisode ?? 0);
  const safeTotal = Number.isFinite(totalEpisodesTrained) ? totalEpisodesTrained : 0;
  const safeGrand = Number.isFinite(grandTotalEpisodes) ? grandTotalEpisodes : 0;
  const stageHistoryEntries = Object.entries(stageHistory)
    .filter(([, v]) => v > 0)
    .sort(([a], [b]) => Number(a) - Number(b));
  const busy = running || clearing || phase === "restoring" || phase === "restore_failed";
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

  const [activeSubTab, setActiveSubTab] = useState<"console" | "curriculum" | "metrics" | "help">("console");
  const [glossarySearch, setGlossarySearch] = useState("");
  const [highlightedTerm, setHighlightedTerm] = useState<string | null>(null);
  const [adminOpen, setAdminOpen] = useState(false);

  const navigateToHelp = (term: string) => {
    setActiveSubTab("help");
    setGlossarySearch(term);
    setHighlightedTerm(term);
    setTimeout(() => setHighlightedTerm(null), 2500);
  };

  return (
    <section className="training-card" aria-label="Training controls" style={{ display: "flex", flexDirection: "column", height: "100%", maxHeight: "100%", minHeight: 0, overflow: "hidden" }}>
      {/* 1. FIXED HEADER */}
      <div className="training-card__header" style={{ flexShrink: 0, marginBottom: "0.5rem" }}>
        <div>
          <p className="eyebrow">Curriculum learning</p>
          <h2 style={{ fontSize: "1.3rem", margin: "0.1rem 0" }}>Training</h2>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
          <CopyAgentDataButton
            trainingStatus={trainingStatus}
            checkpointIndex={checkpointIndex}
            curriculumStage={curriculumStage}
          />
          <span className={`pill ${running ? "pill--live" : "pill--muted"}`}>{phase}</span>
        </div>
      </div>

      {/* 2. FIXED SUB-TABS */}
      <div style={{ display: "flex", borderBottom: "1px solid var(--panel-border)", gap: "0.15rem", marginBottom: "0.6rem", flexShrink: 0 }}>
        {([
          { id: "console", label: "Console", desc: "Run Controls" },
          { id: "curriculum", label: "Curriculum", desc: "Stage Progression" },
          { id: "metrics", label: "Metrics", desc: "Performance Stats" },
          { id: "help", label: "Help & Terms", desc: "Glossary" }
        ] as const).map(tab => {
          const isActive = activeSubTab === tab.id;
          return (
            <button
              key={tab.id}
              type="button"
              onClick={() => setActiveSubTab(tab.id)}
              style={{
                background: "transparent",
                border: "none",
                borderBottom: isActive ? "3px solid var(--accent)" : "3px solid transparent",
                borderRadius: 0,
                padding: "0.35rem 0.15rem",
                color: isActive ? "var(--text)" : "var(--muted)",
                cursor: "pointer",
                transition: "all 150ms ease",
                textAlign: "center",
                transform: "none",
                flex: 1
              }}
            >
              <div style={{ fontWeight: "700", color: isActive ? "var(--accent)" : "var(--text)", fontSize: "0.76rem", whiteSpace: "nowrap" }}>{tab.label}</div>
              <div style={{ fontSize: "0.56rem", color: "var(--muted)", marginTop: "0.05rem", whiteSpace: "nowrap" }}>{tab.desc}</div>
            </button>
          );
        })}
      </div>

      {/* 3. SCROLLABLE INNER CONTENT WRAPPER */}
      <div style={{ flex: 1, overflowY: "auto", paddingRight: "0.2rem", minHeight: 0, display: "flex", flexDirection: "column", gap: "0.75rem" }}>
        
        {/* SUB-TAB: CONSOLE */}
        <div style={{
          height: activeSubTab === "console" ? "auto" : 0,
          overflow: "hidden",
          opacity: activeSubTab === "console" ? 1 : 0,
          pointerEvents: activeSubTab === "console" ? "auto" : "none",
          display: "flex",
          flexDirection: "column",
          gap: "0.75rem",
          flexShrink: 0
        }}>
          {/* Quick Curriculum Stage Summary */}
          <div style={{
            background: "rgba(10, 20, 35, 0.4)",
            border: "1px solid var(--panel-border)",
            borderRadius: "0.6rem",
            padding: "0.6rem 0.8rem",
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center"
          }}>
            <div>
              <span style={{ fontSize: "0.7rem", color: "var(--accent)", textTransform: "uppercase", fontWeight: "600", display: "block" }}>
                Active Curriculum Stage
              </span>
              <strong style={{ fontSize: "0.9rem", color: "var(--text)" }}>Stage {curriculumStage}</strong>
              <span style={{ fontSize: "0.72rem", color: "var(--muted)", marginLeft: "0.5rem" }}>({stageDesc})</span>
            </div>
            <button
              type="button"
              onClick={() => setActiveSubTab("curriculum")}
              title="Manage stages and curriculum details"
              style={{ padding: "0.3rem 0.6rem", fontSize: "0.7rem" }}
            >
              Curriculum &rarr;
            </button>
          </div>

          {antiCollapseWarning?.triggered && (
            <div style={{
              background: "rgba(239, 68, 68, 0.12)",
              border: "1px solid rgba(239, 68, 68, 0.25)",
              borderLeft: "4px solid #ef4444",
              padding: "0.6rem 0.8rem",
              borderRadius: "0 0.5rem 0.5rem 0",
              fontSize: "0.75rem",
              color: "#fca5a5",
              lineHeight: "1.4",
              display: "flex",
              flexDirection: "column",
              gap: "4px"
            }}>
              <strong style={{ display: "flex", alignItems: "center", gap: "6px", color: "#f87171" }}>
                ⚠️ POLICY COLLAPSE DETECTED
              </strong>
              <span>{antiCollapseWarning.message}</span>
              {antiCollapseWarning.recommendation && (
                <span style={{ fontStyle: "italic", fontSize: "0.7rem", color: "#fca5a5" }}>
                  Recommendation: {antiCollapseWarning.recommendation}
                </span>
              )}
            </div>
          )}

          {/* Training Control Card */}
          <div style={{ display: "flex", flexDirection: "column", gap: "0.65rem" }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <span style={{ fontSize: "0.82rem", fontWeight: "600", color: "var(--text)" }}>
                Batch Configuration
              </span>
              <button
                type="button"
                onClick={() => navigateToHelp("episodes")}
                style={{ background: "transparent", border: "none", color: "var(--accent)", cursor: "pointer", padding: "0.1rem 0.3rem", fontSize: "0.8rem" }}
                title="What is an episode/batch?"
              >
                ❓ Help
              </button>
            </div>

            <div style={{ display: "flex", flexDirection: "column", gap: "0.35rem" }}>
              <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
                <div style={{ flex: 1 }}>
                  <label htmlFor="episodes-input" style={{ fontSize: "0.72rem", color: "var(--muted)", display: "block", marginBottom: "0.2rem" }}>
                    Episodes this batch
                  </label>
                  <input
                    id="episodes-input"
                    type="number"
                    min={1}
                    max={1000}
                    value={episodes}
                    onChange={(event) => onEpisodesChange(Number(event.target.value) || 1)}
                    disabled={busy}
                    style={{ padding: "0.5rem", fontSize: "0.85rem" }}
                  />
                </div>
                <div style={{ display: "flex", alignSelf: "flex-end" }}>
                  {episodes !== recommendedEpisodes && !busy ? (
                    <button
                      type="button"
                      onClick={() => onEpisodesChange(recommendedEpisodes)}
                      style={{ padding: "0.5rem 0.6rem", fontSize: "0.72rem" }}
                      title={`Use suggested ${recommendedEpisodes} episodes for Stage ${curriculumStage}`}
                    >
                      Use suggested ({recommendedEpisodes})
                    </button>
                  ) : null}
                </div>
              </div>
            </div>

            <div style={{
              background: "rgba(59, 130, 246, 0.1)",
              border: "1px solid rgba(59, 130, 246, 0.2)",
              borderRadius: "0.5rem",
              padding: "0.6rem",
              fontSize: "0.75rem",
              color: "#93c5fd",
              lineHeight: "1.4",
              display: "flex",
              flexDirection: "column",
              gap: "4px",
            }}>
              <span style={{ fontWeight: "700" }}>💡 Tip: Restore / Fork Past Checkpoints</span>
              <span>
                To restore or fork training from a previous stage or episode, go to the <strong>W&B Model</strong> tab, select the target checkpoint, and choose <strong>Restore weights</strong> or <strong>Fork Run...</strong>.
              </span>
            </div>

            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", background: "rgba(148, 163, 184, 0.05)", padding: "0.5rem", borderRadius: "0.5rem" }}>
              <label style={{ display: "flex", alignItems: "center", gap: "0.45rem", cursor: "pointer", fontSize: "0.78rem" }}>
                <input
                  type="checkbox"
                  checked={enableInstincts}
                  onChange={(event) => onEnableInstinctsChange(event.target.checked)}
                  disabled={busy}
                />
                <span>Enable instinct rewards</span>
              </label>
              <button
                type="button"
                onClick={() => navigateToHelp("instincts")}
                style={{ background: "transparent", border: "none", color: "var(--accent)", cursor: "pointer", padding: 0, fontSize: "0.78rem" }}
                title="What are instincts?"
              >
                ❓
              </button>
            </div>
          </div>

          {curriculumStage === 0 ? (
            <div className="warning-box warning-box--error" role="status" style={{ margin: 0, padding: "0.5rem", fontSize: "0.72rem" }}>
              Stage 0 is the full problem — dogs rarely discover the pen from scratch.
              Promote to <strong>Stage 1</strong> to start simple.
            </div>
          ) : null}

          {/* Progress bar */}
          <div style={{ display: "flex", flexDirection: "column", gap: "0.3rem" }}>
            <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.75rem", color: "var(--muted)" }}>
              <span>Batch Progress</span>
              <span>{progressPct}%</span>
            </div>
            <div
              className="progress-shell"
              role="progressbar"
              aria-label="Current batch progress"
              aria-valuemin={0}
              aria-valuemax={100}
              aria-valuenow={progressPct}
              style={{ margin: 0, height: "10px", borderRadius: "999px" }}
            >
              <div className="progress-shell__bar" style={{ width: `${progress * 100}%` }} />
            </div>
            <div style={{ fontSize: "0.72rem", color: "var(--muted)", marginTop: "0.1rem", display: "flex", justifyContent: "space-between", padding: "0 0.2rem" }}>
              {running && currentEpisode !== null ? (
                <>
                  <span>Episode {activeEpisode} of {denominator || "—"}</span>
                  <span>{displayEpisodeIndex} env episodes herded</span>
                </>
              ) : (
                <div style={{ width: "100%", textAlign: "center" }}>
                  {completedDisplay}/{denominator || "—"} completed
                </div>
              )}
            </div>
          </div>

          {/* Core Controls Buttons */}
          <div style={{ display: "flex", flexDirection: "column", gap: "0.5rem", marginTop: "0.2rem" }}>
            <button
              type="button"
              className="button-row__primary"
              onClick={onStartTraining}
              disabled={busy}
              style={{
                width: "100%",
                padding: "0.8rem",
                fontWeight: "700",
                fontSize: "0.9rem",
                display: "flex",
                justifyContent: "center",
                alignItems: "center"
              }}
              title="Starts running training episodes on the server"
            >
              {running ? "Training..." : `Train ${episodes} more`}
            </button>

            {running ? (
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0.5rem" }}>
                <button
                  type="button"
                  onClick={onPauseTraining}
                  disabled={clearing}
                  style={{ padding: "0.6rem 0.5rem", fontSize: "0.75rem", whiteSpace: "nowrap" }}
                  title="Pause after the current checkpoint completes. The remaining batch episodes will be saved so you can resume."
                >
                  Pause after checkpoint
                </button>
                <button
                  type="button"
                  className="button-row__danger"
                  onClick={onStopTraining}
                  disabled={clearing}
                  style={{ padding: "0.6rem 0.5rem", fontSize: "0.75rem", whiteSpace: "nowrap" }}
                  title="Stop training after the current checkpoint completes and discard remaining batch episodes."
                >
                  Stop after checkpoint
                </button>
              </div>
            ) : canResume ? (
              <button
                type="button"
                onClick={onResumeTraining}
                disabled={clearing}
                style={{
                  width: "100%",
                  padding: "0.65rem",
                  background: "rgba(74, 222, 128, 0.15)",
                  borderColor: "var(--good)"
                }}
                title="Resume training where it was paused or interrupted."
              >
                Resume {resumeRemainingEpisodes} remaining
              </button>
            ) : null}

            <button
              type="button"
              onClick={onCloseApp}
              style={{
                width: "100%",
                padding: "0.65rem",
                background: "rgba(251, 113, 133, 0.1)",
                borderColor: "rgba(251, 113, 133, 0.3)",
                color: "var(--danger)",
                marginTop: "0.4rem"
              }}
              title="Gracefully pause training, save progress, and shut down the backend server."
            >
              🔌 Close Application
            </button>
          </div>

          {/* Quick Stats Grid */}
          <div style={{
            display: "grid",
            gridTemplateColumns: "1fr 1fr",
            gap: "0.5rem",
            background: "rgba(148, 163, 184, 0.04)",
            border: "1px solid var(--panel-border)",
            borderRadius: "0.6rem",
            padding: "0.7rem",
            fontSize: "0.75rem",
            marginTop: "0.4rem"
          }}>
            <div style={{ display: "flex", flexDirection: "column", gridColumn: "span 2" }}>
              <span style={{ color: "var(--muted)" }}>Status message</span>
              <span style={{ fontSize: "0.75rem", fontWeight: "600", color: "var(--text)", wordBreak: "break-word" }}>{message}</span>
            </div>
            <div style={{ display: "flex", flexDirection: "column" }}>
              <span style={{ color: "var(--muted)" }}>Success Rate</span>
              <strong style={{ color: successGood ? "var(--good)" : undefined }}>
                {successPct}
              </strong>
            </div>
            <div style={{ display: "flex", flexDirection: "column", marginTop: "0.4rem" }}>
              <span style={{ color: "var(--muted)" }}>Total Trained</span>
              <strong>{(safeGrand || safeTotal).toLocaleString()}</strong>
            </div>
            <div style={{ display: "flex", flexDirection: "column", marginTop: "0.4rem" }}>
              <span style={{ color: "var(--muted)" }}>Auto-promote</span>
              <strong>{autoPromote ? "Enabled" : "Disabled"}</strong>
            </div>
          </div>

          {/* Administrative Actions */}
          <div style={{
            marginTop: "0.5rem",
            border: "1px solid rgba(251, 113, 133, 0.15)",
            borderRadius: "0.5rem",
            padding: "0.6rem",
            display: "flex",
            flexDirection: "column",
            gap: "0.5rem",
            background: "rgba(10, 15, 25, 0.2)"
          }}>
            <span style={{ fontSize: "0.72rem", color: "var(--danger)", fontWeight: "600" }}>⚠️ Administrative Tools</span>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0.4rem" }}>
              <button
                type="button"
                className="button-row__danger"
                onClick={onClearTraining}
                disabled={busy}
                style={{ padding: "0.45rem 0.25rem", fontSize: "0.7rem" }}
                title="Deletes current checkpoints, evaluations, and archives permanently."
              >
                {clearing ? "Clearing..." : "Clear"}
              </button>
              <button
                type="button"
                onClick={onResetJourney}
                disabled={busy}
                style={{ padding: "0.45rem 0.25rem", fontSize: "0.7rem" }}
                title="Archives progress and starts a fresh journey from Stage 1."
              >
                Reset Journey
              </button>
            </div>
          </div>
        </div>

        {/* SUB-TAB: CURRICULUM */}
        <div style={{
          height: activeSubTab === "curriculum" ? "auto" : 0,
          overflow: "hidden",
          opacity: activeSubTab === "curriculum" ? 1 : 0,
          pointerEvents: activeSubTab === "curriculum" ? "auto" : "none",
          display: "flex",
          flexDirection: "column",
          gap: "0.75rem",
          flexShrink: 0
        }}>
          {/* Stage Chip */}
          <div className="stage-row" style={{ display: "flex", flexDirection: "column", gap: "0.4rem", margin: 0 }}>
            <div className="stage-chip" style={{ width: "100%" }}>
              <span className="stage-chip__label">Stage {curriculumStage}</span>
              <span className="stage-chip__desc">{stageDesc}</span>
            </div>
            {canPromote ? (
              <button
                type="button"
                className="button-row__promote"
                onClick={onPromote}
                disabled={!readyToPromote}
                style={{ width: "100%", padding: "0.6rem" }}
              >
                Promote → Stage {curriculumStage + 1}
              </button>
            ) : curriculumStage >= maxCurriculumStage ? (
              <span className="pill pill--live" style={{ alignSelf: "center" }}>Max stage reached</span>
            ) : null}
          </div>

          {/* Promotion lock/success warning status boxes */}
          {canPromote && !readyToPromote ? (
            <div className="warning-box" role="status" style={{ margin: 0, padding: "0.5rem 0.75rem", fontSize: "0.72rem" }}>
              Promotion locked until Stage {curriculumStage} reaches &ge; {Math.round(PROMOTE_THRESHOLD * 100)}% success.
            </div>
          ) : null}

          {readyToPromote ? (
            <div className="warning-box warning-box--success" role="status" style={{ margin: 0, padding: "0.5rem 0.75rem", fontSize: "0.72rem" }}>
              ✓ {Math.round(successRate! * 100)}% success — ready to promote to Stage {curriculumStage + 1}
            </div>
          ) : successRate !== null && successRate < PROMOTE_THRESHOLD && !running && curriculumStage < maxCurriculumStage ? (
            <div className="warning-box" role="status" style={{ margin: 0, padding: "0.5rem 0.75rem", fontSize: "0.72rem" }}>
              {Math.round(successRate * 100)}% success — target &ge; {Math.round(PROMOTE_THRESHOLD * 100)}% to promote.
            </div>
          ) : null}

          {/* Auto-promotion controls */}
          <div style={{ background: "rgba(148, 163, 184, 0.04)", border: "1px solid var(--panel-border)", borderRadius: "0.6rem", padding: "0.75rem", display: "flex", flexDirection: "column", gap: "0.5rem" }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <label style={{ display: "flex", alignItems: "center", gap: "0.45rem", cursor: "pointer", fontSize: "0.78rem" }}>
                <input
                  type="checkbox"
                  checked={autoPromote}
                  onChange={(event) => onAutoPromoteChange(event.target.checked)}
                  disabled={busy}
                />
                <strong>Auto-promote stages</strong>
              </label>
              <button
                type="button"
                onClick={() => navigateToHelp("auto-promote")}
                style={{ background: "transparent", border: "none", color: "var(--accent)", cursor: "pointer", padding: 0, fontSize: "0.78rem" }}
              >
                ❓
              </button>
            </div>

            {hasAutoPromoteGate ? (
              <div style={{ borderTop: "1px solid var(--panel-border)", paddingTop: "0.5rem", marginTop: "0.2rem" }}>
                <span style={{ fontSize: "0.72rem", color: "var(--muted)", fontWeight: "600", display: "block", marginBottom: "0.3rem" }}>
                  Auto-promotion Checklist
                </span>
                <div style={{ display: "flex", flexDirection: "column", gap: "0.25rem", fontSize: "0.7rem" }}>
                  <div style={{ display: "flex", justifyContent: "space-between" }}>
                    <span>Verdict</span>
                    <strong className={decisionToneClass} style={{ padding: "0.05rem 0.35rem", borderRadius: "3px" }}>
                      {autoPromoteGate.decision.toUpperCase()}
                    </strong>
                  </div>
                  <div style={{ display: "flex", justifyContent: "space-between" }}>
                    <span>Success Gate</span>
                    <strong className={gateToneClass(autoPromoteGate.success_rate_ok)}>
                      {autoPromoteGate.success_rate_ok ? "PASS" : "FAIL"}
                    </strong>
                  </div>
                  <div style={{ display: "flex", justifyContent: "space-between" }}>
                    <span>Seed progress</span>
                    <strong className={gateToneClass(autoPromoteGate.seed_gate_target_met)}>
                      {autoPromoteGate.seed_gate_hits}/{autoPromoteGate.min_seed_gate_hits}
                    </strong>
                  </div>
                  <div style={{ display: "flex", justifyContent: "space-between" }}>
                    <span>Streak</span>
                    <strong className={gateToneClass(autoPromoteGate.qualified_streak >= autoPromoteGate.min_qualified_streak)}>
                      {autoPromoteGate.qualified_streak}/{autoPromoteGate.min_qualified_streak}
                    </strong>
                  </div>
                </div>
              </div>
            ) : null}
          </div>

          {/* Per-Stage History charts (scrollable internally) */}
          {stageHistoryEntries.length > 0 ? (
            <div style={{ display: "flex", flexDirection: "column", gap: "0.4rem" }}>
              <span style={{ fontSize: "0.72rem", color: "var(--muted)", fontWeight: "600" }}>Stage History (Episodes Trained)</span>
              <div style={{
                maxHeight: "150px",
                overflowY: "auto",
                border: "1px solid var(--panel-border)",
                borderRadius: "0.5rem",
                padding: "0.5rem",
                background: "rgba(10, 15, 25, 0.2)"
              }}>
                <div className="stage-history" style={{ margin: 0, padding: 0 }}>
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
                  </div>
                </div>
              </div>
            </div>
          ) : null}

          {/* Manual Stage Override */}
          <div style={{
            background: "rgba(10, 15, 25, 0.3)",
            border: "1px solid var(--panel-border)",
            borderRadius: "0.5rem",
            padding: "0.6rem"
          }}>
            <label htmlFor="stage-manual-input" style={{ fontSize: "0.75rem", color: "var(--muted)", display: "flex", flexDirection: "column", gap: "0.25rem" }}>
              <span style={{ fontWeight: "600", color: "var(--text)" }}>Stage (manual)</span>
              <input
                id="stage-manual-input"
                type="number"
                min={0}
                max={maxCurriculumStage}
                value={curriculumStage}
                onChange={(event) => onCurriculumStageChange(Number(event.target.value) || 0)}
                disabled={busy}
                style={{ padding: "0.45rem", fontSize: "0.85rem" }}
              />
            </label>
            <div style={{ fontSize: "0.65rem", color: "var(--muted)", marginTop: "0.3rem" }}>
              ⚠️ Changing stage manually overrides curriculum flow.
            </div>
          </div>
        </div>

        {/* SUB-TAB: METRICS */}
        <div style={{
          height: activeSubTab === "metrics" ? "auto" : 0,
          overflow: "hidden",
          opacity: activeSubTab === "metrics" ? 1 : 0,
          pointerEvents: activeSubTab === "metrics" ? "auto" : "none",
          display: "flex",
          flexDirection: "column",
          gap: "0.75rem",
          flexShrink: 0
        }}>
          {/* Best performance cards */}
          {(currentBestEntry || previousBestEntry) ? (
            <div style={{ display: "flex", flexDirection: "column", gap: "0.5rem" }}>
              <span style={{ fontSize: "0.72rem", color: "var(--muted)", fontWeight: "600" }}>Record Checkpoints</span>
              <div className="best-perf" style={{ display: "grid", gridTemplateColumns: previousBestEntry ? "1fr 1fr" : "1fr", gap: "0.4rem", margin: 0, padding: 0 }}>
                {currentBestEntry ? (
                  <div className="best-perf__col best-perf__col--current" style={{ padding: "0.5rem" }}>
                    <span className="best-perf__label" style={{ fontSize: "0.6rem" }}>★ Best overall</span>
                    <span className="best-perf__ep" style={{ fontSize: "0.72rem", fontWeight: "700" }}>{bestEntryLabel(currentBestEntry)}</span>
                    <span className="best-perf__rate" style={{ fontSize: "1.1rem", color: currentBestEntry.success_rate >= 0.5 ? "var(--good)" : undefined }}>
                      {Math.round(currentBestEntry.success_rate * 100)}%
                    </span>
                    <span className="best-perf__reward" style={{ fontSize: "0.65rem" }}>
                      {currentBestEntry.average_reward.toFixed(1)} R · {currentBestEntry.average_completion_steps != null ? `${Math.round(currentBestEntry.average_completion_steps)} st` : ""}
                    </span>
                  </div>
                ) : null}

                {previousBestEntry ? (
                  <div className="best-perf__col" style={{ padding: "0.5rem" }}>
                    <span className="best-perf__label" style={{ fontSize: "0.6rem" }}>Previous best</span>
                    <span className="best-perf__ep" style={{ fontSize: "0.72rem", fontWeight: "700" }}>{bestEntryLabel(previousBestEntry)}</span>
                    <span className="best-perf__rate" style={{ fontSize: "1.1rem", color: previousBestEntry.success_rate >= 0.5 ? "var(--good)" : "var(--muted)" }}>
                      {Math.round(previousBestEntry.success_rate * 100)}%
                    </span>
                    <span className="best-perf__reward" style={{ fontSize: "0.65rem" }}>
                      {previousBestEntry.average_reward.toFixed(1)} R
                    </span>
                  </div>
                ) : null}
              </div>
            </div>
          ) : null}

          {/* Latest metrics grid */}
          {hasLatestMetrics ? (
            <div style={{ display: "flex", flexDirection: "column", gap: "0.4rem" }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                <span style={{ fontSize: "0.72rem", color: "var(--muted)", fontWeight: "600" }}>
                  Latest Checkpoint Stats {latestCheckpointEpisode != null ? `(ep ${latestCheckpointEpisode})` : ""}
                </span>
                <button
                  type="button"
                  onClick={() => navigateToHelp("success rate")}
                  style={{ background: "transparent", border: "none", color: "var(--accent)", cursor: "pointer", padding: 0, fontSize: "0.72rem" }}
                >
                  ❓
                </button>
              </div>

              <div style={{
                display: "grid",
                gridTemplateColumns: "1fr 1fr",
                gap: "0.4rem",
                background: "rgba(10, 20, 35, 0.3)",
                border: "1px solid var(--panel-border)",
                borderRadius: "0.6rem",
                padding: "0.6rem"
              }}>
                {[
                  { label: "Success Rate", value: fmtPct(latestSuccessRate), ok: latestSuccessRate != null && latestSuccessRate >= 0.5, keyword: "success rate" },
                  { label: "Avg penned", value: fmtNum(latestAvgSheepPenned), keyword: null },
                  { label: "Avg reward", value: fmtNum(latestAvgReward, 1), keyword: null },
                  { label: "Timeout rate", value: fmtPct(latestTimeoutRate), keyword: "timeout" },
                  { label: "No-progress stop", value: fmtPct(latestStoppedRate), keyword: "no-progress" },
                  { label: "Dist-to-pen", value: fmtNum(latestAvgDistanceToPen, 1), keyword: null },
                  { label: "Flock spread", value: fmtNum(latestAvgFlockSpread, 1), keyword: null },
                  { label: "Farthest-to-pen", value: fmtNum(latestAvgFarthestDistanceToPen, 1), keyword: null },
                  { label: "Farthest-to-flock", value: fmtNum(latestAvgFarthestDistanceToFlockCenter, 1), keyword: null },
                  { label: "No-progress steps", value: fmtNum(latestAvgNoProgressSteps, 1), keyword: null },
                ].map((item, idx) => (
                  <div
                    key={idx}
                    style={{
                      display: "flex",
                      justifyContent: "space-between",
                      alignItems: "center",
                      borderBottom: "1px solid rgba(148, 163, 184, 0.05)",
                      paddingBottom: "0.25rem",
                      fontSize: "0.72rem"
                    }}
                  >
                    <span style={{ color: "var(--muted)", display: "flex", alignItems: "center", gap: "0.2rem" }}>
                      {item.label}
                      {item.keyword && (
                        <button
                          type="button"
                          onClick={() => navigateToHelp(item.keyword!)}
                          style={{ background: "transparent", border: "none", color: "var(--accent)", cursor: "pointer", padding: 0, fontSize: "0.65rem", transform: "none" }}
                        >
                          ❓
                        </button>
                      )}
                    </span>
                    <strong style={{ color: item.ok ? "var(--good)" : "var(--text)" }}>{item.value}</strong>
                  </div>
                ))}
              </div>
            </div>
          ) : (
            <div style={{ textAlign: "center", padding: "2rem", color: "var(--muted)", fontSize: "0.78rem" }}>
              No metrics available yet. Start training to collect logs.
            </div>
          )}

          {/* Active Network Architecture Configuration banner */}
          {hasActiveConfig ? (
            <div style={{
              background: "rgba(244, 197, 66, 0.08)",
              border: "1px solid rgba(244, 197, 66, 0.2)",
              borderRadius: "0.5rem",
              padding: "0.5rem",
              fontSize: "0.7rem",
              display: "flex",
              flexDirection: "column",
              gap: "0.2rem"
            }}>
              <span style={{ fontWeight: "700", color: "var(--accent)" }}>Runtime Config:</span>
              <div style={{ display: "flex", flexWrap: "wrap", gap: "0.4rem", color: "var(--muted)" }}>
                {activeTrainerType ? <span>• {activeTrainerType}</span> : null}
                {activePolicyType ? <span>• {activePolicyType}</span> : null}
                {activeStageLabel ? <span>• {activeStageLabel}</span> : null}
                {activeInstinctsLabel ? (
                  <span style={{ color: activeInstincts ? "var(--good)" : undefined }}>
                    • {activeInstinctsLabel}
                  </span>
                ) : null}
              </div>
            </div>
          ) : null}
        </div>

        {/* SUB-TAB: HELP & TERMS */}
        <div style={{
          height: activeSubTab === "help" ? "auto" : 0,
          overflow: "hidden",
          opacity: activeSubTab === "help" ? 1 : 0,
          pointerEvents: activeSubTab === "help" ? "auto" : "none",
          display: "flex",
          flexDirection: "column",
          gap: "0.75rem",
          flexShrink: 0
        }}>
          {/* Glossary Header & Search */}
          <div style={{ display: "flex", flexDirection: "column", gap: "0.4rem" }}>
            <span style={{ fontSize: "0.72rem", color: "var(--muted)", fontWeight: "600" }}>ML Terms Glossary</span>
            <input
              type="text"
              placeholder="Search terms (e.g. instincts, PPO)..."
              value={glossarySearch}
              onChange={(e) => setGlossarySearch(e.target.value)}
              style={{
                padding: "0.45rem",
                fontSize: "0.8rem",
                borderRadius: "0.4rem",
                border: "1px solid var(--panel-border)",
                background: "rgba(8, 15, 25, 0.6)",
                color: "var(--text)"
              }}
            />
          </div>

          {/* Glossary list */}
          <div style={{
            display: "flex",
            flexDirection: "column",
            gap: "0.5rem",
            maxHeight: "300px",
            overflowY: "auto",
            paddingRight: "0.15rem"
          }}>
            {GLOSSARY_ITEMS.filter(item => {
              const q = glossarySearch.toLowerCase();
              return item.term.toLowerCase().includes(q) || item.definition.toLowerCase().includes(q) || item.category.toLowerCase().includes(q) || item.keyword.includes(q);
            }).map((item, idx) => {
              const isHighlighted = highlightedTerm === item.keyword;
              return (
                <div
                  key={idx}
                  className={isHighlighted ? "glossary-highlight" : ""}
                  style={{
                    background: isHighlighted ? "rgba(244, 197, 66, 0.12)" : "rgba(148, 163, 184, 0.05)",
                    border: isHighlighted ? "1px solid var(--accent)" : "1px solid var(--panel-border)",
                    borderRadius: "0.55rem",
                    padding: "0.6rem",
                    display: "flex",
                    flexDirection: "column",
                    gap: "0.25rem",
                    transition: "all 300ms ease"
                  }}
                >
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                    <strong style={{ fontSize: "0.82rem", color: "var(--accent)" }}>{item.term}</strong>
                    <span style={{
                      fontSize: "0.62rem",
                      background: "rgba(148, 163, 184, 0.12)",
                      padding: "0.1rem 0.35rem",
                      borderRadius: "4px",
                      color: "var(--muted)"
                    }}>
                      {item.category}
                    </span>
                  </div>
                  <p style={{ fontSize: "0.72rem", color: "var(--text)", margin: 0, lineHeight: "1.35" }}>
                    {item.definition}
                  </p>
                  <div style={{ fontSize: "0.65rem", color: "var(--muted)", fontStyle: "italic", borderTop: "1px solid rgba(148, 163, 184, 0.08)", paddingTop: "0.25rem", marginTop: "0.1rem" }}>
                    <strong>In Sheepdog:</strong> {item.relevance}
                  </div>
                </div>
              );
            })}
          </div>
        </div>

      </div>

      {/* 4. FIXED STATUS / NOTICES / ERROR DISPLAY AT BOTTOM */}
      <div style={{ flexShrink: 0, marginTop: "0.5rem" }}>
        {/* Advanced settings collapsible details box — now in Console tab but accessible globally or under Advanced details */}
        <details className="training-advanced" style={{ background: "rgba(148, 163, 184, 0.03)", padding: "0.4rem 0.5rem", borderRadius: "0.4rem" }}>
          <summary style={{ fontSize: "0.72rem" }}>Advanced run settings</summary>
          <div className="training-grid" style={{ gap: "0.4rem", marginTop: "0.3rem" }}>
            <label className="training-toggle" style={{ fontSize: "0.72rem" }}>
              <input
                type="checkbox"
                checked={fastMode}
                onChange={(event) => onFastModeChange(event.target.checked)}
                disabled={busy}
              />
              <span>Fast mode</span>
            </label>

            <label className="training-toggle" style={{ fontSize: "0.72rem" }}>
              <input
                type="checkbox"
                checked={debugRewardBreakdown}
                onChange={(event) => onDebugRewardBreakdownChange(event.target.checked)}
                disabled={busy}
              />
              <span>Debug rewards</span>
            </label>
          </div>
          <div className="warning-box" style={{ margin: "0.3rem 0 0", padding: "0.4rem", fontSize: "0.65rem" }}>
            Old weights trained without instinct rewards may not transfer cleanly. Clear training data before curriculum changes.
          </div>
        </details>

        {running ? (
          <div className="warning-box" role="status" aria-live="polite" style={{ margin: "0.4rem 0 0", padding: "0.4rem 0.6rem", fontSize: "0.7rem" }}>
            Training runs server-side — you can switch tabs and training continues.
          </div>
        ) : null}

        {error ? (
          <div className="warning-box warning-box--error" role="alert" style={{ margin: "0.4rem 0 0", padding: "0.5rem 0.75rem", fontSize: "0.7rem" }}>
            <strong>{errorType ? `${errorType}: ` : ""}</strong>
            {error}
            {traceback ? (
              <details className="training-advanced" style={{ marginTop: "0.4rem" }}>
                <summary>Technical traceback</summary>
                <pre style={{ whiteSpace: "pre-wrap", margin: "0.4rem 0 0", fontSize: "0.65rem" }}>{traceback}</pre>
              </details>
            ) : null}
          </div>
        ) : null}
      </div>
    </section>
  );
}

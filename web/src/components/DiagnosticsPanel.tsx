import { useEffect, useMemo, useState } from "react";
import type { CheckpointEntry, CheckpointIndex, TrainingStatus } from "../state/types";
import { CopyAgentDataButton } from "./CopyAgentDataButton";

/** Number of most-recent checkpoints to watch for a plateau. */
const PLATEAU_WINDOW = 5;
/** Minimum absolute improvement (success_rate) needed to not call it a plateau. */
const PLATEAU_MIN_DELTA = 0.02;
/** Below this success_rate the run is considered "cliff" (stuck at zero). */
const CLIFF_THRESHOLD = 0.05;
/** Minimum checkpoints in the plateau window before we flag a cliff. */
const CLIFF_MIN_CHECKPOINTS = 8;

const STAGE_COLORS: Record<number, string> = {
  0: "#9ca3af",
  1: "#60a5fa",
  2: "#34d399",
  3: "#f59e0b",
  4: "#f472b6",
  5: "#c084fc",
};

const PROMOTE_THRESHOLD = 0.5;

function getSuccessThreshold(stage: number): number {
  if (stage >= 2) {
    return 0.90;
  }
  return 0.80;
}
const RECENT_WINDOW = 5;

function stageColor(stage: number | undefined): string {
  return STAGE_COLORS[stage ?? 0] ?? "#9ca3af";
}

function formatPercent(value: number | null | undefined): string {
  return value != null ? `${Math.round(value * 100)}%` : "—";
}

function formatNumber(value: number | null | undefined, decimals = 1): string {
  return value != null ? value.toFixed(decimals) : "—";
}

function average(values: number[]): number | null {
  if (values.length === 0) return null;
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

type DecisionTone = "good" | "warn" | "danger" | "muted";

function getCheckpointStage(c: CheckpointEntry): number {
  if (c.reward_config?.instincts?.curriculum_stage !== undefined && c.reward_config?.instincts?.curriculum_stage !== null) {
    return c.reward_config.instincts.curriculum_stage;
  }
  if (c.environment_config?.curriculum_stage !== undefined && c.environment_config?.curriculum_stage !== null) {
    return c.environment_config.curriculum_stage;
  }
  return -1;
}

interface DecisionSignal {
  title: string;
  body: string;
  tone: DecisionTone;
  badge: string;
}

function buildDecisionSignal(params: {
  checkpointCount: number;
  latestSuccessRate: number | null;
  latestReward: number | null;
  latestTimeoutRate: number | null;
  plateauKind: "plateau-low" | "plateau-high" | "converged" | "cliff" | "spike" | null;
  stage: number;
  abovePromotionThreshold: boolean;
  improving: boolean;
}): DecisionSignal {
  const {
    checkpointCount,
    latestSuccessRate,
    latestReward,
    latestTimeoutRate,
    plateauKind,
    stage,
    abovePromotionThreshold,
    improving,
  } = params;

  if (checkpointCount === 0) {
    return {
      title: "Collect baseline data",
      body: "Run a few checkpoints before judging readiness. Professional workflows wait for a trend, not a single score.",
      tone: "muted",
      badge: "No history",
    };
  }

  if (plateauKind === "cliff" || (latestSuccessRate != null && latestSuccessRate < 0.05 && checkpointCount >= 8)) {
    return {
      title: "Investigate the training setup",
      body:
        "The model is still failing consistently. At this point engineers would inspect reward shaping, curriculum difficulty, and whether the run is actually learnable from scratch.",
      tone: "danger",
      badge: "Cliff",
    };
  }

  if (latestTimeoutRate != null && latestTimeoutRate >= 0.6) {
    return {
      title: "Too many timeouts",
      body:
        "Episodes are ending by timeout more often than success. That usually means the policy is finding partial motion but not a stable penning strategy yet.",
      tone: "warn",
      badge: "Failure mode",
    };
  }

  if (latestSuccessRate != null && abovePromotionThreshold && improving && plateauKind !== "spike") {
    return {
      title: "Promote to the next stage",
      body:
        "Success is at or above the promotion bar and the recent window is still moving in the right direction. This is the safest handoff point for curriculum learning.",
      tone: "good",
      badge: `Stage ${stage} ready`,
    };
  }

  if (plateauKind === "converged") {
    return {
      title: "Promote to the next stage",
      body:
        "The agent has converged at a high success rate. Training more on this stage yields diminishing returns; consider promoting to advance learning.",
      tone: "good",
      badge: `Stage ${stage} ready`,
    };
  }

  if (plateauKind === "plateau-high") {
    return {
      title: "Continue training or promote",
      body:
        "Performance has stabilized. You can promote to the next stage if this success rate is acceptable, or let it train a little longer to see if it makes further gains.",
      tone: "muted",
      badge: "Stable",
    };
  }

  if (plateauKind === "plateau-low") {
    return {
      title: "Struggling to learn",
      body:
        "The agent is stuck at a low success rate. Consider adjusting the reward function configuration, reducing entropy_coef, or clearing and restarting.",
      tone: "warn",
      badge: "Stuck",
    };
  }

  if (plateauKind === "spike") {
    return {
      title: "Model found something promising, then regressed",
      body:
        "This is the classic PPO oscillation pattern. Keep the best checkpoint, but expect stability checks before advancing.",
      tone: "warn",
      badge: "Volatile",
    };
  }

  return {
    title: "Continue training",
    body:
      latestReward != null
        ? "The model is still improving, but not strongly enough to call the stage complete. Let the batch run and compare the next checkpoint against this one."
        : "There is some signal, but not enough to make a confident promotion call yet.",
    tone: "muted",
    badge: latestSuccessRate != null ? `${Math.round(latestSuccessRate * 100)}% success` : "In progress",
  };
}

// ── Inline SVG line-chart ───────────────────────────────────────────────────

interface ChartPoint {
  x: number;
  y: number;
  stage?: number;
  isBest?: boolean;
  isPrevBest?: boolean;
  /** Custom text to show above a prev-best diamond instead of formatY(y). */
  labelText?: string;
  /** Optional secondary value (e.g. avg_completion_steps) for the right-axis overlay line. */
  secondaryY?: number | null;
}

interface LineChartProps {
  data: ChartPoint[];
  label?: string;
  lineColor: string;
  yMin: number;
  yMax: number;
  formatY: (v: number) => string;
  /** Horizontal reference line (e.g., 0.5 for 50% success threshold). */
  referenceY?: number;
  referenceLabel?: string;
  bestEpisode?: number | null;
  /** When true, draw the prev-best label above each diamond marker. */
  showPrevBestLabels?: boolean;
  /** Right-axis overlay line bounds. When provided, plots data[].secondaryY with fewer=top. */
  secondaryYMin?: number;
  secondaryYMax?: number;
  secondaryLineColor?: string;
  formatSecondaryY?: (v: number) => string;
}

function LineChart({
  data,
  label,
  lineColor,
  yMin,
  yMax,
  formatY,
  referenceY,
  referenceLabel,
  bestEpisode,
  showPrevBestLabels = false,
  secondaryYMin,
  secondaryYMax,
  secondaryLineColor,
  formatSecondaryY,
}: LineChartProps) {
  const W = 900;
  const H = 260;
  const topPad = showPrevBestLabels ? 28 : 18;
  const hasSecondary =
    secondaryYMin !== undefined &&
    secondaryYMax !== undefined &&
    data.some((d) => d.secondaryY != null);
  const effectiveSecColor = secondaryLineColor ?? "rgba(251,146,60,0.9)";
  const PAD = { top: topPad, right: hasSecondary ? 58 : 32, bottom: 36, left: 62 };
  const plotW = W - PAD.left - PAD.right;
  const plotH = H - PAD.top - PAD.bottom;

  const hasData = data.length >= 2;

  const xMin = hasData ? data[0].x : 0;
  const xMax = hasData ? data[data.length - 1].x : 1;
  const xRange = xMax - xMin || 1;
  const yRange = yMax - yMin || 1;

  // Secondary scale — fewer steps = better = top of chart (inverted mapping)
  const secYMin = secondaryYMin ?? 0;
  const secYMax = secondaryYMax ?? 1;
  const secRange = secYMax - secYMin || 1;
  const secDataPoints = hasSecondary ? data.filter((d) => d.secondaryY != null) : [];
  const secTicks = hasSecondary ? [secYMax, (secYMax + secYMin) / 2, secYMin] : [];

  function toSvgX(x: number): number {
    return PAD.left + ((x - xMin) / xRange) * plotW;
  }
  function toSvgY(y: number): number {
    return PAD.top + plotH - ((y - yMin) / yRange) * plotH;
  }
  function toSvgY2(y: number): number {
    // Fewer steps → smaller y value → top of chart (PAD.top)
    return PAD.top + ((y - secYMin) / secRange) * plotH;
  }

  const polyline = hasData
    ? data.map((d) => `${toSvgX(d.x).toFixed(1)},${toSvgY(d.y).toFixed(1)}`).join(" ")
    : "";

  // Y-axis tick values
  const yTicks = [yMin, yMin + yRange * 0.5, yMax];

  // X-axis labels — pick up to 4 evenly spaced
  const xLabelIndices: number[] = [];
  if (hasData) {
    const step = Math.max(1, Math.floor((data.length - 1) / 3));
    for (let i = 0; i < data.length; i += step) {
      xLabelIndices.push(i);
    }
    if (xLabelIndices[xLabelIndices.length - 1] !== data.length - 1) {
      xLabelIndices.push(data.length - 1);
    }
  }

  return (
    <div className="mini-chart">
      {label ? <span className="mini-chart__label">{label}</span> : null}
      <svg
        viewBox={`0 0 ${W} ${H}`}
        className="mini-chart__svg"
        aria-label={label}
        preserveAspectRatio="xMidYMid meet"
        overflow="visible"
      >
        {/* Grid lines */}
        {yTicks.map((v, i) => {
          const sy = toSvgY(v);
          return (
            <g key={i}>
              <line
                x1={PAD.left}
                y1={sy}
                x2={PAD.left + plotW}
                y2={sy}
                stroke="rgba(148,163,184,0.13)"
                strokeWidth={1}
              />
              <text
                x={PAD.left - 4}
                y={sy + 3.5}
                textAnchor="end"
                fontSize={11}
                fill="rgba(148,163,184,0.65)"
              >
                {formatY(v)}
              </text>
            </g>
          );
        })}

        {/* X-axis labels */}
        {xLabelIndices.map((idx) => {
          const d = data[idx];
          return (
            <text
              key={idx}
              x={toSvgX(d.x)}
              y={H - 3}
              textAnchor="middle"
              fontSize={10}
              fill="rgba(148,163,184,0.6)"
            >
              {d.x}
            </text>
          );
        })}

        {/* Reference line */}
        {referenceY !== undefined ? (
          <g>
            <line
              x1={PAD.left}
              y1={toSvgY(referenceY)}
              x2={PAD.left + plotW}
              y2={toSvgY(referenceY)}
              stroke="rgba(74,222,128,0.35)"
              strokeWidth={1}
              strokeDasharray="4 3"
            />
            {referenceLabel ? (
              <text
                x={PAD.left + plotW + 1}
                y={toSvgY(referenceY) + 3.5}
                fontSize={10}
                fill="rgba(74,222,128,0.7)"
              >
                {referenceLabel}
              </text>
            ) : null}
          </g>
        ) : null}

        {/* Line */}
        {hasData ? (
          <polyline
            points={polyline}
            fill="none"
            stroke={lineColor}
            strokeWidth={2.5}
            strokeLinejoin="round"
            strokeLinecap="round"
          />
        ) : null}

        {/* Secondary line (steps) — dashed, right axis, fewer=top */}
        {hasSecondary ? (
          <>
            {secTicks.map((v, i) => (
              <text
                key={i}
                x={PAD.left + plotW + 6}
                y={toSvgY2(v) + 3.5}
                textAnchor="start"
                fontSize={11}
                fill={effectiveSecColor}
                opacity={0.65}
              >
                {formatSecondaryY ? formatSecondaryY(v) : String(Math.round(v))}
              </text>
            ))}
            {secDataPoints.length >= 2 ? (
              <polyline
                points={secDataPoints
                  .map((d) => `${toSvgX(d.x).toFixed(1)},${toSvgY2(d.secondaryY!).toFixed(1)}`)
                  .join(" ")}
                fill="none"
                stroke={effectiveSecColor}
                strokeWidth={2}
                strokeDasharray="5 3"
                strokeLinejoin="round"
                strokeLinecap="round"
                opacity={0.75}
              />
            ) : null}
            {secDataPoints.map((d, i) => (
              <circle
                key={`sec-${d.x}-${i}`}
                cx={toSvgX(d.x)}
                cy={toSvgY2(d.secondaryY!)}
                r={3}
                fill={effectiveSecColor}
                stroke="rgba(8,17,27,0.7)"
                strokeWidth={1}
                opacity={0.8}
              />
            ))}
          </>
        ) : null}

        {/* Dots — colored by stage; prev-bests get a diamond + label; best gets a ring */}
        {data.map((d, i) => {
          const cx = toSvgX(d.x);
          const cy = toSvgY(d.y);
          const fill = stageColor(d.stage);
          const isBest = d.x === bestEpisode;
          const r = isBest ? 6 : d.isPrevBest ? 5 : 4;
          const diamond = `${cx},${cy - r} ${cx + r},${cy} ${cx},${cy + r} ${cx - r},${cy}`;
          return (
            <g key={`${d.x}-${i}`}>
              {isBest ? (
                <circle cx={cx} cy={cy} r={9} fill="none" stroke={fill} strokeWidth={2} opacity={0.6} />
              ) : null}
              {d.isPrevBest ? (
                <polygon points={diamond} fill={fill} stroke="rgba(8,17,27,0.7)" strokeWidth={1} />
              ) : (
                <circle cx={cx} cy={cy} r={r} fill={fill} stroke="rgba(8,17,27,0.7)" strokeWidth={1} />
              )}
              {(d.isPrevBest || isBest) && showPrevBestLabels && d.labelText ? (
                <text
                  x={cx + 4}
                  y={cy - r - 4}
                  transform={`rotate(-90, ${cx + 4}, ${cy - r - 4})`}
                  textAnchor="start"
                  fontSize={10}
                  fontWeight="600"
                  fill={fill}
                  style={{ pointerEvents: "none" }}
                >
                  {d.labelText}
                </text>
              ) : null}
            </g>
          );
        })}

        {/* No-data message */}
        {!hasData ? (
          <text x={W / 2} y={H / 2} textAnchor="middle" fontSize={16} fill="rgba(148,163,184,0.5)">
            Not enough data
          </text>
        ) : null}
      </svg>
    </div>
  );
}

// ── Chart sub-tab types & legend ───────────────────────────────────────────

type ChartTab = "health" | "success" | "reward" | "sheep" | "history" | "learningSignal";
type ViewWindow = "all" | 25 | 50 | 100;
type StageScope = "all" | "current" | "current-journey" | number;

const VIEW_WINDOW_OPTIONS: Array<{ value: ViewWindow; label: string }> = [
  { value: 25, label: "Last 25" },
  { value: 50, label: "Last 50" },
  { value: 100, label: "Last 100" },
  { value: "all", label: "All" },
];

const LEARNING_SIGNAL_WINDOW_OPTIONS: Array<{ value: ViewWindow; label: string }> = [
  { value: 25, label: "Last 25" },
  { value: 50, label: "Last 50" },
  { value: "all", label: "All" },
];

const SMOOTHING_WINDOWS = [25, 50, 100] as const;
type SmoothingWindow = (typeof SMOOTHING_WINDOWS)[number];

interface SignalSummary {
  label: string;
  tone: DecisionTone;
}

interface AdvisorSignal {
  state: "A" | "B" | "C" | "D" | "E";
  title: string;
  body: string;
  tone: DecisionTone;
  actions: string[];
  reason: string;
}

function formatSignalDelta(current: number | null, previous: number | null, percent = false): string {
  if (current == null || previous == null) return "No baseline yet";
  const delta = current - previous;
  if (Math.abs(delta) < 0.0001) return "No meaningful change";
  const prefix = delta > 0 ? "+" : "-";
  if (percent) {
    return `${prefix}${Math.round(Math.abs(delta) * 100)}%`;
  }
  return `${prefix}${Math.abs(delta).toFixed(2)}`;
}

function buildAdvisor(params: {
  currentFlatStreak: number;
  longestHistoricalPlateau: number;
  rewardSignal: SignalSummary;
  timeoutSignal: SignalSummary;
  speedSignal: SignalSummary;
  recentSuccess: number | null;
  recentRewardDelta: number | null;
  recentSpeedDelta: number | null;
  recentTimeout: number | null;
  recentNoProgress: number | null;
  highSuccessStable: boolean;
}): AdvisorSignal {
  const {
    currentFlatStreak,
    longestHistoricalPlateau,
    rewardSignal,
    timeoutSignal,
    speedSignal,
    recentSuccess,
    recentRewardDelta,
    recentSpeedDelta,
    recentTimeout,
    recentNoProgress,
    highSuccessStable,
  } = params;

  const degradingSignals = [rewardSignal.label === "Declining", timeoutSignal.label === "Getting Worse", speedSignal.label === "Slowing"].filter(Boolean).length;

  if (highSuccessStable) {
    return {
      state: "E",
      title: "Ready to Advance — Consider Next Phase",
      body: "Agent has held high success for a sustained window. Plateau at high performance suggests diminishing returns on this stage.",
      tone: "good",
      actions: ["Advance to the next curriculum stage", "Keep this checkpoint as your stage-complete baseline"],
      reason: "Success has been above 80% for a sustained period and has not regressed, which indicates stage mastery.",
    };
  }

  if (degradingSignals >= 2) {
    return {
      state: "D",
      title: "Investigate Training Setup",
      body: "Multiple signals are degrading together, which points to a setup issue rather than a normal plateau.",
      tone: "danger",
      actions: [
        "Review reward shaping balance (penalties vs progress reward)",
        "Re-check curriculum stage difficulty",
        "Run a short scratch restart to test reproducibility",
      ],
      reason: "When reward, timeout, and speed degrade together, training usually needs setup intervention instead of more episodes.",
    };
  }

  const hasMicroProgress =
    (recentRewardDelta != null && recentRewardDelta > 0) ||
    (recentSpeedDelta != null && recentSpeedDelta < 0);
  if (hasMicroProgress && (recentSuccess ?? 0) < 0.8) {
    return {
      state: "B",
      title: "Keep Training — Micro-Progress Detected",
      body: `Success is flat but reward/speed signals are improving (${formatSignalDelta(recentRewardDelta, 0)} reward trend, ${formatSignalDelta(recentSpeedDelta, 0)} steps trend).`,
      tone: "good",
      actions: ["Continue current run", "Re-evaluate after the next 50 checkpoints"],
      reason: "Reward and efficiency often improve before success rate jumps in sparse-reward RL tasks.",
    };
  }

  if (currentFlatStreak > longestHistoricalPlateau && longestHistoricalPlateau > 0) {
    const suggestions: string[] = [];
    if ((recentTimeout ?? 0) > 0.6) {
      suggestions.push("Reduce episode length or simplify curriculum stage");
    }
    if (rewardSignal.label === "Volatile") {
      suggestions.push("Reduce entropy_coef for more stable convergence");
    }
    if (rewardSignal.label === "Declining") {
      suggestions.push("Review reward shaping — penalties may be dominating");
    }
    if ((recentNoProgress ?? 0) > 0) {
      suggestions.push("Review action space/reward conversion — policy is moving but not converting");
    }
    return {
      state: "C",
      title: "Consider Parameter Adjustment",
      body: `Flat streak (${currentFlatStreak} episodes) exceeds your longest known pre-breakthrough plateau (${longestHistoricalPlateau}).`,
      tone: "warn",
      actions: suggestions.length > 0 ? suggestions : ["Adjust exploration/stability hyperparameters", "Try a smaller curriculum step"],
      reason: "Past runs suggest this plateau is longer than your typical pre-breakthrough window.",
    };
  }

  if (currentFlatStreak <= longestHistoricalPlateau || longestHistoricalPlateau === 0) {
    return {
      state: "A",
      title: "Keep Training — Within Breakthrough Window",
      body: `You've been flat for ${currentFlatStreak} episodes. Previous breakthroughs took up to ${longestHistoricalPlateau || currentFlatStreak} episodes of plateau.`,
      tone: "muted",
      actions: ["Continue training", "Watch reward and speed for micro-progress"],
      reason: "Plateaus are normal in PPO; prior runs indicate breakthroughs can arrive after long flat stretches.",
    };
  }

  return {
    state: "C",
    title: "Consider Parameter Adjustment",
    body: "Signals are not improving enough to justify continuing unchanged.",
    tone: "warn",
    actions: ["Adjust entropy/stability settings", "Review curriculum stage"] ,
    reason: "No strong improvement signal detected.",
  };
}

function InfoTip({ text }: { text: string }) {
  return (
    <span className="info-tip" title={text} aria-label={text}>
      i
    </span>
  );
}

type SymbolSpec =
  | { kind: "line"; color: string }
  | { kind: "dash"; color: string }
  | { kind: "dot"; color: string }
  | { kind: "ring"; color: string }
  | { kind: "diamond"; color: string };

interface LegendEntry {
  symbol: SymbolSpec;
  label: string;
  detail: string;
}

interface StatCardProps {
  label: string;
  value: string;
  detail: string;
  tone?: DecisionTone;
}

function StatCard({ label, value, detail, tone = "muted" }: StatCardProps) {
  return (
    <article className={`stat-card stat-card--${tone}`}>
      <span className="stat-card__label">{label}</span>
      <strong className="stat-card__value">{value}</strong>
      <span className="stat-card__detail">{detail}</span>
    </article>
  );
}

function SymIcon({ sym }: { sym: SymbolSpec }) {
  const SW = 32;
  const SH = 16;
  const my = SH / 2;
  const cx = SW / 2;
  if (sym.kind === "line") {
    return (
      <svg width={SW} height={SH} viewBox={`0 0 ${SW} ${SH}`} style={{ flexShrink: 0 }}>
        <line x1={3} y1={my} x2={SW - 3} y2={my} stroke={sym.color} strokeWidth={2.5} strokeLinecap="round" />
      </svg>
    );
  }
  if (sym.kind === "dash") {
    return (
      <svg width={SW} height={SH} viewBox={`0 0 ${SW} ${SH}`} style={{ flexShrink: 0 }}>
        <line x1={3} y1={my} x2={SW - 3} y2={my} stroke={sym.color} strokeWidth={2} strokeDasharray="5 3" strokeLinecap="round" />
      </svg>
    );
  }
  if (sym.kind === "ring") {
    return (
      <svg width={SW} height={SH} viewBox={`0 0 ${SW} ${SH}`} style={{ flexShrink: 0 }}>
        <circle cx={cx} cy={my} r={6} fill="none" stroke={sym.color} strokeWidth={1.8} opacity={0.75} />
        <circle cx={cx} cy={my} r={3.5} fill={sym.color} />
      </svg>
    );
  }
  if (sym.kind === "diamond") {
    const r = 5;
    return (
      <svg width={SW} height={SH} viewBox={`0 0 ${SW} ${SH}`} style={{ flexShrink: 0 }}>
        <polygon
          points={`${cx},${my - r} ${cx + r},${my} ${cx},${my + r} ${cx - r},${my}`}
          fill={sym.color}
        />
      </svg>
    );
  }
  return (
    <svg width={SW} height={SH} viewBox={`0 0 ${SW} ${SH}`} style={{ flexShrink: 0 }}>
      <circle cx={cx} cy={my} r={5} fill={sym.color} />
    </svg>
  );
}

function ChartLegend({ entries }: { entries: LegendEntry[] }) {
  if (!entries.length) return null;
  return (
    <div className="chart-legend">
      {entries.map((e, i) => (
        <div key={i} className="chart-legend__item">
          <SymIcon sym={e.symbol} />
          <div className="chart-legend__text">
            <span className="chart-legend__label">{e.label}</span>
            <span className="chart-legend__detail">{e.detail}</span>
          </div>
        </div>
      ))}
    </div>
  );
}

function trendSummary(current: number | null, previous: number | null, format: (value: number) => string): string {
  if (current == null) return "No data";
  if (previous == null) return format(current);
  const delta = current - previous;
  if (Math.abs(delta) < 0.0001) {
    return `${format(current)} · flat`;
  }
  const sign = delta > 0 ? "+" : "-";
  return `${format(current)} · ${sign}${format(Math.abs(delta))}`;
}

function stageLabel(stage: number): string {
  if (stage === -1) return "Legacy/Unknown";
  return stage === 0 ? "Base difficulty" : `Stage ${stage}`;
}

interface LearningPoint {
  checkpoint: number;
  successRate: number;
  reward: number;
  timeoutRate: number;
  completionSteps: number;
  sheepPenned: number;
  noProgressGuard: number;
  stage: number;
  checkpoint_id?: string;
}

interface FlatZone {
  startCheckpoint: number;
  endCheckpoint: number;
  length: number;
}

interface BreakthroughEvent {
  index: number;
  checkpoint: number;
  flatEpisodesBefore: number;
  fromSuccessRate: number;
  toSuccessRate: number;
  checkpoint_id?: string;
}

interface LearningSignalAnalysis {
  flatZones: FlatZone[];
  breakthroughs: BreakthroughEvent[];
  currentFlatStreak: number;
  stageBestSuccessRate: number;
  smoothedSuccessRate: number[];
  longestHistoricalPlateau: number;
}

function rollingAverage(values: number[], windowSize: number): number[] {
  if (values.length === 0) return [];
  const safeWindow = Math.max(1, windowSize);
  const output: number[] = [];
  for (let i = 0; i < values.length; i += 1) {
    const start = Math.max(0, i - safeWindow + 1);
    const sample = values.slice(start, i + 1);
    output.push(sample.reduce((sum, value) => sum + value, 0) / sample.length);
  }
  return output;
}

function averageNoProgress(entry: CheckpointEntry): number {
  if (!entry.records?.length) return 0;
  const values = entry.records.map((record) => record.no_progress_steps ?? 0);
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function analyzeLearningSignal(points: LearningPoint[], smoothingWindow: number): LearningSignalAnalysis {
  if (points.length === 0) {
    return {
      flatZones: [],
      breakthroughs: [],
      currentFlatStreak: 0,
      stageBestSuccessRate: 0,
      smoothedSuccessRate: [],
      longestHistoricalPlateau: 0,
    };
  }

  let personalBest = points[0].successRate;
  let meaningfulAnchor = points[0].successRate;
  let lastMeaningfulIndex = 0;
  const flatZones: FlatZone[] = [];
  const breakthroughs: BreakthroughEvent[] = [];

  for (let i = 1; i < points.length; i += 1) {
    const point = points[i];
    const flatBefore = i - lastMeaningfulIndex - 1;
    const isBreakthrough = point.successRate > personalBest + 0.10 && flatBefore > 20;
    if (isBreakthrough) {
      breakthroughs.push({
        index: breakthroughs.length + 1,
        checkpoint: point.checkpoint,
        flatEpisodesBefore: flatBefore,
        fromSuccessRate: personalBest,
        toSuccessRate: point.successRate,
        checkpoint_id: point.checkpoint_id,
      });
    }

    const isMeaningful = point.successRate > meaningfulAnchor + 0.05;
    if (isMeaningful) {
      if (flatBefore > 20) {
        flatZones.push({
          startCheckpoint: points[lastMeaningfulIndex + 1]?.checkpoint ?? point.checkpoint,
          endCheckpoint: points[i - 1]?.checkpoint ?? point.checkpoint,
          length: flatBefore,
        });
      }
      meaningfulAnchor = point.successRate;
      lastMeaningfulIndex = i;
    }

    personalBest = Math.max(personalBest, point.successRate);
  }

  const currentFlatStreak = points.length - 1 - lastMeaningfulIndex;
  if (currentFlatStreak > 20 && points.length > 1) {
    flatZones.push({
      startCheckpoint: points[lastMeaningfulIndex + 1]?.checkpoint ?? points[points.length - 1].checkpoint,
      endCheckpoint: points[points.length - 1].checkpoint,
      length: currentFlatStreak,
    });
  }

  const longestHistoricalPlateau = breakthroughs.length
    ? Math.max(...breakthroughs.map((event) => event.flatEpisodesBefore))
    : 0;

  return {
    flatZones,
    breakthroughs,
    currentFlatStreak,
    stageBestSuccessRate: Math.max(...points.map((point) => point.successRate)),
    smoothedSuccessRate: rollingAverage(
      points.map((point) => point.successRate),
      smoothingWindow,
    ),
    longestHistoricalPlateau,
  };
}

function slope(values: number[]): number {
  if (values.length < 2) return 0;
  const first = values[0];
  const last = values[values.length - 1];
  return (last - first) / (values.length - 1);
}

function standardDeviation(values: number[]): number {
  if (values.length < 2) return 0;
  const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
  const variance = values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / values.length;
  return Math.sqrt(variance);
}

type SparkTrendLabel = "Climbing" | "Flat" | "Declining" | "Volatile" | "Improving" | "Stuck" | "Getting Worse" | "Getting Faster" | "Slowing";

function rewardTrendLabel(values: number[]): SparkTrendLabel {
  const s = slope(values);
  const volatility = standardDeviation(values);
  if (volatility > Math.max(2.5, Math.abs((values[values.length - 1] ?? 0) * 0.25))) return "Volatile";
  if (s > 0.02) return "Climbing";
  if (s < -0.02) return "Declining";
  return "Flat";
}

function timeoutTrendLabel(values: number[]): SparkTrendLabel {
  const s = slope(values);
  if (s < -0.002) return "Improving";
  if (s > 0.002) return "Getting Worse";
  return "Stuck";
}

function speedTrendLabel(values: number[]): SparkTrendLabel {
  const s = slope(values);
  if (s < -0.4) return "Getting Faster";
  if (s > 0.4) return "Slowing";
  return "Flat";
}

function trendArrow(label: SparkTrendLabel): string {
  if (label === "Climbing" || label === "Improving" || label === "Getting Faster") return "↗";
  if (label === "Declining" || label === "Getting Worse" || label === "Slowing") return "↘";
  if (label === "Volatile") return "≈";
  return "→";
}

interface SparklineProps {
  values: number[];
  color: string;
  lowerIsBetter?: boolean;
}

function Sparkline({ values, color }: SparklineProps) {
  const W = 260;
  const H = 64;
  if (values.length < 2) {
    return <div className="signal-sparkline signal-sparkline--empty">Not enough data</div>;
  }
  const minV = Math.min(...values);
  const maxV = Math.max(...values);
  const range = maxV - minV || 1;
  const points = values
    .map((value, index) => {
      const x = (index / (values.length - 1)) * (W - 8) + 4;
      const y = H - 6 - ((value - minV) / range) * (H - 14);
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="signal-sparkline" aria-hidden="true">
      <polyline points={points} fill="none" stroke={color} strokeWidth={2.4} strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

interface LearningSignalChartProps {
  points: LearningPoint[];
  smoothedSuccessRate: number[];
  flatZones: FlatZone[];
  breakthroughs: BreakthroughEvent[];
  currentCheckpoint: number | null;
  stageBestSuccessRate: number;
  onBreakthroughClick: (checkpoint: number) => void;
  focusedCheckpoint: number | null;
}

function LearningSignalChart({
  points,
  smoothedSuccessRate,
  flatZones,
  breakthroughs,
  currentCheckpoint,
  stageBestSuccessRate,
  onBreakthroughClick,
  focusedCheckpoint,
}: LearningSignalChartProps) {
  const W = 1100;
  const H = 340;
  const PAD = { top: 22, right: 34, bottom: 38, left: 56 };
  const plotW = W - PAD.left - PAD.right;
  const plotH = H - PAD.top - PAD.bottom;
  const xMin = points[0]?.checkpoint ?? 0;
  const xMax = points[points.length - 1]?.checkpoint ?? 1;
  const xRange = xMax - xMin || 1;

  function toX(checkpoint: number): number {
    return PAD.left + ((checkpoint - xMin) / xRange) * plotW;
  }

  function toY(value: number): number {
    return PAD.top + plotH - value * plotH;
  }

  const baseLine = points.map((point) => `${toX(point.checkpoint).toFixed(1)},${toY(point.successRate).toFixed(1)}`).join(" ");
  const smoothLine = points
    .map((point, index) => `${toX(point.checkpoint).toFixed(1)},${toY(smoothedSuccessRate[index] ?? point.successRate).toFixed(1)}`)
    .join(" ");

  const yTicks = [0, 0.25, 0.5, 0.75, 1];

  return (
    <div className="signal-main-chart">
      <svg viewBox={`0 0 ${W} ${H}`} className="signal-main-chart__svg" role="img" aria-label="Learning signal success history chart">
        {yTicks.map((tick) => (
          <g key={tick}>
            <line x1={PAD.left} y1={toY(tick)} x2={PAD.left + plotW} y2={toY(tick)} stroke="rgba(148,163,184,0.14)" strokeWidth={1} />
            <text x={PAD.left - 6} y={toY(tick) + 4} textAnchor="end" fontSize={11} fill="rgba(148,163,184,0.75)">{Math.round(tick * 100)}%</text>
          </g>
        ))}

        {flatZones.map((zone, idx) => {
          const x1 = toX(zone.startCheckpoint);
          const x2 = toX(zone.endCheckpoint);
          return (
            <rect
              key={`${zone.startCheckpoint}-${zone.endCheckpoint}-${idx}`}
              x={Math.min(x1, x2)}
              y={PAD.top}
              width={Math.max(2, Math.abs(x2 - x1))}
              height={plotH}
              fill="rgba(96,165,250,0.07)"
            />
          );
        })}

        <line
          x1={PAD.left}
          y1={toY(stageBestSuccessRate)}
          x2={PAD.left + plotW}
          y2={toY(stageBestSuccessRate)}
          stroke="rgba(74,222,128,0.42)"
          strokeWidth={1.2}
          strokeDasharray="5 4"
        />

        {currentCheckpoint != null ? (
          <line
            x1={toX(currentCheckpoint)}
            y1={PAD.top}
            x2={toX(currentCheckpoint)}
            y2={PAD.top + plotH}
            stroke="rgba(244,197,66,0.8)"
            strokeWidth={1.4}
            strokeDasharray="4 3"
          />
        ) : null}

        <polyline points={baseLine} fill="none" stroke="rgba(96,165,250,0.9)" strokeWidth={2.2} strokeLinejoin="round" strokeLinecap="round" />
        <polyline points={smoothLine} fill="none" stroke="rgba(244,197,66,0.95)" strokeWidth={2.4} strokeLinejoin="round" strokeLinecap="round" />

        {breakthroughs.map((event, idx) => {
          const x = toX(event.checkpoint);
          const y = toY(event.toSuccessRate);
          const focused = focusedCheckpoint === event.checkpoint;
          const breakthroughKey = event.checkpoint_id 
            ? event.checkpoint_id 
            : `${event.checkpoint}-${idx}`;
          return (
            <g key={breakthroughKey}>
              {focused ? <circle cx={x} cy={y} r={10} fill="none" stroke="rgba(244,197,66,0.6)" strokeWidth={2} /> : null}
              <circle
                cx={x}
                cy={y}
                r={6}
                fill="rgba(244,197,66,0.95)"
                stroke="rgba(8,17,27,0.85)"
                strokeWidth={1.4}
                style={{ cursor: "pointer" }}
                onClick={() => onBreakthroughClick(event.checkpoint)}
              />
              <title>{`Breakthrough #${event.index} at checkpoint ${event.checkpoint}`}</title>
            </g>
          );
        })}

        <text x={PAD.left + 4} y={PAD.top + 14} fontSize={11} fill="rgba(148,163,184,0.75)">success rate</text>
        <text x={PAD.left + plotW - 4} y={PAD.top + plotH + 28} textAnchor="end" fontSize={11} fill="rgba(148,163,184,0.75)">checkpoint</text>
      </svg>
    </div>
  );
}

// ── Plateau / cliff analysis ─────────────────────────────────────────────────

interface PlateauInfo {
  /** converged/plateau-high/low = stable; cliff = never succeeded; spike = regressed */
  kind: "converged" | "plateau-high" | "plateau-low" | "cliff" | "spike";
  window: number;
  bestRate: number;
  /** Highest success rate ever seen across all checkpoints. */
  allTimeBest: number;
  sinceEpisode: number;
}

/** Minimum checkpoints in the current stage before we can flag a plateau/cliff. */
const DIAGNOSTIC_MIN_CHECKPOINTS = 8;
/** Minimum cumulative episodes trained in the current stage before we can flag a plateau/cliff. */
const DIAGNOSTIC_MIN_EPISODES = 150;

function detectPlateau(checkpoints: CheckpointEntry[]): PlateauInfo | null {
  if (checkpoints.length < DIAGNOSTIC_MIN_CHECKPOINTS) return null;

  const firstEp = checkpoints[0].checkpoint_episode;
  const latestEp = checkpoints[checkpoints.length - 1].checkpoint_episode;
  if (latestEp - firstEp < DIAGNOSTIC_MIN_EPISODES) return null;

  const recent = checkpoints.slice(-PLATEAU_WINDOW);
  const allPrior = checkpoints.slice(0, -PLATEAU_WINDOW);
  const bestPrior = allPrior.length > 0 ? Math.max(...allPrior.map((c) => c.success_rate)) : -Infinity;
  const bestRecent = Math.max(...recent.map((c) => c.success_rate));
  const allTimeBest = Math.max(...checkpoints.map((c) => c.success_rate));

  if (bestRecent <= bestPrior + PLATEAU_MIN_DELTA) {
    const everSucceeded = allTimeBest >= CLIFF_THRESHOLD;
    let kind: PlateauInfo["kind"];

    if (bestRecent < CLIFF_THRESHOLD) {
      kind = everSucceeded ? "spike" : "cliff";
    } else if (allTimeBest >= 0.90) {
      kind = "converged";
    } else if (allTimeBest >= 0.50) {
      kind = "plateau-high";
    } else {
      kind = "plateau-low";
    }

    return {
      kind,
      window: PLATEAU_WINDOW,
      bestRate: bestRecent,
      allTimeBest,
      sinceEpisode: recent[0].checkpoint_episode,
    };
  }
  return null;
}

// ── Main component ───────────────────────────────────────────────────────────

interface DiagnosticsPanelProps {
  checkpointIndex: CheckpointIndex | null;
  bestCheckpointEpisode: number | null;
  trainingStatus: TrainingStatus | null;
  effectiveCurriculumStage: number;
}

/** Diagnostics / Learning-Curve tab. */
export function DiagnosticsPanel({
  checkpointIndex,
  bestCheckpointEpisode,
  trainingStatus,
  effectiveCurriculumStage,
}: DiagnosticsPanelProps) {
  const [viewWindow, setViewWindow] = useState<ViewWindow>(() => {
    const saved = localStorage.getItem("sheepdog_insights_view_window");
    if (saved === "all") return "all";
    if (saved === "25" || saved === "50" || saved === "100") {
      return Number(saved) as ViewWindow;
    }
    return "all";
  });
  const [selectedStageScope, setSelectedStageScope] = useState<StageScope>(() => {
    const saved = localStorage.getItem("sheepdog_insights_stage_scope");
    if (saved === "all" || saved === "current" || saved === "current-journey") {
      return saved;
    }
    if (saved !== null) {
      const parsed = Number(saved);
      if (!isNaN(parsed)) return parsed;
    }
    return "current-journey";
  });

  const targetStage = useMemo(() => {
    if (selectedStageScope === "current") return effectiveCurriculumStage;
    if (selectedStageScope === "current-journey") return effectiveCurriculumStage;
    if (selectedStageScope === "all") return effectiveCurriculumStage;
    return Number(selectedStageScope);
  }, [selectedStageScope, effectiveCurriculumStage]);

  const minStreak = useMemo(() => {
    if (trainingStatus?.auto_promote_gate?.min_qualified_streak !== undefined) {
      return trainingStatus.auto_promote_gate.min_qualified_streak;
    }
    return targetStage >= 14 ? 5 : 3;
  }, [trainingStatus?.auto_promote_gate?.min_qualified_streak, targetStage]);

  const checkpoints = useMemo(
    () => checkpointIndex?.checkpoints ?? [],
    [checkpointIndex?.checkpoints],
  );

  const stageScopedCheckpoints = useMemo(
    () => checkpoints.filter((c) => getCheckpointStage(c) === targetStage),
    [checkpoints, targetStage],
  );

  const requiredThreshold = useMemo(() => getSuccessThreshold(targetStage), [targetStage]);

  const { qualifiedStreak, isImproving, stageLatestCheckpoint, stageLatestSuccessRate, stageLatestPolicyVersion, stageLatestCheckpointEpisode, stageLatestCheckpointId, stageEvaluationSeedCount } = useMemo(() => {
    let currentStreak = 0;
    stageScopedCheckpoints.forEach((c) => {
      const isQualified = c.success_rate >= requiredThreshold;
      if (isQualified) {
        currentStreak++;
      } else {
        currentStreak = 0;
      }
    });

    let improving = false;
    if (stageScopedCheckpoints.length >= 3) {
      const last3 = stageScopedCheckpoints.slice(-3);
      if (last3[2].success_rate > last3[0].success_rate + 0.02) {
        improving = true;
      }
    }

    const latest = stageScopedCheckpoints[stageScopedCheckpoints.length - 1];
    return {
      qualifiedStreak: currentStreak,
      isImproving: improving,
      stageLatestCheckpoint: latest,
      stageLatestSuccessRate: latest ? latest.success_rate : 0.0,
      stageLatestPolicyVersion: latest ? (latest.policy_version ?? "N/A") : "N/A",
      stageLatestCheckpointEpisode: latest ? latest.checkpoint_episode : 0,
      stageLatestCheckpointId: latest ? (latest.checkpoint_id ?? "N/A") : "N/A",
      stageEvaluationSeedCount: latest ? (latest.evaluation_seeds ? latest.evaluation_seeds.length : 5) : 5,
    };
  }, [stageScopedCheckpoints, requiredThreshold]);

  const plateauInfo = useMemo(
    () => detectPlateau(stageScopedCheckpoints),
    [stageScopedCheckpoints],
  );

  const plateauRenderData = useMemo(() => {
    if (stageScopedCheckpoints.length === 0) {
      return {
        statusText: `STAGE ${targetStage === -1 ? "LEGACY" : targetStage} EVALUATION PENDING`,
        statusDetail: `Stage ${targetStage === -1 ? "Legacy" : targetStage} evaluation pending — no current-stage performance result is available.`,
        toneClass: " warning-box--warning"
      };
    }
    let statusText = "LEARNING";
    let statusDetail = "No stable plateau yet; the agent is exploring the environment and gathering initial experience.";
    let toneClass = "";

    if (stageLatestSuccessRate >= requiredThreshold && qualifiedStreak >= 5) {
      statusText = "MASTERED / READY TO PROMOTE";
      statusDetail = "The agent has converged and met all promotion criteria. Ready to advance to the next stage!";
      toneClass = " warning-box--success";
    } else if (stageLatestSuccessRate >= requiredThreshold) {
      statusText = `QUALIFIED STREAK ${qualifiedStreak}/${minStreak}`;
      statusDetail = "Performing above threshold; accumulating consecutive successful checkpoints for promotion.";
      toneClass = " warning-box--success";
    } else if (plateauInfo && (plateauInfo.kind === "plateau-low" || plateauInfo.kind === "plateau-high" || plateauInfo.kind === "converged")) {
      statusText = "PLATEAU BELOW GATE";
      statusDetail = "Performance has stabilized, but it remains below the required success threshold for promotion.";
      toneClass = " warning-box--warning";
    } else if (isImproving) {
      statusText = "IMPROVING";
      statusDetail = "Success rate is actively trending upward.";
      toneClass = "";
    } else if (plateauInfo?.kind === "cliff") {
      statusText = "CLIFF DETECTED";
      statusDetail = "The agent has never succeeded after multiple checkpoints. The environment configuration may be too difficult.";
      toneClass = " warning-box--error";
    } else if (plateauInfo?.kind === "spike") {
      statusText = "POLICY INSTABILITY";
      statusDetail = "The agent reached a high success rate but recently regressed. This is typical of PPO oscillation patterns.";
      toneClass = " warning-box--warning";
    }

    return {
      statusText,
      statusDetail,
      toneClass
    };
  }, [stageScopedCheckpoints.length, stageLatestSuccessRate, requiredThreshold, qualifiedStreak, plateauInfo, isImproving, minStreak]);

  const hasArchivedCheckpoints = useMemo(
    () => checkpoints.some((c) => c.journey != null && c.journey !== "current"),
    [checkpoints],
  );

  const currentJourneyCheckpoints = useMemo(
    () => checkpoints.filter((c) => !c.journey || c.journey === "current"),
    [checkpoints],
  );

  const archivedJourneyCheckpoints = useMemo(
    () => checkpoints.filter((c) => c.journey != null && c.journey !== "current"),
    [checkpoints],
  );

  const availableStages = useMemo(
    () => [...new Set(checkpoints.map((c) => getCheckpointStage(c)))].sort((a, b) => a - b),
    [checkpoints],
  );

  const currentJourneyStages = useMemo(
    () => [...new Set(currentJourneyCheckpoints.map((c) => getCheckpointStage(c)))].sort((a, b) => a - b),
    [currentJourneyCheckpoints],
  );

  const archivedStages = useMemo(
    () => [...new Set(archivedJourneyCheckpoints.map((c) => getCheckpointStage(c)))].sort((a, b) => a - b),
    [archivedJourneyCheckpoints],
  );

  const stageScopedViewCheckpoints = useMemo(() => {
    if (selectedStageScope === "all") {
      return checkpoints;
    }
    if (selectedStageScope === "current-journey") {
      return currentJourneyCheckpoints;
    }
    const targetStage = selectedStageScope === "current" ? effectiveCurriculumStage : selectedStageScope;
    return checkpoints.filter(
      (c) => getCheckpointStage(c) === targetStage,
    );
  }, [checkpoints, currentJourneyCheckpoints, selectedStageScope, effectiveCurriculumStage]);

  const filteredCheckpoints = useMemo(() => {
    if (viewWindow === "all") return stageScopedViewCheckpoints;
    return stageScopedViewCheckpoints.slice(-viewWindow);
  }, [stageScopedViewCheckpoints, viewWindow]);

  const stages = useMemo(
    () => filteredCheckpoints.map((c) => getCheckpointStage(c)),
    [filteredCheckpoints],
  );

  const successData: ChartPoint[] = useMemo(() => {
    let runningMaxRate = -Infinity;
    let runningMinSteps = Infinity;
    return filteredCheckpoints.map((c, i) => {
      const rate = c.success_rate;
      const steps = c.average_completion_steps ?? Infinity;
      const betterRate = rate > runningMaxRate;
      const betterSteps = rate === runningMaxRate && steps < runningMinSteps;
      const isPrevBest = betterRate || betterSteps;
      if (isPrevBest) {
        if (betterRate) {
          runningMaxRate = rate;
          runningMinSteps = steps;
        } else {
          runningMinSteps = steps;
        }
      }
      const labelText =
        (isPrevBest || c.checkpoint_episode === bestCheckpointEpisode) &&
        c.average_completion_steps != null
          ? String(Math.round(c.average_completion_steps))
          : undefined;
      return {
        x: c.checkpoint_episode,
        y: rate,
        stage: stages[i],
        isBest: c.checkpoint_episode === bestCheckpointEpisode,
        isPrevBest,
        labelText,
        secondaryY: c.average_completion_steps ?? null,
      };
    });
  }, [filteredCheckpoints, stages, bestCheckpointEpisode]);

  const rewardData: ChartPoint[] = useMemo(
    () =>
      filteredCheckpoints.map((c, i) => ({
        x: c.checkpoint_episode,
        y: c.average_reward,
        stage: stages[i],
        isBest: c.checkpoint_episode === bestCheckpointEpisode,
      })),
    [filteredCheckpoints, stages, bestCheckpointEpisode],
  );

  const sheepData: ChartPoint[] = useMemo(
    () =>
      filteredCheckpoints.map((c, i) => ({
        x: c.checkpoint_episode,
        y: c.average_sheep_penned,
        stage: stages[i],
        isBest: c.checkpoint_episode === bestCheckpointEpisode,
      })),
    [filteredCheckpoints, stages, bestCheckpointEpisode],
  );

  const rewardRange = useMemo(() => {
    if (!rewardData.length) return { min: 0, max: 1 };
    const vals = rewardData.map((d) => d.y);
    const minV = Math.min(...vals);
    const maxV = Math.max(...vals);
    const pad = (maxV - minV) * 0.12 || 1;
    return { min: minV - pad, max: maxV + pad };
  }, [rewardData]);

  const maxSheepPenned = useMemo(() => {
    if (!sheepData.length) return 1;
    return Math.max(...sheepData.map((d) => d.y), 1);
  }, [sheepData]);

  const stepsRange = useMemo(() => {
    const vals = filteredCheckpoints
      .map((c) => c.average_completion_steps)
      .filter((v): v is number => v != null && v > 0);
    if (!vals.length) return { min: 0, max: 500 };
    const minV = Math.min(...vals);
    const maxV = Math.max(...vals);
    const pad = (maxV - minV) * 0.15 || 30;
    return { min: Math.max(0, minV - pad), max: maxV + pad };
  }, [filteredCheckpoints]);

  // Reverse-order rows for the table (newest first)
  const tableRows = useMemo(() => [...filteredCheckpoints].reverse(), [filteredCheckpoints]);

  const isLiveTraining = trainingStatus?.running ?? false;
  const uniqueStages = useMemo(() => [...new Set(stages)].sort((a, b) => a - b), [stages]);
  const [activeChart, setActiveChart] = useState<ChartTab>(() => {
    const saved = localStorage.getItem("sheepdog_insights_active_chart") as ChartTab | null;
    const validCharts: ChartTab[] = ["health", "success", "reward", "sheep", "history", "learningSignal"];
    if (saved && validCharts.includes(saved)) {
      return saved;
    }
    return "success";
  });
  const [learningSignalWindow, setLearningSignalWindow] = useState<ViewWindow>(() => {
    const saved = localStorage.getItem("sheepdog_insights_learning_signal_window");
    if (saved === "all") return "all";
    if (saved === "25" || saved === "50" || saved === "100") {
      return Number(saved) as ViewWindow;
    }
    return "all";
  });
  const [learningSignalSmoothWindow, setLearningSignalSmoothWindow] = useState<SmoothingWindow>(() => {
    const saved = localStorage.getItem("sheepdog_insights_learning_signal_smooth_window");
    if (saved === "25" || saved === "50" || saved === "100") {
      return Number(saved) as SmoothingWindow;
    }
    return 50;
  });

  useEffect(() => {
    localStorage.setItem("sheepdog_insights_view_window", String(viewWindow));
  }, [viewWindow]);

  useEffect(() => {
    localStorage.setItem("sheepdog_insights_stage_scope", String(selectedStageScope));
  }, [selectedStageScope]);

  useEffect(() => {
    localStorage.setItem("sheepdog_insights_active_chart", activeChart);
  }, [activeChart]);

  useEffect(() => {
    localStorage.setItem("sheepdog_insights_learning_signal_window", String(learningSignalWindow));
  }, [learningSignalWindow]);

  useEffect(() => {
    localStorage.setItem("sheepdog_insights_learning_signal_smooth_window", String(learningSignalSmoothWindow));
  }, [learningSignalSmoothWindow]);
  const [focusedBreakthroughCheckpoint, setFocusedBreakthroughCheckpoint] = useState<number | null>(null);
  const [advisorExplainOpen, setAdvisorExplainOpen] = useState(false);
  const [breakthroughNotes, setBreakthroughNotes] = useState<Record<number, string>>({});
  const [isHelpOpen, setIsHelpOpen] = useState(false);

  const latestCheckpoint = checkpoints[checkpoints.length - 1] ?? null;
  const recentCheckpoints = useMemo(() => checkpoints.slice(-RECENT_WINDOW), [checkpoints]);
  const previousWindow = useMemo(() => checkpoints.slice(-(RECENT_WINDOW * 2), -RECENT_WINDOW), [checkpoints]);

  const latestSuccessRate = latestCheckpoint?.success_rate ?? null;
  const latestReward = latestCheckpoint?.average_reward ?? null;
  const latestSheepPenned = latestCheckpoint?.average_sheep_penned ?? null;
  const latestTimeoutRate = latestCheckpoint?.timeout_rate ?? null;
  const latestSteps = latestCheckpoint?.average_completion_steps ?? null;
  const latestNoProgress = trainingStatus?.latest_avg_no_progress_steps ?? null;

  const recentSuccessRate = average(recentCheckpoints.map((entry) => entry.success_rate));
  const recentReward = average(recentCheckpoints.map((entry) => entry.average_reward));
  const recentTimeoutRate = average(recentCheckpoints.map((entry) => entry.timeout_rate));
  const recentSheepPenned = average(recentCheckpoints.map((entry) => entry.average_sheep_penned));
  const recentSteps = average(recentCheckpoints.map((entry) => entry.average_completion_steps));

  const priorSuccessRate = average(previousWindow.map((entry) => entry.success_rate));
  const priorReward = average(previousWindow.map((entry) => entry.average_reward));
  const priorTimeoutRate = average(previousWindow.map((entry) => entry.timeout_rate));
  const priorSheepPenned = average(previousWindow.map((entry) => entry.average_sheep_penned));
  const priorSteps = average(previousWindow.map((entry) => entry.average_completion_steps));

  const recentSuccessDelta = recentSuccessRate != null && priorSuccessRate != null ? recentSuccessRate - priorSuccessRate : null;
  const recentRewardDelta = recentReward != null && priorReward != null ? recentReward - priorReward : null;
  const recentTimeoutDelta = recentTimeoutRate != null && priorTimeoutRate != null ? recentTimeoutRate - priorTimeoutRate : null;
  const recentSheepDelta = recentSheepPenned != null && priorSheepPenned != null ? recentSheepPenned - priorSheepPenned : null;
  const recentStepsDelta = recentSteps != null && priorSteps != null ? recentSteps - priorSteps : null;

  const improvementScore =
    (recentSuccessDelta ?? 0) +
    ((recentRewardDelta ?? 0) / 100) +
    ((recentSheepDelta ?? 0) / 10) -
    ((recentTimeoutDelta ?? 0) / 2) -
    ((recentStepsDelta ?? 0) / 500);

  const stageCheckpoints = checkpoints.filter(
    (entry) => getCheckpointStage(entry) === effectiveCurriculumStage,
  );
  const stageBestSuccessRate = stageCheckpoints.length ? Math.max(...stageCheckpoints.map((entry) => entry.success_rate)) : null;
  const stageBestReward = stageCheckpoints.length ? Math.max(...stageCheckpoints.map((entry) => entry.average_reward)) : null;
  const stageMedianSteps = stageCheckpoints.length
    ? [...stageCheckpoints.map((entry) => entry.average_completion_steps)].sort((a, b) => a - b)[Math.floor(stageCheckpoints.length / 2)]
    : null;
  const currentStageBestSuccessRate =
    stageCheckpoints.length > 0 ? Math.max(...stageCheckpoints.map((entry) => entry.success_rate)) : 0;
  const plateauKind = plateauInfo?.kind ?? null;
  const abovePromotionThreshold = latestSuccessRate != null && latestSuccessRate >= PROMOTE_THRESHOLD;
  const recentImproving = improvementScore > 0.01;
  const decisionSignal = buildDecisionSignal({
    checkpointCount: checkpoints.length,
    latestSuccessRate,
    latestReward,
    latestTimeoutRate,
    plateauKind,
    stage: effectiveCurriculumStage,
    abovePromotionThreshold,
    improving: recentImproving,
  });

  const readinessTone: DecisionTone =
    decisionSignal.tone === "danger"
      ? "danger"
      : decisionSignal.tone === "good"
        ? "good"
        : decisionSignal.tone === "warn"
          ? "warn"
          : "muted";

  const learningSignalSource = useMemo(() => {
    if (learningSignalWindow === "all") return stageScopedViewCheckpoints;
    return stageScopedViewCheckpoints.slice(-learningSignalWindow);
  }, [stageScopedViewCheckpoints, learningSignalWindow]);

  const learningSignalPoints = useMemo<LearningPoint[]>(
    () =>
      learningSignalSource.map((entry) => ({
        checkpoint: entry.checkpoint_episode,
        successRate: entry.success_rate,
        reward: entry.average_reward,
        timeoutRate: entry.timeout_rate,
        completionSteps: entry.average_completion_steps,
        sheepPenned: entry.average_sheep_penned,
        noProgressGuard: averageNoProgress(entry),
        stage: getCheckpointStage(entry),
        checkpoint_id: entry.checkpoint_id,
      })),
    [learningSignalSource],
  );

  const learningSignalAnalysis = useMemo(
    () => analyzeLearningSignal(learningSignalPoints, learningSignalSmoothWindow),
    [learningSignalPoints, learningSignalSmoothWindow],
  );

  const rewardValues = learningSignalPoints.map((point) => point.reward);
  const timeoutValues = learningSignalPoints.map((point) => point.timeoutRate);
  const speedValues = learningSignalPoints.map((point) => point.completionSteps);
  const noProgressValues = learningSignalPoints.map((point) => point.noProgressGuard);

  const rewardTrend = useMemo(() => rewardTrendLabel(rewardValues), [rewardValues]);
  const timeoutTrend = useMemo(() => timeoutTrendLabel(timeoutValues), [timeoutValues]);
  const speedTrend = useMemo(() => speedTrendLabel(speedValues), [speedValues]);

  const recentWindow = 50;
  const currentRewardWindow = rewardValues.slice(-recentWindow);
  const priorRewardWindow = rewardValues.slice(-recentWindow * 2, -recentWindow);
  const currentSpeedWindow = speedValues.slice(-recentWindow);
  const priorSpeedWindow = speedValues.slice(-recentWindow * 2, -recentWindow);
  const rewardDelta =
    currentRewardWindow.length && priorRewardWindow.length
      ? average(currentRewardWindow)! - average(priorRewardWindow)!
      : null;
  const speedDelta =
    currentSpeedWindow.length && priorSpeedWindow.length
      ? average(currentSpeedWindow)! - average(priorSpeedWindow)!
      : null;

  const recentSuccess = average(learningSignalPoints.slice(-recentWindow).map((point) => point.successRate));
  const recentTimeout = average(learningSignalPoints.slice(-recentWindow).map((point) => point.timeoutRate));
  const recentNoProgress = average(noProgressValues.slice(-recentWindow));
  const highSuccessStable =
    learningSignalPoints.length >= 50 &&
    learningSignalPoints.slice(-50).every((point) => point.successRate >= 0.8) &&
    learningSignalPoints[learningSignalPoints.length - 1].successRate >=
      learningSignalPoints[Math.max(0, learningSignalPoints.length - 50)].successRate - 0.02;

  const rewardSignalSummary: SignalSummary = {
    label: rewardTrend,
    tone: rewardTrend === "Climbing" ? "good" : rewardTrend === "Declining" ? "danger" : rewardTrend === "Volatile" ? "warn" : "muted",
  };
  const timeoutSignalSummary: SignalSummary = {
    label: timeoutTrend,
    tone: timeoutTrend === "Improving" ? "good" : timeoutTrend === "Getting Worse" ? "danger" : "warn",
  };
  const speedSignalSummary: SignalSummary = {
    label: speedTrend,
    tone: speedTrend === "Getting Faster" ? "good" : speedTrend === "Slowing" ? "warn" : "muted",
  };

  const learningAdvisor = useMemo(
    () =>
      buildAdvisor({
        currentFlatStreak: learningSignalAnalysis.currentFlatStreak,
        longestHistoricalPlateau: learningSignalAnalysis.longestHistoricalPlateau,
        rewardSignal: rewardSignalSummary,
        timeoutSignal: timeoutSignalSummary,
        speedSignal: speedSignalSummary,
        recentSuccess,
        recentRewardDelta: rewardDelta,
        recentSpeedDelta: speedDelta,
        recentTimeout,
        recentNoProgress,
        highSuccessStable,
      }),
    [
      learningSignalAnalysis.currentFlatStreak,
      learningSignalAnalysis.longestHistoricalPlateau,
      rewardSignalSummary,
      timeoutSignalSummary,
      speedSignalSummary,
      recentSuccess,
      rewardDelta,
      speedDelta,
      recentTimeout,
      recentNoProgress,
      highSuccessStable,
    ],
  );

  const flatContextStatus =
    learningSignalAnalysis.currentFlatStreak <= learningSignalAnalysis.longestHistoricalPlateau ||
    learningSignalAnalysis.longestHistoricalPlateau === 0
      ? `Within normal plateau range — previous breakthroughs took up to ${learningSignalAnalysis.longestHistoricalPlateau || learningSignalAnalysis.currentFlatStreak} episodes`
      : "Exceeding longest known plateau — consider intervention";

  // ── Empty state ───────────────────────────────────────────────────────────
  if (checkpoints.length === 0) {
    return (
      <section className="training-card" aria-label="Diagnostics">
        <div className="training-card__header">
          <div>
            <p className="eyebrow">Insights</p>
            <h2>Learning Curve</h2>
          </div>
        </div>
        <div className="warning-box" role="status">
          No checkpoints yet — run a batch of training to see diagnostics.
        </div>
        <div className="diag-explainer">
          <strong>How training resumes</strong>
          <p>
            You never need to pick a starting episode. The trainer automatically loads the best saved
            model each time you click Start Training. Episode numbers are cumulative counters, not
            scenario selectors.
          </p>
        </div>
      </section>
    );
  }

  // ── Full panel ────────────────────────────────────────────────────────────
  return (
    <section className="training-card training-card--insights" aria-label="Diagnostics">
      <div className="training-card__header">
        <div>
          <p className="eyebrow">Insights</p>
          <h2>Learning Curve</h2>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
          {isLiveTraining ? <span className="pill pill--live">live</span> : null}
          <span className="pill pill--muted">{checkpoints.length} pts</span>
          <CopyAgentDataButton
            trainingStatus={trainingStatus}
            checkpointIndex={checkpointIndex}
            curriculumStage={effectiveCurriculumStage}
          />
          <button
            onClick={() => setIsHelpOpen(true)}
            className="insights-help-btn"
            title="What this page means?"
            aria-label="What this page means?"
          >
            <svg
              width="13"
              height="13"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2.5"
              strokeLinecap="round"
              strokeLinejoin="round"
              style={{ marginRight: "0.35rem" }}
            >
              <circle cx="12" cy="12" r="10" />
              <path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3" />
              <line x1="12" y1="17" x2="12.01" y2="17" />
            </svg>
            What this page means?
          </button>
        </div>
      </div>

      {/* Plateau / cliff / spike alert */}
      {plateauRenderData ? (
        <div
          className={`warning-box${plateauRenderData.toneClass}`}
          role="status"
          style={{ display: "flex", flexDirection: "column", gap: "0.5rem" }}
        >
          <div>
            <strong>Status: <span style={{ textDecoration: "underline" }}>{plateauRenderData.statusText}</span></strong> — {plateauRenderData.statusDetail}
          </div>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))", gap: "0.25rem", fontSize: "0.9em", borderTop: "1px solid rgba(255,255,255,0.1)", paddingTop: "0.5rem", marginTop: "0.25rem" }}>
            <div>• <strong>Stage:</strong> {targetStage === -1 ? "Legacy/Unknown" : targetStage === 0 ? "Base difficulty" : `Stage ${targetStage}`}</div>
            <div>• <strong>Latest Checkpoint:</strong> {stageLatestCheckpointEpisode > 0 ? `ep ${stageLatestCheckpointEpisode}` : "None"} ({stageLatestCheckpointId})</div>
            <div>• <strong>Policy Version:</strong> {stageLatestPolicyVersion}</div>
            <div>• <strong>Evaluation Seeds:</strong> {stageEvaluationSeedCount} seeds</div>
            <div>• <strong>Success Rate:</strong> {stageLatestCheckpoint ? `${Math.round(stageLatestSuccessRate * 100)}%` : "N/A"}</div>
            <div>• <strong>Required Threshold:</strong> {Math.round(requiredThreshold * 100)}%</div>
            <div>• <strong>Qualified Streak:</strong> {qualifiedStreak} / {minStreak}</div>
          </div>
        </div>
      ) : null}

      {/* How training works explainer */}
      <details className="diag-explainer-details">
        <summary>How training resumes · episode control</summary>
        <div className="diag-explainer">
          <p>
            <strong>You do not control the start episode.</strong> Each time you click Start
            Training, the trainer automatically loads the best saved model checkpoint and runs the
            next batch of episodes from there. The episode counter is cumulative — it is not a
            scenario you replay from.
          </p>
          <p>
            If training is stuck, the best levers are: promote to the next curriculum stage (simpler
            environment), add more episodes per batch, or Clear Training and try again from a fresh
            start at Stage 1.
          </p>
        </div>
      </details>

      {/* View window filter */}
      <div className="view-filter">
        <label className="view-filter__label" htmlFor="insights-stage-scope">Stage</label>
        <select
          id="insights-stage-scope"
          aria-label="Stage scope"
          className="view-filter__select"
          value={selectedStageScope === "all" ? "all" : selectedStageScope === "current" ? "current" : selectedStageScope === "current-journey" ? "current-journey" : String(selectedStageScope)}
          onChange={(event) => {
            const nextValue = event.target.value;
            if (nextValue === "all" || nextValue === "current" || nextValue === "current-journey") {
              setSelectedStageScope(nextValue);
              return;
            }
            setSelectedStageScope(Number.parseInt(nextValue, 10));
          }}
        >
          <option value="current-journey">Current journey</option>
          <option value="current">Current stage ({stageLabel(effectiveCurriculumStage)})</option>
          {hasArchivedCheckpoints && (
            <option value="all">All journeys</option>
          )}
          {currentJourneyStages.length > 1 && (
            <optgroup label="Current journey stages">
              {currentJourneyStages.map((stage) => (
                <option key={`current-stage-${stage}`} value={String(stage)}>
                  {stageLabel(stage)}
                </option>
              ))}
            </optgroup>
          )}
          {hasArchivedCheckpoints && archivedStages.length > 0 && (
            <optgroup label="Archived journey stages">
              {archivedStages.map((stage) => (
                <option key={`archived-stage-${stage}`} value={String(stage)}>
                  {stageLabel(stage)}
                </option>
              ))}
            </optgroup>
          )}
        </select>
        <span className="view-filter__label">Window</span>
        <div className="chart-tabs" role="group" aria-label="Chart window">
          {VIEW_WINDOW_OPTIONS.map(({ value, label }) => (
            <button
              key={String(value)}
              className={`chart-tab${viewWindow === value ? " chart-tab--active" : ""}`}
              onClick={() => setViewWindow(value)}
            >
              {label}
            </button>
          ))}
        </div>
      </div>

      {/* Chart sub-tabs */}
      <div className="chart-tabs" role="tablist">
        {(["success", "reward", "sheep", "history", "health", "learningSignal"] as ChartTab[]).map((id) => {
          const labels: Record<ChartTab, string> = {
            success: "Success Rate",
            reward: "Avg Reward",
            sheep: "Sheep Penned",
            history: "History",
            health: "Health",
            learningSignal: "Learning Signal",
          };
          return (
            <button
              key={id}
              role="tab"
              aria-selected={activeChart === id}
              className={`chart-tab${activeChart === id ? " chart-tab--active" : ""}`}
              onClick={() => setActiveChart(id)}
            >
              {labels[id]}
            </button>
          );
        })}
      </div>

      <div className="insights-tab-content">
      {activeChart === "learningSignal" && (
        <div className="chart-view">
          <section className="learning-signal" aria-label="Learning Signal">
            <header className="learning-signal__header">
              <div>
                <h3>
                  Learning Signal
                  <InfoTip text="A learning-focused view to decide whether to keep training, tune parameters, or advance stage." />
                </h3>
                <p>Is the model still learning, or do I need to intervene?</p>
              </div>
              <div className="learning-signal__controls">
                <div className="learning-signal__pill-group" role="group" aria-label="Learning signal data window">
                  {LEARNING_SIGNAL_WINDOW_OPTIONS.map(({ value, label }) => (
                    <button
                      key={`learning-window-${String(value)}`}
                      className={`chart-tab${learningSignalWindow === value ? " chart-tab--active" : ""}`}
                      onClick={() => setLearningSignalWindow(value)}
                    >
                      {label}
                    </button>
                  ))}
                </div>
                <div className="learning-signal__pill-group" role="group" aria-label="Smoothing window">
                  {SMOOTHING_WINDOWS.map((windowSize) => (
                    <button
                      key={`smooth-${windowSize}`}
                      className={`chart-tab${learningSignalSmoothWindow === windowSize ? " chart-tab--active" : ""}`}
                      onClick={() => setLearningSignalSmoothWindow(windowSize)}
                    >
                      Smooth {windowSize}
                    </button>
                  ))}
                </div>
              </div>
            </header>

            <section className="learning-signal__section">
              <div className="learning-signal__section-title">
                <h4>
                  The Learning Curve
                  <InfoTip text="Flat stretches can be normal in PPO. Breakthrough markers indicate first strong jumps after prolonged plateaus." />
                </h4>
              </div>
              <LearningSignalChart
                points={learningSignalPoints}
                smoothedSuccessRate={learningSignalAnalysis.smoothedSuccessRate}
                flatZones={learningSignalAnalysis.flatZones}
                breakthroughs={learningSignalAnalysis.breakthroughs}
                currentCheckpoint={learningSignalPoints[learningSignalPoints.length - 1]?.checkpoint ?? null}
                stageBestSuccessRate={currentStageBestSuccessRate}
                focusedCheckpoint={focusedBreakthroughCheckpoint}
                onBreakthroughClick={(checkpoint) => setFocusedBreakthroughCheckpoint(checkpoint)}
              />
            </section>

            <section className="learning-signal__section">
              <div className="learning-signal__section-title">
                <h4>
                  Flat Streak Context
                  <InfoTip text="Compares your current flat streak against pre-breakthrough plateaus from your own training history." />
                </h4>
              </div>
              <div className="flat-context">
                <div className="flat-context__main">
                  <span className="flat-context__label">Current flat streak</span>
                  <strong>{learningSignalAnalysis.currentFlatStreak} episodes</strong>
                </div>
                <div className="flat-context__bars">
                  {learningSignalAnalysis.breakthroughs.length === 0 ? (
                    <div className="flat-context__empty">No historical breakthroughs yet</div>
                  ) : (
                    learningSignalAnalysis.breakthroughs.map((event) => {
                      const denom = Math.max(
                        learningSignalAnalysis.longestHistoricalPlateau,
                        learningSignalAnalysis.currentFlatStreak,
                        1,
                      );
                      const width = `${Math.max(8, (event.flatEpisodesBefore / denom) * 100)}%`;
                      return (
                        <div key={`bar-${event.checkpoint}`} className="flat-context__bar-row">
                          <span className="flat-context__bar-label">Before B{event.index}</span>
                          <div className="flat-context__bar-track">
                            <span className="flat-context__bar-fill" style={{ width }} />
                          </div>
                          <span className="flat-context__bar-value">{event.flatEpisodesBefore}</span>
                        </div>
                      );
                    })
                  )}
                </div>
                <div
                  className={`warning-box${
                    flatContextStatus.startsWith("Exceeding") ? " warning-box--warning" : " warning-box--success"
                  }`}
                >
                  {flatContextStatus}
                </div>
              </div>
            </section>

            <section className="learning-signal__section">
              <div className="learning-signal__section-title">
                <h4>
                  Multi-Signal Trend Panel
                  <InfoTip text="Cross-check reward, timeout, and speed to detect hidden progress even when success is flat." />
                </h4>
              </div>
              <div className="signal-grid">
                <article className="signal-card">
                  <header>
                    <h5>Reward Trend</h5>
                    <span className={`signal-arrow signal-arrow--${rewardSignalSummary.tone}`}>{trendArrow(rewardTrend)}</span>
                  </header>
                  <Sparkline values={rewardValues} color="rgba(244,197,66,0.9)" />
                  <p>{rewardTrend}</p>
                </article>
                <article className="signal-card">
                  <header>
                    <h5>Timeout Trend</h5>
                    <span className={`signal-arrow signal-arrow--${timeoutSignalSummary.tone}`}>{trendArrow(timeoutTrend)}</span>
                  </header>
                  <Sparkline values={timeoutValues} color="rgba(251,113,133,0.9)" />
                  <p>{timeoutTrend} (lower is better)</p>
                </article>
                <article className="signal-card">
                  <header>
                    <h5>Completion Speed</h5>
                    <span className={`signal-arrow signal-arrow--${speedSignalSummary.tone}`}>{trendArrow(speedTrend)}</span>
                  </header>
                  <Sparkline values={speedValues} color="rgba(125,211,252,0.9)" />
                  <p>{speedTrend} (fewer steps is better)</p>
                </article>
              </div>
            </section>

            <section className="learning-signal__section">
              <div className="learning-signal__section-title">
                <h4>
                  Intervention Advisor
                  <InfoTip text="Actionable guidance generated from plateau duration, trend direction, and convergence stability." />
                </h4>
              </div>
              <div className={`advisor advisor--${learningAdvisor.tone}`}>
                <header className="advisor__header">
                  <span className={`pill pill--${learningAdvisor.tone}`}>State {learningAdvisor.state}</span>
                  <h5>{learningAdvisor.title}</h5>
                </header>
                <p>{learningAdvisor.body}</p>
                <ul className="advisor__actions">
                  {learningAdvisor.actions.map((action, index) => (
                    <li key={`action-${index}`}>{action}</li>
                  ))}
                </ul>
                <button className="advisor__why" onClick={() => setAdvisorExplainOpen((open) => !open)}>
                  {advisorExplainOpen ? "Hide" : "Show"} Why this recommendation?
                </button>
                {advisorExplainOpen ? <div className="advisor__reason">{learningAdvisor.reason}</div> : null}
              </div>
            </section>

            <section className="learning-signal__section">
              <div className="learning-signal__section-title">
                <h4>
                  Breakthrough History
                  <InfoTip text="Track where breakthroughs happened and annotate what changed so you can learn from past interventions." />
                </h4>
              </div>
              <div className="diag-table-wrap">
                <table className="diag-table learning-signal__table">
                  <thead>
                    <tr>
                      <th>Breakthrough #</th>
                      <th>At Checkpoint</th>
                      <th>After Flat Episodes</th>
                      <th>Success Jump</th>
                      <th>What changed</th>
                    </tr>
                  </thead>
                  <tbody>
                    {learningSignalAnalysis.breakthroughs.length === 0 ? (
                      <tr>
                        <td colSpan={5} className="learning-signal__table-empty">No breakthroughs detected yet.</td>
                      </tr>
                    ) : (
                      learningSignalAnalysis.breakthroughs.map((event) => (
                        <tr key={`breakthrough-row-${event.checkpoint}`}>
                          <td>#{event.index}</td>
                          <td>
                            <button
                              className="learning-signal__jump-btn"
                              onClick={() => setFocusedBreakthroughCheckpoint(event.checkpoint)}
                            >
                              {event.checkpoint}
                            </button>
                          </td>
                          <td>{event.flatEpisodesBefore}</td>
                          <td>
                            {Math.round(event.fromSuccessRate * 100)}% → {Math.round(event.toSuccessRate * 100)}%
                          </td>
                          <td>
                            <input
                              className="learning-signal__note-input"
                              value={breakthroughNotes[event.checkpoint] ?? ""}
                              placeholder="manual note"
                              onChange={(e) =>
                                setBreakthroughNotes((prev) => ({
                                  ...prev,
                                  [event.checkpoint]: e.target.value,
                                }))
                              }
                            />
                          </td>
                        </tr>
                      ))
                    )}
                  </tbody>
                </table>
              </div>
            </section>
          </section>
        </div>
      )}

      {activeChart === "health" && (
        <div className="chart-view">
          <section className="health-dashboard" aria-label="Training health overview">
            <div className="health-dashboard__hero">
              <div>
                <p className="eyebrow">Training health</p>
                <h3>{decisionSignal.title}</h3>
                <p className="health-dashboard__copy">{decisionSignal.body}</p>
              </div>
              <div className="health-dashboard__badge-wrap">
                <span className={`pill pill--${readinessTone}`}>{decisionSignal.badge}</span>
                <span className="pill pill--muted">{checkpoints.length} checkpoints</span>
              </div>
            </div>

            <div className="health-dashboard__grid">
              <StatCard
                label="Latest success"
                value={formatPercent(latestSuccessRate)}
                detail={trendSummary(latestSuccessRate, priorSuccessRate, (value) => `${Math.round(value * 100)}%`) + ` · promo bar ${Math.round(PROMOTE_THRESHOLD * 100)}%`}
                tone={latestSuccessRate != null && latestSuccessRate >= PROMOTE_THRESHOLD ? "good" : "warn"}
              />
              <StatCard
                label="Latest reward"
                value={formatNumber(latestReward, 1)}
                detail={trendSummary(latestReward, priorReward, (value) => value.toFixed(1))}
                tone={latestReward != null && recentRewardDelta != null && recentRewardDelta > 0 ? "good" : "muted"}
              />
              <StatCard
                label="Latest sheep penned"
                value={formatNumber(latestSheepPenned, 1)}
                detail={trendSummary(latestSheepPenned, priorSheepPenned, (value) => value.toFixed(1))}
                tone={latestSheepPenned != null && recentSheepDelta != null && recentSheepDelta > 0 ? "good" : "muted"}
              />
              <StatCard
                label="Latest timeout rate"
                value={formatPercent(latestTimeoutRate)}
                detail={trendSummary(latestTimeoutRate, priorTimeoutRate, (value) => `${Math.round(value * 100)}%`) + " · lower is better"}
                tone={latestTimeoutRate != null && latestTimeoutRate >= 0.6 ? "danger" : "muted"}
              />
              <StatCard
                label="Latest completion steps"
                value={formatNumber(latestSteps, 0)}
                detail={trendSummary(latestSteps, priorSteps, (value) => `${Math.round(value)}`) + " · fewer is better"}
                tone={latestSteps != null && recentStepsDelta != null && recentStepsDelta < 0 ? "good" : "muted"}
              />
              <StatCard
                label="No-progress guard"
                value={formatNumber(latestNoProgress, 0)}
                detail="High values usually mean the policy is moving but not converting motion into penning"
                tone={latestNoProgress != null && latestNoProgress > 0 ? "warn" : "muted"}
              />
            </div>

            <div className="health-dashboard__callout health-dashboard__callout--neutral">
              <div>
                <strong>Live training diagnostics</strong>
                <p>
                  Monitor rolling success, timeout rate, reward trend, completion steps, and stage-over-stage deltas to assess stability, convergence quality, and curriculum generalization in the current run.
                </p>
              </div>
              <div className="health-dashboard__callout-metrics">
                <span><strong>{formatPercent(recentSuccessRate)}</strong> recent success</span>
                <span><strong>{formatNumber(recentReward, 1)}</strong> recent reward</span>
                <span><strong>{formatPercent(recentTimeoutRate)}</strong> recent timeout</span>
                <span><strong>{formatNumber(stageBestSuccessRate, 2)}</strong> stage best success</span>
                <span><strong>{formatNumber(stageBestReward, 1)}</strong> stage best reward</span>
                <span><strong>{formatNumber(stageMedianSteps, 0)}</strong> stage median steps</span>
              </div>
            </div>
          </section>
        </div>
      )}

      {activeChart === "success" && (
        <div className="chart-view">
          <LineChart
            data={successData}
            lineColor="var(--good)"
            yMin={0}
            yMax={1}
            formatY={(v) => `${Math.round(v * 100)}%`}
            referenceY={0.5}
            referenceLabel="50%"
            bestEpisode={bestCheckpointEpisode}
            showPrevBestLabels
            secondaryYMin={stepsRange.min}
            secondaryYMax={stepsRange.max}
            secondaryLineColor="rgba(251,146,60,0.85)"
            formatSecondaryY={(v) => String(Math.round(v))}
          />
          <ChartLegend
            entries={[
              { symbol: { kind: "line", color: "var(--good)" }, label: "Success rate", detail: "fraction of eval episodes where all sheep were penned" },
              { symbol: { kind: "dash", color: "rgba(251,146,60,0.85)" }, label: "Avg steps", detail: "average completion steps for successful episodes — fewer is faster (right axis, top = fewer)" },
              { symbol: { kind: "dash", color: "rgba(74,222,128,0.65)" }, label: "50% target", detail: "recommended threshold to promote to next curriculum stage" },
              { symbol: { kind: "diamond", color: "#9ca3af" }, label: "Running best", detail: "each point where a new personal best success rate was set — label shows the rate" },
              { symbol: { kind: "ring", color: "#9ca3af" }, label: "All-time best", detail: "the model currently loaded for inference — also shown with an outer ring" },
              ...uniqueStages.map((s) => ({ symbol: { kind: "dot" as const, color: stageColor(s) }, label: stageLabel(s), detail: s === 0 ? "no curriculum — base difficulty" : `curriculum stage ${s}` })),
            ]}
          />
        </div>
      )}

      {activeChart === "reward" && (
        <div className="chart-view">
          <LineChart
            data={rewardData}
            lineColor="var(--accent)"
            yMin={rewardRange.min}
            yMax={rewardRange.max}
            formatY={(v) => v.toFixed(1)}
            bestEpisode={bestCheckpointEpisode}
          />
          <ChartLegend
            entries={[
              { symbol: { kind: "line", color: "var(--accent)" }, label: "Avg reward", detail: "mean total reward per episode — higher is better, but success rate matters more" },
              ...uniqueStages.map((s) => ({ symbol: { kind: "dot" as const, color: stageColor(s) }, label: stageLabel(s), detail: s === 0 ? "no curriculum — base difficulty" : `curriculum stage ${s}` })),
              { symbol: { kind: "ring", color: "#9ca3af" }, label: "Best checkpoint", detail: "the model currently loaded for inference — shown with an outer ring" },
            ]}
          />
        </div>
      )}

      {activeChart === "sheep" && (
        <div className="chart-view">
          <LineChart
            data={sheepData}
            lineColor="#c084fc"
            yMin={0}
            yMax={maxSheepPenned}
            formatY={(v) => v.toFixed(1)}
            bestEpisode={bestCheckpointEpisode}
          />
          <ChartLegend
            entries={[
              { symbol: { kind: "line", color: "#c084fc" }, label: "Avg sheep penned", detail: "average sheep penned per episode — flat at 0 = cliff; rising trend = learning" },
              ...uniqueStages.map((s) => ({ symbol: { kind: "dot" as const, color: stageColor(s) }, label: stageLabel(s), detail: s === 0 ? "no curriculum — base difficulty" : `curriculum stage ${s}` })),
              { symbol: { kind: "ring", color: "#9ca3af" }, label: "Best checkpoint", detail: "the model currently loaded for inference — shown with an outer ring" },
            ]}
          />
        </div>
      )}

      {activeChart === "history" && (
        <div className="chart-view">
          <div className="diag-table-wrap">
            <table className="diag-table">
              <thead>
                <tr>
                  <th>Ep</th>
                  <th>St</th>
                  <th>Success</th>
                  <th>Reward</th>
                  <th>Sheep</th>
                  <th>Timeout</th>
                  <th>Steps</th>
                </tr>
              </thead>
              <tbody>
                {tableRows.map((c) => {
                  const isBest = c.checkpoint_episode === bestCheckpointEpisode;
                  const cStage = getCheckpointStage(c);
                  const isArchived = c.journey != null && c.journey !== "current";
                  return (
                    <tr
                      key={`${c.journey ?? "current"}-${c.checkpoint_episode}`}
                      className={isBest ? "diag-table__row--best" : isArchived ? "diag-table__row--archived" : undefined}
                    >
                      <td>
                        <span
                        className="diag-table__stage-dot"
                        style={{ background: stageColor(cStage) }}
                      />
                      {isBest ? "★ " : ""}
                      {c.checkpoint_episode}
                      {isArchived && <span className="diag-table__archived-badge" title={c.journey}>⏪</span>}
                    </td>
                    <td>{cStage ?? "—"}</td>
                    <td
                      style={{
                        color:
                          c.success_rate >= 0.5
                            ? "var(--good)"
                            : c.success_rate > 0
                              ? "var(--warn)"
                              : undefined,
                      }}
                    >
                      {Math.round(c.success_rate * 100)}%
                    </td>
                    <td>{c.average_reward.toFixed(1)}</td>
                    <td>{c.average_sheep_penned.toFixed(1)}</td>
                    <td
                      style={{
                        color: c.timeout_rate > 0.7 ? "var(--danger)" : undefined,
                      }}
                    >
                      {Math.round(c.timeout_rate * 100)}%
                    </td>
                    <td>
                      {c.average_completion_steps != null
                        ? Math.round(c.average_completion_steps)
                        : "—"}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
          <ChartLegend
            entries={[
              { symbol: { kind: "dot", color: "#f4c542" }, label: "★ Best", detail: "best saved model — loaded for inference and Watch tab replay" },
              { symbol: { kind: "dot", color: "var(--good)" }, label: "≥50% success", detail: "meets the promotion threshold — consider advancing to next stage" },
              { symbol: { kind: "dot", color: "var(--warn)" }, label: ">0% success", detail: "agent is learning but not yet at threshold" },
              ...uniqueStages.map((s) => ({ symbol: { kind: "dot" as const, color: stageColor(s) }, label: `Stage ${s} dot`, detail: `dot in Ep column — checkpoint recorded at stage ${s}` })),
            ]}
          />
        </div>
      )}
      </div>

      {isHelpOpen && (
        <div className="insights-help-overlay" onClick={() => setIsHelpOpen(false)}>
          <div className="insights-help-panel" onClick={(e) => e.stopPropagation()}>
            <header className="insights-help-panel__header">
              <div>
                <span className="eyebrow">Interpretive Guide</span>
                <h3>Understanding Training Progress</h3>
              </div>
              <button
                className="insights-help-panel__close"
                onClick={() => setIsHelpOpen(false)}
                aria-label="Close guide"
              >
                &times;
              </button>
            </header>

            <div className="insights-help-panel__body">
              <section className="help-section">
                <h4>Overview</h4>
                <div className="help-card">
                  This panel displays real-time performance telemetry for the Reinforcement Learning (RL) sheepdog agent. As the agent interacts with the environment, it learns optimal movement control policies to herd and pen sheep. Use this tab to audit convergence stability, evaluate pathfinding efficiency, and decide when to promote curriculum stages.
                </div>
              </section>

              <section className="help-section">
                <h4>Quick-Reference Playbook</h4>
                
                <div className="help-card">
                  <div className="help-badge help-badge--success">Ready to Promote</div>
                  <strong>Success Rate &ge; 50%</strong>
                  <p>When the success rate reaches 50% and stabilizes over 5+ checkpoints, the agent has successfully generalized the current stage dynamics. It is safe to promote the training to the next curriculum stage.</p>
                </div>

                <div className="help-card">
                  <div className="help-badge help-badge--info">Hidden Learning</div>
                  <strong>Success Rate = 0%, but Reward is Climbing</strong>
                  <p>Do not stop training! Even if the agent has not penned all sheep (0% success), a rising average reward combined with falling steps/timeouts shows the agent is learning to group and steer sheep. A breakthrough is typically imminent.</p>
                </div>

                <div className="help-card">
                  <div className="help-badge help-badge--danger">Cliff State (Stuck)</div>
                  <strong>Success Rate stays at 0% for 8+ checkpoints</strong>
                  <p>The agent is struggling to find the sparse success reward. <strong>Action:</strong> Promote to a simpler curriculum stage, or toggle <em>Instinct Rewards</em> in the Config tab to provide dense proxy shape-rewards.</p>
                </div>

                <div className="help-card">
                  <div className="help-badge help-badge--warning">Policy Instability</div>
                  <strong>Success Rate oscillates or drops sharply</strong>
                  <p>Standard PPO oscillation. The best performing model checkpoint is automatically preserved. Consider reducing the <span className="help-term">entropy_coef</span> in the Config tab to stabilize training convergence.</p>
                </div>
              </section>

              <section className="help-section">
                <h4>Key Metrics & Glossary</h4>
                
                <div className="help-card">
                  <strong>Success Rate</strong>
                  <p>The fraction of evaluation episodes where all sheep are successfully steered into the pen within the step limit. Primary metric for upper management review.</p>
                </div>

                <div className="help-card">
                  <strong>Average Reward</strong>
                  <p>Cumulative step-by-step reinforcement feedback. Composed of positive rewards (sheep proximity, herding alignment) and negative penalties (timeouts, collisions).</p>
                </div>

                <div className="help-card">
                  <strong>PPO (Proximal Policy Optimization)</strong>
                  <p>The core training algorithm. It uses a clip objective function to bound policy updates, ensuring stable updates and preventing sudden performance crashes.</p>
                </div>

                <div className="help-card">
                  <strong>Actor-Critic Architecture</strong>
                  <p>The network structure. The <span className="help-term">Actor</span> maps observations to actions (movement directions). The <span className="help-term">Critic</span> estimates state values to guide the actor's learning.</p>
                </div>

                <div className="help-card">
                  <strong>Entropy Coefficient</strong>
                  <p>Controls exploration. Higher values prevent premature convergence by encouraging random movements. Lower values encourage exploitation of learned paths.</p>
                </div>

                <div className="help-card">
                  <strong>No-Progress Guard</strong>
                  <p>A safety threshold that aborts episodes early if the dogs are inactive or fail to move sheep, avoiding wasting compute on dead-ends.</p>
                </div>
              </section>

              <section className="help-section">
                <h4>Evaluation Examples</h4>
                
                <div className="help-card">
                  <strong>Example A: Ideal Convergence</strong>
                  <p>At Stage 2, Success Rate rises smoothly to 65% by Episode 4,000. Average steps drop from 600 to 280. Reward rises from -150 to +220. <em>Interpretation:</em> The agent has mastered herding. Promote immediately.</p>
                </div>

                <div className="help-card">
                  <strong>Example B: Dense Reward Exploration</strong>
                  <p>At Stage 3, Success Rate remains at 0% for 3,000 episodes. However, Average Reward rises from -300 to -110, and Timeout Rate falls from 100% to 40%. <em>Interpretation:</em> The agent is herding sheep but runs out of time to pen them. Allow training to continue; success will soon follow.</p>
                </div>
              </section>
            </div>
          </div>
        </div>
      )}
    </section>
  );
}

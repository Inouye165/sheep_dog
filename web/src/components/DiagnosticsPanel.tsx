import { useMemo, useState } from "react";
import type { CheckpointEntry, CheckpointIndex, TrainingStatus } from "../state/types";

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

function stageColor(stage: number | undefined): string {
  return STAGE_COLORS[stage ?? 0] ?? "#9ca3af";
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
            {secDataPoints.map((d) => (
              <circle
                key={`sec-${d.x}`}
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
        {data.map((d) => {
          const cx = toSvgX(d.x);
          const cy = toSvgY(d.y);
          const fill = stageColor(d.stage);
          const isBest = d.x === bestEpisode;
          const r = isBest ? 6 : d.isPrevBest ? 5 : 4;
          const diamond = `${cx},${cy - r} ${cx + r},${cy} ${cx},${cy + r} ${cx - r},${cy}`;
          return (
            <g key={d.x}>
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

type ChartTab = "success" | "reward" | "sheep" | "history";
type ViewWindow = "stage" | "all" | 25 | 50 | 100;

const VIEW_WINDOW_OPTIONS: Array<{ value: ViewWindow; label: string }> = [
  { value: "stage", label: "This stage" },
  { value: 25, label: "Last 25" },
  { value: 50, label: "Last 50" },
  { value: 100, label: "Last 100" },
  { value: "all", label: "All" },
];

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

// ── Plateau / cliff analysis ─────────────────────────────────────────────────

interface PlateauInfo {
  /** plateau = improved then stalled; cliff = never succeeded; spike = found success but regressed */
  kind: "plateau" | "cliff" | "spike";
  window: number;
  bestRate: number;
  /** Highest success rate ever seen across all checkpoints. */
  allTimeBest: number;
  sinceEpisode: number;
}

function detectPlateau(checkpoints: CheckpointEntry[]): PlateauInfo | null {
  if (checkpoints.length < PLATEAU_WINDOW) return null;
  const recent = checkpoints.slice(-PLATEAU_WINDOW);
  const allPrior = checkpoints.slice(0, -PLATEAU_WINDOW);
  const bestPrior = allPrior.length > 0 ? Math.max(...allPrior.map((c) => c.success_rate)) : -Infinity;
  const bestRecent = Math.max(...recent.map((c) => c.success_rate));
  const allTimeBest = Math.max(...checkpoints.map((c) => c.success_rate));

  if (bestRecent <= bestPrior + PLATEAU_MIN_DELTA) {
    // spike-and-drop: agent HAS succeeded before but regressed in the recent window
    const everSucceeded = allTimeBest > CLIFF_THRESHOLD;
    const kind =
      checkpoints.length >= CLIFF_MIN_CHECKPOINTS && bestRecent < CLIFF_THRESHOLD
        ? everSucceeded ? "spike" : "cliff"
        : "plateau";
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
  const checkpoints = useMemo(
    () => checkpointIndex?.checkpoints ?? [],
    [checkpointIndex?.checkpoints],
  );

  const plateauInfo = useMemo(() => detectPlateau(checkpoints), [checkpoints]);

  const [viewWindow, setViewWindow] = useState<ViewWindow>("stage");

  const filteredCheckpoints = useMemo(() => {
    if (viewWindow === "all") return checkpoints;
    if (viewWindow === "stage")
      return checkpoints.filter(
        (c) =>
          (c.reward_config?.instincts?.curriculum_stage ?? 0) === effectiveCurriculumStage,
      );
    return checkpoints.slice(-viewWindow);
  }, [checkpoints, viewWindow, effectiveCurriculumStage]);

  const stages = useMemo(
    () => filteredCheckpoints.map((c) => c.reward_config?.instincts?.curriculum_stage ?? 0),
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
  const [activeChart, setActiveChart] = useState<ChartTab>("success");

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
    <section className="training-card" aria-label="Diagnostics">
      <div className="training-card__header">
        <div>
          <p className="eyebrow">Insights</p>
          <h2>Learning Curve</h2>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
          {isLiveTraining ? <span className="pill pill--live">live</span> : null}
          <span className="pill pill--muted">{checkpoints.length} pts</span>
        </div>
      </div>

      {/* Plateau / cliff / spike alert */}
      {plateauInfo ? (
        <div
          className={`warning-box${plateauInfo.kind === "cliff" ? " warning-box--error" : plateauInfo.kind === "spike" ? " warning-box--warning" : ""}`}
          role="alert"
        >
          {plateauInfo.kind === "cliff" ? (
            <>
              <strong>Cliff detected</strong> — the agent has never succeeded after{" "}
              {checkpoints.length} checkpoints.{" "}
              {trainingStatus?.enable_instinct_rewards === false ? (
                <>Try enabling <em>Instinct Rewards</em> and clearing to restart at Stage 1.</>
              ) : (trainingStatus?.curriculum_stage ?? 1) > 1 ? (
                <>Try <em>Clear Training</em> and restart at Stage 1 — the current stage may be too hard to learn from scratch.</>
              ) : (
                <>The current reward config may not be learnable from scratch. Check <em>time_penalty</em> and <em>entropy_coef</em> in the Config tab.</>
              )}
            </>
          ) : plateauInfo.kind === "spike" ? (
            <>
              <strong>Policy instability</strong> — the agent reached{" "}
              {Math.round(plateauInfo.allTimeBest * 100)}% success but regressed. This is a
              PPO oscillation pattern, not a dead-end. The best checkpoint is preserved and
              available. Keep training — or reduce <em>entropy_coef</em> in the Config tab for
              more stable convergence.
            </>
          ) : (
            <>
              <strong>Plateau detected</strong> — no improvement in the last {plateauInfo.window}{" "}
              checkpoints (best {Math.round(plateauInfo.bestRate * 100)}% since ep{" "}
              {plateauInfo.sinceEpisode}). Training will keep trying, but you may want to promote
              to the next stage or clear and restart.
            </>
          )}
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
        <span className="view-filter__label">Show</span>
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
        {(["success", "reward", "sheep", "history"] as ChartTab[]).map((id) => {
          const labels: Record<ChartTab, string> = {
            success: "Success Rate",
            reward: "Avg Reward",
            sheep: "Sheep Penned",
            history: "History",
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
              ...uniqueStages.map((s) => ({ symbol: { kind: "dot" as const, color: stageColor(s) }, label: `Stage ${s}`, detail: s === 0 ? "no curriculum — base difficulty" : `curriculum stage ${s}` })),
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
              ...uniqueStages.map((s) => ({ symbol: { kind: "dot" as const, color: stageColor(s) }, label: `Stage ${s}`, detail: s === 0 ? "no curriculum — base difficulty" : `curriculum stage ${s}` })),
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
              ...uniqueStages.map((s) => ({ symbol: { kind: "dot" as const, color: stageColor(s) }, label: `Stage ${s}`, detail: s === 0 ? "no curriculum — base difficulty" : `curriculum stage ${s}` })),
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
                  const cStage = c.reward_config?.instincts?.curriculum_stage;
                  return (
                    <tr
                      key={c.checkpoint_episode}
                      className={isBest ? "diag-table__row--best" : undefined}
                    >
                      <td>
                        <span
                        className="diag-table__stage-dot"
                        style={{ background: stageColor(cStage) }}
                      />
                      {isBest ? "★ " : ""}
                      {c.checkpoint_episode}
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
    </section>
  );
}

import React, { useMemo, useState } from "react";
import type { CheckpointEntry, TrainingEpisode, AutoPromoteGateDiagnostics } from "../state/types";
import {
  type CanonicalEpisodeRecord,
  type SmoothingWindow,
  type RollingTrainingPoint,
  type FormalEvalMarker,
  computeRollingTrainingSeries,
  buildFormalEvalMarkers,
  selectWindowSlice,
  getCheckpointStage,
} from "../lib/chartPipeline";

export type { SmoothingWindow };
export type XAxisMode = "stage_ep" | "global_ep" | "timesteps";

interface StackedLearningPanelsProps {
  episodes: CanonicalEpisodeRecord[] | TrainingEpisode[];
  fullCanonicalHistory?: CanonicalEpisodeRecord[];
  checkpoints: CheckpointEntry[];
  curriculumStage: number;
  smoothingWindow?: SmoothingWindow;
  onSmoothingWindowChange?: (window: SmoothingWindow) => void;
  xAxisMode: XAxisMode;
  showRawEpisodes?: boolean;
  showRollingAvg?: boolean;
  showFormalEvals?: boolean;
  showPolicySnapshots?: boolean;
  bestCheckpointEpisode?: number | null;
  promotionThreshold?: number;
  autoPromoteGate?: AutoPromoteGateDiagnostics | null;
}

export function StackedLearningPanels({
  episodes,
  fullCanonicalHistory,
  checkpoints,
  curriculumStage,
  smoothingWindow = 50,
  onSmoothingWindowChange,
  xAxisMode,
  showRawEpisodes = true,
  showRollingAvg = true,
  showFormalEvals = true,
  bestCheckpointEpisode,
  promotionThreshold = 0.5,
}: StackedLearningPanelsProps) {
  const [internalSmoothing, setInternalSmoothing] = useState<SmoothingWindow>(smoothingWindow);
  const activeSmoothing = onSmoothingWindowChange ? smoothingWindow : internalSmoothing;

  const handleSmoothingChange = (w: SmoothingWindow) => {
    if (onSmoothingWindowChange) {
      onSmoothingWindowChange(w);
    } else {
      setInternalSmoothing(w);
    }
  };

  const [hoveredXVal, setHoveredXVal] = useState<number | null>(null);
  const [hoveredEvalMarker, setHoveredEvalMarker] = useState<FormalEvalMarker | null>(null);

  // 1. Calculate rolling training metrics across FULL canonical stage history first
  const fullRollingHistory = useMemo(() => {
    const historyToUse = fullCanonicalHistory && fullCanonicalHistory.length > 0
      ? fullCanonicalHistory
      : (episodes as CanonicalEpisodeRecord[]);
    if (!historyToUse || historyToUse.length === 0) return [];
    return computeRollingTrainingSeries(historyToUse, activeSmoothing);
  }, [fullCanonicalHistory, episodes, activeSmoothing]);

  // 2. Window-sliced rolling episodes
  const visibleRollingEpisodes = useMemo(() => {
    if (fullRollingHistory.length === 0) return [];
    if (episodes.length > 0 && episodes.length < fullRollingHistory.length) {
      return fullRollingHistory.slice(-episodes.length);
    }
    return fullRollingHistory;
  }, [fullRollingHistory, episodes.length]);

  const numEpisodes = visibleRollingEpisodes.length;

  // 3. Build formal evaluation markers strictly mapped to this stage
  const formalEvalMarkers = useMemo(() => {
    const pipelineXMode = xAxisMode === "timesteps" ? "timesteps" : "episode";
    return buildFormalEvalMarkers(
      checkpoints,
      curriculumStage,
      pipelineXMode,
      bestCheckpointEpisode,
      curriculumStage
    );
  }, [checkpoints, curriculumStage, xAxisMode, bestCheckpointEpisode]);

  // 4. Determine canonical X value for any episode
  const getEpX = (ep: RollingTrainingPoint | CanonicalEpisodeRecord): number => {
    if (xAxisMode === "global_ep") return ep.global_environment_episode ?? ep.episode_in_stage;
    if (xAxisMode === "timesteps") return ep.global_timestep ?? ep.global_environment_episode ?? ep.episode_in_stage;
    return ep.episode_in_stage ?? ep.global_environment_episode;
  };

  // 5. Shared canonical X-domain calculation across episodes and formal evals
  const { minX, maxX } = useMemo(() => {
    const allX: number[] = [];
    visibleRollingEpisodes.forEach((ep) => allX.push(getEpX(ep)));
    formalEvalMarkers.forEach((m) => allX.push(m.xVal));

    if (allX.length === 0) return { minX: 0, maxX: 100 };
    const min = Math.min(...allX);
    const max = Math.max(...allX);
    if (min === max) {
      return { minX: Math.max(0, min - 10), maxX: max + 10 };
    }
    return { minX: min, maxX: max };
  }, [visibleRollingEpisodes, formalEvalMarkers, xAxisMode]);

  // 6. Shared SVG layout parameters
  const W = 900;
  const H = 190;
  const PAD = { top: 25, right: 45, bottom: 35, left: 60 };
  const plotW = W - PAD.left - PAD.right;
  const plotH = H - PAD.top - PAD.bottom;

  function toSvgX(xVal: number): number {
    const range = maxX - minX || 1;
    return PAD.left + ((xVal - minX) / range) * plotW;
  }

  function toSvgY(val: number, yMin: number, yMax: number): number {
    const range = yMax - yMin || 1;
    return PAD.top + plotH - ((val - yMin) / range) * plotH;
  }

  // 7. Y-Ranges across panels
  const stepsYRange = useMemo(() => {
    const allVals: number[] = [];
    visibleRollingEpisodes.forEach((e) => {
      allVals.push(e.steps);
      if (e.rollingSuccessfulSteps != null) allVals.push(e.rollingSuccessfulSteps);
    });
    formalEvalMarkers.forEach((m) => {
      if (m.medianSuccessfulSteps != null) allVals.push(m.medianSuccessfulSteps);
      if (m.worstSuccessfulSteps != null) allVals.push(m.worstSuccessfulSteps);
    });

    if (allVals.length === 0) return { min: 0, max: 500 };
    const minV = Math.min(...allVals);
    const maxV = Math.max(...allVals);
    const pad = (maxV - minV) * 0.15 || 25;
    return { min: Math.max(0, Math.floor(minV - pad)), max: Math.ceil(maxV + pad) };
  }, [visibleRollingEpisodes, formalEvalMarkers]);

  const rewardYRange = useMemo(() => {
    const allVals: number[] = [];
    visibleRollingEpisodes.forEach((e) => {
      allVals.push(e.reward);
      allVals.push(e.rollingReward);
    });
    formalEvalMarkers.forEach((m) => {
      if (m.avgReward != null) allVals.push(m.avgReward);
    });

    if (allVals.length === 0) return { min: -100, max: 500 };
    const minV = Math.min(...allVals);
    const maxV = Math.max(...allVals);
    const pad = (maxV - minV) * 0.15 || 50;
    return { min: Math.floor(minV - pad), max: Math.ceil(maxV + pad) };
  }, [visibleRollingEpisodes, formalEvalMarkers]);

  // Active hovered point
  const activeHoveredEpisode = useMemo(() => {
    if (hoveredXVal === null || visibleRollingEpisodes.length === 0) return null;
    let closest: RollingTrainingPoint | null = null;
    let minDiff = Infinity;
    for (const ep of visibleRollingEpisodes) {
      const diff = Math.abs(getEpX(ep) - hoveredXVal);
      if (diff < minDiff) {
        minDiff = diff;
        closest = ep;
      }
    }
    return closest;
  }, [hoveredXVal, visibleRollingEpisodes, xAxisMode]);

  // Format X Axis tick labels
  const formatXLabel = (val: number): string => {
    if (xAxisMode === "timesteps") {
      if (val >= 1_000_000) return `${(val / 1_000_000).toFixed(2)}M`;
      if (val >= 1_000) return `${Math.round(val / 1_000)}k`;
      return Math.round(val).toLocaleString();
    }
    return `Ep ${Math.round(val).toLocaleString()}`;
  };

  const hasZeroRollouts = numEpisodes === 0;
  const hasZeroData = numEpisodes === 0 && formalEvalMarkers.length === 0;

  if (hasZeroData) {
    return (
      <div className="warning-box warning-box--info" style={{ marginTop: "1rem" }}>
        No training episodes or evaluation checkpoints recorded yet for Stage {curriculumStage}. Run training to display the stacked learning curve.
      </div>
    );
  }

  const thresholdPct = Math.round(promotionThreshold * 100);

  return (
    <div className="stacked-learning-panels" style={{ display: "flex", flexDirection: "column", gap: "1.25rem", marginTop: "1rem" }}>
      {/* Informative notice if raw rollouts are unavailable */}
      {hasZeroRollouts && (
        <div
          className="warning-box warning-box--info"
          style={{
            margin: "0 0 0.5rem 0",
            background: "rgba(56, 189, 248, 0.08)",
            border: "1px solid rgba(56, 189, 248, 0.3)",
            color: "#38bdf8",
            fontSize: "0.85rem",
          }}
        >
          ℹ️ <strong>Training rollout telemetry unavailable for this range.</strong> Formal deterministic evaluations ({formalEvalMarkers.length} checkpoints) are shown independently below.
        </div>
      )}

      {/* Synchronized Hover Callout Banner */}
      {hoveredEvalMarker ? (
        <div
          className="stacked-hover-banner"
          style={{
            background: "rgba(15, 23, 42, 0.95)",
            border: "1px solid #facc15",
            borderRadius: "6px",
            padding: "0.6rem 1rem",
            fontSize: "0.82rem",
            display: "flex",
            flexWrap: "wrap",
            gap: "1rem",
            alignItems: "center",
            justifyContent: "space-between",
          }}
        >
          <div>
            <strong style={{ color: "#facc15" }}>Formal {hoveredEvalMarker.totalSeeds}-Seed Evaluation</strong>
            <span style={{ marginLeft: "0.5rem", opacity: 0.85 }}>
              (Checkpoint Ep {hoveredEvalMarker.checkpoint.checkpoint_episode})
            </span>
            <span
              style={{
                marginLeft: "0.75rem",
                padding: "0.15rem 0.5rem",
                borderRadius: "4px",
                fontSize: "0.75rem",
                fontWeight: 700,
                background: hoveredEvalMarker.successRatePct >= thresholdPct ? "rgba(74,222,128,0.25)" : "rgba(248,113,113,0.25)",
                color: hoveredEvalMarker.successRatePct >= thresholdPct ? "#4ade80" : "#f87171",
              }}
            >
              {Math.round(hoveredEvalMarker.successRatePct)}% Success ({hoveredEvalMarker.successCount}/{hoveredEvalMarker.totalSeeds})
            </span>
          </div>
          <div style={{ display: "flex", gap: "1.25rem", flexWrap: "wrap" }}>
            <div>
              <span style={{ color: "#94a3b8" }}>Median Succ Steps: </span>
              <strong style={{ color: "#38bdf8" }}>
                {hoveredEvalMarker.medianSuccessfulSteps != null ? Math.round(hoveredEvalMarker.medianSuccessfulSteps) : "—"}
              </strong>
            </div>
            <div>
              <span style={{ color: "#94a3b8" }}>Mean Succ Steps: </span>
              <strong>{hoveredEvalMarker.meanSuccessfulSteps != null ? Math.round(hoveredEvalMarker.meanSuccessfulSteps) : "—"}</strong>
            </div>
            <div>
              <span style={{ color: "#94a3b8" }}>Worst Succ Steps: </span>
              <strong>{hoveredEvalMarker.worstSuccessfulSteps != null ? Math.round(hoveredEvalMarker.worstSuccessfulSteps) : "—"}</strong>
            </div>
            <div>
              <span style={{ color: "#94a3b8" }}>Failed Seeds: </span>
              <strong style={{ color: hoveredEvalMarker.failedSeeds.length > 0 ? "#f87171" : "#4ade80" }}>
                {hoveredEvalMarker.failedSeeds.length > 0 ? hoveredEvalMarker.failedSeeds.join(", ") : "None (100%)"}
              </strong>
            </div>
          </div>
        </div>
      ) : activeHoveredEpisode ? (
        <div
          className="stacked-hover-banner"
          style={{
            background: "rgba(15, 23, 42, 0.95)",
            border: "1px solid var(--panel-border, #334155)",
            borderRadius: "6px",
            padding: "0.6rem 1rem",
            fontSize: "0.82rem",
            display: "flex",
            flexWrap: "wrap",
            gap: "1rem",
            alignItems: "center",
            justifyContent: "space-between",
          }}
        >
          <div>
            <strong>Stage Episode {activeHoveredEpisode.episode_in_stage ?? activeHoveredEpisode.global_environment_episode}</strong>
            <span style={{ marginLeft: "0.5rem", opacity: 0.85 }}>
              (Global Ep {activeHoveredEpisode.global_environment_episode})
            </span>
            <span
              style={{
                marginLeft: "0.75rem",
                padding: "0.1rem 0.4rem",
                borderRadius: "3px",
                fontSize: "0.75rem",
                fontWeight: 600,
                background: activeHoveredEpisode.success ? "rgba(74,222,128,0.2)" : "rgba(248,113,113,0.2)",
                color: activeHoveredEpisode.success ? "#4ade80" : "#f87171",
              }}
            >
              {activeHoveredEpisode.result} ({activeHoveredEpisode.steps} steps)
            </span>
          </div>

          <div style={{ display: "flex", gap: "1.25rem", flexWrap: "wrap" }}>
            <div>
              <span style={{ color: "#94a3b8" }}>Rolling {activeSmoothing} Success: </span>
              <strong style={{ color: activeHoveredEpisode.rollingSuccessRate >= thresholdPct ? "#4ade80" : "#f87171" }}>
                {activeHoveredEpisode.rollingSuccessRate.toFixed(1)}%
              </strong>
            </div>
            <div>
              <span style={{ color: "#94a3b8" }}>Rolling Succ Steps: </span>
              <strong style={{ color: "#38bdf8" }}>
                {activeHoveredEpisode.rollingSuccessfulSteps != null ? Math.round(activeHoveredEpisode.rollingSuccessfulSteps) : "—"}
              </strong>
            </div>
            <div>
              <span style={{ color: "#94a3b8" }}>Rolling Reward: </span>
              <strong style={{ color: "#facc15" }}>{activeHoveredEpisode.rollingReward.toFixed(1)}</strong>
            </div>
          </div>
        </div>
      ) : null}

      {/* ────────────────────────────────────────────────────────────────────── */}
      {/* PANEL 1: SUCCESS / RELIABILITY (%) */}
      {/* ────────────────────────────────────────────────────────────────────── */}
      <div className="panel-card" style={{ background: "rgba(15, 23, 42, 0.6)", borderRadius: "8px", border: "1px solid rgba(255,255,255,0.08)", padding: "0.85rem" }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "0.5rem", flexWrap: "wrap", gap: "0.5rem" }}>
          <div>
            <strong style={{ color: "#e2e8f0", fontSize: "0.9rem" }}>1. SUCCESS / RELIABILITY (%)</strong>
            <span style={{ fontSize: "0.75rem", color: "#94a3b8", marginLeft: "0.5rem" }}>
              (Actual Rollouts vs Formal 10-Seed Evaluations)
            </span>
          </div>

          {/* Rolling Window Selector: 25 | 50 | 100 */}
          <div style={{ display: "flex", alignItems: "center", gap: "0.75rem" }}>
            <div className="chart-tabs" style={{ margin: 0 }}>
              <span style={{ fontSize: "0.75rem", color: "#94a3b8", marginRight: "0.3rem" }}>Rolling:</span>
              {([25, 50, 100] as SmoothingWindow[]).map((w) => (
                <button
                  key={`roll-win-${w}`}
                  type="button"
                  className={`chart-tab${activeSmoothing === w ? " chart-tab--active" : ""}`}
                  style={{ padding: "0.2rem 0.5rem", fontSize: "0.75rem" }}
                  onClick={() => handleSmoothingChange(w)}
                >
                  {w}
                </button>
              ))}
            </div>

            <div style={{ display: "flex", gap: "0.75rem", fontSize: "0.75rem", color: "#94a3b8", flexWrap: "wrap" }}>
              {!hasZeroRollouts && showRawEpisodes && <span><span style={{ color: "rgba(148, 163, 184, 0.6)" }}>●</span> Raw Rollout (0/100%)</span>}
              {!hasZeroRollouts && showRollingAvg && <span><span style={{ color: "#4ade80", fontWeight: "bold" }}>━</span> Rolling {activeSmoothing} Training Avg</span>}
              {showFormalEvals && <span><span style={{ color: "#facc15" }}>◆</span> Formal 10-Seed Eval</span>}
              <span style={{ color: "rgba(74,222,128,0.7)" }}>- - {thresholdPct}% Threshold</span>
            </div>
          </div>
        </div>

        <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", overflow: "visible" }}>
          {/* Y-axis grid & labels */}
          {[0, 25, 50, 75, 100].map((v) => {
            const sy = toSvgY(v, 0, 100);
            return (
              <g key={`p1-grid-${v}`}>
                <line x1={PAD.left} y1={sy} x2={PAD.left + plotW} y2={sy} stroke="rgba(255,255,255,0.07)" strokeDasharray="3 3" />
                <text x={PAD.left - 8} y={sy + 3.5} textAnchor="end" fontSize={11} fill="#94a3b8">{v}%</text>
              </g>
            );
          })}

          {/* Promotion threshold reference line */}
          <line
            x1={PAD.left}
            y1={toSvgY(thresholdPct, 0, 100)}
            x2={PAD.left + plotW}
            y2={toSvgY(thresholdPct, 0, 100)}
            stroke="rgba(74,222,128,0.6)"
            strokeWidth={1.5}
            strokeDasharray="4 4"
          />

          {/* Faint Raw Rollout Dots (0% or 100%) */}
          {!hasZeroRollouts && showRawEpisodes &&
            visibleRollingEpisodes.map((ep) => {
              const cx = toSvgX(getEpX(ep));
              const cy = toSvgY(ep.success ? 100 : 0, 0, 100);
              return (
                <circle
                  key={`p1-raw-${ep.id}`}
                  cx={cx}
                  cy={cy}
                  r={2.2}
                  fill={ep.success ? "rgba(74, 222, 128, 0.4)" : "rgba(248, 113, 113, 0.25)"}
                />
              );
            })}

          {/* Prominent Rolling Training Polyline */}
          {!hasZeroRollouts && showRollingAvg && visibleRollingEpisodes.length >= 2 && (
            <polyline
              points={visibleRollingEpisodes
                .map((ep) => `${toSvgX(getEpX(ep)).toFixed(1)},${toSvgY(ep.rollingSuccessRate, 0, 100).toFixed(1)}`)
                .join(" ")}
              fill="none"
              stroke="#4ade80"
              strokeWidth={2.8}
              strokeLinejoin="round"
            />
          )}

          {/* Thin connecting line between formal evaluation markers */}
          {showFormalEvals && formalEvalMarkers.length >= 2 && (
            <polyline
              points={formalEvalMarkers
                .map((m) => `${toSvgX(m.xVal).toFixed(1)},${toSvgY(m.successRatePct, 0, 100).toFixed(1)}`)
                .join(" ")}
              fill="none"
              stroke="rgba(250, 204, 21, 0.4)"
              strokeWidth={1.5}
              strokeDasharray="2 2"
            />
          )}

          {/* Formal Deterministic Evaluation Diamonds */}
          {showFormalEvals &&
            formalEvalMarkers.map((m, i) => {
              const cx = toSvgX(m.xVal);
              const cy = toSvgY(m.successRatePct, 0, 100);
              const r = 6.5;
              const diamond = `${cx},${cy - r} ${cx + r},${cy} ${cx},${cy + r} ${cx - r},${cy}`;
              const isHovered = hoveredEvalMarker?.checkpoint.checkpoint_episode === m.checkpoint.checkpoint_episode;

              return (
                <g
                  key={`p1-eval-${i}`}
                  style={{ cursor: "pointer" }}
                  onMouseEnter={() => {
                    setHoveredEvalMarker(m);
                    setHoveredXVal(m.xVal);
                  }}
                  onMouseLeave={() => {
                    setHoveredEvalMarker(null);
                    setHoveredXVal(null);
                  }}
                >
                  <polygon
                    points={diamond}
                    fill={isHovered ? "#ffffff" : "#facc15"}
                    stroke="#0f172a"
                    strokeWidth={1.8}
                  />
                  <text
                    x={cx}
                    y={cy - 9}
                    textAnchor="middle"
                    fontSize={10}
                    fontWeight="bold"
                    fill="#facc15"
                  >
                    {Math.round(m.successRatePct)}%
                  </text>
                </g>
              );
            })}

          {/* Synchronized Hover Crosshair */}
          {hoveredXVal !== null && (
            <line
              x1={toSvgX(hoveredXVal)}
              y1={PAD.top}
              x2={toSvgX(hoveredXVal)}
              y2={PAD.top + plotH}
              stroke="#38bdf8"
              strokeWidth={1.5}
              strokeDasharray="3 3"
              style={{ pointerEvents: "none" }}
            />
          )}

          {/* Interactive Mouse Hover Overlay */}
          {!hasZeroRollouts &&
            visibleRollingEpisodes.map((ep) => {
              const cx = toSvgX(getEpX(ep));
              const sliceWidth = Math.max(2, plotW / (visibleRollingEpisodes.length || 1));
              return (
                <rect
                  key={`p1-overlay-${ep.id}`}
                  x={cx - sliceWidth / 2}
                  y={PAD.top}
                  width={sliceWidth}
                  height={plotH}
                  fill="transparent"
                  onMouseEnter={() => setHoveredXVal(getEpX(ep))}
                  onMouseLeave={() => setHoveredXVal(null)}
                  style={{ cursor: "pointer" }}
                />
              );
            })}

          {/* X-axis labels */}
          {[minX, Math.round((minX + maxX) / 2), maxX].map((xVal, idx) => (
            <text
              key={`p1-x-${idx}`}
              x={toSvgX(xVal)}
              y={H - 8}
              textAnchor="middle"
              fontSize={11}
              fill="#94a3b8"
            >
              {formatXLabel(xVal)}
            </text>
          ))}
        </svg>
      </div>

      {/* ────────────────────────────────────────────────────────────────────── */}
      {/* PANEL 2: SUCCESSFUL COMPLETION EFFICIENCY (STEPS — LOWER IS BETTER) */}
      {/* ────────────────────────────────────────────────────────────────────── */}
      <div className="panel-card" style={{ background: "rgba(15, 23, 42, 0.6)", borderRadius: "8px", border: "1px solid rgba(255,255,255,0.08)", padding: "0.85rem" }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "0.5rem", flexWrap: "wrap", gap: "0.5rem" }}>
          <div>
            <strong style={{ color: "#e2e8f0", fontSize: "0.9rem" }}>2. SUCCESSFUL COMPLETION EFFICIENCY (STEPS)</strong>
            <span style={{ fontSize: "0.75rem", color: "#38bdf8", marginLeft: "0.5rem", fontWeight: 600 }}>
              ↓ Lower is Better
            </span>
          </div>

          <div style={{ display: "flex", gap: "1rem", fontSize: "0.75rem", color: "#94a3b8", flexWrap: "wrap" }}>
            {showFormalEvals && <span><span style={{ color: "#38bdf8" }}>◆</span> Formal Median Successful Steps</span>}
            {!hasZeroRollouts && showRollingAvg && <span><span style={{ color: "#f59e0b" }}>━</span> Rolling {activeSmoothing} Successful Rollout Steps</span>}
          </div>
        </div>

        <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", overflow: "visible" }}>
          {/* Y-axis grid & labels */}
          {[stepsYRange.min, Math.round((stepsYRange.min + stepsYRange.max) / 2), stepsYRange.max].map((v) => {
            const sy = toSvgY(v, stepsYRange.min, stepsYRange.max);
            return (
              <g key={`p2-grid-${v}`}>
                <line x1={PAD.left} y1={sy} x2={PAD.left + plotW} y2={sy} stroke="rgba(255,255,255,0.07)" strokeDasharray="3 3" />
                <text x={PAD.left - 8} y={sy + 3.5} textAnchor="end" fontSize={11} fill="#94a3b8">{v}</text>
              </g>
            );
          })}

          {/* Rolling Successful Training Steps Polyline */}
          {!hasZeroRollouts && showRollingAvg && (
            <polyline
              points={visibleRollingEpisodes
                .filter((ep) => ep.rollingSuccessfulSteps !== null)
                .map((ep) => `${toSvgX(getEpX(ep)).toFixed(1)},${toSvgY(ep.rollingSuccessfulSteps!, stepsYRange.min, stepsYRange.max).toFixed(1)}`)
                .join(" ")}
              fill="none"
              stroke="#f59e0b"
              strokeWidth={2.2}
              strokeLinejoin="round"
            />
          )}

          {/* Thin connecting line between formal evaluation efficiency diamonds */}
          {showFormalEvals && formalEvalMarkers.filter((m) => m.medianSuccessfulSteps != null).length >= 2 && (
            <polyline
              points={formalEvalMarkers
                .filter((m) => m.medianSuccessfulSteps != null)
                .map((m) => `${toSvgX(m.xVal).toFixed(1)},${toSvgY(m.medianSuccessfulSteps!, stepsYRange.min, stepsYRange.max).toFixed(1)}`)
                .join(" ")}
              fill="none"
              stroke="rgba(56, 189, 248, 0.4)"
              strokeWidth={1.5}
              strokeDasharray="2 2"
            />
          )}

          {/* Formal Median Successful Steps Diamonds */}
          {showFormalEvals &&
            formalEvalMarkers.map((m, i) => {
              if (m.medianSuccessfulSteps === null) return null;
              const cx = toSvgX(m.xVal);
              const cy = toSvgY(m.medianSuccessfulSteps, stepsYRange.min, stepsYRange.max);
              const r = 6.5;
              const diamond = `${cx},${cy - r} ${cx + r},${cy} ${cx},${cy + r} ${cx - r},${cy}`;
              const isHovered = hoveredEvalMarker?.checkpoint.checkpoint_episode === m.checkpoint.checkpoint_episode;

              return (
                <g
                  key={`p2-eval-${i}`}
                  style={{ cursor: "pointer" }}
                  onMouseEnter={() => {
                    setHoveredEvalMarker(m);
                    setHoveredXVal(m.xVal);
                  }}
                  onMouseLeave={() => {
                    setHoveredEvalMarker(null);
                    setHoveredXVal(null);
                  }}
                >
                  <polygon
                    points={diamond}
                    fill={isHovered ? "#ffffff" : "#38bdf8"}
                    stroke="#0f172a"
                    strokeWidth={1.8}
                  />
                  <text
                    x={cx}
                    y={cy - 9}
                    textAnchor="middle"
                    fontSize={10}
                    fontWeight="bold"
                    fill="#38bdf8"
                  >
                    {Math.round(m.medianSuccessfulSteps)}
                  </text>
                </g>
              );
            })}

          {/* Synchronized Hover Crosshair */}
          {hoveredXVal !== null && (
            <line
              x1={toSvgX(hoveredXVal)}
              y1={PAD.top}
              x2={toSvgX(hoveredXVal)}
              y2={PAD.top + plotH}
              stroke="#38bdf8"
              strokeWidth={1.5}
              strokeDasharray="3 3"
              style={{ pointerEvents: "none" }}
            />
          )}

          {/* Interactive Mouse Hover Overlay */}
          {!hasZeroRollouts &&
            visibleRollingEpisodes.map((ep) => {
              const cx = toSvgX(getEpX(ep));
              const sliceWidth = Math.max(2, plotW / (visibleRollingEpisodes.length || 1));
              return (
                <rect
                  key={`p2-overlay-${ep.id}`}
                  x={cx - sliceWidth / 2}
                  y={PAD.top}
                  width={sliceWidth}
                  height={plotH}
                  fill="transparent"
                  onMouseEnter={() => setHoveredXVal(getEpX(ep))}
                  onMouseLeave={() => setHoveredXVal(null)}
                  style={{ cursor: "pointer" }}
                />
              );
            })}

          {/* X-axis labels */}
          {[minX, Math.round((minX + maxX) / 2), maxX].map((xVal, idx) => (
            <text
              key={`p2-x-${idx}`}
              x={toSvgX(xVal)}
              y={H - 8}
              textAnchor="middle"
              fontSize={11}
              fill="#94a3b8"
            >
              {formatXLabel(xVal)}
            </text>
          ))}
        </svg>
      </div>

      {/* ────────────────────────────────────────────────────────────────────── */}
      {/* PANEL 3: TOTAL REWARD */}
      {/* ────────────────────────────────────────────────────────────────────── */}
      <div className="panel-card" style={{ background: "rgba(15, 23, 42, 0.6)", borderRadius: "8px", border: "1px solid rgba(255,255,255,0.08)", padding: "0.85rem" }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "0.5rem", flexWrap: "wrap", gap: "0.5rem" }}>
          <div>
            <strong style={{ color: "#e2e8f0", fontSize: "0.9rem" }}>3. TOTAL REWARD</strong>
            <span style={{ fontSize: "0.75rem", color: "#94a3b8", marginLeft: "0.5rem" }}>
              (Policy Optimization Objective)
            </span>
          </div>

          <div style={{ display: "flex", gap: "1rem", fontSize: "0.75rem", color: "#94a3b8", flexWrap: "wrap" }}>
            {!hasZeroRollouts && <span><span style={{ color: "rgba(250, 204, 21, 0.6)" }}>━</span> Rolling {activeSmoothing} Avg Reward</span>}
            {showFormalEvals && <span><span style={{ color: "#facc15" }}>◆</span> Formal Benchmark Reward</span>}
          </div>
        </div>

        <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", overflow: "visible" }}>
          {/* Y-axis grid & labels */}
          {[rewardYRange.min, Math.round((rewardYRange.min + rewardYRange.max) / 2), rewardYRange.max].map((v) => {
            const sy = toSvgY(v, rewardYRange.min, rewardYRange.max);
            return (
              <g key={`p3-grid-${v}`}>
                <line x1={PAD.left} y1={sy} x2={PAD.left + plotW} y2={sy} stroke="rgba(255,255,255,0.07)" strokeDasharray="3 3" />
                <text x={PAD.left - 8} y={sy + 3.5} textAnchor="end" fontSize={11} fill="#94a3b8">{v}</text>
              </g>
            );
          })}

          {/* Rolling Reward Polyline */}
          {!hasZeroRollouts && (
            <polyline
              points={visibleRollingEpisodes
                .map((ep) => `${toSvgX(getEpX(ep)).toFixed(1)},${toSvgY(ep.rollingReward, rewardYRange.min, rewardYRange.max).toFixed(1)}`)
                .join(" ")}
              fill="none"
              stroke="#facc15"
              strokeWidth={2.4}
              strokeLinejoin="round"
            />
          )}

          {/* Formal Benchmark Reward Diamonds */}
          {showFormalEvals &&
            formalEvalMarkers.map((m, i) => {
              if (m.avgReward === null) return null;
              const cx = toSvgX(m.xVal);
              const cy = toSvgY(m.avgReward, rewardYRange.min, rewardYRange.max);
              const r = 6;
              const diamond = `${cx},${cy - r} ${cx + r},${cy} ${cx},${cy + r} ${cx - r},${cy}`;
              const isHovered = hoveredEvalMarker?.checkpoint.checkpoint_episode === m.checkpoint.checkpoint_episode;

              return (
                <g
                  key={`p3-eval-${i}`}
                  style={{ cursor: "pointer" }}
                  onMouseEnter={() => {
                    setHoveredEvalMarker(m);
                    setHoveredXVal(m.xVal);
                  }}
                  onMouseLeave={() => {
                    setHoveredEvalMarker(null);
                    setHoveredXVal(null);
                  }}
                >
                  <polygon
                    points={diamond}
                    fill={isHovered ? "#ffffff" : "#facc15"}
                    stroke="#0f172a"
                    strokeWidth={1.8}
                  />
                  <text
                    x={cx}
                    y={cy - 9}
                    textAnchor="middle"
                    fontSize={10}
                    fontWeight="bold"
                    fill="#facc15"
                  >
                    {m.avgReward.toFixed(1)}
                  </text>
                </g>
              );
            })}

          {/* Synchronized Hover Crosshair */}
          {hoveredXVal !== null && (
            <line
              x1={toSvgX(hoveredXVal)}
              y1={PAD.top}
              x2={toSvgX(hoveredXVal)}
              y2={PAD.top + plotH}
              stroke="#38bdf8"
              strokeWidth={1.5}
              strokeDasharray="3 3"
              style={{ pointerEvents: "none" }}
            />
          )}

          {/* Interactive Mouse Hover Overlay */}
          {!hasZeroRollouts &&
            visibleRollingEpisodes.map((ep) => {
              const cx = toSvgX(getEpX(ep));
              const sliceWidth = Math.max(2, plotW / (visibleRollingEpisodes.length || 1));
              return (
                <rect
                  key={`p3-overlay-${ep.id}`}
                  x={cx - sliceWidth / 2}
                  y={PAD.top}
                  width={sliceWidth}
                  height={plotH}
                  fill="transparent"
                  onMouseEnter={() => setHoveredXVal(getEpX(ep))}
                  onMouseLeave={() => setHoveredXVal(null)}
                  style={{ cursor: "pointer" }}
                />
              );
            })}

          {/* X-axis labels */}
          {[minX, Math.round((minX + maxX) / 2), maxX].map((xVal, idx) => (
            <text
              key={`p3-x-${idx}`}
              x={toSvgX(xVal)}
              y={H - 8}
              textAnchor="middle"
              fontSize={11}
              fill="#94a3b8"
            >
              {formatXLabel(xVal)}
            </text>
          ))}
        </svg>
      </div>
    </div>
  );
}

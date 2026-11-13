import React, { useMemo, useState } from "react";
import type { CheckpointEntry, SessionChangeEvent, TrainingEpisode } from "../state/types";
import {
  buildFormalEvalMarkers,
  computeRollingTrainingSeries,
  processCanonicalHistory,
  type FormalEvalMarker,
  type RollingTrainingPoint,
  type SmoothingWindow,
  type StageScope,
} from "../lib/chartPipeline";

export type StudioChartMode = "combined" | "success" | "steps" | "reward" | "stacked";
export type EvalWindowSize = 15 | 25 | 50 | 100 | "all";

export interface StudioLearningChartsProps {
  rawEpisodes: TrainingEpisode[];
  checkpoints: CheckpointEntry[];
  sessionEvents?: SessionChangeEvent[];
  effectiveStage: number;
  activeRunId?: string;
  bestCheckpointEpisode?: number | null;
  selectedCheckpointEpisode?: number | null;
  promotionTargetPct?: number;
  defaultMode?: StudioChartMode;
  onSelectCheckpoint?: (checkpointEpisode: number) => void;
  onInspectInArena?: (checkpointEpisode: number) => void;
}

export const StudioLearningCharts: React.FC<StudioLearningChartsProps> = ({
  rawEpisodes,
  checkpoints,
  sessionEvents = [],
  effectiveStage,
  activeRunId,
  bestCheckpointEpisode,
  selectedCheckpointEpisode,
  promotionTargetPct = 90,
  defaultMode = "combined",
  onSelectCheckpoint,
  onInspectInArena,
}) => {
  const [mode, setMode] = useState<StudioChartMode>(defaultMode);
  const [stageScope, setStageScope] = useState<StageScope>("current");
  // Default to Last 25 evaluations (matching v1) so recent evaluations are crystal clear
  const [evalWindow, setEvalWindow] = useState<EvalWindowSize>(25);
  const [smoothing] = useState<SmoothingWindow>(25);
  const [showRollouts] = useState<boolean>(true);
  const [showSessionCues, setShowSessionCues] = useState<boolean>(true);
  const [hoveredEvalIdx, setHoveredEvalIdx] = useState<number | null>(null);

  // Build sorted, deduplicated formal evaluation markers for the selected scope
  const allFormalEvals = useMemo(() => {
    const rawMarkers = buildFormalEvalMarkers(
      checkpoints,
      stageScope,
      "episode",
      bestCheckpointEpisode,
      effectiveStage
    );
    // Deduplicate by stage + checkpoint_episode in ascending episode order
    const byKey = new Map<string, FormalEvalMarker>();
    for (const m of rawMarkers) {
      byKey.set(`${m.stage}:${m.checkpoint.checkpoint_episode}`, m);
    }
    return Array.from(byKey.values()).sort(
      (a, b) => a.checkpoint.checkpoint_episode - b.checkpoint.checkpoint_episode
    );
  }, [checkpoints, stageScope, bestCheckpointEpisode, effectiveStage]);

  // Slice formal evaluations to the user-selected window (e.g. Last 25)
  const visibleEvals = useMemo(() => {
    if (evalWindow === "all") return allFormalEvals;
    return allFormalEvals.slice(-evalWindow);
  }, [allFormalEvals, evalWindow]);

  // Downsample visibleEvals only if viewing "all" and there are > 180 points
  const plottedEvals = useMemo(() => {
    if (visibleEvals.length <= 180) return visibleEvals;
    const step = Math.ceil(visibleEvals.length / 180);
    const sampled: FormalEvalMarker[] = [];
    for (let i = 0; i < visibleEvals.length; i += step) {
      sampled.push(visibleEvals[i]);
    }
    const last = visibleEvals[visibleEvals.length - 1];
    if (
      last &&
      sampled[sampled.length - 1]?.checkpoint.checkpoint_episode !==
        last.checkpoint.checkpoint_episode
    ) {
      sampled.push(last);
    }
    return sampled;
  }, [visibleEvals]);

  // Process raw training rollout episodes if available
  const canonicalHistory = useMemo(
    () => processCanonicalHistory(rawEpisodes, stageScope, activeRunId, effectiveStage),
    [rawEpisodes, stageScope, activeRunId, effectiveStage]
  );

  const fullRollingSeries = useMemo(
    () => computeRollingTrainingSeries(canonicalHistory, smoothing),
    [canonicalHistory, smoothing]
  );

  // Compute X-axis episode bounds from visibleEvals (primary) or rollout series (fallback)
  const xBounds = useMemo(() => {
    if (plottedEvals.length > 0) {
      const minEp = plottedEvals[0].checkpoint.checkpoint_episode;
      const maxEp = plottedEvals[plottedEvals.length - 1].checkpoint.checkpoint_episode;
      return { min: minEp, max: Math.max(maxEp, minEp + 5) };
    }
    if (fullRollingSeries.length > 0) {
      const sliced =
        evalWindow === "all" ? fullRollingSeries : fullRollingSeries.slice(-evalWindow * 10);
      const minEp = sliced[0].global_environment_episode;
      const maxEp = sliced[sliced.length - 1].global_environment_episode;
      return { min: minEp, max: Math.max(maxEp, minEp + 10) };
    }
    return { min: 0, max: 100 };
  }, [plottedEvals, fullRollingSeries, evalWindow]);

  // Filter rollout series to the visible X window
  const displayRollouts = useMemo(() => {
    if (!showRollouts || fullRollingSeries.length === 0) return [];
    const inRange = fullRollingSeries.filter(
      (p) =>
        p.global_environment_episode >= xBounds.min &&
        p.global_environment_episode <= xBounds.max
    );
    if (inRange.length <= 250) return inRange;
    const step = Math.ceil(inRange.length / 250);
    const sampled: RollingTrainingPoint[] = [];
    for (let i = 0; i < inRange.length; i += step) {
      sampled.push(inRange[i]);
    }
    return sampled;
  }, [showRollouts, fullRollingSeries, xBounds]);

  // Completion steps bounds (inverted in Dual-Axis mode just like v1 so UP = FASTER)
  const stepsBounds = useMemo(() => {
    const vals = [
      ...plottedEvals
        .map((m) => m.medianSuccessfulSteps ?? m.checkpoint.average_completion_steps)
        .filter((v): v is number => typeof v === "number" && v > 0),
      ...displayRollouts
        .map((p) => p.rollingSuccessfulSteps ?? p.rollingAllSteps)
        .filter((v): v is number => typeof v === "number" && v > 0),
    ];
    if (vals.length === 0) return { min: 200, max: 1600 };
    const minV = Math.min(...vals);
    const maxV = Math.max(...vals);
    const pad = Math.max(40, (maxV - minV) * 0.14);
    return {
      min: Math.max(50, Math.floor(minV - pad)),
      max: Math.ceil(maxV + pad),
    };
  }, [plottedEvals, displayRollouts]);

  // Reward bounds
  const rewardBounds = useMemo(() => {
    const rVals = [
      ...plottedEvals
        .map((m) => m.avgReward ?? m.checkpoint.average_reward)
        .filter((v): v is number => typeof v === "number" && !Number.isNaN(v)),
      ...displayRollouts.map((p) => p.rollingReward),
    ];
    if (rVals.length === 0) return { min: -500, max: 200 };
    const min = Math.min(...rVals);
    const max = Math.max(...rVals);
    const pad = Math.max(15, (max - min) * 0.14);
    return { min: Math.floor(min - pad), max: Math.ceil(max + pad) };
  }, [plottedEvals, displayRollouts]);

  // Map session events (parameter / architecture changes) into the visible X window
  const visibleSessionEvents = useMemo(() => {
    if (!showSessionCues || !sessionEvents.length || plottedEvals.length === 0) return [];
    const results: Array<{ event: SessionChangeEvent; ep: number; label: string; color: string }> =
      [];
    for (const evt of sessionEvents) {
      const anyEvt = evt as any;
      const matchCp = checkpoints.find(
        (c) =>
          (c.session_id && c.session_id === evt.session_id) ||
          (anyEvt.checkpoint_id && c.checkpoint_id === anyEvt.checkpoint_id)
      );
      const ep = matchCp?.checkpoint_episode ?? evt.checkpoint_episode ?? null;
      if (ep != null && ep >= xBounds.min && ep <= xBounds.max) {
        const isArch =
          Boolean(anyEvt.architecture_changed) ||
          (anyEvt.diffs ?? []).some((d: any) => d.category === "architecture");
        const isParam =
          Boolean(anyEvt.hyperparameters_changed) ||
          (anyEvt.diffs ?? []).some((d: any) => d.category !== "architecture");
        if (!isArch && !isParam) continue;
        results.push({
          event: evt,
          ep,
          label: isArch ? "🧠 Arch" : "⚙ Params",
          color: isArch ? "#c084fc" : "#f59e0b",
        });
      }
    }
    return results;
  }, [showSessionCues, sessionEvents, checkpoints, plottedEvals.length, xBounds]);

  // Stage-level summary telemetry (matching v1's bottom insights bar)
  const stageStats = useMemo(() => {
    if (allFormalEvals.length === 0) {
      return {
        allTimeAvgPct: 0,
        recentRollingPct: 0,
        peakPct: 0,
        latestEval: null as FormalEvalMarker | null,
        prevEval: null as FormalEvalMarker | null,
      };
    }
    const allTimeSum = allFormalEvals.reduce((acc, m) => acc + m.successRatePct, 0);
    const allTimeAvgPct = allTimeSum / allFormalEvals.length;

    const recentSlice = allFormalEvals.slice(-5);
    const recentSum = recentSlice.reduce((acc, m) => acc + m.successRatePct, 0);
    const recentRollingPct = recentSum / recentSlice.length;

    const peakPct = Math.max(...allFormalEvals.map((m) => m.successRatePct));
    const latestEval = allFormalEvals[allFormalEvals.length - 1] ?? null;
    const prevEval = allFormalEvals.length >= 2 ? allFormalEvals[allFormalEvals.length - 2] : null;

    return {
      allTimeAvgPct,
      recentRollingPct,
      peakPct,
      latestEval,
      prevEval,
    };
  }, [allFormalEvals]);

  // Recent 10 evaluations in reverse chronological order (newest first) for the quick strip
  const recentTenEvals = useMemo(() => {
    return [...allFormalEvals].slice(-10).reverse();
  }, [allFormalEvals]);

  const activeHoveredEval =
    hoveredEvalIdx !== null && plottedEvals[hoveredEvalIdx]
      ? plottedEvals[hoveredEvalIdx]
      : stageStats.latestEval;

  const W = 780;
  const H = mode === "stacked" ? 165 : 220;
  const padL = 44;
  const padR = mode === "combined" ? 50 : 22;
  const padT = 22;
  const padB = 26;
  const plotW = W - padL - padR;
  const plotH = H - padT - padB;

  // When viewing a discrete window of evaluations (<= 100), space points evenly across X (sequential index)
  // just like v1's clean evaluation curve so clustered episodes don't overlap!
  const useSequentialEvalX = plottedEvals.length > 1 && evalWindow !== "all";

  const toXByEpisode = (epVal: number) => {
    const span = Math.max(1, xBounds.max - xBounds.min);
    return padL + Math.max(0, Math.min(1, (epVal - xBounds.min) / span)) * plotW;
  };

  const toXEval = (idx: number, epVal: number) => {
    if (useSequentialEvalX && plottedEvals.length > 1) {
      return padL + (idx / (plottedEvals.length - 1)) * plotW;
    }
    return toXByEpisode(epVal);
  };

  const toYSuccess = (pct: number) => {
    return padT + (1 - Math.max(0, Math.min(100, pct)) / 100) * plotH;
  };

  // Inverted steps axis (fewer steps = higher Y on chart, matching v1!)
  const toYStepsInverted = (steps: number) => {
    const span = Math.max(1, stepsBounds.max - stepsBounds.min);
    const clamped = Math.max(stepsBounds.min, Math.min(stepsBounds.max, steps));
    // min steps -> top (padT), max steps -> bottom (padT + plotH)
    return padT + ((clamped - stepsBounds.min) / span) * plotH;
  };

  const toYReward = (rew: number) => {
    const span = Math.max(1, rewardBounds.max - rewardBounds.min);
    return padT + (1 - Math.max(0, Math.min(1, (rew - rewardBounds.min) / span))) * plotH;
  };

  const buildPath = (pts: Array<{ x: number; y: number }>) => {
    if (pts.length === 0) return "";
    return pts
      .map((p, i) => `${i === 0 ? "M" : "L"}${p.x.toFixed(1)},${p.y.toFixed(1)}`)
      .join(" ");
  };

  const buildArea = (pts: Array<{ x: number; y: number }>, baselineY: number) => {
    if (pts.length < 2) return "";
    const line = buildPath(pts);
    const first = pts[0];
    const last = pts[pts.length - 1];
    return `${line} L${last.x.toFixed(1)},${baselineY.toFixed(1)} L${first.x.toFixed(1)},${baselineY.toFixed(1)} Z`;
  };

  const renderSingleSvgChart = (
    chartType: "combined" | "success" | "steps" | "reward",
    title?: string
  ) => {
    const evalSuccessPts = plottedEvals.map((m, idx) => ({
      x: toXEval(idx, m.checkpoint.checkpoint_episode),
      y: toYSuccess(m.successRatePct),
      marker: m,
      idx,
    }));

    const evalStepsPts = plottedEvals
      .map((m, idx) => {
        const steps = m.medianSuccessfulSteps ?? m.checkpoint.average_completion_steps;
        if (steps == null || steps <= 0) return null;
        return {
          x: toXEval(idx, m.checkpoint.checkpoint_episode),
          y: toYStepsInverted(steps),
          steps: Math.round(steps),
          marker: m,
          idx,
        };
      })
      .filter((v): v is NonNullable<typeof v> => v !== null);

    const evalRewardPts = plottedEvals.map((m, idx) => ({
      x: toXEval(idx, m.checkpoint.checkpoint_episode),
      y: toYReward(m.avgReward ?? m.checkpoint.average_reward ?? 0),
      marker: m,
      idx,
    }));

    const rolloutSuccessPts = displayRollouts.map((p) => ({
      x: toXByEpisode(p.global_environment_episode),
      y: toYSuccess(p.rollingSuccessRate),
    }));

    const showPointLabels = plottedEvals.length > 0 && plottedEvals.length <= 30;
    const dotRadius = plottedEvals.length <= 30 ? 4.2 : plottedEvals.length <= 60 ? 3.2 : 2.2;

    return (
      <div className="sd-chart-canvas-wrap" style={{ minHeight: mode === "stacked" ? 170 : 215 }}>
        {title && (
          <div
            style={{
              fontSize: "0.72rem",
              fontWeight: 700,
              color: "var(--sd-text-secondary)",
              marginBottom: "0.2rem",
              display: "flex",
              justifyContent: "space-between",
            }}
          >
            <span>{title}</span>
          </div>
        )}
        {plottedEvals.length === 0 && displayRollouts.length === 0 ? (
          <div className="sd-empty-state" style={{ minHeight: 165 }}>
            <div>No evaluation checkpoints or rollout telemetry recorded for Stage {effectiveStage} yet.</div>
          </div>
        ) : (
          <svg
            className="sd-chart-svg"
            viewBox={`0 0 ${W} ${H}`}
            onMouseLeave={() => setHoveredEvalIdx(null)}
            onMouseMove={(e) => {
              if (plottedEvals.length === 0) return;
              const rect = e.currentTarget.getBoundingClientRect();
              const relX = ((e.clientX - rect.left) / rect.width) * W;
              let bestI = 0;
              let bestDist = Infinity;
              for (let i = 0; i < plottedEvals.length; i++) {
                const px = toXEval(i, plottedEvals[i].checkpoint.checkpoint_episode);
                const d = Math.abs(px - relX);
                if (d < bestDist) {
                  bestDist = d;
                  bestI = i;
                }
              }
              setHoveredEvalIdx(bestI);
            }}
          >
            <defs>
              <linearGradient id={`sdEvalSuccessArea-${chartType}`} x1="0" x2="0" y1="0" y2="1">
                <stop offset="0%" stopColor="#34d399" stopOpacity="0.22" />
                <stop offset="100%" stopColor="#34d399" stopOpacity="0.0" />
              </linearGradient>
              <linearGradient id={`sdEvalRewardArea-${chartType}`} x1="0" x2="0" y1="0" y2="1">
                <stop offset="0%" stopColor="#818cf8" stopOpacity="0.24" />
                <stop offset="100%" stopColor="#818cf8" stopOpacity="0.0" />
              </linearGradient>
            </defs>

            {/* Horizontal Grid Lines & Left/Right Axis Labels */}
            {[0, 0.25, 0.5, 0.75, 1].map((frac) => {
              const y = padT + (1 - frac) * plotH;
              let leftLabel = "";
              if (chartType === "combined" || chartType === "success") {
                leftLabel = `${Math.round(frac * 100)}%`;
              } else if (chartType === "steps") {
                // Inverted steps: top (frac=1) is min steps, bottom (frac=0) is max steps
                const sVal = stepsBounds.max - frac * (stepsBounds.max - stepsBounds.min);
                leftLabel = `${Math.round(sVal)}s`;
              } else {
                const val = rewardBounds.min + frac * (rewardBounds.max - rewardBounds.min);
                leftLabel = `${Math.round(val)}`;
              }

              // Right axis for combined mode: inverted completion steps (top = fastest, bottom = slowest)
              const rightStepVal = Math.round(
                stepsBounds.max - frac * (stepsBounds.max - stepsBounds.min)
              );

              return (
                <g key={frac}>
                  <line
                    x1={padL}
                    y1={y}
                    x2={W - padR}
                    y2={y}
                    stroke="rgba(148, 163, 184, 0.1)"
                    strokeWidth="1"
                  />
                  <text
                    x={padL - 7}
                    y={y + 3}
                    textAnchor="end"
                    fill="#94a3b8"
                    fontSize="9.5"
                    fontFamily="var(--sd-font-mono)"
                  >
                    {leftLabel}
                  </text>
                  {chartType === "combined" && (
                    <text
                      x={W - padR + 7}
                      y={y + 3}
                      textAnchor="start"
                      fill="#f59e0b"
                      fontSize="9.5"
                      fontFamily="var(--sd-font-mono)"
                    >
                      {rightStepVal}s
                    </text>
                  )}
                </g>
              );
            })}

            {/* Promotion Target Threshold Line (Success / Combined) */}
            {(chartType === "combined" || chartType === "success") && (
              <g>
                <line
                  x1={padL}
                  y1={toYSuccess(promotionTargetPct)}
                  x2={W - padR}
                  y2={toYSuccess(promotionTargetPct)}
                  stroke="rgba(52, 211, 153, 0.55)"
                  strokeWidth="1.3"
                  strokeDasharray="5 4"
                />
                <text
                  x={W - padR - 4}
                  y={toYSuccess(promotionTargetPct) - 4}
                  textAnchor="end"
                  fill="#34d399"
                  fontSize="9.5"
                  fontWeight="700"
                >
                  {promotionTargetPct}% Target
                </text>
              </g>
            )}

            {/* Session / Parameter Event Vertical Cues */}
            {visibleSessionEvents.map((se, i) => {
              // Find closest plotted eval index for X coordinate
              let evtX = toXByEpisode(se.ep);
              if (useSequentialEvalX && plottedEvals.length > 0) {
                let bestI = 0;
                let bestDist = Infinity;
                plottedEvals.forEach((m, idx) => {
                  const d = Math.abs(m.checkpoint.checkpoint_episode - se.ep);
                  if (d < bestDist) {
                    bestDist = d;
                    bestI = idx;
                  }
                });
                evtX = toXEval(bestI, se.ep);
              }
              return (
                <g key={`se-${se.ep}-${i}`}>
                  <line
                    x1={evtX}
                    y1={padT}
                    x2={evtX}
                    y2={padT + plotH}
                    stroke={se.color}
                    strokeWidth="1.2"
                    strokeDasharray="3 3"
                    opacity="0.75"
                  />
                  <rect
                    x={evtX - 28}
                    y={padT - 15}
                    width={56}
                    height={14}
                    rx={7}
                    fill="#0f172a"
                    stroke={se.color}
                    strokeWidth="1"
                  />
                  <text
                    x={evtX}
                    y={padT - 5}
                    textAnchor="middle"
                    fill={se.color}
                    fontSize="8.5"
                    fontWeight="700"
                  >
                    {se.label}
                  </text>
                </g>
              );
            })}

            {/* Optional Background Rollout Success Curve */}
            {(chartType === "combined" || chartType === "success") &&
              !useSequentialEvalX &&
              rolloutSuccessPts.length > 1 && (
                <path
                  d={buildPath(rolloutSuccessPts)}
                  fill="none"
                  stroke="rgba(56, 189, 248, 0.35)"
                  strokeWidth="1.4"
                />
              )}

            {/* 1. Formal 10-Seed Evaluation Success Line & Area (Emerald Green, matching v1) */}
            {(chartType === "combined" || chartType === "success") &&
              evalSuccessPts.length > 1 && (
                <>
                  <path
                    d={buildArea(evalSuccessPts, padT + plotH)}
                    fill={`url(#sdEvalSuccessArea-${chartType})`}
                  />
                  <path
                    d={buildPath(evalSuccessPts)}
                    fill="none"
                    stroke="#34d399"
                    strokeWidth="2.4"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                </>
              )}

            {/* 2. Completion Steps Dashed Line (Amber/Orange, inverted so UP = FASTER, matching v1) */}
            {(chartType === "combined" || chartType === "steps") && evalStepsPts.length > 1 && (
              <path
                d={buildPath(evalStepsPts)}
                fill="none"
                stroke="#f59e0b"
                strokeWidth="2"
                strokeDasharray="5 3"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            )}

            {/* 3. Reward Line */}
            {chartType === "reward" && evalRewardPts.length > 1 && (
              <>
                <path
                  d={buildArea(evalRewardPts, padT + plotH)}
                  fill={`url(#sdEvalRewardArea-${chartType})`}
                />
                <path
                  d={buildPath(evalRewardPts)}
                  fill="none"
                  stroke="#818cf8"
                  strokeWidth="2.2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
              </>
            )}

            {/* Completion Steps Nodes */}
            {(chartType === "combined" || chartType === "steps") &&
              evalStepsPts.map(({ x, y, steps, marker, idx }) => (
                <g
                  key={`eval-step-${marker.stage}-${marker.checkpoint.checkpoint_episode}-${idx}`}
                  transform={`translate(${x}, ${y})`}
                  style={{ cursor: "pointer" }}
                  onMouseEnter={() => setHoveredEvalIdx(idx)}
                  onClick={(e) => {
                    e.stopPropagation();
                    onSelectCheckpoint?.(marker.checkpoint.checkpoint_episode);
                  }}
                >
                  <circle
                    r={dotRadius * 0.85}
                    fill="#f59e0b"
                    stroke="#090d16"
                    strokeWidth="1.2"
                  />
                  <title>{`Ep ${marker.checkpoint.checkpoint_episode}: ${steps} steps`}</title>
                </g>
              ))}

            {/* Formal 10-Seed Evaluation Success Nodes */}
            {(chartType === "combined" || chartType === "success") &&
              evalSuccessPts.map(({ x, y, marker, idx }) => {
                const isSelected =
                  selectedCheckpointEpisode === marker.checkpoint.checkpoint_episode;
                const isHovered = hoveredEvalIdx === idx;
                const isLatest = idx === evalSuccessPts.length - 1;
                const r =
                  marker.isBest || isSelected || isHovered || isLatest
                    ? dotRadius + 1.6
                    : dotRadius;

                return (
                  <g
                    key={`eval-m-${marker.stage}-${marker.checkpoint.checkpoint_episode}-${idx}`}
                    transform={`translate(${x}, ${y})`}
                    style={{ cursor: "pointer" }}
                    onMouseEnter={() => setHoveredEvalIdx(idx)}
                    onClick={(e) => {
                      e.stopPropagation();
                      onSelectCheckpoint?.(marker.checkpoint.checkpoint_episode);
                    }}
                  >
                    {(marker.isBest || isSelected || isLatest) && (
                      <circle
                        r={r + 3.5}
                        fill="none"
                        stroke={marker.isBest ? "#fbbf24" : isSelected ? "#ffffff" : "#34d399"}
                        strokeWidth="1.4"
                        opacity="0.75"
                      />
                    )}
                    <circle
                      r={r}
                      fill={
                        marker.isBest
                          ? "#fbbf24"
                          : marker.successRatePct >= promotionTargetPct
                          ? "#34d399"
                          : "#6ee7b7"
                      }
                      stroke={isSelected ? "#ffffff" : "#090d16"}
                      strokeWidth={isSelected ? "2" : "1.4"}
                    />
                    {showPointLabels &&
                      (idx === evalSuccessPts.length - 1 ||
                        marker.isBest ||
                        isHovered ||
                        idx % 4 === 0) && (
                        <text
                          x={0}
                          y={-r - 5}
                          textAnchor="middle"
                          fill={
                            marker.successRatePct >= promotionTargetPct ? "#34d399" : "#e2e8f0"
                          }
                          fontSize="8.5"
                          fontWeight="700"
                          fontFamily="var(--sd-font-mono)"
                        >
                          {Math.round(marker.successRatePct)}%
                        </text>
                      )}
                  </g>
                );
              })}

            {/* Hover Crosshair Line */}
            {hoveredEvalIdx !== null && plottedEvals[hoveredEvalIdx] && (
              <line
                x1={toXEval(
                  hoveredEvalIdx,
                  plottedEvals[hoveredEvalIdx].checkpoint.checkpoint_episode
                )}
                y1={padT}
                x2={toXEval(
                  hoveredEvalIdx,
                  plottedEvals[hoveredEvalIdx].checkpoint.checkpoint_episode
                )}
                y2={padT + plotH}
                stroke="rgba(248, 250, 252, 0.35)"
                strokeWidth="1"
                strokeDasharray="2 2"
                pointerEvents="none"
              />
            )}

            {/* X Axis Episode Labels */}
            {[0, 0.25, 0.5, 0.75, 1].map((frac) => {
              const x = padL + frac * plotW;
              let epVal = Math.round(xBounds.min + frac * (xBounds.max - xBounds.min));
              if (useSequentialEvalX && plottedEvals.length > 0) {
                const targetIdx = Math.min(
                  plottedEvals.length - 1,
                  Math.round(frac * (plottedEvals.length - 1))
                );
                epVal = plottedEvals[targetIdx].checkpoint.checkpoint_episode;
              }
              return (
                <text
                  key={`x-${frac}`}
                  x={x}
                  y={H - 6}
                  textAnchor="middle"
                  fill="#94a3b8"
                  fontSize="9.5"
                  fontFamily="var(--sd-font-mono)"
                >
                  Ep {epVal.toLocaleString()}
                </text>
              );
            })}
          </svg>
        )}
      </div>
    );
  };

  return (
    <div className="sd-chart-card">
      {/* Header & Controls */}
      <div className="sd-chart-header">
        <div>
          <div className="sd-panel-title">
            <span className="sd-panel-title__icon">📈</span>
            <span>Learning Curve & Recent 10-Seed Evaluations</span>
            <span className="sd-badge sd-badge--emerald">
              Showing{" "}
              {evalWindow === "all"
                ? `All ${allFormalEvals.length}`
                : `Last ${Math.min(Number(evalWindow), allFormalEvals.length)}`}{" "}
              of {allFormalEvals.length} Evals
            </span>
          </div>
        </div>

        <div className="sd-chart-controls">
          {/* Window Slice (Defaults to Last 25 just like v1!) */}
          <div className="sd-pill-group" role="group" aria-label="Evaluation window">
            {(
              [
                { id: 15, label: "Last 15" },
                { id: 25, label: "Last 25" },
                { id: 50, label: "Last 50" },
                { id: 100, label: "Last 100" },
                { id: "all", label: "All" },
              ] as const
            ).map((w) => (
              <button
                key={String(w.id)}
                type="button"
                className={`sd-pill-btn ${evalWindow === w.id ? "sd-pill-btn--active" : ""}`}
                onClick={() => setEvalWindow(w.id)}
              >
                {w.label}
              </button>
            ))}
          </div>

          {/* Chart Mode Selector */}
          <div className="sd-pill-group" role="group" aria-label="Chart metric mode">
            {(
              [
                { id: "combined", label: "Overview (Dual-Axis)" },
                { id: "success", label: "Success %" },
                { id: "steps", label: "Steps" },
                { id: "reward", label: "Return" },
                { id: "stacked", label: "3-Panel" },
              ] as const
            ).map((item) => (
              <button
                key={item.id}
                type="button"
                className={`sd-pill-btn ${mode === item.id ? "sd-pill-btn--active" : ""}`}
                onClick={() => setMode(item.id)}
              >
                {item.label}
              </button>
            ))}
          </div>

          {/* Stage Scope */}
          <div className="sd-pill-group" role="group" aria-label="Stage scope">
            <button
              type="button"
              className={`sd-pill-btn ${stageScope === "current" ? "sd-pill-btn--active" : ""}`}
              onClick={() => setStageScope("current")}
            >
              Stage {effectiveStage}
            </button>
            <button
              type="button"
              className={`sd-pill-btn ${
                stageScope === "current-journey" ? "sd-pill-btn--active" : ""
              }`}
              onClick={() => setStageScope("current-journey")}
            >
              All Stages
            </button>
          </div>
        </div>
      </div>

      {/* Live Telemetry & Hovered Evaluation Readout Bar */}
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          flexWrap: "wrap",
          gap: "0.45rem 0.9rem",
          padding: "0.35rem 0.65rem",
          background: "rgba(9, 13, 22, 0.6)",
          border: "1px solid var(--sd-border-subtle)",
          borderRadius: "var(--sd-radius-sm)",
          fontSize: "0.72rem",
        }}
      >
        <div className="sd-chart-legend">
          <span className="sd-chart-legend__item">
            <span className="sd-chart-legend__swatch" style={{ background: "#34d399" }} />
            <span>Formal 10-Seed Eval Pass %</span>
          </span>
          {(mode === "combined" || mode === "steps" || mode === "stacked") && (
            <span className="sd-chart-legend__item">
              <span
                className="sd-chart-legend__swatch"
                style={{
                  background: "repeating-linear-gradient(90deg,#f59e0b 0 4px,transparent 4px 7px)",
                }}
              />
              <span>Avg Completion Steps (higher = faster)</span>
            </span>
          )}
          <span className="sd-chart-legend__item">
            <span
              style={{
                display: "inline-block",
                width: 8,
                height: 8,
                borderRadius: "50%",
                background: "#fbbf24",
              }}
            />
            <span>Stage Best</span>
          </span>
          <label
            style={{
              display: "inline-flex",
              alignItems: "center",
              gap: "0.25rem",
              cursor: "pointer",
              color: "var(--sd-text-secondary)",
            }}
          >
            <input
              type="checkbox"
              checked={showSessionCues}
              onChange={(e) => setShowSessionCues(e.target.checked)}
            />
            <span>⚙ Session Cues</span>
          </label>
        </div>

        {activeHoveredEval && (
          <div style={{ display: "flex", alignItems: "center", gap: "0.55rem", flexWrap: "wrap" }}>
            <span
              className={`sd-badge ${
                activeHoveredEval.successRatePct >= promotionTargetPct
                  ? "sd-badge--emerald"
                  : activeHoveredEval.successRatePct >= 60
                  ? "sd-badge--amber"
                  : "sd-badge--rose"
              }`}
            >
              {hoveredEvalIdx !== null ? "Inspecting" : "Latest"} Ep{" "}
              {activeHoveredEval.checkpoint.checkpoint_episode.toLocaleString()}:{" "}
              {Math.round(activeHoveredEval.successRatePct)}% ({activeHoveredEval.successCount}/
              {activeHoveredEval.totalSeeds})
            </span>
            {(activeHoveredEval.medianSuccessfulSteps ??
              activeHoveredEval.checkpoint.average_completion_steps) != null && (
              <span className="sd-mono" style={{ color: "#fbbf24", fontWeight: 600 }}>
                {Math.round(
                  (activeHoveredEval.medianSuccessfulSteps ??
                    activeHoveredEval.checkpoint.average_completion_steps)!
                )}{" "}
                steps
              </span>
            )}
            <span className="sd-mono" style={{ color: "var(--sd-text-secondary)" }}>
              Return {(activeHoveredEval.avgReward ?? 0).toFixed(1)}
            </span>
            {onInspectInArena && (
              <button
                type="button"
                className="sd-btn sd-btn--sm sd-btn--ghost"
                style={{ padding: "0.1rem 0.45rem", fontSize: "0.66rem" }}
                onClick={() =>
                  onInspectInArena(activeHoveredEval.checkpoint.checkpoint_episode)
                }
              >
                Inspect 10 Seeds →
              </button>
            )}
          </div>
        )}
      </div>

      {/* SVG Chart Area */}
      {mode === "stacked" ? (
        <div style={{ display: "flex", flexDirection: "column", gap: "0.55rem" }}>
          {renderSingleSvgChart("success", "1. Formal 10-Seed Evaluation Pass Rate (%)")}
          {renderSingleSvgChart(
            "steps",
            "2. Completion Steps (Inverted: Higher = Fewer Steps / Faster)"
          )}
          {renderSingleSvgChart("reward", "3. Average Episode Return")}
        </div>
      ) : (
        renderSingleSvgChart(mode)
      )}

      {/* COMPACT RECENT EVALUATIONS PICKER (Newest -> Oldest) */}
      {recentTenEvals.length > 0 && (
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: "0.35rem",
            flexWrap: "wrap",
            paddingTop: "0.4rem",
            borderTop: "1px solid var(--sd-border-subtle)",
            fontSize: "0.68rem",
          }}
        >
          <span
            style={{
              fontWeight: 700,
              color: "var(--sd-text-muted)",
              textTransform: "uppercase",
              letterSpacing: "0.04em",
              marginRight: "0.15rem",
            }}
            title={`Stage ${effectiveStage} All-Time Avg: ${stageStats.allTimeAvgPct.toFixed(
              1
            )}% · Recent 5 Avg: ${stageStats.recentRollingPct.toFixed(
              1
            )}% · Stage Peak: ${Math.round(stageStats.peakPct)}%`}
          >
            Recent Evals:
          </span>

          {recentTenEvals.slice(0, 8).map((ev, i) => {
            const ep = ev.checkpoint.checkpoint_episode;
            const isSelected = selectedCheckpointEpisode === ep;
            const steps = Math.round(
              ev.medianSuccessfulSteps ?? ev.checkpoint.average_completion_steps ?? 0
            );
            const failedSeeds = (ev.checkpoint.records ?? [])
              .filter((r) => !r.success)
              .map((r) => `#${r.seed}`);
            const rate = Math.round(ev.successRatePct);
            const toneColor =
              rate >= promotionTargetPct
                ? "var(--sd-emerald-bright)"
                : rate >= 70
                ? "var(--sd-amber-bright)"
                : "var(--sd-rose-bright)";

            return (
              <button
                key={`recent-ev-${ev.stage}-${ep}-${i}`}
                type="button"
                onClick={() => onSelectCheckpoint?.(ep)}
                className="sd-mono"
                title={`Ep ${ep.toLocaleString()} — ${rate}% Pass (${
                  steps > 0 ? `${steps} steps` : "—"
                })${
                  failedSeeds.length > 0
                    ? ` · Failed: ${failedSeeds.join(", ")}`
                    : " · All 10 seeds passed!"
                }`}
                style={{
                  display: "inline-flex",
                  alignItems: "center",
                  gap: "0.3rem",
                  padding: "0.2rem 0.48rem",
                  borderRadius: "var(--sd-radius-sm)",
                  background: isSelected
                    ? "rgba(99, 102, 241, 0.22)"
                    : "rgba(9, 13, 22, 0.65)",
                  border: isSelected
                    ? "1px solid var(--sd-accent-bright)"
                    : "1px solid var(--sd-border-subtle)",
                  cursor: "pointer",
                  color: "var(--sd-text-primary)",
                  fontSize: "0.67rem",
                }}
              >
                <span style={{ fontWeight: 700 }}>
                  {i === 0 ? "★ " : ""}Ep {ep.toLocaleString()}
                </span>
                <strong style={{ color: toneColor }}>{rate}%</strong>
                {steps > 0 && (
                  <span style={{ color: "var(--sd-text-muted)", fontSize: "0.62rem" }}>
                    {steps}s
                  </span>
                )}
              </button>
            );
          })}

          <span
            className="sd-mono"
            style={{
              marginLeft: "auto",
              fontSize: "0.65rem",
              color: "var(--sd-text-secondary)",
            }}
          >
            Recent 5 Avg:{" "}
            <strong
              style={{
                color:
                  stageStats.recentRollingPct >= promotionTargetPct
                    ? "var(--sd-emerald-bright)"
                    : "var(--sd-amber-bright)",
              }}
            >
              {stageStats.recentRollingPct.toFixed(0)}%
            </strong>{" "}
            · Peak:{" "}
            <strong style={{ color: "var(--sd-cyan)" }}>
              {Math.round(stageStats.peakPct)}%
            </strong>
          </span>
        </div>
      )}
    </div>
  );
};

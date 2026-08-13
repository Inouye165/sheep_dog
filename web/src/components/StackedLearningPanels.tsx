import React, { useMemo, useState } from "react";
import type { CheckpointEntry, TrainingEpisode } from "../state/types";

export type SmoothingWindow = 1 | 10 | 25 | 50;
export type XAxisMode = "stage_ep" | "global_ep" | "timesteps";

interface StackedLearningPanelsProps {
  episodes: TrainingEpisode[];
  checkpoints: CheckpointEntry[];
  curriculumStage: number;
  smoothingWindow: SmoothingWindow;
  xAxisMode: XAxisMode;
  showRawEpisodes: boolean;
  showRollingAvg: boolean;
  showFormalEvals: boolean;
  showPolicySnapshots: boolean;
  bestCheckpointEpisode?: number | null;
}

export function StackedLearningPanels({
  episodes,
  checkpoints,
  curriculumStage,
  smoothingWindow,
  xAxisMode,
  showRawEpisodes,
  showRollingAvg,
  showFormalEvals,
  showPolicySnapshots,
  bestCheckpointEpisode,
}: StackedLearningPanelsProps) {
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null);

  // 1. Sort and filter episodes
  const sortedEpisodes = useMemo(() => {
    return [...episodes].sort((a, b) => {
      const aX = a.episode_in_stage ?? a.global_environment_episode;
      const bX = b.episode_in_stage ?? b.global_environment_episode;
      return aX - bX;
    });
  }, [episodes]);

  const numEpisodes = sortedEpisodes.length;

  // 2. Map canonical X value for each episode
  const episodeXValues = useMemo(() => {
    return sortedEpisodes.map((ep) => {
      if (xAxisMode === "global_ep") {
        return ep.global_environment_episode ?? ep.episode_in_stage;
      }
      if (xAxisMode === "timesteps") {
        return ep.global_timestep ?? ep.global_environment_episode ?? ep.episode_in_stage;
      }
      return ep.episode_in_stage ?? ep.global_environment_episode;
    });
  }, [sortedEpisodes, xAxisMode]);

  // 3. Compute rolling metrics for each episode index
  const rollingMetrics = useMemo(() => {
    if (numEpisodes === 0) return [];
    const windowSize = smoothingWindow;

    return sortedEpisodes.map((ep, idx) => {
      const start = Math.max(0, idx - windowSize + 1);
      const slice = sortedEpisodes.slice(start, idx + 1);
      const sliceLen = slice.length;

      // Success Rate %
      const successes = slice.filter((e) => e.success).length;
      const successRatePct = (successes / sliceLen) * 100;

      // All-Episode Steps
      const totalStepsSum = slice.reduce((sum, e) => sum + e.steps, 0);
      const avgAllSteps = totalStepsSum / sliceLen;

      // Successful-Only Steps
      const succSlice = slice.filter((e) => e.success);
      const avgSuccSteps =
        succSlice.length > 0
          ? succSlice.reduce((sum, e) => sum + e.steps, 0) / succSlice.length
          : null;

      // Total Reward
      const avgReward = slice.reduce((sum, e) => sum + e.reward, 0) / slice.length;

      // Component Breakdowns (if recorded)
      const episodesWithBreakdown = slice.filter((e) => e.reward_breakdown != null && typeof e.reward_breakdown === "object");
      let breakdownAvg: Record<string, number> | null = null;
      if (episodesWithBreakdown.length > 0) {
        const componentTotals: Record<string, number> = {};
        for (const e of episodesWithBreakdown) {
          if (e.reward_breakdown) {
            for (const [k, v] of Object.entries(e.reward_breakdown)) {
              componentTotals[k] = (componentTotals[k] || 0) + (typeof v === "number" ? v : 0);
            }
          }
        }
        breakdownAvg = {};
        for (const [k, totalVal] of Object.entries(componentTotals)) {
          breakdownAvg[k] = totalVal / episodesWithBreakdown.length;
        }
      }

      return {
        idx,
        episode: ep,
        xVal: episodeXValues[idx],
        successRatePct,
        avgAllSteps,
        avgSuccSteps,
        avgReward,
        breakdownAvg,
        windowCount: sliceLen,
      };
    });
  }, [sortedEpisodes, episodeXValues, numEpisodes, smoothingWindow]);

  // 4. Determine Y-ranges across panels
  const stepsYRange = useMemo(() => {
    if (numEpisodes === 0) return { min: 0, max: 600 };
    const allVals: number[] = [];
    sortedEpisodes.forEach((e) => allVals.push(e.steps));
    rollingMetrics.forEach((m) => {
      allVals.push(m.avgAllSteps);
      if (m.avgSuccSteps != null) allVals.push(m.avgSuccSteps);
    });
    const minV = Math.min(...allVals);
    const maxV = Math.max(...allVals);
    const pad = (maxV - minV) * 0.15 || 20;
    return { min: Math.max(0, Math.floor(minV - pad)), max: Math.ceil(maxV + pad) };
  }, [sortedEpisodes, rollingMetrics, numEpisodes]);

  const rewardYRange = useMemo(() => {
    if (numEpisodes === 0) return { min: -100, max: 500 };
    const allVals: number[] = [];
    sortedEpisodes.forEach((e) => allVals.push(e.reward));
    rollingMetrics.forEach((m) => allVals.push(m.avgReward));
    const minV = Math.min(...allVals);
    const maxV = Math.max(...allVals);
    const pad = (maxV - minV) * 0.15 || 50;
    return { min: Math.floor(minV - pad), max: Math.ceil(maxV + pad) };
  }, [sortedEpisodes, rollingMetrics, numEpisodes]);

  // Active breakdown terms
  const breakdownTerms = useMemo(() => {
    const terms = new Set<string>();
    rollingMetrics.forEach((m) => {
      if (m.breakdownAvg) {
        Object.keys(m.breakdownAvg).forEach((k) => terms.add(k));
      }
    });
    return Array.from(terms).sort();
  }, [rollingMetrics]);

  const breakdownYRange = useMemo(() => {
    if (breakdownTerms.length === 0) return { min: -50, max: 150 };
    const vals: number[] = [];
    rollingMetrics.forEach((m) => {
      if (m.breakdownAvg) {
        Object.values(m.breakdownAvg).forEach((v) => vals.push(v));
      }
    });
    if (vals.length === 0) return { min: -50, max: 150 };
    const minV = Math.min(...vals);
    const maxV = Math.max(...vals);
    const pad = (maxV - minV) * 0.15 || 20;
    return { min: Math.floor(minV - pad), max: Math.ceil(maxV + pad) };
  }, [rollingMetrics, breakdownTerms]);

  // 5. Formal benchmark evals mapped to stage episode landmark points
  const formalEvalPoints = useMemo(() => {
    return checkpoints
      .filter((c) => {
        const cStage = c.reward_config?.instincts?.curriculum_stage ?? c.environment_config?.curriculum_stage ?? c.curriculum_stage ?? -1;
        return cStage === curriculumStage;
      })
      .map((c) => {
        const targetEp = c.total_training_episodes ?? c.cumulative_environment_episodes ?? c.checkpoint_episode;
        // Find closest index in sortedEpisodes or scale by xVal
        return {
          checkpoint: c,
          stageEpisode: targetEp,
          successRatePct: c.success_rate * 100,
          avgSteps: c.average_completion_steps,
          avgReward: c.average_reward,
        };
      });
  }, [checkpoints, curriculumStage]);

  if (numEpisodes === 0) {
    return (
      <div className="warning-box warning-box--info" style={{ marginTop: "1rem" }}>
        No training episodes recorded yet for Stage {curriculumStage}. Run training episodes to display the stacked learning curve.
      </div>
    );
  }

  // Common SVG layout parameters
  const W = 900;
  const H = 190;
  const PAD = { top: 25, right: 45, bottom: 35, left: 55 };
  const plotW = W - PAD.left - PAD.right;
  const plotH = H - PAD.top - PAD.bottom;

  function toSvgX(idx: number): number {
    if (numEpisodes <= 1) return PAD.left + plotW / 2;
    return PAD.left + (idx / (numEpisodes - 1)) * plotW;
  }

  function toSvgY(val: number, yMin: number, yMax: number): number {
    const range = yMax - yMin || 1;
    return PAD.top + plotH - ((val - yMin) / range) * plotH;
  }

  // Hovered item data
  const activeHoveredPoint = hoveredIndex !== null && hoveredIndex >= 0 && hoveredIndex < numEpisodes
    ? rollingMetrics[hoveredIndex]
    : null;

  // Colors for breakdown terms
  const TERM_COLORS: Record<string, string> = {
    progress_to_pen: "#4ade80",
    sheep_penned: "#38bdf8",
    flock_cohesion: "#a78bfa",
    terminal_success: "#34d399",
    gate_progress: "#facc15",
    time_penalty: "#f87171",
    no_progress_penalty: "#fb923c",
    scatter_penalty: "#f472b6",
    wall_pressure_penalty: "#e879f9",
    wait_penalty: "#cbd5e1",
    sprint_cost: "#94a3b8",
    terminal_failure: "#ef4444",
  };

  function getTermColor(term: string, idx: number): string {
    if (TERM_COLORS[term]) return TERM_COLORS[term];
    const defaultPalette = ["#60a5fa", "#f59e0b", "#10b981", "#ec4899", "#8b5cf6", "#06b6d4"];
    return defaultPalette[idx % defaultPalette.length];
  }

  return (
    <div className="stacked-learning-panels" style={{ display: "flex", flexDirection: "column", gap: "1.25rem", marginTop: "1rem" }}>
      {/* Synchronized Hover Header / Callout */}
      {activeHoveredPoint ? (
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
            justify: "space-between",
          }}
        >
          <div>
            <strong>Stage Episode {activeHoveredPoint.episode.episode_in_stage ?? activeHoveredPoint.episode.global_environment_episode}</strong>
            <span style={{ marginLeft: "0.5rem", opacity: 0.85 }}>
              (Global Ep {activeHoveredPoint.episode.global_environment_episode})
            </span>
            <span
              style={{
                marginLeft: "0.75rem",
                padding: "0.1rem 0.4rem",
                borderRadius: "3px",
                fontSize: "0.75rem",
                fontWeight: 600,
                background: activeHoveredPoint.episode.success ? "rgba(74,222,128,0.2)" : "rgba(248,113,113,0.2)",
                color: activeHoveredPoint.episode.success ? "#4ade80" : "#f87171",
              }}
            >
              {activeHoveredPoint.episode.result}
            </span>
          </div>

          <div style={{ display: "flex", gap: "1.25rem", flexWrap: "wrap" }}>
            <div>
              <span style={{ color: "#94a3b8" }}>Success (Rolling {activeHoveredPoint.windowCount}): </span>
              <strong style={{ color: activeHoveredPoint.successRatePct >= 50 ? "#4ade80" : "#f87171" }}>
                {activeHoveredPoint.successRatePct.toFixed(1)}%
              </strong>
            </div>
            <div>
              <span style={{ color: "#94a3b8" }}>All Ep Steps: </span>
              <strong>{Math.round(activeHoveredPoint.avgAllSteps)}</strong>
            </div>
            <div>
              <span style={{ color: "#94a3b8" }}>Succ-Only Steps: </span>
              <strong style={{ color: "#f59e0b" }}>
                {activeHoveredPoint.avgSuccSteps != null ? Math.round(activeHoveredPoint.avgSuccSteps) : "—"}
              </strong>
            </div>
            <div>
              <span style={{ color: "#94a3b8" }}>Avg Reward: </span>
              <strong style={{ color: "#38bdf8" }}>{activeHoveredPoint.avgReward.toFixed(1)}</strong>
            </div>
          </div>
        </div>
      ) : null}

      {/* ────────────────────────────────────────────────────────────────────── */}
      {/* PANEL 1: SUCCESS RATE (%) */}
      {/* ────────────────────────────────────────────────────────────────────── */}
      <div className="panel-card" style={{ background: "rgba(15, 23, 42, 0.6)", borderRadius: "8px", border: "1px solid rgba(255,255,255,0.08)", padding: "0.75rem" }}>
        <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "0.25rem", fontSize: "0.85rem" }}>
          <strong style={{ color: "#e2e8f0" }}>PANEL 1 — SUCCESS RATE (%)</strong>
          <div style={{ display: "flex", gap: "1rem", fontSize: "0.75rem", color: "#94a3b8" }}>
            {showRawEpisodes && <span><span style={{ color: "#38bdf8" }}>●</span> Raw Episode</span>}
            {showRollingAvg && <span><span style={{ color: "#4ade80" }}>━</span> Rolling Training Avg</span>}
            {showFormalEvals && <span><span style={{ color: "#facc15" }}>◆</span> Formal 10-Seed Eval</span>}
            <span style={{ color: "rgba(129,201,149,0.9)" }}>- - 50% Promotion Bar</span>
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
          <line x1={PAD.left} y1={toSvgY(50, 0, 100)} x2={PAD.left + plotW} y2={toSvgY(50, 0, 100)} stroke="rgba(74,222,128,0.6)" strokeWidth={1.5} strokeDasharray="4 4" />

          {/* Raw Episode Points */}
          {showRawEpisodes &&
            rollingMetrics.map((m) => {
              const cx = toSvgX(m.idx);
              const cy = toSvgY(m.episode.success ? 100 : 0, 0, 100);
              return (
                <circle
                  key={`p1-raw-${m.idx}`}
                  cx={cx}
                  cy={cy}
                  r={2.5}
                  fill={m.episode.success ? "rgba(74,222,128,0.5)" : "rgba(248,113,113,0.35)"}
                />
              );
            })}

          {/* Rolling Success Polyline */}
          {showRollingAvg && numEpisodes >= 2 && (
            <polyline
              points={rollingMetrics.map((m) => `${toSvgX(m.idx).toFixed(1)},${toSvgY(m.successRatePct, 0, 100).toFixed(1)}`).join(" ")}
              fill="none"
              stroke="#4ade80"
              strokeWidth={2.5}
              strokeLinejoin="round"
            />
          )}

          {/* Formal Confidence Evaluation Landmark Diamond Markers */}
          {showFormalEvals &&
            formalEvalPoints.map((pt, i) => {
              // Find matching x position
              const matchingIdx = rollingMetrics.findIndex((m) => m.xVal >= pt.stageEpisode);
              const idxToUse = matchingIdx !== -1 ? matchingIdx : numEpisodes - 1;
              const cx = toSvgX(idxToUse);
              const cy = toSvgY(pt.successRatePct, 0, 100);
              const r = 6;
              const diamond = `${cx},${cy - r} ${cx + r},${cy} ${cx},${cy + r} ${cx - r},${cy}`;
              return (
                <g key={`p1-eval-${i}`}>
                  <polygon points={diamond} fill="#facc15" stroke="rgba(15,23,42,0.9)" strokeWidth={1.5} />
                  <text x={cx} y={cy - 9} textAnchor="middle" fontSize={10} fontWeight="bold" fill="#facc15">
                    {Math.round(pt.successRatePct)}%
                  </text>
                </g>
              );
            })}

          {/* Synchronized Hover Crosshair Line & Mouse Overlay */}
          {rollingMetrics.map((m) => {
            const cx = toSvgX(m.idx);
            const isHovered = hoveredIndex === m.idx;
            return (
              <rect
                key={`p1-overlay-${m.idx}`}
                x={cx - (plotW / numEpisodes) / 2}
                y={PAD.top}
                width={plotW / numEpisodes || 1}
                height={plotH}
                fill="transparent"
                onMouseEnter={() => setHoveredIndex(m.idx)}
                onMouseLeave={() => setHoveredIndex(null)}
                style={{ cursor: "pointer" }}
              />
            );
          })}

          {hoveredIndex !== null && (
            <line
              x1={toSvgX(hoveredIndex)}
              y1={PAD.top}
              x2={toSvgX(hoveredIndex)}
              y2={PAD.top + plotH}
              stroke="#38bdf8"
              strokeWidth={1.5}
              strokeDasharray="3 3"
              style={{ pointerEvents: "none" }}
            />
          )}

          {/* X-axis labels */}
          {rollingMetrics.length > 0 &&
            Array.from(new Set([0, Math.floor((numEpisodes - 1) / 2), numEpisodes - 1])).map((idx) => {
              const m = rollingMetrics[idx];
              if (!m) return null;
              return (
                <text key={`p1-x-${idx}`} x={toSvgX(idx)} y={H - 8} textAnchor="middle" fontSize={11} fill="#94a3b8">
                  Ep {m.xVal}
                </text>
              );
            })}
        </svg>
      </div>

      {/* ────────────────────────────────────────────────────────────────────── */}
      {/* PANEL 2: STEPS & EFFICIENCY */}
      {/* ────────────────────────────────────────────────────────────────────── */}
      <div className="panel-card" style={{ background: "rgba(15, 23, 42, 0.6)", borderRadius: "8px", border: "1px solid rgba(255,255,255,0.08)", padding: "0.75rem" }}>
        <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "0.25rem", fontSize: "0.85rem" }}>
          <strong style={{ color: "#e2e8f0" }}>PANEL 2 — STEPS / EFFICIENCY (FEWER IS FASTER)</strong>
          <div style={{ display: "flex", gap: "1rem", fontSize: "0.75rem", color: "#94a3b8" }}>
            <span><span style={{ color: "#38bdf8" }}>━</span> All Episodes Rolling Avg</span>
            <span><span style={{ color: "#f59e0b" }}>- -</span> Successful-Only Rolling Avg</span>
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

          {/* Line A: All-Episode Rolling Avg Steps */}
          {showRollingAvg && numEpisodes >= 2 && (
            <polyline
              points={rollingMetrics.map((m) => `${toSvgX(m.idx).toFixed(1)},${toSvgY(m.avgAllSteps, stepsYRange.min, stepsYRange.max).toFixed(1)}`).join(" ")}
              fill="none"
              stroke="#38bdf8"
              strokeWidth={2.5}
              strokeLinejoin="round"
            />
          )}

          {/* Line B: Successful-Only Rolling Avg Steps */}
          {showRollingAvg && numEpisodes >= 2 && (
            <polyline
              points={rollingMetrics
                .filter((m) => m.avgSuccSteps != null)
                .map((m) => `${toSvgX(m.idx).toFixed(1)},${toSvgY(m.avgSuccSteps!, stepsYRange.min, stepsYRange.max).toFixed(1)}`)
                .join(" ")}
              fill="none"
              stroke="#f59e0b"
              strokeWidth={2}
              strokeDasharray="5 3"
              strokeLinejoin="round"
            />
          )}

          {/* Hover Crosshair & Overlay */}
          {rollingMetrics.map((m) => (
            <rect
              key={`p2-overlay-${m.idx}`}
              x={toSvgX(m.idx) - (plotW / numEpisodes) / 2}
              y={PAD.top}
              width={plotW / numEpisodes || 1}
              height={plotH}
              fill="transparent"
              onMouseEnter={() => setHoveredIndex(m.idx)}
              onMouseLeave={() => setHoveredIndex(null)}
              style={{ cursor: "pointer" }}
            />
          ))}

          {hoveredIndex !== null && (
            <line
              x1={toSvgX(hoveredIndex)}
              y1={PAD.top}
              x2={toSvgX(hoveredIndex)}
              y2={PAD.top + plotH}
              stroke="#38bdf8"
              strokeWidth={1.5}
              strokeDasharray="3 3"
              style={{ pointerEvents: "none" }}
            />
          )}

          {/* X-axis labels */}
          {rollingMetrics.length > 0 &&
            Array.from(new Set([0, Math.floor((numEpisodes - 1) / 2), numEpisodes - 1])).map((idx) => {
              const m = rollingMetrics[idx];
              if (!m) return null;
              return (
                <text key={`p2-x-${idx}`} x={toSvgX(idx)} y={H - 8} textAnchor="middle" fontSize={11} fill="#94a3b8">
                  Ep {m.xVal}
                </text>
              );
            })}
        </svg>
      </div>

      {/* ────────────────────────────────────────────────────────────────────── */}
      {/* PANEL 3: TOTAL REWARD */}
      {/* ────────────────────────────────────────────────────────────────────── */}
      <div className="panel-card" style={{ background: "rgba(15, 23, 42, 0.6)", borderRadius: "8px", border: "1px solid rgba(255,255,255,0.08)", padding: "0.75rem" }}>
        <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "0.25rem", fontSize: "0.85rem" }}>
          <strong style={{ color: "#e2e8f0" }}>PANEL 3 — TOTAL REWARD</strong>
          <div style={{ display: "flex", gap: "1rem", fontSize: "0.75rem", color: "#94a3b8" }}>
            {showRawEpisodes && <span><span style={{ color: "rgba(168,85,247,0.5)" }}>●</span> Raw Episode Reward</span>}
            {showRollingAvg && <span><span style={{ color: "#a78bfa" }}>━</span> Rolling Avg Reward</span>}
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

          {/* Raw Episode Points */}
          {showRawEpisodes &&
            rollingMetrics.map((m) => {
              const cx = toSvgX(m.idx);
              const cy = toSvgY(m.episode.reward, rewardYRange.min, rewardYRange.max);
              return <circle key={`p3-raw-${m.idx}`} cx={cx} cy={cy} r={2.5} fill="rgba(168,85,247,0.45)" />;
            })}

          {/* Rolling Reward Line */}
          {showRollingAvg && numEpisodes >= 2 && (
            <polyline
              points={rollingMetrics.map((m) => `${toSvgX(m.idx).toFixed(1)},${toSvgY(m.avgReward, rewardYRange.min, rewardYRange.max).toFixed(1)}`).join(" ")}
              fill="none"
              stroke="#a78bfa"
              strokeWidth={2.5}
              strokeLinejoin="round"
            />
          )}

          {/* Hover Overlay */}
          {rollingMetrics.map((m) => (
            <rect
              key={`p3-overlay-${m.idx}`}
              x={toSvgX(m.idx) - (plotW / numEpisodes) / 2}
              y={PAD.top}
              width={plotW / numEpisodes || 1}
              height={plotH}
              fill="transparent"
              onMouseEnter={() => setHoveredIndex(m.idx)}
              onMouseLeave={() => setHoveredIndex(null)}
              style={{ cursor: "pointer" }}
            />
          ))}

          {hoveredIndex !== null && (
            <line
              x1={toSvgX(hoveredIndex)}
              y1={PAD.top}
              x2={toSvgX(hoveredIndex)}
              y2={PAD.top + plotH}
              stroke="#38bdf8"
              strokeWidth={1.5}
              strokeDasharray="3 3"
              style={{ pointerEvents: "none" }}
            />
          )}

          {/* X-axis labels */}
          {rollingMetrics.length > 0 &&
            Array.from(new Set([0, Math.floor((numEpisodes - 1) / 2), numEpisodes - 1])).map((idx) => {
              const m = rollingMetrics[idx];
              if (!m) return null;
              return (
                <text key={`p3-x-${idx}`} x={toSvgX(idx)} y={H - 8} textAnchor="middle" fontSize={11} fill="#94a3b8">
                  Ep {m.xVal}
                </text>
              );
            })}
        </svg>
      </div>

      {/* ────────────────────────────────────────────────────────────────────── */}
      {/* PANEL 4: REWARD & PENALTY COMPONENT BREAKDOWN */}
      {/* ────────────────────────────────────────────────────────────────────── */}
      <div className="panel-card" style={{ background: "rgba(15, 23, 42, 0.6)", borderRadius: "8px", border: "1px solid rgba(255,255,255,0.08)", padding: "0.75rem" }}>
        <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "0.25rem", fontSize: "0.85rem" }}>
          <strong style={{ color: "#e2e8f0" }}>PANEL 4 — REWARDS & PENALTIES BREAKDOWN</strong>
          <div style={{ display: "flex", gap: "0.8rem", fontSize: "0.75rem", color: "#94a3b8", flexWrap: "wrap" }}>
            {breakthroughNoticeMessage(breakdownTerms)}
          </div>
        </div>

        {breakdownTerms.length === 0 ? (
          <div style={{ fontSize: "0.78rem", color: "#94a3b8", fontStyle: "italic", padding: "1rem 0" }}>
            ℹ️ Detailed per-episode reward component breakdown was not recorded for these historical episodes. Future training episodes will automatically capture and persist component breakdowns.
          </div>
        ) : (
          <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", overflow: "visible" }}>
            {/* Y-axis grid & labels */}
            {[breakdownYRange.min, 0, breakdownYRange.max].map((v) => {
              const sy = toSvgY(v, breakdownYRange.min, breakdownYRange.max);
              return (
                <g key={`p4-grid-${v}`}>
                  <line x1={PAD.left} y1={sy} x2={PAD.left + plotW} y2={sy} stroke={v === 0 ? "rgba(255,255,255,0.2)" : "rgba(255,255,255,0.07)"} strokeDasharray={v === 0 ? undefined : "3 3"} />
                  <text x={PAD.left - 8} y={sy + 3.5} textAnchor="end" fontSize={11} fill="#94a3b8">{v}</text>
                </g>
              );
            })}

            {/* Render line per breakdown term */}
            {breakdownTerms.map((term, tIdx) => {
              const color = getTermColor(term, tIdx);
              const pointsStr = rollingMetrics
                .map((m) => {
                  const val = m.breakdownAvg ? m.breakdownAvg[term] ?? 0 : 0;
                  return `${toSvgX(m.idx).toFixed(1)},${toSvgY(val, breakdownYRange.min, breakdownYRange.max).toFixed(1)}`;
                })
                .join(" ");

              return (
                <polyline
                  key={`p4-term-${term}`}
                  points={pointsStr}
                  fill="none"
                  stroke={color}
                  strokeWidth={2}
                  strokeLinejoin="round"
                  opacity={0.85}
                />
              );
            })}

            {/* Hover Overlay */}
            {rollingMetrics.map((m) => (
              <rect
                key={`p4-overlay-${m.idx}`}
                x={toSvgX(m.idx) - (plotW / numEpisodes) / 2}
                y={PAD.top}
                width={plotW / numEpisodes || 1}
                height={plotH}
                fill="transparent"
                onMouseEnter={() => setHoveredIndex(m.idx)}
                onMouseLeave={() => setHoveredIndex(null)}
                style={{ cursor: "pointer" }}
              />
            ))}

            {hoveredIndex !== null && (
              <line
                x1={toSvgX(hoveredIndex)}
                y1={PAD.top}
                x2={toSvgX(hoveredIndex)}
                y2={PAD.top + plotH}
                stroke="#38bdf8"
                strokeWidth={1.5}
                strokeDasharray="3 3"
                style={{ pointerEvents: "none" }}
              />
            )}

            {/* X-axis labels */}
            {rollingMetrics.length > 0 &&
              Array.from(new Set([0, Math.floor((numEpisodes - 1) / 2), numEpisodes - 1])).map((idx) => {
                const m = rollingMetrics[idx];
                if (!m) return null;
                return (
                  <text key={`p4-x-${idx}`} x={toSvgX(idx)} y={H - 8} textAnchor="middle" fontSize={11} fill="#94a3b8">
                    Ep {m.xVal}
                  </text>
                );
              })}
          </svg>
        )}
      </div>
    </div>
  );
}

function breakthroughNoticeMessage(terms: string[]) {
  if (terms.length === 0) return null;
  return terms.slice(0, 6).map((term, i) => (
    <span key={term} style={{ display: "inline-flex", alignItems: "center", gap: "0.25rem" }}>
      <span style={{ display: "inline-block", width: 8, height: 8, borderRadius: "50%", background: termColorSimple(term, i) }} />
      {term}
    </span>
  ));
}

function termColorSimple(term: string, idx: number): string {
  const map: Record<string, string> = {
    progress_to_pen: "#4ade80",
    sheep_penned: "#38bdf8",
    flock_cohesion: "#a78bfa",
    terminal_success: "#34d399",
    time_penalty: "#f87171",
    no_progress_penalty: "#fb923c",
    scatter_penalty: "#f472b6",
  };
  return map[term] || ["#60a5fa", "#f59e0b", "#10b981"][idx % 3];
}

import React, { useState, useEffect, useCallback, useRef } from "react";
import { loadTrainingHistory } from "../lib/api";
import type { TelemetryRecord } from "../state/types";
import { RecentEpisodesViewer } from "./RecentEpisodesViewer";

function getStageName(stage: number): string {
  if (stage === 10) return "10A";
  if (stage === 11) return "10B";
  if (stage === 12) return "10C";
  if (stage === 13) return "10D";
  return String(stage);
}

function isFiniteNumber(val: unknown): val is number {
  return typeof val === "number" && Number.isFinite(val);
}

export function HistoryTab() {
  const [history, setHistory] = useState<TelemetryRecord[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const activeRef = useRef(true);
  const hasLoadedOnceRef = useRef(false);

  const fetchHistory = useCallback(async (isBackgroundPoll = false) => {
    try {
      if (!isBackgroundPoll && !hasLoadedOnceRef.current) {
        setLoading(true);
      }
      if (!isBackgroundPoll) {
        setError(null);
      }
      const data = await loadTrainingHistory();
      if (!activeRef.current) return;
      if (!Array.isArray(data)) {
        if (!hasLoadedOnceRef.current) {
          setHistory([]);
        }
        return;
      }
      // Sort history by step/timestamp just in case
      const sorted = [...data].sort((a, b) => (a.step ?? 0) - (b.step ?? 0));
      setHistory(sorted);
      setError(null);
      hasLoadedOnceRef.current = true;
    } catch (err) {
      if (!activeRef.current) return;
      console.error(err);
      if (!hasLoadedOnceRef.current) {
        setError("Failed to load training history telemetry records.");
      }
    } finally {
      if (activeRef.current) {
        setLoading(false);
      }
    }
  }, []);

  useEffect(() => {
    activeRef.current = true;
    void fetchHistory(false);
    // Set up auto-refresh every 8 seconds without resetting loading=true
    const interval = setInterval(() => {
      void fetchHistory(true);
    }, 8000);
    return () => {
      activeRef.current = false;
      clearInterval(interval);
    };
  }, [fetchHistory]);

  // Compute metrics
  const totalRecords = history.length;
  const latestRecord = totalRecords > 0 ? history[totalRecords - 1] : null;
  
  const peakSuccess = useMemoPeakSuccess(history);
  const totalSteps = isFiniteNumber(latestRecord?.step) ? latestRecord.step : 0;
  const currentStageName = latestRecord && isFiniteNumber(latestRecord.stage) ? getStageName(latestRecord.stage) : "N/A";

  // Render SVG Chart for Success Rate
  const chartWidth = 800;
  const chartHeight = 240;
  const paddingLeft = 50;
  const paddingRight = 30;
  const paddingTop = 20;
  const paddingBottom = 40;

  const chartPoints = React.useMemo(() => {
    if (totalRecords < 2) return [];

    return history.map((record, i) => {
      const x = paddingLeft + (i / (totalRecords - 1)) * (chartWidth - paddingLeft - paddingRight);
      const sr = isFiniteNumber(record.success_rate) ? record.success_rate : 0;
      // y goes from top (0) to bottom (chartHeight). Success rate is 0.0 to 1.0.
      const y = chartHeight - paddingBottom - sr * (chartHeight - paddingTop - paddingBottom);
      return { x, y, record };
    });
  }, [history, totalRecords]);

  const svgPath = React.useMemo(() => {
    if (chartPoints.length < 2) return "";
    return chartPoints
      .map((p, i) => `${i === 0 ? "M" : "L"} ${isFiniteNumber(p.x) ? p.x.toFixed(1) : "0.0"} ${isFiniteNumber(p.y) ? p.y.toFixed(1) : "0.0"}`)
      .join(" ");
  }, [chartPoints]);

  const svgAreaPath = React.useMemo(() => {
    if (chartPoints.length < 2) return "";
    const first = chartPoints[0];
    const last = chartPoints[chartPoints.length - 1];
    const baselineY = chartHeight - paddingBottom;
    const lastX = isFiniteNumber(last.x) ? last.x.toFixed(1) : "0.0";
    const firstX = isFiniteNumber(first.x) ? first.x.toFixed(1) : "0.0";
    return `${svgPath} L ${lastX} ${baselineY} L ${firstX} ${baselineY} Z`;
  }, [chartPoints, svgPath]);

  return (
    <div className="network-tab" style={{ padding: "1.5rem", overflowY: "auto", height: "100%" }}>
      <div className="network-tab__header" style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "1.5rem" }}>
        <div>
          <h2>Training History Telemetry</h2>
          <p className="network-tab__intro">Real-time learning metrics, hyperparameters, and PPO agent diagnostics.</p>
        </div>
        <button 
          onClick={() => void fetchHistory(false)} 
          className="control-bar__button"
          style={{ padding: "0.5rem 1rem", backgroundColor: "var(--theme-accent)", color: "white", borderRadius: "4px", border: "none", cursor: "pointer", fontWeight: "bold" }}
        >
          Refresh Data
        </button>
      </div>

      {loading && totalRecords === 0 ? (
        <div style={{ padding: "3rem", textAlign: "center", color: "var(--text-muted)" }}>
          Loading training history...
        </div>
      ) : error ? (
        <div style={{ padding: "2rem", backgroundColor: "rgba(239, 68, 68, 0.1)", color: "#ef4444", borderRadius: "8px", border: "1px solid #ef4444", marginBottom: "1.5rem" }}>
          {error}
        </div>
      ) : (
        <>
          {/* Recent Episode History Viewer & Replay */}
          <div style={{ marginBottom: "2rem" }}>
            <RecentEpisodesViewer />
          </div>

          {/* KPI Cards */}
          <div className="network-tab__kpis" style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))", gap: "1rem", marginBottom: "2rem" }}>
            <div className="network-tab__card" style={{ padding: "1rem", borderRadius: "8px", border: "1px solid var(--border-color)" }}>
              <span style={{ fontSize: "0.85rem", color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.05em" }}>Current Stage</span>
              <strong style={{ display: "block", fontSize: "1.75rem", margin: "0.5rem 0" }}>Stage {currentStageName}</strong>
              <span style={{ fontSize: "0.8rem", color: "var(--text-muted)" }}>Granular curriculum smoothed</span>
            </div>

            <div className="network-tab__card" style={{ padding: "1rem", borderRadius: "8px", border: "1px solid var(--border-color)" }}>
              <span style={{ fontSize: "0.85rem", color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.05em" }}>Peak Success Rate</span>
              <strong style={{ display: "block", fontSize: "1.75rem", margin: "0.5rem 0", color: "#10b981" }}>
                {isFiniteNumber(peakSuccess) ? `${(peakSuccess * 100).toFixed(0)}%` : "0%"}
              </strong>
              <span style={{ fontSize: "0.8rem", color: "var(--text-muted)" }}>Best evaluation run</span>
            </div>

            <div className="network-tab__card" style={{ padding: "1rem", borderRadius: "8px", border: "1px solid var(--border-color)" }}>
              <span style={{ fontSize: "0.85rem", color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.05em" }}>Total Steps</span>
              <strong style={{ display: "block", fontSize: "1.75rem", margin: "0.5rem 0" }}>{totalSteps.toLocaleString()}</strong>
              <span style={{ fontSize: "0.8rem", color: "var(--text-muted)" }}>Environment interactions</span>
            </div>

            <div className="network-tab__card" style={{ padding: "1rem", borderRadius: "8px", border: "1px solid var(--border-color)" }}>
              <span style={{ fontSize: "0.85rem", color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.05em" }}>Latest KL Divergence</span>
              <strong style={{ display: "block", fontSize: "1.75rem", margin: "0.5rem 0" }}>
                {isFiniteNumber(latestRecord?.metrics?.approx_kl) ? latestRecord.metrics.approx_kl.toFixed(5) : "N/A"}
              </strong>
              <span style={{ fontSize: "0.8rem", color: "var(--text-muted)" }}>Policy update divergence</span>
            </div>
          </div>

          {/* SVG Line Chart */}
          <div className="network-tab__card" style={{ padding: "1.5rem", borderRadius: "8px", border: "1px solid var(--border-color)", marginBottom: "2rem" }}>
            <h3 style={{ marginTop: 0, marginBottom: "1rem" }}>Evaluation Success Rate Trend</h3>
            {totalRecords < 2 ? (
              <div style={{ height: "200px", display: "flex", alignItems: "center", justifyContent: "center", color: "var(--text-muted)" }}>
                Waiting for more checkpoints to plot learning curve...
              </div>
            ) : (
              <div style={{ overflowX: "auto" }}>
                <svg width="100%" height={chartHeight} viewBox={`0 0 ${chartWidth} ${chartHeight}`} style={{ overflow: "visible" }}>
                  <defs>
                    <linearGradient id="chartGradient" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="#3b82f6" stopOpacity="0.4" />
                      <stop offset="100%" stopColor="#3b82f6" stopOpacity="0.0" />
                    </linearGradient>
                  </defs>

                  {/* Y Axis Gridlines */}
                  {[0, 0.25, 0.5, 0.75, 1.0].map((val) => {
                    const y = chartHeight - paddingBottom - val * (chartHeight - paddingTop - paddingBottom);
                    return (
                      <g key={val}>
                        <line 
                          x1={paddingLeft} 
                          y1={y} 
                          x2={chartWidth - paddingRight} 
                          y2={y} 
                          stroke="var(--border-color)" 
                          strokeDasharray="4 4" 
                        />
                        <text 
                          x={paddingLeft - 8} 
                          y={y + 4} 
                          fill="var(--text-muted)" 
                          fontSize="10" 
                          textAnchor="end"
                        >
                          {`${val * 100}%`}
                        </text>
                      </g>
                    );
                  })}

                  {/* Gradient Area */}
                  {svgAreaPath && (
                    <path d={svgAreaPath} fill="url(#chartGradient)" />
                  )}

                  {/* Line path */}
                  {svgPath && (
                    <path 
                      d={svgPath} 
                      fill="none" 
                      stroke="#3b82f6" 
                      strokeWidth="2.5" 
                      strokeLinecap="round" 
                      strokeLinejoin="round" 
                    />
                  )}

                  {/* Dots for checkpoints */}
                  {chartPoints.map((pt, idx) => {
                    const baseKey = pt.record.checkpoint_id 
                      ? pt.record.checkpoint_id 
                      : pt.record.evaluation_id 
                        ? pt.record.evaluation_id 
                        : `${pt.record.run_id || "unknown"}-${pt.record.stage}-${pt.record.global_episode || pt.record.step}-${pt.record.episode_in_stage || pt.record.step}-${pt.record.recorded_at || pt.record.timestamp}`;
                    const dotKey = `${baseKey}-${idx}`;
                    const ptStep = isFiniteNumber(pt.record.step) ? pt.record.step.toLocaleString() : "0";
                    const ptSr = isFiniteNumber(pt.record.success_rate) ? (pt.record.success_rate * 100).toFixed(0) : "0";
                    const ptRew = isFiniteNumber(pt.record.metrics?.average_reward) ? pt.record.metrics.average_reward.toFixed(1) : "N/A";
                    return (
                      <g key={dotKey} className="chart-dot">
                        <circle 
                          cx={pt.x} 
                          cy={pt.y} 
                          r="4" 
                          fill="#3b82f6" 
                          stroke="var(--bg-primary, #1e1e1e)" 
                          strokeWidth="1.5" 
                        />
                        <title>
                          {`Step: ${ptStep}\nStage: ${getStageName(pt.record.stage)}\nSuccess: ${ptSr}%\nReward: ${ptRew}`}
                        </title>
                      </g>
                    );
                  })}

                  {/* X Axis labels */}
                  {chartPoints.length > 1 && (
                    <>
                      <text x={paddingLeft} y={chartHeight - 15} fill="var(--text-muted)" fontSize="10" textAnchor="middle">
                        {isFiniteNumber(history[0]?.step) ? history[0].step.toLocaleString() : "0"}
                      </text>
                      <text x={(chartWidth - paddingRight + paddingLeft) / 2} y={chartHeight - 15} fill="var(--text-muted)" fontSize="10" textAnchor="middle">
                        Steps (Timesteps)
                      </text>
                      <text x={chartWidth - paddingRight} y={chartHeight - 15} fill="var(--text-muted)" fontSize="10" textAnchor="middle">
                        {isFiniteNumber(history[history.length - 1]?.step) ? history[history.length - 1].step.toLocaleString() : "0"}
                      </text>
                    </>
                  )}
                </svg>
              </div>
            )}
          </div>

          {/* Database Table */}
          <div className="network-tab__card" style={{ padding: "1.5rem", borderRadius: "8px", border: "1px solid var(--border-color)" }}>
            <h3 style={{ marginTop: 0, marginBottom: "1rem" }}>Telemetry Event Records</h3>
            {totalRecords === 0 ? (
              <div style={{ textAlign: "center", padding: "2rem", color: "var(--text-muted)" }}>
                No history entries recorded yet.
              </div>
            ) : (
              <div style={{ overflowX: "auto", maxHeight: "400px" }}>
                <table className="saved-scenarios-panel__table" style={{ width: "100%", borderCollapse: "collapse" }}>
                  <thead>
                    <tr>
                      <th style={{ textAlign: "left", padding: "0.75rem", borderBottom: "2px solid var(--border-color)" }}>Step</th>
                      <th style={{ textAlign: "left", padding: "0.75rem", borderBottom: "2px solid var(--border-color)" }}>Stage</th>
                      <th style={{ textAlign: "left", padding: "0.75rem", borderBottom: "2px solid var(--border-color)" }}>Success</th>
                      <th style={{ textAlign: "left", padding: "0.75rem", borderBottom: "2px solid var(--border-color)" }}>Avg Reward</th>
                      <th style={{ textAlign: "left", padding: "0.75rem", borderBottom: "2px solid var(--border-color)" }}>KL Div</th>
                      <th style={{ textAlign: "left", padding: "0.75rem", borderBottom: "2px solid var(--border-color)" }}>Clip Frac</th>
                      <th style={{ textAlign: "left", padding: "0.75rem", borderBottom: "2px solid var(--border-color)" }}>Expl. Var</th>
                      <th style={{ textAlign: "left", padding: "0.75rem", borderBottom: "2px solid var(--border-color)" }}>Active Hyperparams</th>
                    </tr>
                  </thead>
                  <tbody>
                    {[...history].reverse().map((record, index) => {
                      const baseKey = record.checkpoint_id 
                        ? record.checkpoint_id 
                        : record.evaluation_id 
                          ? record.evaluation_id 
                          : `${record.run_id || "unknown"}-${record.stage}-${record.global_episode || record.step}-${record.episode_in_stage || record.step}-${record.recorded_at || record.timestamp}`;
                      const rowKey = `${baseKey}-${index}`;
                      const sr = isFiniteNumber(record.success_rate) ? record.success_rate : 0;
                      const avgReward = record.metrics?.average_reward;
                      const approxKl = record.metrics?.approx_kl;
                      const clipFrac = record.metrics?.clip_fraction;
                      const explVar = record.metrics?.explained_variance;
                      const lr = record.hyperparameters?.learning_rate;
                      return (
                        <tr key={rowKey} style={{ borderBottom: "1px solid var(--border-color)" }}>
                          <td style={{ padding: "0.75rem" }}>{isFiniteNumber(record.step) ? record.step.toLocaleString() : "-"}</td>
                        <td style={{ padding: "0.75rem" }}>
                          <span style={{ padding: "0.2rem 0.5rem", borderRadius: "4px", backgroundColor: "var(--theme-bg-accent, #333)", fontWeight: "bold" }}>
                            {isFiniteNumber(record.stage) ? getStageName(record.stage) : "-"}
                          </span>
                        </td>
                        <td style={{ padding: "0.75rem", fontWeight: "bold", color: sr >= 0.8 ? "#10b981" : sr >= 0.5 ? "#f59e0b" : "#ef4444" }}>
                          {isFiniteNumber(record.success_rate) ? `${(record.success_rate * 100).toFixed(0)}%` : "-"}
                        </td>
                        <td style={{ padding: "0.75rem" }}>{isFiniteNumber(avgReward) ? avgReward.toFixed(2) : "-"}</td>
                        <td style={{ padding: "0.75rem", fontFamily: "monospace" }}>
                          {isFiniteNumber(approxKl) ? approxKl.toFixed(5) : "-"}
                        </td>
                        <td style={{ padding: "0.75rem" }}>
                          {isFiniteNumber(clipFrac) ? clipFrac.toFixed(3) : "-"}
                        </td>
                        <td style={{ padding: "0.75rem" }}>
                          {isFiniteNumber(explVar) ? explVar.toFixed(3) : "-"}
                        </td>
                        <td style={{ padding: "0.75rem", fontSize: "0.8rem", color: "var(--text-muted)" }}>
                          lr: {isFiniteNumber(lr) ? lr.toExponential(1) : "-"},{" "}
                          ent: {record.hyperparameters?.entropy_coef ?? "-"},{" "}
                          gae: {record.hyperparameters?.gae_lambda ?? "-"}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
                </table>
              </div>
            )}
          </div>
        </>
      )}
    </div>
  );
}

function useMemoPeakSuccess(history: TelemetryRecord[]): number {
  return React.useMemo(() => {
    const valid = history.map((r) => r.success_rate).filter(isFiniteNumber);
    if (!valid.length) return 0;
    return Math.max(...valid);
  }, [history]);
}

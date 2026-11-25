import React, { useEffect, useState, useCallback, useMemo, useRef } from "react";
import type { StageBottleneckReport, ZoneMetricSummary } from "../state/types";
import { loadStageDiagnostics } from "../lib/api";

interface StageBottlenecksPanelProps {
  currentStage: number;
  runId?: string | null;
}

const ZONE_GRID_LAYOUT: { id: string; label: string; row: number; col: number; isCorner?: boolean; isWall?: boolean }[] = [
  { id: "top_left", label: "Top-Left Corner", row: 0, col: 0, isCorner: true },
  { id: "top_wall", label: "Top Wall", row: 0, col: 1, isWall: true },
  { id: "top_right", label: "Top-Right Corner", row: 0, col: 2, isCorner: true },
  { id: "left_wall", label: "Left Wall", row: 1, col: 0, isWall: true },
  { id: "center", label: "Center (Open Field)", row: 1, col: 1 },
  { id: "right_wall", label: "Right Wall", row: 1, col: 2, isWall: true },
  { id: "bottom_left", label: "Bottom-Left Corner", row: 2, col: 0, isCorner: true },
  { id: "bottom_wall", label: "Bottom Wall", row: 2, col: 1, isWall: true },
  { id: "bottom_right", label: "Bottom-Right Corner", row: 2, col: 2, isCorner: true },
];

export const StageBottlenecksPanel: React.FC<StageBottlenecksPanelProps> = ({
  currentStage,
  runId,
}) => {
  const [selectedStage, setSelectedStage] = useState<number>(currentStage || 1);
  const [report, setReport] = useState<StageBottleneckReport | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const requestSeqRef = useRef<number>(0);
  const isMountedRef = useRef<boolean>(true);

  useEffect(() => {
    isMountedRef.current = true;
    return () => {
      isMountedRef.current = false;
      requestSeqRef.current += 1;
    };
  }, []);

  const prevStageRef = useRef<number>(currentStage);
  // Sync selected stage only if currentStage actually changes to a new stage
  useEffect(() => {
    if (currentStage && currentStage > 0 && currentStage !== prevStageRef.current) {
      prevStageRef.current = currentStage;
      setSelectedStage(currentStage);
    }
  }, [currentStage]);

  const stageOptions = useMemo(() => {
    const stages = new Set<number>();
    for (let s = 1; s <= 38; s += 1) {
      stages.add(s);
    }
    if (currentStage > 0) stages.add(currentStage);
    if (selectedStage > 0) stages.add(selectedStage);
    return Array.from(stages).sort((a, b) => a - b);
  }, [currentStage, selectedStage]);

  const fetchDiagnostics = useCallback(async () => {
    const reqSeq = ++requestSeqRef.current;
    setLoading(true);
    setError(null);
    try {
      const data = await loadStageDiagnostics(selectedStage, runId || undefined);
      if (!isMountedRef.current || reqSeq !== requestSeqRef.current) return;
      if (data) {
        setReport(data);
      } else {
        setError("Failed to load stage diagnostics data.");
      }
    } catch (err) {
      if (!isMountedRef.current || reqSeq !== requestSeqRef.current) return;
      setError(String(err));
    } finally {
      if (isMountedRef.current && reqSeq === requestSeqRef.current) {
        setLoading(false);
      }
    }
  }, [selectedStage, runId]);

  useEffect(() => {
    void fetchDiagnostics();
    return () => {
      requestSeqRef.current += 1;
    };
  }, [fetchDiagnostics]);

  const getZoneTone = (rate: number, total: number): "empty" | "pass" | "warn" | "fail" => {
    if (total === 0) return "empty";
    if (rate >= 0.75) return "pass";
    if (rate >= 0.45) return "warn";
    return "fail";
  };

  const getRateTone = (rate: number): "pass" | "warn" | "fail" => {
    if (rate >= 0.75) return "pass";
    if (rate >= 0.45) return "warn";
    return "fail";
  };

  const getSeverityVariant = (severity: string): "critical" | "warning" | "info" | "success" => {
    switch (severity) {
      case "critical":
        return "critical";
      case "warning":
        return "warning";
      case "info":
        return "info";
      default:
        return "success";
    }
  };

  const sortedPenStats = useMemo(() => {
    return Object.values(report?.pen_stats ?? {}).sort(
      (a, b) => a.win_rate - b.win_rate || b.total - a.total || a.placement.localeCompare(b.placement)
    );
  }, [report]);

  const sortedSetupStats = useMemo(() => {
    return Object.values(report?.setup_stats ?? {}).sort(
      (a, b) => a.win_rate - b.win_rate || b.total - a.total || a.setup.localeCompare(b.setup)
    );
  }, [report]);

  const overallRateTone = report ? getRateTone(report.overall_success_rate) : "fail";

  return (
    <div className="ui-bottlenecks">
      {/* Header & Stage Selection Controls */}
      <div className="ui-bottlenecks__header">
        <div className="ui-bottlenecks__title-wrap">
          <h2 className="ui-bottlenecks__title">
            Stage Learning Bottlenecks & Spatial Heatmap
          </h2>
          <p className="ui-bottlenecks__subtitle">
            Historical failure and entrapment diagnosis across the lifetime of curriculum stages.
          </p>
        </div>

        <div className="ui-bottlenecks__controls">
          <label htmlFor="bottlenecks-stage-select" className="ui-bottlenecks__stage-label">
            Curriculum Stage:
          </label>
          <select
            id="bottlenecks-stage-select"
            value={selectedStage}
            onChange={(e) => setSelectedStage(Number(e.target.value))}
            className="ui-bottlenecks__select"
          >
            {stageOptions.map((s) => (
              <option key={s} value={s}>
                Stage {s} {s === currentStage ? "(Active)" : ""}
              </option>
            ))}
          </select>
          <button
            type="button"
            onClick={() => void fetchDiagnostics()}
            disabled={loading}
            className="ui-bottlenecks__refresh-btn"
          >
            {loading ? "Analyzing..." : "Refresh"}
          </button>
        </div>
      </div>

      {error && (
        <div className="ui-bottlenecks__error">
          {error}
        </div>
      )}

      {/* Stage Summary Statistics Banner */}
      {report && (
        <div className="ui-bottlenecks__kpi-grid">
          <div className="ui-bottlenecks__kpi-card">
            <div className="ui-bottlenecks__kpi-label">Stage Episodes</div>
            <div className="ui-bottlenecks__kpi-value">{report.total_episodes}</div>
          </div>
          <div className="ui-bottlenecks__kpi-card">
            <div className="ui-bottlenecks__kpi-label">Overall Win Rate</div>
            <div className={`ui-bottlenecks__kpi-value ui-bottlenecks__kpi-value--${overallRateTone}`}>
              {`${Math.round(report.overall_success_rate * 100)}%`}
            </div>
          </div>
          <div className="ui-bottlenecks__kpi-card">
            <div className="ui-bottlenecks__kpi-label">Corner Stuck Fails</div>
            <div className={`ui-bottlenecks__kpi-value ${report.corner_stuck_count > 0 ? "ui-bottlenecks__kpi-value--fail" : ""}`}>
              {report.corner_stuck_count}
            </div>
          </div>
          <div className="ui-bottlenecks__kpi-card">
            <div className="ui-bottlenecks__kpi-label">Avg Corner Time</div>
            <div className="ui-bottlenecks__kpi-value">
              {`${Math.round(report.avg_corner_time_pct * 100)}%`}
            </div>
          </div>
          <div className="ui-bottlenecks__kpi-card">
            <div className="ui-bottlenecks__kpi-label">Avg Wall Time</div>
            <div className="ui-bottlenecks__kpi-value">
              {`${Math.round(report.avg_wall_time_pct * 100)}%`}
            </div>
          </div>
          <div className="ui-bottlenecks__kpi-card">
            <div className="ui-bottlenecks__kpi-label">Avg Episode Steps</div>
            <div className="ui-bottlenecks__kpi-value">{report.avg_steps}</div>
          </div>
        </div>
      )}

      {/* Identified Root Cause Bottleneck Insights */}
      {report && report.insights && report.insights.length > 0 && (
        <div className="ui-bottlenecks__section">
          <h3 className="ui-bottlenecks__section-title">
            Identified Bottlenecks & Diagnostic Insights
          </h3>
          <div className="ui-bottlenecks__insights-grid">
            {report.insights.map((insight, idx) => (
              <div key={idx} className="ui-bottlenecks__insight-card">
                <div className="ui-bottlenecks__insight-header">
                  <span className="ui-bottlenecks__insight-title">
                    {insight.title}
                  </span>
                  <span className={`ui-bottlenecks__severity ui-bottlenecks__severity--${getSeverityVariant(insight.severity)}`}>
                    {insight.severity}
                  </span>
                </div>
                <p className="ui-bottlenecks__insight-message">{insight.message}</p>
                {insight.metric && (
                  <div className="ui-bottlenecks__insight-metric">
                    {insight.metric}
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* 2D Spatial Arena Heatmap Matrix */}
      {report && report.zone_stats && (
        <div className="ui-bottlenecks__section">
          <div className="ui-bottlenecks__section-header">
            <h3 className="ui-bottlenecks__section-title">
              Spatial Zone Success Matrix (Initial Sheep Positions)
            </h3>
            <span className="ui-bottlenecks__section-meta">
              Discretized 9-Zone Field Matrix
            </span>
          </div>

          <div className="ui-bottlenecks__zone-grid">
            {ZONE_GRID_LAYOUT.map((layout) => {
              const stat: ZoneMetricSummary | undefined = report.zone_stats[layout.id];
              const total = stat ? stat.total : 0;
              const wins = stat ? stat.wins : 0;
              const winRate = stat ? stat.win_rate : 0.0;
              const trapped = stat ? stat.trapped_at_end : 0;
              const tone = getZoneTone(winRate, total);

              return (
                <div
                  key={layout.id}
                  className={`ui-bottlenecks__zone-card ui-bottlenecks__zone-card--${tone}`}
                >
                  <div>
                    <div className="ui-bottlenecks__zone-header">
                      <span className="ui-bottlenecks__zone-label">{layout.label}</span>
                      {layout.isCorner && (
                        <span className="ui-bottlenecks__zone-tag ui-bottlenecks__zone-tag--corner">
                          Corner
                        </span>
                      )}
                      {layout.isWall && (
                        <span className="ui-bottlenecks__zone-tag ui-bottlenecks__zone-tag--wall">
                          Wall
                        </span>
                      )}
                    </div>

                    <div className="ui-bottlenecks__zone-metrics">
                      <span className="ui-bottlenecks__zone-rate">
                        {total > 0 ? `${Math.round(winRate * 100)}%` : "—"}
                      </span>
                      <span className="ui-bottlenecks__zone-count">
                        {wins} / {total} wins
                      </span>
                    </div>

                    {total > 0 && (
                      <div className="ui-bottlenecks__zone-bar">
                        <div
                          className={`ui-bottlenecks__zone-bar-fill ui-bottlenecks__zone-bar-fill--${tone}`}
                          style={{ width: `${Math.round(winRate * 100)}%` }}
                        />
                      </div>
                    )}
                  </div>

                  <div className="ui-bottlenecks__zone-footer">
                    <span>
                      Trapped fails:{" "}
                      <strong className={trapped > 0 ? "ui-bottlenecks__trapped--fail" : ""}>
                        {trapped}
                      </strong>
                    </span>
                    {stat && stat.avg_corner_pct > 0 && (
                      <span>
                        Corner: <strong>{Math.round(stat.avg_corner_pct * 100)}%</strong>
                      </span>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Pen Placement & Setup Mix Breakdown Tables (worst win-rate first) */}
      {report && (
        <div className="ui-bottlenecks__tables-grid">
          {/* Pen Placement Breakdown */}
          <div className="ui-bottlenecks__table-section">
            <h4 className="ui-bottlenecks__table-title">
              Pen Placement Performance
            </h4>
            <div className="ui-bottlenecks__table-wrap">
              <table className="ui-bottlenecks__table">
                <thead>
                  <tr>
                    <th>Pen Location</th>
                    <th>Episodes</th>
                    <th>Wins</th>
                    <th>Win %</th>
                  </tr>
                </thead>
                <tbody>
                  {sortedPenStats.length === 0 ? (
                    <tr>
                      <td colSpan={4} className="ui-bottlenecks__table-empty">
                        No pen placement records recorded.
                      </td>
                    </tr>
                  ) : (
                    sortedPenStats.map((p) => (
                      <tr key={p.placement}>
                        <td className="ui-bottlenecks__cell-label">
                          {p.placement.replace(/_/g, " ")}
                        </td>
                        <td className="ui-bottlenecks__cell-num">{p.total}</td>
                        <td className="ui-bottlenecks__cell-num">{p.wins}</td>
                        <td className={`ui-bottlenecks__rate ui-bottlenecks__rate--${getRateTone(p.win_rate)}`}>
                          {`${Math.round(p.win_rate * 100)}%`}
                        </td>
                      </tr>
                    ))
                  )}
                </tbody>
              </table>
            </div>
          </div>

          {/* Setup / Spawn Mode Breakdown */}
          <div className="ui-bottlenecks__table-section">
            <h4 className="ui-bottlenecks__table-title">
              Setup & Spawn Mode Performance
            </h4>
            <div className="ui-bottlenecks__table-wrap">
              <table className="ui-bottlenecks__table">
                <thead>
                  <tr>
                    <th>Setup / Mode</th>
                    <th>Episodes</th>
                    <th>Wins</th>
                    <th>Win %</th>
                  </tr>
                </thead>
                <tbody>
                  {sortedSetupStats.length === 0 ? (
                    <tr>
                      <td colSpan={4} className="ui-bottlenecks__table-empty">
                        No setup configuration records recorded.
                      </td>
                    </tr>
                  ) : (
                    sortedSetupStats.map((s) => (
                      <tr key={s.setup}>
                        <td className="ui-bottlenecks__cell-label">
                          {s.setup.replace(/_/g, " ")}
                        </td>
                        <td className="ui-bottlenecks__cell-num">{s.total}</td>
                        <td className="ui-bottlenecks__cell-num">{s.wins}</td>
                        <td className={`ui-bottlenecks__rate ui-bottlenecks__rate--${getRateTone(s.win_rate)}`}>
                          {`${Math.round(s.win_rate * 100)}%`}
                        </td>
                      </tr>
                    ))
                  )}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

import React, { useEffect, useState, useCallback, useRef } from "react";
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

  const prevStageRef = useRef<number>(currentStage);
  // Sync selected stage only if currentStage actually changes to a new stage
  useEffect(() => {
    if (currentStage && currentStage > 0 && currentStage !== prevStageRef.current) {
      prevStageRef.current = currentStage;
      setSelectedStage(currentStage);
    }
  }, [currentStage]);

  const fetchDiagnostics = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await loadStageDiagnostics(selectedStage, runId || undefined);
      if (data) {
        setReport(data);
      } else {
        setError("Failed to load stage diagnostics data.");
      }
    } catch (err) {
      setError(String(err));
    } finally {
      setLoading(false);
    }
  }, [selectedStage, runId]);

  useEffect(() => {
    void fetchDiagnostics();
  }, [fetchDiagnostics]);

  const getWinRateColor = (rate: number, total: number): { bg: string; border: string; text: string } => {
    if (total === 0) return { bg: "bg-slate-900/40", border: "border-slate-800", text: "text-slate-500" };
    if (rate >= 0.75) return { bg: "bg-emerald-950/40", border: "border-emerald-700/60", text: "text-emerald-400" };
    if (rate >= 0.45) return { bg: "bg-amber-950/40", border: "border-amber-700/60", text: "text-amber-400" };
    return { bg: "bg-rose-950/50", border: "border-rose-700/70", text: "text-rose-400" };
  };

  const getSeverityBadge = (severity: string) => {
    switch (severity) {
      case "critical":
        return "bg-rose-500/20 text-rose-300 border-rose-500/40";
      case "warning":
        return "bg-amber-500/20 text-amber-300 border-amber-500/40";
      case "info":
        return "bg-sky-500/20 text-sky-300 border-sky-500/40";
      default:
        return "bg-emerald-500/20 text-emerald-300 border-emerald-500/40";
    }
  };

  return (
    <div className="space-y-6 mt-6 p-6 bg-slate-950 border border-slate-800 rounded-xl shadow-2xl text-slate-200">
      {/* Header & Stage Selection Controls */}
      <div className="flex flex-wrap items-center justify-between gap-4 border-b border-slate-800 pb-4">
        <div>
          <div className="flex items-center gap-3">
            <span className="text-2xl">🎯</span>
            <h2 className="text-xl font-bold tracking-tight text-slate-100">
              Stage Learning Bottlenecks & Spatial Heatmap
            </h2>
          </div>
          <p className="text-xs text-slate-400 mt-1">
            Historical failure and entrapment diagnosis across the lifetime of curriculum stages.
          </p>
        </div>

        <div className="flex items-center gap-3">
          <label className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
            Curriculum Stage:
          </label>
          <select
            value={selectedStage}
            onChange={(e) => setSelectedStage(Number(e.target.value))}
            className="bg-slate-900 border border-slate-700 text-slate-200 text-sm rounded-lg px-3 py-1.5 focus:ring-2 focus:ring-indigo-500 focus:outline-none"
          >
            {[1, 2, 3, 4, 5, 6, 7, 8].map((s) => (
              <option key={s} value={s}>
                Stage {s} {s === currentStage ? "(Active)" : ""}
              </option>
            ))}
          </select>
          <button
            onClick={() => void fetchDiagnostics()}
            disabled={loading}
            className="px-3 py-1.5 bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-medium rounded-lg border border-slate-700 transition flex items-center gap-1.5 disabled:opacity-50"
          >
            {loading ? "Analyzing..." : "Refresh"}
          </button>
        </div>
      </div>

      {error && (
        <div className="p-4 bg-rose-950/40 border border-rose-800 text-rose-300 rounded-lg text-sm">
          {error}
        </div>
      )}

      {/* Stage Summary Statistics Banner */}
      {report && (
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
          <div className="p-3 bg-slate-900/60 border border-slate-800 rounded-lg">
            <div className="text-[11px] font-semibold text-slate-400 uppercase">Stage Episodes</div>
            <div className="text-xl font-bold text-slate-100 mt-0.5">{report.total_episodes}</div>
          </div>
          <div className="p-3 bg-slate-900/60 border border-slate-800 rounded-lg">
            <div className="text-[11px] font-semibold text-slate-400 uppercase">Overall Win Rate</div>
            <div className={`text-xl font-bold mt-0.5 ${report.overall_success_rate >= 0.75 ? "text-emerald-400" : report.overall_success_rate >= 0.45 ? "text-amber-400" : "text-rose-400"}`}>
              {`${Math.round(report.overall_success_rate * 100)}%`}
            </div>
          </div>
          <div className="p-3 bg-slate-900/60 border border-slate-800 rounded-lg">
            <div className="text-[11px] font-semibold text-slate-400 uppercase">Corner Stuck Fails</div>
            <div className="text-xl font-bold text-rose-400 mt-0.5">{report.corner_stuck_count}</div>
          </div>
          <div className="p-3 bg-slate-900/60 border border-slate-800 rounded-lg">
            <div className="text-[11px] font-semibold text-slate-400 uppercase">Avg Corner Time</div>
            <div className="text-xl font-bold text-slate-100 mt-0.5">
              {`${Math.round(report.avg_corner_time_pct * 100)}%`}
            </div>
          </div>
          <div className="p-3 bg-slate-900/60 border border-slate-800 rounded-lg">
            <div className="text-[11px] font-semibold text-slate-400 uppercase">Avg Wall Time</div>
            <div className="text-xl font-bold text-slate-100 mt-0.5">
              {`${Math.round(report.avg_wall_time_pct * 100)}%`}
            </div>
          </div>
          <div className="p-3 bg-slate-900/60 border border-slate-800 rounded-lg">
            <div className="text-[11px] font-semibold text-slate-400 uppercase">Avg Episode Steps</div>
            <div className="text-xl font-bold text-slate-100 mt-0.5">{report.avg_steps}</div>
          </div>
        </div>
      )}

      {/* Identified Root Cause Bottleneck Insights */}
      {report && report.insights && report.insights.length > 0 && (
        <div className="space-y-3">
          <h3 className="text-sm font-bold uppercase tracking-wider text-slate-400 flex items-center gap-2">
            <span>🔍</span> Identified Bottlenecks & Diagnostic Insights
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {report.insights.map((insight, idx) => (
              <div
                key={idx}
                className="p-4 bg-slate-900/80 border border-slate-800/80 rounded-xl space-y-2 relative overflow-hidden"
              >
                <div className="flex items-center justify-between gap-2">
                  <span className="font-semibold text-slate-200 text-sm flex items-center gap-2">
                    {insight.severity === "critical" && "🚨"}
                    {insight.severity === "warning" && "⚠️"}
                    {insight.severity === "info" && "ℹ️"}
                    {insight.severity === "success" && "✅"}
                    {insight.title}
                  </span>
                  <span className={`text-[10px] font-bold px-2 py-0.5 rounded-full border uppercase ${getSeverityBadge(insight.severity)}`}>
                    {insight.severity}
                  </span>
                </div>
                <p className="text-xs text-slate-300 leading-relaxed">{insight.message}</p>
                {insight.metric && (
                  <div className="text-[11px] font-mono text-indigo-300 bg-indigo-950/40 px-2 py-1 rounded border border-indigo-800/40 inline-block">
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
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-bold uppercase tracking-wider text-slate-400 flex items-center gap-2">
              <span>🗺️</span> Spatial Zone Success Matrix (Initial Sheep Positions)
            </h3>
            <span className="text-xs text-slate-500">
              Discretized 9-Zone Field Matrix
            </span>
          </div>

          <div className="grid grid-cols-3 gap-3">
            {ZONE_GRID_LAYOUT.map((layout) => {
              const stat: ZoneMetricSummary | undefined = report.zone_stats[layout.id];
              const total = stat ? stat.total : 0;
              const wins = stat ? stat.wins : 0;
              const winRate = stat ? stat.win_rate : 0.0;
              const trapped = stat ? stat.trapped_at_end : 0;
              const colors = getWinRateColor(winRate, total);

              return (
                <div
                  key={layout.id}
                  className={`p-4 rounded-xl border transition duration-200 ${colors.bg} ${colors.border} flex flex-col justify-between min-h-[140px]`}
                >
                  <div>
                    <div className="flex items-center justify-between gap-1 mb-1">
                      <span className="text-xs font-bold text-slate-200 truncate">{layout.label}</span>
                      {layout.isCorner && (
                        <span className="text-[10px] px-1.5 py-0.2 bg-purple-950 text-purple-300 border border-purple-800 rounded font-semibold uppercase">
                          Corner
                        </span>
                      )}
                      {layout.isWall && (
                        <span className="text-[10px] px-1.5 py-0.2 bg-blue-950 text-blue-300 border border-blue-800 rounded font-semibold uppercase">
                          Wall
                        </span>
                      )}
                    </div>

                    <div className="mt-2 flex items-baseline justify-between">
                      <span className={`text-2xl font-black ${colors.text}`}>
                        {total > 0 ? `${Math.round(winRate * 100)}%` : "—"}
                      </span>
                      <span className="text-xs text-slate-400 font-medium">
                        {wins} / {total} wins
                      </span>
                    </div>

                    {total > 0 && (
                      <div className="w-full bg-slate-800/80 rounded-full h-1.5 mt-2 overflow-hidden">
                        <div
                          className={`h-1.5 rounded-full ${winRate >= 0.75 ? "bg-emerald-500" : winRate >= 0.45 ? "bg-amber-500" : "bg-rose-500"}`}
                          style={{ width: `${Math.round(winRate * 100)}%` }}
                        />
                      </div>
                    )}
                  </div>

                  <div className="mt-3 pt-2 border-t border-slate-800/60 flex items-center justify-between text-[11px] text-slate-400">
                    <span>
                      Trapped fails: <strong className={trapped > 0 ? "text-rose-400 font-bold" : "text-slate-300"}>{trapped}</strong>
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

      {/* Pen Placement & Setup Mix Breakdown Tables */}
      {report && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 pt-2">
          {/* Pen Placement Breakdown */}
          <div className="space-y-3">
            <h4 className="text-xs font-bold uppercase tracking-wider text-slate-400">
              Pen Placement Performance
            </h4>
            <div className="overflow-hidden border border-slate-800 rounded-xl bg-slate-900/40">
              <table className="w-full text-left text-xs">
                <thead className="bg-slate-900/90 text-slate-400 border-b border-slate-800 uppercase font-semibold">
                  <tr>
                    <th className="py-2.5 px-3">Pen Location</th>
                    <th className="py-2.5 px-3">Episodes</th>
                    <th className="py-2.5 px-3">Wins</th>
                    <th className="py-2.5 px-3">Win %</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/60">
                  {Object.values(report.pen_stats).length === 0 ? (
                    <tr>
                      <td colSpan={4} className="py-4 text-center text-slate-500">
                        No pen placement records recorded.
                      </td>
                    </tr>
                  ) : (
                    Object.values(report.pen_stats).map((p, idx) => (
                      <tr key={idx} className="hover:bg-slate-900/60 transition">
                        <td className="py-2.5 px-3 font-medium text-slate-200 capitalize">
                          {p.placement.replace("_", " ")}
                        </td>
                        <td className="py-2.5 px-3 text-slate-400">{p.total}</td>
                        <td className="py-2.5 px-3 text-slate-300">{p.wins}</td>
                        <td className={`py-2.5 px-3 font-bold ${p.win_rate >= 0.75 ? "text-emerald-400" : p.win_rate >= 0.45 ? "text-amber-400" : "text-rose-400"}`}>
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
          <div className="space-y-3">
            <h4 className="text-xs font-bold uppercase tracking-wider text-slate-400">
              Setup & Spawn Mode Performance
            </h4>
            <div className="overflow-hidden border border-slate-800 rounded-xl bg-slate-900/40">
              <table className="w-full text-left text-xs">
                <thead className="bg-slate-900/90 text-slate-400 border-b border-slate-800 uppercase font-semibold">
                  <tr>
                    <th className="py-2.5 px-3">Setup / Mode</th>
                    <th className="py-2.5 px-3">Episodes</th>
                    <th className="py-2.5 px-3">Wins</th>
                    <th className="py-2.5 px-3">Win %</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/60">
                  {Object.values(report.setup_stats).length === 0 ? (
                    <tr>
                      <td colSpan={4} className="py-4 text-center text-slate-500">
                        No setup configuration records recorded.
                      </td>
                    </tr>
                  ) : (
                    Object.values(report.setup_stats).map((s, idx) => (
                      <tr key={idx} className="hover:bg-slate-900/60 transition">
                        <td className="py-2.5 px-3 font-medium text-slate-200 capitalize">
                          {s.setup.replace("_", " ")}
                        </td>
                        <td className="py-2.5 px-3 text-slate-400">{s.total}</td>
                        <td className="py-2.5 px-3 text-slate-300">{s.wins}</td>
                        <td className={`py-2.5 px-3 font-bold ${s.win_rate >= 0.75 ? "text-emerald-400" : s.win_rate >= 0.45 ? "text-amber-400" : "text-rose-400"}`}>
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

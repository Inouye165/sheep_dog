import { useCallback, useEffect, useMemo, useState } from "react";
import { loadModelHistory, loadStageTimings } from "../lib/api";
import type {
  ModelHistoryResponse,
  ModelPassRecord,
  ModelStageComparisonDelta,
  ModelStageComparisonRun,
  StageTimingRecord,
  StageTimingsResponse,
} from "../state/types";
import { ExplainTip } from "./ExplainTip";

type ChartMetric = "wall_seconds" | "training_seconds" | "policy_timesteps";

const METRIC_LABELS: Record<ChartMetric, string> = {
  wall_seconds: "Wall time (seconds)",
  training_seconds: "Training time (seconds)",
  policy_timesteps: "Policy timesteps (PPO steps)",
};

const SERIES_COLORS = ["#38bdf8", "#f97316", "#a78bfa", "#22c55e", "#f43f5e", "#eab308"];

export function formatDuration(seconds: number | null | undefined): string {
  if (seconds === null || seconds === undefined || !Number.isFinite(seconds)) return "—";
  const total = Math.round(seconds);
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;
  if (h > 0) return `${h}h ${m}m ${s}s`;
  if (m > 0) return `${m}m ${s}s`;
  return `${s}s`;
}

function formatNumber(value: number | null | undefined, digits = 0): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "—";
  return value.toLocaleString(undefined, { maximumFractionDigits: digits });
}

function formatPct(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "—";
  return `${value > 0 ? "+" : ""}${value.toFixed(1)}%`;
}

function formatSuccessRate(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "—";
  return `${Math.round(value <= 1 ? value * 100 : value)}%`;
}

function shortRun(runId: string | null): string {
  return runId ? runId.replace(/^run_/, "") : "unknown";
}

function passDisplayName(pass: Pick<ModelPassRecord, "pass_id" | "label">): string {
  return pass.label?.trim() ? pass.label.trim() : pass.pass_id;
}

function statusBadgeClass(status: string): string {
  const lower = status.toLowerCase();
  if (lower === "completed" || lower === "promoted" || lower === "mastered") {
    return "ui-badge--pass";
  }
  if (lower === "running") {
    return "ui-badge--accent";
  }
  if (lower === "failed" || lower === "stopped" || lower === "interrupted") {
    return "ui-badge--warn";
  }
  return "";
}

function renderDeltaSummary(delta: ModelStageComparisonDelta) {
  return (
    <div className="stage-lineage-delta ui-mono">
      <span>Active: {formatPct(delta.active_seconds_pct)}</span>
      <span>Train: {formatPct(delta.training_seconds_pct)}</span>
      {delta.policy_timesteps_pct !== null && delta.policy_timesteps_pct !== undefined ? (
        <span>Steps: {formatPct(delta.policy_timesteps_pct)}</span>
      ) : null}
    </div>
  );
}

function renderComparisonRunCell(run: ModelStageComparisonRun) {
  return (
    <div key={`${run.pass_id}-${run.sequence_order}`} className="stage-lineage-cell">
      <div className="stage-lineage-cell__row">
        <span className="ui-mono">{formatDuration(run.active_seconds)}</span>
        <span className="stage-lineage-cell__muted">active</span>
        <span className="ui-mono">{formatDuration(run.training_seconds)}</span>
        <span className="stage-lineage-cell__muted">train</span>
      </div>
      <div className="stage-lineage-cell__row">
        <span className="ui-mono">{formatNumber(run.policy_timesteps)} steps</span>
        <span className="ui-mono">
          {formatSuccessRate(run.initial_success_rate)} → {formatSuccessRate(run.final_success_rate)}
        </span>
      </div>
      <div className="stage-lineage-cell__row">
        <span className={`ui-badge ${statusBadgeClass(run.status)}`}>{run.status}</span>
      </div>
    </div>
  );
}

async function loadModelHistorySafe(
  passId: string | undefined,
  stage: number | null,
): Promise<ModelHistoryResponse | null> {
  try {
    return typeof loadModelHistory === "function"
      ? await loadModelHistory({ passId, stage })
      : null;
  } catch {
    return null;
  }
}

export function StageTimingPanel({ stage = null }: { stage?: number | null }) {
  const [data, setData] = useState<StageTimingsResponse | null>(null);
  const [modelHistory, setModelHistory] = useState<ModelHistoryResponse | null>(null);
  const [selectedPassId, setSelectedPassId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [metric, setMetric] = useState<ChartMetric>("wall_seconds");

  const refresh = useCallback(async () => {
    const [timingsResult, historyResult] = await Promise.allSettled([
      loadStageTimings(),
      loadModelHistorySafe(selectedPassId ?? undefined, stage),
    ]);
    if (timingsResult.status === "fulfilled") {
      setData(timingsResult.value);
      setError(null);
    } else {
      setError(
        timingsResult.reason instanceof Error
          ? timingsResult.reason.message
          : "Failed to load stage timings",
      );
    }
    if (historyResult.status === "fulfilled") {
      setModelHistory(historyResult.value);
    } else {
      setModelHistory(null);
    }
  }, [selectedPassId, stage]);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const passes = useMemo(() => modelHistory?.passes ?? [], [modelHistory]);
  const passesById = useMemo(() => {
    const map = new Map<string, ModelPassRecord>();
    for (const pass of passes) {
      map.set(pass.pass_id, pass);
    }
    return map;
  }, [passes]);

  const selectedLineage = modelHistory?.selected_lineage ?? null;
  const inspectedPassId =
    selectedPassId ?? selectedLineage?.target_pass_id ?? modelHistory?.active_pass_id ?? null;

  const comparisonRows = useMemo(() => {
    const rows = selectedLineage?.stage_comparison ?? [];
    return stage === null ? rows : rows.filter((row) => row.stage === stage);
  }, [selectedLineage, stage]);

  const records = useMemo(
    () => (data?.records ?? []).filter((rec) => stage === null || rec.stage === stage),
    [data, stage],
  );
  const runs = useMemo(() => {
    const seen: string[] = [];
    for (const rec of records) {
      const id = rec.run_id ?? "unknown";
      if (!seen.includes(id)) seen.push(id);
    }
    return seen;
  }, [records]);

  const series = useMemo(() => {
    return runs.map((run, index) => {
      const points = records
        .filter((rec) => (rec.run_id ?? "unknown") === run && typeof rec[metric] === "number")
        .map((rec) => ({ stage: rec.stage, value: rec[metric] as number }))
        .sort((a, b) => a.stage - b.stage);
      return { run, color: SERIES_COLORS[index % SERIES_COLORS.length], points };
    });
  }, [records, runs, metric]);

  const chart = useMemo(() => {
    const all = series.flatMap((s) => s.points);
    if (all.length === 0) return null;
    const minStage = Math.min(...all.map((p) => p.stage));
    const maxStage = Math.max(...all.map((p) => p.stage));
    const maxValue = Math.max(...all.map((p) => p.value), 1);
    const W = 560;
    const H = 220;
    const pad = 36;
    const x = (stageValue: number) =>
      maxStage === minStage
        ? W / 2
        : pad + ((stageValue - minStage) / (maxStage - minStage)) * (W - pad * 2);
    const y = (value: number) => H - pad - (value / maxValue) * (H - pad * 2);
    return { W, H, pad, x, y, minStage, maxStage, maxValue };
  }, [series]);

  const latestComparisons = (data?.comparison ?? []).filter((c) => c.vs_previous);

  const sortedRows: StageTimingRecord[] = useMemo(
    () => [...records].sort((a, b) => String(b.ended_at ?? "").localeCompare(String(a.ended_at ?? ""))),
    [records],
  );

  return (
    <section aria-labelledby="stage-timing-heading" className="ui-panel stage-timing-panel">
      <div className="ui-panel__header">
        <div>
          <h3 id="stage-timing-heading" className="ui-panel__title">
            Stage timings: how long each stage took to train
          </h3>
          <p className="ui-panel__subtitle">
            Cross-run promotion durations and policy-step efficiency
          </p>
        </div>
        <div className="ui-control-row">
          <ExplainTip
            title="How Stage Timing Works"
            buttonText="How timing works"
            align="right"
            plain="Each time a curriculum stage is completed (promoted), the time it took is saved so you can compare runs and see how a change affects learning speed."
            technical={[
              "Training time counts only the training loop; it excludes pauses, stops, offline gaps, and evaluation.",
              "Wall time is the clock time between a stage's start and its promotion and includes pauses, stops, offline gaps, and evaluation.",
              "Steps/s = policy timesteps per second of training time.",
              "Set SHEEPDOG_RUN_LABEL (for example baseline) before starting the backend to name a run.",
              "Rows marked derived were reconstructed from older promotion logs (wall time only; no training time, config, or assistance info) and are not comparable with recorded rows.",
            ]}
          />
          <button type="button" className="ui-btn--sm" onClick={() => void refresh()}>
            Refresh timings
          </button>
        </div>
      </div>

      <div className="ui-panel__body ui-control-stack">
        {error ? (
          <p role="alert" className="ui-inline-error">
            {error}
          </p>
        ) : null}

        {passes.length > 0 ? (
          <section aria-labelledby="model-passes-heading" className="stage-lineage-section">
            <div className="stage-lineage-section__header">
              <h4 id="model-passes-heading" className="stage-lineage-section__title">
                Model Training Passes &amp; Lineage
              </h4>
              <span className="ui-panel__subtitle">
                {passes.length} recorded {passes.length === 1 ? "pass" : "passes"}
              </span>
            </div>

            <div className="ui-table-wrap">
              <table className="ui-table" aria-label="Recorded Model Passes">
                <caption>Recorded Model Passes</caption>
                <thead>
                  <tr>
                    <th scope="col">Pass</th>
                    <th scope="col">Parent pass</th>
                    <th scope="col">Stages</th>
                    <th scope="col">Completed</th>
                    <th scope="col">Status</th>
                    <th scope="col">Active time</th>
                    <th scope="col">Training time</th>
                    <th scope="col">Eval time</th>
                    <th scope="col">Wall time</th>
                    <th scope="col">Policy timesteps</th>
                    <th scope="col">Assistance</th>
                    <th scope="col">Lineage</th>
                  </tr>
                </thead>
                <tbody>
                  {passes.map((pass) => {
                    const parent = pass.parent_pass_id
                      ? passesById.get(pass.parent_pass_id)
                      : undefined;
                    const parentText = !pass.parent_pass_id
                      ? "fresh init"
                      : parent
                        ? passDisplayName(parent)
                        : pass.parent_pass_id;
                    const isInspected = inspectedPassId === pass.pass_id;
                    return (
                      <tr
                        key={pass.pass_id}
                        className={isInspected ? "stage-lineage-row--selected" : undefined}
                      >
                        <td>
                          <div>{passDisplayName(pass)}</div>
                          {pass.label ? (
                            <div className="ui-mono stage-lineage-cell__muted">{pass.pass_id}</div>
                          ) : null}
                          {pass.resulting_checkpoint_sha256 ? (
                            <div className="ui-mono stage-lineage-cell__muted">
                              sha {pass.resulting_checkpoint_sha256.slice(0, 7)}
                            </div>
                          ) : null}
                        </td>
                        <td>{parentText}</td>
                        <td className="ui-mono">{pass.stage_spec}</td>
                        <td className="ui-mono">
                          {pass.completed_stages_count} / {pass.planned_stages.length}
                        </td>
                        <td>
                          <span className={`ui-badge ${statusBadgeClass(pass.status)}`}>
                            {pass.status}
                          </span>
                        </td>
                        <td className="ui-mono">{formatDuration(pass.total_active_seconds)}</td>
                        <td className="ui-mono">{formatDuration(pass.total_training_seconds)}</td>
                        <td className="ui-mono">{formatDuration(pass.total_evaluation_seconds)}</td>
                        <td className="ui-mono">{formatDuration(pass.total_wall_seconds)}</td>
                        <td className="ui-mono">{formatNumber(pass.total_policy_timesteps)}</td>
                        <td>{pass.assistance_mode ?? "—"}</td>
                        <td>
                          <button
                            type="button"
                            className={`ui-btn--sm${isInspected ? " ui-btn--primary" : ""}`}
                            aria-pressed={isInspected}
                            onClick={() => setSelectedPassId(pass.pass_id)}
                          >
                            Inspect lineage
                          </button>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>

            {selectedLineage ? (
              <div className="stage-lineage-summary" aria-label="Selected Model Lineage Summary">
                <div className="stage-lineage-breadcrumb">
                  <span className="stage-lineage-breadcrumb__label">Selected Model Lineage:</span>
                  <span className="ui-mono">
                    {selectedLineage.lineage.map((p) => passDisplayName(p)).join(" → ")}
                  </span>
                </div>

                <div className="ui-kpi-grid ui-kpi-grid--4">
                  <div className="ui-kpi-card">
                    <span className="ui-kpi-card__label">Passes in lineage</span>
                    <span className="ui-kpi-card__value">
                      {selectedLineage.cumulative.pass_count}
                    </span>
                    <span className="ui-kpi-card__sub">
                      {selectedLineage.cumulative.completed_stage_runs_count} /{" "}
                      {selectedLineage.cumulative.stage_runs_count} stage runs
                    </span>
                  </div>
                  <div className="ui-kpi-card">
                    <span className="ui-kpi-card__label">Cumulative Active</span>
                    <span className="ui-kpi-card__value">
                      {formatDuration(selectedLineage.cumulative.total_active_seconds)}
                    </span>
                    <span className="ui-kpi-card__sub">
                      Wall {formatDuration(selectedLineage.cumulative.total_wall_seconds)}
                    </span>
                  </div>
                  <div className="ui-kpi-card">
                    <span className="ui-kpi-card__label">Cumulative Training</span>
                    <span className="ui-kpi-card__value">
                      {formatDuration(selectedLineage.cumulative.total_training_seconds)}
                    </span>
                    <span className="ui-kpi-card__sub">
                      Eval {formatDuration(selectedLineage.cumulative.total_evaluation_seconds)}
                    </span>
                  </div>
                  <div className="ui-kpi-card">
                    <span className="ui-kpi-card__label">Cumulative Timesteps</span>
                    <span className="ui-kpi-card__value">
                      {formatNumber(selectedLineage.cumulative.total_policy_timesteps)}
                    </span>
                    <span className="ui-kpi-card__sub">Across selected ancestry</span>
                  </div>
                </div>

                {comparisonRows.length > 0 ? (
                  <div className="ui-table-wrap">
                    <table
                      className="ui-table"
                      aria-label="Stage-by-Stage Progression Across Lineage"
                    >
                      <caption>Stage-by-Stage Progression Across Lineage</caption>
                      <thead>
                        <tr>
                          <th scope="col">Stage</th>
                          {selectedLineage.lineage.map((pass, idx) => (
                            <th key={pass.pass_id} scope="col">
                              {`Pass ${idx + 1}: ${passDisplayName(pass)}`}
                            </th>
                          ))}
                          <th scope="col">Δ vs Previous</th>
                          <th scope="col">Δ vs First</th>
                        </tr>
                      </thead>
                      <tbody>
                        {comparisonRows.map((row) => (
                          <tr key={row.stage}>
                            <td className="ui-mono">{row.stage}</td>
                            {selectedLineage.lineage.map((pass, idx) => {
                              const matchingRuns = row.runs.filter(
                                (r) => r.pass_id === pass.pass_id || r.pass_index === idx,
                              );
                              return (
                                <td key={`${row.stage}-${pass.pass_id}`}>
                                  {matchingRuns.length > 0
                                    ? matchingRuns.map((run) => renderComparisonRunCell(run))
                                    : "—"}
                                </td>
                              );
                            })}
                            <td>{renderDeltaSummary(row.delta_vs_previous)}</td>
                            <td>{renderDeltaSummary(row.delta_vs_first)}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                ) : null}
              </div>
            ) : null}
          </section>
        ) : null}

        {data && records.length === 0 ? (
          <p role="status" className="ui-inline-note">
            <strong>No stage timings recorded yet{stage !== null ? ` for stage ${stage}` : ""}.</strong>{" "}
            Complete a stage promotion to record one. Data is stored in{" "}
            <code>artifacts/stage-timing.jsonl</code>.
          </p>
        ) : null}
        {data && records.length > 0 && runs.length === 1 ? (
          <p role="status" className="ui-inline-note">
            <strong>Only one run recorded; run again to compare.</strong> Data is stored in{" "}
            <code>artifacts/stage-timing.jsonl</code>.
          </p>
        ) : null}

        {records.length > 0 ? (
          <>
            <div role="group" aria-label="Chart metric" className="ui-control-row">
              {(Object.keys(METRIC_LABELS) as ChartMetric[]).map((key) => (
                <button
                  key={key}
                  type="button"
                  className={`ui-btn--sm${metric === key ? " ui-btn--primary" : ""}`}
                  aria-pressed={metric === key}
                  onClick={() => setMetric(key)}
                >
                  {METRIC_LABELS[key]}
                </button>
              ))}
            </div>
            {chart ? (
              <figure style={{ margin: 0 }}>
                <svg
                  role="img"
                  aria-label={`${METRIC_LABELS[metric]} per curriculum stage, one line per run`}
                  viewBox={`0 0 ${chart.W} ${chart.H}`}
                  style={{ width: "100%", maxWidth: 640 }}
                >
                  <line
                    x1={chart.pad}
                    y1={chart.H - chart.pad}
                    x2={chart.W - chart.pad}
                    y2={chart.H - chart.pad}
                    stroke="currentColor"
                    opacity={0.4}
                  />
                  <line
                    x1={chart.pad}
                    y1={chart.pad}
                    x2={chart.pad}
                    y2={chart.H - chart.pad}
                    stroke="currentColor"
                    opacity={0.4}
                  />
                  <text x={4} y={chart.pad} fontSize="10" fill="currentColor">
                    {formatNumber(chart.maxValue)}
                  </text>
                  <text x={chart.W / 2} y={chart.H - 6} fontSize="10" textAnchor="middle" fill="currentColor">
                    Curriculum stage
                  </text>
                  {series.map((s) => (
                    <g key={s.run}>
                      <polyline
                        fill="none"
                        stroke={s.color}
                        strokeWidth={2}
                        points={s.points.map((p) => `${chart.x(p.stage)},${chart.y(p.value)}`).join(" ")}
                      />
                      {s.points.map((p) => (
                        <circle key={p.stage} cx={chart.x(p.stage)} cy={chart.y(p.value)} r={3} fill={s.color}>
                          <title>{`${shortRun(s.run)} stage ${p.stage}: ${formatNumber(p.value, 1)}`}</title>
                        </circle>
                      ))}
                    </g>
                  ))}
                </svg>
                <figcaption style={{ fontSize: "0.8rem", color: "var(--ui-text-secondary, #94a3b8)" }}>
                  {series.map((s) => (
                    <span key={s.run} style={{ marginRight: "1rem" }}>
                      <span style={{ color: s.color }} aria-hidden="true">
                        ●
                      </span>{" "}
                      {shortRun(s.run)}
                    </span>
                  ))}
                </figcaption>
              </figure>
            ) : (
              <p>No values available for {METRIC_LABELS[metric]}.</p>
            )}
          </>
        ) : null}

        {latestComparisons.length > 0 ? (
          <div className="ui-table-wrap">
            <table className="ui-table" aria-label="Latest run versus previous run">
              <caption>Latest run vs previous run, per stage (negative = faster)</caption>
              <thead>
                <tr>
                  <th scope="col">Stage</th>
                  <th scope="col">Wall time Δ</th>
                  <th scope="col">Policy timesteps Δ</th>
                  <th scope="col">Steps/s Δ</th>
                  <th scope="col">Comparable?</th>
                </tr>
              </thead>
              <tbody>
                {latestComparisons.map((c) => {
                  const pair = c.vs_previous!;
                  return (
                    <tr key={c.stage}>
                      <td className="ui-mono">{c.stage}</td>
                      <td className="ui-mono">{formatPct(pair.deltas.wall_seconds?.pct_delta)}</td>
                      <td className="ui-mono">{formatPct(pair.deltas.policy_timesteps?.pct_delta)}</td>
                      <td className="ui-mono">{formatPct(pair.deltas.steps_per_second?.pct_delta)}</td>
                      <td>{pair.comparable ? "yes" : `no: ${pair.notes.join("; ")}`}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        ) : null}

        {sortedRows.length > 0 ? (
          <div className="ui-table-wrap">
            <table className="ui-table" aria-label="Stage timing records">
              <thead>
                <tr>
                  <th scope="col">Run</th>
                  <th scope="col">Stage</th>
                  <th scope="col">Date</th>
                  <th scope="col">Wall time</th>
                  <th scope="col">Training time</th>
                  <th scope="col">Policy timesteps</th>
                  <th scope="col">Steps/s</th>
                  <th scope="col">Success</th>
                  <th scope="col">Label</th>
                  <th scope="col">Git rev</th>
                  <th scope="col">Source</th>
                </tr>
              </thead>
              <tbody>
                {sortedRows.map((rec) => (
                  <tr key={rec.promotion_event_id ?? `${rec.run_id}-${rec.stage}-${rec.ended_at}`}>
                    <td className="ui-mono">{shortRun(rec.run_id)}</td>
                    <td className="ui-mono">{rec.stage}</td>
                    <td>{rec.ended_at ? new Date(rec.ended_at).toLocaleString() : "—"}</td>
                    <td className="ui-mono">{formatDuration(rec.wall_seconds)}</td>
                    <td className="ui-mono">{formatDuration(rec.training_seconds)}</td>
                    <td className="ui-mono">{formatNumber(rec.policy_timesteps)}</td>
                    <td className="ui-mono">{formatNumber(rec.steps_per_second, 1)}</td>
                    <td className="ui-mono">
                      {rec.success_rate === null || rec.success_rate === undefined
                        ? "—"
                        : `${Math.round(rec.success_rate * 100)}%${rec.success_count !== null && rec.success_count !== undefined ? ` (${rec.success_count})` : ""}`}
                    </td>
                    <td>{rec.run_label ?? "—"}</td>
                    <td className="ui-mono">
                      {rec.git_revision ? `${rec.git_revision.slice(0, 7)}${rec.git_dirty ? "+dirty" : ""}` : "—"}
                    </td>
                    <td>
                      <span className={`badge badge--${rec.source}`}>{rec.source}</span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : null}
      </div>
    </section>
  );
}

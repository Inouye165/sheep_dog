import React, { useCallback, useEffect, useRef, useState } from "react";
import { acknowledgeEarlyWarning, loadEarlyWarnings } from "../lib/api";
import type { EarlyWarningItem, EarlyWarningsPayload } from "../state/types";

export interface EarlyWarningsPanelProps {
  curriculumStage?: number;
  isOpen: boolean;
  onToggle: () => void;
  isReadOnlyPreview?: boolean;
  onOpenReplay?: (replayPath: string, step?: number) => void;
}

export function EarlyWarningsPanel({
  curriculumStage = 1,
  isOpen,
  onToggle,
  isReadOnlyPreview = false,
  onOpenReplay,
}: EarlyWarningsPanelProps) {
  const [data, setData] = useState<EarlyWarningsPayload | null>(null);
  const [loading, setLoading] = useState(false);
  const [filterTab, setFilterTab] = useState<"active" | "acknowledged">("active");
  const [ackBusyId, setAckBusyId] = useState<string | null>(null);
  const [ackSuccessId, setAckSuccessId] = useState<string | null>(null);

  const requestSeqRef = useRef<number>(0);
  const mountedRef = useRef<boolean>(true);
  const ackTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const ackInFlightRef = useRef<boolean>(false);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      if (ackTimerRef.current) {
        clearTimeout(ackTimerRef.current);
      }
    };
  }, []);

  const fetchWarnings = useCallback(async (force = false) => {
    if (!force && ackInFlightRef.current) {
      return;
    }
    const reqSeq = ++requestSeqRef.current;
    if (mountedRef.current) {
      setLoading(true);
    }
    try {
      const payload = await loadEarlyWarnings(curriculumStage, force);
      if (mountedRef.current && reqSeq === requestSeqRef.current && payload) {
        setData(payload);
      }
    } finally {
      if (mountedRef.current && reqSeq === requestSeqRef.current) {
        setLoading(false);
      }
    }
  }, [curriculumStage]);

  useEffect(() => {
    void fetchWarnings();
    const timer = window.setInterval(() => {
      void fetchWarnings(false);
    }, 10000);
    return () => window.clearInterval(timer);
  }, [fetchWarnings]);

  const handleAcknowledge = async (warningId: string) => {
    requestSeqRef.current += 1;
    ackInFlightRef.current = true;
    setAckBusyId(warningId);
    try {
      const res = await acknowledgeEarlyWarning(warningId, "Operator acknowledged via Early Warnings panel");
      if (res && res.success) {
        if (mountedRef.current) {
          setAckSuccessId(warningId);
        }
        await fetchWarnings(true);
        if (mountedRef.current) {
          if (ackTimerRef.current) {
            clearTimeout(ackTimerRef.current);
          }
          ackTimerRef.current = setTimeout(() => {
            if (mountedRef.current) {
              setAckSuccessId(null);
            }
          }, 3000);
        }
      }
    } finally {
      ackInFlightRef.current = false;
      if (mountedRef.current) {
        setAckBusyId(null);
      }
    }
  };

  const activeItems = data?.active_warnings ?? [];
  const ackItems = data?.acknowledged_warnings ?? [];
  const displayedItems = filterTab === "active" ? activeItems : ackItems;

  const summary = data?.summary ?? {
    critical_count: 0,
    warning_count: 0,
    info_count: 0,
    total_active: 0,
    total_acknowledged: 0,
  };

  const curEval = data?.current_evaluation;

  return (
    <div
      className="sd-early-warnings-container"
      data-testid="early-warnings-panel-container"
      style={{
        border: "1px solid var(--sd-border-subtle, rgba(255, 255, 255, 0.12))",
        borderRadius: "8px",
        background: "var(--sd-bg-surface, #0f172a)",
        marginBottom: "0.75rem",
        overflow: "hidden",
      }}
    >
      {/* 1. Header Bar: Compact Summary & Toggle Button */}
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          padding: "0.55rem 0.9rem",
          background: "rgba(15, 23, 42, 0.88)",
          cursor: "pointer",
          userSelect: "none",
        }}
        onClick={onToggle}
        role="button"
        tabIndex={0}
        aria-expanded={isOpen}
        onKeyDown={(e) => {
          if (e.key === "Enter" || e.key === " ") {
            e.preventDefault();
            onToggle();
          }
        }}
        data-testid="early-warnings-toggle"
      >
        <div style={{ display: "flex", alignItems: "center", gap: "0.6rem" }}>
          <span style={{ fontSize: "1.05rem" }}>
            {summary.critical_count > 0 ? "🚨" : summary.warning_count > 0 ? "⚠️" : "🛡️"}
          </span>
          <span style={{ fontWeight: 600, fontSize: "0.85rem", color: "var(--sd-text-primary, #f8fafc)" }}>
            Advisory Early Warnings
          </span>
          <span
            className={`sd-badge ${
              summary.critical_count > 0
                ? "sd-badge--rose"
                : summary.warning_count > 0
                ? "sd-badge--amber"
                : "sd-badge--emerald"
            }`}
            style={{ fontSize: "0.72rem", padding: "0.15rem 0.45rem" }}
            data-testid="early-warnings-count-badge"
          >
            {summary.total_active === 0
              ? "All Clear"
              : `${summary.total_active} Active ${summary.total_active === 1 ? "Alert" : "Alerts"}`}
          </span>
          <span
            className="sd-badge sd-badge--indigo"
            style={{ fontSize: "0.68rem", opacity: 0.85 }}
            title="Advisory evaluation monitoring only warns; it never pauses or stops runs"
          >
            Completed Evaluations Only · Advisory
          </span>
        </div>

        <div style={{ display: "flex", alignItems: "center", gap: "0.6rem" }}>
          {(isReadOnlyPreview || data?.is_read_only_preview) && (
            <span className="sd-badge sd-badge--cyan" style={{ fontSize: "0.68rem" }}>
              READ-ONLY PREVIEW
            </span>
          )}
          <button
            type="button"
            className="sd-btn sd-btn--xs sd-btn--ghost"
            style={{ fontSize: "0.74rem" }}
            aria-label={isOpen ? "Collapse early warnings" : "Expand early warnings"}
          >
            {isOpen ? "Hide ▲" : "Details ▼"}
          </button>
        </div>
      </div>

      {/* 2. Expanded Body Panel */}
      {isOpen && (
        <div
          style={{
            padding: "0.85rem 0.95rem",
            borderTop: "1px solid var(--sd-border-subtle, rgba(255, 255, 255, 0.08))",
            background: "rgba(10, 15, 29, 0.95)",
          }}
          data-testid="early-warnings-panel-body"
        >
          {/* Read-Only Preview Warning Banner */}
          {(isReadOnlyPreview || data?.is_read_only_preview) && (
            <div
              className="sd-alert sd-alert--info"
              style={{
                display: "flex",
                alignItems: "center",
                gap: "0.5rem",
                padding: "0.45rem 0.75rem",
                marginBottom: "0.75rem",
                fontSize: "0.75rem",
                borderRadius: "6px",
                background: "rgba(14, 165, 233, 0.12)",
                border: "1px solid rgba(14, 165, 233, 0.3)",
                color: "#38bdf8",
              }}
              data-testid="early-warnings-preview-banner"
            >
              <span>🔬</span>
              <span>
                <strong>Preview Mode:</strong> Displaying authentic recorded evaluation evidence.
                Active training sessions are not running in isolated preview.
              </span>
            </div>
          )}

          {/* Current Evaluation Evidence Strip */}
          {curEval && (
            <div
              style={{
                marginBottom: "0.75rem",
                padding: "0.6rem 0.8rem",
                borderRadius: "6px",
                background: "rgba(30, 41, 59, 0.6)",
                border: "1px solid rgba(255, 255, 255, 0.08)",
                fontSize: "0.74rem",
                display: "flex",
                flexDirection: "column",
                gap: "0.35rem",
              }}
              data-testid="evaluation-evidence-card"
            >
              <div style={{ display: "flex", justifyContent: "space-between", flexWrap: "wrap", gap: "0.4rem" }}>
                <div>
                  <span style={{ color: "var(--sd-text-secondary, #94a3b8)" }}>Evaluation: </span>
                  <strong style={{ color: "#f8fafc" }}>{curEval.evaluation_id || "None completed"}</strong>
                  {curEval.checkpoint_episode != null && (
                    <span style={{ marginLeft: "0.5rem", color: "#38bdf8" }}>
                      (Checkpoint Ep {curEval.checkpoint_episode})
                    </span>
                  )}
                </div>
                <div>
                  <span style={{ color: "var(--sd-text-secondary, #94a3b8)" }}>Sample Size: </span>
                  <strong style={{ color: "#f8fafc" }}>{curEval.sample_count} seeds</strong>
                </div>
              </div>

              <div style={{ display: "flex", justifyContent: "space-between", flexWrap: "wrap", gap: "0.4rem" }}>
                <div>
                  <span style={{ color: "var(--sd-text-secondary, #94a3b8)" }}>Repetition Metrics: </span>
                  <span
                    className={`sd-badge ${
                      curEval.repetition_metrics_available ? "sd-badge--emerald" : "sd-badge--amber"
                    }`}
                    style={{ fontSize: "0.68rem" }}
                  >
                    {curEval.repetition_metric_status}
                  </span>
                  {curEval.repetition_metrics && (
                    <span style={{ marginLeft: "0.5rem", color: "#e2e8f0" }}>
                      Fraction: <strong>{Math.round((curEval.repetition_metrics.repeated_movement_fraction ?? 0) * 100)}%</strong> ({curEval.repetition_metrics.repeated_movement_steps} / {curEval.repetition_metrics.total_world_steps} steps)
                    </span>
                  )}
                </div>
                <div>
                  <span style={{ color: "var(--sd-text-secondary, #94a3b8)" }}>Regression Status: </span>
                  <span style={{ color: "#cbd5e1" }}>{curEval.comparison_status}</span>
                </div>
              </div>
            </div>
          )}

          {/* Monitored vs Deferred Disclosure */}
          <div
            style={{
              padding: "0.4rem 0.6rem",
              marginBottom: "0.75rem",
              borderRadius: "4px",
              background: "rgba(0, 0, 0, 0.25)",
              fontSize: "0.68rem",
              color: "#94a3b8",
              lineHeight: 1.4,
            }}
          >
            <div>
              <strong style={{ color: "#38bdf8" }}>Monitored in this version:</strong> Unproductive repetition (v2.0) and compatible success regression from completed formal evaluations.
            </div>
            <div>
              <strong style={{ color: "#e2e8f0" }}>Deferred enhancements:</strong> Live training window monitoring, PPO loss alerts, recovery-relapse tracking, stale training detection, and automatic resolution.
            </div>
          </div>

          {/* Filter Tabs & Refresh Controls */}
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
              marginBottom: "0.75rem",
            }}
          >
            <div style={{ display: "flex", gap: "0.4rem" }}>
              <button
                type="button"
                className={`sd-btn sd-btn--xs ${
                  filterTab === "active" ? "sd-btn--primary" : "sd-btn--ghost"
                }`}
                onClick={() => setFilterTab("active")}
                data-testid="filter-active-tab"
              >
                Active Alerts ({activeItems.length})
              </button>
              <button
                type="button"
                className={`sd-btn sd-btn--xs ${
                  filterTab === "acknowledged" ? "sd-btn--primary" : "sd-btn--ghost"
                }`}
                onClick={() => setFilterTab("acknowledged")}
                data-testid="filter-acknowledged-tab"
              >
                Acknowledged (Seen) ({ackItems.length})
              </button>
            </div>

            <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
              <button
                type="button"
                className="sd-btn sd-btn--xs sd-btn--ghost"
                onClick={() => fetchWarnings(true)}
                disabled={loading}
                title="Force refresh early warning telemetry"
              >
                {loading ? "Refreshing..." : "↻ Refresh"}
              </button>
            </div>
          </div>

          {/* Warning Cards List */}
          {displayedItems.length === 0 ? (
            <div
              style={{
                textAlign: "center",
                padding: "1.2rem",
                fontSize: "0.8rem",
                color: "var(--sd-text-secondary, #94a3b8)",
                border: "1px dashed var(--sd-border-subtle, rgba(255, 255, 255, 0.1))",
                borderRadius: "6px",
              }}
              data-testid="no-warnings-placeholder"
            >
              {filterTab === "active"
                ? `No active advisory warnings for Stage ${curriculumStage}. Evaluation metrics are within normal operating bounds.`
                : `No acknowledged warning records on file for Stage ${curriculumStage}.`}
            </div>
          ) : (
            <div style={{ display: "flex", flexDirection: "column", gap: "0.65rem" }}>
              {displayedItems.map((item) => (
                <WarningCard
                  key={item.id}
                  item={item}
                  onAcknowledge={handleAcknowledge}
                  isAckBusy={ackBusyId === item.id}
                  isAckSuccess={ackSuccessId === item.id}
                  onOpenReplay={onOpenReplay}
                />
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

interface WarningCardProps {
  item: EarlyWarningItem;
  onAcknowledge: (id: string) => void;
  isAckBusy: boolean;
  isAckSuccess: boolean;
  onOpenReplay?: (replayPath: string, step?: number) => void;
}

function WarningCard({ item, onAcknowledge, isAckBusy, isAckSuccess, onOpenReplay }: WarningCardProps) {
  const isAck = item.status === "acknowledged";

  const sevBadgeClass =
    item.severity === "critical"
      ? "sd-badge--rose"
      : item.severity === "warning"
      ? "sd-badge--amber"
      : "sd-badge--cyan";

  const handleLaunchReplay = () => {
    if (item.replay_link && onOpenReplay) {
      onOpenReplay(item.replay_link, item.replay_step ?? 0);
    } else if (item.replay_link) {
      const replayParam = encodeURIComponent(item.replay_link);
      const stepParam = item.replay_step != null ? `&step=${item.replay_step}` : "";
      window.location.href = `${window.location.pathname || "/"}?workspace=temporal&replay=${replayParam}${stepParam}`;
    }
  };

  return (
    <div
      style={{
        border: "1px solid var(--sd-border-subtle, rgba(255, 255, 255, 0.1))",
        borderRadius: "6px",
        background: "rgba(15, 23, 42, 0.65)",
        padding: "0.75rem",
        display: "flex",
        flexDirection: "column",
        gap: "0.5rem",
      }}
      data-testid={`warning-card-${item.id}`}
    >
      {/* Card Header: Title & Badges */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", gap: "0.5rem" }}>
        <div>
          <div style={{ display: "flex", alignItems: "center", gap: "0.45rem", marginBottom: "0.2rem" }}>
            <span className={`sd-badge ${sevBadgeClass}`} style={{ fontSize: "0.68rem" }}>
              {item.severity.toUpperCase()}
            </span>
            <span
              className={`sd-badge ${isAck ? "sd-badge--cyan" : "sd-badge--amber"}`}
              style={{ fontSize: "0.68rem" }}
            >
              {isAck ? "ACKNOWLEDGED (SEEN)" : "ACTIVE"}
            </span>
            <span style={{ fontSize: "0.7rem", color: "var(--sd-text-secondary, #94a3b8)" }}>
              {item.source_layer.replace(/_/g, " ")}
            </span>
          </div>
          <strong style={{ fontSize: "0.85rem", color: "var(--sd-text-primary, #f8fafc)" }}>
            {item.title}
          </strong>
        </div>

        {/* Action button: Acknowledge */}
        <div style={{ display: "flex", flexDirection: "column", alignItems: "flex-end", gap: "0.2rem" }}>
          <button
            type="button"
            className="sd-btn sd-btn--xs sd-btn--ghost"
            style={{ fontSize: "0.72rem", flexShrink: 0 }}
            onClick={() => onAcknowledge(item.id)}
            disabled={isAckBusy || isAck}
            title={
              isAck
                ? "Warning already acknowledged (seen)"
                : "Acknowledge awareness (does not clear measurements or claim recovery)"
            }
            data-testid={`ack-btn-${item.id}`}
          >
            {isAckSuccess
              ? "✓ Acknowledged"
              : isAck
              ? "Acknowledged (Seen)"
              : isAckBusy
              ? "Saving..."
              : "Mark Acknowledged"}
          </button>
          {!isAck && (
            <span style={{ fontSize: "0.62rem", color: "var(--sd-text-muted, #64748b)" }}>
              Records review; does not claim recovery
            </span>
          )}
        </div>
      </div>

      {/* Observed Facts vs Candidate Causes */}
      <div style={{ fontSize: "0.76rem", lineHeight: 1.45, color: "var(--sd-text-secondary, #cbd5e1)" }}>
        <p style={{ margin: "0 0 0.4rem 0" }}>
          <strong style={{ color: "var(--sd-text-primary, #f8fafc)" }}>Observed Facts: </strong>
          {item.plain_language_observation}
        </p>

        {item.candidate_causes && item.candidate_causes.length > 0 && (
          <div style={{ marginTop: "0.3rem", padding: "0.35rem 0.55rem", background: "rgba(0,0,0,0.25)", borderRadius: "4px" }}>
            <span style={{ fontWeight: 600, color: "#93c5fd" }}>Candidate Causes (Hypotheses — Not Proven):</span>
            <ul style={{ margin: "0.2rem 0 0 1.1rem", padding: 0, fontSize: "0.73rem" }}>
              {item.candidate_causes.map((cause, idx) => (
                <li key={idx}>{cause}</li>
              ))}
            </ul>
          </div>
        )}
      </div>

      {/* Evaluation Identity, Metric Version & Replay Strip */}
      <div
        style={{
          display: "flex",
          flexWrap: "wrap",
          alignItems: "center",
          justifyContent: "space-between",
          gap: "0.8rem",
          fontSize: "0.7rem",
          color: "var(--sd-text-muted, #64748b)",
          paddingTop: "0.35rem",
          borderTop: "1px solid rgba(255, 255, 255, 0.05)",
        }}
      >
        <div style={{ display: "flex", flexWrap: "wrap", alignItems: "center", gap: "0.6rem" }}>
          {item.checkpoint_episode != null && (
            <span>
              <strong>Checkpoint:</strong> Ep {item.checkpoint_episode}
            </span>
          )}
          {item.sample_count > 0 && (
            <span>
              <strong>Sample Count:</strong> {item.sample_count} seeds
            </span>
          )}
          {item.metric_version && (
            <span>
              <strong>Metric Version:</strong> v{item.metric_version}
            </span>
          )}
        </div>

        {/* Replay Link or Unavailable State */}
        <div>
          {item.replay_available && item.replay_link ? (
            <button
              type="button"
              className="sd-btn sd-btn--xs sd-btn--ghost"
              style={{ fontSize: "0.72rem", padding: "0.15rem 0.5rem", color: "var(--sd-cyan, #06b6d4)" }}
              onClick={handleLaunchReplay}
              title={`Inspect replay in viewer at lead-in step ${item.replay_step ?? 0}`}
              data-testid={`replay-btn-${item.id}`}
            >
              🔬 Open Evaluation Replay {item.replay_step != null ? `(Step ${item.replay_step})` : ""}
            </button>
          ) : (
            <span style={{ fontStyle: "italic", color: "var(--sd-text-muted, #64748b)" }}>
              {item.link_explanation || "Replay file not available on disk"}
            </span>
          )}
        </div>
      </div>
    </div>
  );
}

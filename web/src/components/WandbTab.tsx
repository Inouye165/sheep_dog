import { useMemo, useState, useEffect } from "react";
import type { CheckpointIndex, TrainingStatus, CheckpointEntry } from "../state/types";
import {
  restoreCheckpoint,
  forkCheckpoint,
  archiveActiveRun,
  loadCheckpointDetails,
  loadConfigActive,
} from "../lib/api";

interface WandbTabProps {
  checkpointIndex: CheckpointIndex | null;
  trainingStatus: TrainingStatus | null;
  effectiveConfig: Record<string, unknown> | null;
  onRefreshData: () => void;
  onWatchReplay: (episode: number) => void;
}

function asRecord(value: unknown): Record<string, unknown> | null {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null;
}

export function WandbTab({
  checkpointIndex,
  trainingStatus,
  effectiveConfig,
  onRefreshData,
  onWatchReplay,
}: WandbTabProps) {
  const rootConfig = asRecord(effectiveConfig);
  const trainingConfig = asRecord(rootConfig?.training);

  const [selectedCheckpoint, setSelectedCheckpoint] = useState<CheckpointEntry | null>(null);
  const [selectedDetails, setSelectedDetails] = useState<any>(null);
  const [showForkForm, setShowForkForm] = useState(false);
  const [showCompare, setShowCompare] = useState(false);
  const [activeConfig, setActiveConfig] = useState<Record<string, any> | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);
  const [actionSuccess, setActionSuccess] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  // Fork overrides state
  const [forkEpisodes, setForkEpisodes] = useState(100);
  const [forkFastMode, setForkFastMode] = useState(true);
  const [forkAutoPromote, setForkAutoPromote] = useState(true);
  const [forkEntropyCoef, setForkEntropyCoef] = useState(0.01);
  const [forkLearningRate, setForkLearningRate] = useState(0.0001);
  const [forkBatchSize, setForkBatchSize] = useState(64);
  const [forkRolloutSteps, setForkRolloutSteps] = useState(2048);
  const [forkClipRange, setForkClipRange] = useState(0.2);

  const checkpoints = useMemo(() => {
    return checkpointIndex?.checkpoints ?? [];
  }, [checkpointIndex]);

  // Load detailed checkpoint payload
  useEffect(() => {
    if (selectedCheckpoint) {
      setActionError(null);
      loadCheckpointDetails(
        selectedCheckpoint.checkpoint_episode,
        selectedCheckpoint.journey,
        selectedCheckpoint.checkpoint_id
      )
        .then((details) => {
          setSelectedDetails(details);
          // Set initial fork overrides from selected checkpoint values
          const policyConf = (details.policy_config || {}) as Record<string, any>;
          setForkEntropyCoef(policyConf.entropy_coef ?? 0.01);
          setForkLearningRate(policyConf.learning_rate ?? 0.0001);
          setForkBatchSize(policyConf.batch_size ?? 64);
          setForkRolloutSteps(policyConf.rollout_steps ?? 2048);
          setForkClipRange(policyConf.clip_range ?? 0.2);
        })
        .catch((err) => {
          console.error("Failed to load checkpoint details:", err);
          setActionError("Failed to fetch full checkpoint metadata details.");
          setSelectedDetails(null);
        });
    } else {
      setSelectedDetails(null);
    }
  }, [selectedCheckpoint]);

  // Load active training config for comparison
  useEffect(() => {
    if (showCompare && selectedCheckpoint) {
      loadConfigActive()
        .then((config) => {
          setActiveConfig(config);
        })
        .catch((err) => {
          console.error("Failed to load active config:", err);
          setActionError("Failed to fetch active training configuration.");
          setActiveConfig(null);
        });
    } else {
      setActiveConfig(null);
    }
  }, [showCompare, selectedCheckpoint]);

  const handleRestore = async () => {
    if (!selectedCheckpoint) return;
    if (trainingStatus?.running) {
      setActionError("Cannot restore checkpoint while training is running.");
      return;
    }
    if (
      !window.confirm(
        `Are you sure you want to restore checkpoint episode ${selectedCheckpoint.checkpoint_episode} as the active model? This will replace your active model weights.`
      )
    ) {
      return;
    }

    try {
      setSubmitting(true);
      setActionError(null);
      setActionSuccess(null);
      const res = await restoreCheckpoint(
        selectedCheckpoint.checkpoint_episode,
        selectedCheckpoint.journey,
        selectedCheckpoint.checkpoint_id
      );
      setActionSuccess(res.message || "Checkpoint restored successfully.");
      onRefreshData();
    } catch (err) {
      setActionError(err instanceof Error ? err.message : "Failed to restore checkpoint.");
    } finally {
      setSubmitting(false);
    }
  };

  const handleFork = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!selectedCheckpoint) return;
    if (trainingStatus?.running) {
      setActionError("Cannot fork checkpoint while training is running.");
      return;
    }

    try {
      setSubmitting(true);
      setActionError(null);
      setActionSuccess(null);

      const overrides = {
        episodes: forkEpisodes,
        fast_mode: forkFastMode,
        auto_promote: forkAutoPromote,
        entropy_coef: forkEntropyCoef,
        learning_rate: forkLearningRate,
        batch_size: forkBatchSize,
        rollout_steps: forkRolloutSteps,
        clip_range: forkClipRange,
      };

      const res = await forkCheckpoint(
        selectedCheckpoint.checkpoint_episode,
        selectedCheckpoint.journey,
        overrides,
        selectedCheckpoint.checkpoint_id
      );
      setActionSuccess(res.message || "Training run forked successfully.");
      setShowForkForm(false);
      onRefreshData();
    } catch (err) {
      setActionError(err instanceof Error ? err.message : "Failed to fork checkpoint run.");
    } finally {
      setSubmitting(false);
    }
  };

  const handleArchive = async () => {
    if (trainingStatus?.running) {
      setActionError("Cannot archive active run while training is running.");
      return;
    }
    if (
      !window.confirm(
        "Are you sure you want to manually archive the active run? This moves current active checkpoints and models to the journey history log."
      )
    ) {
      return;
    }

    try {
      setSubmitting(true);
      setActionError(null);
      setActionSuccess(null);
      const res = await archiveActiveRun();
      setActionSuccess(`Active run archived successfully to: ${res.archive_dir}`);
      onRefreshData();
    } catch (err) {
      setActionError(err instanceof Error ? err.message : "Failed to archive active run.");
    } finally {
      setSubmitting(false);
    }
  };

  // Check if latest checkpoint is worse than best
  const peakSuccess = useMemo(() => {
    if (checkpoints.length === 0) return 0;
    return Math.max(...checkpoints.map((cp) => cp.success_rate));
  }, [checkpoints]);

  const latestSuccess = checkpoints[checkpoints.length - 1]?.success_rate ?? 0;
  const hasRegressed = peakSuccess >= 0.8 && latestSuccess < peakSuccess - 0.25;

  const styles = {
    container: {
      padding: "24px",
      display: "flex",
      flexDirection: "column" as const,
      gap: "20px",
      height: "100%",
      overflowY: "auto" as const,
      color: "#f8fafc",
      fontFamily: "Inter, system-ui, -apple-system, sans-serif",
    },
    header: {
      borderBottom: "1px solid rgba(255, 255, 255, 0.1)",
      paddingBottom: "16px",
      display: "flex",
      justifyContent: "space-between",
      alignItems: "flex-end",
    },
    eyebrow: {
      fontSize: "0.75rem",
      textTransform: "uppercase" as const,
      letterSpacing: "0.1em",
      color: "#f59e0b",
      fontWeight: "bold",
      margin: "0 0 8px 0",
    },
    title: {
      fontSize: "1.75rem",
      fontWeight: "700",
      margin: "0 0 8px 0",
      background: "linear-gradient(135deg, #ffffff 0%, #cbd5e1 100%)",
      WebkitBackgroundClip: "text",
      WebkitTextFillColor: "transparent",
    },
    desc: {
      fontSize: "0.875rem",
      color: "#94a3b8",
      lineHeight: "1.5",
      margin: 0,
    },
    warningBanner: {
      background: "rgba(239, 68, 68, 0.15)",
      borderLeft: "4px solid #ef4444",
      padding: "14px 20px",
      borderRadius: "0 8px 8px 0",
      fontSize: "0.875rem",
      color: "#fca5a5",
      lineHeight: "1.5",
    },
    collapseBanner: {
      background: "rgba(245, 158, 11, 0.15)",
      borderLeft: "4px solid #f59e0b",
      padding: "14px 20px",
      borderRadius: "0 8px 8px 0",
      fontSize: "0.875rem",
      color: "#fde047",
      lineHeight: "1.5",
    },
    alertHeader: {
      fontWeight: "700",
      fontSize: "0.95rem",
      marginBottom: "4px",
      display: "flex",
      alignItems: "center",
      gap: "8px",
    },
    kpis: {
      display: "grid",
      gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))",
      gap: "16px",
    },
    kpiCard: {
      background: "rgba(30, 41, 59, 0.7)",
      border: "1px solid rgba(255, 255, 255, 0.05)",
      borderRadius: "12px",
      padding: "16px",
      display: "flex",
      flexDirection: "column" as const,
      gap: "4px",
      boxShadow: "0 4px 6px -1px rgba(0, 0, 0, 0.1)",
    },
    kpiLabel: {
      fontSize: "0.75rem",
      color: "#94a3b8",
      fontWeight: "500",
      textTransform: "uppercase" as const,
      letterSpacing: "0.05em",
    },
    kpiValue: {
      fontSize: "1.1rem",
      color: "#f1f5f9",
      fontWeight: "600",
      whiteSpace: "nowrap" as const,
      overflow: "hidden",
      textOverflow: "ellipsis",
    },
    actionBtn: {
      background: "linear-gradient(90deg, #f59e0b 0%, #d97706 100%)",
      color: "#0f172a",
      border: "none",
      borderRadius: "6px",
      padding: "8px 16px",
      fontSize: "0.85rem",
      fontWeight: "700",
      cursor: "pointer",
      display: "inline-flex",
      alignItems: "center",
      justifyContent: "center",
      boxShadow: "0 2px 8px rgba(245, 158, 11, 0.3)",
      transition: "all 0.2s",
    },
    secondaryBtn: {
      background: "rgba(255, 255, 255, 0.08)",
      color: "#f8fafc",
      border: "1px solid rgba(255, 255, 255, 0.1)",
      borderRadius: "6px",
      padding: "8px 16px",
      fontSize: "0.85rem",
      fontWeight: "600",
      cursor: "pointer",
      transition: "all 0.2s",
    },
    dangerBtn: {
      background: "rgba(239, 68, 68, 0.15)",
      color: "#fca5a5",
      border: "1px solid rgba(239, 68, 68, 0.3)",
      borderRadius: "6px",
      padding: "8px 16px",
      fontSize: "0.85rem",
      fontWeight: "600",
      cursor: "pointer",
      transition: "all 0.2s",
    },
    sectionTitle: {
      fontSize: "1.15rem",
      fontWeight: "600",
      color: "#f1f5f9",
      margin: "0 0 12px 0",
    },
    tableContainer: {
      background: "rgba(15, 23, 42, 0.5)",
      border: "1px solid rgba(255, 255, 255, 0.05)",
      borderRadius: "12px",
      overflow: "hidden",
    },
    table: {
      width: "100%",
      borderCollapse: "collapse" as const,
      textAlign: "left" as const,
      fontSize: "0.85rem",
    },
    th: {
      background: "rgba(30, 41, 59, 0.6)",
      padding: "12px 14px",
      color: "#94a3b8",
      fontWeight: "600",
      borderBottom: "1px solid rgba(255, 255, 255, 0.08)",
      fontSize: "0.75rem",
      textTransform: "uppercase" as const,
      letterSpacing: "0.05em",
    },
    tr: {
      borderBottom: "1px solid rgba(255, 255, 255, 0.04)",
      cursor: "pointer",
      transition: "background 0.1s ease",
    },
    td: {
      padding: "12px 14px",
      color: "#e2e8f0",
    },
    detailGrid: {
      display: "grid",
      gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))",
      gap: "16px",
    },
    configCard: {
      background: "rgba(30, 41, 59, 0.4)",
      border: "1px solid rgba(255, 255, 255, 0.04)",
      borderRadius: "10px",
      padding: "16px",
    },
    configTitle: {
      fontSize: "0.9rem",
      fontWeight: "600",
      color: "#f1f5f9",
      margin: "0 0 12px 0",
      borderBottom: "1px solid rgba(255, 255, 255, 0.06)",
      paddingBottom: "6px",
    },
    dl: {
      display: "grid",
      gridTemplateColumns: "1fr 1fr",
      gap: "8px 12px",
      margin: 0,
    },
    dt: {
      fontSize: "0.75rem",
      color: "#94a3b8",
      fontWeight: "500",
    },
    dd: {
      fontSize: "0.75rem",
      color: "#f1f5f9",
      fontWeight: "600",
      margin: 0,
      textAlign: "right" as const,
      fontFamily: "monospace",
    },
    modalOverlay: {
      position: "fixed" as const,
      top: 0,
      left: 0,
      right: 0,
      bottom: 0,
      backgroundColor: "rgba(15, 23, 42, 0.8)",
      backdropFilter: "blur(4px)",
      display: "flex",
      alignItems: "center",
      justifyContent: "center",
      zIndex: 1000,
    },
    modalContent: {
      backgroundColor: "#1e293b",
      border: "1px solid rgba(255, 255, 255, 0.1)",
      borderRadius: "12px",
      padding: "24px",
      width: "90%",
      maxWidth: "700px",
      maxHeight: "85vh",
      overflowY: "auto" as const,
      boxShadow: "0 20px 25px -5px rgba(0, 0, 0, 0.5)",
    },
    formGroup: {
      marginBottom: "14px",
      display: "flex",
      flexDirection: "column" as const,
      gap: "6px",
    },
    input: {
      backgroundColor: "#0f172a",
      border: "1px solid rgba(255, 255, 255, 0.1)",
      borderRadius: "6px",
      padding: "8px 12px",
      color: "#f8fafc",
      fontSize: "0.875rem",
    },
    checkboxGroup: {
      display: "flex",
      alignItems: "center",
      gap: "8px",
      cursor: "pointer",
      userSelect: "none" as const,
      fontSize: "0.875rem",
    },
  };

  return (
    <div style={styles.container}>
      <header style={styles.header}>
        <div>
          <p style={styles.eyebrow}>Training Checkpoints & Resume Auditing</p>
          <h2 style={styles.title}>Policy History & Restore / Fork Control</h2>
          <p style={styles.desc}>
            View safety checks, environment compatibility hashes, and launch options for historical checkpoints.
          </p>
        </div>
        <div style={{ display: "flex", gap: "10px" }}>
          <button style={styles.secondaryBtn} onClick={onRefreshData}>
            ↻ Sync State
          </button>
          {!trainingStatus?.running && (
            <button style={styles.dangerBtn} onClick={handleArchive}>
              Archive Active Run
            </button>
          )}
        </div>
      </header>

      {/* Collapse Alerts */}
      {trainingStatus?.anti_collapse_warning?.triggered && (
        <div style={styles.warningBanner}>
          <div style={styles.alertHeader}>
            ⚠️ POLICY COLLAPSE DETECTED
          </div>
          <p style={{ margin: "4px 0 0 0" }}>
            {trainingStatus.anti_collapse_warning.message}
          </p>
          {trainingStatus.anti_collapse_warning.recommendation && (
            <p style={{ margin: "4px 0 0 0", fontStyle: "italic", fontSize: "0.8rem" }}>
              Recommendation: {trainingStatus.anti_collapse_warning.recommendation}
            </p>
          )}
        </div>
      )}

      {hasRegressed && !trainingStatus?.anti_collapse_warning?.triggered && (
        <div style={styles.collapseBanner}>
          <div style={styles.alertHeader}>
            📈 POLICY REGRESSION WARNING
          </div>
          <p style={{ margin: "4px 0 0 0" }}>
            Latest success rate ({Math.round(latestSuccess * 100)}%) has fallen significantly below the peak performance ({Math.round(peakSuccess * 100)}%) reached earlier. Consider restoring or forking from an earlier successful checkpoint (e.g. around episode 896).
          </p>
        </div>
      )}

      {actionError && (
        <div style={{ padding: "12px 16px", backgroundColor: "rgba(239, 68, 68, 0.15)", borderLeft: "4px solid #ef4444", borderRadius: "0 6px 6px 0", color: "#fca5a5", fontSize: "0.85rem" }}>
          Error: {actionError}
        </div>
      )}

      {actionSuccess && (
        <div style={{ padding: "12px 16px", backgroundColor: "rgba(16, 185, 129, 0.15)", borderLeft: "4px solid #10b981", borderRadius: "0 6px 6px 0", color: "#a7f3d0", fontSize: "0.85rem" }}>
          {actionSuccess}
        </div>
      )}

      {/* KPI Panel */}
      <section style={styles.kpis}>
        <div style={styles.kpiCard}>
          <span style={styles.kpiLabel}>Active Run ID</span>
          <span style={styles.kpiValue} title={trainingStatus?.run_id || "N/A"}>
            {trainingStatus?.run_id || "N/A"}
          </span>
        </div>
        <div style={styles.kpiCard}>
          <span style={styles.kpiLabel}>Model Source</span>
          <span style={{ ...styles.kpiValue, textTransform: "capitalize" }}>
            {trainingStatus?.active_model_source || "latest"}
          </span>
        </div>
        {trainingStatus?.parent_run_id && (
          <div style={styles.kpiCard}>
            <span style={styles.kpiLabel}>Parent Run ID</span>
            <span style={styles.kpiValue} title={trainingStatus.parent_run_id}>
              {trainingStatus.parent_run_id}
            </span>
          </div>
        )}
        <div style={styles.kpiCard}>
          <span style={styles.kpiLabel}>Total Timesteps</span>
          <span style={styles.kpiValue}>
            {latestSuccess ? checkpoints[checkpoints.length - 1]?.global_timesteps?.toLocaleString() || "N/A" : "N/A"}
          </span>
        </div>
      </section>

      {/* Checkpoints Table */}
      <div>
        <h3 style={styles.sectionTitle}>Checkpoint Database</h3>
        <div style={styles.tableContainer}>
          <table style={styles.table}>
            <thead>
              <tr>
                <th style={styles.th}>Episode</th>
                <th style={styles.th}>Run ID</th>
                <th style={styles.th}>Evaluation Mode</th>
                <th style={styles.th}>Success Rate</th>
                <th style={styles.th}>Avg Reward</th>
                <th style={styles.th}>Avg Steps</th>
                <th style={styles.th}>Compatibility</th>
              </tr>
            </thead>
            <tbody>
              {checkpoints.length === 0 ? (
                <tr>
                  <td colSpan={7} style={{ ...styles.td, textAlign: "center", color: "#64748b" }}>
                    No checkpoints recorded yet. Start training to log checkpoints.
                  </td>
                </tr>
              ) : (
                [...checkpoints].reverse().map((cp, idx) => {
                  const isSelected = selectedCheckpoint
                    ? (selectedCheckpoint.checkpoint_id && cp.checkpoint_id
                        ? selectedCheckpoint.checkpoint_id === cp.checkpoint_id
                        : selectedCheckpoint.checkpoint_episode === cp.checkpoint_episode && selectedCheckpoint.run_id === cp.run_id)
                    : false;
                  const isCompatible = cp.observation_schema_hash === checkpoints[checkpoints.length - 1]?.observation_schema_hash;
                  const rowKey = cp.checkpoint_id
                    ? cp.checkpoint_id
                    : cp.run_id
                      ? `${cp.run_id}-ep-${cp.checkpoint_episode}`
                      : `ep-${cp.checkpoint_episode}-${idx}`;

                  return (
                    <tr
                      key={rowKey}
                      onClick={() => setSelectedCheckpoint(cp)}
                      style={{
                        ...styles.tr,
                        background: isSelected ? "rgba(245, 158, 11, 0.15)" : "transparent",
                      }}
                      onMouseEnter={(e) => {
                        if (!isSelected) e.currentTarget.style.backgroundColor = "rgba(255, 255, 255, 0.03)";
                      }}
                      onMouseLeave={(e) => {
                        if (!isSelected) e.currentTarget.style.backgroundColor = "transparent";
                      }}
                    >
                      <td style={styles.td}><strong>{cp.checkpoint_episode}</strong></td>
                      <td style={{ ...styles.td, fontFamily: "monospace", fontSize: "0.8rem" }}>
                        {cp.run_id ? cp.run_id.substring(0, 16) + "..." : "N/A"}
                      </td>
                      <td style={styles.td}>
                        <span style={{ fontSize: "0.75rem", padding: "2px 6px", borderRadius: "4px", backgroundColor: cp.deterministic_evaluation ? "rgba(59, 130, 246, 0.2)" : "rgba(100, 116, 139, 0.2)", color: cp.deterministic_evaluation ? "#93c5fd" : "#cbd5e1" }}>
                          {cp.deterministic_evaluation ? "Deterministic" : "Stochastic"}
                        </span>
                      </td>
                      <td style={{ ...styles.td, color: cp.success_rate >= 0.9 ? "#10b981" : cp.success_rate >= 0.5 ? "#fbbf24" : "#f87171", fontWeight: "bold" }}>
                        {(cp.success_rate * 100).toFixed(0)}%
                      </td>
                      <td style={styles.td}>{cp.average_reward.toFixed(1)}</td>
                      <td style={styles.td}>{cp.average_completion_steps.toFixed(0)}</td>
                      <td style={styles.td}>
                        <span style={{ fontSize: "0.75rem", padding: "2px 6px", borderRadius: "4px", backgroundColor: isCompatible ? "rgba(16, 185, 129, 0.15)" : "rgba(239, 68, 68, 0.15)", color: isCompatible ? "#34d399" : "#f87171" }}>
                          {isCompatible ? "✓ Compatible" : "✗ Structural Change"}
                        </span>
                      </td>
                    </tr>
                  );
                })
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* Selected Checkpoint Detail Card */}
      {selectedCheckpoint && (
        <section style={{ ...styles.configCard, border: "1px solid rgba(245, 158, 11, 0.2)" }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: "16px", borderBottom: "1px solid rgba(255, 255, 255, 0.08)", paddingBottom: "12px" }}>
            <div>
              <h3 style={{ margin: 0, fontSize: "1.1rem" }}>
                Selected Checkpoint: Episode {selectedCheckpoint.checkpoint_episode}
              </h3>
              <p style={{ margin: "4px 0 0 0", fontSize: "0.8rem", color: "#94a3b8" }}>
                Run ID: {selectedDetails?.run_id || selectedCheckpoint.run_id || "N/A"}
              </p>
            </div>
            <div style={{ display: "flex", gap: "10px" }}>
              <button style={styles.secondaryBtn} onClick={() => onWatchReplay(selectedCheckpoint.checkpoint_episode)}>
                Watch Replay
              </button>
              <button style={styles.secondaryBtn} onClick={() => setShowCompare(true)}>
                Compare with Active
              </button>
              {!trainingStatus?.running && (
                <>
                  <button style={styles.secondaryBtn} onClick={handleRestore}>
                    Restore weights
                  </button>
                  <button style={styles.actionBtn} onClick={() => setShowForkForm(true)}>
                    Fork Run...
                  </button>
                </>
              )}
            </div>
          </div>

          <div style={styles.detailGrid}>
            <div style={{ display: "flex", flexDirection: "column", gap: "10px" }}>
              <h4 style={{ margin: "0 0 4px 0", fontSize: "0.85rem", color: "#cbd5e1" }}>Structural Hashes</h4>
              <dl style={styles.dl}>
                <dt>Obs Schema Hash</dt>
                <dd title={selectedDetails?.observation_schema_hash}>{selectedDetails?.observation_schema_hash?.substring(0, 8) || "N/A"}</dd>
                <dt>Action Space Hash</dt>
                <dd title={selectedDetails?.action_space_hash}>{selectedDetails?.action_space_hash?.substring(0, 8) || "N/A"}</dd>
                <dt>Env Config Vers.</dt>
                <dd>{selectedDetails?.environment_config_version || "1.0"}</dd>
                <dt>Reward Schema Vers.</dt>
                <dd>{selectedDetails?.reward_schema_version || "1.0"}</dd>
              </dl>
            </div>

            <div style={{ display: "flex", flexDirection: "column", gap: "10px" }}>
              <h4 style={{ margin: "0 0 4px 0", fontSize: "0.85rem", color: "#cbd5e1" }}>Hyperparameters</h4>
              <dl style={styles.dl}>
                <dt>learning_rate</dt>
                <dd>{selectedDetails?.policy_config?.learning_rate ?? "N/A"}</dd>
                <dt>entropy_coef</dt>
                <dd>{selectedDetails?.policy_config?.entropy_coef ?? "N/A"}</dd>
                <dt>batch_size</dt>
                <dd>{selectedDetails?.policy_config?.batch_size ?? "N/A"}</dd>
                <dt>rollout_steps</dt>
                <dd>{selectedDetails?.policy_config?.rollout_steps ?? "N/A"}</dd>
              </dl>
            </div>

            <div style={{ display: "flex", flexDirection: "column", gap: "10px" }}>
              <h4 style={{ margin: "0 0 4px 0", fontSize: "0.85rem", color: "#cbd5e1" }}>Evaluation Info</h4>
              <dl style={styles.dl}>
                <dt>Deterministic Eval</dt>
                <dd>{selectedCheckpoint.deterministic_evaluation ? "True" : "False"}</dd>
                <dt>Eval Seed Size</dt>
                <dd>{selectedDetails?.evaluation_seeds ? selectedDetails.evaluation_seeds.length : "3"}</dd>
                <dt>Parent Run ID</dt>
                <dd>{selectedDetails?.parent_run_id ? selectedDetails.parent_run_id.substring(0, 8) : "None"}</dd>
                <dt>Parent Checkpoint ID</dt>
                <dd>{selectedDetails?.parent_checkpoint_id ? selectedDetails.parent_checkpoint_id.substring(0, 8) : "None"}</dd>
              </dl>
            </div>
          </div>
        </section>
      )}

      {/* Fork Run Modal Dialog */}
      {showForkForm && selectedCheckpoint && (
        <div style={styles.modalOverlay}>
          <div style={styles.modalContent}>
            <h3 style={{ margin: "0 0 16px 0", borderBottom: "1px solid rgba(255,255,255,0.1)", paddingBottom: "10px" }}>
              Fork Training from Episode {selectedCheckpoint.checkpoint_episode}
            </h3>
            <form onSubmit={handleFork}>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "16px" }}>
                <div style={styles.formGroup}>
                  <label style={styles.dt}>Target Training Episodes</label>
                  <input
                    type="number"
                    style={styles.input}
                    value={forkEpisodes}
                    onChange={(e) => setForkEpisodes(parseInt(e.target.value) || 50)}
                  />
                </div>
                <div style={styles.formGroup}>
                  <label style={styles.dt}>Learning Rate</label>
                  <input
                    type="number"
                    step="0.00001"
                    style={styles.input}
                    value={forkLearningRate}
                    onChange={(e) => setForkLearningRate(parseFloat(e.target.value) || 0.0001)}
                  />
                </div>
                <div style={styles.formGroup}>
                  <label style={styles.dt}>Entropy Coefficient (collapse guard recommends lowering)</label>
                  <input
                    type="number"
                    step="0.0001"
                    style={styles.input}
                    value={forkEntropyCoef}
                    onChange={(e) => setForkEntropyCoef(parseFloat(e.target.value) || 0.01)}
                  />
                </div>
                <div style={styles.formGroup}>
                  <label style={styles.dt}>Batch Size</label>
                  <input
                    type="number"
                    style={styles.input}
                    value={forkBatchSize}
                    onChange={(e) => setForkBatchSize(parseInt(e.target.value) || 64)}
                  />
                </div>
                <div style={styles.formGroup}>
                  <label style={styles.dt}>Rollout Steps</label>
                  <input
                    type="number"
                    style={styles.input}
                    value={forkRolloutSteps}
                    onChange={(e) => setForkRolloutSteps(parseInt(e.target.value) || 2048)}
                  />
                </div>
                <div style={styles.formGroup}>
                  <label style={styles.dt}>Clip Range</label>
                  <input
                    type="number"
                    step="0.01"
                    style={styles.input}
                    value={forkClipRange}
                    onChange={(e) => setForkClipRange(parseFloat(e.target.value) || 0.2)}
                  />
                </div>
              </div>

              <div style={{ display: "flex", gap: "20px", margin: "16px 0" }}>
                <label style={styles.checkboxGroup}>
                  <input
                    type="checkbox"
                    checked={forkFastMode}
                    onChange={(e) => setForkFastMode(e.target.checked)}
                  />
                  Fast Mode
                </label>
                <label style={styles.checkboxGroup}>
                  <input
                    type="checkbox"
                    checked={forkAutoPromote}
                    onChange={(e) => setForkAutoPromote(e.target.checked)}
                  />
                  Auto-Promote Curriculum
                </label>
              </div>

              <div style={{ display: "flex", justifyContent: "flex-end", gap: "12px", borderTop: "1px solid rgba(255,255,255,0.1)", paddingTop: "16px" }}>
                <button type="button" style={styles.secondaryBtn} onClick={() => setShowForkForm(false)}>
                  Cancel
                </button>
                <button type="submit" style={styles.actionBtn} disabled={submitting}>
                  {submitting ? "Forking & Launching..." : "Launch Forked Run"}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* Compare Modal Dialog */}
      {showCompare && selectedCheckpoint && (
        <div style={styles.modalOverlay}>
          <div style={{ ...styles.modalContent, maxWidth: "600px" }}>
            <h3 style={{ margin: "0 0 16px 0", borderBottom: "1px solid rgba(255,255,255,0.1)", paddingBottom: "10px" }}>
              Compare Checkpoint vs Active Config
            </h3>
            <table style={{ ...styles.table, fontSize: "0.8rem", width: "100%" }}>
              <thead>
                <tr>
                  <th style={styles.th}>Parameter</th>
                  <th style={styles.th}>Selected Checkpoint</th>
                  <th style={styles.th}>Active Configuration</th>
                </tr>
              </thead>
              <tbody>
                <tr style={styles.tr}>
                  <td style={styles.td}><strong>Success Rate</strong></td>
                  <td style={styles.td}>{(selectedCheckpoint.success_rate * 100).toFixed(0)}%</td>
                  <td style={styles.td}>
                    {activeConfig?.success_rate != null ? `${(activeConfig.success_rate * 100).toFixed(0)}%` : "N/A"}
                  </td>
                </tr>
                <tr style={styles.tr}>
                  <td style={styles.td}><strong>Entropy Coef</strong></td>
                  <td style={styles.td}>{selectedDetails?.policy_config?.entropy_coef ?? "N/A"}</td>
                  <td style={{ ...styles.td, color: (selectedDetails?.policy_config?.entropy_coef !== activeConfig?.entropy_coef) ? "#fbbf24" : "#e2e8f0" }}>
                    {activeConfig?.entropy_coef ?? "N/A"}
                  </td>
                </tr>
                <tr style={styles.tr}>
                  <td style={styles.td}><strong>Learning Rate</strong></td>
                  <td style={styles.td}>{selectedDetails?.policy_config?.learning_rate ?? "N/A"}</td>
                  <td style={{ ...styles.td, color: (selectedDetails?.policy_config?.learning_rate !== activeConfig?.learning_rate) ? "#fbbf24" : "#e2e8f0" }}>
                    {activeConfig?.learning_rate ?? "N/A"}
                  </td>
                </tr>
                <tr style={styles.tr}>
                  <td style={styles.td}><strong>Batch Size</strong></td>
                  <td style={styles.td}>{selectedDetails?.policy_config?.batch_size ?? "N/A"}</td>
                  <td style={{ ...styles.td, color: (selectedDetails?.policy_config?.batch_size !== activeConfig?.batch_size) ? "#fbbf24" : "#e2e8f0" }}>
                    {activeConfig?.batch_size ?? "N/A"}
                  </td>
                </tr>
                <tr style={styles.tr}>
                  <td style={styles.td}><strong>Rollout Steps</strong></td>
                  <td style={styles.td}>{selectedDetails?.policy_config?.rollout_steps ?? "N/A"}</td>
                  <td style={{ ...styles.td, color: (selectedDetails?.policy_config?.rollout_steps !== activeConfig?.rollout_steps) ? "#fbbf24" : "#e2e8f0" }}>
                    {activeConfig?.rollout_steps ?? "N/A"}
                  </td>
                </tr>
                <tr style={styles.tr}>
                  <td style={styles.td}><strong>Observation Hash</strong></td>
                  <td style={{ ...styles.td, fontFamily: "monospace" }}>{selectedDetails?.observation_schema_hash?.substring(0, 8) ?? "N/A"}</td>
                  <td style={{ ...styles.td, fontFamily: "monospace", color: (selectedDetails?.observation_schema_hash !== activeConfig?.observation_schema_hash) ? "#ef4444" : "#34d399" }}>
                    {activeConfig?.observation_schema_hash?.substring(0, 8) ?? "N/A"}
                  </td>
                </tr>
                <tr style={styles.tr}>
                  <td style={styles.td}><strong>Action Space Hash</strong></td>
                  <td style={{ ...styles.td, fontFamily: "monospace" }}>{selectedDetails?.action_space_hash?.substring(0, 8) ?? "N/A"}</td>
                  <td style={{ ...styles.td, fontFamily: "monospace", color: (selectedDetails?.action_space_hash !== activeConfig?.action_space_hash) ? "#ef4444" : "#34d399" }}>
                    {activeConfig?.action_space_hash?.substring(0, 8) ?? "N/A"}
                  </td>
                </tr>
              </tbody>
            </table>

            <div style={{ display: "flex", justifyContent: "flex-end", marginTop: "16px", borderTop: "1px solid rgba(255,255,255,0.1)", paddingTop: "16px" }}>
              <button style={styles.secondaryBtn} onClick={() => setShowCompare(false)}>
                Close
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

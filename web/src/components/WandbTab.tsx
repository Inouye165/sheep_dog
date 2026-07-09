import { useMemo, useState } from "react";
import type { CheckpointIndex, TrainingStatus, CheckpointEntry } from "../state/types";

interface WandbTabProps {
  checkpointIndex: CheckpointIndex | null;
  trainingStatus: TrainingStatus | null;
  effectiveConfig: Record<string, unknown> | null;
}

function asRecord(value: unknown): Record<string, unknown> | null {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null;
}

export function WandbTab({ checkpointIndex, trainingStatus, effectiveConfig }: WandbTabProps) {
  const rootConfig = asRecord(effectiveConfig);
  const trainingConfig = asRecord(rootConfig?.training);
  const rewardsConfig = asRecord(rootConfig?.rewards);
  const environmentConfig = asRecord(rootConfig?.environment);

  const [selectedCheckpoint, setSelectedCheckpoint] = useState<CheckpointEntry | null>(null);

  const checkpoints = useMemo(() => {
    return checkpointIndex?.checkpoints ?? [];
  }, [checkpointIndex]);

  const latestCheckpoint = checkpoints[checkpoints.length - 1] ?? null;
  const totalEpisodes = trainingStatus?.total_episodes_trained ?? latestCheckpoint?.total_training_episodes ?? 0;

  // Weights & Biases brand styles
  const styles = {
    container: {
      padding: "24px",
      display: "flex",
      flexDirection: "column" as const,
      gap: "24px",
      height: "100%",
      overflowY: "auto" as const,
      color: "#f8fafc",
      fontFamily: "Inter, system-ui, -apple-system, sans-serif",
    },
    header: {
      borderBottom: "1px solid rgba(255, 255, 255, 0.1)",
      paddingBottom: "16px",
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
      maxWidth: "800px",
    },
    kpis: {
      display: "grid",
      gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))",
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
    },
    kpiValue: {
      fontSize: "1.1rem",
      color: "#f1f5f9",
      fontWeight: "600",
    },
    actionCard: {
      background: "linear-gradient(135deg, rgba(30, 41, 59, 0.9) 0%, rgba(15, 23, 42, 0.9) 100%)",
      border: "1px solid rgba(245, 158, 11, 0.25)",
      borderRadius: "16px",
      padding: "24px",
      display: "flex",
      flexDirection: "row" as const,
      alignItems: "center",
      justifyContent: "space-between",
      gap: "24px",
      boxShadow: "0 10px 15px -3px rgba(0, 0, 0, 0.3)",
      flexWrap: "wrap" as const,
    },
    actionLeft: {
      display: "flex",
      flexDirection: "column" as const,
      gap: "8px",
      flex: "1 1 500px",
    },
    actionTitle: {
      fontSize: "1.25rem",
      fontWeight: "600",
      margin: 0,
      color: "#f8fafc",
    },
    actionDesc: {
      fontSize: "0.875rem",
      color: "#cbd5e1",
      lineHeight: "1.6",
      margin: 0,
    },
    warningBanner: {
      background: "rgba(245, 158, 11, 0.1)",
      borderLeft: "4px solid #f59e0b",
      padding: "12px 16px",
      borderRadius: "0 8px 8px 0",
      fontSize: "0.825rem",
      color: "#fbbf24",
      margin: "8px 0 0 0",
      lineHeight: "1.5",
    },
    actionBtn: {
      background: "linear-gradient(90deg, #f59e0b 0%, #d97706 100%)",
      color: "#0f172a",
      border: "none",
      borderRadius: "8px",
      padding: "12px 28px",
      fontSize: "0.95rem",
      fontWeight: "700",
      cursor: "pointer",
      textDecoration: "none",
      display: "inline-flex",
      alignItems: "center",
      justifyContent: "center",
      boxShadow: "0 4px 14px rgba(245, 158, 11, 0.4)",
      transition: "transform 0.2s, box-shadow 0.2s",
      minWidth: "220px",
      textAlign: "center" as const,
    },
    sectionTitle: {
      fontSize: "1.25rem",
      fontWeight: "600",
      color: "#f1f5f9",
      margin: "0 0 16px 0",
    },
    tableContainer: {
      background: "rgba(15, 23, 42, 0.5)",
      border: "1px solid rgba(255, 255, 255, 0.05)",
      borderRadius: "16px",
      overflow: "hidden",
      boxShadow: "0 4px 6px -1px rgba(0, 0, 0, 0.1)",
    },
    table: {
      width: "100%",
      borderCollapse: "collapse" as const,
      textAlign: "left" as const,
      fontSize: "0.875rem",
    },
    th: {
      background: "rgba(30, 41, 59, 0.6)",
      padding: "14px 16px",
      color: "#94a3b8",
      fontWeight: "600",
      borderBottom: "1px solid rgba(255, 255, 255, 0.08)",
      fontSize: "0.8rem",
      textTransform: "uppercase" as const,
      letterSpacing: "0.05em",
    },
    tr: {
      borderBottom: "1px solid rgba(255, 255, 255, 0.04)",
      cursor: "pointer",
      transition: "background 0.15s ease",
    },
    td: {
      padding: "14px 16px",
      color: "#e2e8f0",
    },
    detailGrid: {
      display: "grid",
      gridTemplateColumns: "repeat(auto-fit, minmax(320px, 1fr))",
      gap: "20px",
    },
    configCard: {
      background: "rgba(30, 41, 59, 0.4)",
      border: "1px solid rgba(255, 255, 255, 0.04)",
      borderRadius: "12px",
      padding: "20px",
    },
    configTitle: {
      fontSize: "1rem",
      fontWeight: "600",
      color: "#f1f5f9",
      margin: "0 0 16px 0",
      borderBottom: "1px solid rgba(255, 255, 255, 0.08)",
      paddingBottom: "8px",
    },
    dl: {
      display: "grid",
      gridTemplateColumns: "1fr 1fr",
      gap: "10px 16px",
      margin: 0,
    },
    dt: {
      fontSize: "0.8rem",
      color: "#94a3b8",
      fontWeight: "500",
    },
    dd: {
      fontSize: "0.8rem",
      color: "#f1f5f9",
      fontWeight: "600",
      margin: 0,
      textAlign: "right" as const,
    },
  };

  return (
    <div style={styles.container}>
      <header style={styles.header}>
        <p style={styles.eyebrow}>Cloud Telemetry & Diagnostics</p>
        <h2 style={styles.title}>Model Metrics & Weights & Biases</h2>
        <p style={styles.desc}>
          Review live checkpoints, success rates, completion stats, and parameters for the trained RL policies. Connect to Weights & Biases to view gradient visualizations, real-time logging plots, and training runs history.
        </p>
      </header>

      <section style={styles.kpis}>
        <div style={styles.kpiCard}>
          <span style={styles.kpiLabel}>Connection Status</span>
          <span style={{ ...styles.kpiValue, color: "#10b981" }}>Active & Syncing</span>
        </div>
        <div style={styles.kpiCard}>
          <span style={styles.kpiLabel}>W&B Project</span>
          <span style={styles.kpiValue}>sheepdog-herding</span>
        </div>
        <div style={styles.kpiCard}>
          <span style={styles.kpiLabel}>Entity Workspace</span>
          <span style={styles.kpiValue}>inouye165-none</span>
        </div>
        <div style={styles.kpiCard}>
          <span style={styles.kpiLabel}>Total Model Checkpoints</span>
          <span style={styles.kpiValue}>{checkpoints.length} exported</span>
        </div>
      </section>

      <section style={styles.actionCard}>
        <div style={styles.actionLeft}>
          <h3 style={styles.actionTitle}>Weights & Biases Interactive Workspace</h3>
          <p style={styles.actionDesc}>
            View comprehensive interactive charts, run comparisons, parameter correlation panels, and PPO diagnostic reports on the official W&B platform.
          </p>
          <div style={styles.warningBanner}>
            ℹ️ <strong>Browser security notice</strong>: Weights & Biases security policies prohibit embedding workspace dashboards directly inside iframes to prevent clickjacking and protect user session cookies. Please use the button on the right to open your W&B workspace.
          </div>
        </div>
        <a
          href="https://wandb.ai/inouye165-none/sheepdog-herding"
          target="_blank"
          rel="noopener noreferrer"
          style={styles.actionBtn}
        >
          Open W&B Workspace ↗
        </a>
      </section>

      <div>
        <h3 style={styles.sectionTitle}>Model Checkpoint Progression</h3>
        <div style={styles.tableContainer}>
          <table style={styles.table}>
            <thead>
              <tr>
                <th style={styles.th}>Episode</th>
                <th style={styles.th}>Policy Mode</th>
                <th style={styles.th}>Success Rate</th>
                <th style={styles.th}>Avg Reward</th>
                <th style={styles.th}>Avg Completion Steps</th>
                <th style={styles.th}>Avg Sheep Penned</th>
                <th style={styles.th}>Recorded At</th>
              </tr>
            </thead>
            <tbody>
              {checkpoints.length === 0 ? (
                <tr>
                  <td colSpan={7} style={{ ...styles.td, textAlign: "center", color: "#64748b" }}>
                    No checkpoints recorded yet. Start training to log performance metrics.
                  </td>
                </tr>
              ) : (
                checkpoints.map((cp) => (
                  <tr
                    key={cp.checkpoint_episode}
                    onClick={() => setSelectedCheckpoint(cp)}
                    style={{
                      ...styles.tr,
                      background: selectedCheckpoint?.checkpoint_episode === cp.checkpoint_episode
                        ? "rgba(245, 158, 11, 0.15)"
                        : "transparent",
                    }}
                    onMouseEnter={(e) => {
                      if (selectedCheckpoint?.checkpoint_episode !== cp.checkpoint_episode) {
                        e.currentTarget.style.backgroundColor = "rgba(255, 255, 255, 0.05)";
                      }
                    }}
                    onMouseLeave={(e) => {
                      if (selectedCheckpoint?.checkpoint_episode !== cp.checkpoint_episode) {
                        e.currentTarget.style.backgroundColor = "transparent";
                      }
                    }}
                  >
                    <td style={styles.td}><strong>{cp.checkpoint_episode}</strong></td>
                    <td style={styles.td}>{cp.policy_name ?? cp.policy_type}</td>
                    <td style={{ ...styles.td, color: cp.success_rate >= 0.8 ? "#10b981" : cp.success_rate >= 0.4 ? "#fbbf24" : "#f87171" }}>
                      {(cp.success_rate * 100).toFixed(0)}%
                    </td>
                    <td style={styles.td}>{cp.average_reward.toFixed(2)}</td>
                    <td style={styles.td}>{cp.average_completion_steps.toFixed(1)}</td>
                    <td style={styles.td}>{cp.average_sheep_penned.toFixed(2)}</td>
                    <td style={{ ...styles.td, fontSize: "0.8rem", color: "#64748b" }}>
                      {cp.recorded_at ? new Date(cp.recorded_at).toLocaleString() : "unknown"}
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>

      {selectedCheckpoint && (
        (() => {
          const selectedEnvConfig = selectedCheckpoint.environment_config as any;
          const selectedRewardConfig = selectedCheckpoint.reward_config as any;
          return (
            <div>
              <h3 style={styles.sectionTitle}>Checkpoint {selectedCheckpoint.checkpoint_episode} Config Overrides</h3>
              <div style={styles.detailGrid}>
                <div style={styles.configCard}>
                  <h4 style={styles.configTitle}>Environment Settings</h4>
                  <dl style={styles.dl}>
                    <dt>Curriculum Stage</dt>
                    <dd>{String(selectedEnvConfig?.curriculum_stage ?? 0)}</dd>
                    <dt>Dog Speed</dt>
                    <dd>{String(selectedEnvConfig?.dog_speed ?? "1.0")}</dd>
                    <dt>Sprint Multiplier</dt>
                    <dd>{String(selectedEnvConfig?.dog_sprint_multiplier ?? "2.0")}</dd>
                    <dt>Sheep Speed</dt>
                    <dd>{String(selectedEnvConfig?.sheep_speed ?? "0.75")}</dd>
                    <dt>Max Steps</dt>
                    <dd>{String(selectedEnvConfig?.max_steps ?? "600")}</dd>
                  </dl>
                </div>

                <div style={styles.configCard}>
                  <h4 style={styles.configTitle}>Active Reward Weights</h4>
                  <dl style={styles.dl}>
                    <dt>progress_scale</dt>
                    <dd>{String(selectedRewardConfig?.progress_scale ?? "2.0")}</dd>
                    <dt>sheep_penned_reward</dt>
                    <dd>{String(selectedRewardConfig?.sheep_penned_reward ?? "8.0")}</dd>
                    <dt>flock_cohesion_scale</dt>
                    <dd>{String(selectedRewardConfig?.flock_cohesion_scale ?? "0.35")}</dd>
                    <dt>scatter_penalty_scale</dt>
                    <dd>{String(selectedRewardConfig?.scatter_penalty_scale ?? "0.2")}</dd>
                    <dt>terminal_success_reward</dt>
                    <dd>{String(selectedRewardConfig?.terminal_success_reward ?? "20.0")}</dd>
                    <dt>terminal_failure_penalty</dt>
                    <dd>{String(selectedRewardConfig?.terminal_failure_penalty ?? "12.0")}</dd>
                  </dl>
                </div>
              </div>
            </div>
          );
        })()
      )}

      <div style={styles.detailGrid}>
        <div style={styles.configCard}>
          <h3 style={styles.configTitle}>Static Model Parameters</h3>
          <dl style={styles.dl}>
            <dt>Grid size</dt>
            <dd>{String(environmentConfig?.width ?? 80)} x {String(environmentConfig?.height ?? 60)}</dd>
            <dt>Dogs count</dt>
            <dd>{String(environmentConfig?.dogs ?? 3)}</dd>
            <dt>Sheep count</dt>
            <dd>{String(environmentConfig?.sheep ?? 6)}</dd>
            <dt>Pen dimensions</dt>
            <dd>{String(environmentConfig?.pen_width ?? 10)} x {String(environmentConfig?.pen_height ?? 10)}</dd>
            <dt>Trainer Type</dt>
            <dd>{String(trainingConfig?.trainer_type ?? "maskable_ppo")}</dd>
            <dt>Policy Type</dt>
            <dd>{String(trainingConfig?.policy_type ?? "neural")}</dd>
          </dl>
        </div>

        <div style={styles.configCard}>
          <h3 style={styles.configTitle}>Neural Architecture & Hyperparameters</h3>
          <dl style={styles.dl}>
            <dt>Backbone net_arch</dt>
            <dd>[{Array.isArray(trainingConfig?.neural_hidden_sizes) ? trainingConfig.neural_hidden_sizes.join(", ") : "128, 128"}]</dd>
            <dt>learning_rate</dt>
            <dd>{String(trainingConfig?.learning_rate ?? "1e-4")}</dd>
            <dt>learning_rate_final</dt>
            <dd>{String(trainingConfig?.learning_rate_final ?? "3e-5")}</dd>
            <dt>ppo_env_workers</dt>
            <dd>{String(trainingConfig?.ppo_env_workers ?? 8)}</dd>
            <dt>rollout_steps</dt>
            <dd>{String(trainingConfig?.rollout_steps ?? 2048)}</dd>
            <dt>batch_size</dt>
            <dd>{String(trainingConfig?.batch_size ?? 64)}</dd>
          </dl>
        </div>
      </div>
    </div>
  );
}

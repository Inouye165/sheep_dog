import { useState } from "react";
import { loadHyperparams, loadTrainingDiagnostics } from "../lib/api";
import type { TrainingStatus, CheckpointIndex, CheckpointEntry, DiagnosticsResponse } from "../state/types";

const STAGE_DESCRIPTIONS: Record<number, string> = {
  0: "Full problem — 3 dogs, 6 sheep, 80×60 grid",
  1: "1 dog · 1 sheep · fixed easy penning",
  2: "1 dog · 1 sheep · mild start randomization",
  3: "1 dog · 2 sheep · fixed mini-flock",
  4: "1 dog · 2 sheep · randomized mini-flock",
  5: "2 dogs · 3 sheep · fixed teamwork",
  6: "2 dogs · 3 sheep · tiny nearby stray starts",
  7: "2 dogs · 4 sheep · early nearby stray collection",
  8: "3 dogs · 4 sheep · nearby stray emphasis",
  9: "3 dogs · 4 sheep · stronger nearby stray recovery",
  10: "3 dogs · 5 sheep · nearby + first farther strays",
  11: "3 dogs · 5 sheep · farther stray recovery",
  12: "3 dogs · 6 sheep · group + one stray",
  13: "3 dogs · 6 sheep · two nearby strays",
  14: "3 dogs · 6 sheep · split flock (3+3)",
  15: "3 dogs · 6 sheep · partially scattered",
  16: "3 dogs · 6 sheep · scattered sheep",
  17: "3 dogs · 6 sheep · moving pen same wall",
  18: "3 dogs · 6 sheep · any-wall pen",
  19: "3 dogs · 6 sheep · wall pen away from corners",
  20: "3 dogs · 6 sheep · interior pen",
  21: "3 dogs · 6 sheep · random pen + random sheep",
  22: "3 dogs · 6 sheep · wider split/stray recovery",
  23: "3 dogs · 6 sheep · heavy scattered recovery",
  24: "3 dogs · 6 sheep · all-corners starts",
  25: "3 dogs · 6 sheep · bridge: corner-heavy starts before hard random mix",
  26: "3 dogs · 6 sheep · hard spawn mix (no personality bias)",
  27: "3 dogs · 6 sheep · add mild personality variation",
  28: "3 dogs · 6 sheep · moderate personality variation",
  29: "3 dogs · 6 sheep · bridge: weaker cohesion but still pressure-coupled",
  30: "3 dogs · 6 sheep · disable no-pressure cohesion",
  31: "3 dogs · 6 sheep · reduce cohesion + stronger personalities",
  32: "3 dogs · 6 sheep · lowest cohesion + strongest personalities",
};

const PLATEAU_WINDOW = 5;
const PLATEAU_MIN_DELTA = 0.02;
const CLIFF_THRESHOLD = 0.05;
const CLIFF_MIN_CHECKPOINTS = 8;

interface PlateauInfo {
  kind: "plateau" | "cliff" | "spike";
  window: number;
  bestRate: number;
  allTimeBest: number;
  sinceEpisode: number;
}

function detectPlateau(checkpoints: CheckpointEntry[]): PlateauInfo | null {
  if (checkpoints.length < PLATEAU_WINDOW) return null;
  const recent = checkpoints.slice(-PLATEAU_WINDOW);
  const allPrior = checkpoints.slice(0, -PLATEAU_WINDOW);
  const bestPrior = allPrior.length > 0 ? Math.max(...allPrior.map((c) => c.success_rate)) : -Infinity;
  const bestRecent = Math.max(...recent.map((c) => c.success_rate));
  const allTimeBest = Math.max(...checkpoints.map((c) => c.success_rate));

  if (bestRecent <= bestPrior + PLATEAU_MIN_DELTA) {
    const everSucceeded = allTimeBest > CLIFF_THRESHOLD;
    const kind =
      checkpoints.length >= CLIFF_MIN_CHECKPOINTS && bestRecent < CLIFF_THRESHOLD
        ? everSucceeded ? "spike" : "cliff"
        : "plateau";
    return {
      kind,
      window: PLATEAU_WINDOW,
      bestRate: bestRecent,
      allTimeBest,
      sinceEpisode: recent[0].checkpoint_episode,
    };
  }
  return null;
}

interface CopyAgentDataButtonProps {
  trainingStatus: TrainingStatus | null;
  checkpointIndex: CheckpointIndex | null;
  curriculumStage: number;
}

export function CopyAgentDataButton({
  trainingStatus,
  checkpointIndex,
  curriculumStage,
}: CopyAgentDataButtonProps) {
  const [copied, setCopied] = useState(false);
  const [loading, setLoading] = useState(false);
  const [failed, setFailed] = useState(false);

  const handleCopy = async () => {
    if (loading) return;
    setLoading(true);
    setFailed(false);

    try {
      // 1. Fetch live hyperparameters from API
      let hyperparams = null;
      try {
        hyperparams = await loadHyperparams();
      } catch (err) {
        console.error("Failed to load hyperparams for copy report:", err);
      }

      // Fetch diagnostics from API
      let diagnostics: DiagnosticsResponse | null = null;
      let apiErrorDetails: {
        endpoint: string;
        status: number | string;
        message: string;
        responseIsNull: boolean;
        jsonParsingFailed: boolean;
      } | null = null;

      try {
        diagnostics = await loadTrainingDiagnostics(
          trainingStatus?.active_checkpoint_id ?? undefined,
          trainingStatus?.checkpoint_episode ?? undefined
        );
        if (!diagnostics) {
          apiErrorDetails = {
            endpoint: "/api/training/diagnostics",
            status: "unknown",
            message: "Response body was null or undefined",
            responseIsNull: true,
            jsonParsingFailed: false
          };
          setFailed(true);
        } else if (!diagnostics.diagnosticsAvailable || diagnostics.error) {
          apiErrorDetails = {
            endpoint: diagnostics.error?.endpoint || "/api/training/diagnostics",
            status: diagnostics.error ? "500" : "unknown",
            message: diagnostics.error?.message || "Diagnostics unavailable from backend",
            responseIsNull: false,
            jsonParsingFailed: false
          };
          setFailed(true);
        }
      } catch (err) {
        console.error("Failed to load training diagnostics for copy report:", err);
        const fetchError = err as { status?: number; message?: string };
        const isJsonErr = fetchError.message?.includes("JSON") || fetchError.message?.includes("Unexpected token");
        apiErrorDetails = {
          endpoint: "/api/training/diagnostics",
          status: fetchError.status ?? "unknown",
          message: fetchError.message ?? String(err),
          responseIsNull: false,
          jsonParsingFailed: !!isJsonErr
        };
        setFailed(true);
      }

      // 2. Extract checkpoints
      const checkpoints = checkpointIndex?.checkpoints ?? [];

      // 3. Format the markdown report
      const report = formatAgentReport(
        trainingStatus,
        checkpoints,
        hyperparams,
        curriculumStage,
        diagnostics,
        apiErrorDetails
      );

      // 4. Write to clipboard
      await navigator.clipboard.writeText(report);
      if (!apiErrorDetails) {
        setCopied(true);
        setTimeout(() => {
          setCopied(false);
        }, 2000);
      }
    } catch (err) {
      console.error("Failed to copy agent data to clipboard:", err);
      alert("Could not copy agent data. Please check browser permissions.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <button
      onClick={handleCopy}
      className={`agent-copy-btn ${copied ? "agent-copy-btn--success" : ""} ${failed ? "agent-copy-btn--failed" : ""}`}
      disabled={loading}
      title="Copy all training metrics, hyperparameters, and checkpoints for an AI agent"
      aria-label="Copy agent data to clipboard"
    >
      {copied ? (
        <>
          <svg
            width="13"
            height="13"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="3"
            strokeLinecap="round"
            strokeLinejoin="round"
            style={{ marginRight: "0.35rem" }}
          >
            <polyline points="20 6 9 17 4 12" />
          </svg>
          Agent Data Copied!
        </>
      ) : failed ? (
        <>
          <svg
            width="13"
            height="13"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="3"
            strokeLinecap="round"
            strokeLinejoin="round"
            style={{ marginRight: "0.35rem" }}
          >
            <circle cx="12" cy="12" r="10" />
            <line x1="12" y1="8" x2="12" y2="12" />
            <line x1="12" y1="16" x2="12.01" y2="16" />
          </svg>
          Diagnostics unavailable
        </>
      ) : (
        <>
          <svg
            width="13"
            height="13"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2.5"
            strokeLinecap="round"
            strokeLinejoin="round"
            style={{ marginRight: "0.35rem" }}
          >
            <path d="M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2" />
            <rect x="8" y="2" width="8" height="4" rx="1" ry="1" />
          </svg>
          Copy Agent Data
        </>
      )}
    </button>
  );
}

function formatVal(val: any, reason?: string): string {
  if (val === null || val === undefined) {
    return reason ? `Unavailable (${reason})` : "Unavailable";
  }
  return val.toString();
}

function formatPct(val: any, reason?: string): string {
  if (val === null || val === undefined) {
    return reason ? `Unavailable (${reason})` : "Unavailable";
  }
  return `${Math.round(val * 100)}%`;
}

function formatNum(val: any, decimals = 2, reason?: string): string {
  if (val === null || val === undefined || isNaN(val)) {
    return reason ? `Unavailable (${reason})` : "Unavailable";
  }
  return typeof val === "number" ? val.toFixed(decimals) : val;
}

function formatAgentReport(
  status: TrainingStatus | null,
  checkpoints: CheckpointEntry[],
  hyperparams: any,
  currentStage: number,
  diagnostics: DiagnosticsResponse | null,
  apiErrorDetails?: {
    endpoint: string;
    status: number | string;
    message: string;
    responseIsNull: boolean;
    jsonParsingFailed: boolean;
  } | null
): string {
  const timestamp = new Date().toISOString();
  const stageDesc = STAGE_DESCRIPTIONS[currentStage] ?? "Unknown Stage";

  const stageScoped = checkpoints.filter(
    (c) => (c.reward_config?.instincts?.curriculum_stage ?? 0) === currentStage
  );
  const plateau = detectPlateau(stageScoped);

  let learningCurveStatus = "STILL_LEARNING (Normal progress)";
  let analysisExplanation = "The agent is showing active performance gains on this stage.";
  
  if (plateau) {
    if (plateau.kind === "cliff") {
      learningCurveStatus = "ROADBLOCK (Cliff - stuck at zero)";
      analysisExplanation = `Success rate remains under ${Math.round(CLIFF_THRESHOLD * 100)}% for the last ${plateau.window} checkpoints. Recommend rolling back or adjusting hyperparams (e.g. increase learning rate, adjust reward scaling, or check policy inputs).`;
    } else if (plateau.kind === "spike") {
      learningCurveStatus = "ROADBLOCK (Spike and drop - regression)";
      analysisExplanation = `Agent previously achieved ${Math.round(plateau.allTimeBest * 100)}% success but has regressed below ${Math.round(CLIFF_THRESHOLD * 100)}% in the recent window. High risk of policy collapse. Recommend rolling back to a stable checkpoint and lowering learning rate or increasing entropy coefficient.`;
    } else if (plateau.kind === "plateau") {
      learningCurveStatus = "PLATEAU (No recent improvement)";
      analysisExplanation = `Success rate has stagnated around ${Math.round(plateau.bestRate * 100)}% for the last ${plateau.window} checkpoints. If success rate is high (>= 50%), you should promote manually or wait for auto-promotion. If success rate is low, adjust training parameters.`;
    }
  }

  // Helpers
  const fVal = (v: any, r = "Unavailable") => formatVal(v, r);
  const fPct = (v: any, r = "Unavailable") => formatPct(v, r);
  const fNum = (v: any, d = 2, r = "Unavailable") => formatNum(v, d, r);

  // Compute Plateau and Derivative analysis
  const dSuccess3 = computeDerivative(checkpoints, 3, "success_rate");
  const dSuccess5 = computeDerivative(checkpoints, 5, "success_rate");
  const dSuccess10 = computeDerivative(checkpoints, 10, "success_rate");
  const dSteps3 = computeDerivative(checkpoints, 3, "average_completion_steps");
  const dSteps5 = computeDerivative(checkpoints, 5, "average_completion_steps");
  const dSteps10 = computeDerivative(checkpoints, 10, "average_completion_steps");

  const snapshotData = diagnostics?.snapshot;
  const snap: any = snapshotData?.snapshot || {};
  const completeness: any = snapshotData?.completeness || {};
  const modelArch: any = snapshotData?.neural_architecture || {};
  const reconciliations: any[] = snapshotData?.reward_reconciliations || [];
  const geometryValidations: Record<string, any> = snapshotData?.eval_geometry_validations || {};
  const failedTrajectories: Record<string, any[]> = snapshotData?.failed_seed_trajectories || {};
  const counterReconciliation: any = snapshotData?.counter_reconciliation || {};
  const apiWarnings: string[] = snapshotData?.health_warnings || [];
  const evalRecords: any[] = snapshotData?.evaluation_records || [];

  // Action count aggregates
  let totalSteps = 0;
  let totalWaits = 0;
  let totalSprints = 0;
  let totalInvalidActions = 0;
  let hasValidActionMetrics = false;

  for (const r of evalRecords) {
    if (r.steps !== undefined && r.steps !== null) {
      totalSteps += r.steps;
    }
    if (r.num_waits !== undefined && r.num_waits !== null) {
      totalWaits += r.num_waits;
      hasValidActionMetrics = true;
    }
    if (r.num_sprints !== undefined && r.num_sprints !== null) {
      totalSprints += r.num_sprints;
      hasValidActionMetrics = true;
    }
    if (r.num_invalid_actions !== undefined && r.num_invalid_actions !== null) {
      totalInvalidActions += r.num_invalid_actions;
      hasValidActionMetrics = true;
    }
  }

  // Unified health warnings and readiness calculations
  const healthWarnings: string[] = [];
  let readinessReasons: string[] = [];
  let readiness = "READY";

  if (!diagnostics || !diagnostics.diagnosticsAvailable || !snapshotData) {
    readiness = "NOT READY";
    readinessReasons = ["Diagnostics API request failed or returned an error."];
    healthWarnings.push("Critical warning: Diagnostics API snapshot is null/failed.");
    healthWarnings.push("Missing active run identity.");
    healthWarnings.push("Missing active checkpoint identity.");
    healthWarnings.push("Missing policy version for the current instrumented run.");
    healthWarnings.push("Missing neural-network architecture details.");
    healthWarnings.push("Missing per-seed evaluation records.");
    healthWarnings.push("Missing counter reconciliation data.");
  } else {
    readiness = completeness.readiness || "READY";
    readinessReasons = completeness.reasons || [];

    // Add health warnings from backend
    if (apiWarnings && apiWarnings.length > 0) {
      for (const w of apiWarnings) {
        healthWarnings.push(w);
      }
    }

    // Missing PPO stats warning
    const ppoMetrics = snapshotData.ppo_metrics || [];
    if (!ppoMetrics || ppoMetrics.length === 0) {
      healthWarnings.push("Missing PPO training progress metrics.");
    }
    // Missing observation diagnostics warning
    if (!snapshotData.observation_diagnostics) {
      healthWarnings.push("Missing observation diagnostics.");
    }
    // Missing action diagnostics warning
    if (hasValidActionMetrics && totalSteps === 0) {
      healthWarnings.push("Missing action choice metrics.");
    }

    // Fractional completed episodes check
    if (status?.completed_episodes !== undefined && !Number.isInteger(status.completed_episodes)) {
      healthWarnings.push(`Fractional completed episodes detected in status: completed_episodes is ${status.completed_episodes}.`);
    }

    // Inconsistencies check
    if (status) {
      if (status.latest_success_rate !== null && status.latest_success_rate !== undefined && evalRecords && evalRecords.length > 0) {
        const recordSuccessCount = evalRecords.filter((r: any) => r.success).length;
        const recordSuccessRate = recordSuccessCount / evalRecords.length;
        if (Math.abs(status.latest_success_rate - recordSuccessRate) > 0.01) {
          healthWarnings.push(`Success-rate/per-seed inconsistency: Status success rate is ${Math.round(status.latest_success_rate * 100)}% but evaluation records success rate is ${Math.round(recordSuccessRate * 100)}% (${recordSuccessCount}/${evalRecords.length}).`);
        }
        
        const recordStoppedCount = evalRecords.filter((r: any) => r.stop_reason === "stopped" || r.stopped).length;
        const recordStoppedRate = recordStoppedCount / evalRecords.length;
        if (status.latest_stopped_rate !== null && status.latest_stopped_rate !== undefined) {
          if (Math.abs(status.latest_stopped_rate - recordStoppedRate) > 0.01) {
            healthWarnings.push(`Stopped-rate/per-seed inconsistency: Status stopped rate is ${Math.round(status.latest_stopped_rate * 100)}% but evaluation records stopped rate is ${Math.round(recordStoppedRate * 100)}% (${recordStoppedCount}/${evalRecords.length}).`);
          }
        }
      }
    }
  }

  let readinessExplanation = "All critical diagnostic sections are fully populated.";
  if (readiness === "PARTIAL") {
    readinessExplanation = "Some secondary diagnostic information is missing, but core identifiers are present.";
  } else if (readiness === "NOT READY") {
    readinessExplanation = "Critical diagnostic sections are missing or unavailable. Do not use this report for hyperparameter decisions.";
  }

  let md = `# SHEEPDOG AGENT DIAGNOSTICS REPORT\n`;
  md += `Generated: ${timestamp}\n\n`;

  if (apiErrorDetails) {
    md += `> [!CAUTION]\n`;
    md += `> **DIAGNOSTICS API FAILURE**\n`;
    md += `> - **Endpoint Called**: \`${apiErrorDetails.endpoint}\`\n`;
    md += `> - **HTTP Status**: ${apiErrorDetails.status}\n`;
    md += `> - **Error Message**: ${apiErrorDetails.message}\n`;
    md += `> - **Response Is Null**: ${apiErrorDetails.responseIsNull ? "Yes" : "No"}\n`;
    md += `> - **JSON Parsing Failed**: ${apiErrorDetails.jsonParsingFailed ? "Yes" : "No"}\n\n`;
  }

  // 1. Diagnostic Completeness Table and AI Readiness
  md += `## 1. Diagnostic Completeness and AI Readiness\n`;
  md += `- **AI Review Readiness**: **${readiness}**\n`;
  if (readinessReasons.length > 0) {
    md += `  - *Reasons for readiness status*:\n`;
    for (const r of readinessReasons) {
      md += `    - ${r}\n`;
    }
  } else {
    md += `  - *Explanation*: ${readinessExplanation}\n`;
  }
  md += `\n| Diagnostic Area | Completeness Status | Source | Missing/Unrecorded Fields |\n`;
  md += `|---|---|---|---|\n`;
  if (completeness.table) {
    for (const row of completeness.table) {
      md += `| ${row.area} | ${row.status} | ${row.source} | ${row.missing.join(", ") || "None"} |\n`;
    }
  } else {
    md += `| Table data unavailable | | | |\n`;
  }
  md += `\n`;

  // 2. System Overview and Report Identity
  md += `## 2. System Overview and Report Identity\n`;
  md += `- **Report Generation Time**: ${timestamp}\n`;
  md += `- **Snapshot Timestamp**: ${fVal(snap.snapshot_timestamp)}\n`;
  md += `- **Active Run ID**: ${fVal(snap.active_run_id)}\n`;
  md += `- **Loaded Model ID**: ${fVal(snap.loaded_model_id)}\n`;
  md += `- **Active Checkpoint ID**: ${fVal(snap.active_checkpoint_id)}\n`;
  if (snap.is_legacy) {
    md += `- **Legacy Policy Version**: unknown\n`;
    md += `- **Legacy PPO Update Count**: unknown\n`;
    md += `- **Diagnostic Baseline Version**: 0\n`;
    md += `- **Updates since Instrumentation**: 0\n`;
  } else {
    md += `- **Policy Version / Update Number**: ${fVal(snap.policy_version)}\n`;
    md += `- **PPO Update Count**: ${fVal(snap.ppo_update_count)}\n`;
    md += `- **Diagnostic Baseline Version**: 0\n`;
    md += `- **Updates since Instrumentation**: ${snap.ppo_update_count !== null && snap.ppo_update_count !== undefined ? snap.ppo_update_count : 0}\n`;
  }
  md += `- **Current Curriculum Stage**: Stage ${currentStage} (${stageDesc})\n`;
  md += `- **Training Phase**: ${fVal(status?.phase)} (${fVal(status?.message)})\n`;
  md += `- **Learning Curve Status**: **${learningCurveStatus}**\n`;
  md += `  - *Explanation*: ${analysisExplanation}\n`;
  
  if (status?.anti_collapse_warning) {
    md += `- **Anti-Collapse Warning**: **TRIGGERED!**\n`;
    md += `  - *Message*: ${status.anti_collapse_warning.message}\n`;
    if (status.anti_collapse_warning.recommendation) {
      md += `  - *Recommendation*: ${status.anti_collapse_warning.recommendation}\n`;
    }
  } else {
    md += `- **Anti-Collapse Warning**: None triggered.\n`;
  }

  md += `\n### Compatibility & Data Freshness\n`;
  md += `- **Observation Schema Hash**: ${fVal(snap.observation_schema_hash)}\n`;
  md += `- **Action Space Hash**: ${fVal(snap.action_space_hash)}\n`;
  md += `- **Reward Schema Version**: ${fVal(snap.reward_schema_version)}\n`;
  md += `- **Environment Config Version**: ${fVal(status?.latest_checkpoint_episode ? "v1.0" : null)}\n`;
  md += `- **Training Start Time**: ${fVal(status?.training_start_time)}\n`;
  md += `- **Last Policy Update Time**: ${fVal(status?.last_policy_update_time)}\n`;
  md += `- **Last Evaluation Time**: ${fVal(snap.evaluation_timestamp)}\n`;

  if (healthWarnings.length > 0) {
    md += `\n### ⚠️ Automated Health Warnings\n`;
    for (const w of healthWarnings) {
      md += `- **WARNING**: ${w}\n`;
    }
  } else {
    md += `\n### ⚠️ Automated Health Warnings\n`;
    md += `- No active warnings. System metrics are consistent and fresh.\n`;
  }

  if (snapshotData?.environment_mismatches && snapshotData.environment_mismatches.length > 0) {
    md += `\n### ⚠️ Training vs Evaluation Environment Mismatches\n`;
    for (const m of snapshotData.environment_mismatches) {
      md += `- **[${m.component.toUpperCase()} MISMATCH]** Field \`${m.field}\` differs: Training value = \`${m.training_value}\` vs Evaluation value = \`${m.evaluation_value}\` (${m.severity}).\n`;
    }
  }

  // 3. Evaluation Metrics
  md += `\n## 3. Evaluation Metrics\n`;

  md += `\n### Latest Completed Evaluation\n`;
  const evalVersion = snap.evaluation_policy_version;
  const currentPolicyVersion = status?.policy_version;
  const isStaleEvaluation = evalVersion !== null && currentPolicyVersion !== undefined && evalVersion < currentPolicyVersion;
  if (isStaleEvaluation) {
    md += `> [!WARNING]\n`;
    md += `> **Latest completed evaluation is stale relative to the active policy.**\n`;
    md += `> - Evaluated Policy Version: \`v${evalVersion}\` (Checkpoint: ${snap.evaluation_checkpoint_id || "N/A"})\n`;
    md += `> - Active Policy Version: \`v${currentPolicyVersion}\` (Currently training/queued)\n\n`;
  }

  if (snap.evaluation_timestamp && snap.evaluation_timestamp !== "Unknown") {
    md += `- **Checkpoint Evaluated**: ${fVal(snap.evaluation_checkpoint_id || (snap.evaluation_checkpoint_episode !== undefined ? `ep ${snap.evaluation_checkpoint_episode}` : null))}\n`;
    md += `- **Policy Version Evaluated**: v${fVal(evalVersion)}\n`;
    md += `- **Evaluation Timestamp**: ${fVal(snap.evaluation_timestamp)}\n`;
    if (status) {
      md += `- **Success Rate**: ${status.latest_success_rate !== null && status.latest_success_rate !== undefined ? `${Math.round(status.latest_success_rate * 100)}%` : "N/A"}\n`;
      md += `- **Avg Reward**: ${status.latest_avg_reward ?? "N/A"}\n`;
      md += `- **Timeout Rate**: ${status.latest_timeout_rate !== null && status.latest_timeout_rate !== undefined ? `${Math.round(status.latest_timeout_rate * 100)}%` : "N/A"}\n`;
      md += `- **Stopped Rate**: ${status.latest_stopped_rate !== null && status.latest_stopped_rate !== undefined ? `${Math.round(status.latest_stopped_rate * 100)}%` : "N/A"}\n`;
      md += `- **Avg No-Progress Steps**: ${status.latest_avg_no_progress_steps ?? "N/A"}\n`;
      md += `- **Avg Sheep Penned**: ${status.latest_avg_sheep_penned ?? "N/A"}\n`;
      md += `- **Avg Distance to Pen**: ${status.latest_avg_distance_to_pen ?? "N/A"}\n`;
      md += `- **Avg Flock Spread**: ${status.latest_avg_flock_spread ?? "N/A"}\n`;
      md += `- **Avg Farthest Distance to Pen**: ${status.latest_avg_farthest_distance_to_pen ?? "N/A"}\n`;
      md += `- **Avg Farthest Distance to Flock Center**: ${status.latest_avg_farthest_distance_to_flock_center ?? "N/A"}\n`;
    }
  } else {
    md += `- **Status**: No completed evaluations found.\n`;
  }

  md += `\n### Current Active Policy\n`;
  if (currentPolicyVersion !== undefined && currentPolicyVersion !== null) {
    md += `- **Active Policy Version**: v${currentPolicyVersion}\n`;
    const hasEvaluatedCurrent = evalVersion !== null && evalVersion === currentPolicyVersion;
    if (hasEvaluatedCurrent) {
      md += `- **Evaluation Status**: Evaluated (Results match the Latest Completed Evaluation above)\n`;
    } else {
      md += `- **Evaluation Status**: Evaluation pending (No evaluation has run for version v${currentPolicyVersion} yet)\n`;
    }
  } else {
    md += `- **Active Policy Version**: Unknown\n`;
    md += `- **Evaluation Status**: Evaluation pending / unknown\n`;
  }

  if (status) {
    md += `- **Total Episodes in Current Run**: ${status.completed_episodes} / ${status.requested_episodes}\n`;
    if (status.estimated_equivalent_episodes !== undefined && status.estimated_equivalent_episodes !== null) {
      md += `- **Estimated Equivalent Episodes**: ${status.estimated_equivalent_episodes.toFixed(3)}\n`;
    }
    md += `- **Grand Total Trained Episodes**: ${status.total_episodes_trained ?? 0}\n`;
  } else {
    md += `No training status details available.\n`;
  }

  if (status?.auto_promote_gate) {
    const gate = status.auto_promote_gate;
    md += `\n### Auto-Promote Gate Diagnostics\n`;
    md += `- **Gate Decision**: \`${gate.decision.toUpperCase()}\` (Reason: ${gate.reason})\n`;
    md += `- **Streak Count**: ${gate.qualified_streak} / ${gate.min_qualified_streak} qualified batches\n`;
    md += `- **Seed Check**: ${gate.seed_count} seeds evaluated (Gate target: ${gate.seed_gate_target_met ? "MET" : "NOT MET"})\n`;
    md += `- **Success Rate Check**: ${gate.success_count} fully success seeds (Gate target: ${gate.success_threshold * 100}% threshold, current average is ${gate.success_rate_ok ? "OK" : "BELOW TARGET"})\n`;
    md += `- **Timeout Rate Check**: ${gate.timeout_ok ? "OK" : "TOO HIGH"}\n`;
  }

  if (snapshotData?.scenario_coverage) {
    const cov = snapshotData.scenario_coverage;
    md += `\n### Training Scenario Coverage & Exposure\n`;
    md += `- **Unique Seeds Trained On**: ${cov.unique_seeds_count}\n`;
    md += `- **Unique Configurations (Positions) Encountered**: ${cov.unique_configs_count}\n`;
    md += `- **Starting Sheep-to-Pen Distance**: Min: ${fNum(cov.sheep_to_pen_distance?.min)}, Max: ${fNum(cov.sheep_to_pen_distance?.max)}, Avg: ${fNum(cov.sheep_to_pen_distance?.avg)}\n`;
    md += `- **Starting Dog-to-Sheep Distance**: Min: ${fNum(cov.dog_to_sheep_distance?.min)}, Max: ${fNum(cov.dog_to_sheep_distance?.max)}, Avg: ${fNum(cov.dog_to_sheep_distance?.avg)}\n`;
    md += `- **Resemblance to Evaluation Layouts**:\n`;
    for (const seed of [11, 23, 37, 41, 53]) {
      const seedStr = seed.toString();
      const count = cov.resemblance_counts?.[seedStr] ?? 0;
      const succ = cov.resemblance_successes?.[seedStr] ?? 0;
      const rate = count > 0 ? `${Math.round((succ / count) * 100)}%` : "N/A";
      md += `  - **Seed ${seed}**: Resembled in ${count} training episodes (Success Rate: ${rate} [${succ}/${count}])\n`;
    }
  }

  // 4. Active Hyperparameters
  md += `\n## 3. Active Hyperparameters\n`;
  if (hyperparams) {
    if (hyperparams.training) {
      md += `### Training Configuration\n`;
      md += `- **Learning Rate**: ${hyperparams.training.learning_rate} (Final: ${hyperparams.training.learning_rate_final})\n`;
      md += `- **Entropy Coefficient**: ${hyperparams.training.entropy_coef}\n`;
      md += `- **Clip Range**: ${hyperparams.training.clip_range}\n`;
      md += `- **Rollout Steps**: ${hyperparams.training.rollout_steps}\n`;
      md += `- **Batch Size**: ${hyperparams.training.batch_size}\n`;
      md += `- **Gamma**: ${hyperparams.training.gamma}\n`;
      md += `- **GAE Lambda**: ${hyperparams.training.gae_lambda}\n`;
      md += `- **Value Coefficient**: ${hyperparams.training.value_coef}\n`;
    }
    if (hyperparams.rewards) {
      md += `\n### Reward Weights\n`;
      md += `- **Terminal Success Reward**: ${hyperparams.rewards.terminal_success_reward}\n`;
      md += `- **Terminal Failure Penalty**: ${hyperparams.rewards.terminal_failure_penalty}\n`;
      md += `- **Time Penalty**: ${hyperparams.rewards.time_penalty}\n`;
      md += `- **Progress Scale**: ${hyperparams.rewards.progress_scale}\n`;
      md += `- **Sheep Penned Reward**: ${hyperparams.rewards.sheep_penned_reward}\n`;
      md += `- **Wait Penalty**: ${hyperparams.rewards.wait_penalty}\n`;
      md += `- **No Progress Penalty**: ${hyperparams.rewards.no_progress_penalty}\n`;
      md += `- **Flock Cohesion Scale**: ${hyperparams.rewards.flock_cohesion_scale}\n`;
      md += `- **Scatter Penalty Scale**: ${hyperparams.rewards.scatter_penalty_scale}\n`;
      md += `- **Sprint Cost Scale**: ${hyperparams.rewards.sprint_cost_scale}\n`;
    }
    if (hyperparams.environment) {
      md += `\n### Environment Settings\n`;
      md += `- **Sheep Speed**: ${hyperparams.environment.sheep_speed}\n`;
      md += `- **Sheep Vision**: ${hyperparams.environment.sheep_vision}\n`;
      md += `- **Dog Speed**: ${hyperparams.environment.dog_speed} (Sprint: ${hyperparams.environment.dog_sprint_multiplier}x)\n`;
      md += `- **Dog Vision**: ${hyperparams.environment.dog_vision}\n`;
      md += `- **Flock Radius**: ${hyperparams.environment.flock_radius}\n`;
      md += `- **Sheep Personality Strength**: ${hyperparams.environment.sheep_personality_strength}\n`;
    }
  }

  if (snapshotData?.config_snapshot) {
    md += `\n### Config Snapshot & Precedence\n`;
    md += `| Parameter | Default | UI | Stage Override | Checkpoint | Active | Source |\n`;
    md += `|---|---|---|---|---|---|---|\n`;
    for (const [path, info] of Object.entries(snapshotData.config_snapshot)) {
      const inf = info as any;
      md += `| ${path} | ${fVal(inf.default)} | ${fVal(inf.ui)} | ${fVal(inf.stage)} | ${fVal(inf.checkpoint)} | ${fVal(inf.active)} | **${inf.source.toUpperCase()}** |\n`;
    }
    if (snapshotData.config_anomalies && snapshotData.config_anomalies.length > 0) {
      md += `\n**Hyperparameter Anomalies/Conflicts:**\n`;
      for (const anomaly of snapshotData.config_anomalies) {
        md += `- ⚠️ ${anomaly}\n`;
      }
    }
  }

  // 5. Neural Network Architecture
  md += `\n## 4. Neural-Network Architecture\n`;
  if (modelArch.status === "COMPLETE") {
    md += `- **Algorithm**: ${modelArch.algorithm}\n`;
    md += `- **Policy Class**: ${modelArch.policy_class}\n`;
    md += `- **Feed-forward or Recurrent**: ${modelArch.feed_forward_or_recurrent}\n`;
    md += `- **Observation Space**: shape = ${JSON.stringify(modelArch.observation_space_shape)}, type = ${modelArch.observation_data_type} (${modelArch.observation_feature_count} features)\n`;
    md += `- **Feature Extractor**: ${modelArch.feature_extractor_class} (Output Dim: ${modelArch.feature_extractor_output_dimension})\n`;
    md += `- **Actor Layers**: ${JSON.stringify(modelArch.actor_hidden_layers)}\n`;
    md += `- **Critic Layers**: ${JSON.stringify(modelArch.critic_hidden_layers)}\n`;
    md += `- **Shared Layers**: ${JSON.stringify(modelArch.shared_layers)}\n`;
    md += `- **Activation Function**: ${modelArch.activation_function}\n`;
    md += `- **Action Space**: type = ${modelArch.action_space_type} (${modelArch.action_count} actions)\n`;
    md += `- **Ordered Action Mapping**: ${JSON.stringify(modelArch.ordered_action_mapping)}\n`;
    md += `- **Distribution Type**: ${modelArch.distribution_type}\n`;
    md += `- **Orthogonal Initialization Setting**: ${modelArch.orthogonal_initialization_setting ? "Enabled (True)" : "Disabled (False)"}\n`;
    md += `- **Normalization Settings**: ${modelArch.normalization_settings}\n`;
    md += `- **Total Trainable Parameter Count**: ${modelArch.total_trainable_parameter_count}\n`;
    md += `- **Computation Device**: ${modelArch.device}\n`;
    md += `- **Configured net_arch (Checkpoint)**: ${modelArch.configured_architecture}\n`;
    md += `- **Loaded net_arch (PyTorch)**: ${modelArch.loaded_architecture}\n`;
    md += `- **Compatibility Status**: **${modelArch.compatibility_status}**\n`;
  } else {
    md += `Neural network architecture details are **${fVal(modelArch.status)}**.\n`;
    if (modelArch.message) {
      md += `  - *Detail*: ${modelArch.message}\n`;
    }
  }

  // 6. PPO Training Progress and Optimizer Stats
  md += `\n## 5. PPO Training Progress and Optimizer Stats\n`;

  if (status && (status.approx_kl !== undefined && status.approx_kl !== null || status.clip_fraction !== undefined && status.clip_fraction !== null || status.explained_variance !== undefined && status.explained_variance !== null)) {
    md += `### Live Optimizer Metrics (Active Policy Version: v${status.policy_version ?? 0})\n`;
    md += `- **Approximate KL**: ${fNum(status.approx_kl, 6)}\n`;
    md += `- **Clip Fraction**: ${fPct(status.clip_fraction)}\n`;
    md += `- **Explained Variance**: ${fNum(status.explained_variance)}\n`;
    md += `- **Policy Loss**: Live training (unrecorded)\n`;
    md += `- **Value Loss**: Live training (unrecorded)\n\n`;
  }

  const ppoMetrics = snapshotData?.ppo_metrics || [];
  if (ppoMetrics && ppoMetrics.length > 0) {
    md += `### Historical Checkpoint Metrics\n`;
    md += `| Checkpoint | Policy Grad Loss | Value Loss | Entropy Loss | Total Loss | KL Divergence | Clip Frac | Expl Var |\n`;
    md += `|---|---|---|---|---|---|---|---|\n`;
    const recentMetrics = [...ppoMetrics].slice(-10).reverse();
    for (const cp of recentMetrics) {
      md += `| ${cp.checkpoint_episode} | ${fNum(cp.policy_gradient_loss)} | ${fNum(cp.value_loss)} | ${fNum(cp.entropy_loss)} | ${fNum(cp.loss)} | ${fNum(cp.approx_kl, 6)} | ${fPct(cp.clip_fraction)} | ${fNum(cp.explained_variance)} |\n`;
    }
  } else {
    md += `No historical training progress optimizer stats registered.\n`;
  }

  if (snapshotData?.version_history) {
    md += `\n### Checkpoint Progress by Policy Version\n`;
    md += `| Policy Version | Checkpoint Episode | Success Rate | Avg Reward | Avg Steps | Failed Seeds |\n`;
    md += `|---|---|---|---|---|---|\n`;
    for (const [ver, info] of Object.entries(snapshotData.version_history)) {
      const inf = info as any;
      md += `| ${ver} | Episode ${inf.checkpoint_episode} | ${Math.round(inf.success_rate * 100)}% | ${fNum(inf.average_reward)} | ${fNum(inf.average_completion_steps, 1)} | ${inf.failures.join(", ") || "None"} |\n`;
    }
  }
  
  if (snapshotData?.failed_seed_trends) {
    md += `\n### Historical Trends for Failed Seeds\n`;
    md += `| Seed | Currently Failing | Distance Delta | Reward Delta | Status | Trend Classification |\n`;
    md += `|---|---|---|---|---|---|\n`;
    for (const [seed, trend] of Object.entries(snapshotData.failed_seed_trends)) {
      const tr = trend as any;
      const statusStr = tr.is_plateau ? "⚠️ PLATEAU DETECTED" : "Improving";
      md += `| Seed ${seed} | ${tr.currently_failing ? "YES" : "No"} | ${tr.delta_distance.toFixed(2)} | ${tr.delta_reward.toFixed(2)} | ${statusStr} | ${tr.classification || "Unknown"} |\n`;
    }
  }

  // 7. Latest Per-Seed Evaluation Diagnostics
  md += `\n## 6. Latest Per-Seed Evaluation Diagnostics\n`;
  if (evalRecords && evalRecords.length > 0) {
    md += `| Seed | Success | Steps | Stop Reason | Initial Dist | Min Dist | Final Dog-Sheep Dist | Waits / Sprints / Invalid | Top Action | Oscillation? |\n`;
    md += `|---|---|---|---|---|---|---|---|---|---|\n`;
    for (const r of evalRecords) {
      const waitSprintMask = `${fVal(r.num_waits)} / ${fVal(r.num_sprints)} / ${fVal(r.num_invalid_actions)}`;
      md += `| ${r.seed} | ${r.success ? "SUCCESS" : "FAILED"} | ${r.steps} | ${r.stop_reason || "-"} | ${fNum(r.initial_sheep_distance_to_pen)} | ${fNum(r.min_sheep_distance_to_pen)} | ${fNum(r.final_dog_to_sheep_distance)} | ${waitSprintMask} | ${fVal(r.most_frequent_action)} | ${r.oscillation_detected ? "YES" : "No"} |\n`;
    }

    if (reconciliations.length > 0) {
      md += `\n### Reward Component Reconciliation\n`;
      md += `| Seed | Reported Reward | Sum of Components | Reconciliation Difference | Status |\n`;
      md += `|---|---|---|---|---|\n`;
      for (const rec of reconciliations) {
        md += `| Seed ${rec.seed} | ${fNum(rec.reported_reward)} | ${fNum(rec.summed_components)} | ${fNum(rec.difference, 4)} | ${rec.status === "RECONCILED" ? "✅ Reconciled" : rec.status === "PARTIAL LEGACY DATA" ? "⚠️ Partial Legacy Data" : "⚠️ MISMATCH"} |\n`;
      }
    }
    
    if (Object.keys(geometryValidations).length > 0) {
      md += `\n### Heuristic Geometry Validation (Per Evaluation Seed)\n`;
      md += `| Seed | Dog Start | Sheep Start | Pen Position / Dimensions | Grid Size | Overlaps? | Boundary Violation? | Spacing Violation? | Can Enter Pen? (Heuristic) | Space Behind? (Heuristic) |\n`;
      md += `|---|---|---|---|---|---|---|---|---|---|\n`;
      for (const [seed, val] of Object.entries(geometryValidations)) {
        const v = val as any;
        if (v.error) {
          md += `| Seed ${seed} | ERROR: ${v.error} | | | | | | | | |\n`;
        } else {
          const dogsStr = v.dog_start_positions.map((p: any) => `(${p[0].toFixed(1)},${p[1].toFixed(1)})`).join(" ");
          const sheepStr = v.sheep_start_positions.map((p: any) => `(${p[0].toFixed(1)},${p[1].toFixed(1)})`).join(" ");
          const penStr = `(${v.pen_position[0].toFixed(1)},${v.pen_position[1].toFixed(1)}) / ${v.pen_dimensions[0]}x${v.pen_dimensions[1]}`;
          md += `| Seed ${seed} | ${dogsStr} | ${sheepStr} | ${penStr} | ${v.grid_dimensions[0]}x${v.grid_dimensions[1]} | ${v.overlap_detected ? "YES ⚠️" : "No"} | ${v.boundary_violation ? "YES ⚠️" : "No"} | ${v.spacing_violation ? "YES ⚠️" : "No"} | ${v.can_enter_pen_heuristic ? "Yes" : "No"} | ${v.dog_has_space_behind_heuristic ? "Yes" : "No"} |\n`;
        }
      }
    }
  } else {
    md += `No evaluation records registered for the active checkpoint.\n`;
  }

  // 8. Failed-Seed Trajectory Summary
  md += `\n## 7. Failed-Seed Trajectory Summary\n`;
  if (!evalRecords || evalRecords.length === 0) {
    md += `Seed records unavailable. Detailed evaluation seed history cannot be retrieved.\n`;
  } else {
    const failedRecords = evalRecords.filter((r: any) => !r.success);
    if (failedRecords.length > 0) {
      for (const r of failedRecords) {
        md += `### Failed Seed ${r.seed} Details\n`;
        md += `- **Stop Reason**: ${r.stop_reason || "Timeout"}\n`;
        md += `- **Oscillation**: ${r.oscillation_detected ? "Detected (Dog repeated a 2-position loop)" : "None"}\n`;
        
        const traj = failedTrajectories[r.seed] || [];
        const stoppedIdx = traj.findIndex((s: any) => s.no_progress_counter > 0);
        md += `- **First step progress stopped**: ${stoppedIdx !== -1 ? `Step ${traj[stoppedIdx].step} (No progress counter = ${traj[stoppedIdx].no_progress_counter})` : "N/A (Progress continued until end)"}\n`;
        
        const lastActions = traj.slice(-20).map((s: any) => s.selected_actions);
        if (lastActions.length > 0) {
          md += `- **Last 20 Actions Taken**: ${JSON.stringify(lastActions)}\n`;
        }
        
        if (traj.length > 0) {
          md += `\n#### Trajectory step sample:\n`;
          md += `| Step | Event | Dog Positions | Sheep Positions | Sheep-Pen Dist | Dog-Sheep Dist | Selected Actions | Step Reward | No-Progress | Explanation |\n`;
          md += `|---|---|---|---|---|---|---|---|---|---|\n`;
          for (const s of traj) {
            const dogPos = s.dog_positions.map((p: any) => `(${p[0].toFixed(1)},${p[1].toFixed(1)})`).join(" ");
            const sheepPos = s.sheep_positions.map((p: any) => `(${p[0].toFixed(1)},${p[1].toFixed(1)})`).join(" ");
            md += `| ${s.step} | **${s.event}** | ${dogPos} | ${sheepPos} | ${s.sheep_distance_to_pen.toFixed(2)} | ${s.dog_to_sheep_distance.toFixed(2)} | ${s.selected_actions.join(" ")} | ${s.reward.toFixed(3)} | ${s.no_progress_counter} | ${s.no_progress_explanation || "-"} |\n`;
          }
        }
        md += `\n---\n`;
      }
    } else {
      const hasEval = status?.latest_success_rate !== null && status?.latest_success_rate !== undefined;
      const hasFailures = hasEval && status!.latest_success_rate! < 1.0;
      if (hasFailures) {
        md += `Failures exist, but no failed seed details were captured in the evaluation records.\n`;
      } else {
        md += `All recorded seeds succeeded! No failed seed trajectory diagnostics needed.\n`;
      }
    }
  }

  // 9. Observation Normalization and Limits
  md += `\n## 8. Observation Diagnostics\n`;
  const firstObsDiag = snapshotData?.observation_diagnostics;
  if (firstObsDiag?.feature_names) {
    md += `Summary of features normalization and limits:\n`;
    md += `| Feature Name | Min | Max | Mean | Std Dev | Status |\n`;
    md += `|---|---|---|---|---|---|\n`;
    for (let i = 0; i < firstObsDiag.feature_names.length; i++) {
      const name = firstObsDiag.feature_names[i];
      const min_v = firstObsDiag.min_values[i];
      const max_v = firstObsDiag.max_values[i];
      const mean_v = firstObsDiag.mean_values[i];
      const std_v = firstObsDiag.std_values[i];
      
      let statusStr = "OK";
      if (firstObsDiag.nan_or_inf_features.includes(name)) statusStr = "⚠️ NaN/Inf";
      else if (firstObsDiag.constant_features.includes(name)) statusStr = "Constant (No variance)";
      else if (firstObsDiag.saturated_features.includes(name)) statusStr = "Saturated (Constantly maxed)";
      
      md += `| ${name} | ${min_v.toFixed(4)} | ${max_v.toFixed(4)} | ${mean_v.toFixed(4)} | ${std_v.toFixed(4)} | ${statusStr} |\n`;
    }
  } else {
    md += `No observation diagnostics captured for this checkpoint evaluation.\n`;
  }

  // 10. Action and Mask Diagnostics
  md += `\n## 9. Action and Mask Diagnostics\n`;
  if (hasValidActionMetrics && totalSteps > 0) {
    const invalidRate = totalInvalidActions / totalSteps;
    md += `- **Total Evaluation Steps**: ${totalSteps}\n`;
    md += `- **Waits Selected**: ${totalWaits} (${((totalWaits / totalSteps) * 100).toFixed(1)}% of actions)\n`;
    md += `- **Sprints Selected**: ${totalSprints} (${((totalSprints / totalSteps) * 100).toFixed(1)}% of actions)\n`;
    md += `- **Action Mask Violations (Invalid Actions)**: ${totalInvalidActions} (${(invalidRate * 100).toFixed(2)}% of choices)\n`;
    md += `- **Wait/Sprint Choice Ratio**: ${totalSprints > 0 ? (totalWaits / totalSprints).toFixed(2) : "N/A"}\n`;
  } else {
    md += `Action choice metrics are ${fVal(null, "No action counts recorded in active checkpoint")}.\n`;
  }

  // 11. Plateau and Derivative Analysis
  md += `\n## 10. Plateau and Derivative Analysis\n`;
  md += `- **Success Rate Derivatives (rate of change per episode)**:\n`;
  md += `  - Over last 3 checkpoints: \`${dSuccess3}\` success rate change/episode\n`;
  md += `  - Over last 5 checkpoints: \`${dSuccess5}\` success rate change/episode\n`;
  md += `  - Over last 10 checkpoints: \`${dSuccess10}\` success rate change/episode\n`;
  md += `- **Average Completion Steps Derivatives (rate of change per episode)**:\n`;
  md += `  - Over last 3 checkpoints: \`${dSteps3}\` steps change/episode\n`;
  md += `  - Over last 5 checkpoints: \`${dSteps5}\` steps change/episode\n`;
  md += `  - Over last 10 checkpoints: \`${dSteps10}\` steps change/episode\n`;

  // 12. Counter Reconciliation
  md += `\n## 11. Counter Reconciliation\n`;
  if (counterReconciliation.rows && counterReconciliation.rows.length > 0) {
    md += `| Counter Name | Value | Unit | Authoritative Source | Definition |\n`;
    md += `|---|---|---|---|---|\n`;
    for (const r of counterReconciliation.rows) {
      md += `| ${r.counter} | ${fVal(r.value)} | ${r.unit} | ${r.source} | ${r.definition} |\n`;
    }
    if (counterReconciliation.warnings && counterReconciliation.warnings.length > 0) {
      md += `\n**Counter Anomalies/Warnings:**\n`;
      for (const w of counterReconciliation.warnings) {
        md += `- ⚠️ ${w}\n`;
      }
    }
  } else {
    md += `Counter reconciliation statistics are unavailable.\n`;
  }

  // 13. Machine-Readable Appendix (JSON)
  md += `\n## 12. Machine-Readable Appendix (JSON)\n`;
  md += `\`\`\`json\n`;
  
  // Construct non-null, structured machine-readable JSON object in the appendix under all execution outcomes
  const appendixObject = {
    schemaVersion: "3.0",
    snapshotIdentity: {
      activeRunId: snap?.active_run_id || "No active run",
      activeCheckpointId: snap?.active_checkpoint_id || "No active checkpoint",
      policyVersion: snap?.policy_version ?? null,
      ppoUpdateCount: snap?.ppo_update_count ?? 0
    },
    diagnosticCompleteness: {
      readiness: readiness,
      reasons: readinessReasons,
      table: completeness?.table || []
    },
    warnings: healthWarnings,
    diagnostics: diagnostics,
    apiErrorDetails: apiErrorDetails || null
  };

  md += JSON.stringify(appendixObject, null, 2);
  md += `\n\`\`\`\n`;

  return md;
}

function computeDerivative(checkpoints: CheckpointEntry[], window: number, field: "success_rate" | "average_completion_steps"): string {
  if (checkpoints.length <= window) return "Unavailable (Insufficient checkpoints)";
  const latest = checkpoints[checkpoints.length - 1];
  const prior = checkpoints[checkpoints.length - 1 - window];
  const yLatest = latest[field];
  const yPrior = prior[field];
  if (yLatest === undefined || yPrior === undefined || yLatest === null || yPrior === null) {
    return "Unavailable";
  }
  const dy = yLatest - yPrior;
  const dx = latest.checkpoint_episode - prior.checkpoint_episode;
  if (dx === 0) return "0.000000";
  return (dy / dx).toFixed(6);
}

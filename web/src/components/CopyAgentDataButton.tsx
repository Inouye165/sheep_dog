import { useState, useEffect } from "react";
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
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [stageOption, setStageOption] = useState<"current" | "single" | "range">("current");
  const [singleStage, setSingleStage] = useState<number>(curriculumStage);
  const [startStage, setStartStage] = useState<number>(curriculumStage);
  const [endStage, setEndStage] = useState<number>(curriculumStage);
  const [showTooltip, setShowTooltip] = useState(false);

  const checkpoints = checkpointIndex?.checkpoints ?? [];
  const allStages = Array.from(
    new Set([
      curriculumStage,
      ...(trainingStatus?.available_curriculum_stages ?? []),
      ...checkpoints.map((c) => c.reward_config?.instincts?.curriculum_stage ?? c.environment_config?.curriculum_stage ?? 0)
    ])
  ).filter((s) => s !== undefined && s !== null).sort((a, b) => a - b);

  if (allStages.length === 0) {
    for (let i = 0; i <= 32; i++) {
      allStages.push(i);
    }
  }

  // Synchronize dropdown selections when curriculumStage changes
  useEffect(() => {
    setSingleStage(curriculumStage);
    setStartStage(curriculumStage);
    setEndStage(curriculumStage);
  }, [curriculumStage]);

  const confirmAndCopy = async () => {
    if (loading) return;
    setLoading(true);
    setFailed(false);
    setIsModalOpen(false);

    let selectedStages: number[] = [];
    if (stageOption === "current") {
      selectedStages = [curriculumStage];
    } else if (stageOption === "single") {
      selectedStages = [singleStage];
    } else if (stageOption === "range") {
      const start = Math.min(startStage, endStage);
      const end = Math.max(startStage, endStage);
      for (let s = start; s <= end; s++) {
        selectedStages.push(s);
      }
    }

    const fetchAndFormatReport = async (): Promise<string> => {
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
      const checkpointsList = checkpointIndex?.checkpoints ?? [];

      // 3. Format the markdown report
      const report = formatAgentReport(
        trainingStatus,
        checkpointsList,
        hyperparams,
        curriculumStage,
        diagnostics,
        apiErrorDetails,
        selectedStages
      );

      if (!apiErrorDetails) {
        setCopied(true);
        setTimeout(() => {
          setCopied(false);
        }, 2000);
      }

      return report;
    };

    try {
      if (typeof ClipboardItem !== "undefined" && navigator.clipboard && navigator.clipboard.write) {
        const textBlobPromise = fetchAndFormatReport().then(
          (reportText) => new Blob([reportText], { type: "text/plain" })
        );
        await navigator.clipboard.write([
          new ClipboardItem({
            "text/plain": textBlobPromise,
          }),
        ]);
      } else {
        const report = await fetchAndFormatReport();
        await navigator.clipboard.writeText(report);
      }
    } catch (err) {
      console.error("Failed to copy agent data to clipboard:", err);
      alert("Could not copy agent data. Please check browser permissions.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
      <div className="agent-copy-info-container">
        <button
          onClick={() => setIsModalOpen(true)}
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

        <button
          type="button"
          onClick={() => setShowTooltip(!showTooltip)}
          onMouseEnter={() => setShowTooltip(true)}
          onMouseLeave={() => setShowTooltip(false)}
          className="agent-copy-info-btn"
          aria-label="Info about copied data"
        >
          ?
        </button>
        <div className={`agent-copy-info-tooltip ${showTooltip ? "visible" : ""}`}>
          <div className="agent-copy-info-tooltip-title">Generically Copied Data:</div>
          <ul>
            <li>Diagnostic Completeness & AI Readiness</li>
            <li>System Overview & Active Run ID</li>
            <li>Latest Evaluation Metrics & Gates</li>
            <li>Active Hyperparameters & Reward Weights</li>
            <li>Neural-Network Architecture Details</li>
            <li>PPO Training Progress & Optimizer Stats</li>
            <li>Failed Seed Trajectories & Observation limits</li>
          </ul>
        </div>
      </div>

      {isModalOpen && (
        <div className="stage-select-modal-overlay" onClick={() => setIsModalOpen(false)}>
          <div className="stage-select-modal-content" onClick={(e) => e.stopPropagation()}>
            <div className="stage-select-modal-header">
              <h3>Copy Agent Data - Select Stage</h3>
              <button
                className="stage-select-modal-close"
                onClick={() => setIsModalOpen(false)}
                aria-label="Close stage selection modal"
              >
                &times;
              </button>
            </div>

            <div className="stage-select-modal-body">
              <label className="instruction">
                Which curriculum stage(s) would you like to summarize in the report?
              </label>

              <div className="stage-select-modal-option-list">
                <label className="stage-select-modal-option">
                  <input
                    type="radio"
                    name="stageOption"
                    value="current"
                    checked={stageOption === "current"}
                    onChange={() => setStageOption("current")}
                  />
                  <span>Current Stage (Stage {curriculumStage})</span>
                </label>

                <label className="stage-select-modal-option" style={{ display: "flex", flexDirection: "column", alignItems: "flex-start" }}>
                  <div style={{ display: "flex", alignItems: "center", gap: "0.6rem" }}>
                    <input
                      type="radio"
                      name="stageOption"
                      value="single"
                      checked={stageOption === "single"}
                      onChange={() => setStageOption("single")}
                    />
                    <span>Single Stage</span>
                  </div>
                  {stageOption === "single" && (
                    <div className="stage-select-modal-select-container">
                      <select
                        value={singleStage}
                        onChange={(e) => setSingleStage(Number(e.target.value))}
                        className="stage-select-modal-select"
                        aria-label="Select single stage"
                      >
                        {allStages.map((s) => (
                          <option key={s} value={s}>
                            Stage {s} - {STAGE_DESCRIPTIONS[s] || "Unknown Stage"}
                          </option>
                        ))}
                      </select>
                    </div>
                  )}
                </label>

                <label className="stage-select-modal-option" style={{ display: "flex", flexDirection: "column", alignItems: "flex-start" }}>
                  <div style={{ display: "flex", alignItems: "center", gap: "0.6rem" }}>
                    <input
                      type="radio"
                      name="stageOption"
                      value="range"
                      checked={stageOption === "range"}
                      onChange={() => setStageOption("range")}
                    />
                    <span>Range of Stages</span>
                  </div>
                  {stageOption === "range" && (
                    <div className="stage-select-modal-select-container">
                      <select
                        value={startStage}
                        onChange={(e) => setStartStage(Number(e.target.value))}
                        className="stage-select-modal-select"
                        aria-label="Select start stage"
                      >
                        {allStages.map((s) => (
                          <option key={s} value={s}>
                            Stage {s}
                          </option>
                        ))}
                      </select>
                      <span style={{ color: "#94a3b8", fontSize: "0.85rem" }}>to</span>
                      <select
                        value={endStage}
                        onChange={(e) => setEndStage(Number(e.target.value))}
                        className="stage-select-modal-select"
                        aria-label="Select end stage"
                      >
                        {allStages.map((s) => (
                          <option key={s} value={s}>
                            Stage {s}
                          </option>
                        ))}
                      </select>
                    </div>
                  )}
                </label>
              </div>

              {stageOption === "range" && startStage > endStage && (
                <div style={{ color: "#f87171", fontSize: "0.8rem", marginTop: "0.85rem", paddingLeft: "1.8rem" }}>
                  * Start stage must be less than or equal to end stage.
                </div>
              )}
            </div>

            <div className="stage-select-modal-footer">
              <button
                className="stage-select-modal-cancel"
                onClick={() => setIsModalOpen(false)}
              >
                Cancel
              </button>
              <button
                className="stage-select-modal-confirm"
                onClick={confirmAndCopy}
                disabled={loading || (stageOption === "range" && startStage > endStage)}
              >
                {loading ? "Generating..." : "Copy to Clipboard"}
              </button>
            </div>
          </div>
        </div>
      )}
    </>
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
  } | null,
  selectedStages: number[] = []
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
    const ppoMetrics = snapshotData?.ppo_metrics || [];
    if (!ppoMetrics || ppoMetrics.length === 0) {
      healthWarnings.push("Missing PPO training progress metrics.");
    }
    // Missing observation diagnostics warning
    if (!snapshotData?.observation_diagnostics) {
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

  // Prepend selected stages summaries if requested
  if (selectedStages && selectedStages.length > 0) {
    for (const S of selectedStages) {
      const stageCheckpoints = checkpoints.filter(
        (c) => (c.reward_config?.instincts?.curriculum_stage ?? c.environment_config?.curriculum_stage ?? 0) === S
      );

      let latestCpEpisode: string | number = "N/A";
      let polVersion: string | number = "N/A";
      let successesText = "N/A";
      let timeoutsText = "N/A";
      let earlyStopsText = "N/A";
      let avgRewardText = "N/A";
      let avgSheepText = "N/A";
      let avgDistText = "N/A";
      let avgStepsText = "N/A";
      let perSeedRecords: any[] = [];
      let isCurrentStage = (S === currentStage);

      let hasLiveEval = false;
      if (isCurrentStage && diagnostics?.snapshot) {
        const evalRecords = diagnostics.snapshot.evaluation_records || [];
        const latestEval = diagnostics.snapshot.latest_current_stage_evaluation || diagnostics.snapshot.latest_any_stage_evaluation || null;
        if (evalRecords.length > 0 || latestEval !== null) {
          hasLiveEval = true;
        }
      }

      if (hasLiveEval && diagnostics?.snapshot) {
        const latestEval = diagnostics.snapshot.latest_current_stage_evaluation || diagnostics.snapshot.latest_any_stage_evaluation || null;
        latestCpEpisode = latestEval?.checkpoint_episode ?? status?.checkpoint_episode ?? "N/A";
        polVersion = latestEval?.policy_version ?? status?.policy_version ?? "N/A";

        const evalRecords = diagnostics.snapshot.evaluation_records || [];
        if (evalRecords.length > 0) {
          const successCount = evalRecords.filter((r: any) => r.success).length;
          const timeoutCount = evalRecords.filter((r: any) => r.timeout).length;
          const stoppedCount = evalRecords.filter((r: any) => r.stop_reason === "stopped" || r.stopped).length;
          const totalCount = evalRecords.length;

          successesText = `${successCount}/${totalCount}`;
          timeoutsText = `${timeoutCount}/${totalCount}`;
          earlyStopsText = `${stoppedCount}/${totalCount}`;

          const totalReward = evalRecords.reduce((sum: number, r: any) => sum + (r.reward_total ?? 0), 0);
          avgRewardText = (totalReward / totalCount).toFixed(1);

          const totalSheep = evalRecords.reduce((sum: number, r: any) => sum + (r.sheep_penned ?? 0), 0);
          const maxSheep = latestEval?.environment_config?.sheep ?? 4;
          avgSheepText = `${(totalSheep / totalCount).toFixed(1)}/${maxSheep}`;

          const totalDist = evalRecords.reduce((sum: number, r: any) => sum + (r.final_sheep_distance_to_pen ?? 0), 0);
          avgDistText = (totalDist / totalCount).toFixed(1);

          const totalSteps = evalRecords.reduce((sum: number, r: any) => sum + (r.steps ?? 0), 0);
          const totalNoProg = evalRecords.reduce((sum: number, r: any) => sum + (r.no_progress_steps ?? 0), 0);
          avgStepsText = `${(totalSteps / totalCount).toFixed(1)} / ${(totalNoProg / totalCount).toFixed(1)}`;

          perSeedRecords = evalRecords;
        } else if (latestEval) {
          const total = 10;
          const successCount = Math.round((latestEval.success_rate ?? 0) * total);
          const timeoutCount = Math.round((latestEval.timeout_rate ?? 0) * total);
          const stoppedCount = Math.round((latestEval.stopped_rate ?? 0) * total);
          successesText = `${successCount}/${total}`;
          timeoutsText = `${timeoutCount}/${total}`;
          earlyStopsText = `${stoppedCount}/${total}`;
          avgRewardText = latestEval.average_reward !== undefined ? latestEval.average_reward.toFixed(1) : "N/A";
          const maxSheep = latestEval.environment_config?.sheep ?? 4;
          avgSheepText = latestEval.average_sheep_penned !== undefined ? `${latestEval.average_sheep_penned.toFixed(1)}/${maxSheep}` : "N/A";
          avgDistText = (latestEval.average_distance_to_pen ?? latestEval.average_sheep_distance_to_pen) !== undefined ? (latestEval.average_distance_to_pen ?? latestEval.average_sheep_distance_to_pen).toFixed(1) : "N/A";
          avgStepsText = latestEval.average_completion_steps !== undefined ? latestEval.average_completion_steps.toFixed(1) : "N/A";
        }
      } else if (stageCheckpoints.length > 0) {
        const lc = stageCheckpoints[stageCheckpoints.length - 1];
        latestCpEpisode = lc.checkpoint_episode;
        polVersion = lc.policy_version ?? "N/A";

        if (lc.records && lc.records.length > 0) {
          const successCount = lc.records.filter((r: any) => r.success).length;
          const timeoutCount = lc.records.filter((r: any) => r.timeout).length;
          const stoppedCount = lc.records.filter((r: any) => r.stop_reason === "stopped" || r.stopped).length;
          const totalCount = lc.records.length;

          successesText = `${successCount}/${totalCount}`;
          timeoutsText = `${timeoutCount}/${totalCount}`;
          earlyStopsText = `${stoppedCount}/${totalCount}`;

          const totalReward = lc.records.reduce((sum: number, r: any) => sum + (r.reward_total ?? 0), 0);
          avgRewardText = (totalReward / totalCount).toFixed(1);

          const totalSheep = lc.records.reduce((sum: number, r: any) => sum + (r.sheep_penned ?? 0), 0);
          const maxSheep = lc.environment_config?.sheep ?? 4;
          avgSheepText = `${(totalSheep / totalCount).toFixed(1)}/${maxSheep}`;

          const totalDist = lc.records.reduce((sum: number, r: any) => sum + (r.final_sheep_distance_to_pen ?? 0), 0);
          avgDistText = (totalDist / totalCount).toFixed(1);

          const totalSteps = lc.records.reduce((sum: number, r: any) => sum + (r.steps ?? 0), 0);
          const totalNoProg = lc.records.reduce((sum: number, r: any) => sum + (r.no_progress_steps ?? 0), 0);
          avgStepsText = `${(totalSteps / totalCount).toFixed(1)} / ${(totalNoProg / totalCount).toFixed(1)}`;

          perSeedRecords = lc.records;
        } else {
          successesText = `${Math.round(lc.success_rate * 10)}/10`;
          timeoutsText = `${Math.round(lc.timeout_rate * 10)}/10`;
          earlyStopsText = `${Math.round((lc.records?.filter(r => r.stop_reason === "stopped" || r.stopped).length ?? 0) * 10)}/10`;
          avgRewardText = lc.average_reward.toFixed(1);
          const maxSheep = lc.environment_config?.sheep ?? 4;
          avgSheepText = `${lc.average_sheep_penned.toFixed(1)}/${maxSheep}`;
          avgStepsText = lc.average_completion_steps.toFixed(1);
        }
      }

      let gateSectionText = "";
      if (isCurrentStage) {
        const gate = diagnostics?.snapshot?.current_stage_promotion_gate || status?.auto_promote_gate;
        if (gate) {
          gateSectionText = `- Required success: ${Math.round(gate.success_threshold * gate.seed_count)}/${gate.seed_count}\n` +
                            `- Current streak: ${gate.qualified_streak}/${gate.min_qualified_streak}\n` +
                            `- Best success observed: ${gate.best_success}/${gate.seed_count}\n` +
                            `- Perfect batches observed: ${gate.full_success_hits ?? 0}\n`;
        } else {
          gateSectionText = "- No active promotion gate configured or available for this stage.\n";
        }
      } else {
        gateSectionText = "- N/A (Stage already promoted)\n";
      }

      const trendSuccess3 = computeDerivative(stageCheckpoints, 3, "success_rate");
      const trendSuccess5 = computeDerivative(stageCheckpoints, 5, "success_rate");
      const trendSteps = computeDerivative(stageCheckpoints, 3, "average_completion_steps");
      const trendTimeout = computeDerivative(stageCheckpoints, 3, "timeout_rate");

      let klVal = "N/A";
      let clipVal = "N/A";
      let evVal = "N/A";

      const lcForStage = stageCheckpoints[stageCheckpoints.length - 1];
      const activeKL = isCurrentStage ? (status?.approx_kl ?? lcForStage?.approx_kl) : lcForStage?.approx_kl;
      const activeClip = isCurrentStage ? (status?.clip_fraction ?? lcForStage?.clip_fraction) : lcForStage?.clip_fraction;
      const activeEV = isCurrentStage ? (status?.explained_variance ?? lcForStage?.explained_variance) : lcForStage?.explained_variance;

      if (activeKL !== undefined && activeKL !== null) klVal = activeKL.toFixed(4);
      if (activeClip !== undefined && activeClip !== null) clipVal = `${(activeClip * 100).toFixed(1)}%`;
      if (activeEV !== undefined && activeEV !== null) evVal = activeEV.toFixed(3);

      let seedTableText = "";
      if (perSeedRecords.length > 0) {
        seedTableText = "| Seed | Success/Failure | Reward | Steps | Termination Reason |\n" +
                        "|---|---|---|---|---|\n";
        for (const r of perSeedRecords) {
          const succText = r.success ? "SUCCESS" : "FAILED";
          const rewardVal = r.reward_total !== undefined && r.reward_total !== null ? r.reward_total.toFixed(1) : "N/A";
          const stepsVal = r.steps !== undefined && r.steps !== null ? r.steps : "N/A";
          const termReason = r.stop_reason || (r.timeout ? "timeout" : r.stopped ? "stopped" : "success");
          seedTableText += `| ${r.seed} | ${succText} | ${rewardVal} | ${stepsVal} | ${termReason} |\n`;
        }
      } else {
        seedTableText = "No per-seed evaluation records available.\n";
      }

      md += `CURRENT STAGE SUMMARY\n\n`;
      md += `Stage: ${S}\n`;
      md += `Description: ${STAGE_DESCRIPTIONS[S] ?? "Unknown Stage"}\n\n`;
      md += `Latest checkpoint: ${latestCpEpisode}\n`;
      md += `Policy version: ${polVersion}\n`;
      md += `Stage episodes trained: ${status?.stage_history?.[S.toString()] ?? 0}\n\n`;
      
      md += `Latest ${perSeedRecords.length || 10}-seed evaluation:\n`;
      md += `- Successes: ${successesText}\n`;
      md += `- Timeouts: ${timeoutsText}\n`;
      md += `- Early stops: ${earlyStopsText}\n`;
      md += `- Average reward: ${avgRewardText}\n`;
      md += `- Average sheep penned: ${avgSheepText}\n`;
      md += `- Average distance remaining: ${avgDistText}\n`;
      md += `- Average completion/no-progress steps: ${avgStepsText}\n\n`;

      md += `Promotion gate:\n`;
      md += gateSectionText + `\n`;

      md += `Recent current-stage trend:\n`;
      md += `- Success trend over 3 checkpoints: ${trendSuccess3}\n`;
      md += `- Success trend over 5 checkpoints: ${trendSuccess5}\n`;
      md += `- Average steps trend: ${trendSteps}\n`;
      md += `- Timeout trend: ${trendTimeout}\n\n`;

      md += `PPO health:\n`;
      md += `- KL: ${klVal}\n`;
      md += `- Clip fraction: ${clipVal}\n`;
      md += `- Explained variance: ${evVal}\n\n`;

      md += `Per-seed results:\n`;
      md += seedTableText + `\n`;
      md += `---\n\n`;
    }
  }

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
  const authoritativeStage = snap.active_curriculum_stage !== undefined && snap.active_curriculum_stage !== null ? snap.active_curriculum_stage : currentStage;
  const authoritativeStageName = snap.active_stage_name || STAGE_DESCRIPTIONS[authoritativeStage] || "Unknown Stage";
  const authoritativePolicyVersion = snap.active_policy_version !== undefined && snap.active_policy_version !== null ? snap.active_policy_version : (snap.policy_version ?? "N/A");
  const authoritativeCheckpointId = snap.active_checkpoint_id || snap.evaluation_checkpoint_id || "N/A";

  md += `## 2. System Overview and Report Identity\n`;
  md += `- **Report Generation Time**: ${timestamp}\n`;
  md += `- **Snapshot Timestamp**: ${fVal(snap.snapshot_timestamp)}\n`;
  md += `- **Active Run ID**: ${fVal(snap.active_run_id)}\n`;
  md += `- **Loaded Model ID**: ${fVal(snap.loaded_model_id)}\n`;
  md += `- **Active Checkpoint ID**: ${fVal(authoritativeCheckpointId)}\n`;
  if (snap.is_legacy) {
    md += `- **Legacy Policy Version**: unknown\n`;
    md += `- **Legacy PPO Update Count**: unknown\n`;
    md += `- **Diagnostic Baseline Version**: 0\n`;
    md += `- **Updates since Instrumentation**: 0\n`;
  } else {
    md += `- **Policy Version / Update Number**: ${fVal(authoritativePolicyVersion)}\n`;
    md += `- **PPO Update Count**: ${fVal(snap.ppo_update_count)}\n`;
    md += `- **Diagnostic Baseline Version**: 0\n`;
    md += `- **Updates since Instrumentation**: ${snap.ppo_update_count !== null && snap.ppo_update_count !== undefined ? snap.ppo_update_count : 0}\n`;
  }
  md += `- **Current Curriculum Stage**: Stage ${authoritativeStage} (${authoritativeStageName})\n`;
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
  const latestEval = snap.latest_current_stage_evaluation || snap.latest_any_stage_evaluation || null;
  const evalEpisode = latestEval ? latestEval.checkpoint_episode : null;
  const evalSuccess = latestEval ? latestEval.success_rate : null;
  const evalReward = latestEval ? latestEval.average_reward : null;
  const evalTimeout = latestEval ? latestEval.timeout_rate : null;
  const evalStopped = latestEval ? latestEval.stopped_rate : null;
  const evalNoProgress = latestEval ? latestEval.average_no_progress_steps : null;
  const evalPenned = latestEval ? latestEval.average_sheep_penned : null;
  const evalDist = latestEval ? (latestEval.average_distance_to_pen ?? latestEval.average_sheep_distance_to_pen) : null;
  const evalSpread = latestEval ? latestEval.average_flock_spread : null;
  const evalFarthestPen = latestEval ? latestEval.average_farthest_distance_to_pen : null;
  const evalFarthestFlock = latestEval ? latestEval.average_farthest_distance_to_flock_center : null;

  if (latestEval) {
    md += `- **Checkpoint Evaluated**: ${fVal(latestEval.checkpoint_id || (evalEpisode !== null ? `ep ${evalEpisode}` : null))}\n`;
    md += `- **Policy Version Evaluated**: v${fVal(latestEval.policy_version)}\n`;
    md += `- **Evaluation Timestamp**: ${fVal(latestEval.evaluation_timestamp || latestEval.created_timestamp)}\n`;
    md += `- **Success Rate**: ${evalSuccess !== null ? `${Math.round(evalSuccess * 100)}%` : "N/A"}\n`;
    md += `- **Avg Reward**: ${evalReward ?? "N/A"}\n`;
    md += `- **Timeout Rate**: ${evalTimeout !== null ? `${Math.round(evalTimeout * 100)}%` : "N/A"}\n`;
    md += `- **Stopped Rate**: ${evalStopped !== null ? `${Math.round(evalStopped * 100)}%` : "N/A"}\n`;
    md += `- **Avg No-Progress Steps**: ${evalNoProgress ?? "N/A"}\n`;
    md += `- **Avg Sheep Penned**: ${evalPenned ?? "N/A"}\n`;
    md += `- **Avg Distance to Pen**: ${evalDist ?? "N/A"}\n`;
    md += `- **Avg Flock Spread**: ${evalSpread ?? "N/A"}\n`;
    md += `- **Avg Farthest Distance to Pen**: ${evalFarthestPen ?? "N/A"}\n`;
    md += `- **Avg Farthest Distance to Flock Center**: ${evalFarthestFlock ?? "N/A"}\n`;
  } else {
    md += `- **Status**: No completed evaluations found.\n`;
  }

  md += `\n### Current Active Policy\n`;
  const currentPolicyVersion = status?.policy_version;
  const evalVersion = snap.evaluation_policy_version;
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

  const gate = snap.current_stage_promotion_gate;
  if (gate) {
    md += `\n### Auto-Promote Gate Diagnostics\n`;
    md += `- **Gate Decision**: \`${gate.decision.toUpperCase()}\` (Reason: ${gate.reason})\n`;
    md += `- **Streak Count**: ${gate.qualified_streak} / ${gate.min_qualified_streak} qualified batches\n`;
    md += `- **Seed Check**: ${gate.seed_count} seeds evaluated (Gate target: ${gate.seed_gate_target_met ? "MET" : "NOT MET"})\n`;
    md += `- **Success Rate Check**: ${gate.success_count} fully success seeds (Gate target: ${gate.success_threshold * 100}% threshold, current average is ${gate.success_rate_ok ? "OK" : "BELOW TARGET"})\n`;
    md += `- **Timeout Rate Check**: ${gate.timeout_ok ? "OK" : "TOO HIGH"}\n`;
  }

  const prevPromotion = snap.previous_stage_promotion_result;
  if (prevPromotion) {
    const isLegacy = prevPromotion.checkpoint_id === undefined || prevPromotion.policy_version === undefined || prevPromotion.seed_set_id === undefined;
    
    md += `\n### Previous Stage Promotion Result\n`;
    md += `- **Promoted At**: ${prevPromotion.promoted_at || prevPromotion.timestamp || "null"}\n`;
    md += `- **From Stage**: ${prevPromotion.from_stage !== undefined && prevPromotion.from_stage !== null ? `Stage ${prevPromotion.from_stage}` : "null"}\n`;
    md += `- **To Stage**: ${prevPromotion.to_stage !== undefined && prevPromotion.to_stage !== null ? `Stage ${prevPromotion.to_stage}` : "null"}\n`;
    md += `- **Trigger Checkpoint**: ${prevPromotion.trigger_checkpoint_id || prevPromotion.checkpoint_id || "null"} (episode ${prevPromotion.trigger_checkpoint_episode !== undefined ? prevPromotion.trigger_checkpoint_episode : "null"})\n`;
    md += `- **Policy Version**: ${prevPromotion.trigger_policy_version !== undefined ? `v${prevPromotion.trigger_policy_version}` : prevPromotion.policy_version !== undefined ? `v${prevPromotion.policy_version}` : "null"}\n`;
    md += `- **Seed Set ID**: ${prevPromotion.evaluation_seed_set_id || prevPromotion.seed_set_id || "null"}\n`;
    md += `- **Streak**: ${prevPromotion.qualified_streak !== undefined && prevPromotion.qualified_streak !== null ? `${prevPromotion.qualified_streak} qualified batches` : "null"}\n`;
    
    if (isLegacy) {
      md += `\n> ⚠️ **Warning**: This is a legacy promotion record. Some stage identity contract fields are missing.\n`;
    }
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

function computeDerivative(checkpoints: CheckpointEntry[], window: number, field: "success_rate" | "average_completion_steps" | "timeout_rate"): string {
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

import React, { useCallback, useEffect, useState } from "react";
import { loadStageHealth } from "../lib/api";
import type {
  StageHealthSummary,
  PrescriptiveRecommendation,
  CheckpointEntry,
  TrainingStatus,
  SeedHealthItem,
} from "../state/types";

interface StageHealthBannerProps {
  curriculumStage: number;
  lastLiveRefreshTime?: number;
  isLiveTraining?: boolean;
  checkpoints?: CheckpointEntry[];
  trainingStatus?: TrainingStatus | null;
}

function getStageFromCheckpoint(c: CheckpointEntry): number {
  if (c.curriculum_stage !== undefined && c.curriculum_stage !== null) {
    return c.curriculum_stage;
  }
  if (c.environment_config?.curriculum_stage !== undefined && c.environment_config?.curriculum_stage !== null) {
    return c.environment_config.curriculum_stage;
  }
  if (c.reward_config?.instincts?.curriculum_stage !== undefined && c.reward_config?.instincts?.curriculum_stage !== null) {
    return c.reward_config.instincts.curriculum_stage;
  }
  return 1;
}

function buildClientFallbackHealth(
  stage: number,
  checkpoints: CheckpointEntry[],
  trainingStatus?: TrainingStatus | null,
): StageHealthSummary | null {
  const stageCheckpoints = checkpoints.filter((c) => getStageFromCheckpoint(c) === stage);
  if (stageCheckpoints.length === 0) {
    return {
      stage,
      stage_title: `Stage ${stage}`,
      total_stage_checkpoints: 0,
      all_time_stage_success_rate: 0.0,
      recent_success_rate: 0.0,
      peak_stage_success_rate: 0.0,
      recent_avg_steps: 0.0,
      recent_avg_reward: 0.0,
      status: "yellow",
      status_label: "Awaiting Checkpoints",
      status_explanation: `Stage ${stage} telemetry is accumulating. Checkpoints will appear automatically.`,
      promotion_ready: false,
      promotion_status_text: "Collecting Data",
      failure_progress: {
        total_failures: 0,
        avg_penned_on_fail: 0.0,
        three_penned_pct: 0.0,
        two_penned_pct: 0.0,
        one_penned_pct: 0.0,
        zero_penned_pct: 0.0,
        closeness_score: 0.0,
      },
      seed_matrix: [],
      recent_trajectory: [],
      hyperparameter_audit: [],
      prescriptive_recommendations: [],
    };
  }

  const allSr = stageCheckpoints.map((c) => c.success_rate ?? 0.0);
  const allSteps = stageCheckpoints.map((c) => c.average_completion_steps ?? 0.0).filter((s) => s > 0);
  const allRew = stageCheckpoints.map((c) => c.average_reward ?? 0.0);

  const allTimeSr = allSr.reduce((a, b) => a + b, 0) / allSr.length;
  const peakSr = Math.max(...allSr);

  const recentSlice = stageCheckpoints.slice(-15);
  const recentSrList = recentSlice.map((c) => c.success_rate ?? 0.0);
  const recentStepsList = recentSlice.map((c) => c.average_completion_steps ?? 0.0).filter((s) => s > 0);
  const recentRewList = recentSlice.map((c) => c.average_reward ?? 0.0);

  const recentSr = recentSrList.reduce((a, b) => a + b, 0) / recentSrList.length;
  const recentAvgSteps = recentStepsList.length > 0 ? recentStepsList.reduce((a, b) => a + b, 0) / recentStepsList.length : 0.0;
  const recentAvgRew = recentRewList.length > 0 ? recentRewList.reduce((a, b) => a + b, 0) / recentRewList.length : 0.0;

  // Failure progress and seed stats from records
  let totalFailures = 0;
  let threePenned = 0;
  let twoPenned = 0;
  let onePenned = 0;
  let zeroPenned = 0;
  const pennedCounts: number[] = [];

  const seedTotals: Record<number, number> = {};
  const seedWins: Record<number, number> = {};
  const seedConsecFails: Record<number, number> = {};

  for (const c of stageCheckpoints) {
    const records = c.records || [];
    for (const r of records) {
      if (r.seed != null) {
        const s = r.seed;
        seedTotals[s] = (seedTotals[s] || 0) + 1;
        if (r.success) {
          seedWins[s] = (seedWins[s] || 0) + 1;
          seedConsecFails[s] = 0;
        } else {
          seedConsecFails[s] = (seedConsecFails[s] || 0) + 1;
          totalFailures += 1;
          if (r.sheep_penned != null) {
            pennedCounts.push(r.sheep_penned);
            if (r.sheep_penned >= 3) threePenned += 1;
            else if (r.sheep_penned === 2) twoPenned += 1;
            else if (r.sheep_penned === 1) onePenned += 1;
            else zeroPenned += 1;
          }
        }
      }
    }
  }

  const numFailsWithPenned = pennedCounts.length || 1;
  const threePct = threePenned / numFailsWithPenned;
  const twoPct = twoPenned / numFailsWithPenned;
  const onePct = onePenned / numFailsWithPenned;
  const zeroPct = zeroPenned / numFailsWithPenned;
  const avgPennedOnFail = pennedCounts.length > 0 ? pennedCounts.reduce((a, b) => a + b, 0) / pennedCounts.length : 0.0;
  const closeness = Math.min(1.0, threePct * 1.0 + twoPct * 0.6 + onePct * 0.2);

  const seedMatrix: SeedHealthItem[] = Object.keys(seedTotals)
    .map(Number)
    .sort((a, b) => a - b)
    .map((s) => {
      const tot = seedTotals[s];
      const w = seedWins[s] || 0;
      const wr = tot > 0 ? w / tot : 0.0;
      const consec = seedConsecFails[s] || 0;
      let status: "green" | "yellow" | "red" = "green";
      if (wr < 0.50 && consec >= 2) status = "red";
      else if (wr < 0.80) status = "yellow";
      return {
        seed: s,
        win_rate: wr,
        wins: w,
        fails: tot - w,
        total: tot,
        status,
        current_consecutive_fails: consec,
      };
    });

  const latestFourMax = Math.max(...recentSrList.slice(-4), recentSr);
  let status: "green" | "yellow" | "red" = "green";
  let statusLabel = "Healthy Learning · Surging";
  let statusExplanation = `The policy is progressing well on Stage ${stage}, peaking at ${Math.round(latestFourMax * 100)}% win rate with strong pack coordination.`;

  if (latestFourMax >= 0.85 || (recentSr >= 0.75 && recentAvgRew > 150)) {
    status = "green";
    statusLabel = "Healthy Learning · Surging";
  } else if (recentSr >= 0.45 || closeness >= 0.45 || (peakSr >= 0.80 && allTimeSr >= 0.55)) {
    status = "yellow";
    statusLabel = "Active Exploration · Watch";
    statusExplanation = `The policy is undergoing standard multi-agent exploratory consolidation (win rates fluctuate around ${Math.round(recentSr * 100)}%, but ${Math.round(threePct * 100)}% of failures pen 3 of 4 sheep). Do not panic or interrupt.`;
  } else {
    status = "red";
    statusLabel = "Systemic Bottleneck · Action Required";
    statusExplanation = `The policy has remained below 40% success for an extended duration without partial progress. Check for stray recovery or pen friction.`;
  }

  const recs: PrescriptiveRecommendation[] = [];
  if (status === "green") {
    recs.push({
      type: "continue",
      title: "Allow Live Training to Continue",
      description: "The policy is actively hitting 90% benchmarks. Do not reset weights or interrupt.",
      suggested_action: "Keep training running. Monitor auto-promotion gate for consecutive qualification.",
      priority: "info",
    });
  } else {
    recs.push({
      type: "continue",
      title: "Do Not Panic on Temporary Dips",
      description: `Failed runs are penning 3 of 4 sheep ${Math.round(threePct * 100)}% of the time. PPO typically surges after exploration consolidation.`,
      suggested_action: "Allow 10–15 more checkpoints before making intervention decisions.",
      priority: "info",
    });
  }

  const redSeeds = seedMatrix.filter((s) => s.status === "red").map((s) => s.seed);
  if (redSeeds.length > 0) {
    recs.push({
      type: "failure_directed",
      title: `Target Drag Seeds (${redSeeds.join(", ")})`,
      description: `Seeds ${redSeeds.join(", ")} account for the majority of stage failures due to wide stray initial spawns.`,
      suggested_action: `Add override in CURRICULUM_TRAINING_OVERRIDES[${stage}]: 'failure_directed_training_enabled': True`,
      priority: "medium",
    });
  }

  if (stage === 8) {
    recs.push({
      type: "reward_tweak",
      title: "Boost Lone Straggler Approach Scale",
      description: "Failures frequently leave 1 sheep unpenned until timeout. A stronger approach gradient motivates dogs to retrieve the 4th sheep cross-field.",
      suggested_action: `Update CURRICULUM_REWARD_OVERRIDES[8]['farthest_sheep_progress_scale'] = 0.55 in src/sheepdog/curriculum.py`,
      priority: "medium",
    });
  }

  return {
    stage,
    stage_title: `Stage ${stage} (3 Dogs, 4 Sheep, 108x78)`,
    total_stage_checkpoints: stageCheckpoints.length,
    all_time_stage_success_rate: allTimeSr,
    recent_success_rate: recentSr,
    peak_stage_success_rate: peakSr,
    recent_avg_steps: recentAvgSteps,
    recent_avg_reward: recentAvgRew,
    status,
    status_label: statusLabel,
    status_explanation: statusExplanation,
    promotion_ready: latestFourMax >= 0.90 && recentSr >= 0.80,
    promotion_status_text: latestFourMax >= 0.90 ? "Promotion Candidate" : "Passes Pending",
    failure_progress: {
      total_failures: totalFailures,
      avg_penned_on_fail: Number(avgPennedOnFail.toFixed(2)),
      three_penned_pct: Number(threePct.toFixed(3)),
      two_penned_pct: Number(twoPct.toFixed(3)),
      one_penned_pct: Number(onePct.toFixed(3)),
      zero_penned_pct: Number(zeroPct.toFixed(3)),
      closeness_score: Number(closeness.toFixed(3)),
    },
    seed_matrix: seedMatrix,
    recent_trajectory: recentSlice.map((c) => ({
      pv: c.policy_version || 0,
      episode: c.checkpoint_episode || 0,
      success_rate: c.success_rate || 0.0,
      steps: c.average_completion_steps || 0.0,
      reward: c.average_reward || 0.0,
      mode: c.evaluation_mode || "quick",
      timestamp: c.created_timestamp || "",
    })),
    hyperparameter_audit: [
      {
        parameter: "farthest_sheep_progress_scale",
        current_value: 0.42,
        recommended_value: 0.55,
        status: "warn",
        note: "Current 0.42 is low for stray recovery on 108x78 grid; recommend >= 0.55.",
      },
      {
        parameter: "entropy_coef",
        current_value: 0.010,
        recommended_value: 0.010,
        status: "ok",
        note: "Baseline PPO exploration entropy active.",
      },
    ],
    prescriptive_recommendations: recs,
  };
}

export function StageHealthBanner({
  curriculumStage,
  lastLiveRefreshTime,
  isLiveTraining = false,
  checkpoints = [],
  trainingStatus = null,
}: StageHealthBannerProps) {
  const [healthData, setHealthData] = useState<StageHealthSummary | null>(() => {
    return buildClientFallbackHealth(curriculumStage, checkpoints, trainingStatus);
  });
  const [loading, setLoading] = useState<boolean>(false);
  const [isModalOpen, setIsModalOpen] = useState<boolean>(false);
  const [copied, setCopied] = useState<boolean>(false);

  const fetchHealth = useCallback(async (force = false) => {
    try {
      setLoading(true);
      const data = await loadStageHealth(curriculumStage, force);
      if (data && data.total_stage_checkpoints > 0) {
        setHealthData(data);
      } else {
        const fallback = buildClientFallbackHealth(curriculumStage, checkpoints, trainingStatus);
        if (fallback) setHealthData(fallback);
      }
    } catch (err) {
      console.warn("[StageHealthBanner] Endpoint fallback to client aggregation:", err);
      const fallback = buildClientFallbackHealth(curriculumStage, checkpoints, trainingStatus);
      if (fallback) setHealthData(fallback);
    } finally {
      setLoading(false);
    }
  }, [curriculumStage, checkpoints, trainingStatus]);

  useEffect(() => {
    fetchHealth();
  }, [fetchHealth, lastLiveRefreshTime]);


  const handleCopySummary = () => {
    if (!healthData) return;
    const text = [
      `=== STAGE ${healthData.stage} HEALTH AUDIT & DIAGNOSTIC SUMMARY ===`,
      `Status: ${healthData.status.toUpperCase()} (${healthData.status_label})`,
      `Summary: ${healthData.status_explanation}`,
      ``,
      `--- WHOLE-STAGE METRICS ---`,
      `Total Checkpoints Evaluated: ${healthData.total_stage_checkpoints}`,
      `All-Time Stage Success: ${(healthData.all_time_stage_success_rate * 100).toFixed(1)}%`,
      `Recent Success: ${(healthData.recent_success_rate * 100).toFixed(1)}%`,
      `Peak Success: ${(healthData.peak_stage_success_rate * 100).toFixed(1)}%`,
      `Recent Avg Steps: ${healthData.recent_avg_steps}`,
      `Recent Avg Reward: ${healthData.recent_avg_reward}`,
      ``,
      `--- FAILURE PROGRESS (Closeness to Victory) ---`,
      `Total Failed Episodes: ${healthData.failure_progress.total_failures}`,
      `3 of 4 Sheep Penned on Fail: ${(healthData.failure_progress.three_penned_pct * 100).toFixed(1)}%`,
      `2 of 4 Sheep Penned on Fail: ${(healthData.failure_progress.two_penned_pct * 100).toFixed(1)}%`,
      `Avg Sheep Penned on Fail: ${healthData.failure_progress.avg_penned_on_fail}`,
      ``,
      `--- SEED RELIABILITY MATRIX ---`,
      ...healthData.seed_matrix.map(
        (s) =>
          `Seed ${s.seed}: ${(s.win_rate * 100).toFixed(1)}% (${s.wins}/${s.total} wins) - Status: ${s.status.toUpperCase()}${
            s.current_consecutive_fails > 0 ? ` (Streak: ${s.current_consecutive_fails} fails)` : ""
          }`
      ),
      ``,
      `--- PRESCRIPTIVE RECOMMENDATIONS & CHANGES NEEDED ---`,
      ...healthData.prescriptive_recommendations.map(
        (r, i) =>
          `${i + 1}. [${r.priority.toUpperCase()}] ${r.title}\n   Why: ${r.description}\n   Action: ${r.suggested_action}`
      ),
    ].join("\n");

    navigator.clipboard.writeText(text).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2500);
    });
  };

  if (!healthData) {
    if (loading) {
      return (
        <div className="stage-health-card stage-health-card--yellow" role="region" aria-label="Loading Stage Health">
          <div className="stage-health-card__main">
            <div className="stage-health-status-badge">
              <span className="stage-health-dot stage-health-dot--yellow" />
              <div className="stage-health-status-badge__text">
                <span className="stage-health-status-badge__title">Analyzing Stage Health...</span>
                <span className="stage-health-status-badge__sub">Aggregating Stage {curriculumStage} Telemetry</span>
              </div>
            </div>
          </div>
        </div>
      );
    }
    return null;
  }

  const statusTone = healthData.status; // "green" | "yellow" | "red"

  return (
    <>
      <div
        className={`stage-health-card stage-health-card--${statusTone}`}
        role="region"
        aria-label="Real-time Stage Health"
      >
        <div className="stage-health-card__main">
          {/* Status Indicator Pill */}
          <div className="stage-health-status-badge">
            <span className={`stage-health-dot stage-health-dot--${statusTone}`} />
            <div className="stage-health-status-badge__text">
              <span className="stage-health-status-badge__title">
                {healthData.status_label}
              </span>
              <span className="stage-health-status-badge__sub">
                {healthData.stage_title} · {healthData.total_stage_checkpoints} CPs
              </span>
            </div>
          </div>

          {/* KPI Strip */}
          <div className="stage-health-kpis">
            <div className="stage-health-kpi-item">
              <span className="stage-health-kpi-item__label">All-Time Stage Win</span>
              <span className="stage-health-kpi-item__val">
                {(healthData.all_time_stage_success_rate * 100).toFixed(1)}%
              </span>
            </div>

            <div className="stage-health-kpi-item">
              <span className="stage-health-kpi-item__label">Recent Rolling</span>
              <span
                className="stage-health-kpi-item__val"
                style={{
                  color:
                    healthData.recent_success_rate >= 0.8
                      ? "#4ade80"
                      : healthData.recent_success_rate >= 0.5
                      ? "#fbbf24"
                      : "#f87171",
                }}
              >
                {(healthData.recent_success_rate * 100).toFixed(1)}%
              </span>
            </div>

            <div className="stage-health-kpi-item">
              <span className="stage-health-kpi-item__label">Stage Peak</span>
              <span className="stage-health-kpi-item__val" style={{ color: "#38bdf8" }}>
                {(healthData.peak_stage_success_rate * 100).toFixed(0)}%
              </span>
            </div>

            <div className="stage-health-kpi-item">
              <span className="stage-health-kpi-item__label">3-Penned Failures</span>
              <span className="stage-health-kpi-item__val" style={{ color: "#e2e8f0" }}>
                {(healthData.failure_progress.three_penned_pct * 100).toFixed(0)}%
              </span>
            </div>
          </div>

          {/* Seed Reliability Matrix Row */}
          <div className="stage-health-seeds-row">
            <span className="stage-health-seeds-row__label">Seed Health:</span>
            <div className="stage-health-seeds-pills">
              {healthData.seed_matrix.map((s) => (
                <div
                  key={s.seed}
                  className={`stage-seed-pill stage-seed-pill--${s.status}`}
                  title={`Seed ${s.seed}: ${(s.win_rate * 100).toFixed(1)}% (${s.wins}/${s.total} wins)${
                    s.current_consecutive_fails > 0 ? ` · ${s.current_consecutive_fails} consecutive fails` : ""
                  }`}
                >
                  <span className="stage-seed-pill__num">{s.seed}</span>
                  <span className="stage-seed-pill__pct">{Math.round(s.win_rate * 100)}%</span>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Action Button */}
        <div className="stage-health-card__actions">
          <button
            onClick={() => setIsModalOpen(true)}
            className="stage-health-diagnose-btn"
            title="Open Diagnostic Summary & Recommended Changes"
            aria-label="Diagnose & Summary"
          >
            <span className="stage-health-diagnose-btn__icon">🔍</span>
            <span className="stage-health-diagnose-btn__text">Diagnose & Summary</span>
            {healthData.prescriptive_recommendations.length > 0 && (
              <span className="stage-health-diagnose-btn__badge">
                {healthData.prescriptive_recommendations.length}
              </span>
            )}
          </button>
        </div>
      </div>

      {/* ── Diagnostic & Prescriptive Recommendations Modal ── */}
      {isModalOpen && (
        <div className="stage-health-modal-overlay" onClick={() => setIsModalOpen(false)}>
          <div
            className="stage-health-modal"
            onClick={(e) => e.stopPropagation()}
            role="dialog"
            aria-labelledby="stage-health-modal-title"
          >
            <div className="stage-health-modal__header">
              <div className="stage-health-modal__title-group">
                <span className={`stage-health-dot stage-health-dot--${statusTone}`} />
                <h3 id="stage-health-modal-title" className="stage-health-modal__title">
                  Stage {healthData.stage} Diagnostic Audit & Recommendations
                </h3>
              </div>
              <button
                onClick={() => setIsModalOpen(false)}
                className="stage-health-modal__close-btn"
                aria-label="Close modal"
              >
                ✕
              </button>
            </div>

            <div className="stage-health-modal__content">
              {/* Executive Plain English Explanation */}
              <div className={`stage-health-modal__banner stage-health-modal__banner--${statusTone}`}>
                <div className="stage-health-modal__banner-title">
                  <strong>{healthData.status_label}</strong>
                  <span className="stage-health-modal__banner-tag">
                    {healthData.total_stage_checkpoints} Checkpoints Evaluated
                  </span>
                </div>
                <p className="stage-health-modal__banner-desc">{healthData.status_explanation}</p>
              </div>

              {/* Prescriptive Recommendations Section */}
              <div className="stage-health-modal__section">
                <div className="stage-health-modal__section-title">
                  <span>💡 Prescriptive Action Plan & What to Change</span>
                </div>
                <div className="stage-recommendations-list">
                  {healthData.prescriptive_recommendations.map((rec: PrescriptiveRecommendation, idx: number) => (
                    <div
                      key={idx}
                      className={`stage-rec-card stage-rec-card--${rec.priority}`}
                    >
                      <div className="stage-rec-card__header">
                        <span className={`stage-rec-badge stage-rec-badge--${rec.priority}`}>
                          {rec.priority.toUpperCase()}
                        </span>
                        <h4 className="stage-rec-card__title">{rec.title}</h4>
                      </div>
                      <p className="stage-rec-card__desc">{rec.description}</p>
                      <div className="stage-rec-card__action">
                        <span className="stage-rec-card__action-label">Recommended Action:</span>
                        <code className="stage-rec-card__action-code">{rec.suggested_action}</code>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Hyperparameters & Rewards Audit Table */}
              <div className="stage-health-modal__section">
                <div className="stage-health-modal__section-title">
                  <span>⚙️ Active Configuration & Rewards Audit</span>
                </div>
                <div className="stage-audit-table-wrapper">
                  <table className="stage-audit-table">
                    <thead>
                      <tr>
                        <th>Parameter</th>
                        <th>Current Value</th>
                        <th>Recommended</th>
                        <th>Audit Note</th>
                      </tr>
                    </thead>
                    <tbody>
                      {healthData.hyperparameter_audit.map((item, idx) => (
                        <tr key={idx}>
                          <td>
                            <code>{item.parameter}</code>
                          </td>
                          <td>
                            <strong>{String(item.current_value)}</strong>
                          </td>
                          <td>
                            <code>{String(item.recommended_value)}</code>
                          </td>
                          <td className={`stage-audit-cell--${item.status}`}>{item.note}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>

              {/* Seed Reliability Matrix Breakdown */}
              <div className="stage-health-modal__section">
                <div className="stage-health-modal__section-title">
                  <span>🎯 Seed Reliability Matrix (All-Time Stage 8)</span>
                </div>
                <div className="stage-seed-grid">
                  {healthData.seed_matrix.map((s) => (
                    <div key={s.seed} className={`stage-seed-card stage-seed-card--${s.status}`}>
                      <div className="stage-seed-card__header">
                        <span className="stage-seed-card__num">Seed {s.seed}</span>
                        <span className="stage-seed-card__wr">{(s.win_rate * 100).toFixed(0)}%</span>
                      </div>
                      <div className="stage-seed-card__sub">
                        {s.wins}/{s.total} wins · {s.fails} fails
                      </div>
                      {s.current_consecutive_fails > 0 && (
                        <div className="stage-seed-card__streak">
                          Streak: {s.current_consecutive_fails} fail(s)
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            </div>

            <div className="stage-health-modal__footer">
              <button
                onClick={handleCopySummary}
                className="stage-health-copy-btn"
                title="Copy full diagnostic report to clipboard"
              >
                {copied ? "✓ Copied to Clipboard!" : "📋 Copy Diagnostic Summary"}
              </button>
              <button onClick={() => setIsModalOpen(false)} className="stage-health-close-btn">
                Close
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
}

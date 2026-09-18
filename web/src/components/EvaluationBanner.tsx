import React from "react";
import type { TrainingStatus, EvaluationSeedResult } from "../state/types";

export interface EvaluationBannerProps {
  status?: TrainingStatus | null;
  compact?: boolean;
  className?: string;
  onViewReplay?: (replayPath: string) => void;
}

export function EvaluationBanner({
  status,
  compact = false,
  className = "",
}: EvaluationBannerProps) {
  if (!status) return null;

  const evalStatus = status.evaluation_status;
  const isInProgress = Boolean(
    status.evaluation_in_progress ||
    evalStatus?.in_progress ||
    status.phase === "evaluation"
  );

  const isComplete = status.phase === "evaluation_complete";

  // If not in progress and not complete, and no recent results, don't show
  if (!isInProgress && !isComplete && (!evalStatus?.recent_results || evalStatus.recent_results.length === 0)) {
    return null;
  }

  const mode = evalStatus?.mode || status.evaluation_mode || "evaluation";
  const modeLabel = mode.toLowerCase().includes("quick")
    ? "Quick Evaluation"
    : mode.toLowerCase().includes("confidence")
    ? "Confidence Evaluation"
    : "Evaluation Set";

  const currentSeed = evalStatus?.current_seed ?? status.evaluation_seed ?? null;
  const seedIndex = evalStatus?.seed_index ?? status.evaluation_seed_index ?? (currentSeed !== null ? 1 : 0);
  const totalSeeds = evalStatus?.total_seeds ?? status.evaluation_total_seeds ?? (evalStatus?.seeds?.length || 0);
  const completedSeeds = evalStatus?.completed_seeds ?? status.evaluation_completed_seeds ?? 0;
  const successCount = evalStatus?.success_count ?? status.evaluation_success_count ?? 0;
  const timeoutCount = evalStatus?.timeout_count ?? 0;

  const rawRate = evalStatus?.success_rate ?? (completedSeeds > 0 ? successCount / completedSeeds : null);
  const successRatePct = rawRate !== null && rawRate !== undefined ? Math.round(rawRate * 100) : null;

  const inProgressMessage =
    evalStatus?.message ||
    status.evaluation_message ||
    (currentSeed !== null
      ? `Evaluating seed ${currentSeed} (${seedIndex} of ${totalSeeds || 1}) in progress...`
      : "Running evaluation set...");

  const latestResult: EvaluationSeedResult | null =
    evalStatus?.latest_result ||
    status.evaluation_latest_result ||
    (evalStatus?.recent_results && evalStatus.recent_results.length > 0
      ? evalStatus.recent_results[evalStatus.recent_results.length - 1]
      : null);

  const recentResults: EvaluationSeedResult[] =
    evalStatus?.recent_results || status.evaluation_recent_results || [];

  // Determine list of seeds to render in the progress track
  const allSeeds = evalStatus?.seeds && evalStatus.seeds.length > 0
    ? evalStatus.seeds
    : Array.from(new Set([
        ...recentResults.map((r) => r.seed),
        ...(currentSeed !== null ? [currentSeed] : []),
      ]));

  const progressPercent = totalSeeds > 0 ? Math.min(100, Math.round((completedSeeds / totalSeeds) * 100)) : 0;

  return (
    <div
      className={`evaluation-banner ${isInProgress ? "evaluation-banner--live" : "evaluation-banner--complete"} ${compact ? "evaluation-banner--compact" : ""} ${className}`}
      data-testid="evaluation-banner"
      role="region"
      aria-label="Evaluation Set Progress"
      style={{
        background: isInProgress
          ? "linear-gradient(135deg, rgba(30, 27, 75, 0.94) 0%, rgba(15, 23, 42, 0.96) 60%, rgba(30, 58, 138, 0.9) 100%)"
          : "linear-gradient(135deg, rgba(15, 23, 42, 0.94) 0%, rgba(17, 24, 39, 0.96) 100%)",
        border: isInProgress ? "1px solid rgba(99, 102, 241, 0.45)" : "1px solid rgba(148, 163, 184, 0.2)",
        borderRadius: "0.85rem",
        padding: compact ? "0.6rem 0.8rem" : "0.75rem 1.1rem",
        flexShrink: 0,
        boxShadow: isInProgress ? "0 8px 32px rgba(99, 102, 241, 0.2)" : "0 4px 16px rgba(0, 0, 0, 0.25)",
        color: "#e2e8f0",
        backdropFilter: "blur(12px)",
        transition: "all 200ms ease",
      }}
    >
      {/* ── Top Header Row ── */}
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          flexWrap: "wrap",
          gap: "0.5rem",
          marginBottom: "0.5rem",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: "0.6rem", flexWrap: "wrap" }}>
          {/* Animated Status Pill */}
          <div
            style={{
              display: "inline-flex",
              alignItems: "center",
              gap: "0.4rem",
              padding: "0.25rem 0.65rem",
              borderRadius: "9999px",
              fontSize: "0.75rem",
              fontWeight: 700,
              letterSpacing: "0.05em",
              textTransform: "uppercase",
              background: isInProgress
                ? "linear-gradient(90deg, rgba(99, 102, 241, 0.35), rgba(168, 85, 247, 0.35))"
                : "rgba(34, 197, 94, 0.2)",
              border: isInProgress ? "1px solid rgba(168, 85, 247, 0.5)" : "1px solid rgba(34, 197, 94, 0.4)",
              color: isInProgress ? "#c084fc" : "#4ade80",
            }}
          >
            <span
              style={{
                display: "inline-block",
                width: "8px",
                height: "8px",
                borderRadius: "50%",
                background: isInProgress ? "#c084fc" : "#4ade80",
                boxShadow: isInProgress ? "0 0 10px #c084fc" : "none",
                animation: isInProgress ? "pulse 1.5s infinite" : "none",
              }}
            />
            {isInProgress ? `${modeLabel} Running` : `${modeLabel} Complete`}
          </div>

          {/* Current Seed Highlight */}
          {isInProgress && currentSeed !== null && (
            <span
              style={{
                display: "inline-flex",
                alignItems: "center",
                gap: "0.3rem",
                padding: "0.2rem 0.6rem",
                borderRadius: "0.5rem",
                background: "rgba(56, 189, 248, 0.15)",
                border: "1px solid rgba(56, 189, 248, 0.4)",
                color: "#38bdf8",
                fontSize: "0.8rem",
                fontWeight: 600,
              }}
              data-testid="evaluation-current-seed-badge"
            >
              <span>Seed on test:</span>
              <strong style={{ color: "#7dd3fc" }}>{currentSeed}</strong>
              <span style={{ opacity: 0.8 }}>({seedIndex}/{totalSeeds || "?"})</span>
            </span>
          )}
        </div>

        {/* Live Score Overview */}
        <div style={{ display: "flex", alignItems: "center", gap: "0.6rem", fontSize: "0.82rem" }}>
          {completedSeeds > 0 && (
            <div
              style={{
                display: "flex",
                alignItems: "center",
                gap: "0.4rem",
                background: "rgba(15, 23, 42, 0.7)",
                padding: "0.25rem 0.65rem",
                borderRadius: "0.5rem",
                border: "1px solid rgba(148, 163, 184, 0.15)",
              }}
            >
              <span style={{ color: "#94a3b8" }}>Score:</span>
              <strong style={{ color: successCount > 0 ? "#4ade80" : "#f87171" }}>
                {successCount} / {completedSeeds} passed
              </strong>
              {successRatePct !== null && (
                <span
                  style={{
                    color: successRatePct >= 50 ? "#4ade80" : "#fbbf24",
                    fontWeight: 700,
                  }}
                >
                  ({successRatePct}%)
                </span>
              )}
              {timeoutCount > 0 && (
                <span style={{ color: "#f87171", marginLeft: "0.3rem" }}>
                  ({timeoutCount} timeout{timeoutCount > 1 ? "s" : ""})
                </span>
              )}
            </div>
          )}
        </div>
      </div>

      {/* ── Active In-Progress Message ── */}
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: "0.5rem",
          background: "rgba(15, 23, 42, 0.6)",
          padding: "0.4rem 0.75rem",
          borderRadius: "0.5rem",
          border: "1px solid rgba(99, 102, 241, 0.25)",
          marginBottom: "0.6rem",
          fontSize: "0.85rem",
        }}
        data-testid="evaluation-in-progress-message"
      >
        <span style={{ color: isInProgress ? "#a855f7" : "#4ade80", fontSize: "1rem" }}>
          {isInProgress ? "⚡" : "✓"}
        </span>
        <span style={{ color: "#f1f5f9", fontWeight: 500, flex: 1 }}>
          {inProgressMessage}
        </span>
        {totalSeeds > 0 && (
          <span style={{ color: "#94a3b8", fontSize: "0.75rem", whiteSpace: "nowrap" }}>
            {completedSeeds} of {totalSeeds} seeds done ({progressPercent}%)
          </span>
        )}
      </div>

      {/* ── Progress Bar ── */}
      {totalSeeds > 0 && (
        <div
          style={{
            height: "4px",
            width: "100%",
            background: "rgba(148, 163, 184, 0.15)",
            borderRadius: "9999px",
            overflow: "hidden",
            marginBottom: "0.6rem",
          }}
          aria-hidden="true"
        >
          <div
            style={{
              height: "100%",
              width: `${progressPercent}%`,
              background: isInProgress
                ? "linear-gradient(90deg, #6366f1 0%, #a855f7 50%, #38bdf8 100%)"
                : "#22c55e",
              transition: "width 300ms ease",
            }}
          />
        </div>
      )}

      {/* ── Seed Chips Track ("What seed it's on & how it did") ── */}
      {allSeeds.length > 0 && (
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: "0.4rem",
            flexWrap: "wrap",
            marginTop: "0.4rem",
          }}
          data-testid="evaluation-seeds-track"
        >
          <span style={{ fontSize: "0.74rem", color: "#94a3b8", marginRight: "0.2rem" }}>
            Seeds:
          </span>
          {allSeeds.map((seed, sIdx) => {
            const result = recentResults.find((r) => r.seed === seed);
            const isCurrentlyRunning = isInProgress && seed === currentSeed;
            const isFinished = Boolean(result);
            const isWin = Boolean(result?.success);

            let bg = "rgba(30, 41, 59, 0.7)";
            let border = "1px solid rgba(148, 163, 184, 0.2)";
            let color = "#94a3b8";

            if (isCurrentlyRunning) {
              bg = "linear-gradient(135deg, rgba(99, 102, 241, 0.4), rgba(56, 189, 248, 0.35))";
              border = "1px solid #38bdf8";
              color = "#38bdf8";
            } else if (isFinished) {
              if (isWin) {
                bg = "rgba(34, 197, 94, 0.18)";
                border = "1px solid rgba(74, 222, 128, 0.45)";
                color = "#4ade80";
              } else {
                bg = "rgba(239, 68, 68, 0.18)";
                border = "1px solid rgba(248, 113, 113, 0.45)";
                color = "#f87171";
              }
            }

            return (
              <span
                key={`eval-seed-${seed}-${sIdx}`}
                title={
                  result
                    ? `Seed ${seed}: ${result.status} | Reward: ${result.reward} | Penned: ${result.penned}/${result.total_sheep} | Steps: ${result.steps}`
                    : isCurrentlyRunning
                    ? `Seed ${seed} is currently evaluating in progress...`
                    : `Seed ${seed} is scheduled in evaluation set`
                }
                style={{
                  display: "inline-flex",
                  alignItems: "center",
                  gap: "0.25rem",
                  padding: "0.15rem 0.5rem",
                  borderRadius: "0.4rem",
                  fontSize: "0.74rem",
                  fontWeight: isCurrentlyRunning ? 700 : 500,
                  background: bg,
                  border: border,
                  color: color,
                  boxShadow: isCurrentlyRunning ? "0 0 10px rgba(56, 189, 248, 0.4)" : "none",
                  animation: isCurrentlyRunning ? "pulse 1.8s infinite" : "none",
                  cursor: "default",
                }}
              >
                <span>
                  {isCurrentlyRunning ? "⚡" : isFinished ? (isWin ? "✓" : "✗") : "○"}
                </span>
                <span>{seed}</span>
                {result && (
                  <span style={{ fontSize: "0.68rem", opacity: 0.85 }}>
                    ({result.penned}/{result.total_sheep})
                  </span>
                )}
              </span>
            );
          })}
        </div>
      )}

      {/* ── Latest Completed Result Summary ── */}
      {latestResult && (
        <div
          style={{
            marginTop: "0.5rem",
            fontSize: "0.76rem",
            color: "#94a3b8",
            display: "flex",
            alignItems: "center",
            gap: "0.5rem",
            flexWrap: "wrap",
          }}
          data-testid="evaluation-latest-result-summary"
        >
          <span style={{ color: "#cbd5e1" }}>Last completed:</span>
          <strong style={{ color: "#f8fafc" }}>Seed {latestResult.seed}</strong>
          <span
            style={{
              fontWeight: 600,
              color: latestResult.success ? "#4ade80" : "#f87171",
            }}
          >
            {latestResult.status}
          </span>
          <span>· Penned: {latestResult.penned}/{latestResult.total_sheep}</span>
          <span>· Reward: {latestResult.reward.toFixed(2)}</span>
          <span>· Steps: {latestResult.steps}</span>
        </div>
      )}
    </div>
  );
}

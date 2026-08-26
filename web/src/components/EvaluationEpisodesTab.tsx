import React, { useState, useEffect, useMemo, useCallback, useRef } from "react";
import type {
  EvaluationSummaryPayload,
  EvaluationRecordPayload,
  ReplayBundle,
  ReplaySnapshot,
} from "../state/types";
import { loadRecentEvaluations, runLiveReplay, fetchReplayById, loadReplay } from "../lib/api";
import { FieldView } from "./FieldView";

interface EvaluationEpisodesTabProps {
  currentStage?: number;
  runId?: string | null;
}

export function EvaluationEpisodesTab({ currentStage }: EvaluationEpisodesTabProps) {
  const [evaluations, setEvaluations] = useState<EvaluationSummaryPayload[]>([]);
  const [selectedEvalIndex, setSelectedEvalIndex] = useState<number>(0);
  const [selectedSeed, setSelectedSeed] = useState<number | null>(null);
  const [outcomeFilter, setOutcomeFilter] = useState<"all" | "pass" | "fail">("all");
  const [isLoadingEvals, setIsLoadingEvals] = useState<boolean>(true);
  const [evalError, setEvalError] = useState<string | null>(null);

  // Replay playback state
  const [replayBundle, setReplayBundle] = useState<ReplayBundle | null>(null);
  const [currentStep, setCurrentStep] = useState<number>(0);
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const [playbackSpeed, setPlaybackSpeed] = useState<number>(1);
  const [isLoadingReplay, setIsLoadingReplay] = useState<boolean>(false);
  const [replayError, setReplayError] = useState<string | null>(null);

  // Fetch recent 5 evaluations
  const fetchEvals = useCallback(async () => {
    setIsLoadingEvals(true);
    setEvalError(null);
    try {
      const data = await loadRecentEvaluations(5, currentStage);
      if (data && data.length > 0) {
        setEvaluations(data);
        setSelectedEvalIndex(0);
        if (data[0].records && data[0].records.length > 0) {
          setSelectedSeed(data[0].records[0].seed);
        }
      } else {
        // Try fetching without stage filter if no stage-specific ones found
        const fallbackData = await loadRecentEvaluations(5);
        if (fallbackData && fallbackData.length > 0) {
          setEvaluations(fallbackData);
          setSelectedEvalIndex(0);
          if (fallbackData[0].records && fallbackData[0].records.length > 0) {
            setSelectedSeed(fallbackData[0].records[0].seed);
          }
        } else {
          setEvaluations([]);
        }
      }
    } catch (err: any) {
      setEvalError(err?.message || "Failed to load recent evaluations");
    } finally {
      setIsLoadingEvals(false);
    }
  }, [currentStage]);

  useEffect(() => {
    fetchEvals();
  }, [fetchEvals]);

  const activeEval: EvaluationSummaryPayload | null = evaluations[selectedEvalIndex] ?? null;

  // Filter the 10 episode records for the selected evaluation
  const filteredRecords = useMemo(() => {
    if (!activeEval || !activeEval.records) return [];
    if (outcomeFilter === "pass") {
      return activeEval.records.filter((r) => r.success);
    }
    if (outcomeFilter === "fail") {
      return activeEval.records.filter((r) => !r.success);
    }
    return activeEval.records;
  }, [activeEval, outcomeFilter]);

  // Active record
  const activeRecord: EvaluationRecordPayload | null = useMemo(() => {
    if (!activeEval || !activeEval.records || selectedSeed === null) return null;
    return activeEval.records.find((r) => r.seed === selectedSeed) ?? activeEval.records[0] ?? null;
  }, [activeEval, selectedSeed]);

  // Load replay when active record or active evaluation changes
  const loadReplayForRecord = useCallback(
    async (record: EvaluationRecordPayload, evalSummary: EvaluationSummaryPayload) => {
      setIsLoadingReplay(true);
      setReplayError(null);
      setIsPlaying(false);
      setCurrentStep(0);
      setReplayBundle(null);

      try {
        let bundle: ReplayBundle | null = null;

        // 1. Try loading by replay_path if available
        if (record.replay_path && record.replay_path.trim().length > 0) {
          try {
            const rawPath = record.replay_path.replace(/\\/g, "/");
            const filename = rawPath.split("/").pop() || "";
            const replayId = filename.replace(/\.json(\.gz)?$/, "");

            if (replayId) {
              bundle = await fetchReplayById(replayId);
            }
            if (!bundle) {
              bundle = await loadReplay(rawPath);
            }
          } catch {
            // Fall back to live reproduction
          }
        }

        // 2. If no saved bundle, run deterministic simulation
        if (!bundle || !bundle.frames || bundle.frames.length === 0) {
          const runRes = await runLiveReplay({
            seed: record.seed,
            checkpoint_episode: evalSummary.checkpoint_episode,
          });
          if (runRes && (runRes.frames?.length || (runRes as any).move_history?.length)) {
            bundle = runRes;
          }
        }

        if (bundle) {
          setReplayBundle(bundle);
          setCurrentStep(0);
        } else {
          setReplayError("No replay frames available for this episode seed.");
        }
      } catch (err: any) {
        setReplayError(err?.message || "Failed to load episode replay");
      } finally {
        setIsLoadingReplay(false);
      }
    },
    []
  );

  // Trigger replay load when active record changes
  useEffect(() => {
    if (activeRecord && activeEval) {
      loadReplayForRecord(activeRecord, activeEval);
    }
  }, [activeRecord, activeEval, loadReplayForRecord]);

  // Extract frames from replay bundle
  const frames: ReplaySnapshot[] = useMemo(() => {
    if (!replayBundle) return [];
    if (replayBundle.frames && replayBundle.frames.length > 0) {
      return replayBundle.frames.map((f) => f.snapshot);
    }
    const legacyHistory = (replayBundle as any).move_history;
    if (Array.isArray(legacyHistory) && legacyHistory.length > 0) {
      return legacyHistory.map((f: { snapshot: ReplaySnapshot }) => f.snapshot);
    }
    const initialOrFinal = (replayBundle as any).initial_state || replayBundle.final_snapshot;
    if (initialOrFinal) {
      return [initialOrFinal];
    }
    return [];
  }, [replayBundle]);

  const maxSteps = Math.max(0, frames.length - 1);
  const currentSnapshot: ReplaySnapshot | null =
    frames[currentStep] || (replayBundle as any)?.initial_state || replayBundle?.final_snapshot || null;

  // Playback timer loop
  const isPlayingRef = useRef(isPlaying);
  isPlayingRef.current = isPlaying;
  const currentStepRef = useRef(currentStep);
  currentStepRef.current = currentStep;
  const maxStepsRef = useRef(maxSteps);
  maxStepsRef.current = maxSteps;

  useEffect(() => {
    if (!isPlaying) return;

    const intervalMs = Math.max(16, Math.floor(100 / playbackSpeed));
    const timer = setInterval(() => {
      if (!isPlayingRef.current) return;
      if (currentStepRef.current >= maxStepsRef.current) {
        setIsPlaying(false);
        return;
      }
      setCurrentStep((prev) => Math.min(prev + 1, maxStepsRef.current));
    }, intervalMs);

    return () => clearInterval(timer);
  }, [isPlaying, playbackSpeed]);

  const handleTogglePlay = () => {
    if (isPlaying) {
      setIsPlaying(false);
    } else {
      if (currentStep >= maxSteps) {
        setCurrentStep(0);
      }
      setIsPlaying(true);
    }
  };

  const handleStepBack = () => {
    setIsPlaying(false);
    setCurrentStep((prev) => Math.max(0, prev - 1));
  };

  const handleStepForward = () => {
    setIsPlaying(false);
    setCurrentStep((prev) => Math.min(maxSteps, prev + 1));
  };

  const handleResetToStart = () => {
    setIsPlaying(false);
    setCurrentStep(0);
  };

  const handleJumpToEnd = () => {
    setIsPlaying(false);
    setCurrentStep(maxSteps);
  };

  const handleScrubberChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setIsPlaying(false);
    setCurrentStep(Number(e.target.value));
  };

  return (
    <div className="eval-episodes-tab" data-testid="evaluation-episodes-tab">
      {/* ── Top Bar: Header & Controls ── */}
      <div className="eval-episodes-header">
        <div className="eval-episodes-header__info">
          <h3 style={{ margin: 0, fontSize: "1.1rem", color: "#f8fafc", display: "flex", alignItems: "center", gap: "0.5rem" }}>
            <span>🎯</span> Formal Evaluation Benchmark Inspector
          </h3>
          <p style={{ margin: "0.2rem 0 0", fontSize: "0.8rem", color: "#94a3b8" }}>
            Inspect the last 5 formal evaluations with all 10 deterministic passing and failing benchmark episodes. Select any episode to replay.
          </p>
        </div>

        <button
          type="button"
          className="eval-refresh-btn"
          onClick={() => void fetchEvals()}
          disabled={isLoadingEvals}
          title="Refresh evaluation records"
        >
          {isLoadingEvals ? "Refreshing..." : "🔄 Refresh Evals"}
        </button>
      </div>

      {evalError && (
        <div className="warning-box warning-box--error" style={{ marginBottom: "0.75rem" }}>
          {evalError}
        </div>
      )}

      {isLoadingEvals && evaluations.length === 0 && (
        <div style={{ padding: "2rem", textAlign: "center", color: "#94a3b8" }}>
          Loading recent evaluation benchmarks...
        </div>
      )}

      {!isLoadingEvals && evaluations.length === 0 && (
        <div className="warning-box warning-box--info" style={{ marginTop: "1rem" }}>
          No formal evaluation benchmarks found for Stage {currentStage ?? "all"}. Run training evaluation checkpoints to generate formal benchmark episodes.
        </div>
      )}

      {evaluations.length > 0 && (
        <>
          {/* ── Evaluation Benchmark Selector (Last 5 Evaluations) ── */}
          <div className="eval-timeline-selector" role="group" aria-label="Evaluation Runs">
            <span style={{ fontSize: "0.78rem", fontWeight: 600, color: "#94a3b8", alignSelf: "center", marginRight: "0.25rem" }}>
              Last 5 Evals:
            </span>
            {evaluations.map((ev, idx) => {
              const passPct = Math.round(ev.success_rate * 100);
              const passCount = ev.records ? ev.records.filter((r) => r.success).length : 0;
              const totalCount = ev.records ? ev.records.length : 10;
              const isSelected = selectedEvalIndex === idx;
              const isPassingGrade = ev.success_rate >= 0.5;

              return (
                <button
                  key={ev.evaluation_id || `eval-${idx}`}
                  type="button"
                  className={`eval-select-card${isSelected ? " eval-select-card--active" : ""}`}
                  onClick={() => {
                    setSelectedEvalIndex(idx);
                    if (ev.records && ev.records.length > 0) {
                      setSelectedSeed(ev.records[0].seed);
                    }
                  }}
                >
                  <div className="eval-select-card__top">
                    <span className="eval-select-card__ep">
                      Checkpoint #{ev.checkpoint_episode ?? "N/A"}
                    </span>
                    <span className={`eval-pill ${isPassingGrade ? "eval-pill--good" : "eval-pill--warn"}`}>
                      {passPct}% Pass
                    </span>
                  </div>
                  <div className="eval-select-card__meta">
                    <span>Stage {ev.curriculum_stage ?? "—"}</span>
                    <span>·</span>
                    <span>{passCount}/{totalCount} Penned</span>
                    {ev.policy_version != null && (
                      <>
                        <span>·</span>
                        <span>v{ev.policy_version}</span>
                      </>
                    )}
                  </div>
                </button>
              );
            })}
          </div>

          {/* ── Active Evaluation Summary KPI Card ── */}
          {activeEval && (
            <div className="eval-kpi-banner">
              <div className="eval-kpi-item">
                <span className="eval-kpi-label">Checkpoint</span>
                <strong className="eval-kpi-val">#{activeEval.checkpoint_episode}</strong>
                <span className="eval-kpi-sub">Stage {activeEval.curriculum_stage ?? "—"}</span>
              </div>
              <div className="eval-kpi-item">
                <span className="eval-kpi-label">Pass / Success Rate</span>
                <strong className="eval-kpi-val" style={{ color: activeEval.success_rate >= 0.5 ? "#34d399" : "#f87171" }}>
                  {Math.round(activeEval.success_rate * 100)}%
                </strong>
                <span className="eval-kpi-sub">{activeEval.records.filter((r) => r.success).length} of {activeEval.records.length} passed</span>
              </div>
              <div className="eval-kpi-item">
                <span className="eval-kpi-label">Timeout Rate</span>
                <strong className="eval-kpi-val" style={{ color: activeEval.timeout_rate > 0.4 ? "#f87171" : "#94a3b8" }}>
                  {Math.round(activeEval.timeout_rate * 100)}%
                </strong>
                <span className="eval-kpi-sub">{activeEval.records.filter((r) => r.timeout).length} timed out</span>
              </div>
              <div className="eval-kpi-item">
                <span className="eval-kpi-label">Avg Steps</span>
                <strong className="eval-kpi-val">{Math.round(activeEval.average_completion_steps)}</strong>
                <span className="eval-kpi-sub">per episode</span>
              </div>
              <div className="eval-kpi-item">
                <span className="eval-kpi-label">Avg Penned</span>
                <strong className="eval-kpi-val">{activeEval.average_sheep_penned.toFixed(1)}</strong>
                <span className="eval-kpi-sub">sheep</span>
              </div>
              <div className="eval-kpi-item">
                <span className="eval-kpi-label">Avg Total Reward</span>
                <strong className="eval-kpi-val" style={{ color: activeEval.average_reward >= 0 ? "#38bdf8" : "#f87171" }}>
                  {activeEval.average_reward.toFixed(1)}
                </strong>
                <span className="eval-kpi-sub">cumulative return</span>
              </div>
            </div>
          )}

          {/* ── Main Content Split: Episodes List (Left) + Interactive Replay (Right) ── */}
          <div className="eval-episodes-content-grid">
            {/* ── Left Column: 10 Pass/Fail Episodes Grid ── */}
            <div className="eval-episodes-list-col">
              <div className="eval-episodes-list-header">
                <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
                  <strong style={{ fontSize: "0.9rem", color: "#f8fafc" }}>
                    Benchmark Episodes ({activeEval?.records.length ?? 0} Seeds)
                  </strong>
                </div>

                {/* Outcome Filter Chips */}
                <div className="eval-filter-chips">
                  <button
                    type="button"
                    className={`eval-filter-chip${outcomeFilter === "all" ? " eval-filter-chip--active" : ""}`}
                    onClick={() => setOutcomeFilter("all")}
                  >
                    All ({activeEval?.records.length ?? 0})
                  </button>
                  <button
                    type="button"
                    className={`eval-filter-chip${outcomeFilter === "pass" ? " eval-filter-chip--active" : ""}`}
                    onClick={() => setOutcomeFilter("pass")}
                  >
                    Pass ({activeEval?.records.filter((r) => r.success).length ?? 0})
                  </button>
                  <button
                    type="button"
                    className={`eval-filter-chip${outcomeFilter === "fail" ? " eval-filter-chip--active" : ""}`}
                    onClick={() => setOutcomeFilter("fail")}
                  >
                    Fail ({activeEval?.records.filter((r) => !r.success).length ?? 0})
                  </button>
                </div>
              </div>

              <div className="eval-episodes-scroll-list">
                {filteredRecords.map((rec) => {
                  const isSelected = selectedSeed === rec.seed;
                  const isPass = rec.success;

                  return (
                    <div
                      key={`seed-${rec.seed}`}
                      className={`eval-record-card${isSelected ? " eval-record-card--selected" : ""}${isPass ? " eval-record-card--pass" : " eval-record-card--fail"}`}
                      onClick={() => setSelectedSeed(rec.seed)}
                      role="button"
                      tabIndex={0}
                      onKeyDown={(e) => {
                        if (e.key === "Enter" || e.key === " ") {
                          setSelectedSeed(rec.seed);
                        }
                      }}
                    >
                      <div className="eval-record-card__header">
                        <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
                          <span className={`eval-status-badge ${isPass ? "eval-status-badge--pass" : "eval-status-badge--fail"}`}>
                            {isPass ? "✓ PASS" : "✗ FAIL"}
                          </span>
                          <strong className="eval-record-seed">Seed {rec.seed}</strong>
                        </div>
                        <span className="eval-record-steps">
                          {rec.steps} steps
                        </span>
                      </div>

                      <div className="eval-record-card__body">
                        <div className="eval-record-metric">
                          <span className="eval-metric-label">Penned:</span>
                          <span className="eval-metric-val">{rec.sheep_penned} sheep</span>
                        </div>
                        <div className="eval-record-metric">
                          <span className="eval-metric-label">Stop Reason:</span>
                          <span className="eval-metric-val" style={{ color: isPass ? "#34d399" : "#f87171" }}>
                            {rec.stop_reason || (isPass ? "success" : "timeout")}
                          </span>
                        </div>
                        <div className="eval-record-metric">
                          <span className="eval-metric-label">Reward:</span>
                          <span className="eval-metric-val">{rec.reward_total.toFixed(1)}</span>
                        </div>
                        {rec.corner_time_pct != null && rec.corner_time_pct > 0 && (
                          <div className="eval-record-metric">
                            <span className="eval-metric-label">Corner Time:</span>
                            <span className="eval-metric-val" style={{ color: "#fb923c" }}>
                              {Math.round(rec.corner_time_pct * 100)}%
                            </span>
                          </div>
                        )}
                        {rec.role_switches != null && (
                          <div className="eval-record-metric">
                            <span className="eval-metric-label">Role Flips:</span>
                            <span className="eval-metric-val">{rec.role_switches}</span>
                          </div>
                        )}
                      </div>

                      {isSelected && (
                        <div className="eval-record-card__selected-indicator">
                          ▶ Active Replay Target
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            </div>

            {/* ── Right Column: Interactive Replay Visualizer ── */}
            <div className="eval-replay-col">
              <div className="eval-replay-header">
                <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
                  <span style={{ fontSize: "1rem" }}>🎬</span>
                  <strong style={{ fontSize: "0.95rem", color: "#f8fafc" }}>
                    Episode Replay: Checkpoint #{activeEval?.checkpoint_episode ?? "—"} · Seed {activeRecord?.seed ?? "—"}
                  </strong>
                  {activeRecord && (
                    <span className={`eval-status-badge ${activeRecord.success ? "eval-status-badge--pass" : "eval-status-badge--fail"}`}>
                      {activeRecord.success ? "PASS" : "FAIL"}
                    </span>
                  )}
                </div>

                {isLoadingReplay && (
                  <span style={{ fontSize: "0.75rem", color: "#38bdf8", fontStyle: "italic" }}>
                    Generating / loading replay...
                  </span>
                )}
              </div>

              {replayError && (
                <div className="warning-box warning-box--error" style={{ margin: "0.5rem 0" }}>
                  {replayError}
                </div>
              )}

              {/* Field Visualizer Canvas */}
              <div className="eval-field-wrap">
                <FieldView snapshot={currentSnapshot} />
              </div>

              {/* Replay Controls Toolbar */}
              <div className="eval-replay-controls">
                {/* Scrubber Slider */}
                <div className="eval-scrubber-row">
                  <span style={{ fontSize: "0.78rem", color: "#94a3b8", minWidth: "55px" }}>
                    Step {currentStep}
                  </span>
                  <input
                    type="range"
                    min={0}
                    max={maxSteps}
                    value={currentStep}
                    onChange={handleScrubberChange}
                    className="eval-scrubber-slider"
                    aria-label="Replay timeline scrubber"
                    disabled={maxSteps === 0}
                  />
                  <span style={{ fontSize: "0.78rem", color: "#94a3b8", minWidth: "55px", textAlign: "right" }}>
                    Max {maxSteps}
                  </span>
                </div>

                {/* Playback Buttons & Speed Selector */}
                <div className="eval-playback-button-row">
                  <div style={{ display: "flex", gap: "0.35rem", alignItems: "center" }}>
                    <button
                      type="button"
                      className="eval-ctrl-btn"
                      onClick={handleResetToStart}
                      disabled={currentStep === 0}
                      title="Reset to start"
                    >
                      ⏮
                    </button>
                    <button
                      type="button"
                      className="eval-ctrl-btn"
                      onClick={handleStepBack}
                      disabled={currentStep === 0}
                      title="Step backward"
                    >
                      ⏪
                    </button>
                    <button
                      type="button"
                      className={`eval-ctrl-btn eval-ctrl-btn--primary${isPlaying ? " eval-ctrl-btn--playing" : ""}`}
                      onClick={handleTogglePlay}
                      disabled={maxSteps === 0}
                      title={isPlaying ? "Pause" : "Play"}
                    >
                      {isPlaying ? "⏸ Pause" : "▶ Play"}
                    </button>
                    <button
                      type="button"
                      className="eval-ctrl-btn"
                      onClick={handleStepForward}
                      disabled={currentStep >= maxSteps}
                      title="Step forward"
                    >
                      ⏩
                    </button>
                    <button
                      type="button"
                      className="eval-ctrl-btn"
                      onClick={handleJumpToEnd}
                      disabled={currentStep >= maxSteps}
                      title="Jump to end"
                    >
                      ⏭
                    </button>
                  </div>

                  {/* Speed Selector */}
                  <div style={{ display: "flex", alignItems: "center", gap: "0.25rem" }}>
                    <span style={{ fontSize: "0.75rem", color: "#94a3b8", marginRight: "0.2rem" }}>Speed:</span>
                    {[0.5, 1, 2, 5, 10].map((spd) => (
                      <button
                        key={`speed-${spd}`}
                        type="button"
                        className={`eval-speed-btn${playbackSpeed === spd ? " eval-speed-btn--active" : ""}`}
                        onClick={() => setPlaybackSpeed(spd)}
                      >
                        {spd}x
                      </button>
                    ))}
                  </div>
                </div>

                {/* Diagnostic Details at Current Frame */}
                {activeRecord && (
                  <div className="eval-frame-diagnostics">
                    <span>
                      <strong>Result:</strong> {activeRecord.sheep_penned} sheep penned · {activeRecord.steps} total steps · Stop: {activeRecord.stop_reason}
                    </span>
                    {activeRecord.pen_zone && (
                      <span>
                        <strong>Pen:</strong> {activeRecord.pen_zone}
                      </span>
                    )}
                    {activeRecord.final_sheep_zone && (
                      <span>
                        <strong>Sheep Zone:</strong> {activeRecord.final_sheep_zone}
                      </span>
                    )}
                  </div>
                )}
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  );
}

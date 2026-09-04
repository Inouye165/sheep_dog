import React, { useState, useEffect, useMemo, useCallback, useRef } from "react";
import type {
  EvaluationSummaryPayload,
  EvaluationRecordPayload,
  ReplayBundle,
  ReplaySnapshot,
} from "../state/types";
import { loadRecentEvaluations, runLiveReplay, fetchReplayById, loadReplay, pinEvaluation } from "../lib/api";
import { FieldView } from "./FieldView";
import { dogColor } from "./dogPalette";

interface EvaluationEpisodesTabProps {
  currentStage?: number;
  runId?: string | null;
  popupOnly?: boolean;
  isOpen?: boolean;
  standaloneWindow?: boolean;
  onClose?: () => void;
}

function getFenceSegments(snapshot: ReplaySnapshot) {
  const { pen } = snapshot;
  if (!pen || !pen.origin) return [];
  const opening = pen.opening ?? "left";
  const ox = pen.origin.x ?? 0;
  const oy = pen.origin.y ?? 0;
  const right = ox + (pen.width ?? 10);
  const bottom = oy + (pen.height ?? 10);
  const all = [
    { side: "top", x1: ox, y1: oy, x2: right, y2: oy },
    { side: "bottom", x1: ox, y1: bottom, x2: right, y2: bottom },
    { side: "left", x1: ox, y1: oy, x2: ox, y2: bottom },
    { side: "right", x1: right, y1: oy, x2: right, y2: bottom },
  ];
  return all.filter((segment) => segment.side !== opening);
}

function extractBundleFrames(bundle: ReplayBundle | null): ReplaySnapshot[] {
  if (!bundle) return [];
  if (bundle.frames && bundle.frames.length > 0) {
    return bundle.frames.map((f) => f.snapshot);
  }
  const legacyHistory = (bundle as any).move_history;
  if (Array.isArray(legacyHistory) && legacyHistory.length > 0) {
    return legacyHistory.map((f: { snapshot: ReplaySnapshot }) => f.snapshot);
  }
  const initOrFin = (bundle as any).initial_state || bundle.final_snapshot;
  return initOrFin ? [initOrFin] : [];
}

interface SeedMiniCardProps {
  record: EvaluationRecordPayload;
  bundle: ReplayBundle | null;
  loading: boolean;
  error: string | null;
  currentStep: number;
  isSelected: boolean;
  onSelect: () => void;
  onInspect: () => void;
}

function SeedMiniCard({
  record,
  bundle,
  loading,
  error,
  currentStep,
  isSelected,
  onSelect,
  onInspect,
}: SeedMiniCardProps) {
  const frames = useMemo(() => extractBundleFrames(bundle), [bundle]);
  const totalSteps = Math.max(0, frames.length - 1);
  const activeStep = Math.min(currentStep, totalSteps);
  const snapshot: ReplaySnapshot | null =
    frames[activeStep] || (bundle as any)?.initial_state || bundle?.final_snapshot || null;

  const isCompleted = frames.length > 0 && currentStep >= totalSteps;
  const isPass = record.success;
  const pennedCount = snapshot
    ? (snapshot.penned_count ?? snapshot.sheep?.filter((s) => s.penned).length ?? record.sheep_penned)
    : record.sheep_penned;
  const totalSheep = snapshot?.sheep?.length ?? 4;

  const baseWidth = snapshot?.grid_width ?? snapshot?.field_width ?? 40;
  const baseHeight = snapshot?.grid_height ?? snapshot?.field_height ?? 30;
  const width = Math.max(baseWidth, 40);
  const height = Math.max(baseHeight, 30);
  const densityScale = Math.max(width / 40, height / 30, 1);
  const dogRadius = 0.52 * densityScale;
  const sheepRadius = 0.44 * densityScale;
  const fenceStroke = 0.32 * densityScale;
  const penStroke = 0.1 * densityScale;
  const gateRadius = 0.2 * densityScale;

  const fences = snapshot ? getFenceSegments(snapshot) : [];
  const progressRatio = totalSteps > 0 ? activeStep / totalSteps : 0;

  return (
    <div
      className={`eval-seed-card${isSelected ? " eval-seed-card--selected" : ""}${
        isPass ? " eval-seed-card--pass" : " eval-seed-card--fail"
      }`}
      onClick={onSelect}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => {
        if (e.key === "Enter" || e.key === " ") onSelect();
      }}
      aria-label={`Seed ${record.seed} Replay`}
    >
      <div className="eval-seed-card__header">
        <div style={{ display: "flex", alignItems: "center", gap: "0.4rem" }}>
          <strong className="eval-record-seed" style={{ fontSize: "0.85rem" }}>
            Seed {record.seed}
          </strong>
          <span className={`eval-status-badge ${isPass ? "eval-status-badge--pass" : "eval-status-badge--fail"}`}>
            {isPass ? "✓ PASS" : "✗ FAIL"}
          </span>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: "0.35rem" }}>
          <span
            style={{
              fontSize: "0.72rem",
              color: pennedCount === totalSheep ? "#34d399" : "#f1f5f9",
              fontWeight: 600,
            }}
          >
            🐑 {pennedCount}/{totalSheep}
          </span>
          <span
            style={{
              fontSize: "0.68rem",
              color: isCompleted ? (isPass ? "#34d399" : "#f87171") : "#94a3b8",
            }}
          >
            {isCompleted ? (isPass ? "✓ Done" : "✗ Stopped") : `S${activeStep}/${totalSteps || record.steps}`}
          </span>
        </div>
      </div>

      <div className="eval-seed-card__field">
        {loading ? (
          <div
            style={{
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              gap: "0.4rem",
              color: "#38bdf8",
              fontSize: "0.75rem",
            }}
          >
            <span className="eval-spinner" />
            <span>Simulating Seed {record.seed}...</span>
          </div>
        ) : error ? (
          <div style={{ color: "#f87171", fontSize: "0.72rem", padding: "0.5rem", textAlign: "center" }}>
            {error}
          </div>
        ) : snapshot ? (
          <svg
            viewBox={`0 0 ${width} ${height}`}
            className="eval-seed-card__svg"
            role="img"
            aria-label={`Simulation map for Seed ${record.seed}`}
          >
            <defs>
              <linearGradient id={`bgGrad-${record.seed}`} x1="0" y1="0" x2="1" y2="1">
                <stop offset="0%" stopColor="#080e1a" />
                <stop offset="100%" stopColor="#0f1f2e" />
              </linearGradient>
            </defs>
            <rect width="100%" height="100%" fill={`url(#bgGrad-${record.seed})`} />

            {/* Pen Enclosure */}
            {snapshot.pen && snapshot.pen.origin && (
              <rect
                x={snapshot.pen.origin.x}
                y={snapshot.pen.origin.y}
                width={snapshot.pen.width}
                height={snapshot.pen.height}
                rx={0.3 * densityScale}
                fill="rgba(244, 197, 66, 0.12)"
                stroke="rgba(244, 197, 66, 0.5)"
                strokeDasharray={`${0.4 * densityScale} ${0.3 * densityScale}`}
                strokeWidth={penStroke}
              />
            )}

            {/* Fences */}
            {fences.map((seg, idx) => (
              <line
                key={`fence-${idx}`}
                x1={seg.x1}
                y1={seg.y1}
                x2={seg.x2}
                y2={seg.y2}
                stroke="#f4c542"
                strokeWidth={fenceStroke}
                strokeLinecap="round"
              />
            ))}

            {/* Gate markers */}
            {snapshot.pen && (snapshot.pen.opening === "left" || snapshot.pen.opening === undefined) && (
              <>
                <circle cx={snapshot.pen.origin.x} cy={snapshot.pen.origin.y} r={gateRadius} fill="#fde68a" />
                <circle
                  cx={snapshot.pen.origin.x}
                  cy={snapshot.pen.origin.y + snapshot.pen.height}
                  r={gateRadius}
                  fill="#fde68a"
                />
              </>
            )}

            {/* Sheep */}
            {snapshot.sheep?.map((sheep, idx) => {
              const sx = ((sheep as any).position?.x ?? sheep.x ?? 0) + 0.5;
              const sy = ((sheep as any).position?.y ?? sheep.y ?? 0) + 0.5;
              const isPenned = sheep.penned;
              return (
                <g key={`sh-${sheep.index ?? idx}`} transform={`translate(${sx}, ${sy})`}>
                  <circle
                    r={sheepRadius}
                    fill={isPenned ? "#4ade80" : sheep.color ?? "#f8fafc"}
                    stroke={isPenned ? "#86efac" : "#94a3b8"}
                    strokeWidth={penStroke}
                  />
                  <text
                    textAnchor="middle"
                    dominantBaseline="central"
                    fill="#0f172a"
                    fontSize={0.4 * densityScale}
                    fontWeight={800}
                  >
                    S
                  </text>
                </g>
              );
            })}

            {/* Dogs */}
            {snapshot.dogs?.map((dog, idx) => {
              const dx = ((dog as any).position?.x ?? dog.x ?? 0) + 0.5;
              const dy = ((dog as any).position?.y ?? dog.y ?? 0) + 0.5;
              const dColor = dogColor(dog.index ?? idx);
              return (
                <g key={`dg-${dog.index ?? idx}`} transform={`translate(${dx}, ${dy})`}>
                  <circle
                    r={dogRadius}
                    fill={dColor}
                    stroke="rgba(255, 255, 255, 0.85)"
                    strokeWidth={penStroke}
                  />
                  <text
                    textAnchor="middle"
                    dominantBaseline="central"
                    fill="#ffffff"
                    fontSize={0.42 * densityScale}
                    fontWeight={800}
                  >
                    {`D${(dog.index ?? idx) + 1}`}
                  </text>
                </g>
              );
            })}
          </svg>
        ) : (
          <div style={{ color: "#64748b", fontSize: "0.72rem" }}>No frames</div>
        )}
      </div>

      <div className="eval-seed-card__footer">
        <div className="eval-seed-card__progress-bar">
          <div
            className={`eval-seed-card__progress-fill ${
              isPass ? "eval-seed-card__progress-fill--pass" : "eval-seed-card__progress-fill--fail"
            }`}
            style={{ width: `${Math.round(progressRatio * 100)}%` }}
          />
        </div>
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            fontSize: "0.68rem",
          }}
        >
          <span style={{ color: "#94a3b8" }}>
            Reward: <strong style={{ color: "#cbd5e1" }}>{record.reward_total.toFixed(0)}</strong>
          </span>
          <span style={{ color: isPass ? "#34d399" : "#f87171" }}>
            {record.stop_reason || (isPass ? "success" : "timeout")}
          </span>
          <button
            type="button"
            className="eval-seed-inspect-btn"
            onClick={(e) => {
              e.stopPropagation();
              onInspect();
            }}
            title={`Inspect Seed ${record.seed} in single-seed focus mode`}
          >
            🔍 Inspect
          </button>
        </div>
      </div>
    </div>
  );
}

export function EvaluationEpisodesTab({
  currentStage,
  popupOnly = false,
  isOpen,
  standaloneWindow = false,
  onClose,
}: EvaluationEpisodesTabProps) {
  const [evaluations, setEvaluations] = useState<EvaluationSummaryPayload[]>([]);
  const [selectedEvalIndex, setSelectedEvalIndex] = useState<number>(0);
  const [selectedSeed, setSelectedSeed] = useState<number | null>(null);
  const [outcomeFilter, setOutcomeFilter] = useState<"all" | "pass" | "fail">("all");
  const [viewMode, setViewMode] = useState<"grid10" | "single">("grid10");
  const [internalModalOpen, setInternalModalOpen] = useState<boolean>(false);
  const isModalOpen = isOpen !== undefined ? isOpen : internalModalOpen;
  const handleCloseModal = useCallback(() => {
    if (onClose) onClose();
    else setInternalModalOpen(false);
  }, [onClose]);

  const [isLoadingEvals, setIsLoadingEvals] = useState<boolean>(true);
  const [evalError, setEvalError] = useState<string | null>(null);

  // Playback state (synchronized for all 10 seeds and single seed)
  const [currentStep, setCurrentStep] = useState<number>(0);
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const [playbackSpeed, setPlaybackSpeed] = useState<number>(1);
  const [isLooping, setIsLooping] = useState<boolean>(false);

  // Close modal on Escape key
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape" && isModalOpen) {
        handleCloseModal();
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [isModalOpen, handleCloseModal]);

  // Multi-seed replay bundles & cache
  const [multiReplays, setMultiReplays] = useState<
    Record<number, { bundle: ReplayBundle | null; loading: boolean; error: string | null }>
  >({});
  const replayCacheRef = useRef<Map<string, ReplayBundle>>(new Map());

  // Single-seed replay fallback state
  const [isLoadingSingleReplay, setIsLoadingSingleReplay] = useState<boolean>(false);
  const [singleReplayError, setSingleReplayError] = useState<string | null>(null);

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

  // Filter episode records for the selected evaluation
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

  // Active record (for single seed inspector and focus)
  const activeRecord: EvaluationRecordPayload | null = useMemo(() => {
    if (!activeEval || !activeEval.records || selectedSeed === null) return null;
    return activeEval.records.find((r) => r.seed === selectedSeed) ?? activeEval.records[0] ?? null;
  }, [activeEval, selectedSeed]);

  const [isPinning, setIsPinning] = useState<boolean>(false);

  const handleTogglePin = useCallback(async () => {
    if (!activeEval) return;
    const evalId = activeEval.evaluation_id || `evaluation-checkpoint-${activeEval.checkpoint_episode}`;
    const nextPinned = !activeEval.pinned;
    setIsPinning(true);
    try {
      const res = await pinEvaluation(evalId, nextPinned);
      if (res && res.success) {
        setEvaluations((prev) =>
          prev.map((ev, idx) =>
            idx === selectedEvalIndex || ev.evaluation_id === evalId
              ? {
                  ...ev,
                  pinned: nextPinned,
                  retention_status: nextPinned ? "pinned" : ev.retention_status,
                }
              : ev
          )
        );
      }
    } finally {
      setIsPinning(false);
    }
  }, [activeEval, selectedEvalIndex]);

  // Load replays for all seeds of active evaluation in parallel batches
  const loadAllSeedReplays = useCallback(async (evalSummary: EvaluationSummaryPayload) => {
    if (!evalSummary || !evalSummary.records || evalSummary.records.length === 0) return;

    // Initialize multiReplays state with cached bundles or loading
    setMultiReplays((prev) => {
      const next: Record<
        number,
        { bundle: ReplayBundle | null; loading: boolean; error: string | null }
      > = {};
      for (const rec of evalSummary.records) {
        const cacheKey = `${evalSummary.checkpoint_episode}_${rec.seed}`;
        if (replayCacheRef.current.has(cacheKey)) {
          next[rec.seed] = {
            bundle: replayCacheRef.current.get(cacheKey)!,
            loading: false,
            error: null,
          };
        } else {
          next[rec.seed] = prev[rec.seed] ?? { bundle: null, loading: true, error: null };
        }
      }
      return next;
    });

    const recordsToFetch = evalSummary.records.filter((rec) => {
      const cacheKey = `${evalSummary.checkpoint_episode}_${rec.seed}`;
      return !replayCacheRef.current.has(cacheKey);
    });

    if (recordsToFetch.length === 0) return;

    // Load in concurrent batches of 3
    const batchSize = 3;
    for (let i = 0; i < recordsToFetch.length; i += batchSize) {
      const batch = recordsToFetch.slice(i, i + batchSize);
      await Promise.all(
        batch.map(async (record) => {
          try {
            let bundle: ReplayBundle | null = null;
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
                // fall back to live simulation
              }
            }

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
              const cacheKey = `${evalSummary.checkpoint_episode}_${record.seed}`;
              replayCacheRef.current.set(cacheKey, bundle);
              setMultiReplays((prev) => ({
                ...prev,
                [record.seed]: { bundle, loading: false, error: null },
              }));
            } else {
              setMultiReplays((prev) => ({
                ...prev,
                [record.seed]: { bundle: null, loading: false, error: "No replay frames available" },
              }));
            }
          } catch (err: any) {
            setMultiReplays((prev) => ({
              ...prev,
              [record.seed]: {
                bundle: null,
                loading: false,
                error: err?.message || "Failed to load replay",
              },
            }));
          }
        })
      );
    }
  }, []);

  // When active evaluation changes, trigger loading all seed replays
  useEffect(() => {
    if (activeEval) {
      setCurrentStep(0);
      setIsPlaying(false);
      loadAllSeedReplays(activeEval);
    }
  }, [activeEval, loadAllSeedReplays]);

  // Fallback single-seed replay loader (sync with multiReplays cache)
  useEffect(() => {
    if (!activeRecord || !activeEval) return;
    const cacheKey = `${activeEval.checkpoint_episode}_${activeRecord.seed}`;
    if (replayCacheRef.current.has(cacheKey)) {
      setIsLoadingSingleReplay(false);
      setSingleReplayError(null);
      return;
    }

    let isMounted = true;
    setIsLoadingSingleReplay(true);
    setSingleReplayError(null);

    runLiveReplay({
      seed: activeRecord.seed,
      checkpoint_episode: activeEval.checkpoint_episode,
    })
      .then((bundle) => {
        if (!isMounted) return;
        if (bundle) {
          replayCacheRef.current.set(cacheKey, bundle);
          setMultiReplays((prev) => ({
            ...prev,
            [activeRecord.seed]: { bundle, loading: false, error: null },
          }));
        }
      })
      .catch((err) => {
        if (!isMounted) return;
        setSingleReplayError(err?.message || "Failed to load episode replay");
      })
      .finally(() => {
        if (isMounted) setIsLoadingSingleReplay(false);
      });

    return () => {
      isMounted = false;
    };
  }, [activeRecord, activeEval]);

  // Compute master maximum steps across all seeds
  const maxSteps = useMemo(() => {
    if (!activeEval || !activeEval.records) return 0;
    let max = 0;
    for (const rec of activeEval.records) {
      const rep = multiReplays[rec.seed];
      if (rep?.bundle?.frames && rep.bundle.frames.length > 0) {
        max = Math.max(max, rep.bundle.frames.length - 1);
      } else if (rec.steps) {
        max = Math.max(max, rec.steps);
      }
    }
    return max;
  }, [activeEval, multiReplays]);

  // Master Synchronized Playback Loop
  const isPlayingRef = useRef(isPlaying);
  isPlayingRef.current = isPlaying;
  const currentStepRef = useRef(currentStep);
  currentStepRef.current = currentStep;
  const maxStepsRef = useRef(maxSteps);
  maxStepsRef.current = maxSteps;
  const isLoopingRef = useRef(isLooping);
  isLoopingRef.current = isLooping;

  useEffect(() => {
    if (!isPlaying) return;

    const intervalMs = Math.max(16, Math.floor(100 / playbackSpeed));
    const timer = setInterval(() => {
      if (!isPlayingRef.current) return;
      if (currentStepRef.current >= maxStepsRef.current) {
        if (isLoopingRef.current) {
          setCurrentStep(0);
        } else {
          setIsPlaying(false);
        }
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

  // For single-seed focus mode snapshot
  const activeSeedBundle = activeRecord ? multiReplays[activeRecord.seed]?.bundle ?? null : null;
  const singleFrames = useMemo(() => extractBundleFrames(activeSeedBundle), [activeSeedBundle]);
  const singleSnapshot: ReplaySnapshot | null =
    singleFrames[Math.min(currentStep, Math.max(0, singleFrames.length - 1))] ||
    (activeSeedBundle as any)?.initial_state ||
    activeSeedBundle?.final_snapshot ||
    null;

  // Loaded seeds count
  const totalSeedCount = activeEval?.records?.length ?? 10;
  const loadedCount = activeEval?.records
    ? activeEval.records.filter((r) => multiReplays[r.seed]?.bundle != null).length
    : 0;

  // Modal content renderer (shared by tab and popupOnly modes)
  const renderModalContent = () => (
    <div
      className="eval-modal-overlay"
      role="dialog"
      aria-modal="true"
      aria-label="10 Seeds Replay Popup"
      data-testid="eval-modal-popup"
      onClick={handleCloseModal}
    >
      <div
        style={{ width: "100%", height: "100%", display: "flex", flexDirection: "column" }}
        onClick={(e) => e.stopPropagation()}
      >
        {/* Modal Header */}
        <div className="eval-modal-header">
          <div style={{ display: "flex", alignItems: "center", gap: "0.75rem" }}>
            <span style={{ fontSize: "1.2rem" }}>🎯</span>
            <div>
              <h2
                style={{
                  margin: 0,
                  fontSize: "1.05rem",
                  color: "#f8fafc",
                  display: "flex",
                  alignItems: "center",
                  gap: "0.5rem",
                }}
              >
                <span>Evaluation Benchmark #{activeEval?.checkpoint_episode ?? "—"}</span>
                <span
                  className={`eval-pill ${
                    activeEval && activeEval.success_rate >= 0.5 ? "eval-pill--good" : "eval-pill--warn"
                  }`}
                  style={{ fontSize: "0.75rem" }}
                >
                  {Math.round((activeEval?.success_rate ?? 0) * 100)}% Pass
                </span>
              </h2>
              <span style={{ fontSize: "0.72rem", color: "#94a3b8" }}>
                All 10 Seeds Synchronized Replay · Stage {activeEval?.curriculum_stage ?? "—"} ·{" "}
                {activeEval?.records.filter((r) => r.success).length ?? 0}/10 Penned · Step {currentStep} /{" "}
                {maxSteps}
              </span>
            </div>
          </div>

          <div style={{ display: "flex", alignItems: "center", gap: "0.6rem" }}>
            <button
              type="button"
              className={`eval-pin-btn${activeEval?.pinned ? " eval-pin-btn--pinned" : ""}`}
              onClick={handleTogglePin}
              disabled={isPinning || !activeEval}
              title={
                activeEval?.pinned
                  ? "Evaluation replays pinned permanently (click to unpin)"
                  : "Pin this evaluation to keep its replays permanently"
              }
            >
              {isPinning ? "Saving..." : activeEval?.pinned ? "📌 Replays Pinned" : "📌 Pin Replays"}
            </button>
            <button
              type="button"
              className="eval-modal-close-btn"
              onClick={handleCloseModal}
              aria-label={standaloneWindow ? "Close window" : "Close popup"}
              title={standaloneWindow ? "Close window" : "Close popup (Esc)"}
            >
              {standaloneWindow ? "✕ Close Window" : "✕ Close"}
            </button>
          </div>
        </div>

        {/* 3x3 + 1 Grid strictly fitting in 1 screen, NO SCROLLING! */}
        <div className="eval-modal-grid-body">
          {filteredRecords.map((record) => {
            const rep = multiReplays[record.seed] ?? {
              bundle: null,
              loading: true,
              error: null,
            };
            return (
              <SeedMiniCard
                key={`modal-grid-seed-${record.seed}`}
                record={record}
                bundle={rep.bundle}
                loading={rep.loading}
                error={rep.error}
                currentStep={currentStep}
                isSelected={selectedSeed === record.seed}
                onSelect={() => setSelectedSeed(record.seed)}
                onInspect={() => {
                  setSelectedSeed(record.seed);
                  handleCloseModal();
                }}
              />
            );
          })}

          {/* Row 4 Column 2: Evaluation Benchmark Context Card */}
          <div className="eval-context-card" data-testid="eval-context-card-modal">
            <div
              style={{
                display: "flex",
                justifyContent: "space-between",
                alignItems: "flex-start",
                gap: "0.5rem",
              }}
            >
              <div>
                <span
                  style={{
                    fontSize: "0.68rem",
                    textTransform: "uppercase",
                    letterSpacing: "0.06em",
                    color: "#38bdf8",
                    fontWeight: 700,
                  }}
                >
                  Evaluation Benchmark Context
                </span>
                <div style={{ display: "flex", alignItems: "center", gap: "0.4rem", marginTop: "0.15rem" }}>
                  <h4 style={{ margin: 0, fontSize: "0.95rem", color: "#f8fafc" }}>
                    Evaluation #{activeEval?.checkpoint_episode ?? "—"}
                  </h4>
                  {activeEval?.pinned ? (
                    <span className="eval-retention-tag eval-retention-tag--pinned">📌 Pinned</span>
                  ) : activeEval?.is_milestone ? (
                    <span className="eval-retention-tag eval-retention-tag--milestone">⭐ Milestone #{activeEval.evaluation_index}</span>
                  ) : activeEval?.is_first ? (
                    <span className="eval-retention-tag eval-retention-tag--first">🌱 Baseline (#1)</span>
                  ) : selectedEvalIndex === 0 ? (
                    <span className="eval-retention-tag eval-retention-tag--latest">⏱ Rolling Latest</span>
                  ) : null}
                </div>
              </div>
              <span
                className={`eval-pill ${
                  activeEval && activeEval.success_rate >= 0.5 ? "eval-pill--good" : "eval-pill--warn"
                }`}
                style={{ fontSize: "0.74rem" }}
              >
                {Math.round((activeEval?.success_rate ?? 0) * 100)}% Pass Rate
              </span>
            </div>

            <div
              style={{
                display: "grid",
                gridTemplateColumns: "repeat(2, 1fr)",
                gap: "0.25rem 0.5rem",
                fontSize: "0.7rem",
              }}
            >
              <div className="eval-context-metric">
                <span style={{ color: "#64748b" }}>Stage:</span>
                <strong style={{ color: "#f8fafc" }}>Stage {activeEval?.curriculum_stage ?? "—"}</strong>
              </div>
              <div className="eval-context-metric">
                <span style={{ color: "#64748b" }}>Penned:</span>
                <strong style={{ color: "#34d399" }}>
                  {activeEval?.records.filter((r) => r.success).length ?? 0} /{" "}
                  {activeEval?.records.length ?? 10} Seeds
                </strong>
              </div>
              <div className="eval-context-metric">
                <span style={{ color: "#64748b" }}>Avg Steps:</span>
                <strong style={{ color: "#f8fafc" }}>
                  {Math.round(activeEval?.average_completion_steps ?? 0)}
                </strong>
              </div>
              <div className="eval-context-metric">
                <span style={{ color: "#64748b" }}>Timeouts:</span>
                <strong
                  style={{ color: (activeEval?.timeout_rate ?? 0) > 0 ? "#f87171" : "#94a3b8" }}
                >
                  {activeEval?.records.filter((r) => r.timeout).length ?? 0}
                </strong>
              </div>
              <div className="eval-context-metric">
                <span style={{ color: "#64748b" }}>Avg Sheep:</span>
                <strong style={{ color: "#f8fafc" }}>
                  {activeEval?.average_sheep_penned.toFixed(1) ?? "—"}
                </strong>
              </div>
              <div className="eval-context-metric">
                <span style={{ color: "#64748b" }}>Avg Return:</span>
                <strong
                  style={{ color: (activeEval?.average_reward ?? 0) >= 0 ? "#38bdf8" : "#f87171" }}
                >
                  {activeEval?.average_reward.toFixed(1) ?? "—"}
                </strong>
              </div>
            </div>

            <div
              style={{
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
                marginTop: "auto",
                fontSize: "0.68rem",
                color: "#94a3b8",
              }}
            >
              <span>Simulation Status:</span>
              <span
                style={{
                  color: loadedCount === totalSeedCount ? "#34d399" : "#38bdf8",
                  fontWeight: 600,
                }}
              >
                {loadedCount === totalSeedCount
                  ? "✓ All 10 Seeds Ready"
                  : `Loading ${loadedCount} / ${totalSeedCount}...`}
              </span>
            </div>
          </div>

          {/* Row 4 Column 3: Master Playback Controller Card */}
          <div className="eval-master-card" data-testid="eval-master-card-modal">
            <div
              style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}
            >
              <div style={{ display: "flex", alignItems: "center", gap: "0.4rem" }}>
                <span style={{ fontSize: "1.05rem" }}>🎬</span>
                <strong style={{ fontSize: "0.92rem", color: "#f8fafc" }}>
                  Master Playback (All 10)
                </strong>
              </div>
              <span style={{ fontSize: "0.75rem", color: "#38bdf8", fontWeight: 700 }}>
                Step {currentStep} / {maxSteps}
              </span>
            </div>

            <div className="eval-scrubber-row" style={{ margin: "0.1rem 0" }}>
              <input
                type="range"
                min={0}
                max={maxSteps}
                value={currentStep}
                onChange={handleScrubberChange}
                className="eval-scrubber-slider"
                aria-label="Modal Master Replay timeline scrubber"
                disabled={maxSteps === 0}
              />
            </div>

            <div
              style={{
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
                flexWrap: "wrap",
                gap: "0.4rem",
              }}
            >
              <div style={{ display: "flex", gap: "0.25rem", alignItems: "center" }}>
                <button
                  type="button"
                  className="eval-ctrl-btn"
                  onClick={handleResetToStart}
                  disabled={currentStep === 0}
                  title="Reset"
                >
                  ⏮
                </button>
                <button
                  type="button"
                  className="eval-ctrl-btn"
                  onClick={handleStepBack}
                  disabled={currentStep === 0}
                  title="Step back"
                >
                  ⏪
                </button>
                <button
                  type="button"
                  className={`eval-ctrl-btn eval-ctrl-btn--primary${
                    isPlaying ? " eval-ctrl-btn--playing" : ""
                  }`}
                  onClick={handleTogglePlay}
                  disabled={maxSteps === 0}
                  style={{ padding: "0.3rem 0.75rem", fontSize: "0.82rem" }}
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

              <label
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: "0.3rem",
                  fontSize: "0.7rem",
                  color: "#94a3b8",
                  cursor: "pointer",
                }}
              >
                <input
                  type="checkbox"
                  checked={isLooping}
                  onChange={(e) => setIsLooping(e.target.checked)}
                />
                Auto-Loop
              </label>
            </div>

            <div
              style={{
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
                paddingTop: "0.2rem",
                borderTop: "1px solid rgba(255, 255, 255, 0.06)",
              }}
            >
              <span style={{ fontSize: "0.7rem", color: "#94a3b8" }}>Speed:</span>
              <div style={{ display: "flex", gap: "0.25rem" }}>
                {[0.5, 1, 2, 5, 10].map((spd) => (
                  <button
                    key={`modal-speed-${spd}`}
                    type="button"
                    className={`eval-speed-btn${playbackSpeed === spd ? " eval-speed-btn--active" : ""}`}
                    onClick={() => setPlaybackSpeed(spd)}
                  >
                    {spd}x
                  </button>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );

  // If in popup-only mode, only render the modal
  if (popupOnly) {
    if (!isModalOpen) return null;
    return renderModalContent();
  }

  return (
    <div className="eval-episodes-tab" data-testid="evaluation-episodes-tab">
      {/* ── Top Bar: Header, Mode Toggle, and Refresh ── */}
      <div className="eval-episodes-header">
        <div className="eval-episodes-header__info">
          <h3
            style={{
              margin: 0,
              fontSize: "1.1rem",
              color: "#f8fafc",
              display: "flex",
              alignItems: "center",
              gap: "0.5rem",
            }}
          >
            <span>🎯</span> Formal Evaluation Benchmark Inspector
          </h3>
          <p style={{ margin: "0.2rem 0 0", fontSize: "0.8rem", color: "#94a3b8" }}>
            Inspect formal evaluations with all 10 deterministic benchmark seeds playing their animations
            simultaneously.
          </p>
        </div>

        <div style={{ display: "flex", alignItems: "center", gap: "0.6rem" }}>
          {/* View Mode Toggle */}
          <div className="eval-view-mode-tabs" role="group" aria-label="View Mode">
            <button
              type="button"
              className={`eval-view-mode-btn${viewMode === "grid10" ? " eval-view-mode-btn--active" : ""}`}
              onClick={() => setViewMode("grid10")}
              title="Watch all 10 seeds simultaneously in 3x3 + 1 grid"
            >
              ⊞ All 10 Seeds Grid
            </button>
            <button
              type="button"
              className={`eval-view-mode-btn${viewMode === "single" ? " eval-view-mode-btn--active" : ""}`}
              onClick={() => setViewMode("single")}
              title="Inspect a single seed episode in detail"
            >
              ⧉ Single Seed Focus
            </button>
          </div>

          <button
            type="button"
            className="eval-hero-popup-btn"
            onClick={() => setInternalModalOpen(true)}
            title="Pop up all 10 seeds in full screen on one screen"
          >
            🎬 Pop Up 10 Seeds Grid
          </button>

          <button
            type="button"
            className={`eval-pin-btn${activeEval?.pinned ? " eval-pin-btn--pinned" : ""}`}
            onClick={handleTogglePin}
            disabled={isPinning || !activeEval}
            title={
              activeEval?.pinned
                ? "Evaluation replays pinned permanently (click to unpin)"
                : "Pin this evaluation to keep its replays permanently"
            }
          >
            {isPinning ? "Saving..." : activeEval?.pinned ? "📌 Replays Pinned" : "📌 Pin Replays"}
          </button>

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
          No formal evaluation benchmarks found for Stage {currentStage ?? "all"}. Run training evaluation
          checkpoints to generate formal benchmark episodes.
        </div>
      )}

      {evaluations.length > 0 && (
        <>
          {/* ── Evaluation Benchmark Timeline Selector (Last 5 Evaluations) ── */}
          <div className="eval-timeline-selector" role="group" aria-label="Evaluation Runs">
            <span
              style={{
                fontSize: "0.78rem",
                fontWeight: 600,
                color: "#94a3b8",
                alignSelf: "center",
                marginRight: "0.25rem",
              }}
            >
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
                    <span>
                      {passCount}/{totalCount} Penned
                    </span>
                    {ev.pinned ? (
                      <span className="eval-retention-tag eval-retention-tag--pinned">📌 Pinned</span>
                    ) : ev.is_milestone ? (
                      <span className="eval-retention-tag eval-retention-tag--milestone">⭐ M#{ev.evaluation_index}</span>
                    ) : ev.is_first ? (
                      <span className="eval-retention-tag eval-retention-tag--first">🌱 Baseline</span>
                    ) : idx === 0 ? (
                      <span className="eval-retention-tag eval-retention-tag--latest">⏱ Latest</span>
                    ) : null}
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
                <strong
                  className="eval-kpi-val"
                  style={{ color: activeEval.success_rate >= 0.5 ? "#34d399" : "#f87171" }}
                >
                  {Math.round(activeEval.success_rate * 100)}%
                </strong>
                <span className="eval-kpi-sub">
                  {activeEval.records.filter((r) => r.success).length} of {activeEval.records.length} passed
                </span>
              </div>
              <div className="eval-kpi-item">
                <span className="eval-kpi-label">Timeout Rate</span>
                <strong
                  className="eval-kpi-val"
                  style={{ color: activeEval.timeout_rate > 0.4 ? "#f87171" : "#94a3b8" }}
                >
                  {Math.round(activeEval.timeout_rate * 100)}%
                </strong>
                <span className="eval-kpi-sub">
                  {activeEval.records.filter((r) => r.timeout).length} timed out
                </span>
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
                <strong
                  className="eval-kpi-val"
                  style={{ color: activeEval.average_reward >= 0 ? "#38bdf8" : "#f87171" }}
                >
                  {activeEval.average_reward.toFixed(1)}
                </strong>
                <span className="eval-kpi-sub">cumulative return</span>
              </div>
            </div>
          )}
          {/* Pop-up launcher banner */}
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
              background: "linear-gradient(90deg, rgba(30, 58, 138, 0.45), rgba(15, 23, 42, 0.75))",
              border: "1px solid rgba(59, 130, 246, 0.35)",
              borderRadius: "8px",
              padding: "0.6rem 0.9rem",
              marginTop: "0.5rem",
              marginBottom: "0.25rem",
            }}
          >
            <div>
              <strong
                style={{
                  color: "#f8fafc",
                  fontSize: "0.92rem",
                  display: "flex",
                  alignItems: "center",
                  gap: "0.45rem",
                }}
              >
                <span>🎬</span> Watch All 10 Evaluation Seeds in One Screen (Pop-up)
              </strong>
              <p style={{ margin: "0.2rem 0 0", fontSize: "0.75rem", color: "#94a3b8" }}>
                Displays all 10 seeds playing simultaneously on one screen with no scrolling. Click <strong>✕ Close</strong> when done.
              </p>
            </div>
            <button
              type="button"
              className="eval-hero-popup-btn"
              onClick={() => setInternalModalOpen(true)}
              title="Open pop-up with all 10 seeds visible at once"
            >
              ⊞ Open 10-Seed Pop-up ↗
            </button>
          </div>
          {/* VIEW 1: 10-SEED SYNCHRONIZED GRID VIEW (3x3 + 1)                */}
          {/* ═════════════════════════════════════════════════════════════════ */}
          {viewMode === "grid10" && (
            <div style={{ display: "flex", flexDirection: "column", gap: "0.6rem" }}>
              {/* Outcome filter bar with active replay title */}
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                <span style={{ fontSize: "0.85rem", fontWeight: 600, color: "#f8fafc" }}>
                  {`Episode Replay: Checkpoint #${activeEval?.checkpoint_episode ?? "—"} · Seed ${activeRecord?.seed ?? "—"}`}
                </span>
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

              {/* 3x3 + 1 Grid Layout */}
              <div className="eval-multi-grid" role="region" aria-label="10 Seeds Simulation Grid">
                {/* 1. Seed Cards (Seeds 1 to 9, and Seed 10 on Row 4) */}
                {filteredRecords.map((record) => {
                  const rep = multiReplays[record.seed] ?? {
                    bundle: null,
                    loading: true,
                    error: null,
                  };
                  return (
                    <SeedMiniCard
                      key={`grid-seed-${record.seed}`}
                      record={record}
                      bundle={rep.bundle}
                      loading={rep.loading}
                      error={rep.error}
                      currentStep={currentStep}
                      isSelected={selectedSeed === record.seed}
                      onSelect={() => setSelectedSeed(record.seed)}
                      onInspect={() => {
                        setSelectedSeed(record.seed);
                        setViewMode("single");
                      }}
                    />
                  );
                })}

                {/* 2. Empty Area 1 (Row 4, Column 2): Evaluation Benchmark Context Card */}
                <div className="eval-context-card" data-testid="eval-context-card">
                  <div
                    style={{
                      display: "flex",
                      justifyContent: "space-between",
                      alignItems: "flex-start",
                      gap: "0.5rem",
                    }}
                  >
                    <div>
                      <span
                        style={{
                          fontSize: "0.68rem",
                          textTransform: "uppercase",
                          letterSpacing: "0.06em",
                          color: "#38bdf8",
                          fontWeight: 700,
                        }}
                      >
                        Evaluation Benchmark Context
                      </span>
                      <h4 style={{ margin: "0.15rem 0 0", fontSize: "0.95rem", color: "#f8fafc" }}>
                        Evaluation #{activeEval?.checkpoint_episode ?? "—"}
                      </h4>
                    </div>
                    <span
                      className={`eval-pill ${
                        activeEval && activeEval.success_rate >= 0.5 ? "eval-pill--good" : "eval-pill--warn"
                      }`}
                      style={{ fontSize: "0.74rem" }}
                    >
                      {Math.round((activeEval?.success_rate ?? 0) * 100)}% Pass Rate
                    </span>
                  </div>

                  {/* Benchmark Context KPIs */}
                  <div
                    style={{
                      display: "grid",
                      gridTemplateColumns: "repeat(2, 1fr)",
                      gap: "0.35rem 0.55rem",
                      fontSize: "0.72rem",
                    }}
                  >
                    <div className="eval-context-metric">
                      <span style={{ color: "#64748b" }}>Stage:</span>
                      <strong style={{ color: "#f8fafc" }}>Stage {activeEval?.curriculum_stage ?? "—"}</strong>
                    </div>
                    <div className="eval-context-metric">
                      <span style={{ color: "#64748b" }}>Penned:</span>
                      <strong style={{ color: "#34d399" }}>
                        {activeEval?.records.filter((r) => r.success).length ?? 0} /{" "}
                        {activeEval?.records.length ?? 10} Seeds
                      </strong>
                    </div>
                    <div className="eval-context-metric">
                      <span style={{ color: "#64748b" }}>Avg Steps:</span>
                      <strong style={{ color: "#f8fafc" }}>
                        {Math.round(activeEval?.average_completion_steps ?? 0)}
                      </strong>
                    </div>
                    <div className="eval-context-metric">
                      <span style={{ color: "#64748b" }}>Timeouts:</span>
                      <strong style={{ color: (activeEval?.timeout_rate ?? 0) > 0 ? "#f87171" : "#94a3b8" }}>
                        {activeEval?.records.filter((r) => r.timeout).length ?? 0}
                      </strong>
                    </div>
                    <div className="eval-context-metric">
                      <span style={{ color: "#64748b" }}>Avg Sheep:</span>
                      <strong style={{ color: "#f8fafc" }}>
                        {activeEval?.average_sheep_penned.toFixed(1) ?? "—"}
                      </strong>
                    </div>
                    <div className="eval-context-metric">
                      <span style={{ color: "#64748b" }}>Avg Return:</span>
                      <strong
                        style={{ color: (activeEval?.average_reward ?? 0) >= 0 ? "#38bdf8" : "#f87171" }}
                      >
                        {activeEval?.average_reward.toFixed(1) ?? "—"}
                      </strong>
                    </div>
                  </div>

                  {/* Switch Eval & Loading Status */}
                  <div
                    style={{
                      display: "flex",
                      flexDirection: "column",
                      gap: "0.35rem",
                      marginTop: "auto",
                    }}
                  >
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                      <label
                        htmlFor="eval-context-select"
                        style={{ fontSize: "0.68rem", color: "#94a3b8", fontWeight: 600 }}
                      >
                        Switch Eval:
                      </label>
                      <select
                        id="eval-context-select"
                        className="view-filter__select"
                        style={{ padding: "0.15rem 0.4rem", fontSize: "0.72rem", width: "170px" }}
                        value={selectedEvalIndex}
                        onChange={(e) => {
                          const idx = Number(e.target.value);
                          setSelectedEvalIndex(idx);
                          if (evaluations[idx]?.records?.length) {
                            setSelectedSeed(evaluations[idx].records[0].seed);
                          }
                        }}
                      >
                        {evaluations.map((ev, idx) => (
                          <option key={ev.evaluation_id || `opt-${idx}`} value={idx}>
                            Checkpoint #{ev.checkpoint_episode} ({Math.round(ev.success_rate * 100)}% Pass)
                          </option>
                        ))}
                      </select>
                    </div>

                    <div
                      style={{
                        display: "flex",
                        justifyContent: "space-between",
                        alignItems: "center",
                        fontSize: "0.68rem",
                        color: "#94a3b8",
                      }}
                    >
                      <span>Simulation Status:</span>
                      <span
                        style={{
                          color: loadedCount === totalSeedCount ? "#34d399" : "#38bdf8",
                          fontWeight: 600,
                        }}
                      >
                        {loadedCount === totalSeedCount
                          ? "✓ All 10 Seeds Ready"
                          : `Loading ${loadedCount} / ${totalSeedCount}...`}
                      </span>
                    </div>
                  </div>
                </div>

                {/* 3. Empty Area 2 (Row 4, Column 3): Master Synchronized Playback Controller */}
                <div className="eval-master-card" data-testid="eval-master-card">
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                    <div style={{ display: "flex", alignItems: "center", gap: "0.4rem" }}>
                      <span style={{ fontSize: "1.05rem" }}>🎬</span>
                      <strong style={{ fontSize: "0.92rem", color: "#f8fafc" }}>
                        Master Playback (All 10)
                      </strong>
                    </div>
                    <span style={{ fontSize: "0.75rem", color: "#38bdf8", fontWeight: 700 }}>
                      Step {currentStep} / {maxSteps}
                    </span>
                  </div>

                  {/* Scrubber slider controlling all 10 seeds synchronously */}
                  <div className="eval-scrubber-row" style={{ margin: "0.15rem 0" }}>
                    <input
                      type="range"
                      min={0}
                      max={maxSteps}
                      value={currentStep}
                      onChange={handleScrubberChange}
                      className="eval-scrubber-slider"
                      aria-label="Master Replay timeline scrubber"
                      disabled={maxSteps === 0}
                    />
                  </div>

                  {/* Playback Controls Row */}
                  <div
                    style={{
                      display: "flex",
                      justifyContent: "space-between",
                      alignItems: "center",
                      flexWrap: "wrap",
                      gap: "0.4rem",
                    }}
                  >
                    <div style={{ display: "flex", gap: "0.25rem", alignItems: "center" }}>
                      <button
                        type="button"
                        className="eval-ctrl-btn"
                        onClick={handleResetToStart}
                        disabled={currentStep === 0}
                        title="Reset all seeds to start"
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
                        className={`eval-ctrl-btn eval-ctrl-btn--primary${
                          isPlaying ? " eval-ctrl-btn--playing" : ""
                        }`}
                        onClick={handleTogglePlay}
                        disabled={maxSteps === 0}
                        title={isPlaying ? "Pause all 10 animations" : "Play all 10 animations"}
                        style={{ padding: "0.3rem 0.75rem", fontSize: "0.82rem" }}
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
                        title="Jump all seeds to end"
                      >
                        ⏭
                      </button>
                    </div>

                    <label
                      style={{
                        display: "flex",
                        alignItems: "center",
                        gap: "0.3rem",
                        fontSize: "0.7rem",
                        color: "#94a3b8",
                        cursor: "pointer",
                      }}
                    >
                      <input
                        type="checkbox"
                        checked={isLooping}
                        onChange={(e) => setIsLooping(e.target.checked)}
                      />
                      Auto-Loop
                    </label>
                  </div>

                  {/* Playback Speed Row */}
                  <div
                    style={{
                      display: "flex",
                      justifyContent: "space-between",
                      alignItems: "center",
                      paddingTop: "0.25rem",
                      borderTop: "1px solid rgba(255, 255, 255, 0.06)",
                    }}
                  >
                    <span style={{ fontSize: "0.72rem", color: "#94a3b8" }}>Speed:</span>
                    <div style={{ display: "flex", gap: "0.25rem" }}>
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
                </div>
              </div>
            </div>
          )}

          {/* ═════════════════════════════════════════════════════════════════ */}
          {/* VIEW 2: SINGLE-SEED FOCUS VIEW (DEEP-DIVE INSPECTION)           */}
          {/* ═════════════════════════════════════════════════════════════════ */}
          {viewMode === "single" && (
            <div className="eval-episodes-content-grid">
              {/* Left Column: Seed Episodes List */}
              <div className="eval-episodes-list-col">
                <div className="eval-episodes-list-header">
                  <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
                    <strong style={{ fontSize: "0.9rem", color: "#f8fafc" }}>
                      Benchmark Episodes ({activeEval?.records.length ?? 0} Seeds)
                    </strong>
                  </div>

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
                        className={`eval-record-card${isSelected ? " eval-record-card--selected" : ""}${
                          isPass ? " eval-record-card--pass" : " eval-record-card--fail"
                        }`}
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
                            <span
                              className={`eval-status-badge ${
                                isPass ? "eval-status-badge--pass" : "eval-status-badge--fail"
                              }`}
                            >
                              {isPass ? "✓ PASS" : "✗ FAIL"}
                            </span>
                            <strong className="eval-record-seed">Seed {rec.seed}</strong>
                          </div>
                          <span className="eval-record-steps">{rec.steps} steps</span>
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
                          <div className="eval-record-card__selected-indicator">▶ Active Replay Target</div>
                        )}
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* Right Column: Single Seed Replay View */}
              <div className="eval-replay-col">
                <div className="eval-replay-header">
                  <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
                    <span style={{ fontSize: "1rem" }}>🎬</span>
                    <strong style={{ fontSize: "0.95rem", color: "#f8fafc" }}>
                      Episode Replay: Checkpoint #{activeEval?.checkpoint_episode ?? "—"} · Seed{" "}
                      {activeRecord?.seed ?? "—"}
                    </strong>
                    {activeRecord && (
                      <span
                        className={`eval-status-badge ${
                          activeRecord.success ? "eval-status-badge--pass" : "eval-status-badge--fail"
                        }`}
                      >
                        {activeRecord.success ? "PASS" : "FAIL"}
                      </span>
                    )}
                  </div>

                  {isLoadingSingleReplay && (
                    <span style={{ fontSize: "0.75rem", color: "#38bdf8", fontStyle: "italic" }}>
                      Generating / loading replay...
                    </span>
                  )}
                </div>

                {singleReplayError && (
                  <div className="warning-box warning-box--error" style={{ margin: "0.5rem 0" }}>
                    {singleReplayError}
                  </div>
                )}

                {/* Field Visualizer Canvas */}
                <div className="eval-field-wrap">
                  <FieldView snapshot={singleSnapshot} />
                </div>

                {/* Replay Controls Toolbar */}
                <div className="eval-replay-controls">
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
                    <span
                      style={{
                        fontSize: "0.78rem",
                        color: "#94a3b8",
                        minWidth: "55px",
                        textAlign: "right",
                      }}
                    >
                      Max {maxSteps}
                    </span>
                  </div>

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
                        className={`eval-ctrl-btn eval-ctrl-btn--primary${
                          isPlaying ? " eval-ctrl-btn--playing" : ""
                        }`}
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

                    <div style={{ display: "flex", alignItems: "center", gap: "0.25rem" }}>
                      <span style={{ fontSize: "0.75rem", color: "#94a3b8", marginRight: "0.2rem" }}>
                        Speed:
                      </span>
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

                  {activeRecord && (
                    <div className="eval-frame-diagnostics">
                      <span>
                        <strong>Result:</strong> {activeRecord.sheep_penned} sheep penned ·{" "}
                        {activeRecord.steps} total steps · Stop: {activeRecord.stop_reason}
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
          )}
      {/* Fullscreen Popup Modal */}
      {isModalOpen && renderModalContent()}
        </>
      )}
    </div>
  );
}

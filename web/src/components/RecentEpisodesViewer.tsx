import React, { useState, useEffect, useMemo, useCallback } from "react";
import type { EpisodeRecord, EpisodeOutcome, CapturePolicyConfig } from "../state/types";
import { episodeHistoryStore, formatEpisodeLabel, formatFailedEpisodeLabel, convertTrainingEpisodeToRecord } from "../lib/episodeHistoryStore";
import {
  loadTrainingEpisodes,
  loadFailedEpisodes,
  fetchReplayById,
  fetchCapturePolicy,
  updateCapturePolicy,
  reproduceEpisode,
} from "../lib/api";
import { FieldView } from "./FieldView";

export function RecentEpisodesViewer() {
  const [episodes, setEpisodes] = useState<EpisodeRecord[]>(() => episodeHistoryStore.getEpisodes());
  const [selectedEpisodeId, setSelectedEpisodeId] = useState<string | number>("");
  const [currentStep, setCurrentStep] = useState<number>(0);
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const [playbackSpeed, setPlaybackSpeed] = useState<number>(1);
  const [stageFilter, setStageFilter] = useState<string>("all");
  const [outcomeFilter, setOutcomeFilter] = useState<string>("all");
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  // Diagnostic info state
  const [diagnosticInfo, setDiagnosticInfo] = useState<{
    apiCount: number;
    receivedCount: number;
    renderedCount: number;
    rejections: { epId: string | number; reason: string }[];
  }>({ apiCount: 0, receivedCount: 0, renderedCount: 0, rejections: [] });

  // Diagnostic capture policy state
  const [capturePolicy, setCapturePolicy] = useState<CapturePolicyConfig | null>(null);
  const [nextNCount, setNextNCount] = useState<number>(10);
  const [isReproducing, setIsReproducing] = useState<boolean>(false);

  const refreshEpisodes = useCallback(async () => {
    setEpisodes((prev) => {
      if (prev.length === 0) {
        setIsLoading(true);
      }
      return prev;
    });
    setErrorMessage(null);
    try {
      const rawList = await loadFailedEpisodes(25);
      if (rawList && Array.isArray(rawList)) {
        const apiCount = rawList.length;
        const receivedCount = rawList.length;
        const rejections: { epId: string | number; reason: string }[] = [];

        if (rawList.length > 0) {
          episodeHistoryStore.syncWithApiEpisodes(rawList);
        }

        setEpisodes((prevEpisodes) => {
          const storeEpisodes = episodeHistoryStore.getEpisodes();
          const existingMap = new Map(prevEpisodes.map((ep) => [String(ep.episode_id), ep]));
          const validRecords: EpisodeRecord[] = [];

          for (const ep of storeEpisodes) {
            if (ep.outcome === "win") {
              rejections.push({ epId: ep.episode_id, reason: "Episode was successful (win)" });
              continue;
            }
            const existing = existingMap.get(String(ep.episode_id));
            if (existing && existing.move_history && existing.move_history.length > 0) {
              validRecords.push({
                ...ep,
                initial_state: existing.initial_state || ep.initial_state,
                move_history: existing.move_history,
                total_moves: existing.total_moves,
              });
            } else {
              validRecords.push(ep);
            }
          }

          setDiagnosticInfo({
            apiCount,
            receivedCount,
            renderedCount: validRecords.length,
            rejections,
          });

          if (validRecords.length > 0) {
            setSelectedEpisodeId((prev) => {
              const exists = validRecords.some((item) => String(item.episode_id) === String(prev));
              return exists ? prev : validRecords[0].episode_id;
            });
          } else {
            setSelectedEpisodeId("");
          }

          return validRecords;
        });
      } else {
        const currentStore = episodeHistoryStore.getEpisodes();
        if (currentStore.length > 0) {
          setEpisodes(currentStore);
        } else if (rawList === null) {
          setErrorMessage("Failed to load recent training episodes.");
        }
      }
    } catch {
      // Ignore unhandled background poll errors
    } finally {
      setIsLoading(false);
    }
  }, []);

  const refreshPolicy = useCallback(async () => {
    const pol = await fetchCapturePolicy();
    if (pol) {
      setCapturePolicy(pol);
    }
  }, []);

  useEffect(() => {
    const initialList = episodeHistoryStore.getEpisodes();
    setEpisodes(initialList);
    if (initialList.length > 0 && !selectedEpisodeId) {
      setSelectedEpisodeId(initialList[0].episode_id);
    }
    void refreshEpisodes();
    void refreshPolicy();

    const timer = setInterval(() => {
      void refreshEpisodes();
      void refreshPolicy();
    }, 4000);
    return () => clearInterval(timer);
  }, [refreshEpisodes, refreshPolicy]);

  // Filtered episodes list (consumes backend result directly)
  const filteredEpisodes = episodes;

  // Currently selected episode object
  const currentEpisode = useMemo(() => {
    if (!episodes || episodes.length === 0) return null;
    return episodes.find((ep) => String(ep.episode_id) === String(selectedEpisodeId)) || episodes[0] || null;
  }, [episodes, selectedEpisodeId]);

  // Reset step to 0 and pause when selected episode changes
  useEffect(() => {
    setCurrentStep(0);
    setIsPlaying(false);
  }, [selectedEpisodeId]);

  // Fetch authentic replay frames from backend if needed
  useEffect(() => {
    if (
      currentEpisode &&
      currentEpisode.replayAvailable &&
      currentEpisode.replay_id &&
      (!currentEpisode.move_history || currentEpisode.move_history.length === 0)
    ) {
      void fetchReplayById(currentEpisode.replay_id).then((bundle) => {
        if (bundle && bundle.frames) {
          const initialFrame = bundle.frames[0]?.snapshot || bundle.final_snapshot;
          const updatedEp: EpisodeRecord = {
            ...currentEpisode,
            initial_state: initialFrame,
            move_history: bundle.frames,
            total_moves: bundle.frames.length > 0 ? bundle.frames.length - 1 : currentEpisode.total_moves,
          };
          episodeHistoryStore.updateEpisode(updatedEp);
          setEpisodes((prev) =>
            prev.map((ep) => (String(ep.episode_id) === String(updatedEp.episode_id) ? updatedEp : ep))
          );
        }
      });
    }
  }, [currentEpisode]);

  // Current frame snapshot to display
  const currentSnapshot = useMemo(() => {
    if (!currentEpisode || !currentEpisode.replayAvailable || !currentEpisode.move_history || currentEpisode.move_history.length === 0) {
      return null;
    }
    const frame = currentEpisode.move_history[Math.min(currentStep, currentEpisode.move_history.length - 1)];
    return frame ? frame.snapshot : currentEpisode.initial_state || null;
  }, [currentEpisode, currentStep]);

  // Playback timer loop for authentic replays
  useEffect(() => {
    if (!isPlaying || !currentEpisode || !currentEpisode.replayAvailable) return;

    const delayMs = playbackSpeed === 2 ? 150 : playbackSpeed === 0.5 ? 600 : 300;

    const timer = setInterval(() => {
      setCurrentStep((prevStep) => {
        const maxMoves = currentEpisode.total_moves;
        if (prevStep >= maxMoves) {
          setIsPlaying(false);
          return prevStep;
        }
        return prevStep + 1;
      });
    }, delayMs);

    return () => clearInterval(timer);
  }, [isPlaying, playbackSpeed, currentEpisode]);

  const handleSelectEpisode = (e: React.ChangeEvent<HTMLSelectElement>) => {
    setSelectedEpisodeId(e.target.value);
  };

  const handleTogglePlay = () => {
    if (!currentEpisode || !currentEpisode.replayAvailable) return;
    if (currentStep >= currentEpisode.total_moves) {
      setCurrentStep(0);
    }
    setIsPlaying((prev) => !prev);
  };

  const handleStepBackward = () => {
    if (!currentEpisode || !currentEpisode.replayAvailable) return;
    setIsPlaying(false);
    setCurrentStep((prev) => Math.max(0, prev - 1));
  };

  const handleStepForward = () => {
    if (!currentEpisode || !currentEpisode.replayAvailable) return;
    setIsPlaying(false);
    setCurrentStep((prev) => Math.min(currentEpisode.total_moves, prev + 1));
  };

  const handleResetToStart = () => {
    setIsPlaying(false);
    setCurrentStep(0);
  };

  const handleSliderChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setIsPlaying(false);
    setCurrentStep(Number(e.target.value));
  };

  const handleRecordNextN = async (failuresOnly: boolean) => {
    const res = await updateCapturePolicy({
      next_n: nextNCount,
      target_outcome: failuresOnly ? "failures" : "all",
    });
    if (res) {
      setCapturePolicy(res);
    }
  };

  const handleReproduceEpisode = async () => {
    if (!currentEpisode) return;
    setIsReproducing(true);
    try {
      const bundle = await reproduceEpisode(currentEpisode.episode_id);
      if (bundle && bundle.frames) {
        const initialFrame = bundle.frames[0]?.snapshot || bundle.final_snapshot;
        const updatedEp: EpisodeRecord = {
          ...currentEpisode,
          replayAvailable: true,
          replaySource: "reproduced",
          capture_reason: "reproduced",
          capture_status: "available",
          initial_state: initialFrame,
          move_history: bundle.frames,
          total_moves: bundle.frames.length > 0 ? bundle.frames.length - 1 : currentEpisode.total_moves,
        };
        episodeHistoryStore.updateEpisode(updatedEp);
        setEpisodes(episodeHistoryStore.getEpisodes());
      }
    } finally {
      setIsReproducing(false);
    }
  };

  const renderOutcomeBadge = (outcome: EpisodeOutcome, label: string) => {
    let bgColor = "#ef4444";
    let textColor = "#ffffff";

    if (outcome === "win") {
      bgColor = "#10b981";
    } else if (outcome === "timeout") {
      bgColor = "#f59e0b";
    }

    return (
      <span
        style={{
          display: "inline-block",
          padding: "0.15rem 0.5rem",
          borderRadius: "4px",
          backgroundColor: bgColor,
          color: textColor,
          fontWeight: "bold",
          fontSize: "0.75rem",
          textTransform: "uppercase",
          letterSpacing: "0.04em",
        }}
      >
        {label}
      </span>
    );
  };

  const getSourceBadgeLabel = (source?: string | null) => {
    if (source === "training-diagnostic") return "Training Diagnostic Replay";
    if (source === "checkpoint-evaluation") return "Authentic Evaluation Replay (checkpoint-evaluation)";
    if (source === "scenario-evaluation") return "Scenario Replay";
    if (source === "reproduced") return "Reproduced Replay (Rerun)";
    return "Authentic Replay";
  };

  const getReasonText = (ep: EpisodeRecord) => {
    if (ep.capture_reason === "timeout") return "Captured because episode timed out.";
    if (ep.capture_reason === "stopped") return "Captured because episode stopped without progress.";
    if (ep.capture_reason === "unsuccessful_terminal") return "Captured because episode failed.";
    if (ep.capture_reason === "next_n") return "Captured by diagnostic request.";
    if (ep.capture_reason === "sampled_success") return "Sampled successful episode.";
    if (ep.capture_reason === "reproduced") return "Episode rerun from recorded seed and configuration.";
    if (ep.capture_status === "queued") return "Replay queued for writing...";
    if (ep.capture_status === "writing") return "Replay writing in background...";
    if (ep.capture_status === "pruned") return "Replay file pruned due to retention storage limit.";
    return "High-speed training logs summary statistics rather than step-by-step trajectories.";
  };

  return (
    <div
      className="recent-episodes-viewer"
      style={{
        display: "flex",
        flexDirection: "column",
        gap: "1rem",
        padding: "1rem",
        backgroundColor: "var(--card-bg, #1e293b)",
        borderRadius: "10px",
        border: "1px solid var(--border-color, rgba(255, 255, 255, 0.1))",
        color: "var(--text-color, #f8fafc)",
        maxWidth: "100%",
        boxSizing: "border-box",
      }}
    >
      {/* Title & Diagnostic Controls Header */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", flexWrap: "wrap", gap: "0.75rem" }}>
        <div>
          <h2 style={{ margin: 0, fontSize: "1.15rem", fontWeight: 700, lineHeight: 1.2 }}>
            Recent Training Episodes
          </h2>
          <p style={{ margin: "0.2rem 0px 0px 0px", fontSize: "0.78rem", color: "var(--text-muted, #94a3b8)" }}>
            Selective trajectory recording captures authentic coordinates for failure diagnosis and requested episodes.
          </p>
        </div>

        {/* Diagnostic "Record Next Episodes" Widget */}
        <div
          style={{
            display: "flex",
            flexDirection: "column",
            gap: "0.3rem",
            padding: "0.5rem 0.75rem",
            borderRadius: "6px",
            backgroundColor: "#0f172a",
            border: "1px solid #334155",
            fontSize: "0.75rem",
          }}
        >
          <div style={{ fontWeight: 600, color: "#cbd5e1", display: "flex", justifyContent: "space-between", gap: "1rem" }}>
            <span>Record Next Episodes</span>
            {capturePolicy && (
              <span style={{ color: "#94a3b8", fontWeight: "normal" }}>
                Mode: <strong style={{ color: "#3b82f6" }}>{capturePolicy.mode}</strong> | Queued: {capturePolicy.queued_writes} | Written: {capturePolicy.written_count}
              </span>
            )}
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: "0.4rem" }}>
            <select
              value={nextNCount}
              onChange={(e) => setNextNCount(Number(e.target.value))}
              style={{
                padding: "0.2rem 0.4rem",
                borderRadius: "4px",
                backgroundColor: "#1e293b",
                color: "white",
                border: "1px solid #475569",
                fontSize: "0.75rem",
              }}
            >
              <option value={5}>5 Episodes</option>
              <option value={10}>10 Episodes</option>
              <option value={25}>25 Episodes</option>
              <option value={50}>50 Episodes</option>
            </select>
            <button
              type="button"
              onClick={() => void handleRecordNextN(true)}
              style={{
                padding: "0.2rem 0.6rem",
                borderRadius: "4px",
                border: "none",
                backgroundColor: "#f59e0b",
                color: "black",
                fontWeight: "bold",
                cursor: "pointer",
                fontSize: "0.75rem",
              }}
            >
              Record Failures Only
            </button>
            <button
              type="button"
              onClick={() => void handleRecordNextN(false)}
              style={{
                padding: "0.2rem 0.6rem",
                borderRadius: "4px",
                border: "none",
                backgroundColor: "#3b82f6",
                color: "white",
                fontWeight: "bold",
                cursor: "pointer",
                fontSize: "0.75rem",
              }}
            >
              Record All Outcomes
            </button>
          </div>
          {capturePolicy && capturePolicy.next_n_counter > 0 && (
            <div style={{ color: "#10b981", fontWeight: 600 }}>
              ▶ {capturePolicy.next_n_counter} requested captures remaining in rollout pipeline.
            </div>
          )}
        </div>
      </div>

      {/* Loading State */}
      {isLoading && episodes.length === 0 && (
        <div
          data-testid="loading-state"
          style={{
            padding: "1.5rem",
            textAlign: "center",
            backgroundColor: "#0f172a",
            borderRadius: "8px",
            color: "#94a3b8",
            fontSize: "0.9rem",
          }}
        >
          Loading recent training episodes…
        </div>
      )}

      {/* Error State */}
      {errorMessage && episodes.length === 0 && (
        <div
          data-testid="error-state"
          style={{
            padding: "1.5rem",
            textAlign: "center",
            backgroundColor: "#451a1a",
            border: "1px solid #ef4444",
            borderRadius: "8px",
            color: "#fca5a5",
            fontSize: "0.9rem",
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            gap: "0.75rem",
          }}
        >
          <span>{errorMessage}</span>
          <button
            type="button"
            onClick={() => void refreshEpisodes()}
            style={{
              padding: "0.4rem 1rem",
              borderRadius: "4px",
              border: "none",
              backgroundColor: "#ef4444",
              color: "white",
              fontWeight: "bold",
              cursor: "pointer",
            }}
          >
            Retry
          </button>
        </div>
      )}

      {/* Empty State */}
      {!isLoading && !errorMessage && filteredEpisodes.length === 0 && (
        <div
          data-testid="empty-state"
          style={{
            padding: "1.5rem",
            textAlign: "center",
            backgroundColor: "#0f172a",
            borderRadius: "8px",
            color: "#94a3b8",
            fontSize: "0.9rem",
          }}
        >
          No recorded training episodes are available.
        </div>
      )}

      {/* Populated State */}
      {episodes.length > 0 && (
        <div
          style={{
            display: "grid",
            gridTemplateColumns: currentEpisode?.replayAvailable ? "minmax(320px, 420px) 1fr" : "1fr",
            gap: "1rem",
            alignItems: "start",
          }}
        >
          {/* LEFT COLUMN: Filters, Dropdown & Episode Details */}
          <div style={{ display: "flex", flexDirection: "column", gap: "0.85rem" }}>
            {/* Filters */}
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0.5rem" }}>
              <div>
                <label
                  htmlFor="stage-filter"
                  style={{ display: "block", fontSize: "0.75rem", color: "var(--text-muted, #94a3b8)", marginBottom: "0.2rem" }}
                >
                  Filter Stage:
                </label>
                <select
                  id="stage-filter"
                  aria-label="Filter Stage:"
                  value={stageFilter}
                  onChange={(e) => setStageFilter(e.target.value)}
                  style={{
                    width: "100%",
                    padding: "0.3rem 0.5rem",
                    borderRadius: "5px",
                    border: "1px solid var(--border-color, #334155)",
                    backgroundColor: "#0f172a",
                    color: "white",
                    fontSize: "0.8rem",
                  }}
                >
                  <option value="all">All Stages</option>
                  <option value="8">Stage 8</option>
                  <option value="7">Stage 7</option>
                  <option value="9">Stage 9</option>
                  <option value="unknown">Unknown Stage</option>
                </select>
              </div>

              <div>
                <label
                  htmlFor="outcome-filter"
                  style={{ display: "block", fontSize: "0.75rem", color: "var(--text-muted, #94a3b8)", marginBottom: "0.2rem" }}
                >
                  Outcome:
                </label>
                <select
                  id="outcome-filter"
                  aria-label="Outcome:"
                  value={outcomeFilter}
                  onChange={(e) => setOutcomeFilter(e.target.value)}
                  style={{
                    width: "100%",
                    padding: "0.3rem 0.5rem",
                    borderRadius: "5px",
                    border: "1px solid var(--border-color, #334155)",
                    backgroundColor: "#0f172a",
                    color: "white",
                    fontSize: "0.8rem",
                  }}
                >
                  <option value="all">All Outcomes</option>
                  <option value="win">Wins</option>
                  <option value="loss">Losses</option>
                  <option value="timeout">Timeouts</option>
                </select>
              </div>
            </div>

            {/* Main Episode Dropdown */}
            <div>
              <label
                htmlFor="episode-dropdown-select"
                style={{ display: "block", fontSize: "0.8rem", fontWeight: 600, marginBottom: "0.25rem" }}
              >
                Select Failed Episode (Most Recent Top):
              </label>
              <select
                id="episode-dropdown-select"
                aria-label="Recent Episodes Selector"
                value={currentEpisode ? String(currentEpisode.episode_id) : ""}
                onChange={handleSelectEpisode}
                style={{
                  width: "100%",
                  padding: "0.45rem 0.6rem",
                  borderRadius: "6px",
                  border: "1px solid var(--theme-accent, #3b82f6)",
                  backgroundColor: "#0f172a",
                  color: "white",
                  fontSize: "0.85rem",
                  fontWeight: 500,
                  cursor: "pointer",
                }}
              >
                {episodes.length === 0 ? (
                  <option value="">No matching failed episodes with replays</option>
                ) : (
                  episodes.map((ep) => (
                    <option key={`ep-option-${ep.episode_id}`} value={String(ep.episode_id)}>
                      {formatFailedEpisodeLabel(ep)}
                    </option>
                  ))
                )}
              </select>

              {/* Development Diagnostic Banner (when fewer than 25 entries appear) */}
              {diagnosticInfo.renderedCount < 25 && !isLoading && (
                <div
                  data-testid="failed-episodes-diagnostic"
                  style={{
                    marginTop: "0.5rem",
                    padding: "0.6rem 0.8rem",
                    borderRadius: "6px",
                    backgroundColor: "rgba(30, 27, 75, 0.8)",
                    border: "1px solid #6366f1",
                    color: "#c7d2fe",
                    fontSize: "0.75rem",
                    display: "flex",
                    flexDirection: "column",
                    gap: "0.25rem",
                  }}
                >
                  <div style={{ fontWeight: 600, color: "#818cf8" }}>
                    Failed Episode Diagnostics ({diagnosticInfo.renderedCount} / 25 entries)
                  </div>
                  <div>• Number returned by API: {diagnosticInfo.apiCount}</div>
                  <div>• Number received by React: {diagnosticInfo.receivedCount}</div>
                  <div>• Number rendered in dropdown: {diagnosticInfo.renderedCount}</div>
                  <div>
                    • Frontend Rejections:{" "}
                    {diagnosticInfo.rejections.length === 0
                      ? "None"
                      : diagnosticInfo.rejections.map((r) => `Ep #${r.epId} (${r.reason})`).join(", ")}
                  </div>
                </div>
              )}
            </div>

            {/* Episode Summary Card */}
            {currentEpisode && (
              <div
                className="episode-summary-card"
                style={{
                  display: "grid",
                  gridTemplateColumns: "repeat(3, 1fr)",
                  gap: "0.5rem",
                  padding: "0.85rem",
                  borderRadius: "6px",
                  backgroundColor: "rgba(15, 23, 42, 0.7)",
                  border: "1px solid var(--border-color, #334155)",
                  fontSize: "0.8rem",
                }}
              >
                <div>
                  <span style={{ fontSize: "0.7rem", color: "var(--text-muted, #94a3b8)", textTransform: "uppercase" }}>
                    Episode ID
                  </span>
                  <strong style={{ display: "block", fontSize: "0.95rem" }}>
                    Episode #{currentEpisode.episode_id}
                  </strong>
                </div>

                <div>
                  <span style={{ fontSize: "0.7rem", color: "var(--text-muted, #94a3b8)", textTransform: "uppercase" }}>
                    Timestamp
                  </span>
                  <span style={{ display: "block", fontSize: "0.85rem" }}>
                    {currentEpisode.timestamp}
                  </span>
                </div>

                <div>
                  <span style={{ fontSize: "0.7rem", color: "var(--text-muted, #94a3b8)", textTransform: "uppercase" }}>
                    Result
                  </span>
                  <div style={{ marginTop: "0.1rem" }}>
                    {renderOutcomeBadge(currentEpisode.outcome, currentEpisode.outcome_label)}
                  </div>
                </div>

                <div>
                  <span style={{ fontSize: "0.7rem", color: "var(--text-muted, #94a3b8)", textTransform: "uppercase" }}>
                    Total Steps
                  </span>
                  <strong style={{ display: "block", fontSize: "0.9rem" }}>
                    {currentEpisode.total_moves.toLocaleString()}
                  </strong>
                </div>

                <div>
                  <span style={{ fontSize: "0.7rem", color: "var(--text-muted, #94a3b8)", textTransform: "uppercase" }}>
                    Curriculum Stage
                  </span>
                  <strong style={{ display: "block", fontSize: "0.9rem", color: currentEpisode.stage !== null ? "#3b82f6" : "#94a3b8" }}>
                    {currentEpisode.stage !== null ? `Stage ${currentEpisode.stage}` : "Unknown"}
                  </strong>
                </div>

                <div>
                  <span style={{ fontSize: "0.7rem", color: "var(--text-muted, #94a3b8)", textTransform: "uppercase" }}>
                    Penned Sheep
                  </span>
                  <strong style={{ display: "block", fontSize: "0.9rem" }}>
                    {currentEpisode.sheep_penned ?? 0} / {currentEpisode.total_sheep ?? 4}
                  </strong>
                </div>

                {currentEpisode.reward !== undefined && (
                  <div>
                    <span style={{ fontSize: "0.7rem", color: "var(--text-muted, #94a3b8)", textTransform: "uppercase" }}>
                      Reward
                    </span>
                    <strong style={{ display: "block", fontSize: "0.85rem", color: currentEpisode.reward >= 0 ? "#10b981" : "#ef4444" }}>
                      {currentEpisode.reward.toFixed(2)}
                    </strong>
                  </div>
                )}

                {currentEpisode.seed !== undefined && (
                  <div>
                    <span style={{ fontSize: "0.7rem", color: "var(--text-muted, #94a3b8)", textTransform: "uppercase" }}>
                      Seed
                    </span>
                    <span style={{ display: "block", fontSize: "0.85rem" }}>
                      {currentEpisode.seed}
                    </span>
                  </div>
                )}

                {currentEpisode.checkpoint_id && (
                  <div style={{ gridColumn: "span 3" }}>
                    <span style={{ fontSize: "0.7rem", color: "var(--text-muted, #94a3b8)", textTransform: "uppercase" }}>
                      Checkpoint ID
                    </span>
                    <span style={{ display: "block", fontSize: "0.75rem", fontFamily: "monospace", color: "#cbd5e1" }}>
                      {currentEpisode.checkpoint_id}
                    </span>
                  </div>
                )}
              </div>
            )}

            {/* Replay Area: Replay Not Recorded vs Authentic Playback */}
            {currentEpisode && !currentEpisode.replayAvailable && (
              <div
                className="no-replay-banner"
                data-testid="no-replay-banner"
                style={{
                  padding: "1rem",
                  borderRadius: "6px",
                  backgroundColor: "rgba(15, 23, 42, 0.5)",
                  border: "1px dashed var(--border-color, #334155)",
                  display: "flex",
                  flexDirection: "column",
                  gap: "0.5rem",
                }}
              >
                <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: "0.5rem", flexWrap: "wrap" }}>
                  <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
                    <span
                      style={{
                        padding: "0.2rem 0.5rem",
                        borderRadius: "4px",
                        backgroundColor: "#334155",
                        color: "#94a3b8",
                        fontSize: "0.75rem",
                        fontWeight: 600,
                      }}
                    >
                      Replay not recorded
                    </span>
                    <button
                      type="button"
                      disabled
                      style={{
                        padding: "0.2rem 0.5rem",
                        borderRadius: "4px",
                        border: "none",
                        backgroundColor: "#1e293b",
                        color: "#64748b",
                        fontSize: "0.75rem",
                        cursor: "not-allowed",
                      }}
                    >
                      ▶ Play Replay (Disabled)
                    </button>
                  </div>

                  <button
                    type="button"
                    onClick={() => void handleReproduceEpisode()}
                    disabled={isReproducing}
                    style={{
                      padding: "0.25rem 0.75rem",
                      borderRadius: "4px",
                      border: "1px solid #3b82f6",
                      backgroundColor: isReproducing ? "#1e293b" : "#1e40af",
                      color: "white",
                      fontSize: "0.75rem",
                      fontWeight: 600,
                      cursor: isReproducing ? "wait" : "pointer",
                    }}
                  >
                    {isReproducing ? "Reproducing..." : "🔄 Reproduce Episode"}
                  </button>
                </div>

                <p style={{ margin: 0, fontSize: "0.78rem", color: "#94a3b8", lineHeight: 1.4 }}>
                  {getReasonText(currentEpisode)}
                </p>
              </div>
            )}

            {/* Authentic Replay Controls */}
            {currentEpisode && currentEpisode.replayAvailable && (
              <div
                style={{
                  display: "flex",
                  flexDirection: "column",
                  gap: "0.6rem",
                  padding: "0.75rem",
                  borderRadius: "6px",
                  backgroundColor: "rgba(15, 23, 42, 0.5)",
                  border: "1px solid var(--border-color, #334155)",
                }}
              >
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                  <span style={{ fontSize: "0.75rem", color: "#10b981", fontWeight: 600 }}>
                    {getSourceBadgeLabel(currentEpisode.replaySource)}
                  </span>
                  {currentEpisode.capture_reason && (
                    <span style={{ fontSize: "0.7rem", color: "#94a3b8", fontStyle: "italic" }}>
                      {getReasonText(currentEpisode)}
                    </span>
                  )}
                </div>

                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: "0.4rem" }}>
                  <div style={{ display: "flex", gap: "0.3rem" }}>
                    <button
                      type="button"
                      onClick={handleResetToStart}
                      title="Reset to Step 0"
                      style={{
                        padding: "0.35rem 0.55rem",
                        borderRadius: "4px",
                        border: "1px solid #475569",
                        backgroundColor: "#1e293b",
                        color: "white",
                        fontSize: "0.75rem",
                        fontWeight: 600,
                        cursor: "pointer",
                      }}
                    >
                      ⏮ Step 0
                    </button>

                    <button
                      type="button"
                      onClick={handleStepBackward}
                      disabled={currentStep <= 0}
                      title="Step Backward"
                      style={{
                        padding: "0.35rem 0.55rem",
                        borderRadius: "4px",
                        border: "1px solid #475569",
                        backgroundColor: "#1e293b",
                        color: "white",
                        fontSize: "0.75rem",
                        fontWeight: 600,
                        cursor: currentStep <= 0 ? "not-allowed" : "pointer",
                        opacity: currentStep <= 0 ? 0.5 : 1,
                      }}
                    >
                      ◀
                    </button>

                    <button
                      type="button"
                      onClick={handleTogglePlay}
                      style={{
                        padding: "0.35rem 0.85rem",
                        borderRadius: "4px",
                        border: "none",
                        backgroundColor: isPlaying ? "#f59e0b" : "#10b981",
                        color: "white",
                        fontWeight: "bold",
                        fontSize: "0.85rem",
                        cursor: "pointer",
                        minWidth: "70px",
                      }}
                    >
                      {isPlaying ? "⏸ Pause" : "▶ Play"}
                    </button>

                    <button
                      type="button"
                      onClick={handleStepForward}
                      disabled={currentStep >= currentEpisode.total_moves}
                      title="Step Forward"
                      style={{
                        padding: "0.35rem 0.55rem",
                        borderRadius: "4px",
                        border: "1px solid #475569",
                        backgroundColor: "#1e293b",
                        color: "white",
                        fontSize: "0.75rem",
                        fontWeight: 600,
                        cursor: currentStep >= currentEpisode.total_moves ? "not-allowed" : "pointer",
                        opacity: currentStep >= currentEpisode.total_moves ? 0.5 : 1,
                      }}
                    >
                      ▶
                    </button>
                  </div>

                  <div style={{ display: "flex", gap: "0.25rem", alignItems: "center" }}>
                    {[
                      { speed: 1, label: "1x (300ms/step)" },
                      { speed: 2, label: "2x (150ms/step)" },
                    ].map(({ speed, label }) => (
                      <button
                        key={`speed-${speed}`}
                        type="button"
                        onClick={() => setPlaybackSpeed(speed)}
                        style={{
                          padding: "0.25rem 0.5rem",
                          borderRadius: "4px",
                          border: playbackSpeed === speed ? "1px solid #3b82f6" : "1px solid #475569",
                          backgroundColor: playbackSpeed === speed ? "#3b82f6" : "#0f172a",
                          color: "white",
                          fontSize: "0.75rem",
                          fontWeight: playbackSpeed === speed ? "bold" : "normal",
                          cursor: "pointer",
                        }}
                      >
                        {label}
                      </button>
                    ))}
                  </div>
                </div>

                <div style={{ display: "flex", alignItems: "center", gap: "0.6rem" }}>
                  <input
                    type="range"
                    min={0}
                    max={currentEpisode.total_moves}
                    value={currentStep}
                    onChange={handleSliderChange}
                    aria-label="Replay step timeline"
                    style={{
                      flex: 1,
                      accentColor: "#3b82f6",
                      cursor: "pointer",
                      height: "4px",
                    }}
                  />
                  <span style={{ fontSize: "0.8rem", fontWeight: 700, minWidth: "85px", textAlign: "right" }}>
                    Step {currentStep} / {currentEpisode.total_moves}
                  </span>
                </div>
              </div>
            )}
          </div>

          {/* RIGHT COLUMN: Field View (Only rendered for Authentic Replays) */}
          {currentEpisode && currentEpisode.replayAvailable && currentSnapshot && (
            <div
              style={{
                borderRadius: "8px",
                overflow: "hidden",
                border: "1px solid var(--border-color, #334155)",
                maxHeight: "480px",
                display: "flex",
                flexDirection: "column",
                backgroundColor: "#0f172a",
              }}
            >
              <FieldView snapshot={currentSnapshot} />
            </div>
          )}
        </div>
      )}
    </div>
  );
}

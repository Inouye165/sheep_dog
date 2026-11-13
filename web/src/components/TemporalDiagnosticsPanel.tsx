import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  checkBackendHealth,
  fetchReplayById,
  hasCachedTemporalDiagnosticAnalysis,
  loadTemporalDiagnosticAnalysis,
  loadTemporalDiagnosticExamples,
  type ApiError,
  type BackendHealthResult,
} from "../lib/api";
import type {
  ReplayBundle,
  ReplayFrame,
  ReplaySnapshot,
  StepDogEvidence,
  StepSheepEvidence,
  TemporalDiagnosticExample,
  TemporalDiagnosticPayload,
  TemporalInteractionInterval,
  TemporalStepEvidence,
} from "../state/types";
import {
  FieldView,
  type FieldOverlayConfig,
  type SelectedFieldEntity,
} from "./FieldView";
import { dogColor } from "./dogPalette";

export interface TemporalDiagnosticsPanelProps {
  initialReplayId?: string | null;
  activeReplay?: ReplayBundle | null;
  activeReplayId?: string | null;
  onSyncReplayToMainViewer?: (replay: ReplayBundle, worldStep: number) => void;
  backendOffline?: boolean;
  isReadOnlyPreview?: boolean;
  backendStarting?: boolean;
  onCachedReplayAvailableChange?: (available: boolean, replayId: string | null) => void;
  onRequestReconnect?: () => Promise<void> | void;
}

const CLASSIFICATION_META: Record<
  string,
  {
    readableLabel: string;
    shortBadge: string;
    icon: string;
    color: string;
    bg: string;
    border: string;
  }
> = {
  oscillation_loop: {
    readableLabel: "Repeated movement",
    shortBadge: "Repeated movement",
    icon: "⟳",
    color: "#fbbf24",
    bg: "rgba(245, 158, 11, 0.16)",
    border: "rgba(245, 158, 11, 0.55)",
  },
  stationary_hold: {
    readableLabel: "Stationary hold",
    shortBadge: "Stationary hold",
    icon: "⏸",
    color: "#94a3b8",
    bg: "rgba(148, 163, 184, 0.16)",
    border: "rgba(148, 163, 184, 0.5)",
  },
  productive_repositioning: {
    readableLabel: "Productive repositioning",
    shortBadge: "Productive repositioning",
    icon: "↗",
    color: "#4ade80",
    bg: "rgba(74, 222, 128, 0.16)",
    border: "rgba(74, 222, 128, 0.55)",
  },
  counterproductive_pressure: {
    readableLabel: "Counterproductive pressure candidate",
    shortBadge: "Counterproductive pressure",
    icon: "⚠",
    color: "#f87171",
    bg: "rgba(248, 113, 113, 0.16)",
    border: "rgba(248, 113, 113, 0.55)",
  },
  unresolved_ambiguity: {
    readableLabel: "Unresolved pattern",
    shortBadge: "Unresolved pattern",
    icon: "?",
    color: "#c084fc",
    bg: "rgba(192, 132, 252, 0.16)",
    border: "rgba(192, 132, 252, 0.55)",
  },
};

function getClassificationMeta(classification: string, readableLabel?: string) {
  const base = CLASSIFICATION_META[classification] ?? {
    readableLabel: readableLabel ?? classification,
    shortBadge: readableLabel ?? classification,
    icon: "•",
    color: "#38bdf8",
    bg: "rgba(56, 189, 248, 0.15)",
    border: "rgba(56, 189, 248, 0.45)",
  };
  return readableLabel ? { ...base, readableLabel } : base;
}

const SHEEP_LINE_COLORS = [
  "#38bdf8",
  "#f472b6",
  "#a78bfa",
  "#34d399",
  "#fbbf24",
  "#fb7185",
];

function EpistemicBadge({
  kind,
  detail,
}: {
  kind: "recorded" | "derived" | "estimated" | "recomputed" | "truncated" | "not_recorded" | "omitted";
  detail?: string;
}) {
  const styles: Record<
    typeof kind,
    { label: string; color: string; bg: string; border: string }
  > = {
    recorded: {
      label: detail ? `Recorded (${detail})` : "Recorded",
      color: "#86efac",
      bg: "rgba(34, 197, 94, 0.14)",
      border: "rgba(74, 222, 128, 0.4)",
    },
    derived: {
      label: detail ? `Derived (${detail})` : "Derived",
      color: "#67e8f9",
      bg: "rgba(6, 182, 212, 0.14)",
      border: "rgba(34, 211, 238, 0.4)",
    },
    estimated: {
      label: detail ? `Estimated (${detail})` : "Estimated",
      color: "#fde68a",
      bg: "rgba(245, 158, 11, 0.16)",
      border: "rgba(251, 191, 36, 0.45)",
    },
    recomputed: {
      label: detail ? `Recomputed (${detail})` : "Recomputed",
      color: "#d8b4fe",
      bg: "rgba(168, 85, 247, 0.16)",
      border: "rgba(192, 132, 252, 0.45)",
    },
    truncated: {
      label: detail ? `Truncated (${detail})` : "Truncated",
      color: "#fda4af",
      bg: "rgba(244, 63, 94, 0.16)",
      border: "rgba(251, 113, 133, 0.45)",
    },
    omitted: {
      label: detail ? `Omitted (${detail})` : "Omitted",
      color: "#93c5fd",
      bg: "rgba(59, 130, 246, 0.16)",
      border: "rgba(96, 165, 250, 0.45)",
    },
    not_recorded: {
      label: detail ? `Not recorded (${detail})` : "Not recorded",
      color: "#cbd5e1",
      bg: "rgba(100, 116, 139, 0.2)",
      border: "rgba(148, 163, 184, 0.35)",
    },
  };
  const s = styles[kind];
  return (
    <span
      style={{
        display: "inline-flex",
        alignItems: "center",
        padding: "0.1rem 0.45rem",
        borderRadius: "999px",
        fontSize: "0.68rem",
        fontWeight: 700,
        color: s.color,
        background: s.bg,
        border: `1px solid ${s.border}`,
        letterSpacing: "0.02em",
        whiteSpace: "nowrap",
      }}
    >
      {s.label}
    </span>
  );
}

/** Synthesize a ReplaySnapshot from TemporalStepEvidence if raw replay frames are not separately loaded. */
function buildFallbackSnapshotFromStep(
  stepEv: TemporalStepEvidence,
  payload: TemporalDiagnosticPayload,
): ReplaySnapshot {
  const pen = payload.analysis.pen_geometry;
  return {
    step: stepEv.step,
    simulated_seconds: stepEv.step * 0.25,
    grid_width: pen.grid_width,
    grid_height: pen.grid_height,
    dogs: stepEv.dogs.map((d) => ({
      index: d.dog_index,
      x: d.post_step_position.x,
      y: d.post_step_position.y,
      role: d.recorded_role,
      last_action: d.decision?.executed_action ?? "wait",
    })),
    sheep: stepEv.sheep.map((s) => ({
      index: s.sheep_index,
      x: s.post_step_position.x,
      y: s.post_step_position.y,
      penned: s.penned,
      personality: s.personality,
      packed: s.is_packed,
    })),
    pen: {
      origin: pen.origin,
      width: pen.width,
      height: pen.height,
      opening: pen.opening as "left" | "right" | "top" | "bottom",
    },
    penned_count: stepEv.penned_count,
    average_distance_to_pen: stepEv.average_distance_to_pen,
    flock_spread: stepEv.flock_spread,
    no_progress_steps: stepEv.no_progress_steps,
    terminated: false,
    timeout: payload.analysis.outcome_status === "timeout",
    stopped: payload.analysis.outcome_status === "stopped",
    success: payload.analysis.outcome_status === "success",
    status: payload.analysis.outcome_status,
  };
}

export function TemporalDiagnosticsPanel({
  initialReplayId = null,
  activeReplay = null,
  activeReplayId = null,
  onSyncReplayToMainViewer,
  backendOffline = false,
  isReadOnlyPreview = false,
  backendStarting = false,
  onCachedReplayAvailableChange,
  onRequestReconnect,
}: TemporalDiagnosticsPanelProps) {
  const [examples, setExamples] = useState<TemporalDiagnosticExample[]>([]);
  const [selectedReplayId, setSelectedReplayId] = useState<string>(
    initialReplayId?.trim() || activeReplayId?.trim() || "checkpoint-039909-seed-000041",
  );
  const [customReplayInput, setCustomReplayInput] = useState<string>(
    initialReplayId?.trim() || activeReplayId?.trim() || "checkpoint-039909-seed-000041",
  );
  const [payload, setPayload] = useState<TemporalDiagnosticPayload | null>(null);
  const [replayBundle, setReplayBundle] = useState<ReplayBundle | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [localBackendReachable, setLocalBackendReachable] = useState<boolean>(true);
  const [localReadOnlyPreview, setLocalReadOnlyPreview] = useState<boolean>(false);
  const payloadRef = useRef<TemporalDiagnosticPayload | null>(null);
  const onRequestReconnectRef = useRef(onRequestReconnect);
  const prevReachableRef = useRef<boolean>(!backendOffline);

  useEffect(() => {
    onRequestReconnectRef.current = onRequestReconnect;
  }, [onRequestReconnect]);

  useEffect(() => {
    if (backendOffline) {
      prevReachableRef.current = false;
    }
  }, [backendOffline]);

  useEffect(() => {
    payloadRef.current = payload;
    onCachedReplayAvailableChange?.(Boolean(payload), payload?.analysis.replay_id ?? null);
  }, [payload, onCachedReplayAvailableChange]);

  const probeBackendConnection = useCallback(
    async (isManualReconnect = false): Promise<BackendHealthResult> => {
      const res = await checkBackendHealth();
      if (res.ok) {
        const wasUnreachable = !prevReachableRef.current;
        prevReachableRef.current = true;
        setLocalBackendReachable(true);
        setLocalReadOnlyPreview(Boolean(res.readOnlyPreview));
        if (wasUnreachable || isManualReconnect) {
          void onRequestReconnectRef.current?.();
        }
      } else {
        prevReachableRef.current = false;
        setLocalBackendReachable(false);
      }
      return res;
    },
    [],
  );

  const handleReconnectClick = useCallback(async () => {
    await probeBackendConnection(true);
  }, [probeBackendConnection]);

  useEffect(() => {
    let active = true;
    const check = async () => {
      if (!active) return;
      await probeBackendConnection(false);
    };
    void check();
    const timer = window.setInterval(check, 3000);
    return () => {
      active = false;
      window.clearInterval(timer);
    };
  }, [probeBackendConnection]);

  const initialUrlStep = useMemo<number | null>(() => {
    if (typeof window === "undefined") return null;
    const raw = new URLSearchParams(window.location.search).get("step");
    if (raw === null || raw.trim() === "") return null;
    const parsed = Number(raw);
    return Number.isFinite(parsed) && parsed >= 0 ? Math.floor(parsed) : null;
  }, []);
  const initialUrlStepAppliedRef = useRef<boolean>(false);

  // Synchronized playhead state by exact world_step
  const [currentWorldStep, setCurrentWorldStep] = useState<number>(() => initialUrlStep ?? 1);
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const [playbackSpeed, setPlaybackSpeed] = useState<number>(4);
  const [playWindowOnly, setPlayWindowOnly] = useState<boolean>(false);

  // Selected interval and filters
  const [selectedIntervalId, setSelectedIntervalId] = useState<string | null>(null);
  const [unpennedFilter, setUnpennedFilter] = useState<number | "all">("all");

  // Field overlays
  const [overlays, setOverlays] = useState<FieldOverlayConfig>({
    recordedTrails: true,
    estimatedTargets: true,
    derivedWaypoints: true,
    descriptiveGeometry: true,
  });

  // Chart & entity inspector controls
  const [showPerSheepDistances, setShowPerSheepDistances] = useState<boolean>(true);
  const [selectedRewardComponent, setSelectedRewardComponent] = useState<string>("total");
  const [selectedEntity, setSelectedEntity] = useState<SelectedFieldEntity>({
    type: "dog",
    index: 0,
  });
  const [obsSearchQuery, setObsSearchQuery] = useState<string>("");

  const playTimerRef = useRef<number | null>(null);
  const activeReplayRef = useRef<ReplayBundle | null>(activeReplay);
  useEffect(() => {
    activeReplayRef.current = activeReplay;
  }, [activeReplay]);

  // Update selectedReplayId only when initialReplayId prop itself changes
  const prevInitialReplayIdRef = useRef<string | null | undefined>(initialReplayId);
  useEffect(() => {
    if (initialReplayId !== prevInitialReplayIdRef.current) {
      prevInitialReplayIdRef.current = initialReplayId;
      if (initialReplayId && initialReplayId.trim()) {
        const clean = initialReplayId.trim();
        setSelectedReplayId(clean);
        setCustomReplayInput(clean);
      }
    }
  }, [initialReplayId]);

  // Load curated read-only examples on mount
  useEffect(() => {
    let cancelled = false;
    loadTemporalDiagnosticExamples().then((list) => {
      if (cancelled) return;
      setExamples(list);
      if (!initialReplayId && !activeReplayId && list.length > 0) {
        const rec = list.find((item) => item.recommended_default) ?? list[0];
        setSelectedReplayId(rec.replay_id);
        setCustomReplayInput(rec.replay_id);
      }
    });
    return () => {
      cancelled = true;
    };
  }, [initialReplayId, activeReplayId]);

  // Load temporal diagnostic analysis for selectedReplayId
  useEffect(() => {
    if (!selectedReplayId) return;
    let cancelled = false;
    const wasCachedBeforeLoad = hasCachedTemporalDiagnosticAnalysis(selectedReplayId, true);
    if (!wasCachedBeforeLoad || payloadRef.current === null) {
      setLoading(true);
    }
    setErrorMessage(null);
    setIsPlaying(false);

    loadTemporalDiagnosticAnalysis(selectedReplayId, { includeReplay: true })
      .then(async (result) => {
        if (cancelled) return;
        if (wasCachedBeforeLoad) {
          const health = await checkBackendHealth();
          if (!cancelled) {
            prevReachableRef.current = health.ok;
            setLocalBackendReachable(health.ok);
            if (health.ok) {
              setLocalReadOnlyPreview(Boolean(health.readOnlyPreview));
            }
          }
        } else {
          prevReachableRef.current = true;
          setLocalBackendReachable(true);
        }
        setPayload(result);

        let bundle: ReplayBundle | null = result.replay ?? null;
        if (!bundle) {
          try {
            bundle = (await fetchReplayById(selectedReplayId)) as ReplayBundle;
          } catch {
            bundle = activeReplayRef.current ?? null;
          }
        }
        if (cancelled) return;
        setReplayBundle(bundle);

        const intervals = result.analysis.intervals;
        const steps = result.step_evidence;
        const requestedInitialStep =
          !initialUrlStepAppliedRef.current && initialUrlStep !== null ? initialUrlStep : null;
        initialUrlStepAppliedRef.current = true;
        const hasValidRequestedStep =
          requestedInitialStep !== null &&
          (steps.some((s) => s.step === requestedInitialStep) ||
            (requestedInitialStep >= 0 && requestedInitialStep <= result.analysis.total_steps));

        if (intervals.length > 0) {
          const matchingIv =
            hasValidRequestedStep && requestedInitialStep !== null
              ? intervals.find(
                  (iv) =>
                    requestedInitialStep >= iv.lead_in.start_step &&
                    requestedInitialStep <= iv.aftermath.end_step,
                ) ?? intervals[0]
              : intervals[0];
          setSelectedIntervalId(matchingIv.interval_id);
          setCurrentWorldStep(
            hasValidRequestedStep && requestedInitialStep !== null
              ? requestedInitialStep
              : matchingIv.lead_in.start_step,
          );
          if (matchingIv.affected_dog_indices.length > 0) {
            setSelectedEntity({ type: "dog", index: matchingIv.affected_dog_indices[0] });
          }
        } else if (steps.length > 0) {
          setSelectedIntervalId(null);
          setCurrentWorldStep(
            hasValidRequestedStep && requestedInitialStep !== null
              ? requestedInitialStep
              : steps[0].step,
          );
        }
        setLoading(false);
      })
      .catch((err: unknown) => {
        if (cancelled) return;
        const apiErr = err as ApiError | undefined;
        const msg =
          err instanceof Error
            ? err.message
            : "Failed to load temporal diagnostic analysis for this replay.";
        const isConnectivityError =
          Boolean(apiErr?.isNetworkError) ||
          apiErr?.status === 0 ||
          (typeof apiErr?.status === "number" && apiErr.status >= 500) ||
          /network|connection|offline|failed to fetch|expected json/i.test(msg);

        if (isConnectivityError) {
          prevReachableRef.current = false;
          setLocalBackendReachable(false);
          if (payloadRef.current !== null) {
            setCustomReplayInput(payloadRef.current.analysis.replay_id);
            setLoading(false);
            return;
          }
        }
        setErrorMessage(msg);
        setPayload(null);
        setReplayBundle(null);
        setLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [selectedReplayId, initialUrlStep]);

  const effectiveOffline = Boolean(backendOffline || !localBackendReachable);
  const effectiveReadOnly = Boolean(
    !effectiveOffline && (isReadOnlyPreview || localReadOnlyPreview),
  );
  const connectionState:
    | "read_only_preview"
    | "connected_normal"
    | "offline_cached_replay"
    | "backend_offline" = effectiveOffline
    ? payload !== null
      ? "offline_cached_replay"
      : "backend_offline"
    : effectiveReadOnly
      ? "read_only_preview"
      : "connected_normal";

  const stepEvidenceList = useMemo(() => payload?.step_evidence ?? [], [payload]);

  // Map exact world_step -> index in step_evidence
  const stepIndexByWorldStep = useMemo(() => {
    const map = new Map<number, number>();
    stepEvidenceList.forEach((item, idx) => {
      map.set(item.step, idx);
    });
    return map;
  }, [stepEvidenceList]);

  // Find closest available step index for currentWorldStep (handles missing step 0 and frame gaps)
  const currentStepIndex = useMemo(() => {
    if (stepEvidenceList.length === 0) return 0;
    const exact = stepIndexByWorldStep.get(currentWorldStep);
    if (exact !== undefined) return exact;
    let bestIdx = 0;
    let bestDiff = Math.abs(stepEvidenceList[0].step - currentWorldStep);
    for (let i = 1; i < stepEvidenceList.length; i += 1) {
      const diff = Math.abs(stepEvidenceList[i].step - currentWorldStep);
      if (diff < bestDiff) {
        bestDiff = diff;
        bestIdx = i;
      }
    }
    return bestIdx;
  }, [stepEvidenceList, stepIndexByWorldStep, currentWorldStep]);

  const currentStepEvidence: TemporalStepEvidence | null =
    stepEvidenceList[currentStepIndex] ?? null;

  const actualWorldStep = currentStepEvidence?.step ?? currentWorldStep;

  // Map replay frames by step number so FieldView always aligns by exact world_step
  const frameByWorldStep = useMemo(() => {
    const map = new Map<number, { frame: ReplayFrame; index: number }>();
    const frames = replayBundle?.frames ?? [];
    frames.forEach((fr, idx) => {
      const st = fr.step ?? fr.snapshot?.step ?? idx;
      map.set(st, { frame: fr, index: idx });
    });
    return map;
  }, [replayBundle]);

  const currentSnapshot: ReplaySnapshot | null = useMemo(() => {
    const match = frameByWorldStep.get(actualWorldStep);
    if (match?.frame?.snapshot) {
      return {
        ...match.frame.snapshot,
        step: actualWorldStep,
      };
    }
    if (currentStepEvidence && payload) {
      return buildFallbackSnapshotFromStep(currentStepEvidence, payload);
    }
    return null;
  }, [frameByWorldStep, actualWorldStep, currentStepEvidence, payload]);

  const recentFramesForTrails: ReplayFrame[] = useMemo(() => {
    const frames = replayBundle?.frames ?? [];
    const match = frameByWorldStep.get(actualWorldStep);
    if (match && frames.length > 0) {
      const start = Math.max(0, match.index - 18);
      return frames.slice(start, match.index + 1);
    }
    if (!payload || stepEvidenceList.length === 0) return [];
    const start = Math.max(0, currentStepIndex - 18);
    return stepEvidenceList.slice(start, currentStepIndex + 1).map((st) => ({
      step: st.step,
      actions: st.actions,
      snapshot: buildFallbackSnapshotFromStep(st, payload),
      reward: st.reward as unknown as ReplayFrame["reward"],
    }));
  }, [replayBundle, frameByWorldStep, actualWorldStep, payload, stepEvidenceList, currentStepIndex]);

  // Filtered intervals and currently selected interval (preserving stable episode-wide 1-based numbering)
  const allIntervals = useMemo(() => payload?.analysis.intervals ?? [], [payload]);
  const episodeIntervalNumberById = useMemo(() => {
    const map = new Map<string, number>();
    allIntervals.forEach((iv, idx) => {
      map.set(iv.interval_id, idx + 1);
    });
    return map;
  }, [allIntervals]);

  const filteredIntervals = useMemo(() => {
    if (unpennedFilter === "all") return allIntervals;
    return allIntervals.filter((iv) => iv.unpenned_sheep_count === unpennedFilter);
  }, [allIntervals, unpennedFilter]);

  const selectedInterval: TemporalInteractionInterval | null = useMemo(() => {
    if (!allIntervals.length) return null;
    return (
      allIntervals.find((iv) => iv.interval_id === selectedIntervalId) ??
      filteredIntervals[0] ??
      allIntervals[0] ??
      null
    );
  }, [allIntervals, filteredIntervals, selectedIntervalId]);

  const selectedIntervalEpisodeNumber = useMemo(() => {
    if (!selectedInterval) return null;
    return episodeIntervalNumberById.get(selectedInterval.interval_id) ?? 1;
  }, [selectedInterval, episodeIntervalNumberById]);

  // Select an interval and seek to its lead-in world step
  const handleSelectInterval = useCallback(
    (interval: TemporalInteractionInterval, autoPlay = false) => {
      setSelectedIntervalId(interval.interval_id);
      setCurrentWorldStep(interval.lead_in.start_step);
      setPlayWindowOnly(autoPlay);
      setIsPlaying(autoPlay);
      if (replayBundle && onSyncReplayToMainViewer) {
        onSyncReplayToMainViewer(replayBundle, interval.lead_in.start_step);
      }
    },
    [replayBundle, onSyncReplayToMainViewer],
  );

  // Seek to a specific world step
  const seekToWorldStep = useCallback(
    (targetStep: number) => {
      if (stepEvidenceList.length === 0) return;
      const exact = stepIndexByWorldStep.get(targetStep);
      if (exact !== undefined) {
        setCurrentWorldStep(stepEvidenceList[exact].step);
        return;
      }
      const minStep = stepEvidenceList[0].step;
      const maxStep = stepEvidenceList[stepEvidenceList.length - 1].step;
      const clamped = Math.max(minStep, Math.min(maxStep, targetStep));
      let bestStep = stepEvidenceList[0].step;
      let bestDiff = Math.abs(bestStep - clamped);
      for (const st of stepEvidenceList) {
        const diff = Math.abs(st.step - clamped);
        if (diff < bestDiff) {
          bestDiff = diff;
          bestStep = st.step;
        }
      }
      setCurrentWorldStep(bestStep);
    },
    [stepEvidenceList, stepIndexByWorldStep],
  );

  const stepByFrames = useCallback(
    (deltaFrames: number) => {
      if (stepEvidenceList.length === 0) return;
      setIsPlaying(false);
      const nextIdx = Math.max(
        0,
        Math.min(stepEvidenceList.length - 1, currentStepIndex + deltaFrames),
      );
      setCurrentWorldStep(stepEvidenceList[nextIdx].step);
    },
    [stepEvidenceList, currentStepIndex],
  );

  // Playback timer
  useEffect(() => {
    if (!isPlaying || stepEvidenceList.length === 0) {
      if (playTimerRef.current !== null) {
        window.clearInterval(playTimerRef.current);
        playTimerRef.current = null;
      }
      return;
    }

    const intervalMs = Math.max(30, Math.round(220 / Math.max(1, playbackSpeed)));
    playTimerRef.current = window.setInterval(() => {
      setCurrentWorldStep((prevStep) => {
        const idx = stepIndexByWorldStep.get(prevStep) ?? currentStepIndex;
        const maxBoundStep =
          playWindowOnly && selectedInterval
            ? selectedInterval.aftermath.end_step
            : stepEvidenceList[stepEvidenceList.length - 1].step;
        if (idx >= stepEvidenceList.length - 1 || prevStep >= maxBoundStep) {
          setIsPlaying(false);
          return prevStep;
        }
        return stepEvidenceList[idx + 1].step;
      });
    }, intervalMs);

    return () => {
      if (playTimerRef.current !== null) {
        window.clearInterval(playTimerRef.current);
        playTimerRef.current = null;
      }
    };
  }, [
    isPlaying,
    playbackSpeed,
    stepEvidenceList,
    stepIndexByWorldStep,
    currentStepIndex,
    playWindowOnly,
    selectedInterval,
  ]);

  const handleToggleOverlay = useCallback((key: keyof FieldOverlayConfig) => {
    setOverlays((prev) => ({ ...prev, [key]: !prev[key] }));
  }, []);

  // Frame gap detection across episode
  const frameGaps = useMemo(() => {
    const gaps: Array<{ fromStep: number; toStep: number }> = [];
    for (let i = 1; i < stepEvidenceList.length; i += 1) {
      if (!stepEvidenceList[i].has_prior_contiguous_step) {
        gaps.push({
          fromStep: stepEvidenceList[i - 1].step,
          toStep: stepEvidenceList[i].step,
        });
      }
    }
    return gaps;
  }, [stepEvidenceList]);

  const minWorldStep = stepEvidenceList[0]?.step ?? 0;
  const maxWorldStep = stepEvidenceList[stepEvidenceList.length - 1]?.step ?? 1;
  const worldStepSpan = Math.max(1, maxWorldStep - minWorldStep);

  // Determine current playhead phase relative to selectedInterval
  const currentIntervalPhaseLabel = useMemo(() => {
    if (!selectedInterval) return "Episode playback";
    if (
      actualWorldStep >= selectedInterval.lead_in.start_step &&
      actualWorldStep <= selectedInterval.lead_in.end_step &&
      actualWorldStep < selectedInterval.start_step
    ) {
      return `Lead-in (${selectedInterval.lead_in.start_step}..${selectedInterval.lead_in.end_step})`;
    }
    if (
      actualWorldStep >= selectedInterval.start_step &&
      actualWorldStep <= selectedInterval.end_step
    ) {
      return `Inside Interval (${selectedInterval.start_step}..${selectedInterval.end_step})`;
    }
    if (
      actualWorldStep > selectedInterval.end_step &&
      actualWorldStep <= selectedInterval.aftermath.end_step
    ) {
      return `Aftermath (${selectedInterval.aftermath.start_step}..${selectedInterval.aftermath.end_step})`;
    }
    return "Outside selected interval window";
  }, [selectedInterval, actualWorldStep]);

  const currentExampleMeta = useMemo(
    () => examples.find((ex) => ex.replay_id === selectedReplayId) ?? null,
    [examples, selectedReplayId],
  );

  // Keyboard shortcuts when focus is inside the Temporal Diagnostics workspace
  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLDivElement>) => {
      const target = e.target as HTMLElement | null;
      if (
        target &&
        (target.tagName === "INPUT" ||
          target.tagName === "SELECT" ||
          target.tagName === "TEXTAREA")
      ) {
        return;
      }
      if (e.key === "ArrowRight") {
        e.preventDefault();
        stepByFrames(e.shiftKey ? 10 : 1);
      } else if (e.key === "ArrowLeft") {
        e.preventDefault();
        stepByFrames(e.shiftKey ? -10 : -1);
      } else if (e.key === " ") {
        e.preventDefault();
        setPlayWindowOnly(false);
        setIsPlaying((prev) => !prev);
      } else if (e.key === "[" && filteredIntervals.length > 0) {
        e.preventDefault();
        const idx = filteredIntervals.findIndex(
          (iv) => iv.interval_id === selectedInterval?.interval_id,
        );
        const prevIdx = idx > 0 ? idx - 1 : filteredIntervals.length - 1;
        handleSelectInterval(filteredIntervals[prevIdx], false);
      } else if (e.key === "]" && filteredIntervals.length > 0) {
        e.preventDefault();
        const idx = filteredIntervals.findIndex(
          (iv) => iv.interval_id === selectedInterval?.interval_id,
        );
        const nextIdx = idx >= 0 && idx < filteredIntervals.length - 1 ? idx + 1 : 0;
        handleSelectInterval(filteredIntervals[nextIdx], false);
      }
    },
    [stepByFrames, filteredIntervals, selectedInterval, handleSelectInterval],
  );

  return (
    <section
      className="temporal-diagnostics-workspace"
      aria-label="Temporal Interaction Diagnostics"
      tabIndex={0}
      onKeyDown={handleKeyDown}
      style={{
        display: "flex",
        flexDirection: "column",
        gap: "0.55rem",
        outline: "none",
      }}
    >
      {/* Compact Top Bar: Connection Status, Curated Example / Replay Selector, Essential Evidence Warnings & Collapsible Guide/Audit */}
      <div
        className="panel"
        style={{
          padding: "0.48rem 0.75rem",
          background: "rgba(15, 23, 42, 0.9)",
          border: "1px solid rgba(56, 189, 248, 0.25)",
          borderRadius: "0.7rem",
          display: "flex",
          flexDirection: "column",
          gap: "0.4rem",
        }}
      >
        {/* Row 1: Title, Connection Status, Curated Example Select, Replay ID Form, Paired Seed Button */}
        <div
          style={{
            display: "flex",
            flexWrap: "wrap",
            alignItems: "center",
            justifyContent: "space-between",
            gap: "0.45rem 0.65rem",
          }}
        >
          <div style={{ display: "flex", flexWrap: "wrap", alignItems: "center", gap: "0.45rem" }}>
            <h2
              style={{
                margin: 0,
                fontSize: "0.95rem",
                fontWeight: 800,
                color: "#f8fafc",
                letterSpacing: "-0.01em",
                whiteSpace: "nowrap",
              }}
            >
              🔬 Temporal Diagnostics
            </h2>
            <span
              data-testid="temporal-connection-status"
              data-connection-state={connectionState}
              role="status"
              style={{
                display: "inline-flex",
                alignItems: "center",
                gap: "0.3rem",
                padding: "0.14rem 0.5rem",
                borderRadius: "999px",
                fontSize: "0.68rem",
                fontWeight: 700,
                letterSpacing: "0.02em",
                background:
                  connectionState === "read_only_preview"
                    ? "rgba(14, 165, 233, 0.18)"
                    : connectionState === "connected_normal"
                    ? "rgba(16, 185, 129, 0.18)"
                    : connectionState === "offline_cached_replay"
                    ? "rgba(245, 158, 11, 0.22)"
                    : "rgba(239, 68, 68, 0.22)",
                color:
                  connectionState === "read_only_preview"
                    ? "#7dd3fc"
                    : connectionState === "connected_normal"
                    ? "#6ee7b7"
                    : connectionState === "offline_cached_replay"
                    ? "#fde68a"
                    : "#fca5a5",
                border:
                  connectionState === "read_only_preview"
                    ? "1px solid rgba(56, 189, 248, 0.45)"
                    : connectionState === "connected_normal"
                    ? "1px solid rgba(52, 211, 153, 0.45)"
                    : connectionState === "offline_cached_replay"
                    ? "1px solid rgba(251, 191, 36, 0.55)"
                    : "1px solid rgba(248, 113, 113, 0.55)",
              }}
            >
              {connectionState === "read_only_preview"
                ? "● Connected · Read-Only Preview (Training Disabled)"
                : connectionState === "connected_normal"
                ? "● Connected · Normal Backend"
                : connectionState === "offline_cached_replay"
                ? "⚠ Backend Offline · Viewing Cached Replay"
                : backendStarting
                ? "◌ Backend Starting..."
                : "✕ Backend Unreachable"}
            </span>
            <button
              type="button"
              data-testid="temporal-reconnect-btn"
              onClick={() => void handleReconnectClick()}
              style={{
                padding: "0.14rem 0.48rem",
                borderRadius: "0.35rem",
                background: "rgba(30, 41, 59, 0.9)",
                color: "#cbd5e1",
                border: "1px solid rgba(148, 163, 184, 0.35)",
                fontSize: "0.66rem",
                fontWeight: 700,
                cursor: "pointer",
              }}
              title="Probe backend health and refresh connection status"
            >
              {effectiveOffline ? "↻ Reconnect" : "✓ Check Status"}
            </button>
          </div>

          {/* Compact Episode Selector & Replay ID Loader */}
          <div
            style={{
              display: "flex",
              flexWrap: "wrap",
              alignItems: "center",
              gap: "0.4rem",
              flex: "1 1 auto",
              justifyContent: "flex-end",
            }}
          >
            <label
              htmlFor="temporal-example-select"
              style={{ fontSize: "0.7rem", fontWeight: 700, color: "#cbd5e1", whiteSpace: "nowrap" }}
            >
              Example:
            </label>
            <select
              id="temporal-example-select"
              aria-label="Read-only curated episode examples"
              value={examples.some((ex) => ex.replay_id === selectedReplayId) ? selectedReplayId : ""}
              onChange={(e) => {
                if (e.target.value) {
                  setSelectedReplayId(e.target.value);
                  setCustomReplayInput(e.target.value);
                }
              }}
              style={{
                padding: "0.24rem 0.45rem",
                borderRadius: "0.4rem",
                background: "#0f172a",
                color: "#f8fafc",
                border: "1px solid rgba(56, 189, 248, 0.4)",
                fontSize: "0.73rem",
                fontWeight: 600,
                maxWidth: "290px",
              }}
            >
              <option value="" disabled>
                -- Select a read-only example recording --
              </option>
              {examples.map((ex) => (
                <option key={ex.replay_id} value={ex.replay_id}>
                  {ex.recommended_default ? "★ " : ""}
                  {ex.title} [{ex.replay_id}] — unpenned: {ex.unpenned_counts.join(", ")}
                </option>
              ))}
            </select>

            <form
              onSubmit={(e) => {
                e.preventDefault();
                if (customReplayInput.trim()) {
                  setSelectedReplayId(customReplayInput.trim());
                }
              }}
              style={{ display: "inline-flex", gap: "0.3rem", alignItems: "center", flexWrap: "wrap" }}
            >
              <label
                htmlFor="temporal-replay-id-input"
                style={{ fontSize: "0.7rem", fontWeight: 700, color: "#cbd5e1", whiteSpace: "nowrap" }}
              >
                Replay ID:
              </label>
              <input
                id="temporal-replay-id-input"
                aria-label="Or Open Any Approved Replay ID:"
                type="text"
                value={customReplayInput}
                onChange={(e) => setCustomReplayInput(e.target.value)}
                placeholder="e.g. checkpoint-039909-seed-000041"
                style={{
                  padding: "0.24rem 0.48rem",
                  borderRadius: "0.4rem",
                  background: "#0f172a",
                  color: "#f8fafc",
                  border: "1px solid rgba(148, 163, 184, 0.3)",
                  fontSize: "0.73rem",
                  width: "195px",
                }}
              />
              <button
                type="submit"
                style={{
                  padding: "0.24rem 0.6rem",
                  borderRadius: "0.4rem",
                  background: "#0284c7",
                  color: "#f8fafc",
                  border: "1px solid #38bdf8",
                  fontWeight: 700,
                  fontSize: "0.72rem",
                  cursor: "pointer",
                  whiteSpace: "nowrap",
                }}
              >
                Load Analysis
              </button>
              {activeReplayId && activeReplayId !== selectedReplayId ? (
                <button
                  type="button"
                  onClick={() => {
                    setSelectedReplayId(activeReplayId);
                    setCustomReplayInput(activeReplayId);
                  }}
                  style={{
                    padding: "0.24rem 0.55rem",
                    borderRadius: "0.4rem",
                    background: "rgba(30, 41, 59, 0.9)",
                    color: "#bae6fd",
                    border: "1px solid rgba(56, 189, 248, 0.35)",
                    fontWeight: 600,
                    fontSize: "0.71rem",
                    cursor: "pointer",
                    whiteSpace: "nowrap",
                  }}
                >
                  Use Active Replay ({activeReplayId})
                </button>
              ) : null}
              {currentExampleMeta?.paired_replay_id ? (
                <button
                  type="button"
                  onClick={() => {
                    const paired = currentExampleMeta.paired_replay_id!;
                    setSelectedReplayId(paired);
                    setCustomReplayInput(paired);
                  }}
                  style={{
                    padding: "0.24rem 0.55rem",
                    borderRadius: "0.4rem",
                    background: "rgba(16, 185, 129, 0.16)",
                    color: "#a7f3d0",
                    border: "1px solid rgba(52, 211, 153, 0.45)",
                    fontWeight: 700,
                    fontSize: "0.71rem",
                    cursor: "pointer",
                    whiteSpace: "nowrap",
                  }}
                  title={`Switch to Paired Seed ${currentExampleMeta.seed} Comparison (${currentExampleMeta.paired_replay_id})`}
                >
                  ⇄ Switch to Paired Seed {currentExampleMeta.seed} ({currentExampleMeta.paired_replay_id})
                </button>
              ) : null}
            </form>
          </div>
        </div>

        {/* Row 2: Compact 1-Line Evidence Warnings Summary + Collapsible Intro/Guide/Audit Details */}
        {!loading && payload ? (
          <details
            data-testid="temporal-intro-audit-details"
            aria-label="Episode evidence audit summary"
            style={{
              background: "rgba(9, 15, 29, 0.65)",
              border: "1px solid rgba(148, 163, 184, 0.18)",
              borderRadius: "0.45rem",
              padding: "0.18rem 0.55rem",
              fontSize: "0.69rem",
            }}
          >
            <summary
              style={{
                cursor: "pointer",
                fontWeight: 600,
                color: "#cbd5e1",
                userSelect: "none",
                display: "flex",
                flexWrap: "wrap",
                alignItems: "center",
                justifyContent: "space-between",
                gap: "0.3rem 0.6rem",
                listStyle: "none",
              }}
            >
              <span style={{ display: "inline-flex", flexWrap: "wrap", alignItems: "center", gap: "0.35rem" }}>
                <strong style={{ color: "#7dd3fc" }}>▸ Guide &amp; Audit:</strong>
                <code>{payload.analysis.replay_id}</code>
                <span>
                  ({payload.analysis.final_penned_count}/{payload.analysis.total_sheep} penned ·{" "}
                  <code>{payload.analysis.outcome_status}</code>/<code>{payload.analysis.stop_reason}</code> · Steps{" "}
                  {minWorldStep}..{maxWorldStep})
                </span>
                {payload.analysis.evidence_audit.has_initial_step_zero ? (
                  <EpistemicBadge kind="recorded" detail="step 0 present" />
                ) : (
                  <EpistemicBadge kind="not_recorded" detail={`starts at step ${minWorldStep}`} />
                )}
                {payload.analysis.evidence_audit.has_contiguous_frames ? (
                  <EpistemicBadge kind="recorded" detail="contiguous" />
                ) : (
                  <EpistemicBadge kind="estimated" detail={`${frameGaps.length} gap(s) detected`} />
                )}
                {payload.analysis.evidence_audit.guidance_certainty.startsWith("recorded_") ? (
                  <EpistemicBadge
                    kind="recorded"
                    detail={payload.analysis.evidence_audit.guidance_certainty}
                  />
                ) : (
                  <EpistemicBadge
                    kind="estimated"
                    detail={payload.analysis.evidence_audit.guidance_certainty}
                  />
                )}
                {payload.analysis.evidence_audit.telemetry_truncated ? (
                  <EpistemicBadge
                    kind="truncated"
                    detail={`step >= ${payload.analysis.evidence_audit.telemetry_truncation_step ?? "limit"}`}
                  />
                ) : null}
              </span>
              <span style={{ color: "#7dd3fc", fontSize: "0.67rem", fontWeight: 700 }}>
                ℹ Expand Guide, Provenance Key &amp; Penning Events ({payload.analysis.penning_steps.length})
              </span>
            </summary>
            <div
              style={{
                marginTop: "0.38rem",
                paddingTop: "0.38rem",
                borderTop: "1px solid rgba(148, 163, 184, 0.16)",
                display: "flex",
                flexDirection: "column",
                gap: "0.42rem",
                fontSize: "0.74rem",
                color: "#cbd5e1",
              }}
            >
              <p style={{ margin: 0 }}>
                Select a recorded episode and a detected interaction interval to seek to its lead-in, play through the
                interaction, and inspect what values are <strong>recorded</strong>, <strong>derived</strong>,{" "}
                <strong>estimated</strong>, <strong>recomputed</strong>, <strong>truncated</strong>, or{" "}
                <strong>not recorded</strong>. Observational diagnostics highlight candidate patterns and do not
                establish causal proof without a controlled experiment.
              </p>
              <div
                style={{
                  display: "flex",
                  flexWrap: "wrap",
                  alignItems: "center",
                  gap: "0.35rem",
                }}
                aria-label="Epistemic provenance key"
              >
                <span style={{ fontSize: "0.7rem", fontWeight: 700, color: "#94a3b8" }}>
                  Provenance Key:
                </span>
                <EpistemicBadge kind="recorded" />
                <EpistemicBadge kind="derived" />
                <EpistemicBadge kind="estimated" />
                <EpistemicBadge kind="recomputed" />
                <EpistemicBadge kind="truncated" />
                <EpistemicBadge kind="not_recorded" />
              </div>
              {payload.analysis.penning_steps.length > 0 ? (
                <div style={{ display: "flex", flexWrap: "wrap", gap: "0.28rem", alignItems: "center" }}>
                  <strong style={{ color: "#86efac" }}>Recorded Penning Events:</strong>
                  {payload.analysis.penning_steps.map(([pStep, sIdx]) => (
                    <button
                      key={`pen-${pStep}-s${sIdx}`}
                      type="button"
                      onClick={() => seekToWorldStep(pStep)}
                      style={{
                        padding: "0.1rem 0.45rem",
                        borderRadius: "999px",
                        background: "rgba(34, 197, 94, 0.15)",
                        border: "1px solid rgba(74, 222, 128, 0.45)",
                        color: "#bbf7d0",
                        fontSize: "0.69rem",
                        fontWeight: 700,
                        cursor: "pointer",
                      }}
                    >
                      ★ Sheep S{sIdx} penned @ step {pStep}
                    </button>
                  ))}
                </div>
              ) : (
                <div style={{ color: "#94a3b8" }}>
                  <strong>Recorded Penning Events:</strong> None (0 sheep penned).
                </div>
              )}
              {currentExampleMeta ? (
                <div
                  style={{
                    padding: "0.42rem 0.6rem",
                    borderRadius: "0.45rem",
                    background: "rgba(2, 132, 199, 0.12)",
                    border: "1px solid rgba(56, 189, 248, 0.3)",
                    color: "#e0f2fe",
                  }}
                >
                  <strong>Example Guide ({currentExampleMeta.replay_id}):</strong>{" "}
                  {currentExampleMeta.what_to_look_for} ({currentExampleMeta.penned_summary} · Unpenned phases:{" "}
                  {currentExampleMeta.unpenned_counts.join(" → ")} sheep)
                </div>
              ) : null}
              {payload.analysis.summary_notes.length > 0 ? (
                <div style={{ color: "#fde68a", fontSize: "0.72rem" }}>
                  {payload.analysis.summary_notes.map((note, idx) => (
                    <div key={`note-${idx}`}>• [Derived Diagnostic Note] {note}</div>
                  ))}
                </div>
              ) : null}
            </div>
          </details>
        ) : null}
      </div>

      {/* Offline Cached Replay Notice */}
      {connectionState === "offline_cached_replay" && payload ? (
        <div
          data-testid="temporal-offline-cached-banner"
          role="status"
          aria-live="polite"
          className="panel"
          style={{
            padding: "0.45rem 0.85rem",
            background: "rgba(120, 53, 15, 0.32)",
            border: "1px solid rgba(251, 191, 36, 0.55)",
            borderRadius: "0.6rem",
            color: "#fef3c7",
            display: "flex",
            flexWrap: "wrap",
            alignItems: "center",
            justifyContent: "space-between",
            gap: "0.5rem",
            fontSize: "0.74rem",
          }}
        >
          <div>
            <strong>Backend Unreachable — Viewing Cached Replay Evidence:</strong>{" "}
            Replay <code>{payload.analysis.replay_id}</code> ({stepEvidenceList.length} frames,{" "}
            {payload.analysis.intervals.length} intervals) remains interactive in client memory while offline.
          </div>
          <button
            type="button"
            onClick={() => void handleReconnectClick()}
            style={{
              padding: "0.22rem 0.6rem",
              borderRadius: "0.4rem",
              background: "rgba(245, 158, 11, 0.25)",
              color: "#fde68a",
              border: "1px solid rgba(251, 191, 36, 0.6)",
              fontWeight: 700,
              fontSize: "0.71rem",
              cursor: "pointer",
            }}
          >
            ↻ Reconnect to Backend
          </button>
        </div>
      ) : null}

      {/* Loading & Error States */}
      {loading ? (
        <div
          role="status"
          aria-live="polite"
          className="panel"
          style={{
            padding: "1.5rem",
            textAlign: "center",
            background: "rgba(15, 23, 42, 0.75)",
            border: "1px solid rgba(148, 163, 184, 0.2)",
            borderRadius: "0.75rem",
            color: "#e2e8f0",
          }}
        >
          <div style={{ fontSize: "0.95rem", fontWeight: 700, marginBottom: "0.3rem" }}>
            Loading temporal diagnostic analysis for <code>{selectedReplayId}</code>...
          </div>
          <div style={{ fontSize: "0.78rem", color: "#94a3b8" }}>
            Reusing Python temporal analyzer (results are cached after first computation).
          </div>
        </div>
      ) : null}

      {!loading && errorMessage ? (
        <div
          role="alert"
          className="panel"
          style={{
            padding: "1rem 1.25rem",
            background: "rgba(127, 29, 29, 0.28)",
            border: "1px solid rgba(248, 113, 113, 0.5)",
            borderRadius: "0.75rem",
            color: "#fecaca",
          }}
        >
          <div style={{ fontSize: "0.95rem", fontWeight: 800, marginBottom: "0.3rem" }}>
            Unable to Load Temporal Diagnostic Analysis
          </div>
          <div style={{ fontSize: "0.82rem", marginBottom: "0.5rem" }}>{errorMessage}</div>
          <div style={{ fontSize: "0.75rem", color: "#fca5a5" }}>
            Verify that the replay ID exists in an approved artifact directory (e.g.{" "}
            <code>artifacts/evaluations/replays/</code> or <code>artifacts/replays/</code>) and does not contain
            path traversal characters.
          </div>
        </div>
      ) : null}

      {/* Primary Synchronized Diagnostic Workspace: Field/Player Alongside Interval Selection & Synchronized Evidence */}
      {!loading && payload ? (
        <div className="temporal-primary-grid">
          {/* LEFT COLUMN: Primary Field & Playback Controls + Plain-Language Interval Explanation */}
          <div className="temporal-left-col">
            {/* Sticky Playback & Interval Window Transport Bar */}
            <div
              className="panel temporal-sticky-transport"
              aria-label="Synchronized playback and interval controls"
              style={{
                padding: "0.48rem 0.7rem",
                background: "rgba(15, 23, 42, 0.94)",
                border: "1px solid rgba(56, 189, 248, 0.3)",
                borderRadius: "0.65rem",
                display: "flex",
                flexDirection: "column",
                gap: "0.35rem",
              }}
            >
                <div
                  style={{
                    display: "flex",
                    flexWrap: "wrap",
                    alignItems: "center",
                    justifyContent: "space-between",
                    gap: "0.35rem",
                  }}
                >
                  <div style={{ display: "flex", alignItems: "center", gap: "0.28rem", flexWrap: "wrap" }}>
                    <button
                      type="button"
                      onClick={() => {
                        setPlayWindowOnly(false);
                        setIsPlaying((prev) => !prev);
                      }}
                      style={{
                        padding: "0.28rem 0.65rem",
                        borderRadius: "0.4rem",
                        background: isPlaying ? "#f59e0b" : "#10b981",
                        color: "#0f172a",
                        border: "none",
                        fontWeight: 800,
                        fontSize: "0.74rem",
                        cursor: "pointer",
                      }}
                    >
                      {isPlaying ? "⏸ Pause" : "▶ Play"}
                    </button>
                    <button
                      type="button"
                      onClick={() => stepByFrames(-10)}
                      title="Step backward 10 frames (Shift+Left)"
                      style={{
                        padding: "0.25rem 0.45rem",
                        borderRadius: "0.4rem",
                        background: "rgba(30, 41, 59, 0.9)",
                        color: "#e2e8f0",
                        border: "1px solid rgba(148, 163, 184, 0.3)",
                        fontSize: "0.71rem",
                        fontWeight: 700,
                        cursor: "pointer",
                      }}
                    >
                      « -10
                    </button>
                    <button
                      type="button"
                      onClick={() => stepByFrames(-1)}
                      title="Step backward 1 frame (ArrowLeft)"
                      style={{
                        padding: "0.25rem 0.5rem",
                        borderRadius: "0.4rem",
                        background: "rgba(30, 41, 59, 0.9)",
                        color: "#e2e8f0",
                        border: "1px solid rgba(148, 163, 184, 0.3)",
                        fontSize: "0.71rem",
                        fontWeight: 700,
                        cursor: "pointer",
                      }}
                    >
                      ‹ Step -1
                    </button>
                    <button
                      type="button"
                      onClick={() => stepByFrames(1)}
                      title="Step forward 1 frame (ArrowRight)"
                      style={{
                        padding: "0.25rem 0.5rem",
                        borderRadius: "0.4rem",
                        background: "rgba(30, 41, 59, 0.9)",
                        color: "#e2e8f0",
                        border: "1px solid rgba(148, 163, 184, 0.3)",
                        fontSize: "0.71rem",
                        fontWeight: 700,
                        cursor: "pointer",
                      }}
                    >
                      Step +1 ›
                    </button>
                    <button
                      type="button"
                      onClick={() => stepByFrames(10)}
                      title="Step forward 10 frames (Shift+Right)"
                      style={{
                        padding: "0.25rem 0.45rem",
                        borderRadius: "0.4rem",
                        background: "rgba(30, 41, 59, 0.9)",
                        color: "#e2e8f0",
                        border: "1px solid rgba(148, 163, 184, 0.3)",
                        fontSize: "0.71rem",
                        fontWeight: 700,
                        cursor: "pointer",
                      }}
                    >
                      +10 »
                    </button>
                    <label
                      style={{
                        display: "inline-flex",
                        alignItems: "center",
                        gap: "0.25rem",
                        fontSize: "0.71rem",
                        color: "#cbd5e1",
                        marginLeft: "0.15rem",
                      }}
                    >
                      Speed:
                      <select
                        aria-label="Playback speed"
                        value={playbackSpeed}
                        onChange={(e) => setPlaybackSpeed(Number(e.target.value))}
                        style={{
                          padding: "0.18rem 0.35rem",
                          borderRadius: "0.35rem",
                          background: "#0f172a",
                          color: "#f8fafc",
                          border: "1px solid rgba(148, 163, 184, 0.3)",
                          fontSize: "0.71rem",
                        }}
                      >
                        <option value={1}>1x</option>
                        <option value={2}>2x</option>
                        <option value={4}>4x</option>
                        <option value={8}>8x</option>
                        <option value={16}>16x</option>
                      </select>
                    </label>
                  </div>

                  <div style={{ fontSize: "0.73rem", fontWeight: 700, color: "#fde047" }}>
                    World Step {actualWorldStep}{" "}
                    <span style={{ color: "#94a3b8", fontWeight: 500 }}>
                      (Frame {currentStepIndex + 1} / {stepEvidenceList.length})
                    </span>
                  </div>
                </div>

                {/* World-Step Scrubber Slider */}
                <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
                  <span style={{ fontSize: "0.68rem", color: "#94a3b8", minWidth: "3rem" }}>
                    Step {minWorldStep}
                  </span>
                  <input
                    type="range"
                    aria-label="World step scrubber"
                    min={0}
                    max={Math.max(0, stepEvidenceList.length - 1)}
                    value={currentStepIndex}
                    onChange={(e) => {
                      setIsPlaying(false);
                      const idx = Number(e.target.value);
                      if (stepEvidenceList[idx]) {
                        setCurrentWorldStep(stepEvidenceList[idx].step);
                      }
                    }}
                    style={{ flex: 1, cursor: "pointer", height: "16px" }}
                  />
                  <span
                    style={{
                      fontSize: "0.68rem",
                      color: "#94a3b8",
                      minWidth: "3.6rem",
                      textAlign: "right",
                    }}
                  >
                    Step {maxWorldStep}
                  </span>
                </div>

                {/* Selected Interval Quick-Seek Buttons (Lead-In, Interval Start, Interval End, Aftermath) */}
                {selectedInterval ? (
                  <div
                    style={{
                      display: "flex",
                      flexWrap: "wrap",
                      alignItems: "center",
                      gap: "0.28rem",
                      paddingTop: "0.2rem",
                      borderTop: "1px solid rgba(148, 163, 184, 0.14)",
                    }}
                  >
                    <span style={{ fontSize: "0.68rem", fontWeight: 700, color: "#94a3b8" }}>
                      Interval #{selectedIntervalEpisodeNumber}:
                    </span>
                    <button
                      type="button"
                      onClick={() => {
                        setIsPlaying(false);
                        seekToWorldStep(selectedInterval.lead_in.start_step);
                      }}
                      style={{
                        padding: "0.16rem 0.45rem",
                        borderRadius: "0.35rem",
                        background: "rgba(56, 189, 248, 0.16)",
                        border: "1px solid rgba(56, 189, 248, 0.45)",
                        color: "#bae6fd",
                        fontSize: "0.68rem",
                        fontWeight: 700,
                        cursor: "pointer",
                      }}
                    >
                      ⏮ Seek Lead-In (step {selectedInterval.lead_in.start_step})
                    </button>
                    <button
                      type="button"
                      onClick={() => {
                        setIsPlaying(false);
                        seekToWorldStep(selectedInterval.start_step);
                      }}
                      style={{
                        padding: "0.16rem 0.45rem",
                        borderRadius: "0.35rem",
                        background: "rgba(245, 158, 11, 0.16)",
                        border: "1px solid rgba(245, 158, 11, 0.45)",
                        color: "#fde68a",
                        fontSize: "0.68rem",
                        fontWeight: 700,
                        cursor: "pointer",
                      }}
                    >
                      |[ Seek Interval Start (step {selectedInterval.start_step})
                    </button>
                    <button
                      type="button"
                      onClick={() => {
                        setIsPlaying(false);
                        seekToWorldStep(selectedInterval.end_step);
                      }}
                      style={{
                        padding: "0.16rem 0.45rem",
                        borderRadius: "0.35rem",
                        background: "rgba(245, 158, 11, 0.16)",
                        border: "1px solid rgba(245, 158, 11, 0.45)",
                        color: "#fde68a",
                        fontSize: "0.68rem",
                        fontWeight: 700,
                        cursor: "pointer",
                      }}
                    >
                      ]| Seek Interval End (step {selectedInterval.end_step})
                    </button>
                    <button
                      type="button"
                      onClick={() => {
                        setIsPlaying(false);
                        seekToWorldStep(selectedInterval.aftermath.end_step);
                      }}
                      style={{
                        padding: "0.16rem 0.45rem",
                        borderRadius: "0.35rem",
                        background: "rgba(168, 85, 247, 0.16)",
                        border: "1px solid rgba(192, 132, 252, 0.45)",
                        color: "#e9d5ff",
                        fontSize: "0.68rem",
                        fontWeight: 700,
                        cursor: "pointer",
                      }}
                    >
                      ⏭ Seek Aftermath End (step {selectedInterval.aftermath.end_step})
                    </button>
                    <button
                      type="button"
                      onClick={() => handleSelectInterval(selectedInterval, true)}
                      style={{
                        padding: "0.16rem 0.5rem",
                        borderRadius: "0.35rem",
                        background: "rgba(16, 185, 129, 0.2)",
                        border: "1px solid rgba(52, 211, 153, 0.5)",
                        color: "#a7f3d0",
                        fontSize: "0.68rem",
                        fontWeight: 800,
                        cursor: "pointer",
                      }}
                    >
                      ▶ Play Lead-In → Aftermath
                    </button>
                  </div>
                ) : null}
              </div>

            {/* Section A: Animated Field (reusing FieldView) */}
            <div className="temporal-section-player">
              <FieldView
                snapshot={currentSnapshot}
                temporalStepEvidence={currentStepEvidence}
                recentFrames={recentFramesForTrails}
                selectedEntity={selectedEntity}
                onSelectEntity={setSelectedEntity}
                highlightedDogIndices={selectedInterval?.affected_dog_indices ?? []}
                highlightedSheepIndices={selectedInterval?.affected_sheep_indices ?? []}
                sheepVision={payload.sheep_vision}
                overlays={overlays}
                onToggleOverlay={handleToggleOverlay}
                showSheepIndices={true}
              />
            </div>

            {/* Section E: Plain-Language Explanation */}
            <div className="temporal-section-explain">
              <PlainLanguageIntervalCard
                interval={selectedInterval}
                episodeIntervalNumber={selectedIntervalEpisodeNumber}
                onSeekToStep={seekToWorldStep}
              />
            </div>
          </div>

          {/* RIGHT COLUMN: Section B (Synchronized Timeline & Interval Selector), Section C (Charts) & Section D (Entity Inspector) */}
          <div className="temporal-right-col">
            {/* Section B: Synchronized Timeline & Interval Selector (Adjacent to Field on Desktop/Laptop, Above Player on Narrow Screens) */}
            <div
              className="panel temporal-section-timeline"
              aria-label="Synchronized interaction timeline"
              style={{
                padding: "0.55rem 0.75rem",
                background: "rgba(15, 23, 42, 0.88)",
                border: "1px solid rgba(148, 163, 184, 0.22)",
                borderRadius: "0.7rem",
                display: "flex",
                flexDirection: "column",
                gap: "0.42rem",
              }}
            >
              <div
                style={{
                  display: "flex",
                  flexWrap: "wrap",
                  alignItems: "center",
                  justifyContent: "space-between",
                  gap: "0.35rem",
                }}
              >
                <div>
                  <h3 style={{ margin: 0, fontSize: "0.88rem", fontWeight: 800, color: "#f8fafc" }}>
                    B. Synchronized Timeline &amp; Intervals ({filteredIntervals.length} of{" "}
                    {allIntervals.length} shown · stable IDs #1..#{allIntervals.length})
                  </h3>
                  <div style={{ fontSize: "0.68rem", color: "#94a3b8" }}>
                    Click an interval to seek to its <strong>lead-in world step</strong>. Keys:{" "}
                    <code>Space</code> play/pause, <code>←</code>/<code>→</code> step, <code>[</code>/
                    <code>]</code> prev/next interval.
                  </div>
                </div>

                {/* Filter by unpenned sheep count */}
                <div
                  role="group"
                  aria-label="Filter intervals by unpenned sheep count"
                  style={{ display: "flex", flexWrap: "wrap", gap: "0.22rem", alignItems: "center" }}
                >
                  <button
                    type="button"
                    aria-pressed={unpennedFilter === "all"}
                    onClick={() => setUnpennedFilter("all")}
                    style={{
                      padding: "0.14rem 0.42rem",
                      borderRadius: "0.35rem",
                      fontSize: "0.68rem",
                      fontWeight: 700,
                      cursor: "pointer",
                      background: unpennedFilter === "all" ? "#0284c7" : "rgba(30, 41, 59, 0.7)",
                      color: "#f8fafc",
                      border:
                        unpennedFilter === "all"
                          ? "1px solid #38bdf8"
                          : "1px solid rgba(148, 163, 184, 0.25)",
                    }}
                  >
                    All ({allIntervals.length})
                  </button>
                  {[6, 5, 4, 3, 2, 1].map((count) => {
                    const cntIntervalsList = allIntervals.filter(
                      (iv) => iv.unpenned_sheep_count === count,
                    );
                    const cntIntervals = cntIntervalsList.length;
                    const firstStepAtCount = stepEvidenceList.find(
                      (st) => st.unpenned_count === count,
                    );
                    const hasStepsAtCount = Boolean(firstStepAtCount);
                    return (
                      <button
                        key={`filter-unpenned-${count}`}
                        type="button"
                        disabled={cntIntervals === 0 && !hasStepsAtCount}
                        aria-pressed={unpennedFilter === count}
                        onClick={() => {
                          setUnpennedFilter(count);
                          if (selectedInterval && selectedInterval.unpenned_sheep_count === count) {
                            return;
                          }
                          if (cntIntervalsList.length > 0) {
                            handleSelectInterval(cntIntervalsList[0], false);
                          } else if (firstStepAtCount) {
                            seekToWorldStep(firstStepAtCount.step);
                          }
                        }}
                        style={{
                          padding: "0.14rem 0.38rem",
                          borderRadius: "0.35rem",
                          fontSize: "0.68rem",
                          fontWeight: 700,
                          cursor: cntIntervals === 0 && !hasStepsAtCount ? "not-allowed" : "pointer",
                          opacity: cntIntervals === 0 && !hasStepsAtCount ? 0.4 : 1,
                          background:
                            unpennedFilter === count ? "#0284c7" : "rgba(30, 41, 59, 0.7)",
                          color: "#f8fafc",
                          border:
                            unpennedFilter === count
                              ? "1px solid #38bdf8"
                              : "1px solid rgba(148, 163, 184, 0.25)",
                        }}
                      >
                        {count} Sheep ({cntIntervals})
                      </button>
                    );
                  })}
                </div>
              </div>

              {/* Visual World-Step Timeline Track */}
              <div
                style={{
                  position: "relative",
                  height: "46px",
                  background: "#090f1d",
                  border: "1px solid rgba(148, 163, 184, 0.25)",
                  borderRadius: "0.45rem",
                  overflow: "hidden",
                  userSelect: "none",
                }}
                aria-label="World-step interval track"
                onClick={(e) => {
                  const rect = e.currentTarget.getBoundingClientRect();
                  const ratio = Math.max(
                    0,
                    Math.min(1, (e.clientX - rect.left) / Math.max(1, rect.width)),
                  );
                  const targetStep = Math.round(minWorldStep + ratio * worldStepSpan);
                  seekToWorldStep(targetStep);
                }}
              >
                {/* Selected interval Lead-in and Aftermath shading */}
                {selectedInterval ? (
                  <>
                    <div
                      title={`Lead-in: steps ${selectedInterval.lead_in.start_step}..${selectedInterval.lead_in.end_step}`}
                      style={{
                        position: "absolute",
                        top: 0,
                        bottom: 0,
                        left: `${((selectedInterval.lead_in.start_step - minWorldStep) / worldStepSpan) * 100}%`,
                        width: `${Math.max(
                          0.5,
                          ((Math.min(selectedInterval.start_step, selectedInterval.lead_in.end_step) -
                            selectedInterval.lead_in.start_step) /
                            worldStepSpan) *
                            100,
                        )}%`,
                        background: "rgba(56, 189, 248, 0.16)",
                        borderLeft: "2px dashed #38bdf8",
                        pointerEvents: "none",
                      }}
                    />
                    <div
                      title={`Aftermath: steps ${selectedInterval.aftermath.start_step}..${selectedInterval.aftermath.end_step}`}
                      style={{
                        position: "absolute",
                        top: 0,
                        bottom: 0,
                        left: `${((selectedInterval.end_step - minWorldStep) / worldStepSpan) * 100}%`,
                        width: `${Math.max(
                          0.5,
                          ((selectedInterval.aftermath.end_step - selectedInterval.end_step) /
                            worldStepSpan) *
                            100,
                        )}%`,
                        background: "rgba(168, 85, 247, 0.16)",
                        borderRight: "2px dashed #c084fc",
                        pointerEvents: "none",
                      }}
                    />
                  </>
                ) : null}

                {/* Interval segments */}
                {allIntervals.map((iv, idx) => {
                  const epNum = episodeIntervalNumberById.get(iv.interval_id) ?? idx + 1;
                  const meta = getClassificationMeta(iv.classification, iv.readable_label);
                  const leftPct = ((iv.start_step - minWorldStep) / worldStepSpan) * 100;
                  const widthPct = Math.max(
                    1.2,
                    ((iv.end_step - iv.start_step + 1) / worldStepSpan) * 100,
                  );
                  const isSelected = selectedInterval?.interval_id === iv.interval_id;
                  return (
                    <button
                      key={iv.interval_id}
                      type="button"
                      onClick={(e) => {
                        e.stopPropagation();
                        handleSelectInterval(iv, false);
                      }}
                      title={`#${epNum} ${meta.icon} ${meta.readableLabel} (steps ${iv.start_step}..${iv.end_step}, lead-in ${iv.lead_in.start_step}, ${iv.unpenned_sheep_count} unpenned sheep)`}
                      style={{
                        position: "absolute",
                        top: isSelected ? "4px" : "8px",
                        height: isSelected ? "26px" : "20px",
                        left: `${leftPct}%`,
                        width: `${widthPct}%`,
                        background: meta.bg,
                        border: isSelected ? `2px solid #fde047` : `1px solid ${meta.border}`,
                        borderRadius: "0.25rem",
                        color: meta.color,
                        fontSize: "0.62rem",
                        fontWeight: 700,
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "center",
                        overflow: "hidden",
                        whiteSpace: "nowrap",
                        cursor: "pointer",
                        padding: "0 0.15rem",
                        zIndex: isSelected ? 4 : 2,
                      }}
                    >
                      {`#${epNum}`}
                    </button>
                  );
                })}

                {/* Penning markers on timeline */}
                {payload.analysis.penning_steps.map(([pStep, sIdx]) => {
                  const leftPct = ((pStep - minWorldStep) / worldStepSpan) * 100;
                  return (
                    <div
                      key={`tl-pen-${pStep}-${sIdx}`}
                      title={`Sheep S${sIdx} penned at world step ${pStep}`}
                      style={{
                        position: "absolute",
                        top: 0,
                        bottom: 0,
                        left: `${leftPct}%`,
                        width: "2px",
                        background: "#4ade80",
                        zIndex: 3,
                        pointerEvents: "none",
                      }}
                    />
                  );
                })}

                {/* Frame gap markers */}
                {frameGaps.map((gap) => {
                  const leftPct = ((gap.fromStep - minWorldStep) / worldStepSpan) * 100;
                  return (
                    <div
                      key={`tl-gap-${gap.fromStep}-${gap.toStep}`}
                      title={`Frame gap between step ${gap.fromStep} and step ${gap.toStep}`}
                      style={{
                        position: "absolute",
                        top: 0,
                        bottom: 0,
                        left: `${leftPct}%`,
                        width: "3px",
                        background: "#f43f5e",
                        zIndex: 3,
                        pointerEvents: "none",
                      }}
                    />
                  );
                })}

                {/* Playhead vertical line */}
                <div
                  aria-label={`Playhead at world step ${actualWorldStep}`}
                  style={{
                    position: "absolute",
                    top: 0,
                    bottom: 0,
                    left: `${((actualWorldStep - minWorldStep) / worldStepSpan) * 100}%`,
                    width: "2.5px",
                    background: "#fde047",
                    boxShadow: "0 0 8px rgba(253, 224, 71, 0.9)",
                    zIndex: 6,
                    pointerEvents: "none",
                  }}
                />

                {/* Step scale labels */}
                <div
                  style={{
                    position: "absolute",
                    bottom: "1px",
                    left: "5px",
                    right: "5px",
                    display: "flex",
                    justifyContent: "space-between",
                    fontSize: "0.62rem",
                    color: "#94a3b8",
                    pointerEvents: "none",
                  }}
                >
                  <span>
                    Start: {minWorldStep}
                    {!payload.analysis.evidence_audit.has_initial_step_zero ? " (no step 0)" : ""}
                  </span>
                  <span style={{ color: "#fde047", fontWeight: 700 }}>
                    Playhead: Step {actualWorldStep} — {currentIntervalPhaseLabel}
                  </span>
                  <span>End: {maxWorldStep}</span>
                </div>
              </div>

              {/* Selected Interval Step-Range Summary Banner */}
              {selectedInterval ? (
                <div
                  data-testid="temporal-selected-interval-banner"
                  style={{
                    display: "flex",
                    flexWrap: "wrap",
                    alignItems: "center",
                    justifyContent: "space-between",
                    gap: "0.3rem 0.6rem",
                    padding: "0.24rem 0.55rem",
                    borderRadius: "0.4rem",
                    background: "rgba(30, 41, 59, 0.85)",
                    border: "1px solid rgba(253, 224, 71, 0.4)",
                    fontSize: "0.7rem",
                    color: "#f8fafc",
                  }}
                >
                  <span>
                    <strong style={{ color: "#fde047" }}>
                      Selected Interval #{selectedIntervalEpisodeNumber}:
                    </strong>{" "}
                    {
                      getClassificationMeta(
                        selectedInterval.classification,
                        selectedInterval.readable_label,
                      ).readableLabel
                    }{" "}
                    · <strong>Steps {selectedInterval.start_step}..{selectedInterval.end_step}</strong> (
                    {selectedInterval.duration_steps} steps)
                  </span>
                  <span style={{ color: "#bae6fd", fontWeight: 600 }}>
                    Lead-in: <strong>{selectedInterval.lead_in.start_step}..{selectedInterval.lead_in.end_step}</strong>{" "}
                    · Aftermath: <strong>{selectedInterval.aftermath.start_step}..{selectedInterval.aftermath.end_step}</strong>{" "}
                    · <strong>{selectedInterval.unpenned_sheep_count} unpenned</strong>
                  </span>
                </div>
              ) : null}

              {/* Compact Episode-Wide Numbered Interval Selector Listbox */}
              {filteredIntervals.length === 0 ? (
                <div
                  style={{
                    padding: "0.55rem",
                    borderRadius: "0.45rem",
                    background: "rgba(30, 41, 59, 0.5)",
                    color: "#cbd5e1",
                    fontSize: "0.74rem",
                  }}
                >
                  No detected intervals match the selected unpenned-sheep filter. Switch filter to{" "}
                  <strong>All</strong> to view all {allIntervals.length} interval(s).
                </div>
              ) : (
                <div
                  role="listbox"
                  aria-label="Detected temporal intervals"
                  style={{
                    display: "grid",
                    gridTemplateColumns: "repeat(auto-fill, minmax(205px, 1fr))",
                    gap: "0.28rem",
                  }}
                >
                  {filteredIntervals.map((iv, idx) => {
                    const epNum = episodeIntervalNumberById.get(iv.interval_id) ?? idx + 1;
                    const meta = getClassificationMeta(iv.classification, iv.readable_label);
                    const isSelected = selectedInterval?.interval_id === iv.interval_id;
                    return (
                      <button
                        key={iv.interval_id}
                        type="button"
                        role="option"
                        aria-selected={isSelected}
                        onClick={() => handleSelectInterval(iv, false)}
                        style={{
                          textAlign: "left",
                          padding: "0.26rem 0.45rem",
                          borderRadius: "0.42rem",
                          background: isSelected
                            ? "rgba(30, 41, 59, 0.96)"
                            : "rgba(15, 23, 42, 0.78)",
                          border: isSelected
                            ? "2px solid #fde047"
                            : `1px solid ${meta.border}`,
                          color: "#f8fafc",
                          cursor: "pointer",
                          display: "flex",
                          flexDirection: "column",
                          gap: "0.1rem",
                        }}
                      >
                        <div
                          style={{
                            display: "flex",
                            alignItems: "center",
                            justifyContent: "space-between",
                            gap: "0.25rem",
                          }}
                        >
                          <span
                            style={{
                              fontSize: "0.69rem",
                              fontWeight: 800,
                              color: meta.color,
                              overflow: "hidden",
                              textOverflow: "ellipsis",
                              whiteSpace: "nowrap",
                            }}
                          >
                            {meta.icon} #{epNum} {meta.readableLabel}
                          </span>
                          <span
                            style={{
                              fontSize: "0.63rem",
                              fontWeight: 700,
                              padding: "0.05rem 0.35rem",
                              borderRadius: "999px",
                              background: "rgba(56, 189, 248, 0.15)",
                              color: "#bae6fd",
                              flexShrink: 0,
                            }}
                          >
                            {iv.unpenned_sheep_count} unpenned
                          </span>
                        </div>
                        <div style={{ fontSize: "0.66rem", color: "#cbd5e1", fontWeight: 600 }}>
                          Steps {iv.start_step}..{iv.end_step} ({iv.duration_steps} steps) | Lead-in:{" "}
                          {iv.lead_in.start_step}
                        </div>
                      </button>
                    );
                  })}
                </div>
              )}

              {/* Collapsible Full Interval Route & Reward Details */}
              <details
                data-testid="temporal-full-interval-cards-details"
                style={{
                  background: "rgba(9, 15, 29, 0.65)",
                  border: "1px solid rgba(148, 163, 184, 0.18)",
                  borderRadius: "0.42rem",
                  padding: "0.2rem 0.5rem",
                }}
              >
                <summary
                  style={{
                    cursor: "pointer",
                    fontSize: "0.69rem",
                    fontWeight: 700,
                    color: "#94a3b8",
                    userSelect: "none",
                  }}
                >
                  📋 Full Interval Route-Distance &amp; Reward Breakdown ({filteredIntervals.length} of{" "}
                  {allIntervals.length} intervals)
                </summary>
                <div
                  style={{
                    marginTop: "0.35rem",
                    paddingTop: "0.35rem",
                    borderTop: "1px solid rgba(148, 163, 184, 0.14)",
                    display: "grid",
                    gridTemplateColumns: "repeat(auto-fill, minmax(230px, 1fr))",
                    gap: "0.35rem",
                  }}
                >
                  {filteredIntervals.map((iv, idx) => {
                    const epNum = episodeIntervalNumberById.get(iv.interval_id) ?? idx + 1;
                    const meta = getClassificationMeta(iv.classification, iv.readable_label);
                    return (
                      <div
                        key={`full-detail-${iv.interval_id}`}
                        style={{
                          padding: "0.35rem 0.5rem",
                          borderRadius: "0.4rem",
                          background: "rgba(15, 23, 42, 0.8)",
                          border: `1px solid ${meta.border}`,
                          fontSize: "0.68rem",
                          color: "#cbd5e1",
                        }}
                      >
                        <div style={{ fontWeight: 800, color: meta.color }}>
                          #{epNum} {meta.readableLabel} ({iv.unpenned_sheep_count} unpenned)
                        </div>
                        <div>
                          Steps {iv.start_step}..{iv.end_step} ({iv.duration_steps}s) · Lead-in:{" "}
                          {iv.lead_in.start_step}..{iv.lead_in.end_step} · Aftermath:{" "}
                          {iv.aftermath.start_step}..{iv.aftermath.end_step}
                        </div>
                        <div style={{ color: "#94a3b8" }}>
                          Route dist: {iv.avg_route_distance_start.toFixed(1)} →{" "}
                          {iv.avg_route_distance_end.toFixed(1)} (
                          {iv.net_route_progress >= 0 ? "+" : ""}
                          {iv.net_route_progress.toFixed(1)} cells) | Reward:{" "}
                          {iv.reward_summary.total_reward >= 0 ? "+" : ""}
                          {iv.reward_summary.total_reward.toFixed(2)}
                        </div>
                      </div>
                    );
                  })}
                </div>
              </details>
            </div>

            {/* Section C: Progress and Reward Charts */}
            <div className="temporal-section-charts">
              <SynchronizedDiagnosticCharts
                payload={payload}
                stepEvidenceList={stepEvidenceList}
                actualWorldStep={actualWorldStep}
                selectedInterval={selectedInterval}
                showPerSheepDistances={showPerSheepDistances}
                onTogglePerSheepDistances={() => setShowPerSheepDistances((prev) => !prev)}
                selectedRewardComponent={selectedRewardComponent}
                onSelectRewardComponent={setSelectedRewardComponent}
                onSeekToStep={seekToWorldStep}
              />
            </div>

            {/* Section D: Entity Inspector */}
            <div className="temporal-section-inspector">
              <EntityInspectorCard
                payload={payload}
                stepEvidence={currentStepEvidence}
                selectedEntity={selectedEntity}
                onSelectEntity={setSelectedEntity}
                obsSearchQuery={obsSearchQuery}
                onChangeObsSearchQuery={setObsSearchQuery}
              />
            </div>
          </div>
        </div>
      ) : null}
    </section>
  );
}

/* ============================================================================
 * Section C: Synchronized Progress & Reward Charts
 * ========================================================================== */

function SynchronizedDiagnosticCharts({
  payload,
  stepEvidenceList,
  actualWorldStep,
  selectedInterval,
  showPerSheepDistances,
  onTogglePerSheepDistances,
  selectedRewardComponent,
  onSelectRewardComponent,
  onSeekToStep,
}: {
  payload: TemporalDiagnosticPayload;
  stepEvidenceList: TemporalStepEvidence[];
  actualWorldStep: number;
  selectedInterval: TemporalInteractionInterval | null;
  showPerSheepDistances: boolean;
  onTogglePerSheepDistances: () => void;
  selectedRewardComponent: string;
  onSelectRewardComponent: (comp: string) => void;
  onSeekToStep: (step: number) => void;
}) {
  const minStep = stepEvidenceList[0]?.step ?? 0;
  const maxStep = stepEvidenceList[stepEvidenceList.length - 1]?.step ?? 1;
  const stepSpan = Math.max(1, maxStep - minStep);

  // Downsample for SVG polyline performance while preserving exact world-step coordinates
  const sampledSteps = useMemo(() => {
    if (stepEvidenceList.length <= 360) return stepEvidenceList;
    const stride = Math.ceil(stepEvidenceList.length / 360);
    return stepEvidenceList.filter(
      (_, idx) => idx % stride === 0 || idx === stepEvidenceList.length - 1,
    );
  }, [stepEvidenceList]);

  const maxRouteDist = useMemo(() => {
    let m = 10;
    for (const st of sampledSteps) {
      if (st.average_distance_to_pen > m) m = st.average_distance_to_pen;
      for (const s of st.sheep) {
        if (s.route_distance > m) m = s.route_distance;
      }
    }
    return Math.ceil(m * 1.05);
  }, [sampledSteps]);

  const maxNoProgress = useMemo(() => {
    let m = 20;
    for (const st of sampledSteps) {
      if (st.no_progress_steps > m) m = st.no_progress_steps;
    }
    return Math.ceil(m * 1.05);
  }, [sampledSteps]);

  const rewardBounds = useMemo(() => {
    let minR = -0.1;
    let maxR = 0.1;
    for (const st of sampledSteps) {
      const val = st.reward[selectedRewardComponent] ?? 0;
      if (val < minR) minR = val;
      if (val > maxR) maxR = val;
    }
    return { minR, maxR, span: Math.max(1e-4, maxR - minR) };
  }, [sampledSteps, selectedRewardComponent]);

  const noProgressResets = useMemo(
    () => stepEvidenceList.filter((st) => st.no_progress_reset_without_best_gain),
    [stepEvidenceList],
  );

  const currentStepEv = useMemo(
    () => stepEvidenceList.find((st) => st.step === actualWorldStep) ?? stepEvidenceList[0] ?? null,
    [stepEvidenceList, actualWorldStep],
  );

  const handleSvgClick = (e: React.MouseEvent<SVGSVGElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const ratio = Math.max(0, Math.min(1, (e.clientX - rect.left) / Math.max(1, rect.width)));
    const targetStep = Math.round(minStep + ratio * stepSpan);
    onSeekToStep(targetStep);
  };

  const xOfStep = (step: number) => ((step - minStep) / stepSpan) * 600;
  const playheadX = xOfStep(actualWorldStep);

  const renderIntervalShading = (height: number) => {
    if (!selectedInterval) return null;
    const leadStart = xOfStep(selectedInterval.lead_in.start_step);
    const ivStart = xOfStep(selectedInterval.start_step);
    const ivEnd = xOfStep(selectedInterval.end_step);
    const afterEnd = xOfStep(selectedInterval.aftermath.end_step);
    return (
      <g>
        <rect
          x={leadStart}
          y={0}
          width={Math.max(1, ivStart - leadStart)}
          height={height}
          fill="rgba(56, 189, 248, 0.12)"
        />
        <rect
          x={ivStart}
          y={0}
          width={Math.max(2, ivEnd - ivStart)}
          height={height}
          fill="rgba(245, 158, 11, 0.18)"
        />
        <rect
          x={ivEnd}
          y={0}
          width={Math.max(1, afterEnd - ivEnd)}
          height={height}
          fill="rgba(168, 85, 247, 0.12)"
        />
      </g>
    );
  };

  const renderPenningAndTruncationMarkers = (height: number) => (
    <g>
      {payload.analysis.penning_steps.map(([pStep, sIdx]) => {
        const px = xOfStep(pStep);
        return (
          <line
            key={`chart-pen-${pStep}-${sIdx}`}
            x1={px}
            y1={0}
            x2={px}
            y2={height}
            stroke="#4ade80"
            strokeWidth={1.2}
            strokeDasharray="3 2"
          />
        );
      })}
      {payload.analysis.evidence_audit.telemetry_truncated &&
      payload.analysis.evidence_audit.telemetry_truncation_step !== null ? (
        <line
          x1={xOfStep(payload.analysis.evidence_audit.telemetry_truncation_step)}
          y1={0}
          x2={xOfStep(payload.analysis.evidence_audit.telemetry_truncation_step)}
          y2={height}
          stroke="#fb7185"
          strokeWidth={1.5}
          strokeDasharray="2 2"
        />
      ) : null}
    </g>
  );

  const sheepCount = payload.analysis.total_sheep || 6;

  return (
    <div
      className="panel"
      aria-label="Synchronized progress and reward charts"
      style={{
        padding: "0.95rem 1.1rem",
        background: "rgba(15, 23, 42, 0.85)",
        border: "1px solid rgba(148, 163, 184, 0.2)",
        borderRadius: "0.85rem",
        display: "flex",
        flexDirection: "column",
        gap: "0.8rem",
      }}
    >
      <div
        style={{
          display: "flex",
          flexWrap: "wrap",
          alignItems: "center",
          justifyContent: "space-between",
          gap: "0.5rem",
        }}
      >
        <div>
          <h3 style={{ margin: 0, fontSize: "1.02rem", fontWeight: 800, color: "#f8fafc" }}>
            C. Synchronized Progress, Route Distance, Reward &amp; Stall Charts
          </h3>
          <div style={{ fontSize: "0.74rem", color: "#94a3b8" }}>
            Click any chart to seek to that <code>world_step</code>. Each environment <code>world_step</code> executes
            1 joint physics update (or {payload.step_evidence[0]?.dogs.length ?? 3} sequential shared-dog policy transitions
            when <code>discount_timebase=&quot;sequential_shared_dog&quot;</code>).
          </div>
        </div>
        <div style={{ display: "flex", gap: "0.35rem", flexWrap: "wrap" }}>
          <EpistemicBadge kind="recorded" detail="counts, rewards, stall counter" />
          <EpistemicBadge kind="derived" detail="BFS legal-route distances" />
        </div>
      </div>

      {/* Chart 1: Unpenned Sheep Count (sheep) */}
      <div
        style={{
          background: "#090f1d",
          border: "1px solid rgba(148, 163, 184, 0.18)",
          borderRadius: "0.55rem",
          padding: "0.5rem 0.65rem",
        }}
      >
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            fontSize: "0.74rem",
            marginBottom: "0.25rem",
          }}
        >
          <span style={{ fontWeight: 700, color: "#e2e8f0" }}>
            1. Unpenned Sheep Count (unit: sheep, 0..{sheepCount}) [Recorded]
          </span>
          <span style={{ color: "#38bdf8", fontWeight: 700 }}>
            Step {actualWorldStep}: {currentStepEv?.unpenned_count ?? 0} unpenned /{" "}
            {currentStepEv?.penned_count ?? 0} penned
          </span>
        </div>
        <svg
          viewBox="0 0 600 68"
          style={{ width: "100%", height: "68px", cursor: "pointer", display: "block" }}
          onClick={handleSvgClick}
          role="img"
          aria-label="Unpenned sheep count chart"
        >
          {renderIntervalShading(68)}
          {renderPenningAndTruncationMarkers(68)}
          <polyline
            fill="none"
            stroke="#38bdf8"
            strokeWidth={2.2}
            points={sampledSteps
              .map((st) => {
                const x = xOfStep(st.step);
                const y = 60 - (st.unpenned_count / Math.max(1, sheepCount)) * 50;
                return `${x.toFixed(1)},${y.toFixed(1)}`;
              })
              .join(" ")}
          />
          <line x1={playheadX} y1={0} x2={playheadX} y2={68} stroke="#fde047" strokeWidth={2} />
        </svg>
      </div>

      {/* Chart 2: Legal-Route Distance to Pen (grid cells) */}
      <div
        style={{
          background: "#090f1d",
          border: "1px solid rgba(148, 163, 184, 0.18)",
          borderRadius: "0.55rem",
          padding: "0.5rem 0.65rem",
        }}
      >
        <div
          style={{
            display: "flex",
            flexWrap: "wrap",
            justifyContent: "space-between",
            alignItems: "center",
            gap: "0.4rem",
            fontSize: "0.74rem",
            marginBottom: "0.25rem",
          }}
        >
          <span style={{ fontWeight: 700, color: "#e2e8f0" }}>
            2. Legal-Route Distance to Pen Gate (unit: BFS grid cells) [Derived]
          </span>
          <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
            <button
              type="button"
              aria-pressed={showPerSheepDistances}
              onClick={onTogglePerSheepDistances}
              style={{
                padding: "0.14rem 0.5rem",
                borderRadius: "0.35rem",
                fontSize: "0.7rem",
                fontWeight: 700,
                cursor: "pointer",
                background: showPerSheepDistances
                  ? "rgba(45, 212, 191, 0.18)"
                  : "rgba(30, 41, 59, 0.7)",
                color: showPerSheepDistances ? "#99f6e4" : "#94a3b8",
                border: "1px solid rgba(45, 212, 191, 0.4)",
              }}
            >
              {showPerSheepDistances ? "✓ Showing Per-Sheep Lines (S0..S5)" : "Show Per-Sheep Lines"}
            </button>
            <span style={{ color: "#2dd4bf", fontWeight: 700 }}>
              Mean: {currentStepEv?.average_distance_to_pen.toFixed(1) ?? "0.0"} cells
            </span>
          </div>
        </div>
        <svg
          viewBox="0 0 600 92"
          style={{ width: "100%", height: "92px", cursor: "pointer", display: "block" }}
          onClick={handleSvgClick}
          role="img"
          aria-label="Legal-route distance to pen chart"
        >
          {renderIntervalShading(92)}
          {renderPenningAndTruncationMarkers(92)}
          {/* Per-sheep route distance lines */}
          {showPerSheepDistances
            ? Array.from({ length: sheepCount }).map((_, sIdx) => (
                <polyline
                  key={`sheep-dist-line-${sIdx}`}
                  fill="none"
                  stroke={SHEEP_LINE_COLORS[sIdx % SHEEP_LINE_COLORS.length]}
                  strokeOpacity={0.55}
                  strokeWidth={1.2}
                  points={sampledSteps
                    .map((st) => {
                      const x = xOfStep(st.step);
                      const sRec = st.sheep.find((s) => s.sheep_index === sIdx);
                      const dist = sRec?.route_distance ?? 0;
                      const y = 84 - (dist / Math.max(1, maxRouteDist)) * 74;
                      return `${x.toFixed(1)},${y.toFixed(1)}`;
                    })
                    .join(" ")}
                />
              ))
            : null}
          {/* Aggregate mean unpenned route distance */}
          <polyline
            fill="none"
            stroke="#f8fafc"
            strokeWidth={2.2}
            points={sampledSteps
              .map((st) => {
                const x = xOfStep(st.step);
                const y = 84 - (st.average_distance_to_pen / Math.max(1, maxRouteDist)) * 74;
                return `${x.toFixed(1)},${y.toFixed(1)}`;
              })
              .join(" ")}
          />
          <line x1={playheadX} y1={0} x2={playheadX} y2={92} stroke="#fde047" strokeWidth={2} />
        </svg>
        <div
          style={{
            display: "flex",
            flexWrap: "wrap",
            justifyContent: "space-between",
            gap: "0.4rem",
            fontSize: "0.69rem",
            color: "#94a3b8",
            marginTop: "0.2rem",
          }}
        >
          <span>
            White bold = unpenned mean. Colored lines = per-sheep BFS route distance (prevents aggregate mean jumps at penning transitions from hiding stragglers).
          </span>
          {currentStepEv ? (
            <span>
              {currentStepEv.sheep
                .map(
                  (s) =>
                    `S${s.sheep_index}: ${s.penned ? "penned" : `${s.route_distance.toFixed(0)}c`}`,
                )
                .join(" | ")}
            </span>
          ) : null}
        </div>
      </div>

      {/* Chart 3: Step Reward & Components */}
      <div
        style={{
          background: "#090f1d",
          border: "1px solid rgba(148, 163, 184, 0.18)",
          borderRadius: "0.55rem",
          padding: "0.5rem 0.65rem",
        }}
      >
        <div
          style={{
            display: "flex",
            flexWrap: "wrap",
            justifyContent: "space-between",
            alignItems: "center",
            gap: "0.4rem",
            fontSize: "0.74rem",
            marginBottom: "0.25rem",
          }}
        >
          <span style={{ fontWeight: 700, color: "#e2e8f0" }}>
            3. Step Reward &amp; Components (unit: reward / world step) [Recorded]
          </span>
          <div style={{ display: "flex", alignItems: "center", gap: "0.45rem" }}>
            <label htmlFor="reward-comp-select" style={{ fontSize: "0.7rem", color: "#cbd5e1" }}>
              Component:
            </label>
            <select
              id="reward-comp-select"
              aria-label="Reward component selector"
              value={selectedRewardComponent}
              onChange={(e) => onSelectRewardComponent(e.target.value)}
              style={{
                padding: "0.15rem 0.45rem",
                borderRadius: "0.35rem",
                background: "#0f172a",
                color: "#f8fafc",
                border: "1px solid rgba(148, 163, 184, 0.3)",
                fontSize: "0.72rem",
              }}
            >
              {payload.available_reward_components.map((comp) => (
                <option key={comp} value={comp}>
                  {comp}
                </option>
              ))}
            </select>
            <span style={{ color: "#fbbf24", fontWeight: 700 }}>
              {(currentStepEv?.reward[selectedRewardComponent] ?? 0) >= 0 ? "+" : ""}
              {(currentStepEv?.reward[selectedRewardComponent] ?? 0).toFixed(4)}
            </span>
          </div>
        </div>
        <svg
          viewBox="0 0 600 76"
          style={{ width: "100%", height: "76px", cursor: "pointer", display: "block" }}
          onClick={handleSvgClick}
          role="img"
          aria-label="Reward component chart"
        >
          {renderIntervalShading(76)}
          {renderPenningAndTruncationMarkers(76)}
          {/* Zero line */}
          {(() => {
            const zeroY =
              68 - ((0 - rewardBounds.minR) / rewardBounds.span) * 58;
            return (
              <line
                x1={0}
                y1={zeroY}
                x2={600}
                y2={zeroY}
                stroke="rgba(148, 163, 184, 0.35)"
                strokeDasharray="3 3"
              />
            );
          })()}
          <polyline
            fill="none"
            stroke="#fbbf24"
            strokeWidth={1.8}
            points={sampledSteps
              .map((st) => {
                const x = xOfStep(st.step);
                const val = st.reward[selectedRewardComponent] ?? 0;
                const y = 68 - ((val - rewardBounds.minR) / rewardBounds.span) * 58;
                return `${x.toFixed(1)},${y.toFixed(1)}`;
              })
              .join(" ")}
          />
          <line x1={playheadX} y1={0} x2={playheadX} y2={76} stroke="#fde047" strokeWidth={2} />
        </svg>
      </div>

      {/* Chart 4: Stall / No-Progress Counter (no_progress_steps) */}
      <div
        style={{
          background: "#090f1d",
          border: "1px solid rgba(148, 163, 184, 0.18)",
          borderRadius: "0.55rem",
          padding: "0.5rem 0.65rem",
        }}
      >
        <div
          style={{
            display: "flex",
            flexWrap: "wrap",
            justifyContent: "space-between",
            alignItems: "center",
            fontSize: "0.74rem",
            marginBottom: "0.25rem",
          }}
        >
          <span style={{ fontWeight: 700, color: "#e2e8f0" }}>
            4. Stall Counter <code>no_progress_steps</code> (unit: world steps) [Recorded]
          </span>
          <span style={{ color: "#f87171", fontWeight: 700 }}>
            Step {actualWorldStep}: {currentStepEv?.no_progress_steps ?? 0} steps (
            {noProgressResets.length} jitter reset(s) without new best route distance)
          </span>
        </div>
        <svg
          viewBox="0 0 600 68"
          style={{ width: "100%", height: "68px", cursor: "pointer", display: "block" }}
          onClick={handleSvgClick}
          role="img"
          aria-label="Stall counter no_progress_steps chart"
        >
          {renderIntervalShading(68)}
          {renderPenningAndTruncationMarkers(68)}
          <polyline
            fill="none"
            stroke="#f87171"
            strokeWidth={1.8}
            points={sampledSteps
              .map((st) => {
                const x = xOfStep(st.step);
                const y = 60 - (st.no_progress_steps / Math.max(1, maxNoProgress)) * 50;
                return `${x.toFixed(1)},${y.toFixed(1)}`;
              })
              .join(" ")}
          />
          <line x1={playheadX} y1={0} x2={playheadX} y2={68} stroke="#fde047" strokeWidth={2} />
        </svg>
      </div>
    </div>
  );
}

/* ============================================================================
 * Section D: Entity Inspector (Dog & Sheep Pre-Step vs Post-Step Evidence)
 * ========================================================================== */

function EntityInspectorCard({
  payload,
  stepEvidence,
  selectedEntity,
  onSelectEntity,
  obsSearchQuery,
  onChangeObsSearchQuery,
}: {
  payload: TemporalDiagnosticPayload;
  stepEvidence: TemporalStepEvidence | null;
  selectedEntity: SelectedFieldEntity;
  onSelectEntity: (entity: SelectedFieldEntity) => void;
  obsSearchQuery: string;
  onChangeObsSearchQuery: (q: string) => void;
}) {
  if (!stepEvidence) {
    return (
      <div className="panel" style={{ padding: "1rem", borderRadius: "0.85rem" }}>
        No step evidence available at current playhead.
      </div>
    );
  }

  const selectedDog: StepDogEvidence | null =
    selectedEntity.type === "dog"
      ? stepEvidence.dogs.find((d) => d.dog_index === selectedEntity.index) ??
        stepEvidence.dogs[0] ??
        null
      : null;

  const selectedSheep: StepSheepEvidence | null =
    selectedEntity.type === "sheep"
      ? stepEvidence.sheep.find((s) => s.sheep_index === selectedEntity.index) ??
        stepEvidence.sheep[0] ??
        null
      : null;

  return (
    <div
      className="panel"
      aria-label="Entity inspector"
      style={{
        padding: "0.95rem 1.1rem",
        background: "rgba(15, 23, 42, 0.85)",
        border: "1px solid rgba(148, 163, 184, 0.22)",
        borderRadius: "0.85rem",
        display: "flex",
        flexDirection: "column",
        gap: "0.8rem",
      }}
    >
      <div
        style={{
          display: "flex",
          flexWrap: "wrap",
          alignItems: "center",
          justifyContent: "space-between",
          gap: "0.5rem",
        }}
      >
        <div>
          <h3 style={{ margin: 0, fontSize: "1.02rem", fontWeight: 800, color: "#f8fafc" }}>
            D. Synchronized Entity Inspector @ World Step {stepEvidence.step}
          </h3>
          <div style={{ fontSize: "0.74rem", color: "#94a3b8" }}>
            Click any Dog or Sheep below (or directly on the field) to inspect its{" "}
            <strong>pre-step decision</strong> and <strong>post-step result</strong>. Unavailable signals
            explicitly state <em>Not recorded</em> rather than <code>0</code>.
          </div>
        </div>
        {stepEvidence.telemetry_truncated ? (
          <EpistemicBadge kind="truncated" detail="Step telemetry capture limit reached" />
        ) : (stepEvidence as any).telemetry_omitted ? (
          <EpistemicBadge
            kind="omitted"
            detail={
              (stepEvidence as any).telemetry_omitted_range
                ? `Telemetry omitted (windowed capture: ${(stepEvidence as any).telemetry_omitted_range.replace("steps_", "steps ").replace("_to_", " to ")})`
                : "Telemetry omitted (windowed capture)"
            }
          />
        ) : null}
      </div>

      {/* Entity Selection Tabs */}
      <div
        role="tablist"
        aria-label="Inspect dog or sheep"
        style={{ display: "flex", flexWrap: "wrap", gap: "0.35rem" }}
      >
        {stepEvidence.dogs.map((d) => {
          const active = selectedEntity.type === "dog" && selectedEntity.index === d.dog_index;
          return (
            <button
              key={`inspect-dog-${d.dog_index}`}
              type="button"
              role="tab"
              aria-selected={active}
              onClick={() => onSelectEntity({ type: "dog", index: d.dog_index })}
              style={{
                padding: "0.3rem 0.65rem",
                borderRadius: "0.45rem",
                fontSize: "0.76rem",
                fontWeight: 700,
                cursor: "pointer",
                background: active ? "rgba(56, 189, 248, 0.22)" : "rgba(30, 41, 59, 0.75)",
                color: active ? "#fef08a" : "#e2e8f0",
                border: active
                  ? "2px solid #fde047"
                  : `1px solid ${dogColor(d.dog_index)}`,
              }}
            >
              🐕 Dog {d.dog_index + 1} (D{d.dog_index} · {d.assigned_role})
            </button>
          );
        })}
        {stepEvidence.sheep.map((s) => {
          const active = selectedEntity.type === "sheep" && selectedEntity.index === s.sheep_index;
          return (
            <button
              key={`inspect-sheep-${s.sheep_index}`}
              type="button"
              role="tab"
              aria-selected={active}
              onClick={() => onSelectEntity({ type: "sheep", index: s.sheep_index })}
              style={{
                padding: "0.3rem 0.65rem",
                borderRadius: "0.45rem",
                fontSize: "0.76rem",
                fontWeight: 700,
                cursor: "pointer",
                background: active ? "rgba(45, 212, 191, 0.22)" : "rgba(30, 41, 59, 0.75)",
                color: active ? "#fef08a" : s.penned ? "#86efac" : "#e2e8f0",
                border: active
                  ? "2px solid #fde047"
                  : "1px solid rgba(148, 163, 184, 0.3)",
              }}
            >
              🐑 Sheep S{s.sheep_index} ({s.penned ? "Penned" : `${s.route_distance.toFixed(0)}c`})
            </button>
          );
        })}
      </div>

      {/* DOG INSPECTOR */}
      {selectedDog ? (
        <div style={{ display: "flex", flexDirection: "column", gap: "0.75rem" }}>
          {/* 1. Pre-Step Decision Boundary */}
          <div
            style={{
              padding: "0.75rem",
              borderRadius: "0.6rem",
              background: "rgba(9, 15, 29, 0.85)",
              border: "1px solid rgba(56, 189, 248, 0.28)",
              display: "flex",
              flexDirection: "column",
              gap: "0.55rem",
            }}
          >
            <div
              style={{
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
                flexWrap: "wrap",
                gap: "0.4rem",
              }}
            >
              <strong style={{ fontSize: "0.82rem", color: "#bae6fd" }}>
                1. Pre-Step Policy Decision Boundary (Before Step {stepEvidence.step} Executes)
              </strong>
              <EpistemicBadge
                kind={
                  selectedDog.decision?.mask_type === "actual_decision_time"
                    ? "recorded"
                    : "estimated"
                }
                detail={selectedDog.decision?.mask_type ?? "historical_approximate"}
              />
            </div>

            <div
              style={{
                display: "grid",
                gridTemplateColumns: "repeat(auto-fit, minmax(195px, 1fr))",
                gap: "0.5rem",
                fontSize: "0.76rem",
              }}
            >
              <div>
                <div style={{ color: "#94a3b8", fontSize: "0.7rem" }}>Pre-Step Position</div>
                <div style={{ fontWeight: 700, color: "#f8fafc" }}>
                  {selectedDog.pre_step_position ? (
                    <>
                      ({selectedDog.pre_step_position.x}, {selectedDog.pre_step_position.y}){" "}
                      <EpistemicBadge kind="recorded" />
                    </>
                  ) : (
                    <span style={{ color: "#cbd5e1" }}>
                      Not recorded (prior step frame not present)
                    </span>
                  )}
                </div>
              </div>

              <div>
                <div style={{ color: "#94a3b8", fontSize: "0.7rem" }}>Selected Action</div>
                <div style={{ fontWeight: 700, color: "#f8fafc" }}>
                  <code>{selectedDog.decision?.selected_action ?? "Not recorded"}</code>{" "}
                  <EpistemicBadge
                    kind="recorded"
                    detail={
                      stepEvidence.has_recorded_step_telemetry
                        ? "pre-replacement"
                        : "frame actions[]"
                    }
                  />
                </div>
              </div>

              <div>
                <div style={{ color: "#94a3b8", fontSize: "0.7rem" }}>Executed Action</div>
                <div style={{ fontWeight: 700, color: "#f8fafc" }}>
                  <code>{selectedDog.decision?.executed_action ?? "Not recorded"}</code>{" "}
                  <EpistemicBadge kind="recorded" />
                </div>
              </div>

              <div>
                <div style={{ color: "#94a3b8", fontSize: "0.7rem" }}>Action Replacement Reason</div>
                <div style={{ fontWeight: 700, color: "#f8fafc" }}>
                  {selectedDog.decision?.replacement_reason ? (
                    <span style={{ color: "#fbbf24" }}>
                      {selectedDog.decision.replacement_reason}
                    </span>
                  ) : stepEvidence.has_recorded_step_telemetry ? (
                    <span style={{ color: "#86efac" }}>None (executed as selected)</span>
                  ) : (
                    <span style={{ color: "#cbd5e1" }}>
                      Not recorded (historical replay stores executed action only)
                    </span>
                  )}
                </div>
              </div>

              <div>
                <div style={{ color: "#94a3b8", fontSize: "0.7rem" }}>
                  Scripted Role &amp; Target (TeamStrategy)
                </div>
                <div style={{ fontWeight: 700, color: "#f8fafc" }}>
                  <code>{selectedDog.assigned_role}</code> → ({selectedDog.target.x},{" "}
                  {selectedDog.target.y})
                  {selectedDog.target_sheep_index !== null
                    ? ` [focus S${selectedDog.target_sheep_index}]`
                    : ""}{" "}
                  <EpistemicBadge
                    kind={
                      selectedDog.guidance_certainty.startsWith("recorded_")
                        ? "recorded"
                        : "estimated"
                    }
                    detail={selectedDog.guidance_certainty}
                  />
                </div>
              </div>

              <div>
                <div style={{ color: "#94a3b8", fontSize: "0.7rem" }}>Policy Value Estimate V(s)</div>
                <div style={{ fontWeight: 700, color: "#f8fafc" }}>
                  {selectedDog.decision?.value_estimate !== null &&
                  selectedDog.decision?.value_estimate !== undefined ? (
                    <>
                      {selectedDog.decision.value_estimate.toFixed(4)}{" "}
                      <EpistemicBadge
                        kind={
                          selectedDog.decision.value_estimate_source ===
                          "recomputed_diagnostic_forward_pass"
                            ? "recomputed"
                            : "recorded"
                        }
                        detail={selectedDog.decision.value_estimate_source}
                      />
                    </>
                  ) : (
                    <span style={{ color: "#cbd5e1" }}>
                      {(stepEvidence as any).telemetry_omitted
                        ? `Telemetry omitted (windowed capture${
                            (stepEvidence as any).telemetry_omitted_range
                              ? `: ${(stepEvidence as any).telemetry_omitted_range.replace("steps_", "steps ").replace("_to_", " to ")}`
                              : ""
                          })`
                        : stepEvidence.telemetry_truncated
                        ? "Not recorded (telemetry truncated at step limit)"
                        : "Not recorded (critic value estimate not captured in this replay)"}
                    </span>
                  )}
                </div>
              </div>
            </div>

            {/* Decision Action Mask Grid */}
            <div>
              <div
                style={{
                  fontSize: "0.72rem",
                  fontWeight: 700,
                  color: "#cbd5e1",
                  marginBottom: "0.3rem",
                }}
              >
                Decision Action Mask ({payload.action_order.length} actions — source:{" "}
                <code>{selectedDog.decision?.mask_type ?? stepEvidence.mask_type}</code>):
              </div>
              <div style={{ display: "flex", flexWrap: "wrap", gap: "0.3rem" }}>
                {payload.action_order.map((act) => {
                  const allowed = selectedDog.decision?.decision_action_mask?.[act];
                  const isExecuted = selectedDog.decision?.executed_action === act;
                  return (
                    <span
                      key={`mask-${act}`}
                      style={{
                        padding: "0.18rem 0.48rem",
                        borderRadius: "0.35rem",
                        fontSize: "0.71rem",
                        fontWeight: isExecuted ? 800 : 600,
                        background:
                          allowed === false
                            ? "rgba(127, 29, 29, 0.35)"
                            : isExecuted
                              ? "rgba(56, 189, 248, 0.25)"
                              : "rgba(30, 41, 59, 0.75)",
                        color:
                          allowed === false
                            ? "#fca5a5"
                            : isExecuted
                              ? "#fef08a"
                              : "#bbf7d0",
                        border: isExecuted
                          ? "1.5px solid #fde047"
                          : allowed === false
                            ? "1px solid rgba(248, 113, 113, 0.45)"
                            : "1px solid rgba(74, 222, 128, 0.35)",
                      }}
                    >
                      {allowed === false ? "✗ " : "✓ "}
                      {act}
                      {isExecuted ? " ★" : ""}
                    </span>
                  );
                })}
              </div>
            </div>

            {/* Policy Action Probabilities */}
            <div>
              <div
                style={{
                  fontSize: "0.72rem",
                  fontWeight: 700,
                  color: "#cbd5e1",
                  marginBottom: "0.25rem",
                  display: "flex",
                  alignItems: "center",
                  gap: "0.4rem",
                  flexWrap: "wrap",
                }}
              >
                <span>Policy Action Probabilities:</span>
                {selectedDog.decision?.action_probabilities ? (
                  <EpistemicBadge
                    kind={
                      selectedDog.decision.action_probabilities_source ===
                      "recomputed_diagnostic_forward_pass"
                        ? "recomputed"
                        : "recorded"
                    }
                    detail={selectedDog.decision.action_probabilities_source}
                  />
                ) : (
                  <EpistemicBadge
                    kind={
                      (stepEvidence as any).telemetry_omitted
                        ? "omitted"
                        : stepEvidence.telemetry_truncated
                        ? "truncated"
                        : "not_recorded"
                    }
                    detail={
                      (stepEvidence as any).telemetry_omitted
                        ? (stepEvidence as any).telemetry_omitted_range
                          ? (stepEvidence as any).telemetry_omitted_range.replace("steps_", "steps ").replace("_to_", " to ")
                          : "windowed capture"
                        : undefined
                    }
                  />
                )}
              </div>
              {selectedDog.decision?.action_probabilities ? (
                <div
                  style={{
                    display: "grid",
                    gridTemplateColumns: "repeat(auto-fit, minmax(120px, 1fr))",
                    gap: "0.3rem",
                  }}
                >
                  {payload.action_order.map((act) => {
                    const p = selectedDog.decision?.action_probabilities?.[act];
                    const pct = p !== undefined ? p * 100 : null;
                    return (
                      <div
                        key={`prob-${act}`}
                        style={{
                          padding: "0.2rem 0.4rem",
                          background: "rgba(15, 23, 42, 0.9)",
                          border: "1px solid rgba(148, 163, 184, 0.2)",
                          borderRadius: "0.35rem",
                          fontSize: "0.7rem",
                        }}
                      >
                        <div style={{ display: "flex", justifyContent: "space-between" }}>
                          <code>{act}</code>
                          <strong>{pct !== null ? `${pct.toFixed(1)}%` : "Not recorded"}</strong>
                        </div>
                      </div>
                    );
                  })}
                </div>
              ) : (
                <div style={{ fontSize: "0.74rem", color: "#cbd5e1" }}>
                  {(stepEvidence as any).telemetry_omitted
                    ? `Telemetry omitted (windowed capture${
                        (stepEvidence as any).telemetry_omitted_range
                          ? `: ${(stepEvidence as any).telemetry_omitted_range.replace("steps_", "steps ").replace("_to_", " to ")}`
                          : ""
                      })`
                    : `Not recorded — ${
                        stepEvidence.telemetry_truncated
                          ? "telemetry capture limit was reached before this step."
                          : "policy action distribution was not captured in this historical recording."
                      }`}
                </div>
              )}
            </div>

            {/* Labeled Observation Fields */}
            <div>
              <div
                style={{
                  display: "flex",
                  flexWrap: "wrap",
                  alignItems: "center",
                  justifyContent: "space-between",
                  gap: "0.4rem",
                  marginBottom: "0.25rem",
                }}
              >
                <span style={{ fontSize: "0.72rem", fontWeight: 700, color: "#cbd5e1" }}>
                  Observation Vector Fields:{" "}
                  {selectedDog.decision?.observation_labeled?.available ? (
                    <EpistemicBadge
                      kind="recorded"
                      detail={
                        selectedDog.decision.observation_labeled.schema_name ?? "observation"
                      }
                    />
                  ) : (
                    <EpistemicBadge
                      kind={
                        (stepEvidence as any).telemetry_omitted
                          ? "omitted"
                          : stepEvidence.telemetry_truncated
                          ? "truncated"
                          : "not_recorded"
                      }
                      detail={
                        (stepEvidence as any).telemetry_omitted
                          ? (stepEvidence as any).telemetry_omitted_range
                            ? (stepEvidence as any).telemetry_omitted_range.replace("steps_", "steps ").replace("_to_", " to ")
                            : "windowed capture"
                          : undefined
                      }
                    />
                  )}
                </span>
                {selectedDog.decision?.observation_labeled?.available ? (
                  <input
                    type="search"
                    aria-label="Filter observation features"
                    placeholder="Filter observation features..."
                    value={obsSearchQuery}
                    onChange={(e) => onChangeObsSearchQuery(e.target.value)}
                    style={{
                      padding: "0.18rem 0.45rem",
                      borderRadius: "0.35rem",
                      background: "#0f172a",
                      color: "#f8fafc",
                      border: "1px solid rgba(148, 163, 184, 0.3)",
                      fontSize: "0.71rem",
                    }}
                  />
                ) : null}
              </div>
              {selectedDog.decision?.observation_labeled?.available ? (
                <div
                  style={{
                    maxHeight: "170px",
                    overflowY: "auto",
                    border: "1px solid rgba(148, 163, 184, 0.2)",
                    borderRadius: "0.4rem",
                    padding: "0.35rem",
                    display: "grid",
                    gridTemplateColumns: "repeat(auto-fill, minmax(190px, 1fr))",
                    gap: "0.25rem",
                    fontSize: "0.7rem",
                  }}
                >
                  {selectedDog.decision.observation_labeled.fields
                    .filter(
                      (f) =>
                        !obsSearchQuery.trim() ||
                        f.label.toLowerCase().includes(obsSearchQuery.toLowerCase()) ||
                        f.key.toLowerCase().includes(obsSearchQuery.toLowerCase()) ||
                        f.group.toLowerCase().includes(obsSearchQuery.toLowerCase()),
                    )
                    .map((f) => (
                      <div
                        key={`obs-field-${f.index}`}
                        style={{
                          padding: "0.18rem 0.35rem",
                          background: "rgba(30, 41, 59, 0.55)",
                          borderRadius: "0.3rem",
                          display: "flex",
                          justifyContent: "space-between",
                          gap: "0.35rem",
                        }}
                      >
                        <span style={{ color: "#cbd5e1" }} title={`${f.group} · ${f.key}`}>
                          [{f.index}] {f.label}
                        </span>
                        <strong style={{ color: "#f8fafc" }}>{f.value.toFixed(3)}</strong>
                      </div>
                    ))}
                </div>
              ) : (
                <div style={{ fontSize: "0.74rem", color: "#cbd5e1" }}>
                  {selectedDog.decision?.observation_labeled?.reason ??
                    "Not recorded (raw policy observation vectors were not captured in this replay)."}
                </div>
              )}
            </div>
          </div>

          {/* 2. Post-Step Result */}
          <div
            style={{
              padding: "0.75rem",
              borderRadius: "0.6rem",
              background: "rgba(9, 15, 29, 0.85)",
              border: "1px solid rgba(45, 212, 191, 0.28)",
              display: "grid",
              gridTemplateColumns: "repeat(auto-fit, minmax(185px, 1fr))",
              gap: "0.5rem",
              fontSize: "0.76rem",
            }}
          >
            <div style={{ gridColumn: "1 / -1", fontWeight: 800, color: "#5eead4", fontSize: "0.82rem" }}>
              2. Post-Step Result &amp; Descriptive Geometry (After Step {stepEvidence.step})
            </div>
            <div>
              <div style={{ color: "#94a3b8", fontSize: "0.7rem" }}>Post-Step Position</div>
              <div style={{ fontWeight: 700, color: "#f8fafc" }}>
                ({selectedDog.post_step_position.x}, {selectedDog.post_step_position.y}){" "}
                <EpistemicBadge kind="recorded" />
              </div>
            </div>
            <div>
              <div style={{ color: "#94a3b8", fontSize: "0.7rem" }}>Step Displacement</div>
              <div style={{ fontWeight: 700, color: "#f8fafc" }}>
                {selectedDog.displacement ? (
                  <>
                    dx={selectedDog.displacement.dx >= 0 ? "+" : ""}
                    {selectedDog.displacement.dx}, dy={selectedDog.displacement.dy >= 0 ? "+" : ""}
                    {selectedDog.displacement.dy} ({selectedDog.displacement.distance.toFixed(2)} cells){" "}
                    <EpistemicBadge kind="derived" />
                  </>
                ) : (
                  <span style={{ color: "#cbd5e1" }}>
                    Not available (missing contiguous prior step)
                  </span>
                )}
              </div>
            </div>
            <div>
              <div style={{ color: "#94a3b8", fontSize: "0.7rem" }}>Distance to Target</div>
              <div style={{ fontWeight: 700, color: "#f8fafc" }}>
                {selectedDog.distance_to_target.toFixed(2)} cells{" "}
                <EpistemicBadge kind="derived" />
              </div>
            </div>
            <div>
              <div style={{ color: "#94a3b8", fontSize: "0.7rem" }}>Nearest Unpenned Sheep</div>
              <div style={{ fontWeight: 700, color: "#f8fafc" }}>
                {selectedDog.nearest_unpenned_sheep_distance !== null ? (
                  <>
                    {selectedDog.nearest_unpenned_sheep_distance.toFixed(2)} cells (
                    {selectedDog.within_sheep_vision ? "inside" : "outside"} vision={payload.sheep_vision})
                  </>
                ) : (
                  "All sheep penned"
                )}
              </div>
            </div>
            <div>
              <div style={{ color: "#94a3b8", fontSize: "0.7rem" }}>
                Descriptive Alignment &amp; Waypoint Side
              </div>
              <div style={{ fontWeight: 700, color: "#f8fafc" }}>
                align={selectedDog.behind_flock_alignment.toFixed(2)} ·{" "}
                {selectedDog.on_gate_or_waypoint_side
                  ? "On waypoint/gate side (descriptive)"
                  : "Driving/flanking side"}
              </div>
            </div>
          </div>
        </div>
      ) : null}

      {/* SHEEP INSPECTOR */}
      {selectedSheep ? (
        <div
          style={{
            padding: "0.75rem",
            borderRadius: "0.6rem",
            background: "rgba(9, 15, 29, 0.85)",
            border: "1px solid rgba(45, 212, 191, 0.3)",
            display: "flex",
            flexDirection: "column",
            gap: "0.65rem",
            fontSize: "0.76rem",
          }}
        >
          <div style={{ fontWeight: 800, color: "#5eead4", fontSize: "0.84rem" }}>
            Sheep S{selectedSheep.sheep_index} — Pre-Step vs. Post-Step State &amp; Nearby Dog Geometry
          </div>
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fit, minmax(185px, 1fr))",
              gap: "0.5rem",
            }}
          >
            <div>
              <div style={{ color: "#94a3b8", fontSize: "0.7rem" }}>Pre-Step Position</div>
              <div style={{ fontWeight: 700, color: "#f8fafc" }}>
                {selectedSheep.pre_step_position ? (
                  <>
                    ({selectedSheep.pre_step_position.x}, {selectedSheep.pre_step_position.y}){" "}
                    <EpistemicBadge kind="recorded" />
                  </>
                ) : (
                  <span style={{ color: "#cbd5e1" }}>
                    Not recorded (no contiguous prior step)
                  </span>
                )}
              </div>
            </div>
            <div>
              <div style={{ color: "#94a3b8", fontSize: "0.7rem" }}>Post-Step Position &amp; Movement</div>
              <div style={{ fontWeight: 700, color: "#f8fafc" }}>
                ({selectedSheep.post_step_position.x}, {selectedSheep.post_step_position.y}){" "}
                {selectedSheep.displacement
                  ? `[dx=${selectedSheep.displacement.dx >= 0 ? "+" : ""}${selectedSheep.displacement.dx}, dy=${selectedSheep.displacement.dy >= 0 ? "+" : ""}${selectedSheep.displacement.dy}]`
                  : ""}{" "}
                <EpistemicBadge kind="recorded" />
              </div>
            </div>
            <div>
              <div style={{ color: "#94a3b8", fontSize: "0.7rem" }}>Penned Status &amp; Personality</div>
              <div style={{ fontWeight: 700, color: "#f8fafc" }}>
                {selectedSheep.penned ? "✓ Penned" : "Unpenned"} ·{" "}
                <code>{selectedSheep.personality}</code>
                {selectedSheep.is_packed ? " (packed)" : ""}{" "}
                <EpistemicBadge kind="recorded" />
              </div>
            </div>
            <div>
              <div style={{ color: "#94a3b8", fontSize: "0.7rem" }}>
                Legal-Route Distance &amp; Derived Waypoint
              </div>
              <div style={{ fontWeight: 700, color: "#f8fafc" }}>
                {selectedSheep.route_distance.toFixed(1)} cells → Waypoint (
                {selectedSheep.waypoint.x}, {selectedSheep.waypoint.y}){" "}
                <EpistemicBadge kind="derived" />
              </div>
            </div>
            <div>
              <div style={{ color: "#94a3b8", fontSize: "0.7rem" }}>Boundary / Fence Proximity</div>
              <div style={{ fontWeight: 700, color: "#f8fafc" }}>
                Fence adjacent: {selectedSheep.closed_fence_adjacent ? "Yes" : "No"} · Outer wall/corner:{" "}
                {selectedSheep.outer_wall_or_corner ? "Yes" : "No"} (margin {selectedSheep.wall_margin}c)
              </div>
            </div>
          </div>

          {/* Nearby Dog Geometry Table */}
          <div>
            <div style={{ fontWeight: 700, color: "#cbd5e1", fontSize: "0.73rem", marginBottom: "0.3rem" }}>
              Nearby Dog Geometry (Descriptive Evidence — proximity/alignment is not proven physical blockage):
            </div>
            <div style={{ overflowX: "auto" }}>
              <table
                style={{
                  width: "100%",
                  borderCollapse: "collapse",
                  fontSize: "0.73rem",
                  textAlign: "left",
                }}
              >
                <thead>
                  <tr style={{ borderBottom: "1px solid rgba(148, 163, 184, 0.25)", color: "#94a3b8" }}>
                    <th style={{ padding: "0.25rem 0.4rem" }}>Dog</th>
                    <th style={{ padding: "0.25rem 0.4rem" }}>Role</th>
                    <th style={{ padding: "0.25rem 0.4rem" }}>Distance</th>
                    <th style={{ padding: "0.25rem 0.4rem" }}>Within Vision ({payload.sheep_vision}c)</th>
                    <th style={{ padding: "0.25rem 0.4rem" }}>Offset (dx, dy)</th>
                    <th style={{ padding: "0.25rem 0.4rem" }}>Waypoint Alignment</th>
                  </tr>
                </thead>
                <tbody>
                  {selectedSheep.nearby_dogs.map((nd) => (
                    <tr
                      key={`nd-${nd.dog_index}`}
                      style={{ borderBottom: "1px solid rgba(148, 163, 184, 0.12)" }}
                    >
                      <td style={{ padding: "0.25rem 0.4rem", fontWeight: 700 }}>
                        D{nd.dog_index} (Dog {nd.dog_index + 1})
                      </td>
                      <td style={{ padding: "0.25rem 0.4rem" }}>
                        <code>{nd.role}</code>
                      </td>
                      <td style={{ padding: "0.25rem 0.4rem" }}>{nd.distance.toFixed(2)} cells</td>
                      <td style={{ padding: "0.25rem 0.4rem" }}>
                        {nd.within_sheep_vision ? "Yes (active)" : "No (outside vision)"}
                      </td>
                      <td style={{ padding: "0.25rem 0.4rem" }}>
                        ({nd.dx_from_sheep >= 0 ? "+" : ""}
                        {nd.dx_from_sheep}, {nd.dy_from_sheep >= 0 ? "+" : ""}
                        {nd.dy_from_sheep})
                      </td>
                      <td style={{ padding: "0.25rem 0.4rem" }}>
                        {nd.alignment_to_waypoint.toFixed(2)}{" "}
                        {nd.on_waypoint_side ? (
                          <span style={{ color: "#fda4af", fontWeight: 700 }}>
                            (on waypoint side)
                          </span>
                        ) : (
                          <span style={{ color: "#86efac" }}>(behind/side)</span>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      ) : null}
    </div>
  );
}

/* ============================================================================
 * Section E: Plain-Language Explanation Card
 * ========================================================================== */

function PlainLanguageIntervalCard({
  interval,
  episodeIntervalNumber,
  onSeekToStep,
}: {
  interval: TemporalInteractionInterval | null;
  episodeIntervalNumber?: number | null;
  onSeekToStep: (step: number) => void;
}) {
  if (!interval) {
    return (
      <div
        className="panel"
        aria-label="Plain-language interval explanation"
        style={{
          padding: "1rem",
          background: "rgba(15, 23, 42, 0.85)",
          border: "1px solid rgba(148, 163, 184, 0.2)",
          borderRadius: "0.85rem",
          color: "#cbd5e1",
          fontSize: "0.82rem",
        }}
      >
        No repeated-interaction or non-progress interval was detected in this episode. You can still
        scrub across the field and inspect every step in the Entity Inspector.
      </div>
    );
  }

  const meta = getClassificationMeta(interval.classification, interval.readable_label);
  const pl = interval.plain_language;

  return (
    <div
      className="panel"
      aria-label="Plain-language interval explanation"
      style={{
        padding: "0.95rem 1.15rem",
        background: "rgba(15, 23, 42, 0.88)",
        border: `1px solid ${meta.border}`,
        borderRadius: "0.85rem",
        display: "flex",
        flexDirection: "column",
        gap: "0.75rem",
        fontSize: "0.8rem",
      }}
    >
      <div
        style={{
          display: "flex",
          flexWrap: "wrap",
          alignItems: "center",
          justifyContent: "space-between",
          gap: "0.5rem",
        }}
      >
        <div>
          <div
            style={{
              fontSize: "0.7rem",
              fontWeight: 800,
              textTransform: "uppercase",
              letterSpacing: "0.06em",
              color: meta.color,
            }}
          >
            E. Plain-Language Interval Explanation · {meta.icon} {meta.readableLabel}
            {episodeIntervalNumber != null ? ` (Interval #${episodeIntervalNumber})` : ""}
          </div>
          <h3 style={{ margin: "0.15rem 0 0", fontSize: "1.05rem", fontWeight: 800, color: "#f8fafc" }}>
            {episodeIntervalNumber != null ? `Interval #${episodeIntervalNumber}: ` : ""}Steps{" "}
            {interval.start_step}..{interval.end_step} ({interval.duration_steps} steps ·{" "}
            {interval.unpenned_sheep_count} unpenned sheep)
          </h3>
        </div>
        <EpistemicBadge
          kind={interval.guidance_certainty.startsWith("recorded_") ? "recorded" : "estimated"}
          detail={interval.guidance_certainty}
        />
      </div>

      {/* 1. What Was Observed */}
      <div
        style={{
          padding: "0.65rem 0.8rem",
          borderRadius: "0.55rem",
          background: "rgba(9, 15, 29, 0.8)",
          border: "1px solid rgba(148, 163, 184, 0.2)",
        }}
      >
        <div style={{ fontWeight: 800, color: "#86efac", marginBottom: "0.2rem" }}>
          1. What Was Observed [Recorded &amp; Derived]
        </div>
        <div style={{ color: "#e2e8f0", lineHeight: 1.45 }}>
          {pl?.what_was_observed ??
            `Steps ${interval.start_step}..${interval.end_step} (${interval.duration_steps} steps) with ${interval.unpenned_sheep_count} unpenned sheep: route distance changed ${interval.avg_route_distance_start.toFixed(1)} -> ${interval.avg_route_distance_end.toFixed(1)} cells.`}
        </div>
      </div>

      {/* Lead-in and Aftermath Windows */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(210px, 1fr))",
          gap: "0.55rem",
        }}
      >
        <div
          style={{
            padding: "0.6rem 0.75rem",
            borderRadius: "0.55rem",
            background: "rgba(56, 189, 248, 0.1)",
            border: "1px solid rgba(56, 189, 248, 0.35)",
          }}
        >
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
              marginBottom: "0.25rem",
            }}
          >
            <strong style={{ color: "#bae6fd", fontSize: "0.76rem" }}>
              Before (Lead-In: {interval.lead_in.start_step}..{interval.lead_in.end_step})
            </strong>
            <button
              type="button"
              onClick={() => onSeekToStep(interval.lead_in.start_step)}
              style={{
                padding: "0.12rem 0.45rem",
                borderRadius: "0.3rem",
                background: "#0284c7",
                color: "#f8fafc",
                border: "none",
                fontSize: "0.69rem",
                fontWeight: 700,
                cursor: "pointer",
              }}
            >
              Seek {interval.lead_in.start_step}
            </button>
          </div>
          <div style={{ fontSize: "0.74rem", color: "#e0f2fe", lineHeight: 1.38 }}>
            {interval.lead_in.summary}
          </div>
        </div>

        <div
          style={{
            padding: "0.6rem 0.75rem",
            borderRadius: "0.55rem",
            background: "rgba(168, 85, 247, 0.1)",
            border: "1px solid rgba(192, 132, 252, 0.35)",
          }}
        >
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
              marginBottom: "0.25rem",
            }}
          >
            <strong style={{ color: "#e9d5ff", fontSize: "0.76rem" }}>
              Afterward (Aftermath: {interval.aftermath.start_step}..{interval.aftermath.end_step})
            </strong>
            <button
              type="button"
              onClick={() => onSeekToStep(interval.aftermath.start_step)}
              style={{
                padding: "0.12rem 0.45rem",
                borderRadius: "0.3rem",
                background: "#7e22ce",
                color: "#f8fafc",
                border: "none",
                fontSize: "0.69rem",
                fontWeight: 700,
                cursor: "pointer",
              }}
            >
              Seek {interval.aftermath.start_step}
            </button>
          </div>
          <div style={{ fontSize: "0.74rem", color: "#f3e8ff", lineHeight: 1.38 }}>
            {interval.aftermath.summary}
          </div>
        </div>
      </div>

      {/* 2. Candidate Explanation */}
      <div
        style={{
          padding: "0.65rem 0.8rem",
          borderRadius: "0.55rem",
          background: "rgba(245, 158, 11, 0.1)",
          border: "1px solid rgba(251, 191, 36, 0.35)",
        }}
      >
        <div style={{ fontWeight: 800, color: "#fde68a", marginBottom: "0.2rem" }}>
          2. Candidate Explanation (Observational Hypothesis — Not Causal Proof)
        </div>
        <div style={{ color: "#fef3c7", lineHeight: 1.45 }}>
          {pl?.candidate_explanation ??
            `Candidate classification: ${meta.readableLabel}. Candidate mechanisms: ${interval.candidate_mechanisms.join(", ") || "none"}.`}
        </div>
        <div
          style={{
            marginTop: "0.45rem",
            paddingTop: "0.4rem",
            borderTop: "1px solid rgba(251, 191, 36, 0.25)",
            color: "#bae6fd",
            fontSize: "0.75rem",
          }}
        >
          💡{" "}
          {pl?.moving_away_note ??
            "Why moving away from the pen can be productive: Dogs often need to release pressure, circle wide around a fence/corner, or travel away from the gate to collect a straggler before driving sheep into the pen."}
        </div>
      </div>

      {/* 3. Supporting Evidence */}
      <div>
        <div style={{ fontWeight: 800, color: "#67e8f9", marginBottom: "0.25rem" }}>
          3. Supporting Evidence ({interval.supporting_evidence.length} items)
        </div>
        <ul style={{ margin: 0, paddingLeft: "1.15rem", color: "#e2e8f0", lineHeight: 1.42, fontSize: "0.75rem" }}>
          {interval.supporting_evidence.map((ev, idx) => (
            <li key={`sup-ev-${idx}`} style={{ marginBottom: "0.2rem" }}>
              {ev}
            </li>
          ))}
        </ul>
      </div>

      {/* 4. Missing Evidence & Alternatives */}
      <div>
        <div style={{ fontWeight: 800, color: "#fda4af", marginBottom: "0.25rem" }}>
          4. Missing Evidence &amp; Alternative Explanations
        </div>
        <ul style={{ margin: 0, paddingLeft: "1.15rem", color: "#fecdd3", lineHeight: 1.42, fontSize: "0.75rem" }}>
          {(pl?.missing_evidence_and_alternatives ?? interval.alternative_explanations).map(
            (alt, idx) => (
              <li key={`alt-ev-${idx}`} style={{ marginBottom: "0.2rem" }}>
                {alt}
              </li>
            ),
          )}
        </ul>
      </div>

      {/* 5. What Would Require a Controlled Experiment */}
      <div
        style={{
          padding: "0.6rem 0.75rem",
          borderRadius: "0.55rem",
          background: "rgba(30, 41, 59, 0.75)",
          border: "1px solid rgba(148, 163, 184, 0.25)",
          fontSize: "0.75rem",
          color: "#cbd5e1",
          lineHeight: 1.4,
        }}
      >
        <strong style={{ color: "#f8fafc" }}>5. What Requires a Controlled Experiment: </strong>
        {pl?.controlled_experiment_required ??
          "To prove whether a candidate mechanism caused this stall, compare an unchanged baseline against a single isolated change under identical training/evaluation seeds and episode budgets."}
      </div>
    </div>
  );
}

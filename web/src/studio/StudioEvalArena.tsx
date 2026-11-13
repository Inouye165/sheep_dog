import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  fetchReplayById,
  loadRecentEvaluations,
  loadReplay,
  runReplay,
} from "../lib/api";
import {
  analyzePerSeedReliability,
  getCheckpointStage,
} from "../lib/chartPipeline";
import type {
  CheckpointEntry,
  EvaluationSummaryPayload,
} from "../state/types";
import { StudioFieldCanvas, snapshotFromReplay, type AnyReplayPayload } from "./StudioFieldCanvas";

const DEFAULT_BENCHMARK_SEEDS = [11, 23, 37, 41, 53, 59, 61, 67, 71, 73];
const MAX_ARENA_REPLAY_CACHE_ENTRIES = 40;

export interface UnifiedSeedEpisode {
  seed: number;
  success: boolean;
  steps: number;
  total_reward: number;
  sheep_penned: number;
  total_sheep?: number;
  stop_reason?: string;
  replay_id?: string | number | null;
  replay_path?: string | null;
  has_replay: boolean;
}

export interface UnifiedEvaluation {
  key: string;
  evaluation_id?: string | number | null;
  checkpoint_episode: number;
  stage: number;
  recorded_at?: string | null;
  success_rate_pct: number;
  average_reward: number;
  average_sheep_penned: number;
  average_completion_steps: number | null;
  is_best: boolean;
  episodes: UnifiedSeedEpisode[];
}

export interface StudioEvalArenaProps {
  checkpoints: CheckpointEntry[];
  effectiveStage: number;
  bestCheckpointEpisode?: number | null;
  initialCheckpointEpisode?: number | null;
  onSelectSeedForMainPlayer?: (replay: AnyReplayPayload, checkpointEpisode: number, seed: number) => void;
}

function normalizeReplayPath(p?: string | null): string | null {
  if (!p) return null;
  return p.replace(/\\/g, "/");
}

export const StudioEvalArena: React.FC<StudioEvalArenaProps> = ({
  checkpoints,
  effectiveStage,
  bestCheckpointEpisode,
  initialCheckpointEpisode,
  onSelectSeedForMainPlayer,
}) => {
  const [apiEvaluations, setApiEvaluations] = useState<EvaluationSummaryPayload[]>([]);
  const [stageFilter, setStageFilter] = useState<"current" | "all">("current");
  const [selectedEvalKey, setSelectedEvalKey] = useState<string | null>(null);
  const [pinnedEval, setPinnedEval] = useState<boolean>(false);
  const [seedFilter, setSeedFilter] = useState<"all" | "passed" | "failed">("all");
  const [arenaLayout, setArenaLayout] = useState<"grid" | "focus" | "matrix">("grid");
  const [focusedSeed, setFocusedSeed] = useState<number>(11);

  // Playback state
  const [currentStep, setCurrentStep] = useState<number>(0);
  const [isPlaying, setIsPlaying] = useState<boolean>(true);
  const [playbackSpeed, setPlaybackSpeed] = useState<number>(2);
  const [autoLoop, setAutoLoop] = useState<boolean>(true);
  const [showVectors, setShowVectors] = useState<boolean>(false);

  // Loaded replays keyed by `${evalKey}:${seed}`
  const [replaysBySeed, setReplaysBySeed] = useState<Record<number, AnyReplayPayload | null>>({});
  const [loadingSeeds, setLoadingSeeds] = useState<Record<number, boolean>>({});
  const [simulatingAll, setSimulatingAll] = useState<boolean>(false);
  const replayCacheRef = useRef<Map<string, AnyReplayPayload>>(new Map());

  const cacheArenaReplay = useCallback((key: string, payload: AnyReplayPayload) => {
    const map = replayCacheRef.current;
    if (map.has(key)) {
      map.delete(key);
    } else if (map.size >= MAX_ARENA_REPLAY_CACHE_ENTRIES) {
      const oldestKey = map.keys().next().value;
      if (oldestKey !== undefined) {
        map.delete(oldestKey);
      }
    }
    map.set(key, payload);
  }, []);

  // Fetch recent evaluations from API
  const fetchEvaluations = useCallback(async () => {
    try {
      const stageParam = stageFilter === "current" ? effectiveStage : undefined;
      const list = await loadRecentEvaluations(25, stageParam);
      if (Array.isArray(list)) {
        setApiEvaluations(list);
      }
    } catch {
      // Fallback to checkpoint index records silently
    }
  }, [effectiveStage, stageFilter]);

  useEffect(() => {
    fetchEvaluations();
    const timer = window.setInterval(fetchEvaluations, 6000);
    return () => window.clearInterval(timer);
  }, [fetchEvaluations]);

  // Merge API evaluations + CheckpointIndex entries into a unified list sorted newest first
  const unifiedEvaluations = useMemo<UnifiedEvaluation[]>(() => {
    const byStageEp = new Map<string, UnifiedEvaluation>();

    // First populate from checkpoints
    for (const cp of checkpoints) {
      const cpStage = getCheckpointStage(cp);
      if (stageFilter === "current" && cpStage !== effectiveStage) continue;

      const rawRate = cp.success_rate ?? 0;
      const ratePct = rawRate <= 1.0 ? rawRate * 100 : rawRate;
      const records = cp.records ?? [];

      const episodes: UnifiedSeedEpisode[] =
        records.length > 0
          ? records.map((r: any) => ({
              seed: r.seed,
              success: Boolean(r.success),
              steps: r.steps ?? 0,
              total_reward: r.reward_total ?? r.total_reward ?? 0,
              sheep_penned: r.sheep_penned ?? 0,
              total_sheep: undefined,
              stop_reason: r.stop_reason,
              replay_id: r.replay_id ?? null,
              replay_path: normalizeReplayPath(r.replay_path),
              has_replay: Boolean(r.replay_id || r.replay_path),
            }))
          : DEFAULT_BENCHMARK_SEEDS.map((s, idx) => ({
              seed: s,
              success: idx < Math.round((ratePct / 100) * 10),
              steps: cp.average_completion_steps ? Math.round(cp.average_completion_steps) : 0,
              total_reward: cp.average_reward ?? 0,
              sheep_penned: cp.average_sheep_penned ? Math.round(cp.average_sheep_penned) : 0,
              has_replay: false,
            }));

      const mapKey = `${cpStage}:${cp.checkpoint_episode}`;
      byStageEp.set(mapKey, {
        key: `cp-${cpStage}-${cp.checkpoint_episode}`,
        checkpoint_episode: cp.checkpoint_episode,
        stage: cpStage,
        recorded_at: cp.recorded_at ?? (cp as any).evaluation_timestamp ?? (cp as any).created_timestamp,
        success_rate_pct: Math.round(ratePct),
        average_reward: cp.average_reward ?? 0,
        average_sheep_penned: cp.average_sheep_penned ?? 0,
        average_completion_steps: cp.average_completion_steps ?? null,
        is_best: cp.checkpoint_episode === bestCheckpointEpisode && cpStage === effectiveStage,
        episodes: episodes.sort((a, b) => a.seed - b.seed),
      });
    }

    // Enrich / override with API evaluations
    for (let idx = 0; idx < apiEvaluations.length; idx++) {
      const ev = apiEvaluations[idx];
      const evStage = ev.curriculum_stage ?? effectiveStage;
      if (stageFilter === "current" && evStage !== effectiveStage) continue;
      const mapKey = `${evStage}:${ev.checkpoint_episode}`;
      const existing = byStageEp.get(mapKey);
      const rawRate = ev.success_rate ?? 0;
      const ratePct = rawRate <= 1.0 ? rawRate * 100 : rawRate;

      const rawRecords = ev.records ?? (ev as any).episodes ?? [];
      const mergedEpisodes: UnifiedSeedEpisode[] = rawRecords.map((ep: any) => {
        const fallbackEp = existing?.episodes.find((e) => e.seed === ep.seed);
        const rPath = normalizeReplayPath(ep.replay_path) ?? fallbackEp?.replay_path ?? null;
        return {
          seed: ep.seed,
          success: Boolean(ep.success),
          steps: ep.steps ?? fallbackEp?.steps ?? 0,
          total_reward: ep.reward_total ?? ep.total_reward ?? fallbackEp?.total_reward ?? 0,
          sheep_penned: ep.sheep_penned ?? fallbackEp?.sheep_penned ?? 0,
          total_sheep: ep.total_sheep ?? fallbackEp?.total_sheep,
          stop_reason: ep.stop_reason || fallbackEp?.stop_reason,
          replay_id: ep.replay_id ?? fallbackEp?.replay_id ?? null,
          replay_path: rPath,
          has_replay: Boolean(ep.has_replay || ep.replay_id || rPath),
        };
      });

      byStageEp.set(mapKey, {
        key: `eval-${evStage}-${ev.checkpoint_episode}-${ev.evaluation_id ?? idx}`,
        evaluation_id: ev.evaluation_id,
        checkpoint_episode: ev.checkpoint_episode,
        stage: evStage,
        recorded_at: ev.evaluation_timestamp || existing?.recorded_at,
        success_rate_pct: Math.round(ratePct),
        average_reward: ev.average_reward ?? existing?.average_reward ?? 0,
        average_sheep_penned: ev.average_sheep_penned ?? existing?.average_sheep_penned ?? 0,
        average_completion_steps:
          ev.average_completion_steps ?? existing?.average_completion_steps ?? null,
        is_best: ev.checkpoint_episode === bestCheckpointEpisode && evStage === effectiveStage,
        episodes:
          mergedEpisodes.length > 0
            ? mergedEpisodes.sort((a, b) => a.seed - b.seed)
            : existing?.episodes ?? [],
      });
    }

    return Array.from(byStageEp.values()).sort(
      (a, b) => b.stage - a.stage || b.checkpoint_episode - a.checkpoint_episode
    );
  }, [checkpoints, apiEvaluations, stageFilter, effectiveStage, bestCheckpointEpisode]);

  // Select initial or latest evaluation
  useEffect(() => {
    if (initialCheckpointEpisode != null) {
      const match = unifiedEvaluations.find(
        (e) => e.checkpoint_episode === initialCheckpointEpisode
      );
      if (match) {
        setSelectedEvalKey(match.key);
        return;
      }
    }
    if (!pinnedEval && unifiedEvaluations.length > 0) {
      if (!selectedEvalKey || !unifiedEvaluations.some((e) => e.key === selectedEvalKey)) {
        setSelectedEvalKey(unifiedEvaluations[0].key);
      }
    }
  }, [unifiedEvaluations, initialCheckpointEpisode, pinnedEval, selectedEvalKey]);

  const activeEvaluation = useMemo(
    () =>
      unifiedEvaluations.find((e) => e.key === selectedEvalKey) ??
      unifiedEvaluations[0] ??
      null,
    [unifiedEvaluations, selectedEvalKey]
  );

  // Per-seed cross-evaluation reliability analysis
  const reliabilityAnalysis = useMemo(
    () =>
      analyzePerSeedReliability(
        checkpoints,
        stageFilter === "current" ? effectiveStage : undefined,
        10
      ),
    [checkpoints, stageFilter, effectiveStage]
  );

  const reliabilityBySeed = useMemo(() => {
    const map = new Map<number, (typeof reliabilityAnalysis.seeds)[number]>();
    for (const s of reliabilityAnalysis.seeds) {
      map.set(s.seed, s);
    }
    return map;
  }, [reliabilityAnalysis]);

  // Load replays for all 10 seeds of the active evaluation
  useEffect(() => {
    if (!activeEvaluation) {
      setReplaysBySeed({});
      return;
    }

    let cancelled = false;
    const nextReplays: Record<number, AnyReplayPayload | null> = {};
    const toLoad: UnifiedSeedEpisode[] = [];

    for (const ep of activeEvaluation.episodes) {
      const cacheKey = `${activeEvaluation.checkpoint_episode}:${ep.seed}`;
      const cached = replayCacheRef.current.get(cacheKey);
      if (cached) {
        cacheArenaReplay(cacheKey, cached);
        nextReplays[ep.seed] = cached;
      } else {
        nextReplays[ep.seed] = null;
        if (ep.replay_id || ep.replay_path) {
          toLoad.push(ep);
        }
      }
    }

    setReplaysBySeed(nextReplays);
    setCurrentStep(0);

    if (toLoad.length === 0) return;

    const loadAll = async () => {
      const loadingMap: Record<number, boolean> = {};
      for (const ep of toLoad) loadingMap[ep.seed] = true;
      setLoadingSeeds(loadingMap);

      await Promise.all(
        toLoad.map(async (ep) => {
          const cacheKey = `${activeEvaluation.checkpoint_episode}:${ep.seed}`;
          try {
            let payload: AnyReplayPayload | null = null;
            if (ep.replay_id) {
              try {
                payload = await fetchReplayById(String(ep.replay_id));
              } catch {
                payload = null;
              }
            }
            if (!payload && ep.replay_path) {
              try {
                payload = await loadReplay(ep.replay_path);
              } catch {
                payload = null;
              }
            }
            if (!cancelled && payload) {
              cacheArenaReplay(cacheKey, payload);
              setReplaysBySeed((prev) => ({ ...prev, [ep.seed]: payload }));
            }
          } finally {
            if (!cancelled) {
              setLoadingSeeds((prev) => ({ ...prev, [ep.seed]: false }));
            }
          }
        })
      );
    };

    loadAll();
    return () => {
      cancelled = true;
    };
  }, [activeEvaluation, cacheArenaReplay]);

  // On-demand simulation for any seed whose historical replay was pruned
  const handleSimulateSeed = useCallback(
    async (seed: number) => {
      if (!activeEvaluation) return;
      setLoadingSeeds((prev) => ({ ...prev, [seed]: true }));
      try {
        const payload = await runReplay({
          seed,
          checkpoint_episode: activeEvaluation.checkpoint_episode,
          effective_config: {
            curriculum_stage: activeEvaluation.stage > 0 ? activeEvaluation.stage : effectiveStage,
            enable_instinct_rewards: true,
            debug_reward_breakdown: true,
          },
        });
        const cacheKey = `${activeEvaluation.checkpoint_episode}:${seed}`;
        cacheArenaReplay(cacheKey, payload);
        setReplaysBySeed((prev) => ({ ...prev, [seed]: payload }));
      } catch {
        // Ignore error if backend busy
      } finally {
        setLoadingSeeds((prev) => ({ ...prev, [seed]: false }));
      }
    },
    [activeEvaluation, effectiveStage, cacheArenaReplay]
  );

  const handleSimulateMissingSeeds = useCallback(async () => {
    if (!activeEvaluation || simulatingAll) return;
    setSimulatingAll(true);
    try {
      const missing = activeEvaluation.episodes.filter((ep) => !replaysBySeed[ep.seed]);
      for (const ep of missing) {
        await handleSimulateSeed(ep.seed);
      }
    } finally {
      setSimulatingAll(false);
    }
  }, [activeEvaluation, simulatingAll, replaysBySeed, handleSimulateSeed]);

  // Maximum step count across all 10 episodes in the active evaluation
  const maxSteps = useMemo(() => {
    if (!activeEvaluation) return 100;
    let max = 1;
    for (const ep of activeEvaluation.episodes) {
      const replayLen = replaysBySeed[ep.seed]?.frames?.length ?? 0;
      const epSteps = Math.max(ep.steps ?? 0, replayLen > 0 ? replayLen - 1 : 0);
      if (epSteps > max) max = epSteps;
    }
    return max;
  }, [activeEvaluation, replaysBySeed]);

  // Master synchronized playback timer
  useEffect(() => {
    if (!isPlaying || arenaLayout === "matrix") return;
    const intervalMs = Math.max(18, Math.round(110 / playbackSpeed));
    const timer = window.setInterval(() => {
      setCurrentStep((prev) => {
        if (prev >= maxSteps) {
          if (autoLoop) return 0;
          setIsPlaying(false);
          return maxSteps;
        }
        return prev + 1;
      });
    }, intervalMs);
    return () => window.clearInterval(timer);
  }, [isPlaying, playbackSpeed, maxSteps, autoLoop, arenaLayout]);

  const filteredEpisodes = useMemo(() => {
    if (!activeEvaluation) return [];
    return activeEvaluation.episodes.filter((ep) => {
      if (seedFilter === "passed") return ep.success;
      if (seedFilter === "failed") return !ep.success;
      return true;
    });
  }, [activeEvaluation, seedFilter]);

  const passedCount = useMemo(
    () => (activeEvaluation ? activeEvaluation.episodes.filter((e) => e.success).length : 0),
    [activeEvaluation]
  );
  const failedCount = useMemo(
    () => (activeEvaluation ? activeEvaluation.episodes.filter((e) => !e.success).length : 0),
    [activeEvaluation]
  );

  const missingReplayCount = useMemo(() => {
    if (!activeEvaluation) return 0;
    return activeEvaluation.episodes.filter(
      (ep) => !replaysBySeed[ep.seed] && !loadingSeeds[ep.seed]
    ).length;
  }, [activeEvaluation, replaysBySeed, loadingSeeds]);

  const focusedEpisode = useMemo(
    () =>
      activeEvaluation?.episodes.find((e) => e.seed === focusedSeed) ??
      activeEvaluation?.episodes[0] ??
      null,
    [activeEvaluation, focusedSeed]
  );

  return (
    <div className="sd-eval-arena">
      {/* Top Synchronized Master Control Bar */}
      <div className="sd-eval-master-bar">
        {/* Left: Evaluation Selector & Summary */}
        <div className="sd-eval-master-bar__group">
          <span
            style={{
              fontSize: "0.72rem",
              fontWeight: 700,
              textTransform: "uppercase",
              letterSpacing: "0.06em",
              color: "var(--sd-text-muted)",
            }}
          >
            10-Seed Benchmark
          </span>

          <div className="sd-pill-group">
            <button
              type="button"
              className={`sd-pill-btn ${stageFilter === "current" ? "sd-pill-btn--active" : ""}`}
              onClick={() => setStageFilter("current")}
            >
              Stage {effectiveStage}
            </button>
            <button
              type="button"
              className={`sd-pill-btn ${stageFilter === "all" ? "sd-pill-btn--active" : ""}`}
              onClick={() => setStageFilter("all")}
            >
              All Stages
            </button>
          </div>

          <select
            className="sd-select"
            value={activeEvaluation?.key ?? ""}
            onChange={(e) => {
              setSelectedEvalKey(e.target.value);
              setPinnedEval(true);
            }}
            aria-label="Select 10-seed evaluation"
          >
            {unifiedEvaluations.length === 0 ? (
              <option value="">No evaluations recorded yet</option>
            ) : (
              unifiedEvaluations.map((ev) => (
                <option key={ev.key} value={ev.key}>
                  Ep {ev.checkpoint_episode} (Stage {ev.stage}) — {ev.success_rate_pct}% (
                  {ev.episodes.filter((x) => x.success).length}/{ev.episodes.length} seeds)
                  {ev.is_best ? " ★ BEST" : ""}
                </option>
              ))
            )}
          </select>

          <button
            type="button"
            className={`sd-btn sd-btn--sm ${pinnedEval ? "sd-btn--amber" : ""}`}
            onClick={() => setPinnedEval((p) => !p)}
            title="Pin current evaluation so background training doesn't auto-switch it"
          >
            {pinnedEval ? "📌 Pinned" : "Pin"}
          </button>

          {activeEvaluation && (
            <div style={{ display: "flex", alignItems: "center", gap: "0.4rem" }}>
              <span
                className={`sd-badge ${
                  activeEvaluation.success_rate_pct >= 80
                    ? "sd-badge--emerald"
                    : activeEvaluation.success_rate_pct >= 50
                    ? "sd-badge--amber"
                    : "sd-badge--rose"
                }`}
              >
                {activeEvaluation.success_rate_pct}% Pass ({passedCount}/
                {activeEvaluation.episodes.length})
              </span>
              {activeEvaluation.average_completion_steps != null && (
                <span className="sd-badge sd-badge--cyan">
                  Avg {Math.round(activeEvaluation.average_completion_steps)} steps
                </span>
              )}
              <span className="sd-badge sd-badge--neutral">
                Avg Return {activeEvaluation.average_reward.toFixed(1)}
              </span>
            </div>
          )}
        </div>

        {/* Center: Master Synchronized Timeline Scrubber */}
        {arenaLayout !== "matrix" && (
          <div className="sd-eval-master-scrubber">
            <button
              type="button"
              className="sd-btn sd-btn--sm sd-btn--icon"
              onClick={() => setCurrentStep(0)}
              title="Reset to step 0"
            >
              ⏮
            </button>
            <button
              type="button"
              className="sd-btn sd-btn--sm sd-btn--icon"
              onClick={() => {
                setIsPlaying(false);
                setCurrentStep((s) => Math.max(0, s - 1));
              }}
              title="Step backward"
            >
              ◀
            </button>
            <button
              type="button"
              className="sd-btn sd-btn--sm sd-btn--primary"
              onClick={() => setIsPlaying((p) => !p)}
              style={{ minWidth: 68 }}
            >
              {isPlaying ? "⏸ Pause" : "▶ Play"}
            </button>
            <button
              type="button"
              className="sd-btn sd-btn--sm sd-btn--icon"
              onClick={() => {
                setIsPlaying(false);
                setCurrentStep((s) => Math.min(maxSteps, s + 1));
              }}
              title="Step forward"
            >
              ▶
            </button>

            <input
              type="range"
              className="sd-scrubber-slider"
              min={0}
              max={maxSteps}
              value={Math.min(currentStep, maxSteps)}
              onChange={(e) => setCurrentStep(Number(e.target.value))}
              aria-label="Master 10-seed timeline scrubber"
            />

            <span
              className="sd-mono"
              style={{
                fontSize: "0.75rem",
                fontWeight: 600,
                color: "var(--sd-text-primary)",
                minWidth: 82,
                textAlign: "right",
              }}
            >
              {currentStep} / {maxSteps}
            </span>

            <div className="sd-pill-group">
              {[1, 2, 4, 8].map((spd) => (
                <button
                  key={spd}
                  type="button"
                  className={`sd-pill-btn ${playbackSpeed === spd ? "sd-pill-btn--active" : ""}`}
                  onClick={() => setPlaybackSpeed(spd)}
                >
                  {spd}x
                </button>
              ))}
            </div>

            <button
              type="button"
              className={`sd-btn sd-btn--sm ${autoLoop ? "sd-btn--emerald" : "sd-btn--ghost"}`}
              onClick={() => setAutoLoop((l) => !l)}
              title="Auto-loop all 10 episodes"
            >
              🔁 Loop
            </button>
          </div>
        )}

        {/* Right: Pass/Fail Filter & Layout Mode */}
        <div className="sd-eval-master-bar__group">
          {missingReplayCount > 0 && arenaLayout !== "matrix" && (
            <button
              type="button"
              className="sd-btn sd-btn--sm sd-btn--amber"
              disabled={simulatingAll}
              onClick={handleSimulateMissingSeeds}
              title="Simulate missing seed replays on-demand using checkpoint model"
            >
              {simulatingAll
                ? "⚡ Simulating..."
                : `⚡ Simulate ${missingReplayCount} Missing`}
            </button>
          )}

          <div className="sd-pill-group" role="group" aria-label="Filter seeds">
            <button
              type="button"
              className={`sd-pill-btn ${seedFilter === "all" ? "sd-pill-btn--active" : ""}`}
              onClick={() => setSeedFilter("all")}
            >
              All ({activeEvaluation?.episodes.length ?? 10})
            </button>
            <button
              type="button"
              className={`sd-pill-btn ${seedFilter === "passed" ? "sd-pill-btn--active" : ""}`}
              onClick={() => setSeedFilter("passed")}
            >
              ✓ Pass ({passedCount})
            </button>
            <button
              type="button"
              className={`sd-pill-btn ${seedFilter === "failed" ? "sd-pill-btn--active" : ""}`}
              onClick={() => setSeedFilter("failed")}
            >
              ✗ Fail ({failedCount})
            </button>
          </div>

          <div className="sd-pill-group" role="group" aria-label="Arena layout">
            <button
              type="button"
              className={`sd-pill-btn ${arenaLayout === "grid" ? "sd-pill-btn--active" : ""}`}
              onClick={() => setArenaLayout("grid")}
            >
              5×2 Grid
            </button>
            <button
              type="button"
              className={`sd-pill-btn ${arenaLayout === "focus" ? "sd-pill-btn--active" : ""}`}
              onClick={() => setArenaLayout("focus")}
            >
              Focus + Fleet
            </button>
            <button
              type="button"
              className={`sd-pill-btn ${arenaLayout === "matrix" ? "sd-pill-btn--active" : ""}`}
              onClick={() => setArenaLayout("matrix")}
            >
              10-Seed Matrix
            </button>
          </div>
        </div>
      </div>

      {/* BODY VIEW 1: 5x2 Widescreen Synchronized Grid of All 10 Seeds */}
      {arenaLayout === "grid" && (
        <div className="sd-eval-grid-5x2">
          {filteredEpisodes.map((ep) => {
            const replay = replaysBySeed[ep.seed] ?? null;
            const isLoading = Boolean(loadingSeeds[ep.seed]);
            const epMaxFrame = replay?.frames?.length ? replay.frames.length - 1 : ep.steps;
            const clampedFrame = Math.min(currentStep, Math.max(0, epMaxFrame));
            const isFinishedAtStep = currentStep >= epMaxFrame && epMaxFrame > 0;
            const snapshot = snapshotFromReplay(replay, clampedFrame);
            const rel = reliabilityBySeed.get(ep.seed);

            return (
              <div
                key={ep.seed}
                className={`sd-seed-card ${
                  ep.success ? "sd-seed-card--passed" : "sd-seed-card--failed"
                } ${focusedSeed === ep.seed ? "sd-seed-card--selected" : ""}`}
                onClick={() => {
                  setFocusedSeed(ep.seed);
                  if (replay && activeEvaluation && onSelectSeedForMainPlayer) {
                    onSelectSeedForMainPlayer(
                      replay,
                      activeEvaluation.checkpoint_episode,
                      ep.seed
                    );
                  }
                }}
              >
                <div className="sd-seed-card__header">
                  <div style={{ display: "flex", alignItems: "center", gap: "0.35rem" }}>
                    <span className="sd-seed-card__title">Seed #{ep.seed}</span>
                    {rel?.isBlindSpot && (
                      <span
                        className="sd-badge sd-badge--rose"
                        style={{ padding: "0.05rem 0.3rem", fontSize: "0.6rem" }}
                        title={rel.statusText}
                      >
                        Blind Spot
                      </span>
                    )}
                    {rel?.isInefficient && (
                      <span
                        className="sd-badge sd-badge--amber"
                        style={{ padding: "0.05rem 0.3rem", fontSize: "0.6rem" }}
                        title={rel.statusText}
                      >
                        Slow
                      </span>
                    )}
                  </div>

                  <div style={{ display: "flex", alignItems: "center", gap: "0.3rem" }}>
                    <span
                      className={`sd-badge ${
                        ep.success ? "sd-badge--emerald" : "sd-badge--rose"
                      }`}
                      style={{ padding: "0.08rem 0.4rem", fontSize: "0.62rem" }}
                    >
                      {ep.success ? "✓ PASS" : "✗ FAIL"}
                    </span>
                    <button
                      type="button"
                      className="sd-btn sd-btn--sm sd-btn--ghost"
                      style={{ padding: "0.05rem 0.3rem", fontSize: "0.65rem" }}
                      title="Focus this seed in Split Inspector"
                      onClick={(e) => {
                        e.stopPropagation();
                        setFocusedSeed(ep.seed);
                        setArenaLayout("focus");
                      }}
                    >
                      ⤢
                    </button>
                  </div>
                </div>

                <div className="sd-seed-card__canvas">
                  {snapshot ? (
                    <StudioFieldCanvas
                      snapshot={snapshot}
                      compact={true}
                      uniqueId={`seed-${ep.seed}`}
                    />
                  ) : (
                    <div
                      className="sd-empty-state"
                      style={{ padding: "0.5rem", fontSize: "0.7rem", gap: "0.35rem" }}
                    >
                      {isLoading ? (
                        <span>Loading replay...</span>
                      ) : (
                        <>
                          <span>Replay not cached</span>
                          <button
                            type="button"
                            className="sd-btn sd-btn--sm sd-btn--primary"
                            onClick={(e) => {
                              e.stopPropagation();
                              handleSimulateSeed(ep.seed);
                            }}
                          >
                            ⚡ Simulate Seed #{ep.seed}
                          </button>
                        </>
                      )}
                    </div>
                  )}
                </div>

                <div className="sd-seed-card__footer">
                  <span>
                    🐑{" "}
                    <strong style={{ color: "var(--sd-text-primary)" }}>
                      {snapshot?.penned_count ?? ep.sheep_penned}
                    </strong>
                    {ep.total_sheep ? `/${ep.total_sheep}` : ""} penned
                  </span>
                  <span>
                    ⏱{" "}
                    <strong
                      style={{
                        color: isFinishedAtStep
                          ? ep.success
                            ? "var(--sd-emerald-bright)"
                            : "var(--sd-rose-bright)"
                          : "var(--sd-text-primary)",
                      }}
                    >
                      {clampedFrame}/{ep.steps}
                    </strong>{" "}
                    steps
                  </span>
                  <span>
                    R:{" "}
                    <strong style={{ color: "var(--sd-text-primary)" }}>
                      {ep.total_reward.toFixed(1)}
                    </strong>
                  </span>
                </div>
              </div>
            );
          })}
        </div>
      )}

      {/* BODY VIEW 2: Focus Seed + 9-Seed Fleet Split Comparison */}
      {arenaLayout === "focus" && focusedEpisode && (
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "minmax(420px, 50%) 1fr",
            gap: "0.75rem",
            flex: 1,
            minHeight: 0,
          }}
        >
          {/* Left: High-Detail Focus Seed Canvas */}
          <div className="sd-panel" style={{ display: "flex", flexDirection: "column" }}>
            <div className="sd-panel-header">
              <div className="sd-panel-title">
                <span>🔍 Focus Seed #{focusedEpisode.seed}</span>
                <span
                  className={`sd-badge ${
                    focusedEpisode.success ? "sd-badge--emerald" : "sd-badge--rose"
                  }`}
                >
                  {focusedEpisode.success ? "✓ PASSED" : "✗ FAILED"}
                </span>
                <span className="sd-badge sd-badge--neutral">
                  {focusedEpisode.steps} steps · Return {focusedEpisode.total_reward.toFixed(1)}
                </span>
                {focusedEpisode.stop_reason && (
                  <span className="sd-badge sd-badge--amber">
                    {focusedEpisode.stop_reason}
                  </span>
                )}
              </div>

              <div style={{ display: "flex", alignItems: "center", gap: "0.4rem" }}>
                <button
                  type="button"
                  className={`sd-btn sd-btn--sm ${
                    showVectors ? "sd-btn--primary" : "sd-btn--ghost"
                  }`}
                  onClick={() => setShowVectors((v) => !v)}
                >
                  📐 Vectors
                </button>
                <button
                  type="button"
                  className="sd-btn sd-btn--sm"
                  onClick={() => setArenaLayout("grid")}
                >
                  ← Back to 5×2 Grid
                </button>
              </div>
            </div>

            <div className="sd-field-viewport">
              {replaysBySeed[focusedEpisode.seed] ? (
                <StudioFieldCanvas
                  snapshot={snapshotFromReplay(
                    replaysBySeed[focusedEpisode.seed],
                    Math.min(
                      currentStep,
                      Math.max(0, (replaysBySeed[focusedEpisode.seed]?.frames?.length ?? 1) - 1)
                    )
                  )}
                  compact={false}
                  showRoles={true}
                  showVectors={showVectors}
                  uniqueId={`focus-${focusedEpisode.seed}`}
                />
              ) : (
                <div className="sd-empty-state">
                  <button
                    type="button"
                    className="sd-btn sd-btn--primary"
                    onClick={() => handleSimulateSeed(focusedEpisode.seed)}
                  >
                    ⚡ Simulate Seed #{focusedEpisode.seed} Now
                  </button>
                </div>
              )}
            </div>
          </div>

          {/* Right: All 10 Seeds Quick Selector & Mini Synchronized Fleet */}
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(3, minmax(0, 1fr))",
              gridAutoRows: "1fr",
              gap: "0.5rem",
              minHeight: 0,
            }}
          >
            {(activeEvaluation?.episodes ?? []).map((ep) => {
              const replay = replaysBySeed[ep.seed] ?? null;
              const epMax = replay?.frames?.length ? replay.frames.length - 1 : ep.steps;
              const snapshot = snapshotFromReplay(
                replay,
                Math.min(currentStep, Math.max(0, epMax))
              );
              return (
                <div
                  key={ep.seed}
                  className={`sd-seed-card ${
                    ep.success ? "sd-seed-card--passed" : "sd-seed-card--failed"
                  } ${focusedSeed === ep.seed ? "sd-seed-card--selected" : ""}`}
                  onClick={() => setFocusedSeed(ep.seed)}
                >
                  <div className="sd-seed-card__header">
                    <span className="sd-seed-card__title">Seed #{ep.seed}</span>
                    <span
                      className={`sd-badge ${
                        ep.success ? "sd-badge--emerald" : "sd-badge--rose"
                      }`}
                      style={{ fontSize: "0.58rem", padding: "0.05rem 0.3rem" }}
                    >
                      {ep.success ? "PASS" : "FAIL"} · {ep.steps}s
                    </span>
                  </div>
                  <div className="sd-seed-card__canvas">
                    {snapshot ? (
                      <StudioFieldCanvas
                        snapshot={snapshot}
                        compact={true}
                        uniqueId={`fleet-${ep.seed}`}
                      />
                    ) : (
                      <div className="sd-empty-state" style={{ fontSize: "0.65rem" }}>
                        Click to focus
                      </div>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* BODY VIEW 3: Cross-Evaluation 10-Seed Comparison Matrix */}
      {arenaLayout === "matrix" && (
        <div className="sd-panel" style={{ flex: 1, overflow: "auto" }}>
          <div className="sd-panel-header">
            <div>
              <div className="sd-panel-title">
                <span>🎯 Cross-Evaluation 10-Seed Reliability Matrix</span>
                {reliabilityAnalysis.blindSpotCount > 0 && (
                  <span className="sd-badge sd-badge--rose">
                    {reliabilityAnalysis.blindSpotCount} Blind Spot Seed(s):{" "}
                    {reliabilityAnalysis.blindSpotSeeds.map((s) => `#${s}`).join(", ")}
                  </span>
                )}
                {reliabilityAnalysis.overallMedianSuccessfulSteps != null && (
                  <span className="sd-badge sd-badge--cyan">
                    Stage Median Successful Steps: {reliabilityAnalysis.overallMedianSuccessfulSteps}
                  </span>
                )}
              </div>
              <div className="sd-panel-subtitle">
                Compares each deterministic benchmark seed across recent formal evaluations to isolate scenario blind spots vs step efficiency gains.
              </div>
            </div>
          </div>

          <div className="sd-panel-body" style={{ overflowX: "auto" }}>
            <table className="sd-table">
              <thead>
                <tr>
                  <th>Seed</th>
                  <th>Reliability Status</th>
                  <th>Win Rate</th>
                  <th>Median Steps</th>
                  <th>Worst Steps</th>
                  {unifiedEvaluations.slice(0, 8).map((ev) => (
                    <th
                      key={ev.key}
                      style={{
                        textAlign: "center",
                        cursor: "pointer",
                        color:
                          ev.key === activeEvaluation?.key
                            ? "var(--sd-accent-bright)"
                            : undefined,
                      }}
                      onClick={() => {
                        setSelectedEvalKey(ev.key);
                        setArenaLayout("grid");
                      }}
                      title="Click to open this evaluation in the 5×2 Arena"
                    >
                      Ep {ev.checkpoint_episode}
                      <div style={{ fontSize: "0.62rem", opacity: 0.75 }}>
                        {ev.success_rate_pct}%
                      </div>
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {(reliabilityAnalysis.seeds.length > 0
                  ? reliabilityAnalysis.seeds.map((s) => s.seed)
                  : activeEvaluation?.episodes.length
                  ? activeEvaluation.episodes.map((e) => e.seed)
                  : DEFAULT_BENCHMARK_SEEDS
                ).map((seed) => {
                  const rel = reliabilityBySeed.get(seed);
                  return (
                    <tr key={seed}>
                      <td className="sd-mono" style={{ fontWeight: 700 }}>
                        Seed #{seed}
                      </td>
                      <td>
                        {rel ? (
                          <span
                            className={`sd-badge ${
                              rel.isBlindSpot
                                ? "sd-badge--rose"
                                : rel.isInefficient
                                ? "sd-badge--amber"
                                : rel.failureCount > 0
                                ? "sd-badge--neutral"
                                : "sd-badge--emerald"
                            }`}
                          >
                            {rel.statusText}
                          </span>
                        ) : (
                          <span className="sd-badge sd-badge--neutral">No trials</span>
                        )}
                      </td>
                      <td className="sd-mono">
                        {rel ? (
                          <strong
                            style={{
                              color:
                                rel.recentSuccessRate >= 80
                                  ? "var(--sd-emerald-bright)"
                                  : rel.recentSuccessRate >= 50
                                  ? "var(--sd-amber-bright)"
                                  : "var(--sd-rose-bright)",
                            }}
                          >
                            {rel.recentSuccessRate}% ({rel.successCount}/{rel.totalTrials})
                          </strong>
                        ) : (
                          "—"
                        )}
                      </td>
                      <td className="sd-mono" style={{ color: "var(--sd-cyan)" }}>
                        {rel?.typicalSuccessfulSteps != null
                          ? `${rel.typicalSuccessfulSteps}s`
                          : "—"}
                      </td>
                      <td className="sd-mono" style={{ color: "var(--sd-text-muted)" }}>
                        {rel?.worstSuccessfulSteps != null
                          ? `${rel.worstSuccessfulSteps}s`
                          : "—"}
                      </td>
                      {unifiedEvaluations.slice(0, 8).map((ev) => {
                        const ep = ev.episodes.find((x) => x.seed === seed);
                        if (!ep) {
                          return (
                            <td key={ev.key} style={{ textAlign: "center" }}>
                              —
                            </td>
                          );
                        }
                        return (
                          <td
                            key={ev.key}
                            style={{ textAlign: "center", cursor: "pointer" }}
                            onClick={() => {
                              setSelectedEvalKey(ev.key);
                              setFocusedSeed(seed);
                              setArenaLayout("focus");
                            }}
                            title={`Ep ${ev.checkpoint_episode} · Seed #${seed}: ${
                              ep.success ? "PASS" : "FAIL"
                            } in ${ep.steps} steps (Return ${ep.total_reward.toFixed(1)})`}
                          >
                            <span
                              className={`sd-badge ${
                                ep.success ? "sd-badge--emerald" : "sd-badge--rose"
                              }`}
                              style={{
                                minWidth: 54,
                                justifyContent: "center",
                                fontFamily: "var(--sd-font-mono)",
                              }}
                            >
                              {ep.success ? "✓" : "✗"} {ep.steps}s
                            </span>
                          </td>
                        );
                      })}
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
};

import { useCallback, useEffect, useMemo, useState } from "react";
import type { CheckpointEntry, CheckpointIndex, ReplayBundle, ReplaySnapshot } from "../state/types";
import { loadReplay } from "../lib/api";
import { exportResultsVideo, makeResultsVideoFileName } from "../lib/resultsVideo";


const DEFAULT_STAGE_COUNT = 21;
const REPLAYS_PER_STAGE = 5;
const TARGET_WEIGHTS_BY_SLOT_COUNT: Record<number, number[]> = {
  5: [0.34, 0.17, 0.07],
  6: [0.42, 0.24, 0.13, 0.07],
};

interface ResultsPanelProps {
  checkpointIndex: CheckpointIndex | null;
}

interface MilestoneReplay {
  checkpoint: CheckpointEntry;
  bundle: ReplayBundle | null;
  frameIndex: number;
  loading: boolean;
  error: string | null;
}

interface StageMilestones {
  stage: number;
  checkpoints: Array<CheckpointEntry | null>;
}

export interface GridMilestone {
  stage: number;
  slot: number;
  checkpoint: CheckpointEntry | null;
}

function completionSteps(checkpoint: CheckpointEntry): number | null {
  const steps = checkpoint.average_completion_steps;
  return Number.isFinite(steps) ? steps : null;
}

function quantilePick(checkpoints: CheckpointEntry[], count: number): CheckpointEntry[] {
  const sorted = [...checkpoints].sort((a, b) => a.checkpoint_episode - b.checkpoint_episode);
  if (sorted.length <= count) {
    return sorted;
  }

  const picks: number[] = [];
  for (let i = 0; i < count; i += 1) {
    const ratio = i / (count - 1);
    const index = Math.round(ratio * (sorted.length - 1));
    if (!picks.includes(index)) {
      picks.push(index);
    }
  }

  while (picks.length < count) {
    const candidate = Math.min(sorted.length - 1, picks[picks.length - 1] + 1);
    if (picks.includes(candidate)) {
      break;
    }
    picks.push(candidate);
  }

  return picks.slice(0, count).map((index) => sorted[index]);
}

function targetWeightsForSlots(totalSlots: number): number[] {
  const configured = TARGET_WEIGHTS_BY_SLOT_COUNT[totalSlots];
  if (configured) {
    return configured;
  }

  const interior = Math.max(totalSlots - 2, 1);
  const weights: number[] = [];
  let current = 0.42;
  for (let index = 0; index < interior; index += 1) {
    weights.push(Math.max(0.04, current));
    current *= 0.58;
  }
  return weights;
}

function selectEvenMilestones(stageCheckpoints: CheckpointEntry[]): CheckpointEntry[] {
  const sorted = [...stageCheckpoints].sort((a, b) => a.checkpoint_episode - b.checkpoint_episode);
  if (sorted.length <= REPLAYS_PER_STAGE) {
    return sorted;
  }

  const start = sorted[0];
  const end = sorted[sorted.length - 1];
  const startStep = completionSteps(start);
  const endStep = completionSteps(end);

  if (startStep === null || endStep === null) {
    return quantilePick(sorted, REPLAYS_PER_STAGE);
  }

  const totalRange = Math.max(Math.abs(startStep - endStep), 1);
  const improving = startStep >= endStep;
  const targetWeights = targetWeightsForSlots(REPLAYS_PER_STAGE);
  const selectedIndices: number[] = [0];

  for (let slot = 1; slot < REPLAYS_PER_STAGE - 1; slot += 1) {
    const previousIndex = selectedIndices[selectedIndices.length - 1];
    const previousStep = completionSteps(sorted[previousIndex]);
    const remainingSlots = REPLAYS_PER_STAGE - 1 - slot;
    const maxIndex = sorted.length - 1 - remainingSlots;
    const targetGapWeight = targetWeights[Math.min(slot - 1, targetWeights.length - 1)];
    const targetStep = endStep + (startStep - endStep) * targetGapWeight;
    const targetEpisodeRatio = slot / (REPLAYS_PER_STAGE - 1);

    let bestIndex = previousIndex + 1;
    let bestScore = Number.POSITIVE_INFINITY;

    for (let index = previousIndex + 1; index <= maxIndex; index += 1) {
      const candidateStep = completionSteps(sorted[index]);
      if (candidateStep === null) {
        continue;
      }

      const normalizedStepError = Math.abs(candidateStep - targetStep) / totalRange;
      const normalizedEpisodeError = Math.abs(index / (sorted.length - 1) - targetEpisodeRatio);
      let score = normalizedStepError + normalizedEpisodeError * 0.35;

      if (previousStep !== null) {
        const delta = improving ? previousStep - candidateStep : candidateStep - previousStep;
        const minDeltaRatio = Math.max(0.04, 0.22 * Math.pow(0.58, slot - 1));
        const minDelta = totalRange * minDeltaRatio;
        if (delta < minDelta) {
          score += ((minDelta - delta) / totalRange) * 0.8;
        }
        if (delta < -totalRange * 0.02) {
          score += 1.0;
        }
      }

      if (score < bestScore) {
        bestScore = score;
        bestIndex = index;
      }
    }

    selectedIndices.push(bestIndex);
  }

  selectedIndices.push(sorted.length - 1);

  const penultimatePosition = REPLAYS_PER_STAGE - 2;
  const previousPosition = REPLAYS_PER_STAGE - 3;
  const penultimateIndex = selectedIndices[penultimatePosition];
  const penultimateStep = completionSteps(sorted[penultimateIndex]);
  const finalStep = completionSteps(sorted[selectedIndices[REPLAYS_PER_STAGE - 1]]);
  if (penultimateStep !== null && finalStep !== null) {
    const finalGap = Math.abs(penultimateStep - finalStep);
    const minFinalGap = Math.max(4, totalRange * 0.03);
    if (finalGap < minFinalGap) {
      const previousIndex = selectedIndices[previousPosition];
      const maxIndex = selectedIndices[REPLAYS_PER_STAGE - 1] - 1;
      const targetWeight = targetWeights[Math.max(0, targetWeights.length - 1)];
      const targetStep = endStep + (startStep - endStep) * targetWeight;
      let replacement = penultimateIndex;
      let replacementScore = Number.POSITIVE_INFINITY;

      for (let index = previousIndex + 1; index <= maxIndex; index += 1) {
        const candidateStep = completionSteps(sorted[index]);
        if (candidateStep === null) {
          continue;
        }
        const candidateGap = Math.abs(candidateStep - finalStep);
        if (candidateGap < minFinalGap) {
          continue;
        }
        const score = Math.abs(candidateStep - targetStep) / totalRange + Math.abs(index - penultimateIndex) / sorted.length;
        if (score < replacementScore) {
          replacementScore = score;
          replacement = index;
        }
      }

      selectedIndices[penultimatePosition] = replacement;
    }
  }

  const uniqueIndices = Array.from(new Set(selectedIndices)).sort((a, b) => a - b);
  const selected = uniqueIndices.map((index) => sorted[index]);
  if (selected.length < REPLAYS_PER_STAGE) {
    return quantilePick(sorted, REPLAYS_PER_STAGE);
  }

  return selected.slice(0, REPLAYS_PER_STAGE);
}

function resolveStageOrder(checkpoints: CheckpointEntry[]): number[] {
  const maxStageInData = checkpoints.reduce((highest, checkpoint) => {
    const stage = checkpoint.reward_config?.instincts?.curriculum_stage ?? 0;
    return Math.max(highest, stage);
  }, 0);
  const stageCount = Math.max(DEFAULT_STAGE_COUNT, maxStageInData);
  return Array.from({ length: stageCount }, (_, index) => index + 1);
}

function buildStageMilestones(
  checkpoints: CheckpointEntry[],
  stageOrder: number[],
): StageMilestones[] {
  const byStage: Record<number, CheckpointEntry[]> = {};
  const stageSet = new Set(stageOrder);
  for (const checkpoint of checkpoints) {
    const stage = checkpoint.reward_config?.instincts?.curriculum_stage ?? 0;
    if (!stageSet.has(stage)) {
      continue;
    }
    if (!byStage[stage]) {
      byStage[stage] = [];
    }
    byStage[stage].push(checkpoint);
  }

  return stageOrder.map((stage) => {
    const stageCheckpoints = byStage[stage] ?? [];
    const selected = stageCheckpoints.some((checkpoint) => completionSteps(checkpoint) !== null)
      ? selectEvenMilestones(stageCheckpoints)
      : quantilePick(stageCheckpoints, REPLAYS_PER_STAGE);
    const checkpointsForStage: Array<CheckpointEntry | null> = [...selected];
    while (checkpointsForStage.length < REPLAYS_PER_STAGE) {
      checkpointsForStage.push(null);
    }
    return {
      stage,
      checkpoints: checkpointsForStage,
    };
  });
}

export function ResultsPanel({ checkpointIndex }: ResultsPanelProps) {
  const stageOrder = useMemo(
    () => resolveStageOrder(checkpointIndex?.checkpoints ?? []),
    [checkpointIndex],
  );
  const stageMilestones = useMemo(
    () => buildStageMilestones(checkpointIndex?.checkpoints ?? [], stageOrder),
    [checkpointIndex, stageOrder],
  );

  const milestoneCheckpoints = useMemo(
    () => stageMilestones.flatMap((entry) => entry.checkpoints.filter((checkpoint): checkpoint is CheckpointEntry => checkpoint !== null)),
    [stageMilestones],
  );

  const gridMilestones = useMemo(
    () =>
      stageMilestones.flatMap((stageEntry) =>
        stageEntry.checkpoints.map((checkpoint, slotIndex) => ({
          stage: stageEntry.stage,
          slot: slotIndex + 1,
          checkpoint,
        })),
      ),
    [stageMilestones],
  );

  const [replays, setReplays] = useState<Map<number, MilestoneReplay>>(new Map());
  const [playing, setPlaying] = useState(false);
  const [playbackSpeed, setPlaybackSpeed] = useState<"slow" | "normal" | "fast">("normal");
  const [exportStatus, setExportStatus] = useState<"idle" | "exporting" | "success" | "error">("idle");
  const [exportProgress, setExportProgress] = useState<number>(0);
  const [exportError, setExportError] = useState<string | null>(null);

  const isLoadingReplays = useMemo(() => {
    if (milestoneCheckpoints.length === 0) return false;
    if (replays.size === 0) return true;
    return Array.from(replays.values()).some((replay) => replay.loading);
  }, [milestoneCheckpoints, replays]);

  const handleExportVideo = useCallback(async () => {
    setExportStatus("exporting");
    setExportProgress(0);
    setExportError(null);
    try {
      const blob = await exportResultsVideo(gridMilestones, replays, playbackSpeed, (progress) => {
        setExportProgress(progress);
      });

      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = makeResultsVideoFileName();
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);

      setExportStatus("success");
      setTimeout(() => {
        setExportStatus("idle");
      }, 5000);
    } catch (err) {
      setExportStatus("error");
      setExportError(err instanceof Error ? err.message : "Failed to export video");
    }
  }, [gridMilestones, replays, playbackSpeed, setExportProgress]);

  useEffect(() => {
    if (!milestoneCheckpoints.length) {
      setReplays(new Map());
      return;
    }

    const newReplays = new Map<number, MilestoneReplay>();
    for (const checkpoint of milestoneCheckpoints) {
      newReplays.set(checkpoint.checkpoint_episode, {
        checkpoint,
        bundle: null,
        frameIndex: 0,
        loading: true,
        error: null,
      });
    }
    setReplays(newReplays);

    let cancelled = false;

    const loadAll = async () => {
      for (const checkpoint of milestoneCheckpoints) {
        if (cancelled) {
          return;
        }

        const record = checkpoint.records?.[0];
        if (!record) {
          setReplays((prev) => {
            const next = new Map(prev);
            const existing = next.get(checkpoint.checkpoint_episode);
            if (existing) {
              next.set(checkpoint.checkpoint_episode, {
                ...existing,
                loading: false,
                error: "No replay file",
              });
            }
            return next;
          });
          continue;
        }

        try {
          const bundle = await loadReplay(record.replay_path);
          if (cancelled) {
            return;
          }

          setReplays((prev) => {
            const next = new Map(prev);
            const existing = next.get(checkpoint.checkpoint_episode);
            if (existing) {
              next.set(checkpoint.checkpoint_episode, {
                ...existing,
                bundle,
                loading: false,
                error: null,
              });
            }
            return next;
          });
        } catch (error) {
          if (cancelled) {
            return;
          }

          setReplays((prev) => {
            const next = new Map(prev);
            const existing = next.get(checkpoint.checkpoint_episode);
            if (existing) {
              next.set(checkpoint.checkpoint_episode, {
                ...existing,
                loading: false,
                error: error instanceof Error ? error.message : "Failed to load replay",
              });
            }
            return next;
          });
        }
      }
    };

    void loadAll();

    return () => {
      cancelled = true;
    };
  }, [milestoneCheckpoints]);

  useEffect(() => {
    if (!playing) {
      return;
    }

    const delays = { slow: 300, normal: 150, fast: 50 };
    const delay = delays[playbackSpeed];

    const timer = window.setInterval(() => {
      setReplays((prev) => {
        const next = new Map(prev);
        let allComplete = true;

        for (const [episode, replay] of next) {
          if (!replay.bundle || replay.frameIndex >= replay.bundle.frames.length - 1) {
            continue;
          }
          allComplete = false;
          next.set(episode, { ...replay, frameIndex: replay.frameIndex + 1 });
        }

        if (allComplete) {
          setPlaying(false);
        }

        return next;
      });
    }, delay);

    return () => window.clearInterval(timer);
  }, [playing, playbackSpeed]);

  const handlePlayPause = useCallback(() => {
    setPlaying((prev) => !prev);
  }, []);

  const handleReset = useCallback(() => {
    setPlaying(false);
    setReplays((prev) => {
      const next = new Map(prev);
      for (const [episode, replay] of next) {
        next.set(episode, { ...replay, frameIndex: 0 });
      }
      return next;
    });
  }, []);

  const totalSlots = stageMilestones.length * REPLAYS_PER_STAGE;
  const filledSlots = milestoneCheckpoints.length;
  const placeholders = totalSlots - filledSlots;

  if (!milestoneCheckpoints.length) {
    return (
      <div className="results-empty">
        <p className="eyebrow">Learning progression</p>
        <h2>No training results yet</h2>
        <p className="results-empty__message">
          Train your dog team to see milestone replays across curriculum stages.
        </p>
      </div>
    );
  }

  return (
    <div className="results-panel">
      <div className="results-panel__header">
        <div className="results-panel__meta">
          <p className="eyebrow">Learning progression</p>
          <p className="results-panel__description">
            {filledSlots}/{totalSlots} replay slots
            {placeholders > 0 ? ` • ${placeholders} placeholder${placeholders === 1 ? "" : "s"}` : ""}
          </p>
          {exportStatus === "success" && (
            <span style={{ fontSize: "9px", color: "#3fb950", marginLeft: "8px", fontWeight: "bold" }}>
              Video export started
            </span>
          )}
          {exportStatus === "error" && (
            <span style={{ fontSize: "9px", color: "#f85149", marginLeft: "8px", fontWeight: "bold" }}>
              {exportError}
            </span>
          )}
        </div>
        <div className="results-panel__controls">
          <button
            onClick={handlePlayPause}
            disabled={milestoneCheckpoints.length === 0 || Array.from(replays.values()).every((replay) => replay.loading)}
          >
            {playing ? "Pause" : "Play"}
          </button>
          <button onClick={handleReset} disabled={playing}>
            Reset
          </button>
          <select
            value={playbackSpeed}
            onChange={(event) => setPlaybackSpeed(event.target.value as "slow" | "normal" | "fast")}
            disabled={playing}
          >
            <option value="slow">Slow</option>
            <option value="normal">Normal</option>
            <option value="fast">Fast</option>
          </select>
          <button
            onClick={handleExportVideo}
            disabled={isLoadingReplays || exportStatus === "exporting"}
          >
            {exportStatus === "exporting" ? `Exporting (${Math.round(exportProgress * 100)}%)...` : "Export Video"}
          </button>
        </div>
      </div>


      <div className="results-stage-list">
        <div className="results-grid">
          {gridMilestones.map((item) => {
            if (!item.checkpoint) {
              return (
                <PlaceholderCard
                  key={`stage-${item.stage}-placeholder-${item.slot}`}
                  stage={item.stage}
                  slot={item.slot}
                />
              );
            }

            const cpKey = item.checkpoint.checkpoint_id 
              ? item.checkpoint.checkpoint_id 
              : item.checkpoint.run_id 
                ? `${item.checkpoint.run_id}-ep-${item.checkpoint.checkpoint_episode}`
                : `ep-${item.checkpoint.checkpoint_episode}`;

            const replay = replays.get(item.checkpoint.checkpoint_episode);
            if (!replay) {
              return (
                <PlaceholderCard
                  key={`stage-${item.stage}-missing-${cpKey}`}
                  stage={item.stage}
                  slot={item.slot}
                />
              );
            }

            const snapshot = replay.bundle?.frames[replay.frameIndex]?.snapshot ?? replay.bundle?.final_snapshot ?? null;

            return (
              <MilestoneCard
                key={cpKey}
                checkpoint={item.checkpoint}
                snapshot={snapshot}
                stage={item.stage}
                loading={replay.loading}
                error={replay.error}
                frameIndex={replay.frameIndex}
                totalFrames={replay.bundle?.frames.length ?? 0}
              />
            );
          })}
        </div>
      </div>
    </div>
  );
}

function stageColor(stage: number): string {
  const palette: Record<number, string> = {
    0: "#9ca3af",
    1: "#60a5fa",
    2: "#34d399",
    3: "#f59e0b",
    4: "#f472b6",
    5: "#c084fc",
    6: "#fb7185",
    7: "#22d3ee",
    8: "#a3e635",
  };
  if (palette[stage]) {
    return palette[stage];
  }
  const hue = ((stage - 1) * 47) % 360;
  return `hsl(${hue} 74% 64%)`;
}

interface MilestoneCardProps {
  checkpoint: CheckpointEntry;
  snapshot: ReplaySnapshot | null;
  stage: number;
  loading: boolean;
  error: string | null;
  frameIndex: number;
  totalFrames: number;
}

function MilestoneCard({ checkpoint, snapshot, stage, loading, error, frameIndex, totalFrames }: MilestoneCardProps) {
  const baseWidth = snapshot?.grid_width ?? snapshot?.field_width ?? 40;
  const baseHeight = snapshot?.grid_height ?? snapshot?.field_height ?? 30;
  const width = snapshot ? Math.max(baseWidth, 40) : 40;
  const height = snapshot ? Math.max(baseHeight, 30) : 30;

  const densityScale = Math.max(width / 40, height / 30, 1);
  const dogRadius = 0.48 * densityScale;
  const sheepRadius = 0.42 * densityScale;
  const fenceStroke = 0.32 * densityScale;

  const stageAccent = stageColor(stage);
  const fieldAspect = `${width} / ${height}`;
  const successRatio = checkpoint.success_rate;
  const progressRatio = totalFrames > 1 ? frameIndex / (totalFrames - 1) : 0;
  const progressPercent = Math.round(progressRatio * 100);
  const successPercent = Math.round(successRatio * 100);
  const successClassName = successPercent === 100 ? "milestone-card__success milestone-card__success--good" : successPercent === 0 ? "milestone-card__success milestone-card__success--bad" : "milestone-card__success";
  const progressTone = successPercent === 100 ? "milestone-card__progress-fill--good" : progressRatio > 0.6 ? "milestone-card__progress-fill--warn" : "milestone-card__progress-fill--bad";
  const badgeStyle = {
    color: stageAccent,
    background: `color-mix(in srgb, ${stageAccent} 22%, transparent)`,
    borderColor: `color-mix(in srgb, ${stageAccent} 40%, transparent)`,
  };

  function fenceSegments(snap: ReplaySnapshot) {
    const { pen } = snap;
    const opening = pen.opening ?? "left";
    const ox = pen.origin.x;
    const oy = pen.origin.y;
    const right = ox + pen.width;
    const bottom = oy + pen.height;
    const all = [
      { side: "top", x1: ox, y1: oy, x2: right, y2: oy },
      { side: "bottom", x1: ox, y1: bottom, x2: right, y2: bottom },
      { side: "left", x1: ox, y1: oy, x2: ox, y2: bottom },
      { side: "right", x1: right, y1: oy, x2: right, y2: bottom },
    ];
    return all.filter((segment) => segment.side !== opening);
  }

  const fences = snapshot ? fenceSegments(snapshot) : [];

  return (
    <div className="milestone-card" style={{ borderColor: stageAccent, boxShadow: `0 16px 34px color-mix(in srgb, ${stageAccent} 20%, transparent)` }}>
      <div className="milestone-card__header">
        <div className="milestone-card__header-left">
          <span className="milestone-card__badge" style={badgeStyle}>
            S{stage}
          </span>
          <h3 className="milestone-card__title">Episode {checkpoint.checkpoint_episode}</h3>
        </div>
        <span className={successClassName}>{successPercent}%</span>
      </div>

      <div className="milestone-card__field" style={{ aspectRatio: fieldAspect }}>
        {loading ? (
          <div className="milestone-card__loading">Loading...</div>
        ) : error ? (
          <div className="milestone-card__error">{error}</div>
        ) : snapshot ? (
          <svg
            viewBox={`0 0 ${width} ${height}`}
            className="milestone-card__svg"
            style={{ backgroundColor: "rgba(4, 8, 14, 0.7)" }}
            preserveAspectRatio="xMidYMid meet"
          >
            {fences.map((fence, index) => (
              <line
                key={index}
                x1={fence.x1}
                y1={fence.y1}
                x2={fence.x2}
                y2={fence.y2}
                stroke="#86efac"
                strokeWidth={fenceStroke}
                strokeLinecap="round"
              />
            ))}

            {snapshot.sheep.map((sheep, index) => (
              <circle
                key={`sheep-${index}`}
                cx={sheep.x}
                cy={sheep.y}
                r={sheepRadius}
                fill={sheep.penned ? "#86efac" : "#f8fafc"}
                stroke={sheep.penned ? "#86efac" : "#cbd5e1"}
                strokeWidth={0.06 * densityScale}
              />
            ))}

            {snapshot.dogs.map((dog, index) => (
              <circle
                key={`dog-${index}`}
                cx={dog.x}
                cy={dog.y}
                r={dogRadius}
                fill={dogColor(dog.index)}
                stroke="rgba(255,255,255,0.7)"
                strokeWidth={0.08 * densityScale}
              />
            ))}
          </svg>
        ) : (
          <div className="milestone-card__empty">No replay</div>
        )}
      </div>

      {totalFrames > 0 && (
        <div className="milestone-card__progress">
          <div className="milestone-card__progress-bar">
            <div
              className={`milestone-card__progress-fill ${progressTone}`}
              style={{ width: `${(frameIndex / (totalFrames - 1)) * 100}%` }}
            />
          </div>
          <div className="milestone-card__progress-meta">
            <span className="milestone-card__step">{snapshot?.step ?? 0} / {totalFrames - 1}</span>
            <span className="milestone-card__step">{progressPercent}%</span>
          </div>
        </div>
      )}
    </div>
  );
}

interface PlaceholderCardProps {
  stage: number;
  slot: number;
}

function PlaceholderCard({ stage, slot }: PlaceholderCardProps) {
  const stageAccent = stageColor(stage);
  const badgeStyle = {
    color: stageAccent,
    background: `color-mix(in srgb, ${stageAccent} 22%, transparent)`,
    borderColor: `color-mix(in srgb, ${stageAccent} 40%, transparent)`,
  };

  return (
    <div
      className="milestone-card milestone-card--placeholder"
      aria-label={`Stage ${stage} placeholder slot ${slot}`}
      style={{ borderColor: stageAccent, boxShadow: `0 14px 28px color-mix(in srgb, ${stageAccent} 16%, transparent)` }}
    >
      <div className="milestone-card__header">
        <div className="milestone-card__header-left">
          <span className="milestone-card__badge" style={badgeStyle}>S{stage}</span>
          <h3 className="milestone-card__title">Replay slot {slot}</h3>
        </div>
        <span className="milestone-card__success">--</span>
      </div>
      <div className="milestone-card__field" style={{ aspectRatio: "4 / 3" }}>
        <div className="milestone-card__empty">Awaiting more training data</div>
      </div>
      <div className="milestone-card__progress milestone-card__progress--placeholder">
        <div className="milestone-card__progress-bar" />
        <div className="milestone-card__progress-meta">
          <span className="milestone-card__step">No replay yet</span>
          <span className="milestone-card__step">0%</span>
        </div>
      </div>
    </div>
  );
}

function dogColor(index: number): string {
  const palette = ["#3b82f6", "#10b981", "#f59e0b", "#ec4899", "#8b5cf6"];
  return palette[index % palette.length];
}

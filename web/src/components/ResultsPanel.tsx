import { useCallback, useEffect, useMemo, useState } from "react";
import type { CheckpointEntry, CheckpointIndex, ReplayBundle, ReplaySnapshot } from "../state/types";
import { loadReplay } from "../lib/api";

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

/**
 * Select 3-5 representative checkpoints per stage to showcase learning progression.
 * Prioritizes first, middle, and best checkpoints of each stage.
 */
function selectMilestones(checkpoints: CheckpointEntry[]): CheckpointEntry[] {
  if (!checkpoints.length) return [];

  // Group by stage
  const byStage: Record<number, CheckpointEntry[]> = {};
  for (const cp of checkpoints) {
    const stage = cp.reward_config?.instincts?.curriculum_stage ?? 0;
    if (!byStage[stage]) byStage[stage] = [];
    byStage[stage].push(cp);
  }

  const milestones: CheckpointEntry[] = [];

  // For each stage, pick 3-5 milestones
  for (const stage of Object.keys(byStage).sort((a, b) => parseInt(a) - parseInt(b))) {
    const stageCheckpoints = byStage[parseInt(stage)];
    if (!stageCheckpoints.length) continue;

    stageCheckpoints.sort((a, b) => a.checkpoint_episode - b.checkpoint_episode);

    if (stageCheckpoints.length <= 5) {
      // If 5 or fewer, show all
      milestones.push(...stageCheckpoints);
    } else {
      // Show: first, ~25%, ~50%, ~75%, best
      const first = stageCheckpoints[0];
      const q1 = stageCheckpoints[Math.floor(stageCheckpoints.length * 0.25)];
      const mid = stageCheckpoints[Math.floor(stageCheckpoints.length * 0.5)];
      const q3 = stageCheckpoints[Math.floor(stageCheckpoints.length * 0.75)];
      const best = stageCheckpoints.reduce((acc, cp) => {
        if (cp.success_rate > acc.success_rate) return cp;
        if (cp.success_rate === acc.success_rate && (cp.average_completion_steps ?? Infinity) < (acc.average_completion_steps ?? Infinity)) return cp;
        return acc;
      }, stageCheckpoints[stageCheckpoints.length - 1]);

      const selected = new Set([first, q1, mid, q3, best]);
      milestones.push(...selected);
    }
  }

  return milestones;
}

export function ResultsPanel({ checkpointIndex }: ResultsPanelProps) {
  const milestoneCheckpoints = useMemo(
    () => selectMilestones(checkpointIndex?.checkpoints ?? []),
    [checkpointIndex],
  );

  const [replays, setReplays] = useState<Map<number, MilestoneReplay>>(new Map());
  const [playing, setPlaying] = useState(false);
  const [playbackSpeed, setPlaybackSpeed] = useState<"slow" | "normal" | "fast">("normal");

  // Load replays for milestone checkpoints
  useEffect(() => {
    if (!milestoneCheckpoints.length) {
      setReplays(new Map());
      return;
    }

    const newReplays = new Map<number, MilestoneReplay>();
    for (const cp of milestoneCheckpoints) {
      newReplays.set(cp.checkpoint_episode, {
        checkpoint: cp,
        bundle: null,
        frameIndex: 0,
        loading: true,
        error: null,
      });
    }
    setReplays(newReplays);

    let cancelled = false;

    const loadAll = async () => {
      for (const cp of milestoneCheckpoints) {
        if (cancelled) return;

        const record = cp.records[0];
        if (!record) continue;

        try {
          const bundle = await loadReplay(record.replay_path);
          if (cancelled) return;

          setReplays((prev) => {
            const next = new Map(prev);
            const entry = next.get(cp.checkpoint_episode);
            if (entry) {
              next.set(cp.checkpoint_episode, { ...entry, bundle, loading: false });
            }
            return next;
          });
        } catch (error) {
          if (cancelled) return;

          setReplays((prev) => {
            const next = new Map(prev);
            const entry = next.get(cp.checkpoint_episode);
            if (entry) {
              next.set(cp.checkpoint_episode, {
                ...entry,
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

  // Synchronized playback across all replays
  useEffect(() => {
    if (!playing) return;

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
        <div>
          <p className="eyebrow">Learning progression</p>
          <h2>Training milestones</h2>
          <p className="results-panel__description">
            Showing {milestoneCheckpoints.length} milestone{milestoneCheckpoints.length === 1 ? "" : "s"} across curriculum stages
          </p>
        </div>
        <div className="results-panel__controls">
          <button onClick={handlePlayPause} disabled={Array.from(replays.values()).every((r) => r.loading)}>
            {playing ? "⏸ Pause" : "▶ Play All"}
          </button>
          <button onClick={handleReset} disabled={playing}>
            ↻ Reset
          </button>
          <select
            value={playbackSpeed}
            onChange={(e) => setPlaybackSpeed(e.target.value as "slow" | "normal" | "fast")}
            disabled={playing}
          >
            <option value="slow">Slow</option>
            <option value="normal">Normal</option>
            <option value="fast">Fast</option>
          </select>
        </div>
      </div>

      <div className="results-grid">
        {milestoneCheckpoints.map((cp) => {
          const replay = replays.get(cp.checkpoint_episode);
          if (!replay) return null;

          const snapshot = replay.bundle?.frames[replay.frameIndex]?.snapshot ?? replay.bundle?.final_snapshot ?? null;
          const stage = cp.reward_config?.instincts?.curriculum_stage ?? 0;

          return (
            <MilestoneCard
              key={cp.checkpoint_episode}
              checkpoint={cp}
              snapshot={snapshot}
              stage={stage}
              loading={replay.loading}
              error={replay.error}
              frameIndex={replay.frameIndex}
              totalFrames={replay.bundle?.frames.length ?? 0}
            />
          );
        })}
      </div>
    </div>
  );
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

  const stageColors: Record<number, string> = {
    0: "#9ca3af",
    1: "#60a5fa",
    2: "#34d399",
    3: "#f59e0b",
    4: "#f472b6",
    5: "#c084fc",
  };

  const stageColor = stageColors[stage] ?? "#9ca3af";

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
    <div className="milestone-card">
      <div className="milestone-card__header">
        <div>
          <span className="milestone-card__stage" style={{ color: stageColor }}>
            Stage {stage}
          </span>
          <h3 className="milestone-card__title">Episode {checkpoint.checkpoint_episode}</h3>
        </div>
        <div className="milestone-card__stats">
          <span className="milestone-stat milestone-stat--success">
            {(checkpoint.success_rate * 100).toFixed(0)}% success
          </span>
          {checkpoint.average_completion_steps !== null && (
            <span className="milestone-stat">
              {checkpoint.average_completion_steps.toFixed(0)} steps
            </span>
          )}
        </div>
      </div>

      <div className="milestone-card__field">
        {loading ? (
          <div className="milestone-card__loading">Loading...</div>
        ) : error ? (
          <div className="milestone-card__error">{error}</div>
        ) : snapshot ? (
          <svg
            viewBox={`0 0 ${width} ${height}`}
            className="milestone-card__svg"
            style={{ backgroundColor: "rgba(4, 8, 14, 0.7)" }}
          >
            {/* Pen */}
            {fences.map((fence, i) => (
              <line
                key={i}
                x1={fence.x1}
                y1={fence.y1}
                x2={fence.x2}
                y2={fence.y2}
                stroke="#86efac"
                strokeWidth={fenceStroke}
                strokeLinecap="round"
              />
            ))}

            {/* Sheep */}
            {snapshot.sheep.map((sheep, i) => (
              <circle
                key={`sheep-${i}`}
                cx={sheep.x}
                cy={sheep.y}
                r={sheepRadius}
                fill={sheep.penned ? "#86efac" : "#f8fafc"}
                stroke={sheep.penned ? "#86efac" : "#cbd5e1"}
                strokeWidth={0.06 * densityScale}
              />
            ))}

            {/* Dogs */}
            {snapshot.dogs.map((dog, i) => (
              <circle
                key={`dog-${i}`}
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
              className="milestone-card__progress-fill"
              style={{ width: `${(frameIndex / (totalFrames - 1)) * 100}%` }}
            />
          </div>
          <span className="milestone-card__step">
            Step {snapshot?.step ?? 0} / {totalFrames - 1}
          </span>
        </div>
      )}
    </div>
  );
}

function dogColor(index: number): string {
  const palette = ["#3b82f6", "#10b981", "#f59e0b", "#ec4899", "#8b5cf6"];
  return palette[index % palette.length];
}

import type { CheckpointEntry, ReplaySnapshot } from "../state/types";
import type { GridMilestone } from "../components/ResultsPanel";

export interface MilestoneReplay {
  checkpoint: CheckpointEntry;
  bundle: any;
  frameIndex: number;
  loading: boolean;
  error: string | null;
}

export function buildResultsVideoPlan(
  gridMilestones: GridMilestone[],
  replays: Map<number, MilestoneReplay>,
  playbackSpeed: "slow" | "normal" | "fast",
  canvasWidth = 1920,
  canvasHeight = 1080
) {
  let maxFrames = 0;
  for (const milestone of gridMilestones) {
    if (milestone.checkpoint) {
      const replay = replays.get(milestone.checkpoint.checkpoint_episode);
      if (replay?.bundle) {
        maxFrames = Math.max(maxFrames, replay.bundle.frames.length);
      }
    }
  }

  if (maxFrames === 0) {
    maxFrames = 1;
  }

  const delays = { slow: 300, normal: 150, fast: 50 };
  const frameDelayMs = delays[playbackSpeed] ?? 150;
  const maxStage = gridMilestones.reduce((highest, item) => Math.max(highest, item.stage), 0);

  return {
    totalFrames: maxFrames,
    frameDelayMs,
    gridCols: 5,
    gridRows: Math.max(1, maxStage),
    canvasWidth,
    canvasHeight,
  };
}

export function getReplaySnapshotForExportFrame(
  replay: MilestoneReplay,
  frameIndex: number
): ReplaySnapshot | null {
  if (!replay.bundle) return null;
  const frames = replay.bundle.frames;
  if (!frames || frames.length === 0) {
    return replay.bundle.final_snapshot ?? null;
  }
  const idx = Math.min(frameIndex, frames.length - 1);
  return frames[idx]?.snapshot ?? replay.bundle.final_snapshot ?? null;
}

export function makeResultsVideoFileName(date = new Date()): string {
  const pad = (n: number) => String(n).padStart(2, "0");
  const yyyy = date.getFullYear();
  const mm = pad(date.getMonth() + 1);
  const dd = pad(date.getDate());
  const hh = pad(date.getHours());
  const min = pad(date.getMinutes());
  const ss = pad(date.getSeconds());
  return `sheepdog-results-video-${yyyy}${mm}${dd}-${hh}${min}${ss}.webm`;
}

function getStageColor(stage: number): string {
  const colors: Record<number, string> = {
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
  if (colors[stage]) {
    return colors[stage];
  }
  const hue = ((stage - 1) * 47) % 360;
  return `hsl(${hue} 74% 64%)`;
}

function drawRoundedRect(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  w: number,
  h: number,
  r: number
) {
  if (typeof ctx.roundRect === "function") {
    ctx.roundRect(x, y, w, h, r);
  } else {
    ctx.moveTo(x + r, y);
    ctx.lineTo(x + w - r, y);
    ctx.quadraticCurveTo(x + w, y, x + w, y + r);
    ctx.lineTo(x + w, y + h - r);
    ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
    ctx.lineTo(x + r, y + h);
    ctx.quadraticCurveTo(x, y + h, x, y + h - r);
    ctx.lineTo(x, y + r);
    ctx.quadraticCurveTo(x, y, x + r, y);
  }
}

export function drawResultsVideoFrame(
  ctx: CanvasRenderingContext2D,
  gridMilestones: GridMilestone[],
  replays: Map<number, MilestoneReplay>,
  frameIndex: number,
  plan: {
    totalFrames: number;
    frameDelayMs: number;
    gridCols: number;
    gridRows: number;
    canvasWidth: number;
    canvasHeight: number;
  }
): void {
  const { canvasWidth, canvasHeight, gridCols, gridRows } = plan;

  // 1. Draw outer background
  ctx.fillStyle = "#04080e";
  ctx.fillRect(0, 0, canvasWidth, canvasHeight);

  // Layout parameters
  const padding = 16;
  const gap = 8;
  const cardWidth = (canvasWidth - 2 * padding - (gridCols - 1) * gap) / gridCols;
  const cardHeight = (canvasHeight - 2 * padding - (gridRows - 1) * gap) / gridRows;

  // Draw each milestone card
  for (const item of gridMilestones) {
    const col = item.slot - 1;
    const row = item.stage - 1;
    if (col < 0 || col >= gridCols || row < 0 || row >= gridRows) {
      continue;
    }

    const x = padding + col * (cardWidth + gap);
    const y = padding + row * (cardHeight + gap);

    // Save context state for this card
    ctx.save();

    // 2. Draw card container
    ctx.beginPath();
    drawRoundedRect(ctx, x, y, cardWidth, cardHeight, 6);
    ctx.fillStyle = "#0d1527";
    ctx.fill();
    ctx.strokeStyle = getStageColor(item.stage);
    ctx.lineWidth = 1;
    if (!item.checkpoint) {
      ctx.setLineDash([4, 4]);
    } else {
      ctx.setLineDash([]);
    }
    ctx.stroke();
    ctx.setLineDash([]);

    // 3. Draw Header
    const badgeText = `S${item.stage}`;
    ctx.font = "bold 9px sans-serif";
    const badgeWidth = ctx.measureText(badgeText).width + 8;
    const badgeHeight = 12;
    const badgeX = x + 8;
    const badgeY = y + 6;

    // Draw Badge
    ctx.beginPath();
    drawRoundedRect(ctx, badgeX, badgeY, badgeWidth, badgeHeight, 6);
    ctx.fillStyle = getStageColor(item.stage);
    ctx.fill();

    ctx.fillStyle = "#000000";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(badgeText, badgeX + badgeWidth / 2, badgeY + badgeHeight / 2);

    // Draw Title
    ctx.font = "bold 11px sans-serif";
    ctx.fillStyle = "#ffffff";
    ctx.textAlign = "left";
    ctx.textBaseline = "top";
    const titleText = item.checkpoint
      ? `Episode ${item.checkpoint.checkpoint_episode}`
      : `Replay slot ${item.slot}`;
    ctx.fillText(titleText, x + 8 + badgeWidth + 6, y + 6);

    // Draw Success percentage
    ctx.textAlign = "right";
    ctx.textBaseline = "top";
    if (item.checkpoint) {
      const successPercent = Math.round(item.checkpoint.success_rate * 100);
      ctx.font = "bold 10px sans-serif";
      ctx.fillStyle =
        successPercent === 100 ? "#3fb950" : successPercent === 0 ? "#f85149" : "#9ca3af";
      ctx.fillText(`${successPercent}%`, x + cardWidth - 8, y + 6);
    } else {
      ctx.font = "bold 10px sans-serif";
      ctx.fillStyle = "#9ca3af";
      ctx.fillText("--", x + cardWidth - 8, y + 6);
    }

    // 4. Draw field area
    const p = 8;
    const headerHeight = 18;
    const progressHeight = 14;
    const fieldAreaX = x + p;
    const fieldAreaY = y + p + headerHeight;
    const fieldAreaWidth = cardWidth - 2 * p;
    const fieldAreaHeight = cardHeight - 2 * p - headerHeight - progressHeight;

    ctx.fillStyle = "rgba(4, 8, 14, 0.7)";
    ctx.fillRect(fieldAreaX, fieldAreaY, fieldAreaWidth, fieldAreaHeight);

    const replay = item.checkpoint ? replays.get(item.checkpoint.checkpoint_episode) : null;

    if (!item.checkpoint) {
      // Placeholder state
      ctx.font = "9px sans-serif";
      ctx.fillStyle = "#9ca3af";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(
        "Awaiting more training data",
        fieldAreaX + fieldAreaWidth / 2,
        fieldAreaY + fieldAreaHeight / 2
      );
    } else if (!replay) {
      // Missing state
      ctx.font = "9px sans-serif";
      ctx.fillStyle = "#9ca3af";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText("No replay", fieldAreaX + fieldAreaWidth / 2, fieldAreaY + fieldAreaHeight / 2);
    } else if (replay.loading) {
      // Loading state
      ctx.font = "9px sans-serif";
      ctx.fillStyle = "#9ca3af";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText("Loading...", fieldAreaX + fieldAreaWidth / 2, fieldAreaY + fieldAreaHeight / 2);
    } else if (replay.error) {
      // Error state
      ctx.font = "9px sans-serif";
      ctx.fillStyle = "#f85149";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(
        replay.error,
        fieldAreaX + fieldAreaWidth / 2,
        fieldAreaY + fieldAreaHeight / 2
      );
    } else {
      // Standard simulation drawing
      const snapshot = getReplaySnapshotForExportFrame(replay, frameIndex);
      if (snapshot) {
        const baseWidth = snapshot.grid_width ?? snapshot.field_width ?? 40;
        const baseHeight = snapshot.grid_height ?? snapshot.field_height ?? 30;
        const width = Math.max(baseWidth, 40);
        const height = Math.max(baseHeight, 30);
        const aspect = width / height;

        let dw = fieldAreaWidth;
        let dh = fieldAreaHeight;
        if (fieldAreaWidth / fieldAreaHeight > aspect) {
          dw = fieldAreaHeight * aspect;
          dh = fieldAreaHeight;
        } else {
          dw = fieldAreaWidth;
          dh = fieldAreaWidth / aspect;
        }

        const ox = fieldAreaX + (fieldAreaWidth - dw) / 2;
        const oy = fieldAreaY + (fieldAreaHeight - dh) / 2;

        // Draw sub-field background
        ctx.fillStyle = "rgba(4, 8, 14, 0.9)";
        ctx.fillRect(ox, oy, dw, dh);

        const scale = dw / width;
        const densityScale = Math.max(width / 40, height / 30, 1);
        const dogRadius = 0.48 * densityScale * scale;
        const sheepRadius = 0.42 * densityScale * scale;
        const fenceStroke = 0.32 * densityScale * scale;

        // Helper to extract segments
        const { pen } = snapshot;
        const opening = pen.opening ?? "left";
        const penOx = pen.origin.x;
        const penOy = pen.origin.y;
        const penRight = penOx + pen.width;
        const penBottom = penOy + pen.height;
        const allSegments = [
          { side: "top", x1: penOx, y1: penOy, x2: penRight, y2: penOy },
          { side: "bottom", x1: penOx, y1: penBottom, x2: penRight, y2: penBottom },
          { side: "left", x1: penOx, y1: penOy, x2: penOx, y2: penBottom },
          { side: "right", x1: penRight, y1: penOy, x2: penRight, y2: penBottom },
        ];
        const fences = allSegments.filter((segment) => segment.side !== opening);

        // Draw fences
        ctx.strokeStyle = "#86efac";
        ctx.lineWidth = fenceStroke;
        ctx.lineCap = "round";
        for (const fence of fences) {
          ctx.beginPath();
          ctx.moveTo(ox + fence.x1 * scale, oy + fence.y1 * scale);
          ctx.lineTo(ox + fence.x2 * scale, oy + fence.y2 * scale);
          ctx.stroke();
        }

        // Draw sheep
        for (const sheep of snapshot.sheep) {
          ctx.beginPath();
          ctx.arc(ox + sheep.x * scale, oy + sheep.y * scale, sheepRadius, 0, 2 * Math.PI);
          if (sheep.penned) {
            ctx.fillStyle = "#86efac";
            ctx.fill();
            ctx.strokeStyle = "#86efac";
          } else {
            ctx.fillStyle = "#f8fafc";
            ctx.fill();
            ctx.strokeStyle = "#cbd5e1";
          }
          ctx.lineWidth = 0.06 * densityScale * scale;
          ctx.stroke();
        }

        // Draw dogs
        const dogPalette = ["#3b82f6", "#10b981", "#f59e0b", "#ec4899", "#8b5cf6"];
        for (const dog of snapshot.dogs) {
          ctx.beginPath();
          ctx.arc(ox + dog.x * scale, oy + dog.y * scale, dogRadius, 0, 2 * Math.PI);
          ctx.fillStyle = dogPalette[dog.index % dogPalette.length];
          ctx.fill();
          ctx.strokeStyle = "rgba(255,255,255,0.7)";
          ctx.lineWidth = 0.08 * densityScale * scale;
          ctx.stroke();
        }
      }
    }

    // 5. Draw Progress
    const barX = x + p;
    const barY = y + cardHeight - p - progressHeight + 4;
    const barWidth = cardWidth - 2 * p;
    const barHeight = 3;

    ctx.fillStyle = "rgba(148, 163, 184, 0.15)";
    ctx.beginPath();
    drawRoundedRect(ctx, barX, barY, barWidth, barHeight, 2);
    ctx.fill();

    let progressRatio = 0;
    let totalFramesText = "";
    let stepText = "0";
    let progressPercentText = "0%";
    let fillStyleColor = "#f85149";

    if (item.checkpoint && replay?.bundle) {
      const framesCount = replay.bundle.frames.length;
      const currentFrameIdx = Math.min(frameIndex, framesCount - 1);
      progressRatio = framesCount > 1 ? currentFrameIdx / (framesCount - 1) : 0;

      const snapshot = getReplaySnapshotForExportFrame(replay, frameIndex);
      stepText = String(snapshot?.step ?? 0);
      totalFramesText = String(framesCount - 1);
      progressPercentText = `${Math.round(progressRatio * 100)}%`;

      const successPercent = Math.round(item.checkpoint.success_rate * 100);
      if (successPercent === 100) {
        fillStyleColor = "#3fb950";
      } else if (progressRatio > 0.6) {
        fillStyleColor = "#d29922";
      } else {
        fillStyleColor = "#f85149";
      }
    } else if (!item.checkpoint) {
      stepText = "No replay yet";
      totalFramesText = "";
      progressPercentText = "0%";
      progressRatio = 0;
    }

    if (progressRatio > 0) {
      ctx.fillStyle = fillStyleColor;
      ctx.beginPath();
      drawRoundedRect(ctx, barX, barY, barWidth * progressRatio, barHeight, 2);
      ctx.fill();
    }

    const textY = barY + barHeight + 3;
    ctx.font = "8px sans-serif";
    ctx.fillStyle = "#9ca3af";
    ctx.textBaseline = "top";

    ctx.textAlign = "left";
    const leftText = totalFramesText ? `${stepText} / ${totalFramesText}` : stepText;
    ctx.fillText(leftText, barX, textY);

    ctx.textAlign = "right";
    ctx.fillText(progressPercentText, barX + barWidth, textY);

    ctx.restore();
  }
}

export async function exportResultsVideo(
  gridMilestones: GridMilestone[],
  replays: Map<number, MilestoneReplay>,
  playbackSpeed: "slow" | "normal" | "fast",
  onProgress?: (progress: number) => void
): Promise<Blob> {
  if (
    typeof window === "undefined" ||
    !HTMLCanvasElement.prototype.captureStream ||
    !window.MediaRecorder
  ) {
    throw new Error("Video export is not supported in this browser");
  }

  const canvas = document.createElement("canvas");
  canvas.width = 1920;
  canvas.height = 1080;
  const ctx = canvas.getContext("2d");
  if (!ctx) {
    throw new Error("Failed to get 2D context");
  }

  const plan = buildResultsVideoPlan(
    gridMilestones,
    replays,
    playbackSpeed,
    canvas.width,
    canvas.height
  );

  const fps = 30;
  const stream = canvas.captureStream(fps);

  const mimeTypes = [
    "video/webm;codecs=vp9",
    "video/webm;codecs=vp8",
    "video/webm",
  ];
  let selectedMimeType = "";
  for (const mime of mimeTypes) {
    if (MediaRecorder.isTypeSupported(mime)) {
      selectedMimeType = mime;
      break;
    }
  }

  const chunks: Blob[] = [];
  const recorder = new MediaRecorder(
    stream,
    selectedMimeType ? { mimeType: selectedMimeType } : undefined
  );

  recorder.ondataavailable = (event) => {
    if (event.data && event.data.size > 0) {
      chunks.push(event.data);
    }
  };

  const recordPromise = new Promise<Blob>((resolve, reject) => {
    recorder.onstop = () => {
      const blob = new Blob(chunks, { type: selectedMimeType || "video/webm" });
      resolve(blob);
    };
    recorder.onerror = (e) => {
      reject(e);
    };
  });

  recorder.start();

  const frameDurationMs = plan.frameDelayMs;
  const totalFrames = plan.totalFrames;

  let worker: Worker | null = null;
  try {
    const workerCode = `
      self.onmessage = function(e) {
        setTimeout(function() {
          postMessage('done');
        }, e.data.delay);
      };
    `;
    const blob = new Blob([workerCode], { type: "application/javascript" });
    worker = new Worker(URL.createObjectURL(blob));
  } catch (e) {
    console.warn("Failed to create background worker timer, falling back to main-thread timer:", e);
  }

  const startTime = performance.now();
  for (let f = 0; f < totalFrames; f++) {
    drawResultsVideoFrame(ctx, gridMilestones, replays, f, plan);

    if (onProgress) {
      onProgress((f + 1) / totalFrames);
    }

    const nextFrameTime = startTime + (f + 1) * frameDurationMs;
    const now = performance.now();
    const delay = nextFrameTime - now;

    if (delay > 0) {
      if (worker) {
        await new Promise<void>((resolve) => {
          worker!.onmessage = () => resolve();
          worker!.postMessage({ delay });
        });
      } else {
        await new Promise((resolve) => setTimeout(resolve, delay));
      }
    } else {
      // Yield to the event loop to keep the UI responsive
      await new Promise((resolve) => setTimeout(resolve, 0));
    }
  }

  if (worker) {
    worker.terminate();
  }

  recorder.stop();
  stream.getTracks().forEach((track) => track.stop());

  return recordPromise;
}

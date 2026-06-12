import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import {
  buildResultsVideoPlan,
  getReplaySnapshotForExportFrame,
  makeResultsVideoFileName,
  drawResultsVideoFrame,
  exportResultsVideo,
  MilestoneReplay,
} from "./resultsVideo";
import type { GridMilestone } from "../components/ResultsPanel";
import type { CheckpointEntry, ReplaySnapshot } from "../state/types";

describe("resultsVideo helpers", () => {
  beforeEach(() => {
    vi.stubGlobal("URL", {
      createObjectURL: vi.fn().mockReturnValue("blob:http://localhost/mock-blob"),
      revokeObjectURL: vi.fn(),
    });

    const mockCtx = {
      fillRect: vi.fn(),
      fillText: vi.fn(),
      beginPath: vi.fn(),
      arc: vi.fn(),
      fill: vi.fn(),
      stroke: vi.fn(),
      lineTo: vi.fn(),
      moveTo: vi.fn(),
      strokeText: vi.fn(),
      save: vi.fn(),
      restore: vi.fn(),
      setLineDash: vi.fn(),
      measureText: vi.fn().mockReturnValue({ width: 20 }),
      quadraticCurveTo: vi.fn(),
      rect: vi.fn(),
      closePath: vi.fn(),
    } as unknown as CanvasRenderingContext2D;

    vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue(mockCtx);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  it("buildResultsVideoPlan calculates frame delay and total frames", () => {
    const mockReplays = new Map<number, MilestoneReplay>([
      [
        10,
        {
          checkpoint: { checkpoint_episode: 10 } as CheckpointEntry,
          bundle: { frames: new Array(5).fill({}) },
          frameIndex: 0,
          loading: false,
          error: null,
        },
      ],
      [
        20,
        {
          checkpoint: { checkpoint_episode: 20 } as CheckpointEntry,
          bundle: { frames: new Array(12).fill({}) },
          frameIndex: 0,
          loading: false,
          error: null,
        },
      ],
    ]);

    const milestones: GridMilestone[] = [
      { stage: 1, slot: 1, checkpoint: { checkpoint_episode: 10 } as CheckpointEntry },
      { stage: 1, slot: 2, checkpoint: { checkpoint_episode: 20 } as CheckpointEntry },
    ];

    const planSlow = buildResultsVideoPlan(milestones, mockReplays, "slow");
    expect(planSlow.totalFrames).toBe(12);
    expect(planSlow.frameDelayMs).toBe(300);

    const planNormal = buildResultsVideoPlan(milestones, mockReplays, "normal");
    expect(planNormal.frameDelayMs).toBe(150);

    const planFast = buildResultsVideoPlan(milestones, mockReplays, "fast");
    expect(planFast.frameDelayMs).toBe(50);
  });

  it("getReplaySnapshotForExportFrame holds final frame when index exceeds length", () => {
    const mockSnapshot1 = { step: 1 } as ReplaySnapshot;
    const mockSnapshot2 = { step: 2 } as ReplaySnapshot;
    const mockFinalSnapshot = { step: 99 } as ReplaySnapshot;

    const replay: MilestoneReplay = {
      checkpoint: { checkpoint_episode: 10 } as CheckpointEntry,
      bundle: {
        frames: [
          { step: 1, actions: [], snapshot: mockSnapshot1, reward: {} as any },
          { step: 2, actions: [], snapshot: mockSnapshot2, reward: {} as any },
        ],
        final_snapshot: mockFinalSnapshot,
      },
      frameIndex: 0,
      loading: false,
      error: null,
    };

    expect(getReplaySnapshotForExportFrame(replay, 0)).toBe(mockSnapshot1);
    expect(getReplaySnapshotForExportFrame(replay, 1)).toBe(mockSnapshot2);
    // Exceeds frames length, should hold final frame snapshot
    expect(getReplaySnapshotForExportFrame(replay, 2)).toBe(mockSnapshot2);
    expect(getReplaySnapshotForExportFrame(replay, 10)).toBe(mockSnapshot2);
  });

  it("makeResultsVideoFileName matches standard YYYYMMDD-HHMMSS format", () => {
    const fixedDate = new Date(2026, 5, 10, 11, 14, 15); // Month is 0-indexed (5 = June)
    const name = makeResultsVideoFileName(fixedDate);
    expect(name).toBe("sheepdog-results-video-20260610-111415.webm");
  });

  it("drawResultsVideoFrame invokes context methods to render layout", () => {
    const mockCtx = {
      fillRect: vi.fn(),
      fillText: vi.fn(),
      beginPath: vi.fn(),
      arc: vi.fn(),
      fill: vi.fn(),
      stroke: vi.fn(),
      lineTo: vi.fn(),
      moveTo: vi.fn(),
      strokeText: vi.fn(),
      save: vi.fn(),
      restore: vi.fn(),
      setLineDash: vi.fn(),
      measureText: vi.fn().mockReturnValue({ width: 20 }),
      quadraticCurveTo: vi.fn(),
      rect: vi.fn(),
      closePath: vi.fn(),
    } as unknown as CanvasRenderingContext2D;

    const milestones: GridMilestone[] = [
      { stage: 1, slot: 1, checkpoint: { checkpoint_episode: 10, success_rate: 0.8 } as CheckpointEntry },
      { stage: 1, slot: 2, checkpoint: null }, // placeholder
    ];

    const replayBundle = {
      frames: [{ step: 1, snapshot: { step: 1, sheep: [], dogs: [], pen: { origin: { x: 0, y: 0 }, width: 10, height: 10 } } as any }],
      final_snapshot: {} as any,
    };

    const mockReplays = new Map<number, MilestoneReplay>([
      [
        10,
        {
          checkpoint: { checkpoint_episode: 10 } as CheckpointEntry,
          bundle: replayBundle,
          frameIndex: 0,
          loading: false,
          error: null,
        },
      ],
    ]);

    const plan = {
      totalFrames: 1,
      frameDelayMs: 150,
      gridCols: 5,
      gridRows: 5,
      canvasWidth: 1920,
      canvasHeight: 1080,
    };

    drawResultsVideoFrame(mockCtx, milestones, mockReplays, 0, plan);

    // Should have filled rects for backgrounds, badges, and drawn texts
    expect(mockCtx.fillRect).toHaveBeenCalled();
    expect(mockCtx.fillText).toHaveBeenCalled();
  });

  it("exportResultsVideo records canvas and returns WebM blob", async () => {
    class MockMediaRecorder {
      start = vi.fn();
      stop = vi.fn(() => {
        if (this.onstop) this.onstop();
      });
      ondataavailable: ((e: any) => void) | null = null;
      onstop: (() => void) | null = null;
      static isTypeSupported = vi.fn().mockReturnValue(true);
      constructor(_stream: any, _options: any) {
        setTimeout(() => {
          if (this.ondataavailable) {
            this.ondataavailable({ data: new Blob(["webm-data"], { type: "video/webm" }) });
          }
          this.stop();
        }, 20);
      }
    }

    vi.stubGlobal("MediaRecorder", MockMediaRecorder);

    const mockCaptureStream = vi.fn().mockReturnValue({
      getTracks: vi.fn().mockReturnValue([{ stop: vi.fn() }]),
    });
    HTMLCanvasElement.prototype.captureStream = mockCaptureStream;

    const milestones: GridMilestone[] = [
      { stage: 1, slot: 1, checkpoint: { checkpoint_episode: 10 } as CheckpointEntry },
    ];
    const mockReplays = new Map<number, MilestoneReplay>([
      [
        10,
        {
          checkpoint: { checkpoint_episode: 10 } as CheckpointEntry,
          bundle: { frames: [{ step: 1, snapshot: { sheep: [], dogs: [], pen: { origin: { x: 0, y: 0 }, width: 10, height: 10 } } as any }] },
          frameIndex: 0,
          loading: false,
          error: null,
        },
      ],
    ]);

    const blob = await exportResultsVideo(milestones, mockReplays, "fast");
    expect(blob).toBeInstanceOf(Blob);
    expect(blob.type).toContain("video/webm");
  });

  it("exportResultsVideo throws clear error when unsupported", async () => {
    // Temporarily remove API
    vi.stubGlobal("MediaRecorder", undefined);

    await expect(exportResultsVideo([], new Map(), "normal")).rejects.toThrow(
      "Video export is not supported in this browser"
    );
  });
});

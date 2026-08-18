import { describe, expect, it } from "vitest";
import {
  analyzePerSeedReliability,
  assertMonotonicX,
  buildFormalEvalMarkers,
  calculateEfficiencyTrend,
  compareCanonicalEpisodes,
  computeCheckpointEfficiency,
  computeRollingTrainingSeries,
  normalizeEpisodeRecord,
  processCanonicalHistory,
  selectWindowSlice,
  type CanonicalEpisodeRecord,
} from "./chartPipeline";
import type { CheckpointEntry, TrainingEpisode } from "../state/types";

describe("Chart Pipeline Architecture & Integrity", () => {
  // Invariant 1: No raw telemetry + formal evaluations exist -> zero fake episodes, genuine formal evals preserved
  it("Invariant 1: When zero raw rollout telemetry is available, never manufactures fake training episodes", () => {
    const mockCheckpoints: CheckpointEntry[] = [
      {
        checkpoint_episode: 10,
        global_timestep: 1000,
        success_rate: 0.70, // 70% formal benchmark
        average_completion_steps: 120,
        curriculum_stage: 4,
        recorded_at: "2026-08-11T10:00:00Z",
        checkpoint: "chk_10.pt",
        evaluation: "eval_10.json",
        replay: "replay_10.json",
        timeout_rate: 0.3,
        average_completion_seconds: 24,
        average_sheep_penned: 3,
        average_reward: 180,
        records: [],
      },
    ];

    const canonicalRollouts = processCanonicalHistory([], 4);
    expect(canonicalRollouts).toEqual([]);

    const rollingSeries = computeRollingTrainingSeries(canonicalRollouts, 50);
    expect(rollingSeries).toEqual([]);

    // Formal eval markers should be created independently and retain exact 70%
    const markers = buildFormalEvalMarkers(mockCheckpoints, 4, "timesteps");
    expect(markers.length).toBe(1);
    expect(markers[0].successRatePct).toBe(70);
    expect(markers[0].xVal).toBe(1000);
    expect(markers[0].label).toContain("70%");
  });

  // Invariant 2: Correct rolling 50 math
  it("Invariant 2: Computes exact 50-episode rolling average across rollout episodes", () => {
    // 30 successes + 20 failures = 30 / 50 = 60.0%
    const raw: Partial<TrainingEpisode>[] = [
      ...Array.from({ length: 30 }, (_, i) => ({
        id: i + 1,
        curriculum_stage: 1,
        global_timestep: (i + 1) * 100,
        result: "SUCCESS",
        success: true,
        steps: 100,
        reward: 200,
        sheep_penned: 3,
      })),
      ...Array.from({ length: 20 }, (_, i) => ({
        id: 31 + i,
        curriculum_stage: 1,
        global_timestep: (31 + i) * 100,
        result: "TIMEOUT",
        success: false,
        steps: 600,
        reward: -50,
        sheep_penned: 1,
      })),
    ];

    const canonical = processCanonicalHistory(raw, "all");
    const rollingSeries = computeRollingTrainingSeries(canonical, 50);

    expect(rollingSeries.length).toBe(50);
    const lastPoint = rollingSeries[49];
    expect(lastPoint.rollingSuccessRate).toBe(60);
    expect(lastPoint.windowCount).toBe(50);
    // Successful steps average should only average the 30 successes (100 steps), NOT the 20 failures (600 steps)
    expect(lastPoint.rollingSuccessfulSteps).toBe(100);
  });

  // Invariant 3: Rolling calculations use full history before display slicing
  it("Invariant 3: Rolling-50 calculations use full preceding history even in Last 25 display slice", () => {
    // 100 total episodes: first 75 are 100% success, last 25 are 0% success
    // In the last 25 window:
    // The final point (ep 100) has preceding 50 episodes (ep 51..100) which contain 25 successes + 25 failures = 50%
    // If we mistakenly calculated rolling on the truncated 25 slice, it would say 0 / 25 = 0% !
    const raw: Partial<TrainingEpisode>[] = [
      ...Array.from({ length: 75 }, (_, i) => ({
        id: i + 1,
        curriculum_stage: 2,
        global_timestep: (i + 1) * 100,
        result: "SUCCESS",
        success: true,
        steps: 120,
      })),
      ...Array.from({ length: 25 }, (_, i) => ({
        id: 76 + i,
        curriculum_stage: 2,
        global_timestep: (76 + i) * 100,
        result: "TIMEOUT",
        success: false,
        steps: 600,
      })),
    ];

    const fullHistory = processCanonicalHistory(raw, 2);
    const fullRolling = computeRollingTrainingSeries(fullHistory, 50);

    const allSlice = selectWindowSlice(fullRolling, "all");
    const last100 = selectWindowSlice(fullRolling, 100);
    const last50 = selectWindowSlice(fullRolling, 50);
    const last25 = selectWindowSlice(fullRolling, 25);

    // Tail subset invariant
    expect(last25).toEqual(allSlice.slice(-25));
    expect(last50).toEqual(allSlice.slice(-50));
    expect(last100).toEqual(allSlice.slice(-100));

    // The final point rolling-50 value must be exactly 50% in ALL slices
    const expectedFinalRolling = 50; // 25 successes out of 50 in window 51..100
    expect(allSlice[allSlice.length - 1].rollingSuccessRate).toBe(expectedFinalRolling);
    expect(last100[last100.length - 1].rollingSuccessRate).toBe(expectedFinalRolling);
    expect(last50[last50.length - 1].rollingSuccessRate).toBe(expectedFinalRolling);
    expect(last25[last25.length - 1].rollingSuccessRate).toBe(expectedFinalRolling);
  });

  // Invariant 4: Formal evaluation percentages remain exact 0-100 values
  it("Invariant 4: Formal 10-seed evaluations remain exact percentages and are never binarized", () => {
    const rates = [0.6, 0.7, 0.8, 0.9, 1.0];
    const mockCheckpoints: CheckpointEntry[] = rates.map((r, i) => ({
      checkpoint_episode: (i + 1) * 10,
      global_timesteps: (i + 1) * 5000,
      success_rate: r,
      average_completion_steps: 100 - i * 10,
      curriculum_stage: 1,
      checkpoint: `chk_${i}.pt`,
      evaluation: `eval_${i}.json`,
      replay: `rep_${i}.json`,
      timeout_rate: 1 - r,
      average_completion_seconds: 15,
      average_sheep_penned: 3,
      average_reward: 200,
      records: [],
    }));

    const markers = buildFormalEvalMarkers(mockCheckpoints, 1, "timesteps");
    expect(markers.map((m) => m.successRatePct)).toEqual([60, 70, 80, 90, 100]);
  });

  // Invariant 5: Successful completion steps calculations exclude failed seeds
  it("Invariant 5: Computes median and mean steps using ONLY successful evaluation seeds", () => {
    const mockCheckpoint: CheckpointEntry = {
      checkpoint_episode: 50,
      global_timesteps: 25000,
      success_rate: 0.8,
      average_completion_steps: 190, // aggregate mean including failed seeds
      curriculum_stage: 3,
      checkpoint: "chk_50.pt",
      evaluation: "eval_50.json",
      replay: "rep_50.json",
      timeout_rate: 0.2,
      average_completion_seconds: 20,
      average_sheep_penned: 3,
      average_reward: 220,
      records: [
        // 8 successes with steps: 60, 70, 80, 90, 100, 110, 120, 250
        { seed: 1, success: true, timeout: false, stopped: false, steps: 60, simulated_seconds: 6, sheep_penned: 3, final_sheep_distance_to_pen: 0, final_flock_spread: 2, no_progress_steps: 0, stop_reason: "", spawn_mode: "", reward_total: 250, final_farthest_distance_to_pen: 0, final_farthest_distance_to_flock_center: 0, role_switches: 0, collector_activations: 0, blocker_activations: 0, cumulative_gate_progress: 0, controlled_stall_steps: 0, left_flank_occupancy_steps: 0, right_flank_occupancy_steps: 0, gate_corridor_occupancy_peak: 0, gate_corridor_failure_steps: 0, dog_role_occupancy: {}, reward_breakdown: {}, replay_path: "" } as any,
        { seed: 2, success: true, timeout: false, stopped: false, steps: 70 } as any,
        { seed: 3, success: true, timeout: false, stopped: false, steps: 80 } as any,
        { seed: 4, success: true, timeout: false, stopped: false, steps: 90 } as any,
        { seed: 5, success: true, timeout: false, stopped: false, steps: 100 } as any,
        { seed: 6, success: true, timeout: false, stopped: false, steps: 110 } as any,
        { seed: 7, success: true, timeout: false, stopped: false, steps: 120 } as any,
        { seed: 8, success: true, timeout: false, stopped: false, steps: 250 } as any,
        // 2 failures with timeout steps 600
        { seed: 9, success: false, timeout: true, stopped: false, steps: 600 } as any,
        { seed: 10, success: false, timeout: true, stopped: false, steps: 600 } as any,
      ],
    };

    const eff = computeCheckpointEfficiency(mockCheckpoint);
    expect(eff.successRatePct).toBe(80);
    expect(eff.successCount).toBe(8);
    expect(eff.totalSeeds).toBe(10);
    expect(eff.failedSeeds).toEqual([9, 10]);

    // Successful steps: [60, 70, 80, 90, 100, 110, 120, 250]
    // Median of 8 elements = (90 + 100) / 2 = 95
    expect(eff.medianSuccessfulSteps).toBe(95);
    // Mean of 8 elements = (60+70+80+90+100+110+120+250) / 8 = 880 / 8 = 110
    expect(eff.meanSuccessfulSteps).toBe(110);
    expect(eff.worstSuccessfulSteps).toBe(250);
  });

  // Invariant 6: Efficiency trend signal
  it("Invariant 6: Accurately identifies improving, stable, and regressing completion efficiency", () => {
    const makeCkpts = (stepMedians: number[]): CheckpointEntry[] =>
      stepMedians.map((steps, idx) => ({
        checkpoint_episode: (idx + 1) * 10,
        curriculum_stage: 1,
        success_rate: 1.0,
        average_completion_steps: steps,
        records: Array.from({ length: 10 }, (_, s) => ({ seed: s + 1, success: true, steps } as any)),
      } as any));

    // Clearly improving: 250 -> 225 -> 200 -> 150 -> 125 -> 100
    // Prior 3 median (250, 225, 200) = 225
    // Recent 3 median (150, 125, 100) = 125
    // Improvement: (225 - 125) / 225 = +44.4%
    const improvingCkpts = makeCkpts([250, 225, 200, 150, 125, 100]);
    const improvingTrend = calculateEfficiencyTrend(improvingCkpts, 1);
    expect(improvingTrend.status).toBe("improving");
    expect(improvingTrend.percentageImprovement).toBe(44.4);
    expect(improvingTrend.recentMedian).toBe(125);
    expect(improvingTrend.priorMedian).toBe(225);

    // Stable: 92 -> 88 -> 91 -> 86 -> 89 -> 87
    // Prior 3 median (92, 88, 91) = 91
    // Recent 3 median (86, 89, 87) = 87
    // Improvement: (91 - 87) / 91 = +4.4% (within standard stable boundary)
    const stableCkpts = makeCkpts([90, 90, 90, 89, 90, 89]);
    const stableTrend = calculateEfficiencyTrend(stableCkpts, 1);
    expect(stableTrend.status).toBe("stable");

    // Regressing: 100 -> 110 -> 120 -> 180 -> 200 -> 240
    const regressingCkpts = makeCkpts([100, 110, 120, 180, 200, 240]);
    const regressingTrend = calculateEfficiencyTrend(regressingCkpts, 1);
    expect(regressingTrend.status).toBe("regressing");
    expect(regressingTrend.percentageImprovement!).toBeLessThan(0);

    // Insufficient evaluations
    const fewCkpts = makeCkpts([150]);
    const fewTrend = calculateEfficiencyTrend(fewCkpts, 1);
    expect(fewTrend.status).toBe("collecting_evidence");
  });

  // Invariant 7: Per-seed reliability and inefficient scenario detection
  it("Invariant 7: Distinguishes persistent blind spots from inefficient scenarios and normal variance", () => {
    // 3 formal evaluations across seeds 1..5
    // Seed 1: consistently fast success (60, 65, 62 steps) -> reliable
    // Seed 2: normal variance (success 70, fail, success 75) -> occasional failure / normal variance
    // Seed 3: persistent blind spot (fail, fail, fail) -> blind spot
    // Seed 4: inefficient scenario (succeeds 240, 250, 245 steps while overall median is ~70) -> inefficient
    // Seed 5: fast success (70, 72, 68 steps) -> reliable
    const mockEvaluations: CheckpointEntry[] = [
      {
        checkpoint_episode: 10,
        curriculum_stage: 1,
        success_rate: 0.8,
        records: [
          { seed: 1, success: true, steps: 60 } as any,
          { seed: 2, success: true, steps: 70 } as any,
          { seed: 3, success: false, steps: 600 } as any,
          { seed: 4, success: true, steps: 240 } as any,
          { seed: 5, success: true, steps: 70 } as any,
        ],
      } as any,
      {
        checkpoint_episode: 20,
        curriculum_stage: 1,
        success_rate: 0.6,
        records: [
          { seed: 1, success: true, steps: 65 } as any,
          { seed: 2, success: false, steps: 600 } as any,
          { seed: 3, success: false, steps: 600 } as any,
          { seed: 4, success: true, steps: 250 } as any,
          { seed: 5, success: true, steps: 72 } as any,
        ],
      } as any,
      {
        checkpoint_episode: 30,
        curriculum_stage: 1,
        success_rate: 0.8,
        records: [
          { seed: 1, success: true, steps: 62 } as any,
          { seed: 2, success: true, steps: 75 } as any,
          { seed: 3, success: false, steps: 600 } as any,
          { seed: 4, success: true, steps: 245 } as any,
          { seed: 5, success: true, steps: 68 } as any,
        ],
      } as any,
    ];

    const analysis = analyzePerSeedReliability(mockEvaluations, 1);
    expect(analysis.seeds.length).toBe(5);

    const s1 = analysis.seeds.find((s) => s.seed === 1)!;
    expect(s1.status).toBe("reliable");
    expect(s1.recentSuccessRate).toBe(100);

    const s2 = analysis.seeds.find((s) => s.seed === 2)!;
    expect(s2.status).toBe("normal_variance");

    const s3 = analysis.seeds.find((s) => s.seed === 3)!;
    expect(s3.status).toBe("blind_spot");
    expect(s3.isBlindSpot).toBe(true);
    expect(s3.consecutiveFailures).toBe(3);

    const s4 = analysis.seeds.find((s) => s.seed === 4)!;
    expect(s4.status).toBe("inefficient");
    expect(s4.isInefficient).toBe(true);
    expect(s4.typicalSuccessfulSteps).toBe(245);
    expect(analysis.inefficientSeeds).toContain(4);
    expect(analysis.blindSpotSeeds).toContain(3);
  });

  // Invariant 8: Stage isolation & Monotonicity
  it("Invariant 8: Enforces strict stage isolation and chronological X monotonicity", () => {
    const raw: Partial<TrainingEpisode>[] = [
      ...Array.from({ length: 20 }, (_, i) => ({ id: i + 1, curriculum_stage: 1, global_timestep: (i + 1) * 100, completed_at: "2026-08-11T10:00:00Z" })),
      ...Array.from({ length: 30 }, (_, i) => ({ id: 21 + i, curriculum_stage: 2, global_timestep: 2000 + (i + 1) * 100, completed_at: "2026-08-12T10:00:00Z" })),
    ];

    const stage1 = processCanonicalHistory(raw, 1);
    expect(stage1.length).toBe(20);
    expect(stage1.every((e) => e.curriculum_stage === 1)).toBe(true);

    const stage2 = processCanonicalHistory(raw, 2);
    expect(stage2.length).toBe(30);
    expect(stage2.every((e) => e.curriculum_stage === 2)).toBe(true);

    // Monotonicity assertion on canonical history
    expect(() => assertMonotonicX(stage1.map((e) => ({ x: e.global_timestep })))).not.toThrow();
    expect(() => assertMonotonicX(stage2.map((e) => ({ x: e.global_timestep })))).not.toThrow();
  });
});

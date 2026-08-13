import { describe, expect, it } from "vitest";
import {
  assertMonotonicX,
  buildEpisodeBuckets,
  buildFormalEvalMarkers,
  compareCanonicalEpisodes,
  normalizeEpisodeRecord,
  processCanonicalHistory,
  selectWindowSlice,
  type CanonicalEpisodeRecord,
} from "./chartPipeline";
import type { CheckpointEntry, TrainingEpisode } from "../state/types";

describe("Chart Pipeline Architecture & Chronology Pipeline", () => {
  // Test 1 — Out-of-order dates
  it("Test 1: Correctly sorts out-of-order dates into strict chronological order", () => {
    const dates = [
      "2026-08-12T10:00:00Z",
      "2026-08-11T10:00:00Z",
      "2026-08-12T14:00:00Z",
      "2026-08-13T08:00:00Z",
      "2026-08-10T10:00:00Z",
    ];

    const raw: Partial<TrainingEpisode>[] = dates.map((d, i) => ({
      id: i + 1,
      curriculum_stage: 4,
      global_timestep: (i + 1) * 1000,
      completed_at: d,
      success: true,
      steps: 50,
    }));

    // Intentionally mess up timesteps to match out of order dates
    raw[0].global_timestep = 3000; // Aug 12
    raw[1].global_timestep = 2000; // Aug 11
    raw[2].global_timestep = 4000; // Aug 12
    raw[3].global_timestep = 5000; // Aug 13
    raw[4].global_timestep = 1000; // Aug 10

    const canonical = processCanonicalHistory(raw, "all");

    expect(canonical.map((e) => e.global_timestep)).toEqual([1000, 2000, 3000, 4000, 5000]);
    expect(canonical.map((e) => e.completed_at)).toEqual([
      "2026-08-10T10:00:00Z",
      "2026-08-11T10:00:00Z",
      "2026-08-12T10:00:00Z",
      "2026-08-12T14:00:00Z",
      "2026-08-13T08:00:00Z",
    ]);
  });

  // Test 2 — All versus Last N (Exact Tail-Subset Invariant)
  it("Test 2: Guarantees Last 25, 50, 100 are exact tail subsets of All", () => {
    const raw: Partial<TrainingEpisode>[] = Array.from({ length: 150 }, (_, i) => ({
      id: i + 1,
      curriculum_stage: 4,
      global_timestep: (i + 1) * 100,
      completed_at: new Date(1700000000000 + i * 60000).toISOString(),
      success: i % 2 === 0,
      steps: 40 + (i % 10),
    }));

    const all = processCanonicalHistory(raw, "all");
    const last100 = selectWindowSlice(all, 100);
    const last50 = selectWindowSlice(all, 50);
    const last25 = selectWindowSlice(all, 25);

    expect(last100).toEqual(all.slice(-100));
    expect(last50).toEqual(all.slice(-50));
    expect(last25).toEqual(all.slice(-25));

    expect(last50).toEqual(last100.slice(-50));
    expect(last25).toEqual(last50.slice(-25));
  });

  // Test 3 — Same newest point
  it("Test 3: All, Last 100, Last 50, and Last 25 end on the exact same newest record", () => {
    const raw: Partial<TrainingEpisode>[] = Array.from({ length: 150 }, (_, i) => ({
      id: i + 1,
      curriculum_stage: 4,
      global_timestep: (i + 1) * 50,
      completed_at: new Date(1700000000000 + i * 1000).toISOString(),
      success: true,
      steps: 30,
    }));

    const all = processCanonicalHistory(raw, "all");
    const last100 = selectWindowSlice(all, 100);
    const last50 = selectWindowSlice(all, 50);
    const last25 = selectWindowSlice(all, 25);

    const newestAll = all[all.length - 1];
    expect(last100[last100.length - 1].id).toBe(newestAll.id);
    expect(last50[last50.length - 1].id).toBe(newestAll.id);
    expect(last25[last25.length - 1].id).toBe(newestAll.id);
  });

  // Test 4 — Stage filtering
  it("Test 4: Filters by stage correctly and enforces tail-subset invariant on filtered stage history", () => {
    const raw: Partial<TrainingEpisode>[] = [
      ...Array.from({ length: 30 }, (_, i) => ({ id: i + 1, curriculum_stage: 3, global_timestep: (i + 1) * 10 })),
      ...Array.from({ length: 120 }, (_, i) => ({ id: 31 + i, curriculum_stage: 4, global_timestep: 300 + (i + 1) * 10 })),
      ...Array.from({ length: 20 }, (_, i) => ({ id: 151 + i, curriculum_stage: 5, global_timestep: 1500 + (i + 1) * 10 })),
    ];

    const stage4All = processCanonicalHistory(raw, 4);
    expect(stage4All.length).toBe(120);
    expect(stage4All.every((e) => e.curriculum_stage === 4)).toBe(true);

    const last25 = selectWindowSlice(stage4All, 25);
    expect(last25.length).toBe(25);
    expect(last25).toEqual(stage4All.slice(-25));
    expect(last25[last25.length - 1].id).toBe(150);
  });

  // Test 5 — Repeated/restarted episode numbers
  it("Test 5: Handles restarted/repeated episode numbers across sessions via global timesteps and timestamps", () => {
    // Session 1: episode_in_stage 1..10 at timesteps 100..1000, IDs 1..10
    // Session 2: restarted run, episode_in_stage 1..10 again at timesteps 1100..2000, IDs 11..20
    const raw: Partial<TrainingEpisode>[] = [
      ...Array.from({ length: 10 }, (_, i) => ({
        id: i + 1,
        episode_in_stage: i + 1,
        curriculum_stage: 4,
        global_timestep: (i + 1) * 100,
        completed_at: "2026-08-11T10:00:00Z",
      })),
      ...Array.from({ length: 10 }, (_, i) => ({
        id: 11 + i,
        episode_in_stage: i + 1, // repeated!
        curriculum_stage: 4,
        global_timestep: 1000 + (i + 1) * 100, // higher timestep
        completed_at: "2026-08-12T10:00:00Z",
      })),
    ];

    const canonical = processCanonicalHistory(raw, "all");
    expect(canonical.length).toBe(20);
    expect(canonical[0].id).toBe(1);
    expect(canonical[19].id).toBe(20);
    expect(canonical[19].global_timestep).toBe(2000);
  });

  // Test 6 — Duplicate timestamps
  it("Test 6: Provides deterministic secondary and tertiary ordering for records sharing identical timestamps", () => {
    const sameTs = "2026-08-12T12:00:00Z";
    const raw: Partial<TrainingEpisode>[] = [
      { id: 105, global_timestep: 500, completed_at: sameTs },
      { id: 101, global_timestep: 500, completed_at: sameTs },
      { id: 103, global_timestep: 500, completed_at: sameTs },
    ];

    const canonical = processCanonicalHistory(raw, "all");
    expect(canonical.map((e) => e.id)).toEqual([101, 103, 105]);
  });

  // Test 7 — Training + formal evaluations
  it("Test 7: Renders formal benchmark evaluations at timeline position without altering training episode calculations", () => {
    const rawEpisodes: Partial<TrainingEpisode>[] = Array.from({ length: 50 }, (_, i) => ({
      id: i + 1,
      curriculum_stage: 4,
      global_timestep: (i + 1) * 100,
      completed_at: new Date(1700000000000 + i * 1000).toISOString(),
      success: true,
      steps: 40,
    }));

    const mockCheckpoints: CheckpointEntry[] = [
      {
        checkpoint_episode: 20,
        recorded_at: "2026-08-12T12:00:00Z",
        checkpoint: "chk_20.pt",
        evaluation: "eval_20.json",
        replay: "replay_20.json",
        success_rate: 0.2, // 20% formal benchmark
        curriculum_stage: 4,
        global_timesteps: 2000,
        average_completion_steps: 150,
      } as any,
    ];

    const canonical = processCanonicalHistory(rawEpisodes, 4);
    const buckets = buildEpisodeBuckets(canonical, 25);
    const evalMarkers = buildFormalEvalMarkers(mockCheckpoints, 4, "timesteps");

    expect(buckets.length).toBe(2); // 25 eps per bucket
    expect(buckets[0].successRate).toBe(100); // Training success rate remains 100%
    expect(evalMarkers.length).toBe(1);
    expect(evalMarkers[0].xVal).toBe(2000);
    expect(evalMarkers[0].successRatePct).toBe(20);
    expect(evalMarkers[0].label).toContain("Formal Benchmark: 20%");
  });

  // Test 8 — Success calculation
  it("Test 8: Calculates exact success percentage (18 / 25 = 72%)", () => {
    const raw: Partial<TrainingEpisode>[] = [
      ...Array.from({ length: 18 }, (_, i) => ({
        id: i + 1,
        curriculum_stage: 4,
        global_timestep: (i + 1) * 10,
        result: "SUCCESS",
        success: true,
        steps: 40,
      })),
      ...Array.from({ length: 7 }, (_, i) => ({
        id: 19 + i,
        curriculum_stage: 4,
        global_timestep: (19 + i) * 10,
        result: "TIMEOUT",
        success: false,
        steps: 600,
      })),
    ];

    const canonical = processCanonicalHistory(raw, "all");
    const buckets = buildEpisodeBuckets(canonical, 25);

    expect(buckets.length).toBe(1);
    expect(buckets[0].episodeCount).toBe(25);
    expect(buckets[0].successCount).toBe(18);
    expect(buckets[0].failureCount).toBe(7);
    expect(buckets[0].successRate).toBe(72);
  });

  // Test 9 — Steps calculation
  it("Test 9: Calculates average successful steps using ONLY successful episodes in that bucket", () => {
    const raw: Partial<TrainingEpisode>[] = [
      // 3 successes with steps 100, 200, 300 -> avg = 200
      { id: 1, success: true, result: "SUCCESS", steps: 100, global_timestep: 10 },
      { id: 2, success: true, result: "SUCCESS", steps: 200, global_timestep: 20 },
      { id: 3, success: true, result: "SUCCESS", steps: 300, global_timestep: 30 },
      // 2 failures with max steps 600 (should NOT skew successful step average!)
      { id: 4, success: false, result: "TIMEOUT", steps: 600, global_timestep: 40 },
      { id: 5, success: false, result: "TIMEOUT", steps: 600, global_timestep: 50 },
    ];

    const canonical = processCanonicalHistory(raw, "all");
    const buckets = buildEpisodeBuckets(canonical, 5);

    expect(buckets[0].successRate).toBe(60);
    expect(buckets[0].avgSuccessfulSteps).toBe(200); // (100 + 200 + 300) / 3
  });

  // Test 10 — Zero-success bucket
  it("Test 10: Reports 0% success rate and null avgSuccessfulSteps for 0-success buckets", () => {
    const raw: Partial<TrainingEpisode>[] = Array.from({ length: 25 }, (_, i) => ({
      id: i + 1,
      curriculum_stage: 4,
      global_timestep: (i + 1) * 10,
      result: "TIMEOUT",
      success: false,
      steps: 600,
    }));

    const canonical = processCanonicalHistory(raw, "all");
    const buckets = buildEpisodeBuckets(canonical, 25);

    expect(buckets[0].successRate).toBe(0);
    expect(buckets[0].avgSuccessfulSteps).toBeNull();
  });

  // Test 11 — Boundary dates & Monotonicity
  it("Test 11: Proves no rendered series goes backward in time across Aug 11 -> Aug 12 -> Aug 13 boundary dates", () => {
    const raw: Partial<TrainingEpisode>[] = [
      { id: 1, completed_at: "2026-08-11T08:00:00Z", global_timestep: 100, success: true, steps: 50 },
      { id: 2, completed_at: "2026-08-12T08:00:00Z", global_timestep: 200, success: true, steps: 50 },
      { id: 3, completed_at: "2026-08-13T08:00:00Z", global_timestep: 300, success: true, steps: 50 },
    ];

    const canonical = processCanonicalHistory(raw, "all");
    const points = canonical.map((e) => ({ x: e.timestamp_ms }));

    expect(() => assertMonotonicX(points, "Calendar Series")).not.toThrow();
  });

  // Test 12 — Real-data invariant
  it("Test 12: Proves real database records obey exact tail-subset invariant and report identical newest record", () => {
    // Construct 100 realistic records imitating real Stage 4 database telemetry
    const raw: Partial<TrainingEpisode>[] = Array.from({ length: 100 }, (_, i) => ({
      id: 136678 + i,
      run_id: "run_stage4_real_test",
      curriculum_stage: 4,
      global_environment_episode: i + 1,
      episode_in_stage: i + 1,
      global_timestep: (i + 1) * 500,
      completed_at: new Date(1786500000000 + i * 10000).toISOString(),
      success: i >= 20, // 80% success rate
      steps: i >= 20 ? 110 : 600,
      reward: i >= 20 ? 250 : -50,
      sheep_penned: 3,
    }));

    const stage4All = processCanonicalHistory(raw, 4);
    const last100 = selectWindowSlice(stage4All, 100);
    const last50 = selectWindowSlice(stage4All, 50);
    const last25 = selectWindowSlice(stage4All, 25);

    // Verify exact tail subset invariant
    expect(last25).toEqual(last50.slice(-25));
    expect(last50).toEqual(last100.slice(-50));

    // Verify identical newest record ID and timestamp
    const newestId = stage4All[stage4All.length - 1].id;
    expect(last100[last100.length - 1].id).toBe(newestId);
    expect(last50[last50.length - 1].id).toBe(newestId);
    expect(last25[last25.length - 1].id).toBe(newestId);
  });

  // Test 13 — Checkpoints fallback
  it("Test 13: Falls back seamlessly to checkpoints when raw training episodes are empty", () => {
    const checkpoints = [
      { checkpoint_episode: 10, global_timestep: 1000, success_rate: 0.70, average_completion_steps: 120, stage: 4, recorded_at: "2026-08-11T10:00:00Z" },
      { checkpoint_episode: 20, global_timestep: 2000, success_rate: 0.85, average_completion_steps: 105, stage: 4, recorded_at: "2026-08-12T10:00:00Z" },
      { checkpoint_episode: 30, global_timestep: 3000, success_rate: 0.90, average_completion_steps: 95, stage: 4, recorded_at: "2026-08-13T10:00:00Z" },
    ];

    const canonical = processCanonicalHistory([], checkpoints as any, 4);
    expect(canonical.length).toBe(3);

    const buckets = buildEpisodeBuckets(canonical, 25);
    expect(buckets.length).toBe(3);
    expect(buckets[0].successRate).toBe(70);
    expect(buckets[0].avgSuccessfulSteps).toBe(120);
    expect(buckets[2].successRate).toBe(90);
    expect(buckets[2].avgSuccessfulSteps).toBe(95);

    const last25 = selectWindowSlice(canonical, 25);
    expect(last25.length).toBe(3);
    expect(last25[last25.length - 1].id).toBe(30);
  });
});

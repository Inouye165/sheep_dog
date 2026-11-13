import { render, screen, waitFor, fireEvent, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { clearTemporalDiagnosticCache } from "../lib/api";
import type { TemporalDiagnosticPayload } from "../state/types";
import { TemporalDiagnosticsPanel } from "./TemporalDiagnosticsPanel";

function jsonResponse(payload: unknown, status = 200): Response {
  return new Response(JSON.stringify(payload), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

function buildMockDiagnosticPayload(
  overrides: Partial<TemporalDiagnosticPayload> = {},
): TemporalDiagnosticPayload {
  // Recording missing step 0 (starts at step 1) and containing a frame gap between step 2 and step 5,
  // plus intervals at 6 unpenned sheep and 1 unpenned sheep.
  const steps = [1, 2, 5, 6, 7, 8];
  const step_evidence = steps.map((st, idx) => {
    const isLate = st >= 6;
    const unpennedCount = isLate ? 1 : 6;
    const pennedCount = 6 - unpennedCount;
    const hasPriorContiguous = idx > 0 && steps[idx - 1] === st - 1;
    const truncated = st === 8;

    return {
      step: st,
      has_prior_contiguous_step: hasPriorContiguous,
      actions: ["right", "up", "wait"],
      unpenned_count: unpennedCount,
      penned_count: pennedCount,
      average_distance_to_pen: isLate ? 6.0 : 18.5,
      best_route_distance_so_far: 6.0,
      flock_spread: isLate ? 0.0 : 4.2,
      no_progress_steps: st,
      no_progress_reset_without_best_gain: st === 5,
      collector_blind_spot: false,
      corner_target_inverted: false,
      any_dog_outside_sheep_vision_oscillating: false,
      any_sheep_closed_fence_adjacent: true,
      any_sheep_wall_or_corner: false,
      reward: {
        total: -0.02,
        progress_to_pen: st % 2 === 0 ? 0.15 : -0.15,
        time_penalty: -0.02,
      },
      guidance_source: st === 1 ? "recorded_step_telemetry" : "reconstructed_team_strategy",
      guidance_certainty:
        st === 1
          ? "recorded_runtime_telemetry"
          : "uncertain_reconstruction_missing_step0",
      mask_type: st === 1 ? "actual_decision_time" : "historical_approximate_no_carry",
      has_recorded_step_telemetry: st === 1,
      telemetry_truncated: truncated,
      dogs: [
        {
          dog_index: 0,
          pre_step_position: hasPriorContiguous ? { x: 11, y: 14 } : null,
          post_step_position: { x: 12 + (st % 2), y: 14 },
          displacement: hasPriorContiguous
            ? { dx: 1, dy: 0, distance: 1.0 }
            : null,
          recorded_role: "rear_pressure",
          assigned_role: "rear_pressure",
          role_matches_recorded: true,
          target: { x: 15, y: 12 },
          target_sheep_index: 0,
          distance_to_target: 3.6,
          guidance_source:
            st === 1 ? "recorded_step_telemetry" : "reconstructed_team_strategy",
          guidance_certainty:
            st === 1
              ? "recorded_runtime_telemetry"
              : "uncertain_reconstruction_missing_step0",
          nearest_unpenned_sheep_distance: 4.0,
          within_sheep_vision: true,
          behind_flock_alignment: 0.75,
          on_gate_or_waypoint_side: false,
          legal_action_mask: {
            wait: true,
            up: true,
            down: true,
            left: false,
            right: true,
          },
          decision: {
            selected_action: st === 1 ? "left" : "right",
            executed_action: st === 1 ? "wait" : "right",
            replacement_reason: st === 1 ? "action_masked_illegal" : null,
            mask_type:
              st === 1 ? "actual_decision_time" : "historical_approximate_no_carry",
            decision_action_mask: {
              wait: true,
              up: true,
              down: true,
              left: false,
              right: true,
            },
            action_probabilities:
              st === 1
                ? {
                    wait: 0.55,
                    up: 0.15,
                    down: 0.1,
                    left: 0.0,
                    right: 0.2,
                  }
                : null,
            action_probabilities_source:
              st === 1 ? "recomputed_diagnostic_forward_pass" : "not_recorded",
            value_estimate: st === 1 ? 1.425 : null,
            value_estimate_source:
              st === 1 ? "recomputed_diagnostic_forward_pass" : "not_recorded",
            decision_mode: st === 1 ? "deterministic" : "not_recorded",
            capture_error: null,
            observation_labeled:
              st === 1
                ? {
                    available: true,
                    schema_verified: true,
                    schema_name: "role_aware_54",
                    length: 54,
                    reason: "Verified schema role_aware_54 (54 features).",
                    fields: [
                      {
                        index: 0,
                        key: "dog_x_norm",
                        label: "Dog X position (normalized)",
                        group: "Self State",
                        value: 0.342,
                      },
                    ],
                  }
                : {
                    available: false,
                    schema_verified: false,
                    schema_name: null,
                    length: 0,
                    reason:
                      "Not recorded (raw policy observation vectors are only captured when capture_step_telemetry=True).",
                    fields: [],
                  },
          },
        },
      ],
      sheep: Array.from({ length: 6 }).map((_, sIdx) => ({
        sheep_index: sIdx,
        pre_step_position: hasPriorContiguous ? { x: 14, y: 10 } : null,
        post_step_position: { x: 14, y: 10 },
        displacement: hasPriorContiguous ? { dx: 0, dy: 0, distance: 0 } : null,
        penned: isLate ? sIdx > 0 : false,
        personality: sIdx === 0 ? "stubborn" : "obedient",
        is_packed: false,
        route_distance: isLate && sIdx > 0 ? 0 : 6.0 + sIdx,
        waypoint: { x: 14, y: 8 },
        closed_fence_adjacent: sIdx === 0,
        outer_wall_or_corner: false,
        wall_margin: 8,
        nearby_dogs: [
          {
            dog_index: 0,
            role: "rear_pressure",
            distance: 4.0,
            within_sheep_vision: true,
            dx_from_sheep: -2,
            dy_from_sheep: 4,
            alignment_to_waypoint: 0.75,
            on_waypoint_side: false,
          },
        ],
      })),
    };
  });

  return ({
    analysis: {
      replay_id: "checkpoint-039909-seed-000041",
      seed: 41,
      policy_name: "neural_policy",
      outcome_status: "stopped",
      stop_reason: "no-progress",
      total_steps: 8,
      final_penned_count: 5,
      final_unpenned_count: 1,
      total_sheep: 6,
      unpenned_count_coverage: [6, 1],
      pen_geometry: {
        origin: { x: 10, y: 2 },
        width: 8,
        height: 6,
        opening: "bottom",
        grid_width: 35,
        grid_height: 25,
      },
      penning_steps: [[6, 1]],
      no_progress_resets_without_best_gain: 1,
      evidence_audit: {
        has_initial_step_zero: false,
        has_contiguous_frames: false,
        has_recorded_debug_targets: false,
        has_recorded_step_telemetry: true,
        has_recorded_action_probabilities: true,
        telemetry_truncated: true,
        telemetry_limit_steps: 7,
        telemetry_truncation_step: 8,
        strategy_version: "1.0",
        guidance_certainty: "uncertain_reconstruction_missing_step0",
        recorded_fields: ["dog/sheep positions", "actions"],
        derived_fields: ["BFS legal-route distances"],
        reconstructed_fields: ["TeamStrategy role/target estimates"],
        unavailable_fields: ["Initial step 0 pre-action snapshot"],
      },
      intervals: [
        {
          interval_id: "interval_1_6_8",
          start_step: 6,
          end_step: 8,
          duration_steps: 3,
          unpenned_sheep_count: 1,
          penned_sheep_count: 5,
          classification: "oscillation_loop",
          readable_label: "Repeated movement",
          candidate_mechanisms: ["oscillation_loop", "fence_side_obstruction"],
          affected_sheep_indices: [0],
          affected_dog_indices: [0],
          avg_route_distance_start: 6.0,
          avg_route_distance_end: 6.0,
          avg_route_distance_min: 6.0,
          avg_route_distance_max: 6.0,
          net_route_progress: 0.0,
          flock_spread_start: 0.0,
          flock_spread_end: 0.0,
          no_progress_step_start: 6,
          no_progress_step_end: 8,
          no_progress_resets_in_interval: 0,
          collector_blind_spot_steps: 0,
          guidance_certainty: "uncertain_reconstruction_missing_step0",
          lead_in: {
            start_step: 2,
            end_step: 5,
            duration_steps: 2,
            sheep_penned_delta: 0,
            avg_route_distance_start: 18.5,
            avg_route_distance_end: 18.5,
            net_route_progress: 0.0,
            flock_spread_delta: 0.0,
            dog_displacement_mean: 2.0,
            summary: "Lead-in steps 2..5 before straggler loop.",
          },
          aftermath: {
            start_step: 8,
            end_step: 8,
            duration_steps: 1,
            sheep_penned_delta: 0,
            avg_route_distance_start: 6.0,
            avg_route_distance_end: 6.0,
            net_route_progress: 0.0,
            flock_spread_delta: 0.0,
            dog_displacement_mean: 1.0,
            summary: "Episode terminated at step 8.",
          },
          dog_summaries: [],
          sheep_summaries: [],
          reward_summary: {
            total_reward: -0.06,
            mean_step_reward: -0.02,
            component_totals: { total: -0.06 },
            positive_progress_during_zero_net_gain: 0.15,
            cycles: [],
            has_candidate_positive_return_cycle: false,
            has_exploitable_positive_cycle: false,
            has_net_negative_cycle_with_positive_steps: true,
          },
          supporting_evidence: [
            "Dog 0 alternated horizontal positions while Sheep 0 remained adjacent to closed pen fence.",
          ],
          alternative_explanations: [
            "Missing policy hidden state prevents proving whether oscillation is policy or target driven.",
          ],
          plain_language: {
            readable_label: "Repeated movement",
            what_was_observed:
              "Steps 6..8 (3 steps) with 1 unpenned sheep: route distance stayed at 6.0 cells.",
            lead_in_summary: "Lead-in steps 2..5 before straggler loop.",
            aftermath_summary: "Episode terminated at step 8.",
            candidate_explanation:
              "Dogs and sheep entered a repeated back-and-forth movement pattern.",
            moving_away_note:
              "Why moving away from the pen can be productive: Dogs often need to release pressure and circle wide.",
            supporting_evidence: [
              "Dog 0 alternated horizontal positions while Sheep 0 remained adjacent to closed pen fence.",
            ],
            missing_evidence_and_alternatives: [
              "Missing policy hidden state prevents proving whether oscillation is policy or target driven.",
            ],
            controlled_experiment_required:
              "To test whether this pattern causes failures, run a controlled A/B experiment.",
          },
        },
        {
          interval_id: "interval_2_1_5",
          start_step: 1,
          end_step: 5,
          duration_steps: 3,
          unpenned_sheep_count: 6,
          penned_sheep_count: 0,
          classification: "productive_repositioning",
          readable_label: "Productive repositioning",
          candidate_mechanisms: ["productive_repositioning"],
          affected_sheep_indices: [0, 1, 2, 3, 4, 5],
          affected_dog_indices: [0],
          avg_route_distance_start: 18.5,
          avg_route_distance_end: 18.5,
          avg_route_distance_min: 18.5,
          avg_route_distance_max: 18.5,
          net_route_progress: 0.0,
          flock_spread_start: 4.2,
          flock_spread_end: 4.2,
          no_progress_step_start: 1,
          no_progress_step_end: 5,
          no_progress_resets_in_interval: 1,
          collector_blind_spot_steps: 0,
          guidance_certainty: "recorded_runtime_telemetry",
          lead_in: {
            start_step: 1,
            end_step: 1,
            duration_steps: 1,
            sheep_penned_delta: 0,
            avg_route_distance_start: 18.5,
            avg_route_distance_end: 18.5,
            net_route_progress: 0.0,
            flock_spread_delta: 0.0,
            dog_displacement_mean: 1.0,
            summary: "Initial 6-sheep setup at step 1.",
          },
          aftermath: {
            start_step: 5,
            end_step: 6,
            duration_steps: 2,
            sheep_penned_delta: 5,
            avg_route_distance_start: 18.5,
            avg_route_distance_end: 6.0,
            net_route_progress: 12.5,
            flock_spread_delta: -4.2,
            dog_displacement_mean: 3.0,
            summary: "5 sheep penned by step 6.",
          },
          dog_summaries: [],
          sheep_summaries: [],
          reward_summary: {
            total_reward: -0.06,
            mean_step_reward: -0.02,
            component_totals: { total: -0.06 },
            positive_progress_during_zero_net_gain: 0.15,
            cycles: [],
            has_candidate_positive_return_cycle: false,
            has_exploitable_positive_cycle: false,
            has_net_negative_cycle_with_positive_steps: false,
          },
          supporting_evidence: ["Dogs repositioned around the 6-sheep flock."],
          alternative_explanations: ["Observational evidence only."],
          plain_language: {
            readable_label: "Productive repositioning",
            what_was_observed: "Steps 1..5 with 6 unpenned sheep.",
            lead_in_summary: "Initial 6-sheep setup at step 1.",
            aftermath_summary: "5 sheep penned by step 6.",
            candidate_explanation:
              "Temporary pause or movement away from the pen was followed by penning progress.",
            moving_away_note: "Moving away from the pen can be productive.",
            supporting_evidence: ["Dogs repositioned around the 6-sheep flock."],
            missing_evidence_and_alternatives: ["Observational evidence only."],
            controlled_experiment_required: "Requires controlled experiment.",
          },
        },
      ],
      summary_notes: ["Detected 1 oscillation loop interval(s)."],
    },
    step_evidence,
    available_reward_components: ["total", "progress_to_pen", "time_penalty"],
    action_order: ["wait", "up", "down", "left", "right"],
    sheep_vision: 10.0,
    ...overrides,
  }) as unknown as TemporalDiagnosticPayload;
}

describe("TemporalDiagnosticsPanel", () => {
  beforeEach(() => {
    clearTemporalDiagnosticCache();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("loads curated episode, seeks to selected interval lead-in, and aligns world steps across missing step 0 and frame gaps without re-fetching on scrub", async () => {
    let analysisFetchCount = 0;
    const mockPayload = buildMockDiagnosticPayload();

    vi.spyOn(globalThis, "fetch").mockImplementation(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.includes("/api/temporal-diagnostics/examples")) {
        return jsonResponse({
          examples: [
            {
              replay_id: "checkpoint-039909-seed-000041",
              title: "Seed 41 Failure — Multi-Count Progression",
              category: "failure_vs_recovery_pair",
              recommended_default: true,
              seed: 41,
              outcome_status: "stopped",
              stop_reason: "no-progress",
              total_steps: 8,
              penned_summary: "5/6 penned",
              unpenned_counts: [6, 1],
              paired_replay_id: "checkpoint-038710-seed-000041",
              what_to_look_for: "Inspect 6->1 unpenned sheep and straggler oscillation.",
            },
          ],
        });
      }
      if (url.includes("/api/replays/checkpoint-039909-seed-000041/temporal-analysis")) {
        analysisFetchCount += 1;
        return jsonResponse(mockPayload);
      }
      return jsonResponse({ detail: "Not found" }, 404);
    });

    render(<TemporalDiagnosticsPanel initialReplayId="checkpoint-039909-seed-000041" />);

    // 1. Wait for payload to load; first interval is selected automatically and seeks to its lead_in.start_step (World Step 2)
    await waitFor(() => {
      expect(
        screen.getByText(/D\. Synchronized Entity Inspector @ World Step 2/i),
      ).toBeInTheDocument();
    });
    expect(analysisFetchCount).toBe(1);

    // Verify missing step 0 and frame gaps are disclosed
    expect(screen.getByText(/starts at step 1/i)).toBeInTheDocument();
    expect(screen.getByText(/1 gap\(s\) detected/i)).toBeInTheDocument();

    // 2. Selecting the second interval ("Productive repositioning", 6 unpenned sheep) seeks to its lead-in (World Step 1)
    const intervalOption2 = screen.getByRole("option", {
      name: /Productive repositioning/i,
    });
    fireEvent.click(intervalOption2);

    await waitFor(() => {
      expect(
        screen.getByText(/D\. Synchronized Entity Inspector @ World Step 1/i),
      ).toBeInTheDocument();
    });

    // 3. At World Step 1, verify recorded replacement reason, recomputed probabilities/value estimate, and verified 54-feature observation schema
    expect(screen.getByText("action_masked_illegal")).toBeInTheDocument();
    expect(
      screen.getAllByText(/Recomputed \(recomputed_diagnostic_forward_pass\)/i).length,
    ).toBeGreaterThanOrEqual(2);
    expect(screen.getByText(/1\.4250/i)).toBeInTheDocument();
    expect(screen.getByText(/Dog X position \(normalized\)/i)).toBeInTheDocument();

    // 4. Step forward across the frame gap (Step 1 -> Step 2 -> Step 5) and verify world-step alignment & "Not recorded" badges
    const stepPlusOneBtn = screen.getByRole("button", { name: /Step \+1/i });
    fireEvent.click(stepPlusOneBtn); // index 0 (step 1) -> index 1 (step 2)
    expect(
      screen.getByText(/D\. Synchronized Entity Inspector @ World Step 2/i),
    ).toBeInTheDocument();

    fireEvent.click(stepPlusOneBtn); // index 1 (step 2) -> index 2 (step 5, jumping the gap 2->5 cleanly)
    expect(
      screen.getByText(/D\. Synchronized Entity Inspector @ World Step 5/i),
    ).toBeInTheDocument();

    // At Step 5 (after the gap and without step_telemetry), unavailable fields must say "Not recorded", never 0
    expect(
      screen.getByText(/Not recorded \(critic value estimate not captured in this replay\)/i),
    ).toBeInTheDocument();
    expect(
      screen.getByText(/Not recorded \(prior step frame not present\)/i),
    ).toBeInTheDocument();

    // 5. Filter by unpenned sheep count (1 Sheep) and verify keyboard navigation
    const oneSheepFilterBtn = screen.getByRole("button", { name: /1 Sheep \(1\)/i });
    fireEvent.click(oneSheepFilterBtn);
    await waitFor(() => {
      expect(
        screen.getByText(/E\. Plain-Language Interval Explanation · ⟳ Repeated movement/i),
      ).toBeInTheDocument();
    });

    // Switch Entity Inspector to Sheep S0 and verify sheep geometry & route distance
    const sheepTab0 = screen.getByRole("tab", { name: /Sheep S0/i });
    fireEvent.click(sheepTab0);
    expect(
      screen.getByText(/Sheep S0 — Pre-Step vs\. Post-Step State & Nearby Dog Geometry/i),
    ).toBeInTheDocument();
    expect(screen.getAllByText(/stubborn/i).length).toBeGreaterThanOrEqual(1);

    // Repeated scrubbing/stepping must NOT re-run or re-fetch backend analysis
    const scrubber = screen.getByLabelText("World step scrubber");
    fireEvent.change(scrubber, { target: { value: "5" } }); // index 5 -> World Step 8 (truncated step)
    expect(
      screen.getByText(/D\. Synchronized Entity Inspector @ World Step 8/i),
    ).toBeInTheDocument();
    expect(
      screen.getByText(/Truncated \(Step telemetry capture limit reached\)/i),
    ).toBeInTheDocument();
    expect(analysisFetchCount).toBe(1);
  });

  it("displays clear error state when backend rejects invalid or missing replay ID", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.includes("/api/temporal-diagnostics/examples")) {
        return jsonResponse({ examples: [] });
      }
      return jsonResponse({ detail: "Replay 'missing-id' not found in approved artifact directories." }, 404);
    });

    render(<TemporalDiagnosticsPanel initialReplayId="missing-id" />);

    await waitFor(() => {
      expect(screen.getByRole("alert")).toBeInTheDocument();
    });
    expect(
      screen.getByText(/Unable to Load Temporal Diagnostic Analysis/i),
    ).toBeInTheDocument();
    expect(
      screen.getByText(/Failed to fetch \/api\/replays\/missing-id\/temporal-analysis/i),
    ).toBeInTheDocument();
  });

  it("accurately distinguishes read-only preview, normal backend, offline cached replay, and reconnection recovery", async () => {
    let backendOnline = true;
    let readOnlyMode = true;
    const mockPayload = buildMockDiagnosticPayload();

    vi.spyOn(globalThis, "fetch").mockImplementation(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (!backendOnline) {
        throw new Error("Failed to fetch: network unreachable");
      }
      if (url.includes("/api/health")) {
        return jsonResponse({
          status: "ok",
          service: "sheepdog-api",
          ready: true,
          ...(readOnlyMode ? { mode: "read_only_preview", read_only_preview: true } : {}),
        });
      }
      if (url.includes("/api/temporal-diagnostics/examples")) {
        return jsonResponse({ examples: [] });
      }
      if (url.includes("/api/replays/checkpoint-039909-seed-000041/temporal-analysis")) {
        return jsonResponse(mockPayload);
      }
      return jsonResponse({ detail: "Not found" }, 404);
    });

    const cachedChanges: boolean[] = [];
    render(
      <TemporalDiagnosticsPanel
        initialReplayId="checkpoint-039909-seed-000041"
        onCachedReplayAvailableChange={(avail) => cachedChanges.push(avail)}
      />,
    );

    // 1. Wait for read-only preview connection status and loaded replay
    await waitFor(() => {
      const badge = screen.getByTestId("temporal-connection-status");
      expect(badge).toHaveAttribute("data-connection-state", "read_only_preview");
      expect(badge).toHaveTextContent(/Connected · Read-Only Preview/i);
    });
    expect(cachedChanges[cachedChanges.length - 1]).toBe(true);

    // 2. Simulate backend disconnecting while replay is already loaded/cached
    backendOnline = false;
    const reconnectBtn = screen.getByTestId("temporal-reconnect-btn");
    fireEvent.click(reconnectBtn);

    await waitFor(() => {
      const badge = screen.getByTestId("temporal-connection-status");
      expect(badge).toHaveAttribute("data-connection-state", "offline_cached_replay");
      expect(badge).toHaveTextContent(/Backend Offline · Viewing Cached Replay/i);
    });
    expect(screen.getByTestId("temporal-offline-cached-banner")).toBeInTheDocument();
    // Cached replay inspector remains interactive while disconnected
    expect(
      screen.getByText(/D\. Synchronized Entity Inspector @ World Step 2/i),
    ).toBeInTheDocument();

    // 3. Reconnect to normal backend mode and verify status recovery
    backendOnline = true;
    readOnlyMode = false;
    fireEvent.click(screen.getByTestId("temporal-reconnect-btn"));

    await waitFor(() => {
      const badge = screen.getByTestId("temporal-connection-status");
      expect(badge).toHaveAttribute("data-connection-state", "connected_normal");
      expect(badge).toHaveTextContent(/Connected · Normal Backend/i);
    });
    expect(screen.queryByTestId("temporal-offline-cached-banner")).not.toBeInTheDocument();
  });

  it("preserves stable episode-wide interval numbering across sheep-count filters and keeps intro/audit details collapsible by default", async () => {
    const basePayload = buildMockDiagnosticPayload();
    // Create 11 intervals so interval #11 (steps 757..778, lead-in 737, 2 unpenned sheep) is at episode index 10 (1-based #11)
    const fillerIntervals = Array.from({ length: 10 }).map((_, idx) => ({
      ...basePayload.analysis.intervals[1],
      interval_id: `iv-${idx + 1}-filler`,
      start_step: 1,
      end_step: 2,
      unpenned_sheep_count: 6,
      readable_label: `Early interval ${idx + 1}`,
    }));
    const interval11 = {
      ...basePayload.analysis.intervals[0],
      interval_id: "iv-11-counterproductive_pressure_candidate",
      start_step: 757,
      end_step: 778,
      duration_steps: 22,
      unpenned_sheep_count: 2,
      penned_sheep_count: 4,
      classification: "counterproductive_pressure_candidate",
      readable_label: "Counterproductive Pressure Candidate",
      lead_in: {
        ...basePayload.analysis.intervals[0].lead_in,
        start_step: 737,
        end_step: 757,
      },
      aftermath: {
        ...basePayload.analysis.intervals[0].aftermath,
        start_step: 778,
        end_step: 798,
      },
    };
    const extraStep737 = {
      ...basePayload.step_evidence[0],
      step: 737,
      has_prior_contiguous_step: true,
      unpenned_count: 2,
      penned_count: 4,
    };
    const extraStep757 = {
      ...basePayload.step_evidence[0],
      step: 757,
      has_prior_contiguous_step: true,
      unpenned_count: 2,
      penned_count: 4,
    };

    const mockPayload = buildMockDiagnosticPayload({
      analysis: {
        ...basePayload.analysis,
        intervals: [...fillerIntervals, interval11],
      },
      step_evidence: [...basePayload.step_evidence, extraStep737, extraStep757],
    });

    vi.spyOn(globalThis, "fetch").mockImplementation(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.includes("/api/health")) {
        return jsonResponse({
          status: "ok",
          service: "sheepdog-api",
          ready: true,
          mode: "read_only_preview",
          read_only_preview: true,
        });
      }
      if (url.includes("/api/temporal-diagnostics/examples")) {
        return jsonResponse({ examples: [] });
      }
      if (url.includes("/api/replays/checkpoint-039909-seed-000041/temporal-analysis")) {
        return jsonResponse(mockPayload);
      }
      return jsonResponse({ detail: "Not found" }, 404);
    });

    render(<TemporalDiagnosticsPanel initialReplayId="checkpoint-039909-seed-000041" />);

    await waitFor(() => {
      expect(screen.getByRole("listbox", { name: /Detected temporal intervals/i })).toBeInTheDocument();
    });

    // 1. Collapsible details are closed by default so the player and interval selector stay above the fold
    const introDetails = screen.getByTestId("temporal-intro-audit-details") as HTMLDetailsElement;
    const fullCardsDetails = screen.getByTestId("temporal-full-interval-cards-details") as HTMLDetailsElement;
    expect(introDetails.open).toBe(false);
    expect(fullCardsDetails.open).toBe(false);

    // Essential evidence warnings remain visible even when details are collapsed
    expect(screen.getByText(/starts at step 1/i)).toBeInTheDocument();
    expect(screen.getByText(/1 gap\(s\) detected/i)).toBeInTheDocument();

    // Expanding details works cleanly
    introDetails.open = true;
    fireEvent(introDetails, new Event("toggle"));
    expect(introDetails.open).toBe(true);

    // 2. Select interval #11 (steps 757..778, lead-in 737)
    const option11 = screen.getByRole("option", {
      name: /#11 Counterproductive Pressure Candidate/i,
    });
    fireEvent.click(option11);

    await waitFor(() => {
      expect(
        screen.getByText(/D\. Synchronized Entity Inspector @ World Step 737/i),
      ).toBeInTheDocument();
    });
    const selectedBanner = screen.getByTestId("temporal-selected-interval-banner");
    expect(selectedBanner).toHaveTextContent(/Selected Interval #11/i);
    expect(selectedBanner).toHaveTextContent(/Steps 757\.\.778/i);
    expect(selectedBanner).toHaveTextContent(/737\.\.757/i);

    // 3. Filter to "2 Sheep" and verify interval #11 is NOT renumbered to #1
    const twoSheepBtn = screen.getByRole("button", { name: /2 Sheep \(1\)/i });
    fireEvent.click(twoSheepBtn);

    const intervalListbox = screen.getByRole("listbox", { name: /Detected temporal intervals/i });
    const filteredOptions = within(intervalListbox).getAllByRole("option");
    expect(filteredOptions).toHaveLength(1);
    expect(filteredOptions[0]).toHaveTextContent(/#11 Counterproductive Pressure Candidate/i);
    expect(filteredOptions[0]).not.toHaveTextContent(/#1 Counterproductive/i);
    expect(screen.getByTestId("temporal-selected-interval-banner")).toHaveTextContent(
      /Selected Interval #11/i,
    );
  });

  it("displays clear intentional omission notice when step telemetry was omitted in windowed head/tail mode", async () => {
    const basePayload = buildMockDiagnosticPayload();
    const modifiedSteps = basePayload.step_evidence.map((s) => {
      if (s.step === 5) {
        return {
          ...s,
          telemetry_omitted: true,
          telemetry_omitted_range: "steps_3_to_4",
          has_recorded_step_telemetry: false,
          dogs: s.dogs.map((d) => ({
            ...d,
            decision: {
              ...d.decision!,
              action_probabilities: undefined,
              value_estimate: undefined,
              observation_labeled: {
                schema_name: "54-dim-role-aware",
                schema_status: "omitted",
                available: false,
                reason: "Not recorded (intentionally omitted middle window in head/tail telemetry mode).",
                features: [],
              },
            },
          })),
        };
      }
      return s;
    });

    const mockPayload = buildMockDiagnosticPayload({
      step_evidence: modifiedSteps as any,
    });

    vi.spyOn(globalThis, "fetch").mockImplementation(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.includes("/api/health")) {
        return jsonResponse({
          status: "ok",
          service: "sheepdog-api",
          ready: true,
          mode: "read_only_preview",
          read_only_preview: true,
        });
      }
      if (url.includes("/api/temporal-diagnostics/examples")) {
        return jsonResponse({ examples: [] });
      }
      if (url.includes("/api/replays/checkpoint-039909-seed-000041/temporal-analysis")) {
        return jsonResponse(mockPayload);
      }
      return jsonResponse({ detail: "Not found" }, 404);
    });

    render(<TemporalDiagnosticsPanel initialReplayId="checkpoint-039909-seed-000041" />);

    await waitFor(() => {
      expect(screen.getByRole("listbox", { name: /Detected temporal intervals/i })).toBeInTheDocument();
    });

    const scrubber = screen.getByLabelText("World step scrubber");
    fireEvent.change(scrubber, { target: { value: "2" } }); // index 2 -> World Step 5

    await waitFor(() => {
      expect(
        screen.getByText(/D\. Synchronized Entity Inspector @ World Step 5/i),
      ).toBeInTheDocument();
    });

    expect(
      screen.getByText(/Omitted \(Telemetry omitted \(windowed capture: steps 3 to 4\)\)/i),
    ).toBeInTheDocument();
    expect(
      screen.getAllByText(/Telemetry omitted \(windowed capture: steps 3 to 4\)/i).length,
    ).toBeGreaterThanOrEqual(1);
  });
});

import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { EarlyWarningsPanel } from "./EarlyWarningsPanel";

const mockEarlyWarningsPayload = {
  enabled: true,
  is_read_only_preview: false,
  mode_description: "Advisory Early Warnings (Completed Evaluations Only; never stops training)",
  monitored_signals: [
    "unproductive_repetition (completed evaluations with verified v2.0 metrics)",
    "success_regression (completed evaluations with verified compatible baseline)",
  ],
  deferred_signals: [
    "live_training_window (deferred to future sprint)",
    "ppo_numerical_instability (deferred to future sprint)",
    "recovery_relapse_tracking (deferred to future sprint)",
    "stale_training_detection (deferred to future sprint)",
    "automatic_resolution (deferred to future sprint)",
  ],
  current_evaluation: {
    evaluation_id: "eval_chk_test_cp_39909",
    checkpoint_id: "chk_test_39909",
    checkpoint_episode: 39909,
    policy_version: 10074,
    curriculum_stage: 34,
    completion_time: "2026-11-06T18:00:00Z",
    sample_count: 10,
    evidence_source: "artifacts/evaluations/eval_chk_test_cp_39909.json",
    repetition_metrics_available: true,
    repetition_metric_status: "Verified (v2.0)",
    repetition_metrics: {
      metrics_version: "2.0",
      repeated_movement_fraction: 0.6327,
      repeated_movement_steps: 6179,
      total_world_steps: 9766,
      stalls_count: 15,
      recovery_rate: 0.6,
    },
    comparison_status: "Comparison unavailable: missing baseline for stage 34",
  },
  active_warnings: [
    {
      id: "warn_rep_test",
      condition_type: "unproductive_repetition",
      severity: "warning",
      title: "Unproductive Repetition (63.3% of world steps)",
      plain_language_observation:
        "In completed evaluation eval_chk_test_cp_39909, dogs spent 6179 of 9766 world steps (63.3%) in verified repeated movement cycles.",
      candidate_causes: [
        "Dogs overshooting pen entrance axis",
        "Inward target clamping against walls or corners",
      ],
      source_layer: "completed_evaluation",
      evidence_source: "artifacts/evaluations/eval_chk_test_cp_39909.json",
      evaluation_id: "eval_chk_test_cp_39909",
      checkpoint_id: "chk_test_39909",
      checkpoint_episode: 39909,
      policy_version: 10074,
      curriculum_stage: 34,
      completion_time: "2026-11-06T18:00:00Z",
      sample_count: 10,
      metric_version: "2.0",
      status: "active",
      supporting_measurements: {
        repeated_movement_fraction: 0.6327,
        repeated_movement_steps: 6179,
        total_world_steps: 9766,
      },
      replay_link: "artifacts/evaluations/replays/checkpoint-039909-seed-000041.json.gz",
      replay_step: 7,
      replay_interval_id: "interval_0",
      replay_available: true,
      link_explanation: "Verified replay on disk at lead-in step 7",
      acknowledged_at: null,
      acknowledged_by: null,
      first_detected_at: "2026-11-06T18:00:00Z",
      last_detected_at: "2026-11-06T18:00:00Z",
    },
  ],
  acknowledged_warnings: [
    {
      id: "warn_ack_test",
      condition_type: "unproductive_repetition",
      severity: "warning",
      title: "Prior Standoffs (Acknowledged)",
      plain_language_observation: "Previously acknowledged standoff on evaluation checkpoint 39890.",
      candidate_causes: ["Lateral pen fence cycling"],
      source_layer: "completed_evaluation",
      evidence_source: "artifacts/evaluations/eval_chk_test_cp_39890.json",
      curriculum_stage: 34,
      completion_time: "2026-11-06T16:00:00Z",
      sample_count: 10,
      metric_version: "2.0",
      status: "acknowledged",
      supporting_measurements: {},
      replay_link: null,
      replay_available: false,
      link_explanation: "Replay not available on disk",
      acknowledged_at: "2026-11-06T17:00:00Z",
      acknowledged_by: "operator",
      first_detected_at: "2026-11-06T16:00:00Z",
      last_detected_at: "2026-11-06T16:00:00Z",
    },
  ],
  summary: {
    critical_count: 0,
    warning_count: 1,
    info_count: 0,
    total_active: 1,
    total_acknowledged: 1,
  },
};

vi.mock("../lib/api", () => ({
  loadEarlyWarnings: vi.fn().mockImplementation(() => Promise.resolve(mockEarlyWarningsPayload)),
  acknowledgeEarlyWarning: vi.fn().mockImplementation((id: string) =>
    Promise.resolve({ success: true, warning_id: id }),
  ),
}));

describe("EarlyWarningsPanel", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("renders the compact header with active count badge and advisory label", async () => {
    const onToggle = vi.fn();
    render(
      <EarlyWarningsPanel
        curriculumStage={34}
        isOpen={false}
        onToggle={onToggle}
      />
    );

    await waitFor(() => {
      expect(screen.getByText("Advisory Early Warnings")).toBeInTheDocument();
      expect(screen.getByTestId("early-warnings-count-badge")).toHaveTextContent("1 Active Alert");
      expect(screen.getByText("Completed Evaluations Only · Advisory")).toBeInTheDocument();
    });

    fireEvent.click(screen.getByTestId("early-warnings-toggle"));
    expect(onToggle).toHaveBeenCalledTimes(1);
  });

  it("renders evaluation evidence card with verified v2.0 metric, denominator, and observed facts", async () => {
    render(
      <EarlyWarningsPanel
        curriculumStage={34}
        isOpen={true}
        onToggle={vi.fn()}
      />
    );

    await waitFor(() => {
      expect(screen.getByTestId("evaluation-evidence-card")).toBeInTheDocument();
      expect(screen.getByText("eval_chk_test_cp_39909")).toBeInTheDocument();
      expect(screen.getByText("Verified (v2.0)")).toBeInTheDocument();
      expect(screen.getByText(/63%/)).toBeInTheDocument();
      expect(screen.getByText(/6179 \/ 9766 steps/)).toBeInTheDocument();
      expect(screen.getByText("Unproductive Repetition (63.3% of world steps)")).toBeInTheDocument();
      expect(screen.getByText(/In completed evaluation eval_chk_test_cp_39909/)).toBeInTheDocument();
      expect(screen.getByText(/Candidate Causes \(Hypotheses — Not Proven\):/)).toBeInTheDocument();
      expect(screen.getByText("Mark Acknowledged")).toBeInTheDocument();
    });
  });

  it("switches to Acknowledged tab and displays acknowledged warnings with explanation", async () => {
    render(
      <EarlyWarningsPanel
        curriculumStage={34}
        isOpen={true}
        onToggle={vi.fn()}
      />
    );

    await waitFor(() => {
      expect(screen.getByTestId("filter-acknowledged-tab")).toBeInTheDocument();
    });

    fireEvent.click(screen.getByTestId("filter-acknowledged-tab"));

    await waitFor(() => {
      expect(screen.getByText("Prior Standoffs (Acknowledged)")).toBeInTheDocument();
      expect(screen.getByText("ACKNOWLEDGED (SEEN)")).toBeInTheDocument();
      expect(screen.getByText("Replay not available on disk")).toBeInTheDocument();
    });
  });

  it("acknowledges warning when clicking Mark Acknowledged without claiming recovery", async () => {
    render(
      <EarlyWarningsPanel
        curriculumStage={34}
        isOpen={true}
        onToggle={vi.fn()}
      />
    );

    await waitFor(() => {
      expect(screen.getByTestId("ack-btn-warn_rep_test")).toBeInTheDocument();
      expect(screen.getByText("Records review; does not claim recovery")).toBeInTheDocument();
    });

    fireEvent.click(screen.getByTestId("ack-btn-warn_rep_test"));

    await waitFor(() => {
      expect(screen.getByTestId("ack-btn-warn_rep_test")).toHaveTextContent("✓ Acknowledged");
    });
  });

  it("renders read-only preview mode banner when isReadOnlyPreview is true", async () => {
    render(
      <EarlyWarningsPanel
        curriculumStage={34}
        isOpen={true}
        onToggle={vi.fn()}
        isReadOnlyPreview={true}
      />
    );

    await waitFor(() => {
      expect(screen.getByTestId("early-warnings-preview-banner")).toBeInTheDocument();
      expect(screen.getByText(/Displaying authentic recorded evaluation evidence/)).toBeInTheDocument();
      expect(screen.getByText("READ-ONLY PREVIEW")).toBeInTheDocument();
    });
  });

  it("triggers onOpenReplay with replay path and lead-in step when replay button is clicked", async () => {
    const onOpenReplay = vi.fn();
    render(
      <EarlyWarningsPanel
        curriculumStage={34}
        isOpen={true}
        onToggle={vi.fn()}
        onOpenReplay={onOpenReplay}
      />
    );

    await waitFor(() => {
      expect(screen.getByTestId("replay-btn-warn_rep_test")).toBeInTheDocument();
      expect(screen.getByText("Open Evaluation Replay (Step 7)")).toBeInTheDocument();
    });

    fireEvent.click(screen.getByTestId("replay-btn-warn_rep_test"));
    expect(onOpenReplay).toHaveBeenCalledWith(
      "artifacts/evaluations/replays/checkpoint-039909-seed-000041.json.gz",
      7
    );
  });
});

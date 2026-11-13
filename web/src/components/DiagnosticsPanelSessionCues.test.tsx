import { describe, expect, it, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { SessionEventHoverPortal, DiagnosticsPanel } from "./DiagnosticsPanel";
import type { SessionChangeEvent, CheckpointIndex } from "../state/types";
import * as api from "../lib/api";

const mockModelEvent: SessionChangeEvent = {
  id: "evt-model-1",
  session_id: "session_model_abc",
  session_index: 210,
  run_id: "run-1",
  timestamp: "2026-09-27T20:58:00Z",
  timestamp_ms: 1790542680000,
  checkpoint_episode: 29634,
  global_timestep: 5926800,
  event_type: "model_change",
  title: "Net2WiderNet Architecture Upgrade",
  summary: "Widen hidden layers from [128, 128, 128] to [256, 256, 128]",
  changes: [
    {
      param: "actor_layers",
      from_val: "[128, 128, 128] (41K params)",
      to_val: "[256, 256, 128] (115K params)",
      category: "architecture",
    },
    {
      param: "critic_layers",
      from_val: "[128, 128, 128]",
      to_val: "[256, 256, 128]",
      category: "architecture",
    },
  ],
  rationale: "Stage 26 Plateau Remediation: Capacity scaling for multi-modal observation density.",
  badge_icon: "🧠",
  badge_color: "#c084fc",
  session_stats: {
    checkpoint_count: 5,
    avg_success_rate: 82.5,
    avg_reward: 145.2,
    avg_steps: 124,
  },
  is_current: false,
};

const mockParamEvent: SessionChangeEvent = {
  id: "evt-param-1",
  session_id: "session_param_xyz",
  session_index: 216,
  run_id: "run-2",
  timestamp: "2026-09-29T14:15:00Z",
  timestamp_ms: 1790691300000,
  checkpoint_episode: 29919,
  global_timestep: 5983800,
  event_type: "param_change",
  title: "Stage 26 Plateau Remediation & Timeout Extension",
  summary: "max_steps: 1820 -> 2200, entropy_coef: 0.010 -> 0.020",
  changes: [
    {
      param: "max_steps",
      from_val: "1820",
      to_val: "2200",
      category: "tuning",
    },
    {
      param: "entropy_coef",
      from_val: "0.010",
      to_val: "0.020",
      category: "tuning",
    },
  ],
  rationale: "Pen corner deadlocks required +20% longer horizon and higher entropy exploration.",
  badge_icon: "⚡",
  badge_color: "#f59e0b",
  session_stats: {
    checkpoint_count: 8,
    avg_success_rate: 88.0,
    avg_reward: 162.0,
    avg_steps: 140,
  },
  is_current: true,
};

const mockCheckpointIndex: CheckpointIndex = {
  checkpoints: [
    {
      checkpoint: "chk_29634.pt",
      evaluation: "eval_29634.json",
      replay: "replay_29634.json",
      records: [],
      checkpoint_id: "chk_29634",
      checkpoint_episode: 29634,
      global_timestep: 5926800,
      curriculum_stage: 26,
      success_rate: 0.82,
      timeout_rate: 0.05,
      average_reward: 145.0,
      average_completion_steps: 124,
      average_completion_seconds: 24.5,
      average_sheep_penned: 3,
      recorded_at: "2026-09-27T20:58:00Z",
      session_id: "session_model_abc",
    },
    {
      checkpoint: "chk_29919.pt",
      evaluation: "eval_29919.json",
      replay: "replay_29919.json",
      records: [],
      checkpoint_id: "chk_29919",
      checkpoint_episode: 29919,
      global_timestep: 5983800,
      curriculum_stage: 26,
      success_rate: 0.88,
      timeout_rate: 0.02,
      average_reward: 162.0,
      average_completion_steps: 140,
      average_completion_seconds: 28.0,
      average_sheep_penned: 3,
      recorded_at: "2026-09-29T14:15:00Z",
      session_id: "session_param_xyz",
    },
  ],
  latest: null,
};

describe("SessionEventHoverPortal", () => {
  it("renders model change details, diff table, rationale, and session stats", () => {
    const targetRect = {
      left: 300,
      right: 320,
      top: 200,
      bottom: 220,
      width: 20,
      height: 20,
    } as DOMRect;

    render(<SessionEventHoverPortal event={mockModelEvent} targetRect={targetRect} />);

    const tooltip = screen.getByTestId("session-tooltip");
    expect(tooltip).toBeInTheDocument();

    // Title and type pill
    expect(screen.getByText("Net2WiderNet Architecture Upgrade")).toBeInTheDocument();
    expect(screen.getByText("MODEL ARCHITECTURE UPDATE")).toBeInTheDocument();
    expect(screen.getByText("Session #210")).toBeInTheDocument();

    // Parameter changes diff
    expect(screen.getByText("⚡ What Changed:")).toBeInTheDocument();
    expect(screen.getByText("actor_layers")).toBeInTheDocument();
    expect(screen.getByText("[128, 128, 128] (41K params)")).toBeInTheDocument();
    expect(screen.getByText("[256, 256, 128] (115K params)")).toBeInTheDocument();

    // Context / Rationale
    expect(screen.getByText(/Stage 26 Plateau Remediation/)).toBeInTheDocument();

    // Stats
    expect(screen.getByText("Session Performance")).toBeInTheDocument();
    expect(screen.getByText("82.5%")).toBeInTheDocument();
  });

  it("renders parameter tuning details and active run indicator", () => {
    const targetRect = {
      left: 300,
      right: 320,
      top: 200,
      bottom: 220,
      width: 20,
      height: 20,
    } as DOMRect;

    render(<SessionEventHoverPortal event={mockParamEvent} targetRect={targetRect} />);

    expect(screen.getByText("Stage 26 Plateau Remediation & Timeout Extension")).toBeInTheDocument();
    expect(screen.getByText("PARAMETER TUNING")).toBeInTheDocument();
    expect(screen.getByText("Active Run")).toBeInTheDocument();
    expect(screen.getByText("max_steps")).toBeInTheDocument();
    expect(screen.getByText("1820")).toBeInTheDocument();
    expect(screen.getByText("2200")).toBeInTheDocument();
  });
});

describe("DiagnosticsPanel Session Cues & Isolation Controls", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    localStorage.clear();
    vi.spyOn(api, "loadTrainingEpisodes").mockResolvedValue({
      episodes: [],
      latest_id: 0,
      next_after_id: 0,
      has_more: false,
      oldest_available_timestamp: null,
      total_matching: 0,
    });
    vi.spyOn(api, "loadSessionEvents").mockResolvedValue({
      events: [mockModelEvent, mockParamEvent],
      current_session_id: "session_param_xyz",
      total_sessions: 2,
    });
  });

  it("renders session cue pins and toggling layer pill hides/shows them", async () => {
    const { container } = render(
      <DiagnosticsPanel
        checkpointIndex={mockCheckpointIndex}
        bestCheckpointEpisode={29919}
        trainingStatus={null}
        effectiveCurriculumStage={26}
      />
    );

    // Wait for session events to be loaded and rendered
    await waitFor(() => {
      const cueMarkers = container.querySelectorAll(".session-marker-cue");
      expect(cueMarkers.length).toBeGreaterThan(0);
    });

    // Check layer pill is present
    const togglePill = screen.getByLabelText(/Session & Model Cues/i) as HTMLInputElement;
    expect(togglePill).toBeInTheDocument();
    expect(togglePill.checked).toBe(true);

    // Uncheck layer pill
    fireEvent.click(togglePill);
    expect(togglePill.checked).toBe(false);

    // Verify cue markers are now hidden
    const cueMarkersAfter = container.querySelectorAll(".session-marker-cue");
    expect(cueMarkersAfter.length).toBe(0);

    // Toggle back on
    fireEvent.click(togglePill);
    expect(togglePill.checked).toBe(true);
    const cueMarkersRestored = container.querySelectorAll(".session-marker-cue");
    expect(cueMarkersRestored.length).toBeGreaterThan(0);
  });

  it("allows selecting a session via dropdown to isolate view and clearing filter", async () => {
    render(
      <DiagnosticsPanel
        checkpointIndex={mockCheckpointIndex}
        bestCheckpointEpisode={29919}
        trainingStatus={null}
        effectiveCurriculumStage={26}
      />
    );

    // Find session dropdown selector
    const sessionSelect = (await screen.findByLabelText(/^Session$/i)) as HTMLSelectElement;
    expect(sessionSelect).toBeInTheDocument();
    expect(sessionSelect.value).toBe("all");

    // Select the model architecture upgrade session
    fireEvent.change(sessionSelect, { target: { value: "session_model_abc" } });

    // Active session banner should appear
    expect(await screen.findByText(/Isolated View:/i)).toBeInTheDocument();
    expect(screen.getByText("Net2WiderNet Architecture Upgrade")).toBeInTheDocument();

    // Clear filter button
    const clearBtn = screen.getByRole("button", { name: /✕ Show All Sessions/i });
    expect(clearBtn).toBeInTheDocument();
    fireEvent.click(clearBtn);

    // Banner should disappear and dropdown returns to "all"
    await waitFor(() => {
      expect(screen.queryByText(/Isolated View:/i)).not.toBeInTheDocument();
      expect(sessionSelect.value).toBe("all");
    });
  });

  it("displays session hover portal with diff details on mouse enter", async () => {
    const { container } = render(
      <DiagnosticsPanel
        checkpointIndex={mockCheckpointIndex}
        bestCheckpointEpisode={29919}
        trainingStatus={null}
        effectiveCurriculumStage={26}
      />
    );

    let cueMarker: Element | null = null;
    await waitFor(() => {
      cueMarker = container.querySelector(".session-marker-cue");
      expect(cueMarker).not.toBeNull();
    });

    // Hover over cue marker
    fireEvent.mouseEnter(cueMarker!);

    // Session tooltip should appear with what changed diffs
    const tooltip = await screen.findByTestId("session-tooltip");
    expect(tooltip).toBeInTheDocument();
    expect(screen.getByText("⚡ What Changed:")).toBeInTheDocument();

    // Mouse leave hides the tooltip
    fireEvent.mouseLeave(cueMarker!);
    await waitFor(() => {
      expect(screen.queryByTestId("session-tooltip")).not.toBeInTheDocument();
    });
  });
});

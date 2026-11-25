import { cleanup, render, screen, waitFor, within, fireEvent } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { App } from "./App";
import { setIsBackendOffline } from "./lib/api";

function jsonResponse(payload: unknown, status = 200): Response {
  return new Response(JSON.stringify(payload), {
    status,
    headers: {
      "Content-Type": "application/json",
    },
  });
}

const idleTrainingStatus = {
  running: false,
  fast_mode: true,
  trainer_type: "baseline",
  policy_type: "instinct",
  enable_instinct_rewards: false,
  policy_mode: "instinct_only",
  replay_mode: "baseline",
  debug_reward_breakdown: false,
  curriculum_stage: 1,
  requested_episodes: 0,
  completed_episodes: 0,
  batch_total_episodes: 0,
  batch_completed_episodes: 0,
  total_episodes_trained: 0,
  current_episode: null,
  checkpoint_episode: null,
  latest_checkpoint_episode: null,
  latest_seed: null,
  latest_replay_path: null,
  best_score: null,
  phase: "idle",
  message: "Idle",
  error: null,
};

const checkpointIndex = {
  checkpoints: [
    {
      checkpoint_episode: 0,
      checkpoint: "checkpoint-000000.json",
      evaluation: "evaluation-checkpoint-000000.json",
      replay: "/generated/replays/checkpoint-000000-seed-000011.json",
      policy_name: "trained_policy",
      trainer_type: "hill_climb",
      policy_type: "linear",
      policy_mode: "trained_policy",
      replay_mode: "trained_linear",
      total_training_episodes: 4,
      environment_config: {
        dogs: 2,
        sheep: 2,
        width: 80,
        height: 60,
      },
      reward_config: {
        instincts: {
          curriculum_stage: 1,
          enable_instinct_rewards: true,
        },
      },
      success_rate: 0,
      timeout_rate: 1,
      average_completion_steps: 300,
      average_completion_seconds: 300,
      average_sheep_penned: 0,
      average_reward: -12,
      records: [
        {
          seed: 11,
          success: false,
          timeout: true,
          stopped: false,
          steps: 300,
          simulated_seconds: 300,
          sheep_penned: 0,
          final_sheep_distance_to_pen: 17,
          no_progress_steps: 22,
          reward_total: -12,
          reward_breakdown: {
            progress_to_pen: 0,
            sheep_penned: 0,
            flock_cohesion: 0,
            scatter_penalty: 0,
            time_penalty: -0.05,
            no_progress_penalty: -1,
            wall_pressure_penalty: 0,
            wait_penalty: 0,
            terminal_success: 0,
            terminal_failure: -12,
            total: -13.05,
          },
          replay_path: "/generated/replays/checkpoint-000000-seed-000011.json",
        },
      ],
    },
  ],
  latest: {
    checkpoint_episode: 0,
    policy_name: "trained_policy",
    trainer_type: "hill_climb",
    policy_type: "linear",
    records: [
      {
        seed: 11,
        success: false,
        timeout: true,
        stopped: false,
        steps: 300,
        simulated_seconds: 300,
        sheep_penned: 0,
        final_sheep_distance_to_pen: 17,
        no_progress_steps: 22,
        reward_total: -12,
        reward_breakdown: {
          progress_to_pen: 0,
          sheep_penned: 0,
          flock_cohesion: 0,
          scatter_penalty: 0,
          time_penalty: -0.05,
          no_progress_penalty: -1,
          wall_pressure_penalty: 0,
          wait_penalty: 0,
          terminal_success: 0,
          terminal_failure: -12,
          total: -13.05,
        },
        replay_path: "/generated/replays/checkpoint-000000-seed-000011.json",
      },
    ],
    success_rate: 0,
    timeout_rate: 1,
    average_completion_steps: 300,
    average_completion_seconds: 300,
    average_sheep_penned: 0,
    average_reward: -12,
  },
};

const multiStageCheckpointIndex = {
  checkpoints: [
    checkpointIndex.checkpoints[0],
    {
      ...checkpointIndex.checkpoints[0],
      checkpoint_episode: 40,
      checkpoint: "checkpoint-000040.json",
      evaluation: "evaluation-checkpoint-000040.json",
      total_training_episodes: 40,
      reward_config: {
        instincts: {
          curriculum_stage: 2,
          enable_instinct_rewards: true,
        },
      },
      success_rate: 0.42,
      timeout_rate: 0.58,
      average_completion_steps: 210,
      average_completion_seconds: 210,
      average_sheep_penned: 1.3,
      average_reward: 22,
      records: [
        {
          ...checkpointIndex.checkpoints[0].records[0],
          steps: 210,
          reward_total: 22,
        },
      ],
    },
    {
      ...checkpointIndex.checkpoints[0],
      checkpoint_episode: 90,
      checkpoint: "checkpoint-000090.json",
      evaluation: "evaluation-checkpoint-000090.json",
      total_training_episodes: 90,
      reward_config: {
        instincts: {
          curriculum_stage: 3,
          enable_instinct_rewards: true,
        },
      },
      success_rate: 0.71,
      timeout_rate: 0.29,
      average_completion_steps: 160,
      average_completion_seconds: 160,
      average_sheep_penned: 1.8,
      average_reward: 39,
      records: [
        {
          ...checkpointIndex.checkpoints[0].records[0],
          steps: 160,
          reward_total: 39,
        },
      ],
    },
    {
      ...checkpointIndex.checkpoints[0],
      checkpoint_episode: 200,
      checkpoint: "checkpoint-000200.json",
      evaluation: "evaluation-checkpoint-000200.json",
      total_training_episodes: 200,
      journey: "journey-20260625-193907",
      reward_config: {
        instincts: {
          curriculum_stage: 11,
          enable_instinct_rewards: true,
        },
      },
      success_rate: 0.85,
      timeout_rate: 0.15,
      average_completion_steps: 120,
      average_completion_seconds: 120,
      average_sheep_penned: 2.0,
      average_reward: 50,
      records: [
        {
          ...checkpointIndex.checkpoints[0].records[0],
          steps: 120,
          reward_total: 50,
        },
      ],
    },
  ],
  latest: {
    ...checkpointIndex.latest,
    checkpoint_episode: 90,
    success_rate: 0.71,
    timeout_rate: 0.29,
    average_completion_steps: 160,
    average_completion_seconds: 160,
    average_sheep_penned: 1.8,
    average_reward: 39,
  },
};

const replay = {
  seed: 11,
  policy_name: "trained_policy",
  trainer_type: "hill_climb",
  policy_type: "linear",
  policy_mode: "trained_policy",
  replay_mode: "trained_linear",
  environment: {
    dogs: 2,
    sheep: 2,
    width: 80,
    height: 60,
    curriculum_stage: 1,
    enable_instinct_rewards: true,
  },
  final_snapshot: {
    step: 3,
    simulated_seconds: 3,
    dogs: [
      { index: 0, x: 1, y: 1, last_action: "right", role: "rear_pressure" },
      { index: 1, x: 2, y: 1, last_action: "right", role: "left_flanker" },
    ],
    sheep: [
      { index: 0, x: 14, y: 9, penned: false },
      { index: 1, x: 15, y: 10, penned: false },
    ],
    pen: { origin: { x: 16, y: 2 }, width: 5, height: 5 },
    penned_count: 0,
    average_distance_to_pen: 16,
    flock_spread: 1.2,
    no_progress_steps: 2,
    terminated: true,
    timeout: true,
    stopped: false,
    success: false,
    status: "timeout",
  },
  stats: {
    steps: 3,
    simulated_seconds: 3,
    sheep_penned: 0,
    timeout: true,
    terminated: true,
    success: false,
    stopped: false,
    stop_reason: "timeout",
    reward_total: 1.2,
    no_progress_steps: 2,
    final_avg_distance_to_pen: 16,
    final_flock_spread: 1.2,
    role_distribution: { rear_pressure: 3, left_flanker: 3, collector: 1 },
    role_switches: 4,
    collector_activations: 1,
    blocker_activations: 0,
    sheep_split_events: 1,
    final_reward_breakdown: {
      progress_to_pen: 1.4,
      sheep_penned: 0,
      flock_cohesion: 0.1,
      scatter_penalty: 0,
      time_penalty: -0.05,
      no_progress_penalty: 0,
      wall_pressure_penalty: 0,
      wait_penalty: 0,
      terminal_success: 0,
      terminal_failure: 0,
      total: 1.45,
    },
  },
  frames: [
    {
      step: 1,
      actions: ["right", "right"],
      snapshot: {
        step: 1,
        simulated_seconds: 1,
        dogs: [
          { index: 0, x: 1, y: 1, last_action: "right", role: "rear_pressure" },
          { index: 1, x: 2, y: 1, last_action: "right", role: "left_flanker" },
        ],
        sheep: [
          { index: 0, x: 15, y: 9, penned: false },
          { index: 1, x: 15, y: 10, penned: false },
        ],
        pen: { origin: { x: 16, y: 2 }, width: 5, height: 5 },
        penned_count: 0,
        average_distance_to_pen: 16.5,
        flock_spread: 1,
        no_progress_steps: 0,
        terminated: false,
        timeout: false,
        stopped: false,
        success: false,
        status: "running",
      },
      reward: {
        progress_to_pen: 1.4,
        sheep_penned: 0,
        flock_cohesion: 0.1,
        scatter_penalty: 0,
        time_penalty: -0.05,
        no_progress_penalty: 0,
        wall_pressure_penalty: 0,
        wait_penalty: 0,
        terminal_success: 0,
        terminal_failure: 0,
        total: 1.45,
      },
    },
  ],
};

beforeEach(() => {
  setIsBackendOffline(false);
  localStorage.clear();
  vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL) => {
    const path = String(input);
    if (path.includes("/api/training/status")) {
      return jsonResponse(idleTrainingStatus);
    }
    if (path.includes("/api/early-warnings")) {
      return jsonResponse({
        enabled: true,
        summary: { total_active: 0, critical_count: 0, warning_count: 0, info_count: 0, total_acknowledged: 0 },
        active_warnings: [],
        acknowledged_warnings: [],
        current_evaluation: null,
      });
    }
    if (path.includes("checkpoint-index.json")) {
      return jsonResponse(checkpointIndex);
    }
    if (path.includes("checkpoint-000000-seed-000011.json")) {
      return jsonResponse(replay);
    }
    return new Response("not found", { status: 404 });
  }));
});

afterEach(() => {
  cleanup();
  setIsBackendOffline(false);
  window.history.replaceState({}, "", "/");
  window.localStorage.clear();
  vi.unstubAllGlobals();
});

describe("App", () => {
  it("renders the simplified controls and run button", async () => {
    render(<App />);

    await waitFor(
      () => expect(screen.getByRole("button", { name: "Run best model (ep 0)" })).toBeInTheDocument(),
      { timeout: 4000 },
    );
    expect(screen.getByText("Live Replay")).toBeInTheDocument();
    expect(screen.getByLabelText("Playback controls")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Replay selected" })).toBeInTheDocument();
    expect(screen.getByText(/Checkpoint Ep 0 seeds/)).toBeInTheDocument();
    expect(screen.getByRole("option", { name: /Seed 11/ })).toBeInTheDocument();
  });

  it("adopts a higher server curriculum stage on initial load", async () => {
    window.localStorage.setItem("sheepdog_curriculum_stage", "1");

    const fetchMock = vi.mocked(fetch);
    fetchMock.mockImplementation(async (input: RequestInfo | URL) => {
      const path = String(input);
      if (path.includes("/api/training/status")) {
        return jsonResponse({
          ...idleTrainingStatus,
          curriculum_stage: 2,
        });
      }
      if (path.includes("checkpoint-index.json")) {
        return jsonResponse(checkpointIndex);
      }
      if (path.includes("checkpoint-000000-seed-000011.json")) {
        return jsonResponse(replay);
      }
      return new Response("not found", { status: 404 });
    });

    render(<App />);

    await waitFor(() => expect(screen.getByLabelText("Training controls")).toBeInTheDocument());
    expect(screen.getByText("Stage 2", { selector: ".stage-chip__label" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Promote → Stage 3" })).toBeInTheDocument();
    expect(window.localStorage.getItem("sheepdog_curriculum_stage")).toBe("2");
  });

  it("preserves a locally-promoted curriculum stage above the server's reported stage", async () => {
    // User clicked Promote so localStorage is ahead of the server's
    // last-trained stage.  Polling must NOT revert the UI back to the server
    // value while training is idle.
    window.localStorage.setItem("sheepdog_curriculum_stage", "2");

    const fetchMock = vi.mocked(fetch);
    fetchMock.mockImplementation(async (input: RequestInfo | URL) => {
      const path = String(input);
      if (path.includes("/api/training/status")) {
        return jsonResponse({
          ...idleTrainingStatus,
          curriculum_stage: 1,
        });
      }
      if (path.includes("checkpoint-index.json")) {
        return jsonResponse(checkpointIndex);
      }
      if (path.includes("checkpoint-000000-seed-000011.json")) {
        return jsonResponse(replay);
      }
      return new Response("not found", { status: 404 });
    });

    render(<App />);

    await waitFor(() => expect(screen.getByLabelText("Training controls")).toBeInTheDocument());
    expect(screen.getByText("Stage 2", { selector: ".stage-chip__label" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Promote → Stage 3" })).toBeInTheDocument();
    expect(window.localStorage.getItem("sheepdog_curriculum_stage")).toBe("2");
  });

  it("shows current live run status from loaded replay data", async () => {
    render(<App />);

    await waitFor(() => expect(screen.getByText(/rear_pressure, left_flanker/i)).toBeInTheDocument());
    expect(screen.getByText("Live Replay")).toBeInTheDocument();
    expect(within(screen.getByLabelText("Run status")).getByText("idle")).toBeInTheDocument();
    expect(screen.getByLabelText("Run status")).toBeInTheDocument();
    expect(screen.getByText("Trained policy")).toBeInTheDocument();
    expect(screen.getByText("Pen-directed behavior here comes from learned training weights rather than default instinct.")).toBeInTheDocument();
    expect(screen.getByText(/No-progress guard is active/)).toBeInTheDocument();
  });

  it("shows the read-only health dashboard in Progress", async () => {
    const user = userEvent.setup();

    render(<App />);

    await user.click(screen.getByRole("tab", { name: "Progress" }));

    await waitFor(() => expect(screen.getByLabelText("Promotion gate and stage health")).toBeInTheDocument());

    expect(screen.getByText("Stage 1 Progress")).toBeInTheDocument();
    expect(screen.getByText("Formal Evals")).toBeInTheDocument();
    expect(screen.getByText("Qualified (Last 8)")).toBeInTheDocument();
    expect(screen.getByText("Pass Threshold")).toBeInTheDocument();
    expect(screen.getByText("Recent Avg Success")).toBeInTheDocument();
    expect(screen.getByText("Learning Curve & Recent 10-Seed Evaluations")).toBeInTheDocument();

    await user.click(screen.getByRole("tab", { name: /Checkpoints/i }));
    expect(screen.getByText("Recent Checkpoints")).toBeInTheDocument();
  });

  it("lets you browse checkpoints across all stages or one specific stage in Progress", async () => {
    const user = userEvent.setup();
    const fetchMock = vi.mocked(fetch);

    fetchMock.mockImplementation(async (input: RequestInfo | URL) => {
      const path = String(input);
      if (path.includes("/api/training/status")) {
        return jsonResponse({
          ...idleTrainingStatus,
          curriculum_stage: 3,
          latest_checkpoint_episode: 90,
          total_episodes_trained: 90,
        });
      }
      if (path.includes("checkpoint-index.json")) {
        return jsonResponse(multiStageCheckpointIndex);
      }
      if (path.includes("checkpoint-000000-seed-000011.json")) {
        return jsonResponse(replay);
      }
      return new Response("not found", { status: 404 });
    });

    render(<App />);

    await user.click(screen.getByRole("tab", { name: "Progress" }));
    await user.click(screen.getByRole("tab", { name: /Checkpoints/i }));
    const historyTable = await screen.findByRole("table");
    const historyRows = () => within(historyTable).getAllByRole("row").slice(1);
    const rowEpisodes = () =>
      historyRows().map((row) => {
        const cells = within(row).getAllByRole("cell");
        return {
          episode: cells[0].textContent?.replace(/[^0-9]/g, "") ?? "",
          stage: cells[1].textContent?.trim() ?? "",
        };
      });

    // Defaults to the current stage being worked on (stage 3)
    expect(historyRows()).toHaveLength(1);
    expect(rowEpisodes()).toEqual([{ episode: "90", stage: "S3" }]);

    // Switch to all stages to see every current-journey checkpoint (excluding archived journey-20260625-193907 Stage 11)
    const filterSelect = screen.getByLabelText("Checkpoint history filter");
    await user.selectOptions(filterSelect, "all");
    await waitFor(() => expect(historyRows()).toHaveLength(3));
    expect(rowEpisodes()).toEqual(
      expect.arrayContaining([
        { episode: "90", stage: "S3" },
        { episode: "40", stage: "S2" },
        { episode: "0", stage: "S1" },
      ]),
    );
    expect(rowEpisodes()).not.toContainEqual({ episode: "200", stage: "S11" });
    expect(within(filterSelect).queryByRole("option", { name: "Stage 11" })).toBeNull();

    // Select stage 1
    await user.selectOptions(filterSelect, "1");
    await waitFor(() => expect(historyRows()).toHaveLength(1));
    expect(rowEpisodes()).toEqual([{ episode: "0", stage: "S1" }]);

    // Select stage 2
    await user.selectOptions(filterSelect, "2");
    await waitFor(() => expect(historyRows()).toHaveLength(1));
    expect(rowEpisodes()).toEqual([{ episode: "40", stage: "S2" }]);
  });

  it("clears training artifacts from the UI", async () => {
    const user = userEvent.setup();
    const fetchMock = vi.mocked(fetch);
    vi.stubGlobal("confirm", vi.fn(() => true));
    fetchMock.mockImplementation(async (input: RequestInfo | URL) => {
      const path = String(input);
      if (path.includes("/api/training/clear")) {
        return jsonResponse({
          running: false,
          fast_mode: true,
          requested_episodes: 0,
          completed_episodes: 0,
          batch_total_episodes: 0,
          batch_completed_episodes: 0,
          total_episodes_trained: 0,
          current_episode: null,
          checkpoint_episode: null,
          latest_checkpoint_episode: null,
          latest_seed: null,
          latest_replay_path: null,
          best_score: null,
          phase: "idle",
          message: "Training cleared. Baseline replay restored",
          error: null,
        });
      }
      if (path.includes("/api/training/status")) {
        return jsonResponse({
          running: false,
          fast_mode: true,
          requested_episodes: 0,
          completed_episodes: 0,
          batch_total_episodes: 0,
          batch_completed_episodes: 0,
          total_episodes_trained: 0,
          current_episode: null,
          checkpoint_episode: null,
          latest_checkpoint_episode: null,
          latest_seed: null,
          latest_replay_path: null,
          best_score: null,
          phase: "idle",
          message: "Idle",
          error: null,
        });
      }
      if (path.includes("checkpoint-index.json")) {
        return jsonResponse(checkpointIndex);
      }
      if (path.includes("checkpoint-000000-seed-000011.json")) {
        return jsonResponse(replay);
      }
      return new Response("not found", { status: 404 });
    });

    render(<App />);

    await waitFor(() => expect(screen.getByText("Advanced options")).toBeInTheDocument());
    fireEvent.click(screen.getByText("Advanced options"));
    await waitFor(() => expect(screen.getByRole("button", { name: "Clear" })).toBeInTheDocument());
    await user.click(screen.getByRole("button", { name: "Clear" }));

    await waitFor(() => expect(screen.getByText("Training cleared. Baseline replay restored")).toBeInTheDocument());
    expect(within(screen.getByLabelText("Run status")).getByText("11")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Replay selected" })).toBeInTheDocument();
  });

  it("clears saved training data from the training panel", async () => {
    const fetchMock = vi.mocked(fetch);
    fetchMock.mockImplementation(async (input: RequestInfo | URL) => {
      const path = String(input);
      if (path.includes("/api/training/clear")) {
        return jsonResponse({
          ...idleTrainingStatus,
          message: "Training history cleared",
        });
      }
      if (path.includes("/api/training/status")) {
        return jsonResponse(idleTrainingStatus);
      }
      if (path.includes("checkpoint-index.json")) {
        return jsonResponse(checkpointIndex);
      }
      if (path.includes("checkpoint-000000-seed-000011.json")) {
        return jsonResponse(replay);
      }
      return new Response("not found", { status: 404 });
    });

    vi.stubGlobal("confirm", vi.fn(() => true));
    render(<App />);

    await waitFor(() => expect(screen.getByLabelText("Training controls")).toBeInTheDocument());

    fireEvent.click(screen.getByText("Advanced options"));
    fireEvent.click(screen.getByRole("button", { name: "Clear" }));

    await waitFor(() => expect(screen.getByText("Training history cleared")).toBeInTheDocument());
    expect(fetchMock).toHaveBeenCalledWith(
      "/api/training/clear",
      expect.objectContaining({ cache: "no-store", method: "POST" }),
    );
  });

  it("runs the current dogs even when no checkpoints have been exported", async () => {
    const fetchMock = vi.mocked(fetch);
    fetchMock.mockImplementation(async (input: RequestInfo | URL) => {
      const path = String(input);
      if (path.includes("checkpoint-index.json")) {
        return new Response("not found", { status: 404 });
      }
      if (path.includes("/api/training/status")) {
        return jsonResponse(idleTrainingStatus);
      }
      if (path.includes("/api/replay/run")) {
        return jsonResponse({
          ...replay,
          policy_name: "instinct_only",
          trainer_type: "baseline",
          policy_type: "instinct",
          policy_mode: "instinct_only",
          replay_mode: "baseline",
          checkpoint_episode: null,
          environment: {
            dogs: 2,
            sheep: 2,
            width: 80,
            height: 60,
            curriculum_stage: 1,
            enable_instinct_rewards: true,
          },
          seed: 11,
        });
      }
      return new Response("not found", { status: 404 });
    });

    render(<App />);

    await waitFor(() => expect(screen.getByText("Instinct-only dogs do not know the pen. Pen-directed behavior requires training, heuristic expert mode, or a handler target command.")).toBeInTheDocument());
    fireEvent.click(screen.getByRole("button", { name: "Run best model" }));

    await waitFor(() => expect(screen.getByText("Instinct only")).toBeInTheDocument());
    expect(screen.getByText("Instinct-only dogs can chase, circle, avoid diving into the flock, and recover nearby sheep, but they do not know where the pen is.")).toBeInTheDocument();
    expect(fetchMock).toHaveBeenCalledWith(
      "/api/replay/run",
      expect.objectContaining({ cache: "no-store", method: "POST" }),
    );
    const runReplayCall = fetchMock.mock.calls.find(([request]) => String(request).includes("/api/replay/run"));
    expect(runReplayCall?.[1]).toEqual(
      expect.objectContaining({
        body: JSON.stringify({
          seed: 11,
          checkpoint_episode: null,
          trainer_type: "baseline",
          policy_type: "instinct",
          policy_mode: "instinct_only",
          effective_config: {
            enable_instinct_rewards: false,
            curriculum_stage: 1,
            debug_reward_breakdown: false,
          },
        }),
      }),
    );
  });

  it("runs the current dogs with the latest trained checkpoint when trained artifacts exist", async () => {
    const user = userEvent.setup();
    const fetchMock = vi.mocked(fetch);
    fetchMock.mockImplementation(async (input: RequestInfo | URL) => {
      const path = String(input);
      if (path.includes("/api/training/status")) {
        return jsonResponse({
          ...idleTrainingStatus,
          trainer_type: "hill_climb",
          policy_type: "linear",
          policy_mode: "trained_policy",
          replay_mode: "trained_linear",
          total_episodes_trained: 4,
        });
      }
      if (path.includes("checkpoint-index.json")) {
        return jsonResponse(checkpointIndex);
      }
      if (path.includes("checkpoint-000000-seed-000011.json")) {
        return jsonResponse(replay);
      }
      if (path.includes("/api/replay/run")) {
        return jsonResponse(replay);
      }
      return new Response("not found", { status: 404 });
    });

    render(<App />);

    await waitFor(() => expect(screen.getByRole("button", { name: "Run best model (ep 0)" })).toBeInTheDocument());
    await user.click(screen.getByRole("button", { name: "Run best model (ep 0)" }));

    const runReplayCall = fetchMock.mock.calls.find(([request]) => String(request).includes("/api/replay/run"));
    expect(runReplayCall?.[1]).toEqual(
      expect.objectContaining({
        body: JSON.stringify({
          seed: 11,
          checkpoint_episode: null,
          trainer_type: "hill_climb",
          policy_type: "linear",
          policy_mode: "trained_policy",
          effective_config: {
            enable_instinct_rewards: false,
            curriculum_stage: 1,
            debug_reward_breakdown: false,
          },
        }),
      }),
    );
  });

  it("treats the Vite HTML fallback for a missing checkpoint index as no exported checkpoints", async () => {
    const fetchMock = vi.mocked(fetch);
    fetchMock.mockImplementation(async (input: RequestInfo | URL) => {
      const path = String(input);
      if (path.includes("checkpoint-index.json")) {
        return new Response("<!doctype html><html><body>fallback</body></html>", {
          status: 200,
          headers: {
            "Content-Type": "text/html",
          },
        });
      }
      if (path.includes("/api/training/status")) {
        return jsonResponse(idleTrainingStatus);
      }
      return new Response("not found", { status: 404 });
    });

    render(<App />);

    await waitFor(() => expect(screen.getByText("No replay loaded yet.")).toBeInTheDocument());
    expect(screen.queryByText(/Unexpected token/)).not.toBeInTheDocument();
  });

  it.each([false, true])("starts Stage 1 with instincts off (continuous: %s)", async (continuous) => {
    const fetchMock = vi.mocked(fetch);
    fetchMock.mockImplementation(async (input: RequestInfo | URL) => {
      const path = String(input);
      if (path.includes("/api/training/start")) {
        return jsonResponse({
          ...idleTrainingStatus,
          running: true,
          requested_episodes: 5,
          batch_total_episodes: 5,
          enable_instinct_rewards: false,
          curriculum_stage: 1,
          message: "Queued training job",
        });
      }
      if (path.includes("/api/training/status")) {
        return jsonResponse(idleTrainingStatus);
      }
      if (path.includes("checkpoint-index.json")) {
        return jsonResponse(checkpointIndex);
      }
      if (path.includes("checkpoint-000000-seed-000011.json")) {
        return jsonResponse(replay);
      }
      return new Response("not found", { status: 404 });
    });

    render(<App />);

    await waitFor(() => expect(screen.getByLabelText("Training controls")).toBeInTheDocument());
    if (continuous) {
      fireEvent.click(screen.getByRole("checkbox", { name: "Continuous training" }));
      expect(screen.getByLabelText("Episode target per batch")).toBeInTheDocument();
    }
    fireEvent.click(screen.getByRole("button", { name: continuous ? "Start continuous training" : "Train 50 more" }));

    const trainingStartCall = fetchMock.mock.calls.find(([request]) =>
      String(request).includes("/api/training/start"),
    );
    expect(trainingStartCall).toBeDefined();
    expect(trainingStartCall?.[1]).toEqual(
      expect.objectContaining({
        cache: "no-store",
        method: "POST",
        body: JSON.stringify({
          episodes: 50,
          fast_mode: true,
          enable_instinct_rewards: false,
          curriculum_stage: 1,
          debug_reward_breakdown: false,
          auto_promote: true,
          ...(continuous ? { continuous: true } : {}),
        }),
      }),
    );
  });

  it("requests a graceful pause from the training panel while a run is active", async () => {
    const fetchMock = vi.mocked(fetch);
    fetchMock.mockImplementation(async (input: RequestInfo | URL) => {
      const path = String(input);
      if (path.includes("/api/training/pause")) {
        return jsonResponse({
          ...idleTrainingStatus,
          running: true,
          phase: "paused",
          message: "Pause requested; waiting for the current checkpoint to finish",
        });
      }
      if (path.includes("/api/training/status")) {
        return jsonResponse({
          ...idleTrainingStatus,
          running: true,
          requested_episodes: 50,
          batch_total_episodes: 50,
          batch_completed_episodes: 12,
          current_episode: 12,
          message: "Training in progress",
        });
      }
      if (path.includes("checkpoint-index.json")) {
        return jsonResponse(checkpointIndex);
      }
      if (path.includes("checkpoint-000000-seed-000011.json")) {
        return jsonResponse(replay);
      }
      return new Response("not found", { status: 404 });
    });

    render(<App />);

    await waitFor(() => expect(screen.getByRole("button", { name: "Pause after checkpoint" })).toBeInTheDocument());
    fireEvent.click(screen.getByRole("button", { name: "Pause after checkpoint" }));

    const pauseCall = fetchMock.mock.calls.find(([request]) =>
      String(request).includes("/api/training/pause"),
    );
    expect(pauseCall?.[1]).toEqual(
      expect.objectContaining({
        cache: "no-store",
        method: "POST",
      }),
    );
  });

  it.each([false, true])("resumes a saved training session (continuous: %s)", async (continuous) => {
    const fetchMock = vi.mocked(fetch);
    fetchMock.mockImplementation(async (input: RequestInfo | URL) => {
      const path = String(input);
      if (path.includes("/api/training/start")) {
        return jsonResponse({
          ...idleTrainingStatus,
          running: true,
          requested_episodes: 12,
          batch_total_episodes: 12,
          curriculum_stage: 4,
          enable_instinct_rewards: true,
          message: "Queued training job",
        });
      }
      if (path.includes("/api/training/status")) {
        return jsonResponse({
          ...idleTrainingStatus,
          phase: "paused",
          message: "Training paused; 12 episodes remain for resume",
          curriculum_stage: 4,
          enable_instinct_rewards: true,
          resume_available: true,
          resume_remaining_episodes: continuous ? 0 : 12,
          resume_request: {
            episodes: 50,
            fast_mode: true,
            enable_instinct_rewards: true,
            curriculum_stage: 4,
            debug_reward_breakdown: false,
            auto_promote: true,
            ...(continuous ? { continuous: true } : {}),
          },
        });
      }
      if (path.includes("checkpoint-index.json")) {
        return jsonResponse(checkpointIndex);
      }
      if (path.includes("checkpoint-000000-seed-000011.json")) {
        return jsonResponse(replay);
      }
      return new Response("not found", { status: 404 });
    });

    render(<App />);

    const resumeLabel = continuous ? "Resume continuous training" : "Resume 12 remaining";
    await waitFor(() => expect(screen.getByRole("button", { name: resumeLabel })).toBeInTheDocument());
    fireEvent.click(screen.getByRole("button", { name: resumeLabel }));

    const resumeCall = fetchMock.mock.calls.find(([request]) =>
      String(request).includes("/api/training/start"),
    );
    expect(resumeCall?.[1]).toEqual(
      expect.objectContaining({
        cache: "no-store",
        method: "POST",
        body: JSON.stringify({
          episodes: continuous ? 50 : 12,
          fast_mode: true,
          enable_instinct_rewards: true,
          curriculum_stage: 4,
          debug_reward_breakdown: false,
          auto_promote: true,
          ...(continuous ? { continuous: true, resume: true } : {}),
        }),
      }),
    );
  });

  it("ends the current replay immediately when the episode is clearly bad", async () => {
    const skippableReplay = {
      ...replay,
      frames: [
        ...replay.frames,
        {
          step: replay.final_snapshot.step,
          actions: ["wait", "wait"],
          snapshot: replay.final_snapshot,
          reward: replay.stats.final_reward_breakdown,
        },
      ],
    };
    const fetchMock = vi.mocked(fetch);
    fetchMock.mockImplementation(async (input: RequestInfo | URL) => {
      const path = String(input);
      if (path.includes("/api/training/status")) {
        return jsonResponse(idleTrainingStatus);
      }
      if (path.includes("checkpoint-index.json")) {
        return jsonResponse(checkpointIndex);
      }
      if (path.includes("checkpoint-000000-seed-000011.json")) {
        return jsonResponse(skippableReplay);
      }
      return new Response("not found", { status: 404 });
    });

    render(<App />);

    await waitFor(() => expect(screen.getByRole("button", { name: "End episode" })).toBeEnabled());

    fireEvent.click(screen.getByRole("button", { name: "End episode" }));

    const statusPanel = screen.getByLabelText("Run status");
    expect(within(statusPanel).getByText("3")).toBeInTheDocument();
    expect(within(statusPanel).getByText("timeout")).toBeInTheDocument();
  });

  it("persists progress view filters in localStorage", async () => {
    const user = userEvent.setup();
    const fetchMock = vi.mocked(fetch);

    fetchMock.mockImplementation(async (input: RequestInfo | URL) => {
      const path = String(input);
      if (path.includes("/api/training/status")) {
        return jsonResponse({
          ...idleTrainingStatus,
          curriculum_stage: 3,
          latest_checkpoint_episode: 90,
          total_episodes_trained: 90,
        });
      }
      if (path.includes("checkpoint-index.json")) {
        return jsonResponse(multiStageCheckpointIndex);
      }
      return new Response("not found", { status: 404 });
    });

    // Populate localStorage with saved progress checkpoint filter
    window.localStorage.setItem("sheepdog_progress_stage_filter", "all");

    render(<App />);

    // Go to Progress tab (defaults to Learning Curve subtab)
    await user.click(screen.getByRole("tab", { name: "Progress" }));

    // Interact with learning curve controls
    await user.click(screen.getByRole("button", { name: "Last 50" }));
    await user.click(screen.getByRole("button", { name: "Return" }));
    await user.click(screen.getByRole("button", { name: "All Stages" }));
    expect(screen.getByRole("button", { name: "Last 50" }).className).toContain("sd-pill-btn--active");
    expect(screen.getByRole("button", { name: "Return" }).className).toContain("sd-pill-btn--active");
    expect(screen.getByRole("button", { name: "All Stages" }).className).toContain("sd-pill-btn--active");

    // Switch to Checkpoints subtab and verify localStorage filter restores "all" (excluding archived Stage 11)
    await user.click(screen.getByRole("tab", { name: /Checkpoints/i }));
    const filterSelect = screen.getByLabelText("Checkpoint history filter") as HTMLSelectElement;
    expect(filterSelect.value).toBe("all");
    const historyTable = await screen.findByRole("table");
    await waitFor(() => expect(within(historyTable).getAllByRole("row").slice(1)).toHaveLength(3));

    // Change checkpoint filter and verify localStorage updates
    await user.selectOptions(filterSelect, "2");
    expect(window.localStorage.getItem("sheepdog_progress_stage_filter")).toBe("2");
    await waitFor(() => expect(within(historyTable).getAllByRole("row").slice(1)).toHaveLength(1));
  });

  it("displays 'Backend disconnected' when backend polling fails during active training", async () => {
    const fetchMock = vi.mocked(fetch);
    let pollCount = 0;

    fetchMock.mockImplementation(async (input: RequestInfo | URL) => {
      const path = String(input);
      if (path.includes("/api/training/status")) {
        pollCount++;
        if (pollCount === 1) {
          return jsonResponse({
            ...idleTrainingStatus,
            running: true,
            phase: "learning",
            message: "Learning neural policy: segment 1/5",
            curriculum_stage: 9,
            requested_episodes: 200,
            completed_episodes: 20,
            batch_total_episodes: 200,
            batch_completed_episodes: 20,
          });
        }
        throw new TypeError("Failed to fetch");
      }
      if (path.includes("checkpoint-index.json")) {
        return jsonResponse(checkpointIndex);
      }
      return new Response("not found", { status: 404 });
    });

    render(<App />);

    // First poll succeeds (shows training state)
    await waitFor(() => {
      expect(screen.getByText("Learning neural policy: segment 1/5")).toBeInTheDocument();
    });

    // Advance timer or trigger re-poll to simulate backend disconnect failure
    await waitFor(() => {
      expect(screen.getAllByText("Backend disconnected").length).toBeGreaterThan(0);
    });

    // Verify stale 'learning' phase is no longer shown
    expect(screen.queryByText("Learning neural policy: segment 1/5")).not.toBeInTheDocument();
  });

  it("replays selected checkpoint without clobbering running state or sending stage -1", async () => {
    const fetchMock = vi.mocked(fetch);
    const stagelessCheckpointIndex = {
      checkpoints: [
        {
          ...checkpointIndex.checkpoints[0],
          environment_config: { dogs: 2, sheep: 2, width: 80, height: 60 },
          reward_config: undefined,
          records: [
            {
              ...checkpointIndex.checkpoints[0].records[0],
              replay_path: undefined,
            },
          ],
        },
      ],
      latest: checkpointIndex.latest,
    };
    const twoFrameReplay = {
      ...replay,
      frames: [
        replay.frames[0],
        {
          ...replay.frames[0],
          step: 2,
          snapshot: {
            ...replay.frames[0].snapshot,
            step: 2,
            simulated_seconds: 2,
          },
        },
      ],
    };

    fetchMock.mockImplementation(async (input: RequestInfo | URL) => {
      const path = String(input);
      if (path.includes("/api/training/status")) {
        return jsonResponse({ ...idleTrainingStatus, curriculum_stage: 3 });
      }
      if (path.includes("checkpoint-index.json")) {
        return jsonResponse(stagelessCheckpointIndex);
      }
      if (path.includes("/api/replay/run")) {
        return jsonResponse(twoFrameReplay);
      }
      return new Response("not found", { status: 404 });
    });

    render(<App />);

    await waitFor(() => expect(screen.getByRole("button", { name: "Replay selected" })).toBeInTheDocument());
    await waitFor(() => expect(within(screen.getByLabelText("Run status")).getByText("idle")).toBeInTheDocument());

    const callsBeforeClick = fetchMock.mock.calls.filter(([req]) => String(req).includes("/api/replay/run")).length;
    fireEvent.click(screen.getByRole("button", { name: "Replay selected" }));

    await waitFor(() => {
      const callsAfterClick = fetchMock.mock.calls.filter(([req]) => String(req).includes("/api/replay/run"));
      expect(callsAfterClick.length).toBe(callsBeforeClick + 1);
      const lastCallBody = JSON.parse(String(callsAfterClick[callsAfterClick.length - 1][1]?.body ?? "{}"));
      expect(lastCallBody.effective_config?.curriculum_stage).toBe(3);
    });

    // Run status transitions to running (or advances to frame 2), never immediately clobbered by a duplicate load
    await waitFor(() => {
      const statusText = screen.getByLabelText("Run status").textContent ?? "";
      expect(statusText.includes("running") || statusText.includes("2")).toBe(true);
    });
  });

  it("discards stale in-flight status polls across user lifecycle mutations", async () => {
    const fetchMock = vi.mocked(fetch);
    const slowPollHolder: { resolve: ((res: Response) => void) | null } = { resolve: null };
    let statusCallCount = 0;

    fetchMock.mockImplementation((input: RequestInfo | URL) => {
      const path = String(input);
      if (path.includes("/api/training/status")) {
        statusCallCount += 1;
        if (statusCallCount === 1) {
          return Promise.resolve(jsonResponse(idleTrainingStatus));
        }
        if (statusCallCount === 2) {
          return new Promise<Response>((resolve) => {
            slowPollHolder.resolve = resolve;
          });
        }
        return Promise.resolve(
          jsonResponse({
            ...idleTrainingStatus,
            running: true,
            phase: "learning",
            message: "Training batch active",
          }),
        );
      }
      if (path.includes("/api/training/start")) {
        return Promise.resolve(
          jsonResponse({
            ...idleTrainingStatus,
            running: true,
            phase: "learning",
            requested_episodes: 50,
            batch_total_episodes: 50,
            message: "Training batch active",
          }),
        );
      }
      if (path.includes("checkpoint-index.json")) {
        return Promise.resolve(jsonResponse(checkpointIndex));
      }
      if (path.includes("checkpoint-000000-seed-000011.json")) {
        return Promise.resolve(jsonResponse(replay));
      }
      return Promise.resolve(new Response("not found", { status: 404 }));
    });

    render(<App />);

    await waitFor(() => expect(screen.getByRole("button", { name: "Train 50 more" })).toBeInTheDocument());
    await waitFor(() => expect(slowPollHolder.resolve).not.toBeNull());

    fireEvent.click(screen.getByRole("button", { name: "Train 50 more" }));
    await waitFor(() => expect(screen.getByText("Training batch active")).toBeInTheDocument());

    // Now resolve the stale in-flight status poll that started BEFORE clicking Train
    slowPollHolder.resolve?.(
      jsonResponse({ ...idleTrainingStatus, running: false, phase: "idle", message: "Stale idle poll" }),
    );

    // Ensure the stale poll was ignored and did not overwrite "Training batch active"
    await new Promise((r) => setTimeout(r, 50));
    expect(screen.queryByText("Stale idle poll")).not.toBeInTheDocument();
    expect(screen.getByText("Training batch active")).toBeInTheDocument();
  });

  it("renders exactly 4 top-level tabs, zero Studio v2 links, and zero duplicate evaluation banners", async () => {
    const fetchMock = vi.mocked(fetch);
    fetchMock.mockImplementation(async (input: RequestInfo | URL) => {
      const path = String(input);
      if (path.includes("/api/training/status")) {
        return jsonResponse({
          ...idleTrainingStatus,
          phase: "evaluation_complete",
          auto_promote_gate: {
            ready: true,
            decision: "promote_ready",
            reason: "Promotion gate satisfied.",
          },
          evaluation_status: {
            in_progress: false,
            mode: "confidence",
            total_seeds: 10,
            completed_seeds: 10,
            success_count: 10,
            recent_results: [
              {
                seed: 11,
                success: true,
                timeout: false,
                status: "success",
                reward: 45,
                penned: 2,
                total_sheep: 2,
                steps: 80,
              },
            ],
          },
        });
      }
      if (path.includes("checkpoint-index.json")) {
        return jsonResponse(checkpointIndex);
      }
      if (path.includes("checkpoint-000000-seed-000011.json")) {
        return jsonResponse(replay);
      }
      return new Response("not found", { status: 404 });
    });

    render(<App />);

    await waitFor(() => expect(screen.getByLabelText("Training controls")).toBeInTheDocument());

    const topNav = screen.getByRole("tablist", { name: "Workspace sections" });
    const tabs = within(topNav).getAllByRole("tab");
    expect(tabs.map((tab) => tab.textContent)).toEqual([
      "Train",
      "Progress",
      "Trouble Spots & Replays",
      "Settings",
    ]);
    expect(screen.queryByText(/Studio v2/i)).not.toBeInTheDocument();
    expect(screen.queryAllByTestId("evaluation-banner")).toHaveLength(0);
    expect(screen.getAllByRole("button", { name: "Promote → Stage 2" })).toHaveLength(1);
  });

  it("navigates Trouble Spots & Replays sub-views cleanly", async () => {
    const user = userEvent.setup();
    render(<App />);

    await user.click(screen.getByRole("tab", { name: "Trouble Spots & Replays" }));

    await waitFor(() =>
      expect(screen.getByText("Failure Breakdown & Hardest Seeds")).toBeInTheDocument(),
    );
    expect(
      screen.getByText("Stage Learning Bottlenecks & Spatial Heatmap"),
    ).toBeInTheDocument();

    await user.click(screen.getByRole("tab", { name: "Step-by-Step Inspector" }));
    expect(screen.getByRole("tab", { name: "Step-by-Step Inspector" })).toHaveAttribute(
      "aria-selected",
      "true",
    );
    expect(
      screen.getByRole("region", { name: "Temporal Interaction Diagnostics" }),
    ).toBeInTheDocument();

    await user.click(screen.getByRole("tab", { name: "Scenario Sandbox" }));
    expect(screen.getByRole("tab", { name: "Scenario Sandbox" })).toHaveAttribute(
      "aria-selected",
      "true",
    );
    expect(screen.getByLabelText("Scenario status")).toBeInTheDocument();

    await user.click(screen.getByRole("tab", { name: "Scenario & Seed Failures" }));
    expect(screen.getByText("Failure Breakdown & Hardest Seeds")).toBeInTheDocument();
  });

  it("renders Settings with ConfigPanel and collapsed disclosures until expanded", async () => {
    const user = userEvent.setup();
    const fetchMock = vi.mocked(fetch);
    fetchMock.mockImplementation(async (input: RequestInfo | URL) => {
      const path = String(input);
      if (path.includes("/api/training/status")) {
        return jsonResponse(idleTrainingStatus);
      }
      if (path.includes("checkpoint-index.json")) {
        return jsonResponse(checkpointIndex);
      }
      if (path.includes("checkpoint-000000-seed-000011.json")) {
        return jsonResponse(replay);
      }
      if (path.includes("/api/network/topology")) {
        return jsonResponse({
          observation_mode: "guided",
          hidden_layer_sizes: [128, 128],
          observation_size: 54,
          action_size: 9,
          actor_head: { type: "linear", node_count: 9, output: "logits" },
          critic_head: { type: "linear", node_count: 1, output: "value" },
          action_masking_enabled: true,
          connectivity: "dense",
        });
      }
      return new Response("not found", { status: 404 });
    });

    render(<App />);

    await user.click(screen.getByRole("tab", { name: "Settings" }));

    await waitFor(() => expect(screen.getByLabelText("Config")).toBeInTheDocument());

    const disclosureSummaries = [
      "Checkpoint Browser, Restore & Fork",
      "Policy Network Topology",
      "Observation & Action Layers",
      "Curriculum Stages & RL Glossary",
      "Raw Episode Telemetry Log",
    ];
    for (const label of disclosureSummaries) {
      const summary = screen.getByText(label);
      const details = summary.closest("details");
      expect(details).not.toBeNull();
      expect(details).not.toHaveAttribute("open");
    }

    await user.click(screen.getByText("Policy Network Topology"));
    await waitFor(() => {
      expect(screen.getByText("Policy Network Topology").closest("details")).toHaveAttribute(
        "open",
      );
    });
    await waitFor(() => {
      expect(
        fetchMock.mock.calls.some(([req]) => String(req).includes("/api/network/topology")),
      ).toBe(true);
    });
  });

  it("deep-links directly to Step-by-Step Inspector from ?workspace=temporal&replay=...", async () => {
    window.history.replaceState(
      {},
      "",
      "/?workspace=temporal&replay=checkpoint-039909-seed-000041",
    );

    render(<App />);

    await waitFor(() =>
      expect(screen.getByRole("tab", { name: "Trouble Spots & Replays" })).toHaveAttribute(
        "aria-selected",
        "true",
      ),
    );
    expect(screen.getByRole("tab", { name: "Step-by-Step Inspector" })).toHaveAttribute(
      "aria-selected",
      "true",
    );
    const replayInput = screen.getByLabelText(
      "Or Open Any Approved Replay ID:",
    ) as HTMLInputElement;
    expect(replayInput.value).toBe("checkpoint-039909-seed-000041");
  });

  it("displays live PPO timestep progress and rollout telemetry mid-segment even when batch_completed_episodes is 0", async () => {
    const fetchMock = vi.mocked(fetch);
    fetchMock.mockImplementation(async (input: RequestInfo | URL) => {
      const path = String(input);
      if (path.includes("/api/training/status")) {
        return jsonResponse({
          ...idleTrainingStatus,
          running: true,
          phase: "episode_complete",
          curriculum_stage: 7,
          requested_episodes: 34,
          batch_total_episodes: 34,
          batch_completed_episodes: 0,
          batch_total_segments: 35,
          batch_completed_segments: 0,
          message: "Learning neural policy: 17328/32768 timesteps (53%)",
          latest_completed_environment_episode: 19212,
          latest_episode_result: "SUCCESS",
          latest_episode_reward: 347.53,
          live_rollout_window_count: 65,
          live_rollout_success_rate: 0.98,
        });
      }
      if (path.includes("checkpoint-index.json")) {
        return jsonResponse(checkpointIndex);
      }
      if (path.includes("checkpoint-000000-seed-000011.json")) {
        return jsonResponse(replay);
      }
      return new Response("not found", { status: 404 });
    });

    render(<App />);

    await waitFor(() =>
      expect(screen.getByText("17,328 / 32,768 timesteps (53%)")).toBeInTheDocument(),
    );
    expect(screen.getByText("Next checkpoint")).toBeInTheDocument();
    expect(screen.getByRole("progressbar", { name: "Training batch progress" })).toHaveAttribute(
      "aria-valuenow",
      "53",
    );
    expect(
      screen.getByText("Batch checkpoints: 0 / 35 saved · 0 / 34 episodes"),
    ).toBeInTheDocument();

    const rolloutStrip = screen.getByTestId("live-rollout-telemetry");
    expect(within(rolloutStrip).getByText("Ep #19,212")).toBeInTheDocument();
    expect(within(rolloutStrip).getByText("SUCCESS")).toBeInTheDocument();
    expect(within(rolloutStrip).getByText("+347.5")).toBeInTheDocument();
    expect(
      within(rolloutStrip).getByText("65 rollouts since last checkpoint · 98% rollout win rate"),
    ).toBeInTheDocument();
    expect(
      screen.getByText("Learning neural policy: 17328/32768 timesteps (53%)"),
    ).toBeInTheDocument();
    expect(
      screen.getByText(
        "Field shows last saved checkpoint replay — background workers export a new replay at each checkpoint.",
      ),
    ).toBeInTheDocument();
    expect(
      screen.getByText("Training · Ckpt 0/35 (53%) · Ep #19,212"),
    ).toBeInTheDocument();
  });

  it("prevents mid-batch status oscillation and defers insight polling until Progress is active", async () => {
    const user = userEvent.setup();
    const fetchMock = vi.mocked(fetch);
    let statusPollCount = 0;

    fetchMock.mockImplementation(async (input: RequestInfo | URL) => {
      const path = String(input);
      if (path.includes("/api/training/status")) {
        statusPollCount += 1;
        if (statusPollCount === 1) {
          return jsonResponse({
            ...idleTrainingStatus,
            running: true,
            phase: "learning",
            curriculum_stage: 7,
            requested_episodes: 34,
            completed_episodes: 18,
            batch_total_episodes: 34,
            batch_completed_episodes: 18,
            batch_total_segments: 35,
            batch_completed_segments: 0.53,
            current_global_timestep: 500000,
            message: "Learning neural policy: 17328/32768 timesteps (53%)",
            latest_completed_environment_episode: 19212,
            latest_episode_result: "SUCCESS",
            latest_episode_reward: 347.53,
            live_rollout_window_count: 65,
            live_rollout_success_rate: 0.98,
          });
        }
        return jsonResponse({
          ...idleTrainingStatus,
          running: true,
          phase: "episode_complete",
          curriculum_stage: 7,
          requested_episodes: 34,
          completed_episodes: 0,
          batch_total_episodes: 34,
          batch_completed_episodes: 0,
          batch_total_segments: undefined,
          batch_completed_segments: 0,
          current_global_timestep: 499000,
          message: "Learning neural policy: 16000/32768 timesteps (49%)",
          latest_completed_environment_episode: 19213,
          latest_episode_result: "TIMEOUT",
          latest_episode_reward: -12.4,
          live_rollout_window_count: 66,
          live_rollout_success_rate: 0.97,
        });
      }
      if (path.includes("/api/insights/training-episodes")) {
        return jsonResponse({ episodes: [], total: 0 });
      }
      if (path.includes("/api/insights/session-events")) {
        return jsonResponse({ events: [], total_sessions: 0 });
      }
      if (path.includes("checkpoint-index.json")) {
        return jsonResponse(checkpointIndex);
      }
      if (path.includes("checkpoint-000000-seed-000011.json")) {
        return jsonResponse(replay);
      }
      return new Response("not found", { status: 404 });
    });

    render(<App />);

    await waitFor(() =>
      expect(screen.getByText("17,328 / 32,768 timesteps (53%)")).toBeInTheDocument(),
    );
    await waitFor(() => expect(statusPollCount).toBeGreaterThanOrEqual(2), { timeout: 2000 });

    // Non-regression: second poll's zeroed segment/episode counts and lower timestep message must not regress UI.
    expect(screen.getByText("17,328 / 32,768 timesteps (53%)")).toBeInTheDocument();
    expect(
      screen.getByText("Batch checkpoints: 0 / 35 saved · 18 / 34 episodes"),
    ).toBeInTheDocument();
    // Latest rollout metadata from poll 2 still updates live.
    expect(screen.getByText("Ep #19,213")).toBeInTheDocument();

    // Insights endpoints must not be polled while viewing Train tab.
    expect(
      fetchMock.mock.calls.some(
        ([req]) =>
          String(req).includes("/api/insights/training-episodes") ||
          String(req).includes("/api/insights/session-events"),
      ),
    ).toBe(false);

    // Switching to Progress immediately triggers insights polling.
    await user.click(screen.getByRole("tab", { name: "Progress" }));
    await waitFor(() => {
      expect(
        fetchMock.mock.calls.some(([req]) => String(req).includes("/api/insights/training-episodes")),
      ).toBe(true);
      expect(
        fetchMock.mock.calls.some(([req]) => String(req).includes("/api/insights/session-events")),
      ).toBe(true);
    });
  });

  it("renders prominent curriculum stage indicators in the topbar, Train view, and Progress view", async () => {
    const user = userEvent.setup();
    render(<App />);

    await waitFor(() => expect(screen.getByTestId("topbar-stage-pill")).toBeInTheDocument());
    const topbarPill = screen.getByTestId("topbar-stage-pill");
    expect(topbarPill).toHaveTextContent("Stage 1 / 32");
    expect(topbarPill).toHaveTextContent("· 2d / 2s");

    const trainCallout = screen.getByTestId("train-stage-callout");
    expect(trainCallout).toHaveTextContent("Stage 1");
    expect(trainCallout).toHaveTextContent("of 32");

    await user.click(screen.getByRole("tab", { name: "Progress" }));
    await waitFor(() => expect(screen.getByTestId("progress-stage-pill")).toBeInTheDocument());
    expect(screen.getByTestId("progress-stage-pill")).toHaveTextContent("Stage 1 of 32");
  });

  it("validates and launches a stage-queue training pass without calling cross-stage-fork", async () => {
    const user = userEvent.setup();
    const fetchMock = vi.mocked(fetch);
    const startedBodies: Array<Record<string, unknown>> = [];
    let forkCalls = 0;

    fetchMock.mockImplementation(async (input: RequestInfo | URL, init?: RequestInit) => {
      const path = String(input);
      if (path.includes("/api/training/status")) {
        return jsonResponse({
          ...idleTrainingStatus,
          max_curriculum_stage: 38,
        });
      }
      if (path.includes("/api/training/cross-stage-fork")) {
        forkCalls += 1;
        return jsonResponse(idleTrainingStatus);
      }
      if (path.includes("/api/training/start")) {
        const body = JSON.parse(String(init?.body ?? "{}")) as Record<string, unknown>;
        startedBodies.push(body);
        return jsonResponse({
          ...idleTrainingStatus,
          max_curriculum_stage: 38,
          running: true,
          phase: "learning",
          curriculum_stage: 10,
          continuous: true,
          auto_promote: true,
          stage_queue: [10, 11, 12, 15, 16, 17, 18, 19, 20, 25, 33, 34, 35, 36, 37, 38],
          stage_queue_index: 0,
          stage_spec: String(body.stage_spec ?? ""),
          message: "Running stage queue",
        });
      }
      if (path.includes("checkpoint-index.json")) {
        return jsonResponse(checkpointIndex);
      }
      if (path.includes("checkpoint-000000-seed-000011.json")) {
        return jsonResponse(replay);
      }
      return new Response("not found", { status: 404 });
    });

    render(<App />);

    const stageSpecInput = await screen.findByLabelText("Stages to run (optional)");

    // Invalid spec shows alert and disables Start button
    await user.type(stageSpecInput, "39");
    expect(screen.getByRole("alert")).toHaveTextContent(/39/);
    const startWhileInvalid = screen.getByRole("button", { name: "Train 50 more" });
    expect(startWhileInvalid).toBeDisabled();

    // Valid multi-range spec shows preview, pass label input, and Run 16 stages button
    await user.clear(stageSpecInput);
    await user.type(stageSpecInput, "10-12, 15-20, 25, 33-38");

    expect(screen.getByTestId("stage-spec-preview")).toHaveTextContent(
      "Queued 16 stages (10-12, 15-20, 25, 33-38) · uses current trained model",
    );
    const passLabelInput = screen.getByLabelText("Pass label");
    await user.type(passLabelInput, "Hard tail polish");

    const runQueueButton = screen.getByRole("button", { name: "Run 16 stages" });
    expect(runQueueButton).toBeEnabled();
    await user.click(runQueueButton);

    await waitFor(() => expect(startedBodies).toHaveLength(1));
    expect(forkCalls).toBe(0);
    expect(startedBodies[0]).toMatchObject({
      stage_spec: "10-12, 15-20, 25, 33-38",
      pass_label: "Hard tail polish",
      starting_model_source: "latest",
      continuous: true,
      auto_promote: true,
    });

    await waitFor(() =>
      expect(screen.getByTestId("stage-queue-progress")).toHaveTextContent(
        "Stage queue: 1 / 16 · Stage 10 (10-12, 15-20, 25, 33-38)",
      ),
    );
  });
});

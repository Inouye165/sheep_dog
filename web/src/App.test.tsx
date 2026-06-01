import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { App } from "./App";

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
  localStorage.clear();
  vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL) => {
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
    return new Response("not found", { status: 404 });
  }));
});

afterEach(() => {
  window.localStorage.clear();
  vi.unstubAllGlobals();
});

describe("App", () => {
  it("renders the simplified controls and run button", async () => {
    render(<App />);

    await userEvent.click(screen.getByRole("tab", { name: "Watch" }));
    await waitFor(() => expect(screen.getByText("Live Replay")).toBeInTheDocument());
    expect(screen.getByLabelText("Playback controls")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Run best model (ep 0)" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Replay selected" })).toBeInTheDocument();
    expect(screen.getByText(/Checkpoint 0/)).toBeInTheDocument();
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
    expect(window.localStorage.getItem("sheepdog_curriculum_stage")).toBe("2");
  });

  it("shows current live run status from loaded replay data", async () => {
    render(<App />);

    await userEvent.click(screen.getByRole("tab", { name: "Watch" }));
    await waitFor(() => expect(screen.getByText("Live Replay")).toBeInTheDocument());
    await waitFor(() => expect(within(screen.getByLabelText("Run status")).getByText("idle")).toBeInTheDocument());
    expect(screen.getByLabelText("Run status")).toBeInTheDocument();
    expect(screen.getByText("Trained policy")).toBeInTheDocument();
    expect(screen.getByText("Pen-directed behavior here comes from learned training weights rather than default instinct.")).toBeInTheDocument();
    expect(screen.getByText(/No-progress guard is active/)).toBeInTheDocument();
    expect(screen.getByText(/rear_pressure, left_flanker/i)).toBeInTheDocument();
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

    await waitFor(() => expect(screen.getByRole("button", { name: "Clear" })).toBeInTheDocument());
    await user.click(screen.getByRole("button", { name: "Clear" }));

    await waitFor(() => expect(screen.getByText("Training cleared. Baseline replay restored")).toBeInTheDocument());
    await user.click(screen.getByRole("tab", { name: "Watch" }));
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

    await userEvent.click(screen.getByRole("button", { name: "Clear" }));

    await waitFor(() => expect(screen.getByText("Training history cleared")).toBeInTheDocument());
    expect(fetchMock).toHaveBeenCalledWith(
      new URL("/api/training/clear", "http://127.0.0.1:8000"),
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

    await userEvent.click(screen.getByRole("tab", { name: "Watch" }));
    await waitFor(() => expect(screen.getByText("Instinct-only dogs do not know the pen. Pen-directed behavior requires training, heuristic expert mode, or a handler target command.")).toBeInTheDocument());
    await userEvent.click(screen.getByRole("button", { name: "Run best model" }));

    await waitFor(() => expect(screen.getByText("Instinct only")).toBeInTheDocument());
    expect(screen.getByText("Instinct-only dogs can chase, circle, avoid diving into the flock, and recover nearby sheep, but they do not know where the pen is.")).toBeInTheDocument();
    expect(fetchMock).toHaveBeenCalledWith(
      new URL("/api/replay/run", "http://127.0.0.1:8000"),
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

    await user.click(screen.getByRole("tab", { name: "Watch" }));
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

  it("starts training with instincts disabled and curriculum stage one by default", async () => {
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
    await userEvent.click(screen.getByRole("button", { name: "Train 50 more" }));

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

    await userEvent.click(screen.getByRole("tab", { name: "Watch" }));
    await waitFor(() => expect(screen.getByRole("button", { name: "End episode" })).toBeEnabled());

    await userEvent.click(screen.getByRole("button", { name: "End episode" }));

    const statusPanel = screen.getByLabelText("Run status");
    expect(within(statusPanel).getByText("3")).toBeInTheDocument();
    expect(within(statusPanel).getAllByText("timeout")).toHaveLength(2);
  });
});

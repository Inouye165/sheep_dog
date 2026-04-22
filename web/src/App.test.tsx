import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { App } from "./App";

const idleTrainingStatus = {
  running: false,
  fast_mode: true,
  enable_instinct_rewards: true,
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
    policy_name: "trained-checkpoint",
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
  policy_name: "trained-checkpoint",
  final_snapshot: {
    step: 3,
    simulated_seconds: 3,
    dogs: [
      { index: 0, x: 1, y: 1, last_action: "right" },
      { index: 1, x: 2, y: 1, last_action: "right" },
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
          { index: 0, x: 1, y: 1, last_action: "right" },
          { index: 1, x: 2, y: 1, last_action: "right" },
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
  vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL) => {
    const path = String(input);
    if (path.includes("/api/training/status")) {
      return new Response(JSON.stringify(idleTrainingStatus), { status: 200 });
    }
    if (path.includes("checkpoint-index.json")) {
      return new Response(JSON.stringify(checkpointIndex), { status: 200 });
    }
    if (path.includes("checkpoint-000000-seed-000011.json")) {
      return new Response(JSON.stringify(replay), { status: 200 });
    }
    return new Response("not found", { status: 404 });
  }));
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("App", () => {
  it("renders the simplified controls and run button", async () => {
    render(<App />);

    await waitFor(() => expect(screen.getByText("Replay")).toBeInTheDocument());
    expect(screen.getByLabelText("Playback controls")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Run current dogs" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Start replay" })).toBeInTheDocument();
    expect(screen.getByText(/Checkpoint 0/)).toBeInTheDocument();
    expect(screen.getByRole("option", { name: "Seed 11" })).toBeInTheDocument();
  });

  it("shows current live run status from loaded replay data", async () => {
    render(<App />);

    await waitFor(() => expect(screen.getByText("Replay")).toBeInTheDocument());
    await waitFor(() => expect(within(screen.getByLabelText("Run status")).getByText("Live run")).toBeInTheDocument());
    expect(screen.getByLabelText("Run status")).toBeInTheDocument();
    expect(screen.getByText("trained-checkpoint")).toBeInTheDocument();
    expect(screen.getByText(/No-progress guard is active/)).toBeInTheDocument();
  });

  it("clears saved training data from the training panel", async () => {
    const fetchMock = vi.mocked(fetch);
    fetchMock.mockImplementation(async (input: RequestInfo | URL) => {
      const path = String(input);
      if (path.includes("/api/training/reset")) {
        return new Response(
          JSON.stringify({
            ...idleTrainingStatus,
            message: "Training history cleared",
          }),
          { status: 200 },
        );
      }
      if (path.includes("/api/training/status")) {
        return new Response(JSON.stringify(idleTrainingStatus), { status: 200 });
      }
      if (path.includes("checkpoint-index.json")) {
        return new Response(JSON.stringify(checkpointIndex), { status: 200 });
      }
      if (path.includes("checkpoint-000000-seed-000011.json")) {
        return new Response(JSON.stringify(replay), { status: 200 });
      }
      return new Response("not found", { status: 404 });
    });

    vi.stubGlobal("confirm", vi.fn(() => true));
    render(<App />);

    await waitFor(() => expect(screen.getByLabelText("Training controls")).toBeInTheDocument());

    await userEvent.click(screen.getByRole("button", { name: "Clear training data" }));

    await waitFor(() => expect(screen.getByText("Training history cleared")).toBeInTheDocument());
    expect(fetchMock).toHaveBeenCalledWith(
      new URL("/api/training/reset", "http://127.0.0.1:8000"),
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
        return new Response(JSON.stringify(idleTrainingStatus), { status: 200 });
      }
      if (path.includes("/api/replay/run")) {
        return new Response(
          JSON.stringify({
            ...replay,
            policy_name: "instinct-only",
            seed: 11,
          }),
          { status: 200 },
        );
      }
      return new Response("not found", { status: 404 });
    });

    render(<App />);

    await waitFor(() => expect(screen.getByText("Use Run current dogs to watch instinct-only or trained behavior.")).toBeInTheDocument());
    await userEvent.click(screen.getByRole("button", { name: "Run current dogs" }));

    await waitFor(() => expect(screen.getByText("instinct-only")).toBeInTheDocument());
    expect(fetchMock).toHaveBeenCalledWith(
      new URL("/api/replay/run", "http://127.0.0.1:8000"),
      expect.objectContaining({ cache: "no-store", method: "POST" }),
    );
  });

  it("starts training with instincts enabled and curriculum stage one by default", async () => {
    const fetchMock = vi.mocked(fetch);
    fetchMock.mockImplementation(async (input: RequestInfo | URL) => {
      const path = String(input);
      if (path.includes("/api/training/start")) {
        return new Response(
          JSON.stringify({
            ...idleTrainingStatus,
            running: true,
            requested_episodes: 5,
            batch_total_episodes: 5,
            enable_instinct_rewards: true,
            curriculum_stage: 1,
            message: "Queued training job",
          }),
          { status: 200 },
        );
      }
      if (path.includes("/api/training/status")) {
        return new Response(JSON.stringify(idleTrainingStatus), { status: 200 });
      }
      if (path.includes("checkpoint-index.json")) {
        return new Response(JSON.stringify(checkpointIndex), { status: 200 });
      }
      if (path.includes("checkpoint-000000-seed-000011.json")) {
        return new Response(JSON.stringify(replay), { status: 200 });
      }
      return new Response("not found", { status: 404 });
    });

    render(<App />);

    await waitFor(() => expect(screen.getByLabelText("Training controls")).toBeInTheDocument());
    await userEvent.click(screen.getByRole("button", { name: "Train 5 more" }));

    const trainingStartCall = fetchMock.mock.calls.find(([request]) =>
      String(request).includes("/api/training/start"),
    );
    expect(trainingStartCall).toBeDefined();
    expect(trainingStartCall?.[1]).toEqual(
      expect.objectContaining({
        cache: "no-store",
        method: "POST",
        body: JSON.stringify({
          episodes: 5,
          fast_mode: true,
          enable_instinct_rewards: true,
          curriculum_stage: 1,
          debug_reward_breakdown: false,
        }),
      }),
    );
  });
});

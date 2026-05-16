import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { FieldView } from "./FieldView";
import { StatusPanel } from "./StatusPanel";
import { dogColor } from "./dogPalette";

describe("FieldView", () => {
  it("renders the pen and agents", () => {
    render(
      <FieldView
        snapshot={{
          step: 1,
          simulated_seconds: 1,
          grid_width: 80,
          grid_height: 60,
          dogs: [
            { index: 0, x: 2, y: 3, last_action: "right", role: "rear_pressure" },
            { index: 1, x: 4, y: 3, last_action: "left", role: "left_flanker" },
          ],
          sheep: [{ index: 0, x: 10, y: 5, penned: false }],
          pen: { origin: { x: 15, y: 2 }, width: 4, height: 4 },
          penned_count: 0,
          average_distance_to_pen: 12,
          flock_spread: 1,
          no_progress_steps: 0,
          terminated: false,
          timeout: false,
          stopped: false,
          success: false,
          status: "running",
        }}
      />,
    );

    expect(screen.getByLabelText("Simulation field")).toBeInTheDocument();
    expect(screen.getByText("Herding field")).toBeInTheDocument();
    expect(screen.getByText("Dog 1 - Rear")).toBeInTheDocument();
    expect(screen.getByText("Dog 2 - Left flank")).toBeInTheDocument();
    expect(screen.getByText("D1")).toBeInTheDocument();
    expect(screen.getByText("Left flank")).toBeInTheDocument();
    expect(dogColor(0)).not.toBe(dogColor(1));
  });

  it("shows the grid size in status", () => {
    render(
      <StatusPanel
        snapshot={{
          step: 1,
          simulated_seconds: 1,
          grid_width: 80,
          grid_height: 60,
          dogs: [{ index: 0, x: 2, y: 3, last_action: "right", role: "rear_pressure" }],
          sheep: [{ index: 0, x: 10, y: 5, penned: false }],
          pen: { origin: { x: 15, y: 2 }, width: 4, height: 4 },
          penned_count: 0,
          average_distance_to_pen: 12,
          flock_spread: 1,
          no_progress_steps: 0,
          terminated: false,
          timeout: false,
          stopped: false,
          success: false,
          status: "running",
        }}
        replay={{
          seed: 11,
          policy_name: "trained_policy",
          trainer_type: "hill_climb",
          policy_type: "linear",
          policy_mode: "trained_policy",
          replay_mode: "trained_linear",
          environment: {
            dogs: 1,
            sheep: 1,
            width: 80,
            height: 60,
            curriculum_stage: 1,
            enable_instinct_rewards: true,
          },
          final_snapshot: {
            step: 1,
            simulated_seconds: 1,
            grid_width: 80,
            grid_height: 60,
            dogs: [{ index: 0, x: 2, y: 3, last_action: "right", role: "rear_pressure" }],
            sheep: [{ index: 0, x: 10, y: 5, penned: false }],
            pen: { origin: { x: 15, y: 2 }, width: 4, height: 4 },
            penned_count: 0,
            average_distance_to_pen: 12,
            flock_spread: 1,
            no_progress_steps: 0,
            terminated: false,
            timeout: false,
            stopped: false,
            success: false,
            status: "running",
          },
          stats: {
            steps: 1,
            simulated_seconds: 1,
            sheep_penned: 0,
            timeout: false,
            terminated: false,
            success: false,
            stopped: false,
            stop_reason: "",
            reward_total: 0,
            no_progress_steps: 0,
            final_avg_distance_to_pen: 12,
            final_flock_spread: 1,
            role_distribution: { rear_pressure: 1 },
            role_switches: 0,
            collector_activations: 0,
            blocker_activations: 0,
            sheep_split_events: 0,
            final_reward_breakdown: {
              progress_to_pen: 0,
              sheep_penned: 0,
              flock_cohesion: 0,
              scatter_penalty: 0,
              time_penalty: 0,
              no_progress_penalty: 0,
              wall_pressure_penalty: 0,
              wait_penalty: 0,
              terminal_success: 0,
              terminal_failure: 0,
              total: 0,
            },
          },
          frames: [],
        }}
        selectedCheckpoint={null}
        selectedCheckpointEpisode={0}
        bestCheckpointEpisode={null}
        selectedSeed={11}
        runState="running"
      />,
    );

    expect(screen.getByText("Grid size")).toBeInTheDocument();
    expect(screen.getByText("80 x 60")).toBeInTheDocument();
    expect(screen.getByText("Role distribution")).toBeInTheDocument();
    expect(screen.getByText("rear_pressure: 1")).toBeInTheDocument();
    expect(screen.getByText("Avg distance to pen")).toBeInTheDocument();
    expect(screen.getByText("Trainer type")).toBeInTheDocument();
    expect(screen.getByText("hill_climb")).toBeInTheDocument();
    expect(screen.getByText("Policy type")).toBeInTheDocument();
    expect(screen.getByText("linear")).toBeInTheDocument();
    expect(screen.getByText("Replay kind")).toBeInTheDocument();
    expect(screen.getByText("Trained linear")).toBeInTheDocument();
    expect(screen.getByText("Dogs / sheep")).toBeInTheDocument();
    expect(screen.getByText("1 / 1")).toBeInTheDocument();
  });

  it("uses fixed field dimensions instead of scaling to group positions", () => {
    render(
      <FieldView
        snapshot={{
          step: 1,
          simulated_seconds: 1,
          field_width: 120,
          field_height: 90,
          dogs: [{ index: 0, x: 2, y: 58, last_action: "right" }],
          sheep: [{ index: 0, x: 10, y: 60, penned: false }],
          pen: { origin: { x: 15, y: 2 }, width: 4, height: 4 },
          penned_count: 0,
          average_distance_to_pen: 12,
          flock_spread: 1,
          no_progress_steps: 0,
          terminated: false,
          timeout: false,
          stopped: false,
          success: false,
          status: "running",
        }}
      />,
    );

    expect(screen.getByRole("img", { name: "Sheepdog simulation map" })).toHaveAttribute("viewBox", "0 0 120 90");
  });
});

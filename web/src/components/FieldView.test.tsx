import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { FieldView } from "./FieldView";
import { StatusPanel } from "./StatusPanel";

describe("FieldView", () => {
  it("renders the pen and agents", () => {
    render(
      <FieldView
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
      />,
    );

    expect(screen.getByLabelText("Simulation field")).toBeInTheDocument();
    expect(screen.getByText("Herding field")).toBeInTheDocument();
    expect(screen.getByText("RP")).toBeInTheDocument();
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
        replay={null}
        evaluation={null}
        rewardBreakdown={null}
        episodeOutcome="running"
        selectedCheckpointEpisode={0}
        selectedSeed={11}
        runState="running"
        trainingStatus={null}
      />,
    );

    expect(screen.getByText("Grid size")).toBeInTheDocument();
    expect(screen.getByText("80 x 60")).toBeInTheDocument();
  });
});

import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { FieldView } from "./FieldView";

describe("FieldView", () => {
  it("renders the pen and agents", () => {
    render(
      <FieldView
        snapshot={{
          step: 1,
          simulated_seconds: 1,
          field_width: 40,
          field_height: 30,
          dogs: [{ index: 0, x: 2, y: 3, last_action: "right" }],
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

import { render, screen, fireEvent } from "@testing-library/react";
import { describe, expect, it, beforeEach } from "vitest";
import { StageConceptBanner, STAGE_CONCEPTS } from "./StageConceptBanner";

describe("StageConceptBanner", () => {
  beforeEach(() => {
    localStorage.clear();
  });

  it("renders the new concept banner for Stage 24 (Wall-Aligned Pen Placement)", () => {
    render(<StageConceptBanner stage={24} />);

    expect(screen.getByRole("region", { name: /New concept for Stage 24/i })).toBeInTheDocument();
    expect(screen.getByText("NEW CONCEPT INTRODUCED")).toBeInTheDocument();
    expect(screen.getByText("Stage 24")).toBeInTheDocument();
    expect(screen.getByText("Pen Moved Away from Corner")).toBeInTheDocument();
    expect(screen.getByText(/Training Pace Notice:/i)).toBeInTheDocument();
    expect(
      screen.getByText(/The pen moves from the field corner to along the wall/i)
    ).toBeInTheDocument();
    expect(
      screen.getByText(/Guiding sheep along an open boundary without corner backstops/i)
    ).toBeInTheDocument();
  });

  it("renders for other milestone stages like Stage 5 and Stage 19", () => {
    const { rerender } = render(<StageConceptBanner stage={5} />);
    expect(screen.getByText("2-Dog Cooperative Herding")).toBeInTheDocument();

    rerender(<StageConceptBanner stage={19} />);
    expect(screen.getByText("Bifurcated Herd (3+3 Split)")).toBeInTheDocument();
  });

  it("returns null for stages without specific concept definitions", () => {
    const { container } = render(<StageConceptBanner stage={999} />);
    expect(container.firstChild).toBeNull();
  });

  it("allows user to dismiss the banner and remembers the dismissal in localStorage", () => {
    const { unmount } = render(<StageConceptBanner stage={24} />);
    expect(screen.getByText("Pen Moved Away from Corner")).toBeInTheDocument();

    const dismissBtn = screen.getByRole("button", { name: /Dismiss banner/i });
    fireEvent.click(dismissBtn);

    // Banner should be removed from DOM
    expect(screen.queryByText("Pen Moved Away from Corner")).not.toBeInTheDocument();
    expect(localStorage.getItem("sheepdog_concept_banner_dismissed_stage_24")).toBe("true");

    // Rerendering should remain dismissed
    unmount();
    render(<StageConceptBanner stage={24} />);
    expect(screen.queryByText("Pen Moved Away from Corner")).not.toBeInTheDocument();
  });
});

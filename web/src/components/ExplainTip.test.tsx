import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { ExplainTip, ProgressPageGuideModal } from "./ExplainTip";

describe("ExplainTip", () => {
  it("is closed by default and opens on click to show Plain English and For Data Scientists", () => {
    render(
      <ExplainTip
        title="Recent Avg Success"
        plain="Shows how often the dogs penned all 6 sheep recently."
        technical={["Window: last N formal 10-seed evaluations.", "Denominator: 10 fixed seeds per eval."]}
      />,
    );

    const trigger = screen.getByRole("button", { name: "Explain Recent Avg Success" });
    expect(trigger.getAttribute("aria-expanded")).toBe("false");
    expect(screen.queryByRole("dialog", { name: "Recent Avg Success" })).toBeNull();
    expect(screen.queryByText(/Shows how often the dogs penned/)).toBeNull();

    fireEvent.click(trigger);

    expect(trigger.getAttribute("aria-expanded")).toBe("true");
    const dialog = screen.getByRole("dialog", { name: "Recent Avg Success" });
    expect(dialog).toBeInTheDocument();
    expect(screen.getByText("Plain English")).toBeInTheDocument();
    expect(screen.getByText("Shows how often the dogs penned all 6 sheep recently.")).toBeInTheDocument();
    expect(screen.getByText("For Data Scientists")).toBeInTheDocument();
    expect(screen.getByText("Window: last N formal 10-seed evaluations.")).toBeInTheDocument();
    expect(screen.getByText("Denominator: 10 fixed seeds per eval.")).toBeInTheDocument();
  });

  it("renders optional buttonText pill and dismisses on close button, Escape, and outside click", () => {
    render(
      <div>
        <ExplainTip
          title="How Stage Timing Works"
          buttonText="How timing works"
          plain="Saves duration whenever a stage promotes."
        />
        <div data-testid="outside">Outside area</div>
      </div>,
    );

    const trigger = screen.getByRole("button", { name: "Explain How Stage Timing Works" });
    expect(trigger.textContent).toContain("How timing works");

    // Open and close via close button
    fireEvent.click(trigger);
    expect(screen.getByRole("dialog", { name: "How Stage Timing Works" })).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Close explanation" }));
    expect(screen.queryByRole("dialog", { name: "How Stage Timing Works" })).toBeNull();

    // Open and close via Escape
    fireEvent.click(trigger);
    expect(screen.getByRole("dialog", { name: "How Stage Timing Works" })).toBeInTheDocument();
    fireEvent.keyDown(document, { key: "Escape" });
    expect(screen.queryByRole("dialog", { name: "How Stage Timing Works" })).toBeNull();

    // Open and close via outside mousedown
    fireEvent.click(trigger);
    expect(screen.getByRole("dialog", { name: "How Stage Timing Works" })).toBeInTheDocument();
    fireEvent.mouseDown(screen.getByTestId("outside"));
    expect(screen.queryByRole("dialog", { name: "How Stage Timing Works" })).toBeNull();
  });
});

describe("ProgressPageGuideModal", () => {
  it("renders nothing when closed and displays summary, 5 tabs, and 6 glossary terms when open", () => {
    const onClose = vi.fn();
    const { rerender } = render(
      <ProgressPageGuideModal isOpen={false} onClose={onClose} stage={4} promotionTargetPct={90} />,
    );
    expect(screen.queryByRole("dialog", { name: "How to read the Progress page" })).toBeNull();

    rerender(
      <ProgressPageGuideModal isOpen={true} onClose={onClose} stage={4} promotionTargetPct={90} />,
    );

    const modal = screen.getByRole("dialog", { name: "How to read the Progress page" });
    expect(modal).toBeInTheDocument();
    expect(screen.getByText(/Stage 4/)).toBeInTheDocument();
    expect(screen.getByText(/90% bar/)).toBeInTheDocument();

    for (const tabName of [
      "Learning Curve",
      "Checkpoints",
      "Seeds & Outcomes",
      "Speed & Timing",
      "Milestones & Video",
    ]) {
      expect(screen.getByText(tabName)).toBeInTheDocument();
    }

    for (const term of [
      "Curriculum Stage",
      "Formal 10-Seed Eval",
      "Promotion Gate",
      "Checkpoint",
      "World Steps vs Policy Steps",
      "Episode Return",
    ]) {
      expect(screen.getByText(term)).toBeInTheDocument();
    }

    fireEvent.click(screen.getByRole("button", { name: "Close page guide" }));
    expect(onClose).toHaveBeenCalledTimes(1);

    fireEvent.keyDown(document, { key: "Escape" });
    expect(onClose).toHaveBeenCalledTimes(2);
  });
});

import { render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { NetworkTopologyViewer } from "./NetworkTopologyViewer";
import {
  buildTopologyModel,
  getAllDenseConnectionCounts,
  getNodeConnectionSummary,
} from "./networkTopologyModel";

describe("network topology model", () => {
  it("uses hidden layer sizes and actor/critic sizes exactly", () => {
    const model = buildTopologyModel({
      inputSize: 54,
      hiddenSizes: [128, 128],
      actionSize: 9,
      maskEnabled: true,
    });

    expect(model.layers.find((layer) => layer.id === "hidden-1")?.nodeCount).toBe(128);
    expect(model.layers.find((layer) => layer.id === "hidden-2")?.nodeCount).toBe(128);
    expect(model.layers.find((layer) => layer.id === "actor")?.nodeCount).toBe(9);
    expect(model.layers.find((layer) => layer.id === "critic")?.nodeCount).toBe(1);
  });

  it("computes total dense connection count", () => {
    const model = buildTopologyModel({
      inputSize: 54,
      hiddenSizes: [128, 128],
      actionSize: 9,
      maskEnabled: true,
    });

    expect(getAllDenseConnectionCounts(model)).toBe(32768);
  });

  it("reports incoming and outgoing counts for a selected node", () => {
    const model = buildTopologyModel({
      inputSize: 54,
      hiddenSizes: [128, 128],
      actionSize: 9,
      maskEnabled: true,
    });

    const selectedHiddenOneSummary = getNodeConnectionSummary(model, {
      layerIndex: 1,
      nodeIndex: 0,
    });

    expect(selectedHiddenOneSummary.incoming).toBe(54);
    expect(selectedHiddenOneSummary.outgoing).toBe(138);
    expect(selectedHiddenOneSummary.fullyConnectedIncoming).toBe(true);
    expect(selectedHiddenOneSummary.fullyConnectedOutgoing).toBe(true);
  });
});

describe("NetworkTopologyViewer", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("shows simulated mode label", () => {
    vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue(null as never);

    render(
      <NetworkTopologyViewer
        config={{
          inputSize: 54,
          hiddenSizes: [128, 128],
          actionSize: 9,
          maskEnabled: true,
        }}
        observationMode="guided"
        maskEnabled
      />,
    );

    expect(screen.getByTestId("simulated-label").textContent?.toLowerCase()).toContain("simulated");
  });
});

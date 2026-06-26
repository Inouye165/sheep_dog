export type LayerKind = "input" | "hidden" | "actor" | "critic" | "mask";

export interface TopologyLayer {
  id: string;
  label: string;
  kind: LayerKind;
  nodeCount: number;
}

export interface TopologyModel {
  layers: TopologyLayer[];
}

export interface NodeRef {
  layerIndex: number;
  nodeIndex: number;
}

export interface NodeConnectionSummary {
  incoming: number;
  outgoing: number;
  fullyConnectedIncoming: boolean;
  fullyConnectedOutgoing: boolean;
}

export interface NetworkTopologyBuildConfig {
  inputSize: number;
  hiddenSizes: number[];
  actionSize: number;
  maskEnabled: boolean;
}

function safePositiveInt(value: number, fallback: number): number {
  return Number.isFinite(value) && value > 0 ? Math.floor(value) : fallback;
}

export function buildTopologyModel(config: NetworkTopologyBuildConfig): TopologyModel {
  const inputSize = safePositiveInt(config.inputSize, 1);
  const hiddenSizes = (config.hiddenSizes ?? [])
    .map((value) => safePositiveInt(value, 0))
    .filter((value) => value > 0);
  const actionSize = safePositiveInt(config.actionSize, 1);

  const layers: TopologyLayer[] = [
    { id: "input", label: "State input", kind: "input", nodeCount: inputSize },
    ...hiddenSizes.map((size, index) => ({
      id: `hidden-${index + 1}`,
      label: `Dense hidden ${index + 1}`,
      kind: "hidden" as const,
      nodeCount: size,
    })),
    { id: "actor", label: "Actor logits", kind: "actor", nodeCount: actionSize },
    { id: "critic", label: "Critic value", kind: "critic", nodeCount: 1 },
  ];

  if (config.maskEnabled) {
    layers.push({ id: "mask", label: "Action mask", kind: "mask", nodeCount: actionSize });
  }

  return { layers };
}

export function getDenseConnectionCount(fromLayer: TopologyLayer, toLayer: TopologyLayer): number {
  return fromLayer.nodeCount * toLayer.nodeCount;
}

function isDenseForwardConnection(from: TopologyLayer, to: TopologyLayer): boolean {
  if (from.kind === "input" && to.kind === "hidden") return true;
  if (from.kind === "hidden" && to.kind === "hidden") return true;
  if (from.kind === "hidden" && to.kind === "actor") return true;
  if (from.kind === "hidden" && to.kind === "critic") return true;
  return false;
}

function isOneToOneForwardConnection(from: TopologyLayer, to: TopologyLayer): boolean {
  return from.kind === "actor" && to.kind === "mask";
}

export function getNodeConnectionSummary(model: TopologyModel, ref: NodeRef): NodeConnectionSummary {
  const layer = model.layers[ref.layerIndex];
  if (!layer) {
    return {
      incoming: 0,
      outgoing: 0,
      fullyConnectedIncoming: false,
      fullyConnectedOutgoing: false,
    };
  }

  let incoming = 0;
  let outgoing = 0;
  let fullyConnectedIncoming = false;
  let fullyConnectedOutgoing = false;

  for (let i = 0; i < model.layers.length; i += 1) {
    if (i === ref.layerIndex) continue;
    const other = model.layers[i];
    if (i < ref.layerIndex) {
      if (isDenseForwardConnection(other, layer)) {
        incoming += other.nodeCount;
        fullyConnectedIncoming = true;
      } else if (isOneToOneForwardConnection(other, layer)) {
        if (ref.nodeIndex < other.nodeCount) {
          incoming += 1;
        }
      }
    }

    if (i > ref.layerIndex) {
      if (isDenseForwardConnection(layer, other)) {
        outgoing += other.nodeCount;
        fullyConnectedOutgoing = true;
      } else if (isOneToOneForwardConnection(layer, other)) {
        if (ref.nodeIndex < other.nodeCount) {
          outgoing += 1;
        }
      }
    }
  }

  return {
    incoming,
    outgoing,
    fullyConnectedIncoming,
    fullyConnectedOutgoing,
  };
}

export function getAllDenseConnectionCounts(model: TopologyModel): number {
  let total = 0;
  for (let i = 0; i < model.layers.length - 1; i += 1) {
    const from = model.layers[i];
    for (let j = i + 1; j < model.layers.length; j += 1) {
      const to = model.layers[j];
      if (isDenseForwardConnection(from, to)) {
        total += getDenseConnectionCount(from, to);
      }
    }
  }
  return total;
}

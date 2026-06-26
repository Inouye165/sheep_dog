import { useEffect, useMemo, useRef, useState } from "react";
import {
  buildTopologyModel,
  getAllDenseConnectionCounts,
  getNodeConnectionSummary,
  type LayerKind,
  type NetworkTopologyBuildConfig,
  type NodeRef,
  type TopologyModel,
} from "./networkTopologyModel";

type DensityMode = "all" | "selected-only" | "bands";

interface NetworkTopologyViewerProps {
  config: NetworkTopologyBuildConfig;
  observationMode: string;
  maskEnabled: boolean;
}

interface LayerLayout {
  x: number;
  nodes: Array<{ x: number; y: number }>;
}

const LAYER_COLORS: Record<LayerKind, string> = {
  input: "#93c5fd",
  hidden: "#c4b5fd",
  actor: "#818cf8",
  critic: "#6ee7b7",
  mask: "#fde047",
};

const BASE_WIDTH = 2200;
const BASE_HEIGHT = 1200;
const NODE_RADIUS = 6;
const HIT_RADIUS = 9;

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

function simulateNodeValue(layerIndex: number, nodeIndex: number, phase: number): number {
  const angle = phase * 6.28 + layerIndex * 0.62 + nodeIndex * 0.12;
  return Math.sin(angle) * 0.55 + Math.cos(angle * 0.7) * 0.35;
}

function activationFromInput(value: number): number {
  // ReLU-style demo activation for visualization only.
  return Math.max(0, value);
}

function isEdgeActive(model: TopologyModel, fromLayerIdx: number, toLayerIdx: number): boolean {
  const from = model.layers[fromLayerIdx];
  const to = model.layers[toLayerIdx];
  if (!from || !to) return false;
  if (from.kind === "input" && to.kind === "hidden") return true;
  if (from.kind === "hidden" && to.kind === "hidden") return true;
  if (from.kind === "hidden" && to.kind === "actor") return true;
  if (from.kind === "hidden" && to.kind === "critic") return true;
  if (from.kind === "actor" && to.kind === "mask") return true;
  return false;
}

function computeLayerLayout(model: TopologyModel): LayerLayout[] {
  const left = 120;
  const right = BASE_WIDTH - 120;
  const top = 110;
  const bottom = BASE_HEIGHT - 90;
  const layerCount = model.layers.length;
  const xSpacing = layerCount > 1 ? (right - left) / (layerCount - 1) : 0;

  return model.layers.map((layer, layerIndex) => {
    const x = left + layerIndex * xSpacing;
    const rows = layer.nodeCount;
    const ySpacing = rows > 1 ? (bottom - top) / (rows - 1) : 0;
    const nodes = Array.from({ length: rows }, (_, nodeIndex) => ({
      x,
      y: rows > 1 ? top + nodeIndex * ySpacing : (top + bottom) / 2,
    }));
    return { x, nodes };
  });
}

export function NetworkTopologyViewer({ config, observationMode, maskEnabled }: NetworkTopologyViewerProps) {
  const model = useMemo(() => buildTopologyModel(config), [config]);
  const layerLayout = useMemo(() => computeLayerLayout(model), [model]);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);
  const [selectedNode, setSelectedNode] = useState<NodeRef | null>(null);
  const [densityMode, setDensityMode] = useState<DensityMode>("bands");
  const [bgOpacity, setBgOpacity] = useState(0.18);
  const [zoom, setZoom] = useState(0.42);
  const [pan, setPan] = useState({ x: 10, y: 24 });
  const [dragging, setDragging] = useState(false);
  const [simPlaying, setSimPlaying] = useState(false);
  const [simSpeed, setSimSpeed] = useState(1);
  const [simPhase, setSimPhase] = useState(0);
  const [showLabels, setShowLabels] = useState(true);
  const dragStartRef = useRef<{ x: number; y: number; px: number; py: number } | null>(null);

  const selectedSummary = selectedNode ? getNodeConnectionSummary(model, selectedNode) : null;
  const denseCount = getAllDenseConnectionCounts(model);

  useEffect(() => {
    if (!simPlaying) return;
    let raf = 0;
    let lastTs = performance.now();
    const tick = (ts: number) => {
      const dt = (ts - lastTs) / 1000;
      lastTs = ts;
      setSimPhase((phase) => (phase + dt * simSpeed * 0.55) % 1);
      raf = window.requestAnimationFrame(tick);
    };
    raf = window.requestAnimationFrame(tick);
    return () => window.cancelAnimationFrame(raf);
  }, [simPlaying, simSpeed]);

  useEffect(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container) return;

    const rect = container.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.max(1, Math.floor(rect.width * dpr));
    canvas.height = Math.max(1, Math.floor(rect.height * dpr));
    canvas.style.width = `${rect.width}px`;
    canvas.style.height = `${rect.height}px`;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    ctx.clearRect(0, 0, rect.width, rect.height);
    ctx.fillStyle = "#030914";
    ctx.fillRect(0, 0, rect.width, rect.height);

    ctx.save();
    ctx.translate(pan.x, pan.y);
    ctx.scale(zoom, zoom);

    const drawNode = (layerIndex: number, nodeIndex: number, alpha = 1) => {
      const layer = model.layers[layerIndex];
      const node = layerLayout[layerIndex]?.nodes[nodeIndex];
      if (!layer || !node) return;
      const raw = simulateNodeValue(layerIndex, nodeIndex, simPhase);
      const activation = activationFromInput(raw);
      const glow = simPlaying ? clamp(activation, 0, 1) : 0.15;
      const color = LAYER_COLORS[layer.kind];
      ctx.beginPath();
      ctx.arc(node.x, node.y, NODE_RADIUS, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.globalAlpha = alpha;
      ctx.fill();
      ctx.lineWidth = 1.4;
      ctx.strokeStyle = "rgba(241,245,249,0.75)";
      ctx.stroke();

      if (glow > 0.05) {
        ctx.beginPath();
        ctx.arc(node.x, node.y, NODE_RADIUS + 4 + glow * 5, 0, Math.PI * 2);
        ctx.strokeStyle = `rgba(147,197,253,${0.12 + glow * 0.22})`;
        ctx.lineWidth = 2;
        ctx.stroke();
      }
      ctx.globalAlpha = 1;
    };

    const selected = selectedNode;

    for (let i = 0; i < model.layers.length; i += 1) {
      for (let j = i + 1; j < model.layers.length; j += 1) {
        if (!isEdgeActive(model, i, j)) continue;
        const fromLayer = model.layers[i];
        const toLayer = model.layers[j];
        const fromNodes = layerLayout[i].nodes;
        const toNodes = layerLayout[j].nodes;
        const edgeAlphaBase = bgOpacity;
        const highlightMode = densityMode === "selected-only";

        if (densityMode === "bands" || zoom < 0.5) {
          const topY = Math.min(fromNodes[0]?.y ?? 0, toNodes[0]?.y ?? 0);
          const bottomY = Math.max(fromNodes[fromNodes.length - 1]?.y ?? 0, toNodes[toNodes.length - 1]?.y ?? 0);
          ctx.beginPath();
          ctx.moveTo(fromNodes[0].x, topY);
          ctx.lineTo(toNodes[0].x, topY);
          ctx.lineTo(toNodes[0].x, bottomY);
          ctx.lineTo(fromNodes[0].x, bottomY);
          ctx.closePath();
          ctx.fillStyle = `rgba(148,163,184,${edgeAlphaBase * 0.35})`;
          ctx.fill();
        } else {
          for (let a = 0; a < fromNodes.length; a += 1) {
            for (let b = 0; b < toNodes.length; b += 1) {
              let active = true;
              if (fromLayer.kind === "actor" && toLayer.kind === "mask") {
                active = a === b;
              }
              if (!active) continue;

              let edgeAlpha = edgeAlphaBase;
              if (highlightMode && selected) {
                const selectedInFrom = selected.layerIndex === i && selected.nodeIndex === a;
                const selectedInTo = selected.layerIndex === j && selected.nodeIndex === b;
                edgeAlpha = selectedInFrom || selectedInTo ? 0.9 : edgeAlphaBase * 0.1;
              }

              const pulse = simPlaying ? (Math.sin(simPhase * 6.28 - i * 0.8 + (a + b) * 0.02) + 1) * 0.5 : 0;
              ctx.strokeStyle = `rgba(147,197,253,${edgeAlpha + pulse * 0.15})`;
              ctx.lineWidth = 0.9;
              ctx.beginPath();
              ctx.moveTo(fromNodes[a].x, fromNodes[a].y);
              ctx.lineTo(toNodes[b].x, toNodes[b].y);
              ctx.stroke();
            }
          }
        }
      }
    }

    for (let li = 0; li < model.layers.length; li += 1) {
      const layer = model.layers[li];
      for (let ni = 0; ni < layer.nodeCount; ni += 1) {
        const dimBySelection = selected && selected.layerIndex !== li && densityMode === "selected-only";
        drawNode(li, ni, dimBySelection ? 0.35 : 1);
      }

      if (showLabels) {
        const labelNode = layerLayout[li].nodes[0];
        ctx.fillStyle = "#e2e8f0";
        ctx.font = "600 26px 'Segoe UI'";
        ctx.fillText(`${layer.label} (${layer.nodeCount})`, labelNode.x - 40, 44);
      }
    }

    if (selected) {
      const node = layerLayout[selected.layerIndex].nodes[selected.nodeIndex];
      ctx.beginPath();
      ctx.arc(node.x, node.y, NODE_RADIUS + 8, 0, Math.PI * 2);
      ctx.strokeStyle = "rgba(251,191,36,0.95)";
      ctx.lineWidth = 2.5;
      ctx.stroke();
    }

    ctx.restore();

    ctx.fillStyle = "rgba(229,238,247,0.9)";
    ctx.font = "500 12px 'Segoe UI'";
    ctx.fillText("Simulated forward pass (not real model weights or activations)", 12, rect.height - 12);
  }, [
    model,
    layerLayout,
    selectedNode,
    densityMode,
    bgOpacity,
    zoom,
    pan,
    simPlaying,
    simPhase,
    showLabels,
  ]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const handleWheel = (event: WheelEvent) => {
      event.preventDefault();
      const delta = -event.deltaY;
      const factor = delta > 0 ? 1.08 : 0.92;
      setZoom((prev) => clamp(prev * factor, 0.18, 2.2));
    };

    canvas.addEventListener("wheel", handleWheel, { passive: false });
    return () => {
      canvas.removeEventListener("wheel", handleWheel);
    };
  }, []);

  const screenToWorld = (x: number, y: number) => ({
    x: (x - pan.x) / zoom,
    y: (y - pan.y) / zoom,
  });

  const pickNode = (worldX: number, worldY: number): NodeRef | null => {
    for (let li = 0; li < layerLayout.length; li += 1) {
      const layer = layerLayout[li];
      for (let ni = 0; ni < layer.nodes.length; ni += 1) {
        const node = layer.nodes[ni];
        const dx = worldX - node.x;
        const dy = worldY - node.y;
        if (dx * dx + dy * dy <= HIT_RADIUS * HIT_RADIUS) {
          return { layerIndex: li, nodeIndex: ni };
        }
      }
    }
    return null;
  };

  const onMouseDown: React.MouseEventHandler<HTMLCanvasElement> = (event) => {
    const rect = event.currentTarget.getBoundingClientRect();
    const world = screenToWorld(event.clientX - rect.left, event.clientY - rect.top);
    const hit = pickNode(world.x, world.y);
    if (hit) {
      setSelectedNode(hit);
      return;
    }

    setDragging(true);
    dragStartRef.current = {
      x: event.clientX,
      y: event.clientY,
      px: pan.x,
      py: pan.y,
    };
  };

  const onMouseMove: React.MouseEventHandler<HTMLCanvasElement> = (event) => {
    if (!dragging || !dragStartRef.current) return;
    const dx = event.clientX - dragStartRef.current.x;
    const dy = event.clientY - dragStartRef.current.y;
    setPan({ x: dragStartRef.current.px + dx, y: dragStartRef.current.py + dy });
  };

  const onMouseUp: React.MouseEventHandler<HTMLCanvasElement> = () => {
    setDragging(false);
    dragStartRef.current = null;
  };

  const selectedLayer = selectedNode ? model.layers[selectedNode.layerIndex] : null;
  const selectedValues = selectedNode
    ? {
        input: simulateNodeValue(selectedNode.layerIndex, selectedNode.nodeIndex, simPhase),
        output: activationFromInput(
          simulateNodeValue(selectedNode.layerIndex, selectedNode.nodeIndex, simPhase),
        ),
      }
    : null;

  const hidden1 = model.layers.find((layer) => layer.id === "hidden-1")?.nodeCount ?? 0;
  const hidden2 = model.layers.find((layer) => layer.id === "hidden-2")?.nodeCount ?? 0;
  const actorCount = model.layers.find((layer) => layer.id === "actor")?.nodeCount ?? 0;
  const criticCount = model.layers.find((layer) => layer.id === "critic")?.nodeCount ?? 0;

  return (
    <div className="topology-viewer">
      <div className="topology-toolbar">
        <div className="topology-toolbar__group">
          <span data-testid="hidden-1-count">Hidden L1: {hidden1}</span>
          <span data-testid="hidden-2-count">Hidden L2: {hidden2}</span>
          <span data-testid="actor-count">Actor: {actorCount}</span>
          <span data-testid="critic-count">Critic: {criticCount}</span>
        </div>
        <div className="topology-toolbar__group">
          <label>
            Density
            <select value={densityMode} onChange={(e) => setDensityMode(e.target.value as DensityMode)}>
              <option value="all">Show all connections</option>
              <option value="selected-only">Selected-node connections only</option>
              <option value="bands">Connection bands only</option>
            </select>
          </label>
          <label>
            Background opacity
            <input
              type="range"
              min={0.05}
              max={0.6}
              step={0.01}
              value={bgOpacity}
              onChange={(e) => setBgOpacity(Number(e.target.value))}
            />
          </label>
          <label>
            Zoom
            <input
              type="range"
              min={0.18}
              max={2.2}
              step={0.01}
              value={zoom}
              onChange={(e) => setZoom(Number(e.target.value))}
            />
          </label>
        </div>

        <div className="topology-toolbar__group">
          <span className="pill pill--warning">Simulated forward pass</span>
          <button type="button" onClick={() => setSimPlaying((v) => !v)}>
            {simPlaying ? "Pause" : "Play"}
          </button>
          <button
            type="button"
            onClick={() => {
              setSimPhase(0);
              setSimPlaying(false);
            }}
          >
            Reset
          </button>
          <label>
            Speed
            <input
              type="range"
              min={0.25}
              max={3}
              step={0.05}
              value={simSpeed}
              onChange={(e) => setSimSpeed(Number(e.target.value))}
            />
          </label>
          <label className="topology-checkbox">
            <input type="checkbox" checked={showLabels} onChange={(e) => setShowLabels(e.target.checked)} />
            Labels
          </label>
        </div>
      </div>

      <div className="topology-canvas-shell">
        <div className="topology-canvas" ref={containerRef}>
          <canvas
            ref={canvasRef}
            onMouseDown={onMouseDown}
            onMouseMove={onMouseMove}
            onMouseUp={onMouseUp}
            onMouseLeave={onMouseUp}
          />
        </div>

        <aside className="topology-inspector" aria-label="Node inspector">
          <h3>Inspector</h3>
          {selectedNode && selectedLayer && selectedSummary && selectedValues ? (
            <dl>
              <dt>Layer</dt>
              <dd>{selectedLayer.label}</dd>
              <dt>Node index</dt>
              <dd>{selectedNode.nodeIndex}</dd>
              <dt>Incoming connections</dt>
              <dd>{selectedSummary.incoming.toLocaleString()}</dd>
              <dt>Outgoing connections</dt>
              <dd>{selectedSummary.outgoing.toLocaleString()}</dd>
              <dt>Layer connectivity</dt>
              <dd>
                {selectedSummary.fullyConnectedIncoming || selectedSummary.fullyConnectedOutgoing
                  ? "Dense / fully connected"
                  : "Pointwise / constrained"}
              </dd>
              <dt>Simulated input</dt>
              <dd>{selectedValues.input.toFixed(4)}</dd>
              <dt>Simulated activation/output</dt>
              <dd>{selectedValues.output.toFixed(4)}</dd>
              <dt>Formula</dt>
              <dd>y = ReLU(sum(w_i * x_i + b))</dd>
            </dl>
          ) : (
            <p>Select a node in the canvas to inspect connections and simulated activations.</p>
          )}

          <div className="topology-inspector__meta">
            <p data-testid="simulated-label">Simulated only: not model internal activations or weights.</p>
            <p>Observation mode: <strong>{observationMode}</strong></p>
            <p>Action mask: <strong>{maskEnabled ? "enabled" : "disabled"}</strong></p>
            <p>Total dense connections: <strong>{denseCount.toLocaleString()}</strong></p>
          </div>
        </aside>
      </div>
    </div>
  );
}

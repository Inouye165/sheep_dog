import type {
  ReplayFrame,
  ReplaySnapshot,
  TemporalStepEvidence,
} from "../state/types";
import { dogColor } from "./dogPalette";

export interface FieldOverlayConfig {
  recordedTrails: boolean;
  estimatedTargets: boolean;
  derivedWaypoints: boolean;
  descriptiveGeometry: boolean;
}

export interface SelectedFieldEntity {
  type: "dog" | "sheep";
  index: number;
}

interface FieldViewProps {
  snapshot: ReplaySnapshot | null;
  temporalStepEvidence?: TemporalStepEvidence | null;
  recentFrames?: ReplayFrame[];
  selectedEntity?: SelectedFieldEntity | null;
  onSelectEntity?: (entity: SelectedFieldEntity) => void;
  highlightedDogIndices?: number[];
  highlightedSheepIndices?: number[];
  sheepVision?: number;
  overlays?: FieldOverlayConfig;
  onToggleOverlay?: (key: keyof FieldOverlayConfig) => void;
  showSheepIndices?: boolean;
}

type Side = "top" | "bottom" | "left" | "right";

const ROLE_LABELS: Record<string, string> = {
  rear_pressure: "Rear",
  left_flanker: "Left flank",
  right_flanker: "Right flank",
  collector: "Collector",
  blocker: "Blocker",
};

const DEFAULT_SHEEP_COLOR = "#f8fafc";
const PENNED_SHEEP_STROKE = "#86efac";
const ACTIVE_SHEEP_STROKE = "#cbd5e1";

function fenceSegments(snapshot: ReplaySnapshot): Array<{ side: Side; x1: number; y1: number; x2: number; y2: number }> {
  const { pen } = snapshot;
  if (!pen || !pen.origin) return [];
  const opening = pen.opening ?? "left";
  const ox = pen.origin.x ?? 0;
  const oy = pen.origin.y ?? 0;
  const right = ox + (pen.width ?? 10);
  const bottom = oy + (pen.height ?? 10);
  const all: Array<{ side: Side; x1: number; y1: number; x2: number; y2: number }> = [
    { side: "top", x1: ox, y1: oy, x2: right, y2: oy },
    { side: "bottom", x1: ox, y1: bottom, x2: right, y2: bottom },
    { side: "left", x1: ox, y1: oy, x2: ox, y2: bottom },
    { side: "right", x1: right, y1: oy, x2: right, y2: bottom },
  ];
  return all.filter((segment) => segment.side !== opening);
}

function gateEndpoints(snapshot: ReplaySnapshot): {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  midX: number;
  midY: number;
  side: Side;
} | null {
  const { pen } = snapshot;
  if (!pen || !pen.origin) return null;
  const opening: Side = (pen.opening as Side | undefined) ?? "left";
  const ox = pen.origin.x ?? 0;
  const oy = pen.origin.y ?? 0;
  const right = ox + (pen.width ?? 10);
  const bottom = oy + (pen.height ?? 10);
  if (opening === "top") {
    return { x1: ox, y1: oy, x2: right, y2: oy, midX: (ox + right) / 2, midY: oy, side: "top" };
  }
  if (opening === "bottom") {
    return { x1: ox, y1: bottom, x2: right, y2: bottom, midX: (ox + right) / 2, midY: bottom, side: "bottom" };
  }
  if (opening === "right") {
    return { x1: right, y1: oy, x2: right, y2: bottom, midX: right, midY: (oy + bottom) / 2, side: "right" };
  }
  return { x1: ox, y1: oy, x2: ox, y2: bottom, midX: ox, midY: (oy + bottom) / 2, side: "left" };
}

export function FieldView({
  snapshot,
  temporalStepEvidence = null,
  recentFrames = [],
  selectedEntity = null,
  onSelectEntity,
  highlightedDogIndices = [],
  highlightedSheepIndices = [],
  sheepVision = 10,
  overlays,
  onToggleOverlay,
  showSheepIndices = false,
}: FieldViewProps) {
  const baseWidth = snapshot?.grid_width ?? snapshot?.field_width ?? 40;
  const baseHeight = snapshot?.grid_height ?? snapshot?.field_height ?? 30;
  const width = snapshot ? Math.max(baseWidth, 40) : 40;
  const height = snapshot ? Math.max(baseHeight, 30) : 30;
  const fences = snapshot ? fenceSegments(snapshot) : [];
  const gate = snapshot ? gateEndpoints(snapshot) : null;
  const densityScale = Math.max(width / 40, height / 30, 1);
  const dogRadius = 0.48 * densityScale;
  const sheepRadius = 0.42 * densityScale;
  const fenceStroke = 0.32 * densityScale;
  const gateRadius = 0.18 * densityScale;
  const penStroke = 0.08 * densityScale;
  const fontSize = 0.46 * densityScale;
  const transitionStyle = {
    transition: "transform 150ms linear, cx 150ms linear, cy 150ms linear, x 150ms linear, y 150ms linear",
  };
  const roleTagHeight = 0.54 * densityScale;
  const roleTagRadius = 0.12 * densityScale;
  const roleTagStroke = 0.04 * densityScale;
  const roleLabelSize = 0.28 * densityScale;
  const hasDiagnosticMode = Boolean(temporalStepEvidence || overlays || onSelectEntity || showSheepIndices);

  const sheepStepMap = new Map(
    (temporalStepEvidence?.sheep ?? []).map((s) => [s.sheep_index, s]),
  );

  return (
    <section className="field-card" aria-label="Simulation field">
      <div className="field-card__header" style={{ alignItems: "center", flexWrap: "wrap", gap: "0.5rem" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "0.75rem" }}>
          <div style={{ display: "flex", flexDirection: "column" }}>
            <div style={{ display: "flex", alignItems: "center", gap: "0.4rem", marginBottom: "0.15rem" }}>
              <span
                style={{
                  display: "inline-block",
                  width: "7px",
                  height: "7px",
                  borderRadius: "50%",
                  background: snapshot ? "#4ade80" : "#94a3b8",
                  boxShadow: snapshot ? "0 0 8px rgba(74, 222, 128, 0.7)" : "none",
                }}
              />
              <span className="eyebrow" style={{ fontSize: "0.68rem", letterSpacing: "0.08em", fontWeight: 700, color: snapshot ? "#4ade80" : "var(--muted)", textTransform: "uppercase" }}>
                {hasDiagnosticMode ? "Recorded Evidence Replay" : "Live Replay"}
              </span>
            </div>
            <h2 style={{ fontSize: "1.25rem", fontWeight: 700, margin: 0, letterSpacing: "-0.01em" }}>Herding field</h2>
          </div>
        </div>
        {snapshot ? (
          <div className="field-card__meta" style={{ display: "flex", alignItems: "center", gap: "0.45rem", flexWrap: "wrap" }}>
            <span style={{
              background: "rgba(148, 163, 184, 0.08)",
              border: "1px solid rgba(148, 163, 184, 0.16)",
              borderRadius: "999px",
              padding: "0.2rem 0.65rem",
              fontWeight: 600,
              fontSize: "0.76rem",
              color: "#e2e8f0"
            }}>
              Step {snapshot.step ?? 0}
            </span>
            {temporalStepEvidence ? (
              <span
                style={{
                  background: "rgba(56, 189, 248, 0.12)",
                  border: "1px solid rgba(56, 189, 248, 0.35)",
                  borderRadius: "999px",
                  padding: "0.2rem 0.65rem",
                  fontWeight: 600,
                  fontSize: "0.75rem",
                  color: "#bae6fd",
                }}
              >
                Unpenned: {temporalStepEvidence.unpenned_count} | Penned: {temporalStepEvidence.penned_count}
              </span>
            ) : null}
            {snapshot.simulated_seconds !== undefined && snapshot.simulated_seconds !== null ? (
              <span style={{
                background: "rgba(148, 163, 184, 0.08)",
                border: "1px solid rgba(148, 163, 184, 0.16)",
                borderRadius: "999px",
                padding: "0.2rem 0.65rem",
                fontWeight: 600,
                fontSize: "0.76rem",
                color: "#e2e8f0"
              }}>
                {snapshot.simulated_seconds.toFixed(0)}s simulated
              </span>
            ) : null}
          </div>
        ) : null}
      </div>

      {/* Compact Diagnostic Overlays Toggle Bar (Above Field Stage) */}
      {overlays && onToggleOverlay ? (
        <div
          aria-label="Field evidence overlay controls"
          style={{
            display: "flex",
            flexWrap: "wrap",
            alignItems: "center",
            gap: "0.28rem",
            padding: "0.25rem 0.5rem",
            background: "rgba(15, 23, 42, 0.65)",
            border: "1px solid rgba(148, 163, 184, 0.18)",
            borderRadius: "0.45rem",
            fontSize: "0.68rem",
          }}
        >
          <span
            style={{
              fontWeight: 700,
              color: "#94a3b8",
              textTransform: "uppercase",
              fontSize: "0.63rem",
              letterSpacing: "0.04em",
            }}
          >
            Overlays:
          </span>
          <button
            type="button"
            aria-pressed={overlays.recordedTrails}
            onClick={() => onToggleOverlay("recordedTrails")}
            style={{
              padding: "0.12rem 0.42rem",
              borderRadius: "0.35rem",
              border: overlays.recordedTrails
                ? "1px solid #38bdf8"
                : "1px solid rgba(148, 163, 184, 0.25)",
              background: overlays.recordedTrails
                ? "rgba(56, 189, 248, 0.16)"
                : "rgba(30, 41, 59, 0.6)",
              color: overlays.recordedTrails ? "#e0f2fe" : "#94a3b8",
              fontSize: "0.66rem",
              fontWeight: 600,
              cursor: "pointer",
            }}
          >
            {overlays.recordedTrails ? "● " : "○ "}Recorded Movement & Trails
          </button>
          <button
            type="button"
            aria-pressed={overlays.estimatedTargets}
            onClick={() => onToggleOverlay("estimatedTargets")}
            style={{
              padding: "0.12rem 0.42rem",
              borderRadius: "0.35rem",
              border: overlays.estimatedTargets
                ? "1px solid #f59e0b"
                : "1px solid rgba(148, 163, 184, 0.25)",
              background: overlays.estimatedTargets
                ? "rgba(245, 158, 11, 0.16)"
                : "rgba(30, 41, 59, 0.6)",
              color: overlays.estimatedTargets ? "#fef3c7" : "#94a3b8",
              fontSize: "0.66rem",
              fontWeight: 600,
              cursor: "pointer",
            }}
          >
            {overlays.estimatedTargets ? "◆ " : "◇ "}Estimated/Recorded Guidance Targets
          </button>
          <button
            type="button"
            aria-pressed={overlays.derivedWaypoints}
            onClick={() => onToggleOverlay("derivedWaypoints")}
            style={{
              padding: "0.12rem 0.42rem",
              borderRadius: "0.35rem",
              border: overlays.derivedWaypoints
                ? "1px solid #2dd4bf"
                : "1px solid rgba(148, 163, 184, 0.25)",
              background: overlays.derivedWaypoints
                ? "rgba(45, 212, 191, 0.16)"
                : "rgba(30, 41, 59, 0.6)",
              color: overlays.derivedWaypoints ? "#ccfbf1" : "#94a3b8",
              fontSize: "0.66rem",
              fontWeight: 600,
              cursor: "pointer",
            }}
          >
            {overlays.derivedWaypoints ? "■ " : "□ "}Derived Navigation Waypoints
          </button>
          <button
            type="button"
            aria-pressed={overlays.descriptiveGeometry}
            onClick={() => onToggleOverlay("descriptiveGeometry")}
            style={{
              padding: "0.12rem 0.42rem",
              borderRadius: "0.35rem",
              border: overlays.descriptiveGeometry
                ? "1px solid #c084fc"
                : "1px solid rgba(148, 163, 184, 0.25)",
              background: overlays.descriptiveGeometry
                ? "rgba(192, 132, 252, 0.16)"
                : "rgba(30, 41, 59, 0.6)",
              color: overlays.descriptiveGeometry ? "#f3e8ff" : "#94a3b8",
              fontSize: "0.66rem",
              fontWeight: 600,
              cursor: "pointer",
            }}
          >
            {overlays.descriptiveGeometry ? "◎ " : "○ "}Descriptive Vision & Waypoint-Side Geometry
          </button>
        </div>
      ) : null}

      {!hasDiagnosticMode && (snapshot?.dogs?.length || snapshot?.sheep?.length) ? (
        <div style={{
          display: "flex",
          flexWrap: "wrap",
          alignItems: "center",
          justifyContent: "space-between",
          gap: "0.4rem 1rem",
          padding: "0.35rem 0.65rem",
          background: "rgba(15, 23, 42, 0.45)",
          border: "1px solid rgba(148, 163, 184, 0.12)",
          borderRadius: "0.6rem",
          fontSize: "0.76rem"
        }}>
          {(snapshot?.dogs ?? []).length ? (
            <div className="field-card__meta" aria-label="Dog legend" style={{ display: "flex", alignItems: "center", gap: "0.8rem", flexWrap: "wrap" }}>
              {(snapshot?.dogs ?? []).map((dog) => {
                const isSelected = selectedEntity?.type === "dog" && selectedEntity.index === dog.index;
                return (
                  <span
                    key={`legend-${dog.index}`}
                    onClick={onSelectEntity ? () => onSelectEntity({ type: "dog", index: dog.index }) : undefined}
                    style={{
                      display: "inline-flex",
                      alignItems: "center",
                      color: isSelected ? "#fef08a" : "#cbd5e1",
                      fontWeight: isSelected ? 700 : 400,
                      cursor: onSelectEntity ? "pointer" : "default",
                    }}
                  >
                    <span
                      aria-hidden="true"
                      style={{
                        display: "inline-block",
                        width: "0.75rem",
                        height: "0.75rem",
                        borderRadius: "999px",
                        marginRight: "0.35rem",
                        backgroundColor: dogColor(dog.index),
                        border: isSelected ? "2px solid #fde047" : "1px solid rgba(255,255,255,0.7)",
                        boxShadow: `0 0 6px ${dogColor(dog.index)}40`,
                      }}
                    />
                    {`Dog ${dog.index + 1}${dog.role ? ` - ${ROLE_LABELS[dog.role] ?? dog.role}` : ""}`}
                  </span>
                );
              })}
            </div>
          ) : null}

          {(snapshot?.sheep ?? []).length ? (
            <div className="field-card__meta" aria-label="Sheep legend" style={{ display: "flex", alignItems: "center", gap: "0.6rem", flexWrap: "wrap" }}>
              {(() => {
                const groups = new Map<string, { color: string; count: number }>();
                for (const sheep of snapshot?.sheep ?? []) {
                  const key = sheep.personality ?? "obedient";
                  const color = sheep.color ?? DEFAULT_SHEEP_COLOR;
                  const existing = groups.get(key);
                  if (existing) {
                    existing.count += 1;
                  } else {
                    groups.set(key, { color, count: 1 });
                  }
                }
                return Array.from(groups.entries()).map(([personality, info]) => (
                  <span key={`sheep-legend-${personality}`} style={{ display: "inline-flex", alignItems: "center", color: "#cbd5e1" }}>
                    <span
                      aria-hidden="true"
                      style={{
                        display: "inline-block",
                        width: "0.7rem",
                        height: "0.7rem",
                        borderRadius: "999px",
                        marginRight: "0.35rem",
                        backgroundColor: info.color,
                        border: `1px solid ${ACTIVE_SHEEP_STROKE}`,
                      }}
                    />
                    <span>{`${personality} (${info.count})`}</span>
                  </span>
                ));
              })()}
            </div>
          ) : null}
        </div>
      ) : null}

      <div className="field-stage">
        {snapshot ? (
          <svg className="field-stage__svg" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Sheepdog simulation map">
            <defs>
              <linearGradient id="fieldGradient" x1="0" x2="1" y1="0" y2="1">
                <stop offset="0%" stopColor="#0f172a" />
                <stop offset="100%" stopColor="#122a32" />
              </linearGradient>
              <pattern id="grid" width="1" height="1" patternUnits="userSpaceOnUse">
                <path d="M 1 0 L 0 0 0 1" fill="none" stroke="rgba(255,255,255,0.05)" strokeWidth="0.06" />
              </pattern>
            </defs>
            <rect width="100%" height="100%" fill="url(#fieldGradient)" />
            <rect width="100%" height="100%" fill="url(#grid)" />
            {snapshot.pen?.origin ? (
              <rect
                x={snapshot.pen.origin.x ?? 0}
                y={snapshot.pen.origin.y ?? 0}
                width={snapshot.pen.width ?? 10}
                height={snapshot.pen.height ?? 10}
                rx={0.4 * densityScale}
                fill="rgba(244, 197, 66, 0.12)"
                stroke="rgba(244, 197, 66, 0.55)"
                strokeDasharray={`${0.4 * densityScale} ${0.3 * densityScale}`}
                strokeWidth={penStroke}
              />
            ) : null}
            {fences.map((segment) => (
              <line
                key={segment.side}
                x1={segment.x1}
                y1={segment.y1}
                x2={segment.x2}
                y2={segment.y2}
                stroke="#f4c542"
                strokeWidth={fenceStroke}
                strokeLinecap="round"
              />
            ))}
            {/* Gate markers on the open side */}
            {gate ? (
              <g aria-label={`Pen entrance (${gate.side})`}>
                <line
                  x1={gate.x1}
                  y1={gate.y1}
                  x2={gate.x2}
                  y2={gate.y2}
                  stroke="#4ade80"
                  strokeWidth={0.14 * densityScale}
                  strokeDasharray={`${0.28 * densityScale} ${0.22 * densityScale}`}
                />
                <circle cx={gate.x1} cy={gate.y1} r={gateRadius} fill="#fde68a" />
                <circle cx={gate.x2} cy={gate.y2} r={gateRadius} fill="#fde68a" />
                {hasDiagnosticMode ? (
                  <text
                    x={gate.midX}
                    y={gate.side === "bottom" ? gate.midY + 0.55 * densityScale : gate.midY - 0.35 * densityScale}
                    textAnchor="middle"
                    fill="#86efac"
                    fontSize={0.28 * densityScale}
                    fontWeight={700}
                  >
                    GATE ({gate.side.toUpperCase()})
                  </text>
                ) : null}
              </g>
            ) : null}

            {/* 1. Recorded Trails Overlay */}
            {overlays?.recordedTrails && recentFrames.length > 1 ? (
              <g aria-label="Recorded recent movement trails">
                {(snapshot.sheep ?? []).map((sheep) => {
                  const pts = recentFrames
                    .map((fr) => (fr.snapshot?.sheep ?? []).find((s) => s.index === sheep.index))
                    .filter((s): s is NonNullable<typeof s> => Boolean(s))
                    .map((s) => `${s.x + 0.5},${s.y + 0.5}`)
                    .join(" ");
                  if (!pts) return null;
                  return (
                    <polyline
                      key={`sheep-trail-${sheep.index}`}
                      points={pts}
                      fill="none"
                      stroke={sheep.penned ? "rgba(134, 239, 172, 0.45)" : "rgba(45, 212, 191, 0.7)"}
                      strokeWidth={0.11 * densityScale}
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    />
                  );
                })}
                {(snapshot.dogs ?? []).map((dog) => {
                  const pts = recentFrames
                    .map((fr) => (fr.snapshot?.dogs ?? []).find((d) => d.index === dog.index))
                    .filter((d): d is NonNullable<typeof d> => Boolean(d))
                    .map((d) => `${d.x + 0.5},${d.y + 0.5}`)
                    .join(" ");
                  if (!pts) return null;
                  return (
                    <polyline
                      key={`dog-trail-${dog.index}`}
                      points={pts}
                      fill="none"
                      stroke={dogColor(dog.index)}
                      strokeOpacity={0.72}
                      strokeWidth={0.13 * densityScale}
                      strokeDasharray={`${0.25 * densityScale} ${0.14 * densityScale}`}
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    />
                  );
                })}
              </g>
            ) : null}

            {/* 2. Descriptive Geometry Overlay (Sheep Vision Radius & Waypoint-Side Alignment) */}
            {overlays?.descriptiveGeometry && temporalStepEvidence ? (
              <g aria-label="Descriptive sheep vision and alignment overlay">
                {temporalStepEvidence.sheep
                  .filter((s) => !s.penned && (selectedEntity?.type !== "sheep" || selectedEntity.index === s.sheep_index))
                  .map((s) => (
                    <circle
                      key={`vision-${s.sheep_index}`}
                      cx={s.post_step_position.x + 0.5}
                      cy={s.post_step_position.y + 0.5}
                      r={sheepVision}
                      fill="rgba(192, 132, 252, 0.03)"
                      stroke="rgba(192, 132, 252, 0.35)"
                      strokeWidth={0.07 * densityScale}
                      strokeDasharray={`${0.35 * densityScale} ${0.25 * densityScale}`}
                    />
                  ))}
                {temporalStepEvidence.dogs
                  .filter((d) => d.on_gate_or_waypoint_side)
                  .map((d) => (
                    <circle
                      key={`wp-side-dog-${d.dog_index}`}
                      cx={d.post_step_position.x + 0.5}
                      cy={d.post_step_position.y + 0.5}
                      r={dogRadius * 1.45}
                      fill="none"
                      stroke="#f43f5e"
                      strokeWidth={0.09 * densityScale}
                      strokeDasharray={`${0.18 * densityScale} ${0.14 * densityScale}`}
                    >
                      <title>{`D${d.dog_index}: Descriptive geometry indicates dog is on waypoint/gate side within sheep vision (not proven physical blockage)`}</title>
                    </circle>
                  ))}
              </g>
            ) : null}

            {/* 3. Derived Navigation Waypoints Overlay */}
            {overlays?.derivedWaypoints && temporalStepEvidence ? (
              <g aria-label="Derived navigation waypoints overlay">
                {temporalStepEvidence.sheep
                  .filter((s) => !s.penned)
                  .map((s) => {
                    const sx = s.post_step_position.x + 0.5;
                    const sy = s.post_step_position.y + 0.5;
                    const wx = s.waypoint.x + 0.5;
                    const wy = s.waypoint.y + 0.5;
                    const boxSize = 0.36 * densityScale;
                    return (
                      <g key={`wp-sheep-${s.sheep_index}`}>
                        <line
                          x1={sx}
                          y1={sy}
                          x2={wx}
                          y2={wy}
                          stroke="#2dd4bf"
                          strokeWidth={0.09 * densityScale}
                          strokeDasharray={`${0.14 * densityScale} ${0.14 * densityScale}`}
                        />
                        <rect
                          x={wx - boxSize / 2}
                          y={wy - boxSize / 2}
                          width={boxSize}
                          height={boxSize}
                          fill="rgba(45, 212, 191, 0.25)"
                          stroke="#2dd4bf"
                          strokeWidth={0.07 * densityScale}
                        />
                        <text
                          x={wx}
                          y={wy - 0.28 * densityScale}
                          textAnchor="middle"
                          fill="#5eead4"
                          fontSize={0.24 * densityScale}
                          fontWeight={700}
                        >
                          {`W-S${s.sheep_index}`}
                        </text>
                        <title>{`Derived BFS waypoint for S${s.sheep_index}: (${s.waypoint.x}, ${s.waypoint.y}), route distance ${s.route_distance.toFixed(1)} cells`}</title>
                      </g>
                    );
                  })}
              </g>
            ) : null}

            {/* 4. Estimated/Recorded Guidance Targets Overlay */}
            {overlays?.estimatedTargets && temporalStepEvidence ? (
              <g aria-label="Guidance targets overlay">
                {temporalStepEvidence.dogs.map((d) => {
                  const dx = d.post_step_position.x + 0.5;
                  const dy = d.post_step_position.y + 0.5;
                  const tx = d.target.x + 0.5;
                  const ty = d.target.y + 0.5;
                  const dSize = 0.32 * densityScale;
                  const isRecordedTarget = d.guidance_certainty.startsWith("recorded_");
                  const certTag = isRecordedTarget ? "rec" : "est";
                  return (
                    <g key={`target-dog-${d.dog_index}`}>
                      <line
                        x1={dx}
                        y1={dy}
                        x2={tx}
                        y2={ty}
                        stroke={dogColor(d.dog_index)}
                        strokeOpacity={0.85}
                        strokeWidth={0.09 * densityScale}
                        strokeDasharray={`${0.26 * densityScale} ${0.18 * densityScale}`}
                      />
                      <polygon
                        points={`${tx},${ty - dSize} ${tx + dSize},${ty} ${tx},${ty + dSize} ${tx - dSize},${ty}`}
                        fill="rgba(245, 158, 11, 0.28)"
                        stroke={dogColor(d.dog_index)}
                        strokeWidth={0.08 * densityScale}
                      />
                      <text
                        x={tx}
                        y={ty - 0.38 * densityScale}
                        textAnchor="middle"
                        fill="#fde68a"
                        fontSize={0.24 * densityScale}
                        fontWeight={700}
                      >
                        {`T-D${d.dog_index}[${certTag}]`}
                      </text>
                      <title>{`D${d.dog_index} guidance target (${d.target.x}, ${d.target.y}) [${d.guidance_certainty}]`}</title>
                    </g>
                  );
                })}
              </g>
            ) : null}

            {(snapshot.sheep ?? []).map((sheep) => {
              const isSelected = selectedEntity?.type === "sheep" && selectedEntity.index === sheep.index;
              const isHighlighted = highlightedSheepIndices.includes(sheep.index);
              const sStep = sheepStepMap.get(sheep.index);
              const sheepLabel = showSheepIndices ? `S${sheep.index}` : "S";
              return (
                <g
                  key={`sheep-${sheep.index}`}
                  transform={`translate(${sheep.x + 0.5}, ${sheep.y + 0.5})`}
                  onClick={onSelectEntity ? () => onSelectEntity({ type: "sheep", index: sheep.index }) : undefined}
                  style={{ cursor: onSelectEntity ? "pointer" : "default" }}
                  aria-label={showSheepIndices ? `Sheep S${sheep.index}` : undefined}
                >
                  {isSelected || isHighlighted ? (
                    <circle
                      r={sheepRadius * 1.48}
                      fill="none"
                      stroke={isSelected ? "#fde047" : "#38bdf8"}
                      strokeWidth={0.11 * densityScale}
                      strokeDasharray={isSelected ? undefined : `${0.16 * densityScale} ${0.12 * densityScale}`}
                    />
                  ) : null}
                  <circle
                    r={sheepRadius}
                    fill={sheep.color ?? DEFAULT_SHEEP_COLOR}
                    fillOpacity={sheep.penned ? 0.85 : 1}
                    stroke={sheep.penned ? PENNED_SHEEP_STROKE : ACTIVE_SHEEP_STROKE}
                    strokeWidth={sheep.penned ? penStroke * 1.8 : penStroke}
                    style={transitionStyle}
                  />
                  <text
                    textAnchor="middle"
                    dominantBaseline="central"
                    fill="#0f172a"
                    fontSize={showSheepIndices ? fontSize * 0.78 : fontSize}
                    fontWeight={700}
                    style={transitionStyle}
                  >
                    {sheepLabel}
                  </text>
                  {sheep.penned && showSheepIndices ? (
                    <text
                      x={sheepRadius * 0.85}
                      y={-sheepRadius * 0.85}
                      textAnchor="middle"
                      fill="#4ade80"
                      fontSize={0.28 * densityScale}
                      fontWeight={800}
                    >
                      ✓
                    </text>
                  ) : null}
                  <title>{`Sheep S${sheep.index} (Dog/Sheep 1-based #${sheep.index + 1})${sheep.personality ? ` - ${sheep.personality}` : ""}${sheep.penned ? " (penned)" : sStep ? ` (unpenned, route ${sStep.route_distance.toFixed(1)} cells)` : ""}`}</title>
                </g>
              );
            })}
            {(snapshot.dogs ?? []).map((dog) => {
              const roleLabel = ROLE_LABELS[dog.role ?? ""] ?? dog.role ?? "Dog";
              const roleTagWidth = Math.max(1.9, roleLabel.length * 0.28) * densityScale;
              const roleTagX = -roleTagWidth / 2;
              const isSelected = selectedEntity?.type === "dog" && selectedEntity.index === dog.index;
              const isHighlighted = highlightedDogIndices.includes(dog.index);

              // Smart anti-collision placement when dogs are clustered together
              const isClustered = (snapshot.dogs ?? []).some(
                (other) => other.index !== dog.index && Math.hypot(other.x - dog.x, other.y - dog.y) < 2.5
              );
              // Alternate positioning: even dog indexes below, odd dog indexes above (if space permits)
              const placeAbove = isClustered && (dog.index % 2 === 1) && (dog.y > 2);
              const dynamicTagY = placeAbove
                ? -(0.66 * densityScale) - roleTagHeight
                : (0.66 * densityScale);
              const dynamicLabelY = dynamicTagY + (roleTagHeight / 2);

              return (
              <g
                key={`dog-${dog.index}`}
                transform={`translate(${dog.x + 0.5}, ${dog.y + 0.5})`}
                aria-label={`Dog ${dog.index + 1}`}
                onClick={onSelectEntity ? () => onSelectEntity({ type: "dog", index: dog.index }) : undefined}
                style={{ cursor: onSelectEntity ? "pointer" : "default" }}
              >
                {isSelected || isHighlighted ? (
                  <circle
                    r={dogRadius * 1.48}
                    fill="none"
                    stroke={isSelected ? "#fde047" : "#38bdf8"}
                    strokeWidth={0.12 * densityScale}
                    strokeDasharray={isSelected ? undefined : `${0.18 * densityScale} ${0.12 * densityScale}`}
                  />
                ) : null}
                <circle r={dogRadius} fill={dogColor(dog.index)} stroke="rgba(255,255,255,0.88)" strokeWidth={penStroke} style={transitionStyle} />
                <text
                  textAnchor="middle"
                  dominantBaseline="central"
                  fill="#f8fafc"
                  fontSize={fontSize}
                  fontWeight={800}
                  stroke="rgba(15, 23, 42, 0.72)"
                  strokeWidth={0.08 * densityScale}
                  paintOrder="stroke"
                  style={transitionStyle}
                >
                  {`D${dog.index + 1}`}
                </text>
                {dog.role ? (
                  <>
                    <rect
                      x={roleTagX}
                      y={dynamicTagY}
                      width={roleTagWidth}
                      height={roleTagHeight}
                      rx={roleTagRadius}
                      fill="rgba(15, 23, 42, 0.92)"
                      stroke="rgba(248, 250, 252, 0.5)"
                      strokeWidth={roleTagStroke}
                    />
                    <text
                      textAnchor="middle"
                      dominantBaseline="central"
                      y={dynamicLabelY}
                      fill="#f8fafc"
                      fontSize={roleLabelSize}
                      fontWeight={700}
                    >
                      {roleLabel}
                    </text>
                    <title>{`Dog ${dog.index + 1} (D${dog.index}): ${dog.role.replaceAll("_", " ")}`}</title>
                  </>
                ) : null}
              </g>
            );})}
          </svg>
        ) : (
          <div className="field-stage__empty">
            <p>No replay loaded yet.</p>
            <p>Use Run current dogs to watch the current dog team.</p>
            <p>Instinct-only dogs do not know the pen. Pen-directed behavior requires training, heuristic expert mode, or a handler target command.</p>
          </div>
        )}
      </div>

      {/* Compact Legend & Overlay Provenance Footer (Below Field Stage in Diagnostic Mode) */}
      {hasDiagnosticMode && (snapshot?.dogs?.length || snapshot?.sheep?.length) ? (
        <div
          style={{
            display: "flex",
            flexWrap: "wrap",
            alignItems: "center",
            justifyContent: "space-between",
            gap: "0.25rem 0.65rem",
            padding: "0.24rem 0.5rem",
            background: "rgba(15, 23, 42, 0.5)",
            border: "1px solid rgba(148, 163, 184, 0.14)",
            borderRadius: "0.45rem",
            fontSize: "0.66rem",
            color: "#94a3b8",
          }}
        >
          <div
            className="field-card__meta"
            aria-label="Dog legend"
            style={{ display: "flex", alignItems: "center", gap: "0.55rem", flexWrap: "wrap" }}
          >
            {(snapshot?.dogs ?? []).map((dog) => {
              const isSelected = selectedEntity?.type === "dog" && selectedEntity.index === dog.index;
              return (
                <span
                  key={`legend-${dog.index}`}
                  onClick={onSelectEntity ? () => onSelectEntity({ type: "dog", index: dog.index }) : undefined}
                  style={{
                    display: "inline-flex",
                    alignItems: "center",
                    color: isSelected ? "#fef08a" : "#cbd5e1",
                    fontWeight: isSelected ? 700 : 500,
                    cursor: onSelectEntity ? "pointer" : "default",
                  }}
                >
                  <span
                    aria-hidden="true"
                    style={{
                      display: "inline-block",
                      width: "0.58rem",
                      height: "0.58rem",
                      borderRadius: "999px",
                      marginRight: "0.25rem",
                      backgroundColor: dogColor(dog.index),
                      border: isSelected ? "2px solid #fde047" : "1px solid rgba(255,255,255,0.7)",
                    }}
                  />
                  {`D${dog.index} (${DogLabel(dog.index)}${dog.role ? `: ${ROLE_LABELS[dog.role] ?? dog.role}` : ""})`}
                </span>
              );
            })}
          </div>
          <span>
            ◆ Guidance: <code>{temporalStepEvidence?.guidance_certainty ?? "estimated/recorded"}</code> · ■ BFS route waypoints · ◎ Vision ({sheepVision}c, descriptive)
          </span>
        </div>
      ) : null}
    </section>
  );
}

function DogLabel(index: number): string {
  return `Dog ${index + 1}`;
}

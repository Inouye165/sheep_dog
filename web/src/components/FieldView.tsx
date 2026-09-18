import type { ReplaySnapshot } from "../state/types";
import { dogColor } from "./dogPalette";

interface FieldViewProps {
  snapshot: ReplaySnapshot | null;
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

export function FieldView({ snapshot }: FieldViewProps) {
  const baseWidth = snapshot?.grid_width ?? snapshot?.field_width ?? 40;
  const baseHeight = snapshot?.grid_height ?? snapshot?.field_height ?? 30;
  const width = snapshot ? Math.max(baseWidth, 40) : 40;
  const height = snapshot ? Math.max(baseHeight, 30) : 30;
  const fences = snapshot ? fenceSegments(snapshot) : [];
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
  const roleTagY = 0.66 * densityScale;
  const roleTagHeight = 0.54 * densityScale;
  const roleTagRadius = 0.12 * densityScale;
  const roleTagStroke = 0.04 * densityScale;
  const roleLabelSize = 0.28 * densityScale;

  return (
    <section className="field-card" aria-label="Simulation field">
      <div className="field-card__header" style={{ alignItems: "center" }}>
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
                Live Replay
              </span>
            </div>
            <h2 style={{ fontSize: "1.25rem", fontWeight: 700, margin: 0, letterSpacing: "-0.01em" }}>Herding field</h2>
          </div>
        </div>
        {snapshot ? (
          <div className="field-card__meta" style={{ display: "flex", alignItems: "center", gap: "0.45rem" }}>
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

      {/* Unified Legend Strip */}
      {(snapshot?.dogs?.length || snapshot?.sheep?.length) ? (
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
          {snapshot?.dogs.length ? (
            <div className="field-card__meta" aria-label="Dog legend" style={{ display: "flex", alignItems: "center", gap: "0.8rem", flexWrap: "wrap" }}>
              {snapshot.dogs.map((dog) => (
                <span key={`legend-${dog.index}`} style={{ display: "inline-flex", alignItems: "center", color: "#cbd5e1" }}>
                  <span
                    aria-hidden="true"
                    style={{
                      display: "inline-block",
                      width: "0.75rem",
                      height: "0.75rem",
                      borderRadius: "999px",
                      marginRight: "0.35rem",
                      backgroundColor: dogColor(dog.index),
                      border: "1px solid rgba(255,255,255,0.7)",
                      boxShadow: `0 0 6px ${dogColor(dog.index)}40`,
                    }}
                  />
                  {`Dog ${dog.index + 1}${dog.role ? ` - ${ROLE_LABELS[dog.role] ?? dog.role}` : ""}`}
                </span>
              ))}
            </div>
          ) : null}

          {snapshot?.sheep.length ? (
            <div className="field-card__meta" aria-label="Sheep legend" style={{ display: "flex", alignItems: "center", gap: "0.6rem", flexWrap: "wrap" }}>
              {(() => {
                const groups = new Map<string, { color: string; count: number }>();
                for (const sheep of snapshot.sheep) {
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
            <rect
              x={snapshot.pen.origin.x}
              y={snapshot.pen.origin.y}
              width={snapshot.pen.width}
              height={snapshot.pen.height}
              rx={0.4 * densityScale}
              fill="rgba(244, 197, 66, 0.12)"
              stroke="rgba(244, 197, 66, 0.55)"
              strokeDasharray={`${0.4 * densityScale} ${0.3 * densityScale}`}
              strokeWidth={penStroke}
            />
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
            {snapshot.pen.opening === "left" || snapshot.pen.opening === undefined ? (
              <>
                <circle cx={snapshot.pen.origin.x} cy={snapshot.pen.origin.y} r={gateRadius} fill="#fde68a" />
                <circle cx={snapshot.pen.origin.x} cy={snapshot.pen.origin.y + snapshot.pen.height} r={gateRadius} fill="#fde68a" />
              </>
            ) : null}
            {snapshot.sheep.map((sheep) => (
              <g key={`sheep-${sheep.index}`} transform={`translate(${sheep.x + 0.5}, ${sheep.y + 0.5})`}>
                <circle
                  r={sheepRadius}
                  fill={sheep.color ?? DEFAULT_SHEEP_COLOR}
                  fillOpacity={sheep.penned ? 0.85 : 1}
                  stroke={sheep.penned ? PENNED_SHEEP_STROKE : ACTIVE_SHEEP_STROKE}
                  strokeWidth={penStroke}
                  style={transitionStyle}
                />
                <text textAnchor="middle" dominantBaseline="central" fill="#0f172a" fontSize={fontSize} fontWeight={700} style={transitionStyle}>
                  S
                </text>
                <title>{`Sheep ${sheep.index + 1}${sheep.personality ? ` - ${sheep.personality}` : ""}${sheep.penned ? " (penned)" : ""}`}</title>
              </g>
            ))}
            {snapshot.dogs.map((dog) => {
              const roleLabel = ROLE_LABELS[dog.role ?? ""] ?? dog.role ?? "Dog";
              const roleTagWidth = Math.max(1.9, roleLabel.length * 0.28) * densityScale;
              const roleTagX = -roleTagWidth / 2;

              // Smart anti-collision placement when dogs are clustered together
              const isClustered = snapshot.dogs.some(
                (other) => other.index !== dog.index && Math.hypot(other.x - dog.x, other.y - dog.y) < 2.5
              );
              // Alternate positioning: even dog indexes below, odd dog indexes above (if space permits)
              const placeAbove = isClustered && (dog.index % 2 === 1) && (dog.y > 2);
              const dynamicTagY = placeAbove
                ? -(0.66 * densityScale) - roleTagHeight
                : (0.66 * densityScale);
              const dynamicLabelY = dynamicTagY + (roleTagHeight / 2);

              return (
              <g key={`dog-${dog.index}`} transform={`translate(${dog.x + 0.5}, ${dog.y + 0.5})`} aria-label={`Dog ${dog.index + 1}`}>
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
                    <title>{`Dog ${dog.index + 1}: ${dog.role.replaceAll("_", " ")}`}</title>
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
    </section>
  );
}

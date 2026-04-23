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

function fenceSegments(snapshot: ReplaySnapshot): Array<{ side: Side; x1: number; y1: number; x2: number; y2: number }> {
  const { pen } = snapshot;
  const opening = pen.opening ?? "left";
  const ox = pen.origin.x;
  const oy = pen.origin.y;
  const right = ox + pen.width;
  const bottom = oy + pen.height;
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
  const roleLabelY = 0.94 * densityScale;
  const roleLabelSize = 0.28 * densityScale;

  return (
    <section className="field-card" aria-label="Simulation field">
      <div className="field-card__header">
        <div>
          <p className="eyebrow">Live Replay</p>
          <h2>Herding field</h2>
        </div>
        {snapshot ? (
          <div className="field-card__meta">
            <span>Step {snapshot.step}</span>
            <span>{snapshot.simulated_seconds.toFixed(0)}s simulated</span>
          </div>
        ) : null}
      </div>
      {snapshot?.dogs.length ? (
        <div className="field-card__meta" aria-label="Dog legend">
          {snapshot.dogs.map((dog) => (
            <span key={`legend-${dog.index}`}>
              <span
                aria-hidden="true"
                style={{
                  display: "inline-block",
                  width: "0.85rem",
                  height: "0.85rem",
                  borderRadius: "999px",
                  marginRight: "0.4rem",
                  verticalAlign: "middle",
                  backgroundColor: dogColor(dog.index),
                  border: "1px solid rgba(255,255,255,0.55)",
                }}
              />
              {`Dog ${dog.index + 1}${dog.role ? ` - ${ROLE_LABELS[dog.role] ?? dog.role}` : ""}`}
            </span>
          ))}
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
                <circle r={sheepRadius} fill={sheep.penned ? "#bbf7d0" : "#f8fafc"} stroke="#cbd5e1" strokeWidth={penStroke} style={transitionStyle} />
                <text textAnchor="middle" dominantBaseline="central" fill="#0f172a" fontSize={fontSize} fontWeight={700} style={transitionStyle}>
                  S
                </text>
              </g>
            ))}
            {snapshot.dogs.map((dog) => {
              const roleLabel = ROLE_LABELS[dog.role ?? ""] ?? dog.role ?? "Dog";
              const roleTagWidth = Math.max(1.9, roleLabel.length * 0.28) * densityScale;
              const roleTagX = -roleTagWidth / 2;
              return (
              <g key={`dog-${dog.index}`} transform={`translate(${dog.x + 0.5}, ${dog.y + 0.5})`} aria-label={`Dog ${dog.index + 1}`}>
                <circle r={dogRadius} fill={dogColor(dog.index)} stroke="rgba(255,255,255,0.82)" strokeWidth={penStroke} style={transitionStyle} />
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
                      y={roleTagY}
                      width={roleTagWidth}
                      height={roleTagHeight}
                      rx={roleTagRadius}
                      fill="rgba(15, 23, 42, 0.88)"
                      stroke="rgba(248, 250, 252, 0.6)"
                      strokeWidth={roleTagStroke}
                    />
                    <text
                      textAnchor="middle"
                      dominantBaseline="middle"
                      y={roleLabelY}
                      fill="#f8fafc"
                      fontSize={roleLabelSize}
                      fontWeight={800}
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

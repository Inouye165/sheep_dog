import type { ReplaySnapshot } from "../state/types";

interface FieldViewProps {
  snapshot: ReplaySnapshot | null;
}

type Side = "top" | "bottom" | "left" | "right";

const ROLE_LABELS: Record<string, string> = {
  rear_pressure: "RP",
  left_flanker: "LF",
  right_flanker: "RF",
  collector: "CO",
  blocker: "BL",
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
  const gridWidth = snapshot?.grid_width ?? 40;
  const gridHeight = snapshot?.grid_height ?? 30;
  const gridScale = Math.max(gridWidth / 40, gridHeight / 30, 1);
  const margin = 4 * gridScale;
  const width = snapshot ? gridWidth + margin : 40;
  const height = snapshot ? gridHeight + margin : 30;
  const fences = snapshot ? fenceSegments(snapshot) : [];
  const sheepRadius = 0.42 * gridScale;
  const sheepStroke = 0.08 * gridScale;
  const sheepLabelSize = 0.35 * gridScale;
  const dogRadius = 0.48 * gridScale;
  const dogStroke = 0.08 * gridScale;
  const dogLabelSize = 0.34 * gridScale;
  const fenceStroke = 0.32 * gridScale;
  const gateRadius = 0.18 * gridScale;
  const penDash = `${0.4 * gridScale} ${0.3 * gridScale}`;
  const penStroke = 0.08 * gridScale;
  const roleTagX = -0.62 * gridScale;
  const roleTagY = 0.52 * gridScale;
  const roleTagWidth = 1.24 * gridScale;
  const roleTagHeight = 0.38 * gridScale;
  const roleTagRadius = 0.12 * gridScale;
  const roleTagStroke = 0.04 * gridScale;
  const roleLabelY = 0.71 * gridScale;
  const roleLabelSize = 0.22 * gridScale;

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
              rx="0.4"
              fill="rgba(244, 197, 66, 0.12)"
              stroke="rgba(244, 197, 66, 0.55)"
              strokeDasharray={penDash}
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
                <circle r={sheepRadius} fill={sheep.penned ? "#bbf7d0" : "#f8fafc"} stroke="#cbd5e1" strokeWidth={sheepStroke} />
                <text textAnchor="middle" dominantBaseline="central" fill="#0f172a" fontSize={sheepLabelSize} fontWeight={700}>
                  S
                </text>
              </g>
            ))}
            {snapshot.dogs.map((dog) => (
              <g key={`dog-${dog.index}`} transform={`translate(${dog.x + 0.5}, ${dog.y + 0.5})`}>
                <circle r={dogRadius} fill="#1d4ed8" stroke="#dbeafe" strokeWidth={dogStroke} />
                <text textAnchor="middle" dominantBaseline="central" fill="#eff6ff" fontSize={dogLabelSize} fontWeight={700}>
                  D
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
                      stroke="rgba(219, 234, 254, 0.5)"
                      strokeWidth={roleTagStroke}
                    />
                    <text
                      textAnchor="middle"
                      dominantBaseline="middle"
                      y={roleLabelY}
                      fill="#dbeafe"
                      fontSize={roleLabelSize}
                      fontWeight={700}
                    >
                      {ROLE_LABELS[dog.role] ?? dog.role.slice(0, 2).toUpperCase()}
                    </text>
                    <title>{`Dog ${dog.index}: ${dog.role.replaceAll("_", " ")}`}</title>
                  </>
                ) : null}
              </g>
            ))}
          </svg>
        ) : (
          <div className="field-stage__empty">
            <p>No replay loaded yet.</p>
            <p>Run the Python export command to populate the viewer.</p>
          </div>
        )}
      </div>
    </section>
  );
}

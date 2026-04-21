import type { ReplaySnapshot } from "../state/types";

interface FieldViewProps {
  snapshot: ReplaySnapshot | null;
}

type Side = "top" | "bottom" | "left" | "right";

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
  const width = snapshot ? snapshot.pen.origin.x + snapshot.pen.width + 4 : 40;
  const height = snapshot ? Math.max(snapshot.pen.origin.y + snapshot.pen.height + 4, 30) : 30;
  const fences = snapshot ? fenceSegments(snapshot) : [];

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
              strokeDasharray="0.4 0.3"
              strokeWidth="0.08"
            />
            {fences.map((segment) => (
              <line
                key={segment.side}
                x1={segment.x1}
                y1={segment.y1}
                x2={segment.x2}
                y2={segment.y2}
                stroke="#f4c542"
                strokeWidth="0.32"
                strokeLinecap="round"
              />
            ))}
            {/* Gate markers on the open side */}
            {snapshot.pen.opening === "left" || snapshot.pen.opening === undefined ? (
              <>
                <circle cx={snapshot.pen.origin.x} cy={snapshot.pen.origin.y} r="0.18" fill="#fde68a" />
                <circle cx={snapshot.pen.origin.x} cy={snapshot.pen.origin.y + snapshot.pen.height} r="0.18" fill="#fde68a" />
              </>
            ) : null}
            {snapshot.sheep.map((sheep) => (
              <g key={`sheep-${sheep.index}`} transform={`translate(${sheep.x + 0.5}, ${sheep.y + 0.5})`}>
                <circle r="0.42" fill={sheep.penned ? "#bbf7d0" : "#f8fafc"} stroke="#cbd5e1" strokeWidth="0.08" />
                <text textAnchor="middle" dominantBaseline="central" fill="#0f172a" fontSize="0.35" fontWeight={700}>
                  S
                </text>
              </g>
            ))}
            {snapshot.dogs.map((dog) => (
              <g key={`dog-${dog.index}`} transform={`translate(${dog.x + 0.5}, ${dog.y + 0.5})`}>
                <circle r="0.48" fill="#1d4ed8" stroke="#dbeafe" strokeWidth="0.08" />
                <text textAnchor="middle" dominantBaseline="central" fill="#eff6ff" fontSize="0.34" fontWeight={700}>
                  D
                </text>
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

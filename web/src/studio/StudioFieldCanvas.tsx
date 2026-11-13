/* eslint-disable react-refresh/only-export-components */
import React from "react";
import type { ReplayBundle, ReplaySnapshot } from "../state/types";
import { dogColor } from "../components/dogPalette";

type Side = "top" | "bottom" | "left" | "right";

export type AnyReplayPayload = ReplayBundle & {
  grid?: { width?: number; height?: number; pen?: ReplaySnapshot["pen"] };
  summary?: {
    success?: boolean;
    steps?: number;
    total_reward?: number;
    stop_reason?: string;
  };
};

export const ROLE_LABELS: Record<string, string> = {
  rear_pressure: "Rear",
  left_flanker: "Left flank",
  right_flanker: "Right flank",
  collector: "Collector",
  blocker: "Blocker",
};

const DEFAULT_SHEEP_COLOR = "#f8fafc";

export function snapshotFromReplay(
  replay: AnyReplayPayload | null | undefined,
  frameIndex: number
): ReplaySnapshot | null {
  if (!replay) return null;
  const frames = replay.frames ?? [];
  if (frames.length === 0) {
    return replay.final_snapshot ?? null;
  }
  const clampedIndex = Math.max(0, Math.min(frameIndex, frames.length - 1));
  const frame = frames[clampedIndex] as any;
  if (!frame) return replay.final_snapshot ?? null;

  // Standard ReplayBundle has frame.snapshot
  if (frame.snapshot && frame.snapshot.dogs && frame.snapshot.sheep) {
    return frame.snapshot as ReplaySnapshot;
  }

  // Fallback for flat frame format
  const width = replay.environment?.width ?? replay.grid?.width ?? 40;
  const height = replay.environment?.height ?? replay.grid?.height ?? 30;
  const pen = frame.pen ??
    replay.final_snapshot?.pen ??
    replay.grid?.pen ?? {
      origin: { x: 30, y: 10 },
      width: 8,
      height: 10,
      opening: "left",
    };

  return {
    step: frame.step ?? clampedIndex,
    simulated_seconds: frame.simulated_seconds ?? clampedIndex * 0.5,
    grid_width: width,
    grid_height: height,
    field_width: width,
    field_height: height,
    dogs: frame.dogs ?? [],
    sheep: frame.sheep ?? [],
    pen,
    penned_count:
      frame.penned_count ??
      ( Array.isArray(frame.sheep) ? frame.sheep.filter((s: any) => s.penned).length : 0 ),
    average_distance_to_pen: frame.average_distance_to_pen ?? 0,
    flock_spread: frame.flock_spread ?? 0,
    no_progress_steps: frame.no_progress_steps ?? 0,
    terminated: Boolean(frame.done ?? frame.terminated),
    timeout: Boolean(frame.timeout),
    stopped: Boolean(frame.stopped),
    success: Boolean(frame.success ?? replay.stats?.success ?? replay.summary?.success),
    status: frame.stop_reason ?? replay.stats?.stop_reason ?? replay.summary?.stop_reason ?? "active",
  };
}

function fenceSegments(
  snapshot: ReplaySnapshot
): Array<{ side: Side; x1: number; y1: number; x2: number; y2: number }> {
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

export interface StudioFieldCanvasProps {
  snapshot: ReplaySnapshot | null;
  compact?: boolean;
  showRoles?: boolean;
  showVectors?: boolean;
  smoothTransition?: boolean;
  uniqueId?: string;
}

export const StudioFieldCanvas: React.FC<StudioFieldCanvasProps> = ({
  snapshot,
  compact = false,
  showRoles = true,
  showVectors = false,
  smoothTransition = false,
  uniqueId = "main",
}) => {
  if (!snapshot) {
    return (
      <div className="sd-empty-state">
        <div style={{ fontSize: "1.5rem", opacity: 0.5 }}>🐕</div>
        <div style={{ fontWeight: 600, color: "var(--sd-text-secondary)" }}>
          No Episode Replay Loaded
        </div>
        <div style={{ maxWidth: 340 }}>
          Click <strong>Run Live Episode</strong> below or select a checkpoint / evaluation seed to inspect field behavior.
        </div>
      </div>
    );
  }

  const baseWidth = snapshot.grid_width ?? snapshot.field_width ?? 40;
  const baseHeight = snapshot.grid_height ?? snapshot.field_height ?? 30;
  const width = Math.max(baseWidth, 40);
  const height = Math.max(baseHeight, 30);
  const fences = fenceSegments(snapshot);
  const densityScale = Math.max(width / 40, height / 30, 1);

  const sizeMultiplier = compact ? 1.28 : 1.08;
  const dogScale = 0.62 * densityScale * sizeMultiplier;
  const sheepScale = 0.52 * densityScale * sizeMultiplier;
  const fenceStroke = 0.32 * densityScale;
  const gateRadius = 0.2 * densityScale;
  const penStroke = 0.08 * densityScale * sizeMultiplier;
  const roleTagHeight = 0.52 * densityScale;
  const roleTagRadius = 0.14 * densityScale;
  const roleTagStroke = 0.04 * densityScale;
  const roleLabelSize = 0.27 * densityScale;

  const transitionStyle: React.CSSProperties | undefined = smoothTransition
    ? {
        transition:
          "transform 120ms linear, cx 120ms linear, cy 120ms linear, x 120ms linear, y 120ms linear",
      }
    : undefined;

  // Compute flock center of unpenned sheep for optional herding vector visualization
  const activeSheep = (snapshot.sheep ?? []).filter((s) => !s.penned);
  const flockCenter =
    activeSheep.length > 0
      ? {
          x: activeSheep.reduce((acc, s) => acc + s.x + 0.5, 0) / activeSheep.length,
          y: activeSheep.reduce((acc, s) => acc + s.y + 0.5, 0) / activeSheep.length,
        }
      : null;

  const gateCenter = snapshot.pen?.origin
    ? {
        x: snapshot.pen.origin.x,
        y: snapshot.pen.origin.y + (snapshot.pen.height ?? 10) / 2,
      }
    : null;

  const gradId = `sdFieldGrad-${uniqueId}`;
  const gridId = `sdFieldGrid-${uniqueId}`;
  const shadowId = `sdEntityShadow-${uniqueId}`;

  return (
    <svg
      className="sd-field-svg"
      viewBox={`0 0 ${width} ${height}`}
      role="img"
      aria-label="Sheepdog simulation map"
    >
      <defs>
        <linearGradient id={gradId} x1="0" x2="1" y1="0" y2="1">
          <stop offset="0%" stopColor="#09131c" />
          <stop offset="55%" stopColor="#0d1e28" />
          <stop offset="100%" stopColor="#0f252e" />
        </linearGradient>
        <pattern id={gridId} width="2" height="2" patternUnits="userSpaceOnUse">
          <path
            d="M 2 0 L 0 0 0 2"
            fill="none"
            stroke="rgba(148, 163, 184, 0.06)"
            strokeWidth="0.06"
          />
        </pattern>
        <filter id={shadowId} x="-35%" y="-35%" width="170%" height="170%">
          <feDropShadow dx="0" dy={0.08 * densityScale} stdDeviation={0.09 * densityScale} floodColor="#000000" floodOpacity="0.65" />
        </filter>
      </defs>

      <rect width="100%" height="100%" fill={`url(#${gradId})`} />
      <rect width="100%" height="100%" fill={`url(#${gridId})`} />

      {/* Pen Target Zone */}
      {snapshot.pen?.origin && (
        <rect
          x={snapshot.pen.origin.x}
          y={snapshot.pen.origin.y}
          width={snapshot.pen.width}
          height={snapshot.pen.height}
          rx={0.4 * densityScale}
          fill="rgba(245, 158, 11, 0.11)"
          stroke="rgba(245, 158, 11, 0.5)"
          strokeDasharray={`${0.4 * densityScale} ${0.3 * densityScale}`}
          strokeWidth={penStroke}
        />
      )}

      {/* Optional Herding Axis Vector (Flock Center -> Gate Center) */}
      {showVectors && flockCenter && gateCenter && (
        <g opacity={0.65}>
          <line
            x1={flockCenter.x}
            y1={flockCenter.y}
            x2={gateCenter.x}
            y2={gateCenter.y}
            stroke="#38bdf8"
            strokeWidth={0.1 * densityScale}
            strokeDasharray={`${0.35 * densityScale} ${0.25 * densityScale}`}
          />
          <circle
            cx={flockCenter.x}
            cy={flockCenter.y}
            r={0.28 * densityScale}
            fill="none"
            stroke="#38bdf8"
            strokeWidth={0.08 * densityScale}
          />
        </g>
      )}

      {/* Solid Fence Walls */}
      {fences.map((segment) => (
        <line
          key={segment.side}
          x1={segment.x1}
          y1={segment.y1}
          x2={segment.x2}
          y2={segment.y2}
          stroke="#f59e0b"
          strokeWidth={fenceStroke}
          strokeLinecap="round"
        />
      ))}

      {/* Gate Post Markers */}
      {(snapshot.pen?.opening === "left" || snapshot.pen?.opening === undefined) &&
        snapshot.pen?.origin && (
          <>
            <circle
              cx={snapshot.pen.origin.x}
              cy={snapshot.pen.origin.y}
              r={gateRadius}
              fill="#fde68a"
            />
            <circle
              cx={snapshot.pen.origin.x}
              cy={snapshot.pen.origin.y + snapshot.pen.height}
              r={gateRadius}
              fill="#fde68a"
            />
          </>
        )}

      {/* Sheep Entities — Fluffy White Wool Cloud + Dark Face/Ears */}
      {(snapshot.sheep ?? []).map((sheep) => {
        const s = sheepScale;
        const woolFill = sheep.penned ? "#ecfdf5" : "#f8fafc";
        const woolStroke = sheep.penned ? "#10b981" : "#64748b";
        const traitColor =
          sheep.color && sheep.color !== DEFAULT_SHEEP_COLOR ? sheep.color : null;

        return (
          <g
            key={`sheep-${sheep.index}`}
            transform={`translate(${sheep.x + 0.5}, ${sheep.y + 0.5})`}
            style={transitionStyle}
            filter={`url(#${shadowId})`}
          >
            {/* Penned Emerald Halo */}
            {sheep.penned && (
              <circle
                r={s * 1.45}
                fill="rgba(16, 185, 129, 0.16)"
                stroke="#34d399"
                strokeWidth={penStroke * 0.85}
              />
            )}

            {/* Little Dark Sheep Legs */}
            <line
              x1={-s * 0.42}
              y1={s * 0.45}
              x2={-s * 0.42}
              y2={s * 0.88}
              stroke="#0f172a"
              strokeWidth={s * 0.22}
              strokeLinecap="round"
            />
            <line
              x1={s * 0.35}
              y1={s * 0.45}
              x2={s * 0.35}
              y2={s * 0.88}
              stroke="#0f172a"
              strokeWidth={s * 0.22}
              strokeLinecap="round"
            />

            {/* Fluffy Scalloped Wool Coat */}
            <g fill={woolFill} stroke={woolStroke} strokeWidth={s * 0.11}>
              <circle cx={-s * 0.42} cy={-s * 0.08} r={s * 0.46} />
              <circle cx={s * 0.36} cy={-s * 0.08} r={s * 0.44} />
              <circle cx={-s * 0.16} cy={-s * 0.38} r={s * 0.44} />
              <circle cx={s * 0.18} cy={-s * 0.34} r={s * 0.4} />
              <circle cx={-s * 0.2} cy={s * 0.24} r={s * 0.42} />
              <circle cx={s * 0.22} cy={s * 0.22} r={s * 0.42} />
            </g>
            {/* Inner Wool Body (covers internal puff seams) */}
            <ellipse
              cx={-s * 0.03}
              cy={-s * 0.04}
              rx={s * 0.62}
              ry={s * 0.48}
              fill={woolFill}
            />

            {/* Subtle Personality Ear-Tag / Collar Dot (preserves trait info without looking like a dog) */}
            {traitColor && (
              <circle
                cx={-s * 0.38}
                cy={-s * 0.06}
                r={s * 0.22}
                fill={traitColor}
                stroke="#0f172a"
                strokeWidth={s * 0.07}
              />
            )}

            {/* Dark Sheep Head & Floppy Side Ears (positioned on right/front of body) */}
            <g transform={`translate(${s * 0.52}, ${-s * 0.12})`}>
              {/* Top & Bottom Horizontal Sheep Ears */}
              <ellipse
                cx={-s * 0.05}
                cy={-s * 0.38}
                rx={s * 0.16}
                ry={s * 0.26}
                transform="rotate(-28)"
                fill="#1e293b"
                stroke="#0f172a"
                strokeWidth={s * 0.07}
              />
              <ellipse
                cx={-s * 0.05}
                cy={s * 0.38}
                rx={s * 0.16}
                ry={s * 0.26}
                transform="rotate(28)"
                fill="#1e293b"
                stroke="#0f172a"
                strokeWidth={s * 0.07}
              />
              {/* Dark Oval Sheep Face */}
              <ellipse
                cx={0}
                cy={0}
                rx={s * 0.36}
                ry={s * 0.3}
                fill="#1e293b"
                stroke="#0f172a"
                strokeWidth={s * 0.08}
              />
              {/* Little White Wool Forelock tuft */}
              <circle cx={-s * 0.12} cy={0} r={s * 0.14} fill={woolFill} />
              {/* Bright Eyes */}
              <circle cx={s * 0.08} cy={-s * 0.11} r={s * 0.065} fill="#f8fafc" />
              <circle cx={s * 0.08} cy={s * 0.11} r={s * 0.065} fill="#f8fafc" />
            </g>

            <title>{`Sheep ${sheep.index + 1}${
              sheep.personality ? ` (${sheep.personality})` : ""
            }${sheep.penned ? " — Penned" : ""}`}</title>
          </g>
        );
      })}

      {/* Dog Entities — Distinct Canine Head + Pointed Ears + Snout + Colored Harness */}
      {(snapshot.dogs ?? []).map((dog) => {
        const d = dogScale;
        const color = dogColor(dog.index);
        const roleLabel = ROLE_LABELS[dog.role ?? ""] ?? dog.role ?? "Dog";
        const roleTagWidth = Math.max(1.85, roleLabel.length * 0.27) * densityScale;
        const roleTagX = -roleTagWidth / 2;

        // Smart vertical & horizontal offset so clustered dogs don't overlap role tags
        const nearbyDogs = (snapshot.dogs ?? []).filter(
          (other) =>
            other.index !== dog.index &&
            Math.hypot(other.x - dog.x, other.y - dog.y) < 3.4
        );
        const isClustered = nearbyDogs.length > 0;
        const placeAbove =
          (isClustered && dog.index % 2 === 1 && dog.y > 2.2) ||
          dog.y > height - 2.5;
        const extraYOffset =
          isClustered && dog.index === 2 ? 0.42 * densityScale : 0;
        const dynamicTagY = placeAbove
          ? -(0.82 * densityScale) - roleTagHeight - extraYOffset
          : 0.78 * densityScale + extraYOffset;
        const dynamicLabelY = dynamicTagY + roleTagHeight / 2;

        return (
          <g
            key={`dog-${dog.index}`}
            transform={`translate(${dog.x + 0.5}, ${dog.y + 0.5})`}
            aria-label={`Dog ${dog.index + 1}`}
            style={transitionStyle}
            filter={`url(#${shadowId})`}
          >
            {/* Outer Colored Tactical Ring so Dogs are Instantly Spotlights on the Field */}
            <circle
              r={d * 1.18}
              fill="rgba(15, 23, 42, 0.55)"
              stroke={color}
              strokeWidth={d * 0.14}
            />

            {/* Pointed Canine Ears (Top-Left & Top-Right Triangles) */}
            <polygon
              points={`${-d * 0.78},${-d * 0.22} ${-d * 0.52},${-d * 1.08} ${-d * 0.08},${-d * 0.58}`}
              fill="#1e293b"
              stroke={color}
              strokeWidth={d * 0.12}
              strokeLinejoin="round"
            />
            <polygon
              points={`${d * 0.78},${-d * 0.22} ${d * 0.52},${-d * 1.08} ${d * 0.08},${-d * 0.58}`}
              fill="#1e293b"
              stroke={color}
              strokeWidth={d * 0.12}
              strokeLinejoin="round"
            />
            {/* Inner Pink/Amber Ear Accents */}
            <polygon
              points={`${-d * 0.62},${-d * 0.32} ${-d * 0.46},${-d * 0.86} ${-d * 0.2},${-d * 0.54}`}
              fill={color}
            />
            <polygon
              points={`${d * 0.62},${-d * 0.32} ${d * 0.46},${-d * 0.86} ${d * 0.2},${-d * 0.54}`}
              fill={color}
            />

            {/* Canine Head / Coat Badge */}
            <path
              d={`M ${-d * 0.76} ${-d * 0.2}
                  Q ${-d * 0.82} ${-d * 0.72} 0 ${-d * 0.72}
                  Q ${d * 0.82} ${-d * 0.72} ${d * 0.76} ${-d * 0.2}
                  L ${d * 0.48} ${d * 0.58}
                  Q 0 ${d * 0.92} ${-d * 0.48} ${d * 0.58} Z`}
              fill={color}
              stroke="#ffffff"
              strokeWidth={d * 0.13}
              strokeLinejoin="round"
            />

            {/* Border Collie White Blaze & Muzzle */}
            <ellipse
              cx={0}
              cy={d * 0.44}
              rx={d * 0.34}
              ry={d * 0.26}
              fill="#f8fafc"
            />
            {/* Black Dog Nose Tip */}
            <ellipse
              cx={0}
              cy={d * 0.56}
              rx={d * 0.16}
              ry={d * 0.11}
              fill="#0f172a"
            />

            {/* Dog Number Callsign on Forehead/Harness */}
            <text
              y={-d * 0.08}
              textAnchor="middle"
              dominantBaseline="central"
              fill="#ffffff"
              fontSize={d * 0.72}
              fontWeight={900}
              stroke="#0f172a"
              strokeWidth={d * 0.16}
              paintOrder="stroke"
            >
              {`D${dog.index + 1}`}
            </text>

            {showRoles && !compact && dog.role ? (
              <>
                <rect
                  x={roleTagX}
                  y={dynamicTagY}
                  width={roleTagWidth}
                  height={roleTagHeight}
                  rx={roleTagRadius}
                  fill="rgba(9, 13, 22, 0.92)"
                  stroke={color}
                  strokeWidth={roleTagStroke * 1.4}
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
              </>
            ) : null}
            <title>{`Dog ${dog.index + 1}${
              dog.role ? `: ${ROLE_LABELS[dog.role] ?? dog.role}` : ""
            }`}</title>
          </g>
        );
      })}
    </svg>
  );
};

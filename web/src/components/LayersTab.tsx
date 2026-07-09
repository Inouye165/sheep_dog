import React, { useState, useMemo } from "react";

interface LayersTabProps {
  effectiveConfig: Record<string, unknown> | null;
  topologyInfo: {
    observation_mode: string;
    hidden_layer_sizes: number[];
    observation_size: number;
    action_size: number;
    action_masking_enabled: boolean;
  } | null;
}

interface NodeDefinition {
  index: number;
  name: string;
  category: "position" | "distance" | "sensor" | "role" | "target" | "sheep" | "dog" | "shepherd" | "identity";
  description: string;
  details: string;
}

const CATEGORY_LABELS: Record<NodeDefinition["category"], string> = {
  position: "Position (Absolute)",
  distance: "Distance Scalar",
  sensor: "Environment Sensor",
  role: "Dog Role Assignment",
  target: "Target Coordination",
  sheep: "Sheep State (Relative)",
  dog: "Teammate State (Relative)",
  shepherd: "Shepherd Command",
  identity: "Agent Identity",
};

const CATEGORY_COLORS: Record<NodeDefinition["category"], string> = {
  position: "#3b82f6",
  distance: "#10b981",
  sensor: "#f59e0b",
  role: "#8b5cf6",
  target: "#ec4899",
  sheep: "#06b6d4",
  dog: "#14b8a6",
  shepherd: "#f43f5e",
  identity: "#64748b",
};

const GUIDED_NODES: NodeDefinition[] = [
  { index: 0, name: "own_x", category: "position", description: "Dog own absolute X position", details: "Normalized: own_x / (field_width - 1)" },
  { index: 1, name: "own_y", category: "position", description: "Dog own absolute Y position", details: "Normalized: own_y / (field_height - 1)" },
  { index: 2, name: "pen_x", category: "position", description: "Pen center absolute X coordinate", details: "Normalized: pen_center_x / (field_width - 1)" },
  { index: 3, name: "pen_y", category: "position", description: "Pen center absolute Y coordinate", details: "Normalized: pen_center_y / (field_height - 1)" },
  { index: 4, name: "flock_center_x", category: "position", description: "Flock center absolute X coordinate", details: "Normalized: flock_center_x / (field_width - 1)" },
  { index: 5, name: "flock_center_y", category: "position", description: "Flock center absolute Y coordinate", details: "Normalized: flock_center_y / (field_height - 1)" },
  { index: 6, name: "target_x", category: "target", description: "Assigned target absolute X coordinate", details: "Computed by high-level role assignment: target_x / (field_width - 1)" },
  { index: 7, name: "target_y", category: "target", description: "Assigned target absolute Y coordinate", details: "Computed by high-level role assignment: target_y / (field_height - 1)" },
  { index: 8, name: "distance_to_pen", category: "distance", description: "Distance from dog to pen center", details: "Normalized: distance_to_pen / field_diagonal" },
  { index: 9, name: "distance_to_flock", category: "distance", description: "Distance from dog to flock center", details: "Normalized: distance_to_flock / field_diagonal" },
  { index: 10, name: "distance_to_target", category: "distance", description: "Distance from dog to its target position", details: "Normalized: distance_to_target / field_diagonal" },
  { index: 11, name: "flock_spread", category: "sensor", description: "Standard deviation of sheep positions", details: "Normalized: flock_spread / field_diagonal" },
  { index: 12, name: "average_distance_to_pen", category: "distance", description: "Mean distance of all sheep to the pen", details: "Normalized: avg_distance / field_diagonal" },
  { index: 13, name: "wall_left", category: "sensor", description: "Proximity to the left boundary", details: "own_x / (field_width - 1)" },
  { index: 14, name: "wall_right", category: "sensor", description: "Proximity to the right boundary", details: "(field_width - 1 - own_x) / (field_width - 1)" },
  { index: 15, name: "wall_top", category: "sensor", description: "Proximity to the top boundary", details: "own_y / (field_height - 1)" },
  { index: 16, name: "wall_bottom", category: "sensor", description: "Proximity to the bottom boundary", details: "(field_height - 1 - own_y) / (field_height - 1)" },
  { index: 17, name: "blocked_steps", category: "sensor", description: "Steps dog has been blocked/colliding", details: "min(blocked_steps, 10) / 10.0" },
  { index: 18, name: "no_progress_steps", category: "sensor", description: "Steps with no team progress toward pen", details: "min(no_progress, window) / window" },
  { index: 19, name: "revisits_recent_position", category: "sensor", description: "Binary: position revisit loop detected", details: "1.0 if revisited recent cell, 0.0 otherwise" },
  { index: 20, name: "two_position_loop", category: "sensor", description: "Binary: two-position oscillation detected", details: "1.0 if oscillating between two positions, 0.0 otherwise" },
  { index: 21, name: "stray_present", category: "sensor", description: "Binary: any stray sheep exists", details: "1.0 if a sheep is > 1.8 * flock_spread from center, 0.0 otherwise" },
  { index: 22, name: "role_rear_pressure", category: "role", description: "Flag: assigned to REAR_PRESSURE role", details: "One-hot: 1.0 if active. Pushes flock from behind." },
  { index: 23, name: "role_left_flanker", category: "role", description: "Flag: assigned to LEFT_FLANKER role", details: "One-hot: 1.0 if active. Blocks left escapes." },
  { index: 24, name: "role_right_flanker", category: "role", description: "Flag: assigned to RIGHT_FLANKER role", details: "One-hot: 1.0 if active. Blocks right escapes." },
  { index: 25, name: "role_collector", category: "role", description: "Flag: assigned to COLLECTOR role", details: "One-hot: 1.0 if active. Fetches strays." },
  { index: 26, name: "role_blocker", category: "role", description: "Flag: assigned to BLOCKER role", details: "One-hot: 1.0 if active. Holds near pen mouth." },
  { index: 27, name: "focus_sheep_dx", category: "sheep", description: "Relative X to closest unpenned target sheep", details: "(focus_sheep_x - own_x) / (field_width - 1)" },
  { index: 28, name: "focus_sheep_dy", category: "sheep", description: "Relative Y to closest unpenned target sheep", details: "(focus_sheep_y - own_y) / (field_height - 1)" },
  { index: 29, name: "focus_sheep_distance", category: "distance", description: "Distance to closest unpenned target sheep", details: "distance_to_focus / field_diagonal" },
  { index: 30, name: "stray_sheep_dx", category: "sheep", description: "Relative X to designated stray sheep", details: "(stray_x - own_x) / (field_width - 1). 0.0 if no stray." },
  { index: 31, name: "stray_sheep_dy", category: "sheep", description: "Relative Y to designated stray sheep", details: "(stray_y - own_y) / (field_height - 1). 0.0 if no stray." },
  { index: 32, name: "sheep_0_dx", category: "sheep", description: "Relative X to 1st closest sheep", details: "Normalized X offset, sheep slot 0." },
  { index: 33, name: "sheep_0_dy", category: "sheep", description: "Relative Y to 1st closest sheep", details: "Normalized Y offset, sheep slot 0." },
  { index: 34, name: "sheep_0_penned", category: "sheep", description: "Penned state of 1st closest sheep", details: "1.0 if penned, 0.0 if unpenned." },
  { index: 35, name: "sheep_1_dx", category: "sheep", description: "Relative X to 2nd closest sheep", details: "Normalized X offset, sheep slot 1." },
  { index: 36, name: "sheep_1_dy", category: "sheep", description: "Relative Y to 2nd closest sheep", details: "Normalized Y offset, sheep slot 1." },
  { index: 37, name: "sheep_1_penned", category: "sheep", description: "Penned state of 2nd closest sheep", details: "1.0 if penned, 0.0 if unpenned." },
  { index: 38, name: "sheep_2_dx", category: "sheep", description: "Relative X to 3rd closest sheep", details: "Normalized X offset, sheep slot 2." },
  { index: 39, name: "sheep_2_dy", category: "sheep", description: "Relative Y to 3rd closest sheep", details: "Normalized Y offset, sheep slot 2." },
  { index: 40, name: "sheep_2_penned", category: "sheep", description: "Penned state of 3rd closest sheep", details: "1.0 if penned, 0.0 if unpenned." },
  { index: 41, name: "sheep_3_dx", category: "sheep", description: "Relative X to 4th closest sheep", details: "Normalized X offset, sheep slot 3." },
  { index: 42, name: "sheep_3_dy", category: "sheep", description: "Relative Y to 4th closest sheep", details: "Normalized Y offset, sheep slot 3." },
  { index: 43, name: "sheep_3_penned", category: "sheep", description: "Penned state of 4th closest sheep", details: "1.0 if penned, 0.0 if unpenned." },
  { index: 44, name: "sheep_4_dx", category: "sheep", description: "Relative X to 5th closest sheep", details: "Normalized X offset, sheep slot 4." },
  { index: 45, name: "sheep_4_dy", category: "sheep", description: "Relative Y to 5th closest sheep", details: "Normalized Y offset, sheep slot 4." },
  { index: 46, name: "sheep_4_penned", category: "sheep", description: "Penned state of 5th closest sheep", details: "1.0 if penned, 0.0 if unpenned." },
  { index: 47, name: "sheep_5_dx", category: "sheep", description: "Relative X to 6th closest sheep", details: "Normalized X offset, sheep slot 5." },
  { index: 48, name: "sheep_5_dy", category: "sheep", description: "Relative Y to 6th closest sheep", details: "Normalized Y offset, sheep slot 5." },
  { index: 49, name: "sheep_5_penned", category: "sheep", description: "Penned state of 6th closest sheep", details: "1.0 if penned, 0.0 if unpenned." },
  { index: 50, name: "other_dog_0_dx", category: "dog", description: "Relative X to teammate dog 0", details: "(teammate_0_x - own_x) / (field_width - 1)" },
  { index: 51, name: "other_dog_0_dy", category: "dog", description: "Relative Y to teammate dog 0", details: "(teammate_0_y - own_y) / (field_height - 1)" },
  { index: 52, name: "other_dog_1_dx", category: "dog", description: "Relative X to teammate dog 1", details: "(teammate_1_x - own_x) / (field_width - 1)" },
  { index: 53, name: "other_dog_1_dy", category: "dog", description: "Relative Y to teammate dog 1", details: "(teammate_1_y - own_y) / (field_height - 1)" },
];

const HIERARCHICAL_NODES: NodeDefinition[] = [
  ...GUIDED_NODES,
  { index: 54, name: "shepherd_cmd_gather", category: "shepherd", description: "Active Shepherd Command: GATHER", details: "1.0 if active — instructs dog to group dispersed sheep." },
  { index: 55, name: "shepherd_cmd_drive_to_pen", category: "shepherd", description: "Active Shepherd Command: DRIVE_TO_PEN", details: "1.0 if active — push flock toward the pen." },
  { index: 56, name: "shepherd_cmd_hold_left", category: "shepherd", description: "Active Shepherd Command: HOLD_LEFT", details: "1.0 if active — guard the left flank." },
  { index: 57, name: "shepherd_cmd_hold_right", category: "shepherd", description: "Active Shepherd Command: HOLD_RIGHT", details: "1.0 if active — guard the right flank." },
  { index: 58, name: "shepherd_cmd_block_escape", category: "shepherd", description: "Active Shepherd Command: BLOCK_ESCAPE", details: "1.0 if active — general block fallback." },
  { index: 59, name: "shepherd_cmd_apply_pressure", category: "shepherd", description: "Active Shepherd Command: APPLY_PRESSURE", details: "1.0 if active — push from behind into pen." },
  { index: 60, name: "shepherd_cmd_back_off", category: "shepherd", description: "Active Shepherd Command: BACK_OFF", details: "1.0 if active — give sheep room to settle." },
  { index: 61, name: "shepherd_cmd_stop", category: "shepherd", description: "Active Shepherd Command: STOP", details: "1.0 if active — all sheep penned, dogs halt." },
  { index: 62, name: "dog_id_normalized", category: "identity", description: "Normalized index of this dog", details: "dog_index / max(1, dog_count - 1)" },
  { index: 63, name: "dog_count_normalized", category: "identity", description: "Normalized total dog count", details: "dog_count / max(1, MAX_DOG_SLOTS=5)" },
  { index: 64, name: "dog_id_slot_0", category: "identity", description: "Identity one-hot: Slot 0", details: "1.0 if this is dog 0." },
  { index: 65, name: "dog_id_slot_1", category: "identity", description: "Identity one-hot: Slot 1", details: "1.0 if this is dog 1." },
  { index: 66, name: "dog_id_slot_2", category: "identity", description: "Identity one-hot: Slot 2", details: "1.0 if this is dog 2." },
  { index: 67, name: "dog_id_slot_3", category: "identity", description: "Identity one-hot: Slot 3", details: "1.0 if this is dog 3." },
  { index: 68, name: "dog_id_slot_4", category: "identity", description: "Identity one-hot: Slot 4", details: "1.0 if this is dog 4." },
];

const EMERGENT_NODES: NodeDefinition[] = [
  { index: 0, name: "own_x", category: "position", description: "Dog own absolute X position", details: "Normalized: own_x / (field_width - 1)" },
  { index: 1, name: "own_y", category: "position", description: "Dog own absolute Y position", details: "Normalized: own_y / (field_height - 1)" },
  { index: 2, name: "dog_id_slot_0", category: "identity", description: "Identity one-hot: Slot 0", details: "1.0 if this is dog 0." },
  { index: 3, name: "dog_id_slot_1", category: "identity", description: "Identity one-hot: Slot 1", details: "1.0 if this is dog 1." },
  { index: 4, name: "dog_id_slot_2", category: "identity", description: "Identity one-hot: Slot 2", details: "1.0 if this is dog 2." },
  { index: 5, name: "pen_x", category: "position", description: "Pen center absolute X coordinate", details: "Normalized: pen_center_x / (field_width - 1)" },
  { index: 6, name: "pen_y", category: "position", description: "Pen center absolute Y coordinate", details: "Normalized: pen_center_y / (field_height - 1)" },
  { index: 7, name: "flock_center_x", category: "position", description: "Flock center absolute X coordinate", details: "Normalized: flock_center_x / (field_width - 1)" },
  { index: 8, name: "flock_center_y", category: "position", description: "Flock center absolute Y coordinate", details: "Normalized: flock_center_y / (field_height - 1)" },
  { index: 9, name: "distance_to_pen", category: "distance", description: "Distance from dog to pen center", details: "Normalized: distance_to_pen / field_diagonal" },
  { index: 10, name: "distance_to_flock", category: "distance", description: "Distance from dog to flock center", details: "Normalized: distance_to_flock / field_diagonal" },
  { index: 11, name: "flock_spread", category: "sensor", description: "Standard deviation of sheep positions", details: "Normalized: flock_spread / field_diagonal" },
  { index: 12, name: "average_distance_to_pen", category: "distance", description: "Mean distance of all sheep to pen", details: "Normalized: avg_distance / field_diagonal" },
  { index: 13, name: "wall_left", category: "sensor", description: "Proximity to left boundary", details: "own_x / (field_width - 1)" },
  { index: 14, name: "wall_right", category: "sensor", description: "Proximity to right boundary", details: "(field_width - 1 - own_x) / (field_width - 1)" },
  { index: 15, name: "wall_top", category: "sensor", description: "Proximity to top boundary", details: "own_y / (field_height - 1)" },
  { index: 16, name: "wall_bottom", category: "sensor", description: "Proximity to bottom boundary", details: "(field_height - 1 - own_y) / (field_height - 1)" },
  { index: 17, name: "blocked_steps", category: "sensor", description: "Steps dog has been blocked", details: "min(blocked_steps, 10) / 10.0" },
  { index: 18, name: "no_progress_steps", category: "sensor", description: "Steps with no team progress", details: "min(no_progress, window) / window" },
  { index: 19, name: "revisits_recent_position", category: "sensor", description: "Binary: position revisit loop", details: "1.0 if revisited recent cell." },
  { index: 20, name: "two_position_loop", category: "sensor", description: "Binary: two-position oscillation", details: "1.0 if oscillating between two positions." },
  { index: 21, name: "stray_present", category: "sensor", description: "Binary: any stray sheep", details: "1.0 if stray exists." },
  { index: 22, name: "nearest_unpenned_dx", category: "sheep", description: "Relative X to closest unpenned sheep", details: "(nearest_x - own_x) / (field_width - 1)" },
  { index: 23, name: "nearest_unpenned_dy", category: "sheep", description: "Relative Y to closest unpenned sheep", details: "(nearest_y - own_y) / (field_height - 1)" },
  { index: 24, name: "nearest_unpenned_distance", category: "distance", description: "Distance to closest unpenned sheep", details: "distance_to_nearest / field_diagonal" },
  { index: 25, name: "farthest_unpenned_dx", category: "sheep", description: "Relative X to unpenned sheep farthest from pen", details: "(farthest_x - own_x) / (field_width - 1)" },
  { index: 26, name: "farthest_unpenned_dy", category: "sheep", description: "Relative Y to unpenned sheep farthest from pen", details: "(farthest_y - own_y) / (field_height - 1)" },
  { index: 27, name: "farthest_unpenned_distance", category: "distance", description: "Distance to unpenned sheep farthest from pen", details: "distance_to_farthest / field_diagonal" },
  { index: 28, name: "sheep_0_dx", category: "sheep", description: "Relative X to 1st closest sheep", details: "Normalized X offset, slot 0." },
  { index: 29, name: "sheep_0_dy", category: "sheep", description: "Relative Y to 1st closest sheep", details: "Normalized Y offset, slot 0." },
  { index: 30, name: "sheep_0_penned", category: "sheep", description: "Penned state of 1st closest sheep", details: "1.0 if penned." },
  { index: 31, name: "sheep_1_dx", category: "sheep", description: "Relative X to 2nd closest sheep", details: "Normalized X offset, slot 1." },
  { index: 32, name: "sheep_1_dy", category: "sheep", description: "Relative Y to 2nd closest sheep", details: "Normalized Y offset, slot 1." },
  { index: 33, name: "sheep_1_penned", category: "sheep", description: "Penned state of 2nd closest sheep", details: "1.0 if penned." },
  { index: 34, name: "sheep_2_dx", category: "sheep", description: "Relative X to 3rd closest sheep", details: "Normalized X offset, slot 2." },
  { index: 35, name: "sheep_2_dy", category: "sheep", description: "Relative Y to 3rd closest sheep", details: "Normalized Y offset, slot 2." },
  { index: 36, name: "sheep_2_penned", category: "sheep", description: "Penned state of 3rd closest sheep", details: "1.0 if penned." },
  { index: 37, name: "sheep_3_dx", category: "sheep", description: "Relative X to 4th closest sheep", details: "Normalized X offset, slot 3." },
  { index: 38, name: "sheep_3_dy", category: "sheep", description: "Relative Y to 4th closest sheep", details: "Normalized Y offset, slot 3." },
  { index: 39, name: "sheep_3_penned", category: "sheep", description: "Penned state of 4th closest sheep", details: "1.0 if penned." },
  { index: 40, name: "sheep_4_dx", category: "sheep", description: "Relative X to 5th closest sheep", details: "Normalized X offset, slot 4." },
  { index: 41, name: "sheep_4_dy", category: "sheep", description: "Relative Y to 5th closest sheep", details: "Normalized Y offset, slot 4." },
  { index: 42, name: "sheep_4_penned", category: "sheep", description: "Penned state of 5th closest sheep", details: "1.0 if penned." },
  { index: 43, name: "sheep_5_dx", category: "sheep", description: "Relative X to 6th closest sheep", details: "Normalized X offset, slot 5." },
  { index: 44, name: "sheep_5_dy", category: "sheep", description: "Relative Y to 6th closest sheep", details: "Normalized Y offset, slot 5." },
  { index: 45, name: "sheep_5_penned", category: "sheep", description: "Penned state of 6th closest sheep", details: "1.0 if penned." },
  { index: 46, name: "other_dog_0_dx", category: "dog", description: "Relative X to teammate dog 0", details: "(teammate_0_x - own_x) / (field_width - 1)" },
  { index: 47, name: "other_dog_0_dy", category: "dog", description: "Relative Y to teammate dog 0", details: "(teammate_0_y - own_y) / (field_height - 1)" },
  { index: 48, name: "other_dog_1_dx", category: "dog", description: "Relative X to teammate dog 1", details: "(teammate_1_x - own_x) / (field_width - 1)" },
  { index: 49, name: "other_dog_1_dy", category: "dog", description: "Relative Y to teammate dog 1", details: "(teammate_1_y - own_y) / (field_height - 1)" },
];

interface OutputDefinition { index: number; name: string; type: string; description: string; communicating: string; }

const ACTOR_NODES: OutputDefinition[] = [
  { index: 0, name: "up", type: "Step", description: "Move 1 unit up (North)", communicating: "Step-sized movement. Safe, costs no sprint budget." },
  { index: 1, name: "down", type: "Step", description: "Move 1 unit down (South)", communicating: "Step-sized movement southward." },
  { index: 2, name: "left", type: "Step", description: "Move 1 unit left (West)", communicating: "Step-sized movement westward." },
  { index: 3, name: "right", type: "Step", description: "Move 1 unit right (East)", communicating: "Step-sized movement eastward." },
  { index: 4, name: "sprint_up", type: "Sprint", description: "Move 2 units up (North, fast)", communicating: "Double-speed move. Costs extra energy — used to close gaps or chase strays." },
  { index: 5, name: "sprint_down", type: "Sprint", description: "Move 2 units down (South, fast)", communicating: "Double-speed move southward." },
  { index: 6, name: "sprint_left", type: "Sprint", description: "Move 2 units left (West, fast)", communicating: "Double-speed move westward." },
  { index: 7, name: "sprint_right", type: "Sprint", description: "Move 2 units right (East, fast)", communicating: "Double-speed move eastward." },
  { index: 8, name: "wait", type: "Hold", description: "Remain stationary this step", communicating: "Deliberate stillness — used to hold a blocking position or let sheep settle." },
];

const CRITIC_NODES: OutputDefinition[] = [
  { index: 0, name: "V(s)", type: "Scalar Value", description: "Expected total future reward from this state", communicating: "Used by PPO/GAE to compute advantage estimates. Higher = better situation. No activation — can be any real number." },
];

function asRecord(v: unknown): Record<string, unknown> | null {
  return v !== null && typeof v === "object" && !Array.isArray(v) ? (v as Record<string, unknown>) : null;
}

function formatNumber(n: number): string {
  return n >= 1_000_000 ? (n / 1_000_000).toFixed(2) + "M" : n >= 1_000 ? (n / 1_000).toFixed(1) + "K" : String(n);
}

type LayerKey = "input" | { hidden: number } | "actor" | "critic";

// ============================================================
// CLICKABLE ARCHITECTURE DIAGRAM
// ============================================================

function ArchitectureDiagram({ inputSize, hiddenSizes, actorSize, criticSize, onLayerClick }: {
  inputSize: number; hiddenSizes: number[]; actorSize: number; criticSize: number;
  onLayerClick: (key: LayerKey) => void;
}) {
  const btnBase: React.CSSProperties = {
    border: "1px solid var(--panel-border)",
    borderRadius: "0.55rem",
    padding: "0.6rem 0.85rem",
    textAlign: "center",
    minWidth: "100px",
    cursor: "pointer",
    background: "rgba(16, 28, 44, 0.4)",
    transition: "all 150ms ease",
    fontFamily: "monospace",
    boxShadow: "0 2px 8px rgba(0,0,0,0.15)"
  };
  const hov = (color: string) => ({
    boxShadow: "0 0 12px " + color + "60",
    borderColor: color,
    transform: "translateY(-2px)"
  });

  return (
    <div style={{ padding: "0.5rem 0", fontFamily: "inherit" }}>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "center", flexWrap: "wrap", gap: "0.75rem", padding: "1rem 0" }}>
        {/* Input */}
        <ArchBox label="Input Layer" sub={inputSize + " features"} color="#3b82f6" bg="rgba(59,130,246,0.12)" border="rgba(59,130,246,0.3)" btnBase={btnBase} hov={hov} onClick={() => onLayerClick("input")} />
        <Arrow />

        {/* Hidden layers */}
        {hiddenSizes.map((s, i) => (
          <React.Fragment key={i}>
            <ArchBox label={"Hidden " + (i + 1)} sub={s + " Linear"} color="#8b5cf6" bg="rgba(139,92,246,0.1)" border="rgba(139,92,246,0.3)" btnBase={btnBase} hov={hov} onClick={() => onLayerClick({ hidden: i })} />
            <Arrow />
          </React.Fragment>
        ))}

        {/* Fork to outputs */}
        <div style={{ display: "flex", flexDirection: "column", gap: "0.6rem" }}>
          <div style={{ display: "flex", alignItems: "center", gap: "0.3rem" }}>
            <span style={{ color: "rgba(148,163,184,0.4)", fontSize: "0.8rem", userSelect: "none" }}>&#x250c;</span>
            <ArchBox label="Actor Head" sub={actorSize + " actions"} color="#818cf8" bg="rgba(129,140,248,0.12)" border="rgba(129,140,248,0.35)" btnBase={btnBase} hov={hov} onClick={() => onLayerClick("actor")} />
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: "0.3rem" }}>
            <span style={{ color: "rgba(148,163,184,0.4)", fontSize: "0.8rem", userSelect: "none" }}>&#x2514;</span>
            <ArchBox label="Critic Head" sub={criticSize + " value V(s)"} color="#10b981" bg="rgba(16,185,129,0.08)" border="rgba(16,185,129,0.25)" btnBase={btnBase} hov={hov} onClick={() => onLayerClick("critic")} />
          </div>
        </div>
      </div>
    </div>
  );
}

function Arrow() {
  return <div style={{ color: "rgba(148,163,184,0.3)", padding: "0 0.2rem", fontSize: "1.2rem", fontWeight: "bold", userSelect: "none" }}>&#8594;</div>;
}

function ArchBox({ label, sub, color, bg, border, btnBase, hov, onClick }: {
  label: string; sub: string; color: string; bg: string; border: string;
  btnBase: React.CSSProperties; hov: (color: string) => React.CSSProperties; onClick: () => void;
}) {
  const [hovered, setHovered] = React.useState(false);
  return (
    <button
      style={{
        ...btnBase,
        borderColor: hovered ? color : border,
        background: hovered ? bg : "rgba(12, 22, 36, 0.6)",
        color: hovered ? "#ffffff" : "var(--text)",
        ...(hovered ? hov(color) : {})
      }}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      onClick={onClick}
      title={"Click to navigate to: " + label}
    >
      <div style={{ color, fontWeight: "700", fontSize: "0.8rem", letterSpacing: "0.02em" }}>{label}</div>
      <div style={{ color: "var(--muted)", fontSize: "0.68rem", marginTop: "0.15rem" }}>{sub}</div>
      <div style={{ color: color, fontSize: "0.58rem", marginTop: "0.25rem", opacity: 0.85, fontWeight: "600" }}>GO TO TAB &rarr;</div>
    </button>
  );
}

interface GlossaryEntry { term: string; tag: string; tagColor: string; definition: string; usedHere: string; }
const GLOSSARY: GlossaryEntry[] = [
  { term: "MLP (Multi-Layer Perceptron)", tag: "Architecture", tagColor: "#3b82f6", definition: "A fully-connected feedforward neural network. Every neuron in one layer is connected to every neuron in the next. The simplest and most common deep learning building block for fixed-size input vectors.", usedHere: "This network IS an MLP. Input features -> hidden layers -> actor/critic heads." },
  { term: "ReLU (Rectified Linear Unit)", tag: "Activation", tagColor: "#10b981", definition: "f(x) = max(0, x). Outputs zero for negative inputs and passes positive inputs unchanged. Cheap to compute, avoids the vanishing-gradient problem.", usedHere: "Applied after every hidden linear layer." },
  { term: "Tanh", tag: "Activation", tagColor: "#10b981", definition: "f(x) = (e^x - e^-x)/(e^x + e^-x). Outputs in (-1,1). Zero-centered but suffers vanishing gradients for large inputs. Largely replaced by ReLU in hidden layers.", usedHere: "Not used. Shown for comparison." },
  { term: "Sigmoid", tag: "Activation", tagColor: "#10b981", definition: "f(x) = 1/(1+e^-x). Squashes to (0,1). Common in binary classification outputs. Causes vanishing gradients in deep networks.", usedHere: "Not used here. Binary inputs (stray_present etc) are pre-computed 0/1." },
  { term: "Softmax", tag: "Activation", tagColor: "#10b981", definition: "Converts logits to a probability distribution summing to 1. Used in classification heads.", usedHere: "Applied (with masking) to actor logits to produce action probabilities." },
  { term: "CNN (Convolutional Neural Network)", tag: "Architecture", tagColor: "#3b82f6", definition: "Applies learned filters that slide across a spatial input. Excellent for image-like data. Dominant in visual RL (Atari, etc.).", usedHere: "Not used. Hand-crafted vector inputs need no spatial convolution." },
  { term: "PPO (Proximal Policy Optimization)", tag: "RL Algorithm", tagColor: "#ec4899", definition: "Policy-gradient RL algorithm. Updates using a clipped objective that prevents catastrophically large policy updates. Fast and stable.", usedHere: "MaskablePPO variant with action masking support." },
  { term: "GAE (Generalized Advantage Estimation)", tag: "RL Concept", tagColor: "#ec4899", definition: "Blends multi-step returns with critic bootstraps using lambda. Reduces variance while controlling bias.", usedHere: "Used during training. Requires critic V(s)." },
  { term: "Action Masking", tag: "RL Concept", tagColor: "#ec4899", definition: "Sets logits for illegal actions to -infinity before Softmax, giving them exactly 0% probability.", usedHere: "Enabled via MaskablePPO. Environment computes valid moves each step." },
  { term: "One-Hot Encoding", tag: "Input Encoding", tagColor: "#64748b", definition: "Represents a categorical choice as a binary vector with exactly one 1. Avoids implying numerical order.", usedHere: "Used for roles (role_rear_pressure, etc.) and dog identity slots." },
  { term: "Normalization", tag: "Input Encoding", tagColor: "#64748b", definition: "Rescaling raw values to a consistent range (typically [0,1]) for stable training.", usedHere: "All inputs normalized. Coordinates / field dimension; distances / field diagonal." },
];

const LAYER_COMPARISON = [
  { name: "Linear (Dense)", used: true, icon: "▦", color: "#8b5cf6", strength: "Learns any weighted combination of inputs. Universal approximator when stacked.", weakness: "Quadratic parameter growth with width. No spatial structure.", whenToUse: "Fixed-size, non-spatial feature vectors." },
  { name: "Convolutional (CNN)", used: false, icon: "⊞", color: "#3b82f6", strength: "Weight-sharing, vastly fewer parameters for image data.", weakness: "Assumes 2D/3D spatial structure.", whenToUse: "Pixel observations, spatial maps (Atari, Minecraft)." },
  { name: "Recurrent (LSTM/GRU)", used: false, icon: "↺", color: "#f59e0b", strength: "Maintains state across timesteps. Learns temporal patterns.", weakness: "Harder to parallelize. Requires truncated BPTT in RL.", whenToUse: "Partially-observable envs where history matters." },
  { name: "Transformer (Attention)", used: false, icon: "◈", color: "#ec4899", strength: "Dynamic context-aware weighting. Handles variable-length sets.", weakness: "Heavy compute. Overkill for small fixed-size vectors.", whenToUse: "Variable agent counts or very large observation sets." },
];

function Pill({ label, color, bg }: { label: string; color: string; bg: string }) {
  return (
    <span style={{
      fontSize: "0.68rem",
      padding: "0.18rem 0.6rem",
      borderRadius: "999px",
      background: bg,
      border: "1px solid " + color + "30",
      color,
      fontWeight: "600",
      textTransform: "uppercase" as const,
      letterSpacing: "0.05em"
    }}>
      {label}
    </span>
  );
}

function SectionBox({ accent, title, children }: { accent: string; title: string; children: React.ReactNode }) {
  return (
    <div style={{
      background: "rgba(10,20,35,0.4)",
      borderLeft: "4px solid " + accent,
      borderTop: "1px solid rgba(148,163,184,0.06)",
      borderRight: "1px solid rgba(148,163,184,0.06)",
      borderBottom: "1px solid rgba(148,163,184,0.06)",
      borderRadius: "0.6rem",
      padding: "1rem 1.25rem",
      boxShadow: "0 4px 20px rgba(0,0,0,0.15)"
    }}>
      <div style={{
        fontSize: "0.72rem",
        color: accent,
        fontWeight: "700",
        textTransform: "uppercase" as const,
        letterSpacing: "0.08em",
        marginBottom: "0.5rem"
      }}>
        {title}
      </div>
      <div style={{ fontSize: "0.85rem", color: "var(--muted)", lineHeight: "1.6" }}>
        {children}
      </div>
    </div>
  );
}

// ============================================================
// MAIN COMPONENT
// ============================================================

export function LayersTab({ effectiveConfig, topologyInfo }: LayersTabProps) {
  const root = asRecord(effectiveConfig);
  const training = asRecord(root?.training);
  const activeModeFromConfig = (topologyInfo?.observation_mode || (training?.observation_mode as string | undefined) || "guided") as string;

  const [activeSubTab, setActiveSubTab] = useState<"general" | "entry" | "hidden" | "exit">("general");
  const [selectedViewMode, setSelectedViewMode] = useState<string>(activeModeFromConfig);
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedCategory, setSelectedCategory] = useState<string>("all");
  const [glossaryQuery, setGlossaryQuery] = useState("");

  const hiddenLayerSizes: number[] = useMemo(() => {
    if (topologyInfo?.hidden_layer_sizes && topologyInfo.hidden_layer_sizes.length > 0) return topologyInfo.hidden_layer_sizes;
    const nn = asRecord(root?.neural_network);
    const sizes = nn?.hidden_layer_sizes;
    if (Array.isArray(sizes) && sizes.length > 0) return sizes as number[];
    return [256, 256];
  }, [topologyInfo, root]);

  const activeNodesList = useMemo(() => {
    if (selectedViewMode === "hierarchical") return HIERARCHICAL_NODES;
    if (selectedViewMode === "emergent") return EMERGENT_NODES;
    return GUIDED_NODES;
  }, [selectedViewMode]);

  const filteredNodesList = useMemo(() => {
    return activeNodesList.filter(node => {
      const q = searchQuery.toLowerCase();
      const matchesSearch = node.name.toLowerCase().includes(q) || node.description.toLowerCase().includes(q) || node.details.toLowerCase().includes(q);
      const matchesCategory = selectedCategory === "all" || node.category === selectedCategory;
      return matchesSearch && matchesCategory;
    });
  }, [activeNodesList, searchQuery, selectedCategory]);

  const totalParams = useMemo(() => {
    let prev = activeNodesList.length;
    let total = 0;
    for (const sz of hiddenLayerSizes) {
      total += prev * sz + sz;
      prev = sz;
    }
    total += prev * ACTOR_NODES.length + ACTOR_NODES.length;
    total += prev * CRITIC_NODES.length + CRITIC_NODES.length;
    return total;
  }, [activeNodesList.length, hiddenLayerSizes]);

  const filteredGlossary = useMemo(() => {
    return GLOSSARY.filter(item => {
      const q = glossaryQuery.toLowerCase();
      return item.term.toLowerCase().includes(q) || item.definition.toLowerCase().includes(q) || item.tag.toLowerCase().includes(q);
    });
  }, [glossaryQuery]);

  const subTabButtons = (
    <div style={{ display: "flex", borderBottom: "1px solid var(--panel-border)", gap: "0.5rem", marginBottom: "1rem", flexShrink: 0 }}>
      {([
        { id: "general", label: "General", desc: "System Overview" },
        { id: "entry", label: "Entry (Input)", desc: "Observation Senses" },
        { id: "hidden", label: "Hidden Layers", desc: "Internal Transformations" },
        { id: "exit", label: "Exit (Output)", desc: "Action & Evaluation" }
      ] as const).map(tab => {
        const isActive = activeSubTab === tab.id;
        return (
          <button
            key={tab.id}
            onClick={() => setActiveSubTab(tab.id)}
            style={{
              background: "transparent",
              border: "none",
              borderBottom: isActive ? "3px solid var(--accent)" : "3px solid transparent",
              borderRadius: 0,
              padding: "0.6rem 1.25rem",
              color: isActive ? "var(--text)" : "var(--muted)",
              fontWeight: isActive ? "600" : "400",
              fontSize: "0.85rem",
              cursor: "pointer",
              transition: "all 150ms ease",
              textAlign: "left",
              transform: "none"
            }}
          >
            <div style={{ fontWeight: "700", color: isActive ? "var(--accent)" : "var(--text)", fontSize: "0.9rem" }}>{tab.label}</div>
            <div style={{ fontSize: "0.68rem", color: "var(--muted)", marginTop: "0.1rem" }}>{tab.desc}</div>
          </button>
        );
      })}
    </div>
  );

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%", maxHeight: "100%", minHeight: 0, overflow: "hidden" }}>
      {/* Title block */}
      <div style={{ marginBottom: "0.75rem", flexShrink: 0 }}>
        <p className="eyebrow" style={{ color: "var(--accent)", fontWeight: 600, margin: 0 }}>Architecture Reference & ML Glossary</p>
        <h2 style={{ fontSize: "1.6rem", margin: "0.15rem 0 0.35rem" }}>Neural Network Layers</h2>
        <p style={{ margin: 0, fontSize: "0.83rem", color: "var(--muted)", maxWidth: "800px", lineHeight: "1.4" }}>
          Explore what the agents see, how they make decisions, and how the network layers operate. 
          Use the sub-tabs below to study different parts of the network or click elements in the architecture diagram to switch views.
        </p>
      </div>

      {/* Sub tabs nav */}
      {subTabButtons}

      {/* Content wrapper */}
      <div style={{ flex: 1, minHeight: 0, overflowY: "auto", paddingRight: "0.4rem", display: "flex", flexDirection: "column", gap: "1rem" }}>

        {/* GENERAL TAB */}
        {activeSubTab === "general" && (
          <div style={{ display: "flex", flexDirection: "column", gap: "1.25rem" }}>
            {/* KPI Cards Row */}
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit,minmax(160px,1fr))", gap: "0.75rem" }}>
              {[
                { label: "Active Training Mode", value: activeModeFromConfig.toUpperCase(), color: "var(--accent)", badge: "RUNTIME" },
                { label: "Senses (Inputs)", value: `${activeNodesList.length} Nodes`, color: "#3b82f6", badge: null },
                { label: "Hidden Layers", value: `${hiddenLayerSizes.length} Layers`, color: "#8b5cf6", badge: null },
                { label: "Total Neurons", value: formatNumber(hiddenLayerSizes.reduce((a, b) => a + b, 0)), color: "#93c5fd", badge: null },
                { label: "Total Parameters", value: formatNumber(totalParams), color: "#10b981", badge: null },
                { label: "Action Outputs", value: `${ACTOR_NODES.length} Logits`, color: "#818cf8", badge: null },
              ].map((kpi, i) => (
                <div key={i} style={{ background: "rgba(12,24,38,0.5)", border: "1px solid var(--panel-border)", padding: "0.9rem 1.1rem", borderRadius: "0.6rem" }}>
                  <span style={{ fontSize: "0.65rem", textTransform: "uppercase", letterSpacing: "0.06em", color: "var(--muted)", fontWeight: "600" }}>{kpi.label}</span>
                  <div style={{ display: "flex", alignItems: "center", gap: "0.4rem", marginTop: "0.25rem" }}>
                    <strong style={{ fontSize: "1.15rem", color: kpi.color }}>{kpi.value}</strong>
                    {kpi.badge && <span style={{ background: "var(--accent-soft)", color: "var(--accent)", border: "1px solid var(--accent)", fontSize: "0.55rem", padding: "0.1rem 0.35rem", borderRadius: "999px", fontWeight: "bold" }}>{kpi.badge}</span>}
                  </div>
                </div>
              ))}
            </div>

            {/* Architecture diagram container */}
            <section style={{ background: "rgba(10,20,30,0.4)", border: "1px solid var(--panel-border)", borderRadius: "0.75rem", padding: "1rem 1.25rem" }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "0.5rem" }}>
                <h3 style={{ margin: 0, fontSize: "0.95rem", fontWeight: "600", color: "var(--text)" }}>Interactive Network Flow Diagram</h3>
                <span style={{ fontSize: "0.72rem", color: "var(--muted)" }}>Click any box to inspect feature details & formulas</span>
              </div>
              <ArchitectureDiagram
                inputSize={activeNodesList.length}
                hiddenSizes={hiddenLayerSizes}
                actorSize={ACTOR_NODES.length}
                criticSize={CRITIC_NODES.length}
                onLayerClick={(key) => {
                  if (key === "input") {
                    setActiveSubTab("entry");
                  } else if (typeof key === "object" && "hidden" in key) {
                    setActiveSubTab("hidden");
                  } else if (key === "actor" || key === "critic") {
                    setActiveSubTab("exit");
                  }
                }}
              />
            </section>

            {/* Bottom Row - Layer Types & Glossary */}
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "1.25rem", alignItems: "start" }}>
              {/* Layer Types */}
              <div style={{ background: "rgba(12,24,38,0.35)", border: "1px solid var(--panel-border)", borderRadius: "0.75rem", padding: "1.1rem" }}>
                <h3 style={{ margin: "0 0 0.75rem", fontSize: "0.95rem", color: "var(--text)" }}>Deep Learning Layer Types Guide</h3>
                <div style={{ display: "flex", flexDirection: "column", gap: "0.75rem" }}>
                  {LAYER_COMPARISON.map((layer, i) => (
                    <div key={i} style={{ background: "rgba(8,16,28,0.45)", border: "1px solid " + (layer.used ? "rgba(139,92,246,0.25)" : "rgba(148,163,184,0.08)"), borderRadius: "0.55rem", padding: "0.8rem 1rem" }}>
                      <div style={{ display: "flex", alignItems: "center", gap: "0.5rem", marginBottom: "0.4rem" }}>
                        <span style={{ fontSize: "1.15rem", color: layer.color }}>{layer.icon}</span>
                        <strong style={{ fontSize: "0.85rem", color: layer.used ? "var(--text)" : "var(--muted)" }}>{layer.name}</strong>
                        {layer.used ? (
                          <span style={{ fontSize: "0.58rem", padding: "0.08rem 0.4rem", borderRadius: "999px", background: "rgba(74,222,128,0.12)", border: "1px solid rgba(74,222,128,0.35)", color: "#4ade80", fontWeight: "700" }}>ACTIVE</span>
                        ) : (
                          <span style={{ fontSize: "0.58rem", padding: "0.08rem 0.4rem", borderRadius: "999px", background: "rgba(148,163,184,0.06)", border: "1px solid rgba(148,163,184,0.15)", color: "var(--muted)" }}>NOT IN CONFIG</span>
                        )}
                      </div>
                      <div style={{ fontSize: "0.76rem", color: "var(--muted)", lineHeight: "1.45" }}>
                        <strong style={{ color: "var(--text)" }}>Strength:</strong> {layer.strength}<br />
                        <strong style={{ color: "var(--text)" }}>When to Use:</strong> {layer.whenToUse}
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Concepts Glossary */}
              <div style={{ background: "rgba(12,24,38,0.35)", border: "1px solid var(--panel-border)", borderRadius: "0.75rem", padding: "1.1rem", display: "flex", flexDirection: "column", height: "100%", maxHeight: "580px" }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "0.75rem", gap: "0.5rem" }}>
                  <h3 style={{ margin: 0, fontSize: "0.95rem", color: "var(--text)" }}>Reinforcement Learning Glossary</h3>
                  <input
                    type="text"
                    placeholder="Search terminology..."
                    value={glossaryQuery}
                    onChange={(e) => setGlossaryQuery(e.target.value)}
                    style={{
                      borderRadius: "0.4rem",
                      border: "1px solid var(--panel-border)",
                      background: "rgba(8,15,25,0.7)",
                      color: "var(--text)",
                      padding: "0.3rem 0.6rem",
                      fontSize: "0.75rem",
                      width: "160px"
                    }}
                  />
                </div>

                <div style={{ flex: 1, overflowY: "auto", display: "flex", flexDirection: "column", gap: "0.55rem", paddingRight: "0.2rem" }}>
                  {filteredGlossary.map((entry, i) => (
                    <div key={i} style={{ background: "rgba(8,16,28,0.45)", border: "1px solid rgba(148,163,184,0.08)", borderRadius: "0.55rem", padding: "0.75rem 0.9rem" }}>
                      <div style={{ display: "flex", alignItems: "center", gap: "0.45rem", marginBottom: "0.25rem", flexWrap: "wrap" }}>
                        <strong style={{ fontSize: "0.85rem" }}>{entry.term}</strong>
                        <span style={{ fontSize: "0.58rem", padding: "0.08rem 0.35rem", borderRadius: "999px", background: entry.tagColor + "15", border: "1px solid " + entry.tagColor + "30", color: entry.tagColor, fontWeight: "600", letterSpacing: "0.02em" }}>{entry.tag}</span>
                      </div>
                      <p style={{ margin: "0 0 0.35rem", fontSize: "0.76rem", color: "var(--muted)", lineHeight: "1.4" }}>{entry.definition}</p>
                      <div style={{ fontSize: "0.7rem", color: "var(--accent)", background: "rgba(244,197,66,0.04)", borderRadius: "0.3rem", padding: "0.2rem 0.5rem", borderLeft: "2px solid rgba(244,197,66,0.3)" }}>
                        <strong>Usage here:</strong> {entry.usedHere}
                      </div>
                    </div>
                  ))}
                  {filteredGlossary.length === 0 && (
                    <div style={{ textAlign: "center", padding: "2rem", color: "var(--muted)", fontSize: "0.8rem" }}>No matching terms found.</div>
                  )}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* ENTRY (INPUT) TAB */}
        {activeSubTab === "entry" && (
          <div style={{ display: "flex", flexDirection: "column", gap: "1.25rem" }}>
            {/* Context boxes grid */}
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: "0.85rem" }}>
              <SectionBox accent="#3b82f6" title="Observation vector — Senses">
                Raw environments measure metrics like coordinate positions, agent values, teammate coordinates, 
                and role vectors. These are transformed into an array of floats called the <strong>Observation Vector</strong> 
                and sent to the network. Every index maps to an exact sensor node.
              </SectionBox>
              <SectionBox accent="#f59e0b" title="Why input normalization?">
                Raw coordinates or scalar pixels vary wildly. Stacking disparate values causes gradients to explode or vanish, 
                stalling optimization. Dividing offsets by the field size boundaries normalizes all features to roughly 
                equal ranges for stable, consistent learning.
              </SectionBox>
              <SectionBox accent="#8b5cf6" title="Design Decisions: Vector vs Pixels">
                Instead of expensive pixel streams, we feed hand-crafted state coordinates. 
                This design achieves convergence in hours instead of days, 
                but assumes perfect coordinate reporting from the RL environment.
              </SectionBox>
            </div>

            {/* Grid for Selector and Feature Table */}
            <div style={{ background: "rgba(12,24,38,0.4)", border: "1px solid var(--panel-border)", borderRadius: "0.75rem", padding: "1.1rem", display: "flex", flexDirection: "column" }}>
              {/* Controls header */}
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: "0.85rem", borderBottom: "1px solid var(--panel-border)", paddingBottom: "0.75rem", marginBottom: "0.85rem" }}>
                {/* View Mode segmented control */}
                <div style={{ display: "flex", gap: "0.35rem", background: "rgba(8,16,28,0.7)", padding: "0.25rem", borderRadius: "0.55rem", border: "1px solid var(--panel-border)" }}>
                  {["guided", "hierarchical", "emergent"].map(mode => {
                    const isSelected = selectedViewMode === mode;
                    return (
                      <button
                        key={mode}
                        onClick={() => setSelectedViewMode(mode)}
                        style={{
                          background: isSelected ? "var(--accent)" : "transparent",
                          border: "none",
                          color: isSelected ? "#000000" : "var(--muted)",
                          fontSize: "0.75rem",
                          fontWeight: "700",
                          padding: "0.3rem 0.8rem",
                          borderRadius: "0.4rem",
                          cursor: "pointer",
                          transform: "none",
                          display: "flex",
                          alignItems: "center",
                          gap: "0.25rem"
                        }}
                      >
                        {mode.toUpperCase()}
                        {activeModeFromConfig === mode && (
                          <span style={{ width: "5px", height: "5px", background: isSelected ? "#000000" : "var(--good)", borderRadius: "50%" }} />
                        )}
                      </button>
                    );
                  })}
                </div>

                {/* Search query */}
                <div style={{ display: "flex", gap: "0.5rem", flex: "1", maxWidth: "340px" }}>
                  <input
                    type="text"
                    placeholder="Search input features..."
                    value={searchQuery}
                    onChange={e => setSearchQuery(e.target.value)}
                    style={{
                      width: "100%",
                      borderRadius: "0.5rem",
                      border: "1px solid var(--panel-border)",
                      color: "var(--text)",
                      background: "rgba(8,15,25,0.8)",
                      padding: "0.4rem 0.75rem",
                      fontSize: "0.8rem"
                    }}
                  />
                </div>
              </div>

              {/* Category pills filter */}
              <div style={{ display: "flex", flexWrap: "wrap", gap: "0.35rem", marginBottom: "0.85rem" }}>
                <button
                  onClick={() => setSelectedCategory("all")}
                  style={{
                    padding: "0.15rem 0.55rem",
                    fontSize: "0.7rem",
                    borderRadius: "999px",
                    borderColor: selectedCategory === "all" ? "var(--accent)" : "rgba(148,163,184,0.15)",
                    background: selectedCategory === "all" ? "rgba(244,197,66,0.15)" : "rgba(12,24,38,0.4)",
                    color: selectedCategory === "all" ? "var(--accent)" : "var(--muted)",
                    cursor: "pointer",
                    transform: "none"
                  }}
                >
                  All categories
                </button>
                {Object.entries(CATEGORY_LABELS).map(([catKey, label]) => {
                  if (!activeNodesList.some(n => n.category === catKey)) return null;
                  const cc = CATEGORY_COLORS[catKey as NodeDefinition["category"]];
                  const isSelected = selectedCategory === catKey;
                  return (
                    <button
                      key={catKey}
                      onClick={() => setSelectedCategory(catKey)}
                      style={{
                        padding: "0.15rem 0.55rem",
                        fontSize: "0.7rem",
                        borderRadius: "999px",
                        borderColor: isSelected ? cc : "rgba(148,163,184,0.15)",
                        background: isSelected ? cc + "25" : "rgba(12,24,38,0.4)",
                        color: isSelected ? cc : "var(--muted)",
                        cursor: "pointer",
                        transform: "none"
                      }}
                    >
                      {label}
                    </button>
                  );
                })}
              </div>

              {/* Features list table wrapper with bounded height */}
              <div style={{ maxHeight: "380px", overflowY: "auto", border: "1px solid var(--panel-border)", borderRadius: "0.55rem" }}>
                <table style={{ width: "100%", borderCollapse: "collapse", textAlign: "left", fontSize: "0.8rem" }}>
                  <thead>
                    <tr style={{ background: "rgba(8,16,28,0.6)", borderBottom: "1px solid var(--panel-border)", position: "sticky", top: 0, zIndex: 10 }}>
                      <th style={{ padding: "0.6rem 0.8rem", width: "55px", color: "var(--muted)" }}>#</th>
                      <th style={{ padding: "0.6rem 0.8rem", width: "160px", color: "var(--text)" }}>Name</th>
                      <th style={{ padding: "0.6rem 0.8rem", width: "160px", color: "var(--text)" }}>Category</th>
                      <th style={{ padding: "0.6rem 0.8rem", color: "var(--text)" }}>Description &amp; Normalization Scale</th>
                    </tr>
                  </thead>
                  <tbody>
                    {filteredNodesList.length === 0 ? (
                      <tr>
                        <td colSpan={4} style={{ textAlign: "center", padding: "2.5rem", color: "var(--muted)" }}>
                          No features match the criteria.
                        </td>
                      </tr>
                    ) : (
                      filteredNodesList.map(node => {
                        const cc = CATEGORY_COLORS[node.category];
                        return (
                          <tr key={node.index} style={{ borderBottom: "1px solid rgba(148,163,184,0.06)", transition: "background 100ms ease" }} className="layer-row-hover">
                            <td style={{ padding: "0.6rem 0.8rem", fontWeight: "700", color: "var(--muted)", fontFamily: "monospace" }}>#{node.index}</td>
                            <td style={{ padding: "0.6rem 0.8rem", fontWeight: "600", fontFamily: "monospace", color: cc }}>{node.name}</td>
                            <td style={{ padding: "0.6rem 0.8rem" }}>
                              <span style={{ display: "inline-block", padding: "0.1rem 0.4rem", borderRadius: "4px", fontSize: "0.65rem", fontWeight: "600", color: cc, background: cc + "15", border: "1px solid " + cc + "30" }}>
                                {CATEGORY_LABELS[node.category]}
                              </span>
                            </td>
                            <td style={{ padding: "0.6rem 0.8rem" }}>
                              <div style={{ fontWeight: "600", color: "var(--text)" }}>{node.description}</div>
                              <div style={{ fontSize: "0.72rem", color: "var(--muted)", fontStyle: "italic", marginTop: "0.15rem" }}>{node.details}</div>
                            </td>
                          </tr>
                        );
                      })
                    )}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}

        {/* HIDDEN TAB */}
        {activeSubTab === "hidden" && (
          <div style={{ display: "flex", flexDirection: "column", gap: "1.25rem" }}>
            {/* Info cards */}
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: "0.85rem" }}>
              <SectionBox accent="#8b5cf6" title="Hidden Layers role">
                A fully connected linear network transforms inputs through matrix weights. Each hidden layer is a 
                consecutive transformation step, computing high-level features like relative flock position, teammate roles, and strategic coordinates.
              </SectionBox>
              <SectionBox accent="#10b981" title="Activation function: ReLU">
                Stacking linear layers yields only a single linear representation without activation. The 
                <strong>Rectified Linear Unit (ReLU)</strong> introduces crucial non-linearity by capping negative pre-activations to 0: f(x) = max(0, x).
              </SectionBox>
              <SectionBox accent="#3b82f6" title="Parameter Math & Updates">
                Each layer maps I input features to O output features. 
                Weights count: I × O; bias vector size: O. Total parameters: I × O + O. 
                These parameter floats are trained by standard gradient step updates.
              </SectionBox>
            </div>

            {/* Hidden Layers List */}
            <div style={{ display: "flex", flexDirection: "column", gap: "0.75rem" }}>
              <h3 style={{ margin: 0, fontSize: "0.95rem", color: "var(--text)" }}>Network Configuration Details</h3>
              {hiddenLayerSizes.map((size, i) => {
                const prevSize = i === 0 ? activeNodesList.length : hiddenLayerSizes[i - 1];
                const params = prevSize * size + size;
                const purpose = i === 0 
                  ? "Learns basic feature abstractions directly from the environment senses (coordinate differences, distances)." 
                  : i === hiddenLayerSizes.length - 1 
                  ? "Constructs abstract representation states directly upstream of Actor policies and Critic evaluations." 
                  : "Transforms and compresses features to identify coordination relationships.";

                return (
                  <div key={i} style={{ background: "rgba(12,24,38,0.5)", border: "1px solid rgba(139,92,246,0.25)", borderRadius: "0.75rem", padding: "1.1rem" }}>
                    <div style={{ display: "grid", gridTemplateColumns: "auto 1fr auto auto", gap: "1rem", alignItems: "center", borderBottom: "1px solid rgba(148,163,184,0.06)", paddingBottom: "0.6rem", marginBottom: "0.75rem" }}>
                      <div style={{ width: "2rem", height: "2rem", borderRadius: "0.4rem", background: "rgba(139,92,246,0.15)", border: "1px solid rgba(139,92,246,0.35)", display: "flex", alignItems: "center", justifyContent: "center", fontWeight: "700", color: "#8b5cf6" }}>
                        H{i + 1}
                      </div>
                      <div>
                        <div style={{ fontWeight: "700", fontSize: "0.95rem" }}>
                          Hidden Layer {i + 1}
                          {i === hiddenLayerSizes.length - 1 && (
                            <span style={{ marginLeft: "0.5rem", fontSize: "0.6rem", background: "rgba(244,197,66,0.12)", color: "#f4c542", border: "1px solid rgba(244,197,66,0.3)", padding: "0.1rem 0.4rem", borderRadius: "999px", fontWeight: "600", verticalAlign: "middle" }}>OUTPUT ADJACENT</span>
                          )}
                        </div>
                        <div style={{ fontSize: "0.72rem", color: "var(--muted)", marginTop: "0.1rem" }}>Linear({prevSize} &rarr; {size}) + ReLU</div>
                      </div>
                      <div style={{ textAlign: "right" }}>
                        <div style={{ fontSize: "0.6rem", color: "var(--muted)", textTransform: "uppercase" }}>Neurons</div>
                        <strong style={{ fontSize: "1rem", color: "#8b5cf6" }}>{size}</strong>
                      </div>
                      <div style={{ textAlign: "right" }}>
                        <div style={{ fontSize: "0.6rem", color: "var(--muted)", textTransform: "uppercase" }}>Parameters</div>
                        <strong style={{ fontSize: "1rem", color: "#93c5fd" }}>{formatNumber(params)}</strong>
                      </div>
                    </div>

                    <div style={{ display: "flex", flexWrap: "wrap", gap: "0.4rem", marginBottom: "0.75rem" }}>
                      <span style={{ background: "rgba(59,130,246,0.1)", border: "1px solid rgba(59,130,246,0.25)", color: "#93c5fd", padding: "0.2rem 0.5rem", borderRadius: "0.35rem", fontFamily: "monospace", fontSize: "0.7rem" }}>{prevSize} inputs</span>
                      <span style={{ background: "rgba(139,92,246,0.08)", border: "1px solid rgba(139,92,246,0.25)", color: "#c084fc", padding: "0.2rem 0.5rem", borderRadius: "0.35rem", fontFamily: "monospace", fontSize: "0.7rem" }}>&times; {prevSize * size} weights</span>
                      <span style={{ background: "rgba(139,92,246,0.08)", border: "1px solid rgba(139,92,246,0.2)", color: "#c084fc", padding: "0.2rem 0.5rem", borderRadius: "0.35rem", fontFamily: "monospace", fontSize: "0.7rem" }}>+ {size} biases</span>
                      <span style={{ background: "rgba(16,185,129,0.08)", border: "1px solid rgba(16,185,129,0.25)", color: "#6ee7b7", padding: "0.2rem 0.5rem", borderRadius: "0.35rem", fontFamily: "monospace", fontSize: "0.7rem" }}>&rarr; ReLU({size}) activation</span>
                    </div>

                    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0.75rem", fontSize: "0.78rem" }}>
                      <div style={{ background: "rgba(8,16,28,0.4)", borderRadius: "0.4rem", padding: "0.6rem 0.8rem", borderLeft: "2px solid #8b5cf6" }}>
                        <div style={{ color: "#8b5cf6", fontWeight: "600", fontSize: "0.65rem", textTransform: "uppercase", marginBottom: "0.15rem" }}>Strategic Role</div>
                        <div style={{ color: "var(--muted)", lineHeight: "1.4" }}>{purpose}</div>
                      </div>
                      <div style={{ background: "rgba(8,16,28,0.4)", borderRadius: "0.4rem", padding: "0.6rem 0.8rem", borderLeft: "2px solid #10b981" }}>
                        <div style={{ color: "#10b981", fontWeight: "600", fontSize: "0.65rem", textTransform: "uppercase", marginBottom: "0.15rem" }}>Activation Pipeline</div>
                        <div style={{ color: "var(--muted)", lineHeight: "1.4" }}>Zeros out negative pre-activation weights, forcing sparse outputs and preventing saturated gradient nodes.</div>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* EXIT (OUTPUT) TAB */}
        {activeSubTab === "exit" && (
          <div style={{ display: "flex", flexDirection: "column", gap: "1.25rem" }}>
            {/* Info cards */}
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: "0.85rem" }}>
              <SectionBox accent="#818cf8" title="What is the Actor Head?">
                The Actor network layer processes the final hidden representation to output action scores (logits). 
                These logits are passed to Softmax, generating a probability distribution to sample a discrete movement action.
              </SectionBox>
              <SectionBox accent="#10b981" title="What is the Critic Head?">
                The Critic head outputs a single unbounded state value prediction: V(s). It evaluates how advantageous 
                the current configuration is. Used in GAE computation, but omitted during active model evaluation/deployment.
              </SectionBox>
              <SectionBox accent="#ec4899" title="Action Masking Logic">
                Logits for invalid actions (like moving into static field boundaries) are set to -infinity prior to Softmax. 
                This results in a 0% action selection probability, optimizing training speed.
              </SectionBox>
            </div>

            {/* Actor and Critic details layout grid */}
            <div style={{ display: "grid", gridTemplateColumns: "1.6fr 1fr", gap: "1.25rem", alignItems: "start" }}>
              {/* Actor actions details */}
              <div style={{ background: "rgba(12,24,38,0.4)", border: "1px solid var(--panel-border)", borderRadius: "0.75rem", padding: "1.1rem" }}>
                <h3 style={{ margin: "0 0 0.2rem", fontSize: "0.95rem", color: "#818cf8" }}>Actor Output Actions ({ACTOR_NODES.length})</h3>
                <p style={{ fontSize: "0.75rem", color: "var(--muted)", margin: "0 0 0.85rem" }}>
                  The model outputs discrete actions. Exponents are scaled via Softmax logic after action masks filters are processed.
                </p>

                <div style={{ display: "flex", flexDirection: "column", gap: "0.45rem", maxHeight: "360px", overflowY: "auto", paddingRight: "0.2rem" }}>
                  {ACTOR_NODES.map(act => (
                    <div key={act.index} style={{ background: "rgba(8,15,25,0.45)", border: "1px solid rgba(129,140,248,0.15)", borderRadius: "0.55rem", padding: "0.6rem 0.85rem", display: "grid", gridTemplateColumns: "30px 100px 1fr", gap: "0.5rem", alignItems: "center" }}>
                      <span style={{ fontWeight: "700", color: "var(--muted)", fontSize: "0.75rem", fontFamily: "monospace" }}>#{act.index}</span>
                      <span style={{ fontFamily: "monospace", fontWeight: "700", fontSize: "0.82rem", color: "#818cf8" }}>{act.name}</span>
                      <div style={{ fontSize: "0.76rem" }}>
                        <div style={{ display: "flex", alignItems: "center", gap: "0.4rem" }}>
                          <span style={{ fontWeight: "600", color: "var(--text)" }}>{act.description}</span>
                          <span style={{
                            fontSize: "0.58rem",
                            padding: "0.05rem 0.35rem",
                            borderRadius: "999px",
                            background: act.type === "Sprint" ? "rgba(244,197,66,0.12)" : act.type === "Hold" ? "rgba(148,163,184,0.06)" : "rgba(59,130,246,0.1)",
                            color: act.type === "Sprint" ? "#f4c542" : act.type === "Hold" ? "var(--muted)" : "#93c5fd",
                            fontWeight: "700"
                          }}>
                            {act.type.toUpperCase()}
                          </span>
                        </div>
                        <div style={{ color: "var(--muted)", fontSize: "0.68rem", marginTop: "0.15rem" }}>{act.communicating}</div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Critic Head and Action masking details */}
              <div style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>
                {/* Critic Head Info */}
                <div style={{ background: "rgba(12,24,38,0.4)", border: "1px solid rgba(16,185,129,0.25)", borderRadius: "0.75rem", padding: "1.1rem" }}>
                  <h3 style={{ margin: "0 0 0.5rem", fontSize: "0.95rem", color: "#10b981" }}>Critic Evaluator Value V(s)</h3>
                  <div style={{ background: "rgba(8,15,25,0.5)", border: "1px solid rgba(16,185,129,0.15)", borderRadius: "0.45rem", padding: "0.75rem 0.9rem" }}>
                    <div style={{ display: "flex", alignItems: "center", gap: "0.45rem", marginBottom: "0.4rem" }}>
                      <span style={{ fontFamily: "monospace", fontWeight: "700", fontSize: "1.1rem", color: "#10b981" }}>V(s)</span>
                      <span style={{ fontSize: "0.58rem", padding: "0.08rem 0.35rem", borderRadius: "999px", background: "rgba(16,185,129,0.1)", color: "#10b981", fontWeight: "600" }}>SCALAR</span>
                      <span style={{ fontSize: "0.58rem", padding: "0.08rem 0.35rem", borderRadius: "999px", background: "rgba(148,163,184,0.06)", color: "var(--muted)" }}>NO ACTIVATION</span>
                    </div>
                    <div style={{ fontSize: "0.8rem", color: "var(--text)", fontWeight: "600", marginBottom: "0.15rem" }}>
                      Expected discounted cumulative return.
                    </div>
                    <p style={{ margin: 0, fontSize: "0.72rem", color: "var(--muted)", lineHeight: "1.4" }}>
                      Evaluates current state s: V(s) = wᵀ · h + b. Higher values indicate better states. 
                      Enables Advantage estimation during learning phases.
                    </p>
                  </div>
                </div>

                {/* Pipeline overview note */}
                <div style={{ background: "rgba(244,197,66,0.02)", border: "1px solid rgba(244,197,66,0.15)", borderRadius: "0.75rem", padding: "1.1rem" }}>
                  <h4 style={{ margin: "0 0 0.3rem", color: "var(--accent)", fontSize: "0.8rem", textTransform: "uppercase", letterSpacing: "0.05em" }}>Action Masking Pipeline</h4>
                  <p style={{ fontSize: "0.72rem", color: "var(--muted)", margin: 0, lineHeight: "1.45" }}>
                    Logits are intercepted prior to Softmax operations. Masking matrices index invalid movement states 
                    (e.g., collisions) and overwrite their pre-activation logits to -infinity. 
                    This guarantees e^(-infinity) = 0, ensuring invalid actions are never sampled.
                  </p>
                </div>
              </div>
            </div>
          </div>
        )}

      </div>
    </div>
  );
}

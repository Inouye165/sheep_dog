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

// ============================================================
// LAYER DETAIL MODAL — opens when any architecture box is clicked
// ============================================================

type LayerKey = "input" | { hidden: number } | "actor" | "critic";

interface LayerDetailModalProps {
  layerKey: LayerKey;
  nodes: NodeDefinition[];        // active input nodes (for input modal)
  hiddenSizes: number[];
  onClose: () => void;
}

function NodeTable({ nodes }: { nodes: (NodeDefinition | OutputDefinition)[] }) {
  const [q, setQ] = useState("");
  const [cat, setCat] = useState("all");
  const isInput = "category" in nodes[0];
  const filtered = nodes.filter((n) => {
    const text = n.name + " " + n.description + ("details" in n ? n.details : "") + ("communicating" in n ? n.communicating : "");
    const matchQ = text.toLowerCase().includes(q.toLowerCase());
    const matchC = cat === "all" || ("category" in n && (n as NodeDefinition).category === cat);
    return matchQ && matchC;
  });

  const cats = isInput ? Array.from(new Set((nodes as NodeDefinition[]).map(n => n.category))) : [];

  return (
    <div>
      <div style={{ display: "flex", gap: "0.5rem", marginBottom: "0.75rem", flexWrap: "wrap" }}>
        <input value={q} onChange={e => setQ(e.target.value)} placeholder="Search nodes..." style={{ flex: 1, minWidth: "140px", borderRadius: "0.4rem", border: "1px solid rgba(148,163,184,0.25)", background: "rgba(8,15,25,0.7)", color: "var(--text)", padding: "0.4rem 0.7rem", fontSize: "0.8rem" }} />
        {isInput && (
          <select value={cat} onChange={e => setCat(e.target.value)} style={{ borderRadius: "0.4rem", border: "1px solid rgba(148,163,184,0.25)", background: "rgba(8,15,25,0.7)", color: "var(--text)", padding: "0.4rem 0.6rem", fontSize: "0.8rem" }}>
            <option value="all">All categories</option>
            {cats.map(c => <option key={c} value={c}>{CATEGORY_LABELS[c as NodeDefinition["category"]]}</option>)}
          </select>
        )}
      </div>
      <div style={{ overflowY: "auto", maxHeight: "340px", scrollbarWidth: "thin", scrollbarColor: "rgba(148,163,184,0.3) transparent" }}>
        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "0.8rem" }}>
          <thead>
            <tr style={{ borderBottom: "1px solid rgba(148,163,184,0.15)" }}>
              <th style={{ padding: "0.5rem 0.4rem", color: "var(--muted)", width: "50px", textAlign: "left" }}>#</th>
              <th style={{ padding: "0.5rem 0.4rem", textAlign: "left" }}>Name</th>
              {isInput && <th style={{ padding: "0.5rem 0.4rem", textAlign: "left" }}>Category</th>}
              <th style={{ padding: "0.5rem 0.4rem", textAlign: "left" }}>Description</th>
              <th style={{ padding: "0.5rem 0.4rem", color: "var(--muted)", textAlign: "left" }}>{isInput ? "Normalization" : "Role"}</th>
            </tr>
          </thead>
          <tbody>
            {filtered.length === 0 && <tr><td colSpan={5} style={{ textAlign: "center", padding: "1.5rem", color: "var(--muted)" }}>No nodes match.</td></tr>}
            {filtered.map(n => {
              const isNode = "category" in n;
              const nd = n as NodeDefinition;
              const od = n as OutputDefinition;
              const cc = isNode ? CATEGORY_COLORS[nd.category] : "#818cf8";
              return (
                <tr key={n.index} style={{ borderBottom: "1px solid rgba(148,163,184,0.06)" }}>
                  <td style={{ padding: "0.55rem 0.4rem", fontWeight: "700", color: "var(--muted)", fontFamily: "monospace" }}>#{n.index}</td>
                  <td style={{ padding: "0.55rem 0.4rem", fontFamily: "monospace", fontWeight: "600", color: cc }}>{n.name}</td>
                  {isNode && <td style={{ padding: "0.55rem 0.4rem" }}><span style={{ fontSize: "0.68rem", padding: "0.1rem 0.45rem", borderRadius: "4px", color: cc, background: cc + "18", border: "1px solid " + cc + "40" }}>{CATEGORY_LABELS[nd.category]}</span></td>}
                  <td style={{ padding: "0.55rem 0.4rem", color: "var(--text)" }}>{n.description}</td>
                  <td style={{ padding: "0.55rem 0.4rem", color: "var(--muted)", fontSize: "0.72rem", fontStyle: "italic" }}>{isNode ? nd.details : od.communicating}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <div style={{ fontSize: "0.72rem", color: "var(--muted)", marginTop: "0.5rem" }}>{filtered.length} of {nodes.length} nodes shown</div>
    </div>
  );
}

function Pill({ label, color, bg }: { label: string; color: string; bg: string }) {
  return <span style={{ fontSize: "0.68rem", padding: "0.15rem 0.55rem", borderRadius: "999px", background: bg, border: "1px solid " + color + "50", color, fontWeight: "600", textTransform: "uppercase" as const, letterSpacing: "0.05em" }}>{label}</span>;
}
function SectionBox({ accent, title, children }: { accent: string; title: string; children: React.ReactNode }) {
  return (
    <div style={{ background: "rgba(8,15,25,0.45)", borderLeft: "3px solid " + accent, borderRadius: "0.5rem", padding: "0.9rem 1rem", marginBottom: "1rem" }}>
      <div style={{ fontSize: "0.68rem", color: accent, fontWeight: "700", textTransform: "uppercase" as const, letterSpacing: "0.07em", marginBottom: "0.4rem" }}>{title}</div>
      <div style={{ fontSize: "0.83rem", color: "var(--muted)", lineHeight: "1.6" }}>{children}</div>
    </div>
  );
}

function InputLayerModal({ nodes }: { nodes: NodeDefinition[] }) {
  return (
    <div>
      <div style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap", marginBottom: "1.25rem" }}>
        <Pill label={nodes.length + " nodes"} color="#3b82f6" bg="rgba(59,130,246,0.12)" />
        <Pill label="fed every step" color="#10b981" bg="rgba(16,185,129,0.1)" />
        <Pill label="all values in [0, 1]" color="#f59e0b" bg="rgba(245,158,11,0.1)" />
      </div>
      <SectionBox accent="#3b82f6" title="For beginners — What is the input layer?">
        Think of the input layer as the dog&apos;s <strong style={{ color: "var(--text)" }}>senses</strong>. Before the agent takes any action, the environment measures everything it can observe — where the sheep are, where the pen is, how far away things are, which role the dog has — and writes each measurement as a number between 0 and 1. These numbers are written side by side into a list called the <em>observation vector</em>, and that list is what the neural network receives as its input.<br /><br />
        The network cannot see images or hear sounds. It only reads this vector of numbers. Every entry in the table below is one &quot;sense&quot; — one slot in that list. The order matters: the network always expects the same feature at the same position.
      </SectionBox>
      <SectionBox accent="#f59e0b" title="Why are all values normalized to [0, 1]?">
        Raw values like &quot;dog is at X=45 on a 60-wide field&quot; or &quot;distance is 28 cells&quot; vary wildly in scale. If we fed raw numbers directly, neurons connected to large-valued inputs would dominate the learning signal and training would be unstable. Dividing by the field size or field diagonal squashes everything into [0, 1] so every input has roughly equal influence at the start of training. This is called <strong style={{ color: "var(--text)" }}>normalization</strong>.
      </SectionBox>
      <SectionBox accent="#8b5cf6" title="For experienced readers — observation space design">
        This is a <strong style={{ color: "var(--text)" }}>hand-crafted observation space</strong> — each feature was deliberately chosen and computed by the environment rather than learned from raw pixels. The design tradeoffs are: (1) fewer parameters needed vs. a CNN, (2) faster convergence, (3) requires domain knowledge to design well, (4) may miss emergent features a CNN could discover. The <em>Guided</em> mode includes role assignments as one-hot flags, giving the policy explicit role context. <em>Emergent</em> mode strips roles out, forcing the policy to discover coordination patterns without hints.
      </SectionBox>
      <div style={{ borderTop: "1px solid rgba(148,163,184,0.12)", paddingTop: "1rem", marginTop: "0.25rem" }}>
        <h4 style={{ margin: "0 0 0.75rem", fontSize: "0.9rem" }}>All input nodes</h4>
        <NodeTable nodes={nodes} />
      </div>
    </div>
  );
}

function HiddenLayerModal({ layerIndex, inputSize, outputSize, isLast, totalLayers }: { layerIndex: number; inputSize: number; outputSize: number; isLast: boolean; totalLayers: number; }) {
  const params = inputSize * outputSize + outputSize;
  const purposeByIndex: Record<number, string> = {
    0: "First hidden layer — this is where raw input features get their first transformation. The network learns basic combinations: \"how far am I from the sheep?\" \"am I near a wall?\" \"is the flock spread out?\" — patterns that are one step above the raw inputs.",
    1: "Second hidden layer — builds on the first layer's abstractions. Here the network learns higher-level patterns like \"the flock is between me and the pen\" or \"my teammate is already covering the left flank, so I should take the right\".",
  };
  const purpose = purposeByIndex[layerIndex] ?? (isLast ? "Final hidden layer — immediately upstream of both output heads. Neurons here encode a compact, abstract representation of the full situation. Both the Actor (action choice) and Critic (value estimate) read from this same layer." : "Intermediate hidden layer — continues building increasingly abstract representations from the previous layer's output.");

  return (
    <div>
      <div style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap", marginBottom: "1.25rem" }}>
        <Pill label={outputSize + " neurons"} color="#818cf8" bg="rgba(129,140,248,0.12)" />
        <Pill label={formatNumber(params) + " parameters"} color="#93c5fd" bg="rgba(147,197,253,0.1)" />
        <Pill label={"Layer " + (layerIndex + 1) + " of " + totalLayers} color="#f4c542" bg="rgba(244,197,66,0.1)" />
        {isLast && <Pill label="feeds both output heads" color="#6ee7b7" bg="rgba(110,231,183,0.1)" />}
      </div>

      <SectionBox accent="#818cf8" title="For beginners — What does a hidden layer do?">
        A hidden layer is like a team of detectives. Each neuron in the layer reads <em>all</em> {inputSize} numbers from the previous layer, gives each one a weight (importance), adds them all together, and produces a single number as its conclusion. With {outputSize} neurons, the layer produces {outputSize} such conclusions simultaneously.<br /><br />
        These conclusions are then passed to the next layer, which does the same thing again — forming opinions about opinions. By the time information reaches the output, the network has had {totalLayers} rounds of reasoning to transform raw sensor readings into a strategic decision.
      </SectionBox>

      <SectionBox accent="#10b981" title="Activation: ReLU — f(x) = max(0, x)">
        After each neuron computes its weighted sum, the result is passed through <strong style={{ color: "var(--text)" }}>ReLU</strong>. If the sum is negative, ReLU outputs zero — the neuron is &quot;off&quot;. If positive, it passes the value straight through — the neuron is &quot;on&quot;.<br /><br />
        <strong style={{ color: "var(--text)" }}>Why not just use a straight line?</strong> Without an activation function, stacking multiple linear layers is mathematically identical to a single linear layer. ReLU introduces the non-linearity that allows deep networks to approximate arbitrarily complex functions. It was chosen over Sigmoid/Tanh because it: (1) doesn&apos;t saturate for large positive inputs, avoiding vanishing gradients; (2) is computationally trivial; (3) produces sparse activations (many zeros), which acts as natural regularization.
      </SectionBox>

      <SectionBox accent="#f4c542" title={"What layer " + (layerIndex + 1) + " specifically learns"}>
        {purpose}
      </SectionBox>

      <SectionBox accent="#93c5fd" title="For experienced readers — parameter math">
        This layer has <strong style={{ color: "var(--text)" }}>{inputSize} inputs</strong> and <strong style={{ color: "var(--text)" }}>{outputSize} outputs</strong>.<br />
        Weight matrix: {inputSize} &times; {outputSize} = {(inputSize * outputSize).toLocaleString()} learnable weights<br />
        Bias vector: {outputSize} biases<br />
        <strong style={{ color: "var(--text)" }}>Total: {params.toLocaleString()} parameters</strong> adjusted during every PPO update.<br /><br />
        Each parameter update step: loss &rarr; backprop &rarr; gradients &rarr; Adam optimizer step &rarr; weights shift slightly in the direction that reduces the PPO clipped surrogate loss.
      </SectionBox>
    </div>
  );
}

function ActorHeadModal() {
  return (
    <div>
      <div style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap", marginBottom: "1.25rem" }}>
        <Pill label={ACTOR_NODES.length + " output logits"} color="#818cf8" bg="rgba(129,140,248,0.12)" />
        <Pill label="Softmax + Action Masking" color="#ec4899" bg="rgba(236,72,153,0.1)" />
        <Pill label="Discrete action space" color="#f4c542" bg="rgba(244,197,66,0.1)" />
      </div>

      <SectionBox accent="#818cf8" title="For beginners — Why is it called the Actor?">
        In reinforcement learning, we split the neural network into two &quot;roles&quot;:<br /><br />
        <strong style={{ color: "var(--text)" }}>The Actor</strong> decides <em>what to do</em>. It reads the final hidden layer and outputs a score for each possible action. The action with the highest score (after adjusting for what&apos;s legal) gets chosen. Think of the actor as the part of the brain that moves the muscles — it produces the behaviour you actually see.<br /><br />
        <strong style={{ color: "var(--text)" }}>The Critic</strong> (the other head) judges <em>how good the situation is</em>. It doesn&apos;t choose actions — it gives the actor feedback during training.<br /><br />
        Both heads share the same hidden layers. They are just two different &quot;readout&quot; layers bolted onto the same backbone.
      </SectionBox>

      <SectionBox accent="#ec4899" title="What are logits and how does Softmax work?">
        The actor outputs raw numbers called <strong style={{ color: "var(--text)" }}>logits</strong> — one per action. A logit can be any real number: positive, negative, large, small. By themselves they are meaningless as probabilities.<br /><br />
        <strong style={{ color: "var(--text)" }}>Softmax</strong> converts logits to probabilities: it exponentiates each logit (e<sup>x</sup>) and divides by the sum of all exponentiated logits. The result is a probability distribution that sums to 1. A logit of +5 becomes a much higher probability than -1.<br /><br />
        <strong style={{ color: "var(--text)" }}>Action masking</strong> happens before Softmax: logits for illegal moves (e.g. walking into a wall) are set to &minus;&infin;, so their Softmax output is exactly 0%. This means the network never wastes gradient on impossible actions.
      </SectionBox>

      <SectionBox accent="#f4c542" title="For experienced readers — training the actor">
        The actor is trained using PPO&apos;s clipped surrogate objective: L<sup>CLIP</sup> = E[min(r_t &middot; A_t, clip(r_t, 1-&epsilon;, 1+&epsilon;) &middot; A_t)], where r_t is the probability ratio (new policy / old policy) and A_t is the advantage estimated by GAE. The clip prevents large policy jumps. The advantage A_t comes from the critic — positive advantage means &quot;this action was better than average, reinforce it&quot;; negative means &quot;worse than average, suppress it&quot;.
      </SectionBox>

      <div style={{ borderTop: "1px solid rgba(148,163,184,0.12)", paddingTop: "1rem", marginTop: "0.25rem" }}>
        <h4 style={{ margin: "0 0 0.75rem", fontSize: "0.9rem" }}>All {ACTOR_NODES.length} output actions</h4>
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: "0.5rem" }}>
          {ACTOR_NODES.map(act => (
            <div key={act.index} style={{ background: "rgba(8,15,25,0.5)", border: "1px solid rgba(129,140,248,0.2)", borderRadius: "0.5rem", padding: "0.65rem 0.8rem" }}>
              <div style={{ display: "flex", alignItems: "center", gap: "0.4rem", marginBottom: "0.3rem" }}>
                <span style={{ fontWeight: "700", color: "var(--muted)", fontSize: "0.7rem", fontFamily: "monospace" }}>#{act.index}</span>
                <span style={{ fontFamily: "monospace", fontWeight: "700", color: "#818cf8", fontSize: "0.85rem" }}>{act.name}</span>
                <span style={{ marginLeft: "auto", fontSize: "0.62rem", padding: "0.05rem 0.35rem", borderRadius: "999px", background: act.type === "Sprint" ? "rgba(244,197,66,0.12)" : act.type === "Hold" ? "rgba(148,163,184,0.08)" : "rgba(59,130,246,0.1)", color: act.type === "Sprint" ? "#f4c542" : act.type === "Hold" ? "var(--muted)" : "#93c5fd", border: "1px solid " + (act.type === "Sprint" ? "rgba(244,197,66,0.3)" : act.type === "Hold" ? "rgba(148,163,184,0.15)" : "rgba(59,130,246,0.3)"), fontWeight: "600" }}>{act.type}</span>
              </div>
              <div style={{ fontSize: "0.78rem", color: "var(--text)", marginBottom: "0.2rem" }}>{act.description}</div>
              <div style={{ fontSize: "0.7rem", color: "var(--muted)", lineHeight: "1.4" }}>{act.communicating}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function CriticHeadModal() {
  return (
    <div>
      <div style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap", marginBottom: "1.25rem" }}>
        <Pill label="1 scalar output" color="#6ee7b7" bg="rgba(110,231,183,0.1)" />
        <Pill label="V(s) — state value" color="#10b981" bg="rgba(16,185,129,0.1)" />
        <Pill label="Training only" color="#f4c542" bg="rgba(244,197,66,0.1)" />
        <Pill label="No activation" color="#64748b" bg="rgba(100,116,139,0.1)" />
      </div>

      <SectionBox accent="#6ee7b7" title="For beginners — Why is it called the Critic?">
        Imagine a sports coach (the critic) watching a player (the actor) make moves. The player decides what to do; the coach rates how good the <em>overall situation</em> is and gives feedback afterward. The coach doesn&apos;t make moves — they observe and judge.<br /><br />
        In RL, the <strong style={{ color: "var(--text)" }}>Critic</strong> outputs a single number called <strong style={{ color: "var(--text)" }}>V(s)</strong> — the &quot;value&quot; of the current state <em>s</em>. It is the network&apos;s prediction of the total future reward it will collect from this moment onward, assuming the current policy is followed. A high V(s) means &quot;we are in a good position — plenty of reward ahead&quot;. A low V(s) means &quot;we are struggling&quot;.<br /><br />
        <strong style={{ color: "var(--text)" }}>The critic is only used during training.</strong> At inference time (watching the dogs run), only the actor head matters — the critic output is ignored.
      </SectionBox>

      <SectionBox accent="#10b981" title="How V(s) drives learning — Advantage estimation">
        During training, after collecting a batch of experience (states, actions, rewards), the algorithm asks: &quot;Was that action <em>better</em> or <em>worse</em> than expected?&quot;<br /><br />
        The answer is the <strong style={{ color: "var(--text)" }}>advantage</strong>: A(s, a) = actual return &minus; V(s). If the dog took an action that led to more reward than V(s) predicted, A is positive — reinforce that action. If less reward than expected, A is negative — suppress it.<br /><br />
        <strong style={{ color: "var(--text)" }}>GAE (Generalized Advantage Estimation)</strong> refines this by blending multi-step returns with bootstrapped critic estimates, controlled by lambda (&lambda;). Higher lambda = more actual experience, less critic bias. Lower lambda = more critic, less variance.
      </SectionBox>

      <SectionBox accent="#f4c542" title="Why no activation function on V(s)?">
        Unlike the actor (which needs probabilities between 0 and 1), V(s) can be <em>any real number</em>. If an episode goes perfectly, the total reward might be +250. If it goes badly, it might be &minus;50. Putting Sigmoid or Tanh on the output would clip V(s) to a fixed range and make it impossible to represent high-value or low-value states accurately. The output neuron is therefore <strong style={{ color: "var(--text)" }}>linear</strong> — whatever the weighted sum is, that is V(s).
      </SectionBox>

      <SectionBox accent="#93c5fd" title="For experienced readers — critic loss">
        The critic is trained separately from the actor using a <strong style={{ color: "var(--text)" }}>value loss</strong>: L<sup>V</sup> = E[(V&theta;(s_t) &minus; V<sup>target</sup>_t)<sup>2</sup>], where V<sup>target</sup>_t is computed from the discounted return or TD target. PPO often clips the value function update similarly to the policy update to prevent instability. The &quot;value coefficient&quot; (c_v in PPO) scales the value loss relative to the policy loss during joint optimization.
      </SectionBox>

      <div style={{ borderTop: "1px solid rgba(148,163,184,0.12)", paddingTop: "1rem", marginTop: "0.25rem" }}>
        <h4 style={{ margin: "0 0 0.75rem", fontSize: "0.9rem" }}>Output</h4>
        <div style={{ background: "rgba(8,15,25,0.5)", border: "1px solid rgba(110,231,183,0.25)", borderRadius: "0.5rem", padding: "0.9rem 1rem" }}>
          <div style={{ display: "flex", alignItems: "center", gap: "0.6rem", marginBottom: "0.4rem" }}>
            <span style={{ fontFamily: "monospace", fontWeight: "700", fontSize: "1.1rem", color: "#6ee7b7" }}>V(s)</span>
            <Pill label="Unbounded scalar" color="#6ee7b7" bg="rgba(110,231,183,0.08)" />
            <Pill label="Linear output — no activation" color="#64748b" bg="rgba(100,116,139,0.08)" />
          </div>
          <div style={{ fontSize: "0.83rem", color: "var(--text)", marginBottom: "0.3rem" }}>Expected total future reward from this state onwards</div>
          <div style={{ fontSize: "0.77rem", color: "var(--muted)", lineHeight: "1.5" }}>
            Computed as: V&theta;(s) = w<sup>T</sup> &middot; h<sub>final</sub> + b, where h<sub>final</sub> is the final hidden layer output. Used exclusively during training to compute GAE advantages. Discarded at inference time.
          </div>
        </div>
      </div>
    </div>
  );
}

function LayerDetailModal({ layerKey, nodes, hiddenSizes, onClose }: LayerDetailModalProps) {
  const isInput = layerKey === "input";
  const isActor = layerKey === "actor";
  const isCritic = layerKey === "critic";
  const isHidden = typeof layerKey === "object" && "hidden" in layerKey;
  const hiddenIndex = isHidden ? (layerKey as { hidden: number }).hidden : 0;

  let title = "";
  let subtitle = "";
  let accentColor = "#3b82f6";
  let icon = "";

  if (isInput) { title = "Input Layer"; subtitle = "The agent senses — " + nodes.length + " observation features"; accentColor = "#3b82f6"; icon = "&#9654;"; }
  else if (isActor) { title = "Actor Head"; subtitle = "The decision maker — " + ACTOR_NODES.length + " action logits"; accentColor = "#818cf8"; icon = "&#9654;"; }
  else if (isCritic) { title = "Critic Head"; subtitle = "The evaluator — state value V(s)"; accentColor = "#6ee7b7"; icon = "&#9654;"; }
  else { title = "Hidden Layer " + (hiddenIndex + 1); const sz = hiddenSizes[hiddenIndex]; const inp = hiddenIndex === 0 ? nodes.length : hiddenSizes[hiddenIndex - 1]; subtitle = "Linear(" + inp + " \u2192 " + sz + ") + ReLU  \u00b7  " + formatNumber(inp * sz + sz) + " parameters"; accentColor = "#818cf8"; icon = "&#9650;"; }

  return (
    <div style={{ position: "fixed", inset: 0, zIndex: 1100, background: "rgba(4,8,15,0.88)", backdropFilter: "blur(10px)", display: "flex", alignItems: "center", justifyContent: "center", padding: "1rem" }} onClick={onClose}>
      <div style={{ background: "linear-gradient(160deg,rgba(10,20,35,0.99),rgba(6,12,22,0.99))", border: "1px solid " + accentColor + "40", borderRadius: "1.25rem", boxShadow: "0 0 0 1px " + accentColor + "20, 0 48px 140px rgba(0,0,0,0.75)", width: "min(860px,96vw)", maxHeight: "90vh", display: "flex", flexDirection: "column", overflow: "hidden" }} onClick={e => e.stopPropagation()}>
        {/* Header */}
        <div style={{ padding: "1.5rem 2rem 1.1rem", borderBottom: "1px solid " + accentColor + "25", background: "linear-gradient(135deg," + accentColor + "08, transparent)", flexShrink: 0 }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
            <div>
              <div style={{ fontSize: "0.68rem", textTransform: "uppercase" as const, letterSpacing: "0.12em", color: accentColor, fontWeight: "700", marginBottom: "0.3rem" }}>
                {isInput ? "Layer 0 — Input" : isActor ? "Output Head A" : isCritic ? "Output Head B" : "Layer " + (hiddenIndex + 1) + " — Hidden"}
              </div>
              <h2 style={{ margin: 0, fontSize: "1.6rem", lineHeight: 1.1, color: "var(--text)" }}>{title}</h2>
              <p style={{ margin: "0.35rem 0 0", color: "var(--muted)", fontSize: "0.84rem" }}>{subtitle}</p>
            </div>
            <button onClick={onClose} style={{ background: "rgba(148,163,184,0.08)", border: "1px solid rgba(148,163,184,0.18)", borderRadius: "0.5rem", color: "var(--muted)", padding: "0.4rem 0.9rem", fontSize: "0.85rem", flexShrink: 0, cursor: "pointer" }}>&#10005; Close</button>
          </div>
        </div>
        {/* Body */}
        <div style={{ overflowY: "auto", padding: "1.5rem 2rem", flex: 1, scrollbarWidth: "thin" as const, scrollbarColor: "rgba(148,163,184,0.3) transparent" }}>
          {isInput && <InputLayerModal nodes={nodes} />}
          {isHidden && <HiddenLayerModal layerIndex={hiddenIndex} inputSize={hiddenIndex === 0 ? nodes.length : hiddenSizes[hiddenIndex - 1]} outputSize={hiddenSizes[hiddenIndex]} isLast={hiddenIndex === hiddenSizes.length - 1} totalLayers={hiddenSizes.length} />}
          {isActor && <ActorHeadModal />}
          {isCritic && <CriticHeadModal />}
        </div>
      </div>
    </div>
  );
}

// ============================================================
// CLICKABLE ARCHITECTURE DIAGRAM
// ============================================================

function ArchitectureDiagram({ inputSize, hiddenSizes, actorSize, criticSize, onLayerClick }: {
  inputSize: number; hiddenSizes: number[]; actorSize: number; criticSize: number;
  onLayerClick: (key: LayerKey) => void;
}) {
  const btnBase: React.CSSProperties = { border: "1px solid", borderRadius: "0.5rem", padding: "0.5rem 0.75rem", textAlign: "center", minWidth: "80px", cursor: "pointer", background: "transparent", transition: "transform 100ms ease, box-shadow 100ms ease", fontFamily: "inherit" };
  const hov = (color: string) => ({ boxShadow: "0 0 0 2px " + color, transform: "translateY(-2px)" });

  return (
    <div style={{ padding: "1rem 0.5rem", fontFamily: "monospace", fontSize: "0.75rem" }}>
      <p style={{ margin: "0 0 0.75rem", fontSize: "0.73rem", color: "var(--muted)" }}>
        &#128161; Click any box below to open a detailed explanation.
      </p>
      <div style={{ display: "flex", alignItems: "center", flexWrap: "nowrap", overflowX: "auto", gap: 0 }}>
        {/* Input */}
        <ArchBox label="Input" sub={inputSize + " nodes"} color="#3b82f6" bg="rgba(59,130,246,0.12)" border="rgba(59,130,246,0.4)" btnBase={btnBase} hov={() => hov("#3b82f6")} onClick={() => onLayerClick("input")} />
        <Arrow />

        {/* Hidden layers */}
        {hiddenSizes.map((s, i) => (
          <React.Fragment key={i}>
            <ArchBox label={"Hidden " + (i + 1)} sub={s + " \u00d7 ReLU"} color="#818cf8" bg="rgba(129,140,248,0.10)" border="rgba(129,140,248,0.35)" btnBase={btnBase} hov={() => hov("#818cf8")} onClick={() => onLayerClick({ hidden: i })} />
            <Arrow />
          </React.Fragment>
        ))}

        {/* Fork to outputs */}
        <div style={{ display: "flex", flexDirection: "column", gap: "0.4rem", flexShrink: 0 }}>
          <div style={{ display: "flex", alignItems: "center", gap: "0.25rem" }}>
            <span style={{ color: "rgba(148,163,184,0.4)", fontSize: "0.7rem" }}>&#x250c;</span>
            <ArchBox label="Actor Head" sub={actorSize + " logits"} color="#818cf8" bg="rgba(129,140,248,0.12)" border="rgba(129,140,248,0.4)" btnBase={btnBase} hov={() => hov("#818cf8")} onClick={() => onLayerClick("actor")} />
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: "0.25rem" }}>
            <span style={{ color: "rgba(148,163,184,0.4)", fontSize: "0.7rem" }}>&#x2514;</span>
            <ArchBox label="Critic Head" sub={criticSize + " value"} color="#6ee7b7" bg="rgba(110,231,183,0.08)" border="rgba(110,231,183,0.3)" btnBase={btnBase} hov={() => hov("#6ee7b7")} onClick={() => onLayerClick("critic")} />
          </div>
        </div>
      </div>
    </div>
  );
}

function Arrow() {
  return <div style={{ color: "rgba(148,163,184,0.4)", padding: "0 0.3rem", flexShrink: 0, fontSize: "0.9rem" }}>&#8594;</div>;
}

function ArchBox({ label, sub, color, bg, border, btnBase, hov, onClick }: {
  label: string; sub: string; color: string; bg: string; border: string;
  btnBase: React.CSSProperties; hov: () => React.CSSProperties; onClick: () => void;
}) {
  const [hovered, setHovered] = React.useState(false);
  return (
    <button
      style={{ ...btnBase, borderColor: hovered ? color : border, background: hovered ? bg : bg + "aa", ...(hovered ? hov() : {}) }}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      onClick={onClick}
      title={"Click to learn about: " + label}
    >
      <div style={{ color, fontWeight: "700", fontSize: "0.7rem" }}>{label}</div>
      <div style={{ color: "var(--muted)", fontSize: "0.63rem", marginTop: "0.15rem" }}>{sub}</div>
      <div style={{ color, fontSize: "0.55rem", marginTop: "0.2rem", opacity: 0.7 }}>&#128269; click</div>
    </button>
  );
}

// ============================================================
// HIDDEN LAYER CARD (collapsible inline card on page)
// ============================================================

interface HiddenLayerCardProps { index: number; inputSize: number; outputSize: number; isLast: boolean; onOpenModal: () => void; }

function HiddenLayerCard({ index, inputSize, outputSize, isLast, onOpenModal }: HiddenLayerCardProps) {
  const params = inputSize * outputSize + outputSize;
  const [open, setOpen] = useState(false);
  const purposeText = index === 0
    ? "First hidden layer — performs the initial feature transformation. Learns which combinations of raw inputs are meaningful."
    : isLast
    ? "Final hidden layer — refines the representation before being read by both the actor and critic heads."
    : "Intermediate hidden layer — re-combines features into increasingly abstract representations.";
  return (
    <div style={{ background: "rgba(12,24,38,0.6)", border: "1px solid rgba(129,140,248,0.25)", borderRadius: "0.75rem", overflow: "hidden", transition: "border-color 150ms ease" }}
      onMouseEnter={e => (e.currentTarget.style.borderColor = "rgba(129,140,248,0.55)")}
      onMouseLeave={e => (e.currentTarget.style.borderColor = "rgba(129,140,248,0.25)")}>
      <div style={{ display: "grid", gridTemplateColumns: "auto 1fr auto auto auto auto", gap: "0.75rem", alignItems: "center", padding: "0.9rem 1.25rem" }}>
        <div style={{ width: "2.25rem", height: "2.25rem", borderRadius: "0.5rem", background: "linear-gradient(135deg,rgba(129,140,248,0.2),rgba(99,102,241,0.1))", border: "1px solid rgba(129,140,248,0.4)", display: "flex", alignItems: "center", justifyContent: "center", fontWeight: "700", fontSize: "0.8rem", color: "#818cf8", flexShrink: 0 }}>
          H{index + 1}
        </div>
        <div>
          <div style={{ fontWeight: "600", fontSize: "0.92rem" }}>
            Hidden Layer {index + 1}
            {isLast && <span style={{ marginLeft: "0.5rem", fontSize: "0.62rem", background: "rgba(244,197,66,0.12)", color: "#f4c542", border: "1px solid rgba(244,197,66,0.3)", padding: "0.1rem 0.4rem", borderRadius: "999px", fontWeight: "600", verticalAlign: "middle" }}>OUTPUT ADJACENT</span>}
          </div>
          <div style={{ fontSize: "0.73rem", color: "var(--muted)", marginTop: "0.1rem" }}>Linear({inputSize} &#x2192; {outputSize}) + ReLU</div>
        </div>
        <div style={{ textAlign: "center" }}>
          <div style={{ fontSize: "0.6rem", color: "var(--muted)", textTransform: "uppercase" as const, letterSpacing: "0.05em" }}>Neurons</div>
          <div style={{ fontWeight: "700", fontSize: "1.05rem", color: "#818cf8" }}>{outputSize}</div>
        </div>
        <div style={{ textAlign: "center" }}>
          <div style={{ fontSize: "0.6rem", color: "var(--muted)", textTransform: "uppercase" as const, letterSpacing: "0.05em" }}>Params</div>
          <div style={{ fontWeight: "700", fontSize: "1.05rem", color: "#93c5fd" }}>{formatNumber(params)}</div>
        </div>
        <button onClick={onOpenModal} style={{ background: "rgba(129,140,248,0.08)", border: "1px solid rgba(129,140,248,0.3)", borderRadius: "0.5rem", color: "#818cf8", padding: "0.35rem 0.75rem", fontSize: "0.75rem", fontWeight: "600", cursor: "pointer" }}>
          Deep Dive &#128269;
        </button>
        <button onClick={() => setOpen(v => !v)} aria-expanded={open}
          style={{ background: "transparent", border: "none", color: "var(--muted)", fontSize: "0.85rem", cursor: "pointer", padding: "0.2rem 0.4rem", transform: open ? "rotate(180deg)" : "rotate(0deg)", transition: "transform 200ms ease" }}>
          &#9662;
        </button>
      </div>

      {open && (
        <div style={{ padding: "0 1.25rem 1.25rem", borderTop: "1px solid rgba(148,163,184,0.08)" }}>
          <div style={{ display: "flex", alignItems: "center", gap: "0.5rem", margin: "0.9rem 0", fontSize: "0.78rem", flexWrap: "wrap" }}>
            {[
              [inputSize + " inputs", "#93c5fd", "rgba(59,130,246,0.12)", "rgba(59,130,246,0.3)"],
              ["× " + inputSize + "×" + outputSize + " weights", "#818cf8", "rgba(129,140,248,0.1)", "rgba(129,140,248,0.3)"],
              ["+ " + outputSize + " biases", "#818cf8", "rgba(129,140,248,0.08)", "rgba(129,140,248,0.2)"],
              ["→ ReLU(" + outputSize + ")", "#6ee7b7", "rgba(16,185,129,0.1)", "rgba(16,185,129,0.3)"],
              ["→ " + outputSize + " activations", "#818cf8", "rgba(129,140,248,0.12)", "rgba(129,140,248,0.3)"],
            ].map(([lbl, c, bg, b], i) => (
              <span key={i} style={{ background: bg, border: "1px solid " + b, color: c, padding: "0.28rem 0.6rem", borderRadius: "0.4rem", fontFamily: "monospace" }}>{lbl}</span>
            ))}
          </div>
          <div style={{ background: "rgba(8,15,25,0.4)", borderRadius: "0.5rem", padding: "0.8rem 1rem", borderLeft: "3px solid #818cf8", marginBottom: "0.6rem" }}>
            <div style={{ fontSize: "0.67rem", color: "#818cf8", textTransform: "uppercase" as const, letterSpacing: "0.06em", marginBottom: "0.35rem", fontWeight: "600" }}>What this layer does</div>
            <p style={{ margin: 0, fontSize: "0.8rem", color: "var(--muted)", lineHeight: "1.55" }}>{purposeText}</p>
          </div>
          <div style={{ background: "rgba(16,185,129,0.05)", borderRadius: "0.5rem", padding: "0.8rem 1rem", borderLeft: "3px solid #10b981" }}>
            <div style={{ fontSize: "0.67rem", color: "#10b981", textTransform: "uppercase" as const, letterSpacing: "0.06em", marginBottom: "0.35rem", fontWeight: "600" }}>Activation: ReLU — f(x) = max(0, x)</div>
            <p style={{ margin: 0, fontSize: "0.8rem", color: "var(--muted)", lineHeight: "1.55" }}>
              Negative pre-activations become 0 (neuron &ldquo;off&rdquo;); positive values pass through unchanged (neuron &ldquo;on&rdquo;). Without this, stacking linear layers would be equivalent to a single linear layer.
            </p>
          </div>
        </div>
      )}
    </div>
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
  { name: "Linear (Dense)", used: true, icon: "▦", color: "#818cf8", strength: "Learns any weighted combination of inputs. Universal approximator when stacked.", weakness: "Quadratic parameter growth with width. No spatial structure.", whenToUse: "Fixed-size, non-spatial feature vectors." },
  { name: "Convolutional (CNN)", used: false, icon: "⊞", color: "#3b82f6", strength: "Weight-sharing, vastly fewer parameters for image data.", weakness: "Assumes 2D/3D spatial structure.", whenToUse: "Pixel observations, spatial maps (Atari, Minecraft)." },
  { name: "Recurrent (LSTM/GRU)", used: false, icon: "↺", color: "#f59e0b", strength: "Maintains state across timesteps. Learns temporal patterns.", weakness: "Harder to parallelize. Requires truncated BPTT in RL.", whenToUse: "Partially-observable envs where history matters." },
  { name: "Transformer (Attention)", used: false, icon: "◈", color: "#ec4899", strength: "Dynamic context-aware weighting. Handles variable-length sets.", weakness: "Heavy compute. Overkill for small fixed-size vectors.", whenToUse: "Variable agent counts or very large observation sets." },
];

function InfoModal({ onClose }: { onClose: () => void }) {
  const [activeSection, setActiveSection] = useState<"overview" | "glossary" | "layertypes">("overview");
  return (
    <div style={{ position: "fixed", inset: 0, zIndex: 1000, background: "rgba(4,8,15,0.85)", backdropFilter: "blur(8px)", display: "flex", alignItems: "center", justifyContent: "center", padding: "1rem" }} onClick={onClose}>
      <div style={{ background: "linear-gradient(160deg,rgba(12,24,38,0.98),rgba(8,15,25,0.99))", border: "1px solid rgba(148,163,184,0.2)", borderRadius: "1.25rem", boxShadow: "0 40px 120px rgba(0,0,0,0.7)", width: "min(860px,96vw)", maxHeight: "88vh", display: "flex", flexDirection: "column", overflow: "hidden" }} onClick={e => e.stopPropagation()}>
        <div style={{ padding: "1.5rem 2rem 1rem", borderBottom: "1px solid rgba(148,163,184,0.12)", display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
          <div>
            <div style={{ fontSize: "0.68rem", textTransform: "uppercase" as const, letterSpacing: "0.1em", color: "#f4c542", fontWeight: "700", marginBottom: "0.3rem" }}>Teaching Guide</div>
            <h2 style={{ margin: 0, fontSize: "1.4rem" }}>Understanding This Page</h2>
            <p style={{ margin: "0.35rem 0 0", color: "var(--muted)", fontSize: "0.83rem" }}>A deep-dive into the neural network architecture powering the sheepdog agents.</p>
          </div>
          <button onClick={onClose} style={{ background: "rgba(148,163,184,0.08)", border: "1px solid rgba(148,163,184,0.2)", borderRadius: "0.5rem", color: "var(--muted)", padding: "0.4rem 0.8rem", fontSize: "0.83rem", flexShrink: 0, cursor: "pointer" }}>&#10005; Close</button>
        </div>
        <div style={{ display: "flex", borderBottom: "1px solid rgba(148,163,184,0.12)", padding: "0 2rem", flexShrink: 0 }}>
          {(["overview", "glossary", "layertypes"] as const).map(sec => {
            const labels = { overview: "Page Overview", glossary: "Glossary", layertypes: "Layer Types" };
            return <button key={sec} onClick={() => setActiveSection(sec)} style={{ background: "transparent", border: "none", borderBottom: activeSection === sec ? "2px solid #f4c542" : "2px solid transparent", borderRadius: "0", padding: "0.7rem 1rem", fontSize: "0.8rem", fontWeight: activeSection === sec ? "600" : "400", color: activeSection === sec ? "var(--text)" : "var(--muted)", cursor: "pointer", marginBottom: "-1px" }}>{labels[sec]}</button>;
          })}
        </div>
        <div style={{ overflowY: "auto", padding: "1.5rem 2rem", flex: 1, scrollbarWidth: "thin" as const, scrollbarColor: "rgba(148,163,184,0.3) transparent" }}>
          {activeSection === "overview" && (
            <div style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>
              <div style={{ background: "rgba(244,197,66,0.06)", border: "1px solid rgba(244,197,66,0.2)", borderRadius: "0.75rem", padding: "1.1rem" }}>
                <h3 style={{ margin: "0 0 0.4rem", color: "#f4c542", fontSize: "0.95rem" }}>What is this page?</h3>
                <p style={{ margin: 0, fontSize: "0.84rem", color: "var(--muted)", lineHeight: "1.6" }}>The <strong style={{ color: "var(--text)" }}>Layers tab</strong> is a complete map of the neural network controlling every sheepdog agent. Click any box in the Architecture Flow diagram to open a detailed explanation of that layer. The input, hidden layers, actor head, and critic head all have their own educational modal.</p>
              </div>
              {[{ n: "1", color: "#3b82f6", title: "Click any architecture box", body: "The Input, Hidden Layer, Actor Head, and Critic Head boxes in the Network Architecture Flow diagram are all clickable. Each opens a modal explaining what that layer is, why it exists, and what it does — with both beginner and advanced sections." }, { n: "2", color: "#818cf8", title: "Hidden Layers section", body: "Each layer card can be expanded inline for a quick summary, or use the Deep Dive button to open the full detail modal." }, { n: "3", color: "#10b981", title: "Input Layer nodes table", body: "Switch between observation modes, filter by category, search for features. The Input architecture box also opens this full table." }, { n: "4", color: "#ec4899", title: "Output heads", body: "Actor Head = the 9 action logits. Critic Head = the state value V(s). Click each in the diagram for a full explanation." }].map((item, i) => (
                <div key={i} style={{ display: "flex", gap: "0.9rem", alignItems: "flex-start" }}>
                  <div style={{ width: "1.8rem", height: "1.8rem", borderRadius: "50%", background: item.color + "20", border: "1px solid " + item.color + "50", color: item.color, display: "flex", alignItems: "center", justifyContent: "center", fontWeight: "700", flexShrink: 0, fontSize: "0.82rem" }}>{item.n}</div>
                  <div><div style={{ fontWeight: "600", fontSize: "0.87rem", marginBottom: "0.2rem" }}>{item.title}</div><p style={{ margin: 0, fontSize: "0.8rem", color: "var(--muted)", lineHeight: "1.5" }}>{item.body}</p></div>
                </div>
              ))}
            </div>
          )}
          {activeSection === "glossary" && (
            <div style={{ display: "flex", flexDirection: "column", gap: "0.65rem" }}>
              {GLOSSARY.map((entry, i) => (
                <div key={i} style={{ background: "rgba(12,24,38,0.5)", border: "1px solid rgba(148,163,184,0.1)", borderRadius: "0.6rem", padding: "0.9rem 1rem" }}>
                  <div style={{ display: "flex", alignItems: "center", gap: "0.55rem", marginBottom: "0.4rem", flexWrap: "wrap" }}>
                    <strong style={{ fontSize: "0.9rem" }}>{entry.term}</strong>
                    <span style={{ fontSize: "0.63rem", padding: "0.1rem 0.45rem", borderRadius: "999px", background: entry.tagColor + "18", border: "1px solid " + entry.tagColor + "40", color: entry.tagColor, fontWeight: "600", textTransform: "uppercase" as const, letterSpacing: "0.05em" }}>{entry.tag}</span>
                  </div>
                  <p style={{ margin: "0 0 0.4rem", fontSize: "0.8rem", color: "var(--muted)", lineHeight: "1.5" }}>{entry.definition}</p>
                  <div style={{ fontSize: "0.73rem", color: "#f4c542", background: "rgba(244,197,66,0.06)", borderRadius: "0.35rem", padding: "0.25rem 0.55rem", borderLeft: "2px solid rgba(244,197,66,0.4)" }}><strong>Used here:</strong> {entry.usedHere}</div>
                </div>
              ))}
            </div>
          )}
          {activeSection === "layertypes" && (
            <div style={{ display: "flex", flexDirection: "column", gap: "0.9rem" }}>
              {LAYER_COMPARISON.map((layer, i) => (
                <div key={i} style={{ background: layer.used ? "rgba(129,140,248,0.06)" : "rgba(12,24,38,0.4)", border: "1px solid " + (layer.used ? "rgba(129,140,248,0.3)" : "rgba(148,163,184,0.1)"), borderRadius: "0.75rem", padding: "1rem 1.1rem" }}>
                  <div style={{ display: "flex", alignItems: "center", gap: "0.65rem", marginBottom: "0.65rem", flexWrap: "wrap" }}>
                    <span style={{ fontSize: "1.3rem", color: layer.color }}>{layer.icon}</span>
                    <strong style={{ fontSize: "0.95rem", color: layer.used ? layer.color : "var(--text)" }}>{layer.name}</strong>
                    {layer.used ? <span style={{ fontSize: "0.62rem", padding: "0.1rem 0.45rem", borderRadius: "999px", background: "rgba(74,222,128,0.12)", border: "1px solid rgba(74,222,128,0.35)", color: "#4ade80", fontWeight: "700", textTransform: "uppercase" as const }}>&#10003; Used</span> : <span style={{ fontSize: "0.62rem", padding: "0.1rem 0.45rem", borderRadius: "999px", background: "rgba(148,163,184,0.08)", border: "1px solid rgba(148,163,184,0.2)", color: "var(--muted)", fontWeight: "600", textTransform: "uppercase" as const }}>Not used</span>}
                  </div>
                  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0.6rem", fontSize: "0.8rem" }}>
                    <div style={{ background: "rgba(74,222,128,0.05)", borderRadius: "0.4rem", padding: "0.55rem 0.7rem", borderLeft: "2px solid rgba(74,222,128,0.4)" }}>
                      <div style={{ color: "#4ade80", fontWeight: "600", fontSize: "0.66rem", textTransform: "uppercase" as const, marginBottom: "0.25rem" }}>Strengths</div>
                      <p style={{ margin: 0, color: "var(--muted)", lineHeight: "1.4" }}>{layer.strength}</p>
                    </div>
                    <div style={{ background: "rgba(251,113,133,0.05)", borderRadius: "0.4rem", padding: "0.55rem 0.7rem", borderLeft: "2px solid rgba(251,113,133,0.3)" }}>
                      <div style={{ color: "#fb7185", fontWeight: "600", fontSize: "0.66rem", textTransform: "uppercase" as const, marginBottom: "0.25rem" }}>Weaknesses</div>
                      <p style={{ margin: 0, color: "var(--muted)", lineHeight: "1.4" }}>{layer.weakness}</p>
                    </div>
                  </div>
                  <div style={{ marginTop: "0.6rem", background: "rgba(244,197,66,0.05)", borderRadius: "0.4rem", padding: "0.5rem 0.7rem", fontSize: "0.78rem", color: "var(--muted)", borderLeft: "2px solid rgba(244,197,66,0.35)" }}>
                    <strong style={{ color: "#f4c542" }}>When to use: </strong>{layer.whenToUse}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
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

  const [selectedViewMode, setSelectedViewMode] = useState<string>(activeModeFromConfig);
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedCategory, setSelectedCategory] = useState<string>("all");
  const [showInfoModal, setShowInfoModal] = useState(false);
  const [activeLayerModal, setActiveLayerModal] = useState<LayerKey | null>(null);

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
    for (const sz of hiddenLayerSizes) { total += prev * sz + sz; prev = sz; }
    total += prev * ACTOR_NODES.length + ACTOR_NODES.length;
    total += prev * CRITIC_NODES.length + CRITIC_NODES.length;
    return total;
  }, [activeNodesList.length, hiddenLayerSizes]);

  return (
    <>
      {showInfoModal && <InfoModal onClose={() => setShowInfoModal(false)} />}
      {activeLayerModal !== null && (
        <LayerDetailModal layerKey={activeLayerModal} nodes={activeNodesList} hiddenSizes={hiddenLayerSizes} onClose={() => setActiveLayerModal(null)} />
      )}

      <section className="network-tab" aria-label="Model Layers Detailed Card" style={{ paddingBottom: "3rem" }}>

        {/* Header */}
        <div className="network-tab__header" style={{ marginBottom: "1.5rem" }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", gap: "1rem", flexWrap: "wrap" }}>
            <div>
              <p className="eyebrow" style={{ color: "var(--accent)", fontWeight: 600 }}>Model Card</p>
              <h2 style={{ fontSize: "2rem", margin: "0.25rem 0 0.5rem" }}>Neural Network Layers</h2>
              <p className="network-tab__intro" style={{ maxWidth: "700px", color: "var(--muted)" }}>
                A precise map of what the sheepdog agents see, how they process it, and what decisions they output. Click any box in the architecture diagram to explore that layer in depth.
              </p>
            </div>
            <button id="layers-info-button" onClick={() => setShowInfoModal(true)}
              style={{ display: "flex", alignItems: "center", gap: "0.5rem", background: "linear-gradient(135deg,rgba(244,197,66,0.12),rgba(244,197,66,0.04))", border: "1px solid rgba(244,197,66,0.4)", color: "#f4c542", padding: "0.65rem 1.1rem", borderRadius: "0.75rem", fontWeight: "600", fontSize: "0.85rem", flexShrink: 0, cursor: "pointer" }}>
              &#128161; What is this page?
            </button>
          </div>
        </div>

        {/* KPI Row */}
        <div className="network-tab__kpis" style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit,minmax(175px,1fr))", gap: "0.85rem", marginBottom: "2rem" }}>
          {[
            { label: "Active Mode", value: activeModeFromConfig.toUpperCase(), color: "#f4c542", badge: "RUNTIME" },
            { label: "Input Features", value: `${activeNodesList.length}`, color: "#93c5fd", badge: null },
            { label: "Hidden Layers", value: `${hiddenLayerSizes.length}`, color: "#818cf8", badge: null },
            { label: "Total Neurons", value: formatNumber(hiddenLayerSizes.reduce((a, b) => a + b, 0)), color: "#818cf8", badge: null },
            { label: "Total Parameters", value: formatNumber(totalParams), color: "#6ee7b7", badge: null },
            { label: "Actions (Actor)", value: `${ACTOR_NODES.length}`, color: "#818cf8", badge: null },
          ].map((kpi, i) => (
            <div key={i} style={{ background: "rgba(12,24,38,0.5)", border: "1px solid var(--panel-border)", padding: "1.05rem 1.1rem", borderRadius: "0.75rem" }}>
              <span style={{ fontSize: "0.68rem", textTransform: "uppercase" as const, letterSpacing: "0.05em", color: "var(--muted)" }}>{kpi.label}</span>
              <div style={{ display: "flex", alignItems: "center", gap: "0.5rem", marginTop: "0.2rem" }}>
                <strong style={{ fontSize: "1.3rem", color: kpi.color }}>{kpi.value}</strong>
                {kpi.badge && <span style={{ background: "var(--accent-soft)", color: "var(--accent)", border: "1px solid var(--accent)", fontSize: "0.58rem", padding: "0.12rem 0.4rem", borderRadius: "999px", fontWeight: "bold" }}>{kpi.badge}</span>}
              </div>
            </div>
          ))}
        </div>

        {/* Architecture Diagram */}
        <section className="network-tab__card" style={{ padding: "1.1rem 1.4rem", marginBottom: "1.5rem" }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "0.15rem", flexWrap: "wrap", gap: "0.5rem" }}>
            <h3 style={{ margin: 0, fontSize: "1rem" }}>Network Architecture Flow</h3>
            <span style={{ fontSize: "0.73rem", color: "var(--muted)" }}>{hiddenLayerSizes.length} hidden layer{hiddenLayerSizes.length !== 1 ? "s" : ""} &middot; {formatNumber(totalParams)} total parameters</span>
          </div>
          <ArchitectureDiagram inputSize={activeNodesList.length} hiddenSizes={hiddenLayerSizes} actorSize={ACTOR_NODES.length} criticSize={CRITIC_NODES.length} onLayerClick={key => setActiveLayerModal(key)} />
        </section>

        {/* Hidden Layers */}
        <section style={{ marginBottom: "2rem" }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "0.75rem", flexWrap: "wrap", gap: "0.5rem" }}>
            <h3 style={{ margin: 0, fontSize: "1.05rem" }}>
              Hidden Layers
              <span style={{ marginLeft: "0.55rem", fontSize: "0.68rem", background: "rgba(129,140,248,0.12)", border: "1px solid rgba(129,140,248,0.3)", color: "#818cf8", padding: "0.12rem 0.5rem", borderRadius: "999px", fontWeight: "600", verticalAlign: "middle" }}>
                {hiddenLayerSizes.length} LAYER{hiddenLayerSizes.length !== 1 ? "S" : ""}
              </span>
            </h3>
            <span style={{ fontSize: "0.78rem", color: "var(--muted)" }}>Expand inline or click Deep Dive for a full explanation</span>
          </div>
          <div style={{ display: "flex", flexDirection: "column", gap: "0.7rem" }}>
            {hiddenLayerSizes.map((size, i) => (
              <HiddenLayerCard key={i} index={i} inputSize={i === 0 ? activeNodesList.length : hiddenLayerSizes[i - 1]} outputSize={size} isLast={i === hiddenLayerSizes.length - 1} onOpenModal={() => setActiveLayerModal({ hidden: i })} />
            ))}
          </div>
        </section>

        {/* Mode Switcher + Search */}
        <div style={{ display: "flex", flexWrap: "wrap", justifyContent: "space-between", alignItems: "center", gap: "1rem", borderBottom: "1px solid var(--panel-border)", paddingBottom: "1rem", marginBottom: "1.5rem" }}>
          <div style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap" }}>
            {["guided", "hierarchical", "emergent"].map(modeName => (
              <button key={modeName} onClick={() => setSelectedViewMode(modeName)}
                style={{ background: selectedViewMode === modeName ? "linear-gradient(180deg,var(--accent-soft),rgba(244,197,66,0.05))" : "transparent", borderColor: selectedViewMode === modeName ? "var(--accent)" : "var(--panel-border)", color: selectedViewMode === modeName ? "var(--text)" : "var(--muted)", fontSize: "0.83rem", padding: "0.45rem 0.9rem", borderRadius: "0.5rem", fontWeight: selectedViewMode === modeName ? "600" : "400", display: "flex", alignItems: "center", gap: "0.4rem", cursor: "pointer" }}>
                {modeName.toUpperCase()}
                {activeModeFromConfig === modeName && <span style={{ width: "6px", height: "6px", background: "var(--good)", borderRadius: "50%" }} />}
              </button>
            ))}
          </div>
          <div style={{ display: "flex", gap: "0.5rem", flex: "1", maxWidth: "380px" }}>
            <input type="text" placeholder="Search input features..." value={searchQuery} onChange={e => setSearchQuery(e.target.value)}
              style={{ width: "100%", borderRadius: "0.5rem", border: "1px solid var(--panel-border)", color: "var(--text)", background: "rgba(8,15,25,0.8)", padding: "0.45rem 0.75rem", fontSize: "0.83rem" }} />
          </div>
        </div>

        {/* Input + Output Panels */}
        <div style={{ display: "grid", gridTemplateColumns: "1fr 390px", gap: "1.5rem", minHeight: 0 }}>

          {/* INPUT PANEL */}
          <section className="network-tab__card" style={{ padding: "1.5rem", margin: "0", display: "flex", flexDirection: "column", minHeight: 0 }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "0.9rem" }}>
              <h3 style={{ margin: 0 }}>Input Layer Nodes ({filteredNodesList.length} of {activeNodesList.length})</h3>
              <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
                <span style={{ fontSize: "0.78rem", color: "var(--muted)" }}>Mode: {selectedViewMode}</span>
                <button onClick={() => setActiveLayerModal("input")} style={{ background: "rgba(59,130,246,0.08)", border: "1px solid rgba(59,130,246,0.3)", borderRadius: "0.4rem", color: "#93c5fd", padding: "0.25rem 0.6rem", fontSize: "0.72rem", fontWeight: "600", cursor: "pointer" }}>About &#128269;</button>
              </div>
            </div>
            <div style={{ display: "flex", flexWrap: "wrap", gap: "0.35rem", marginBottom: "1rem" }}>
              <button onClick={() => setSelectedCategory("all")} style={{ padding: "0.18rem 0.55rem", fontSize: "0.72rem", borderRadius: "999px", borderColor: selectedCategory === "all" ? "var(--accent)" : "rgba(148,163,184,0.15)", background: selectedCategory === "all" ? "rgba(244,197,66,0.1)" : "rgba(12,24,38,0.4)", cursor: "pointer" }}>All</button>
              {Object.entries(CATEGORY_LABELS).map(([catKey, label]) => {
                if (!activeNodesList.some(n => n.category === catKey)) return null;
                const cc = CATEGORY_COLORS[catKey as NodeDefinition["category"]];
                return <button key={catKey} onClick={() => setSelectedCategory(catKey)} style={{ padding: "0.18rem 0.55rem", fontSize: "0.72rem", borderRadius: "999px", borderColor: selectedCategory === catKey ? cc : "rgba(148,163,184,0.15)", background: selectedCategory === catKey ? cc + "20" : "rgba(12,24,38,0.4)", color: selectedCategory === catKey ? cc : "var(--muted)", cursor: "pointer" }}>{label}</button>;
              })}
            </div>
            <div style={{ overflowY: "auto", flex: 1, minHeight: 0, paddingRight: "0.4rem", scrollbarWidth: "thin" as const, scrollbarColor: "rgba(148,163,184,0.3) transparent" }}>
              <table style={{ width: "100%", borderCollapse: "collapse", textAlign: "left", fontSize: "0.82rem" }}>
                <thead>
                  <tr style={{ borderBottom: "1px solid var(--panel-border)" }}>
                    <th style={{ padding: "0.65rem 0.4rem", width: "60px", color: "var(--muted)" }}>#</th>
                    <th style={{ padding: "0.65rem 0.4rem", width: "170px" }}>Feature</th>
                    <th style={{ padding: "0.65rem 0.4rem", width: "160px" }}>Category</th>
                    <th style={{ padding: "0.65rem 0.4rem" }}>Description &amp; Normalization</th>
                  </tr>
                </thead>
                <tbody>
                  {filteredNodesList.length === 0
                    ? <tr><td colSpan={4} style={{ textAlign: "center", padding: "2rem", color: "var(--muted)" }}>No nodes match.</td></tr>
                    : filteredNodesList.map(node => {
                        const cc = CATEGORY_COLORS[node.category];
                        return (
                          <tr key={node.index} style={{ borderBottom: "1px solid rgba(148,163,184,0.07)", transition: "background 100ms ease" }} className="layer-row-hover">
                            <td style={{ padding: "0.65rem 0.4rem", fontWeight: "700", color: "var(--muted)", fontFamily: "monospace" }}>#{node.index}</td>
                            <td style={{ padding: "0.65rem 0.4rem", fontWeight: "600", fontFamily: "monospace", color: "var(--text)" }}>{node.name}</td>
                            <td style={{ padding: "0.65rem 0.4rem" }}><span style={{ display: "inline-block", padding: "0.12rem 0.45rem", borderRadius: "4px", fontSize: "0.67rem", fontWeight: "600", color: cc, background: cc + "18", border: "1px solid " + cc + "40" }}>{CATEGORY_LABELS[node.category]}</span></td>
                            <td style={{ padding: "0.65rem 0.4rem" }}>
                              <div style={{ fontWeight: "500" }}>{node.description}</div>
                              <div style={{ fontSize: "0.71rem", color: "var(--muted)", fontStyle: "italic", marginTop: "0.12rem" }}>{node.details}</div>
                            </td>
                          </tr>
                        );
                      })
                  }
                </tbody>
              </table>
            </div>
          </section>

          {/* OUTPUT PANELS */}
          <div style={{ display: "flex", flexDirection: "column", gap: "1.25rem", overflowY: "auto", paddingRight: "0.4rem", scrollbarWidth: "thin" as const, scrollbarColor: "rgba(148,163,184,0.3) transparent" }}>

            <section className="network-tab__card" style={{ padding: "1.25rem", margin: "0" }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "0.75rem" }}>
                <h3 style={{ margin: 0, color: "#818cf8", fontSize: "0.95rem" }}>Actor Head — Policy Logits</h3>
                <button onClick={() => setActiveLayerModal("actor")} style={{ background: "rgba(129,140,248,0.08)", border: "1px solid rgba(129,140,248,0.3)", borderRadius: "0.4rem", color: "#818cf8", padding: "0.25rem 0.6rem", fontSize: "0.72rem", fontWeight: "600", cursor: "pointer" }}>About &#128269;</button>
              </div>
              <p style={{ fontSize: "0.76rem", color: "var(--muted)", margin: "0 0 0.85rem" }}>{ACTOR_NODES.length} logits &rarr; Softmax + Action Masking &rarr; discrete action</p>
              <div style={{ display: "flex", flexDirection: "column", gap: "0.4rem" }}>
                {ACTOR_NODES.map(act => (
                  <div key={act.index} style={{ background: "rgba(8,15,25,0.4)", border: "1px solid rgba(148,163,184,0.07)", borderRadius: "0.45rem", padding: "0.55rem 0.75rem", display: "grid", gridTemplateColumns: "30px 80px 1fr", gap: "0.4rem", alignItems: "center" }}>
                    <span style={{ fontWeight: "700", color: "var(--muted)", fontSize: "0.75rem", fontFamily: "monospace" }}>#{act.index}</span>
                    <span style={{ fontFamily: "monospace", fontWeight: "700", fontSize: "0.8rem", color: "#818cf8" }}>{act.name}</span>
                    <div style={{ fontSize: "0.72rem" }}>
                      <div style={{ fontWeight: "600", color: "var(--text)" }}>{act.description}</div>
                      <div style={{ color: "var(--muted)", fontSize: "0.67rem", marginTop: "0.08rem" }}>{act.communicating}</div>
                    </div>
                  </div>
                ))}
              </div>
            </section>

            <section className="network-tab__card" style={{ padding: "1.25rem", margin: "0" }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "0.75rem" }}>
                <h3 style={{ margin: 0, color: "#6ee7b7", fontSize: "0.95rem" }}>Critic Head — Value Head</h3>
                <button onClick={() => setActiveLayerModal("critic")} style={{ background: "rgba(110,231,183,0.07)", border: "1px solid rgba(110,231,183,0.3)", borderRadius: "0.4rem", color: "#6ee7b7", padding: "0.25rem 0.6rem", fontSize: "0.72rem", fontWeight: "600", cursor: "pointer" }}>About &#128269;</button>
              </div>
              <p style={{ fontSize: "0.76rem", color: "var(--muted)", margin: "0 0 0.85rem" }}>Single scalar V(s) &rarr; used by GAE during training only</p>
              <div>
                {CRITIC_NODES.map(crit => (
                  <div key={crit.index} style={{ background: "rgba(8,15,25,0.4)", border: "1px solid rgba(110,231,183,0.15)", borderRadius: "0.45rem", padding: "0.7rem 0.85rem" }}>
                    <div style={{ display: "flex", alignItems: "center", gap: "0.6rem", marginBottom: "0.35rem" }}>
                      <span style={{ fontFamily: "monospace", fontWeight: "700", fontSize: "1rem", color: "#6ee7b7" }}>{crit.name}</span>
                      <span style={{ fontSize: "0.67rem", padding: "0.1rem 0.45rem", borderRadius: "999px", background: "rgba(110,231,183,0.1)", border: "1px solid rgba(110,231,183,0.3)", color: "#6ee7b7", fontWeight: "600" }}>{crit.type}</span>
                      <span style={{ fontSize: "0.67rem", padding: "0.1rem 0.45rem", borderRadius: "999px", background: "rgba(100,116,139,0.08)", border: "1px solid rgba(100,116,139,0.2)", color: "var(--muted)", fontWeight: "600" }}>No activation</span>
                    </div>
                    <div style={{ fontSize: "0.8rem", color: "var(--text)", marginBottom: "0.2rem" }}>{crit.description}</div>
                    <div style={{ fontSize: "0.72rem", color: "var(--muted)", lineHeight: "1.45" }}>{crit.communicating}</div>
                  </div>
                ))}
              </div>
            </section>

            <section className="network-tab__card" style={{ padding: "1.1rem", margin: "0", background: "rgba(244,197,66,0.03)", borderColor: "rgba(244,197,66,0.2)" }}>
              <h4 style={{ margin: "0 0 0.45rem", color: "var(--accent)", fontSize: "0.83rem" }}>Action Masking Pipeline</h4>
              <p style={{ fontSize: "0.73rem", color: "var(--muted)", margin: 0, lineHeight: "1.5" }}>
                Actor logits are intercepted by <strong>MaskablePPO</strong>. Invalid moves are set to &minus;&infin; before Softmax, giving exactly 0% probability. This focuses all gradient on legal actions.
              </p>
            </section>
          </div>
        </div>
      </section>
    </>
  );
}

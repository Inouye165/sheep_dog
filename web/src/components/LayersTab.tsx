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

// Guided (Role-aware) nodes - 54 nodes
const GUIDED_NODES: NodeDefinition[] = [
  { index: 0, name: "own_x", category: "position", description: "Dog's own absolute X position", details: "Normalized coordinate: own_x / (field_width - 1)" },
  { index: 1, name: "own_y", category: "position", description: "Dog's own absolute Y position", details: "Normalized coordinate: own_y / (field_height - 1)" },
  { index: 2, name: "pen_x", category: "position", description: "Pen center absolute X coordinate", details: "Normalized coordinate: pen_center_x / (field_width - 1)" },
  { index: 3, name: "pen_y", category: "position", description: "Pen center absolute Y coordinate", details: "Normalized coordinate: pen_center_y / (field_height - 1)" },
  { index: 4, name: "flock_center_x", category: "position", description: "Flock center absolute X coordinate", details: "Normalized coordinate: flock_center_x / (field_width - 1)" },
  { index: 5, name: "flock_center_y", category: "position", description: "Flock center absolute Y coordinate", details: "Normalized coordinate: flock_center_y / (field_height - 1)" },
  { index: 6, name: "target_x", category: "target", description: "Dog's assigned target absolute X coordinate", details: "Computed by high-level role assignment: target_x / (field_width - 1)" },
  { index: 7, name: "target_y", category: "target", description: "Dog's assigned target absolute Y coordinate", details: "Computed by high-level role assignment: target_y / (field_height - 1)" },
  { index: 8, name: "distance_to_pen", category: "distance", description: "Distance from dog to pen center", details: "Normalized distance: distance_to_pen / field_diagonal" },
  { index: 9, name: "distance_to_flock", category: "distance", description: "Distance from dog to flock center", details: "Normalized distance: distance_to_flock / field_diagonal" },
  { index: 10, name: "distance_to_target", category: "distance", description: "Distance from dog to its target position", details: "Normalized distance: distance_to_target / field_diagonal" },
  { index: 11, name: "flock_spread", category: "sensor", description: "Standard deviation of sheep positions", details: "Normalized spread of the flock: flock_spread / field_diagonal" },
  { index: 12, name: "average_distance_to_pen", category: "distance", description: "Mean distance of all sheep to the pen", details: "Normalized average distance: avg_distance / field_diagonal" },
  { index: 13, name: "wall_left", category: "sensor", description: "Proximity to the left boundary", details: "Calculated as: own_x / (field_width - 1)" },
  { index: 14, name: "wall_right", category: "sensor", description: "Proximity to the right boundary", details: "Calculated as: (field_width - 1 - own_x) / (field_width - 1)" },
  { index: 15, name: "wall_top", category: "sensor", description: "Proximity to the top boundary", details: "Calculated as: own_y / (field_height - 1)" },
  { index: 16, name: "wall_bottom", category: "sensor", description: "Proximity to the bottom boundary", details: "Calculated as: (field_height - 1 - own_y) / (field_height - 1)" },
  { index: 17, name: "blocked_steps", category: "sensor", description: "Steps dog has been blocked/colliding", details: "Scaled output: min(blocked_steps, 10) / 10.0" },
  { index: 18, name: "no_progress_steps", category: "sensor", description: "Steps team has made no progress towards pen", details: "Scaled: min(no_progress, window) / window" },
  { index: 19, name: "revisits_recent_position", category: "sensor", description: "Binary indicator of position revisit loop", details: "1.0 if dog has revisited a recent cell in the past few steps, 0.0 otherwise" },
  { index: 20, name: "two_position_loop", category: "sensor", description: "Binary indicator of two-position oscillation", details: "1.0 if dog is oscillating back and forth between two positions, 0.0 otherwise" },
  { index: 21, name: "stray_present", category: "sensor", description: "Binary indicator if any stray sheep exists", details: "1.0 if at least one sheep is stray (distance to flock > 1.8 * spread), 0.0 otherwise" },
  { index: 22, name: "role_rear_pressure", category: "role", description: "Flag: Assigned to REAR_PRESSURE role", details: "One-hot role encoding: 1.0 if active, 0.0 otherwise. Role focuses on pushing flock from behind." },
  { index: 23, name: "role_left_flanker", category: "role", description: "Flag: Assigned to LEFT_FLANKER role", details: "One-hot role encoding: 1.0 if active, 0.0 otherwise. Role blocks left escapes." },
  { index: 24, name: "role_right_flanker", category: "role", description: "Flag: Assigned to RIGHT_FLANKER role", details: "One-hot role encoding: 1.0 if active, 0.0 otherwise. Role blocks right escapes." },
  { index: 25, name: "role_collector", category: "role", description: "Flag: Assigned to COLLECTOR role", details: "One-hot role encoding: 1.0 if active, 0.0 otherwise. Role fetches outliers/strays." },
  { index: 26, name: "role_blocker", category: "role", description: "Flag: Assigned to BLOCKER role", details: "One-hot role encoding: 1.0 if active, 0.0 otherwise. Role holds position near the pen mouth." },
  { index: 27, name: "focus_sheep_dx", category: "sheep", description: "Relative X distance to closest unpenned target sheep", details: "Normalized: (focus_sheep_x - own_x) / (field_width - 1)" },
  { index: 28, name: "focus_sheep_dy", category: "sheep", description: "Relative Y distance to closest unpenned target sheep", details: "Normalized: (focus_sheep_y - own_y) / (field_height - 1)" },
  { index: 29, name: "focus_sheep_distance", category: "distance", description: "Distance to closest unpenned target sheep", details: "Normalized: distance_to_focus / field_diagonal" },
  { index: 30, name: "stray_sheep_dx", category: "sheep", description: "Relative X distance to designated stray sheep", details: "Normalized offset: (stray_x - own_x) / (field_width - 1). 0.0 if no stray." },
  { index: 31, name: "stray_sheep_dy", category: "sheep", description: "Relative Y distance to designated stray sheep", details: "Normalized offset: (stray_y - own_y) / (field_height - 1). 0.0 if no stray." },
  
  // Sheep Slots (indices 32-49)
  { index: 32, name: "sheep_0_dx", category: "sheep", description: "Relative X offset to 1st closest sheep", details: "Normalized X offset to closest sheep slot 0." },
  { index: 33, name: "sheep_0_dy", category: "sheep", description: "Relative Y offset to 1st closest sheep", details: "Normalized Y offset to closest sheep slot 0." },
  { index: 34, name: "sheep_0_penned", category: "sheep", description: "Penned state of 1st closest sheep", details: "1.0 if penned, 0.0 if unpenned (or if slot empty)." },
  
  { index: 35, name: "sheep_1_dx", category: "sheep", description: "Relative X offset to 2nd closest sheep", details: "Normalized X offset to sheep slot 1." },
  { index: 36, name: "sheep_1_dy", category: "sheep", description: "Relative Y offset to 2nd closest sheep", details: "Normalized Y offset to sheep slot 1." },
  { index: 37, name: "sheep_1_penned", category: "sheep", description: "Penned state of 2nd closest sheep", details: "1.0 if penned, 0.0 if unpenned." },
  
  { index: 38, name: "sheep_2_dx", category: "sheep", description: "Relative X offset to 3rd closest sheep", details: "Normalized X offset to sheep slot 2." },
  { index: 39, name: "sheep_2_dy", category: "sheep", description: "Relative Y offset to 3rd closest sheep", details: "Normalized Y offset to sheep slot 2." },
  { index: 40, name: "sheep_2_penned", category: "sheep", description: "Penned state of 3rd closest sheep", details: "1.0 if penned, 0.0 if unpenned." },
  
  { index: 41, name: "sheep_3_dx", category: "sheep", description: "Relative X offset to 4th closest sheep", details: "Normalized X offset to sheep slot 3." },
  { index: 42, name: "sheep_3_dy", category: "sheep", description: "Relative Y offset to 4th closest sheep", details: "Normalized Y offset to sheep slot 3." },
  { index: 43, name: "sheep_3_penned", category: "sheep", description: "Penned state of 4th closest sheep", details: "1.0 if penned, 0.0 if unpenned." },
  
  { index: 44, name: "sheep_4_dx", category: "sheep", description: "Relative X offset to 5th closest sheep", details: "Normalized X offset to sheep slot 4." },
  { index: 45, name: "sheep_4_dy", category: "sheep", description: "Relative Y offset to 5th closest sheep", details: "Normalized Y offset to sheep slot 4." },
  { index: 46, name: "sheep_4_penned", category: "sheep", description: "Penned state of 5th closest sheep", details: "1.0 if penned, 0.0 if unpenned." },
  
  { index: 47, name: "sheep_5_dx", category: "sheep", description: "Relative X offset to 6th closest sheep", details: "Normalized X offset to sheep slot 5." },
  { index: 48, name: "sheep_5_dy", category: "sheep", description: "Relative Y offset to 6th closest sheep", details: "Normalized Y offset to sheep slot 5." },
  { index: 49, name: "sheep_5_penned", category: "sheep", description: "Penned state of 6th closest sheep", details: "1.0 if penned, 0.0 if unpenned." },
  
  // Teammate slots (indices 50-53)
  { index: 50, name: "other_dog_0_dx", category: "dog", description: "Relative X offset to teammate dog 0", details: "Normalized offset: (teammate_0_x - own_x) / (field_width - 1)" },
  { index: 51, name: "other_dog_0_dy", category: "dog", description: "Relative Y offset to teammate dog 0", details: "Normalized offset: (teammate_0_y - own_y) / (field_height - 1)" },
  { index: 52, name: "other_dog_1_dx", category: "dog", description: "Relative X offset to teammate dog 1", details: "Normalized offset: (teammate_1_x - own_x) / (field_width - 1)" },
  { index: 53, name: "other_dog_1_dy", category: "dog", description: "Relative Y offset to teammate dog 1", details: "Normalized offset: (teammate_1_y - own_y) / (field_height - 1)" },
];

// Hierarchical nodes - 69 nodes
const HIERARCHICAL_NODES: NodeDefinition[] = [
  ...GUIDED_NODES,
  { index: 54, name: "shepherd_cmd_gather", category: "shepherd", description: "Active Shepherd Command: GATHER", details: "1.0 if active. Instructs dog to group dispersed sheep." },
  { index: 55, name: "shepherd_cmd_drive_to_pen", category: "shepherd", description: "Active Shepherd Command: DRIVE_TO_PEN", details: "1.0 if active. Instructs dog to push the flock towards the pen." },
  { index: 56, name: "shepherd_cmd_hold_left", category: "shepherd", description: "Active Shepherd Command: HOLD_LEFT", details: "1.0 if active. Instructs dog to guard the left flank relative to the pen." },
  { index: 57, name: "shepherd_cmd_hold_right", category: "shepherd", description: "Active Shepherd Command: HOLD_RIGHT", details: "1.0 if active. Instructs dog to guard the right flank relative to the pen." },
  { index: 58, name: "shepherd_cmd_block_escape", category: "shepherd", description: "Active Shepherd Command: BLOCK_ESCAPE", details: "1.0 if active. General block fallback when any escape is detected." },
  { index: 59, name: "shepherd_cmd_apply_pressure", category: "shepherd", description: "Active Shepherd Command: APPLY_PRESSURE", details: "1.0 if active. Push from behind to funnel sheep into the pen." },
  { index: 60, name: "shepherd_cmd_back_off", category: "shepherd", description: "Active Shepherd Command: BACK_OFF", details: "1.0 if active. Back off to let sheep settle down." },
  { index: 61, name: "shepherd_cmd_stop", category: "shepherd", description: "Active Shepherd Command: STOP", details: "1.0 if active. All sheep are penned, dogs may halt." },
  { index: 62, name: "dog_id_normalized", category: "identity", description: "Normalized index of this dog", details: "Calculated as: dog_index / max(1, dog_count - 1) if dog_count > 1 else 0.0" },
  { index: 63, name: "dog_count_normalized", category: "identity", description: "Normalized total dog count", details: "Calculated as: dog_count / max(1, MAX_DOG_SLOTS = 5)" },
  { index: 64, name: "dog_id_slot_0", category: "identity", description: "Identity one-hot: Slot 0", details: "1.0 if this is dog 0, 0.0 otherwise." },
  { index: 65, name: "dog_id_slot_1", category: "identity", description: "Identity one-hot: Slot 1", details: "1.0 if this is dog 1, 0.0 otherwise." },
  { index: 66, name: "dog_id_slot_2", category: "identity", description: "Identity one-hot: Slot 2", details: "1.0 if this is dog 2, 0.0 otherwise." },
  { index: 67, name: "dog_id_slot_3", category: "identity", description: "Identity one-hot: Slot 3", details: "1.0 if this is dog 3, 0.0 otherwise." },
  { index: 68, name: "dog_id_slot_4", category: "identity", description: "Identity one-hot: Slot 4", details: "1.0 if this is dog 4, 0.0 otherwise." },
];

// Emergent nodes - 50 nodes
const EMERGENT_NODES: NodeDefinition[] = [
  { index: 0, name: "own_x", category: "position", description: "Dog's own absolute X position", details: "Normalized coordinate: own_x / (field_width - 1)" },
  { index: 1, name: "own_y", category: "position", description: "Dog's own absolute Y position", details: "Normalized coordinate: own_y / (field_height - 1)" },
  { index: 2, name: "dog_id_slot_0", category: "identity", description: "Identity one-hot: Slot 0", details: "1.0 if this is dog 0, 0.0 otherwise (out of HERD_DOG_SLOTS = 3)." },
  { index: 3, name: "dog_id_slot_1", category: "identity", description: "Identity one-hot: Slot 1", details: "1.0 if this is dog 1, 0.0 otherwise." },
  { index: 4, name: "dog_id_slot_2", category: "identity", description: "Identity one-hot: Slot 2", details: "1.0 if this is dog 2, 0.0 otherwise." },
  { index: 5, name: "pen_x", category: "position", description: "Pen center absolute X coordinate", details: "Normalized coordinate: pen_center_x / (field_width - 1)" },
  { index: 6, name: "pen_y", category: "position", description: "Pen center absolute Y coordinate", details: "Normalized coordinate: pen_center_y / (field_height - 1)" },
  { index: 7, name: "flock_center_x", category: "position", description: "Flock center absolute X coordinate", details: "Normalized coordinate: flock_center_x / (field_width - 1)" },
  { index: 8, name: "flock_center_y", category: "position", description: "Flock center absolute Y coordinate", details: "Normalized coordinate: flock_center_y / (field_height - 1)" },
  { index: 9, name: "distance_to_pen", category: "distance", description: "Distance from dog to pen center", details: "Normalized distance: distance_to_pen / field_diagonal" },
  { index: 10, name: "distance_to_flock", category: "distance", description: "Distance from dog to flock center", details: "Normalized distance: distance_to_flock / field_diagonal" },
  { index: 11, name: "flock_spread", category: "sensor", description: "Standard deviation of sheep positions", details: "Normalized spread of the flock: flock_spread / field_diagonal" },
  { index: 12, name: "average_distance_to_pen", category: "distance", description: "Mean distance of all sheep to the pen", details: "Normalized average distance: avg_distance / field_diagonal" },
  { index: 13, name: "wall_left", category: "sensor", description: "Proximity to the left boundary", details: "Calculated as: own_x / (field_width - 1)" },
  { index: 14, name: "wall_right", category: "sensor", description: "Proximity to the right boundary", details: "Calculated as: (field_width - 1 - own_x) / (field_width - 1)" },
  { index: 15, name: "wall_top", category: "sensor", description: "Proximity to the top boundary", details: "Calculated as: own_y / (field_height - 1)" },
  { index: 16, name: "wall_bottom", category: "sensor", description: "Proximity to the bottom boundary", details: "Calculated as: (field_height - 1 - own_y) / (field_height - 1)" },
  { index: 17, name: "blocked_steps", category: "sensor", description: "Steps dog has been blocked/colliding", details: "Scaled output: min(blocked_steps, 10) / 10.0" },
  { index: 18, name: "no_progress_steps", category: "sensor", description: "Steps team has made no progress towards pen", details: "Scaled: min(no_progress, window) / window" },
  { index: 19, name: "revisits_recent_position", category: "sensor", description: "Binary indicator of position revisit loop", details: "1.0 if dog has revisited a recent cell in the past few steps, 0.0 otherwise" },
  { index: 20, name: "two_position_loop", category: "sensor", description: "Binary indicator of two-position oscillation", details: "1.0 if dog is oscillating back and forth between two positions, 0.0 otherwise" },
  { index: 21, name: "stray_present", category: "sensor", description: "Binary indicator if any stray sheep exists", details: "1.0 if at least one sheep is stray (distance to flock > 1.8 * spread), 0.0 otherwise" },
  { index: 22, name: "nearest_unpenned_dx", category: "sheep", description: "Relative X distance to closest unpenned sheep", details: "Normalized: (nearest_x - own_x) / (field_width - 1). Script-free raw sensor." },
  { index: 23, name: "nearest_unpenned_dy", category: "sheep", description: "Relative Y distance to closest unpenned sheep", details: "Normalized: (nearest_y - own_y) / (field_height - 1). Script-free raw sensor." },
  { index: 24, name: "nearest_unpenned_distance", category: "distance", description: "Distance to closest unpenned sheep", details: "Normalized distance: distance_to_nearest / field_diagonal" },
  { index: 25, name: "farthest_unpenned_dx", category: "sheep", description: "Relative X distance to unpenned sheep farthest from pen", details: "Normalized: (farthest_x - own_x) / (field_width - 1). Guides targeting of critical strays." },
  { index: 26, name: "farthest_unpenned_dy", category: "sheep", description: "Relative Y distance to unpenned sheep farthest from pen", details: "Normalized: (farthest_y - own_y) / (field_height - 1). Guides targeting of critical strays." },
  { index: 27, name: "farthest_unpenned_distance", category: "distance", description: "Distance to unpenned sheep farthest from pen", details: "Normalized distance: distance_to_farthest / field_diagonal" },
  
  // Sheep Slots (indices 28-45)
  { index: 28, name: "sheep_0_dx", category: "sheep", description: "Relative X offset to 1st closest sheep", details: "Normalized X offset to closest sheep slot 0." },
  { index: 29, name: "sheep_0_dy", category: "sheep", description: "Relative Y offset to 1st closest sheep", details: "Normalized Y offset to closest sheep slot 0." },
  { index: 30, name: "sheep_0_penned", category: "sheep", description: "Penned state of 1st closest sheep", details: "1.0 if penned, 0.0 if unpenned." },
  
  { index: 31, name: "sheep_1_dx", category: "sheep", description: "Relative X offset to 2nd closest sheep", details: "Normalized X offset to sheep slot 1." },
  { index: 32, name: "sheep_1_dy", category: "sheep", description: "Relative Y offset to 2nd closest sheep", details: "Normalized Y offset to sheep slot 1." },
  { index: 33, name: "sheep_1_penned", category: "sheep", description: "Penned state of 2nd closest sheep", details: "1.0 if penned, 0.0 if unpenned." },
  
  { index: 34, name: "sheep_2_dx", category: "sheep", description: "Relative X offset to 3rd closest sheep", details: "Normalized X offset to sheep slot 2." },
  { index: 35, name: "sheep_2_dy", category: "sheep", description: "Relative Y offset to 3rd closest sheep", details: "Normalized Y offset to sheep slot 2." },
  { index: 36, name: "sheep_2_penned", category: "sheep", description: "Penned state of 3rd closest sheep", details: "1.0 if penned, 0.0 if unpenned." },
  
  { index: 37, name: "sheep_3_dx", category: "sheep", description: "Relative X offset to 4th closest sheep", details: "Normalized X offset to sheep slot 3." },
  { index: 38, name: "sheep_3_dy", category: "sheep", description: "Relative Y offset to 4th closest sheep", details: "Normalized Y offset to sheep slot 3." },
  { index: 39, name: "sheep_3_penned", category: "sheep", description: "Penned state of 4th closest sheep", details: "1.0 if penned, 0.0 if unpenned." },
  
  { index: 40, name: "sheep_4_dx", category: "sheep", description: "Relative X offset to 5th closest sheep", details: "Normalized X offset to sheep slot 4." },
  { index: 41, name: "sheep_4_dy", category: "sheep", description: "Relative Y offset to 5th closest sheep", details: "Normalized Y offset to sheep slot 4." },
  { index: 42, name: "sheep_4_penned", category: "sheep", description: "Penned state of 5th closest sheep", details: "1.0 if penned, 0.0 if unpenned." },
  
  { index: 43, name: "sheep_5_dx", category: "sheep", description: "Relative X offset to 6th closest sheep", details: "Normalized X offset to sheep slot 5." },
  { index: 44, name: "sheep_5_dy", category: "sheep", description: "Relative Y offset to 6th closest sheep", details: "Normalized Y offset to sheep slot 5." },
  { index: 45, name: "sheep_5_penned", category: "sheep", description: "Penned state of 6th closest sheep", details: "1.0 if penned, 0.0 if unpenned." },
  
  // Teammate slots (indices 46-49)
  { index: 46, name: "other_dog_0_dx", category: "dog", description: "Relative X offset to teammate dog 0", details: "Normalized offset: (teammate_0_x - own_x) / (field_width - 1)" },
  { index: 47, name: "other_dog_0_dy", category: "dog", description: "Relative Y offset to teammate dog 0", details: "Normalized offset: (teammate_0_y - own_y) / (field_height - 1)" },
  { index: 48, name: "other_dog_1_dx", category: "dog", description: "Relative X offset to teammate dog 1", details: "Normalized offset: (teammate_1_x - own_x) / (field_width - 1)" },
  { index: 49, name: "other_dog_1_dy", category: "dog", description: "Relative Y offset to teammate dog 1", details: "Normalized offset: (teammate_1_y - own_y) / (field_height - 1)" },
];

interface OutputDefinition {
  index: number;
  name: string;
  type: string;
  description: string;
  communicating: string;
}

const ACTOR_NODES: OutputDefinition[] = [
  { index: 0, name: "up", type: "Movement (Step)", description: "Move 1 unit up", communicating: "Commanding dog to step North on the 2D grid" },
  { index: 1, name: "down", type: "Movement (Step)", description: "Move 1 unit down", communicating: "Commanding dog to step South on the 2D grid" },
  { index: 2, name: "left", type: "Movement (Step)", description: "Move 1 unit left", communicating: "Commanding dog to step West on the 2D grid" },
  { index: 3, name: "right", type: "Movement (Step)", description: "Move 1 unit right", communicating: "Commanding dog to step East on the 2D grid" },
  { index: 4, name: "sprint_up", type: "Movement (Sprint)", description: "Move 2 units up", communicating: "Commanding dog to sprint North (double speed)" },
  { index: 5, name: "sprint_down", type: "Movement (Sprint)", description: "Move 2 units down", communicating: "Commanding dog to sprint South (double speed)" },
  { index: 6, name: "sprint_left", type: "Movement (Sprint)", description: "Move 2 units left", communicating: "Commanding dog to sprint West (double speed)" },
  { index: 7, name: "sprint_right", type: "Movement (Sprint)", description: "Move 2 units right", communicating: "Commanding dog to sprint East (double speed)" },
  { index: 8, name: "wait", type: "Action Void", description: "Stay in place", communicating: "Instructing the dog to remain stationary this step" },
];

const CRITIC_NODES: OutputDefinition[] = [
  { index: 0, name: "value", type: "State Value Estimate", description: "Expected future return", communicating: "Estimating the state value V(s) to compute generalized advantage estimation (GAE)" },
];

function asRecord(value: unknown): Record<string, unknown> | null {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null;
}

export function LayersTab({ effectiveConfig, topologyInfo }: LayersTabProps) {
  // Read active mode from config/topologyInfo
  const root = asRecord(effectiveConfig);
  const training = asRecord(root?.training);
  const activeModeFromConfig = (topologyInfo?.observation_mode || training?.observation_mode || "guided") as string;
  
  // Handle local selector state
  const [selectedViewMode, setSelectedViewMode] = useState<string>(activeModeFromConfig);
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedCategory, setSelectedCategory] = useState<string>("all");

  const activeNodesList = useMemo(() => {
    if (selectedViewMode === "hierarchical") return HIERARCHICAL_NODES;
    if (selectedViewMode === "emergent") return EMERGENT_NODES;
    return GUIDED_NODES;
  }, [selectedViewMode]);

  const filteredNodesList = useMemo(() => {
    return activeNodesList.filter((node) => {
      const matchesSearch =
        node.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
        node.description.toLowerCase().includes(searchQuery.toLowerCase()) ||
        node.details.toLowerCase().includes(searchQuery.toLowerCase());
      
      const matchesCategory = selectedCategory === "all" || node.category === selectedCategory;

      return matchesSearch && matchesCategory;
    });
  }, [activeNodesList, searchQuery, selectedCategory]);

  return (
    <section className="network-tab" aria-label="Model Layers Detailed Card" style={{ paddingBottom: "3rem" }}>
      <div className="network-tab__header" style={{ marginBottom: "2rem" }}>
        <div>
          <p className="eyebrow" style={{ color: "var(--accent)", fontWeight: 600 }}>Model Card</p>
          <h2 style={{ fontSize: "2rem", margin: "0.25rem 0 0.5rem" }}>Neural Network Layers</h2>
          <p className="network-tab__intro" style={{ maxWidth: "800px", color: "var(--muted)" }}>
            A precise mapping of all input features fed into the MLP (Multi-Layer Perceptron) policy network, 
            along with output actions and value estimations.
          </p>
        </div>
      </div>

      {/* Overview Cards */}
      <div className="network-tab__kpis" style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))", gap: "1rem", marginBottom: "2rem" }}>
        <div style={{ background: "rgba(12, 24, 38, 0.5)", border: "1px solid var(--panel-border)", padding: "1.25rem", borderRadius: "0.75rem" }}>
          <span style={{ fontSize: "0.75rem", textTransform: "uppercase", letterSpacing: "0.05em", color: "var(--muted)" }}>Active Mode</span>
          <div style={{ display: "flex", alignItems: "center", gap: "0.5rem", marginTop: "0.25rem" }}>
            <strong style={{ fontSize: "1.25rem" }}>{activeModeFromConfig.toUpperCase()}</strong>
            <span style={{ background: "var(--accent-soft)", color: "var(--accent)", border: "1px solid var(--accent)", fontSize: "0.65rem", padding: "0.15rem 0.4rem", borderRadius: "999px", fontWeight: "bold" }}>
              ACTIVE IN RUNTIME
            </span>
          </div>
        </div>
        <div style={{ background: "rgba(12, 24, 38, 0.5)", border: "1px solid var(--panel-border)", padding: "1.25rem", borderRadius: "0.75rem" }}>
          <span style={{ fontSize: "0.75rem", textTransform: "uppercase", letterSpacing: "0.05em", color: "var(--muted)" }}>Input Layer Size</span>
          <strong style={{ fontSize: "1.5rem", display: "block", marginTop: "0.25rem", color: "#93c5fd" }}>
            {activeNodesList.length} Nodes
          </strong>
        </div>
        <div style={{ background: "rgba(12, 24, 38, 0.5)", border: "1px solid var(--panel-border)", padding: "1.25rem", borderRadius: "0.75rem" }}>
          <span style={{ fontSize: "0.75rem", textTransform: "uppercase", letterSpacing: "0.05em", color: "var(--muted)" }}>Output Action Size</span>
          <strong style={{ fontSize: "1.5rem", display: "block", marginTop: "0.25rem", color: "#818cf8" }}>
            Discrete({ACTOR_NODES.length})
          </strong>
        </div>
        <div style={{ background: "rgba(12, 24, 38, 0.5)", border: "1px solid var(--panel-border)", padding: "1.25rem", borderRadius: "0.75rem" }}>
          <span style={{ fontSize: "0.75rem", textTransform: "uppercase", letterSpacing: "0.05em", color: "var(--muted)" }}>Value Head size</span>
          <strong style={{ fontSize: "1.5rem", display: "block", marginTop: "0.25rem", color: "#6ee7b7" }}>
            {CRITIC_NODES.length} Node
          </strong>
        </div>
      </div>

      {/* Mode Switcher */}
      <div style={{ display: "flex", flexWrap: "wrap", justifyContent: "space-between", alignItems: "center", gap: "1rem", borderBottom: "1px solid var(--panel-border)", paddingBottom: "1rem", marginBottom: "1.5rem" }}>
        <div style={{ display: "flex", gap: "0.5rem" }}>
          {["guided", "hierarchical", "emergent"].map((modeName) => (
            <button
              key={modeName}
              onClick={() => setSelectedViewMode(modeName)}
              style={{
                background: selectedViewMode === modeName ? "linear-gradient(180deg, var(--accent-soft), rgba(244, 197, 66, 0.05))" : "transparent",
                borderColor: selectedViewMode === modeName ? "var(--accent)" : "var(--panel-border)",
                color: selectedViewMode === modeName ? "var(--text)" : "var(--muted)",
                fontSize: "0.85rem",
                padding: "0.5rem 1rem",
                borderRadius: "0.5rem",
                fontWeight: selectedViewMode === modeName ? "600" : "400",
                display: "flex",
                alignItems: "center",
                gap: "0.4rem",
              }}
            >
              {modeName.toUpperCase()}
              {activeModeFromConfig === modeName && (
                <span style={{ width: "6px", height: "6px", background: "var(--good)", borderRadius: "50%" }} title="Active Mode" />
              )}
            </button>
          ))}
        </div>

        {/* Search Input */}
        <div style={{ display: "flex", gap: "0.5rem", flex: "1", maxWidth: "400px" }}>
          <input
            type="text"
            placeholder="Search input features..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            style={{
              width: "100%",
              borderRadius: "0.5rem",
              border: "1px solid var(--panel-border)",
              color: "var(--text)",
              background: "rgba(8, 15, 25, 0.8)",
              padding: "0.5rem 0.8rem",
              fontSize: "0.85rem"
            }}
          />
        </div>
      </div>

      {/* Layout Split: Left for Inputs, Right for Outputs */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 400px", gap: "1.5rem", minHeight: 0, height: "100%" }}>
        
        {/* INPUT LAYER PANEL */}
        <section className="network-tab__card" style={{ padding: "1.5rem", margin: "0", display: "flex", flexDirection: "column", minHeight: 0, height: "100%" }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "1rem" }}>
            <h3 style={{ margin: "0" }}>Input Layer Nodes ({filteredNodesList.length} of {activeNodesList.length})</h3>
            <span style={{ fontSize: "0.8rem", color: "var(--muted)" }}>Mode: {selectedViewMode}</span>
          </div>

          {/* Category Filter Badges */}
          <div style={{ display: "flex", flexWrap: "wrap", gap: "0.4rem", marginBottom: "1.25rem" }}>
            <button
              onClick={() => setSelectedCategory("all")}
              style={{
                padding: "0.2rem 0.6rem",
                fontSize: "0.75rem",
                borderRadius: "999px",
                borderColor: selectedCategory === "all" ? "var(--accent)" : "rgba(148, 163, 184, 0.15)",
                background: selectedCategory === "all" ? "rgba(244, 197, 66, 0.1)" : "rgba(12, 24, 38, 0.4)",
              }}
            >
              All
            </button>
            {Object.entries(CATEGORY_LABELS).map(([catKey, label]) => {
              const hasItems = activeNodesList.some((n) => n.category === catKey);
              if (!hasItems) return null;
              return (
                <button
                  key={catKey}
                  onClick={() => setSelectedCategory(catKey)}
                  style={{
                    padding: "0.2rem 0.6rem",
                    fontSize: "0.75rem",
                    borderRadius: "999px",
                    borderColor: selectedCategory === catKey ? CATEGORY_COLORS[catKey as NodeDefinition["category"]] : "rgba(148, 163, 184, 0.15)",
                    background: selectedCategory === catKey ? `${CATEGORY_COLORS[catKey as NodeDefinition["category"]]}20` : "rgba(12, 24, 38, 0.4)",
                    color: selectedCategory === catKey ? CATEGORY_COLORS[catKey as NodeDefinition["category"]] : "var(--muted)",
                  }}
                >
                  {label}
                </button>
              );
            })}
          </div>

          {/* Nodes Table */}
          <div style={{ overflowY: "auto", flex: 1, minHeight: 0, paddingRight: "0.5rem", scrollbarWidth: "thin", scrollbarColor: "rgba(148, 163, 184, 0.3) transparent" }}>
            <table style={{ width: "100%", borderCollapse: "collapse", textAlign: "left", fontSize: "0.85rem" }}>
              <thead>
                <tr style={{ borderBottom: "1px solid var(--panel-border)" }}>
                  <th style={{ padding: "0.75rem 0.5rem", width: "70px", color: "var(--muted)" }}>Index</th>
                  <th style={{ padding: "0.75rem 0.5rem", width: "180px" }}>Feature Name</th>
                  <th style={{ padding: "0.75rem 0.5rem", width: "180px" }}>Category</th>
                  <th style={{ padding: "0.75rem 0.5rem" }}>Description & Normalization Scale</th>
                </tr>
              </thead>
              <tbody>
                {filteredNodesList.length === 0 ? (
                  <tr>
                    <td colSpan={4} style={{ textAlign: "center", padding: "2rem", color: "var(--muted)" }}>
                      No nodes matching current search/filters.
                    </td>
                  </tr>
                ) : (
                  filteredNodesList.map((node) => (
                    <tr
                      key={node.index}
                      style={{
                        borderBottom: "1px solid rgba(148, 163, 184, 0.08)",
                        transition: "background 100ms ease",
                      }}
                      className="layer-row-hover"
                    >
                      <td style={{ padding: "0.75rem 0.5rem", fontWeight: "bold", color: "var(--muted)" }}>
                        #{node.index}
                      </td>
                      <td style={{ padding: "0.75rem 0.5rem", fontWeight: "600", fontFamily: "monospace", color: "var(--text)" }}>
                        {node.name}
                      </td>
                      <td style={{ padding: "0.75rem 0.5rem" }}>
                        <span
                          style={{
                            display: "inline-block",
                            padding: "0.15rem 0.5rem",
                            borderRadius: "4px",
                            fontSize: "0.7rem",
                            fontWeight: "600",
                            color: CATEGORY_COLORS[node.category],
                            background: `${CATEGORY_COLORS[node.category]}18`,
                            border: `1px solid ${CATEGORY_COLORS[node.category]}40`,
                          }}
                        >
                          {CATEGORY_LABELS[node.category]}
                        </span>
                      </td>
                      <td style={{ padding: "0.75rem 0.5rem" }}>
                        <div style={{ fontWeight: "500" }}>{node.description}</div>
                        <div style={{ fontSize: "0.75rem", color: "var(--muted)", fontStyle: "italic", marginTop: "0.15rem" }}>
                          {node.details}
                        </div>
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </section>

        {/* OUTPUT LAYER PANEL */}
        <div style={{ display: "flex", flexDirection: "column", gap: "1.5rem", overflowY: "auto", height: "100%", paddingRight: "0.5rem", scrollbarWidth: "thin", scrollbarColor: "rgba(148, 163, 184, 0.3) transparent" }}>
          
          {/* ACTOR HEAD */}
          <section className="network-tab__card" style={{ padding: "1.5rem", margin: "0" }}>
            <h3 style={{ margin: "0 0 0.25rem", color: "#818cf8" }}>Actor Head (Policy Logits)</h3>
            <p style={{ fontSize: "0.78rem", color: "var(--muted)", margin: "0 0 1rem" }}>
              Outputs 9 raw logits representing dog movement actions.
            </p>
            <div style={{ display: "flex", flexDirection: "column", gap: "0.5rem" }}>
              {ACTOR_NODES.map((act) => (
                <div
                  key={act.index}
                  style={{
                    background: "rgba(8, 15, 25, 0.4)",
                    border: "1px solid rgba(148, 163, 184, 0.08)",
                    borderRadius: "0.5rem",
                    padding: "0.6rem 0.8rem",
                    display: "grid",
                    gridTemplateColumns: "35px 90px 1fr",
                    gap: "0.5rem",
                    alignItems: "center"
                  }}
                >
                  <span style={{ fontWeight: "bold", color: "var(--muted)", fontSize: "0.8rem" }}>#{act.index}</span>
                  <span style={{ fontFamily: "monospace", fontWeight: "600", fontSize: "0.85rem", color: "#818cf8" }}>{act.name}</span>
                  <div style={{ fontSize: "0.75rem" }}>
                    <div style={{ fontWeight: "600", color: "var(--text)" }}>{act.description}</div>
                    <div style={{ color: "var(--muted)", fontSize: "0.7rem", marginTop: "0.1rem" }}>{act.communicating}</div>
                  </div>
                </div>
              ))}
            </div>
          </section>

          {/* CRITIC HEAD */}
          <section className="network-tab__card" style={{ padding: "1.5rem", margin: "0" }}>
            <h3 style={{ margin: "0 0 0.25rem", color: "#6ee7b7" }}>Critic Head (Value Head)</h3>
            <p style={{ fontSize: "0.78rem", color: "var(--muted)", margin: "0 0 1rem" }}>
              Outputs a single value estimate for GAE advantage computation.
            </p>
            <div>
              {CRITIC_NODES.map((crit) => (
                <div
                  key={crit.index}
                  style={{
                    background: "rgba(8, 15, 25, 0.4)",
                    border: "1px solid rgba(148, 163, 184, 0.08)",
                    borderRadius: "0.5rem",
                    padding: "0.6rem 0.8rem",
                    display: "grid",
                    gridTemplateColumns: "35px 90px 1fr",
                    gap: "0.5rem",
                    alignItems: "center"
                  }}
                >
                  <span style={{ fontWeight: "bold", color: "var(--muted)", fontSize: "0.8rem" }}>#{crit.index}</span>
                  <span style={{ fontFamily: "monospace", fontWeight: "600", fontSize: "0.85rem", color: "#6ee7b7" }}>{crit.name}</span>
                  <div style={{ fontSize: "0.75rem" }}>
                    <div style={{ fontWeight: "600", color: "var(--text)" }}>{crit.description}</div>
                    <div style={{ color: "var(--muted)", fontSize: "0.7rem", marginTop: "0.1rem" }}>{crit.communicating}</div>
                  </div>
                </div>
              ))}
            </div>
          </section>

          {/* IMPLEMENTATION NOTE */}
          <section className="network-tab__card" style={{ padding: "1.25rem", margin: "0", background: "rgba(244, 197, 66, 0.03)", borderColor: "rgba(244, 197, 66, 0.2)" }}>
            <h4 style={{ margin: "0 0 0.5rem", color: "var(--accent)", fontSize: "0.85rem" }}>Action Masking Pipeline</h4>
            <p style={{ fontSize: "0.75rem", color: "var(--muted)", margin: "0", lineHeight: "1.4" }}>
              Logits outputted by the Actor head are intercepted by the <strong>MaskablePPO</strong> policy. 
              The environment computes valid moves (e.g. avoiding blocking borders) and maps invalid moves to -inf 
              prior to softmax, ensuring invalid actions have absolute 0% probability.
            </p>
          </section>

        </div>
      </div>
    </section>
  );
}

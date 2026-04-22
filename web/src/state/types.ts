export interface Point {
  x: number;
  y: number;
}

export interface Pen {
  origin: Point;
  width: number;
  height: number;
  opening?: "left" | "right" | "top" | "bottom";
}

export interface AgentSnapshot {
  index: number;
  x: number;
  y: number;
  penned?: boolean;
  last_action?: string;
}

export interface RewardBreakdown {
  progress_to_pen: number;
  sheep_penned: number;
  flock_cohesion: number;
  scatter_penalty: number;
  time_penalty: number;
  no_progress_penalty: number;
  wall_pressure_penalty: number;
  wait_penalty: number;
  terminal_success: number;
  terminal_failure: number;
  total: number;
}

export interface PressureDebugDog {
  index: number;
  desired_pressure_target: Point;
  distance_to_pressure_target: number;
  pressure_side_alignment: number;
  between_flock_and_pen: boolean;
  inside_or_too_close_to_flock: boolean;
  distance_to_flock?: number;
  flock_buffer_radius?: number;
  focus_mode?: string;
  distance_to_focus_sheep?: number | null;
  holding_pressure_position?: boolean;
  role_slot?: number;
}

export interface ReplayDebugSnapshot {
  curriculum_stage: number;
  enable_instinct_rewards: boolean;
  policy_mode?: string;
  allow_instinct_target_awareness?: boolean;
  handler_target_enabled?: boolean;
  flock_center?: Point | null;
  dogs: PressureDebugDog[];
}

export interface ReplaySnapshot {
  step: number;
  simulated_seconds: number;
  field_width?: number;
  field_height?: number;
  dogs: AgentSnapshot[];
  sheep: AgentSnapshot[];
  pen: Pen;
  fence_cells?: Array<[number, number]>;
  penned_count: number;
  average_distance_to_pen: number;
  flock_spread: number;
  no_progress_steps: number;
  terminated: boolean;
  timeout: boolean;
  stopped: boolean;
  success: boolean;
  status: string;
  debug?: ReplayDebugSnapshot;
}

export interface ReplayFrame {
  step: number;
  actions: string[];
  snapshot: ReplaySnapshot;
  reward: RewardBreakdown;
}

export interface ReplayBundle {
  seed: number;
  policy_name: string;
  final_snapshot: ReplaySnapshot;
  stats: {
    steps: number;
    simulated_seconds: number;
    sheep_penned: number;
    timeout: boolean;
    terminated: boolean;
    success: boolean;
    stopped: boolean;
    stop_reason: string;
    reward_total: number;
    no_progress_steps: number;
    final_avg_distance_to_pen: number;
    final_flock_spread: number;
    final_reward_breakdown: RewardBreakdown;
  };
  frames: ReplayFrame[];
}

export interface TrainingStatus {
  running: boolean;
  fast_mode: boolean;
  enable_instinct_rewards: boolean;
  policy_mode?: string;
  allow_instinct_target_awareness?: boolean;
  handler_target_enabled?: boolean;
  debug_reward_breakdown: boolean;
  curriculum_stage: number;
  requested_episodes: number;
  completed_episodes: number;
  batch_total_episodes: number;
  batch_completed_episodes: number;
  total_episodes_trained: number;
  current_episode: number | null;
  checkpoint_episode: number | null;
  latest_checkpoint_episode: number | null;
  latest_seed: number | null;
  latest_replay_path: string | null;
  best_score: number | null;
  phase: string;
  message: string;
  error: string | null;
}

export interface TrainingStartRequest {
  episodes: number;
  fast_mode: boolean;
  enable_instinct_rewards: boolean;
  curriculum_stage: number;
  debug_reward_breakdown: boolean;
}

export interface ReplayRunRequest {
  seed: number;
}

export interface EvaluationRecord {
  seed: number;
  success: boolean;
  timeout: boolean;
  stopped: boolean;
  steps: number;
  simulated_seconds: number;
  sheep_penned: number;
  final_sheep_distance_to_pen: number;
  no_progress_steps: number;
  reward_total: number;
  reward_breakdown: RewardBreakdown;
  replay_path: string;
}

export interface CheckpointEntry {
  checkpoint_episode: number;
  checkpoint: string;
  evaluation: string;
  replay: string;
  success_rate: number;
  timeout_rate: number;
  average_completion_steps: number;
  average_completion_seconds: number;
  average_sheep_penned: number;
  average_reward: number;
  records: EvaluationRecord[];
}

export interface EvaluationSummary {
  checkpoint_episode: number;
  policy_name: string;
  records: EvaluationRecord[];
  success_rate: number;
  timeout_rate: number;
  average_completion_steps: number;
  average_completion_seconds: number;
  average_sheep_penned: number;
  average_reward: number;
}

export interface CheckpointIndex {
  checkpoints: CheckpointEntry[];
  latest: EvaluationSummary;
}

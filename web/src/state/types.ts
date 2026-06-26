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
  role?: string;
  personality?: string | null;
  color?: string | null;
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
  grid_width?: number;
  grid_height?: number;
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
  trainer_type?: string;
  policy_type?: string;
  policy_mode?: string;
  replay_mode?: string;
  checkpoint_episode?: number | null;
  total_training_episodes?: number | null;
  environment?: {
    dogs: number;
    sheep: number;
    width: number;
    height: number;
    curriculum_stage: number;
    enable_instinct_rewards: boolean;
  };
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
    role_distribution: Record<string, number>;
    role_switches: number;
    collector_activations: number;
    blocker_activations: number;
    sheep_split_events: number;
    final_reward_breakdown: RewardBreakdown;
  };
  frames: ReplayFrame[];
  scenario_id?: string;
  scenario_name?: string;
}

export interface TrainingStatus {
  running: boolean;
  fast_mode: boolean;
  trainer_type?: string;
  policy_type?: string;
  enable_instinct_rewards: boolean;
  policy_mode?: string;
  replay_mode?: string;
  allow_instinct_target_awareness?: boolean;
  handler_target_enabled?: boolean;
  debug_reward_breakdown: boolean;
  auto_promote?: boolean;
  auto_promote_threshold?: number;
  auto_promote_stages_completed?: number;
  auto_promote_gate?: AutoPromoteGateDiagnostics;
  available_curriculum_stages?: number[];
  max_curriculum_stage?: number;
  curriculum_stage: number;
  requested_episodes: number;
  completed_episodes: number;
  batch_total_episodes: number;
  batch_completed_episodes: number;
  total_episodes_trained: number;
  stage_history: Record<string, number>;
  grand_total_episodes: number;
  current_episode: number | null;
  checkpoint_episode: number | null;
  latest_checkpoint_episode: number | null;
  latest_seed: number | null;
  latest_replay_path: string | null;
  best_score: number | null;
  latest_success_rate: number | null;
  latest_avg_sheep_penned: number | null;
  latest_avg_reward: number | null;
  latest_timeout_rate: number | null;
  latest_stopped_rate: number | null;
  latest_avg_no_progress_steps: number | null;
  latest_avg_distance_to_pen: number | null;
  latest_avg_flock_spread: number | null;
  latest_avg_farthest_distance_to_pen: number | null;
  latest_avg_farthest_distance_to_flock_center: number | null;
  phase: string;
  message: string;
  error: string | null;
  error_type?: string | null;
  traceback?: string | null;
  seed_episode?: number | null;
  starting_episode: number | null;
}

export interface AutoPromoteGateDiagnostics {
  decision: "pending" | "hold" | "promote";
  reason: string;
  seed_count: number;
  success_count: number;
  best_success: number;
  best_reward: number | null;
  seed_gate_ok: boolean;
  success_rate_ok: boolean;
  timeout_ok: boolean;
  reward_close_ok: boolean;
  qualified_streak: number;
  min_qualified_streak: number;
  seed_gate_hits: number;
  min_seed_gate_hits: number;
  seed_gate_target_met: boolean;
  full_success_hits: number;
  min_full_success_hits: number;
  full_success_target_met: boolean;
  success_threshold: number;
  max_timeout_rate: number;
  reward_tolerance_ratio: number;
}

export interface TrainingStartRequest {
  episodes: number;
  fast_mode: boolean;
  enable_instinct_rewards: boolean;
  curriculum_stage: number;
  debug_reward_breakdown: boolean;
  auto_promote?: boolean;
  promote_from_checkpoint_episode?: number | null;
}

export interface ReplayRunRequest {
  seed: number;
  checkpoint_episode?: number | null;
  policy_mode?: string;
  trainer_type?: string;
  policy_type?: string;
  effective_config?: {
    enable_instinct_rewards: boolean;
    curriculum_stage: number;
    debug_reward_breakdown: boolean;
  };
  /** Per-run environment tweaks (not persisted to Config hyperparams). */
  environment_overrides?: {
    sheep_personality_strength?: number;
  };
}

export interface EvaluationRecord {
  seed: number;
  success: boolean;
  timeout: boolean;
  stopped: boolean;
  stop_reason?: string;
  spawn_mode?: string;
  steps: number;
  simulated_seconds: number;
  sheep_penned: number;
  final_sheep_distance_to_pen: number;
  final_flock_spread?: number;
  final_farthest_distance_to_pen?: number;
  final_farthest_distance_to_flock_center?: number;
  no_progress_steps: number;
  reward_total: number;
  reward_breakdown: RewardBreakdown;
  replay_path: string;
}

export interface CheckpointEntry {
  checkpoint_episode: number;
  recorded_at?: string;
  checkpoint: string;
  evaluation: string;
  replay: string;
  policy_name?: string;
  trainer_type?: string;
  policy_type?: string;
  policy_mode?: string;
  replay_mode?: string;
  total_training_episodes?: number;
  environment_config?: {
    dogs: number;
    sheep: number;
    width: number;
    height: number;
  };
  reward_config?: {
    instincts?: {
      curriculum_stage?: number;
      enable_instinct_rewards?: boolean;
    };
  };
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
  trainer_type?: string;
  policy_type?: string;
  records: EvaluationRecord[];
  success_rate: number;
  timeout_rate: number;
  average_completion_steps: number;
  average_distance_to_pen?: number;
  average_flock_spread?: number;
  stopped_rate?: number;
  average_no_progress_steps?: number;
  average_farthest_distance_to_pen?: number;
  average_farthest_distance_to_flock_center?: number;
  average_completion_seconds: number;
  average_sheep_penned: number;
  average_reward: number;
}

export interface CheckpointIndex {
  checkpoints: CheckpointEntry[];
  latest: EvaluationSummary | null;
}

export interface SavedScenario {
  id: string;
  name: string;
  created_at: string;
  seed: number;
  width: number;
  height: number;
  dogs: AgentSnapshot[];
  sheep: AgentSnapshot[];
  pen: Pen;
  sheep_personality_strength?: number;
  sheep_personality_seed_offset?: number;
  seed_offset?: number;
  description?: string;
  notes?: string;
}

export interface ScenarioRunResult {
  scenario_id: string;
  checkpoint_episode: number;
  success: boolean;
  sheep_penned: number;
  steps: number;
  timeout: boolean;
  stopped: boolean;
  reward_total: number;
  replay_path: string;
}

export interface ScenarioIndex {
  scenarios: SavedScenario[];
  best_by_scenario: Record<string, ScenarioRunResult & { checkpoint_episode: number }>;
  runs: ScenarioRunResult[];
  latest_checkpoint_episode: number | null;
  latest_runs: ScenarioRunResult[];
}

export interface ConfigTrainingSettings {
  episodes: number;
  fast_mode: boolean;
  enable_instinct_rewards: boolean;
  curriculum_stage: number;
  debug_reward_breakdown: boolean;
}

export interface ConfigRevision {
  id: number;
  timestamp: string;
  label: string;
  source: "training_start" | "manual";
  training_settings: ConfigTrainingSettings;
  config: Record<string, unknown>;
}

export interface ConfigHistory {
  revisions: ConfigRevision[];
}

export interface UserHyperparamsEnvironment {
  sheep_personality_strength: number;
  sheep_speed: number;
  sheep_vision: number;
  flock_radius: number;
  dog_speed: number;
  dog_sprint_multiplier: number;
  dog_vision: number;
}

export interface UserHyperparamsTraining {
  learning_rate: number;
  learning_rate_final: number;
  entropy_coef: number;
  gamma: number;
  gae_lambda: number;
  clip_range: number;
  rollout_steps: number;
  batch_size: number;
  value_coef: number;
}

export interface UserHyperparamsRewards {
  time_penalty: number;
  progress_scale: number;
  sheep_penned_reward: number;
  wait_penalty: number;
  no_progress_penalty: number;
  terminal_success_reward: number;
  terminal_failure_penalty: number;
  flock_cohesion_scale: number;
  scatter_penalty_scale: number;
  sprint_cost_scale: number;
}

export interface UserHyperparams {
  environment: UserHyperparamsEnvironment;
  training: UserHyperparamsTraining;
  rewards: UserHyperparamsRewards;
}

export interface NetworkTopologyInfo {
  observation_mode: string;
  hidden_layer_sizes: number[];
  observation_size: number;
  action_size: number;
  actor_head: {
    type: string;
    node_count: number;
    output: string;
  };
  critic_head: {
    type: string;
    node_count: number;
    output: string;
  };
  action_masking_enabled: boolean;
  connectivity: string;
}

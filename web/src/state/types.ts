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

export type EpisodeOutcome = "win" | "loss" | "timeout";

export interface EpisodeRecord {
  episode_id: number | string;
  timestamp: string;
  stage: number | null;
  outcome: EpisodeOutcome;
  outcome_label: string;
  total_moves: number;
  reward?: number;
  sheep_penned?: number;
  total_sheep?: number;
  seed?: number;
  policy_name?: string;
  checkpoint_id?: string | null;
  replayAvailable: boolean;
  replaySource?: "training-diagnostic" | "checkpoint-evaluation" | "scenario-evaluation" | "reproduced" | null;
  replayUrl?: string | null;
  replay_id?: string | null;
  capture_reason?: string | null;
  capture_status?: "not_requested" | "queued" | "writing" | "available" | "failed" | "pruned" | null;
  initial_state?: ReplaySnapshot;
  move_history?: ReplayFrame[];
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
  active_curriculum_stage?: number;
  curriculum_stage: number;
  requested_episodes: number;
  completed_episodes: number;
  estimated_equivalent_episodes?: number;
  batch_total_episodes: number;
  batch_completed_episodes: number;
  batch_total_segments?: number;
  batch_completed_segments?: number;
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
  resume_available?: boolean;
  resume_remaining_episodes?: number | null;
  resume_request?: TrainingStartRequest | null;
  run_id?: string;
  parent_run_id?: string;
  parent_checkpoint_id?: string;
  active_model_source?: string;
  active_checkpoint_id?: string;
  training_start_time?: string;
  last_policy_update_time?: string;
  last_evaluation_time?: string;
  policy_version?: number;
  ppo_update_count?: number;
  approx_kl?: number;
  clip_fraction?: number;
  explained_variance?: number;
  anti_collapse_warning?: {
    triggered: boolean;
    message: string;
    recommendation?: string;
  } | null;
  runtime?: TrainingRuntimeSummary;
  episodes_in_stage?: number;
  stage_success_count?: number;
  stage_success_rate?: number;
  total_timesteps?: number;
  checkpoint_save_interval?: number;
  active_policy_identity?: string;
  adaptive_lr_stage?: number;
  adaptive_lr_stage_max?: number;
  adaptive_lr_stage_label?: string;
  adaptive_lr_multiplier?: number;
  effective_learning_rate?: number;
  effective_mutation_scale?: number;

  // Explicit Live Telemetry Fields
  current_stage_environment_episode?: number;
  latest_completed_environment_episode?: number;
  latest_completed_episode_id?: number;
  live_rollout_window_count?: number;
  live_rollout_success_count?: number;
  live_rollout_failure_count?: number;
  live_rollout_stopped_count?: number;
  live_rollout_timeout_count?: number;
  live_rollout_success_rate?: number | null;
  episodes_since_latest_confidence_evaluation?: number;
  latest_confidence_environment_episode?: number;
  current_global_timestep?: number;
  latest_checkpoint_global_timestep?: number;
  timesteps_since_latest_checkpoint?: number;
  latest_episode_completed_at?: string;
  latest_episode_reward?: number;
  latest_episode_result?: string;
  telemetry_dropped_count?: number;
  telemetry_error_count?: number;
  evaluation_schedule_unit?: string;
  latest_evaluated_environment_episode?: number;
  next_evaluation_environment_episode?: number;
  episodes_until_next_evaluation?: number;
}

export interface RuntimeSessionRecord {
  session_id: string;
  run_id?: string | null;
  started_at: string;
  last_heartbeat_at: string;
  ended_at?: string | null;
  end_reason?: string | null;
  status: string;
  current_phase?: string | null;
}

export interface TrainingRuntimeSummary {
  training_seconds: number;
  evaluation_seconds: number;
  replay_capture_seconds: number;
  replay_serialization_seconds: number;
  checkpoint_save_seconds: number;
  paused_seconds: number;
  active_seconds_total: number;
  wall_clock_seconds: number;
  offline_or_unknown_seconds: number;
  session_id?: string | null;
  current_phase?: string | null;
  session_count: number;
  sessions: RuntimeSessionRecord[];
  episodes_per_active_hour?: number | null;
  timesteps_per_training_second?: number | null;
  training_time_percentage?: number | null;
}

export interface AutoPromoteGateDiagnostics {
  ready?: boolean;
  decision: "pending" | "hold" | "promote" | "promote_ready" | "blocked" | string;
  status_text?: "READY TO PROMOTE" | "NOT READY" | "COLLECTING EVIDENCE" | string;
  reason: string;
  blocking_reasons?: string[];
  stage?: number;
  success_threshold?: number;
  window_size?: number;
  formal_evaluations_available?: number;
  formal_evaluations_required?: number;
  qualified_evaluations?: number;
  qualified_evaluations_required?: number;
  recent_average_success?: number;
  persistent_seed_failure?: boolean;
  blocking_seed?: number | null;
  blocking_seed_consecutive_failures?: number;
  blocking_seeds?: number[];
  seed_count?: number;
  success_count?: number;
  best_success?: number;
  best_reward?: number | null;
  seed_gate_ok?: boolean;
  success_rate_ok?: boolean;
  timeout_ok?: boolean;
  reward_close_ok?: boolean;
  minimum_required_evaluations?: number;
  minimum_seed_trials?: number;
  total_seed_trials?: number;
  total_successes?: number;
  aggregate_success_rate?: number;
  aggregate_timeout_rate?: number;
  latest_success_rate?: number;
  recent_qualifying_checkpoints?: number;
  recent_checkpoints_considered?: number;
  latest_floor_passed?: boolean;
  reward_guard_passed?: boolean;
  seed_consistency_passed?: boolean;
  step_efficiency_improving?: boolean;
  step_efficiency_delta_pct?: number | null;
  recent_avg_steps?: number | null;
  step_improvement_plateaued?: boolean;
  [key: string]: any;
}

export interface TrainingStartRequest {
  episodes: number;
  fast_mode: boolean;
  enable_instinct_rewards: boolean;
  curriculum_stage: number;
  debug_reward_breakdown: boolean;
  auto_promote?: boolean;
  promote_from_checkpoint_episode?: number | null;
  resume?: boolean;
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
  policy_version?: number;
  initial_sheep_distance_to_pen?: number;
  min_sheep_distance_to_pen?: number;
  final_dog_to_sheep_distance?: number;
  final_dog_positions?: [number, number][];
  final_sheep_positions?: [number, number][];
  pen_position?: [number, number];
  num_waits?: number;
  num_sprints?: number;
  num_invalid_actions?: number;
  most_frequent_action?: string;
  oscillation_detected?: boolean;
  observation_diagnostics?: {
    feature_names: string[];
    vector_length: number;
    min_values: number[];
    max_values: number[];
    mean_values: number[];
    std_values: number[];
    constant_features: string[];
    nan_or_inf_features: string[];
    saturated_features: string[];
    bounds_mismatch: boolean;
  };
  failed_trajectory_summary?: {
    step: number;
    dog_positions: [number, number][];
    sheep_positions: [number, number][];
    sheep_distance_to_pen: number;
    dog_to_sheep_distance: number;
    selected_actions: string[];
    reward: number;
    reward_breakdown: Record<string, number>;
    no_progress_counter: number;
    event?: string;
  }[];
  last_actions_before_failure?: string[][];
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
  policy_state_path?: string;
  policy_config?: any;
  run_id?: string;
  checkpoint_id?: string;
  parent_run_id?: string;
  parent_checkpoint_id?: string;
  global_timesteps?: number;
  observation_schema_hash?: string;
  action_space_hash?: string;
  environment_config_version?: string;
  reward_schema_version?: string;
  deterministic_evaluation?: boolean;
  evaluation_seeds?: number[];
  evaluation_seed_count?: number;
  replay_mode?: string;
  total_training_episodes?: number;
  cumulative_environment_episodes?: number;
  policy_version?: number;
  policy_gradient_loss?: number;
  value_loss?: number;
  entropy_loss?: number;
  loss?: number;
  approx_kl?: number;
  clip_fraction?: number;
  explained_variance?: number;
  environment_config?: {
    dogs: number;
    sheep: number;
    width: number;
    height: number;
    curriculum_stage?: number;
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
  /** Journey identifier — "current" for the active journey, "journey-YYYYMMDD-HHMMSS" for archived ones. */
  journey?: string;
  stopped_rate?: number;
  average_distance_to_pen?: number;
  average_sheep_distance_to_pen?: number;
  average_flock_spread?: number;
  global_timestep?: number | null;
  created_timestamp?: string | null;
  active_runtime_seconds_total?: number | null;
  training_seconds_total?: number | null;
  evaluation_seconds_total?: number | null;
  wall_clock_elapsed_seconds?: number | null;
  session_id?: string | null;
  evaluation_mode?: "quick" | "confidence" | string;
  promotion_eligible?: boolean;
  curriculum_stage?: number;
  promotion_gate?: AutoPromoteGateDiagnostics | null;
  evaluation_timestamp?: string | null;
  adaptive_lr_stage?: number;
  adaptive_lr_stage_max?: number;
  adaptive_lr_stage_label?: string;
  adaptive_lr_multiplier?: number;
  effective_learning_rate?: number;
  effective_mutation_scale?: number;
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

export interface TelemetryRecord {
  timestamp: string;
  step: number;
  stage: number;
  success_rate: number;
  metrics: {
    average_reward: number;
    timeout_rate: number;
    average_sheep_penned: number;
    approx_kl?: number;
    clip_fraction?: number;
    explained_variance?: number;
  };
  hyperparameters: {
    learning_rate: number;
    learning_rate_final: number;
    entropy_coef: number;
    gae_lambda: number;
  };
  run_id?: string;
  checkpoint_id?: string;
  evaluation_id?: string;
  global_episode?: number;
  episode_in_stage?: number;
  recorded_at?: string;
}

export interface SnapshotIdentity {
  snapshot_timestamp: string;
  active_run_id: string;
  active_checkpoint_id: string;
  loaded_model_id: string;
  policy_version: number | null;
  ppo_update_count: number;
  global_timestep: number;
  current_rollout_progress: string;
  current_curriculum_stage: number;
  config_hash: string;
  observation_schema_hash: string;
  action_space_hash: string;
  reward_schema_version: string;
  evaluation_timestamp: string;
  evaluation_policy_version: number | null;
  evaluation_checkpoint_id: string | null;
  latest_current_stage_evaluation?: CheckpointEntry | null;
  latest_any_stage_evaluation?: CheckpointEntry | null;
  current_stage_promotion_gate?: AutoPromoteGateDiagnostics | null;
}

export interface CompletenessRow {
  area: string;
  status: string;
  source: string;
  missing: string[];
}

export interface DiagnosticCompleteness {
  table: CompletenessRow[];
  readiness: string;
  reasons: string[];
}

export interface NeuralArchitecture {
  status: string;
  algorithm?: string;
  policy_class?: string;
  feed_forward_or_recurrent?: string;
  observation_space_shape?: number[];
  observation_data_type?: string;
  observation_feature_count?: number;
  feature_extractor_class?: string;
  feature_extractor_output_dimension?: number;
  actor_hidden_layers?: number[];
  critic_hidden_layers?: number[];
  shared_layers?: number[];
  activation_function?: string;
  action_space_type?: string;
  action_count?: number;
  ordered_action_mapping?: string[];
  distribution_type?: string;
  orthogonal_initialization_setting?: boolean;
  normalization_settings?: string;
  total_trainable_parameter_count?: number;
  device?: string;
  configured_architecture?: string;
  loaded_architecture?: string;
  compatibility_status?: string;
  message?: string;
}

export interface PPOMetric {
  checkpoint_episode: number;
  policy_gradient_loss: number | null;
  value_loss: number | null;
  entropy_loss: number | null;
  loss: number | null;
  approx_kl: number | null;
  clip_fraction: number | null;
  explained_variance: number | null;
}



export interface CounterRow {
  counter: string;
  value: number | null;
  unit: string;
  source: string;
  definition: string;
}

export interface CounterReconciliation {
  rows: CounterRow[];
  warnings: string[];
}

export interface RewardReconciliation {
  seed: number;
  success: boolean;
  reported_reward: number;
  summed_components: number;
  difference: number;
  status: string;
  breakdown: Record<string, number>;
}

export interface ConfigSnapshotValue {
  default: any;
  ui: any;
  stage: any;
  checkpoint: any;
  active: any;
  source: string;
}

export interface DiagnosticsSnapshot {
  snapshot: SnapshotIdentity;
  completeness: DiagnosticCompleteness;
  config_snapshot: Record<string, ConfigSnapshotValue>;
  config_anomalies: string[];
  environment_mismatches: Array<{
    component: string;
    field: string;
    training_value: any;
    evaluation_value: any;
    severity: string;
  }>;
  scenario_coverage: {
    unique_seeds_count: number;
    unique_configs_count: number;
    sheep_to_pen_distance: { min: number; max: number; avg: number };
    dog_to_sheep_distance: { min: number; max: number; avg: number };
    resemblance_counts: Record<string, number>;
    resemblance_successes: Record<string, number>;
  };
  version_history: Record<string, {
    checkpoint_episode: number;
    success_rate: number;
    average_reward: number;
    average_completion_steps: number;
    failures: number[];
  }>;
  failed_seed_trends: Record<string, {
    currently_failing: boolean;
    distance_delta: number;
    reward_delta: number;
    status: string;
    trend_classification: string;
  }>;
  reward_reconciliations: RewardReconciliation[];
  eval_geometry_validations: Record<string, {
    error?: string;
    pen_origin?: number[];
    pen_dimensions?: number[];
    grid_dimensions?: number[];
    overlap_detected?: boolean;
    boundary_violation?: boolean;
    spacing_violation?: boolean;
    can_enter_pen_heuristic?: boolean;
    dog_has_space_behind_heuristic?: boolean;
    material_difficulty_difference?: boolean;
  }>;
  neural_architecture: NeuralArchitecture;
  ppo_metrics: PPOMetric[];
  evaluation_records: EvaluationRecord[];
  failed_seed_trajectories: Record<string, any[]>;
  observation_diagnostics: any;
  counter_reconciliation: CounterReconciliation;
  health_warnings: string[];
  training_status: TrainingStatus;
}

export interface DiagnosticsResponse {
  diagnosticsAvailable: boolean;
  snapshot: DiagnosticsSnapshot | null;
  error: {
    code: string;
    message: string;
    exceptionType: string;
    endpoint: string;
  } | null;
}

export interface TrainingEpisode {
  id: number;
  event_key: string;
  run_id: string | null;
  session_id: string | null;
  global_environment_episode: number;
  episode_in_stage: number;
  curriculum_stage: number;
  global_timestep: number | null;
  policy_version: number | null;
  completed_at: string;
  active_runtime_seconds_total: number | null;
  reward: number;
  result: string;
  success: boolean;
  timeout: boolean;
  stopped: boolean;
  sheep_penned: number;
  total_sheep: number;
  steps: number;
  seed: number | null;
  checkpoint_id: string | null;
  replay_available?: boolean;
  replay_id?: string | null;
  replay_path?: string | null;
  replay_source?: string | null;
  capture_reason?: string | null;
  capture_status?: string | null;
  reward_breakdown?: Record<string, number> | null;
}

export interface CapturePolicyConfig {
  mode: "off" | "failures" | "selective" | "next_n" | "all";
  next_n_counter: number;
  success_sample_rate: number;
  target_stage: number | null;
  target_outcome: string;
  queued_writes: number;
  written_count: number;
  dropped_count: number;
  failure_count: number;
  max_replays_per_stage?: number;
  max_total_replays?: number;
  max_disk_mb?: number;
}

export interface TrainingEpisodesResponse {
  episodes: TrainingEpisode[];
  latest_id: number;
  next_after_id: number;
  has_more: boolean;
  oldest_available_timestamp: string | null;
  total_matching: number;
  max_id?: number;
}

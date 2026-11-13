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
  source_stage?: number;
  target_environment_stage?: number;
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
  early_warnings_summary?: EarlyWarningsSummary;
  early_warnings_monitoring_status?: string;
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
  evaluation_in_progress?: boolean;
  evaluation_status?: EvaluationProgressStatus;
  evaluation_mode?: string | null;
  evaluation_seed?: number | null;
  evaluation_seed_index?: number;
  evaluation_total_seeds?: number;
  evaluation_completed_seeds?: number;
  evaluation_success_count?: number;
  evaluation_message?: string | null;
  evaluation_latest_result?: EvaluationSeedResult | null;
  evaluation_recent_results?: EvaluationSeedResult[];
  mode?: string;
  read_only_preview?: boolean;
}

export interface EvaluationSeedResult {
  seed: number;
  success: boolean;
  timeout: boolean;
  status: string;
  reward: number;
  penned: number;
  total_sheep: number;
  steps: number;
}

export interface EvaluationProgressStatus {
  in_progress: boolean;
  mode?: string | null;
  current_seed?: number | null;
  seed_index?: number;
  total_seeds?: number;
  completed_seeds?: number;
  success_count?: number;
  timeout_count?: number;
  success_rate?: number | null;
  seeds?: number[];
  message?: string | null;
  latest_result?: EvaluationSeedResult | null;
  recent_results?: EvaluationSeedResult[];
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

export interface SessionEventParamDiff {
  param: string;
  full_path?: string;
  from_val: string;
  to_val: string;
  category?: "architecture" | "tuning" | "environment" | "rewards" | "training" | string;
}

export interface SessionEventStats {
  checkpoint_count?: number;
  first_episode?: number | null;
  last_episode?: number | null;
  first_timestep?: number | null;
  last_timestep?: number | null;
  avg_success_rate?: number | null;
  avg_reward?: number | null;
  avg_steps?: number | null;
  training_seconds?: number | null;
  active_seconds_total?: number | null;
  end_reason?: string | null;
  status?: string | null;
}

export interface SessionChangeEvent {
  id: string;
  session_id: string;
  session_index: number;
  run_id?: string | null;
  timestamp: string;
  timestamp_ms: number;
  ended_at?: string | null;
  ended_ms?: number | null;
  checkpoint_episode?: number | null;
  global_timestep?: number | null;
  event_type: "model_change" | "param_change" | "session_start" | "promotion";
  title: string;
  summary: string;
  changes: SessionEventParamDiff[];
  rationale?: string | null;
  badge_icon: string;
  badge_color: string;
  session_stats?: SessionEventStats;
  is_current?: boolean;
}

export interface SessionEventsResponse {
  events: SessionChangeEvent[];
  current_session_id?: string | null;
  total_sessions: number;
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
  qualifying_tail_passed?: boolean;
  qualifying_tail_required?: number;
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
  environment_episodes_total?: number;
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

export interface ZoneMetricSummary {
  zone: string;
  total: number;
  wins: number;
  win_rate: number;
  timeouts: number;
  stopped: number;
  trapped_at_end: number;
  avg_steps: number;
  avg_corner_pct: number;
  avg_wall_pct: number;
  is_corner: boolean;
  is_wall: boolean;
}

export interface PenPlacementSummary {
  placement: string;
  total: number;
  wins: number;
  win_rate: number;
  timeouts: number;
  avg_steps: number;
}

export interface SetupSummary {
  setup: string;
  total: number;
  wins: number;
  win_rate: number;
  timeouts: number;
  avg_steps: number;
}

export interface StageBottleneckInsight {
  severity: "critical" | "warning" | "info" | "success";
  type: string;
  title: string;
  message: string;
  metric?: string;
}

export interface StageBottleneckReport {
  curriculum_stage: number;
  total_episodes: number;
  success_count: number;
  timeout_count: number;
  stopped_count: number;
  corner_stuck_count: number;
  overall_success_rate: number;
  avg_steps: number;
  avg_corner_time_pct: number;
  avg_wall_time_pct: number;
  earliest_timestamp: string | null;
  latest_timestamp: string | null;
  zone_stats: Record<string, ZoneMetricSummary>;
  pen_stats: Record<string, PenPlacementSummary>;
  setup_stats: Record<string, SetupSummary>;
  terminal_failure_heatmap: Record<string, number>;
  insights: StageBottleneckInsight[];
  error?: string;
}

export interface EvaluationRecordPayload {
  seed: number;
  success: boolean;
  timeout?: boolean;
  stopped?: boolean;
  steps: number;
  simulated_seconds?: number;
  sheep_penned: number;
  stop_reason: string;
  spawn_mode?: string;
  reward_total: number;
  replay_path?: string;
  policy_version?: number | null;
  final_sheep_distance_to_pen?: number;
  final_flock_spread?: number;
  final_sheep_zone?: string | null;
  pen_zone?: string | null;
  corner_time_pct?: number | null;
  wall_time_pct?: number | null;
  role_switches?: number;
  collector_activations?: number;
  blocker_activations?: number;
  dog_role_occupancy?: Record<string, Record<string, number>>;
  reward_breakdown?: Record<string, number>;
}

export interface EvaluationSummaryPayload {
  checkpoint_episode: number;
  policy_name?: string;
  records: EvaluationRecordPayload[];
  success_rate: number;
  timeout_rate: number;
  average_completion_steps: number;
  average_sheep_penned: number;
  average_reward: number;
  curriculum_stage?: number;
  run_id?: string | null;
  checkpoint_id?: string | null;
  policy_version?: number | null;
  evaluation_timestamp?: string | null;
  evaluation_id?: string | null;
  evaluation_mode?: string;
  pinned?: boolean;
  evaluation_index?: number;
  is_milestone?: boolean;
  is_first?: boolean;
  retention_status?: string;
}

export interface SeedHealthItem {
  seed: number;
  win_rate: number;
  wins: number;
  fails: number;
  total: number;
  status: "green" | "yellow" | "red";
  current_consecutive_fails: number;
}

export interface FailureProgressStats {
  total_failures: number;
  avg_penned_on_fail: number;
  three_penned_pct: number;
  two_penned_pct: number;
  one_penned_pct: number;
  zero_penned_pct: number;
  closeness_score: number;
}

export interface HyperparameterAuditItem {
  parameter: string;
  current_value: unknown;
  recommended_value: unknown;
  status: "ok" | "warn" | "danger";
  note: string;
}

export interface PrescriptiveRecommendation {
  type: string;
  title: string;
  description: string;
  suggested_action: string;
  priority: "high" | "medium" | "info";
}

export interface StageHealthSummary {
  stage: number;
  stage_title: string;
  total_stage_checkpoints: number;
  all_time_stage_success_rate: number;
  recent_success_rate: number;
  peak_stage_success_rate: number;
  recent_avg_steps: number;
  recent_avg_reward: number;
  status: "green" | "yellow" | "red";
  status_label: string;
  status_explanation: string;
  promotion_ready: boolean;
  promotion_status_text: string;
  failure_progress: FailureProgressStats;
  seed_matrix: SeedHealthItem[];
  recent_trajectory: Array<{
    pv: number;
    episode: number;
    success_rate: number;
    steps: number;
    reward: number;
    mode: string;
    timestamp: string;
  }>;
  hyperparameter_audit: HyperparameterAuditItem[];
  prescriptive_recommendations: PrescriptiveRecommendation[];
}

export interface EarlyWarningItem {
  id: string;
  condition_type: "unproductive_repetition" | "success_regression" | string;
  severity: "critical" | "warning" | "info";
  title: string;
  plain_language_observation: string;
  candidate_causes: string[];
  source_layer: "completed_evaluation" | string;
  evidence_source: string;
  evaluation_id?: string | null;
  checkpoint_id?: string | null;
  checkpoint_episode?: number | null;
  policy_version?: number | null;
  curriculum_stage: number;
  completion_time: string;
  sample_count: number;
  metric_version?: string | null;
  status: "active" | "acknowledged";
  supporting_measurements: Record<string, any>;
  compatibility_factors?: Record<string, any> | null;
  comparison_status?: string | null;
  replay_link?: string | null;
  replay_step?: number | null;
  replay_interval_id?: string | null;
  replay_available: boolean;
  link_explanation?: string | null;
  acknowledged_at?: string | null;
  acknowledged_by?: string | null;
  first_detected_at: string;
  last_detected_at: string;
}

export interface EarlyWarningsSummary {
  critical_count: number;
  warning_count: number;
  info_count: number;
  total_active: number;
  total_acknowledged: number;
  total_resolved?: number;
  monitoring_status?: string;
}

export interface CurrentEvaluationContext {
  evaluation_id?: string | null;
  checkpoint_id?: string | null;
  checkpoint_episode?: number | null;
  policy_version?: number | null;
  curriculum_stage: number;
  completion_time?: string | null;
  sample_count: number;
  evidence_source?: string | null;
  repetition_metrics_available: boolean;
  repetition_metric_status: string;
  repetition_metrics?: Record<string, any> | null;
  comparison_status: string;
}

export interface EarlyWarningsPayload {
  enabled: boolean;
  is_read_only_preview: boolean;
  mode_description: string;
  monitored_signals: string[];
  deferred_signals: string[];
  current_evaluation: CurrentEvaluationContext;
  active_warnings: EarlyWarningItem[];
  acknowledged_warnings: EarlyWarningItem[];
  summary: EarlyWarningsSummary;
}

export type TemporalIntervalClassification =
  | "oscillation_loop"
  | "stationary_hold"
  | "productive_repositioning"
  | "counterproductive_pressure"
  | "unresolved_ambiguity"
  | string;

export interface ReplayEvidenceAudit {
  has_initial_step_zero: boolean;
  has_contiguous_frames: boolean;
  has_recorded_debug_targets: boolean;
  has_recorded_step_telemetry: boolean;
  has_recorded_action_probabilities: boolean;
  has_full_environment_config: boolean;
  missing_environment_config_fields: string[];
  guidance_certainty: string;
  reconstruction_limitations: string[];
  discount_factor_available: boolean;
  discount_factor: number | null;
  discount_timebase: string | null;
  strategy_version: string | null;
  trainer_type: string | null;
  checkpoint_gamma: number | null;
  evaluation_gamma: number | null;
  gamma_mismatch: boolean;
  telemetry_truncated: boolean;
  telemetry_limit_steps: number | null;
  telemetry_truncation_step: number | null;
  recorded_fields: string[];
  derived_fields: string[];
  unavailable_fields: string[];
}

export interface IntervalContextWindowSummary {
  window_type: "lead_in" | "aftermath" | string;
  start_step: number;
  end_step: number;
  duration_steps: number;
  unpenned_sheep_start: number;
  unpenned_sheep_end: number;
  sheep_penned_delta: number;
  avg_route_distance_start: number;
  avg_route_distance_end: number;
  net_route_progress: number;
  flock_spread_start: number;
  flock_spread_end: number;
  dog_role_switches: number;
  behind_flock_alignment_start: number;
  behind_flock_alignment_end: number;
  summary: string;
}

export interface DogIntervalSummary {
  dog_index: number;
  dominant_role: string;
  role_distribution: Record<string, number>;
  role_switches: number;
  start_position: [number, number];
  end_position: [number, number];
  net_displacement: number;
  total_path_length: number;
  tortuosity_ratio: number;
  unique_positions_count: number;
  bounding_box_width: number;
  bounding_box_height: number;
  bounding_box_span: number;
  action_reversal_rate: number;
  wait_rate: number;
  sprint_rate: number;
  top_actions: Array<{ action: string; count: number }>;
  mean_distance_to_nearest_sheep: number;
  within_sheep_vision_rate: number;
  on_gate_or_waypoint_side_rate: number;
  behind_flock_alignment_mean: number;
  behind_flock_alignment_delta: number;
  guidance_source: string;
  guidance_certainty: string;
  evidence_limitations: string[];
  start_target: [number, number];
  end_target: [number, number];
  mean_distance_to_target: number;
  legal_action_mask_source: string;
  mask_type: string;
  mean_legal_action_count: number;
  action_probabilities_available: boolean;
  selected_actions_available: boolean;
  action_replacements_count: number;
  value_estimates_available: boolean;
}

export interface SheepIntervalSummary {
  sheep_index: number;
  personality: string;
  penned_at_start: boolean;
  penned_at_end: boolean;
  start_position: [number, number];
  end_position: [number, number];
  start_waypoint: [number, number];
  end_waypoint: [number, number];
  start_route_distance: number;
  end_route_distance: number;
  min_route_distance: number;
  max_route_distance: number;
  net_route_progress: number;
  unique_positions_count: number;
  bounding_box_span: number;
  total_path_length: number;
  position_revisit_rate: number;
  closed_fence_adjacent_rate: number;
  outer_wall_or_corner_rate: number;
  packed_rate: number;
}

export interface RewardCycleRecord {
  cycle_id: string;
  start_step: number;
  end_step: number;
  duration_steps: number;
  unpenned_sheep_count: number;
  recurrence_type: string;
  position_signature_matched: boolean;
  hidden_state_matched: boolean;
  unobserved_hidden_state_fields: string[];
  total_reward: number;
  discounted_return: number | null;
  discount_factor: number | null;
  discount_timebase: string | null;
  discounting_status: string;
  component_totals: Record<string, number>;
  positive_progress_steps: number;
  negative_progress_steps: number;
  net_route_progress: number;
  cycle_classification: string;
  description: string;
}

export interface RewardIntervalSummary {
  total_reward: number;
  mean_step_reward: number;
  component_totals: Record<string, number>;
  positive_progress_reward_steps: number;
  negative_progress_reward_steps: number;
  zero_net_progress_with_positive_reward_steps: boolean;
  cycles: RewardCycleRecord[];
  has_candidate_positive_return_cycle: boolean;
  has_exploitable_positive_cycle: boolean;
  has_net_negative_cycle_with_positive_steps: boolean;
  has_incomplete_cycle: boolean;
}

export interface IntervalPlainLanguageExplanation {
  readable_label: string;
  what_was_observed: string;
  candidate_explanation: string;
  supporting_evidence: string[];
  missing_evidence_and_alternatives: string[];
  controlled_experiment_required: string;
  moving_away_note: string;
}

export interface TemporalInteractionInterval {
  interval_id: string;
  replay_id: string;
  classification: TemporalIntervalClassification;
  readable_label?: string;
  start_step: number;
  end_step: number;
  duration_steps: number;
  unpenned_sheep_count: number;
  affected_dog_indices: number[];
  affected_sheep_indices: number[];
  avg_route_distance_start: number;
  avg_route_distance_end: number;
  net_route_progress: number;
  flock_spread_start: number;
  flock_spread_end: number;
  no_progress_steps_start: number;
  no_progress_steps_end: number;
  no_progress_resets_without_best_route_gain: number;
  collector_blind_spot_steps: number;
  corner_target_inversion_steps: number;
  dogs_outside_sheep_vision_steps: number;
  lead_in: IntervalContextWindowSummary;
  aftermath: IntervalContextWindowSummary;
  dog_summaries: DogIntervalSummary[];
  sheep_summaries: SheepIntervalSummary[];
  reward_summary: RewardIntervalSummary;
  candidate_mechanisms: string[];
  supporting_evidence: string[];
  alternative_explanations: string[];
  unavailable_signals: string[];
  guidance_certainty: string;
  evidence_limitations: string[];
  plain_language?: IntervalPlainLanguageExplanation;
}

export interface EpisodeTemporalAnalysis {
  replay_id: string;
  replay_path: string | null;
  seed: number | null;
  policy_name: string;
  outcome_status: string;
  stop_reason: string;
  total_steps: number;
  total_sheep: number;
  final_penned_count: number;
  final_unpenned_count: number;
  pen_geometry: {
    origin: Point;
    width: number;
    height: number;
    opening: string;
    grid_width: number;
    grid_height: number;
  };
  penning_steps: Array<[number, number]>;
  evidence_audit: ReplayEvidenceAudit;
  intervals: TemporalInteractionInterval[];
  unpenned_count_coverage: number[];
  summary_notes: string[];
}

export interface LabeledObservationField {
  index: number;
  key: string;
  label: string;
  group: string;
  value: number;
}

export interface LabeledObservationPayload {
  available: boolean;
  schema_status: "verified" | "unverified_schema" | "truncated" | "not_recorded" | string;
  schema_name: string | null;
  vector_length: number;
  reason: string;
  fields: LabeledObservationField[];
}

export interface StepDogDecisionEvidence {
  dog_index: number;
  world_step: number;
  observation: number[] | null;
  observation_labeled?: LabeledObservationPayload;
  decision_action_mask: Record<string, boolean>;
  recomputed_physical_mask: Record<string, boolean>;
  selected_action: string;
  executed_action: string;
  replacement_reason: string | null;
  action_probabilities: Record<string, number> | null;
  logits: Record<string, number> | null;
  value_estimate: number | null;
  decision_mode: string;
  policy_version: number | null;
  checkpoint_id: string | null;
  mask_type: string;
  action_probabilities_source: string;
  value_estimate_source: string;
  capture_error?: string | null;
}

export interface StepDogEvidence {
  dog_index: number;
  pre_step_position: Point | null;
  post_step_position: Point;
  displacement: { dx: number; dy: number; distance: number } | null;
  recorded_role: string;
  assigned_role: string;
  target: Point;
  target_sheep_index: number | null;
  target_reason: string;
  guidance_source: string;
  guidance_certainty: string;
  distance_to_target: number;
  nearest_unpenned_sheep_distance: number | null;
  within_sheep_vision: boolean;
  behind_flock_alignment: number;
  on_gate_or_waypoint_side: boolean;
  decision: StepDogDecisionEvidence | null;
}

export interface NearbyDogGeometry {
  dog_index: number;
  role: string;
  distance: number;
  within_sheep_vision: boolean;
  dx_from_sheep: number;
  dy_from_sheep: number;
  alignment_to_waypoint: number;
  on_waypoint_side: boolean;
}

export interface StepSheepEvidence {
  sheep_index: number;
  pre_step_position: Point | null;
  post_step_position: Point;
  displacement: { dx: number; dy: number; distance: number } | null;
  penned: boolean;
  penned_at_pre_step: boolean | null;
  personality: string;
  is_packed: boolean;
  route_distance: number;
  waypoint: Point;
  closed_fence_adjacent: boolean;
  outer_wall_or_corner: boolean;
  wall_margin: number;
  nearby_dogs: NearbyDogGeometry[];
}

export interface TemporalStepEvidence {
  step: number;
  has_prior_contiguous_step: boolean;
  actions: string[];
  unpenned_count: number;
  penned_count: number;
  average_distance_to_pen: number;
  best_route_distance_so_far: number | null;
  flock_spread: number;
  no_progress_steps: number;
  no_progress_reset_without_best_gain: boolean;
  collector_blind_spot: boolean;
  corner_target_inverted: boolean;
  any_dog_outside_sheep_vision_oscillating: boolean;
  guidance_source: string;
  guidance_certainty: string;
  legal_action_mask_source: string;
  mask_type: string;
  has_recorded_step_telemetry: boolean;
  has_recorded_debug_targets: boolean;
  telemetry_truncated: boolean;
  reward: Record<string, number>;
  strategy: {
    flock_center: Point | null;
    stray_sheep_index: number | null;
    collector_active: boolean;
  };
  dogs: StepDogEvidence[];
  sheep: StepSheepEvidence[];
}

export interface TemporalDiagnosticPayload {
  replay_id: string;
  sheep_vision: number;
  action_order: string[];
  available_reward_components: string[];
  analysis: EpisodeTemporalAnalysis;
  step_evidence: TemporalStepEvidence[];
  replay?: ReplayBundle;
}

export interface TemporalDiagnosticExample {
  replay_id: string;
  title: string;
  category: string;
  seed: number | null;
  outcome_status: string;
  stop_reason: string;
  penned_summary: string;
  unpenned_counts: number[];
  has_step_zero: boolean;
  paired_replay_id: string | null;
  what_to_look_for: string;
  recommended_default: boolean;
}



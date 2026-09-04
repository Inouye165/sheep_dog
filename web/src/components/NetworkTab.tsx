import { useMemo } from "react";
import type { CheckpointIndex, TrainingStatus } from "../state/types";
import { NetworkTopologyViewer } from "./NetworkTopologyViewer";

interface NetworkTabProps {
  checkpointIndex: CheckpointIndex | null;
  trainingStatus: TrainingStatus | null;
  effectiveConfig: Record<string, unknown> | null;
  topologyInfo: {
    observation_mode: string;
    hidden_layer_sizes: number[];
    observation_size: number;
    action_size: number;
    action_masking_enabled: boolean;
  } | null;
}

type TrainingConfigView = {
  hiddenSizes: number[];
  invalidActionMasking: boolean | null;
  observationMode: string | null;
};

const ACTION_LABELS = [
  "up",
  "down",
  "left",
  "right",
  "sprint_up",
  "sprint_down",
  "sprint_left",
  "sprint_right",
  "wait",
] as const;

function asRecord(value: unknown): Record<string, unknown> | null {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null;
}

function asBoolean(value: unknown): boolean | null {
  return typeof value === "boolean" ? value : null;
}

function asString(value: unknown): string | null {
  return typeof value === "string" ? value : null;
}

function asNumberArray(value: unknown): number[] {
  if (!Array.isArray(value)) {
    return [];
  }
  return value
    .map((item) => (typeof item === "number" && Number.isFinite(item) ? item : null))
    .filter((item): item is number => item !== null);
}

function parseTrainingConfig(effectiveConfig: Record<string, unknown> | null): TrainingConfigView {
  const root = asRecord(effectiveConfig);
  const training = asRecord(root?.training);
  return {
    hiddenSizes: asNumberArray(training?.neural_hidden_sizes),
    invalidActionMasking: asBoolean(training?.invalid_action_masking),
    observationMode: asString(training?.observation_mode),
  };
}

function trainerLabel(value: string | null | undefined): string {
  if (value === "maskable_ppo") return "MaskablePPO";
  if (value === "hierarchical_maskable_ppo") return "Hierarchical MaskablePPO";
  if (value === "hill_climb") return "Hill-climb baseline";
  return value ?? "unknown";
}

function policyLabel(value: string | null | undefined): string {
  if (value === "neural") return "Neural policy";
  if (value === "linear") return "Linear policy";
  if (value === "instinct") return "Instinct policy";
  return value ?? "unknown";
}

export function NetworkTab({ checkpointIndex, trainingStatus, effectiveConfig, topologyInfo }: NetworkTabProps) {
  const trainingConfig = useMemo(() => parseTrainingConfig(effectiveConfig), [effectiveConfig]);
  const latestCheckpoint = checkpointIndex?.checkpoints[checkpointIndex.checkpoints.length - 1] ?? null;
  const trainerType = trainingStatus?.trainer_type ?? latestCheckpoint?.trainer_type ?? null;
  const policyType = trainingStatus?.policy_type ?? latestCheckpoint?.policy_type ?? null;
  const hiddenSizes = topologyInfo?.hidden_layer_sizes?.length
    ? topologyInfo.hidden_layer_sizes
    : trainingConfig.hiddenSizes.length > 0
      ? trainingConfig.hiddenSizes
      : [128, 128];
  const mode = topologyInfo?.observation_mode ?? trainingConfig.observationMode ?? "guided";
  const observationSize = topologyInfo?.observation_size ?? 54;
  const actionSize = topologyInfo?.action_size ?? ACTION_LABELS.length;
  const maskEnabled = topologyInfo?.action_masking_enabled ?? trainingConfig.invalidActionMasking ?? true;

  return (
    <section className="network-tab network-tab--scrollable" aria-label="Neural network architecture">
      <div className="network-tab__header">
        <div>
          <p className="eyebrow">Model Card</p>
          <h2>Neural network architecture</h2>
          <p className="network-tab__intro">
            Source of truth: runtime config + trainer code paths in NeuralPolicy, ShepherdNeuralDogPolicy,
            SheepdogRLAdapter, and JointActionRLEnv.
          </p>
        </div>
      </div>

      <div className="network-tab__kpis">
        <div>
          <span>Trainer</span>
          <strong>{trainerLabel(trainerType)}</strong>
        </div>
        <div>
          <span>Policy class</span>
          <strong>MlpPolicy</strong>
        </div>
        <div>
          <span>Policy type</span>
          <strong>{policyLabel(policyType)}</strong>
        </div>
        <div>
          <span>Action space</span>
          <strong>Discrete({actionSize})</strong>
        </div>
        <div>
          <span>Observation size</span>
          <strong>{observationSize}</strong>
        </div>
        <div>
          <span>Shared layers</span>
          <strong>[{hiddenSizes.join(", ")}]</strong>
        </div>
        <div>
          <span>Observation mode</span>
          <strong>{mode}</strong>
        </div>
      </div>

      <NetworkTopologyViewer
        config={{
          inputSize: observationSize,
          hiddenSizes,
          actionSize,
          maskEnabled,
        }}
        observationMode={mode}
        maskEnabled={maskEnabled}
      />

      <div className="network-tab__grid">
        <section className="network-tab__card">
          <h3>Action masking path</h3>
          <p>
            Legal actions are computed per dog from <strong>action_mask_for_dog()</strong> and passed to
            MaskablePPO prediction/training calls. Illegal actions are removed before selection.
          </p>
          <div className="network-tab__chips">
            {ACTION_LABELS.map((label) => (
              <span key={label} className="pill pill--muted">{label}</span>
            ))}
          </div>
          <dl className="network-tab__specs">
            <dt>invalid_action_masking</dt>
            <dd>{trainingConfig.invalidActionMasking == null ? "unknown" : String(trainingConfig.invalidActionMasking)}</dd>
            <dt>Mask source</dt>
            <dd>environment.action_mask_for_dog</dd>
          </dl>
        </section>


        <section className="network-tab__card">
          <h3>Implementation notes</h3>
          <ul className="network-tab__list">
            <li>Model family: sb3_contrib MaskablePPO with MlpPolicy.</li>
            <li>net_arch is configured by training.neural_hidden_sizes.</li>
            <li>Actor and critic heads branch from the shared backbone.</li>
            <li>Activation function is not overridden in project code; library policy defaults apply.</li>
            <li>Hierarchical trainer extends observations with shepherd command + dog identity features.</li>
          </ul>
        </section>
      </div>
    </section>
  );
}

import type { CheckpointIndex, TrainingStatus } from "../state/types";

interface NeuralNetworkPanelProps {
  checkpointIndex: CheckpointIndex | null;
  trainingStatus: TrainingStatus | null;
  effectiveCurriculumStage: number;
}

function formatPercent(value: number | null): string {
  if (value == null) {
    return "--";
  }
  return `${Math.round(value * 100)}%`;
}

export function NeuralNetworkPanel({
  checkpointIndex,
  trainingStatus,
  effectiveCurriculumStage,
}: NeuralNetworkPanelProps) {
  const latestCheckpoint = checkpointIndex?.checkpoints[checkpointIndex.checkpoints.length - 1] ?? null;
  const latestSuccess =
    trainingStatus?.latest_success_rate ??
    latestCheckpoint?.success_rate ??
    null;
  const hiddenWidth = effectiveCurriculumStage >= 12 ? 96 : effectiveCurriculumStage >= 6 ? 80 : 64;

  return (
    <section className="network-card" aria-label="Neural policy architecture">
      <div className="network-card__header">
        <div>
          <p className="eyebrow">Policy Design</p>
          <h2>Neural network</h2>
        </div>
        <span className="pill pill--muted">PPO</span>
      </div>

      <div className="network-figure" role="img" aria-label="Observation encoder with policy and value heads">
        <svg viewBox="0 0 920 210" className="network-figure__svg">
          <defs>
            <linearGradient id="networkStroke" x1="0" x2="1" y1="0" y2="0">
              <stop offset="0%" stopColor="rgba(125, 211, 252, 0.8)" />
              <stop offset="100%" stopColor="rgba(244, 197, 66, 0.7)" />
            </linearGradient>
          </defs>
          <rect x="40" y="62" width="170" height="82" rx="14" className="network-node network-node--input" />
          <text x="125" y="96" textAnchor="middle" className="network-label">Observations</text>
          <text x="125" y="118" textAnchor="middle" className="network-sublabel">state, flock, pen</text>

          <rect x="290" y="32" width="190" height="62" rx="14" className="network-node network-node--hidden" />
          <text x="385" y="64" textAnchor="middle" className="network-label">Encoder</text>
          <text x="385" y="84" textAnchor="middle" className="network-sublabel">Dense {hiddenWidth}</text>

          <rect x="290" y="116" width="190" height="62" rx="14" className="network-node network-node--hidden" />
          <text x="385" y="148" textAnchor="middle" className="network-label">Temporal context</text>
          <text x="385" y="168" textAnchor="middle" className="network-sublabel">Dense {Math.max(48, hiddenWidth - 16)}</text>

          <rect x="560" y="32" width="150" height="62" rx="14" className="network-node network-node--head" />
          <text x="635" y="64" textAnchor="middle" className="network-label">Policy head</text>
          <text x="635" y="84" textAnchor="middle" className="network-sublabel">dog action logits</text>

          <rect x="560" y="116" width="150" height="62" rx="14" className="network-node network-node--head" />
          <text x="635" y="148" textAnchor="middle" className="network-label">Value head</text>
          <text x="635" y="168" textAnchor="middle" className="network-sublabel">state value</text>

          <rect x="780" y="62" width="112" height="82" rx="14" className="network-node network-node--output" />
          <text x="836" y="96" textAnchor="middle" className="network-label">Dog team</text>
          <text x="836" y="118" textAnchor="middle" className="network-sublabel">coordinated drive</text>

          <path className="network-link" d="M 210 103 L 290 63" />
          <path className="network-link" d="M 210 103 L 290 147" />
          <path className="network-link" d="M 480 63 L 560 63" />
          <path className="network-link" d="M 480 147 L 560 147" />
          <path className="network-link" d="M 710 63 L 780 103" />
          <path className="network-link" d="M 710 147 L 780 103" />
        </svg>
      </div>

      <div className="network-metrics">
        <div>
          <span>Curriculum stage</span>
          <strong>{effectiveCurriculumStage}</strong>
        </div>
        <div>
          <span>Latest success</span>
          <strong>{formatPercent(latestSuccess)}</strong>
        </div>
        <div>
          <span>Total episodes</span>
          <strong>{(trainingStatus?.total_episodes_trained ?? 0).toLocaleString()}</strong>
        </div>
      </div>
    </section>
  );
}

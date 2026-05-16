import { useCallback, useEffect, useState } from "react";
import { loadConfigHistory, loadEffectiveConfig, saveConfigRevision } from "../lib/api";
import type { ConfigHistory, ConfigRevision } from "../state/types";

// ── helpers ──────────────────────────────────────────────────────────────────

function formatTimestamp(iso: string): string {
  try {
    return new Date(iso).toLocaleString(undefined, {
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  } catch {
    return iso;
  }
}

// ── sub-components ───────────────────────────────────────────────────────────

interface ParamRowProps {
  label: string;
  value: unknown;
  note?: string;
}

function ParamRow({ label, value, note }: ParamRowProps) {
  const display =
    value === null || value === undefined
      ? "—"
      : typeof value === "boolean"
        ? value ? "ON" : "OFF"
        : String(value);
  return (
    <tr className="config-param-row">
      <td className="config-param-row__label">{label}</td>
      <td className="config-param-row__value">
        <code>{display}</code>
        {note ? <span className="config-param-row__note">{note}</span> : null}
      </td>
    </tr>
  );
}

interface SectionProps {
  title: string;
  children: React.ReactNode;
}

function Section({ title, children }: SectionProps) {
  return (
    <div className="config-section">
      <h3 className="config-section__title">{title}</h3>
      <table className="config-param-table">
        <tbody>{children}</tbody>
      </table>
    </div>
  );
}

// ── revision card ─────────────────────────────────────────────────────────────

interface RevisionCardProps {
  revision: ConfigRevision;
  isExpanded: boolean;
  onToggle: () => void;
}

function RevisionCard({ revision, isExpanded, onToggle }: RevisionCardProps) {
  const ts = revision.training_settings;
  return (
    <div className={`config-revision${isExpanded ? " config-revision--expanded" : ""}`}>
      <button className="config-revision__header" onClick={onToggle} aria-expanded={isExpanded}>
        <span className="config-revision__index">#{revision.id}</span>
        <span className="config-revision__label">{revision.label}</span>
        <span className="config-revision__meta">
          {formatTimestamp(revision.timestamp)}
          <span className={`pill ${revision.source === "training_start" ? "pill--live" : "pill--muted"}`}>
            {revision.source === "training_start" ? "auto" : "manual"}
          </span>
        </span>
      </button>
      {isExpanded ? (
        <div className="config-revision__body">
          <div className="config-revision__settings">
            <span><strong>Episodes:</strong> {ts.episodes}</span>
            <span><strong>Mode:</strong> {ts.fast_mode ? "fast (2k steps)" : "full (25k steps)"}</span>
            <span><strong>Stage:</strong> {ts.curriculum_stage}</span>
            <span><strong>Instincts:</strong> {ts.enable_instinct_rewards ? "ON" : "OFF"}</span>
          </div>
          <details className="config-revision__raw-details">
            <summary>Full config JSON</summary>
            <pre className="config-revision__raw">{JSON.stringify(revision.config, null, 2)}</pre>
          </details>
        </div>
      ) : null}
    </div>
  );
}

// ── main component ────────────────────────────────────────────────────────────

export function ConfigPanel() {
  const [config, setConfig] = useState<Record<string, unknown> | null>(null);
  const [history, setHistory] = useState<ConfigHistory | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [expandedId, setExpandedId] = useState<number | null>(null);
  const [saving, setSaving] = useState(false);

  const refresh = useCallback(async () => {
    try {
      const [cfg, hist] = await Promise.all([loadEffectiveConfig(), loadConfigHistory()]);
      setConfig(cfg);
      setHistory(hist);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load config");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { void refresh(); }, [refresh]);

  const handleSaveSnapshot = useCallback(async () => {
    if (!config) return;
    setSaving(true);
    try {
      const label = `Manual snapshot · ${new Date().toLocaleString()}`;
      const updated = await saveConfigRevision({
        label,
        source: "manual",
        training_settings: {
          episodes: 0,
          fast_mode: false,
          enable_instinct_rewards: false,
          curriculum_stage: 0,
          debug_reward_breakdown: false,
        },
        config,
      });
      setHistory(updated);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to save snapshot");
    } finally {
      setSaving(false);
    }
  }, [config]);

  if (loading) {
    return (
      <section className="training-card" aria-label="Config">
        <div className="training-card__header">
          <div><p className="eyebrow">Config</p><h2>Parameters</h2></div>
        </div>
        <div className="warning-box" role="status">Loading config…</div>
      </section>
    );
  }

  // ── extract nested config values ──────────────────────────────────────────
  const env = config?.environment as Record<string, unknown> | undefined;
  const rewards = config?.rewards as Record<string, unknown> | undefined;
  const instincts = rewards?.instincts as Record<string, unknown> | undefined;
  const training = config?.training as Record<string, unknown> | undefined;
  const policy = config?.policy as Record<string, unknown> | undefined;

  const revisions = history?.revisions ? [...history.revisions].reverse() : [];

  return (
    <section className="training-card config-panel" aria-label="Config">
      <div className="training-card__header">
        <div><p className="eyebrow">Config</p><h2>Parameters &amp; History</h2></div>
        <button
          className="btn-secondary"
          onClick={() => void handleSaveSnapshot()}
          disabled={saving || !config}
          title="Save a manual snapshot of the current config to revision history"
        >
          {saving ? "Saving…" : "Save snapshot"}
        </button>
      </div>

      {error ? <div className="warning-box warning-box--error" role="alert">{error}</div> : null}

      {config ? (
        <div className="config-sections">
          <Section title="Environment">
            <ParamRow label="Grid size" value={env ? `${String(env.width)} × ${String(env.height)}` : "—"} />
            <ParamRow label="Dogs" value={env?.dogs} />
            <ParamRow label="Sheep" value={env?.sheep} />
            <ParamRow label="Max steps" value={env?.max_steps} />
            <ParamRow label="Dog speed" value={env?.dog_speed} />
            <ParamRow label="Sheep speed" value={env?.sheep_speed} />
          </Section>

          <Section title="Rewards">
            <ParamRow label="time_penalty" value={rewards?.time_penalty} note="per-step penalty; higher = faster pressure to complete" />
            <ParamRow label="progress_scale" value={rewards?.progress_scale} />
            <ParamRow label="success_reward" value={rewards?.success_reward} />
            <ParamRow label="failure_penalty" value={rewards?.failure_penalty} />
            <ParamRow label="cohesion_weight" value={rewards?.cohesion_weight} />
            <ParamRow label="scatter_penalty_weight" value={rewards?.scatter_penalty_weight} />
            <ParamRow label="no_progress_penalty" value={rewards?.no_progress_penalty} />
          </Section>

          <Section title="Instinct Rewards">
            <ParamRow label="enable_instinct_rewards" value={instincts?.enable_instinct_rewards} />
            <ParamRow label="curriculum_stage" value={instincts?.curriculum_stage} />
            <ParamRow label="safe_pressure_weight" value={instincts?.safe_pressure_weight} note="reward for staying 2–6 cells from flock" />
            <ParamRow label="overpressure_penalty_weight" value={instincts?.overpressure_penalty_weight} />
            <ParamRow label="engagement_weight" value={instincts?.engagement_weight} />
            <ParamRow label="between_flock_and_pen_weight" value={instincts?.between_flock_and_pen_weight} />
          </Section>

          <Section title="Training Hyperparameters">
            <ParamRow label="trainer_type" value={training?.trainer_type} />
            <ParamRow label="total_timesteps" value={training?.total_timesteps} />
            <ParamRow label="rollout_steps" value={training?.rollout_steps} />
            <ParamRow label="batch_size" value={training?.batch_size} />
            <ParamRow label="learning_rate" value={training?.learning_rate} />
            <ParamRow label="n_epochs" value={training?.n_epochs} />
            <ParamRow label="entropy_coef" value={training?.entropy_coef} note="higher = more exploration; too high causes instability after success" />
            <ParamRow label="clip_range" value={training?.clip_range} />
            <ParamRow label="evaluation_seeds" value={Array.isArray(training?.evaluation_seeds) ? String((training?.evaluation_seeds as unknown[]).length) + " seeds" : "—"} />
            <ParamRow label="fast_mode steps/ep" value="2 000 (fast) / 25 000 (full)" note="set in UI before starting training" />
          </Section>

          <Section title="Policy">
            <ParamRow label="policy_mode" value={policy?.policy_mode} />
            <ParamRow label="policy_type" value={training?.policy_type} />
            <ParamRow label="allow_instinct_target_awareness" value={policy?.allow_instinct_target_awareness} />
            <ParamRow label="handler_target_enabled" value={policy?.handler_target_enabled} />
          </Section>
        </div>
      ) : null}

      {/* Revision history */}
      <div className="config-history">
        <h3 className="config-section__title">
          Revision History
          <span className="pill pill--muted" style={{ marginLeft: "0.5rem" }}>{revisions.length}</span>
        </h3>
        {revisions.length === 0 ? (
          <p className="config-history__empty">No revisions yet — start a training run to record the first one.</p>
        ) : (
          <div className="config-history__list">
            {revisions.map((rev) => (
              <RevisionCard
                key={rev.id}
                revision={rev}
                isExpanded={expandedId === rev.id}
                onToggle={() => setExpandedId(expandedId === rev.id ? null : rev.id)}
              />
            ))}
          </div>
        )}
      </div>
    </section>
  );
}

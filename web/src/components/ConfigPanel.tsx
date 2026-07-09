import { useCallback, useEffect, useRef, useState } from "react";
import { loadConfigHistory, loadEffectiveConfig, loadHyperparams, saveConfigRevision, saveHyperparams } from "../lib/api";
import type { ConfigHistory, ConfigRevision, UserHyperparams } from "../state/types";

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

interface OptionListRowProps {
  label: string;
  items: Array<{ name: string; description: string }>;
}

function OptionListRow({ label, items }: OptionListRowProps) {
  return (
    <tr className="config-param-row">
      <td className="config-param-row__label">{label}</td>
      <td className="config-param-row__value">
        <ul style={{ margin: 0, paddingLeft: "1.1rem" }}>
          {items.map((item) => (
            <li key={item.name}>
              <code>{item.name}</code>
              <span className="config-param-row__note"> — {item.description}</span>
            </li>
          ))}
        </ul>
      </td>
    </tr>
  );
}

// ── HyperparamsEditor ─────────────────────────────────────────────────────────

const HYPERPARAM_DEFAULTS: UserHyperparams = {
  environment: {
    sheep_personality_strength: 0.0,
    sheep_speed: 0.75,
    sheep_vision: 12,
    flock_radius: 10,
    dog_speed: 1.0,
    dog_sprint_multiplier: 2.0,
    dog_vision: 16,
  },
  training: {
    learning_rate: 0.0001,
    learning_rate_final: 0.00003,
    entropy_coef: 0.01,
    gamma: 0.99,
    gae_lambda: 0.95,
    clip_range: 0.2,
    rollout_steps: 2048,
    batch_size: 64,
    value_coef: 0.5,
  },
  rewards: {
    time_penalty: 0.05,
    progress_scale: 2.0,
    sheep_penned_reward: 8.0,
    wait_penalty: 0.05,
    no_progress_penalty: 0.1,
    terminal_success_reward: 20.0,
    terminal_failure_penalty: 12.0,
    flock_cohesion_scale: 0.35,
    scatter_penalty_scale: 0.2,
    sprint_cost_scale: 0.12,
  },
};

interface HParamFieldDef {
  key: string;
  label: string;
  note: string;
  min: number;
  max: number;
  step: number;
}

const ENV_FIELDS: HParamFieldDef[] = [
  { key: "sheep_personality_strength", label: "Personality strength", note: "0 = all obedient; 0.1 mild; 0.5+ pronounced", min: 0, max: 3, step: 0.05 },
  { key: "sheep_speed", label: "Sheep speed", note: "cells per step", min: 0.1, max: 3, step: 0.05 },
  { key: "sheep_vision", label: "Sheep vision", note: "radius (cells) in which sheep react to dogs", min: 3, max: 30, step: 1 },
  { key: "flock_radius", label: "Flock radius", note: "cohesion goal distance (cells)", min: 2, max: 25, step: 1 },
  { key: "dog_speed", label: "Dog speed", note: "cells per step", min: 0.5, max: 4, step: 0.1 },
  { key: "dog_sprint_multiplier", label: "Sprint multiplier", note: "speed boost for sprint actions", min: 1, max: 5, step: 0.1 },
  { key: "dog_vision", label: "Dog vision", note: "radius (cells)", min: 5, max: 40, step: 1 },
];

const TRAINING_FIELDS: HParamFieldDef[] = [
  { key: "learning_rate", label: "Learning rate", note: "initial PPO LR (e.g. 1e-4)", min: 1e-6, max: 0.01, step: 1e-5 },
  { key: "learning_rate_final", label: "LR final", note: "annealed final value", min: 1e-6, max: 0.005, step: 1e-5 },
  { key: "entropy_coef", label: "Entropy coef", note: "exploration bonus; >0.05 risks instability", min: 0, max: 0.2, step: 0.001 },
  { key: "gamma", label: "Gamma (γ)", note: "discount factor for future rewards", min: 0.9, max: 1, step: 0.001 },
  { key: "gae_lambda", label: "GAE lambda (λ)", note: "advantage estimation smoothing", min: 0.8, max: 1, step: 0.005 },
  { key: "clip_range", label: "Clip range", note: "PPO clip ε", min: 0.05, max: 0.5, step: 0.01 },
  { key: "rollout_steps", label: "Rollout steps", note: "steps per update buffer", min: 512, max: 8192, step: 512 },
  { key: "batch_size", label: "Batch size", note: "mini-batch size for gradient updates", min: 32, max: 2048, step: 32 },
  { key: "value_coef", label: "Value coef", note: "weight of the value-function loss", min: 0.1, max: 1, step: 0.05 },
];

const REWARD_FIELDS: HParamFieldDef[] = [
  { key: "time_penalty", label: "Time penalty", note: "per-step penalty; keep ≤0.05 to avoid PPO instability", min: 0, max: 0.5, step: 0.005 },
  { key: "progress_scale", label: "Progress scale", note: "reward per step of flock movement toward pen", min: 0, max: 10, step: 0.1 },
  { key: "sheep_penned_reward", label: "Penned reward", note: "bonus per sheep entering pen", min: 0, max: 50, step: 0.5 },
  { key: "wait_penalty", label: "Wait penalty", note: "cost for the wait action", min: 0, max: 0.5, step: 0.005 },
  { key: "no_progress_penalty", label: "No-progress penalty", note: "applied when no flock movement detected", min: 0, max: 0.5, step: 0.005 },
  { key: "terminal_success_reward", label: "Success reward", note: "terminal bonus for all sheep penned", min: 0, max: 100, step: 1 },
  { key: "terminal_failure_penalty", label: "Failure penalty", note: "terminal penalty on timeout", min: 0, max: 100, step: 1 },
  { key: "flock_cohesion_scale", label: "Cohesion scale", note: "reward for keeping flock together", min: 0, max: 2, step: 0.05 },
  { key: "scatter_penalty_scale", label: "Scatter penalty", note: "penalty when flock spreads", min: 0, max: 2, step: 0.05 },
  { key: "sprint_cost_scale", label: "Sprint cost", note: "cost multiplier for using sprint", min: 0, max: 1, step: 0.01 },
];

interface HyperparamsEditorProps {
  hyperparams: UserHyperparams;
  onSaved: (updated: UserHyperparams) => void;
}

function HyperparamsEditor({ hyperparams, onSaved }: HyperparamsEditorProps) {
  type SectionKey = keyof UserHyperparams;
  type Draft = { environment: Record<string, number>; training: Record<string, number>; rewards: Record<string, number> };
  const toDraft = (hp: UserHyperparams): Draft => ({
    environment: { ...hp.environment as unknown as Record<string, number> },
    training: { ...hp.training as unknown as Record<string, number> },
    rewards: { ...hp.rewards as unknown as Record<string, number> },
  });

  const [draft, setDraft] = useState<Draft>(() => toDraft(hyperparams));
  const [saving, setSaving] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);
  const [savedOk, setSavedOk] = useState(false);
  const savedTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const autoSaveTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  // True only when the user has typed a change that hasn't been saved yet.
  const userEditedRef = useRef(false);

  // Sync draft when hyperparams prop changes (e.g. after external load).
  // Reset the dirty flag so the sync itself doesn't trigger an auto-save.
  useEffect(() => {
    setDraft(toDraft(hyperparams));
    userEditedRef.current = false;
  }, [hyperparams]);

  const handleChange = (section: SectionKey, key: string, raw: string) => {
    const num = parseFloat(raw);
    if (!Number.isNaN(num)) {
      userEditedRef.current = true;
      setDraft(prev => ({ ...prev, [section]: { ...prev[section], [key]: num } }));
    }
  };

  const handleReset = () => {
    userEditedRef.current = true;
    setDraft(toDraft(HYPERPARAM_DEFAULTS));
  };

  const handleSave = async () => {
    userEditedRef.current = false;
    setSaving(true);
    setSaveError(null);
    setSavedOk(false);
    try {
      const payload = draft as unknown as UserHyperparams;
      const updated = await saveHyperparams(payload);
      onSaved(updated);
      setSavedOk(true);
      if (savedTimerRef.current) clearTimeout(savedTimerRef.current);
      savedTimerRef.current = setTimeout(() => setSavedOk(false), 3000);
    } catch (err) {
      setSaveError(err instanceof Error ? err.message : "Save failed");
    } finally {
      setSaving(false);
    }
  };

  // Auto-save 800 ms after the last user-initiated change.
  useEffect(() => {
    if (!userEditedRef.current) return;
    if (autoSaveTimerRef.current) clearTimeout(autoSaveTimerRef.current);
    autoSaveTimerRef.current = setTimeout(() => void handleSave(), 800);
    return () => {
      if (autoSaveTimerRef.current) clearTimeout(autoSaveTimerRef.current);
    };
  // handleSave is stable within this render; draft dep ensures we re-schedule on each keystroke.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [draft]);

  const isModified = (section: SectionKey, key: string): boolean => {
    const def = (HYPERPARAM_DEFAULTS[section] as unknown as Record<string, number>)[key];
    return draft[section][key] !== def;
  };

  const renderField = (section: SectionKey, field: HParamFieldDef) => {
    const modified = isModified(section, field.key);
    return (
      <tr key={field.key} className="hyperparam-row">
        <td className="hyperparam-row__label">
          {field.label}
          {modified ? <span className="hyperparam-badge--modified" title="Modified from default">●</span> : null}
        </td>
        <td className="hyperparam-row__input">
          <input
            type="number"
            className="hyperparam-input"
            value={draft[section][field.key] ?? ""}
            min={field.min}
            max={field.max}
            step={field.step}
            onChange={e => handleChange(section, field.key, e.target.value)}
          />
          <span className="hyperparam-row__note">{field.note}</span>
        </td>
      </tr>
    );
  };

  return (
    <div className="hyperparam-editor">
      <div className="hyperparam-editor__actions">
        <button
          className="btn-secondary"
          onClick={() => void handleSave()}
          disabled={saving}
          title="Persist hyperparameters — survives clear and server restart"
        >
          {saving ? "Saving…" : savedOk ? "Saved ✓" : "Save hyperparams"}
        </button>
        <button
          className="btn-secondary btn-secondary--ghost"
          onClick={handleReset}
          disabled={saving}
          title="Restore all values to their original defaults"
        >
          Reset to defaults
        </button>
        {saveError ? <span className="hyperparam-editor__error">{saveError}</span> : null}
      </div>
      <p className="hyperparam-editor__note">
        Changes auto-save after you stop typing. Values persist across
        <strong> clear&nbsp;&amp;&nbsp;retrain</strong> and server restarts.
        Takes effect on the next training run or live replay.
      </p>

      <div className="hyperparam-editor__sections">
        <div className="hyperparam-section">
          <h4 className="hyperparam-section__title">🐑 Sheep Behavior</h4>
          <table className="hyperparam-table"><tbody>
            {ENV_FIELDS.map(f => renderField("environment", f))}
          </tbody></table>
        </div>
        <div className="hyperparam-section">
          <h4 className="hyperparam-section__title">🏆 Rewards</h4>
          <table className="hyperparam-table"><tbody>
            {REWARD_FIELDS.map(f => renderField("rewards", f))}
          </tbody></table>
        </div>
        <div className="hyperparam-section">
          <h4 className="hyperparam-section__title">🧠 Training (PPO)</h4>
          <table className="hyperparam-table"><tbody>
            {TRAINING_FIELDS.map(f => renderField("training", f))}
          </tbody></table>
        </div>
      </div>
    </div>
  );
}

// ── catalogs ──────────────────────────────────────────────────────────────────
// Mirrors backend definitions in src/sheepdog/entities.py and ACTION_DELTAS in
// src/sheepdog/environment.py. Kept in sync manually; if you add a new
// personality or dog action server-side, update the list here too.

const DOG_ACTIONS: Array<{ name: string; description: string }> = [
  { name: "up / down / left / right", description: "single-cell move in cardinal direction" },
  { name: "sprint_up / sprint_down / sprint_left / sprint_right", description: "faster move (costs more), gated by dog_sprint_multiplier" },
  { name: "wait", description: "hold position; used to apply patient pressure" },
  { name: "bark", description: "no movement, but applies extra panic on nearby sheep" },
];

const DOG_ROLES: Array<{ name: string; description: string }> = [
  { name: "rear_pressure", description: "drive the flock from behind toward the pen" },
  { name: "left_flanker", description: "shape the flock from its left side" },
  { name: "right_flanker", description: "shape the flock from its right side" },
  { name: "collector", description: "gather scattered or stragglers back to the group" },
  { name: "blocker", description: "stand on the far side of the pen opening to prevent escape" },
];

const SHEEP_PERSONALITIES: Array<{ name: string; description: string }> = [
  { name: "obedient", description: "neutral baseline; no extra bias" },
  { name: "pen_fearful", description: "proximity-scaled push away from the pen; strongest near the entrance, fades with distance" },
  { name: "pen_shy", description: "mild push away from the pen center" },
  { name: "escapist", description: "when panicked, breaks away from the flock instead of cohering" },
  { name: "bold", description: "ignores distant dogs more; the dog must close the gap (or bark) to apply real pressure" },
];

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
  const [hyperparams, setHyperparams] = useState<UserHyperparams | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [expandedId, setExpandedId] = useState<number | null>(null);
  const [saving, setSaving] = useState(false);

  const refresh = useCallback(async () => {
    try {
      const [cfg, hist, hp] = await Promise.all([
        loadEffectiveConfig(),
        loadConfigHistory(),
        loadHyperparams(),
      ]);
      setConfig(cfg);
      setHistory(hist);
      setHyperparams(hp);
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

      {/* Editable hyperparameter form */}
      {hyperparams ? (
        <details className="hyperparam-editor-details" open>
          <summary className="hyperparam-editor-summary">
            <span className="config-section__title" style={{ display: "inline" }}>Edit Hyperparameters</span>
            <span className="pill pill--muted" style={{ marginLeft: "0.5rem", verticalAlign: "middle" }}>persistent</span>
          </summary>
          <HyperparamsEditor hyperparams={hyperparams} onSaved={setHyperparams} />
        </details>
      ) : null}

      {config ? (
        <div className="config-sections">
          <Section title="Environment">
            <ParamRow label="Grid size" value={env ? `${String(env.width)} × ${String(env.height)}` : "—"} />
            <ParamRow label="Max steps" value={env?.max_steps} />
            <ParamRow label="Pen size" value={env ? `${String(env.pen_width)} × ${String(env.pen_height)}` : "—"} />
            <ParamRow label="Pen opening" value={env?.pen_opening} />
          </Section>

          <Section title="Dogs">
            <ParamRow label="Count" value={env?.dogs} />
            <ParamRow label="Speed" value={env?.dog_speed} note="cells per step" />
            <ParamRow label="Sprint multiplier" value={env?.dog_sprint_multiplier} note="speed boost for sprint_* actions" />
            <ParamRow label="Vision" value={env?.dog_vision} note="cells; sheep react to dogs within this radius" />
            <ParamRow label="Policy mode" value={policy?.policy_mode} note="how the dog selects its action each step" />
            <OptionListRow label="Available actions" items={DOG_ACTIONS} />
            <OptionListRow label="Available roles" items={DOG_ROLES} />
          </Section>

          <Section title="Sheep">
            <ParamRow label="Count" value={env?.sheep} />
            <ParamRow label="Speed" value={env?.sheep_speed} note="cells per step" />
            <ParamRow label="Vision" value={env?.sheep_vision} />
            <ParamRow label="Flock radius" value={env?.flock_radius} />
            <ParamRow
              label="Personality colors"
              value={
                env?.sheep_personality_colors && typeof env.sheep_personality_colors === "object"
                  ? Object.entries(env.sheep_personality_colors as Record<string, unknown>)
                      .map(([name, color]) => `${name}: ${String(color)}`)
                      .join(", ")
                  : undefined
              }
              note="hex color shown in the replay viewer for each personality"
            />
            <ParamRow
              label="Personality strength"
              value={env?.sheep_personality_strength}
              note="0.0 disables (all sheep obedient, single color); ~0.25–0.5 is mild; assigned at episode reset and held fixed"
            />
            <OptionListRow label="Available personalities" items={SHEEP_PERSONALITIES} />
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

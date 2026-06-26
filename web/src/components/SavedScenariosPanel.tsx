import type { CheckpointEntry, ReplayBundle, SavedScenario, ScenarioIndex, ScenarioRunResult } from "../state/types";
import type { CheckpointMode } from "../lib/api";

interface SavedScenariosPanelProps {
  scenarioIndex: ScenarioIndex | null;
  checkpoints: CheckpointEntry[];
  latestCheckpointEpisode: number | null;
  bestCheckpointEpisode: number | null;
  checkpointMode: CheckpointMode;
  specificCheckpointEpisode: number | null;
  selectedScenarioId: string | null;
  scenarioReplay: ReplayBundle | null;
  saveSnapshotSource: "initial" | "final";
  running: boolean;
  disabled?: boolean;
  onCheckpointModeChange: (mode: CheckpointMode) => void;
  onSpecificCheckpointChange: (episode: number | null) => void;
  onSelectScenario: (scenarioId: string | null) => void;
  onSaveSnapshotSourceChange: (source: "initial" | "final") => void;
  onSaveFromReplay: () => void;
  onEvaluate: () => void;
  onRunScenario: () => void;
  canSaveFromReplay: boolean;
}

function formatResult(result: ScenarioRunResult): string {
  const status = result.success ? "success" : result.timeout ? "timeout" : "incomplete";
  return `${status} · ${result.sheep_penned} penned · ${result.steps} steps · reward ${result.reward_total.toFixed(1)}`;
}

export function SavedScenariosPanel({
  scenarioIndex,
  checkpoints,
  latestCheckpointEpisode,
  bestCheckpointEpisode,
  checkpointMode,
  specificCheckpointEpisode,
  selectedScenarioId,
  scenarioReplay,
  saveSnapshotSource,
  running,
  disabled,
  onCheckpointModeChange,
  onSpecificCheckpointChange,
  onSelectScenario,
  onSaveSnapshotSourceChange,
  onSaveFromReplay,
  onEvaluate,
  onRunScenario,
  canSaveFromReplay,
}: SavedScenariosPanelProps) {
  const scenarios = scenarioIndex?.scenarios ?? [];
  const effectiveSpecific =
    specificCheckpointEpisode ??
    latestCheckpointEpisode ??
    checkpoints[checkpoints.length - 1]?.checkpoint_episode ??
    null;

  const resolvedCheckpointLabel = (() => {
    if (checkpointMode === "latest") return latestCheckpointEpisode != null ? `ep ${latestCheckpointEpisode}` : "—";
    if (checkpointMode === "global_best") return bestCheckpointEpisode != null ? `ep ${bestCheckpointEpisode}` : "—";
    if (checkpointMode === "scenario_best") {
      const best = selectedScenarioId ? scenarioIndex?.best_by_scenario[selectedScenarioId] : null;
      return best ? `ep ${best.checkpoint_episode}` : "—";
    }
    return effectiveSpecific != null ? `ep ${effectiveSpecific}` : "—";
  })();

  const latestRuns = scenarioIndex?.runs ?? [];
  const runByScenario = new Map<string, ScenarioRunResult>();
  for (const run of latestRuns) {
    if (typeof run === "object" && run !== null && "scenario_id" in run) {
      runByScenario.set(run.scenario_id, run as ScenarioRunResult);
    }
  }
  const bestByScenario = scenarioIndex?.best_by_scenario ?? {};

  return (
    <section className="controls-card saved-scenarios-panel" aria-label="Saved test scenarios">
      <h3 className="saved-scenarios-panel__title">Saved scenarios</h3>
      <p className="scenario-panel__intro">
        Saved layouts reset the simulation to exact dog, sheep, pen, and field positions. Evaluate any
        checkpoint against them; per-scenario bests are tracked separately from the global best.
      </p>

      <div className="controls-grid">
        <label>
          <span>Checkpoint</span>
          <select
            className="hyperparam-input"
            value={checkpointMode}
            onChange={(event) => onCheckpointModeChange(event.target.value as CheckpointMode)}
            disabled={disabled || running}
          >
            <option value="latest">Latest checkpoint</option>
            <option value="global_best">Global best</option>
            <option value="scenario_best" disabled={!selectedScenarioId}>
              Best for this scenario
            </option>
            <option value="specific">Specific checkpoint…</option>
          </select>
        </label>

        {checkpointMode === "specific" ? (
          <label>
            <span>Episode</span>
            <select
              className="hyperparam-input"
              value={effectiveSpecific ?? ""}
              onChange={(event) => {
                const parsed = Number.parseInt(event.target.value, 10);
                onSpecificCheckpointChange(Number.isNaN(parsed) ? null : parsed);
              }}
              disabled={disabled || running || !checkpoints.length}
            >
              {checkpoints.map((entry) => (
                <option key={entry.checkpoint_episode} value={entry.checkpoint_episode}>
                  ep {entry.checkpoint_episode}
                  {entry.checkpoint_episode === latestCheckpointEpisode ? " (latest)" : ""}
                  {entry.checkpoint_episode === bestCheckpointEpisode ? " (global best)" : ""}
                </option>
              ))}
            </select>
          </label>
        ) : null}

        <label>
          <span>Scenario</span>
          <select
            className="hyperparam-input"
            value={selectedScenarioId ?? ""}
            onChange={(event) => onSelectScenario(event.target.value || null)}
            disabled={disabled || running || !scenarios.length}
          >
            <option value="">Select a scenario…</option>
            {scenarios.map((scenario: SavedScenario) => (
              <option key={scenario.id} value={scenario.id}>
                {scenario.name}
              </option>
            ))}
          </select>
        </label>
      </div>

      <p className="scenario-panel__hint">
        Resolved checkpoint: <strong>{resolvedCheckpointLabel}</strong>
      </p>

      <fieldset className="saved-scenarios-panel__fieldset">
        <legend>Save layout from replay</legend>
        <label className="training-toggle training-toggle--inline">
          <input
            type="radio"
            name="snapshot-source"
            checked={saveSnapshotSource === "initial"}
            onChange={() => onSaveSnapshotSourceChange("initial")}
            disabled={disabled || running}
          />
          <span>Starting position (frame 0)</span>
        </label>
        <label className="training-toggle training-toggle--inline">
          <input
            type="radio"
            name="snapshot-source"
            checked={saveSnapshotSource === "final"}
            onChange={() => onSaveSnapshotSourceChange("final")}
            disabled={disabled || running}
          />
          <span>Current / final position</span>
        </label>
      </fieldset>

      <div className="button-row">
        <button
          type="button"
          className="button-row__secondary"
          onClick={onSaveFromReplay}
          disabled={disabled || running || !canSaveFromReplay}
        >
          Save named scenario
        </button>
        <button
          type="button"
          className="button-row__primary"
          onClick={onEvaluate}
          disabled={disabled || running || !selectedScenarioId}
        >
          {running ? "Working…" : "Evaluate on scenario"}
        </button>
        <button
          type="button"
          className="button-row__secondary"
          onClick={onRunScenario}
          disabled={disabled || running || !selectedScenarioId}
        >
          Replay on scenario
        </button>
      </div>

      {scenarioReplay?.scenario_name ? (
        <p className="scenario-panel__policy">
          Last replay: <strong>{scenarioReplay.scenario_name}</strong>
          {scenarioReplay.checkpoint_episode != null ? ` · ep ${scenarioReplay.checkpoint_episode}` : null}
        </p>
      ) : null}

      {scenarios.length ? (
        <details className="training-advanced">
          <summary>Scenario performance table</summary>
          <div className="saved-scenarios-panel__table-wrap">
            <table className="saved-scenarios-panel__table">
              <thead>
                <tr>
                  <th>Scenario</th>
                  <th>Last run</th>
                  <th>Best for scenario</th>
                </tr>
              </thead>
              <tbody>
                {scenarios.map((scenario) => {
                  const latest = runByScenario.get(scenario.id);
                  const best = bestByScenario[scenario.id];
                  return (
                    <tr
                      key={scenario.id}
                      className={selectedScenarioId === scenario.id ? "saved-scenarios-panel__row--selected" : ""}
                    >
                      <td>
                        <button
                          type="button"
                          className="saved-scenarios-panel__name"
                          onClick={() => onSelectScenario(scenario.id)}
                        >
                          {scenario.name}
                        </button>
                        {scenario.description ? (
                          <span className="saved-scenarios-panel__desc">{scenario.description}</span>
                        ) : null}
                      </td>
                      <td>{latest ? formatResult(latest) : "—"}</td>
                      <td>
                        {best
                          ? `ep ${best.checkpoint_episode} · ${formatResult(best)}`
                          : "—"}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </details>
      ) : (
        <p className="scenario-panel__hint">No saved scenarios yet. Run a live episode, then save its layout.</p>
      )}
    </section>
  );
}

interface ScenarioPanelProps {
  personalityStrength: number;
  seed: number;
  fastMode: boolean;
  running: boolean;
  canEndEpisode: boolean;
  hasReplay: boolean;
  policyLabel: string;
  onPersonalityStrengthChange: (value: number) => void;
  onSeedChange: (seed: number) => void;
  onFastModeChange: (enabled: boolean) => void;
  onRun: () => void;
  onRestart: () => void;
  onEndEpisode: () => void;
  disabled?: boolean;
}

export function ScenarioPanel({
  personalityStrength,
  seed,
  fastMode,
  running,
  canEndEpisode,
  hasReplay,
  policyLabel,
  onPersonalityStrengthChange,
  onSeedChange,
  onFastModeChange,
  onRun,
  onRestart,
  onEndEpisode,
  disabled,
}: ScenarioPanelProps) {
  return (
    <section className="controls-card scenario-panel" aria-label="Scenario test controls">
      <p className="scenario-panel__intro">
        Run the current best model on a fresh episode with your scenario settings. This is a live
        simulation, not a saved checkpoint replay.
      </p>

      <div className="scenario-panel__field">
        <label htmlFor="scenario-personality-strength">
          <span>Personality strength</span>
          <output htmlFor="scenario-personality-strength" className="scenario-panel__value">
            {personalityStrength.toFixed(2)}
          </output>
        </label>
        <input
          id="scenario-personality-strength"
          type="range"
          min={0}
          max={3}
          step={0.05}
          value={personalityStrength}
          onChange={(event) => onPersonalityStrengthChange(Number(event.target.value))}
          disabled={disabled || running}
        />
        <p className="scenario-panel__hint">
          0 = all obedient; ~0.25–0.5 mild variety; higher = stronger pen-shy, escapist, and bold
          behaviors.
        </p>
      </div>

      <div className="controls-grid">
        <label>
          <span>Seed</span>
          <input
            type="number"
            className="hyperparam-input"
            min={0}
            max={999999}
            step={1}
            value={seed}
            onChange={(event) => {
              const parsed = Number.parseInt(event.target.value, 10);
              if (!Number.isNaN(parsed)) {
                onSeedChange(parsed);
              }
            }}
            disabled={disabled || running}
          />
        </label>

        <label className="training-toggle training-toggle--inline">
          <input
            type="checkbox"
            checked={fastMode}
            onChange={(event) => onFastModeChange(event.target.checked)}
            disabled={disabled}
          />
          <span>Fast playback</span>
        </label>
      </div>

      <p className="scenario-panel__policy">
        Policy: <strong>{policyLabel}</strong>
      </p>

      <div className="button-row">
        <button type="button" className="button-row__primary" onClick={onRun} disabled={disabled || running}>
          {running ? "Running scenario…" : "Run scenario"}
        </button>
        <button
          type="button"
          className="button-row__secondary"
          onClick={onRestart}
          disabled={disabled || !hasReplay || running}
        >
          Restart playback
        </button>
        <button
          type="button"
          className="button-row__secondary"
          onClick={onEndEpisode}
          disabled={disabled || !canEndEpisode}
        >
          End episode
        </button>
      </div>
    </section>
  );
}

import { useState, type ReactNode } from "react";
import { ConfigPanel } from "../components/ConfigPanel";
import { HistoryTab } from "../components/HistoryTab";
import { LayersTab } from "../components/LayersTab";
import { NetworkTab } from "../components/NetworkTab";
import { StagesTab } from "../components/StagesTab";
import { WandbTab } from "../components/WandbTab";
import type { CheckpointIndex, NetworkTopologyInfo, TrainingStatus } from "../state/types";

export interface SettingsViewProps {
  checkpointIndex: CheckpointIndex | null;
  trainingStatus: TrainingStatus | null;
  effectiveConfig: Record<string, unknown> | null;
  topologyInfo: NetworkTopologyInfo | null;
  running?: boolean;
  clearing?: boolean;
  isReadOnlyPreview?: boolean;
  onResetJourney: () => void;
  onClearTraining: () => void;
  onCloseApp: () => void;
  onRefreshData: () => void;
  onWatchReplay: (episode: number) => void;
  onExpandNetwork?: () => void;
}

interface LazyDisclosureProps {
  summary: string;
  onOpen?: () => void;
  children: ReactNode;
}

function LazyDisclosure({ summary, onOpen, children }: LazyDisclosureProps) {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <details
      className="ui-disclosure"
      onToggle={(e) => {
        const nextOpen = e.currentTarget.open;
        setIsOpen(nextOpen);
        if (nextOpen) {
          onOpen?.();
        }
      }}
    >
      <summary>{summary}</summary>
      <div className="ui-disclosure__body">{isOpen ? children : null}</div>
    </details>
  );
}

export function SettingsView({
  checkpointIndex,
  trainingStatus,
  effectiveConfig,
  topologyInfo,
  running = false,
  clearing = false,
  isReadOnlyPreview = false,
  onResetJourney,
  onClearTraining,
  onCloseApp,
  onRefreshData,
  onWatchReplay,
  onExpandNetwork,
}: SettingsViewProps) {
  const isTraining = running || Boolean(trainingStatus?.running);
  const mutationsDisabled = isTraining || clearing || isReadOnlyPreview;

  return (
    <div className="ui-view-stack">
      {/* 1. Hyperparameters & Configuration */}
      <div className="ui-panel">
        <ConfigPanel />
      </div>

      {/* 2. Session & Lifecycle Actions */}
      <section className="ui-panel ui-action-bar" aria-label="Session and lifecycle actions">
        <div className="ui-action-bar__copy">
          <h2 className="ui-panel__title">Session &amp; Lifecycle Actions</h2>
          <p className="ui-panel__subtitle">
            Manage the active curriculum journey, clear stored training artifacts, or shut down the backend server.
          </p>
        </div>
        <div className="ui-action-bar__buttons">
          <button
            type="button"
            onClick={onResetJourney}
            disabled={mutationsDisabled}
          >
            Reset Journey to Stage 1
          </button>
          <button
            type="button"
            className="ui-btn--danger"
            onClick={onClearTraining}
            disabled={mutationsDisabled}
          >
            {clearing ? "Clearing…" : "Clear Saved Training"}
          </button>
          <button
            type="button"
            className="ui-btn--danger"
            onClick={onCloseApp}
            disabled={isReadOnlyPreview}
          >
            Shut Down Server
          </button>
        </div>
      </section>

      {/* 3. Collapsible Reference & Advanced Tools */}
      <div className="ui-disclosure-list">
        <LazyDisclosure summary="Checkpoint Browser, Restore & Fork">
          <WandbTab
            checkpointIndex={checkpointIndex}
            trainingStatus={trainingStatus}
            effectiveConfig={effectiveConfig}
            onRefreshData={onRefreshData}
            onWatchReplay={onWatchReplay}
          />
        </LazyDisclosure>

        <LazyDisclosure summary="Policy Network Topology" onOpen={onExpandNetwork}>
          <NetworkTab
            checkpointIndex={checkpointIndex}
            trainingStatus={trainingStatus}
            effectiveConfig={effectiveConfig}
            topologyInfo={topologyInfo}
          />
        </LazyDisclosure>

        <LazyDisclosure summary="Observation & Action Layers">
          <LayersTab
            effectiveConfig={effectiveConfig}
            topologyInfo={topologyInfo}
          />
        </LazyDisclosure>

        <LazyDisclosure summary="Curriculum Stages & RL Glossary">
          <StagesTab />
        </LazyDisclosure>

        <LazyDisclosure summary="Raw Episode Telemetry Log">
          <HistoryTab />
        </LazyDisclosure>
      </div>
    </div>
  );
}

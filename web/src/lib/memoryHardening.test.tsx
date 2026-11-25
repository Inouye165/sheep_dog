import { useRef, useState } from "react";
import { act, cleanup, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  loadCheckpointIndexConditional,
  loadRecentEvaluations,
  loadReplay,
  setIsBackendOffline,
  slimCheckpointIndex,
  slimEvaluationSummary,
} from "./api";
import {
  episodeHistoryStore,
  MAX_CACHED_REPLAY_HISTORIES,
  MAX_EPISODE_HISTORY_BUFFER_SIZE,
} from "./episodeHistoryStore";
import { useTrainingStatusPolling } from "../hooks/useTrainingSync";
import type {
  CheckpointIndex,
  EpisodeRecord,
  EvaluationSummaryPayload,
  ReplayBundle,
  TrainingStatus,
} from "../state/types";
import { ProgressView } from "../views/ProgressView";
import { SettingsView } from "../views/SettingsView";
import { TrainView } from "../views/TrainView";

vi.mock("../components/EarlyWarningsPanel", () => ({
  EarlyWarningsPanel: () => <div data-testid="mock-early-warnings" />,
}));
vi.mock("../studio/StudioLearningCharts", () => ({
  StudioLearningCharts: () => <div data-testid="mock-learning-charts" />,
}));
vi.mock("../components/ResultsPanel", () => ({
  ResultsPanel: () => <div data-testid="mock-results-panel" />,
}));
vi.mock("../components/ConfigPanel", () => ({
  ConfigPanel: () => <div data-testid="mock-config-panel" />,
}));
vi.mock("../components/WandbTab", () => ({
  WandbTab: () => <div data-testid="mock-wandb-tab" />,
}));
vi.mock("../components/HistoryTab", () => ({
  HistoryTab: () => <div data-testid="mock-history-tab" />,
}));
vi.mock("../components/NetworkTab", () => ({
  NetworkTab: () => <div data-testid="mock-network-tab" />,
}));
vi.mock("../components/LayersTab", () => ({
  LayersTab: () => <div data-testid="mock-layers-tab" />,
}));
vi.mock("../components/StagesTab", () => ({
  StagesTab: () => <div data-testid="mock-stages-tab" />,
}));

function setDocumentVisibility(state: DocumentVisibilityState): void {
  Object.defineProperty(document, "visibilityState", {
    configurable: true,
    get: () => state,
  });
}

describe("Memory & CPU Hardening", () => {
  const originalFetch = global.fetch;

  beforeEach(() => {
    vi.restoreAllMocks();
    setIsBackendOffline(false);
    episodeHistoryStore.clear();
    setDocumentVisibility("visible");
  });

  afterEach(() => {
    cleanup();
    global.fetch = originalFetch;
    setIsBackendOffline(false);
    episodeHistoryStore.clear();
    setDocumentVisibility("visible");
  });

  it("1. slimCheckpointIndex strips bloat fields and excess config keys while preserving needed env and instincts fields", () => {
    const raw = {
      checkpoints: [
        {
          checkpoint_episode: 100,
          checkpoint: "chk_100.pt",
          evaluation: "eval_100.json",
          replay: "rep_100.json",
          success_rate: 0.9,
          timeout_rate: 0.1,
          average_completion_steps: 90,
          average_completion_seconds: 22.5,
          average_sheep_penned: 5.5,
          average_reward: 110,
          failure_directed_telemetry: { large: [1, 2, 3] },
          failure_directed_weights: { seed101: 5.0 },
          policy_weights: new Array(1000).fill(0.1),
          training_scenario_coverage: { buckets: [1, 2] },
          policy_config: { hidden_sizes: [128, 128] },
          environment_config: {
            dogs: 3,
            sheep: 6,
            width: 80,
            height: 60,
            curriculum_stage: 12,
            excess_noise_matrix: [[1, 2]],
            extra_spawn_rules: { corner: true },
          },
          reward_config: {
            instincts: {
              curriculum_stage: 12,
              enable_instinct_rewards: true,
              extra_instinct_debug: { huge: true },
            },
            shaping: { progress_scale: 1.0 },
          },
          records: [
            {
              seed: 11,
              success: false,
              timeout: true,
              stopped: false,
              steps: 300,
              simulated_seconds: 75,
              sheep_penned: 4,
              final_sheep_distance_to_pen: 8,
              no_progress_steps: 10,
              reward_total: -5,
              reward_breakdown: {} as any,
              replay_path: "/generated/replays/cp100_11.json",
              failed_trajectory_summary: [{ step: 1 }],
              observation_diagnostics: { vector_length: 54 },
            },
          ],
        },
        {
          checkpoint_episode: 200,
          checkpoint: "chk_200.pt",
          evaluation: "eval_200.json",
          replay: "rep_200.json",
          success_rate: 1.0,
          timeout_rate: 0,
          average_completion_steps: 70,
          average_completion_seconds: 17.5,
          average_sheep_penned: 6,
          average_reward: 140,
          reward_config: { shaping: { progress_scale: 2.0 } },
          records: [],
        },
      ],
      latest: {
        checkpoint_episode: 200,
        policy_name: "neural_policy",
        success_rate: 1.0,
        timeout_rate: 0,
        average_completion_steps: 70,
        average_completion_seconds: 17.5,
        average_sheep_penned: 6,
        average_reward: 140,
        failure_directed_telemetry: { huge: true },
        policy_weights: [1, 2, 3],
        records: [
          {
            seed: 101,
            success: true,
            timeout: false,
            stopped: false,
            steps: 70,
            simulated_seconds: 17.5,
            sheep_penned: 6,
            final_sheep_distance_to_pen: 0,
            no_progress_steps: 0,
            reward_total: 140,
            reward_breakdown: {} as any,
            replay_path: "/generated/replays/cp200_101.json",
            failed_trajectory_summary: [{ step: 1 }],
            observation_diagnostics: { vector_length: 54 },
          },
        ],
      },
    } as unknown as CheckpointIndex;

    const slimmed = slimCheckpointIndex(raw);
    const cp0 = slimmed.checkpoints[0] as unknown as Record<string, unknown>;
    expect(cp0.failure_directed_telemetry).toBeUndefined();
    expect(cp0.failure_directed_weights).toBeUndefined();
    expect(cp0.policy_weights).toBeUndefined();
    expect(cp0.training_scenario_coverage).toBeUndefined();
    expect(cp0.policy_config).toBeUndefined();
    expect(slimmed.checkpoints[0].environment_config).toEqual({
      dogs: 3,
      sheep: 6,
      width: 80,
      height: 60,
      curriculum_stage: 12,
    });
    expect(slimmed.checkpoints[0].reward_config).toEqual({
      instincts: {
        curriculum_stage: 12,
        enable_instinct_rewards: true,
      },
    });
    expect(slimmed.checkpoints[0].records[0].failed_trajectory_summary).toBeUndefined();
    expect(slimmed.checkpoints[0].records[0].observation_diagnostics).toBeUndefined();
    expect(slimmed.checkpoints[0].records[0].replay_path).toBe("/generated/replays/cp100_11.json");

    expect(slimmed.checkpoints[1].reward_config).toBeUndefined();

    const slimLatest = slimmed.latest as unknown as Record<string, unknown>;
    expect(slimLatest.failure_directed_telemetry).toBeUndefined();
    expect(slimLatest.policy_weights).toBeUndefined();
    expect(slimmed.latest?.records[0].failed_trajectory_summary).toBeUndefined();
    expect(slimmed.latest?.records[0].observation_diagnostics).toBeUndefined();
  });

  it("2. loadCheckpointIndexConditional sends If-None-Match, handles 304 without .json(), handles HTML fallback, and loadReplay deduplicates in-flight calls", async () => {
    const jsonSpy = vi.fn().mockResolvedValue({ checkpoints: [], latest: null });
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce({
        ok: false,
        status: 304,
        headers: new Headers({ etag: '"etag-v1"' }),
        json: jsonSpy,
      } as unknown as Response)
      .mockResolvedValueOnce(
        new Response("<!doctype html><html><body>fallback</body></html>", {
          status: 200,
          headers: { "Content-Type": "text/html" },
        }),
      );

    global.fetch = fetchMock;

    const notModRes = await loadCheckpointIndexConditional('"etag-v1"');
    expect(notModRes).toEqual({ status: "not_modified", etag: '"etag-v1"' });
    expect(jsonSpy).not.toHaveBeenCalled();
    expect(fetchMock).toHaveBeenCalledWith(
      "/generated/checkpoint-index.json",
      expect.objectContaining({
        cache: "no-store",
        headers: { "If-None-Match": '"etag-v1"' },
      }),
    );

    const htmlRes = await loadCheckpointIndexConditional(null);
    expect(htmlRes).toEqual({ status: "empty" });

    let resolveReplay!: (res: Response) => void;
    const pendingReplay = new Promise<Response>((resolve) => {
      resolveReplay = resolve;
    });
    const replayFetchMock = vi.fn().mockReturnValue(pendingReplay);
    global.fetch = replayFetchMock;

    const fakeBundle = {
      seed: 11,
      policy_name: "neural_policy",
      final_snapshot: {} as ReplayBundle["final_snapshot"],
      stats: {} as ReplayBundle["stats"],
      frames: [],
    } satisfies ReplayBundle;

    const p1 = loadReplay("/custom/path/replay-11.json");
    const p2 = loadReplay("/custom/path/replay-11.json");
    expect(replayFetchMock).toHaveBeenCalledTimes(1);

    resolveReplay(
      new Response(JSON.stringify(fakeBundle), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }),
    );

    const [b1, b2] = await Promise.all([p1, p2]);
    expect(b1).toEqual(fakeBundle);
    expect(b2).toEqual(fakeBundle);

    // Subsequent sequential call after settlement fetches fresh (no permanent cache retention)
    replayFetchMock.mockResolvedValueOnce(
      new Response(JSON.stringify(fakeBundle), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }),
    );
    await loadReplay("/custom/path/replay-11.json");
    expect(replayFetchMock).toHaveBeenCalledTimes(2);
  });

  it("3. episodeHistoryStore retains at most MAX_CACHED_REPLAY_HISTORIES move_history arrays while keeping 50 scalar records", () => {
    expect(MAX_CACHED_REPLAY_HISTORIES).toBe(3);

    for (let i = 1; i <= MAX_EPISODE_HISTORY_BUFFER_SIZE; i += 1) {
      const record: EpisodeRecord = {
        episode_id: i,
        timestamp: "12:00:00",
        stage: 8,
        outcome: "loss",
        outcome_label: "STOPPED",
        total_moves: 100 + i,
        replayAvailable: true,
        replay_id: `rep_${i}`,
      };
      if (i <= 5) {
        record.initial_state = { step: 0 } as EpisodeRecord["initial_state"];
        record.move_history = new Array(101 + i).fill({ step: 0 }) as EpisodeRecord["move_history"];
      }
      episodeHistoryStore.addEpisode(record);
    }

    const episodes = episodeHistoryStore.getEpisodes();
    expect(episodes).toHaveLength(50);

    const withMoveHistory = episodes.filter((ep) => ep.move_history && ep.move_history.length > 0);
    expect(withMoveHistory).toHaveLength(MAX_CACHED_REPLAY_HISTORIES);
    expect(withMoveHistory.map((ep) => ep.episode_id).sort((a, b) => Number(a) - Number(b))).toEqual([
      3,
      4,
      5,
    ]);

    const evictedEp1 = episodes.find((ep) => ep.episode_id === 1);
    const evictedEp2 = episodes.find((ep) => ep.episode_id === 2);
    expect(evictedEp1?.move_history).toBeUndefined();
    expect(evictedEp1?.initial_state).toBeUndefined();
    expect(evictedEp1?.total_moves).toBe(101);
    expect(evictedEp2?.move_history).toBeUndefined();
    expect(evictedEp2?.initial_state).toBeUndefined();
    expect(evictedEp2?.total_moves).toBe(102);

    episodeHistoryStore.clear();
    expect(episodeHistoryStore.getEpisodes()).toHaveLength(0);
  });

  it("4. ProgressView does not mount ResultsPanel until Milestones & Video tab is opened and unmounts it when switching tabs", async () => {
    const user = userEvent.setup();
    render(
      <ProgressView
        checkpoints={[]}
        checkpointIndex={null}
        trainingEpisodes={[]}
        sessionEvents={[]}
        trainingStatus={null}
        effectiveCurriculumStage={1}
        bestCheckpointEpisode={null}
        selectedCheckpointEpisode={null}
        onInspectTroubleSpots={vi.fn()}
        onSelectCheckpointForReplay={vi.fn()}
        onInspectCheckpointInTroubleSpots={vi.fn()}
        onOpenReplayInInspector={vi.fn()}
      />,
    );

    expect(screen.queryByTestId("mock-results-panel")).not.toBeInTheDocument();

    await user.click(screen.getByRole("tab", { name: "Milestones & Video" }));
    await waitFor(() => {
      expect(screen.getByTestId("mock-results-panel")).toBeInTheDocument();
    });

    await user.click(screen.getByRole("tab", { name: "Learning Curve" }));
    await waitFor(() => {
      expect(screen.queryByTestId("mock-results-panel")).not.toBeInTheDocument();
    });
  });

  it("5. SettingsView does not mount HistoryTab or WandbTab until their disclosures are opened", async () => {
    const user = userEvent.setup();
    render(
      <SettingsView
        checkpointIndex={null}
        trainingStatus={null}
        effectiveConfig={null}
        topologyInfo={null}
        onResetJourney={vi.fn()}
        onClearTraining={vi.fn()}
        onCloseApp={vi.fn()}
        onRefreshData={vi.fn()}
        onWatchReplay={vi.fn()}
      />,
    );

    expect(screen.queryByTestId("mock-wandb-tab")).not.toBeInTheDocument();
    expect(screen.queryByTestId("mock-history-tab")).not.toBeInTheDocument();

    await user.click(screen.getByText("Checkpoint Browser, Restore & Fork"));
    await waitFor(() => {
      expect(screen.getByTestId("mock-wandb-tab")).toBeInTheDocument();
    });
    expect(screen.queryByTestId("mock-history-tab")).not.toBeInTheDocument();

    await user.click(screen.getByText("Raw Episode Telemetry Log"));
    await waitFor(() => {
      expect(screen.getByTestId("mock-history-tab")).toBeInTheDocument();
    });

    await user.click(screen.getByText("Raw Episode Telemetry Log"));
    await waitFor(() => {
      expect(screen.queryByTestId("mock-history-tab")).not.toBeInTheDocument();
    });
  });

  it("6. useTrainingStatusPolling pauses network calls when visibilityState is hidden and resumes when visible", async () => {
    setDocumentVisibility("hidden");
    const fetchMock = vi.fn().mockImplementation(async () =>
      new Response(
        JSON.stringify({
          running: false,
          fast_mode: true,
          enable_instinct_rewards: false,
          debug_reward_breakdown: false,
          curriculum_stage: 1,
          requested_episodes: 0,
          completed_episodes: 0,
          batch_total_episodes: 0,
          batch_completed_episodes: 0,
          total_episodes_trained: 0,
          stage_history: {},
          grand_total_episodes: 0,
          current_episode: null,
          checkpoint_episode: null,
          latest_checkpoint_episode: null,
          latest_seed: null,
          latest_replay_path: null,
          best_score: null,
          latest_success_rate: null,
          latest_avg_sheep_penned: null,
          latest_avg_reward: null,
          latest_timeout_rate: null,
          latest_stopped_rate: null,
          latest_avg_no_progress_steps: null,
          latest_avg_distance_to_pen: null,
          latest_avg_flock_spread: null,
          latest_avg_farthest_distance_to_pen: null,
          latest_avg_farthest_distance_to_flock_center: null,
          phase: "idle",
          message: "Idle",
          error: null,
          starting_episode: null,
        } satisfies TrainingStatus),
        { status: 200, headers: { "Content-Type": "application/json" } },
      ),
    );
    global.fetch = fetchMock;

    function PollingHarness() {
      const [, setStatus] = useState<TrainingStatus | null>(null);
      const seqRef = useRef(0);
      const manualRef = useRef(false);
      useTrainingStatusPolling({
        activePollDelayMs: 50,
        statusMutationSeqRef: seqRef,
        manualStageOverrideRef: manualRef,
        onStatusReceived: setStatus,
        syncTrainingStageFromStatus: () => {},
      });
      return null;
    }

    render(<PollingHarness />);

    await new Promise((r) => setTimeout(r, 120));
    expect(fetchMock).not.toHaveBeenCalled();

    await act(async () => {
      setDocumentVisibility("visible");
      document.dispatchEvent(new Event("visibilitychange"));
    });

    await waitFor(() => {
      expect(fetchMock).toHaveBeenCalledTimes(1);
    });

    const callsWhenVisible = fetchMock.mock.calls.length;
    await act(async () => {
      setDocumentVisibility("hidden");
      document.dispatchEvent(new Event("visibilitychange"));
    });

    await new Promise((r) => setTimeout(r, 130));
    expect(fetchMock.mock.calls.length).toBeLessThanOrEqual(callsWhenVisible + 1);
    const settledCount = fetchMock.mock.calls.length;
    await new Promise((r) => setTimeout(r, 120));
    expect(fetchMock.mock.calls.length).toBe(settledCount);
  });

  it("7. TrainView windows checkpoint dropdown to latest 200 entries while preserving selected and best checkpoints", () => {
    const checkpoints = Array.from({ length: 250 }, (_, idx) => ({
      checkpoint_episode: idx + 1,
      checkpoint: `chk_${idx + 1}.pt`,
      evaluation: `eval_${idx + 1}.json`,
      replay: `rep_${idx + 1}.json`,
      curriculum_stage: 3,
      success_rate: 0.5,
      timeout_rate: 0.5,
      average_completion_steps: 100,
      average_completion_seconds: 25,
      average_sheep_penned: 3,
      average_reward: 10,
      records: [],
    }));

    render(
      <TrainView
        checkpoints={checkpoints}
        selectedCheckpoint={checkpoints[4]}
        selectedCheckpointEpisode={5}
        bestCheckpointEpisode={10}
        seedOptions={[]}
        selectedSeed={null}
        replay={null}
        snapshot={null}
        frameIndex={0}
        isPlaying={false}
        statusLabel="idle"
        canEndEpisode={false}
        runningCurrent={false}
        loadingSelectedReplay={false}
        playbackFastMode={false}
        trainingStatus={null}
        curriculumStage={3}
        maxCurriculumStage={32}
        episodes={50}
        fastMode={true}
        enableInstincts={false}
        autoPromote={true}
        debugRewardBreakdown={false}
        startingModelSource="latest"
        running={false}
        clearing={false}
        isStartingTraining={false}
        lifecycleAction={null}
        latestSuccessRate={0.5}
        onSelectCheckpointEpisode={vi.fn()}
        onSelectSeed={vi.fn()}
        onSelectSeedAndReplay={vi.fn()}
        onRunCurrentReplay={vi.fn()}
        onReplaySelected={vi.fn()}
        onEndEpisode={vi.fn()}
        onTogglePlay={vi.fn()}
        onStepFrame={vi.fn()}
        onScrubFrame={vi.fn()}
        onPlaybackFastModeChange={vi.fn()}
        onOpenInspector={vi.fn()}
        onOpenTroubleSpots={vi.fn()}
        onOpenProgress={vi.fn()}
        onPromote={vi.fn()}
        onCurriculumStageChange={vi.fn()}
        onEpisodesChange={vi.fn()}
        onStartTraining={vi.fn()}
        onResumeTraining={vi.fn()}
        onPauseTraining={vi.fn()}
        onStopTraining={vi.fn()}
        onEnableInstinctsChange={vi.fn()}
        onAutoPromoteChange={vi.fn()}
        onStartingModelSourceChange={vi.fn()}
        onFastModeChange={vi.fn()}
        onDebugRewardBreakdownChange={vi.fn()}
        onStartRemediation={vi.fn()}
        onClearTraining={vi.fn()}
      />,
    );

    const select = screen.getByLabelText("Checkpoint") as HTMLSelectElement;
    expect(select.options).toHaveLength(202);
    expect(select.options[0].value).toBe("5");
    expect(select.options[1].value).toBe("10");
    expect(select.options[1].text).toContain("(Best)");
    expect(select.options[2].value).toBe("51");
    expect(select.options[201].value).toBe("250");
  });

  it("8. slimEvaluationSummary and loadRecentEvaluations strip heavy diagnostic blobs non-mutatingly while preserving metrics and metadata", async () => {
    const rawEval = {
      checkpoint_episode: 3500,
      policy_name: "neural_policy",
      curriculum_stage: 37,
      success_rate: 0.9,
      timeout_rate: 0.1,
      average_completion_steps: 142,
      average_sheep_penned: 5.8,
      average_reward: 112.4,
      pinned: true,
      is_milestone: true,
      is_first: false,
      failure_directed_telemetry: { large: [1, 2, 3] },
      failure_directed_weights: { seed101: 2.5 },
      policy_weights: new Array(200).fill(0.25),
      training_scenario_coverage: { buckets: [1, 2, 3] },
      records: [
        {
          seed: 101,
          success: true,
          timeout: false,
          stopped: false,
          steps: 120,
          simulated_seconds: 30,
          sheep_penned: 6,
          final_sheep_distance_to_pen: 0,
          no_progress_steps: 0,
          reward_total: 135.5,
          replay_path: "/generated/replays/checkpoint-003500-seed-000101.json",
          reward_breakdown: { progress_to_pen: 12.5, terminal_success: 50 },
          spatial_metrics: { pen_approach_time: 18 },
          failed_trajectory_summary: [{ step: 1, event: "stall" }],
          observation_diagnostics: { vector_length: 54 },
        },
      ],
    } as unknown as EvaluationSummaryPayload;

    const slimmed = slimEvaluationSummary(rawEval);
    const slimRec = slimmed.records[0] as unknown as Record<string, unknown>;
    const slimObj = slimmed as unknown as Record<string, unknown>;

    expect(slimObj.failure_directed_telemetry).toBeUndefined();
    expect(slimObj.failure_directed_weights).toBeUndefined();
    expect(slimObj.policy_weights).toBeUndefined();
    expect(slimObj.training_scenario_coverage).toBeUndefined();
    expect(slimRec.failed_trajectory_summary).toBeUndefined();
    expect(slimRec.observation_diagnostics).toBeUndefined();

    // Preserves metrics and metadata
    expect(slimmed.pinned).toBe(true);
    expect(slimmed.is_milestone).toBe(true);
    expect(slimmed.is_first).toBe(false);
    expect(slimmed.records[0].seed).toBe(101);
    expect(slimmed.records[0].replay_path).toBe("/generated/replays/checkpoint-003500-seed-000101.json");
    expect(slimmed.records[0].reward_breakdown).toEqual({ progress_to_pen: 12.5, terminal_success: 50 });
    expect(slimRec.spatial_metrics).toEqual({ pen_approach_time: 18 });

    // Original raw object and record remain unmutated
    expect((rawEval as unknown as Record<string, unknown>).policy_weights).toBeDefined();
    expect((rawEval.records[0] as unknown as Record<string, unknown>).failed_trajectory_summary).toBeDefined();

    // Verify loadRecentEvaluations applies slimEvaluationSummary
    global.fetch = vi.fn().mockResolvedValueOnce(
      new Response(JSON.stringify({ evaluations: [rawEval] }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }),
    );

    const fetched = await loadRecentEvaluations(25, 37);
    expect(fetched).toHaveLength(1);
    const fetched0 = fetched![0] as unknown as Record<string, unknown>;
    const fetchedRec0 = fetched![0].records[0] as unknown as Record<string, unknown>;
    expect(fetched0.failure_directed_telemetry).toBeUndefined();
    expect(fetched0.policy_weights).toBeUndefined();
    expect(fetchedRec0.failed_trajectory_summary).toBeUndefined();
    expect(fetchedRec0.observation_diagnostics).toBeUndefined();
    expect(fetched![0].pinned).toBe(true);
    expect(fetched![0].is_milestone).toBe(true);
    expect(fetched![0].records[0].replay_path).toBe("/generated/replays/checkpoint-003500-seed-000101.json");
  });
});

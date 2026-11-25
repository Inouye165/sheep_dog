import type {
  CheckpointEntry,
  CheckpointIndex,
  ConfigHistory,
  ConfigRevision,
  EvaluationSummary,
  NetworkTopologyInfo,
  ReplayBundle,
  ReplayRunRequest,
  SavedScenario,
  ScenarioIndex,
  ScenarioRunResult,
  TrainingStartRequest,
  StageTrainingReport,
  TrainingStatus,
  UserHyperparams,
  TelemetryRecord,
  DiagnosticsResponse,
  TrainingEpisodesResponse,
  TrainingEpisode,
  CapturePolicyConfig,
  StageBottleneckReport,
  EvaluationSummaryPayload,
  StageHealthSummary,
  SessionEventsResponse,
  TemporalDiagnosticExample,
  TemporalDiagnosticPayload,
  EarlyWarningsPayload,
  StageTimingsResponse,
  ModelHistoryResponse,
  ModelLineageResponse,
} from "../state/types";

export const API_BASE_URL = import.meta.env.VITE_API_BASE_URL?.trim() || "";

export interface ApiError extends Error {
  status?: number;
  contentType?: string;
  isNetworkError?: boolean;
  /** Machine-readable error code from the backend JSON body (e.g. "BACKEND_STARTING"). */
  code?: string;
}

/** True when the backend is up but still restoring training state (HTTP 503 BACKEND_STARTING). */
export function isBackendStartingError(error: unknown): boolean {
  const apiError = error as ApiError | null;
  return Boolean(apiError) && apiError?.status === 503 && apiError?.code === "BACKEND_STARTING";
}

const OFFLINE_COOLDOWN_MS = 2_000;
let backendOfflineState = false;
let offlineRetryAfterMs = 0;

function markBackendOffline(): void {
  backendOfflineState = true;
  offlineRetryAfterMs = Date.now() + OFFLINE_COOLDOWN_MS;
}

function clearBackendOffline(): void {
  backendOfflineState = false;
  offlineRetryAfterMs = 0;
}

export function isBackendMarkedOffline(): boolean {
  if (!backendOfflineState) {
    return false;
  }
  if (Date.now() >= offlineRetryAfterMs) {
    backendOfflineState = false;
    offlineRetryAfterMs = 0;
    return false;
  }
  return true;
}

export function getIsBackendOffline(): boolean {
  return isBackendMarkedOffline();
}

export function setIsBackendOffline(offline: boolean): void {
  if (offline) {
    markBackendOffline();
  } else {
    clearBackendOffline();
  }
}

function isJsonResponse(response: Response): boolean {
  const contentType = response.headers.get("content-type") ?? "";
  return contentType.toLowerCase().includes("application/json");
}

async function fetchJson<T>(path: string, init?: RequestInit, baseUrl?: string): Promise<T> {
  const requestUrl = baseUrl ? new URL(path, baseUrl) : path;
  try {
    const response = await fetch(requestUrl, { cache: "no-store", ...init });
    clearBackendOffline();
    if (!response.ok) {
      let errMsg = `Failed to fetch ${path}: ${response.status}`;
      let errCode: string | undefined;
      try {
        if (isJsonResponse(response)) {
          const errJson = await response.clone().json();
          if (errJson && typeof errJson === "object" && "error" in errJson) {
            errMsg = String(errJson.error);
          }
          if (errJson && typeof errJson === "object" && typeof errJson.code === "string") {
            errCode = errJson.code;
          }
        }
      } catch {
        // ignore
      }
      const error = new Error(errMsg) as ApiError;
      error.status = response.status;
      error.code = errCode;
      throw error;
    }

    if (!isJsonResponse(response)) {
      const error = new Error(`Expected JSON from ${path}, received ${response.headers.get("content-type") ?? "unknown content type"}`) as ApiError;
      error.status = response.status;
      error.contentType = response.headers.get("content-type") ?? undefined;
      throw error;
    }

    return (await response.json()) as T;
  } catch (error) {
    if (error instanceof TypeError || (error instanceof Error && error.name === "TypeError")) {
      markBackendOffline();
      const netErr = new Error(`Network or connection error fetching ${path}: ${error.message}`) as ApiError;
      netErr.status = 0;
      netErr.isNetworkError = true;
      throw netErr;
    }
    throw error;
  }
}

export interface BackendHealthResult {
  ok: boolean;
  service?: string;
  ready?: boolean;
  status?: string;
  mode?: string;
  readOnlyPreview?: boolean;
  starting?: boolean;
  error?: string;
}

export async function checkBackendHealth(): Promise<BackendHealthResult> {
  try {
    const res = await fetchJson<{
      status: string;
      service?: string;
      ready?: boolean;
      mode?: string;
      read_only_preview?: boolean;
      code?: string;
    }>("/api/health", undefined, API_BASE_URL);
    clearBackendOffline();
    const isStarting = res.status === "starting" || res.code === "BACKEND_STARTING";
    const isReadOnly = Boolean(res.read_only_preview || res.mode === "read_only_preview");
    return {
      ok: res.status === "ok" || isStarting,
      service: res.service,
      ready: res.ready ?? res.status === "ok",
      status: res.status,
      mode: res.mode ?? (isReadOnly ? "read_only_preview" : "normal"),
      readOnlyPreview: isReadOnly,
      starting: isStarting,
    };
  } catch (err) {
    if (isBackendStartingError(err)) {
      clearBackendOffline();
      return {
        ok: true,
        ready: false,
        status: "starting",
        starting: true,
      };
    }
    const fetchErr = err as ApiError;
    if (fetchErr?.status === 404) {
      clearBackendOffline();
      return {
        ok: true,
        ready: true,
        status: "unimplemented_health_mock",
        mode: "normal",
        readOnlyPreview: false,
        starting: false,
      };
    }
    if (fetchErr?.isNetworkError || fetchErr?.status === 0 || !fetchErr?.status || err instanceof TypeError) {
      markBackendOffline();
    }
    return { ok: false, error: fetchErr.message || String(err) };
  }
}

const DROPPED_EVAL_SUMMARY_KEYS = [
  "failure_directed_telemetry",
  "failure_directed_weights",
  "policy_weights",
  "training_scenario_coverage",
] as const;

const DROPPED_CHECKPOINT_KEYS = [
  ...DROPPED_EVAL_SUMMARY_KEYS,
  "policy_config",
] as const;

const ALLOWED_ENV_CONFIG_KEYS = [
  "dogs",
  "sheep",
  "width",
  "height",
  "curriculum_stage",
] as const;

function slimEvaluationRecord<T>(rec: T): T {
  if (!rec || typeof rec !== "object") {
    return rec;
  }
  const recObj = rec as Record<string, unknown>;
  if (!("failed_trajectory_summary" in recObj) && !("observation_diagnostics" in recObj)) {
    return rec;
  }
  const trimmedRec = { ...recObj };
  delete trimmedRec.failed_trajectory_summary;
  delete trimmedRec.observation_diagnostics;
  return trimmedRec as T;
}

function stripKeysAndRecords<T extends object>(obj: T, keysToDrop: readonly string[]): T {
  if (!obj || typeof obj !== "object") {
    return obj;
  }
  const slim: Record<string, unknown> = { ...(obj as unknown as Record<string, unknown>) };
  for (const key of keysToDrop) {
    delete slim[key];
  }
  if (Array.isArray(slim.records)) {
    slim.records = slim.records.map((rec) => slimEvaluationRecord(rec));
  }
  return slim as unknown as T;
}

/** Strip unused multi-MB payloads and excess nested metadata from a checkpoint entry. */
export function slimCheckpointEntry(entry: CheckpointEntry): CheckpointEntry {
  if (!entry || typeof entry !== "object") {
    return entry;
  }
  const slim = stripKeysAndRecords(entry, DROPPED_CHECKPOINT_KEYS) as unknown as Record<string, unknown>;

  const rawEnv = slim.environment_config;
  if (rawEnv && typeof rawEnv === "object") {
    const rawEnvRecord = rawEnv as Record<string, unknown>;
    const trimmedEnv: Record<string, unknown> = {};
    for (const key of ALLOWED_ENV_CONFIG_KEYS) {
      if (key in rawEnvRecord && rawEnvRecord[key] !== undefined) {
        trimmedEnv[key] = rawEnvRecord[key];
      }
    }
    slim.environment_config = trimmedEnv;
  }

  if ("reward_config" in slim) {
    const rawReward = slim.reward_config;
    const rawInstincts =
      rawReward && typeof rawReward === "object"
        ? (rawReward as Record<string, unknown>).instincts
        : undefined;
    if (rawInstincts && typeof rawInstincts === "object") {
      const instinctsRecord = rawInstincts as Record<string, unknown>;
      const trimmedInstincts: Record<string, unknown> = {};
      if ("curriculum_stage" in instinctsRecord && instinctsRecord.curriculum_stage !== undefined) {
        trimmedInstincts.curriculum_stage = instinctsRecord.curriculum_stage;
      }
      if (
        "enable_instinct_rewards" in instinctsRecord &&
        instinctsRecord.enable_instinct_rewards !== undefined
      ) {
        trimmedInstincts.enable_instinct_rewards = instinctsRecord.enable_instinct_rewards;
      }
      slim.reward_config = { instincts: trimmedInstincts };
    } else {
      delete slim.reward_config;
    }
  }

  return slim as unknown as CheckpointEntry;
}

/** Strip unused heavy fields from an evaluation summary payload and its per-seed records. */
export function slimEvaluationSummary(summary: EvaluationSummaryPayload): EvaluationSummaryPayload {
  return stripKeysAndRecords(summary, DROPPED_EVAL_SUMMARY_KEYS);
}

/** Trim unused heavy fields across all checkpoint entries and `latest` summary. */
export function slimCheckpointIndex(index: CheckpointIndex): CheckpointIndex {
  if (!index || typeof index !== "object") {
    return index;
  }
  return {
    ...index,
    checkpoints: Array.isArray(index.checkpoints)
      ? index.checkpoints.map((entry) => slimCheckpointEntry(entry))
      : [],
    latest: index.latest
      ? (slimCheckpointEntry(index.latest as unknown as CheckpointEntry) as unknown as EvaluationSummary)
      : null,
  };
}

export type CheckpointIndexLoadResult =
  | { status: "ok"; index: CheckpointIndex; etag: string | null }
  | { status: "not_modified"; etag: string | null }
  | { status: "empty" };

// The checkpoint index grows with every checkpoint (tens of MB on long runs) and is
// served by the same Python process that runs training. Share one in-flight request
// between concurrent callers so a single status change doesn't trigger several
// full downloads that compete with the trainer for CPU. Never retain resolved payloads here.
let checkpointIndexConditionalInFlight: Promise<CheckpointIndexLoadResult> | null = null;

/** Slow fallback refresh for the checkpoint index; normal refreshes are status-driven. */
export const CHECKPOINT_INDEX_SAFETY_REFRESH_MS = 60_000;

export function loadCheckpointIndexConditional(
  ifNoneMatch?: string | null,
): Promise<CheckpointIndexLoadResult> {
  if (isBackendMarkedOffline()) {
    console.debug("[Sheepdog API] Backend server is in offline cooldown; skipping checkpoint index fetch.");
    return Promise.resolve({ status: "empty" });
  }
  if (checkpointIndexConditionalInFlight) {
    return checkpointIndexConditionalInFlight;
  }
  const request = loadCheckpointIndexConditionalUncached(ifNoneMatch).finally(() => {
    if (checkpointIndexConditionalInFlight === request) {
      checkpointIndexConditionalInFlight = null;
    }
  });
  checkpointIndexConditionalInFlight = request;
  return request;
}

async function loadCheckpointIndexConditionalUncached(
  ifNoneMatch?: string | null,
): Promise<CheckpointIndexLoadResult> {
  if (isBackendMarkedOffline()) {
    console.debug("[Sheepdog API] Backend server is in offline cooldown; skipping checkpoint index fetch.");
    return { status: "empty" };
  }
  const hasEtag = Boolean(ifNoneMatch);
  const path = hasEtag
    ? "/generated/checkpoint-index.json"
    : `/generated/checkpoint-index.json?t=${Date.now()}`;
  const requestUrl = API_BASE_URL ? new URL(path, API_BASE_URL) : path;
  const init: RequestInit = hasEtag
    ? { cache: "no-store", headers: { "If-None-Match": ifNoneMatch! } }
    : { cache: "no-store" };

  try {
    const response = await fetch(requestUrl, init);
    clearBackendOffline();

    const getHeader = (name: string): string | null =>
      typeof response.headers?.get === "function" ? response.headers.get(name) ?? null : null;

    if (response.status === 304) {
      return {
        status: "not_modified",
        etag: getHeader("etag") ?? ifNoneMatch ?? null,
      };
    }

    if (response.status === 404 || response.status === 500) {
      console.info(
        `Checkpoint index "/generated/checkpoint-index.json" is not available yet (HTTP ${response.status}). This is expected during startup or if no checkpoints have been saved yet.`
      );
      return { status: "empty" };
    }

    if (!response.ok) {
      const error = new Error(`Failed to fetch ${path}: ${response.status}`) as ApiError;
      error.status = response.status;
      throw error;
    }

    const contentType = getHeader("content-type") ?? "";
    if (contentType.toLowerCase().includes("text/html")) {
      console.info(
        `Checkpoint index request returned HTML (status ${response.status}). Treating as not available yet.`
      );
      return { status: "empty" };
    }

    if (contentType && !contentType.toLowerCase().includes("application/json")) {
      const error = new Error(
        `Expected JSON from ${path}, received ${contentType || "unknown content type"}`
      ) as ApiError;
      error.status = response.status;
      error.contentType = contentType;
      throw error;
    }

    const raw = (await response.json()) as CheckpointIndex;
    return {
      status: "ok",
      index: slimCheckpointIndex(raw),
      etag: getHeader("etag"),
    };
  } catch (error) {
    const fetchError = error as ApiError;
    if (fetchError?.status === 404 || fetchError?.status === 500) {
      return { status: "empty" };
    }
    if (
      error instanceof TypeError ||
      (error instanceof Error && error.name === "TypeError") ||
      fetchError?.isNetworkError ||
      fetchError?.status === 0
    ) {
      markBackendOffline();
      console.debug(
        `Backend server offline or unreachable on port 8000 when fetching checkpoint index (${fetchError?.message || String(error)}). Treating as not available yet.`
      );
      return { status: "empty" };
    }
    throw error;
  }
}

export async function loadCheckpointIndex(): Promise<CheckpointIndex | null> {
  const res = await loadCheckpointIndexConditional(null);
  return res.status === "ok" ? res.index : null;
}

export interface LoadEpisodesOptions {
  afterId?: number;
  beforeId?: number;
  stage?: number;
  runId?: string;
  limit?: number;
  order?: "asc" | "desc";
}

export async function loadTrainingEpisodes(options: LoadEpisodesOptions = {}): Promise<TrainingEpisodesResponse | null> {
  if (isBackendMarkedOffline()) {
    return null;
  }
  const queryParams = new URLSearchParams();
  if (options.afterId !== undefined) queryParams.set("after_id", String(options.afterId));
  if (options.beforeId !== undefined) queryParams.set("before_id", String(options.beforeId));
  if (options.stage !== undefined) queryParams.set("stage", String(options.stage));
  if (options.runId) queryParams.set("run_id", options.runId);
  if (options.limit !== undefined) queryParams.set("limit", String(options.limit));
  if (options.order) queryParams.set("order", options.order);

  const queryString = queryParams.toString();
  const path = `/api/insights/training-episodes${queryString ? `?${queryString}` : ""}`;

  try {
    return await fetchJson<TrainingEpisodesResponse>(path, undefined, API_BASE_URL);
  } catch (error) {
    const fetchErr = error as ApiError;
    if (fetchErr.isNetworkError || fetchErr.status === 0 || !fetchErr.status || error instanceof TypeError) {
      markBackendOffline();
    }
    return null;
  }
}

export async function loadFailedEpisodes(limit: number = 25): Promise<TrainingEpisode[] | null> {
  if (isBackendMarkedOffline()) {
    return null;
  }
  const path = `/api/insights/failed-episodes?limit=${limit}`;
  try {
    const res = await fetchJson<{ episodes: TrainingEpisode[] } | TrainingEpisode[]>(path, undefined, API_BASE_URL);
    if (!res) return null;
    if (Array.isArray(res)) return res;
    if (Array.isArray(res.episodes)) return res.episodes;
    return [];
  } catch (error) {
    const fetchErr = error as ApiError;
    if (fetchErr.isNetworkError || fetchErr.status === 0 || !fetchErr.status || error instanceof TypeError) {
      markBackendOffline();
    }
    return null;
  }
}

export async function loadRecentEvaluations(
  limit: number = 5,
  stage?: number
): Promise<EvaluationSummaryPayload[] | null> {
  if (isBackendMarkedOffline()) {
    return null;
  }
  const query = new URLSearchParams();
  query.set("limit", String(limit));
  if (stage !== undefined) {
    query.set("stage", String(stage));
  }
  const path = `/api/insights/evaluations?${query.toString()}`;
  try {
    const res = await fetchJson<{ evaluations: EvaluationSummaryPayload[] }>(path, undefined, API_BASE_URL);
    return (res?.evaluations ?? []).map((entry) => slimEvaluationSummary(entry));
  } catch (error) {
    const fetchErr = error as ApiError;
    if (fetchErr.isNetworkError || fetchErr.status === 0 || !fetchErr.status || error instanceof TypeError) {
      markBackendOffline();
    }
    return null;
  }
}

export async function loadSessionEvents(): Promise<SessionEventsResponse | null> {
  try {
    return await fetchJson<SessionEventsResponse>("/api/insights/session-events", undefined, API_BASE_URL);
  } catch {
    return null;
  }
}

const replayInFlight = new Map<string, Promise<ReplayBundle>>();
const replayByIdInFlight = new Map<string, Promise<ReplayBundle | null>>();

export function loadReplay(path: string): Promise<ReplayBundle> {
  if (!path || !path.trim()) {
    return Promise.reject(new Error("No replay path specified."));
  }
  const cleanPath = path.replace(/\\/g, "/");
  const existing = replayInFlight.get(cleanPath);
  if (existing) {
    return existing;
  }
  const request = (async (): Promise<ReplayBundle> => {
    const filename = cleanPath.split("/").pop() || "";
    const replayId = filename.replace(/\.json(\.gz)?$/, "");
    if (
      replayId &&
      (cleanPath.includes("evaluations") ||
        cleanPath.includes("replays") ||
        cleanPath.startsWith("artifacts/"))
    ) {
      const fromApi = await fetchReplayById(replayId);
      if (fromApi) return fromApi;
    }
    return fetchJson<ReplayBundle>(path, undefined, API_BASE_URL);
  })().finally(() => {
    if (replayInFlight.get(cleanPath) === request) {
      replayInFlight.delete(cleanPath);
    }
  });
  replayInFlight.set(cleanPath, request);
  return request;
}

export function fetchReplayById(replayId: string): Promise<ReplayBundle | null> {
  const cleanId = replayId.trim();
  const existing = replayByIdInFlight.get(cleanId);
  if (existing) {
    return existing;
  }
  const request = (async (): Promise<ReplayBundle | null> => {
    try {
      return await fetchJson<ReplayBundle>(`/api/replays/${cleanId}`, undefined, API_BASE_URL);
    } catch {
      return null;
    }
  })().finally(() => {
    if (replayByIdInFlight.get(cleanId) === request) {
      replayByIdInFlight.delete(cleanId);
    }
  });
  replayByIdInFlight.set(cleanId, request);
  return request;
}

export async function pinEvaluation(
  evaluationId: string,
  pinned: boolean = true
): Promise<{ success: boolean; evaluation_id: string; pinned: boolean } | null> {
  try {
    return await fetchJson<{ success: boolean; evaluation_id: string; pinned: boolean }>(
      "/api/evaluations/pin",
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ evaluation_id: evaluationId, pinned }),
      },
      API_BASE_URL
    );
  } catch {
    return null;
  }
}

export async function fetchCapturePolicy(): Promise<CapturePolicyConfig | null> {
  try {
    return await fetchJson<CapturePolicyConfig>("/api/training/capture-policy", undefined, API_BASE_URL);
  } catch {
    return null;
  }
}

export async function updateCapturePolicy(params: Record<string, unknown>): Promise<CapturePolicyConfig | null> {
  try {
    return await fetchJson<CapturePolicyConfig>("/api/training/capture-policy", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(params),
    }, API_BASE_URL);
  } catch {
    return null;
  }
}

export async function reproduceEpisode(episodeId: number | string): Promise<ReplayBundle | null> {
  try {
    return await fetchJson<ReplayBundle>(`/api/episodes/${episodeId}/reproduce`, {
      method: "POST",
    }, API_BASE_URL);
  } catch {
    return null;
  }
}

export async function loadTrainingStatus(): Promise<TrainingStatus> {
  return fetchJson<TrainingStatus>("/api/training/status", undefined, API_BASE_URL);
}

export async function loadTrainingStatusSafe(): Promise<{ status: TrainingStatus | null; isOffline: boolean }> {
  try {
    const status = await loadTrainingStatus();
    return { status, isOffline: false };
  } catch (err) {
    const fetchErr = err as ApiError;
    if (fetchErr.isNetworkError || fetchErr.status === 0) {
      return { status: null, isOffline: true };
    }
    throw err;
  }
}

export async function startTraining(request: TrainingStartRequest): Promise<TrainingStatus> {
  return fetchJson<TrainingStatus>("/api/training/start", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(request),
  }, API_BASE_URL);
}

export async function loadStageTrainingReport(
  stage: number,
  runId?: string | null,
  signal?: AbortSignal,
  journey?: string,
): Promise<StageTrainingReport> {
  const query = new URLSearchParams({ stage: String(stage) });
  if (runId) query.set("run_id", runId);
  if (journey) query.set("journey", journey);
  return fetchJson<StageTrainingReport>(`/api/insights/stage-report?${query}`, { signal }, API_BASE_URL);
}

export async function pauseTraining(): Promise<TrainingStatus> {
  return fetchJson<TrainingStatus>("/api/training/pause", {
    method: "POST",
  }, API_BASE_URL);
}

export async function stopTraining(): Promise<TrainingStatus> {
  return fetchJson<TrainingStatus>("/api/training/stop", {
    method: "POST",
  }, API_BASE_URL);
}

export async function clearTraining(): Promise<TrainingStatus> {
  return fetchJson<TrainingStatus>("/api/training/clear", {
    method: "POST",
  }, API_BASE_URL);
}

export async function resetTraining(): Promise<TrainingStatus> {
  return clearTraining();
}

export async function resetJourneyTraining(): Promise<TrainingStatus> {
  return fetchJson<TrainingStatus>("/api/training/reset-journey", {
    method: "POST",
  }, API_BASE_URL);
}

export async function rewindTraining(stage: number): Promise<TrainingStatus> {
  return fetchJson<TrainingStatus>("/api/training/rewind", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ stage }),
  }, API_BASE_URL);
}

export async function startRemediationFork(targetStage = 9, canaryEpisodes = 20): Promise<TrainingStatus> {
  return fetchJson<TrainingStatus>("/api/training/remediation", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ target_stage: targetStage, canary_episodes: canaryEpisodes }),
  }, API_BASE_URL);
}

export async function crossStageFork(request: {
  target_stage: number;
  starting_model_source: string;
  source_checkpoint_id?: string;
}): Promise<TrainingStatus> {
  return fetchJson<TrainingStatus>("/api/training/cross-stage-fork", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(request),
  }, API_BASE_URL);
}

export async function runReplay(request: ReplayRunRequest): Promise<ReplayBundle> {
  return fetchJson<ReplayBundle>("/api/replay/run", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(request),
  }, API_BASE_URL);
}

export const runLiveReplay = runReplay;

export async function loadEffectiveConfig(): Promise<Record<string, unknown>> {
  return fetchJson<Record<string, unknown>>("/api/config", undefined, API_BASE_URL);
}

export async function loadConfigHistory(): Promise<ConfigHistory> {
  return fetchJson<ConfigHistory>("/api/config/history", undefined, API_BASE_URL);
}

export async function saveConfigRevision(
  payload: Omit<ConfigRevision, "id" | "timestamp">,
): Promise<ConfigHistory> {
  return fetchJson<ConfigHistory>("/api/config/history", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  }, API_BASE_URL);
}

export async function loadHyperparams(): Promise<UserHyperparams> {
  return fetchJson<UserHyperparams>("/api/config/hyperparams", undefined, API_BASE_URL);
}

export async function loadTrainingDiagnostics(checkpointId?: string, episode?: number): Promise<DiagnosticsResponse> {
  let url = "/api/training/diagnostics";
  const params: string[] = [];
  if (checkpointId) {
    params.push(`checkpoint_id=${encodeURIComponent(checkpointId)}`);
  }
  if (episode !== undefined) {
    params.push(`episode=${episode}`);
  }
  if (params.length > 0) {
    url += "?" + params.join("&");
  }
  return fetchJson<DiagnosticsResponse>(url, undefined, API_BASE_URL);
}

export async function loadNetworkTopology(): Promise<NetworkTopologyInfo> {
  return fetchJson<NetworkTopologyInfo>("/api/network/topology", undefined, API_BASE_URL);
}

export async function saveHyperparams(payload: UserHyperparams): Promise<UserHyperparams> {
  return fetchJson<UserHyperparams>("/api/config/hyperparams", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  }, API_BASE_URL);
}

/** Load scenarios and evaluation results from the training API (source of truth). */
export async function loadScenarioIndex(): Promise<ScenarioIndex> {
  return fetchJson<ScenarioIndex>("/api/scenarios", undefined, API_BASE_URL);
}

export async function saveScenario(payload: {
  name: string;
  seed: number;
  snapshot: ReplayBundle["final_snapshot"];
  sheep_personality_strength?: number;
  description?: string;
  snapshot_source?: "initial" | "final";
}): Promise<SavedScenario> {
  return fetchJson<SavedScenario>("/api/scenarios", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  }, API_BASE_URL);
}

export type CheckpointMode = "latest" | "global_best" | "scenario_best" | "specific";

export interface ScenarioCheckpointRequest {
  checkpoint_mode: CheckpointMode;
  checkpoint_episode?: number;
  policy_mode?: string;
  trainer_type?: string;
  policy_type?: string;
  effective_config?: Record<string, unknown>;
}

export async function evaluateScenario(
  scenarioId: string,
  request: ScenarioCheckpointRequest,
): Promise<{ checkpoint_episode: number; result: ScenarioRunResult; index: ScenarioIndex }> {
  return fetchJson(`/api/scenarios/${scenarioId}/evaluate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
  }, API_BASE_URL);
}

export async function replayScenario(
  scenarioId: string,
  request: ScenarioCheckpointRequest,
): Promise<ReplayBundle> {
  return fetchJson<ReplayBundle>(`/api/scenarios/${scenarioId}/replay`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
  }, API_BASE_URL);
}

export async function loadTrainingHistory(): Promise<TelemetryRecord[]> {
  return fetchJson<TelemetryRecord[]>("/api/training/history", undefined, API_BASE_URL);
}

export async function shutdownApp(): Promise<{ status: string }> {
  return fetchJson<{ status: string }>("/api/shutdown", {
    method: "POST",
  }, API_BASE_URL);
}

export async function loadCheckpointDetails(
  episode: number | null,
  journey?: string,
  checkpointId?: string,
): Promise<Record<string, unknown>> {
  let url = "/api/checkpoint/details?";
  const params: string[] = [];
  if (checkpointId) {
    params.push(`checkpoint_id=${encodeURIComponent(checkpointId)}`);
  }
  if (episode !== null && episode !== undefined) {
    params.push(`episode=${episode}`);
  }
  if (journey) {
    params.push(`journey=${encodeURIComponent(journey)}`);
  }
  url += params.join("&");
  return fetchJson<Record<string, unknown>>(url, undefined, API_BASE_URL);
}

export async function restoreCheckpoint(
  episode: number | null,
  journey?: string,
  checkpointId?: string,
): Promise<{ status: string; message: string }> {
  return fetchJson<{ status: string; message: string }>("/api/training/restore", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ episode, journey, checkpoint_id: checkpointId }),
  }, API_BASE_URL);
}

export async function forkCheckpoint(
  episode: number | null,
  journey?: string,
  hyperparams?: Record<string, unknown>,
  checkpointId?: string,
): Promise<{ status: string; message: string; run_id: string }> {
  return fetchJson<{ status: string; message: string; run_id: string }>("/api/training/fork", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ episode, journey, hyperparams, checkpoint_id: checkpointId }),
  }, API_BASE_URL);
}

export async function archiveActiveRun(): Promise<{ status: string; archive_dir: string }> {
  return fetchJson<{ status: string; archive_dir: string }>("/api/training/archive-active", {
    method: "POST",
  }, API_BASE_URL);
}

export async function loadConfigEditable(): Promise<Record<string, unknown>> {
  return fetchJson<Record<string, unknown>>("/api/config/editable", undefined, API_BASE_URL);
}

export async function loadConfigActive(): Promise<Record<string, unknown>> {
  return fetchJson<Record<string, unknown>>("/api/config/active", undefined, API_BASE_URL);
}

export async function loadConfigNextRun(): Promise<Record<string, unknown>> {
  return fetchJson<Record<string, unknown>>("/api/config/next-run", undefined, API_BASE_URL);
}

export async function loadStageDiagnostics(
  stage?: number,
  runId?: string,
): Promise<StageBottleneckReport | null> {
  const queryParts: string[] = [];
  if (stage !== undefined && stage !== null) {
    queryParts.push(`stage=${stage}`);
  }
  if (runId) {
    queryParts.push(`run_id=${encodeURIComponent(runId)}`);
  }
  const query = queryParts.length > 0 ? `?${queryParts.join("&")}` : "";
  try {
    return await fetchJson<StageBottleneckReport>(`/api/stage-diagnostics${query}`, undefined, API_BASE_URL);
  } catch (err) {
    console.warn("[Sheepdog API] Failed to load stage diagnostics:", err);
    return null;
  }
}

export async function loadStageHealth(
  stage?: number,
  force?: boolean,
): Promise<StageHealthSummary | null> {
  const queryParts: string[] = [];
  if (stage !== undefined && stage !== null) {
    queryParts.push(`stage=${stage}`);
  }
  if (force) {
    queryParts.push("force=1");
  }
  const query = queryParts.length > 0 ? `?${queryParts.join("&")}` : "";
  try {
    return await fetchJson<StageHealthSummary>(`/api/stage-health${query}`, undefined, API_BASE_URL);
  } catch (err) {
    console.warn("[Sheepdog API] Failed to load stage health:", err);
    return null;
  }
}

export async function loadEarlyWarnings(
  stage?: number,
  force?: boolean,
): Promise<EarlyWarningsPayload | null> {
  const queryParts: string[] = [];
  if (stage !== undefined && stage !== null) {
    queryParts.push(`stage=${stage}`);
  }
  if (force) {
    queryParts.push("force=1");
  }
  const query = queryParts.length > 0 ? `?${queryParts.join("&")}` : "";
  try {
    return await fetchJson<EarlyWarningsPayload>(`/api/early-warnings${query}`, undefined, API_BASE_URL);
  } catch (err) {
    console.warn("[Sheepdog API] Failed to load early warnings:", err);
    return null;
  }
}

export async function acknowledgeEarlyWarning(
  warningId: string,
  note = "",
): Promise<{ success: boolean; warning_id: string } | null> {
  try {
    return await fetchJson<{ success: boolean; warning_id: string }>(
      "/api/early-warnings/acknowledge",
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ warning_id: warningId, note }),
      },
      API_BASE_URL,
    );
  } catch (err) {
    console.warn("[Sheepdog API] Failed to acknowledge early warning:", err);
    return null;
  }
}

const MAX_TEMPORAL_DIAGNOSTIC_CACHE_ENTRIES = 4;
const temporalDiagnosticPayloadCache = new Map<string, Promise<TemporalDiagnosticPayload>>();
const resolvedTemporalDiagnosticPayloads = new Map<string, TemporalDiagnosticPayload>();

function setBoundedMapEntry<K, V>(map: Map<K, V>, key: K, value: V, maxEntries: number): void {
  if (map.has(key)) {
    map.delete(key);
  } else if (map.size >= maxEntries) {
    const oldestKey = map.keys().next().value;
    if (oldestKey !== undefined) {
      map.delete(oldestKey);
    }
  }
  map.set(key, value);
}

export function clearTemporalDiagnosticCache(): void {
  temporalDiagnosticPayloadCache.clear();
  resolvedTemporalDiagnosticPayloads.clear();
}

export function hasCachedTemporalDiagnosticAnalysis(
  replayId: string,
  includeReplay = true,
): boolean {
  const cleanId = replayId.trim();
  const cacheKey = `${cleanId}::includeReplay=${includeReplay ? "1" : "0"}`;
  return resolvedTemporalDiagnosticPayloads.has(cacheKey);
}

export async function loadTemporalDiagnosticExamples(): Promise<TemporalDiagnosticExample[]> {
  try {
    const res = await fetchJson<{ examples: TemporalDiagnosticExample[] }>(
      "/api/temporal-diagnostics/examples",
      undefined,
      API_BASE_URL,
    );
    return Array.isArray(res.examples) ? res.examples : [];
  } catch (err) {
    console.warn("[Sheepdog API] Failed to load temporal diagnostic examples:", err);
    return [];
  }
}

export async function loadTemporalDiagnosticAnalysis(
  replayId: string,
  options?: { includeReplay?: boolean; forceRefresh?: boolean },
): Promise<TemporalDiagnosticPayload> {
  const cleanId = replayId.trim();
  const includeReplay = Boolean(options?.includeReplay);
  const cacheKey = `${cleanId}::includeReplay=${includeReplay ? "1" : "0"}`;
  if (!options?.forceRefresh) {
    const cached = temporalDiagnosticPayloadCache.get(cacheKey);
    if (cached) {
      setBoundedMapEntry(temporalDiagnosticPayloadCache, cacheKey, cached, MAX_TEMPORAL_DIAGNOSTIC_CACHE_ENTRIES);
      const resolved = resolvedTemporalDiagnosticPayloads.get(cacheKey);
      if (resolved) {
        setBoundedMapEntry(resolvedTemporalDiagnosticPayloads, cacheKey, resolved, MAX_TEMPORAL_DIAGNOSTIC_CACHE_ENTRIES);
      }
      return cached;
    }
  }
  const query = includeReplay ? "?include_replay=1" : "";
  const requestPromise = fetchJson<TemporalDiagnosticPayload>(
    `/api/replays/${encodeURIComponent(cleanId)}/temporal-analysis${query}`,
    undefined,
    API_BASE_URL,
  )
    .then((payload) => {
      setBoundedMapEntry(resolvedTemporalDiagnosticPayloads, cacheKey, payload, MAX_TEMPORAL_DIAGNOSTIC_CACHE_ENTRIES);
      return payload;
    })
    .catch((err) => {
      temporalDiagnosticPayloadCache.delete(cacheKey);
      throw err;
    });
  setBoundedMapEntry(temporalDiagnosticPayloadCache, cacheKey, requestPromise, MAX_TEMPORAL_DIAGNOSTIC_CACHE_ENTRIES);
  return requestPromise;
}




export async function loadStageTimings(): Promise<StageTimingsResponse> {
  return fetchJson<StageTimingsResponse>("/api/stage-timings", undefined, API_BASE_URL);
}

export async function loadModelHistory(params?: {
  passId?: string;
  stage?: number | null;
  limit?: number;
}): Promise<ModelHistoryResponse> {
  const query = new URLSearchParams();
  if (params?.passId) query.set("pass_id", params.passId);
  if (params?.stage !== undefined && params?.stage !== null) query.set("stage", String(params.stage));
  if (params?.limit !== undefined) query.set("limit", String(params.limit));
  const qs = query.toString();
  return fetchJson<ModelHistoryResponse>(
    `/api/model-history${qs ? `?${qs}` : ""}`,
    undefined,
    API_BASE_URL,
  );
}

export async function loadModelLineage(passId: string): Promise<ModelLineageResponse> {
  const query = new URLSearchParams({ pass_id: passId });
  return fetchJson<ModelLineageResponse>(
    `/api/model-history/lineage?${query.toString()}`,
    undefined,
    API_BASE_URL,
  );
}

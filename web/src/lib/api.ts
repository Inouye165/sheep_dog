import type {
  CheckpointIndex,
  ConfigHistory,
  ConfigRevision,
  NetworkTopologyInfo,
  ReplayBundle,
  ReplayRunRequest,
  SavedScenario,
  ScenarioIndex,
  ScenarioRunResult,
  TrainingStartRequest,
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

// The checkpoint index grows with every checkpoint (tens of MB on long runs) and is
// served by the same Python process that runs training. Share one in-flight request
// between concurrent callers so a single status change doesn't trigger several
// full downloads that compete with the trainer for CPU.
let checkpointIndexInFlight: Promise<CheckpointIndex | null> | null = null;

/** Slow fallback refresh for the checkpoint index; normal refreshes are status-driven. */
export const CHECKPOINT_INDEX_SAFETY_REFRESH_MS = 60_000;

export function loadCheckpointIndex(): Promise<CheckpointIndex | null> {
  if (checkpointIndexInFlight) {
    return checkpointIndexInFlight;
  }
  const request = loadCheckpointIndexUncached();
  checkpointIndexInFlight = request;
  const clear = () => {
    if (checkpointIndexInFlight === request) {
      checkpointIndexInFlight = null;
    }
  };
  request.then(clear, clear);
  return request;
}

async function loadCheckpointIndexUncached(): Promise<CheckpointIndex | null> {
  if (isBackendMarkedOffline()) {
    console.debug("[Sheepdog API] Backend server is in offline cooldown; skipping checkpoint index fetch.");
    return null;
  }
  try {
    return await fetchJson<CheckpointIndex>(`/generated/checkpoint-index.json?t=${Date.now()}`, undefined, API_BASE_URL);
  } catch (error) {
    const fetchError = error as ApiError;
    // Treat 404 and 500 (Vite serves 500 when the file doesn't exist yet)
    // as "not available yet" rather than a hard error.
    if (fetchError.status === 404 || fetchError.status === 500) {
      console.info(
        `Checkpoint index "/generated/checkpoint-index.json" is not available yet (HTTP ${fetchError.status}). This is expected during startup or if no checkpoints have been saved yet.`
      );
      return null;
    }
    // Also treat network/connection errors (status 0 / ERR_CONNECTION_REFUSED) as not available yet.
    if (fetchError.isNetworkError || fetchError.status === 0 || !fetchError.status || error instanceof TypeError) {
      markBackendOffline();
      console.debug(
        `Backend server offline or unreachable on port 8000 when fetching checkpoint index (${fetchError.message || String(error)}). Treating as not available yet.`
      );
      return null;
    }
    if (fetchError.status === 200 && fetchError.contentType?.toLowerCase().includes("text/html")) {
      console.info(
        `Checkpoint index request returned HTML (status 200). Treating as not available yet.`
      );
      return null;
    }
    throw error;
  }
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
    return res?.evaluations ?? [];
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

export async function loadReplay(path: string): Promise<ReplayBundle> {
  if (!path || !path.trim()) {
    throw new Error("No replay path specified.");
  }
  const cleanPath = path.replace(/\\/g, "/");
  const filename = cleanPath.split("/").pop() || "";
  const replayId = filename.replace(/\.json(\.gz)?$/, "");
  if (replayId && (cleanPath.includes("evaluations") || cleanPath.includes("replays") || cleanPath.startsWith("artifacts/"))) {
    const fromApi = await fetchReplayById(replayId);
    if (fromApi) return fromApi;
  }
  return fetchJson<ReplayBundle>(path, undefined, API_BASE_URL);
}

export async function fetchReplayById(replayId: string): Promise<ReplayBundle | null> {
  try {
    return await fetchJson<ReplayBundle>(`/api/replays/${replayId}`, undefined, API_BASE_URL);
  } catch {
    return null;
  }
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

const MAX_TEMPORAL_DIAGNOSTIC_CACHE_ENTRIES = 12;
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




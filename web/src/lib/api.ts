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
} from "../state/types";

const API_BASE_URL = "http://127.0.0.1:8000";

function isJsonResponse(response: Response): boolean {
  const contentType = response.headers.get("content-type") ?? "";
  return contentType.toLowerCase().includes("application/json");
}

async function fetchJson<T>(path: string, init?: RequestInit, baseUrl?: string): Promise<T> {
  const requestUrl = baseUrl ? new URL(path, baseUrl) : path;
  const response = await fetch(requestUrl, { cache: "no-store", ...init });
  if (!response.ok) {
    let errMsg = `Failed to fetch ${path}: ${response.status}`;
    try {
      if (isJsonResponse(response)) {
        const errJson = await response.clone().json();
        if (errJson && typeof errJson === "object" && "error" in errJson) {
          errMsg = String(errJson.error);
        }
      }
    } catch {
      // ignore
    }
    const error = new Error(errMsg) as Error & { status?: number };
    error.status = response.status;
    throw error;
  }

  if (!isJsonResponse(response)) {
    const error = new Error(`Expected JSON from ${path}, received ${response.headers.get("content-type") ?? "unknown content type"}`) as Error & { status?: number; contentType?: string };
    error.status = response.status;
    error.contentType = response.headers.get("content-type") ?? undefined;
    throw error;
  }

  return (await response.json()) as T;
}

export async function loadCheckpointIndex(): Promise<CheckpointIndex | null> {
  try {
    return await fetchJson<CheckpointIndex>("/generated/checkpoint-index.json", undefined, API_BASE_URL);
  } catch (error) {
    const fetchError = error as { status?: number; contentType?: string; message?: string };
    // Treat 404 and 500 (Vite serves 500 when the file doesn't exist yet)
    // as "not available yet" rather than a hard error.
    if (fetchError.status === 404 || fetchError.status === 500) {
      console.info(
        `Checkpoint index "/generated/checkpoint-index.json" is not available yet (HTTP ${fetchError.status}). This is expected during startup or if no checkpoints have been saved yet.`
      );
      return null;
    }
    // Also treat network/connection errors (e.g. TypeError: Failed to fetch) as not available yet.
    if (error instanceof TypeError || !fetchError.status) {
      console.warn(
        `Network or connection error fetching checkpoint index: ${fetchError.message || String(error)}. Treating as not available yet.`
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

export async function loadReplay(path: string): Promise<ReplayBundle> {
  if (!path || !path.trim()) {
    throw new Error("No replay path specified.");
  }
  return fetchJson<ReplayBundle>(path, undefined, API_BASE_URL);
}

export async function loadTrainingStatus(): Promise<TrainingStatus> {
  return fetchJson<TrainingStatus>("/api/training/status", undefined, API_BASE_URL);
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

export async function runReplay(request: ReplayRunRequest): Promise<ReplayBundle> {
  return fetchJson<ReplayBundle>("/api/replay/run", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(request),
  }, API_BASE_URL);
}

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

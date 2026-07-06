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
    const error = new Error(`Failed to fetch ${path}: ${response.status}`) as Error & { status?: number };
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
    return await fetchJson<CheckpointIndex>("/generated/checkpoint-index.json");
  } catch (error) {
    const fetchError = error as { status?: number; contentType?: string };
    // Treat 404 and 500 (Vite serves 500 when the file doesn't exist yet)
    // as "not available yet" rather than a hard error.
    if (fetchError.status === 404 || fetchError.status === 500) {
      return null;
    }
    if (fetchError.status === 200 && fetchError.contentType?.toLowerCase().includes("text/html")) {
      return null;
    }
    throw error;
  }
}

export async function loadReplay(path: string): Promise<ReplayBundle> {
  return fetchJson<ReplayBundle>(path);
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

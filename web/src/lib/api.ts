import type { CheckpointIndex, ConfigHistory, ConfigRevision, ReplayBundle, ReplayRunRequest, TrainingStartRequest, TrainingStatus, UserHyperparams } from "../state/types";

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
    if (fetchError.status === 404) {
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

export async function clearTraining(): Promise<TrainingStatus> {
  return fetchJson<TrainingStatus>("/api/training/clear", {
    method: "POST",
  }, API_BASE_URL);
}

export async function resetTraining(): Promise<TrainingStatus> {
  return clearTraining();
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

export async function saveHyperparams(payload: UserHyperparams): Promise<UserHyperparams> {
  return fetchJson<UserHyperparams>("/api/config/hyperparams", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  }, API_BASE_URL);
}

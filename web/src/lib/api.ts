import type { CheckpointIndex, ReplayBundle, ReplayRunRequest, TrainingStartRequest, TrainingStatus } from "../state/types";

const API_BASE_URL = "http://127.0.0.1:8000";

async function fetchJson<T>(path: string, init?: RequestInit, baseUrl?: string): Promise<T> {
  const requestUrl = baseUrl ? new URL(path, baseUrl) : path;
  const response = await fetch(requestUrl, { cache: "no-store", ...init });
  if (!response.ok) {
    const error = new Error(`Failed to fetch ${path}: ${response.status}`) as Error & { status?: number };
    error.status = response.status;
    throw error;
  }
  return (await response.json()) as T;
}

export async function loadCheckpointIndex(): Promise<CheckpointIndex | null> {
  try {
    return await fetchJson<CheckpointIndex>("/generated/checkpoint-index.json");
  } catch (error) {
    if ((error as { status?: number }).status === 404) {
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

export async function resetTraining(): Promise<TrainingStatus> {
  return fetchJson<TrainingStatus>("/api/training/reset", {
    method: "POST",
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


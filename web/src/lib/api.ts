import type { CheckpointIndex, ReplayBundle, TrainingStartRequest, TrainingStatus } from "../state/types";

const API_BASE_URL = "http://127.0.0.1:8000";

async function fetchJson<T>(path: string, init?: RequestInit, baseUrl?: string): Promise<T> {
  const requestUrl = baseUrl ? new URL(path, baseUrl) : path;
  const response = await fetch(requestUrl, { cache: "no-store", ...init });
  if (!response.ok) {
    throw new Error(`Failed to fetch ${path}: ${response.status}`);
  }
  return (await response.json()) as T;
}

export async function loadCheckpointIndex(): Promise<CheckpointIndex> {
  return fetchJson<CheckpointIndex>("/generated/checkpoint-index.json");
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


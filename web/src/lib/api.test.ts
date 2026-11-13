import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import {
  checkBackendHealth,
  isBackendStartingError,
  loadCheckpointIndex,
  loadTrainingStatus,
  loadTrainingStatusSafe,
  ApiError,
} from "./api";

describe("api client network error and health check handling", () => {
  const originalFetch = global.fetch;

  beforeEach(() => {
    vi.restoreAllMocks();
  });

  afterEach(() => {
    global.fetch = originalFetch;
  });

  it("checkBackendHealth returns ok=true when /api/health responds with status ok", async () => {
    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      headers: new Headers({ "content-type": "application/json" }),
      json: async () => ({ status: "ok", service: "sheepdog-api" }),
    } as Response);

    const health = await checkBackendHealth();
    expect(health.ok).toBe(true);
    expect(health.service).toBe("sheepdog-api");
  });

  it("checkBackendHealth returns ok=false when network request fails (server offline)", async () => {
    global.fetch = vi.fn().mockRejectedValue(new TypeError("Failed to fetch"));

    const health = await checkBackendHealth();
    expect(health.ok).toBe(false);
    expect(health.error).toContain("Failed to fetch");
  });

  it("loadCheckpointIndex returns null gracefully on network error (connection refused)", async () => {
    global.fetch = vi.fn().mockRejectedValue(new TypeError("Failed to fetch"));

    const index = await loadCheckpointIndex();
    expect(index).toBeNull();
  });

  it("loadTrainingStatusSafe returns isOffline=true on network error", async () => {
    global.fetch = vi.fn().mockRejectedValue(new TypeError("Failed to fetch"));

    const res = await loadTrainingStatusSafe();
    expect(res.isOffline).toBe(true);
    expect(res.status).toBeNull();
  });

  it("recognizes a 503 BACKEND_STARTING response as restoring, not offline", async () => {
    const body = { code: "BACKEND_STARTING", phase: "restoring", error: "Restoring training state (12s elapsed)" };
    global.fetch = vi.fn().mockResolvedValue({
      ok: false,
      status: 503,
      headers: new Headers({ "content-type": "application/json" }),
      clone: () => ({ json: async () => body }),
      json: async () => body,
    } as unknown as Response);

    const error = await loadTrainingStatus().catch((err: unknown) => err as ApiError);
    expect(isBackendStartingError(error)).toBe(true);
    expect((error as ApiError).code).toBe("BACKEND_STARTING");
    expect((error as ApiError).message).toContain("Restoring training state");

    const safe = await loadTrainingStatusSafe().catch((err: unknown) => err as ApiError);
    expect(isBackendStartingError(safe)).toBe(true);
  });

  it("does not treat other failures as backend startup", () => {
    const generic = Object.assign(new Error("boom"), { status: 503 }) as ApiError;
    const network = Object.assign(new Error("refused"), { status: 0, isNetworkError: true }) as ApiError;
    expect(isBackendStartingError(generic)).toBe(false);
    expect(isBackendStartingError(network)).toBe(false);
    expect(isBackendStartingError(null)).toBe(false);
  });
});

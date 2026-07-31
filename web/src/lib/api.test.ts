import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import {
  checkBackendHealth,
  loadCheckpointIndex,
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
});

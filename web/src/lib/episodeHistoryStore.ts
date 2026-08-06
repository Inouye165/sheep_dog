import type { EpisodeRecord, EpisodeOutcome, ReplayFrame, ReplaySnapshot, AgentSnapshot, TrainingEpisode } from "../state/types";

export const MAX_EPISODE_HISTORY_BUFFER_SIZE = 50;

/**
 * Format label according to specification:
 * `Episode <ID> - <Timestamp> | <Total Moves> moves | <Outcome>`
 * Example: "Episode 1405 - 07:38:31 | 1040 moves | TIMEOUT"
 */
export function formatEpisodeLabel(record: EpisodeRecord): string {
  let outcomeText = record.outcome_label;
  if (!outcomeText) {
    if (record.outcome === "win") outcomeText = "WIN";
    else if (record.outcome === "timeout") outcomeText = "TIMEOUT";
    else outcomeText = "LOSS";
  }
  return `Episode ${record.episode_id} - ${record.timestamp} | ${record.total_moves} moves | ${outcomeText}`;
}

export function formatFailedEpisodeLabel(record: EpisodeRecord): string {
  const epId = record.episode_id;
  const stage = record.stage !== null && record.stage !== undefined ? record.stage : "Unknown";
  const seed = record.seed !== undefined && record.seed !== null ? record.seed : "Unknown";
  return `Episode ${epId} — Stage ${stage} — Seed ${seed} — Failed`;
}

function formatTimeString(isoString: string): string {
  try {
    const d = new Date(isoString);
    if (isNaN(d.getTime())) return isoString;
    const hours = String(d.getHours()).padStart(2, "0");
    const minutes = String(d.getMinutes()).padStart(2, "0");
    const seconds = String(d.getSeconds()).padStart(2, "0");
    return `${hours}:${minutes}:${seconds}`;
  } catch {
    return isoString;
  }
}

/**
 * Convert backend TrainingEpisode object into frontend EpisodeRecord.
 * Represented honestly as a scalar telemetry summary (no synthetic trajectory frames).
 */
export function convertTrainingEpisodeToRecord(item: TrainingEpisode): EpisodeRecord {
  const epId = item.global_environment_episode || item.id;
  const outcome: EpisodeOutcome = item.success ? "win" : item.timeout ? "timeout" : "loss";
  const outcomeLabel = item.result || (item.success ? "SUCCESS" : item.timeout ? "TIMEOUT" : "STOPPED");
  const totalMoves = item.steps ?? 0;
  const stage = item.curriculum_stage !== undefined && item.curriculum_stage !== null ? item.curriculum_stage : null;
  const seed = item.seed !== undefined && item.seed !== null ? item.seed : undefined;
  const isReplayAvail = Boolean(item.replay_available ?? (item as any).replayAvailable);
  const repId = item.replay_id || (item as any).replayId || null;
  const replaySrc = (item.replay_source as any) || (item as any).replaySource || (isReplayAvail ? "training-diagnostic" : null);
  const capReason = item.capture_reason || (item as any).captureReason || null;
  const capStatus = (item.capture_status as any) || (item as any).captureStatus || (isReplayAvail ? "available" : "not_requested");

  return {
    episode_id: epId,
    timestamp: formatTimeString(item.completed_at),
    stage,
    outcome,
    outcome_label: outcomeLabel,
    total_moves: totalMoves,
    reward: item.reward,
    sheep_penned: item.sheep_penned ?? 0,
    total_sheep: item.total_sheep ?? 4,
    seed,
    checkpoint_id: item.checkpoint_id ?? null,
    policy_name: "PPO_Agent",
    replayAvailable: isReplayAvail,
    replay_id: repId,
    replaySource: replaySrc,
    replayUrl: repId ? `/api/replays/${repId}` : null,
    capture_reason: capReason,
    capture_status: capStatus,
  };
}

/**
 * Test fixture generator for generating mock records strictly in test environments.
 * MUST NOT run automatically in production.
 */
export function createPreseededEpisodesForTesting(): EpisodeRecord[] {
  const records: EpisodeRecord[] = [];
  const baseTime = new Date("2026-08-05T07:30:00Z").getTime();

  for (let i = 50; i >= 1; i--) {
    const epId = 1350 + i;
    const epTime = new Date(baseTime + i * 10000);
    const hours = String(epTime.getHours()).padStart(2, "0");
    const minutes = String(epTime.getMinutes()).padStart(2, "0");
    const seconds = String(epTime.getSeconds()).padStart(2, "0");
    const timestampStr = `${hours}:${minutes}:${seconds}`;

    let stage: number | null = 8;
    if (i % 5 === 0) stage = 7;
    else if (i % 7 === 0) stage = 9;

    let outcome: EpisodeOutcome = "win";
    let outcomeLabel = "SUCCESS";
    let moves = 25 + (i * 3) % 25;

    if (i % 3 === 0) {
      outcome = "timeout";
      outcomeLabel = "TIMEOUT";
      moves = 50;
    } else if (i % 5 === 2 || i === 8 || i === 18 || i === 28 || i === 38 || i === 48) {
      outcome = "loss";
      outcomeLabel = "STOPPED";
      moves = 18 + (i % 12);
    }

    const seed = 1000 + epId;

    records.push({
      episode_id: epId,
      timestamp: timestampStr,
      stage,
      outcome,
      outcome_label: outcomeLabel,
      total_moves: moves,
      seed,
      policy_name: "PPO_Stage8_v2",
      sheep_penned: 2,
      total_sheep: 4,
      replayAvailable: false,
      replaySource: null,
    });
  }

  return records.sort((a, b) => Number(b.episode_id) - Number(a.episode_id));
}

class EpisodeHistoryStore {
  private episodes: EpisodeRecord[] = [];

  constructor() {
    // Production store starts empty. No preseeded mock data added.
    this.episodes = [];
  }

  public getEpisodes(): EpisodeRecord[] {
    return [...this.episodes];
  }

  public getEpisodeById(id: number | string): EpisodeRecord | undefined {
    return this.episodes.find((ep) => String(ep.episode_id) === String(id));
  }

  /**
   * Synchronize stored episode records with API data.
   */
  public syncWithApiEpisodes(apiEpisodes: TrainingEpisode[]): void {
    if (!apiEpisodes || apiEpisodes.length === 0) {
      return;
    }

    const map = new Map<string, EpisodeRecord>();
    for (const ep of this.episodes) {
      map.set(String(ep.episode_id), ep);
    }

    for (const item of apiEpisodes) {
      const freshRecord = convertTrainingEpisodeToRecord(item);
      const key = String(freshRecord.episode_id);
      const existing = map.get(key);
      if (existing) {
        map.set(key, {
          ...freshRecord,
          replayAvailable: existing.replayAvailable || freshRecord.replayAvailable,
          replay_id: freshRecord.replay_id || existing.replay_id,
          replaySource: existing.replaySource || freshRecord.replaySource,
          capture_reason: existing.capture_reason || freshRecord.capture_reason,
          capture_status: freshRecord.capture_status !== "not_requested" ? freshRecord.capture_status : existing.capture_status,
          initial_state: existing.initial_state || freshRecord.initial_state,
          move_history: existing.move_history || freshRecord.move_history,
          total_moves: (existing.move_history && existing.move_history.length > 0) ? (existing.move_history.length - 1) : freshRecord.total_moves,
        });
      } else {
        map.set(key, freshRecord);
      }
    }

    this.setEpisodes(Array.from(map.values()));
  }

  /**
   * Sets the episodes list to explicit records, capped at MAX_EPISODE_HISTORY_BUFFER_SIZE.
   */
  public setEpisodes(records: EpisodeRecord[]): void {
    this.episodes = [...records].sort((a, b) => Number(b.episode_id) - Number(a.episode_id));
    if (this.episodes.length > MAX_EPISODE_HISTORY_BUFFER_SIZE) {
      this.episodes = this.episodes.slice(0, MAX_EPISODE_HISTORY_BUFFER_SIZE);
    }
  }

  /**
   * Adds an episode record to the rolling buffer.
   */
  public addEpisode(record: EpisodeRecord): void {
    this.episodes = this.episodes.filter((ep) => String(ep.episode_id) !== String(record.episode_id));
    this.episodes.push(record);
    this.episodes.sort((a, b) => Number(b.episode_id) - Number(a.episode_id));

    if (this.episodes.length > MAX_EPISODE_HISTORY_BUFFER_SIZE) {
      this.episodes = this.episodes.slice(0, MAX_EPISODE_HISTORY_BUFFER_SIZE);
    }
  }

  public updateEpisode(record: EpisodeRecord): void {
    this.addEpisode(record);
  }

  public clear(): void {
    this.episodes = [];
  }
}

export const episodeHistoryStore = new EpisodeHistoryStore();

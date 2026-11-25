export interface ParsedStageSpec {
  stages: number[];
  error: string | null;
}

/**
 * Parse a user-entered stage specification into an ordered list of stage numbers.
 * Supports single stages ("25"), inclusive ranges ("10-12"), en-dashes ("10–12"),
 * spaces around dashes ("10 - 12"), and comma/semicolon lists ("10-12, 15-20, 25, 33-38").
 * An empty string returns `{ stages: [], error: null }` (single-stage mode).
 */
export function parseStageSpec(
  input: string,
  minStage = 1,
  maxStage = 38,
): ParsedStageSpec {
  const normalized = input.replace(/–/g, "-").trim();
  if (!normalized) {
    return { stages: [], error: null };
  }

  const rawTokens = normalized.split(/[,;]/).map((token) => token.trim());
  if (rawTokens.length > 1 && rawTokens[rawTokens.length - 1] === "") {
    rawTokens.pop();
  }

  const stages: number[] = [];
  for (const token of rawTokens) {
    if (!token) {
      return {
        stages: [],
        error: "Stage list contains an empty entry. Remove extra commas or semicolons.",
      };
    }

    const rangeMatch = token.match(/^(-?\d+)\s*-\s*(-?\d+)$/);
    if (rangeMatch) {
      const start = Number(rangeMatch[1]);
      const end = Number(rangeMatch[2]);
      if (start > end) {
        return {
          stages: [],
          error: `Invalid stage range "${token}": start (${start}) cannot exceed end (${end}).`,
        };
      }
      if (start < minStage || end > maxStage) {
        return {
          stages: [],
          error: `Stage range "${token}" is out of bounds (must be between ${minStage} and ${maxStage}).`,
        };
      }
      for (let stage = start; stage <= end; stage += 1) {
        stages.push(stage);
      }
      continue;
    }

    const singleIntMatch = token.match(/^(-?\d+)$/);
    if (singleIntMatch) {
      const stage = Number(singleIntMatch[1]);
      if (stage < minStage || stage > maxStage) {
        return {
          stages: [],
          error: `Stage ${stage} is out of bounds (must be between ${minStage} and ${maxStage}).`,
        };
      }
      stages.push(stage);
      continue;
    }

    if (token.includes("-")) {
      return {
        stages: [],
        error: `Invalid stage range "${token}". Use integers like 10-12.`,
      };
    }

    return {
      stages: [],
      error: `Invalid stage "${token}". Enter integers or ranges between ${minStage} and ${maxStage}.`,
    };
  }

  return { stages, error: null };
}

/**
 * Format an ordered stage list into a compact specification string,
 * collapsing consecutive ascending runs of length >= 2 into `start-end`.
 */
export function formatStageSpec(stages: number[]): string {
  if (stages.length === 0) {
    return "";
  }

  const parts: string[] = [];
  let runStart = stages[0];
  let runEnd = stages[0];

  for (let i = 1; i < stages.length; i += 1) {
    const current = stages[i];
    if (current === runEnd + 1) {
      runEnd = current;
    } else {
      parts.push(runStart === runEnd ? String(runStart) : `${runStart}-${runEnd}`);
      runStart = current;
      runEnd = current;
    }
  }
  parts.push(runStart === runEnd ? String(runStart) : `${runStart}-${runEnd}`);

  return parts.join(", ");
}

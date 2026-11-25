import { describe, expect, it } from "vitest";
import { formatStageSpec, parseStageSpec } from "./stageSpec";

describe("parseStageSpec", () => {
  it("treats empty or whitespace-only input as empty single-stage mode", () => {
    expect(parseStageSpec("")).toEqual({ stages: [], error: null });
    expect(parseStageSpec("   \t  ")).toEqual({ stages: [], error: null });
  });

  it("parses full range 1-38", () => {
    const result = parseStageSpec("1-38");
    expect(result.error).toBeNull();
    expect(result.stages).toHaveLength(38);
    expect(result.stages[0]).toBe(1);
    expect(result.stages[37]).toBe(38);
  });

  it("parses mixed ranges and single stages in exact order", () => {
    const result = parseStageSpec("10-12, 15-20, 25, 33-38");
    expect(result).toEqual({
      stages: [10, 11, 12, 15, 16, 17, 18, 19, 20, 25, 33, 34, 35, 36, 37, 38],
      error: null,
    });
  });

  it("preserves exact order and intentional duplicates", () => {
    expect(parseStageSpec("25, 36, 25")).toEqual({
      stages: [25, 36, 25],
      error: null,
    });
  });

  it("supports en-dash, semicolons, spaces around dashes, and trailing comma", () => {
    expect(parseStageSpec("10 – 12; 15 - 16, ")).toEqual({
      stages: [10, 11, 12, 15, 16],
      error: null,
    });
  });

  it("rejects out-of-bounds, inverted ranges, empty inner tokens, and non-integers", () => {
    const overMax = parseStageSpec("39");
    expect(overMax.stages).toEqual([]);
    expect(overMax.error).toMatch(/39/);
    expect(overMax.error).toMatch(/1 and 38/);

    const zero = parseStageSpec("0");
    expect(zero.stages).toEqual([]);
    expect(zero.error).toMatch(/0/);

    const inverted = parseStageSpec("30-20");
    expect(inverted.stages).toEqual([]);
    expect(inverted.error).toMatch(/30-20/);

    const nonInt = parseStageSpec("abc");
    expect(nonInt.stages).toEqual([]);
    expect(nonInt.error).toMatch(/abc/);

    const innerEmpty = parseStageSpec("1,,2");
    expect(innerEmpty.stages).toEqual([]);
    expect(innerEmpty.error).toBeTruthy();

    const loneComma = parseStageSpec(",");
    expect(loneComma.stages).toEqual([]);
    expect(loneComma.error).toBeTruthy();
  });
});

describe("formatStageSpec", () => {
  it("collapses ascending consecutive runs of 2 or more", () => {
    expect(formatStageSpec([])).toBe("");
    expect(formatStageSpec([25])).toBe("25");
    expect(formatStageSpec([10, 11, 12, 15, 16, 25])).toBe("10-12, 15-16, 25");
    expect(formatStageSpec([25, 36, 25])).toBe("25, 36, 25");
  });
});

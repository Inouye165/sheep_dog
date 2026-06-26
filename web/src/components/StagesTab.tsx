import React, { useState, useEffect } from "react";

interface StageConfig {
  stage: number;
  dogs: number;
  sheep: number;
  width: number;
  height: number;
  pen_width: number;
  pen_height: number;
  pen_placement: string;
  dog_speed: number;
  sheep_speed: number;
  max_steps: number;
  no_progress_window: number;
  spawn_mix: Record<string, number>;
  summary: string;
  details?: string;
  sheep_personality_strength?: number;
  sheep_flock_cohesion_weight?: number;
}

const STAGES_DATA: StageConfig[] = [
  {
    stage: 1,
    dogs: 1,
    sheep: 1,
    width: 60,
    height: 45,
    pen_width: 12,
    pen_height: 12,
    pen_placement: "corner",
    dog_speed: 1.75,
    sheep_speed: 1.0,
    max_steps: 600,
    no_progress_window: 240,
    spawn_mix: { fixed_easy: 1.0 },
    summary: "1 dog, 1 sheep, fixed easy placement.",
    details: "First basic penning task. Placements are completely predictable."
  },
  {
    stage: 2,
    dogs: 1,
    sheep: 1,
    width: 60,
    height: 45,
    pen_width: 12,
    pen_height: 12,
    pen_placement: "corner",
    dog_speed: 1.75,
    sheep_speed: 1.0,
    max_steps: 640,
    no_progress_window: 250,
    spawn_mix: { fixed_easy: 0.7, randomized_flock: 0.3 },
    summary: "1 dog, 1 sheep, mild random starts.",
    details: "Introduces slight randomness to sheep and dog starting positions."
  },
  {
    stage: 3,
    dogs: 1,
    sheep: 2,
    width: 60,
    height: 45,
    pen_width: 12,
    pen_height: 12,
    pen_placement: "corner",
    dog_speed: 1.8,
    sheep_speed: 1.0,
    max_steps: 700,
    no_progress_window: 260,
    spawn_mix: { fixed_easy: 0.8, randomized_flock: 0.2 },
    summary: "1 dog, 2 sheep, fixed easy flock.",
    details: "Adds a second sheep, introducing flocking interactions."
  },
  {
    stage: 4,
    dogs: 1,
    sheep: 2,
    width: 72,
    height: 54,
    pen_width: 12,
    pen_height: 12,
    pen_placement: "corner",
    dog_speed: 1.8,
    sheep_speed: 1.0,
    max_steps: 760,
    no_progress_window: 280,
    spawn_mix: { fixed_easy: 0.6, randomized_flock: 0.4 },
    summary: "1 dog, 2 sheep, randomized flock start.",
    details: "Enlarges the field boundary and increases randomization."
  },
  {
    stage: 5,
    dogs: 2,
    sheep: 3,
    width: 84,
    height: 60,
    pen_width: 12,
    pen_height: 12,
    pen_placement: "corner",
    dog_speed: 1.9,
    sheep_speed: 1.0,
    max_steps: 860,
    no_progress_window: 220,
    spawn_mix: { fixed_easy: 0.8, randomized_flock: 0.2 },
    summary: "2 dogs, 3 sheep, fixed easy teamwork.",
    details: "Introduces dog coordination and teamwork for the first time."
  },
  {
    stage: 6,
    dogs: 2,
    sheep: 3,
    width: 96,
    height: 72,
    pen_width: 12,
    pen_height: 12,
    pen_placement: "corner",
    dog_speed: 1.9,
    sheep_speed: 1.0,
    max_steps: 920,
    no_progress_window: 230,
    spawn_mix: { fixed_easy: 0.6, randomized_flock: 0.35, nearby_stray: 0.05 },
    summary: "2 dogs, 3 sheep, mostly grouped with occasional tiny nearby stray.",
    details: "Introduces strays that wander away slightly from the flock."
  },
  {
    stage: 7,
    dogs: 2,
    sheep: 4,
    width: 96,
    height: 72,
    pen_width: 12,
    pen_height: 12,
    pen_placement: "corner",
    dog_speed: 1.95,
    sheep_speed: 1.0,
    max_steps: 980,
    no_progress_window: 240,
    spawn_mix: { fixed_easy: 0.45, randomized_flock: 0.45, nearby_stray: 0.1 },
    summary: "2 dogs, 4 sheep, early nearby stray collection starts (6-9 cells).",
    details: "Increases number of sheep and introduces nearby strays."
  },
  {
    stage: 8,
    dogs: 3,
    sheep: 4,
    width: 108,
    height: 78,
    pen_width: 14,
    pen_height: 14,
    pen_placement: "corner",
    dog_speed: 2.0,
    sheep_speed: 1.0,
    max_steps: 1040,
    no_progress_window: 260,
    spawn_mix: { fixed_easy: 0.3, randomized_flock: 0.55, nearby_stray: 0.15 },
    summary: "3 dogs, 4 sheep, larger field with frequent nearby strays (7-10 cells).",
    details: "Enlarges field and pen, and adds a third dog teammate."
  },
  {
    stage: 9,
    dogs: 3,
    sheep: 4,
    width: 120,
    height: 84,
    pen_width: 14,
    pen_height: 14,
    pen_placement: "corner",
    dog_speed: 2.0,
    sheep_speed: 1.0,
    max_steps: 1180,
    no_progress_window: 320,
    spawn_mix: { fixed_easy: 0.15, randomized_flock: 0.6, nearby_stray: 0.25 },
    summary: "3 dogs, 4 sheep, nearby stray recovery emphasis (8-12 cells).",
    details: "Greater focus on bringing strays back to the main group."
  },
  {
    stage: 10,
    dogs: 3,
    sheep: 5,
    width: 120,
    height: 84,
    pen_width: 14,
    pen_height: 14,
    pen_placement: "corner",
    dog_speed: 2.0,
    sheep_speed: 1.0,
    max_steps: 1260,
    no_progress_window: 360,
    spawn_mix: { fixed_easy: 0.1, randomized_flock: 0.5, nearby_stray: 0.3, farther_stray: 0.1 },
    summary: "3 dogs, 5 sheep, nearby strays plus first farther stray (18-26 cells).",
    details: "Introduces much larger stray distances requiring collection."
  },
  {
    stage: 11,
    dogs: 3,
    sheep: 5,
    width: 120,
    height: 84,
    pen_width: 14,
    pen_height: 14,
    pen_placement: "corner",
    dog_speed: 2.0,
    sheep_speed: 1.0,
    max_steps: 1340,
    no_progress_window: 390,
    spawn_mix: { randomized_flock: 0.35, nearby_stray: 0.3, farther_stray: 0.25, split_flock: 0.1 },
    summary: "3 dogs, 5 sheep, stronger farther stray recovery (18-28 cells).",
    details: "Eliminates fixed easy spawns, increases farther strays, adds split flock."
  },
  {
    stage: 12,
    dogs: 3,
    sheep: 6,
    width: 120,
    height: 84,
    pen_width: 14,
    pen_height: 14,
    pen_placement: "corner",
    dog_speed: 2.0,
    sheep_speed: 1.0,
    max_steps: 1420,
    no_progress_window: 430,
    spawn_mix: { randomized_flock: 0.35, nearby_stray: 0.35, farther_stray: 0.2, split_flock: 0.1 },
    summary: "3 dogs, 6 sheep, group + one stray collection.",
    details: "Increases max sheep to 6 with mixed group/stray distribution."
  },
  {
    stage: 13,
    dogs: 3,
    sheep: 6,
    width: 126,
    height: 90,
    pen_width: 14,
    pen_height: 14,
    pen_placement: "corner",
    dog_speed: 2.0,
    sheep_speed: 1.0,
    max_steps: 1480,
    no_progress_window: 450,
    spawn_mix: { nearby_stray: 0.4, two_strays: 0.3, farther_stray: 0.2, split_flock: 0.1 },
    summary: "3 dogs, 6 sheep, two nearby strays collection.",
    details: "Enlarges field and focuses on retrieving two stray sheep."
  },
  {
    stage: 14,
    dogs: 3,
    sheep: 6,
    width: 126,
    height: 90,
    pen_width: 14,
    pen_height: 14,
    pen_placement: "corner",
    dog_speed: 2.0,
    sheep_speed: 1.0,
    max_steps: 1520,
    no_progress_window: 460,
    spawn_mix: { split_flock: 0.5, randomized_flock: 0.2, nearby_stray: 0.2, farther_stray: 0.1 },
    summary: "3 dogs, 6 sheep, split flock (3+3) recovery.",
    details: "Splits the herd into two separate groups spawned far apart."
  },
  {
    stage: 15,
    dogs: 3,
    sheep: 6,
    width: 126,
    height: 90,
    pen_width: 14,
    pen_height: 14,
    pen_placement: "corner",
    dog_speed: 2.0,
    sheep_speed: 1.0,
    max_steps: 1560,
    no_progress_window: 470,
    spawn_mix: { partial_scattered: 0.5, split_flock: 0.25, nearby_stray: 0.15, farther_stray: 0.1 },
    summary: "3 dogs, 6 sheep, partially scattered recovery.",
    details: "Sheep are loosely dispersed rather than in a tight flock."
  },
  {
    stage: 16,
    dogs: 3,
    sheep: 6,
    width: 126,
    height: 90,
    pen_width: 14,
    pen_height: 14,
    pen_placement: "corner",
    dog_speed: 2.0,
    sheep_speed: 1.0,
    max_steps: 1600,
    no_progress_window: 480,
    spawn_mix: { scattered_sheep: 0.55, partial_scattered: 0.25, split_flock: 0.1, farther_stray: 0.1 },
    summary: "3 dogs, 6 sheep, scattered sheep recovery.",
    details: "Extremely scattered spawn configuration."
  },
  {
    stage: 17,
    dogs: 3,
    sheep: 6,
    width: 132,
    height: 96,
    pen_width: 14,
    pen_height: 14,
    pen_placement: "same_wall",
    dog_speed: 2.0,
    sheep_speed: 1.0,
    max_steps: 1660,
    no_progress_window: 500,
    spawn_mix: { scattered_sheep: 0.35, partial_scattered: 0.25, split_flock: 0.2, farther_stray: 0.1, nearby_stray: 0.1 },
    summary: "3 dogs, 6 sheep, pen moves on same wall.",
    details: "Pen is no longer locked to a single corner; moves along the bottom wall."
  },
  {
    stage: 18,
    dogs: 3,
    sheep: 6,
    width: 132,
    height: 96,
    pen_width: 14,
    pen_height: 14,
    pen_placement: "any_wall",
    dog_speed: 2.0,
    sheep_speed: 1.0,
    max_steps: 1720,
    no_progress_window: 520,
    spawn_mix: { scattered_sheep: 0.35, partial_scattered: 0.2, split_flock: 0.2, farther_stray: 0.15, nearby_stray: 0.1 },
    summary: "3 dogs, 6 sheep, pen can be on any wall.",
    details: "Pen can spawn along any outer wall, requiring rotational mapping."
  },
  {
    stage: 19,
    dogs: 3,
    sheep: 6,
    width: 132,
    height: 96,
    pen_width: 14,
    pen_height: 14,
    pen_placement: "away_from_corner",
    dog_speed: 2.0,
    sheep_speed: 1.0,
    max_steps: 1760,
    no_progress_window: 530,
    spawn_mix: { scattered_sheep: 0.35, partial_scattered: 0.2, split_flock: 0.2, farther_stray: 0.15, nearby_stray: 0.1 },
    summary: "3 dogs, 6 sheep, pen away from corners.",
    details: "Pen placement restricted away from helpful corner walls."
  },
  {
    stage: 20,
    dogs: 3,
    sheep: 6,
    width: 132,
    height: 96,
    pen_width: 14,
    pen_height: 14,
    pen_placement: "interior",
    dog_speed: 2.0,
    sheep_speed: 1.0,
    max_steps: 1820,
    no_progress_window: 560,
    spawn_mix: { scattered_sheep: 0.4, partial_scattered: 0.2, split_flock: 0.2, farther_stray: 0.1, nearby_stray: 0.1 },
    summary: "3 dogs, 6 sheep, interior pen challenge.",
    details: "Pen spawned inside the field, allowing sheep to escape around all sides."
  },
  {
    stage: 21,
    dogs: 3,
    sheep: 6,
    width: 132,
    height: 96,
    pen_width: 14,
    pen_height: 14,
    pen_placement: "random",
    dog_speed: 2.0,
    sheep_speed: 1.0,
    max_steps: 1900,
    no_progress_window: 600,
    spawn_mix: { randomized_flock: 0.25, nearby_stray: 0.2, farther_stray: 0.2, split_flock: 0.15, partial_scattered: 0.1, scattered_sheep: 0.1 },
    summary: "3 dogs, 6 sheep, random pen + random sheep.",
    details: "Combines fully random pen placements with mixed spawn patterns."
  },
  {
    stage: 22,
    dogs: 3,
    sheep: 6,
    width: 144,
    height: 104,
    pen_width: 14,
    pen_height: 14,
    pen_placement: "random",
    dog_speed: 2.0,
    sheep_speed: 1.0,
    max_steps: 1980,
    no_progress_window: 640,
    spawn_mix: { farther_stray: 0.4, split_flock: 0.3, two_strays: 0.3 },
    summary: "3 dogs, 6 sheep, very far stray/split recovery.",
    details: "Enlarges field again and raises difficulty of strays."
  },
  {
    stage: 23,
    dogs: 3,
    sheep: 6,
    width: 144,
    height: 104,
    pen_width: 14,
    pen_height: 14,
    pen_placement: "random",
    dog_speed: 2.0,
    sheep_speed: 1.0,
    max_steps: 2060,
    no_progress_window: 680,
    spawn_mix: { scattered_sheep: 0.6, randomized_flock: 0.2, split_flock: 0.2 },
    summary: "3 dogs, 6 sheep, fully random scattered placements.",
    details: "Highly scattered start positions with random pen positions."
  },
  {
    stage: 24,
    dogs: 3,
    sheep: 6,
    width: 144,
    height: 104,
    pen_width: 14,
    pen_height: 14,
    pen_placement: "random",
    dog_speed: 2.0,
    sheep_speed: 1.0,
    max_steps: 2140,
    no_progress_window: 700,
    spawn_mix: { all_corners: 1.0 },
    summary: "3 dogs, 6 sheep, all sheep spawned in field corners.",
    details: "Forces dogs to travel to all corners to gather the herd."
  },
  {
    stage: 25,
    dogs: 3,
    sheep: 6,
    width: 144,
    height: 104,
    pen_width: 14,
    pen_height: 14,
    pen_placement: "random",
    dog_speed: 2.0,
    sheep_speed: 1.0,
    max_steps: 2240,
    no_progress_window: 740,
    spawn_mix: { all_corners: 0.35, scattered_sheep: 0.35, split_flock: 0.2, farther_stray: 0.1 },
    sheep_personality_strength: 0.0,
    summary: "3 dogs, 6 sheep, pen-fearful sheep in hard random layouts.",
    details: "Removes default sheep personality assistance (set to 0.0)."
  },
  {
    stage: 26,
    dogs: 3,
    sheep: 6,
    width: 144,
    height: 104,
    pen_width: 14,
    pen_height: 14,
    pen_placement: "random",
    dog_speed: 2.0,
    sheep_speed: 1.0,
    max_steps: 2280,
    no_progress_window: 760,
    spawn_mix: { all_corners: 0.3, scattered_sheep: 0.4, split_flock: 0.2, farther_stray: 0.1 },
    sheep_personality_strength: 0.35,
    summary: "3 dogs, 6 sheep, phase 1 reduced sheep self-grouping tendency.",
    details: "Sheep personality set to 0.35. Self-grouping begins to decrease."
  },
  {
    stage: 27,
    dogs: 3,
    sheep: 6,
    width: 144,
    height: 104,
    pen_width: 14,
    pen_height: 14,
    pen_placement: "random",
    dog_speed: 2.0,
    sheep_speed: 1.0,
    max_steps: 2320,
    no_progress_window: 780,
    spawn_mix: { all_corners: 0.25, scattered_sheep: 0.45, split_flock: 0.2, farther_stray: 0.1 },
    sheep_personality_strength: 0.7,
    summary: "3 dogs, 6 sheep, phase 2 reduced sheep self-grouping tendency.",
    details: "Sheep personality set to 0.7. Cohesion drops significantly."
  },
  {
    stage: 28,
    dogs: 3,
    sheep: 6,
    width: 144,
    height: 104,
    pen_width: 14,
    pen_height: 14,
    pen_placement: "random",
    dog_speed: 2.0,
    sheep_speed: 1.0,
    max_steps: 2360,
    no_progress_window: 800,
    spawn_mix: { all_corners: 0.2, scattered_sheep: 0.5, split_flock: 0.2, farther_stray: 0.1 },
    sheep_personality_strength: 0.7,
    sheep_flock_cohesion_weight: 0.2,
    summary: "3 dogs, 6 sheep, phase 3 reduced sheep self-grouping tendency.",
    details: "Cohesion weight drops to 0.2, and cohesion without dog pressure is disabled."
  },
  {
    stage: 29,
    dogs: 3,
    sheep: 6,
    width: 144,
    height: 104,
    pen_width: 14,
    pen_height: 14,
    pen_placement: "random",
    dog_speed: 2.0,
    sheep_speed: 1.0,
    max_steps: 2400,
    no_progress_window: 820,
    spawn_mix: { all_corners: 0.15, scattered_sheep: 0.55, split_flock: 0.2, farther_stray: 0.1 },
    sheep_personality_strength: 1.0,
    sheep_flock_cohesion_weight: 0.16,
    summary: "3 dogs, 6 sheep, phase 4 reduced sheep self-grouping tendency.",
    details: "Flock cohesion weight drops to 0.16; sheep personality set to 1.0."
  },
  {
    stage: 30,
    dogs: 3,
    sheep: 6,
    width: 144,
    height: 104,
    pen_width: 14,
    pen_height: 14,
    pen_placement: "random",
    dog_speed: 2.0,
    sheep_speed: 1.0,
    max_steps: 2440,
    no_progress_window: 840,
    spawn_mix: { all_corners: 0.1, scattered_sheep: 0.6, split_flock: 0.2, farther_stray: 0.1 },
    sheep_personality_strength: 1.25,
    sheep_flock_cohesion_weight: 0.12,
    summary: "3 dogs, 6 sheep, phase 5 reduced sheep self-grouping tendency.",
    details: "Cohesion weight drops to 0.12. Sheep scatter easily; personality strength set to 1.25."
  }
];

interface Note {
  id: string;
  title: string;
  contents: string;
  timestamp: string;
}

// Interactive visualization of the stage's starting layout
function StartingPositionPreview({ stage }: { stage: StageConfig }) {
  const w = stage.width;
  const h = stage.height;

  // Let's determine pen location based on rules
  let penX = w - stage.pen_width - 2;
  let penY = h - stage.pen_height - 2;
  let isRandomPen = false;
  let isInteriorPen = false;

  if (stage.pen_placement === "interior") {
    penX = w / 2 - stage.pen_width / 2;
    penY = h / 2 - stage.pen_height / 2;
    isInteriorPen = true;
  } else if (stage.pen_placement === "random" || stage.pen_placement === "any_wall") {
    isRandomPen = true;
  } else if (stage.pen_placement === "same_wall" || stage.pen_placement === "away_from_corner") {
    // representative placement
    penX = w / 2 - stage.pen_width / 2;
    penY = h - stage.pen_height - 1;
  }

  // Draw dogs (up to 3) starting positions (usually near the pen or bottom center)
  const dogsList = [];
  for (let i = 0; i < stage.dogs; i++) {
    // representative starting points
    dogsList.push({
      x: isInteriorPen ? penX - 5 + i * 5 : penX - 4 - i * 3,
      y: isInteriorPen ? penY + stage.pen_height + 4 : h - 4
    });
  }

  // Draw sheep based on spawn mix rules
  const sheepList: { x: number; y: number }[] = [];
  const mix = stage.spawn_mix;

  if (mix.fixed_easy || mix.randomized_flock) {
    // Single main group in the upper left or center
    for (let i = 0; i < stage.sheep; i++) {
      sheepList.push({
        x: w * 0.3 + (i % 3) * 3 + Math.sin(i) * 2,
        y: h * 0.3 + Math.floor(i / 3) * 3 + Math.cos(i) * 2
      });
    }
  } else if (mix.all_corners) {
    // Distributed in corners
    for (let i = 0; i < stage.sheep; i++) {
      const cornerIndex = i % 4;
      let cx = w * 0.15;
      let cy = h * 0.15;
      if (cornerIndex === 1) cx = w * 0.85;
      if (cornerIndex === 2) cy = h * 0.85;
      if (cornerIndex === 3) {
        cx = w * 0.85;
        cy = h * 0.85;
      }
      sheepList.push({
        x: cx + Math.sin(i) * 3,
        y: cy + Math.cos(i) * 3
      });
    }
  } else if (mix.split_flock) {
    // 2 groups
    const half = Math.ceil(stage.sheep / 2);
    for (let i = 0; i < stage.sheep; i++) {
      if (i < half) {
        sheepList.push({
          x: w * 0.25 + (i % 2) * 3 + Math.sin(i) * 2,
          y: h * 0.35 + Math.floor(i / 2) * 3 + Math.cos(i) * 2
        });
      } else {
        sheepList.push({
          x: w * 0.7 + (i % 2) * 3 + Math.sin(i) * 2,
          y: h * 0.25 + Math.floor(i / 2) * 3 + Math.cos(i) * 2
        });
      }
    }
  } else {
    // Scattered
    for (let i = 0; i < stage.sheep; i++) {
      // Deterministic random spots
      const angle = (i * 2 * Math.PI) / stage.sheep;
      const radius = Math.min(w, h) * 0.25;
      sheepList.push({
        x: w / 2 + Math.cos(angle) * radius + Math.sin(i * 9) * 2,
        y: h / 2 + Math.sin(angle) * radius + Math.cos(i * 9) * 2
      });
    }
  }

  return (
    <div style={{ background: "rgba(15, 23, 42, 0.4)", borderRadius: "0.75rem", border: "1px solid var(--panel-border)", padding: "1rem", display: "flex", flexDirection: "column", gap: "0.5rem" }}>
      <span className="eyebrow" style={{ fontSize: "0.75rem" }}>Simulated starting position layout preview</span>
      <div style={{ width: "100%", height: "200px", background: "#0f172a", borderRadius: "0.5rem", position: "relative", overflow: "hidden", display: "flex", alignItems: "center", justifyContent: "center" }}>
        <svg viewBox={`0 0 ${w} ${h}`} style={{ width: "100%", height: "100%", maxHeight: "100%" }}>
          <defs>
            <pattern id="gridPattern" width="6" height="6" patternUnits="userSpaceOnUse">
              <path d="M 6 0 L 0 0 0 6" fill="none" stroke="rgba(255,255,255,0.03)" strokeWidth="0.5" />
            </pattern>
          </defs>
          <rect width="100%" height="100%" fill="url(#gridPattern)" />

          {/* Border */}
          <rect width={w} height={h} fill="none" stroke="rgba(255,255,255,0.1)" strokeWidth="1" />

          {/* Pen */}
          {!isRandomPen ? (
            <g>
              <rect
                x={penX}
                y={penY}
                width={stage.pen_width}
                height={stage.pen_height}
                fill="rgba(244, 197, 66, 0.05)"
                stroke="var(--accent)"
                strokeWidth="1.2"
              />
              {/* Gate opening */}
              {isInteriorPen ? (
                <line x1={penX} y1={penY + stage.pen_height} x2={penX + stage.pen_width} y2={penY + stage.pen_height} stroke="#0f172a" strokeWidth="2" />
              ) : (
                <line x1={penX} y1={penY} x2={penX + stage.pen_width} y2={penY} stroke="#0f172a" strokeWidth="2" />
              )}
            </g>
          ) : (
            // Indication of random pen position
            <g>
              <rect
                x={w * 0.7}
                y={h * 0.6}
                width={stage.pen_width}
                height={stage.pen_height}
                fill="none"
                stroke="var(--accent)"
                strokeDasharray="2,2"
                strokeWidth="1.2"
              />
              <text x={w * 0.7 + stage.pen_width/2} y={h * 0.6 + stage.pen_height/2 + 2} fill="var(--accent)" fontSize="4" textAnchor="middle" fontWeight="bold">Random Pen</text>
            </g>
          )}

          {/* Sheep */}
          {sheepList.map((s, idx) => (
            <circle
              key={`sheep-${idx}`}
              cx={s.x}
              cy={s.y}
              r="1.6"
              fill="#ffffff"
              stroke="#cbd5e1"
              strokeWidth="0.4"
            />
          ))}

          {/* Dogs */}
          {dogsList.map((d, idx) => {
            const colors = ["#3b82f6", "#10b981", "#8b5cf6"];
            return (
              <circle
                key={`dog-${idx}`}
                cx={d.x}
                cy={d.y}
                r="1.8"
                fill={colors[idx % colors.length]}
                stroke="#ffffff"
                strokeWidth="0.5"
              />
            );
          })}
        </svg>
      </div>
      <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.75rem", color: "var(--muted)" }}>
        <span>⬤ White = Sheep ({stage.sheep})</span>
        <span>⬤ Colors = Dogs ({stage.dogs})</span>
        <span>⬜ Gold = Pen ({stage.pen_placement})</span>
      </div>
    </div>
  );
}

export function StagesTab() {
  const [selectedStageNum, setSelectedStageNum] = useState<number>(1);
  const [notes, setNotes] = useState<Note[]>(() => {
    try {
      const saved = localStorage.getItem("sheepdog_master_notes");
      return saved ? JSON.parse(saved) : [];
    } catch {
      return [];
    }
  });

  const [noteTitle, setNoteTitle] = useState("");
  const [noteContents, setNoteContents] = useState("");
  const [selectedNoteIds, setSelectedNoteIds] = useState<Set<string>>(new Set());
  const [copyFeedback, setCopyFeedback] = useState(false);

  // Sync notes to local storage
  useEffect(() => {
    localStorage.setItem("sheepdog_master_notes", JSON.stringify(notes));
  }, [notes]);

  const selectedStage = STAGES_DATA.find((s) => s.stage === selectedStageNum) || STAGES_DATA[0];
  const previousStage = selectedStageNum > 1 ? STAGES_DATA.find((s) => s.stage === selectedStageNum - 1) : null;

  // Calculate differences to make it harder
  const getDiffs = () => {
    if (!previousStage) return ["Baseline stage. No previous stage to compare."];
    const diffs: string[] = [];

    if (selectedStage.dogs > previousStage.dogs) {
      diffs.push(`Increased dog team size to ${selectedStage.dogs} (+${selectedStage.dogs - previousStage.dogs} dog)`);
    }
    if (selectedStage.sheep > previousStage.sheep) {
      diffs.push(`Increased sheep count to ${selectedStage.sheep} (+${selectedStage.sheep - previousStage.sheep} sheep)`);
    }
    if (selectedStage.width > previousStage.width || selectedStage.height > previousStage.height) {
      diffs.push(
        `Expanded field size to ${selectedStage.width}x${selectedStage.height} (previously ${previousStage.width}x${previousStage.height})`
      );
    }
    if (selectedStage.pen_width > previousStage.pen_width) {
      diffs.push(`Changed pen dimensions to ${selectedStage.pen_width}x${selectedStage.pen_height}`);
    }
    if (selectedStage.pen_placement !== previousStage.pen_placement) {
      diffs.push(`Moved pen placement rule to "${selectedStage.pen_placement}" (previously "${previousStage.pen_placement}")`);
    }
    if (selectedStage.dog_speed > previousStage.dog_speed) {
      diffs.push(`Increased dog speed to ${selectedStage.dog_speed}`);
    }
    if (selectedStage.max_steps > previousStage.max_steps) {
      diffs.push(`Extended episode limit to ${selectedStage.max_steps} steps (+${selectedStage.max_steps - previousStage.max_steps})`);
    }
    if (selectedStage.no_progress_window > previousStage.no_progress_window) {
      diffs.push(`Increased progress window to ${selectedStage.no_progress_window} steps`);
    }

    // Compare spawn mix
    const newMixKeys = Object.keys(selectedStage.spawn_mix);
    const oldMixKeys = Object.keys(previousStage.spawn_mix);
    newMixKeys.forEach((key) => {
      if (!oldMixKeys.includes(key)) {
        diffs.push(`Added new spawn pattern: "${key}" (${Math.round(selectedStage.spawn_mix[key] * 100)}% weight)`);
      } else if (selectedStage.spawn_mix[key] > previousStage.spawn_mix[key]) {
        diffs.push(`Increased spawn chance of "${key}" to ${Math.round(selectedStage.spawn_mix[key] * 100)}%`);
      }
    });

    // Personality strength changes
    const curPers = selectedStage.sheep_personality_strength ?? 1.0;
    const prevPers = previousStage.sheep_personality_strength ?? 1.0;
    if (curPers !== prevPers) {
      diffs.push(`Adjusted sheep personality strength to ${curPers} (previously ${prevPers})`);
    }

    // Cohesion changes
    const curCoh = selectedStage.sheep_flock_cohesion_weight ?? 1.0;
    const prevCoh = previousStage.sheep_flock_cohesion_weight ?? 1.0;
    if (curCoh < prevCoh) {
      diffs.push(`Reduced sheep self-grouping cohesion weight to ${curCoh} (harder to flock)`);
    }

    if (diffs.length === 0) {
      diffs.push("Incremental difficulty adjustments and minor coordinate randomizations.");
    }
    return diffs;
  };

  const handleAddNote = (e: React.FormEvent) => {
    e.preventDefault();
    if (!noteTitle.trim() && !noteContents.trim()) return;

    const newNote: Note = {
      id: Math.random().toString(36).substring(2, 9),
      title: noteTitle.trim() || "Untitled Note",
      contents: noteContents.trim() || "(No contents)",
      timestamp: new Date().toLocaleString()
    };

    setNotes([newNote, ...notes]);
    setNoteTitle("");
    setNoteContents("");
  };

  const handleToggleNoteSelect = (id: string) => {
    const next = new Set(selectedNoteIds);
    if (next.has(id)) {
      next.delete(id);
    } else {
      next.add(id);
    }
    setSelectedNoteIds(next);
  };

  const handleSelectAllNotes = () => {
    if (selectedNoteIds.size === notes.length) {
      setSelectedNoteIds(new Set());
    } else {
      setSelectedNoteIds(new Set(notes.map((n) => n.id)));
    }
  };

  const handleDeleteSelected = () => {
    if (selectedNoteIds.size === 0) return;
    if (window.confirm(`Are you sure you want to delete ${selectedNoteIds.size} selected note(s)?`)) {
      setNotes(notes.filter((n) => !selectedNoteIds.has(n.id)));
      setSelectedNoteIds(new Set());
    }
  };

  const handleCopyToClipboard = () => {
    const selected = notes.filter((n) => selectedNoteIds.has(n.id));
    if (selected.length === 0) return;

    const formattedText = selected
      .map((n) => `[${n.timestamp}] ${n.title}\n${n.contents}\n--------------------`)
      .join("\n\n");

    navigator.clipboard.writeText(formattedText).then(() => {
      setCopyFeedback(true);
      setTimeout(() => setCopyFeedback(false), 2000);
    });
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "1.5rem", height: "100%", minHeight: 0 }}>
      {/* Tab Header Banner */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", borderBottom: "1px solid var(--panel-border)", paddingBottom: "1rem" }}>
        <div>
          <p className="eyebrow" style={{ margin: 0 }}>Training curriculum</p>
          <h2 style={{ margin: "0.25rem 0 0" }}>Progression Stages & Notes</h2>
        </div>
        <span className="pill pill--muted">30 Stages</span>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 450px", gap: "1.5rem", flex: 1, minHeight: 0 }}>
        {/* LEFT COLUMN: Stages Progression & Stats */}
        <div style={{ display: "flex", flexDirection: "column", gap: "1rem", minHeight: 0 }}>
          
          {/* Wrapped Grid Selector: fits 30 stages nicely, wraps cleanly and fits any screen */}
          <div style={{ 
            background: "rgba(8, 15, 25, 0.8)", 
            border: "1px solid var(--panel-border)", 
            borderRadius: "0.75rem", 
            padding: "0.75rem", 
            display: "grid", 
            gridTemplateColumns: "repeat(10, 1fr)",
            gap: "0.4rem"
          }}>
            {STAGES_DATA.map((s) => (
              <button
                key={s.stage}
                onClick={() => setSelectedStageNum(s.stage)}
                style={{
                  background: selectedStageNum === s.stage ? "var(--accent)" : "rgba(255, 255, 255, 0.05)",
                  color: selectedStageNum === s.stage ? "#000" : "var(--text)",
                  border: "1px solid transparent",
                  padding: "0.4rem 0.2rem",
                  borderRadius: "0.4rem",
                  cursor: "pointer",
                  fontWeight: "bold",
                  fontSize: "0.8rem",
                  textAlign: "center"
                }}
              >
                S{s.stage}
              </button>
            ))}
          </div>

          {/* Selected Stage Parameters Cards */}
          <div style={{ overflowY: "auto", flex: 1, display: "flex", flexDirection: "column", gap: "1rem", paddingRight: "0.5rem", scrollbarWidth: "thin" }}>
            <div className="network-card" style={{ padding: "1.5rem", margin: 0 }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "0.75rem" }}>
                <span className="eyebrow">Selected configuration</span>
                <span className="pill" style={{ color: "var(--accent)", background: "rgba(244, 197, 66, 0.15)" }}>Stage {selectedStage.stage}</span>
              </div>
              <h3 style={{ margin: "0 0 0.5rem", fontSize: "1.25rem" }}>{selectedStage.summary}</h3>
              <p style={{ color: "var(--muted)", margin: "0 0 1.25rem", fontSize: "0.9rem", lineHeight: "1.4" }}>
                {selectedStage.details}
              </p>

              {/* Grid of details */}
              <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: "1rem", marginBottom: "1.25rem" }}>
                <div style={{ background: "rgba(255, 255, 255, 0.03)", padding: "0.75rem", borderRadius: "0.5rem", border: "1px solid var(--panel-border)" }}>
                  <div style={{ fontSize: "0.75rem", color: "var(--muted)" }}>Dogs</div>
                  <strong style={{ fontSize: "1.2rem", color: "var(--accent)" }}>{selectedStage.dogs}</strong>
                </div>
                <div style={{ background: "rgba(255, 255, 255, 0.03)", padding: "0.75rem", borderRadius: "0.5rem", border: "1px solid var(--panel-border)" }}>
                  <div style={{ fontSize: "0.75rem", color: "var(--muted)" }}>Sheep</div>
                  <strong style={{ fontSize: "1.2rem" }}>{selectedStage.sheep}</strong>
                </div>
                <div style={{ background: "rgba(255, 255, 255, 0.03)", padding: "0.75rem", borderRadius: "0.5rem", border: "1px solid var(--panel-border)" }}>
                  <div style={{ fontSize: "0.75rem", color: "var(--muted)" }}>Env Size</div>
                  <strong style={{ fontSize: "1.1rem" }}>{selectedStage.width} × {selectedStage.height}</strong>
                </div>
                <div style={{ background: "rgba(255, 255, 255, 0.03)", padding: "0.75rem", borderRadius: "0.5rem", border: "1px solid var(--panel-border)" }}>
                  <div style={{ fontSize: "0.75rem", color: "var(--muted)" }}>Max Steps</div>
                  <strong style={{ fontSize: "1.1rem" }}>{selectedStage.max_steps}</strong>
                </div>
              </div>

              {/* Parameters List */}
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "1rem", fontSize: "0.85rem" }}>
                <div>
                  <div style={{ marginBottom: "0.4rem" }}>
                    <span style={{ color: "var(--muted)" }}>Pen Placement:</span> <strong>{selectedStage.pen_placement}</strong>
                  </div>
                  <div style={{ marginBottom: "0.4rem" }}>
                    <span style={{ color: "var(--muted)" }}>Pen Dimensions:</span> <strong>{selectedStage.pen_width} × {selectedStage.pen_height}</strong>
                  </div>
                  <div style={{ marginBottom: "0.4rem" }}>
                    <span style={{ color: "var(--muted)" }}>Dog Speed:</span> <strong>{selectedStage.dog_speed}</strong>
                  </div>
                </div>
                <div>
                  <div style={{ marginBottom: "0.4rem" }}>
                    <span style={{ color: "var(--muted)" }}>No-Progress Window:</span> <strong>{selectedStage.no_progress_window} steps</strong>
                  </div>
                  <div style={{ marginBottom: "0.4rem" }}>
                    <span style={{ color: "var(--muted)" }}>Sheep Cohesion Weight:</span> <strong>{selectedStage.sheep_flock_cohesion_weight ?? 1.0}</strong>
                  </div>
                  <div style={{ marginBottom: "0.4rem" }}>
                    <span style={{ color: "var(--muted)" }}>Spawn Mix:</span>{" "}
                    <strong>
                      {Object.entries(selectedStage.spawn_mix)
                        .map(([k, v]) => `${k} (${Math.round(v * 100)}%)`)
                        .join(", ")}
                    </strong>
                  </div>
                </div>
              </div>
            </div>

            {/* STARTING POSITION SVG PREVIEW */}
            <StartingPositionPreview stage={selectedStage} />

            {/* WHAT IS ADDED TO MAKE IT HARDER */}
            <div className="network-card" style={{ padding: "1.5rem", margin: 0, background: "rgba(244, 197, 66, 0.03)", border: "1px solid rgba(244, 197, 66, 0.2)" }}>
              <h4 style={{ color: "var(--accent)", margin: "0 0 0.75rem", fontSize: "0.95rem", textTransform: "uppercase", letterSpacing: "0.05em" }}>
                What makes this stage harder?
              </h4>
              <ul style={{ margin: 0, paddingLeft: "1.2rem", fontSize: "0.9rem", color: "var(--text)", lineHeight: "1.6" }}>
                {getDiffs().map((diff, index) => (
                  <li key={index} style={{ marginBottom: "0.4rem" }}>
                    {diff}
                  </li>
                ))}
              </ul>
            </div>
          </div>
        </div>

        {/* RIGHT COLUMN: Master Notes Panel */}
        <div style={{ background: "rgba(8, 15, 25, 0.6)", border: "1px solid var(--panel-border)", borderRadius: "0.75rem", padding: "1.25rem", display: "flex", flexDirection: "column", minHeight: 0 }}>
          <h3 style={{ margin: "0 0 1rem" }}>Master Notes Log</h3>

          {/* Add note Form */}
          <form onSubmit={handleAddNote} style={{ display: "flex", flexDirection: "column", gap: "0.75rem", marginBottom: "1.25rem", borderBottom: "1px solid var(--panel-border)", paddingBottom: "1.25rem" }}>
            <input
              type="text"
              placeholder="Note title..."
              value={noteTitle}
              onChange={(e) => setNoteTitle(e.target.value)}
              style={{
                borderRadius: "0.5rem",
                border: "1px solid var(--panel-border)",
                color: "var(--text)",
                background: "rgba(8, 15, 25, 0.8)",
                padding: "0.5rem 0.75rem",
                fontSize: "0.85rem"
              }}
            />
            <textarea
              placeholder="Write observations, reward adjustments, success rate notes..."
              value={noteContents}
              onChange={(e) => setNoteContents(e.target.value)}
              rows={3}
              style={{
                borderRadius: "0.5rem",
                border: "1px solid var(--panel-border)",
                color: "var(--text)",
                background: "rgba(8, 15, 25, 0.8)",
                padding: "0.5rem 0.75rem",
                fontSize: "0.85rem",
                resize: "none"
              }}
            />
            <button
              type="submit"
              style={{
                background: "var(--accent)",
                color: "#000",
                border: "none",
                padding: "0.5rem 1rem",
                borderRadius: "0.5rem",
                cursor: "pointer",
                fontWeight: "bold",
                alignSelf: "flex-end",
                fontSize: "0.85rem"
              }}
            >
              Add note
            </button>
          </form>

          {/* Notes Actions */}
          {notes.length > 0 && (
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "0.75rem" }}>
              <button
                onClick={handleSelectAllNotes}
                style={{
                  background: "transparent",
                  color: "var(--muted)",
                  border: "none",
                  cursor: "pointer",
                  fontSize: "0.75rem",
                  padding: 0
                }}
              >
                {selectedNoteIds.size === notes.length ? "Deselect All" : "Select All"}
              </button>
              <div style={{ display: "flex", gap: "0.5rem" }}>
                <button
                  onClick={handleCopyToClipboard}
                  disabled={selectedNoteIds.size === 0}
                  style={{
                    background: "rgba(255,255,255,0.05)",
                    border: "1px solid var(--panel-border)",
                    color: selectedNoteIds.size > 0 ? "var(--text)" : "var(--muted)",
                    padding: "0.3rem 0.6rem",
                    borderRadius: "0.25rem",
                    fontSize: "0.75rem",
                    cursor: selectedNoteIds.size > 0 ? "pointer" : "default"
                  }}
                >
                  {copyFeedback ? "✓ Copied!" : "Copy selected"}
                </button>
                <button
                  onClick={handleDeleteSelected}
                  disabled={selectedNoteIds.size === 0}
                  style={{
                    background: "rgba(244, 63, 94, 0.15)",
                    border: "1px solid rgba(244, 63, 94, 0.3)",
                    color: selectedNoteIds.size > 0 ? "var(--accent)" : "var(--muted)",
                    padding: "0.3rem 0.6rem",
                    borderRadius: "0.25rem",
                    fontSize: "0.75rem",
                    cursor: selectedNoteIds.size > 0 ? "pointer" : "default"
                  }}
                >
                  Delete selected
                </button>
              </div>
            </div>
          )}

          {/* Notes List Scrollbox */}
          <div style={{ overflowY: "auto", flex: 1, display: "flex", flexDirection: "column", gap: "0.75rem", scrollbarWidth: "thin" }}>
            {notes.length === 0 ? (
              <div style={{ textAlign: "center", color: "var(--muted)", padding: "2rem 0", fontSize: "0.85rem" }}>
                No notes logged yet. Add your first training note above.
              </div>
            ) : (
              notes.map((note) => (
                <div
                  key={note.id}
                  style={{
                    background: "rgba(255, 255, 255, 0.02)",
                    border: "1px solid var(--panel-border)",
                    borderRadius: "0.5rem",
                    padding: "0.75rem",
                    position: "relative",
                    display: "flex",
                    gap: "0.75rem"
                  }}
                >
                  <input
                    type="checkbox"
                    checked={selectedNoteIds.has(note.id)}
                    onChange={() => handleToggleNoteSelect(note.id)}
                    style={{ cursor: "pointer", marginTop: "0.2rem" }}
                  />
                  <div style={{ flex: 1 }}>
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: "0.25rem" }}>
                      <strong style={{ fontSize: "0.9rem" }}>{note.title}</strong>
                      <span style={{ fontSize: "0.7rem", color: "var(--muted)" }}>{note.timestamp}</span>
                    </div>
                    <p style={{ margin: 0, fontSize: "0.85rem", color: "var(--muted)", whiteSpace: "pre-wrap", lineHeight: "1.4" }}>
                      {note.contents}
                    </p>
                  </div>
                </div>
              ))
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

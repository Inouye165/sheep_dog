import React, { useState } from "react";

export interface StageConceptInfo {
  title: string;
  concept: string;
  description: string;
  keyChange: string;
  icon?: string;
}

export const STAGE_CONCEPTS: Record<number, StageConceptInfo> = {
  1: {
    title: "Stage 1: Basic Penning",
    concept: "Single Dog & Sheep Foundations",
    description: "Learn to approach from behind and steer a single sheep into a fixed corner pen.",
    keyChange: "Baseline control mechanics and corner approach angle.",
    icon: "🎯",
  },
  2: {
    title: "Stage 2: Mild Randomization",
    concept: "Variable Starting Positions",
    description: "Sheep and dog positions vary slightly at spawn instead of starting at fixed coordinates.",
    keyChange: "Adapting approach trajectories to dynamic start coordinates.",
    icon: "🎲",
  },
  3: {
    title: "Stage 3: Flocking Introduction",
    concept: "Dual Sheep Flocking",
    description: "A second sheep is introduced. The policy must understand mutual attraction between sheep.",
    keyChange: "Managing multiple animals and keeping them grouped.",
    icon: "🐑",
  },
  4: {
    title: "Stage 4: Arena Expansion",
    concept: "Expanded Arena & Randomized Flock",
    description: "Arena expands to 72x54 with fully randomized initial flock placements.",
    keyChange: "Navigating larger distances and controlling momentum.",
    icon: "📐",
  },
  5: {
    title: "Stage 5: Multi-Dog Teamwork",
    concept: "2-Dog Cooperative Herding",
    description: "A second dog joins the field. Dogs must divide roles (rear driver vs. flanker) to avoid collisions.",
    keyChange: "Multi-agent coordination and spatial role differentiation.",
    icon: "🐕",
  },
  6: {
    title: "Stage 6: Stray Recovery",
    concept: "Nearby Stray Detection",
    description: "Single sheep occasionally drifts 3–5 cells away from the main flock.",
    keyChange: "Recognizing strays and breaking off to retrieve them before penning.",
    icon: "🔍",
  },
  7: {
    title: "Stage 7: Extended Stray Radius",
    concept: "Wide Stray Recovery (4 Sheep)",
    description: "Herd increases to 4 sheep with strays drifting 6–9 cells away.",
    keyChange: "Balancing main herd containment with long-range stray sweeps.",
    icon: "🔭",
  },
  8: {
    title: "Stage 8: 3-Dog Coordination",
    concept: "Full 3-Dog Team Dynamics",
    description: "3 dogs operate simultaneously on a 120x80 arena with frequent stray occurrences.",
    keyChange: "Three-way role division: Driver, Flanker, and Collector.",
    icon: "🤝",
  },
  9: {
    title: "Stage 9: Wall & Corner Recovery",
    concept: "Boundary Entrapment Remediation",
    description: "High concentration of wall and corner entrapment scenarios requiring dislodging maneuvers.",
    keyChange: "Extracting sheep pinned against arena boundaries.",
    icon: "🧱",
  },
  10: {
    title: "Stage 10: Far Stray Collection",
    concept: "5 Sheep & Far Stray Radius",
    description: "Herd grows to 5 sheep with strays spawning 12–16 cells away from the cluster.",
    keyChange: "Long-distance interception while maintaining rear pressure.",
    icon: "🏃",
  },
  16: {
    title: "Stage 16: 6-Sheep Scaling",
    concept: "Full Herd Size Introduction",
    description: "Herd scales to 6 sheep, significantly increasing the perimeter that dogs must patrol.",
    keyChange: "Flock perimeter control and anti-scattering maneuvers.",
    icon: "👥",
  },
  19: {
    title: "Stage 19: Split Flock Recovery",
    concept: "Bifurcated Herd (3+3 Split)",
    description: "The herd begins divided into two separate clusters on opposite sides of the field.",
    keyChange: "Flanking both clusters inwards to merge the herd before penning.",
    icon: "✂️",
  },
  21: {
    title: "Stage 21: Partially Scattered Herd",
    concept: "Dispersed Formation Recovery",
    description: "Sheep spawn loosely scattered across the arena rather than in a tight group.",
    keyChange: "Sequential gathering and active clustering before driving toward pen.",
    icon: "✨",
  },
  22: {
    title: "Stage 22: Fully Scattered Herd",
    concept: "High-Entropy Scatter Recovery",
    description: "55% scattered sheep spawns requiring comprehensive sweep and containment.",
    keyChange: "Wide perimeter sweeps to consolidate high-spread herds.",
    icon: "🌪️",
  },
  24: {
    title: "Stage 24: Wall-Aligned Pen Placement",
    concept: "Pen Moved Away from Corner",
    description: "The pen moves from the field corner to along the wall (same_wall). Dogs cannot rely on corner funnels.",
    keyChange: "Guiding sheep along an open boundary without corner backstops.",
    icon: "🚪",
  },
  25: {
    title: "Stage 25: Detached Pen Placement",
    concept: "Pen Away from All Corners",
    description: "The pen is positioned further away from corners, demanding precise alignment angles.",
    keyChange: "Approach trajectory control without corner trapping assistance.",
    icon: "📍",
  },
  26: {
    title: "Stage 26: Interior Pen Challenge",
    concept: "Open-Field Penning",
    description: "Pen placed in the open interior of the arena away from all walls.",
    keyChange: "360-degree containment; dogs must act as living walls on all sides.",
    icon: "🎪",
  },
  27: {
    title: "Stage 27: Randomized Pen Placement",
    concept: "Fully Dynamic Pen Coordinates",
    description: "Pen location is randomized each episode across corners, walls, and interior.",
    keyChange: "Generalized spatial awareness regardless of pen destination.",
    icon: "🧭",
  },
  30: {
    title: "Stage 30: All-Corners Spawn",
    concept: "Four-Corner Distribution",
    description: "Sheep spawn separated into all four corners of the arena.",
    keyChange: "Multi-stage sweeps across the entire perimeter of the field.",
    icon: "🗺️",
  },
  33: {
    title: "Stage 33: Personality Variation",
    concept: "Heterogeneous Sheep Speeds",
    description: "Individual sheep have varying movement speeds and personal fear responses.",
    keyChange: "Adapting pressure intensity to fast vs. slow sheep in the same herd.",
    icon: "⚡",
  },
  36: {
    title: "Stage 36: Zero-Pressure Cohesion Disabled",
    concept: "Independent Sheep Dynamics",
    description: "Sheep will no longer naturally flock together unless dogs actively apply pressure.",
    keyChange: "Continuous multi-flank pressure required to keep herd unified.",
    icon: "🛡️",
  },
  38: {
    title: "Stage 38: Maximum Complexity",
    concept: "Mastery Benchmark",
    description: "Random pen locations, high scatter, maximum personality variance, and low cohesion.",
    keyChange: "Full generalist herding capability across all combined challenges.",
    icon: "👑",
  },
};

export interface StageConceptBannerProps {
  stage: number;
  className?: string;
}

export function StageConceptBanner({ stage, className = "" }: StageConceptBannerProps) {
  const concept = STAGE_CONCEPTS[stage];

  // Store dismissal per stage in localStorage so users aren't permanently nagged
  const storageKey = `sheepdog_concept_banner_dismissed_stage_${stage}`;
  const [isDismissed, setIsDismissed] = useState<boolean>(() => {
    try {
      return localStorage.getItem(storageKey) === "true";
    } catch {
      return false;
    }
  });

  if (!concept || isDismissed) {
    return null;
  }

  const handleDismiss = () => {
    setIsDismissed(true);
    try {
      localStorage.setItem(storageKey, "true");
    } catch {
      // Ignore storage errors
    }
  };

  return (
    <div
      className={`stage-concept-banner ${className}`}
      role="region"
      aria-label={`New concept for Stage ${stage}`}
    >
      <div className="stage-concept-banner__indicator">
        <span className="stage-concept-banner__icon">{concept.icon || "💡"}</span>
      </div>

      <div className="stage-concept-banner__body">
        <div className="stage-concept-banner__header">
          <div className="stage-concept-banner__tags">
            <span className="stage-concept-banner__badge">NEW CONCEPT INTRODUCED</span>
            <span className="stage-concept-banner__stage-pill">Stage {stage}</span>
          </div>
          <button
            type="button"
            className="stage-concept-banner__dismiss"
            onClick={handleDismiss}
            title="Dismiss notification for this stage"
            aria-label="Dismiss banner"
          >
            ✕
          </button>
        </div>

        <div className="stage-concept-banner__content">
          <h3 className="stage-concept-banner__title">{concept.concept}</h3>
          <p className="stage-concept-banner__notice">
            ⚠️ <strong>Training Pace Notice:</strong> Initial learning rate and episode success may appear slow as the policy acquires this new skill. Zero-shot baseline scores are expected to be low.
          </p>
          <div className="stage-concept-banner__details">
            <div className="stage-concept-banner__detail-item">
              <span className="stage-concept-banner__detail-label">Concept Focus:</span>
              <span className="stage-concept-banner__detail-text">{concept.description}</span>
            </div>
            <div className="stage-concept-banner__detail-item">
              <span className="stage-concept-banner__detail-label">Key Challenge:</span>
              <span className="stage-concept-banner__detail-text">{concept.keyChange}</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

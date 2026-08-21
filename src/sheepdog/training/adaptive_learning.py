"""Adaptive step-size and learning rate controller for herding policies.

Conservatively dampens learning rate (MaskablePPO) and mutation scale (Hill-Climbing)
across discrete stages as herding success improves, preventing catastrophic policy
collapse and regression during fine-tuning.

Automatically resets back to Stage 1 of 4 (1.00x base exploration) whenever a new
curriculum stage begins to preserve adaptation plasticity.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


ADAPTIVE_STAGES: tuple[dict[str, Any], ...] = (
    {
        "stage": 1,
        "min_success": 0.0,
        "max_success": 0.60,
        "multiplier": 1.00,
        "name": "Base Exploration",
        "description": "No modification — full learning plasticity",
    },
    {
        "stage": 2,
        "min_success": 0.60,
        "max_success": 0.75,
        "multiplier": 0.80,
        "name": "Stabilizing",
        "description": "Mild damping — stabilizing flock gathering",
    },
    {
        "stage": 3,
        "min_success": 0.75,
        "max_success": 0.90,
        "multiplier": 0.65,
        "name": "Refining",
        "description": "Moderate damping — refining gate corridor approach",
    },
    {
        "stage": 4,
        "min_success": 0.90,
        "max_success": 1.01,
        "multiplier": 0.50,
        "name": "Holding Gate",
        "description": "Precision hold — minimizing pen bounce for auto-promotion",
    },
)

MAX_ADAPTIVE_STAGES = len(ADAPTIVE_STAGES)


@dataclass(frozen=True, slots=True)
class AdaptiveStepState:
    """Current state snapshot of the adaptive step controller."""

    stage: int
    max_stages: int
    multiplier: float
    name: str
    description: str
    label: str
    curriculum_stage: int
    effective_learning_rate: float
    effective_mutation_scale: float
    ema_success_rate: float
    consecutive_hits: int

    def to_dict(self) -> dict[str, Any]:
        """Serialize state to plain dictionary."""
        return asdict(self)


def _target_stage_for_success(success_rate: float) -> int:
    """Determine target adaptive stage for a given raw or smoothed success rate."""
    if success_rate >= 0.90:
        return 4
    if success_rate >= 0.75:
        return 3
    if success_rate >= 0.60:
        return 2
    return 1


class AdaptiveStepController:
    """Controller managing discrete, conservative learning rate and mutation adjustments."""

    def __init__(
        self,
        base_learning_rate: float = 1e-4,
        base_mutation_scale: float = 0.08,
        ema_alpha: float = 0.40,
        debounce_required_hits: int = 2,
        initial_curriculum_stage: int = 1,
    ) -> None:
        self.base_learning_rate = float(base_learning_rate)
        self.base_mutation_scale = float(base_mutation_scale)
        self.ema_alpha = float(ema_alpha)
        self.debounce_required_hits = max(1, int(debounce_required_hits))
        
        self.current_stage: int = 1
        self.curriculum_stage: int = int(initial_curriculum_stage)
        self.ema_success_rate: float = 0.0
        self.consecutive_hits: int = 0
        self.pending_target_stage: int = 1

    def reset_for_curriculum_stage(self, new_curriculum_stage: int) -> None:
        """Reset the adaptive controller to Stage 1 (1.0x) upon entering a new curriculum stage."""
        self.curriculum_stage = int(new_curriculum_stage)
        self.current_stage = 1
        self.pending_target_stage = 1
        self.consecutive_hits = 0
        self.ema_success_rate = 0.0

    def update(
        self,
        eval_success_rate: float,
        current_curriculum_stage: int,
    ) -> AdaptiveStepState:
        """Update the controller with a new evaluation success rate and return current state."""
        # 1. Automatic curriculum reset check
        if current_curriculum_stage != self.curriculum_stage:
            self.reset_for_curriculum_stage(current_curriculum_stage)
            return self.get_state()

        # 2. Update smoothed exponential moving average
        eval_success = max(0.0, min(1.0, float(eval_success_rate)))
        if self.ema_success_rate == 0.0 and self.current_stage == 1 and self.consecutive_hits == 0:
            self.ema_success_rate = eval_success
        else:
            self.ema_success_rate = (
                self.ema_alpha * eval_success + (1.0 - self.ema_alpha) * self.ema_success_rate
            )

        # 3. Determine proposed target stage based on raw and smoothed success
        raw_target = _target_stage_for_success(eval_success)
        ema_target = _target_stage_for_success(self.ema_success_rate)
        proposed_target = min(raw_target, ema_target)

        # 4. Debounce transitions
        if proposed_target > self.current_stage:
            # Accumulate confirmation count towards stepping up
            if proposed_target >= self.pending_target_stage:
                self.consecutive_hits += 1
                self.pending_target_stage = min(proposed_target, max(self.pending_target_stage, self.current_stage + 1))
            else:
                self.pending_target_stage = proposed_target
                self.consecutive_hits = 1

            if self.consecutive_hits >= self.debounce_required_hits:
                # Step up by at most 1 stage at a time for conservative progression
                self.current_stage = min(self.current_stage + 1, proposed_target)
                self.consecutive_hits = 0
                self.pending_target_stage = self.current_stage
        elif proposed_target < self.current_stage:
            # Performance degraded: release step size smoothly (at most 1 stage per check)
            self.current_stage = max(1, self.current_stage - 1)
            self.pending_target_stage = self.current_stage
            self.consecutive_hits = 0
        else:
            # Sustained in same stage
            self.pending_target_stage = self.current_stage
            self.consecutive_hits = 0

        return self.get_state()

    def get_state(self) -> AdaptiveStepState:
        """Construct the current state snapshot."""
        stage_idx = max(0, min(len(ADAPTIVE_STAGES) - 1, self.current_stage - 1))
        stage_def = ADAPTIVE_STAGES[stage_idx]
        multiplier = float(stage_def["multiplier"])
        name = str(stage_def["name"])
        desc = str(stage_def["description"])

        if self.current_stage == 1:
            label = f"Stage 1 of {MAX_ADAPTIVE_STAGES} ({multiplier:.2f}x • {name}) [No modification]"
        else:
            label = f"Stage {self.current_stage} of {MAX_ADAPTIVE_STAGES} ({multiplier:.2f}x • {name})"

        effective_lr = self.base_learning_rate * multiplier
        effective_mutation = self.base_mutation_scale * multiplier

        return AdaptiveStepState(
            stage=self.current_stage,
            max_stages=MAX_ADAPTIVE_STAGES,
            multiplier=multiplier,
            name=name,
            description=desc,
            label=label,
            curriculum_stage=self.curriculum_stage,
            effective_learning_rate=effective_lr,
            effective_mutation_scale=effective_mutation,
            ema_success_rate=round(self.ema_success_rate, 4),
            consecutive_hits=self.consecutive_hits,
        )

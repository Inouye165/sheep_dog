"""Named test scenarios with fixed layouts for replay evaluation."""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class AgentLayout:
    """Fixed position (and optional personality) for one agent."""

    index: int
    x: int
    y: int
    personality: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> AgentLayout:
        return cls(
            index=int(payload["index"]),
            x=int(payload["x"]),
            y=int(payload["y"]),
            personality=payload.get("personality"),
        )


@dataclass(frozen=True, slots=True)
class PenLayout:
    """Pen geometry for a saved scenario."""

    origin_x: int
    origin_y: int
    width: int
    height: int
    opening: str = "left"

    def to_dict(self) -> dict[str, Any]:
        return {
            "origin": {"x": self.origin_x, "y": self.origin_y},
            "origin_x": self.origin_x,
            "origin_y": self.origin_y,
            "width": self.width,
            "height": self.height,
            "opening": self.opening,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PenLayout:
        origin = payload.get("origin", {})
        if isinstance(origin, dict):
            origin_x = int(origin.get("x", payload.get("origin_x", 0)))
            origin_y = int(origin.get("y", payload.get("origin_y", 0)))
        else:
            origin_x = int(payload.get("origin_x", 0))
            origin_y = int(payload.get("origin_y", 0))
        return cls(
            origin_x=origin_x,
            origin_y=origin_y,
            width=int(payload["width"]),
            height=int(payload["height"]),
            opening=str(payload.get("opening", "left")),
        )


@dataclass(frozen=True, slots=True)
class Scenario:
    """Reproducible episode start: fixed agents, pen, seed, and field size."""

    id: str
    name: str
    created_at: str
    seed: int
    width: int
    height: int
    dogs: tuple[AgentLayout, ...]
    sheep: tuple[AgentLayout, ...]
    pen: PenLayout
    sheep_personality_strength: float = 0.0
    sheep_personality_seed_offset: int = 0
    seed_offset: int = 0
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "created_at": self.created_at,
            "seed": self.seed,
            "width": self.width,
            "height": self.height,
            "dogs": [agent.to_dict() for agent in self.dogs],
            "sheep": [agent.to_dict() for agent in self.sheep],
            "pen": self.pen.to_dict(),
            "sheep_personality_strength": self.sheep_personality_strength,
            "sheep_personality_seed_offset": self.sheep_personality_seed_offset,
            "seed_offset": self.seed_offset,
            "description": self.description,
            # Backward-compatible alias for older clients.
            "notes": self.description,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Scenario:
        description = str(payload.get("description", payload.get("notes", "")))
        return cls(
            id=str(payload["id"]),
            name=str(payload["name"]),
            created_at=str(payload.get("created_at", "")),
            seed=int(payload["seed"]),
            width=int(payload["width"]),
            height=int(payload["height"]),
            dogs=tuple(AgentLayout.from_dict(item) for item in payload["dogs"]),
            sheep=tuple(AgentLayout.from_dict(item) for item in payload["sheep"]),
            pen=PenLayout.from_dict(payload["pen"]),
            sheep_personality_strength=float(payload.get("sheep_personality_strength", 0.0)),
            sheep_personality_seed_offset=int(payload.get("sheep_personality_seed_offset", 0)),
            seed_offset=int(payload.get("seed_offset", 0)),
            description=description,
        )


# Backward-compatible alias used by environment and older imports.
SavedScenario = Scenario


def scenario_from_snapshot(
    *,
    name: str,
    seed: int,
    snapshot: dict[str, Any],
    sheep_personality_strength: float = 0.0,
    sheep_personality_seed_offset: int = 0,
    seed_offset: int = 0,
    description: str = "",
) -> Scenario:
    """Build a scenario from a replay frame or final snapshot dict."""

    dogs = tuple(
        AgentLayout(
            index=int(agent["index"]),
            x=int(agent["x"]),
            y=int(agent["y"]),
        )
        for agent in snapshot.get("dogs", [])
    )
    sheep = tuple(
        AgentLayout(
            index=int(agent["index"]),
            x=int(agent["x"]),
            y=int(agent["y"]),
            personality=agent.get("personality"),
        )
        for agent in snapshot.get("sheep", [])
    )
    width = int(snapshot.get("grid_width") or snapshot.get("field_width") or 80)
    height = int(snapshot.get("grid_height") or snapshot.get("field_height") or 60)
    return Scenario(
        id=uuid.uuid4().hex[:12],
        name=name,
        created_at=datetime.now(UTC).isoformat(),
        seed=seed,
        width=width,
        height=height,
        dogs=dogs,
        sheep=sheep,
        pen=PenLayout.from_dict(snapshot.get("pen", {})),
        sheep_personality_strength=sheep_personality_strength,
        sheep_personality_seed_offset=sheep_personality_seed_offset,
        seed_offset=seed_offset,
        description=description,
    )


class ScenarioStore:
    """Persist scenarios to ``<output>/scenarios/scenarios.json``."""

    FILENAME = "scenarios.json"

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / self.FILENAME
        if not self.path.exists():
            self._write([])

    def list(self) -> tuple[Scenario, ...]:
        payload = self._read()
        scenarios = payload.get("scenarios", [])
        if not isinstance(scenarios, list):
            return ()
        return tuple(Scenario.from_dict(item) for item in scenarios)

    def get(self, scenario_id: str) -> Scenario | None:
        for scenario in self.list():
            if scenario.id == scenario_id:
                return scenario
        return None

    def save(self, scenario: Scenario) -> Scenario:
        scenarios = list(self.list())
        replaced = False
        for index, existing in enumerate(scenarios):
            if existing.id == scenario.id:
                scenarios[index] = scenario
                replaced = True
                break
        if not replaced:
            scenarios.append(scenario)
        self._write(scenarios)
        return scenario

    def delete(self, scenario_id: str) -> bool:
        before = len(self.list())
        scenarios = [scenario for scenario in self.list() if scenario.id != scenario_id]
        if len(scenarios) == before:
            return False
        self._write(scenarios)
        return True

    def _read(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"scenarios": []}
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {"scenarios": []}
        return payload if isinstance(payload, dict) else {"scenarios": []}

    def _write(self, scenarios: list[Scenario]) -> None:
        payload = {"scenarios": [scenario.to_dict() for scenario in scenarios]}
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

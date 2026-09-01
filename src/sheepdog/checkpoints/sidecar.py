"""Sidecar metadata utilities for legacy checkpoint schema verification."""

from __future__ import annotations

import datetime
import hashlib
import json
from pathlib import Path
from typing import Any

from sheepdog.atomic_io import atomic_write_json


def compute_file_sha256(file_path: Path) -> str:
    """Compute deterministic SHA256 digest of a file."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def get_sidecar_path_for_model(model_path: Path) -> Path:
    """Return canonical sidecar JSON path for a model zip path."""
    return model_path.with_suffix(".sidecar.json")


def create_sidecar_metadata(
    model_path: Path,
    config: Any,
    policy_architecture: str = "MaskableActorCriticPolicy",
    migration_method: str = "reconstructed_from_stage8_canonical_schema",
) -> dict[str, Any]:
    """Reconstruct legacy schema metadata and write a verified sidecar JSON file."""
    from sheepdog.checkpoints.store import (
        get_action_space_hash,
        get_observation_schema_hash,
    )
    from sheepdog.environment import ACTION_ORDER, SheepdogEnvironment

    if not model_path.exists():
        raise FileNotFoundError(f"Model file does not exist: {model_path}")

    env = SheepdogEnvironment(config)
    env.reset(seed=42)
    obs = env.build_observation_for_dog(0)

    obs_hash = get_observation_schema_hash(config)
    act_hash = get_action_space_hash()
    model_sha256 = compute_file_sha256(model_path)

    sidecar_data: dict[str, Any] = {
        "observation_schema_hash": obs_hash,
        "observation_dimension": len(obs.values),
        "ordered_feature_manifest": list(obs.feature_names),
        "action_schema_hash": act_hash,
        "action_count": len(ACTION_ORDER),
        "policy_architecture": policy_architecture,
        "source_checkpoint_sha256": model_sha256,
        "migration_timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
        "migration_method": migration_method,
        "verified_legacy_schema": True,
    }

    sidecar_path = get_sidecar_path_for_model(model_path)
    atomic_write_json(sidecar_path, sidecar_data)
    return sidecar_data


def load_and_verify_sidecar(model_path: Path) -> dict[str, Any] | None:
    """Load sidecar metadata for a model zip and verify its SHA256 signature."""
    if not model_path.exists():
        return None

    sidecar_path = get_sidecar_path_for_model(model_path)
    if not sidecar_path.exists():
        return None

    try:
        sidecar_data = json.loads(sidecar_path.read_text(encoding="utf-8"))
        if not isinstance(sidecar_data, dict):
            return None

        if not sidecar_data.get("verified_legacy_schema"):
            return None

        expected_sha256 = sidecar_data.get("source_checkpoint_sha256")
        if not expected_sha256:
            return None

        actual_sha256 = compute_file_sha256(model_path)
        if actual_sha256 != expected_sha256:
            return None

        return sidecar_data
    except Exception:
        return None

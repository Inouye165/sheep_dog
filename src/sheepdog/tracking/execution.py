"""Runtime execution and startup tracking for Sheepdog training runs.

Captures verified execution context at process startup (Git commit hash, dirty worktree status,
patch hash) and records active runtime activations in the tracking registry.
"""

from __future__ import annotations

import datetime
import logging
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sheepdog.tracking.registry import (
    ActivationRecord,
    ExperimentRecord,
    TrackingRegistry,
    capture_current_git_provenance,
)

logger = logging.getLogger(__name__)

# Process-level startup boundary tracking
_PROCESS_STARTUP_PROVENANCE: tuple[str | None, int, str | None] = (None, 0, None)
_BOUNDARY_EXPLICITLY_ESTABLISHED: bool = False


def establish_process_startup_boundary(repo_dir: str | Path | None = None) -> tuple[str | None, int, str | None]:
    """Explicitly capture and establish process startup source provenance boundary."""
    global _PROCESS_STARTUP_PROVENANCE, _BOUNDARY_EXPLICITLY_ESTABLISHED
    if _BOUNDARY_EXPLICITLY_ESTABLISHED:
        return _PROCESS_STARTUP_PROVENANCE
    _PROCESS_STARTUP_PROVENANCE = capture_current_git_provenance(cwd=repo_dir)
    _BOUNDARY_EXPLICITLY_ESTABLISHED = True
    return _PROCESS_STARTUP_PROVENANCE


def get_process_startup_provenance() -> tuple[str | None, int, str | None]:
    """Return process initialization git provenance."""
    return _PROCESS_STARTUP_PROVENANCE


def is_process_startup_boundary_established() -> bool:
    """Return True if the process startup boundary was explicitly established."""
    return _BOUNDARY_EXPLICITLY_ESTABLISHED


def reset_process_startup_boundary() -> None:
    """Reset the process startup boundary to unestablished state."""
    global _PROCESS_STARTUP_PROVENANCE, _BOUNDARY_EXPLICITLY_ESTABLISHED
    _PROCESS_STARTUP_PROVENANCE = (None, 0, None)
    _BOUNDARY_EXPLICITLY_ESTABLISHED = False


# Backward compatibility aliases
def get_process_provenance() -> tuple[str | None, int, str | None]:
    """Return process initialization git provenance (backward compatibility alias)."""
    return get_process_startup_provenance()


def set_process_provenance(prov: tuple[str | None, int, str | None]) -> None:
    """Set process initialization git provenance (backward compatibility alias)."""
    global _PROCESS_STARTUP_PROVENANCE, _BOUNDARY_EXPLICITLY_ESTABLISHED
    _PROCESS_STARTUP_PROVENANCE = prov
    _BOUNDARY_EXPLICITLY_ESTABLISHED = True


@dataclass(frozen=True, slots=True)
class ExecutionContext:
    """Immutable execution context established at process startup."""

    experiment_id: str
    activation_id: str
    run_id: str
    status: str = "active"
    change_id: str | None = None
    source_revision: str | None = None
    dirty_worktree: int = 0
    worktree_patch_hash: str | None = None
    verified_at: str = ""
    is_resumed: bool = False
    verification_notes: str | None = None


class RuntimeExecutionTracker:
    """Manages process startup capture and verified activation records."""

    def __init__(self, registry: TrackingRegistry | None = None) -> None:
        self.registry = registry

    @classmethod
    def from_db(cls, db_path: str | Path | sqlite3.Connection | None) -> RuntimeExecutionTracker:
        """Create tracker with registry attached to db_path if provided."""
        if db_path is None:
            return cls(registry=None)
        try:
            reg = TrackingRegistry(db_path)
            return cls(registry=reg)
        except Exception as exc:
            logger.warning("Could not initialize TrackingRegistry from %s: %s", db_path, exc)
            return cls(registry=None)

    def initialize_or_resume(
        self,
        run_id: str,
        *,
        experiment_id: str | None = None,
        change_id: str | None = None,
        starting_episode: int = 0,
        checkpoint_label_episode: int | None = None,
        starting_policy_transition: int = 0,
        starting_world_step: int | None = None,
        policy_state_path: str | None = None,
        weights_sha256: str | None = None,
        config_hash: str | None = None,
        is_resumed: bool = False,
        repo_dir: str | Path | None = None,
        target_stage: int | None = None,
        assistance_mode: str = "guided",
        hyperparameters: dict[str, Any] | None = None,
    ) -> ExecutionContext:
        """Capture startup provenance and establish verified runtime activation.

        Compares workspace disk provenance against the process initialization provenance.
        If diverged, records divergence in verification_notes and sets status='unverified'.
        If an experiment already exists for experiment_id or run_id, reuses it to
        prevent duplicate experiment records.
        """
        now_utc = datetime.datetime.now(datetime.UTC).isoformat()
        timestamp_slug = datetime.datetime.now(datetime.UTC).strftime("%Y%m%d_%H%M%S_%f")

        # Compare current disk provenance against process initialization provenance
        current_rev, current_dirty, current_patch = capture_current_git_provenance(cwd=repo_dir)

        # Model weights verification note
        weights_sha_desc = (
            f"weights SHA-256: {weights_sha256}"
            if weights_sha256
            else "weights SHA unverified"
        )
        weights_desc = (
            f"Loaded weights path: {policy_state_path or 'initial_scratch'} ({weights_sha_desc}); "
            f"config hash: {config_hash or 'unspecified'}"
        )

        if not _BOUNDARY_EXPLICITLY_ESTABLISHED:
            status = "unverified"
            workspace_diverged = False
            proc_rev, proc_dirty, proc_patch = (None, 0, None)
            retained_rev = None
            retained_dirty = current_dirty
            retained_patch = current_patch
            notes = (
                "Process startup boundary was not explicitly established prior to training module execution; "
                f"loaded-code identity is unverified. {weights_desc}"
            )
        else:
            proc_rev, proc_dirty, proc_patch = _PROCESS_STARTUP_PROVENANCE
            workspace_diverged = (
                current_rev != proc_rev
                or current_dirty != proc_dirty
                or current_patch != proc_patch
            )

            if workspace_diverged:
                status = "unverified"
                # Retain process startup provenance - do NOT substitute later workspace contents for loaded code!
                retained_rev = proc_rev
                retained_dirty = proc_dirty
                retained_patch = proc_patch
                notes = (
                    f"Workspace diverged from process start (process: rev={proc_rev[:8] if proc_rev else 'unknown'}, "
                    f"dirty={proc_dirty}, patch={proc_patch[:8] if proc_patch else 'none'}; "
                    f"disk: rev={current_rev[:8] if current_rev else 'unknown'}, dirty={current_dirty}, "
                    f"patch={current_patch[:8] if current_patch else 'none'}). "
                    f"Running process cannot verify loaded code identity for later disk changes. "
                    f"{weights_desc}"
                )
            else:
                status = "active"
                retained_rev = proc_rev
                retained_dirty = proc_dirty
                retained_patch = proc_patch
                notes = (
                    f"PID {os.getpid()} runtime activation: process source rev={proc_rev[:8] if proc_rev else 'unknown'}, "
                    f"dirty={proc_dirty}. {weights_desc}"
                )

        # Determine experiment identity
        exp_id = experiment_id or f"exp_{run_id}"

        # Determine activation identity
        activation_id = f"act_{run_id}_{timestamp_slug}_{os.getpid()}"
        method = "trainer_resume" if is_resumed else "trainer_startup"

        ctx = ExecutionContext(
            experiment_id=exp_id,
            activation_id=activation_id,
            run_id=run_id,
            status=status,
            change_id=change_id,
            source_revision=retained_rev,
            dirty_worktree=retained_dirty,
            worktree_patch_hash=retained_patch,
            verified_at=now_utc if status == "active" else "",
            is_resumed=is_resumed,
            verification_notes=notes,
        )

        if self.registry is not None:
            try:
                # Ensure foreign key validity for change_id
                valid_change_id = None
                if change_id:
                    existing_chg = self.registry.get_change(change_id)
                    if existing_chg:
                        valid_change_id = change_id

                # Ensure experiment record exists without duplicating
                existing_exp = self.registry.get_experiment(exp_id)
                if not existing_exp:
                    exp_record = ExperimentRecord(
                        experiment_id=exp_id,
                        change_id=valid_change_id,
                        created_at=now_utc,
                        name=f"Training Run {run_id}",
                        hypothesis=None,
                        parent_checkpoint_id=None,
                        parent_checkpoint_path=policy_state_path,
                        assistance_mode=assistance_mode,
                        target_stage=target_stage,
                        hyperparameters=hyperparameters,
                        status="active",
                        metadata={"run_id": run_id, "startup_pid": os.getpid()},
                    )
                    self.registry.record_experiment(exp_record)

                # Record verified activation
                act_record = ActivationRecord(
                    activation_id=activation_id,
                    experiment_id=exp_id,
                    change_id=valid_change_id,
                    run_id=run_id,
                    status=status,
                    activation_method=method,
                    verified_at=now_utc if status == "active" else None,
                    activated_at_episode=starting_episode,
                    activated_at_policy_transition=starting_policy_transition,
                    activated_at_world_step=starting_world_step,
                    verified_source_revision=retained_rev,
                    verification_notes=notes,
                    metadata={
                        "pid": os.getpid(),
                        "checkpoint_label_episode": checkpoint_label_episode,
                        "process_source_revision": proc_rev,
                        "process_dirty_worktree": proc_dirty,
                        "process_worktree_patch_hash": proc_patch,
                        "disk_source_revision": current_rev,
                        "disk_dirty_worktree": current_dirty,
                        "disk_worktree_patch_hash": current_patch,
                        "workspace_diverged": workspace_diverged,
                        "boundary_explicitly_established": _BOUNDARY_EXPLICITLY_ESTABLISHED,
                        "is_resumed": is_resumed,
                        "policy_state_path": policy_state_path,
                        "weights_sha256": weights_sha256,
                        "config_hash": config_hash,
                        "world_steps_status": "unmeasured" if starting_world_step is None else "measured",
                    },
                )
                self.registry.record_activation(act_record)
            except Exception as exc:
                logger.warning("Could not persist execution activation to registry: %s", exc)

        return ctx

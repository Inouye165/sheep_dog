"""Automated stage milestone and hourly snapshot backup manager for sheepdog training."""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sheepdog.atomic_io import atomic_write_json

logger = logging.getLogger("sheepdog.training.backup")


class TrainingBackupManager:
    """Manages immutable completed stage backups and rolling hourly training snapshots."""

    def __init__(self, backup_root: str | Path = "artifacts/backups") -> None:
        self.backup_root = Path(backup_root).resolve()
        self.stages_dir = self.backup_root / "stages"
        self.hourly_dir = self.backup_root / "hourly"
        self._ensure_dirs()
        self._last_hourly_backup_time: float = time.time()
        self._last_hourly_active_seconds: float = 0.0

    def _ensure_dirs(self) -> None:
        """Ensure backup directory structure exists."""
        self.stages_dir.mkdir(parents=True, exist_ok=True)
        self.hourly_dir.mkdir(parents=True, exist_ok=True)

    def backup_completed_stage(
        self,
        stage: int,
        model_path: str | Path | None = None,
        checkpoint_payload: dict[str, Any] | None = None,
        evaluation_payload: dict[str, Any] | None = None,
        config_dict: dict[str, Any] | None = None,
        metrics: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create an immutable milestone backup of a completed curriculum stage."""
        self._ensure_dirs()
        stage_dir = self.stages_dir / f"stage_{stage}"
        stage_dir.mkdir(parents=True, exist_ok=True)

        copied_model_path: str | None = None
        if model_path:
            src_model = Path(model_path)
            if src_model.exists():
                dst_model = stage_dir / f"stage_{stage}_best_model.zip"
                try:
                    shutil.copy2(src_model, dst_model)
                    copied_model_path = str(dst_model)
                    logger.info("Backed up Stage %d model to %s", stage, dst_model)
                    # Also copy sidecar if present
                    sidecar = src_model.with_suffix(".zip.sidecar.json")
                    if not sidecar.exists():
                        sidecar = src_model.with_name(f"{src_model.name}.sidecar.json")
                    if sidecar.exists():
                        shutil.copy2(sidecar, stage_dir / f"stage_{stage}_best_model.zip.sidecar.json")
                except Exception as exc:
                    logger.warning("Failed to copy model for Stage %d backup: %s", stage, exc)

        if checkpoint_payload:
            dst_chk = stage_dir / f"stage_{stage}_checkpoint.json"
            atomic_write_json(dst_chk, checkpoint_payload)

        if evaluation_payload:
            dst_eval = stage_dir / f"stage_{stage}_evaluation.json"
            atomic_write_json(dst_eval, evaluation_payload)

        if config_dict:
            dst_cfg = stage_dir / f"stage_{stage}_config.json"
            atomic_write_json(dst_cfg, config_dict)

        now_utc = datetime.now(timezone.utc).isoformat()
        manifest = {
            "stage": stage,
            "stage_name": f"Stage {stage}",
            "completed_at": now_utc,
            "has_model": copied_model_path is not None,
            "model_path": copied_model_path,
            "metrics": metrics or {},
            "checkpoint_id": (checkpoint_payload or {}).get("checkpoint_id"),
            "policy_version": (checkpoint_payload or {}).get("policy_version"),
            "success_rate": (checkpoint_payload or evaluation_payload or {}).get("success_rate"),
        }
        manifest_path = stage_dir / f"stage_{stage}_manifest.json"
        atomic_write_json(manifest_path, manifest)
        logger.info("Completed Stage %d milestone backup in %s", stage, stage_dir)
        return manifest

    def backup_hourly_snapshot(
        self,
        stage: int,
        model_path: str | Path | None,
        training_state: dict[str, Any] | None = None,
        checkpoint_payload: dict[str, Any] | None = None,
        active_runtime_seconds: float | None = None,
        interval_seconds: float = 3600.0,
        max_snapshots_per_stage: int = 24,
        force: bool = False,
    ) -> Path | None:
        """Create a rolling hourly backup of active training progress."""
        now = time.time()
        time_elapsed = now - self._last_hourly_backup_time

        if active_runtime_seconds is not None:
            active_elapsed = active_runtime_seconds - self._last_hourly_active_seconds
        else:
            active_elapsed = time_elapsed

        if not force and active_elapsed < interval_seconds and time_elapsed < interval_seconds:
            return None

        self._ensure_dirs()
        timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        snapshot_prefix = f"snapshot_stage{stage}_{timestamp_str}"

        dst_model: Path | None = None
        if model_path:
            src_model = Path(model_path)
            if src_model.exists():
                dst_model = self.hourly_dir / f"{snapshot_prefix}.zip"
                try:
                    shutil.copy2(src_model, dst_model)
                    sidecar = src_model.with_suffix(".zip.sidecar.json")
                    if not sidecar.exists():
                        sidecar = src_model.with_name(f"{src_model.name}.sidecar.json")
                    if sidecar.exists():
                        shutil.copy2(sidecar, self.hourly_dir / f"{snapshot_prefix}.zip.sidecar.json")
                except Exception as exc:
                    logger.warning("Failed to copy active model for hourly backup: %s", exc)

        metadata = {
            "snapshot_id": snapshot_prefix,
            "stage": stage,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model_saved": dst_model is not None and dst_model.exists(),
            "training_state": training_state or {},
            "checkpoint_payload": checkpoint_payload or {},
            "active_runtime_seconds": active_runtime_seconds,
        }
        metadata_path = self.hourly_dir / f"{snapshot_prefix}.json"
        atomic_write_json(metadata_path, metadata)

        self._last_hourly_backup_time = now
        if active_runtime_seconds is not None:
            self._last_hourly_active_seconds = active_runtime_seconds

        logger.info("Saved hourly training snapshot: %s", snapshot_prefix)
        self.prune_old_hourly_snapshots(stage=stage, max_keep=max_snapshots_per_stage)
        return metadata_path

    def prune_old_hourly_snapshots(self, stage: int | None = None, max_keep: int = 24) -> int:
        """Prune older hourly snapshots to conserve disk space while keeping recent ones."""
        if not self.hourly_dir.exists() or max_keep <= 0:
            return 0

        pattern = f"snapshot_stage{stage}_*.json" if stage is not None else "snapshot_stage*.json"
        meta_files = sorted(self.hourly_dir.glob(pattern), key=lambda p: p.stat().st_mtime)

        pruned_count = 0
        if len(meta_files) > max_keep:
            to_remove = meta_files[:-max_keep]
            for meta_p in to_remove:
                base_stem = meta_p.stem
                try:
                    meta_p.unlink(missing_ok=True)
                    (self.hourly_dir / f"{base_stem}.zip").unlink(missing_ok=True)
                    (self.hourly_dir / f"{base_stem}.zip.sidecar.json").unlink(missing_ok=True)
                    pruned_count += 1
                except Exception:
                    pass
        return pruned_count

    def list_backups(self) -> dict[str, Any]:
        """List all available completed stage milestones and hourly snapshots."""
        self._ensure_dirs()
        stages: list[dict[str, Any]] = []
        if self.stages_dir.exists():
            for stage_folder in sorted(self.stages_dir.glob("stage_*")):
                if not stage_folder.is_dir():
                    continue
                manifest_path = stage_folder / f"{stage_folder.name}_manifest.json"
                if manifest_path.exists():
                    try:
                        with open(manifest_path, "r", encoding="utf-8") as f:
                            stages.append(json.load(f))
                    except Exception:
                        pass
                else:
                    try:
                        stg_num = int(stage_folder.name.split("_")[1])
                        stages.append({
                            "stage": stg_num,
                            "stage_name": f"Stage {stg_num}",
                            "completed_at": None,
                            "has_model": (stage_folder / f"stage_{stg_num}_best_model.zip").exists(),
                            "path": str(stage_folder),
                        })
                    except Exception:
                        pass

        hourly: list[dict[str, Any]] = []
        if self.hourly_dir.exists():
            for json_file in sorted(self.hourly_dir.glob("snapshot_*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
                try:
                    with open(json_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        data["metadata_file"] = json_file.name
                        data["model_file"] = f"{json_file.stem}.zip" if (self.hourly_dir / f"{json_file.stem}.zip").exists() else None
                        hourly.append(data)
                except Exception:
                    pass

        return {
            "completed_stages": stages,
            "hourly_snapshots": hourly,
            "backup_dir": str(self.backup_root),
        }

    def restore_stage_milestone(
        self,
        stage: int,
        output_root: str | Path = "artifacts",
    ) -> dict[str, Any]:
        """Restore a completed stage milestone back into the active workspace."""
        out = Path(output_root).resolve()
        stage_dir = self.stages_dir / f"stage_{stage}"
        if not stage_dir.exists():
            raise FileNotFoundError(f"Stage {stage} milestone backup does not exist in {stage_dir}")

        models_dir = out / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        checkpoints_dir = out / "checkpoints"
        checkpoints_dir.mkdir(parents=True, exist_ok=True)

        restored_model: Path | None = None
        src_model = stage_dir / f"stage_{stage}_best_model.zip"
        if src_model.exists():
            dst_best = models_dir / "best-model.zip"
            shutil.copy2(src_model, dst_best)
            restored_model = dst_best
            sidecar = stage_dir / f"stage_{stage}_best_model.zip.sidecar.json"
            if sidecar.exists():
                shutil.copy2(sidecar, models_dir / "best-model.zip.sidecar.json")

        src_chk = stage_dir / f"stage_{stage}_checkpoint.json"
        chk_payload: dict[str, Any] = {}
        if src_chk.exists():
            with open(src_chk, "r", encoding="utf-8") as f:
                chk_payload = json.load(f)

        src_manifest = stage_dir / f"stage_{stage}_manifest.json"
        manifest_payload: dict[str, Any] = {}
        if src_manifest.exists():
            with open(src_manifest, "r", encoding="utf-8") as f:
                manifest_payload = json.load(f)

        return {
            "stage": stage,
            "restored_model_path": str(restored_model) if restored_model else None,
            "checkpoint": chk_payload,
            "manifest": manifest_payload,
            "status": "success",
            "message": f"Restored Stage {stage} milestone model and configuration.",
        }

    def restore_hourly_snapshot(
        self,
        snapshot_id: str,
        output_root: str | Path = "artifacts",
    ) -> dict[str, Any]:
        """Restore a specific hourly snapshot model and state back into active workspace."""
        out = Path(output_root).resolve()
        clean_id = Path(snapshot_id).stem
        json_path = self.hourly_dir / f"{clean_id}.json"
        zip_path = self.hourly_dir / f"{clean_id}.zip"

        if not json_path.exists():
            raise FileNotFoundError(f"Snapshot metadata {json_path} not found")

        with open(json_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        models_dir = out / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        restored_model: Path | None = None
        if zip_path.exists():
            dst_best = models_dir / "best-model.zip"
            shutil.copy2(zip_path, dst_best)
            restored_model = dst_best
            sidecar = self.hourly_dir / f"{clean_id}.zip.sidecar.json"
            if sidecar.exists():
                shutil.copy2(sidecar, models_dir / "best-model.zip.sidecar.json")

        return {
            "snapshot_id": clean_id,
            "stage": metadata.get("stage"),
            "restored_model_path": str(restored_model) if restored_model else None,
            "metadata": metadata,
            "status": "success",
            "message": f"Restored snapshot {clean_id} as active model.",
        }

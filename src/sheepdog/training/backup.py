"""Automated stage milestone and hourly snapshot backup manager for sheepdog training."""

from __future__ import annotations

import contextlib
import json
import logging
import shutil
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sheepdog.atomic_io import atomic_replace, atomic_write_json
from sheepdog.checkpoints.sidecar import get_sidecar_path_for_model

logger = logging.getLogger("sheepdog.training.backup")


def _find_sidecar_for_model(model_path: Path) -> Path | None:
    """Return existing canonical (`*.sidecar.json`) or legacy (`*.zip.sidecar.json`) sidecar."""
    canonical = get_sidecar_path_for_model(model_path)
    if canonical.exists():
        return canonical
    legacy = model_path.with_name(f"{model_path.name}.sidecar.json")
    if legacy.exists():
        return legacy
    return None


def _atomic_copy_file(src: Path, dst: Path) -> None:
    """Atomically copy *src* to *dst* via a unique sibling temporary file."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(f"{dst.stem}-{uuid.uuid4().hex}.tmp")
    try:
        shutil.copy2(src, tmp)
        atomic_replace(tmp, dst)
    except BaseException:
        with contextlib.suppress(OSError):
            tmp.unlink(missing_ok=True)
        raise


def _copy_model_with_canonical_sidecar(src_model: Path, dst_model: Path) -> None:
    """Copy *src_model* to *dst_model* and write canonical sidecar if present."""
    _atomic_copy_file(src_model, dst_model)
    sidecar = _find_sidecar_for_model(src_model)
    if sidecar is not None:
        _atomic_copy_file(sidecar, get_sidecar_path_for_model(dst_model))


def _resolve_path_under_output_root(raw_path: str | Path, output_root: Path) -> Path | None:
    """Resolve *raw_path* if and only if it stays within *output_root*."""
    p = Path(raw_path)
    if p.is_absolute():
        resolved = p.resolve()
        return resolved if resolved.is_relative_to(output_root) else None
    direct = (output_root / p).resolve()
    if p.parts and p.parts[0] == output_root.name:
        via_parent = (output_root.parent / p).resolve()
        if via_parent.is_relative_to(output_root):
            return via_parent
    return direct if direct.is_relative_to(output_root) else None


class TrainingBackupManager:
    """Manages immutable completed stage backups and rolling hourly training snapshots."""

    def __init__(
        self,
        backup_root: str | Path = "artifacts/backups",
        output_root: str | Path | None = None,
    ) -> None:
        self.backup_root = Path(backup_root).resolve()
        self.output_root = (
            Path(output_root).resolve()
            if output_root is not None
            else self.backup_root.parent
        )
        self.stages_dir = self.backup_root / "stages"
        self.hourly_dir = self.backup_root / "hourly"
        self._ensure_dirs()
        self._last_hourly_backup_time: float = time.time()
        self._last_hourly_active_seconds: float = 0.0
        self._seed_hourly_backup_watermark()

    def _seed_hourly_backup_watermark(self) -> None:
        """Seed hourly backup watermarks from the latest valid snapshot on disk."""
        if not self.hourly_dir.exists():
            return
        dated_snapshots: list[tuple[float, Path]] = []
        for snap_path in self.hourly_dir.glob("snapshot_stage*.json"):
            try:
                if snap_path.is_file():
                    dated_snapshots.append((float(snap_path.stat().st_mtime), snap_path))
            except OSError:
                continue
        dated_snapshots.sort(key=lambda item: item[0])
        for latest_mtime, latest_path in reversed(dated_snapshots):
            try:
                payload = json.loads(latest_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if not isinstance(payload, dict):
                continue
            self._last_hourly_backup_time = float(latest_mtime)
            active_sec = payload.get("active_runtime_seconds")
            if isinstance(active_sec, (int, float)) and not isinstance(active_sec, bool):
                self._last_hourly_active_seconds = float(active_sec)
            return

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
                canonical_sidecar_dst = get_sidecar_path_for_model(dst_model)
                legacy_sidecar_dst = dst_model.with_name(f"{dst_model.name}.sidecar.json")
                try:
                    _atomic_copy_file(src_model, dst_model)
                    # Also copy sidecar if present (canonical and legacy naming)
                    sidecar = _find_sidecar_for_model(src_model)
                    if sidecar is not None:
                        _atomic_copy_file(sidecar, canonical_sidecar_dst)
                        _atomic_copy_file(sidecar, legacy_sidecar_dst)
                    copied_model_path = str(dst_model)
                    logger.info("Backed up Stage %d model to %s", stage, dst_model)
                except Exception as exc:
                    for cleanup_path in (dst_model, canonical_sidecar_dst, legacy_sidecar_dst):
                        with contextlib.suppress(OSError):
                            cleanup_path.unlink(missing_ok=True)
                    copied_model_path = None
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

        now_utc = datetime.now(UTC).isoformat()
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
        timestamp_str = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")
        snapshot_prefix = f"snapshot_stage{stage}_{timestamp_str}"

        dst_model: Path | None = None
        model_copied = False
        if model_path:
            src_model = Path(model_path)
            if src_model.exists():
                dst_model = self.hourly_dir / f"{snapshot_prefix}.zip"
                canonical_sidecar_dst = get_sidecar_path_for_model(dst_model)
                legacy_sidecar_dst = dst_model.with_name(f"{dst_model.name}.sidecar.json")
                try:
                    _atomic_copy_file(src_model, dst_model)
                    sidecar = _find_sidecar_for_model(src_model)
                    if sidecar is not None:
                        _atomic_copy_file(sidecar, canonical_sidecar_dst)
                        _atomic_copy_file(sidecar, legacy_sidecar_dst)
                    model_copied = True
                except Exception as exc:
                    for cleanup_path in (dst_model, canonical_sidecar_dst, legacy_sidecar_dst):
                        with contextlib.suppress(OSError):
                            cleanup_path.unlink(missing_ok=True)
                    model_copied = False
                    logger.warning("Failed to copy active model for hourly backup: %s", exc)

        metadata = {
            "snapshot_id": snapshot_prefix,
            "stage": stage,
            "created_at": datetime.now(UTC).isoformat(),
            "model_saved": model_copied,
            "training_state": training_state or {},
            "checkpoint_payload": checkpoint_payload or {},
            "active_runtime_seconds": active_runtime_seconds,
        }
        metadata_path = self.hourly_dir / f"{snapshot_prefix}.json"
        try:
            atomic_write_json(metadata_path, metadata)
        except BaseException:
            if dst_model is not None and model_copied:
                for cleanup_path in (
                    dst_model,
                    get_sidecar_path_for_model(dst_model),
                    dst_model.with_name(f"{dst_model.name}.sidecar.json"),
                ):
                    with contextlib.suppress(OSError):
                        cleanup_path.unlink(missing_ok=True)
            raise

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
                    zip_file = self.hourly_dir / f"{base_stem}.zip"
                    meta_p.unlink(missing_ok=True)
                    zip_file.unlink(missing_ok=True)
                    get_sidecar_path_for_model(zip_file).unlink(missing_ok=True)
                    (self.hourly_dir / f"{base_stem}.zip.sidecar.json").unlink(missing_ok=True)
                    pruned_count += 1
                except Exception as exc:
                    logger.warning("Failed to prune hourly snapshot %s: %s", meta_p, exc)
        return pruned_count

    _prune_hourly_backups = prune_old_hourly_snapshots

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
                        with open(manifest_path, encoding="utf-8") as f:
                            stages.append(json.load(f))
                    except Exception as exc:
                        logger.warning(
                            "Failed to read stage backup manifest %s: %s", manifest_path, exc
                        )
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
                    except Exception as exc:
                        logger.warning(
                            "Failed to inspect stage backup directory %s: %s", stage_folder, exc
                        )

        hourly: list[dict[str, Any]] = []
        if self.hourly_dir.exists():
            for json_file in sorted(self.hourly_dir.glob("snapshot_*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
                try:
                    with open(json_file, encoding="utf-8") as f:
                        data = json.load(f)
                        data["metadata_file"] = json_file.name
                        data["model_file"] = f"{json_file.stem}.zip" if (self.hourly_dir / f"{json_file.stem}.zip").exists() else None
                        hourly.append(data)
                except Exception as exc:
                    logger.warning("Failed to read hourly backup metadata %s: %s", json_file, exc)

        return {
            "completed_stages": stages,
            "hourly_snapshots": hourly,
            "backup_dir": str(self.backup_root),
        }

    def restore_stage_milestone(
        self,
        stage: int,
        output_root: str | Path | None = None,
    ) -> dict[str, Any]:
        """Restore a completed stage milestone back into the active workspace."""
        out = Path(output_root).resolve() if output_root is not None else self.output_root
        stage_dir = self.stages_dir / f"stage_{stage}"
        if not stage_dir.exists():
            raise FileNotFoundError(f"Stage {stage} milestone backup does not exist in {stage_dir}")

        models_dir = out / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        checkpoints_dir = out / "checkpoints"
        checkpoints_dir.mkdir(parents=True, exist_ok=True)

        src_chk = stage_dir / f"stage_{stage}_checkpoint.json"
        chk_payload: dict[str, Any] = {}
        if src_chk.exists():
            with open(src_chk, encoding="utf-8") as f:
                chk_payload = json.load(f)

        src_manifest = stage_dir / f"stage_{stage}_manifest.json"
        manifest_payload: dict[str, Any] = {}
        if src_manifest.exists():
            with open(src_manifest, encoding="utf-8") as f:
                manifest_payload = json.load(f)

        training_state = manifest_payload.get("training_state") or chk_payload.get("training_state")

        restored_model: Path | None = None
        src_model = stage_dir / f"stage_{stage}_best_model.zip"
        if src_model.exists():
            dst_best = models_dir / "best-model.zip"
            _copy_model_with_canonical_sidecar(src_model, dst_best)
            restored_model = dst_best
            if stage >= 9:
                stage_best = checkpoints_dir / f"stage{stage}" / f"stage{stage}-best-model.zip"
                _copy_model_with_canonical_sidecar(src_model, stage_best)
                restored_model = stage_best
                if isinstance(training_state, dict) and training_state.get("policy_state_path"):
                    psp = _resolve_path_under_output_root(training_state["policy_state_path"], out)
                    if psp is not None and psp != stage_best and psp != dst_best:
                        _copy_model_with_canonical_sidecar(src_model, psp)

        if isinstance(training_state, dict) and training_state:
            updated_state = dict(training_state)
            if restored_model is not None:
                updated_state["best_model_path"] = str(restored_model)
                updated_state["best_model_curriculum_stage"] = int(stage)
                psp_raw = updated_state.get("policy_state_path")
                psp_resolved = (
                    _resolve_path_under_output_root(psp_raw, out) if psp_raw else None
                )
                if psp_resolved is None or not psp_resolved.exists():
                    updated_state["policy_state_path"] = str(restored_model)
            atomic_write_json(out / "training-state.json", updated_state)

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
        output_root: str | Path | None = None,
    ) -> dict[str, Any]:
        """Restore a specific hourly snapshot model and state back into active workspace."""
        out = Path(output_root).resolve() if output_root is not None else self.output_root
        clean_id = Path(snapshot_id).stem
        json_path = self.hourly_dir / f"{clean_id}.json"
        zip_path = self.hourly_dir / f"{clean_id}.zip"

        if not json_path.exists():
            raise FileNotFoundError(f"Snapshot metadata {json_path} not found")

        with open(json_path, encoding="utf-8") as f:
            metadata = json.load(f)

        raw_stage = metadata.get("stage")
        snap_stage = int(raw_stage) if raw_stage is not None else None
        training_state = metadata.get("training_state")

        models_dir = out / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        restored_model: Path | None = None
        if zip_path.exists():
            dst_best = models_dir / "best-model.zip"
            _copy_model_with_canonical_sidecar(zip_path, dst_best)
            restored_model = dst_best
            if snap_stage is not None and snap_stage >= 9:
                stage_best = out / "checkpoints" / f"stage{snap_stage}" / f"stage{snap_stage}-best-model.zip"
                _copy_model_with_canonical_sidecar(zip_path, stage_best)
                restored_model = stage_best
            if isinstance(training_state, dict) and training_state.get("policy_state_path"):
                psp = _resolve_path_under_output_root(training_state["policy_state_path"], out)
                if psp is not None and psp != dst_best and psp != restored_model:
                    _copy_model_with_canonical_sidecar(zip_path, psp)

        if isinstance(training_state, dict) and training_state:
            updated_state = dict(training_state)
            if restored_model is not None:
                psp_raw = updated_state.get("policy_state_path")
                psp_resolved = (
                    _resolve_path_under_output_root(psp_raw, out) if psp_raw else None
                )
                if psp_resolved is None or not psp_resolved.exists():
                    updated_state["policy_state_path"] = str(restored_model)
                else:
                    updated_state["policy_state_path"] = str(psp_resolved)
                if snap_stage is not None and snap_stage >= 9:
                    bmp_raw = updated_state.get("best_model_path")
                    bmp_resolved = (
                        _resolve_path_under_output_root(bmp_raw, out) if bmp_raw else None
                    )
                    if bmp_resolved is None or not bmp_resolved.exists():
                        updated_state["best_model_path"] = str(restored_model)
                    updated_state.setdefault("best_model_curriculum_stage", snap_stage)
            atomic_write_json(out / "training-state.json", updated_state)

        return {
            "snapshot_id": clean_id,
            "stage": snap_stage,
            "restored_model_path": str(restored_model) if restored_model else None,
            "metadata": metadata,
            "status": "success",
            "message": f"Restored snapshot {clean_id} as active model.",
        }

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

from pydantic import ValidationError

from lisai.config import settings
from lisai.infra.fs.run_naming import parse_run_dir_name
from lisai.infra.paths import Paths
from lisai.infra.paths.model_subfolder import (
    group_path_from_model_subfolder,
    normalize_model_subfolder,
)

from .code_state import collect_code_state
from .identifiers import generate_run_id
from .io import read_run_metadata, write_run_metadata_atomic
from .schema import (
    SCHEMA_VERSION,
    LiveRuntimeStats,
    RunMetadata,
    RuntimeStats,
    TrainingSignature,
    utc_now,
)

_PATHS = Paths(settings)


def stored_run_path(run_dir: str | Path) -> str:
    path = Path(run_dir).resolve()
    data_root = _PATHS.data_root()

    try:
        return path.relative_to(data_root).as_posix()
    except ValueError:
        return path.as_posix()


def create_run_metadata(
    run_dir: str | Path,
    *,
    dataset: str,
    model_subfolder: str | None = None,
    max_epoch: int | None = None,
    group_path: str | None = None,
    preserve_existing: bool = False,
) -> RunMetadata:
    run_dir = Path(run_dir)
    run_name, run_index = parse_run_dir_name(run_dir.name)
    normalized_model_subfolder = normalize_model_subfolder(model_subfolder)
    if normalized_model_subfolder is None:
        normalized_model_subfolder = run_dir.parent.name.strip() or "unknown_model_subfolder"
    group_path = group_path if group_path is not None else group_path_from_model_subfolder(normalized_model_subfolder)
    path_text = stored_run_path(run_dir)
    now = utc_now()
    existing = _read_existing_metadata(run_dir) if preserve_existing else None

    if existing is None:
        metadata = RunMetadata(
            schema_version=SCHEMA_VERSION,
            run_id=generate_run_id(),
            run_name=run_name,
            run_index=run_index,
            dataset=dataset.strip(),
            model_subfolder=normalized_model_subfolder,
            status="running",
            closed_cleanly=False,
            created_at=now,
            updated_at=now,
            ended_at=None,
            last_heartbeat_at=now,
            last_epoch=None,
            max_epoch=max_epoch,
            best_val_loss=None,
            path=path_text,
            group_path=group_path,
            code=collect_code_state(),
        )
    else:
        metadata = existing.model_copy(
            update={
                "run_id": existing.run_id,
                "run_name": run_name,
                "run_index": run_index,
                "dataset": dataset.strip(),
                "model_subfolder": normalized_model_subfolder,
                "status": "running",
                "closed_cleanly": False,
                "updated_at": now,
                "ended_at": None,
                "last_heartbeat_at": now,
                "max_epoch": max_epoch if max_epoch is not None else existing.max_epoch,
                "path": path_text,
                "group_path": group_path,
            }
        )

    write_run_metadata_atomic(run_dir, metadata)
    return metadata


def update_run_heartbeat(run_dir: str | Path) -> RunMetadata:
    metadata = read_run_metadata(run_dir)
    now = utc_now()
    updated = metadata.model_copy(
        update={
            "status": "running",
            "closed_cleanly": False,
            "updated_at": now,
            "ended_at": None,
            "last_heartbeat_at": now,
        }
    )
    write_run_metadata_atomic(run_dir, updated)
    return updated


def update_run_progress(
    run_dir: str | Path,
    *,
    last_epoch: int | None = None,
    max_epoch: int | None = None,
    val_loss: float | None = None,
    epoch_duration_s: float | None = None,
) -> RunMetadata:
    metadata = read_run_metadata(run_dir)
    now = utc_now()
    best_val_loss = metadata.best_val_loss
    live_runtime_stats = metadata.live_runtime_stats
    if val_loss is not None:
        val_loss = float(val_loss)
        if best_val_loss is None or val_loss < best_val_loss:
            best_val_loss = val_loss
    if epoch_duration_s is not None:
        duration = float(epoch_duration_s)
        if duration < 0:
            raise ValueError("epoch_duration_s must be >= 0.")
        recent = [] if live_runtime_stats is None else list(live_runtime_stats.recent_epoch_durations_s)
        recent.append(duration)
        recent = recent[-3:]
        live_runtime_stats = LiveRuntimeStats(
            last_epoch_duration_s=duration,
            recent_epoch_durations_s=recent,
        )

    updated = metadata.model_copy(
        update={
            "status": "running",
            "closed_cleanly": False,
            "updated_at": now,
            "ended_at": None,
            "last_heartbeat_at": now,
            "last_epoch": metadata.last_epoch if last_epoch is None else last_epoch,
            "max_epoch": metadata.max_epoch if max_epoch is None else max_epoch,
            "best_val_loss": best_val_loss,
            "live_runtime_stats": live_runtime_stats,
        }
    )
    write_run_metadata_atomic(run_dir, updated)
    return updated


def update_run_runtime_details(
    run_dir: str | Path,
    *,
    training_signature: TrainingSignature | Mapping[str, object] | None = None,
    peak_gpu_mem_mb: int | None = None,
    total_training_time_sec: float | None = None,
    training_time_per_epoch_sec: float | None = None,
) -> RunMetadata:
    metadata = read_run_metadata(run_dir)
    now = utc_now()

    resolved_signature = metadata.training_signature
    if training_signature is not None:
        if isinstance(training_signature, TrainingSignature):
            resolved_signature = training_signature
        else:
            resolved_signature = TrainingSignature.model_validate(training_signature)

    resolved_stats = metadata.runtime_stats
    peak_mb_value = None if resolved_stats is None else resolved_stats.peak_gpu_mem_mb
    total_time_value = None if resolved_stats is None else resolved_stats.total_training_time_sec
    time_per_epoch_value = None if resolved_stats is None else resolved_stats.training_time_per_epoch_sec
    runtime_stats_changed = False

    if peak_gpu_mem_mb is not None:
        peak_mb = int(peak_gpu_mem_mb)
        if peak_mb < 0:
            raise ValueError("peak_gpu_mem_mb must be >= 0.")
        peak_mb_value = peak_mb if peak_mb_value is None else max(peak_mb_value, peak_mb)
        runtime_stats_changed = True

    if total_training_time_sec is not None:
        total_time = float(total_training_time_sec)
        if total_time < 0:
            raise ValueError("total_training_time_sec must be >= 0.")
        total_time_value = total_time
        runtime_stats_changed = True

    if training_time_per_epoch_sec is not None:
        per_epoch = float(training_time_per_epoch_sec)
        if per_epoch < 0:
            raise ValueError("training_time_per_epoch_sec must be >= 0.")
        time_per_epoch_value = per_epoch
        runtime_stats_changed = True

    if runtime_stats_changed:
        resolved_stats = RuntimeStats(
            peak_gpu_mem_mb=peak_mb_value,
            total_training_time_sec=total_time_value,
            training_time_per_epoch_sec=time_per_epoch_value,
        )

    updated = metadata.model_copy(
        update={
            "updated_at": now,
            "training_signature": resolved_signature,
            "runtime_stats": resolved_stats,
        }
    )
    write_run_metadata_atomic(run_dir, updated)
    return updated

def update_run_recovery_info(
    run_dir: str | Path,
    *,
    failure_reason: str | None = None,
    recovery_checkpoint_filename: str | None = None,
    recovery_strategy: str | None = None,
    last_safe_epoch: int | None = None,
    last_safe_batch_id: int | None = None,
    safe_resume_fail_count: int | None = None,
) -> RunMetadata:
    metadata = read_run_metadata(run_dir)
    now = utc_now()
    updated = metadata.model_copy(
        update={
            "updated_at": now,
            "failure_reason": failure_reason,
            "recovery_checkpoint_filename": recovery_checkpoint_filename,
            "recovery_strategy": recovery_strategy,
            "last_safe_epoch": last_safe_epoch,
            "last_safe_batch_id": last_safe_batch_id,
            "safe_resume_fail_count": (
                metadata.safe_resume_fail_count
                if safe_resume_fail_count is None
                else max(int(safe_resume_fail_count), 0)
            ),
        }
    )
    write_run_metadata_atomic(run_dir, updated)
    return updated


def update_run_failure_reason(
    run_dir: str | Path,
    *,
    failure_reason: str | None = None,
) -> RunMetadata:
    metadata = read_run_metadata(run_dir)
    now = utc_now()
    updated = metadata.model_copy(
        update={
            "updated_at": now,
            "failure_reason": failure_reason,
        }
    )
    write_run_metadata_atomic(run_dir, updated)
    return updated


def finalize_run_completed(run_dir: str | Path) -> RunMetadata:
    return _finalize_run(run_dir, status="completed")


def finalize_run_stopped(run_dir: str | Path) -> RunMetadata:
    return _finalize_run(run_dir, status="stopped")


def finalize_run_failed(run_dir: str | Path) -> RunMetadata:
    return _finalize_run(run_dir, status="failed")


def finalize_setup_failed(run_dir: str | Path) -> RunMetadata:
    return _finalize_run(run_dir, status="failed")


def _finalize_run(run_dir: str | Path, *, status: str) -> RunMetadata:
    metadata = read_run_metadata(run_dir)
    now = utc_now()
    updated = metadata.model_copy(
        update={
            "status": status,
            "closed_cleanly": True,
            "updated_at": now,
            "ended_at": now,
            "last_heartbeat_at": now,
        }
    )
    write_run_metadata_atomic(run_dir, updated)
    return updated


def _read_existing_metadata(run_dir: str | Path) -> RunMetadata | None:
    try:
        return read_run_metadata(run_dir)
    except (FileNotFoundError, OSError, ValidationError, ValueError, json.JSONDecodeError):
        return None


__all__ = [
    "create_run_metadata",
    "collect_code_state",
    "finalize_run_completed",
    "finalize_run_failed",
    "finalize_setup_failed",
    "finalize_run_stopped",
    "group_path_from_model_subfolder",
    "normalize_model_subfolder",
    "stored_run_path",
    "update_run_failure_reason",
    "update_run_heartbeat",
    "update_run_progress",
    "update_run_recovery_info",
    "update_run_runtime_details",
]

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Mapping

from pydantic import BaseModel, ConfigDict, Field, field_validator

from lisai.infra.paths import Paths

from .storage import ensure_queue_dirs

QUEUE_CONTROL_SCHEMA_VERSION = 1
QUEUE_CONTROL_FILENAME = "control.json"


class QueueControl(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = Field(default=QUEUE_CONTROL_SCHEMA_VERSION)
    paused: bool = False
    max_concurrent_runs_per_gpu: int = Field(default=1, ge=1)

    @field_validator("schema_version")
    @classmethod
    def _validate_schema_version(cls, value: int) -> int:
        if value != QUEUE_CONTROL_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported queue control schema_version {value!r}. "
                f"Expected {QUEUE_CONTROL_SCHEMA_VERSION}."
            )
        return value


def queue_control_path(*, queue_root: str | Path | None = None) -> Path:
    root = ensure_queue_dirs(queue_root=queue_root)
    return root / QUEUE_CONTROL_FILENAME


def default_queue_control(*, paths: Paths) -> QueueControl:
    return QueueControl(
        paused=bool(paths.settings.project_cfg.queue.paused),
        max_concurrent_runs_per_gpu=int(paths.settings.project_cfg.queue.max_concurrent_runs_per_gpu),
    )


def read_queue_control(*, queue_root: str | Path, paths: Paths) -> QueueControl:
    path = queue_control_path(queue_root=queue_root)
    if not path.exists():
        return default_queue_control(paths=paths)

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return QueueControl.model_validate(payload)


def write_queue_control(
    control: QueueControl | Mapping[str, object],
    *,
    queue_root: str | Path | None = None,
) -> QueueControl:
    model = control if isinstance(control, QueueControl) else QueueControl.model_validate(control)
    path = queue_control_path(queue_root=queue_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
        text=True,
    )
    tmp_path = Path(tmp_name)

    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(model.model_dump(mode="json"), handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())

        os.replace(tmp_path, path)
        _fsync_directory(path.parent)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise

    return model


def update_queue_control(
    *,
    paused: bool | None = None,
    max_concurrent_runs_per_gpu: int | None = None,
    queue_root: str | Path,
    paths: Paths,
) -> QueueControl:
    current = read_queue_control(queue_root=queue_root, paths=paths)
    updates: dict[str, object] = {}

    if paused is not None:
        updates["paused"] = bool(paused)
    if max_concurrent_runs_per_gpu is not None:
        updates["max_concurrent_runs_per_gpu"] = int(max_concurrent_runs_per_gpu)

    if not updates:
        return current

    next_value = current.model_copy(update=updates)
    return write_queue_control(next_value, queue_root=queue_root)


def _fsync_directory(path: Path) -> None:
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


__all__ = [
    "QueueControl",
    "QUEUE_CONTROL_FILENAME",
    "QUEUE_CONTROL_SCHEMA_VERSION",
    "default_queue_control",
    "queue_control_path",
    "read_queue_control",
    "update_queue_control",
    "write_queue_control",
]

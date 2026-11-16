from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from lisai.infra.paths import Paths

from .schema import JOB_STATUSES, JobStatus, QueueJob


@dataclass(frozen=True)
class DiscoveredJob:
    job: QueueJob
    path: Path
    status: JobStatus


@dataclass(frozen=True)
class InvalidQueueJob:
    path: Path
    kind: str
    message: str


def default_queue_root(*, paths: Paths) -> Path:
    env_override = os.environ.get("LISAI_QUEUE_ROOT")
    if env_override:
        return Path(env_override).expanduser().resolve()
    return (paths.workspace_root() / ".lisai" / "queue").resolve()


def queue_state_dir(status: JobStatus, *, queue_root: str | Path | None = None) -> Path:
    return _queue_root_path(queue_root) / status


def queue_logs_dir(*, queue_root: str | Path | None = None) -> Path:
    return _queue_root_path(queue_root) / "logs"


def ensure_queue_dirs(*, queue_root: str | Path | None = None) -> Path:
    root = _queue_root_path(queue_root)
    root.mkdir(parents=True, exist_ok=True)
    for status in JOB_STATUSES:
        queue_state_dir(status, queue_root=root).mkdir(parents=True, exist_ok=True)
    queue_logs_dir(queue_root=root).mkdir(parents=True, exist_ok=True)
    return root


def job_filename(job_id: str, *, selector: str | None = None) -> str:
    if selector:
        return f"job_{selector}_{job_id}.json"
    return f"{job_id}.json"


def job_log_filename(job_id: str, *, selector: str | None = None) -> str:
    if selector:
        return f"job_{selector}_{job_id}.log"
    return f"{job_id}.log"


def read_job(path: str | Path) -> QueueJob:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return QueueJob.model_validate(payload)


def write_job_atomic(path: str | Path, job: QueueJob | Mapping[str, object]) -> Path:
    destination = Path(path)
    model = job if isinstance(job, QueueJob) else QueueJob.model_validate(job)
    destination.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=str(destination.parent),
        text=True,
    )
    tmp_path = Path(tmp_name)

    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(model.model_dump(mode="json"), handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())

        os.replace(tmp_path, destination)
        _fsync_directory(destination.parent)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise

    return destination


def discover_jobs(
    *,
    status: JobStatus | None = None,
    queue_root: str | Path | None = None,
) -> tuple[tuple[DiscoveredJob, ...], tuple[InvalidQueueJob, ...]]:
    root = ensure_queue_dirs(queue_root=queue_root)
    statuses = (status,) if status is not None else JOB_STATUSES
    discovered: list[DiscoveredJob] = []
    invalid: list[InvalidQueueJob] = []

    for state in statuses:
        state_dir = queue_state_dir(state, queue_root=root)
        for path in sorted(state_dir.glob("*.json")):
            try:
                job = read_job(path)
                if job.status != state:
                    job = job.model_copy(update={"status": state})
                discovered.append(DiscoveredJob(job=job, path=path, status=state))
            except json.JSONDecodeError as exc:
                invalid.append(InvalidQueueJob(path=path, kind="json_parse_error", message=str(exc)))
            except Exception as exc:
                invalid.append(InvalidQueueJob(path=path, kind="schema_validation_error", message=str(exc)))

    discovered.sort(key=lambda item: item.job.submitted_at)
    return tuple(discovered), tuple(invalid)


def find_job(job_id: str, *, queue_root: str | Path | None = None) -> DiscoveredJob | None:
    jobs, _invalid = discover_jobs(queue_root=queue_root)
    for record in jobs:
        if record.job.job_id == job_id:
            return record
    return None


def transition_job(
    record: DiscoveredJob,
    *,
    to_status: JobStatus,
    updates: Mapping[str, object] | None = None,
    queue_root: str | Path | None = None,
) -> DiscoveredJob:
    root = ensure_queue_dirs(queue_root=queue_root)
    updates = dict(updates or {})
    updates["status"] = to_status

    destination = queue_state_dir(to_status, queue_root=root) / record.path.name
    if record.path.resolve() != destination.resolve():
        destination.parent.mkdir(parents=True, exist_ok=True)
        record.path.replace(destination)
    updated_job = record.job.model_copy(update=updates)
    write_job_atomic(destination, updated_job)
    return DiscoveredJob(job=updated_job, path=destination, status=to_status)


def remove_job_file(record: DiscoveredJob) -> None:
    record.path.unlink(missing_ok=True)


def _queue_root_path(queue_root: str | Path | None) -> Path:
    if queue_root is None:
        raise ValueError("queue_root must be resolved at the runner/CLI boundary")
    return Path(queue_root).expanduser().resolve()


def _fsync_directory(path: Path):
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


__all__ = [
    "DiscoveredJob",
    "InvalidQueueJob",
    "default_queue_root",
    "discover_jobs",
    "ensure_queue_dirs",
    "find_job",
    "job_log_filename",
    "job_filename",
    "queue_logs_dir",
    "queue_state_dir",
    "read_job",
    "remove_job_file",
    "transition_job",
    "write_job_atomic",
]

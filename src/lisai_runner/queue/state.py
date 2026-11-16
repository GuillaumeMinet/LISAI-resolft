from __future__ import annotations

from datetime import datetime
from pathlib import Path

from lisai.runs.identifiers import generate_run_id
from lisai.runs.schema import TrainingSignature, utc_now

from .selectors import allocate_selector
from .schema import JobPriority, QueueJob, ResourceClass
from .storage import DiscoveredJob, job_filename, queue_state_dir, transition_job, write_job_atomic


def create_queued_job(
    *,
    config_path: str | Path,
    resource_class: ResourceClass,
    device: str,
    priority: JobPriority = "normal",
    queue_root: str | Path | None = None,
    run_id: str | None = None,
    source_config: str | Path | None = None,
    workspace_root: str | Path | None = None,
    config_root: str | Path | None = None,
    dataset: str | None = None,
    model_subfolder: str | None = None,
    run_name: str | None = None,
    training_signature: TrainingSignature | None = None,
    now: datetime | None = None,
) -> DiscoveredJob:
    submitted = utc_now() if now is None else now
    job_id = f"job_{generate_run_id()}"
    selector = allocate_selector(queue_root=queue_root)
    job = QueueJob(
        job_id=job_id,
        selector=selector,
        config=str(Path(config_path).resolve()),
        source_config=None if source_config is None else str(Path(source_config).resolve()),
        workspace_root=None if workspace_root is None else str(Path(workspace_root).resolve()),
        config_root=None if config_root is None else str(Path(config_root).resolve()),
        status="queued",
        priority=priority,
        device=device,
        submitted_at=submitted,
        updated_at=submitted,
        resource_class=resource_class,
        run_id=run_id,
        dataset=dataset,
        model_subfolder=model_subfolder,
        run_name=run_name,
        training_signature=training_signature,
    )
    path = queue_state_dir("queued", queue_root=queue_root) / job_filename(
        job.job_id,
        selector=job.selector,
    )
    write_job_atomic(path, job)
    return DiscoveredJob(job=job, path=path, status="queued")


def mark_job_running(
    record: DiscoveredJob,
    *,
    pid: int,
    log_path: str | Path | None = None,
    now: datetime | None = None,
    queue_root: str | Path | None = None,
) -> DiscoveredJob:
    timestamp = utc_now() if now is None else now
    return transition_job(
        record,
        to_status="running",
        queue_root=queue_root,
        updates={
            "updated_at": timestamp,
            "launched_at": timestamp,
            "pid": pid,
            "log_path": None if log_path is None else str(Path(log_path).resolve()),
            "error": None,
        },
    )


def mark_job_done(
    record: DiscoveredJob,
    *,
    exit_code: int,
    now: datetime | None = None,
    queue_root: str | Path | None = None,
) -> DiscoveredJob:
    timestamp = utc_now() if now is None else now
    return transition_job(
        record,
        to_status="done",
        queue_root=queue_root,
        updates={
            "updated_at": timestamp,
            "finished_at": timestamp,
            "exit_code": int(exit_code),
            "error": None,
        },
    )


def mark_job_blocked(
    record: DiscoveredJob,
    *,
    error: str,
    now: datetime | None = None,
    queue_root: str | Path | None = None,
) -> DiscoveredJob:
    timestamp = utc_now() if now is None else now
    return transition_job(
        record,
        to_status="blocked",
        queue_root=queue_root,
        updates={
            "updated_at": timestamp,
            "finished_at": timestamp,
            "exit_code": None,
            "error": error,
        },
    )


def mark_job_failed(
    record: DiscoveredJob,
    *,
    exit_code: int | None,
    error: str | None = None,
    now: datetime | None = None,
    queue_root: str | Path | None = None,
) -> DiscoveredJob:
    timestamp = utc_now() if now is None else now
    return transition_job(
        record,
        to_status="failed",
        queue_root=queue_root,
        updates={
            "updated_at": timestamp,
            "finished_at": timestamp,
            "exit_code": None if exit_code is None else int(exit_code),
            "error": error,
        },
    )


def set_job_run_id(
    record: DiscoveredJob,
    *,
    run_id: str,
    now: datetime | None = None,
    queue_root: str | Path | None = None,
) -> DiscoveredJob:
    timestamp = utc_now() if now is None else now
    return transition_job(
        record,
        to_status=record.status,
        queue_root=queue_root,
        updates={
            "updated_at": timestamp,
            "run_id": run_id,
        },
    )


__all__ = [
    "create_queued_job",
    "mark_job_blocked",
    "mark_job_done",
    "mark_job_failed",
    "mark_job_running",
    "set_job_run_id",
]

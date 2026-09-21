from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
import re
import shutil
import signal
import sys
import time
from collections import deque
from datetime import timedelta
from pathlib import Path
from typing import Sequence

from lisai.config import load_yaml, resolve_config, save_yaml
from lisai.evaluation import run_evaluate
from lisai.evaluation.defaults import resolve_evaluate_config
from lisai.runs.schema import format_timestamp_local, utc_now
from lisai.runs.scanner import DiscoveredRun, scan_runs
from lisai.training.cli import resolve_config_path

from .control import read_queue_control, update_queue_control
from .history import load_scheduling_context
from .schema import (
    JOB_PRIORITIES,
    JOB_STATUSES,
    RESOURCE_CLASSES,
    JobPriority,
    QueueJob,
    ResourceClass,
    is_queue_selector,
)
from .selectors import reset_selector_index
from .state import create_queued_job, mark_job_failed
from .storage import (
    DiscoveredJob,
    discover_jobs,
    ensure_queue_dirs,
    find_job,
    job_log_filename,
    queue_logs_dir,
    remove_job_file,
)
from .worker import QueueWorker

ERROR_LINE_HINTS = (
    "error",
    "exception",
    "traceback",
    "failed",
    "fatal",
    "critical",
    "runtimeerror",
    "valueerror",
)

POST_TRAINING_INFERENCE_CONFIG = "post_training"
QUEUE_MANAGED_SNAPSHOTS_DIRNAME = "submitted_configs"


def submit_job(
    *,
    config: str,
    resource_class: ResourceClass,
    device: str,
    priority: JobPriority = "normal",
    stdout=None,
) -> int:
    out = sys.stdout if stdout is None else stdout
    resolved_config = resolve_config_path(config)
    snapshot_config = _copy_config_snapshot(resolved_config)
    context = load_scheduling_context(snapshot_config)

    record = create_queued_job(
        config_path=snapshot_config,
        source_config=resolved_config,
        resource_class=resource_class,
        device=device,
        priority=priority,
        dataset=context.dataset,
        model_subfolder=context.model_subfolder,
        run_name=context.run_name,
        training_signature=context.training_signature,
    )
    selector = record.job.selector or "-"
    print(f"Submitted job {selector} ({record.job.job_id})", file=out)
    print(f"  label: {_job_display_label(record.job)}", file=out)
    print(f"  config: {record.job.config}", file=out)
    print(f"  source_config: {record.job.source_config or '-'}", file=out)
    print(f"  resource_class: {record.job.resource_class}", file=out)
    print(f"  priority: {record.job.priority}", file=out)
    print(f"  device: {record.job.device}", file=out)
    return 0


def submit_sweep(
    *,
    file: str,
    resource_class: ResourceClass,
    device: str,
    priority: JobPriority = "normal",
    stdout=None,
    stderr=None,
) -> int:
    out = sys.stdout if stdout is None else stdout
    err = sys.stderr if stderr is None else stderr

    try:
        sweep_path = resolve_config_path(file)
        payload = load_yaml(sweep_path)
    except Exception as exc:
        print(
            f"Failed to read sweep file {file!r} (resolved via configs/training search): "
            f"{type(exc).__name__}: {exc}",
            file=err,
        )
        return 1

    base_config = payload.get("base_config")
    runs = payload.get("runs")
    if not isinstance(base_config, str) or not base_config.strip():
        print("Sweep file must define a non-empty 'base_config'.", file=err)
        return 1
    if not isinstance(runs, list) or not runs:
        print("Sweep file must define a non-empty list under 'runs'.", file=err)
        return 1

    try:
        resolved_base_config = _resolve_sweep_base_config(base_config, sweep_path=sweep_path)
        base_cfg = load_yaml(resolved_base_config)
    except Exception as exc:
        print(f"Failed to resolve base_config {base_config!r}: {type(exc).__name__}: {exc}", file=err)
        return 1

    queue_root = ensure_queue_dirs()
    generated_root = queue_root / "generated_configs"
    generated_root.mkdir(parents=True, exist_ok=True)

    submitted = 0
    for index, run_entry in enumerate(runs, start=1):
        if not isinstance(run_entry, dict):
            print(f"Sweep run #{index} must be an object.", file=err)
            return 1

        run_name = run_entry.get("run_name")
        overrides = run_entry.get("overrides", {})
        if not isinstance(run_name, str) or not run_name.strip():
            print(f"Sweep run #{index} is missing a non-empty 'run_name'.", file=err)
            return 1
        if not isinstance(overrides, dict):
            print(f"Sweep run #{index} has non-dictionary overrides.", file=err)
            return 1

        run_cfg = deepcopy(base_cfg)
        for key, value in overrides.items():
            if not isinstance(key, str) or not key.strip():
                print(f"Sweep run #{index} has an invalid override key {key!r}.", file=err)
                return 1
            _set_dotted_value(run_cfg, key.strip(), value)

        # Sweep run_name is authoritative for each generated run config.
        _set_dotted_value(run_cfg, "experiment.exp_name", run_name.strip())

        generated_config = _next_sweep_config_path(
            generated_root=generated_root,
            run_name=run_name,
            index=index,
        )
        save_yaml(run_cfg, generated_config)
        snapshot_config = _copy_config_snapshot(generated_config, queue_root=queue_root)
        context = load_scheduling_context(snapshot_config)
        record = create_queued_job(
            config_path=snapshot_config,
            source_config=generated_config,
            resource_class=resource_class,
            device=device,
            priority=priority,
            dataset=context.dataset,
            model_subfolder=context.model_subfolder,
            run_name=context.run_name,
            training_signature=context.training_signature,
        )
        submitted += 1
        print(
            f"Submitted job {record.job.selector or '-'} ({record.job.job_id}) "
            f"from sweep run {run_name}.",
            file=out,
        )

    print(f"Sweep submission complete: submitted={submitted} file={sweep_path}", file=out)
    return 0


def list_jobs(
    *,
    status: str | None = None,
    full: bool = False,
    stdout=None,
    stderr=None,
) -> int:
    out = sys.stdout if stdout is None else stdout
    err = sys.stderr if stderr is None else stderr

    jobs, invalid = discover_jobs(status=status)
    if jobs:
        print(render_jobs_table([record.job for record in jobs], full=full), file=out)
    else:
        print("No queue jobs found.", file=out)

    for item in invalid:
        print(
            f"warning: skipped invalid queue job {item.path} ({item.kind}: {item.message})",
            file=err,
        )
    return 0


def show_job(
    *,
    target: str | None,
    run_name: str | None = None,
    dataset: str | None = None,
    model_subfolder: str | None = None,
    lines: int = 20,
    errors: bool = False,
    as_json: bool = False,
    stdout=None,
    stderr=None,
) -> int:
    out = sys.stdout if stdout is None else stdout
    err = sys.stderr if stderr is None else stderr

    record = _resolve_single_job(
        target=target,
        run_name=run_name,
        dataset=dataset,
        model_subfolder=model_subfolder,
        stdout=out,
        stderr=err,
    )
    if record is None:
        return 1

    if as_json:
        print(json.dumps(record.job.model_dump(mode="json"), indent=2, sort_keys=True), file=out)
        return 0

    job = record.job
    print(f"selector      : {job.selector or '-'}", file=out)
    print(f"label         : {_job_display_label(job)}", file=out)
    print(f"status        : {job.status}", file=out)
    print(f"priority      : {job.priority}", file=out)
    print(f"job_id        : {job.job_id}", file=out)
    print(f"dataset       : {job.dataset or '-'}", file=out)
    print(f"subfolder     : {job.model_subfolder or '-'}", file=out)
    print(f"run_name      : {job.run_name or '-'}", file=out)
    print(f"run_id        : {job.run_id or '-'}", file=out)
    print(f"device        : {job.device}", file=out)
    print(f"pid           : {'-' if job.pid is None else job.pid}", file=out)
    print(f"submitted_at  : {format_timestamp_local(job.submitted_at)}", file=out)
    print(
        f"launched_at   : {'-' if job.launched_at is None else format_timestamp_local(job.launched_at)}",
        file=out,
    )
    print(
        f"finished_at   : {'-' if job.finished_at is None else format_timestamp_local(job.finished_at)}",
        file=out,
    )
    print(f"config        : {job.config}", file=out)
    print(f"source_config : {job.source_config or '-'}", file=out)
    print(f"log_path      : {_resolve_log_path(job)}", file=out)
    print(f"error         : {job.error or '-'}", file=out)

    if lines <= 0:
        return 0

    log_path = _resolve_log_path(job)
    if not log_path.exists():
        print("No log file found for this job.", file=out)
        return 0

    snippet = _tail_lines(log_path, limit=lines, errors_only=errors)
    if not snippet:
        print("No matching log lines.", file=out)
        return 0

    print("", file=out)
    print(f"Last {len(snippet)} log lines ({'errors only' if errors else 'all'}):", file=out)
    for line in snippet:
        print(line, file=out)
    return 0


def logs_job(
    *,
    target: str | None,
    run_name: str | None = None,
    dataset: str | None = None,
    model_subfolder: str | None = None,
    lines: int = 100,
    follow: bool = False,
    errors: bool = False,
    stdout=None,
    stderr=None,
) -> int:
    out = sys.stdout if stdout is None else stdout
    err = sys.stderr if stderr is None else stderr

    record = _resolve_single_job(
        target=target,
        run_name=run_name,
        dataset=dataset,
        model_subfolder=model_subfolder,
        stdout=out,
        stderr=err,
    )
    if record is None:
        return 1

    log_path = _resolve_log_path(record.job)
    if not log_path.exists():
        print(f"Log file not found: {log_path}", file=err)
        return 1

    initial = _tail_lines(log_path, limit=lines, errors_only=errors)
    for line in initial:
        print(line, file=out)

    if not follow:
        return 0

    try:
        _follow_log(log_path, errors_only=errors, stdout=out)
        return 0
    except KeyboardInterrupt:
        print("", file=out)
        return 0


def cancel_jobs(
    *,
    target: str | None,
    cancel_all: bool = False,
    run_name: str | None = None,
    dataset: str | None = None,
    model_subfolder: str | None = None,
    assume_yes: bool = False,
    force: bool = False,
    eval_override: bool | None = None,
    stdin=None,
    stdout=None,
    stderr=None,
) -> int:
    in_stream = sys.stdin if stdin is None else stdin
    out = sys.stdout if stdout is None else stdout
    err = sys.stderr if stderr is None else stderr

    if cancel_all:
        records = _filter_jobs(
            discover_jobs()[0],
            status=None,
            include_statuses={"queued", "running"},
        )
        if not records:
            print("No queued/running jobs to cancel.", file=out)
            return 0

        if not assume_yes:
            confirmed = _prompt_yes_no(
                f"Cancel {len(records)} queued/running jobs? [y/N]: ",
                stdin=in_stream,
                stdout=out,
            )
            if confirmed is None:
                print("Confirmation required. Rerun with --yes to continue non-interactively.", file=err)
                return 1
            if not confirmed:
                print("Cancel aborted.", file=err)
                return 1

        cancelled = 0
        failed = 0
        for record in records:
            ok = _cancel_single_record(
                record,
                force=force,
                allow_force_prompt=False,
                eval_override=eval_override,
                stdin=in_stream,
                stdout=out,
                stderr=err,
            )
            if ok:
                cancelled += 1
            else:
                failed += 1

        print(f"Cancel summary: cancelled={cancelled}, failed={failed}", file=out)
        return 0 if failed == 0 else 1

    record = _resolve_single_job(
        target=target,
        run_name=run_name,
        dataset=dataset,
        model_subfolder=model_subfolder,
        stdout=out,
        stderr=err,
    )
    if record is None:
        return 1

    if record.job.status in {"blocked", "done", "failed"}:
        print(f"Job {_job_selector_text(record.job)} is already {record.job.status}. Nothing to cancel.", file=out)
        return 0

    if not assume_yes:
        confirmed = _prompt_yes_no(
            f"Cancel job {_job_selector_text(record.job)} ({_job_display_label(record.job)})? [y/N]: ",
            stdin=in_stream,
            stdout=out,
        )
        if confirmed is None:
            print("Confirmation required. Rerun with --yes to continue non-interactively.", file=err)
            return 1
        if not confirmed:
            print("Cancel aborted.", file=err)
            return 1

    ok = _cancel_single_record(
        record,
        force=force,
        allow_force_prompt=True,
        eval_override=eval_override,
        stdin=in_stream,
        stdout=out,
        stderr=err,
    )
    return 0 if ok else 1


def start_worker(
    *,
    poll_seconds: int | None,
    safety_margin_mb: int | None,
    once: bool = False,
    stdout=None,
    stderr=None,
) -> int:
    worker = QueueWorker(
        poll_seconds=poll_seconds,
        safety_margin_mb=safety_margin_mb,
        stdout=stdout,
        stderr=stderr,
    )
    if once:
        result = worker.run_once()
        worker._report_cycle_status(result, force=True)
        return 0
    return worker.run_forever()


def pause_queue(*, stdout=None, stderr=None) -> int:
    out = sys.stdout if stdout is None else stdout
    err = sys.stderr if stderr is None else stderr
    try:
        control = update_queue_control(paused=True)
    except Exception as exc:
        print(f"Failed to pause queue: {type(exc).__name__}: {exc}", file=err)
        return 1
    print(
        "Queue paused "
        f"(max_concurrent_runs_per_gpu={control.max_concurrent_runs_per_gpu}).",
        file=out,
    )
    return 0


def resume_queue(*, stdout=None, stderr=None) -> int:
    out = sys.stdout if stdout is None else stdout
    err = sys.stderr if stderr is None else stderr
    try:
        control = update_queue_control(paused=False)
    except Exception as exc:
        print(f"Failed to resume queue: {type(exc).__name__}: {exc}", file=err)
        return 1
    print(
        "Queue resumed "
        f"(max_concurrent_runs_per_gpu={control.max_concurrent_runs_per_gpu}).",
        file=out,
    )
    return 0


def show_queue_control(*, stdout=None, stderr=None) -> int:
    out = sys.stdout if stdout is None else stdout
    err = sys.stderr if stderr is None else stderr
    try:
        control = read_queue_control()
    except Exception as exc:
        print(f"Failed to read queue control: {type(exc).__name__}: {exc}", file=err)
        return 1
    print(f"paused: {control.paused}", file=out)
    print(f"max_concurrent_runs_per_gpu: {control.max_concurrent_runs_per_gpu}", file=out)
    return 0


def set_queue_concurrency(*, max_runs_per_gpu: int, stdout=None, stderr=None) -> int:
    out = sys.stdout if stdout is None else stdout
    err = sys.stderr if stderr is None else stderr
    try:
        control = update_queue_control(max_concurrent_runs_per_gpu=max_runs_per_gpu)
    except Exception as exc:
        print(f"Failed to update queue concurrency: {type(exc).__name__}: {exc}", file=err)
        return 1
    print(
        "Queue concurrency updated "
        f"(paused={control.paused}, max_concurrent_runs_per_gpu={control.max_concurrent_runs_per_gpu}).",
        file=out,
    )
    return 0


def clean_jobs(
    *,
    older_than: str | None,
    status: str | None,
    clean_all: bool,
    assume_yes: bool = False,
    reset_selector: bool = False,
    stdin=None,
    stdout=None,
    stderr=None,
) -> int:
    in_stream = sys.stdin if stdin is None else stdin
    out = sys.stdout if stdout is None else stdout
    err = sys.stderr if stderr is None else stderr

    if not clean_all and status is None and older_than is None:
        print("At least one filter is required: --older-than, --status, or --all.", file=err)
        return 1
    if status == "running":
        print("Refusing to clean running jobs.", file=err)
        return 1

    threshold = None
    if older_than is not None:
        age_delta = parse_age_duration(older_than)
        threshold = utc_now() - age_delta

    if clean_all:
        target_statuses = ("queued", "blocked", "done", "failed")
    elif status is not None:
        target_statuses = (status,)
    else:
        target_statuses = ("blocked", "done", "failed")

    removed_jobs = 0
    removed_logs = 0
    removed_snapshots = 0
    skipped_recent = 0
    skipped_running = len(discover_jobs(status="running")[0]) if clean_all else 0
    removed_by_status = {key: 0 for key in target_statuses}

    for state in target_statuses:
        records, invalid = discover_jobs(status=state)
        for item in invalid:
            print(
                f"warning: skipped invalid queue job {item.path} ({item.kind}: {item.message})",
                file=err,
            )
        for record in records:
            if threshold is not None:
                reference = record.job.finished_at or record.job.updated_at
                if reference >= threshold:
                    skipped_recent += 1
                    continue
            if not assume_yes and not clean_all and status in {"queued"}:
                confirmed = _prompt_yes_no(
                    f"Clean queued job {_job_selector_text(record.job)}? [y/N]: ",
                    stdin=in_stream,
                    stdout=out,
                )
                if not confirmed:
                    continue
            remove_job_file(record)
            removed_jobs += 1
            removed_by_status[state] += 1
            removed_logs += int(_remove_log_for_job(record.job))
            removed_snapshots += int(_remove_managed_snapshot_for_job(record.job))

    summary = [
        f"removed_jobs={removed_jobs}",
        f"removed_logs={removed_logs}",
        f"removed_snapshots={removed_snapshots}",
    ]
    if skipped_recent:
        summary.append(f"skipped_recent={skipped_recent}")
    if skipped_running:
        summary.append(f"skipped_running={skipped_running}")
    print("Clean summary: " + ", ".join(summary), file=out)
    if removed_jobs:
        details = ", ".join(f"{state}:{count}" for state, count in removed_by_status.items())
        print(f"Removed by status: {details}", file=out)

    if reset_selector:
        if _queue_is_empty():
            reset_selector_index()
            print("Selector index reset to q0001.", file=out)
        else:
            print("Selector index not reset because the queue is not empty.", file=err)
            return 1
    elif clean_all and _queue_is_empty() and _is_interactive(in_stream):
        confirmed = _prompt_yes_no(
            "Queue is empty. Reset selector index to q0001? [y/N]: ",
            stdin=in_stream,
            stdout=out,
        )
        if confirmed:
            reset_selector_index()
            print("Selector index reset to q0001.", file=out)

    return 0


def render_jobs_table(jobs: list[QueueJob], *, full: bool = False) -> str:
    if not jobs:
        return ""

    if full:
        headers = [
            "id",
            "name",
            "status",
            "priority",
            "dataset",
            "subfolder",
            "job_id",
            "config",
            "source_config",
            "device",
            "submitted_at",
            "run_id",
            "pid",
            "log_path",
            "error",
        ]
        rows = [
            [
                _job_selector_text(job),
                _job_display_label(job),
                job.status,
                job.priority,
                "-" if job.dataset is None else job.dataset,
                "-" if job.model_subfolder is None else job.model_subfolder,
                job.job_id,
                job.config,
                "-" if job.source_config is None else job.source_config,
                job.device,
                format_timestamp_local(job.submitted_at),
                "-" if job.run_id is None else job.run_id,
                "-" if job.pid is None else str(job.pid),
                str(_resolve_log_path(job)),
                "-" if job.error is None else _single_line(job.error),
            ]
            for job in jobs
        ]
    else:
        headers = [
            "id",
            "name",
            "status",
            "priority",
            "device",
            "submitted_at",
            "pid",
            "config",
        ]
        rows = [
            [
                _job_selector_text(job),
                _job_display_label(job),
                job.status,
                job.priority,
                job.device,
                format_timestamp_local(job.submitted_at),
                "-" if job.pid is None else str(job.pid),
                _config_name(job.source_config or job.config),
            ]
            for job in jobs
        ]

    widths = [max(len(header), *(len(row[idx]) for row in rows)) for idx, header in enumerate(headers)]
    lines = [
        "  ".join(header.ljust(widths[idx]) for idx, header in enumerate(headers)),
        "  ".join("-" * widths[idx] for idx in range(len(headers))),
    ]
    lines.extend(
        "  ".join(value.ljust(widths[idx]) for idx, value in enumerate(row))
        for row in rows
    )
    return "\n".join(lines)


def parse_age_duration(value: str) -> timedelta:
    match = re.fullmatch(r"\s*(\d+)\s*([smhd])\s*", value)
    if match is None:
        raise ValueError("Duration must match <int><unit>, e.g. 7d, 12h, 30m.")
    amount = int(match.group(1))
    unit = match.group(2)
    if amount < 0:
        raise ValueError("Duration amount must be >= 0.")
    if unit == "s":
        return timedelta(seconds=amount)
    if unit == "m":
        return timedelta(minutes=amount)
    if unit == "h":
        return timedelta(hours=amount)
    return timedelta(days=amount)


def _queue_managed_snapshots_dir(*, queue_root: str | Path | None = None) -> Path:
    root = ensure_queue_dirs(queue_root=queue_root)
    snapshots_dir = root / QUEUE_MANAGED_SNAPSHOTS_DIRNAME
    snapshots_dir.mkdir(parents=True, exist_ok=True)
    return snapshots_dir


def _copy_config_snapshot(source_config: str | Path, *, queue_root: str | Path | None = None) -> Path:
    source_path = Path(source_config).expanduser().resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"Training config not found for snapshot copy: {source_path}")
    if not source_path.is_file():
        raise ValueError(f"Training config is not a file: {source_path}")

    snapshots_dir = _queue_managed_snapshots_dir(queue_root=queue_root)
    suffix = source_path.suffix if source_path.suffix.lower() in {".yaml", ".yml"} else ".yml"
    safe_stem = _sanitize_filename_component(source_path.stem)
    token = time.time_ns()
    candidate = snapshots_dir / f"{safe_stem}_{token}{suffix}"
    counter = 2
    while candidate.exists():
        candidate = snapshots_dir / f"{safe_stem}_{token}_{counter}{suffix}"
        counter += 1

    shutil.copyfile(source_path, candidate)
    return candidate.resolve()


def _remove_managed_snapshot_for_job(job: QueueJob, *, queue_root: str | Path | None = None) -> bool:
    snapshots_root = _queue_managed_snapshots_dir(queue_root=queue_root).resolve()
    config_path = Path(job.config).expanduser().resolve()
    if not _is_path_within(config_path, snapshots_root):
        return False
    if not config_path.exists() or not config_path.is_file():
        return False
    config_path.unlink(missing_ok=True)
    return True


def _is_path_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _resolve_sweep_base_config(base_config: str, *, sweep_path: Path) -> Path:
    candidate = Path(base_config).expanduser()
    if not candidate.is_absolute():
        candidate_from_sweep = (sweep_path.parent / candidate).resolve()
        if candidate_from_sweep.exists():
            return resolve_config_path(str(candidate_from_sweep))
    return resolve_config_path(base_config)


def _next_sweep_config_path(*, generated_root: Path, run_name: str, index: int) -> Path:
    generated_root.mkdir(parents=True, exist_ok=True)
    safe_name = _sanitize_filename_component(run_name)
    base_name = f"sweep_{index:04d}_{safe_name}"
    candidate = generated_root / f"{base_name}.yml"
    if not candidate.exists():
        return candidate

    suffix = 2
    while True:
        candidate = generated_root / f"{base_name}_{suffix}.yml"
        if not candidate.exists():
            return candidate
        suffix += 1


def _sanitize_filename_component(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    cleaned = cleaned.strip("._-")
    return cleaned or "run"


def _set_dotted_value(target: dict, dotted_key: str, value: object) -> None:
    keys = [part for part in dotted_key.split(".") if part]
    if not keys:
        raise ValueError("Override path must not be empty.")

    cursor = target
    for part in keys[:-1]:
        existing = cursor.get(part)
        if existing is None:
            next_node: dict[str, object] = {}
            cursor[part] = next_node
            cursor = next_node
            continue
        if not isinstance(existing, dict):
            raise ValueError(
                f"Cannot set nested override {dotted_key!r}: "
                f"component {part!r} is not an object."
            )
        cursor = existing
    cursor[keys[-1]] = value


def run_submit_from_args(args: argparse.Namespace) -> int:
    return submit_job(
        config=args.config,
        resource_class=args.resource_class,
        device=args.device,
        priority=args.priority,
    )


def run_submit_sweep_from_args(args: argparse.Namespace) -> int:
    return submit_sweep(
        file=args.file,
        resource_class=args.resource_class,
        device=args.device,
        priority=args.priority,
    )


def run_list_from_args(args: argparse.Namespace) -> int:
    return list_jobs(
        status=args.status,
        full=args.full,
    )


def run_show_from_args(args: argparse.Namespace) -> int:
    if not _has_selector_input(args):
        print("Provide a job selector or semantic filters (e.g. --run-name).", file=sys.stderr)
        return 1
    return show_job(
        target=args.target,
        run_name=args.run_name,
        dataset=args.dataset,
        model_subfolder=args.model_subfolder,
        lines=args.lines,
        errors=args.errors,
        as_json=args.json,
    )


def run_logs_from_args(args: argparse.Namespace) -> int:
    if not _has_selector_input(args):
        print("Provide a job selector or semantic filters (e.g. --run-name).", file=sys.stderr)
        return 1
    return logs_job(
        target=args.target,
        run_name=args.run_name,
        dataset=args.dataset,
        model_subfolder=args.model_subfolder,
        lines=args.lines,
        follow=args.follow,
        errors=args.errors,
    )


def run_cancel_from_args(args: argparse.Namespace) -> int:
    if args.all and _has_selector_input(args):
        print("Use either --all or a specific selector/filter, not both.", file=sys.stderr)
        return 1
    if not args.all and not _has_selector_input(args):
        print("Provide a job selector/filter or use --all.", file=sys.stderr)
        return 1
    eval_override = _parse_optional_bool_text(getattr(args, "eval_override", None))
    return cancel_jobs(
        target=args.target,
        cancel_all=args.all,
        run_name=args.run_name,
        dataset=args.dataset,
        model_subfolder=args.model_subfolder,
        assume_yes=args.yes,
        force=args.force,
        eval_override=eval_override,
    )


def run_worker_from_args(args: argparse.Namespace) -> int:
    return start_worker(
        poll_seconds=args.poll_seconds,
        safety_margin_mb=args.safety_margin_mb,
        once=args.once,
    )


def run_clean_from_args(args: argparse.Namespace) -> int:
    return clean_jobs(
        older_than=args.older_than,
        status=args.status,
        clean_all=args.all,
        assume_yes=args.yes,
        reset_selector=args.reset_selector,
    )


def run_pause_from_args(_args: argparse.Namespace) -> int:
    return pause_queue()


def run_resume_from_args(_args: argparse.Namespace) -> int:
    return resume_queue()


def run_control_from_args(_args: argparse.Namespace) -> int:
    return show_queue_control()


def run_concurrency_from_args(args: argparse.Namespace) -> int:
    return set_queue_concurrency(max_runs_per_gpu=args.max_runs_per_gpu)


def _add_queue_commands(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    submit_parser = subparsers.add_parser(
        "submit",
        help="Submit a training config to the local queue.",
        description="Submit a training config to the local queue.",
    )
    submit_parser.add_argument("--config", required=True, help="Training config path or short name.")
    submit_parser.add_argument(
        "--resource-class",
        choices=RESOURCE_CLASSES,
        default="medium",
        help="Resource class used when no historical VRAM data is available.",
    )
    submit_parser.add_argument("--device", default="cuda:0", help="Target device (default: cuda:0).")
    submit_parser.add_argument(
        "--priority",
        choices=JOB_PRIORITIES,
        default="normal",
        help="Queue priority (default: normal).",
    )
    submit_parser.set_defaults(handler=run_submit_from_args)

    submit_sweep_parser = subparsers.add_parser(
        "submit-sweep",
        help="Submit multiple runs from an explicit sweep file.",
        description="Submit multiple runs from an explicit sweep file.",
    )
    submit_sweep_parser.add_argument(
        "--file",
        required=True,
        help=(
            "Sweep YAML path or training-config-relative path. "
            "Use --file sweeps/<name>.yaml for files under configs/training/sweeps; "
            "base_config may use a short training config name such as hdn.yml."
        ),
    )
    submit_sweep_parser.add_argument(
        "--resource-class",
        choices=RESOURCE_CLASSES,
        default="medium",
        help="Resource class used when no historical VRAM data is available.",
    )
    submit_sweep_parser.add_argument("--device", default="cuda:0", help="Target device (default: cuda:0).")
    submit_sweep_parser.add_argument(
        "--priority",
        choices=JOB_PRIORITIES,
        default="normal",
        help="Queue priority for all generated sweep jobs (default: normal).",
    )
    submit_sweep_parser.set_defaults(handler=run_submit_sweep_from_args)

    list_parser = subparsers.add_parser(
        "list",
        help="List queue jobs.",
        description="List queue jobs.",
    )
    list_parser.add_argument("--status", choices=JOB_STATUSES, help="Filter jobs by queue status.")
    list_parser.add_argument(
        "--full",
        action="store_true",
        help="Display full details (paths, run_id, pid, log_path, and errors).",
    )
    list_parser.set_defaults(handler=run_list_from_args)

    show_parser = subparsers.add_parser(
        "show",
        help="Show details for a single queue job.",
        description="Show details for a single queue job.",
    )
    _add_job_selector_arguments(show_parser)
    show_parser.add_argument(
        "--lines",
        type=int,
        default=20,
        help="Number of log lines to include in the summary (default: 20).",
    )
    show_parser.add_argument(
        "--errors",
        action="store_true",
        help="Only show log lines likely related to errors.",
    )
    show_parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw queue job JSON.",
    )
    show_parser.set_defaults(handler=run_show_from_args)

    logs_parser = subparsers.add_parser(
        "logs",
        help="Display logs for a queue job.",
        description="Display logs for a queue job.",
    )
    _add_job_selector_arguments(logs_parser)
    logs_parser.add_argument(
        "-n",
        "--lines",
        type=int,
        default=100,
        help="Number of lines to display initially (default: 100).",
    )
    logs_parser.add_argument(
        "-f",
        "--follow",
        action="store_true",
        help="Follow log output continuously.",
    )
    logs_parser.add_argument(
        "--errors",
        action="store_true",
        help="Only show log lines likely related to errors.",
    )
    logs_parser.set_defaults(handler=run_logs_from_args)

    cancel_parser = subparsers.add_parser(
        "cancel",
        help="Cancel queued/running jobs.",
        description="Cancel queued/running jobs.",
    )
    _add_job_selector_arguments(cancel_parser)
    cancel_parser.add_argument("--all", action="store_true", help="Cancel all queued/running jobs.")
    cancel_parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Skip confirmation prompts.",
    )
    cancel_parser.add_argument(
        "--force",
        action="store_true",
        help="Allow SIGKILL escalation for unresponsive running jobs.",
    )
    cancel_parser.add_argument(
        "--eval",
        dest="eval_override",
        type=lambda value: value.strip().lower(),
        choices=("true", "false"),
        help=(
            "Override post-cancel evaluation behavior for running jobs. "
            "When omitted, follows experiment.post_training_inference from the training config."
        ),
    )
    cancel_parser.set_defaults(handler=run_cancel_from_args)

    worker_parser = subparsers.add_parser(
        "worker",
        help="Run the local queue worker loop.",
        description="Run the local queue worker loop.",
    )
    worker_parser.add_argument("--poll-seconds", type=int, default=None, help="Worker poll interval.")
    worker_parser.add_argument(
        "--safety-margin-mb",
        type=int,
        default=None,
        help="Extra free VRAM required before launch.",
    )
    worker_parser.add_argument(
        "--once",
        action="store_true",
        help="Run a single worker cycle and exit.",
    )
    worker_parser.set_defaults(handler=run_worker_from_args)

    clean_parser = subparsers.add_parser(
        "clean",
        help="Clean queue records and associated queue logs.",
        description="Clean queue records and associated queue logs.",
    )
    clean_parser.add_argument(
        "--older-than",
        help="Age threshold like 7d, 12h, or 30m.",
    )
    mode_group = clean_parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--status",
        choices=JOB_STATUSES,
        help="Only clean jobs with this status (running is rejected).",
    )
    mode_group.add_argument(
        "--all",
        action="store_true",
        help="Clean all non-running jobs (queued/blocked/done/failed).",
    )
    clean_parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Skip confirmation prompts.",
    )
    clean_parser.add_argument(
        "--reset-selector",
        action="store_true",
        help="Reset selector index to q0001 if the queue becomes empty after cleaning.",
    )
    clean_parser.set_defaults(handler=run_clean_from_args)

    pause_parser = subparsers.add_parser(
        "pause",
        help="Pause queue launches.",
        description="Pause queue launches.",
    )
    pause_parser.set_defaults(handler=run_pause_from_args)

    resume_parser = subparsers.add_parser(
        "resume",
        help="Resume queue launches.",
        description="Resume queue launches.",
    )
    resume_parser.set_defaults(handler=run_resume_from_args)

    control_parser = subparsers.add_parser(
        "control",
        help="Show persistent queue control values.",
        description="Show persistent queue control values.",
    )
    control_parser.set_defaults(handler=run_control_from_args)

    concurrency_parser = subparsers.add_parser(
        "concurrency",
        help="Set max concurrent runs per GPU.",
        description="Set max concurrent runs per GPU.",
    )
    concurrency_parser.add_argument(
        "--max-runs-per-gpu",
        type=int,
        required=True,
        help="Maximum concurrent CUDA jobs per GPU.",
    )
    concurrency_parser.set_defaults(handler=run_concurrency_from_args)


def _add_job_selector_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "target",
        nargs="?",
        help="Job selector (qNNNN), full job_id, or unique job_id prefix.",
    )
    parser.add_argument("--run-name", help="Filter by run_name.")
    parser.add_argument("--dataset", help="Filter by dataset name.")
    parser.add_argument(
        "--model-subfolder",
        "--subfolder",
        dest="model_subfolder",
        help="Filter by model_subfolder.",
    )


def add_queue_subparser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]):
    parser = subparsers.add_parser(
        "queue",
        help="Manage local training queue jobs.",
        description="Manage local training queue jobs.",
    )
    queue_subparsers = parser.add_subparsers(dest="queue_command")
    queue_subparsers.required = True
    _add_queue_commands(queue_subparsers)
    return parser


def build_parser(*, prog: str = "lisai-runner") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage local training queue jobs.", prog=prog)
    subparsers = parser.add_subparsers(dest="queue_command")
    subparsers.required = True
    _add_queue_commands(subparsers)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    return args.handler(args)


def _resolve_single_job(
    *,
    target: str | None,
    run_name: str | None,
    dataset: str | None,
    model_subfolder: str | None,
    stdout,
    stderr,
    include_statuses: set[str] | None = None,
) -> DiscoveredJob | None:
    records = discover_jobs()[0]
    matches = _filter_jobs(
        records,
        target=target,
        run_name=run_name,
        dataset=dataset,
        model_subfolder=model_subfolder,
        include_statuses=include_statuses,
    )
    if not matches:
        print("No matching queue job found.", file=stderr)
        return None
    if len(matches) > 1:
        print("Multiple queue jobs match the selector/filters.", file=stderr)
        print(render_jobs_table([item.job for item in matches]), file=stdout)
        print("Narrow the selector (prefer qNNNN) and retry.", file=stderr)
        return None
    return matches[0]


def _filter_jobs(
    records: tuple[DiscoveredJob, ...],
    *,
    target: str | None = None,
    run_name: str | None = None,
    dataset: str | None = None,
    model_subfolder: str | None = None,
    status: str | None = None,
    include_statuses: set[str] | None = None,
) -> list[DiscoveredJob]:
    out = list(records)
    if include_statuses is not None:
        allowed = {item.strip().lower() for item in include_statuses}
        out = [record for record in out if record.job.status in allowed]
    if status is not None:
        out = [record for record in out if record.job.status == status]
    if target:
        out = _match_target(out, target)
    if run_name:
        out = [record for record in out if record.job.run_name == run_name]
    if dataset:
        out = [record for record in out if record.job.dataset == dataset]
    if model_subfolder:
        out = [record for record in out if record.job.model_subfolder == model_subfolder]
    return out


def _match_target(records: list[DiscoveredJob], target: str) -> list[DiscoveredJob]:
    token = target.strip()
    if not token:
        return records

    lowered = token.lower()

    exact_selector = [record for record in records if (record.job.selector or "").lower() == lowered]
    if exact_selector:
        return exact_selector

    exact_job_id = [record for record in records if record.job.job_id.lower() == lowered]
    if exact_job_id:
        return exact_job_id

    if is_queue_selector(token):
        return []

    prefix_matches = [record for record in records if record.job.job_id.lower().startswith(lowered)]
    if prefix_matches:
        return prefix_matches
    return []


def _cancel_single_record(
    record: DiscoveredJob,
    *,
    force: bool,
    allow_force_prompt: bool,
    eval_override: bool | None,
    stdin,
    stdout,
    stderr,
) -> bool:
    if record.job.status == "queued":
        latest = find_job(record.job.job_id)
        if latest is None:
            print(f"warning: queue job {record.job.job_id} disappeared before cancellation.", file=stderr)
            return False
        mark_job_failed(
            latest,
            exit_code=None,
            error="cancelled_by_user(queued)",
        )
        print(f"Cancelled queued job {_job_selector_text(record.job)}.", file=stdout)
        return True

    if record.job.status != "running":
        print(
            f"Job {_job_selector_text(record.job)} is already {record.job.status}. Nothing to cancel.",
            file=stdout,
        )
        return True

    pid = record.job.pid
    if pid is None:
        latest = find_job(record.job.job_id)
        if latest is None:
            print(f"warning: queue job {record.job.job_id} disappeared before cancellation.", file=stderr)
            return False
        mark_job_failed(
            latest,
            exit_code=None,
            error="cancelled_by_user(running_no_pid)",
        )
        print(f"Cancelled running job {_job_selector_text(record.job)} (missing pid metadata).", file=stdout)
        _maybe_run_post_cancel_evaluation(latest.job, eval_override=eval_override, stdout=stdout, stderr=stderr)
        return True

    if not _pid_exists(pid):
        latest = find_job(record.job.job_id)
        if latest is None:
            print(f"warning: queue job {record.job.job_id} disappeared before cancellation.", file=stderr)
            return False
        mark_job_failed(
            latest,
            exit_code=None,
            error="cancelled_by_user(process_not_found)",
        )
        print(
            f"Marked {_job_selector_text(record.job)} as cancelled because PID {pid} no longer exists.",
            file=stdout,
        )
        _maybe_run_post_cancel_evaluation(latest.job, eval_override=eval_override, stdout=stdout, stderr=stderr)
        return True

    if not _send_signal(pid, signal.SIGINT, stderr=stderr):
        return False
    print(f"Sent SIGINT to PID {pid}. Waiting up to 30s...", file=stdout)
    if _wait_for_exit(pid, timeout_seconds=30):
        return _finalize_cancelled_running(
            record,
            reason="sigint",
            eval_override=eval_override,
            stdout=stdout,
            stderr=stderr,
        )

    if not _send_signal(pid, signal.SIGTERM, stderr=stderr):
        return False
    print(f"Process still alive. Sent SIGTERM to PID {pid}. Waiting up to 10s...", file=stdout)
    if _wait_for_exit(pid, timeout_seconds=10):
        return _finalize_cancelled_running(
            record,
            reason="sigterm",
            eval_override=eval_override,
            stdout=stdout,
            stderr=stderr,
        )

    if not force and allow_force_prompt:
        prompt = _prompt_yes_no(
            "Job is still running after SIGINT/SIGTERM. Force kill with SIGKILL? [y/N]: ",
            stdin=stdin,
            stdout=stdout,
        )
        force = bool(prompt)

    if not force:
        print(
            "Job is still running after SIGINT/SIGTERM. Use --force to allow SIGKILL.",
            file=stderr,
        )
        return False

    if not _send_signal(pid, signal.SIGKILL, stderr=stderr):
        return False
    print(f"Sent SIGKILL to PID {pid}.", file=stdout)
    if _wait_for_exit(pid, timeout_seconds=5):
        return _finalize_cancelled_running(
            record,
            reason="sigkill",
            eval_override=eval_override,
            stdout=stdout,
            stderr=stderr,
        )

    print(f"Failed to terminate PID {pid}.", file=stderr)
    return False


def _finalize_cancelled_running(
    record: DiscoveredJob,
    *,
    reason: str,
    eval_override: bool | None,
    stdout,
    stderr,
) -> bool:
    latest = find_job(record.job.job_id)
    if latest is None:
        print(f"warning: queue job {record.job.job_id} disappeared before final cancellation mark.", file=stderr)
        return False
    if latest.job.status != "running":
        print(
            f"Job {_job_selector_text(latest.job)} already transitioned to {latest.job.status}.",
            file=stdout,
        )
        return True
    mark_job_failed(
        latest,
        exit_code=None,
        error=f"cancelled_by_user({reason})",
    )
    print(f"Cancelled running job {_job_selector_text(latest.job)}.", file=stdout)
    _maybe_run_post_cancel_evaluation(latest.job, eval_override=eval_override, stdout=stdout, stderr=stderr)
    return True


def _send_signal(pid: int, sig: signal.Signals, *, stderr) -> bool:
    try:
        os.kill(pid, sig)
        return True
    except ProcessLookupError:
        print(f"warning: PID {pid} no longer exists.", file=stderr)
        return False
    except PermissionError:
        print(f"warning: missing permission to signal PID {pid}.", file=stderr)
        return False
    except OSError as exc:
        print(f"warning: failed to signal PID {pid}: {type(exc).__name__}: {exc}", file=stderr)
        return False


def _pid_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _wait_for_exit(pid: int, *, timeout_seconds: int, poll_seconds: float = 0.5) -> bool:
    deadline = time.monotonic() + float(timeout_seconds)
    while time.monotonic() < deadline:
        if not _pid_exists(pid):
            return True
        time.sleep(poll_seconds)
    return not _pid_exists(pid)


def _maybe_run_post_cancel_evaluation(
    job: QueueJob,
    *,
    eval_override: bool | None,
    stdout,
    stderr,
) -> None:
    should_eval = _should_evaluate_after_cancel(job, eval_override=eval_override, stderr=stderr)
    if not should_eval:
        return

    run = _resolve_run_for_queue_job(job, stderr=stderr)
    if run is None:
        return

    if run.metadata.last_epoch is None:
        print(
            f"Skipping post-cancel evaluation for {_job_selector_text(job)}: no completed epoch recorded.",
            file=stderr,
        )
        return

    print(
        f"Running post-cancel evaluation for {_job_selector_text(job)} "
        f"(run={run.metadata.run_id}, model={run.run_dir.name}).",
        file=stdout,
    )
    try:
        eval_cfg = resolve_evaluate_config(config=POST_TRAINING_INFERENCE_CONFIG)
        run_evaluate(
            cfg=eval_cfg,
            dataset_name=run.dataset,
            model_subfolder=run.model_subfolder,
            model_name=run.run_dir.name,
        )
    except Exception as exc:
        print(
            f"warning: post-cancel evaluation failed for {_job_selector_text(job)}: "
            f"{type(exc).__name__}: {exc}",
            file=stderr,
        )


def _should_evaluate_after_cancel(
    job: QueueJob,
    *,
    eval_override: bool | None,
    stderr,
) -> bool:
    if eval_override is not None:
        return bool(eval_override)

    try:
        cfg = resolve_config(job.config)
    except Exception as exc:
        print(
            f"warning: could not resolve config for {_job_selector_text(job)} "
            f"to read post_training_inference: {type(exc).__name__}: {exc}",
            file=stderr,
        )
        return False

    experiment = getattr(cfg, "experiment", None)
    return bool(getattr(experiment, "post_training_inference", False))


def _resolve_run_for_queue_job(job: QueueJob, *, stderr) -> DiscoveredRun | None:
    scan_result = scan_runs()

    candidates: list[DiscoveredRun] = []
    if job.run_id is not None:
        candidates = [run for run in scan_result.runs if run.metadata.run_id == job.run_id]
        if len(candidates) == 1:
            return candidates[0]

    if job.run_name and job.dataset and job.model_subfolder:
        candidates = [
            run
            for run in scan_result.runs
            if run.metadata.run_name == job.run_name
            and run.dataset == job.dataset
            and run.model_subfolder == job.model_subfolder
        ]

    if not candidates:
        print(
            f"warning: no matching run found for {_job_selector_text(job)}; cannot run evaluation.",
            file=stderr,
        )
        return None

    if len(candidates) > 1:
        ids = ", ".join(run.metadata.run_id for run in candidates[:5])
        suffix = "" if len(candidates) <= 5 else ", ..."
        print(
            f"warning: ambiguous run match for {_job_selector_text(job)} ({len(candidates)} candidates: "
            f"{ids}{suffix}); skipping evaluation.",
            file=stderr,
        )
        return None

    return candidates[0]


def _has_selector_input(args: argparse.Namespace) -> bool:
    return bool(
        getattr(args, "target", None)
        or getattr(args, "run_name", None)
        or getattr(args, "dataset", None)
        or getattr(args, "model_subfolder", None)
    )


def _resolve_log_path(job: QueueJob) -> Path:
    if job.log_path:
        return Path(job.log_path).expanduser().resolve()
    return queue_logs_dir() / job_log_filename(job.job_id, selector=job.selector)


def _remove_log_for_job(job: QueueJob) -> bool:
    path = _resolve_log_path(job)
    if not path.exists() or not path.is_file():
        return False
    path.unlink(missing_ok=True)
    return True


def _tail_lines(path: Path, *, limit: int, errors_only: bool = False) -> list[str]:
    if limit <= 0:
        return []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        lines = [line.rstrip("\n") for line in deque(handle, maxlen=limit)]
    if not errors_only:
        return lines
    return [line for line in lines if _looks_like_error_line(line)]


def _follow_log(path: Path, *, errors_only: bool, stdout) -> None:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        handle.seek(0, os.SEEK_END)
        while True:
            line = handle.readline()
            if not line:
                time.sleep(0.5)
                continue
            text = line.rstrip("\n")
            if errors_only and not _looks_like_error_line(text):
                continue
            print(text, file=stdout, flush=True)


def _looks_like_error_line(line: str) -> bool:
    lowered = line.lower()
    return any(token in lowered for token in ERROR_LINE_HINTS)


def _queue_is_empty() -> bool:
    records, invalid = discover_jobs()
    return not records and not invalid


def _prompt_yes_no(prompt: str, *, stdin, stdout) -> bool | None:
    if not _is_interactive(stdin):
        return None
    print(prompt, end="", file=stdout, flush=True)
    answer = stdin.readline()
    if answer == "":
        return None
    return answer.strip().lower() in {"y", "yes"}


def _is_interactive(stream) -> bool:
    is_tty = getattr(stream, "isatty", None)
    return callable(is_tty) and bool(is_tty())


def _config_name(path: str) -> str:
    return Path(path).name


def _job_selector_text(job: QueueJob) -> str:
    return job.selector or job.job_id


def _job_display_label(job: QueueJob) -> str:
    run_name = _single_line(job.run_name or _config_name(job.source_config or job.config))
    if job.selector:
        return f"{job.selector}_{run_name}"
    return run_name


def _single_line(value: str) -> str:
    return value.replace("\n", " ").replace("\r", " ").strip()


def _parse_optional_bool_text(value: str | None) -> bool | None:
    if value is None:
        return None
    text = value.strip().lower()
    if text == "true":
        return True
    if text == "false":
        return False
    raise ValueError(f"Unsupported boolean text: {value!r}")


__all__ = [
    "add_queue_subparser",
    "build_parser",
    "cancel_jobs",
    "clean_jobs",
    "list_jobs",
    "logs_job",
    "main",
    "parse_age_duration",
    "pause_queue",
    "render_jobs_table",
    "resume_queue",
    "set_queue_concurrency",
    "show_job",
    "show_queue_control",
    "start_worker",
    "submit_job",
    "submit_sweep",
]

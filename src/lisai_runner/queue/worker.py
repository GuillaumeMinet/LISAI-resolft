from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import IO

from lisai.infra.paths import Paths
from lisai.runs.listing import active_heartbeat_timeout, is_run_likely_active, write_invalid_run_warnings
from lisai.runs.scanner import DiscoveredRun, scan_runs
from lisai.runs.schema import utc_now

from .control import QueueControl, default_queue_control, read_queue_control
from .gpu import parse_cuda_device_index
from .schema import QueueJob, parse_queue_selector
from .state import (
    mark_job_blocked,
    mark_job_done,
    mark_job_failed,
    mark_job_running,
    set_job_run_id,
)
from .storage import (
    DiscoveredJob,
    InvalidQueueJob,
    discover_jobs,
    ensure_queue_dirs,
    find_job,
    job_log_filename,
    queue_logs_dir,
    queue_state_dir,
    write_job_atomic,
)

PRIORITY_RANK = {
    "high": 0,
    "normal": 1,
    "low": 2,
}


@dataclass(frozen=True)
class LaunchDecision:
    should_launch: bool
    reason: str
    blocked_error: str | None = None


@dataclass(frozen=True)
class WorkerCycleResult:
    status_key: str
    status_message: str
    launched: bool = False


@dataclass(frozen=True)
class ProcessRefreshResult:
    finished_jobs: int
    failed_jobs: int


@dataclass
class RunningProcess:
    record: DiscoveredJob
    process: subprocess.Popen
    log_handle: IO[bytes]


class QueueWorker:
    def __init__(
        self,
        *,
        paths: Paths,
        queue_root: str | Path,
        poll_seconds: int | None = None,
        safety_margin_mb: int | None = None,
        fixed_margin_pct: float | None = None,
        heartbeat_seconds: int = 120,
        stdout=None,
        stderr=None,
    ):
        self.paths = paths
        self.queue_root = ensure_queue_dirs(queue_root=queue_root)
        self.poll_seconds = int(
            self.paths.settings.project_cfg.queue.poll_seconds if poll_seconds is None else poll_seconds
        )
        # Kept for compatibility with previous worker constructor.
        self.safety_margin_mb = int(
            self.paths.settings.project_cfg.queue.safety_margin_mb if safety_margin_mb is None else safety_margin_mb
        )
        self.fixed_margin_pct = float(
            self.paths.settings.project_cfg.queue.fixed_margin_pct if fixed_margin_pct is None else fixed_margin_pct
        )
        self.heartbeat_seconds = int(heartbeat_seconds)
        self.stdout = sys.stdout if stdout is None else stdout
        self.stderr = sys.stderr if stderr is None else stderr
        self._running_processes: dict[str, RunningProcess] = {}
        self._started = False
        self._last_cycle_result: WorkerCycleResult | None = None
        self._last_cycle_report_monotonic: float | None = None
        self._last_control: QueueControl | None = None

    def run_forever(self) -> int:
        self._ensure_started()
        while True:
            result = self.run_once()
            self._report_cycle_status(result)
            time.sleep(self.poll_seconds)

    def run_once(self) -> WorkerCycleResult:
        self._ensure_started()

        scan_result = scan_runs(paths=self.paths)
        write_invalid_run_warnings(scan_result.invalid, stderr=self.stderr)

        refresh = self._refresh_finished_processes()
        self._reconcile_running_jobs(scan_result.runs)

        queued_records, invalid_jobs = discover_jobs(status="queued", queue_root=self.queue_root)
        for invalid in invalid_jobs:
            quarantined = self._quarantine_invalid_queued_job(invalid)
            if quarantined is None:
                self._emit(
                    f"warning: skipped invalid queue job {invalid.path} ({invalid.kind}: {invalid.message})",
                    stream=self.stderr,
                )
                continue
            self._emit(
                f"quarantined invalid queued job {invalid.path} -> {quarantined.path} "
                f"({invalid.kind}: {invalid.message})",
                stream=self.stderr,
            )

        control = self._load_control()
        self._report_control_change(control)

        if control.paused:
            return WorkerCycleResult(
                status_key="paused",
                status_message="paused",
            )

        queued_records = self._sorted_queued_records(queued_records)
        if not queued_records:
            if refresh.failed_jobs:
                return WorkerCycleResult(
                    status_key="failed",
                    status_message=f"failed {refresh.failed_jobs} job(s)",
                )
            if refresh.finished_jobs:
                return WorkerCycleResult(
                    status_key="finished",
                    status_message=f"finished {refresh.finished_jobs} job(s)",
                )
            return WorkerCycleResult(
                status_key="idle",
                status_message="idle: no queued jobs",
            )

        running_cuda_by_gpu = self._running_cuda_counts()
        waiting_due_to_capacity = False
        blocked_this_cycle = 0

        for record in queued_records:
            decision = self._launch_decision(
                record.job,
                running_cuda_by_gpu=running_cuda_by_gpu,
                max_concurrent_runs_per_gpu=control.max_concurrent_runs_per_gpu,
            )
            if decision.blocked_error:
                mark_job_blocked(
                    record,
                    error=decision.blocked_error,
                    queue_root=self.queue_root,
                )
                blocked_this_cycle += 1
                self._emit(
                    f"blocked job {record.job.job_id}: {decision.blocked_error}",
                    stream=self.stderr,
                )
                continue

            if not decision.should_launch:
                if decision.reason == "device_busy":
                    waiting_due_to_capacity = True
                continue

            launched = self._launch_job(record)
            if launched:
                return WorkerCycleResult(
                    status_key="launched",
                    status_message=f"launched: {record.job.selector or record.job.job_id}",
                    launched=True,
                )
            return WorkerCycleResult(
                status_key="failed",
                status_message=f"failed: {record.job.selector or record.job.job_id}",
            )

        if blocked_this_cycle:
            queued_after, _invalid = discover_jobs(status="queued", queue_root=self.queue_root)
            if not queued_after:
                return WorkerCycleResult(
                    status_key="idle",
                    status_message="idle: no queued jobs",
                )

        if waiting_due_to_capacity:
            return WorkerCycleResult(
                status_key="waiting/device_busy",
                status_message="waiting: device busy / concurrency limit reached",
            )

        return WorkerCycleResult(
            status_key="waiting/no_eligible",
            status_message="waiting: no eligible job",
        )

    def _launch_decision(
        self,
        job: QueueJob,
        *,
        running_cuda_by_gpu: dict[int, int],
        max_concurrent_runs_per_gpu: int,
    ) -> LaunchDecision:
        blocked_error = self._prelaunch_block_reason(job)
        if blocked_error is not None:
            return LaunchDecision(
                should_launch=False,
                reason="blocked",
                blocked_error=blocked_error,
            )

        device_index = parse_cuda_device_index(job.device)
        if device_index is None:
            return LaunchDecision(should_launch=True, reason="non_cuda")

        running_on_device = running_cuda_by_gpu.get(device_index, 0)
        if running_on_device >= max_concurrent_runs_per_gpu:
            return LaunchDecision(
                should_launch=False,
                reason="device_busy",
            )
        return LaunchDecision(should_launch=True, reason="eligible")

    def _prelaunch_block_reason(self, job: QueueJob) -> str | None:
        try:
            parse_cuda_device_index(job.device)
        except Exception as exc:
            return f"invalid_device_spec:{type(exc).__name__}: {exc}"

        config_path = Path(job.config).expanduser().resolve()
        if not config_path.exists():
            return f"prelaunch_validation_failed: config not found ({config_path})"
        if not config_path.is_file():
            return f"prelaunch_validation_failed: config is not a file ({config_path})"
        if job.workspace_root is not None:
            # Catch uninitialized/moved workspaces before starting Python/GPU.
            owned_configs = Path(job.workspace_root) / "configs"
            missing = [owned_configs / name for name in ("workspace.yml", "data.yml")
                       if not (owned_configs / name).is_file()]
            if missing:
                return ("prelaunch_validation_failed: selected workspace is missing "
                        + ", ".join(map(str, missing))
                        + f" (run 'lisai workspace init {job.workspace_root}')")
        elif job.config_root is not None:
            old = Path(job.config_root)
            if not (old / "project_config.yml").is_file():
                return ("prelaunch_validation_failed: old job lacks workspace_root and "
                        f"{old} is not a legacy config directory; resubmit this job")
        return None

    def _running_cuda_counts(self) -> dict[int, int]:
        counts: dict[int, int] = {}
        running_records, _invalid = discover_jobs(status="running", queue_root=self.queue_root)
        for record in running_records:
            try:
                device_index = parse_cuda_device_index(record.job.device)
            except Exception:
                continue
            if device_index is None:
                continue
            counts[device_index] = counts.get(device_index, 0) + 1
        return counts

    def _sorted_queued_records(self, records: tuple[DiscoveredJob, ...]) -> tuple[DiscoveredJob, ...]:
        return tuple(
            sorted(
                records,
                key=lambda record: (
                    PRIORITY_RANK.get(record.job.priority, PRIORITY_RANK["normal"]),
                    record.job.submitted_at,
                    self._selector_index(record.job.selector),
                    record.job.job_id,
                ),
            )
        )

    def _selector_index(self, selector: str | None) -> int:
        if selector is None:
            return sys.maxsize
        parsed = parse_queue_selector(selector)
        return sys.maxsize if parsed is None else parsed

    def _launch_job(self, record: DiscoveredJob) -> bool:
        job = record.job
        log_path = queue_logs_dir(queue_root=self.queue_root) / job_log_filename(
            job.job_id,
            selector=job.selector,
        )
        log_handle: IO[bytes] | None = None
        try:
            log_handle = log_path.open("ab")

            env = os.environ.copy()
            device_index = parse_cuda_device_index(job.device)
            if device_index is not None:
                env["CUDA_VISIBLE_DEVICES"] = str(device_index)
            # Queue worker logs are non-interactive files; disable tqdm redraw spam.
            env["LISAI_DISABLE_TQDM"] = "1"

            command = [
                sys.executable,
                "-m",
                "lisai_runner.training_entrypoint",
                "--config",
                job.config,
            ]
            if job.workspace_root is not None:
                command.extend([
                    "--workspace", job.workspace_root,
                    "--config-root", job.config_root,
                ])
            elif job.config_root is not None:
                # Old queued jobs still reference a legacy three-file config dir.
                command.extend(["--settings-config-dir", job.config_root])
            else:
                # Job created without explicit identity by a legacy in-process
                # caller: preserve the worker-selected roots as an explicit handoff.
                command.extend([
                    "--workspace", str(self.paths.workspace_root()),
                    "--config-root", str(self.paths.settings.CONFIGS_ROOT),
                ])
            process = subprocess.Popen(
                command,
                cwd=str(self.queue_root),
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
            running_record = mark_job_running(
                record,
                pid=process.pid,
                log_path=log_path,
                queue_root=self.queue_root,
            )
            self._running_processes[job.job_id] = RunningProcess(
                record=running_record,
                process=process,
                log_handle=log_handle,
            )
            self._emit(
                f"launched job {job.job_id} ({job.selector or '-'}) config={job.config} "
                f"pid={process.pid} device={job.device}",
            )
            return True
        except Exception as exc:
            if log_handle is not None:
                log_handle.close()
            mark_job_failed(
                record,
                exit_code=None,
                error=f"{type(exc).__name__}: {exc}",
                queue_root=self.queue_root,
            )
            self._emit(
                f"failed job {job.job_id}: launch error {type(exc).__name__}: {exc}",
                stream=self.stderr,
            )
            return False

    def _refresh_finished_processes(self) -> ProcessRefreshResult:
        finished: list[str] = []
        finished_jobs = 0
        failed_jobs = 0
        for job_id, running in self._running_processes.items():
            exit_code = running.process.poll()
            if exit_code is None:
                continue

            running.log_handle.close()
            latest_record = find_job(job_id, queue_root=self.queue_root)
            if latest_record is not None:
                if latest_record.status != "running":
                    finished.append(job_id)
                    continue
                if exit_code == 0:
                    mark_job_done(
                        latest_record,
                        exit_code=exit_code,
                        queue_root=self.queue_root,
                    )
                    finished_jobs += 1
                    self._emit(
                        f"finished job {job_id} exit_code={exit_code}",
                    )
                else:
                    mark_job_failed(
                        latest_record,
                        exit_code=exit_code,
                        queue_root=self.queue_root,
                    )
                    failed_jobs += 1
                    self._emit(
                        f"failed job {job_id} exit_code={exit_code}",
                        stream=self.stderr,
                    )
            finished.append(job_id)

        for job_id in finished:
            self._running_processes.pop(job_id, None)

        return ProcessRefreshResult(finished_jobs=finished_jobs, failed_jobs=failed_jobs)

    def _reconcile_running_jobs(self, runs: tuple[DiscoveredRun, ...]) -> None:
        running_records, _invalid = discover_jobs(status="running", queue_root=self.queue_root)
        if not running_records:
            return

        by_run_id = {run.metadata.run_id: run for run in runs}
        for record in running_records:
            linked = record
            if linked.job.run_id is None:
                maybe_run_id = self._infer_run_id_for_job(linked.job, runs)
                if maybe_run_id is not None:
                    linked = set_job_run_id(
                        linked,
                        run_id=maybe_run_id,
                        queue_root=self.queue_root,
                    )

            if linked.job.job_id in self._running_processes:
                continue

            run_id = linked.job.run_id
            if run_id is None:
                continue
            run = by_run_id.get(run_id)
            if run is None:
                continue

            if run.metadata.status in {"completed", "stopped"} and run.metadata.closed_cleanly:
                mark_job_done(linked, exit_code=0, queue_root=self.queue_root)
                self._emit(f"finished job {linked.job.job_id} (reconciled from run metadata)")
            elif run.metadata.status == "failed" and run.metadata.closed_cleanly:
                mark_job_failed(linked, exit_code=1, queue_root=self.queue_root)
                self._emit(
                    f"failed job {linked.job.job_id} (reconciled from run metadata)",
                    stream=self.stderr,
                )

    def _infer_run_id_for_job(
        self,
        job: QueueJob,
        runs: tuple[DiscoveredRun, ...],
    ) -> str | None:
        if job.run_name is None or job.dataset is None or job.model_subfolder is None:
            return None

        candidates = [
            run
            for run in runs
            if run.metadata.run_name == job.run_name
            and run.dataset == job.dataset
            and run.model_subfolder == job.model_subfolder
        ]
        if not candidates:
            return None

        if job.launched_at is not None:
            oldest_allowed = job.launched_at - timedelta(minutes=10)
            candidates = [run for run in candidates if run.metadata.created_at >= oldest_allowed]
            if not candidates:
                return None

        active_candidates = [run for run in candidates if is_run_likely_active(run, fallback_timeout=active_heartbeat_timeout(paths=self.paths))]
        if len(active_candidates) == 1:
            return active_candidates[0].metadata.run_id
        if len(active_candidates) > 1:
            return None

        if len(candidates) == 1:
            return candidates[0].metadata.run_id
        return None

    def _ensure_started(self) -> None:
        if self._started:
            return
        self._started = True
        self._emit("worker started")
        self._report_startup_snapshot()

    def _report_startup_snapshot(self) -> None:
        queued_count = len(discover_jobs(status="queued", queue_root=self.queue_root)[0])
        running_count = len(discover_jobs(status="running", queue_root=self.queue_root)[0])
        blocked_count = len(discover_jobs(status="blocked", queue_root=self.queue_root)[0])
        control = self._load_control()
        self._last_control = control
        self._emit(
            "worker snapshot: "
            f"queued={queued_count} "
            f"running={running_count} "
            f"blocked={blocked_count} "
            f"paused={control.paused} "
            f"max_concurrent_runs_per_gpu={control.max_concurrent_runs_per_gpu}"
        )

    def _load_control(self) -> QueueControl:
        try:
            return read_queue_control(queue_root=self.queue_root, paths=self.paths)
        except Exception as exc:
            self._emit(
                f"warning: failed to read queue control file ({type(exc).__name__}: {exc}); "
                "using defaults.",
                stream=self.stderr,
            )
            return default_queue_control(paths=self.paths)

    def _report_control_change(self, control: QueueControl) -> None:
        previous = self._last_control
        if previous is None:
            self._last_control = control
            return
        if previous == control:
            return

        self._emit(
            "control change: "
            f"paused={control.paused} "
            f"max_concurrent_runs_per_gpu={control.max_concurrent_runs_per_gpu}"
        )
        if previous.paused != control.paused:
            if control.paused:
                self._emit("paused")
            else:
                self._emit("resumed")
        self._last_control = control

    def _report_cycle_status(self, result: WorkerCycleResult, *, force: bool = False) -> None:
        now = time.monotonic()
        changed = (
            force
            or self._last_cycle_result is None
            or self._last_cycle_result.status_key != result.status_key
            or self._last_cycle_result.status_message != result.status_message
        )

        if changed:
            self._emit(result.status_message)
            self._last_cycle_result = result
            self._last_cycle_report_monotonic = now
            return

        if self._last_cycle_report_monotonic is None:
            self._emit(result.status_message)
            self._last_cycle_report_monotonic = now
            return

        if (now - self._last_cycle_report_monotonic) >= self.heartbeat_seconds:
            self._emit(f"heartbeat: {result.status_message}")
            self._last_cycle_report_monotonic = now

    def _emit(self, message: str, *, stream=None) -> None:
        target = self.stdout if stream is None else stream
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {message}", file=target, flush=True)

    def _quarantine_invalid_queued_job(self, invalid: InvalidQueueJob) -> DiscoveredJob | None:
        queued_dir = queue_state_dir("queued", queue_root=self.queue_root).resolve()
        invalid_path = invalid.path.resolve()
        if invalid_path.parent != queued_dir:
            return None
        if not invalid_path.exists():
            return None
        if not invalid_path.is_file():
            return None

        payload = self._load_invalid_payload(invalid_path)
        selector, job_id = self._job_identity_from_filename(invalid_path)
        now = utc_now()

        config_value = str(invalid_path)
        source_config_value: str | None = None
        if payload is not None:
            raw_config = payload.get("config")
            if isinstance(raw_config, str) and raw_config.strip():
                config_value = raw_config.strip()
            raw_source = payload.get("source_config")
            if isinstance(raw_source, str) and raw_source.strip():
                source_config_value = raw_source.strip()
            raw_job_id = payload.get("job_id")
            if isinstance(raw_job_id, str) and raw_job_id.strip():
                job_id = raw_job_id.strip()
            raw_selector = payload.get("selector")
            if isinstance(raw_selector, str) and raw_selector.strip():
                selector = raw_selector.strip()

        quarantined = QueueJob(
            job_id=job_id,
            selector=selector,
            config=config_value,
            source_config=source_config_value,
            status="failed",
            device="unknown",
            submitted_at=now,
            updated_at=now,
            resource_class="medium",
            finished_at=now,
            error=f"quarantined_invalid_queued_job:{invalid.kind}: {invalid.message}",
        )

        destination = self._failed_quarantine_path(invalid_path)
        write_job_atomic(destination, quarantined)
        invalid_path.unlink(missing_ok=True)
        return DiscoveredJob(job=quarantined, path=destination, status="failed")

    def _load_invalid_payload(self, path: Path) -> dict[str, object] | None:
        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception:
            return None
        if not isinstance(payload, dict):
            return None
        return payload

    def _job_identity_from_filename(self, path: Path) -> tuple[str | None, str]:
        stem = path.stem
        if not stem.startswith("job_"):
            fallback = f"job_invalid_{int(utc_now().timestamp())}"
            return None, fallback

        body = stem[len("job_") :]
        if body.startswith("q") and "_job_" in body:
            maybe_selector, tail = body.split("_job_", 1)
            if tail.strip():
                return maybe_selector.strip(), f"job_{tail.strip()}"
        if stem.strip():
            return None, stem.strip()
        fallback = f"job_invalid_{int(utc_now().timestamp())}"
        return None, fallback

    def _failed_quarantine_path(self, source_path: Path) -> Path:
        failed_dir = queue_state_dir("failed", queue_root=self.queue_root)
        candidate = failed_dir / source_path.name
        if not candidate.exists():
            return candidate

        suffix = 2
        while True:
            candidate = failed_dir / f"{source_path.stem}_{suffix}{source_path.suffix}"
            if not candidate.exists():
                return candidate
            suffix += 1


__all__ = [
    "LaunchDecision",
    "ProcessRefreshResult",
    "QueueWorker",
    "RunningProcess",
    "WorkerCycleResult",
]

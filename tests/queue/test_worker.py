from __future__ import annotations

import io
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import lisai.queue.worker as worker_mod
import lisai.runs.listing as runs_listing
from lisai.queue.control import update_queue_control
from lisai.queue.schema import QueueJob
from lisai.queue.state import create_queued_job, mark_job_running
from lisai.queue.storage import discover_jobs, queue_state_dir
from lisai.queue.worker import QueueWorker, WorkerCycleResult
from lisai.runs.scanner import DiscoveredRun, ScanResults
from lisai.runs.schema import RunMetadata, TrainingSignature


def _run(
    *,
    run_id: str,
    status: str,
    closed_cleanly: bool,
    last_heartbeat_at: str,
    signature: TrainingSignature | None,
    peak_gpu_mem_mb: int | None,
    live_runtime_stats: dict | None = None,
    created_at: str = "2026-03-20T10:00:00Z",
) -> DiscoveredRun:
    updated_at = last_heartbeat_at
    metadata = RunMetadata.model_validate(
        {
            "schema_version": 2,
            "run_id": run_id,
            "run_name": "demo",
            "run_index": 0,
            "dataset": "Gag",
            "model_subfolder": "HDN",
            "status": status,
            "closed_cleanly": closed_cleanly,
            "created_at": created_at,
            "updated_at": updated_at,
            "ended_at": None if status == "running" else "2026-03-20T11:00:00Z",
            "last_heartbeat_at": last_heartbeat_at,
            "last_epoch": 10,
            "max_epoch": 10,
            "best_val_loss": 0.1,
            "path": "datasets/Gag/models/HDN/demo_00",
            "group_path": None,
            "training_signature": None if signature is None else signature.model_dump(mode="json"),
            "runtime_stats": None if peak_gpu_mem_mb is None else {"peak_gpu_mem_mb": peak_gpu_mem_mb},
            "live_runtime_stats": live_runtime_stats,
        }
    )
    run_dir = Path("/tmp/Gag/demo_00")
    return DiscoveredRun(
        metadata=metadata,
        metadata_path=run_dir / ".lisai_run_meta.json",
        run_dir=run_dir,
        dataset="Gag",
        model_subfolder="HDN",
        group_path=None,
        path="datasets/Gag/models/HDN/demo_00",
        path_consistent=True,
        consistency_issues=(),
    )


def _queued_job(
    tmp_path: Path,
    *,
    resource_class: str = "medium",
    priority: str = "normal",
    device: str = "cuda:0",
    now: datetime | None = None,
):
    timestamp = now or datetime(2026, 3, 20, 12, 0, tzinfo=timezone.utc)
    cfg_path = tmp_path / "cfg.yml"
    cfg_path.write_text("{}", encoding="utf-8")
    return create_queued_job(
        config_path=cfg_path,
        resource_class=resource_class,
        priority=priority,
        device=device,
        queue_root=tmp_path / ".lisai" / "queue",
        dataset="Gag",
        model_subfolder="HDN",
        run_name="demo",
        training_signature=TrainingSignature(architecture="unet", batch_size=8, patch_size=128),
        now=timestamp,
    )


def _running_job(tmp_path: Path, *, device: str = "cuda:0"):
    created = _queued_job(tmp_path, device=device)
    queued_records, _invalid = discover_jobs(status="queued", queue_root=tmp_path / ".lisai" / "queue")
    assert len(queued_records) == 1
    return mark_job_running(
        queued_records[0],
        pid=4242,
        log_path=tmp_path / "queue.log",
        queue_root=tmp_path / ".lisai" / "queue",
    )


def test_worker_launch_order_is_priority_first_then_fifo(monkeypatch, tmp_path):
    queue_root = tmp_path / ".lisai" / "queue"
    _queued_job(
        tmp_path,
        priority="normal",
        now=datetime(2026, 3, 20, 12, 0, tzinfo=timezone.utc),
    )
    first_high = _queued_job(
        tmp_path,
        priority="high",
        now=datetime(2026, 3, 20, 12, 1, tzinfo=timezone.utc),
    )
    _queued_job(
        tmp_path,
        priority="high",
        now=datetime(2026, 3, 20, 12, 2, tzinfo=timezone.utc),
    )

    monkeypatch.setattr(worker_mod, "scan_runs", lambda: ScanResults(runs=(), invalid=()))
    launched: list[str] = []
    monkeypatch.setattr(
        QueueWorker,
        "_launch_job",
        lambda self, record: launched.append(record.job.job_id) or True,
    )

    worker = QueueWorker(queue_root=queue_root)
    result = worker.run_once()

    assert result.status_key == "launched"
    assert launched == [first_high.job.job_id]


def test_worker_blocks_invalid_device_spec(monkeypatch, tmp_path):
    queue_root = tmp_path / ".lisai" / "queue"
    created = _queued_job(tmp_path, device="cuda:abc")
    monkeypatch.setattr(worker_mod, "scan_runs", lambda: ScanResults(runs=(), invalid=()))
    monkeypatch.setattr(QueueWorker, "_launch_job", lambda self, record: True)

    worker = QueueWorker(queue_root=queue_root)
    result = worker.run_once()

    assert result.status_key in {"idle", "waiting/no_eligible"}
    queued_records, _invalid = discover_jobs(status="queued", queue_root=queue_root)
    blocked_records, _invalid = discover_jobs(status="blocked", queue_root=queue_root)
    assert queued_records == ()
    assert len(blocked_records) == 1
    assert blocked_records[0].job.job_id == created.job.job_id
    assert "invalid_device_spec" in (blocked_records[0].job.error or "")


def test_worker_waits_when_concurrency_limit_is_reached(monkeypatch, tmp_path):
    queue_root = tmp_path / ".lisai" / "queue"
    _running_job(tmp_path, device="cuda:0")
    _queued_job(
        tmp_path,
        now=datetime(2026, 3, 20, 12, 5, tzinfo=timezone.utc),
    )
    monkeypatch.setattr(worker_mod, "scan_runs", lambda: ScanResults(runs=(), invalid=()))

    launched: list[str] = []
    monkeypatch.setattr(
        QueueWorker,
        "_launch_job",
        lambda self, record: launched.append(record.job.job_id) or True,
    )

    worker = QueueWorker(queue_root=queue_root)
    result = worker.run_once()

    assert result.status_key == "waiting/device_busy"
    assert launched == []


def test_worker_allows_launch_when_concurrency_override_increases_capacity(monkeypatch, tmp_path):
    queue_root = tmp_path / ".lisai" / "queue"
    _running_job(tmp_path, device="cuda:0")
    queued = _queued_job(
        tmp_path,
        now=datetime(2026, 3, 20, 12, 5, tzinfo=timezone.utc),
    )
    update_queue_control(queue_root=queue_root, max_concurrent_runs_per_gpu=2)

    monkeypatch.setattr(worker_mod, "scan_runs", lambda: ScanResults(runs=(), invalid=()))
    launched: list[str] = []
    monkeypatch.setattr(
        QueueWorker,
        "_launch_job",
        lambda self, record: launched.append(record.job.job_id) or True,
    )

    worker = QueueWorker(queue_root=queue_root)
    result = worker.run_once()

    assert result.status_key == "launched"
    assert launched == [queued.job.job_id]


def test_worker_paused_control_prevents_launch(monkeypatch, tmp_path):
    queue_root = tmp_path / ".lisai" / "queue"
    _queued_job(tmp_path)
    update_queue_control(queue_root=queue_root, paused=True)

    monkeypatch.setattr(worker_mod, "scan_runs", lambda: ScanResults(runs=(), invalid=()))
    launched: list[str] = []
    monkeypatch.setattr(
        QueueWorker,
        "_launch_job",
        lambda self, record: launched.append(record.job.job_id) or True,
    )

    worker = QueueWorker(queue_root=queue_root)
    result = worker.run_once()

    assert result.status_key == "paused"
    assert launched == []


def test_worker_status_reporter_prints_on_change_and_heartbeat(monkeypatch, tmp_path):
    output = io.StringIO()
    worker = QueueWorker(
        queue_root=tmp_path / ".lisai" / "queue",
        heartbeat_seconds=30,
        stdout=output,
    )

    ticks = iter([0.0, 10.0, 45.0])
    monkeypatch.setattr(worker_mod.time, "monotonic", lambda: next(ticks))

    result = WorkerCycleResult(status_key="idle", status_message="idle: no queued jobs")
    worker._report_cycle_status(result)
    worker._report_cycle_status(result)
    worker._report_cycle_status(result)

    lines = [line for line in output.getvalue().splitlines() if line.strip()]
    assert len(lines) == 2
    assert "idle: no queued jobs" in lines[0]
    assert "heartbeat: idle: no queued jobs" in lines[1]


def test_worker_treats_old_running_heartbeat_as_stale_not_active(monkeypatch, tmp_path, settings):
    _queued_job(tmp_path)
    stale_run = _run(
        run_id="01ARZ3NDEKTSV4RRFFQ69G5FAC",
        status="running",
        closed_cleanly=False,
        last_heartbeat_at="2026-03-20T10:00:00Z",
        signature=None,
        peak_gpu_mem_mb=None,
    )

    monkeypatch.setattr(worker_mod, "scan_runs", lambda: ScanResults(runs=(stale_run,), invalid=()))
    monkeypatch.setattr(settings.project_cfg.run_tracking, "active_heartbeat_timeout_minutes", 10)
    monkeypatch.setattr(runs_listing, "utc_now", lambda: datetime(2026, 3, 20, 12, 0, tzinfo=timezone.utc))
    monkeypatch.setattr(QueueWorker, "_launch_job", lambda self, record: True)

    worker = QueueWorker(queue_root=tmp_path / ".lisai" / "queue")
    result = worker.run_once()

    assert result.status_key == "launched"


def test_worker_infer_run_id_prefers_single_dynamic_active_candidate(monkeypatch, tmp_path):
    now = datetime(2026, 3, 20, 12, 0, tzinfo=timezone.utc)
    stale_candidate = _run(
        run_id="01ARZ3NDEKTSV4RRFFQ69G5FAD",
        status="running",
        closed_cleanly=False,
        last_heartbeat_at="2026-03-20T11:58:00Z",
        signature=None,
        peak_gpu_mem_mb=None,
        live_runtime_stats={
            "last_epoch_duration_s": 30.0,
            "recent_epoch_durations_s": [30.0],
        },
        created_at="2026-03-20T11:55:00Z",
    )
    active_candidate = _run(
        run_id="01ARZ3NDEKTSV4RRFFQ69G5FAE",
        status="running",
        closed_cleanly=False,
        last_heartbeat_at="2026-03-20T11:59:30Z",
        signature=None,
        peak_gpu_mem_mb=None,
        live_runtime_stats={
            "last_epoch_duration_s": 150.0,
            "recent_epoch_durations_s": [120.0, 150.0, 180.0],
        },
        created_at="2026-03-20T11:55:30Z",
    )
    job = QueueJob(
        job_id="job_demo",
        config=str(tmp_path / "cfg.yml"),
        status="running",
        device="cuda:0",
        submitted_at=now - timedelta(minutes=2),
        updated_at=now,
        resource_class="medium",
        run_id=None,
        dataset="Gag",
        model_subfolder="HDN",
        run_name="demo",
        launched_at=now,
    )

    monkeypatch.setattr(runs_listing, "utc_now", lambda: now)
    worker = QueueWorker(queue_root=tmp_path / ".lisai" / "queue")

    inferred = worker._infer_run_id_for_job(job, (stale_candidate, active_candidate))
    assert inferred == "01ARZ3NDEKTSV4RRFFQ69G5FAE"


def test_worker_launch_sets_env_flag_to_disable_tqdm(monkeypatch, tmp_path):
    created = _queued_job(tmp_path)
    queued_records, _invalid = discover_jobs(status="queued", queue_root=tmp_path / ".lisai" / "queue")
    record = queued_records[0]
    assert record.job.job_id == created.job.job_id

    captured: dict[str, object] = {}

    class DummyProcess:
        pid = 12345

        def poll(self):
            return None

    def fake_popen(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return DummyProcess()

    monkeypatch.setattr(worker_mod.subprocess, "Popen", fake_popen)

    worker = QueueWorker(queue_root=tmp_path / ".lisai" / "queue")
    launched = worker._launch_job(record)

    assert launched is True
    popen_kwargs = captured["kwargs"]
    assert isinstance(popen_kwargs, dict)
    assert popen_kwargs["env"]["LISAI_DISABLE_TQDM"] == "1"
    running = worker._running_processes.pop(record.job.job_id)
    running.log_handle.close()


def test_worker_quarantines_invalid_queued_job_once(monkeypatch, tmp_path):
    queue_root = tmp_path / ".lisai" / "queue"
    queued_dir = queue_state_dir("queued", queue_root=queue_root)
    invalid_path = queued_dir / "job_q0099_job_invalid_payload.json"
    invalid_path.parent.mkdir(parents=True, exist_ok=True)
    invalid_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "job_id": "job_invalid_payload",
                "selector": "q0099",
                "config": "/tmp/example.yml",
                "status": "queued",
                "device": "cuda:0",
                "submitted_at": "2026-03-20T12:00:00Z",
                "updated_at": "2026-03-20T12:00:00Z",
                "resource_class": "medium",
                "unexpected_field": "boom",
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(worker_mod, "scan_runs", lambda: ScanResults(runs=(), invalid=()))
    stderr = io.StringIO()
    worker = QueueWorker(queue_root=queue_root, stdout=io.StringIO(), stderr=stderr)

    first = worker.run_once()
    second = worker.run_once()

    assert first.status_key == "idle"
    assert second.status_key == "idle"

    queued_jobs, queued_invalid = discover_jobs(status="queued", queue_root=queue_root)
    failed_jobs, failed_invalid = discover_jobs(status="failed", queue_root=queue_root)
    assert queued_jobs == ()
    assert queued_invalid == ()
    assert failed_invalid == ()
    assert len(failed_jobs) == 1
    assert failed_jobs[0].job.job_id == "job_invalid_payload"
    assert "quarantined_invalid_queued_job" in (failed_jobs[0].job.error or "")

    err_text = stderr.getvalue()
    assert err_text.count("quarantined invalid queued job") == 1
    assert "skipped invalid queue job" not in err_text


def test_worker_quarantine_does_not_block_valid_launch(monkeypatch, tmp_path):
    queue_root = tmp_path / ".lisai" / "queue"
    queued_dir = queue_state_dir("queued", queue_root=queue_root)
    invalid_path = queued_dir / "job_q0100_job_invalid_and_valid.json"
    invalid_path.parent.mkdir(parents=True, exist_ok=True)
    invalid_path.write_text("{", encoding="utf-8")

    valid = _queued_job(tmp_path)
    monkeypatch.setattr(worker_mod, "scan_runs", lambda: ScanResults(runs=(), invalid=()))
    launched: list[str] = []
    monkeypatch.setattr(
        QueueWorker,
        "_launch_job",
        lambda self, record: launched.append(record.job.job_id) or True,
    )

    worker = QueueWorker(queue_root=queue_root, stdout=io.StringIO(), stderr=io.StringIO())
    result = worker.run_once()

    assert result.status_key == "launched"
    assert launched == [valid.job.job_id]
    failed_jobs, _invalid = discover_jobs(status="failed", queue_root=queue_root)
    assert any("quarantined_invalid_queued_job" in (record.job.error or "") for record in failed_jobs)

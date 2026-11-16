"""M1 Patch F: per-operation paths and parent->child queue config identity."""
from __future__ import annotations

import io
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from lisai.config.io.yaml import load_yaml, save_yaml
from lisai.config.settings import Settings
from lisai.infra.paths import Paths
from lisai.runs.lifecycle import create_run_metadata, stored_run_path
from lisai.runs.listing import active_heartbeat_timeout
from lisai.runs.scanner import scan_runs
from lisai_runner.queue import cli as queue_cli
from lisai_runner.queue.control import default_queue_control, read_queue_control
from lisai_runner.queue.history import SchedulingContext, load_scheduling_context
from lisai_runner.queue.schema import QueueJob
from lisai_runner.queue.state import create_queued_job
from lisai_runner.queue.storage import (
    default_queue_root, discover_jobs, ensure_queue_dirs, read_job,
)
from lisai_runner.queue.worker import QueueWorker
from lisai_runner import training_entrypoint
from lisai.runs.schema import TrainingSignature, utc_now


def _queue_settings(settings: Settings, *, timeout_minutes: int, paused: bool, concurrency: int) -> Settings:
    project_file = settings.CONFIGS_ROOT / "project_config.yml"
    raw = load_yaml(project_file)
    raw["run_tracking"]["active_heartbeat_timeout_minutes"] = timeout_minutes
    raw["queue"]["paused"] = paused
    raw["queue"]["max_concurrent_runs_per_gpu"] = concurrency
    save_yaml(raw, project_file)
    return Settings.from_config_dir(settings.CONFIGS_ROOT)


def _signature():
    return TrainingSignature(architecture="unet", train_batch_size=8)


def _context(config_path):
    return SchedulingContext(
        config_path=Path(config_path), dataset="my_dataset", model_subfolder="my_model",
        run_name="my_run", training_signature=_signature(),
    )


def test_run_scanner_lifecycle_and_heartbeat_a_b_a(settings_factory):
    a = Paths(_queue_settings(settings_factory("run_a"), timeout_minutes=3, paused=False, concurrency=1))
    b = Paths(_queue_settings(settings_factory("run_b"), timeout_minutes=9, paused=True, concurrency=2))

    for paths in (a, b):
        run_dir = paths.run_dir(dataset_name="my_dataset", models_subfolder="my_model", exp_name="my_run")
        run_dir.mkdir(parents=True)
        metadata = create_run_metadata(run_dir, dataset="my_dataset", model_subfolder="my_model", paths=paths)
        assert metadata.path == stored_run_path(run_dir, paths=paths)
        scan = scan_runs(paths=paths)
        assert [run.run_dir for run in scan.runs] == [run_dir]
        assert not scan.invalid

    for paths, timeout in ((a, 3), (b, 9), (a, 3)):
        assert int(active_heartbeat_timeout(paths=paths).total_seconds()) == 60 * timeout
        scan = scan_runs(paths=paths)
        assert len(scan.runs) == 1
        assert scan.runs[0].run_dir == paths.run_dir(dataset_name="my_dataset", models_subfolder="my_model", exp_name="my_run")
    assert a.workspace_root() != b.workspace_root()


def test_queue_storage_and_control_a_b_a(settings_factory):
    a = Paths(_queue_settings(settings_factory("queue_a"), timeout_minutes=3, paused=False, concurrency=1))
    b = Paths(_queue_settings(settings_factory("queue_b"), timeout_minutes=9, paused=True, concurrency=2))
    assert default_queue_root(paths=a) != default_queue_root(paths=b)
    for paths, paused, max_runs in ((a, False, 1), (b, True, 2), (a, False, 1)):
        root = default_queue_root(paths=paths)
        assert root == paths.project_root() / ".lisai" / "queue"
        assert default_queue_control(paths=paths).paused is paused
        assert read_queue_control(queue_root=root, paths=paths).max_concurrent_runs_per_gpu == max_runs
        assert read_queue_control(queue_root=root, paths=paths).paused is paused
    with pytest.raises(ValueError, match="queue_root"):
        ensure_queue_dirs()


def test_submit_job_remembers_selected_config_and_queue_root_a_b_a(settings_factory, monkeypatch):
    a, b = Paths(settings_factory("submit_a")), Paths(settings_factory("submit_b"))
    monkeypatch.setattr(queue_cli, "load_scheduling_context", lambda cfg, *, paths: _context(cfg))

    for paths in (a, b, a):
        src = paths.settings.CONFIGS_ROOT / "training" / "submitted.yml"
        src.parent.mkdir(exist_ok=True)
        src.write_text("experiment:\n  exp_name: my_run\n", encoding="utf-8")
        result = queue_cli.submit_job(
            config=str(src), resource_class="light", device="cpu", paths=paths,
            stdout=io.StringIO(),
        )
        assert result == 0
        jobs, invalid = discover_jobs(queue_root=default_queue_root(paths=paths))
        assert not invalid
        assert jobs[-1].job.config_root == str(paths.settings.CONFIGS_ROOT)
        assert jobs[-1].job.source_config == str(src)
        snapshot = Path(jobs[-1].job.config)
        assert snapshot.is_file() and snapshot.parent.name == "submitted_configs"
        assert snapshot.read_text(encoding="utf-8") == src.read_text(encoding="utf-8")
    assert len(discover_jobs(queue_root=default_queue_root(paths=a))[0]) == 2
    assert len(discover_jobs(queue_root=default_queue_root(paths=b))[0]) == 1


def test_worker_preserves_config_identity_when_worker_and_job_settings_differ(settings_factory, monkeypatch, tmp_path):
    import lisai_runner.queue.worker as worker_mod
    a, b = Paths(settings_factory("worker_a")), Paths(settings_factory("worker_b"))
    root = tmp_path / "shared_queue"
    cfg = tmp_path / "config_snapshot.yml"
    cfg.write_text("experiment: {}\n", encoding="utf-8")
    record = create_queued_job(
        config_path=cfg, config_root=b.settings.CONFIGS_ROOT, resource_class="light",
        device="cpu", queue_root=root,
    )
    assert read_job(record.path).config_root == str(b.settings.CONFIGS_ROOT)
    captured = {}

    class FakeProcess:
        pid = 999
        def poll(self):
            return None

    def fake_popen(args, **kw):
        captured.update(args=args, **kw)
        return FakeProcess()

    monkeypatch.setattr(worker_mod.subprocess, "Popen", fake_popen)
    worker = QueueWorker(paths=a, queue_root=root, stdout=io.StringIO(), stderr=io.StringIO())
    assert worker._launch_job(record)
    cmd = captured["args"]
    assert cmd[cmd.index("--settings-config-dir") + 1] == str(b.settings.CONFIGS_ROOT)
    assert cmd[cmd.index("--config") + 1] == str(cfg)
    # Working directory remains the legacy workspace, but never decides configuration.
    assert captured["cwd"] == str(a.project_root())
    assert captured["stdin"] == worker_mod.subprocess.DEVNULL
    assert captured["env"]["LISAI_DISABLE_TQDM"] == "1"
    worker._running_processes[record.job.job_id].log_handle.close()


def test_training_child_reconstructs_submitted_context(settings_factory, monkeypatch, tmp_path):
    import lisai.api as api
    a, b = Paths(settings_factory("child_a")), Paths(settings_factory("child_b"))
    cfg = tmp_path / "submitted.yml"
    cfg.write_text("experiment: {}\n", encoding="utf-8")
    seen = []
    monkeypatch.setattr(api, "train", lambda config, *, settings: seen.append((Path(config), settings.CONFIGS_ROOT)))
    for chosen in (a, b, a):
        exit_code = training_entrypoint.main([
            "--config", str(cfg), "--settings-config-dir", str(chosen.settings.CONFIGS_ROOT),
        ])
        assert exit_code == 0
    assert seen == [(cfg, p.settings.CONFIGS_ROOT) for p in (a, b, a)]


def test_scheduling_context_and_resource_defaults_are_selected(settings_factory, monkeypatch, tmp_path):
    import lisai_runner.queue.history as history
    a, b = Paths(settings_factory("history_a")), Paths(settings_factory("history_b"))
    file = tmp_path / "dummy.yml"
    file.write_text("{}", encoding="utf-8")
    captured = []

    def fake_resolve(path, *, stg):
        captured.append(stg)
        return SimpleNamespace(
            data=SimpleNamespace(dataset_name="dataset"),
            routing=SimpleNamespace(models_subfolder="model"),
            experiment=SimpleNamespace(exp_name="run"),
        )

    monkeypatch.setattr(history, "resolve_config", fake_resolve)
    monkeypatch.setattr(history, "build_training_signature_from_resolved_config", lambda cfg: _signature())
    for paths in (a, b, a):
        assert load_scheduling_context(file, paths=paths).dataset == "dataset"
        defaults = history.resource_class_defaults_mb(paths=paths)
        assert defaults["light"] > 0
    assert captured == [a.settings, b.settings, a.settings]

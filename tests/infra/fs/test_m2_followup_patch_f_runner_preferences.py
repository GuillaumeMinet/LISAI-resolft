"""Patch F: queue identities are pinned; developer preferences are read at launch."""
from __future__ import annotations

import io
from pathlib import Path
from types import SimpleNamespace

import pytest

from lisai.config.io.yaml import save_yaml
from lisai.config.selection import load_settings_for_selected_locations, resolve_cli_locations
from lisai.config.settings import Settings
from lisai.infra.paths import Paths
from lisai_runner import training_entrypoint
from lisai_runner.queue import cli as queue_cli
from lisai_runner.queue.storage import default_queue_root, discover_jobs
from lisai_runner.queue.worker import QueueWorker


@pytest.mark.parametrize("mode", ["repo", "workspace", "custom"])
def test_all_developer_modes_keep_submitted_roots_and_read_current_preferences(
    mode, settings_factory, tmp_path, monkeypatch,
):
    import lisai.api as api
    import lisai_runner.queue.worker as worker_mod

    first = settings_factory(f"queue_{mode}_a").WORKSPACE_ROOT
    second = settings_factory(f"queue_{mode}_b").WORKSPACE_ROOT
    checkout = tmp_path / "checkout"
    custom = tmp_path / "extra-configs"
    root = {"repo": checkout / "configs", "workspace": first / "configs", "custom": custom}[mode]
    root.mkdir(parents=True, exist_ok=True)
    local = checkout / "configs/local.yml"
    local.parent.mkdir(parents=True, exist_ok=True)
    developer = {
        "dev": {"enabled": True},
        "workspace": {"root": str(first)},
        "config": {"mode": mode, **({"root": str(custom)} if mode == "custom" else {})},
        "console": {"progress_bar": True},
    }
    save_yaml(developer, local)
    locations = resolve_cli_locations(environ={"LISAI_HOME": str(second)}, editable_source_root=checkout)
    assert locations.experiment_config_root == root
    assert locations.local_config_path == local
    paths = Paths(load_settings_for_selected_locations(locations))
    source = root / "training" / "experiment.yml"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text("experiment: {}\n", encoding="utf-8")
    monkeypatch.setattr(queue_cli, "_resolve_training_reference", lambda ref, *, paths: source)
    monkeypatch.setattr(queue_cli, "load_scheduling_context", lambda cfg, *, paths: SimpleNamespace(
        dataset="test", model_subfolder="model", run_name="demo", training_signature=None,
    ))
    assert queue_cli.submit_job(config="experiment", resource_class="light", device="cpu",
                                paths=paths, stdout=io.StringIO()) == 0
    queue_root = default_queue_root(paths=paths)
    records, invalid = discover_jobs(queue_root=queue_root)
    assert not invalid and len(records) == 1
    job = records[0].job
    assert (job.workspace_root, job.config_root, job.local_preferences_path) == (
        str(first), str(root), str(local),
    )

    # A later developer selection edit must not change this job's W/R;
    # a permitted preference edit must be visible when the child launches.
    developer["workspace"]["root"] = str(second)
    developer["console"]["progress_bar"] = False
    save_yaml(developer, local)

    seen = {}
    class FakeProcess:
        pid = 9876
        def poll(self):
            return None
    def fake_popen(command, **kwargs):
        seen["command"] = command
        return FakeProcess()
    monkeypatch.setattr(worker_mod.subprocess, "Popen", fake_popen)
    worker = QueueWorker(paths=paths, queue_root=queue_root, stdout=io.StringIO(), stderr=io.StringIO())
    assert worker._prelaunch_block_reason(job) is None
    assert worker._launch_job(records[0])
    cmd = seen["command"]
    assert cmd[cmd.index("--workspace") + 1] == str(first)
    assert cmd[cmd.index("--config-root") + 1] == str(root)
    assert cmd[cmd.index("--local-preferences") + 1] == str(local)
    worker._running_processes[job.job_id].log_handle.close()

    observed = []
    monkeypatch.setattr(api, "train", lambda cfg, *, settings: observed.append((
        settings.WORKSPACE_ROOT, settings.CONFIGS_ROOT, settings.LOCAL_PROGRESS_BAR,
    )))
    assert training_entrypoint.main(cmd[3:]) == 0
    assert observed == [(first, root, False)]
    selected = queue_cli._job_paths(job, paths)
    assert (selected.workspace_root(), selected.settings.CONFIGS_ROOT,
            selected.settings.LOCAL_PROGRESS_BAR) == (first, root, False)


def test_standalone_jobs_ignore_workspace_local_and_forward_no_preferences(
    settings_factory, tmp_path, monkeypatch,
):
    import lisai_runner.queue.worker as worker_mod
    workspace = settings_factory("queue_standalone").WORKSPACE_ROOT
    save_yaml({"console": {"progress_bar": True}}, workspace / "configs/local.yml")
    paths = Paths(Settings.from_workspace(workspace))
    assert paths.settings.local_preferences_path is None
    assert paths.settings.LOCAL_PROGRESS_BAR is None

    cfg = tmp_path / "cfg.yml"
    cfg.write_text("experiment: {}\n", encoding="utf-8")
    monkeypatch.setattr(queue_cli, "_resolve_training_reference", lambda ref, *, paths: cfg)
    monkeypatch.setattr(queue_cli, "load_scheduling_context", lambda cfg, *, paths: SimpleNamespace(
        dataset="test", model_subfolder="model", run_name="demo", training_signature=None,
    ))
    assert queue_cli.submit_job(config="experiment", resource_class="light", device="cpu",
                                paths=paths, stdout=io.StringIO()) == 0
    queue_root = default_queue_root(paths=paths)
    records, invalid = discover_jobs(queue_root=queue_root)
    assert not invalid and len(records) == 1
    assert records[0].job.local_preferences_path is None

    seen = {}
    class FakeProcess:
        pid = 9877
        def poll(self):
            return None
    monkeypatch.setattr(worker_mod.subprocess, "Popen", lambda args, **kw: (seen.update(args=args), FakeProcess())[1])
    worker = QueueWorker(paths=paths, queue_root=queue_root, stdout=io.StringIO(), stderr=io.StringIO())
    assert worker._launch_job(records[0])
    assert "--local-preferences" not in seen["args"]
    worker._running_processes[records[0].job.job_id].log_handle.close()


def test_missing_submitted_preference_file_blocks_before_launch(settings_factory, tmp_path):
    from lisai_runner.queue.state import create_queued_job
    ws = settings_factory("queue_pref_missing").WORKSPACE_ROOT
    cfg = tmp_path / "cfg.yml"
    cfg.write_text("{}\n", encoding="utf-8")
    missing = tmp_path / "repo/configs/local.yml"
    record = create_queued_job(config_path=cfg, workspace_root=ws, config_root=ws / "configs",
                               local_preferences_path=missing, resource_class="light", device="cpu",
                               queue_root=tmp_path / "queue")
    worker = QueueWorker(paths=Paths(Settings.from_workspace(ws)), queue_root=tmp_path / "queue")
    assert "developer preferences file missing" in worker._prelaunch_block_reason(record.job)

"""M2 F: workspace-owned queues, two-root worker handoff, source provenance."""
from __future__ import annotations

import io
from pathlib import Path
from types import SimpleNamespace

import pytest

from lisai.config.io.yaml import save_yaml
from lisai.config.settings import Settings
from lisai.infra.paths import Paths
from lisai.runs.code_state import collect_code_state
from lisai_runner import training_entrypoint
from lisai_runner.queue import cli as queue_cli
from lisai_runner.queue.state import create_queued_job
from lisai_runner.queue.storage import default_queue_root, discover_jobs, read_job
from lisai_runner.queue.worker import QueueWorker


@pytest.fixture
def contexts(settings_factory, tmp_path, monkeypatch):
    monkeypatch.delenv("LISAI_QUEUE_ROOT", raising=False)
    legacy_a = settings_factory("f_workspace_a")
    legacy_b = settings_factory("f_workspace_b")
    editable = tmp_path / "editable_checkout" / "configs"
    editable.mkdir(parents=True)
    save_yaml({
        "dev": {"enabled": True},
        "workspace": {"root": str(legacy_a.WORKSPACE_ROOT)},
        "config": {"mode": "repo"},
        "console": {"progress_bar": True},
    }, editable / "local.yml")
    a = Paths(Settings.from_workspace(legacy_a.WORKSPACE_ROOT, config_root=editable,
                                      local_config=editable / "local.yml"))
    b = Paths(Settings.from_workspace(legacy_b.WORKSPACE_ROOT, config_root=editable,
                                      local_config=editable / "local.yml"))
    return a, b, editable


def test_queue_and_tensorboard_are_per_workspace_a_b_a(contexts):
    a, b, editable = contexts
    assert a.settings.CONFIGS_ROOT == b.settings.CONFIGS_ROOT == editable
    assert a.workspace_root() != b.workspace_root()
    assert a.settings.local_cfg.console.progress_bar is True
    assert b.settings.local_cfg.console.progress_bar is True
    for paths in (a, b, a):
        expected = paths.workspace_root()
        assert default_queue_root(paths=paths) == expected / ".lisai" / "queue"
        assert paths.tensorboard_dir(dataset_name="my_dataset", tensorboard_subfolder="experiment") == (
            expected / "tensorboard/my_dataset/experiment"
        )


def test_queue_override_remains_explicit(contexts, tmp_path, monkeypatch):
    a, b, _ = contexts
    custom = tmp_path / "shared-external-queue"
    monkeypatch.setenv("LISAI_QUEUE_ROOT", str(custom))
    assert default_queue_root(paths=a) == default_queue_root(paths=b) == custom


def test_queue_submission_persists_both_selected_roots(contexts, monkeypatch):
    a, b, editable = contexts
    source = editable / "training/local/sweep.yml"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text("experiment:\n  exp_name: test\n", encoding="utf-8")
    monkeypatch.setattr(queue_cli, "load_scheduling_context", lambda config, *, paths: SimpleNamespace(
        dataset="dummy", model_subfolder="demo", run_name="run",
        training_signature=None,
    ))
    for paths in (a, b, a):
        assert queue_cli.submit_job(config=str(source), resource_class="light", device="cpu",
                                    paths=paths, stdout=io.StringIO()) == 0
        records, bad = discover_jobs(queue_root=default_queue_root(paths=paths))
        assert not bad
        saved = read_job(records[-1].path)
        assert saved.workspace_root == str(paths.workspace_root())
        assert saved.config_root == str(editable)
        assert saved.local_preferences_path == str(editable / "local.yml")
        assert Path(saved.config).is_file()
        assert Path(saved.config).parent.name == "submitted_configs"
        assert saved.source_config == str(source)
    assert len(discover_jobs(queue_root=default_queue_root(paths=a))[0]) == 2
    assert len(discover_jobs(queue_root=default_queue_root(paths=b))[0]) == 1


def test_foreign_workspace_job_preserves_two_roots_across_worker_and_child(
    contexts, tmp_path, monkeypatch,
):
    import lisai.api as api
    import lisai_runner.queue.worker as worker_module

    a, b, editable = contexts
    queue_root = tmp_path / "shared-queue"
    cfg = tmp_path / "absolute-snapshot.yml"
    cfg.write_text("experiment: {}\n", encoding="utf-8")
    record = create_queued_job(
        config_path=cfg, workspace_root=b.workspace_root(), config_root=editable,
        local_preferences_path=editable / "local.yml",
        resource_class="light", device="cpu", queue_root=queue_root,
    )
    seen = {}

    class FakeProcess:
        pid = 56789
        def poll(self):
            return None

    def fake_popen(cmd, **kwargs):
        seen.update(command=cmd, **kwargs)
        return FakeProcess()

    monkeypatch.setattr(worker_module.subprocess, "Popen", fake_popen)
    worker = QueueWorker(paths=a, queue_root=queue_root, stdout=io.StringIO(), stderr=io.StringIO())
    assert worker._launch_job(record)
    args = seen["command"]
    assert args[args.index("--workspace") + 1] == str(b.workspace_root())
    assert args[args.index("--config-root") + 1] == str(editable)
    assert args[args.index("--config") + 1] == str(cfg)
    assert args[args.index("--local-preferences") + 1] == str(editable / "local.yml")
    assert seen["cwd"] == str(queue_root.resolve())
    assert seen["stdin"] == worker_module.subprocess.DEVNULL
    assert seen["env"]["LISAI_DISABLE_TQDM"] == "1"
    worker._running_processes[record.job.job_id].log_handle.close()

    observed = []
    def fake_train(config, *, settings):
        observed.append((Path(config), settings.WORKSPACE_ROOT,
                         settings.CONFIGS_ROOT, settings.local_cfg.console.progress_bar))
    monkeypatch.setattr(api, "train", fake_train)
    assert training_entrypoint.main(args[3:]) == 0
    assert observed == [(cfg, b.workspace_root(), editable, True)]


def test_queue_job_postprocessing_uses_submission_context(contexts, tmp_path):
    a, b, editable = contexts
    config = tmp_path / "snapshot.yml"
    config.write_text("{}\n", encoding="utf-8")
    from lisai_runner.queue.cli import _job_paths
    record = create_queued_job(
        config_path=config, workspace_root=b.workspace_root(), config_root=editable,
        local_preferences_path=editable / "local.yml",
        resource_class="light", device="cpu", queue_root=tmp_path / "queue",
    )
    selected = _job_paths(record.job, a)
    assert selected.workspace_root() == b.workspace_root()
    assert selected.settings.CONFIGS_ROOT == editable
    assert selected.settings.local_cfg.console.progress_bar is True


def test_old_queue_job_without_workspace_rejects_ambiguous_config_root(contexts, tmp_path):
    a, _, editable = contexts
    from lisai_runner.queue.cli import _job_paths
    record = create_queued_job(
        config_path=tmp_path / "config.yml", config_root=editable,
        resource_class="light", device="cpu", queue_root=tmp_path / "queue",
    )
    with pytest.raises(ValueError, match="lacks workspace_root"):
        _job_paths(record.job, a)


def test_child_rejects_incomplete_root_handoff(tmp_path):
    with pytest.raises(ValueError, match="requires --workspace"):
        training_entrypoint.run_training(tmp_path / "snapshot.yml", config_root=tmp_path)
    with pytest.raises(ValueError, match="requires --config-root"):
        training_entrypoint.run_training(tmp_path / "snapshot.yml", workspace=tmp_path)
    with pytest.raises(ValueError, match="requires --workspace"):
        training_entrypoint.run_training(tmp_path / "snapshot.yml", local_preferences=tmp_path)


def test_workspace_git_repository_is_not_mistaken_for_lisai_source(contexts, monkeypatch):
    import lisai.config.selection as selector
    import lisai.runs.code_state as code_state

    a, _, _ = contexts
    (a.workspace_root() / ".git").mkdir()
    monkeypatch.setattr(selector, "_source_checkout_for_import", lambda: None)
    monkeypatch.setattr(code_state, "_lisai_version", lambda: "0.1.0")
    def not_git(*args, **kwargs):
        raise AssertionError("Git must not be interrogated when no LISAI source checkout is verified")
    monkeypatch.setattr(code_state.subprocess, "run", not_git)
    state = collect_code_state(paths=a)
    assert state.git_commit is None
    assert state.git_branch is None
    assert state.git_dirty is None
    assert state.git_remote is None
    assert state.lisai_version == "0.1.0"


def test_verified_source_repo_used_for_git_not_workspace(contexts, tmp_path, monkeypatch):
    import lisai.config.selection as selector
    import lisai.runs.code_state as code_state

    a, _, _ = contexts
    trusted = tmp_path / "verified-lisai-source"
    trusted.mkdir()
    monkeypatch.setattr(selector, "_source_checkout_for_import", lambda: trusted)
    monkeypatch.setattr(code_state, "_lisai_version", lambda: "0.1.1")
    seen = []
    answers = {
        ("rev-parse", "HEAD"): "verified-commit\n",
        ("branch", "--show-current"): "main\n",
        ("config", "--get", "remote.origin.url"): "git@example/LISAI\n",
        ("status", "--porcelain"): "",
    }
    def fake_git(command, **kwargs):
        assert kwargs["cwd"] == trusted
        seen.append(tuple(command[1:]))
        return SimpleNamespace(returncode=0, stdout=answers[tuple(command[1:])])
    monkeypatch.setattr(code_state.subprocess, "run", fake_git)
    state = collect_code_state(paths=a)
    assert state.git_commit == "verified-commit"
    assert state.git_dirty is False
    assert state.lisai_version == "0.1.1"
    assert len(seen) == 4


def test_incomplete_new_job_identity_is_rejected(tmp_path):
    from pydantic import ValidationError
    with pytest.raises(ValidationError, match="requires config_root"):
        create_queued_job(
            config_path=tmp_path / "cfg.yml", workspace_root=tmp_path / "space",
            resource_class="light", device="cpu", queue_root=tmp_path / "queue",
        )


def test_worker_prelaunch_blocks_missing_workspace_globals(contexts, tmp_path):
    a, b, editable = contexts
    missing = tmp_path / "not-initialized"
    config = tmp_path / "snapshot.yml"
    config.write_text("{}\n", encoding="utf-8")
    record = create_queued_job(
        config_path=config, workspace_root=missing, config_root=editable,
        resource_class="light", device="cpu", queue_root=tmp_path / "queue",
    )
    worker = QueueWorker(paths=a, queue_root=tmp_path / "queue")
    reason = worker._prelaunch_block_reason(record.job)
    assert "workspace init" in reason
    assert "workspace.yml" in reason
    assert "data.yml" in reason


def test_real_child_process_is_cwd_independent_and_uses_both_roots(contexts, tmp_path):
    import json
    import os
    import subprocess
    import sys

    a, b, editable = contexts
    cwd = tmp_path / "arbitrary-non-source-cwd"
    cwd.mkdir()
    cfg = tmp_path / "run_snapshot.yml"
    cfg.write_text("experiment: {}\n", encoding="utf-8")
    source_root = Path(__file__).resolve().parents[3]
    script = (
        "import json, sys; import lisai.api as api; "
        "api.train=lambda path,*,settings: print(json.dumps({"
        "'workspace':str(settings.WORKSPACE_ROOT),"
        "'configs':str(settings.CONFIGS_ROOT),"
        "'progress':settings.local_cfg.console.progress_bar})); "
        "from lisai_runner.training_entrypoint import main; "
        "raise SystemExit(main(sys.argv[1:]))"
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (str(source_root / "src"), env.get("PYTHONPATH"))))
    for selected in (a, b, a):
        proc = subprocess.run([
            sys.executable, "-c", script, "--config", str(cfg),
            "--workspace", str(selected.workspace_root()),
            "--config-root", str(editable),
            "--local-preferences", str(editable / "local.yml"),
        ], cwd=cwd, env=env, capture_output=True, text=True, timeout=40)
        assert proc.returncode == 0, proc.stderr
        payload = json.loads(proc.stdout.strip())
        assert payload == {
            "workspace": str(selected.workspace_root()),
            "configs": str(editable),
            "progress": True,
        }


def test_sweep_jobs_record_workspace_and_independent_config_root(contexts, monkeypatch):
    a, b, editable = contexts
    from lisai.config.io.yaml import save_yaml

    src = editable / "training/local/base.yml"
    src.parent.mkdir(parents=True, exist_ok=True)
    save_yaml({"experiment": {"exp_name": "base"}}, src)
    sweep = editable / "training/local/sweep.yml"
    save_yaml({
        "base_config": str(src),
        "runs": [{"run_name": "sweep_one", "overrides": {}}],
    }, sweep)
    monkeypatch.setattr(queue_cli, "load_scheduling_context", lambda config, *, paths: SimpleNamespace(
        dataset="dataset", model_subfolder="model", run_name="sweep_one",
        training_signature=None,
    ))
    for selected in (a, b):
        assert queue_cli.submit_sweep(
            file=str(sweep), resource_class="light", device="cpu",
            paths=selected, stdout=io.StringIO(), stderr=io.StringIO(),
        ) == 0
        saved, errors = discover_jobs(queue_root=default_queue_root(paths=selected))
        assert not errors
        assert len(saved) == 1
        assert saved[0].job.workspace_root == str(selected.workspace_root())
        assert saved[0].job.config_root == str(editable)
        assert saved[0].job.local_preferences_path == str(editable / "local.yml")
        assert Path(saved[0].job.config).is_file()

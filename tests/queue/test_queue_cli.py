from __future__ import annotations

import io
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import lisai_runner.queue.cli as queue_cli
from lisai_runner.cli import main as root_main
from lisai.config import load_yaml
from lisai.config.settings import load_default_settings
from lisai.infra.paths import Paths
from lisai_runner.queue.history import SchedulingContext
from lisai_runner.queue.schema import QueueJob
from lisai_runner.queue.storage import DiscoveredJob, discover_jobs, queue_logs_dir, queue_state_dir, write_job_atomic
from lisai.runs.schema import TrainingSignature


def test_queue_submit_and_list_via_root_cli(monkeypatch, tmp_path, capsys):
    queue_root = tmp_path / ".lisai" / "queue"
    config_path = tmp_path / "cfg.yml"
    config_path.write_text("{}", encoding="utf-8")

    monkeypatch.setenv("LISAI_QUEUE_ROOT", str(queue_root))
    monkeypatch.setattr(queue_cli, "resolve_config_path", lambda _, *, resolver: config_path)
    monkeypatch.setattr(
        queue_cli,
        "load_scheduling_context",
        lambda _, *, paths: SchedulingContext(
            config_path=config_path,
            dataset="Gag",
            model_subfolder="HDN",
            run_name="demo",
            training_signature=TrainingSignature(architecture="unet", batch_size=8, patch_size=128),
        ),
    )

    submit_exit = root_main(["queue", "submit", "--config", "cfg.yml", "--resource-class", "light"])
    submit_captured = capsys.readouterr()
    assert submit_exit == 0
    assert "Submitted job" in submit_captured.out
    assert "priority: normal" in submit_captured.out

    list_exit = root_main(["queue", "list"])
    listed = capsys.readouterr()
    assert list_exit == 0
    assert "id" in listed.out
    assert "name" in listed.out
    assert "status" in listed.out
    assert "queued" in listed.out


def test_queue_submit_copies_source_config_to_managed_snapshot(monkeypatch, tmp_path):
    queue_root = tmp_path / ".lisai" / "queue"
    source_config = tmp_path / "cfg_source.yaml"
    source_config.write_text("key: value\n# keep exact bytes\n", encoding="utf-8")
    expected_bytes = source_config.read_bytes()

    monkeypatch.setenv("LISAI_QUEUE_ROOT", str(queue_root))
    monkeypatch.setattr(queue_cli, "resolve_config_path", lambda _, *, resolver: source_config)
    monkeypatch.setattr(
        queue_cli,
        "load_scheduling_context",
        lambda path, *, paths: SchedulingContext(
            config_path=Path(path).resolve(),
            dataset="Gag",
            model_subfolder="HDN",
            run_name="demo",
            training_signature=TrainingSignature(architecture="unet", batch_size=8, patch_size=128),
        ),
    )

    exit_code = queue_cli.submit_job(
        config="cfg_source.yaml",
        resource_class="light",
        device="cuda:0",
        stdout=io.StringIO(),
    )
    assert exit_code == 0

    queued_jobs, _invalid = discover_jobs(status="queued", queue_root=queue_root)
    assert len(queued_jobs) == 1
    job = queued_jobs[0].job

    snapshot_path = Path(job.config)
    assert snapshot_path.exists()
    assert snapshot_path.is_file()
    assert snapshot_path.suffix == ".yaml"
    assert snapshot_path.parent == (queue_root / queue_cli.QUEUE_MANAGED_SNAPSHOTS_DIRNAME)
    assert snapshot_path.read_bytes() == expected_bytes
    assert job.source_config == str(source_config.resolve())
    assert job.config != str(source_config.resolve())

    source_config.write_text("key: changed_after_submit\n", encoding="utf-8")
    assert snapshot_path.read_bytes() == expected_bytes


def test_queue_clean_removes_old_done_and_failed_jobs(monkeypatch, tmp_path):
    queue_root = tmp_path / ".lisai" / "queue"
    monkeypatch.setenv("LISAI_QUEUE_ROOT", str(queue_root))

    old_time = datetime(2026, 3, 1, 12, 0, tzinfo=timezone.utc)
    recent_time = datetime(2026, 3, 20, 12, 0, tzinfo=timezone.utc)

    old_done = QueueJob(
        job_id="job_old_done",
        config=str(tmp_path / "a.yml"),
        status="done",
        device="cuda:0",
        submitted_at=old_time - timedelta(hours=1),
        updated_at=old_time,
        resource_class="medium",
        finished_at=old_time,
        exit_code=0,
    )
    recent_failed = QueueJob(
        job_id="job_recent_failed",
        config=str(tmp_path / "b.yml"),
        status="failed",
        device="cuda:0",
        submitted_at=recent_time - timedelta(hours=1),
        updated_at=recent_time,
        resource_class="medium",
        finished_at=recent_time,
        exit_code=1,
        error="boom",
    )

    write_job_atomic(queue_state_dir("done", queue_root=queue_root) / "job_old_done.json", old_done)
    write_job_atomic(
        queue_state_dir("failed", queue_root=queue_root) / "job_recent_failed.json",
        recent_failed,
    )

    monkeypatch.setattr(queue_cli, "utc_now", lambda: datetime(2026, 3, 20, 12, 0, tzinfo=timezone.utc))
    exit_code = queue_cli.clean_jobs(older_than="7d", status=None, clean_all=False)
    assert exit_code == 0

    done_jobs, _ = discover_jobs(status="done", queue_root=queue_root)
    failed_jobs, _ = discover_jobs(status="failed", queue_root=queue_root)
    assert done_jobs == ()
    assert len(failed_jobs) == 1
    assert failed_jobs[0].job.job_id == "job_recent_failed"


def test_queue_clean_only_removes_managed_snapshots(monkeypatch, tmp_path):
    queue_root = tmp_path / ".lisai" / "queue"
    monkeypatch.setenv("LISAI_QUEUE_ROOT", str(queue_root))

    old_time = datetime(2026, 3, 1, 12, 0, tzinfo=timezone.utc)
    source_config = tmp_path / "source.yml"
    source_config.write_text("source: keep\n", encoding="utf-8")
    second_source = tmp_path / "source2.yml"
    second_source.write_text("source: also_keep\n", encoding="utf-8")

    managed_dir = queue_root / queue_cli.QUEUE_MANAGED_SNAPSHOTS_DIRNAME
    managed_dir.mkdir(parents=True, exist_ok=True)
    managed_snapshot = managed_dir / "managed_snapshot.yml"
    managed_snapshot.write_text("snapshot: managed\n", encoding="utf-8")
    external_snapshot = tmp_path / "external_snapshot.yml"
    external_snapshot.write_text("snapshot: external\n", encoding="utf-8")

    managed_job = QueueJob(
        job_id="job_managed_snapshot",
        config=str(managed_snapshot),
        source_config=str(source_config),
        status="done",
        device="cuda:0",
        submitted_at=old_time - timedelta(hours=1),
        updated_at=old_time,
        resource_class="medium",
        finished_at=old_time,
        exit_code=0,
    )
    external_job = QueueJob(
        job_id="job_external_snapshot",
        config=str(external_snapshot),
        source_config=str(second_source),
        status="done",
        device="cuda:0",
        submitted_at=old_time - timedelta(hours=2),
        updated_at=old_time,
        resource_class="medium",
        finished_at=old_time,
        exit_code=0,
    )
    write_job_atomic(
        queue_state_dir("done", queue_root=queue_root) / "job_managed_snapshot.json",
        managed_job,
    )
    write_job_atomic(
        queue_state_dir("done", queue_root=queue_root) / "job_external_snapshot.json",
        external_job,
    )

    exit_code = queue_cli.clean_jobs(older_than=None, status="done", clean_all=False, assume_yes=True)
    assert exit_code == 0

    assert not managed_snapshot.exists()
    assert external_snapshot.exists()
    assert source_config.exists()
    assert second_source.exists()


def test_queue_list_status_filter(monkeypatch, tmp_path, capsys):
    queue_root = tmp_path / ".lisai" / "queue"
    monkeypatch.setenv("LISAI_QUEUE_ROOT", str(queue_root))
    now = datetime(2026, 3, 20, 12, 0, tzinfo=timezone.utc)

    queued_job = QueueJob(
        job_id="job_queued",
        selector="q0001",
        config=str(tmp_path / "queued.yml"),
        status="queued",
        device="cuda:0",
        submitted_at=now - timedelta(minutes=10),
        updated_at=now - timedelta(minutes=10),
        resource_class="medium",
    )
    done_job = QueueJob(
        job_id="job_done",
        selector="q0002",
        config=str(tmp_path / "done.yml"),
        status="done",
        device="cuda:0",
        submitted_at=now - timedelta(hours=2),
        updated_at=now - timedelta(hours=1),
        resource_class="medium",
        finished_at=now - timedelta(hours=1),
        exit_code=0,
    )
    write_job_atomic(queue_state_dir("queued", queue_root=queue_root) / "job_q0001_job_queued.json", queued_job)
    write_job_atomic(queue_state_dir("done", queue_root=queue_root) / "job_q0002_job_done.json", done_job)

    exit_code = root_main(["queue", "list", "--status", "queued"])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "q0001" in captured.out
    assert "job_done" not in captured.out


def test_queue_list_uses_local_timestamp_formatter(monkeypatch, tmp_path, capsys):
    queue_root = tmp_path / ".lisai" / "queue"
    monkeypatch.setenv("LISAI_QUEUE_ROOT", str(queue_root))
    now = datetime(2026, 3, 20, 12, 0, tzinfo=timezone.utc)

    queued_job = QueueJob(
        job_id="job_with_local_ts",
        selector="q0004",
        config=str(tmp_path / "queued.yml"),
        status="queued",
        device="cuda:0",
        submitted_at=now - timedelta(minutes=10),
        updated_at=now - timedelta(minutes=10),
        resource_class="medium",
    )
    write_job_atomic(
        queue_state_dir("queued", queue_root=queue_root) / "job_q0004_job_with_local_ts.json",
        queued_job,
    )

    monkeypatch.setattr(queue_cli, "format_timestamp_local", lambda _value: "LOCAL_TS")

    exit_code = root_main(["queue", "list", "--status", "queued"])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "LOCAL_TS" in captured.out


def test_queue_clean_removes_associated_logs(monkeypatch, tmp_path):
    queue_root = tmp_path / ".lisai" / "queue"
    monkeypatch.setenv("LISAI_QUEUE_ROOT", str(queue_root))
    old_time = datetime(2026, 3, 1, 12, 0, tzinfo=timezone.utc)

    done_job = QueueJob(
        job_id="job_done_for_log_cleanup",
        selector="q0007",
        config=str(tmp_path / "done.yml"),
        status="done",
        device="cuda:0",
        submitted_at=old_time - timedelta(hours=1),
        updated_at=old_time,
        resource_class="medium",
        finished_at=old_time,
        exit_code=0,
    )
    write_job_atomic(
        queue_state_dir("done", queue_root=queue_root) / "job_q0007_job_done_for_log_cleanup.json",
        done_job,
    )
    log_path = queue_logs_dir(queue_root=queue_root) / "job_q0007_job_done_for_log_cleanup.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("hello\n", encoding="utf-8")

    monkeypatch.setattr(queue_cli, "utc_now", lambda: datetime(2026, 3, 20, 12, 0, tzinfo=timezone.utc))
    exit_code = queue_cli.clean_jobs(older_than="7d", status=None, clean_all=False)
    assert exit_code == 0
    assert not log_path.exists()


def test_queue_cancel_queued_job_by_selector(monkeypatch, tmp_path):
    queue_root = tmp_path / ".lisai" / "queue"
    monkeypatch.setenv("LISAI_QUEUE_ROOT", str(queue_root))
    now = datetime(2026, 3, 20, 12, 0, tzinfo=timezone.utc)

    queued_job = QueueJob(
        job_id="job_cancel_me",
        selector="q0003",
        config=str(tmp_path / "queued.yml"),
        status="queued",
        device="cuda:0",
        submitted_at=now - timedelta(minutes=5),
        updated_at=now - timedelta(minutes=5),
        resource_class="medium",
    )
    write_job_atomic(
        queue_state_dir("queued", queue_root=queue_root) / "job_q0003_job_cancel_me.json",
        queued_job,
    )

    exit_code = root_main(["queue", "cancel", "q0003", "--yes"])
    assert exit_code == 0
    queued_jobs, _ = discover_jobs(status="queued", queue_root=queue_root)
    failed_jobs, _ = discover_jobs(status="failed", queue_root=queue_root)
    assert queued_jobs == ()
    assert len(failed_jobs) == 1
    assert failed_jobs[0].job.job_id == "job_cancel_me"
    assert failed_jobs[0].job.error == "cancelled_by_user(queued)"


def test_cancel_eval_policy_follows_config_default(monkeypatch, tmp_path):
    job = QueueJob(
        job_id="job_eval_policy",
        selector="q0009",
        config=str(tmp_path / "cfg.yml"),
        status="running",
        device="cuda:0",
        submitted_at=datetime(2026, 3, 20, 12, 0, tzinfo=timezone.utc),
        updated_at=datetime(2026, 3, 20, 12, 1, tzinfo=timezone.utc),
        resource_class="medium",
        launched_at=datetime(2026, 3, 20, 12, 1, tzinfo=timezone.utc),
    )

    monkeypatch.setattr(
        queue_cli,
        "resolve_config",
        lambda _path, *, stg: SimpleNamespace(experiment=SimpleNamespace(post_training_inference=True)),
    )
    assert queue_cli._should_evaluate_after_cancel(job, paths=Paths(load_default_settings()), eval_override=None, stderr=io.StringIO()) is True

    monkeypatch.setattr(
        queue_cli,
        "resolve_config",
        lambda _path, *, stg: SimpleNamespace(experiment=SimpleNamespace(post_training_inference=False)),
    )
    assert queue_cli._should_evaluate_after_cancel(job, paths=Paths(load_default_settings()), eval_override=None, stderr=io.StringIO()) is False


def test_cancel_eval_policy_override_takes_priority(monkeypatch, tmp_path):
    job = QueueJob(
        job_id="job_eval_override",
        selector="q0010",
        config=str(tmp_path / "cfg.yml"),
        status="running",
        device="cuda:0",
        submitted_at=datetime(2026, 3, 20, 12, 0, tzinfo=timezone.utc),
        updated_at=datetime(2026, 3, 20, 12, 1, tzinfo=timezone.utc),
        resource_class="medium",
        launched_at=datetime(2026, 3, 20, 12, 1, tzinfo=timezone.utc),
    )

    monkeypatch.setattr(
        queue_cli,
        "resolve_config",
        lambda _path, *, stg: SimpleNamespace(experiment=SimpleNamespace(post_training_inference=False)),
    )
    assert queue_cli._should_evaluate_after_cancel(job, paths=Paths(load_default_settings()), eval_override=True, stderr=io.StringIO()) is True
    assert queue_cli._should_evaluate_after_cancel(job, paths=Paths(load_default_settings()), eval_override=False, stderr=io.StringIO()) is False


def test_cancel_running_process_not_found_triggers_eval_hook(monkeypatch, tmp_path):
    queue_root = tmp_path / ".lisai" / "queue"
    monkeypatch.setenv("LISAI_QUEUE_ROOT", str(queue_root))
    now = datetime(2026, 3, 20, 12, 0, tzinfo=timezone.utc)

    running_job = QueueJob(
        job_id="job_cancel_running_eval",
        selector="q0011",
        config=str(tmp_path / "cfg.yml"),
        status="running",
        device="cuda:0",
        submitted_at=now - timedelta(minutes=10),
        updated_at=now - timedelta(minutes=1),
        resource_class="medium",
        launched_at=now - timedelta(minutes=9),
        pid=424242,
    )
    write_job_atomic(
        queue_state_dir("running", queue_root=queue_root) / "job_q0011_job_cancel_running_eval.json",
        running_job,
    )
    records, _invalid = discover_jobs(status="running", queue_root=queue_root)
    assert len(records) == 1
    record: DiscoveredJob = records[0]

    called: list[tuple[str, bool | None]] = []
    monkeypatch.setattr(queue_cli, "_pid_exists", lambda _pid: False)
    monkeypatch.setattr(
        queue_cli,
        "_maybe_run_post_cancel_evaluation",
        lambda job, *, eval_override, stdout, stderr, paths: called.append((job.job_id, eval_override)),
    )

    ok = queue_cli._cancel_single_record(
        record,
        paths=Paths(load_default_settings()),
        queue_root=queue_root,
        force=False,
        allow_force_prompt=False,
        eval_override=True,
        stdin=io.StringIO(""),
        stdout=io.StringIO(),
        stderr=io.StringIO(),
    )
    assert ok is True
    assert called == [("job_cancel_running_eval", True)]


def test_queue_control_commands_pause_resume_and_concurrency(monkeypatch, tmp_path, capsys):
    queue_root = tmp_path / ".lisai" / "queue"
    monkeypatch.setenv("LISAI_QUEUE_ROOT", str(queue_root))

    exit_code = root_main(["queue", "control"])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "paused: False" in captured.out
    assert "max_concurrent_runs_per_gpu: 1" in captured.out

    exit_code = root_main(["queue", "pause"])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Queue paused" in captured.out

    exit_code = root_main(["queue", "control"])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "paused: True" in captured.out

    exit_code = root_main(["queue", "concurrency", "--max-runs-per-gpu", "3"])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "max_concurrent_runs_per_gpu=3" in captured.out

    exit_code = root_main(["queue", "resume"])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Queue resumed" in captured.out

    exit_code = root_main(["queue", "control"])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "paused: False" in captured.out
    assert "max_concurrent_runs_per_gpu: 3" in captured.out


def test_queue_submit_sweep_creates_multiple_jobs_with_overrides(monkeypatch, tmp_path):
    queue_root = tmp_path / ".lisai" / "queue"
    monkeypatch.setenv("LISAI_QUEUE_ROOT", str(queue_root))

    base_config = tmp_path / "HDN_base.yaml"
    base_config.write_text(
        "\n".join(
            [
                "experiment:",
                "  exp_name: HDN_BASE",
                "routing:",
                "  models_subfolder: HDN",
                "data:",
                "  dataset_name: Gag",
                "training_prm:",
                "  betaKL: 0.1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    sweep_file = tmp_path / "hdn_betaKL.yaml"
    sweep_file.write_text(
        "\n".join(
            [
                f"base_config: {base_config}",
                "runs:",
                "  - run_name: HDN_Gag_KL03",
                "    overrides:",
                "      training_prm.betaKL: 0.3",
                "  - run_name: HDN_Gag_KL05",
                "    overrides:",
                "      training_prm.betaKL: 0.5",
                "  - run_name: HDN_Gag_KL07",
                "    overrides:",
                "      training_prm.betaKL: 0.7",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(queue_cli, "resolve_config_path", lambda arg, *, resolver: Path(arg).resolve())

    def fake_context(path: Path, *, paths) -> SchedulingContext:
        cfg = load_yaml(path)
        return SchedulingContext(
            config_path=Path(path).resolve(),
            dataset=str(cfg["data"]["dataset_name"]),
            model_subfolder=str(cfg["routing"]["models_subfolder"]),
            run_name=str(cfg["experiment"]["exp_name"]),
            training_signature=TrainingSignature(architecture="unet", batch_size=8, patch_size=128),
        )

    monkeypatch.setattr(queue_cli, "load_scheduling_context", fake_context)

    exit_code = queue_cli.submit_sweep(
        file=str(sweep_file),
        resource_class="light",
        device="cuda:0",
        priority="high",
        stdout=io.StringIO(),
        stderr=io.StringIO(),
    )
    assert exit_code == 0

    queued_records, _invalid = discover_jobs(status="queued", queue_root=queue_root)
    assert len(queued_records) == 3
    assert [record.job.priority for record in queued_records] == ["high", "high", "high"]
    assert [record.job.run_name for record in queued_records] == [
        "HDN_Gag_KL03",
        "HDN_Gag_KL05",
        "HDN_Gag_KL07",
    ]

    generated_values: list[float] = []
    for record in queued_records:
        cfg = load_yaml(record.job.config)
        generated_values.append(float(cfg["training_prm"]["betaKL"]))
        assert cfg["experiment"]["exp_name"] in {"HDN_Gag_KL03", "HDN_Gag_KL05", "HDN_Gag_KL07"}

    assert sorted(generated_values) == [0.3, 0.5, 0.7]


def test_queue_submit_sweep_resolves_file_from_training_sweeps_dir(monkeypatch, tmp_path):
    queue_root = tmp_path / ".lisai" / "queue"
    monkeypatch.setenv("LISAI_QUEUE_ROOT", str(queue_root))

    training_dir = tmp_path / "configs" / "training"
    sweep_dir = training_dir / "sweeps"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    base_config = training_dir / "hdn.yml"
    base_config.write_text(
        "\n".join(
            [
                "experiment:",
                "  exp_name: HDN_BASE",
                "routing:",
                "  models_subfolder: HDN",
                "data:",
                "  dataset_name: Gag",
                "training:",
                "  learning_rate: 0.00001",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    sweep_file = sweep_dir / "hdn_learning_rate.yaml"
    sweep_file.write_text(
        "\n".join(
            [
                "base_config: hdn.yml",
                "runs:",
                "  - run_name: HDN_Gag_LR1e-05",
                "    overrides:",
                "      training.learning_rate: 0.00001",
                "  - run_name: HDN_Gag_LR5e-05",
                "    overrides:",
                "      training.learning_rate: 0.00005",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    def fake_context(path: Path, *, paths) -> SchedulingContext:
        cfg = load_yaml(path)
        return SchedulingContext(
            config_path=Path(path).resolve(),
            dataset=str(cfg["data"]["dataset_name"]),
            model_subfolder=str(cfg["routing"]["models_subfolder"]),
            run_name=str(cfg["experiment"]["exp_name"]),
            training_signature=TrainingSignature(architecture="unet", batch_size=8, patch_size=128),
        )

    def fake_resolve_config_path(arg: str, *, resolver) -> Path:
        path = Path(arg)
        if path.is_absolute():
            return path.resolve()
        return (training_dir / path).resolve()

    monkeypatch.setattr(queue_cli, "resolve_config_path", fake_resolve_config_path)
    monkeypatch.setattr(queue_cli, "load_scheduling_context", fake_context)

    exit_code = queue_cli.submit_sweep(
        file="sweeps/hdn_learning_rate.yaml",
        resource_class="light",
        device="cuda:0",
        priority="normal",
        stdout=io.StringIO(),
        stderr=io.StringIO(),
    )
    assert exit_code == 0

    queued_records, _invalid = discover_jobs(status="queued", queue_root=queue_root)
    assert len(queued_records) == 2
    assert sorted(record.job.run_name for record in queued_records) == ["HDN_Gag_LR1e-05", "HDN_Gag_LR5e-05"]


def test_queue_show_still_works_after_top_level_unification(monkeypatch, tmp_path, capsys):
    queue_root = tmp_path / ".lisai" / "queue"
    monkeypatch.setenv("LISAI_QUEUE_ROOT", str(queue_root))
    now = datetime(2026, 3, 20, 12, 0, tzinfo=timezone.utc)

    queued_job = QueueJob(
        job_id="job_queue_show_regression",
        selector="q0099",
        config=str(tmp_path / "cfg.yml"),
        status="queued",
        device="cuda:0",
        submitted_at=now,
        updated_at=now,
        resource_class="medium",
        run_name="queue_show_regression",
    )
    write_job_atomic(
        queue_state_dir("queued", queue_root=queue_root) / "job_q0099_job_queue_show_regression.json",
        queued_job,
    )

    exit_code = root_main(["queue", "show", "q0099", "--lines", "0"])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "selector      : q0099" in captured.out
    assert "status        : queued" in captured.out

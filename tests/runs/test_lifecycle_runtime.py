from __future__ import annotations

from lisai.runs.io import read_run_metadata, write_run_metadata_atomic
from lisai.runs.lifecycle import (
    create_run_metadata,
    update_run_failure_reason,
    update_run_recovery_info,
    update_run_runtime_details,
)
from lisai.runs.schema import CodeState, RunMetadata


def _payload(run_dir, **overrides):
    payload = {
        "schema_version": 2,
        "run_id": "01ARZ3NDEKTSV4RRFFQ69G5FAV",
        "run_name": "demo",
        "run_index": 0,
        "dataset": "Gag",
        "model_subfolder": "HDN",
        "status": "running",
        "closed_cleanly": False,
        "created_at": "2026-03-20T10:14:00Z",
        "updated_at": "2026-03-20T10:15:00Z",
        "ended_at": None,
        "last_heartbeat_at": "2026-03-20T10:15:00Z",
        "last_epoch": 2,
        "max_epoch": 10,
        "best_val_loss": 0.5,
        "path": f"datasets/Gag/models/HDN/{run_dir.name}",
        "group_path": None,
    }
    payload.update(overrides)
    return payload


def test_update_run_runtime_details_sets_signature_and_peak(tmp_path):
    run_dir = tmp_path / "demo_00"
    write_run_metadata_atomic(run_dir, RunMetadata.model_validate(_payload(run_dir)))

    update_run_runtime_details(
        run_dir,
        training_signature={
            "architecture": "unet",
            "batch_size": 16,
            "patch_size": 96,
        },
        peak_gpu_mem_mb=5120,
        total_training_time_sec=120.0,
        training_time_per_epoch_sec=12.0,
    )
    metadata = read_run_metadata(run_dir)

    assert metadata.training_signature is not None
    assert metadata.training_signature.architecture == "unet"
    assert metadata.training_signature.batch_size == 16
    assert metadata.training_signature.patch_size == 96
    assert metadata.runtime_stats is not None
    assert metadata.runtime_stats.peak_gpu_mem_mb == 5120
    assert metadata.runtime_stats.total_training_time_sec == 120.0
    assert metadata.runtime_stats.training_time_per_epoch_sec == 12.0
    assert metadata.status == "running"


def test_create_run_metadata_persists_code_state(monkeypatch, tmp_path, settings):
    run_dir = tmp_path / "demo_00"
    expected = CodeState(
        git_commit="abc1234def5678",
        git_branch="main",
        git_dirty=True,
        git_remote="git@github.com:GuillaumeMinet/LISAI.git",
        lisai_version="0.1.0",
    )
    monkeypatch.setattr("lisai.runs.lifecycle.collect_code_state", lambda *, paths: expected)

    from lisai.infra.paths import Paths
    create_run_metadata(run_dir, dataset="Gag", model_subfolder="HDN", max_epoch=10, paths=Paths(settings))
    metadata = read_run_metadata(run_dir)

    assert metadata.code == expected


def test_update_run_runtime_details_keeps_max_peak(tmp_path):
    run_dir = tmp_path / "demo_00"
    write_run_metadata_atomic(
        run_dir,
        RunMetadata.model_validate(
            _payload(
                run_dir,
                runtime_stats={
                    "peak_gpu_mem_mb": 4000,
                },
            )
        ),
    )

    update_run_runtime_details(run_dir, peak_gpu_mem_mb=3500)
    metadata = read_run_metadata(run_dir)
    assert metadata.runtime_stats is not None
    assert metadata.runtime_stats.peak_gpu_mem_mb == 4000


def test_update_run_runtime_details_preserves_legacy_peak_only_stats(tmp_path):
    run_dir = tmp_path / "demo_00"
    write_run_metadata_atomic(
        run_dir,
        RunMetadata.model_validate(
            _payload(
                run_dir,
                runtime_stats={
                    "peak_gpu_mem_mb": 4096,
                },
            )
        ),
    )

    update_run_runtime_details(
        run_dir,
        training_signature={
            "architecture": "unet_rcan",
            "train_batch_size": 8,
            "train_patch_size": 96,
            "val_patch_size": 128,
            "input_channels": 1,
            "upsampling_factor": 2,
        },
    )
    metadata = read_run_metadata(run_dir)

    assert metadata.training_signature is not None
    assert metadata.training_signature.architecture == "unet_rcan"
    assert metadata.runtime_stats is not None
    assert metadata.runtime_stats.peak_gpu_mem_mb == 4096
    assert metadata.runtime_stats.total_training_time_sec is None
    assert metadata.runtime_stats.training_time_per_epoch_sec is None


def test_update_run_recovery_info_updates_failure_and_safe_resume_fields(tmp_path):
    run_dir = tmp_path / "demo_00"
    write_run_metadata_atomic(run_dir, RunMetadata.model_validate(_payload(run_dir)))

    update_run_recovery_info(
        run_dir,
        failure_reason="manual_pause",
        recovery_checkpoint_filename="safe_on_divergence.pth",
        recovery_strategy="hdn_safe_resume_v1",
        last_safe_epoch=5,
        last_safe_batch_id=42,
        safe_resume_fail_count=3,
    )
    metadata = read_run_metadata(run_dir)

    assert metadata.failure_reason == "manual_pause"
    assert metadata.recovery_checkpoint_filename == "safe_on_divergence.pth"
    assert metadata.recovery_strategy == "hdn_safe_resume_v1"
    assert metadata.last_safe_epoch == 5
    assert metadata.last_safe_batch_id == 42
    assert metadata.safe_resume_fail_count == 3


def test_update_run_failure_reason_updates_only_failure_summary(tmp_path):
    run_dir = tmp_path / "demo_00"
    write_run_metadata_atomic(
        run_dir,
        RunMetadata.model_validate(
            _payload(
                run_dir,
                failure_reason="old failure",
                recovery_checkpoint_filename="safe_on_divergence.pth",
                recovery_strategy="hdn_safe_resume_v1",
                last_safe_epoch=4,
                last_safe_batch_id=21,
                safe_resume_fail_count=2,
            )
        ),
    )

    update_run_failure_reason(run_dir, failure_reason="new failure")
    metadata = read_run_metadata(run_dir)

    assert metadata.failure_reason == "new failure"
    assert metadata.recovery_checkpoint_filename == "safe_on_divergence.pth"
    assert metadata.recovery_strategy == "hdn_safe_resume_v1"
    assert metadata.last_safe_epoch == 4
    assert metadata.last_safe_batch_id == 21
    assert metadata.safe_resume_fail_count == 2

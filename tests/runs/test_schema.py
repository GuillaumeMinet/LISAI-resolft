from __future__ import annotations

import re
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from lisai.runs.schema import RunMetadata, format_timestamp_local


def _payload(**overrides):
    payload = {
        "schema_version": 2,
        "run_id": "01ARZ3NDEKTSV4RRFFQ69G5FAV",
        "run_name": "HDN_Gag_KL07",
        "run_index": 1,
        "dataset": "Gag",
        "model_subfolder": "HDN",
        "status": "running",
        "closed_cleanly": False,
        "created_at": "2026-03-20T10:14:00Z",
        "updated_at": "2026-03-20T10:15:00Z",
        "ended_at": None,
        "last_heartbeat_at": "2026-03-20T10:15:00Z",
        "last_epoch": 17,
        "max_epoch": 100,
        "best_val_loss": None,
        "path": "datasets/Gag/models/HDN/HDN_Gag_KL07_01",
        "group_path": None,
    }
    payload.update(overrides)
    return payload


def test_run_metadata_accepts_running_payload():
    metadata = RunMetadata.model_validate(_payload())

    assert metadata.status == "running"
    assert metadata.kept is False
    assert metadata.ended_at is None
    assert metadata.model_subfolder == "HDN"
    assert metadata.run_name == "HDN_Gag_KL07"
    assert metadata.run_index == 1
    assert metadata.safe_resume_fail_count == 0


def test_run_metadata_accepts_pause_related_non_terminal_statuses():
    for status in ("pause_requested", "paused", "resuming"):
        metadata = RunMetadata.model_validate(
            _payload(
                status=status,
                closed_cleanly=False,
                ended_at=None,
            )
        )
        assert metadata.status == status


def test_run_metadata_accepts_legacy_retry_orchestration_fields():
    metadata = RunMetadata.model_validate(
        _payload(
            retry_attempt=3,
            max_retry_attempts=5,
            retry_state={"status": "retrying"},
        )
    )
    payload = metadata.model_dump(mode="python")

    assert "retry_attempt" not in payload
    assert "max_retry_attempts" not in payload
    assert "retry_state" not in payload


def test_run_metadata_accepts_optional_training_signature_and_runtime_stats():
    metadata = RunMetadata.model_validate(
        _payload(
            training_signature={
                "architecture": "unet",
                "batch_size": 8,
                "patch_size": 128,
            },
            runtime_stats={
                "peak_gpu_mem_mb": 4096,
            },
            live_runtime_stats={
                "last_epoch_duration_s": 62.5,
                "recent_epoch_durations_s": [58.0, 60.0, 62.5],
            },
        )
    )

    assert metadata.training_signature is not None
    assert metadata.training_signature.architecture == "unet"
    assert metadata.training_signature.batch_size == 8
    assert metadata.training_signature.patch_size == 128
    assert metadata.runtime_stats is not None
    assert metadata.runtime_stats.peak_gpu_mem_mb == 4096
    assert metadata.live_runtime_stats is not None
    assert metadata.live_runtime_stats.last_epoch_duration_s == pytest.approx(62.5)
    assert metadata.live_runtime_stats.recent_epoch_durations_s == pytest.approx([58.0, 60.0, 62.5])
    assert metadata.live_runtime_stats.median_epoch_duration_s == pytest.approx(60.0)


def test_run_metadata_accepts_optional_code_state():
    metadata = RunMetadata.model_validate(
        _payload(
            code={
                "git_commit": "abc1234def5678",
                "git_branch": "main",
                "git_dirty": True,
                "git_remote": "git@github.com:GuillaumeMinet/LISAI.git",
                "lisai_version": "0.1.0",
            }
        )
    )

    assert metadata.code is not None
    assert metadata.code.git_commit == "abc1234def5678"
    assert metadata.code.git_branch == "main"
    assert metadata.code.git_dirty is True
    assert metadata.code.git_remote == "git@github.com:GuillaumeMinet/LISAI.git"
    assert metadata.code.lisai_version == "0.1.0"


def test_run_metadata_accepts_payload_without_code_state():
    metadata = RunMetadata.model_validate(_payload())

    assert metadata.code is None


def test_run_metadata_accepts_legacy_import_provenance():
    metadata = RunMetadata.model_validate(
        _payload(
            status="completed",
            closed_cleanly=True,
            ended_at="2026-03-20T10:20:00Z",
            provenance={
                "source": "legacy_import",
                "source_path": "/mnt/e/dl_monalisa/deepL_resolft_dataset/Trained_models/CARE",
                "source_config": "config_train.json",
                "imported_at": "2026-03-20T10:19:00Z",
                "notes": "Imported by src/scripts/import_legacy_models.py",
            },
        )
    )
    dumped = metadata.model_dump(mode="json")

    assert metadata.provenance is not None
    assert metadata.provenance.source == "legacy_import"
    assert metadata.provenance.source_config == "config_train.json"
    assert dumped["provenance"]["imported_at"] == "2026-03-20T10:19:00Z"


def test_run_metadata_rejects_negative_live_epoch_duration():
    with pytest.raises(ValidationError):
        RunMetadata.model_validate(
            _payload(
                live_runtime_stats={
                    "last_epoch_duration_s": -1.0,
                }
            )
        )





@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("status", "stale"),
        ("run_id", "not-a-valid-ulid"),
        ("created_at", "not-a-timestamp"),
        ("extra_field", True),

        ("models_subfolder", "HDN"),
    ],
)
def test_run_metadata_rejects_invalid_values(field: str, value):
    payload = _payload()
    payload[field] = value

    with pytest.raises(ValidationError):
        RunMetadata.model_validate(payload)


def test_run_metadata_rejects_missing_required_field():
    payload = _payload()
    payload.pop("dataset")

    with pytest.raises(ValidationError):
        RunMetadata.model_validate(payload)


def test_format_timestamp_local_uses_cli_friendly_layout():
    formatted = format_timestamp_local(datetime(2026, 3, 20, 10, 14, 37, tzinfo=timezone.utc))

    assert re.fullmatch(r"\d{4}-\d{2}-\d{2} - \d{2}:\d{2}", formatted) is not None
    assert "T" not in formatted

from __future__ import annotations

from pathlib import Path

from lisai.runs.scanner import DiscoveredRun
from lisai.runs.schema import RunMetadata, TrainingSignature

from lisai_runner.queue.history import estimate_expected_vram_mb


def _run(
    *,
    run_id: str,
    dataset: str,
    signature: TrainingSignature,
    peak_gpu_mem_mb: int,
) -> DiscoveredRun:
    metadata = RunMetadata.model_validate(
        {
            "schema_version": 2,
            "run_id": run_id,
            "run_name": "demo",
            "run_index": 0,
            "dataset": dataset,
            "model_subfolder": "HDN",
            "status": "completed",
            "closed_cleanly": True,
            "created_at": "2026-03-20T10:00:00Z",
            "updated_at": "2026-03-20T11:00:00Z",
            "ended_at": "2026-03-20T11:00:00Z",
            "last_heartbeat_at": "2026-03-20T11:00:00Z",
            "last_epoch": 10,
            "max_epoch": 10,
            "best_val_loss": 0.1,
            "path": f"datasets/{dataset}/models/HDN/demo_00",
            "group_path": None,
            "training_signature": signature.model_dump(mode="json"),
            "runtime_stats": {"peak_gpu_mem_mb": peak_gpu_mem_mb},
        }
    )
    run_dir = Path(f"/tmp/{dataset}/demo_00")
    return DiscoveredRun(
        metadata=metadata,
        metadata_path=run_dir / ".lisai_run_meta.json",
        run_dir=run_dir,
        dataset=dataset,
        model_subfolder="HDN",
        group_path=None,
        path=f"datasets/{dataset}/models/HDN/demo_00",
        path_consistent=True,
        consistency_issues=(),
    )


def _signature(
    *,
    architecture: str = "unet",
    train_batch_size: int = 8,
    train_patch_size: int = 128,
    val_patch_size: int = 128,
    input_channels: int = 1,
    upsampling_factor: int = 1,
    trainable_params: int | None = None,
) -> TrainingSignature:
    return TrainingSignature(
        architecture=architecture,
        train_batch_size=train_batch_size,
        train_patch_size=train_patch_size,
        val_patch_size=val_patch_size,
        input_channels=input_channels,
        upsampling_factor=upsampling_factor,
        trainable_params=trainable_params,
    )


def test_estimate_expected_vram_prefers_matching_history_over_resource_class():
    target_signature = _signature()
    runs = (
        _run(
            run_id="01ARZ3NDEKTSV4RRFFQ69G5FAA",
            dataset="A",
            signature=target_signature,
            peak_gpu_mem_mb=4500,
        ),
        _run(
            run_id="01ARZ3NDEKTSV4RRFFQ69G5FAB",
            dataset="B",
            signature=target_signature,
            peak_gpu_mem_mb=5200,
        ),
        _run(
            run_id="01ARZ3NDEKTSV4RRFFQ69G5FAC",
            dataset="B",
            signature=_signature(train_batch_size=4),
            peak_gpu_mem_mb=3000,
        ),
    )

    expected, source = estimate_expected_vram_mb(
        signature=target_signature,
        resource_class="light",
        runs=runs,
        resource_defaults_mb={"light": 2000, "medium": 4000, "heavy": 6000},
    )

    assert expected == 5200
    assert source == "history"


def test_estimate_expected_vram_falls_back_to_resource_class_without_history():
    expected, source = estimate_expected_vram_mb(
        signature=_signature(architecture="lvae", train_batch_size=2, train_patch_size=64, val_patch_size=64),
        resource_class="heavy",
        runs=(),
        resource_defaults_mb={"light": 2000, "medium": 4000, "heavy": 6000},
    )

    assert expected == 6000
    assert source == "resource_class:heavy"


def test_estimate_expected_vram_uses_relaxed_matching_when_strict_is_empty():
    target_signature = _signature(
        architecture="unet_rcan",
        train_batch_size=8,
        train_patch_size=96,
        val_patch_size=240,
        input_channels=3,
        upsampling_factor=2,
    )
    runs = (
        _run(
            run_id="01ARZ3NDEKTSV4RRFFQ69G5FAD",
            dataset="A",
            signature=_signature(
                architecture="unet_rcan",
                train_batch_size=4,
                train_patch_size=64,
                val_patch_size=240,
                input_channels=3,
                upsampling_factor=2,
            ),
            peak_gpu_mem_mb=5300,
        ),
        _run(
            run_id="01ARZ3NDEKTSV4RRFFQ69G5FAE",
            dataset="A",
            signature=_signature(
                architecture="unet_rcan",
                train_batch_size=4,
                train_patch_size=64,
                val_patch_size=192,
                input_channels=3,
                upsampling_factor=2,
            ),
            peak_gpu_mem_mb=6000,
        ),
    )

    expected, source = estimate_expected_vram_mb(
        signature=target_signature,
        resource_class="light",
        runs=runs,
        resource_defaults_mb={"light": 2000, "medium": 4000, "heavy": 6000},
    )

    assert expected == 5300
    assert source == "history"


def test_estimate_expected_vram_applies_trainable_params_tiebreaker():
    target_signature = _signature(trainable_params=1_000)
    runs = (
        _run(
            run_id="01ARZ3NDEKTSV4RRFFQ69G5FAF",
            dataset="A",
            signature=_signature(trainable_params=900),
            peak_gpu_mem_mb=4500,
        ),
        _run(
            run_id="01ARZ3NDEKTSV4RRFFQ69G5FAG",
            dataset="A",
            signature=_signature(trainable_params=1_100),
            peak_gpu_mem_mb=4700,
        ),
        _run(
            run_id="01ARZ3NDEKTSV4RRFFQ69G5FAH",
            dataset="A",
            signature=_signature(trainable_params=1_500),
            peak_gpu_mem_mb=6200,
        ),
        _run(
            run_id="01ARZ3NDEKTSV4RRFFQ69G5FAJ",
            dataset="A",
            signature=_signature(trainable_params=None),
            peak_gpu_mem_mb=7000,
        ),
    )

    expected, source = estimate_expected_vram_mb(
        signature=target_signature,
        resource_class="light",
        runs=runs,
        resource_defaults_mb={"light": 2000, "medium": 4000, "heavy": 6000},
    )

    assert expected == 4700
    assert source == "history"

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from lisai.config import resolve_config, settings
from lisai.runs.scanner import DiscoveredRun
from lisai.runs.schema import TrainingSignature
from lisai.runs.signature import build_training_signature_from_resolved_config

from .schema import ResourceClass


@dataclass(frozen=True)
class SchedulingContext:
    config_path: Path
    dataset: str
    model_subfolder: str
    run_name: str
    training_signature: TrainingSignature


def load_scheduling_context(config_path: str | Path) -> SchedulingContext:
    resolved_path = Path(config_path).expanduser().resolve()
    cfg = resolve_config(resolved_path, stg=settings)
    return SchedulingContext(
        config_path=resolved_path,
        dataset=str(cfg.data.dataset_name).strip(),
        model_subfolder=str(cfg.routing.models_subfolder).strip(),
        run_name=str(cfg.experiment.exp_name).strip(),
        training_signature=build_training_signature_from_resolved_config(cfg),
    )


def resource_class_defaults_mb() -> dict[ResourceClass, int]:
    configured = settings.project_cfg.queue.resource_class_vram_mb
    return {
        "light": int(configured.light),
        "medium": int(configured.medium),
        "heavy": int(configured.heavy),
    }


def _patch_key(value: int | list[int] | None) -> tuple[int, ...] | int | None:
    if isinstance(value, list):
        return tuple(int(item) for item in value)
    if value is None:
        return None
    return int(value)


def _strict_match(left: TrainingSignature, right: TrainingSignature) -> bool:
    return (
        left.architecture == right.architecture
        and int(left.train_batch_size) == int(right.train_batch_size)
        and _patch_key(left.train_patch_size) == _patch_key(right.train_patch_size)
        and _patch_key(left.val_patch_size) == _patch_key(right.val_patch_size)
        and left.input_channels == right.input_channels
        and left.upsampling_factor == right.upsampling_factor
    )


def _relaxed_match(left: TrainingSignature, right: TrainingSignature) -> bool:
    return (
        left.architecture == right.architecture
        and _patch_key(left.val_patch_size) == _patch_key(right.val_patch_size)
        and left.input_channels == right.input_channels
        and left.upsampling_factor == right.upsampling_factor
    )


def training_signature_key(
    signature: TrainingSignature,
) -> tuple[str, int, tuple[int, ...] | int | None, tuple[int, ...] | int | None, int | None, int | None]:
    return (
        signature.architecture,
        int(signature.train_batch_size),
        _patch_key(signature.train_patch_size),
        _patch_key(signature.val_patch_size),
        signature.input_channels,
        signature.upsampling_factor,
    )


def signatures_match(left: TrainingSignature, right: TrainingSignature) -> bool:
    return _strict_match(left, right)


def _eligible_historical_runs(
    runs: list[DiscoveredRun] | tuple[DiscoveredRun, ...],
) -> list[DiscoveredRun]:
    candidates: list[DiscoveredRun] = []
    for run in runs:
        metadata = run.metadata
        if not metadata.closed_cleanly:
            continue
        if metadata.training_signature is None:
            continue
        if metadata.runtime_stats is None or metadata.runtime_stats.peak_gpu_mem_mb is None:
            continue
        candidates.append(run)
    return candidates


def _apply_trainable_params_tiebreaker(
    candidates: list[DiscoveredRun],
    *,
    target_signature: TrainingSignature,
) -> list[DiscoveredRun]:
    if target_signature.trainable_params is None:
        return candidates

    with_params: list[tuple[int, DiscoveredRun]] = []
    for candidate in candidates:
        signature = candidate.metadata.training_signature
        if signature is None or signature.trainable_params is None:
            continue
        diff = abs(int(signature.trainable_params) - int(target_signature.trainable_params))
        with_params.append((diff, candidate))

    if not with_params:
        return candidates

    min_diff = min(diff for diff, _candidate in with_params)
    return [candidate for diff, candidate in with_params if diff == min_diff]


def _matching_history_runs(
    *,
    signature: TrainingSignature,
    runs: list[DiscoveredRun] | tuple[DiscoveredRun, ...],
) -> list[DiscoveredRun]:
    eligible = _eligible_historical_runs(runs)
    strict = [run for run in eligible if _strict_match(run.metadata.training_signature, signature)]
    if strict:
        return _apply_trainable_params_tiebreaker(strict, target_signature=signature)

    relaxed = [run for run in eligible if _relaxed_match(run.metadata.training_signature, signature)]
    if relaxed:
        return _apply_trainable_params_tiebreaker(relaxed, target_signature=signature)

    return []


def matching_peak_vram_mb(
    *,
    signature: TrainingSignature,
    runs: list[DiscoveredRun] | tuple[DiscoveredRun, ...],
) -> list[int]:
    candidates = _matching_history_runs(signature=signature, runs=runs)
    peaks: list[int] = []
    for run in candidates:
        metadata = run.metadata
        if metadata.runtime_stats is None or metadata.runtime_stats.peak_gpu_mem_mb is None:
            continue
        peaks.append(int(metadata.runtime_stats.peak_gpu_mem_mb))
    return peaks


def estimate_expected_vram_mb(
    *,
    signature: TrainingSignature | None,
    resource_class: ResourceClass,
    runs: list[DiscoveredRun] | tuple[DiscoveredRun, ...],
    resource_defaults_mb: Mapping[str, int] | None = None,
) -> tuple[int, str]:
    defaults = dict(resource_defaults_mb or resource_class_defaults_mb())

    if signature is not None:
        peaks = matching_peak_vram_mb(signature=signature, runs=runs)
        if peaks:
            return max(peaks), "history"

    return int(defaults[resource_class]), f"resource_class:{resource_class}"


__all__ = [
    "SchedulingContext",
    "build_training_signature_from_resolved_config",
    "estimate_expected_vram_mb",
    "load_scheduling_context",
    "matching_peak_vram_mb",
    "resource_class_defaults_mb",
    "signatures_match",
    "training_signature_key",
]

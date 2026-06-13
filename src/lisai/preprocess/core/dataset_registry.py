from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol

from lisai.data.dataset_registry import (
    load_dataset_registry,
    save_dataset_registry,
    summarize_numeric_range,
)


class PipelineResultLike(Protocol):
    n_files: int
    n_frames: int | None
    snr_levels: int | list[int] | None
    timepoints: int | list[int] | None


DEFAULT_KEYS = ("input", "target", "eval_gt")


def _output_ref(output: dict[str, Any]) -> str:
    if "path" in output:
        return str(output["path"])
    return str(output.get("key") or "")


def _single_output_by_role(outputs: list[dict[str, Any]], role: str) -> str | None:
    candidates = [output for output in outputs if output.get("role") == role]
    if len(candidates) == 1:
        return _output_ref(candidates[0])
    return None


def _defaults_from_outputs(outputs: list[dict[str, Any]], *, usage: str) -> dict[str, str | None]:
    defaults: dict[str, str | None] = {
        "input": _single_output_by_role(outputs, "inp"),
        "target": None,
        "eval_gt": None,
    }
    target_name = _single_output_by_role(outputs, "gt")

    if target_name is not None:
        if usage == "training":
            defaults["target"] = target_name
        defaults["eval_gt"] = target_name
    return defaults


def _apply_default_overrides(
    defaults: dict[str, str | None],
    overrides: Mapping[str, str | None] | None,
) -> dict[str, str | None]:
    if not overrides:
        return defaults

    merged = dict(defaults)
    for key, value in overrides.items():
        if key in DEFAULT_KEYS:
            merged[key] = value
    return merged


@dataclass
class DatasetRegistry:
    path: Path
    data: dict[str, Any] | None = None

    def __post_init__(self):
        self.path = Path(self.path)
        self.data = load_dataset_registry(self.path)

    def save(self) -> None:
        save_dataset_registry(self.data or {}, self.path)

    def ensure_dataset(self, dataset_name: str) -> dict[str, Any]:
        datasets = self.data
        if dataset_name not in datasets:
            datasets[dataset_name] = {
                "data_format": None,
                "for_training": True,
                "usage": "training",
                "defaults": {},
                "outputs": {},
                "size": {},
                "split": {},
                "structure": {},
            }
        return datasets[dataset_name]

    def update_after_preprocess(
        self,
        *,
        dataset_name: str,
        data_type: str,
        data_format: str | None,
        structure: list[str],
        outputs: list[dict[str, Any]],
        result: PipelineResultLike,
        usage: str = "training",
        default_overrides: Mapping[str, str | None] | None = None,
        split_summary: dict[str, Any] | None = None,
    ) -> None:
        ds = self.ensure_dataset(dataset_name)

        if data_format is not None:
            ds["data_format"] = data_format
        ds["usage"] = usage
        ds["for_training"] = usage == "training"

        ds.setdefault("defaults", {})
        ds.setdefault("outputs", {})
        ds.setdefault("size", {})
        ds.setdefault("structure", {})
        ds.setdefault("split", {})

        size_entry: dict[str, Any] = {"n_files": result.n_files}
        if result.n_frames is not None:
            size_entry["n_frames"] = result.n_frames

        if result.snr_levels is not None:
            size_entry["snr_levels"] = summarize_numeric_range(result.snr_levels)
        if result.timepoints is not None:
            size_entry["timepoints"] = summarize_numeric_range(result.timepoints)

        ds["size"][data_type] = size_entry
        ds["structure"][data_type] = structure
        ds["outputs"][data_type] = outputs
        ds["defaults"][data_type] = _apply_default_overrides(
            _defaults_from_outputs(outputs, usage=usage),
            default_overrides,
        )
        ds["split"][data_type] = split_summary or {}

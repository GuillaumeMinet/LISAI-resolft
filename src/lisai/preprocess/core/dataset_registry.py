from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from lisai.data.dataset_registry import load_dataset_registry, save_dataset_registry


class PipelineResultLike(Protocol):
    n_files: int
    n_frames: int | None
    snr_levels: int | list[int] | None


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
        result: PipelineResultLike,
        split_summary: dict[str, Any] | None = None,
    ) -> None:
        ds = self.ensure_dataset(dataset_name)

        if data_format is not None:
            ds["data_format"] = data_format

        ds.setdefault("size", {})
        ds.setdefault("structure", {})
        ds.setdefault("split", {})

        size_entry: dict[str, Any] = {"n_files": result.n_files}
        if result.n_frames is not None:
            size_entry["n_frames"] = result.n_frames

        if result.snr_levels is not None:
            size_entry["snr_levels"] = result.snr_levels

        ds["size"][data_type] = size_entry
        ds["structure"][data_type] = structure
        ds["split"][data_type] = split_summary or {}

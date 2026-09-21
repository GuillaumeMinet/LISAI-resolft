from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class EvalSource:
    """Dataset source represented by one saved evaluation folder."""

    usage: Literal["training", "evaluation"]
    name: str

    @classmethod
    def training_split(cls, split: str) -> "EvalSource":
        split = str(split).strip().lower()
        if split not in {"train", "val", "test"}:
            raise ValueError("Training evaluation split must be 'train', 'val', or 'test'.")
        return cls(usage="training", name=split)

    @classmethod
    def dataset(cls, dataset_name: str) -> "EvalSource":
        name = str(dataset_name).strip()
        if not name:
            raise ValueError("Evaluation dataset name must not be empty.")
        return cls(usage="evaluation", name=name)

    @property
    def folder_name(self) -> str:
        if self.usage == "training":
            return f"training_{self.name}"
        return self.name.replace("\\", "__").replace("/", "__")


__all__ = ["EvalSource"]

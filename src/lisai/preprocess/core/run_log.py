from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from lisai.config import save_yaml

from .sources import Item


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _serialize(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _serialize(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize(item) for item in value]
    if isinstance(value, set):
        return sorted(_serialize(item) for item in value)
    return value


def _source_name(item: Item) -> str:
    if item.source_name is not None:
        return item.source_name
    return item.paths[0].name


def _source_relpaths(item: Item) -> list[str]:
    if item.source_relpaths:
        return list(item.source_relpaths)
    return [path.name for path in item.paths]


@dataclass
class PreprocessRunLog:
    path: Path
    data: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def start(
        cls,
        *,
        path: str | Path,
        dataset_name: str,
        pipeline_name: str,
        data_type: str,
        fmt: str,
        usage: str,
        pipeline_cfg: Mapping[str, Any],
        log_cfg: Mapping[str, Any],
        split_cfg: Mapping[str, Any],
        preprocess_dir: str | Path,
    ) -> "PreprocessRunLog":
        return cls(
            path=Path(path),
            data={
                "version": 1,
                "status": "running",
                "started_at": _now_iso(),
                "dataset_name": dataset_name,
                "pipeline": pipeline_name,
                "data_type": data_type,
                "fmt": fmt,
                "usage": usage,
                "preprocess_dir": str(preprocess_dir),
                "config": {
                    "pipeline_cfg": _serialize(dict(pipeline_cfg)),
                    "log": _serialize(dict(log_cfg)),
                    "split": _serialize(dict(split_cfg)),
                },
                "items": [],
            },
        )

    def record_item(
        self,
        *,
        index: int,
        sample_id: str,
        split: str | None,
        item: Item,
        saved_outputs: Mapping[str, str],
    ) -> None:
        self.data["items"].append(
            {
                "index": index,
                "sample_id": sample_id,
                "split": split,
                "source_key": item.key,
                "source_name": _source_name(item),
                "source_relpaths": _source_relpaths(item),
                "source_paths": [str(path) for path in item.paths],
                **(
                    {
                        "auxiliary_source_paths": {
                            name: str(path) for name, path in item.auxiliary_paths.items()
                        }
                    }
                    if item.auxiliary_paths
                    else {}
                ),
                "saved_outputs": _serialize(dict(saved_outputs)),
            }
        )

    def finalize_success(
        self,
        *,
        result,
        structure: list[str],
        split_summary: dict[str, Any] | None,
    ) -> None:
        self.data["status"] = "success"
        self.data["finished_at"] = _now_iso()
        self.data["summary"] = {
            "result": {
                "n_files": int(result.n_files),
                "n_frames": result.n_frames,
                "snr_levels": _serialize(result.snr_levels),
            },
            "structure": list(structure),
            "split": _serialize(split_summary),
        }
        self.save()

    def finalize_failure(
        self,
        *,
        error: Exception,
        structure: list[str],
        n_files: int,
        stats: Mapping[str, Any],
        split_summary: dict[str, Any] | None,
    ) -> None:
        self.data["status"] = "failed"
        self.data["finished_at"] = _now_iso()
        self.data["summary"] = {
            "partial_result": {
                "n_files": int(n_files),
                "stats": _serialize(dict(stats)),
            },
            "structure": list(structure),
            "split": _serialize(split_summary),
        }
        self.data["error"] = {
            "type": type(error).__name__,
            "message": str(error),
        }
        self.save()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        save_yaml(_serialize(self.data), self.path)

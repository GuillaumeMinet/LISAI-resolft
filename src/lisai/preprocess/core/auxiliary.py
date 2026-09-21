from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
import re
from typing import Iterable, Literal, Mapping

from .constants import MAIN_OUTPUT_KEY
from .sources import Item, Source


AuxiliaryMatching = Literal["required", "optional"]
AuxiliaryMatchBy = Literal["filename", "timestamp"]
AuxiliaryTimestampRelation = Literal["before_primary", "after_primary", "either"]

_TIMESTAMP_RE = re.compile(
    r"(?<!\d)(?P<hours>\d+)h(?P<minutes>\d{1,2})m(?P<seconds>\d{1,2})s"
)


@dataclass(frozen=True)
class AuxiliarySubfolderConfig:
    """Configuration for one named auxiliary image folder."""

    matching: AuxiliaryMatching = field(
        metadata={
            "description": (
                "Whether every primary sample requires an auxiliary match ('required') "
                "or unmatched primary samples are kept without that auxiliary ('optional')."
            )
        },
    )
    match_by: AuxiliaryMatchBy = field(
        default="filename",
        metadata={
            "description": (
                "How auxiliary files are matched to primary samples. Exact filename matching "
                "is the default; timestamp matching uses an HHhMMmSSs timestamp in filenames."
            )
        },
    )
    max_time_delta_s: float | None = field(
        default=None,
        metadata={
            "description": (
                "Maximum allowed timestamp difference in seconds. Required when match_by is 'timestamp'."
            )
        },
    )
    timestamp_relation: AuxiliaryTimestampRelation = field(
        default="either",
        metadata={
            "description": (
                "For timestamp matching, restrict auxiliary acquisitions to before the primary, "
                "after the primary, or either side."
            )
        },
    )


def _validate_auxiliary_config(
    name: str,
    cfg: AuxiliarySubfolderConfig,
    *,
    provided_keys: set[str] | None = None,
) -> AuxiliarySubfolderConfig:
    if cfg.matching not in {"required", "optional"}:
        raise ValueError(
            f"Unsupported matching mode '{cfg.matching}' for auxiliary '{name}'. "
            "Supported values: ['optional', 'required']."
        )

    if cfg.match_by not in {"filename", "timestamp"}:
        raise ValueError(
            f"Unsupported match_by mode '{cfg.match_by}' for auxiliary '{name}'. "
            "Supported values: ['filename', 'timestamp']."
        )

    if cfg.timestamp_relation not in {"before_primary", "after_primary", "either"}:
        raise ValueError(
            f"Unsupported timestamp_relation '{cfg.timestamp_relation}' for auxiliary '{name}'. "
            "Supported values: ['after_primary', 'before_primary', 'either']."
        )

    if cfg.match_by == "timestamp":
        if cfg.max_time_delta_s is None:
            raise ValueError(
                f"Auxiliary '{name}' with match_by='timestamp' requires max_time_delta_s."
            )
        if isinstance(cfg.max_time_delta_s, bool) or not isinstance(cfg.max_time_delta_s, (int, float)):
            raise ValueError(
                f"Auxiliary '{name}' max_time_delta_s must be a non-negative number."
            )
        if cfg.max_time_delta_s < 0:
            raise ValueError(
                f"Auxiliary '{name}' max_time_delta_s must be a non-negative number."
            )
    else:
        if cfg.max_time_delta_s is not None:
            raise ValueError(
                f"Auxiliary '{name}' sets max_time_delta_s but match_by='filename'. "
                "max_time_delta_s is only valid for timestamp matching."
            )
        if provided_keys is not None and "timestamp_relation" in provided_keys:
            raise ValueError(
                f"Auxiliary '{name}' sets timestamp_relation but match_by='filename'. "
                "timestamp_relation is only valid for timestamp matching."
            )
        if provided_keys is None and cfg.timestamp_relation != "either":
            raise ValueError(
                f"Auxiliary '{name}' sets timestamp_relation but match_by='filename'. "
                "timestamp_relation is only valid for timestamp matching."
            )

    return cfg


def normalize_auxiliary_subfolders(
    raw: Mapping[str, AuxiliarySubfolderConfig | Mapping[str, object]] | None,
    *,
    reserved_names: Iterable[str] = (),
) -> dict[str, AuxiliarySubfolderConfig]:
    """Normalize nested auxiliary config while BasePipeline parsing remains shallow."""

    if not raw:
        return {}

    reserved = set(reserved_names) | {MAIN_OUTPUT_KEY}
    normalized: dict[str, AuxiliarySubfolderConfig] = {}

    for name, value in raw.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Auxiliary subfolder names must be non-empty strings.")
        if Path(name).name != name or name in {".", ".."}:
            raise ValueError(
                f"Auxiliary subfolder '{name}' must be a direct folder name. "
                "Custom auxiliary paths are not supported yet."
            )
        if name in reserved:
            raise ValueError(f"Auxiliary subfolder name '{name}' conflicts with a pipeline output name.")

        if isinstance(value, AuxiliarySubfolderConfig):
            cfg = _validate_auxiliary_config(name, value)
        elif isinstance(value, Mapping):
            allowed = {"matching", "match_by", "max_time_delta_s", "timestamp_relation"}
            unknown = sorted(set(value) - allowed)
            if unknown:
                raise ValueError(
                    f"Unknown auxiliary config keys for '{name}': {unknown}. "
                    f"Allowed keys: {sorted(allowed)}"
                )
            if "matching" not in value:
                raise ValueError(
                    f"Auxiliary '{name}' requires an explicit 'matching' value: "
                    "'required' or 'optional'."
                )

            cfg = AuxiliarySubfolderConfig(
                matching=str(value["matching"]),  # type: ignore[arg-type]
                match_by=str(value.get("match_by", "filename")),  # type: ignore[arg-type]
                max_time_delta_s=value.get("max_time_delta_s"),  # type: ignore[arg-type]
                timestamp_relation=str(value.get("timestamp_relation", "either")),  # type: ignore[arg-type]
            )
            cfg = _validate_auxiliary_config(name, cfg, provided_keys=set(value))
        else:
            raise ValueError(
                f"Auxiliary subfolder '{name}' must be configured as a mapping, e.g. "
                f"{name}: {{matching: required}}."
            )

        normalized[name] = cfg

    return normalized


def _timestamp_seconds(filename: str) -> int:
    match = _TIMESTAMP_RE.search(filename)
    if match is None:
        raise ValueError(
            f"Could not extract timestamp from filename '{filename}'. "
            "Expected a timestamp such as '00h22m40s'."
        )

    hours = int(match.group("hours"))
    minutes = int(match.group("minutes"))
    seconds = int(match.group("seconds"))
    if minutes >= 60 or seconds >= 60:
        raise ValueError(
            f"Invalid timestamp in filename '{filename}': minutes and seconds must be < 60."
        )
    return hours * 3600 + minutes * 60 + seconds


class AuxiliaryFolderMatcher:
    """Resolve configured auxiliary folders against primary source filenames."""

    def __init__(
        self,
        *,
        root: Path,
        auxiliary_subfolders: Mapping[str, AuxiliarySubfolderConfig | Mapping[str, object]] | None,
        exts: tuple[str, ...],
        reserved_names: Iterable[str] = (),
    ) -> None:
        self.root = Path(root)
        self.configs = normalize_auxiliary_subfolders(
            auxiliary_subfolders,
            reserved_names=reserved_names,
        )
        self.indices: dict[str, dict[str, Path]] = {}

        for name in self.configs:
            folder = self.root / name
            if not folder.is_dir():
                raise FileNotFoundError(f"Auxiliary folder does not exist: {folder}")

            self.indices[name] = {
                path.name: path
                for path in sorted(folder.iterdir())
                if path.is_file() and path.suffix.lower() in exts
            }

    @property
    def requires_batch_matching(self) -> bool:
        return any(cfg.match_by == "timestamp" for cfg in self.configs.values())

    def match(self, source_name: str) -> dict[str, Path]:
        """Match one item when all configured auxiliaries use filename matching."""
        if self.requires_batch_matching:
            raise RuntimeError("Timestamp auxiliary matching requires batch matching.")

        matches: dict[str, Path] = {}
        for name, cfg in self.configs.items():
            path = self.indices[name].get(source_name)
            if path is None:
                if cfg.matching == "required":
                    raise ValueError(
                        f"Missing required auxiliary '{name}' for source file '{source_name}'. "
                        f"Expected: {self.root / name / source_name}"
                    )
                continue
            matches[name] = path
        return matches

    def match_many(self, source_names: list[str]) -> list[dict[str, Path]]:
        """Match auxiliaries for a batch of primary filenames."""
        matches: list[dict[str, Path]] = [{} for _ in source_names]

        for name, cfg in self.configs.items():
            if cfg.match_by == "filename":
                for index, source_name in enumerate(source_names):
                    path = self.indices[name].get(source_name)
                    if path is None:
                        if cfg.matching == "required":
                            raise ValueError(
                                f"Missing required auxiliary '{name}' for source file '{source_name}'. "
                                f"Expected: {self.root / name / source_name}"
                            )
                        continue
                    matches[index][name] = path
                continue

            timestamp_matches = self._match_timestamps(
                auxiliary_name=name,
                cfg=cfg,
                source_names=source_names,
            )
            for index, path in timestamp_matches.items():
                matches[index][name] = path

        return matches

    def _match_timestamps(
        self,
        *,
        auxiliary_name: str,
        cfg: AuxiliarySubfolderConfig,
        source_names: list[str],
    ) -> dict[int, Path]:
        import numpy as np
        from scipy.optimize import linear_sum_assignment

        max_delta = float(cfg.max_time_delta_s)  # validated during normalization
        auxiliary_index = self.indices[auxiliary_name]
        auxiliary_names = sorted(auxiliary_index)

        primary_times = [_timestamp_seconds(name) for name in source_names]
        auxiliary_times = [_timestamp_seconds(name) for name in auxiliary_names]

        n_primary = len(source_names)
        n_auxiliary = len(auxiliary_names)
        if n_primary == 0:
            return {}

        unmatched_cost = max_delta + 1.0
        invalid_cost = unmatched_cost * (n_primary + n_auxiliary + 2) + 1.0
        cost = np.full(
            (n_primary, n_auxiliary + n_primary),
            unmatched_cost,
            dtype=float,
        )
        if n_auxiliary:
            cost[:, :n_auxiliary] = invalid_cost

        for primary_index, primary_time in enumerate(primary_times):
            for auxiliary_index_pos, auxiliary_time in enumerate(auxiliary_times):
                if cfg.timestamp_relation == "before_primary" and auxiliary_time > primary_time:
                    continue
                if cfg.timestamp_relation == "after_primary" and auxiliary_time < primary_time:
                    continue

                delta = abs(primary_time - auxiliary_time)
                if delta <= max_delta:
                    cost[primary_index, auxiliary_index_pos] = float(delta)

        row_indices, column_indices = linear_sum_assignment(cost)
        resolved: dict[int, Path] = {}
        for row, column in zip(row_indices.tolist(), column_indices.tolist()):
            if column >= n_auxiliary:
                continue
            if cost[row, column] > max_delta:
                continue
            resolved[row] = auxiliary_index[auxiliary_names[column]]

        if cfg.matching == "required" and len(resolved) != n_primary:
            unmatched = [
                source_names[index]
                for index in range(n_primary)
                if index not in resolved
            ]
            raise ValueError(
                f"Missing required auxiliary '{auxiliary_name}' timestamp match for "
                f"{len(unmatched)} source file(s) within max_time_delta_s={max_delta:g}: {unmatched}"
            )

        return resolved


@dataclass(frozen=True)
class AuxiliarySource(Source):
    """Decorate a primary Source with named auxiliary paths."""

    source: Source
    matcher: AuxiliaryFolderMatcher

    def iter_items(self):
        if not self.matcher.requires_batch_matching:
            for item in self.source.iter_items():
                source_name = item.source_name or item.paths[0].name
                yield replace(
                    item,
                    auxiliary_paths=self.matcher.match(source_name),
                )
            return

        items = list(self.source.iter_items())
        source_names = [item.source_name or item.paths[0].name for item in items]
        auxiliary_matches = self.matcher.match_many(source_names)
        for item, matches in zip(items, auxiliary_matches):
            yield replace(item, auxiliary_paths=matches)

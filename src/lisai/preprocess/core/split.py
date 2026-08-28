from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable, Mapping, Sequence

from lisai.config import load_yaml

from .config import PreprocessSplitConfig, SplitMatchBy
from .sources import Item

VALID_SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class SplitPlan:
    assignments: dict[int, str]
    mode: str
    counts: dict[str, int]
    details: dict[str, Any]

    def split_for(self, index: int) -> str:
        return self.assignments.get(index, "train")

    def summary(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "mode": self.mode,
            "counts": dict(self.counts),
            **self.details,
        }


def _normalize_identifier(value: str) -> str:
    return value.replace("\\", "/")


def _identifier_aliases(value: str, *, include_stem: bool) -> set[str]:
    normalized = _normalize_identifier(value)
    path = PurePosixPath(normalized)
    aliases = {normalized}
    if path.name:
        aliases.add(path.name)
    if include_stem:
        aliases.add(path.stem)
        without_suffix = path.with_suffix("").as_posix()
        if without_suffix:
            aliases.add(without_suffix)
    return {alias for alias in aliases if alias}


def _item_source_name(item: Item) -> str:
    if item.source_name is not None:
        return item.source_name
    return item.paths[0].name


def _item_source_relpath(item: Item) -> str:
    if item.source_relpaths:
        return item.source_relpaths[0]
    return _item_source_name(item)


def _item_aliases(item: Item, *, match_by: SplitMatchBy, sample_id: str) -> set[str]:
    if match_by == "sample_id":
        return {sample_id}
    if match_by == "source_relpath":
        return _identifier_aliases(_item_source_relpath(item), include_stem=True)
    return _identifier_aliases(_item_source_name(item), include_stem=True)


def _build_alias_lookup(
    items: list[Item],
    *,
    match_by: SplitMatchBy,
    sample_ids: list[str],
) -> tuple[dict[str, int], set[str]]:
    alias_to_index: dict[str, int] = {}
    ambiguous: set[str] = set()

    for index, (item, sample_id) in enumerate(zip(items, sample_ids, strict=True)):
        for alias in _item_aliases(item, match_by=match_by, sample_id=sample_id):
            existing = alias_to_index.get(alias)
            if existing is not None and existing != index:
                ambiguous.add(alias)
            else:
                alias_to_index[alias] = index

    for alias in ambiguous:
        alias_to_index.pop(alias, None)

    return alias_to_index, ambiguous


def _resolve_indices(
    identifiers: Iterable[str],
    *,
    alias_to_index: dict[str, int],
    ambiguous: set[str],
) -> list[int]:
    resolved: list[int] = []
    missing: list[str] = []
    ambiguous_hits: list[str] = []

    for identifier in identifiers:
        normalized = _normalize_identifier(identifier)
        if normalized in ambiguous:
            ambiguous_hits.append(identifier)
            continue
        index = alias_to_index.get(normalized)
        if index is None:
            missing.append(identifier)
            continue
        resolved.append(index)

    if ambiguous_hits:
        raise ValueError(f"Ambiguous split identifiers: {sorted(ambiguous_hits)}")
    if missing:
        raise ValueError(f"Unknown split identifiers: {sorted(missing)}")
    return resolved


def _counts_from_assignments(assignments: dict[int, str], n_items: int) -> dict[str, int]:
    counts = {split: 0 for split in VALID_SPLITS}
    for index in range(n_items):
        counts[assignments.get(index, "train")] += 1
    return counts


def _resolve_manifest_path(
    *,
    manifest_path: str | None,
    dataset_name: str | None,
    data_type: str | None,
    current_data_type: str,
    paths,
) -> Path:
    if manifest_path is not None:
        return Path(manifest_path).expanduser().resolve()
    if dataset_name is None:
        raise ValueError("Reuse split requires either `manifest_path` or `dataset_name`.")
    return paths.preprocess_log_path(
        dataset_name=dataset_name,
        data_type=data_type or current_data_type,
    )


def _manifest_identifier(item: dict[str, Any], *, match_by: SplitMatchBy) -> str | None:
    if match_by == "sample_id":
        value = item.get("sample_id")
    elif match_by == "source_relpath":
        relpaths = item.get("source_relpaths") or []
        value = relpaths[0] if relpaths else item.get("source_name")
    else:
        value = item.get("source_name")
    if value is None:
        return None
    return _normalize_identifier(str(value))


def _empty_split_bucket() -> dict[str, Any]:
    return {
        "count": 0,
        "source_names": [],
        "output_names": [],
    }


def summarize_processed_splits(
    *,
    processed_items: Sequence[Mapping[str, Any]],
    split_plan: SplitPlan | None,
    include_names: bool,
) -> dict[str, Any]:
    if split_plan is None and all(item.get("split") is None for item in processed_items):
        return {"enabled": False}

    buckets = {split_name: _empty_split_bucket() for split_name in VALID_SPLITS}

    for processed_item in processed_items:
        split_name = str(processed_item.get("split") or "train")
        if split_name not in VALID_SPLITS:
            raise ValueError(f"Unknown split name '{split_name}' in processed split summary.")

        bucket = buckets[split_name]
        bucket["count"] += 1
        if include_names:
            bucket["source_names"].append(str(processed_item.get("source_name") or ""))
            bucket["output_names"].append(
                str(processed_item.get("output_name") or processed_item.get("sample_id") or "")
            )

    counts = {split_name: buckets[split_name]["count"] for split_name in VALID_SPLITS}
    summary: dict[str, Any] = {
        "enabled": split_plan is not None,
        "mode": split_plan.mode if split_plan is not None else None,
        "counts": counts,
    }
    if split_plan is not None:
        summary.update(split_plan.details)
    if include_names:
        summary.update({split_name: buckets[split_name] for split_name in VALID_SPLITS})
    return summary


def plan_split(
    *,
    items: list[Item],
    split_cfg: PreprocessSplitConfig,
    sample_id_fn: Callable[[int], str],
    dataset_name: str,
    data_type: str,
    paths,
) -> SplitPlan:
    if not split_cfg.enabled:
        raise ValueError("plan_split should only be called when split.enabled is true.")

    sample_ids = [sample_id_fn(index) for index in range(len(items))]

    if split_cfg.mode == "random":
        random_cfg = split_cfg.random
        indices = list(range(len(items)))
        rng = random.Random(random_cfg.seed)
        rng.shuffle(indices)

        n_val = round(random_cfg.val_fraction * len(items))
        n_test = round(random_cfg.test_fraction * len(items))

        assignments = {index: "train" for index in range(len(items))}
        for index in indices[:n_val]:
            assignments[index] = "val"
        for index in indices[n_val : n_val + n_test]:
            assignments[index] = "test"

        return SplitPlan(
            assignments=assignments,
            mode="random",
            counts=_counts_from_assignments(assignments, len(items)),
            details={
                "seed": random_cfg.seed,
                "val_fraction": random_cfg.val_fraction,
                "test_fraction": random_cfg.test_fraction,
            },
        )

    match_by: SplitMatchBy
    assignments = {index: "train" for index in range(len(items))}

    if split_cfg.mode == "manual":
        manual_cfg = split_cfg.manual
        match_by = manual_cfg.match_by
        alias_to_index, ambiguous = _build_alias_lookup(items, match_by=match_by, sample_ids=sample_ids)
        val_indices = _resolve_indices(manual_cfg.val, alias_to_index=alias_to_index, ambiguous=ambiguous)
        test_indices = _resolve_indices(manual_cfg.test, alias_to_index=alias_to_index, ambiguous=ambiguous)

        overlap = sorted(set(val_indices) & set(test_indices))
        if overlap:
            raise ValueError(f"Manual split targets overlap after resolution: {overlap}")

        for index in val_indices:
            assignments[index] = "val"
        for index in test_indices:
            assignments[index] = "test"

        return SplitPlan(
            assignments=assignments,
            mode="manual",
            counts=_counts_from_assignments(assignments, len(items)),
            details={"match_by": match_by},
        )

    reuse_cfg = split_cfg.reuse
    if reuse_cfg is None:
        raise ValueError("`split.reuse` must be provided when `split.mode='reuse'`.")

    match_by = reuse_cfg.match_by
    alias_to_index, ambiguous = _build_alias_lookup(items, match_by=match_by, sample_ids=sample_ids)
    manifest_path = _resolve_manifest_path(
        manifest_path=reuse_cfg.manifest_path,
        dataset_name=reuse_cfg.dataset_name,
        data_type=reuse_cfg.data_type,
        current_data_type=data_type,
        paths=paths,
    )
    manifest = load_yaml(manifest_path)
    missing: list[str] = []

    for logged_item in manifest.get("items", []):
        identifier = _manifest_identifier(logged_item, match_by=match_by)
        if identifier is None:
            continue
        if identifier in ambiguous:
            raise ValueError(f"Ambiguous reuse split identifier: {identifier}")
        index = alias_to_index.get(identifier)
        if index is None:
            missing.append(identifier)
            continue
        split_name = str(logged_item.get("split") or "train")
        if split_name not in VALID_SPLITS:
            raise ValueError(f"Unknown split name '{split_name}' in manifest {manifest_path}")
        assignments[index] = split_name

    if missing:
        raise ValueError(
            f"Could not match manifest identifiers in current preprocess source: {sorted(missing)}"
        )

    return SplitPlan(
        assignments=assignments,
        mode="reuse",
        counts=_counts_from_assignments(assignments, len(items)),
        details={
            "match_by": match_by,
            "manifest_path": str(manifest_path),
            "source_dataset": reuse_cfg.dataset_name,
            "source_data_type": reuse_cfg.data_type or data_type,
        },
    )


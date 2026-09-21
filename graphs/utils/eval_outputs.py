from __future__ import annotations

import warnings
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path, PurePosixPath
from typing import Iterable, Mapping, Sequence

import numpy as np
from tifffile import TiffFile, imread

from lisai.config import load_yaml, settings
from lisai.data.dataset_registry import (
    load_dataset_info,
    registry_data_types,
    registry_output_axes,
    registry_value_for_data_type,
)
from lisai.evaluation import load_outputs_manifest
from lisai.infra.paths import Paths

from graphs.utils.eval_folder import RunEvaluation


_OUTPUT_KEYS = (
    "inp",
    "gt",
    "pred",
    "samples",
)

_TIFF_SUFFIXES = {
    ".tif",
    ".tiff",
}




_UNKNOWN = "unknown"


class UnalignedComparisonWarning(UserWarning):
    """Comparing runs without source-sample alignment can change interpretation."""


@dataclass(frozen=True)
class OutputSpec:
    files: tuple[Path, ...]
    sample_axis: int | None

    def sample_count(self) -> int:
        if self.sample_axis is None:
            return len(self.files)
        if len(self.files) != 1:
            raise ValueError("A stacked manifest output must reference exactly one file.")
        shape = _tiff_shape(self.files[0])
        axis = self.sample_axis if self.sample_axis >= 0 else self.sample_axis + len(shape)
        if not 0 <= axis < len(shape):
            raise ValueError(f"Invalid sample_axis={self.sample_axis} for {self.files[0]} with shape {shape}.")
        return int(shape[axis])

    def load(self, positions: Sequence[int]) -> tuple[np.ndarray, ...]:
        positions = tuple(positions)
        if self.sample_axis is None:
            return tuple(np.asarray(imread(self.files[i])) for i in positions)

        array = np.asarray(imread(self.files[0]))
        axis = self.sample_axis if self.sample_axis >= 0 else self.sample_axis + array.ndim
        if not 0 <= axis < array.ndim:
            raise ValueError(f"Invalid sample_axis={self.sample_axis} for {self.files[0]} with shape {array.shape}.")
        selected = np.take(array, positions, axis=axis)
        selected = np.moveaxis(selected, axis, 0)
        return tuple(selected[i] for i in range(len(positions)))


@dataclass(frozen=True)
class DatasetOutputSpec:
    path: str
    axes: str | None


@dataclass(frozen=True)
class DatasetOutputsSpec:
    dataset_name: str
    usage: str
    data_type: str
    data_dir: Path
    recorded_data_dir: Path | None
    input_path: str
    outputs: Mapping[str, DatasetOutputSpec]
    gt: DatasetOutputSpec | None = None


@dataclass(frozen=True)
class ManifestEvalItem:
    name: str
    input_id: str
    gt_id: str | None
    source_axis: str | None
    source_indices: tuple[int, ...] | str | None
    outputs: Mapping[str, OutputSpec]
    dataset_outputs: DatasetOutputsSpec | None = None

    @property
    def identity(self) -> tuple[str, str | None]:
        return self.input_id, self.gt_id

    @property
    def sample_count(self) -> int:
        if isinstance(self.source_indices, tuple):
            return len(self.source_indices)
        return next(iter(self.outputs.values())).sample_count()

    def select(self, positions: Sequence[int]) -> "SelectedEvalItem":
        positions = tuple(int(i) for i in positions)
        if any(i < 0 or i >= self.sample_count for i in positions):
            raise IndexError(f"Selection is out of range for evaluation item {self.name!r}.")
        return SelectedEvalItem(self, positions)


@dataclass(frozen=True)
class SelectedEvalItem:
    item: ManifestEvalItem
    positions: tuple[int, ...]

    @property
    def source_indices(self) -> tuple[int, ...] | str | None:
        indices = self.item.source_indices
        if isinstance(indices, tuple):
            return tuple(indices[i] for i in self.positions)
        return indices

    def load_samples(self, key: str, *, squeeze: bool = False) -> tuple[np.ndarray, ...]:
        if key in self.item.outputs:
            samples = self.item.outputs[key].load(self.positions)
            if squeeze:
                samples = tuple(np.squeeze(sample) for sample in samples)
            return samples

        dataset_outputs = self.item.dataset_outputs
        if key == "gt" and dataset_outputs is not None and dataset_outputs.gt is not None:
            return self._load_dataset_output(dataset_outputs.gt, dataset_outputs=dataset_outputs, squeeze=squeeze)
        raise KeyError(f"Evaluation item {self.item.name!r} has no {key!r} output.")

    def load_gt_pred(self, *, squeeze: bool = True) -> tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...]]:
        return (
            self.load_samples("gt", squeeze=squeeze),
            self.load_samples("pred", squeeze=squeeze),
        )

    def load_aux_data(self, name: str, *, squeeze: bool = True) -> tuple[np.ndarray, ...]:
        """Load one registered auxiliary dataset output aligned to this selected item."""
        dataset_outputs = self.item.dataset_outputs
        if dataset_outputs is None:
            raise KeyError(
                f"Evaluation item {self.item.name!r} has no registered auxiliary dataset outputs."
            )
        if name not in dataset_outputs.outputs:
            available = ", ".join(sorted(dataset_outputs.outputs)) or "<none>"
            raise KeyError(
                f"Auxiliary output {name!r} is not registered for dataset {dataset_outputs.dataset_name!r}. "
                f"Available auxiliary outputs: {available}."
            )

        return self._load_dataset_output(
            dataset_outputs.outputs[name],
            dataset_outputs=dataset_outputs,
            squeeze=squeeze,
        )

    def load_dataset_gt(
        self,
        *,
        dataset_outputs: DatasetOutputsSpec | None = None,
        squeeze: bool = True,
    ) -> tuple[np.ndarray, ...]:
        """Load the registered dataset GT aligned to this selected item."""
        resolved = dataset_outputs or self.item.dataset_outputs
        if resolved is None or resolved.gt is None:
            raise KeyError(f"Evaluation item {self.item.name!r} has no registered dataset GT.")
        return self._load_dataset_output(resolved.gt, dataset_outputs=resolved, squeeze=squeeze)

    def _load_dataset_output(
        self,
        output: DatasetOutputSpec,
        *,
        dataset_outputs: DatasetOutputsSpec,
        squeeze: bool,
    ) -> tuple[np.ndarray, ...]:
        if dataset_outputs is None:
            raise KeyError(f"Evaluation item {self.item.name!r} has no registered dataset outputs.")
        relative_path = _relative_input_item_path(self.item.input_id, dataset_outputs.input_path)
        path = dataset_outputs.data_dir / output.path / relative_path
        if not path.is_file():
            raise FileNotFoundError(
                f"Dataset output {output.path!r} for evaluation item {self.item.name!r} was not found: {path}"
            )

        array = np.asarray(imread(path))
        if self.item.source_axis is None:
            samples = (array,)
        elif not _output_has_source_axis(output.axes):
            samples = tuple(array for _ in self.positions)
        else:
            indices = self.source_indices
            if not isinstance(indices, tuple):
                raise ValueError(
                    f"Cannot align dataset output {output.path!r} for evaluation item {self.item.name!r}: "
                    f"{self.item.source_axis} source indices are unavailable."
                )
            if array.ndim == 0:
                raise ValueError(f"Dataset output {path} has no source axis to select.")
            if indices and max(indices) >= array.shape[0]:
                raise IndexError(
                    f"Dataset output {path} has {array.shape[0]} entries on its source axis, "
                    f"but index {max(indices)} is required."
                )
            selected = np.take(array, indices, axis=0)
            samples = tuple(selected[i] for i in range(len(indices)))

        if squeeze:
            samples = tuple(np.squeeze(sample) for sample in samples)
        return samples


@dataclass(frozen=True)
class ComparisonOutputs:
    evaluations: tuple[RunEvaluation, ...]
    items_by_run: Mapping[Path, tuple[SelectedEvalItem, ...]]
    aligned: bool
    gt_reference: str | RunEvaluation
    dataset_reference: DatasetOutputsSpec | None = None

    def items_for(self, evaluation: RunEvaluation) -> tuple[SelectedEvalItem, ...]:
        return self.items_by_run[evaluation.run_dir]

    def load_reference_gt(self, item: SelectedEvalItem, *, squeeze: bool = True) -> tuple[np.ndarray, ...]:
        """Load the GT selected for this comparison."""
        if self.gt_reference == "per_run":
            return item.load_samples("gt", squeeze=squeeze)
        if self.gt_reference == "dataset":
            if self.dataset_reference is None:
                raise ValueError("This comparison has no shared dataset GT reference.")
            return item.load_dataset_gt(dataset_outputs=self.dataset_reference, squeeze=squeeze)

        reference = self.gt_reference
        assert isinstance(reference, RunEvaluation)
        reference_item = self._matching_item(reference, item)
        return reference_item.load_samples("gt", squeeze=squeeze)

    def load_gt_pred(
        self,
        item: SelectedEvalItem,
        *,
        squeeze: bool = True,
    ) -> tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...]]:
        """Load a prediction paired with this comparison's selected GT reference."""
        return self.load_reference_gt(item, squeeze=squeeze), item.load_samples("pred", squeeze=squeeze)

    def _matching_item(self, evaluation: RunEvaluation, item: SelectedEvalItem) -> SelectedEvalItem:
        matches = [
            candidate
            for candidate in self.items_for(evaluation)
            if candidate.item.identity == item.item.identity
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Could not resolve one reference item for {item.item.identity!r} in {evaluation.folder}."
            )
        reference_item = matches[0]
        if reference_item.source_indices != item.source_indices:
            raise ValueError(
                f"Reference item {item.item.identity!r} is not aligned to the same source indices."
            )
        return reference_item


def load_eval_output_items(
    evaluation: RunEvaluation,
    *,
    required: Sequence[str] = ("gt", "pred"),
) -> tuple[ManifestEvalItem, ...]:
    """Load output files and provenance from an evaluation manifest."""
    manifest_path = evaluation.folder / "outputs_manifest.yaml"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"{evaluation.folder} has no outputs_manifest.yaml; exact alignment is unavailable."
        )

    dataset_outputs = _load_dataset_outputs_spec(evaluation)
    items = tuple(
        _manifest_item(evaluation.folder, record, dataset_outputs=dataset_outputs)
        for record in load_outputs_manifest(manifest_path)["items"]
    )
    if not items:
        raise FileNotFoundError(f"Evaluation manifest contains no outputs: {manifest_path}")

    for item in items:
        available = set(item.outputs)
        if item.dataset_outputs is not None and item.dataset_outputs.gt is not None:
            available.add("gt")
        missing = [key for key in required if key not in available]
        if missing:
            raise ValueError(f"Evaluation item {item.name!r} is missing required outputs {missing}.")
        counts = {item.outputs[key].sample_count() for key in required if key in item.outputs}
        counts.add(item.sample_count)
        if len(counts) != 1:
            raise ValueError(f"Evaluation item {item.name!r} has inconsistent sample counts: {sorted(counts)}.")
    return items


def prepare_comparison_outputs(
    evaluations: Sequence[RunEvaluation],
    *,
    required: Sequence[str] = ("gt", "pred"),
    align: bool = True,
    gt_reference: str | RunEvaluation = "per_run",
) -> ComparisonOutputs:
    """Load aligned outputs with explicit GT semantics.

    ``gt_reference='per_run'`` preserves the historical behavior. ``'dataset'``
    loads one canonical registered dataset GT for every run, while passing one of
    the supplied ``RunEvaluation`` objects reuses that evaluation's GT.
    """
    evaluations = tuple(evaluations)
    if not evaluations:
        raise ValueError("No evaluations supplied for comparison.")
    if isinstance(gt_reference, str) and gt_reference not in {"per_run", "dataset"}:
        raise ValueError("gt_reference must be 'per_run', 'dataset', or one of the supplied evaluations.")
    if isinstance(gt_reference, RunEvaluation) and gt_reference.run_dir not in {e.run_dir for e in evaluations}:
        raise ValueError("A RunEvaluation used as gt_reference must be included in the comparison.")

    by_run = {
        evaluation.run_dir: load_eval_output_items(evaluation, required=required)
        for evaluation in evaluations
    }
    dataset_reference = _shared_dataset_reference(evaluations, by_run) if gt_reference == "dataset" else None

    if not align:
        if len(evaluations) > 1:
            warnings.warn(
                "Evaluation alignment is disabled. Runs may contain different source items "
                "or source indices, so metric/FRC differences can partly reflect sample "
                "composition rather than model performance.",
                UnalignedComparisonWarning,
                stacklevel=2,
            )
        selected = {
            run_dir: tuple(item.select(range(item.sample_count)) for item in items)
            for run_dir, items in by_run.items()
        }
        return ComparisonOutputs(
            evaluations,
            selected,
            aligned=False,
            gt_reference=gt_reference,
            dataset_reference=dataset_reference,
        )

    if len(evaluations) == 1:
        run_dir = evaluations[0].run_dir
        selected = tuple(item.select(range(item.sample_count)) for item in by_run[run_dir])
        return ComparisonOutputs(
            evaluations,
            {run_dir: selected},
            aligned=True,
            gt_reference=gt_reference,
            dataset_reference=dataset_reference,
        )

    return ComparisonOutputs(
        evaluations,
        _align_items(evaluations, by_run),
        aligned=True,
        gt_reference=gt_reference,
        dataset_reference=dataset_reference,
    )


def _align_items(
    evaluations: tuple[RunEvaluation, ...],
    by_run: Mapping[Path, tuple[ManifestEvalItem, ...]],
) -> dict[Path, tuple[SelectedEvalItem, ...]]:
    maps = {}
    for evaluation in evaluations:
        mapping = {item.identity: item for item in by_run[evaluation.run_dir]}
        if len(mapping) != len(by_run[evaluation.run_dir]):
            raise ValueError(f"Duplicate input/GT provenance in {evaluation.folder}; alignment is ambiguous.")
        maps[evaluation.run_dir] = mapping

    common = set.intersection(*(set(mapping) for mapping in maps.values()))
    if not common:
        raise ValueError("Selected evaluations have no common input/GT items to align.")

    first_run = evaluations[0].run_dir
    ordered = [item.identity for item in by_run[first_run] if item.identity in common]
    selected = {evaluation.run_dir: [] for evaluation in evaluations}

    for identity in ordered:
        items = {evaluation.run_dir: maps[evaluation.run_dir][identity] for evaluation in evaluations}
        axes = {item.source_axis for item in items.values()}
        if len(axes) != 1:
            raise ValueError(f"Common item {identity!r} has inconsistent source axes: {axes}.")

        axis = next(iter(axes))
        if axis is None:
            if {item.sample_count for item in items.values()} != {1}:
                raise _alignment_error(identity, "multiple samples have no source-axis provenance")
            for evaluation in evaluations:
                selected[evaluation.run_dir].append(items[evaluation.run_dir].select((0,)))
            continue

        if any(item.source_indices == _UNKNOWN for item in items.values()):
            raise _alignment_error(identity, f"{axis} source indices are unknown")

        indices_by_run = {}
        for evaluation in evaluations:
            indices = items[evaluation.run_dir].source_indices
            if not isinstance(indices, tuple):
                raise _alignment_error(identity, f"{axis} source indices are unavailable")
            if len(set(indices)) != len(indices):
                raise _alignment_error(identity, f"{axis} source indices contain duplicates")
            indices_by_run[evaluation.run_dir] = indices

        common_indices = set.intersection(*(set(indices) for indices in indices_by_run.values()))
        if not common_indices:
            continue
        ordered_indices = tuple(index for index in indices_by_run[first_run] if index in common_indices)

        for evaluation in evaluations:
            positions = {index: pos for pos, index in enumerate(indices_by_run[evaluation.run_dir])}
            selected[evaluation.run_dir].append(
                items[evaluation.run_dir].select(tuple(positions[index] for index in ordered_indices))
            )

    if not any(selected.values()):
        raise ValueError("Selected evaluations have no common source samples after alignment.")
    return {run_dir: tuple(items) for run_dir, items in selected.items()}


def _alignment_error(identity, reason: str) -> ValueError:
    return ValueError(
        f"Cannot exactly align item {identity!r}: {reason}. "
        "Pass align=False to proceed explicitly without source-sample alignment."
    )


def _manifest_item(
    folder: Path,
    record: Mapping[str, object],
    *,
    dataset_outputs: DatasetOutputsSpec | None = None,
) -> ManifestEvalItem:
    name = str(record.get("name", ""))
    input_id = record.get("input_id")
    outputs_raw = record.get("outputs")
    if not name or not isinstance(input_id, str) or not isinstance(outputs_raw, Mapping):
        raise ValueError(f"Invalid outputs manifest item in {folder}: {record!r}")

    axis_raw = record.get("source_axis")
    source_axis = None if axis_raw is None else str(axis_raw)
    indices_raw = record.get("source_indices")
    if isinstance(indices_raw, list):
        source_indices: tuple[int, ...] | str | None = tuple(int(i) for i in indices_raw)
    elif indices_raw in (None, _UNKNOWN):
        source_indices = indices_raw
    else:
        raise ValueError(f"Invalid source_indices for evaluation item {name!r}: {indices_raw!r}")
    if (source_axis is None) != (source_indices is None):
        raise ValueError(f"Evaluation item {name!r} must define source_axis and source_indices together.")

    outputs = {}
    for key, value in outputs_raw.items():
        if not isinstance(value, Mapping) or not isinstance(value.get("files"), list):
            raise ValueError(f"Invalid output {key!r} for evaluation item {name!r}.")
        sample_axis = value.get("sample_axis")
        if sample_axis is not None and not isinstance(sample_axis, int):
            raise ValueError(f"Invalid sample_axis for evaluation item {name!r}, output {key!r}.")
        outputs[str(key)] = OutputSpec(
            tuple(folder / str(file) for file in value["files"]),
            sample_axis,
        )

    gt_raw = record.get("gt_id")
    gt_id = None if gt_raw is None else str(gt_raw)
    return ManifestEvalItem(name, input_id, gt_id, source_axis, source_indices, outputs, dataset_outputs)


def _load_dataset_outputs_spec(evaluation: RunEvaluation) -> DatasetOutputsSpec | None:
    metadata_path = evaluation.folder / "evaluation.yaml"
    if not metadata_path.is_file():
        return None

    metadata = load_yaml(metadata_path)
    if not isinstance(metadata, Mapping):
        return None
    dataset = metadata.get("dataset")
    if not isinstance(dataset, Mapping) or dataset.get("usage") not in {"evaluation", "training"}:
        return None

    dataset_name = dataset.get("name")
    usage = str(dataset["usage"])
    data_type = dataset.get("data_type")
    data_dir = dataset.get("data_dir")
    input_path = dataset.get("input")
    gt_path = dataset.get("eval_gt")
    if dataset_name is None or input_path is None:
        return None
    recorded_data_dir = None if data_dir is None else Path(data_dir)

    dataset_info = load_dataset_info(Paths(settings).dataset_registry_path(), str(dataset_name))
    if not isinstance(dataset_info, Mapping):
        return None

    data_type = _resolve_registry_data_type(
        dataset_info,
        data_type,
        data_dir=data_dir,
        input_path=input_path,
        gt_path=gt_path,
    )
    if data_type is None:
        return None

    resolved_data_dir = _canonical_dataset_data_dir(
        dataset_name=str(dataset_name),
        data_type=data_type,
        usage=usage,
        recorded_data_dir=recorded_data_dir,
    )

    input_output = _registered_output_spec(dataset_info, data_type, input_path)
    resolved_input_path = str(input_path) if input_output is None else input_output.path

    registered_outputs = registry_value_for_data_type(dataset_info, "outputs", data_type)
    auxiliary_outputs = {}
    if isinstance(registered_outputs, list):
        for output in registered_outputs:
            if not isinstance(output, Mapping) or output.get("role") != "aux" or output.get("key") is None:
                continue
            key = str(output["key"])
            path = output.get("path")
            value = key if path is None else str(path)
            auxiliary_outputs[key] = DatasetOutputSpec(
                path=value,
                axes=None if output.get("axes") is None else str(output.get("axes")),
            )

    return DatasetOutputsSpec(
        dataset_name=str(dataset_name),
        usage=usage,
        data_type=data_type,
        data_dir=resolved_data_dir,
        recorded_data_dir=recorded_data_dir,
        input_path=resolved_input_path,
        outputs=auxiliary_outputs,
        gt=None if gt_path is None else _registered_output_spec(dataset_info, data_type, gt_path),
    )


def _canonical_dataset_data_dir(
    *,
    dataset_name: str,
    data_type: str,
    usage: str,
    recorded_data_dir: Path | None = None,
) -> Path:
    data_dir = Paths(settings).dataset_preprocess_dir(
        dataset_name=dataset_name,
        data_type=data_type,
        usage=usage,
    )
    if data_dir.exists() or recorded_data_dir is None:
        return data_dir
    return recorded_data_dir


def _resolve_registry_data_type(
    dataset_info: Mapping[str, object],
    recorded_data_type: object,
    *,
    data_dir: object,
    input_path: object,
    gt_path: object,
) -> str | None:
    if recorded_data_type is not None:
        return str(recorded_data_type)

    known = sorted(registry_data_types(dataset_info))
    if len(known) == 1:
        return known[0]

    data_dir_name = PurePosixPath(str(data_dir).replace("\\", "/")).name
    if data_dir_name in known:
        return data_dir_name

    candidates = []
    for data_type in known:
        if _registered_output_spec(dataset_info, data_type, input_path) is None:
            continue
        if gt_path is not None and _registered_output_spec(dataset_info, data_type, gt_path) is None:
            continue
        candidates.append(data_type)
    return candidates[0] if len(candidates) == 1 else None


def _registered_output_spec(
    dataset_info: Mapping[str, object],
    data_type: str,
    value: object,
) -> DatasetOutputSpec | None:
    if value is None:
        return None
    text = str(value)
    registered_outputs = registry_value_for_data_type(dataset_info, "outputs", data_type)
    if isinstance(registered_outputs, list):
        for output in registered_outputs:
            if not isinstance(output, Mapping):
                continue
            key = output.get("key")
            path = output.get("path")
            if text not in {str(key), str(path)}:
                continue
            resolved_path = key if path is None else path
            if resolved_path is None:
                continue
            return DatasetOutputSpec(
                path=str(resolved_path),
                axes=registry_output_axes(dataset_info, data_type, value),
            )
    return None


def _shared_dataset_reference(
    evaluations: tuple[RunEvaluation, ...],
    by_run: Mapping[Path, tuple[ManifestEvalItem, ...]],
) -> DatasetOutputsSpec:
    specs = []
    for evaluation in evaluations:
        items = by_run[evaluation.run_dir]
        spec = items[0].dataset_outputs if items else None
        if spec is None or spec.gt is None:
            raise ValueError(
                f"Cannot use gt_reference='dataset': {evaluation.folder} does not resolve a registered dataset GT."
            )
        specs.append(spec)

    first = specs[0]
    first_identity = _dataset_reference_identity(first)
    conflicts = [spec for spec in specs[1:] if _dataset_reference_identity(spec) != first_identity]
    if conflicts:
        descriptions = sorted({_dataset_reference_description(spec) for spec in specs})
        raise ValueError(
            "Cannot use one shared dataset GT because selected evaluations resolve different dataset references: "
            + "; ".join(descriptions)
        )
    return first


def _dataset_reference_identity(spec: DatasetOutputsSpec) -> tuple[str, str, str, str, str]:
    assert spec.gt is not None
    return (
        spec.dataset_name,
        spec.usage,
        spec.data_type,
        spec.input_path,
        spec.gt.path,
    )


def _dataset_reference_description(spec: DatasetOutputsSpec) -> str:
    assert spec.gt is not None
    root = str(spec.data_dir)
    recorded = (
        ""
        if spec.recorded_data_dir in (None, spec.data_dir)
        else f", recorded_root={str(spec.recorded_data_dir)!r}"
    )
    return (
        f"{spec.dataset_name}/{spec.usage}/{spec.data_type}: "
        f"input={spec.input_path!r}, gt={spec.gt.path!r}, root={root!r}{recorded}"
    )


def _output_has_source_axis(axes: str | None) -> bool:
    if axes is None:
        # Preserve legacy behavior when axes metadata is unavailable.
        return True
    axes = axes.upper()
    return len(axes) > 2 and axes[0] in {"T", "S"}


def _relative_input_item_path(input_id: str, input_path: str) -> Path:
    source = PurePosixPath(str(input_id).replace("\\", "/"))
    input_text = str(input_path).replace("\\", "/").strip("/")
    if not input_text:
        return Path(*source.parts)

    try:
        relative = source.relative_to(PurePosixPath(input_text))
    except ValueError as exc:
        raise ValueError(
            f"Evaluation input id {input_id!r} is not inside the recorded input path {input_path!r}; "
            "cannot resolve matching auxiliary data."
        ) from exc
    return Path(*relative.parts)


@lru_cache(maxsize=256)
def _tiff_shape(path: Path) -> tuple[int, ...]:
    with TiffFile(path) as tif:
        return tuple(int(value) for value in tif.series[0].shape)


@dataclass(frozen=True)
class EvalItemFiles:
    name: str
    folder: Path
    inp: Path | None = None
    gt: Path | None = None
    pred: Path | None = None
    samples: Path | None = None

    def path_for(
        self,
        key: str,
    ) -> Path | None:
        return getattr(self, key)


def discover_eval_items(
    folder: str | Path,
    *,
    required: Sequence[str] = ("gt", "pred"),
) -> list[EvalItemFiles]:
    """
    Discover evaluation outputs by their _<kind>.tif suffix.

    Works with both individual images and the newer item-level
    timelapse stacks.
    """

    folder = Path(folder)

    if not folder.is_dir():
        raise FileNotFoundError(
            f"Evaluation folder does not exist: {folder}"
        )

    by_item: dict[str, dict[str, Path]] = {}

    for path in folder.iterdir():
        if not path.is_file():
            continue

        if path.suffix.lower() not in _TIFF_SUFFIXES:
            continue

        stem = path.stem

        for key in _OUTPUT_KEYS:
            marker = f"_{key}"

            if stem.endswith(marker):
                item_name = stem[:-len(marker)]

                by_item.setdefault(
                    item_name,
                    {},
                )[key] = path

                break

    required = tuple(required)

    missing_by_item = {
        name: [
            key
            for key in required
            if key not in outputs
        ]
        for name, outputs in by_item.items()
        if any(
            key not in outputs
            for key in required
        )
    }

    if missing_by_item:
        details = "; ".join(
            f"{name}: missing {missing}"
            for name, missing
            in sorted(missing_by_item.items())
        )

        raise ValueError(
            f"Incomplete evaluation outputs in {folder}: "
            f"{details}"
        )

    items = [
        EvalItemFiles(
            name=name,
            folder=folder,
            inp=outputs.get("inp"),
            gt=outputs.get("gt"),
            pred=outputs.get("pred"),
            samples=outputs.get("samples"),
        )
        for name, outputs
        in sorted(by_item.items())
        if all(
            key in outputs
            for key in required
        )
    ]

    if not items:
        raise FileNotFoundError(
            f"No evaluation items with required outputs "
            f"{required} found in {folder}"
        )

    return items


def eval_items_by_name(
    folder: str | Path,
    *,
    required: Sequence[str] = ("gt", "pred"),
) -> dict[str, EvalItemFiles]:

    return {
        item.name: item
        for item in discover_eval_items(
            folder,
            required=required,
        )
    }


def common_eval_item_names(
    folders: Iterable[str | Path],
    *,
    required: Sequence[str] = ("gt", "pred"),
) -> list[str]:
    """Return item names present in every evaluation folder."""

    name_sets = [
        set(
            eval_items_by_name(
                folder,
                required=required,
            )
        )
        for folder in folders
    ]

    if not name_sets:
        return []

    return sorted(
        set.intersection(*name_sets)
    )


def as_image_frames(
    array: np.ndarray,
) -> np.ndarray:
    """
    Normalize an image/image stack to (N, Y, X).

    Leading singleton channel/batch axes from saved evaluation
    outputs are therefore harmless.
    """

    array = np.asarray(array)

    if array.ndim < 2:
        raise ValueError(
            f"Expected at least a 2D image, "
            f"got shape {array.shape}"
        )

    if array.ndim == 2:
        return array[None, ...]

    return array.reshape(
        -1,
        array.shape[-2],
        array.shape[-1],
    )


def load_item_frames(
    item: EvalItemFiles,
    key: str,
) -> np.ndarray:

    path = item.path_for(key)

    if path is None:
        raise ValueError(
            f"Evaluation item {item.name!r} "
            f"has no {key!r} output"
        )

    return as_image_frames(
        imread(path)
    )


def load_gt_pred_frames(
    item: EvalItemFiles,
) -> tuple[np.ndarray, np.ndarray]:

    gt = load_item_frames(
        item,
        "gt",
    )

    pred = load_item_frames(
        item,
        "pred",
    )

    if gt.shape != pred.shape:
        raise ValueError(
            f"GT/pred shape mismatch for "
            f"{item.name!r}: "
            f"{gt.shape} vs {pred.shape}"
        )

    return gt, pred


def center_trim_frames(
    frames: np.ndarray,
    n_frames: int,
) -> np.ndarray:
    """Keep the central n_frames of an image stack."""

    frames = as_image_frames(frames)

    if (
        n_frames < 0
        or n_frames > len(frames)
    ):
        raise ValueError(
            f"Cannot keep {n_frames} frames "
            f"from stack with {len(frames)} frames"
        )

    start = (
        len(frames) - n_frames
    ) // 2

    return frames[
        start:start + n_frames
    ]

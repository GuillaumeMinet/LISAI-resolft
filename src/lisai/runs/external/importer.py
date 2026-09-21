from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from tifffile import imread, imwrite

from lisai.config import settings
from lisai.config.io.yaml import save_yaml
from lisai.config.models.training.data import DataSection
from lisai.data.dataset_registry import (
    load_dataset_info,
    registry_data_format_for_output,
    registry_data_types,
    registry_mapping_for_data_type,
    registry_value_for_data_type,
)
from lisai.evaluation import EvalSource
from lisai.evaluation.data import EvalItem, EvalSampleSource
from lisai.evaluation.io import save_outputs_manifest
from lisai.infra.paths import Paths

from .prediction_formats import convert_prediction
from .schema import (
    EXTERNAL_RUN_METADATA_FILENAME,
    ExternalDatasetSelection,
    ExternalRunMetadata,
)


@dataclass(frozen=True)
class ExternalImportResult:
    run_dir: Path
    evaluation_dir: Path
    prediction_count: int


def _natural_key(path: Path):
    return tuple(int(part) if part.isdigit() else part.casefold() for part in re.split(r"(\d+)", path.name))


def _resolve_data_type(dataset_name: str, info: Mapping[str, Any], options: Mapping[str, Any]) -> str:
    explicit = options.get("data_type")
    known = registry_data_types(info)
    if explicit is not None:
        value = str(explicit)
        if known and value not in known:
            raise ValueError(
                f"Dataset {dataset_name!r} has no data type {value!r}. Available: {', '.join(sorted(known))}."
            )
        return value
    if len(known) == 1:
        return next(iter(known))
    if not known:
        raise ValueError(f"Dataset {dataset_name!r} does not define any data types in the registry.")
    raise ValueError(
        f"Dataset {dataset_name!r} has multiple data types ({', '.join(sorted(known))}). "
        "Specify data_type in the import data options."
    )


def _role_paths(info: Mapping[str, Any], data_type: str, role: str) -> list[str]:
    outputs = registry_value_for_data_type(info, "outputs", data_type)
    result: list[str] = []
    if not isinstance(outputs, list):
        return result
    for output in outputs:
        if not isinstance(output, Mapping) or output.get("role") != role:
            continue
        value = output.get("path", output.get("key"))
        if value is not None:
            result.append(str(value))
    return result


def _resolve_path_choice(
    *,
    dataset_name: str,
    info: Mapping[str, Any],
    data_type: str,
    options: Mapping[str, Any],
    option_name: str,
    default_name: str,
    role: str,
    allow_none: bool,
) -> str | None:
    if option_name in options and options[option_name] is not None:
        return str(options[option_name])

    defaults = registry_mapping_for_data_type(info, "defaults", data_type) or {}
    default = defaults.get(default_name)
    if default is not None:
        return str(default)

    candidates = _role_paths(info, data_type, role)
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        raise ValueError(
            f"Dataset {dataset_name!r}/{data_type} has multiple {role!r} outputs: {candidates}. "
            f"Specify {option_name!r} in the import data options."
        )
    if allow_none:
        return None
    raise ValueError(
        f"Dataset {dataset_name!r}/{data_type} has no unambiguous {role!r} output. "
        f"Specify {option_name!r} in the import data options."
    )


def _resolve_dataset_selection(
    dataset_name: str,
    *,
    usage: str,
    options: Mapping[str, Any] | None,
) -> tuple[ExternalDatasetSelection, DataSection]:
    options = dict(options or {})
    paths = Paths(settings)
    info = load_dataset_info(paths.dataset_registry_path(), dataset_name)
    if not isinstance(info, Mapping):
        raise ValueError(f"Dataset {dataset_name!r} is not registered.")

    data_type = _resolve_data_type(dataset_name, info, options)
    input_path = _resolve_path_choice(
        dataset_name=dataset_name,
        info=info,
        data_type=data_type,
        options=options,
        option_name="input",
        default_name="input",
        role="inp",
        allow_none=False,
    )
    default_gt_name = "eval_gt" if usage == "evaluation" else "target"
    gt_path = _resolve_path_choice(
        dataset_name=dataset_name,
        info=info,
        data_type=data_type,
        options=options,
        option_name="gt",
        default_name=default_gt_name,
        role="gt",
        allow_none=True,
    )

    data_format = registry_data_format_for_output(info, data_type, input_path) or str(info.get("data_format") or "single")
    snr_idx = options.get("snr_idx")
    if data_format == "mltpl_snr" and snr_idx is None:
        raise ValueError(
            f"Dataset {dataset_name!r}/{data_type} uses multiple SNR levels. "
            "Specify snr_idx in the import data options."
        )

    selection = ExternalDatasetSelection(
        dataset=dataset_name,
        data_type=data_type,
        input=input_path,
        gt=gt_path,
        snr_idx=snr_idx,
    )
    data_dir = paths.dataset_preprocess_dir(dataset_name=dataset_name, data_type=data_type, usage=usage)
    payload: dict[str, Any] = {
        "dataset_name": dataset_name,
        "prep_before": True,
        "already_split": True,
        "paired": gt_path is not None,
        "input": input_path,
        "target": gt_path,
        "data_format": data_format,
        "data_dir": data_dir,
        "dataset_info": dict(info),
        "registry_data_type": data_type,
        "registry_checked": True,
    }
    if snr_idx is not None:
        payload["mltpl_snr_prm"] = {"snr_idx": snr_idx}
    return selection, DataSection.model_validate(payload)


def _expected_items(
    *,
    trained_on: str,
    evaluated_on: EvalSource,
    training_data: Mapping[str, Any] | None,
    evaluation_data: Mapping[str, Any] | None,
) -> tuple[ExternalDatasetSelection, ExternalDatasetSelection, tuple[EvalItem, ...], DataSection]:
    training_selection, _ = _resolve_dataset_selection(
        trained_on,
        usage="training",
        options=training_data,
    )

    if evaluated_on.usage == "evaluation":
        evaluation_selection, config = _resolve_dataset_selection(
            evaluated_on.name,
            usage="evaluation",
            options=evaluation_data,
        )
        source = EvalSampleSource.from_config(config, use_split=False)
    else:
        evaluation_selection, config = _resolve_dataset_selection(
            trained_on,
            usage="training",
            options=evaluation_data or training_data,
        )
        config = DataSection.model_validate(config.model_dump() | {"split": evaluated_on.name, "data_dir": config.data_dir, "dataset_info": config.dataset_info})
        source = EvalSampleSource.from_config(config, use_split=True)

    if not source.items:
        raise ValueError(f"Resolved evaluation source {evaluated_on} contains no items.")
    return training_selection, evaluation_selection, source.items, config


def _collect_prediction_files(path: Path, patterns: Sequence[str]) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"Evaluation prediction path does not exist: {path}")
    files: set[Path] = set()
    for pattern in patterns:
        files.update(p for p in path.glob(pattern) if p.is_file())
    return sorted(files, key=_natural_key)


def _normalize_prediction_for_item(array: np.ndarray, item: EvalItem, *, source_path: Path) -> tuple[np.ndarray, int | None]:
    array = np.asarray(array)
    if item.sample_count == 1:
        return array, None
    if array.ndim == 0 or array.shape[0] != item.sample_count:
        raise ValueError(
            f"Prediction {source_path.name!r} maps to evaluation item {item.name!r}, which expects "
            f"{item.sample_count} samples, but converted prediction shape is {array.shape}."
        )
    return array, 0


def _evaluation_metadata(
    *,
    source: EvalSource,
    selection: ExternalDatasetSelection,
    config: DataSection,
    run_name: str,
    trained_on: str,
) -> dict[str, Any]:
    dataset: dict[str, Any] = {
        "name": selection.dataset,
        "usage": source.usage,
        "data_type": selection.data_type,
        "data_dir": str(config.data_dir),
        "input": selection.input,
        "eval_gt": selection.gt,
    }
    if source.usage == "training":
        dataset["split"] = source.name
    return {
        "version": 1,
        "model": {
            "source": "external_run",
            "experiment_name": run_name,
            "training_dataset": trained_on,
        },
        "dataset": dataset,
        "evaluation": {"imported": True},
    }


def import_external_run(
    *,
    run_name: str,
    trained_on: str,
    checkpoint_path: str | Path | None,
    evaluation_data_path: str | Path,
    evaluated_on: EvalSource,
    prediction_format: str = "image",
    prediction_glob: str | Sequence[str] = ("*.tif", "*.tiff"),
    config_path: str | Path | None = None,
    training_data: Mapping[str, Any] | None = None,
    evaluation_data: Mapping[str, Any] | None = None,
    uses_current_split: bool = True,
    overwrite: bool = False,
    notes: str | None = None,
) -> ExternalImportResult:
    """Import an externally trained model and one already-computed evaluation into LISAI."""
    if not uses_current_split:
        raise NotImplementedError(
            "External runs using a different train/val/test split are not supported yet. "
            "Import with uses_current_split=True or add explicit split mapping support first."
        )

    patterns = (prediction_glob,) if isinstance(prediction_glob, str) else tuple(prediction_glob)
    training_selection, evaluation_selection, items, eval_config = _expected_items(
        trained_on=trained_on,
        evaluated_on=evaluated_on,
        training_data=training_data,
        evaluation_data=evaluation_data,
    )
    prediction_files = _collect_prediction_files(Path(evaluation_data_path), patterns)
    if len(prediction_files) != len(items):
        raise ValueError(
            f"Found {len(prediction_files)} prediction file(s), but {len(items)} evaluation item(s) are expected "
            f"for {evaluated_on.folder_name!r}. Predictions are paired to LISAI items by deterministic file order, "
            "so the counts must match exactly."
        )

    paths = Paths(settings)
    run_dir = paths.external_run_dir(dataset_name=trained_on, run_name=run_name)
    if run_dir.exists():
        if not overwrite:
            raise FileExistsError(f"External run already exists: {run_dir}")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_rel = None
    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint file does not exist: {checkpoint_path}")
        checkpoint_dir = run_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        copied = checkpoint_dir / checkpoint_path.name
        shutil.copy2(checkpoint_path, copied)
        checkpoint_rel = copied.relative_to(run_dir).as_posix()

    config_rel = None
    if config_path is not None:
        config_path = Path(config_path)
        if not config_path.is_file():
            raise FileNotFoundError(f"External config file does not exist: {config_path}")
        config_dir = run_dir / "config"
        config_dir.mkdir(parents=True, exist_ok=True)
        copied = config_dir / config_path.name
        shutil.copy2(config_path, copied)
        config_rel = copied.relative_to(run_dir).as_posix()

    metadata = ExternalRunMetadata(
        run_name=run_name,
        trained_on=training_selection,
        checkpoint=checkpoint_rel,
        config=config_rel,
        uses_current_split=uses_current_split,
        notes=notes,
    )
    save_yaml(metadata.model_dump(mode="json", exclude_none=True), run_dir / EXTERNAL_RUN_METADATA_FILENAME)

    evaluation_dir = run_dir / "evaluations" / evaluated_on.folder_name / "imported"
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    save_yaml(
        _evaluation_metadata(
            source=evaluated_on,
            selection=evaluation_selection,
            config=eval_config,
            run_name=run_name,
            trained_on=trained_on,
        ),
        evaluation_dir / "evaluation.yaml",
    )

    manifest_items: list[dict[str, Any]] = []
    for item, prediction_file in zip(items, prediction_files):
        converted = convert_prediction(np.asarray(imread(prediction_file)), prediction_format)
        converted, sample_axis = _normalize_prediction_for_item(converted, item, source_path=prediction_file)
        output_name = f"{item.name}_pred.tif"
        output_path = evaluation_dir / output_name
        imwrite(output_path, converted, photometric="minisblack")
        record: dict[str, Any] = {
            "name": item.name,
            "input_id": item.input_id or item.inp_path.as_posix(),
            "gt_id": item.gt_id,
            "outputs": {"pred": {"files": [output_name], "sample_axis": sample_axis}},
        }
        if item.source_axis is not None:
            record["source_axis"] = item.source_axis
            record["source_indices"] = item.source_indices
        manifest_items.append(record)

    save_outputs_manifest(evaluation_dir, manifest_items)
    return ExternalImportResult(run_dir=run_dir, evaluation_dir=evaluation_dir, prediction_count=len(prediction_files))


__all__ = ["ExternalImportResult", "import_external_run"]

"""Prepare a runnable *fresh* unsupervised HDN training config from raw recordings.

Training itself is launched separately, e.g. with lisai_runner.training_entrypoint.
No second training engine, preprocessor, or split-manifest implementation lives here.
"""
from __future__ import annotations

import json
import math
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Mapping

import numpy as np
import tifffile

from lisai.config.catalog import resolve_training_preset_path, save_training_config_yaml
from lisai.config.io.metadata import strip_config_metadata
from lisai.config.io.resolver import resolve_config_dict
from lisai.config.io.yaml import load_yaml
from lisai.config.models.training import DataSection
from lisai.config.models.training.data import SplitRatios
from lisai.config.models.training.normalization import DataNormalizationParams
from lisai.data.data_loaders.split_manifest import collect_input_files, make_unprepared_split_manifest
from lisai.data.dataset_registry import load_dataset_registry
from lisai.infra.paths import Paths

if TYPE_CHECKING:
    from lisai.config.settings import Settings


_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


@dataclass(frozen=True)
class QuickHDNTrainingPlan:
    config_path: Path
    dataset_name: str
    experiment_name: str
    noise_model_name: str
    recording_count: int
    split_counts: Mapping[str, int]
    split_filenames: Mapping[str, tuple[str, ...]]
    config: dict


def _require_safe_name(value: str, label: str) -> str:
    if not isinstance(value, str) or not _SAFE_NAME.fullmatch(value) or value in {".", ".."}:
        raise ValueError(f"Invalid {label} {value!r}; use letters, digits, '.', '-' or '_'.")
    return value


def _check_noise_model(paths: Paths, name: str) -> None:
    model = paths.noise_model_path(noiseModel_name=name)
    norm = paths.noise_model_norm_prm_path(noiseModel_name=name)
    if not model.is_file():
        raise FileNotFoundError(f"HDN noise model {name!r} is missing: {model}")
    if not norm.is_file():
        raise FileNotFoundError(f"HDN noise-model normalization is missing: {norm}")
    try:
        with np.load(model, allow_pickle=False) as archive:
            required = {"min_signal", "max_signal", "trained_weight", "min_sigma"}
            if not required.issubset(archive.files):
                raise ValueError(f"Noise-model archive missing fields: {sorted(required - set(archive.files))}")
            weights = archive["trained_weight"]
            if weights.ndim != 2 or weights.shape[0] < 3 or weights.shape[0] % 3 or weights.shape[1] < 1:
                raise ValueError("Noise-model trained_weight must have shape (3*n_gaussians, n_coefficients).")
            for key in required:
                if not np.isfinite(archive[key]).all():
                    raise ValueError(f"Noise-model field {key!r} is not finite.")
            if (archive["min_sigma"].size != 1 or float(archive["min_sigma"].item()) <= 0
                    or archive["min_signal"].size != 1 or archive["max_signal"].size != 1
                    or float(archive["max_signal"].item()) <= float(archive["min_signal"].item())):
                raise ValueError("Noise model requires finite scalar min/max signal and positive min_sigma.")
    except (OSError, KeyError) as exc:
        raise ValueError(f"Could not read HDN noise-model archive {model}: {exc}") from exc
    try:
        with norm.open(encoding="utf-8") as handle:
            metadata = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid noise-model normalization JSON {norm}: {exc}") from exc
    if not isinstance(metadata, dict):
        raise ValueError(f"Noise-model normalization must be a JSON object: {norm}")
    DataNormalizationParams.model_validate(metadata)
    # Do not invent normalization defaults; the existing training loader owns the
    # interpretation of these saved scientific calibration parameters.


def prepare_quick_hdn_training(
    *,
    settings: Settings,
    dataset_name: str,
    noise_model_name: str,
    experiment_name: str,
    epochs: int,
    patch_size: int,
    split_ratios: SplitRatios | Mapping[str, float] | None = None,
    seed: int = 0,
    learning_rate: float | None = None,
    beta_kl: float | None = None,
    write: bool = False,
) -> QuickHDNTrainingPlan:
    """Validate a new HDN experiment; optionally persist its YAML.

    ``write=False`` is a read-only preview. The manifest is *not* saved here;
    the standard training-run setup writes its unchanged v1 manifest.
    """
    dataset_name = _require_safe_name(dataset_name, "dataset name")
    experiment_name = _require_safe_name(experiment_name, "experiment name")
    noise_model_name = _require_safe_name(noise_model_name, "noise model name")
    if type(epochs) is not int or epochs <= 0:
        raise ValueError("epochs must be a positive integer.")
    if type(patch_size) is not int or patch_size <= 0:
        raise ValueError("patch_size must be a positive integer.")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer.")
    if learning_rate is not None and (not math.isfinite(learning_rate) or learning_rate <= 0):
        raise ValueError("learning_rate must be finite and positive.")
    if beta_kl is not None and (not math.isfinite(beta_kl) or beta_kl < 0):
        raise ValueError("beta_kl must be finite and nonnegative.")
    ratios = (SplitRatios(train=0.8, val=0.2, test=0.0)
              if split_ratios is None else SplitRatios.model_validate(split_ratios))
    paths = Paths(settings)
    registry = load_dataset_registry(paths.dataset_registry_path())
    info = registry.get(dataset_name)
    if not isinstance(info, dict):
        raise ValueError(f"Dataset {dataset_name!r} is not registered in workspace {paths.workspace_root()}.")
    if info.get("usage") != "training" or info.get("for_training") is False:
        raise ValueError(f"Dataset {dataset_name!r} is not a training dataset.")
    data_format = info.get("data_format")
    if data_format not in {"single", "timelapse"}:
        raise ValueError(f"Quick HDN supports registered 'single' or 'timelapse' data, got {data_format!r}.")
    dataset_dir = paths.dataset_dir(dataset_name=dataset_name)
    if not (dataset_dir / "raw").is_dir():
        raise FileNotFoundError(f"Expected raw TIFF recordings in {dataset_dir / 'raw'}.")
    _check_noise_model(paths, noise_model_name)

    preset = strip_config_metadata(load_yaml(resolve_training_preset_path("denoising_hdn_unsup", stg=settings)))
    preset["experiment"].update({"mode": "train", "exp_name": experiment_name,
                                 "overwrite": False, "post_training_inference": False})
    preset["routing"].update({"data_subfolder": "", "models_subfolder": "HDN"})
    data = preset["data"]
    data.update({
        "dataset_name": dataset_name, "data_format": data_format, "canonical_load": True,
        "prep_before": False, "already_split": False, "paired": False,
        "input": "raw", "target": None, "patch_size": patch_size,
        "val_patch_size": patch_size, "split_manifest": {
            "seed": seed, "ratios": ratios.model_dump(), "retrain_policy": "reuse"
        },
    })
    if data_format == "timelapse":
        data["timelapse_prm"] = {"context_length": None, "timelapse_max_frames": None,
                                 "shuffle": False}
    else:
        data.pop("timelapse_prm", None)
    preset["noise_model"] = {"name": noise_model_name}
    preset["normalization"]["load_from_noise_model"] = True
    preset["training"]["n_epochs"] = epochs
    if beta_kl is not None:
        preset["experiment"]["task"]["betaKL"] = beta_kl
    if learning_rate is not None:
        preset["training"]["learning_rate"] = learning_rate

    # Check the exact files through the original split code (and seed), not a
    # GUI-side approximation. TIFF shape checks are fast metadata operations.
    file_cfg = DataSection.model_validate({**data, "data_dir": dataset_dir})
    found_files = collect_input_files(file_cfg)
    if len(found_files) == 1:
        raise ValueError(
            "Quick HDN needs at least two separate recordings for train/validation; "
            "a single timelapse is not split into frames."
        )
    manifest = make_unprepared_split_manifest(file_cfg, data_format=data_format)
    input_files = [entry["input"] for entries in manifest["splits"].values() for entry in entries]
    for filename in input_files:
        source = dataset_dir / "raw" / filename
        with tifffile.TiffFile(source) as tiff:
            shape = tiff.series[0].shape
            expected = 2 if data_format == "single" else 3
            # TIFF metadata can squeeze T=1 even when tifffile.imread keeps TYX.
            if expected == 3 and len(shape) == 2:
                shape = tifffile.imread(source).shape
            if len(shape) != expected or min(shape[-2:]) < patch_size:
                raise ValueError(f"{source.name}: expected a {data_format} image with Y/X >= {patch_size}, got {shape}.")

    # Type/architecture/cross-section rules use the normal experiment resolver.
    resolved = resolve_config_dict(preset, stg=settings)
    if resolved.model is None or resolved.model.architecture != "lvae":
        raise ValueError("Packaged quick HDN preset did not resolve to LVAE.")
    target = settings.TRAINING_CONFIG_DIR / "improcess" / f"{experiment_name}.yml"
    if target.exists():
        raise FileExistsError(f"Training config already exists: {target}")
    run_dir = paths.run_dir(dataset_name=dataset_name, models_subfolder="HDN", exp_name=experiment_name)
    if run_dir.exists():
        raise FileExistsError(f"Training run already exists: {run_dir}")
    if write:
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(prefix=f".{experiment_name}-", suffix=".yml", dir=target.parent)
        os.close(fd)
        temp = Path(temp_name)
        try:
            save_training_config_yaml(preset, temp, schema_root=settings.CONFIGS_ROOT / "schema")
            # Exclusive creation: a second process must not replace a config.
            with target.open("x", encoding="utf-8") as handle:
                try:
                    handle.write(temp.read_text(encoding="utf-8"))
                except BaseException:
                    handle.close()
                    target.unlink(missing_ok=True)
                    raise
        finally:
            temp.unlink(missing_ok=True)

    counts = {split: len(entries) for split, entries in manifest["splits"].items()}
    filenames = {split: tuple(entry["input"] for entry in entries)
                 for split, entries in manifest["splits"].items()}
    return QuickHDNTrainingPlan(
        config_path=target, dataset_name=dataset_name, experiment_name=experiment_name,
        noise_model_name=noise_model_name, recording_count=len(input_files),
        split_counts=counts, split_filenames=filenames, config=preset,
    )


__all__ = ["QuickHDNTrainingPlan", "prepare_quick_hdn_training"]

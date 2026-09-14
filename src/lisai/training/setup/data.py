"""Training data preparation helpers.

This module owns the data-side setup boundary of training. It resolves dataset
location and normalization, builds the training and validation loaders, and
returns the typed artifact that later training steps consume.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

from lisai.data.data_loaders import make_training_loaders
from lisai.data.dataset_registry import load_dataset_info
from lisai.data.data_loaders.split_manifest import (
    read_split_manifest,
    resolve_split_manifest_path,
)

from .noise_model import resolve_noise_model_metadata

if TYPE_CHECKING:
    from lisai.config.models import ResolvedExperiment
    from lisai.training.runtime import TrainingRuntime

logger = logging.getLogger("lisai.prepare_data")


@dataclass
class PreparedTrainingData:
    """Typed output of training data preparation.

    This artifact bundles the effective loaders and the normalization / patch
    metadata discovered during data setup so the rest of the training pipeline
    can consume one explicit object.
    """

    train_loader: Any
    val_loader: Any
    data_norm_prm: dict | None
    model_norm_prm: dict | None
    patch_info: Any | None
    split_manifest: dict | None = None


def _load_required_split_manifest(*, paths, run_dir: str | Path, purpose: str) -> dict:
    manifest_path = resolve_split_manifest_path(paths, run_dir)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing split manifest for {purpose}: {manifest_path}")
    return read_split_manifest(manifest_path)


def _resolve_origin_split_manifest(cfg: ResolvedExperiment, runtime: TrainingRuntime, data_cfg) -> dict | None:
    if bool(getattr(data_cfg, "prep_before", True)):
        return None

    mode = getattr(cfg.experiment, "mode", "train")
    if mode == "train":
        return None

    origin_run_dir = getattr(cfg.experiment, "origin_run_dir", None)
    if not origin_run_dir:
        raise ValueError(f"Mode '{mode}' with `data.prep_before=false` requires `experiment.origin_run_dir`.")

    if mode == "continue_training":
        return _load_required_split_manifest(
            paths=runtime.paths,
            run_dir=origin_run_dir,
            purpose="continue_training",
        )

    if mode == "retrain":
        policy = data_cfg.split_manifest.retrain_policy
        if policy == "new":
            return None
        if policy == "reuse":
            return _load_required_split_manifest(
                paths=runtime.paths,
                run_dir=origin_run_dir,
                purpose="retrain with `data.split_manifest.retrain_policy='reuse'`",
            )
        raise ValueError(f"Unknown retrain split manifest policy: {policy!r}")

    return None


def prepare_data(
    cfg: ResolvedExperiment,
    runtime: TrainingRuntime,
) -> PreparedTrainingData:
    """Resolve training data configuration, normalization, and loaders."""
    is_lvae = cfg.model.architecture == "lvae"
    is_volumetric = cfg.model.architecture == "unet3d"

    data_norm_prm = None
    if is_lvae:
        data_norm_prm = resolve_noise_model_metadata(cfg, runtime.paths)
    if data_norm_prm is None:
        normalization = getattr(cfg, "normalization", None)
        if isinstance(normalization, Mapping):
            data_norm_prm = normalization.get("norm_prm")
        elif normalization is not None:
            data_norm_prm = normalization.norm_prm_dict()

    data_dir = runtime.paths.dataset_dir(
        dataset_name=cfg.data.dataset_name,
        data_subfolder=cfg.routing.data_subfolder,
        usage="training",
    )

    registry_path = Path(runtime.paths.dataset_registry_path())
    if not registry_path.exists():
        logger.warning("Dataset registry not found")
    dataset_info = load_dataset_info(registry_path, cfg.data.dataset_name)

    data_cfg = cfg.data.resolved(
        data_dir=data_dir,
        norm_prm=data_norm_prm,
        dataset_info=dataset_info,
        volumetric=is_volumetric,
    )

    split_manifest = _resolve_origin_split_manifest(cfg, runtime, data_cfg)

    train_loader, val_loader, model_norm_prm, patch_info, split_manifest = make_training_loaders(
        config=data_cfg,
        split_manifest=split_manifest,
        return_split_manifest=True,
    )

    if split_manifest is not None and runtime.run_dir is None:
        logger.warning(
            "`data.prep_before=false` produced an in-memory split manifest, but saving is disabled; "
            "`split_manifest.json` will not be available for evaluation or continue/retrain reuse."
        )

    return PreparedTrainingData(
        train_loader=train_loader,
        val_loader=val_loader,
        data_norm_prm=data_norm_prm,
        model_norm_prm=model_norm_prm,
        patch_info=patch_info,
        split_manifest=split_manifest,
    )

"""Config-side evaluation helpers for loading a saved training run.

This module owns the evaluation boundary that turns a saved `config_train.yaml`
into a small, immutable `SavedTrainingRun` object. It is intentionally separate
from the live inference runtime so evaluation keeps a clean split between
saved configuration and process-time resources.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping

from lisai.config import load_yaml, settings
from lisai.config.models import ResolvedExperiment
from lisai.data.data_loaders.split_manifest import read_split_manifest, resolve_split_manifest_path
from lisai.infra.paths import Paths
from lisai.models.params import AnyModelParams

CheckpointMethod = Literal["state_dict", "full_model"]

_DEFAULT_TILING_SIZE_BY_ARCHITECTURE = {
    "lvae": 300,
    "unet": 1024,
    "unet3d": 512,
    "unetrcan": 512,
}


def resolve_run_dir(*, dataset_name: str, subfolder: str, exp_name: str) -> Path:
    """Resolve the canonical training run directory from experiment routing fields."""
    paths = Paths(settings)
    return paths.run_dir(dataset_name=dataset_name, models_subfolder=subfolder, exp_name=exp_name)



def _noise_model_name(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        name = value.get("name")
        return str(name) if name is not None else None
    name = getattr(value, "name", None)
    return str(name) if name is not None else None



def _checkpoint_methods(cfg: ResolvedExperiment) -> tuple[CheckpointMethod, ...]:
    """Extract the checkpoint formats that the saved run can be loaded from."""
    methods: list[CheckpointMethod] = []
    if bool(cfg.saving.state_dict):
        methods.append("state_dict")
    if bool(cfg.saving.entire_model):
        methods.append("full_model")
    if not methods:
        raise ValueError("Saved training config does not enable any checkpoint format.")
    return tuple(methods)



def _architecture_default_tiling_size(architecture: str) -> int | None:
    """Resolve the default inference tiling size for a model architecture."""
    for key in (architecture, architecture.replace("_", "")):
        if key in _DEFAULT_TILING_SIZE_BY_ARCHITECTURE:
            return int(_DEFAULT_TILING_SIZE_BY_ARCHITECTURE[key])
    return None


def _optional_int_tiling_size(value: Any, *, source: str) -> int | None:
    if isinstance(value, bool):
        raise ValueError(f"{source} must be a positive integer, 'auto', 'off', or null.")
    try:
        resolved = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{source} must be a positive integer, 'auto', 'off', or null.") from None
    if resolved <= 0:
        raise ValueError(f"{source} must be greater than 0.")
    return resolved


def _legacy_compat_default_tiling_size(
    *,
    architecture: str,
    run_dir: Path,
    models_subfolder: str,
) -> int | None:
    """Compatibility default for already-imported legacy runs without metadata."""
    normalized_architecture = architecture.replace("_", "")
    if normalized_architecture != "unetrcan":
        return None

    hints = [str(run_dir), models_subfolder]
    if any("legacy" in hint.lower() for hint in hints):
        return 2000
    return None


def _saved_inference_default_tiling_size(cfg: ResolvedExperiment) -> Any:
    inference = getattr(cfg, "inference", None)
    if inference is None:
        return None
    if isinstance(inference, Mapping):
        return inference.get("default_tiling_size")
    return getattr(inference, "default_tiling_size", None)


def _default_tiling_size(
    architecture: str,
    *,
    cfg: ResolvedExperiment,
    run_dir: Path,
) -> int | None:
    """Resolve the saved-run default tiling size used by inference auto mode."""
    saved_default = _saved_inference_default_tiling_size(cfg)
    if isinstance(saved_default, str):
        normalized = saved_default.strip().lower()
        if normalized in {"", "auto"}:
            saved_default = None
        elif normalized in {"off", "none", "disable", "disabled"}:
            return None
        else:
            return _optional_int_tiling_size(saved_default, source="inference.default_tiling_size")
    elif saved_default is not None:
        return _optional_int_tiling_size(saved_default, source="inference.default_tiling_size")

    compat_default = _legacy_compat_default_tiling_size(
        architecture=architecture,
        run_dir=run_dir,
        models_subfolder=cfg.routing.models_subfolder,
    )
    if compat_default is not None:
        return compat_default

    return _architecture_default_tiling_size(architecture)


@dataclass(frozen=True)
class SavedTrainingRun:
    """Immutable evaluation-facing summary of one saved training run.

    The object keeps only the subset of the saved training configuration that
    evaluation needs: model description, normalization settings, checkpoint
    formats, and data-loading hints.
    """

    run_dir: Path
    experiment_name: str
    dataset_name: str
    data_subfolder: str
    data_cfg: dict[str, Any]
    model_architecture: str
    model_parameters: AnyModelParams
    data_norm_prm: dict[str, Any] | None
    model_norm_prm: dict[str, Any] | None
    noise_model_name: str | None
    checkpoint_methods: tuple[CheckpointMethod, ...]
    patch_size: int | None
    downsamp_factor: int
    upsampling_factor: int
    context_length: int | None
    default_tiling_size: int | None
    split_manifest: dict[str, Any] | None = None

    @property
    def is_lvae(self) -> bool:
        """Whether this saved run uses the LVAE architecture."""
        return self.model_architecture == "lvae"

    @classmethod
    def from_resolved(cls, cfg: ResolvedExperiment, *, run_dir: Path) -> "SavedTrainingRun":
        """Project a validated `ResolvedExperiment` into the evaluation config boundary."""
        if cfg.model is None:
            raise ValueError("Saved training config is missing model information.")
        architecture = cfg.model.architecture
        if not architecture:
            raise ValueError("Saved training config is missing model.architecture.")

        model_parameters = cfg.model.parameters
        normalization = getattr(cfg, "normalization", None)
        if isinstance(normalization, Mapping):
            norm_prm = normalization.get("norm_prm")
            data_norm_prm = dict(norm_prm) if isinstance(norm_prm, Mapping) else None
        elif normalization is not None:
            data_norm_prm = normalization.norm_prm_dict()
        else:
            data_norm_prm = None
        model_norm_prm = dict(cfg.model_norm_prm) if isinstance(cfg.model_norm_prm, Mapping) else None
        context_length = None
        if cfg.data.timelapse_prm is not None:
            context_length = cfg.data.timelapse_prm.context_length

        paths = Paths(settings)
        manifest_path = resolve_split_manifest_path(paths, run_dir)
        split_manifest = read_split_manifest(manifest_path) if manifest_path.exists() else None

        return cls(
            run_dir=Path(run_dir),
            experiment_name=Path(run_dir).name,
            dataset_name=cfg.data.dataset_name,
            data_subfolder=cfg.routing.data_subfolder,
            data_cfg=cfg.data.model_dump(exclude_none=True),
            model_architecture=architecture,
            model_parameters=model_parameters,
            data_norm_prm=data_norm_prm,
            model_norm_prm=model_norm_prm,
            noise_model_name=_noise_model_name(cfg.noise_model),
            checkpoint_methods=_checkpoint_methods(cfg),
            patch_size=cfg.data.model_patch_size,
            downsamp_factor=cfg.data.downsampling_factor,
            upsampling_factor=model_parameters.effective_upsampling_factor(),
            context_length=int(context_length) if context_length is not None else None,
            default_tiling_size=_default_tiling_size(architecture, cfg=cfg, run_dir=run_dir),
            split_manifest=split_manifest,
        )



def load_saved_run(run_dir: Path) -> SavedTrainingRun:
    """Load, validate, and project a saved training config from a run directory."""
    run_dir = Path(run_dir)
    cfg_path = Paths(settings).cfg_train_path(run_dir=run_dir)
    cfg = ResolvedExperiment.model_validate(load_yaml(cfg_path))
    return SavedTrainingRun.from_resolved(cfg, run_dir=run_dir)



__all__ = ["CheckpointMethod", "SavedTrainingRun", "load_saved_run", "resolve_run_dir"]

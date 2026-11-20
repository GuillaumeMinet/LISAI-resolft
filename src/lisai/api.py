"""Convenient Python boundaries; internal inference functions receive explicit paths."""

from pathlib import Path
from typing import Literal, overload

import numpy as np

from lisai.config.models.inference import ApplyOverrides
from lisai.config.settings import Settings, load_default_settings
from lisai.evaluation.inference.results import PredictionInfo, PredictionResult
from lisai.infra.paths import Paths
from lisai.model_sources import RunRef


def train(config_path, *, progress_bar: bool | None = None, settings: Settings | None = None):
    """Run training; import its optional dependencies only when requested."""
    from lisai.training.run_training import run_training

    return run_training(config_path, progress_bar=progress_bar, settings=settings)


def train_from_dict(config: dict, *, progress_bar: bool | None = None, settings: Settings | None = None):
    """Run training from a dict without requiring training dependencies at import time."""
    from lisai.training.run_training import run_training_from_config_dict

    return run_training_from_config_dict(config, progress_bar=progress_bar, settings=settings)


def run_apply_model(*args, **kwargs):
    """Defer the file-based inference engine until prediction is requested."""
    from lisai.evaluation.run_apply_model import run_apply_model as _run_apply_model

    return _run_apply_model(*args, **kwargs)


def run_evaluate(*args, **kwargs):
    """Defer evaluation-specific dependencies until evaluation is requested."""
    from lisai.evaluation.run_evaluate import run_evaluate as _run_evaluate

    return _run_evaluate(*args, **kwargs)


@overload
def predict(
    model: str | RunRef, data: np.ndarray, *, settings: Settings,
    output_dir: None = None, config: str | Path | None = None,
    overrides: ApplyOverrides | None = None, device: str = "auto",
    overwrite: Literal[False] = False, return_info: Literal[False] = False,
) -> np.ndarray: ...


@overload
def predict(
    model: str | RunRef, data: np.ndarray, *, settings: Settings,
    output_dir: None = None, config: str | Path | None = None,
    overrides: ApplyOverrides | None = None, device: str = "auto",
    overwrite: Literal[False] = False, return_info: Literal[True],
) -> tuple[np.ndarray, PredictionInfo]: ...


@overload
def predict(
    model: str | RunRef, data: str | Path, *, settings: Settings,
    output_dir: str | Path | None = None, config: str | Path | None = None,
    overrides: ApplyOverrides | None = None, device: str = "auto",
    overwrite: bool = False, return_info: Literal[False] = False,
) -> PredictionResult: ...


def predict(
    model: str | RunRef,
    data: np.ndarray | str | Path,
    *,
    settings: Settings,
    output_dir: str | Path | None = None,
    config: str | Path | None = None,
    overrides: ApplyOverrides | None = None,
    device: str = "auto",
    overwrite: bool = False,
    return_info: bool = False,
) -> np.ndarray | tuple[np.ndarray, PredictionInfo] | PredictionResult:
    """Predict from an installed model name or an explicit saved-run reference.

    A string means a promoted model, while RunRef(run_id=...) means a training
    run in the supplied workspace. Arrays never create intermediate TIFFs.
    """
    if not isinstance(model, RunRef) and (not isinstance(model, str) or not model.strip()):
        raise TypeError("model must be a promoted-model name or RunRef(run_id=...).")
    if not isinstance(settings, Settings):
        raise TypeError("settings must be an explicitly constructed Settings instance.")
    if overrides is not None and not isinstance(overrides, ApplyOverrides):
        raise TypeError("overrides must be an ApplyOverrides instance.")
    is_array = isinstance(data, np.ndarray)
    is_path = isinstance(data, (str, Path))
    if not is_array and not is_path:
        raise TypeError("data must be a numpy.ndarray, TIFF path or directory path.")
    if is_array:
        if output_dir is not None or overwrite:
            raise ValueError("output_dir and overwrite apply only to TIFF/path prediction.")
        saving = overrides.saving if overrides is not None else None
        if saving is not None and any(
            key in saving.model_fields_set for key in ("mode", "in_place", "save_folder")
        ):
            raise ValueError("Array prediction cannot use explicit output-routing overrides.saving settings.")
        if data.ndim not in (2, 3, 4):
            raise ValueError("Array data must have YX, TYX or TZYX dimensions.")
    else:
        if return_info:
            raise ValueError("return_info is for arrays only; PredictionResult already includes file metadata.")
        if isinstance(data, str) and not data.strip():
            raise ValueError("data path must not be empty.")
        input_path = Path(data)
        if input_path.is_file() and input_path.suffix.lower() not in (".tif", ".tiff"):
            raise ValueError("Path input must be a TIFF (.tif/.tiff) file or directory.")

    from lisai.evaluation.defaults import resolve_apply_config
    from lisai.evaluation.runtime import initialize_runtime
    from lisai.evaluation.saved_run import load_saved_run
    from lisai.evaluation.run_apply_model import compute_prediction_array

    paths = Paths(settings)
    if isinstance(model, RunRef):
        from lisai.runs.lookup import resolve_run_ref

        selected_run = resolve_run_ref(model, paths=paths)
        saved_run = load_saved_run(selected_run.run_dir, paths=paths)
        promoted = None
    else:
        from lisai.promoted_models.package import load_promoted_model

        selected_run = None
        promoted = load_promoted_model(model.strip(), paths=paths)
        saved_run = promoted.saved_run
    sparse = overrides.model_dump(exclude_unset=True) if overrides is not None else {}
    if output_dir is not None:
        dest = Path(output_dir).expanduser().resolve()
        chosen = sparse.get("saving", {})
        if "save_folder" in chosen:
            authored = chosen["save_folder"]
            if authored is None or Path(authored).expanduser().resolve() != dest:
                raise ValueError("Conflicting explicit output_dir and overrides.saving.save_folder.")
        if chosen.get("mode") is not None or chosen.get("in_place") is not None:
            raise ValueError("Conflicting explicit output_dir and overrides.saving output mode.")
        sparse = {**sparse, "saving": {**chosen, "save_folder": str(dest)}}
    resolved_overrides = ApplyOverrides.model_validate(sparse) if sparse else None
    cfg = resolve_apply_config(
        model_config=promoted.inference_config_path if promoted is not None else None,
        config=config, overrides=resolved_overrides, stg=settings,
    )
    if is_path:
        # The new public TIFF adapter accepts both extensions regardless of
        # legacy experiment-specific CLI filters in a bundled inference YAML.
        cfg = cfg.model_copy(update={
            "input": cfg.input.model_copy(update={"filters": ["tif", "tiff"]})
        })
    if is_array:
        # Sampling outputs are a file-saving option; avoid calculating them for arrays.
        cfg = cfg.model_copy(update={
            "saving": cfg.saving.model_copy(update={"lvae_save_samples": False})
        })
        runtime_options = dict(
            saved_run=saved_run, paths=paths,
            tiling_size=cfg.inference.tiling_size,
            device=None if device == "auto" else device,
        )
        if promoted is not None:
            runtime_options.update(
                checkpoint_path=promoted.weights_path,
                noise_model_path=promoted.noise_model_path,
                noise_model_norm_prm_path=promoted.noise_model_norm_prm_path,
            )
        else:
            runtime_options.update(
                best_or_last=cfg.checkpoint.best_or_last,
                epoch_number=cfg.checkpoint.epoch_number,
            )
        runtime = initialize_runtime(**runtime_options)
        result, _, _, info, _ = compute_prediction_array(
            data, saved_run=saved_run, runtime=runtime, cfg=cfg,
            verbose=False,
        )
        return (result, info) if return_info else result

    # Path adapter uses the same computation function through run_apply_model,
    # preserving output-numbering, in-place and explicit overwrite behavior.
    return run_apply_model(
        cfg,
        model_dataset=selected_run.dataset if selected_run is not None else "",
        model_subfolder=selected_run.model_subfolder if selected_run is not None else "promoted",
        model_name=selected_run.run_dir.name if selected_run is not None else model,
        data_path=Path(data), paths=paths,
        promoted_model_name=model if promoted is not None else None,
        saved_run_dir=selected_run.run_dir if selected_run is not None else None,
        device=None if device == "auto" else device,
        overwrite=overwrite, return_result=True,
    )


def evaluate(cfg, *, settings: Settings | None = None, **options):
    stg = load_default_settings() if settings is None else settings
    return run_evaluate(cfg, paths=Paths(stg), **options)

"""Convenient Python boundaries; internal inference functions receive explicit paths."""

from pathlib import Path
from typing import Callable, Literal, overload

import numpy as np

from lisai.config.models.inference import ApplyOverrides
from lisai.config.settings import Settings, load_default_settings
from lisai.evaluation.inference.results import ArrayPrediction, PredictionInfo, PredictionResult
from lisai.evaluation.inference.progress import InferenceProgressEvent, InferenceProgressReporter
from lisai.hdn_performance import HDNPerformanceOptions, InferencePlan
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


def _resolve_prediction_setup(model, *, settings: Settings, config,
                              overrides: ApplyOverrides | None, device: str,
                              performance: HDNPerformanceOptions | None = None,
                              output_dir=None):
    """Load source metadata/config once; preserve sparse override precedence."""
    from lisai.evaluation.defaults import resolve_apply_config
    from lisai.evaluation.saved_run import load_saved_run

    if not isinstance(model, RunRef) and (not isinstance(model, str) or not model.strip()):
        raise TypeError("model must be a promoted-model name or RunRef(run_id=...).")
    if not isinstance(settings, Settings):
        raise TypeError("settings must be an explicitly constructed Settings instance.")
    if overrides is not None and not isinstance(overrides, ApplyOverrides):
        raise TypeError("overrides must be an ApplyOverrides instance.")
    if performance is not None and not isinstance(performance, HDNPerformanceOptions):
        raise TypeError("performance must be an HDNPerformanceOptions instance.")
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
    # Keyword performance has per-field precedence over typed overrides/YAML.
    if performance is not None:
        # An explicitly supplied empty options object means "Auto / Auto";
        # otherwise merge only authored fields to preserve YAML precision/batch.
        explicit = (performance.model_dump(exclude_unset=True)
                    if performance.model_fields_set else performance.model_dump())
        if explicit:
            inference = dict(sparse.get("inference") or {})
            inference["performance"] = {**dict(inference.get("performance") or {}), **explicit}
            sparse = {**sparse, "inference": inference}
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
    if not saved_run.is_lvae and (
        performance is not None
        or (overrides is not None and overrides.inference is not None
            and overrides.inference.performance is not None)
    ):
        raise ValueError("Explicit HDN performance options are invalid for non-HDN models.")
    return paths, selected_run, saved_run, promoted, cfg


def _load_prediction_runtime(*, saved_run, promoted, paths, cfg, device):
    from lisai.evaluation.runtime import initialize_runtime
    runtime_options = dict(
        saved_run=saved_run, paths=paths, tiling_size=cfg.inference.tiling_size,
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
    return initialize_runtime(**runtime_options)


@overload
def predict(
    model: str | RunRef, data: np.ndarray, *, settings: Settings,
    output_dir: None = None, config: str | Path | None = None,
    overrides: ApplyOverrides | None = None, device: str = "auto",
    performance: HDNPerformanceOptions | None = None,
    measure_gpu_memory: bool = False,
    on_progress: Callable[[InferenceProgressEvent], None] | None = None,
    cancel_check: Callable[[], None] | None = None,
    return_samples: bool = False,
    overwrite: Literal[False] = False, return_info: Literal[False] = False,
) -> np.ndarray | ArrayPrediction: ...


@overload
def predict(
    model: str | RunRef, data: np.ndarray, *, settings: Settings,
    output_dir: None = None, config: str | Path | None = None,
    overrides: ApplyOverrides | None = None, device: str = "auto",
    performance: HDNPerformanceOptions | None = None,
    measure_gpu_memory: bool = False,
    on_progress: Callable[[InferenceProgressEvent], None] | None = None,
    cancel_check: Callable[[], None] | None = None,
    return_samples: bool = False,
    overwrite: Literal[False] = False, return_info: Literal[True],
) -> tuple[np.ndarray, PredictionInfo] | ArrayPrediction: ...


@overload
def predict(
    model: str | RunRef, data: str | Path, *, settings: Settings,
    output_dir: str | Path | None = None, config: str | Path | None = None,
    overrides: ApplyOverrides | None = None, device: str = "auto",
    performance: HDNPerformanceOptions | None = None,
    measure_gpu_memory: bool = False,
    on_progress: Callable[[InferenceProgressEvent], None] | None = None,
    cancel_check: Callable[[], None] | None = None,
    return_samples: bool = False,
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
    performance: HDNPerformanceOptions | None = None,
    measure_gpu_memory: bool = False,
    on_progress: Callable[[InferenceProgressEvent], None] | None = None,
    cancel_check: Callable[[], None] | None = None,
    return_samples: bool = False,
    overwrite: bool = False,
    return_info: bool = False,
) -> np.ndarray | tuple[np.ndarray, PredictionInfo] | ArrayPrediction | PredictionResult:
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

    from lisai.evaluation.run_apply_model import compute_prediction_array

    if return_samples and not is_array:
        raise ValueError("return_samples is only supported for NumPy array prediction.")
    reporter = (InferenceProgressReporter(on_progress, cancel_check)
                if on_progress is not None or cancel_check is not None else None)
    if reporter is not None:
        reporter.stage("loading")

    paths, selected_run, saved_run, promoted, cfg = _resolve_prediction_setup(
        model, settings=settings, config=config, overrides=overrides,
        device=device, performance=performance, output_dir=output_dir,
    )
    if return_samples and not saved_run.is_lvae:
        raise ValueError("return_samples requires an HDN/LVAE model.")
    if is_path:
        # The new public TIFF adapter accepts both extensions regardless of
        # legacy experiment-specific CLI filters in a bundled inference YAML.
        cfg = cfg.model_copy(update={
            "input": cfg.input.model_copy(update={"filters": ["tif", "tiff"]})
        })
    if is_array:
        # The caller explicitly opts into the additional memory required to
        # retain individual HDN stochastic draws. Ordinary arrays stay cheap.
        cfg = cfg.model_copy(update={
            "saving": cfg.saving.model_copy(update={"lvae_save_samples": bool(return_samples)})
        })
        runtime = _load_prediction_runtime(
            saved_run=saved_run, promoted=promoted, paths=paths, cfg=cfg, device=device,
        )
        if reporter is not None:
            reporter.checkpoint()
        result, samples, _, info, _ = compute_prediction_array(
            data, saved_run=saved_run, runtime=runtime, cfg=cfg,
            verbose=False, measure_gpu_memory=measure_gpu_memory,
            progress_reporter=reporter,
        )
        if return_samples:
            if samples is None:
                raise RuntimeError("HDN inference did not return requested samples.")
            return ArrayPrediction(result, samples, info)
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
        measure_gpu_memory=measure_gpu_memory,
        progress_reporter=reporter,
    )


def plan_inference(
    model: str | RunRef,
    data: np.ndarray | str | Path,
    *,
    settings: Settings,
    config: str | Path | None = None,
    overrides: ApplyOverrides | None = None,
    device: str = "auto",
    performance: HDNPerformanceOptions | None = None,
) -> InferencePlan | None:
    """Preview the HDN plan without making stochastic draws or writing TIFFs.

    Loads the selected checkpoint, reads the given image (if TIFF), normalizes
    it and queries CUDA memory. For a directory containing multiple TIFFs,
    plan each file separately; a directory has no unique effective shape.
    Non-HDN models return None.
    """
    from tifffile import imread
    from lisai.evaluation.run_apply_model import compute_prediction_plan

    if isinstance(data, np.ndarray):
        image = data
    elif isinstance(data, (str, Path)):
        path = Path(data)
        if not path.is_file() or path.suffix.lower() not in (".tif", ".tiff"):
            raise ValueError("Plan preview requires one NumPy array or TIFF file, not a directory.")
        image = imread(path)
    else:
        raise TypeError("Plan preview requires one NumPy array or TIFF file.")
    if image.ndim not in (2, 3, 4):
        raise ValueError("Plan preview requires YX, TYX or TZYX input.")
    paths, _, saved_run, promoted, cfg = _resolve_prediction_setup(
        model, settings=settings, config=config, overrides=overrides,
        device=device, performance=performance,
    )
    runtime = _load_prediction_runtime(
        saved_run=saved_run, promoted=promoted, paths=paths, cfg=cfg, device=device,
    )
    return compute_prediction_plan(image, saved_run=saved_run, runtime=runtime, cfg=cfg)


def evaluate(cfg, *, settings: Settings | None = None, **options):
    stg = load_default_settings() if settings is None else settings
    return run_evaluate(cfg, paths=Paths(stg), **options)

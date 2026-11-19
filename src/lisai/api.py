"""Convenient Python boundaries; internal inference functions receive explicit paths."""

from typing import Mapping

from lisai.config.models.inference import InputNormalization
from lisai.config.settings import Settings, load_default_settings
from lisai.infra.paths import Paths


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


def predict(
    cfg,
    *,
    settings: Settings | None = None,
    input_normalization: InputNormalization | Mapping[str, object] | None = None,
    **options,
):
    """Predict with optional input intensity adaptation (training/image/manual).

    ``input_normalization`` overrides ``cfg.input.normalization`` for this
    prediction; pass e.g. {"mode": "manual", "gain": 10.0, "offset": -7.3}.
    """
    stg = load_default_settings() if settings is None else settings
    if input_normalization is not None:
        options["input_normalization"] = InputNormalization.model_validate(input_normalization)
    return run_apply_model(cfg, paths=Paths(stg), **options)


def evaluate(cfg, *, settings: Settings | None = None, **options):
    stg = load_default_settings() if settings is None else settings
    return run_evaluate(cfg, paths=Paths(stg), **options)

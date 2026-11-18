"""Convenient Python boundaries; internal inference functions receive explicit paths."""

from typing import Mapping

from lisai.config.models.inference import InputNormalization
from lisai.config.settings import Settings, load_default_settings
from lisai.evaluation import run_evaluate, run_apply_model
from lisai.infra.paths import Paths
from lisai.training.run_training import (
    run_training as train,
    run_training_from_config_dict as train_from_dict,
)


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

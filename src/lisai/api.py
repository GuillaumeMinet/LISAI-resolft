"""Convenient Python boundaries; internal inference functions receive explicit paths."""

from lisai.config.settings import Settings, load_default_settings
from lisai.evaluation import run_evaluate, run_apply_model
from lisai.infra.paths import Paths
from lisai.training.run_training import (
    run_training as train,
    run_training_from_config_dict as train_from_dict,
)


def predict(cfg, *, settings: Settings | None = None, **options):
    stg = load_default_settings() if settings is None else settings
    return run_apply_model(cfg, paths=Paths(stg), **options)


def evaluate(cfg, *, settings: Settings | None = None, **options):
    stg = load_default_settings() if settings is None else settings
    return run_evaluate(cfg, paths=Paths(stg), **options)

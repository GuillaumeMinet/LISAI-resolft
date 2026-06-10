from .io import load_yaml, prune_config_for_saving, resolve_config, resolve_config_dict, save_yaml
from .models import (
    ContinueTrainingConfig,
    DataConfig,
    ExperimentConfig,
    ProjectConfig,
    ResolvedExperiment,
    RetrainConfig,
)
from .settings import settings

_JSON_SCHEMA_EXPORTS = {
    "continue_training_json_schema",
    "experiment_json_schema",
    "experiment_template_json_schema",
    "preprocess_json_schema",
    "retrain_json_schema",
    "write_continue_training_json_schema",
    "write_experiment_json_schema",
    "write_experiment_template_json_schema",
    "write_preprocess_json_schema",
    "write_retrain_json_schema",
}


def __getattr__(name: str):
    if name in _JSON_SCHEMA_EXPORTS:
        from . import json_schema as _json_schema

        return getattr(_json_schema, name)
    raise AttributeError(name)

__all__ = [
    "settings",
    "resolve_config",
    "resolve_config_dict",
    "prune_config_for_saving",
    "load_yaml",
    "save_yaml",
    "ProjectConfig",
    "DataConfig",
    "ExperimentConfig",
    "ContinueTrainingConfig",
    "RetrainConfig",
    "ResolvedExperiment",
    "experiment_json_schema",
    "experiment_template_json_schema",
    "continue_training_json_schema",
    "retrain_json_schema",
    "write_experiment_json_schema",
    "write_experiment_template_json_schema",
    "write_continue_training_json_schema",
    "write_retrain_json_schema",
    "preprocess_json_schema",
    "write_preprocess_json_schema",
]

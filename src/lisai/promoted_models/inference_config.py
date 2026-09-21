from __future__ import annotations

from pathlib import Path

from lisai.config import load_yaml
from lisai.config.io.config_paths import ConfigPathResolver
from lisai.config.models.inference import InferenceOverrides

PROMOTED_INFERENCE_CONFIG_FILENAME = "config_inference.yaml"


def resolve_promoted_inference_config_source(config: str | Path) -> Path:
    """Resolve and validate an inference config before attaching it to a model.

    Promoted-model inference configs are intentionally sparse, but they must
    define an ``apply`` section because their purpose is to provide the model's
    default settings for ``lisai apply --model``.
    """
    config_path = ConfigPathResolver("inference").resolve(config)
    if config_path is None:
        raise FileNotFoundError(f"Inference config not found: {config}")

    parsed = InferenceOverrides.model_validate(load_yaml(config_path))
    if parsed.apply is None:
        raise ValueError(
            f"Inference config '{config_path}' does not define an 'apply' section. "
            "Promoted-model inference configs must define defaults for `lisai apply --model`."
        )
    return config_path


__all__ = [
    "PROMOTED_INFERENCE_CONFIG_FILENAME",
    "resolve_promoted_inference_config_source",
]

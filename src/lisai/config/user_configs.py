"""Explicit user-config scaffolding; shared by workspace/dev init in Patch E.

No function here is called by Settings construction or normal CLI dispatch.
"""
from __future__ import annotations

from pathlib import Path

import yaml

from lisai.config.models.inference import ResolvedInferenceConfig
from lisai.config.models.inference.presets import POST_TRAINING_OVERRIDES
from lisai.config.settings import Settings

INFERENCE_DEFAULTS_HINT = "# yaml-language-server: $schema=../schema/inference-defaults.schema.json"
INFERENCE_OVERRIDES_HINT = "# yaml-language-server: $schema=../schema/inference.schema.json"


def seed_inference_local_configs(config_root: str | Path) -> list[Path]:
    """Create *missing* flat user-editable inference files, never overwrite existing ones."""
    root = Path(config_root).expanduser().resolve() / "inference"
    candidates = (
        (root / "defaults.yml", ResolvedInferenceConfig().model_dump(mode="json"), INFERENCE_DEFAULTS_HINT),
        (root / "post_training.yml", POST_TRAINING_OVERRIDES, INFERENCE_OVERRIDES_HINT),
    )
    created: list[Path] = []
    for target, values, schema_hint in candidates:
        if target.exists():
            continue
        root.mkdir(parents=True, exist_ok=True)
        # Exclusive creation protects against accidental overwrite even in a
        # simultaneous explicit initialization by two processes.
        with target.open("x", encoding="utf-8") as handle:
            handle.write(schema_hint + "\n" + yaml.safe_dump(values, sort_keys=False))
        created.append(target)
    return created


def refresh_user_schemas(*, stg: Settings) -> list[Path]:
    """Regenerate editor-facing JSON schemas for the selected config root.

    The generated schemas are a cache; user-authored YAML files are untouched.
    Dataset-aware experiment schemas read the *selected* workspace registry.
    """
    from lisai.config.json_schema import (
        write_continue_training_json_schema, write_experiment_json_schema,
        write_experiment_template_json_schema, write_inference_defaults_json_schema,
        write_inference_json_schema, write_local_config_json_schema,
        write_preprocess_json_schema, write_retrain_json_schema,
    )

    root = stg.CONFIGS_ROOT / "schema"
    return [
        write_experiment_json_schema(root / "experiment.schema.json", stg=stg),
        write_experiment_template_json_schema(root / "experiment-template.schema.json", stg=stg),
        write_continue_training_json_schema(root / "continue_training.schema.json"),
        write_retrain_json_schema(root / "retrain.schema.json", stg=stg),
        write_inference_defaults_json_schema(root / "inference-defaults.schema.json"),
        write_inference_json_schema(root / "inference.schema.json"),
        write_preprocess_json_schema(root / "preprocess.schema.json", stg=stg),
        write_local_config_json_schema(root / "local-config.schema.json"),
    ]

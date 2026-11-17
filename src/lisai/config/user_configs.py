"""Explicit user-config scaffolding; shared by workspace/dev init in Patch E.

No function here is called by Settings construction or normal CLI dispatch.
"""
from __future__ import annotations

from pathlib import Path

import yaml

from lisai.config.models.inference import ResolvedInferenceConfig
from lisai.config.models.inference.presets import POST_TRAINING_OVERRIDES
from lisai.config.settings import Settings
from lisai.config.io.schema_hints import schema_header


def inference_seed_files(config_root: str | Path) -> dict[Path, str]:
    """Produce both editable inference files without writing anything.

    The same content is used by explicit initialization and workspace updates;
    the editor hints always point at the selected writable config root.
    """
    config_root = Path(config_root).expanduser().resolve()
    folder = config_root / "inference"
    schema_root = config_root / "schema"
    contents = (
        ("defaults.yml", ResolvedInferenceConfig().model_dump(mode="json"),
         "inference-defaults.schema.json"),
        ("post_training.yml", POST_TRAINING_OVERRIDES, "inference.schema.json"),
    )
    result: dict[Path, str] = {}
    for filename, values, schema_name in contents:
        target = folder / filename
        header = schema_header(target, schema_root, schema_name)
        text = yaml.safe_dump(values, sort_keys=False)
        result[target] = f"{header}\n{text}" if header else text
    return result


def seed_inference_local_configs(config_root: str | Path) -> list[Path]:
    """Create *missing* flat user-editable inference files, never overwrite existing ones."""
    created: list[Path] = []
    for target, text in inference_seed_files(config_root).items():
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation protects against accidental overwrite even during
        # simultaneous explicit initialization by different processes.
        with target.open("x", encoding="utf-8") as handle:
            handle.write(text)
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

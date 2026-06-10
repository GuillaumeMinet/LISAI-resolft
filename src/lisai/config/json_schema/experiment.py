from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from ..models import ContinueTrainingConfig, ExperimentConfig, RetrainConfig

_PLACEHOLDER_SCHEMA = {
    "type": "string",
    "pattern": r"^(?:<[A-Za-z_][A-Za-z0-9_.-]*>|\$\{[A-Za-z_][A-Za-z0-9_.-]*\})$",
}

_CATALOG_METADATA_SCHEMA = {
    "type": "object",
    "title": "ConfigMetadata",
    "description": "Catalog metadata used by `lisai configs`; stripped before runtime validation.",
    "additionalProperties": True,
    "properties": {
        "kind": {
            "type": "string",
            "enum": ["preset", "template", "example", "local"],
            "description": "Config catalog kind.",
        },
        "name": {"type": "string", "description": "Human-readable config name."},
        "task": {"type": "string", "description": "Associated training task name."},
        "description": {"type": "string", "description": "Short catalog description."},
    },
}


def experiment_json_schema() -> dict:
    return _with_catalog_metadata(ExperimentConfig.model_json_schema())


def experiment_template_json_schema() -> dict:
    schema = _relax_placeholders(experiment_json_schema())
    schema["title"] = "ExperimentTemplateConfig"
    schema["description"] = (
        "Relaxed training config schema for editable templates. Placeholder strings such as "
        "`<dataset_name>` or `${dataset_name}` are allowed and must be replaced before training."
    )
    return schema


def continue_training_json_schema() -> dict:
    return _with_catalog_metadata(ContinueTrainingConfig.model_json_schema())


def retrain_json_schema() -> dict:
    return _with_catalog_metadata(RetrainConfig.model_json_schema())


def write_experiment_json_schema(output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    schema = experiment_json_schema()
    path.write_text(json.dumps(schema, indent=2) + "\n", encoding="utf-8")
    return path


def write_experiment_template_json_schema(output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    schema = experiment_template_json_schema()
    path.write_text(json.dumps(schema, indent=2) + "\n", encoding="utf-8")
    return path


def write_continue_training_json_schema(output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    schema = continue_training_json_schema()
    path.write_text(json.dumps(schema, indent=2) + "\n", encoding="utf-8")
    return path


def write_retrain_json_schema(output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    schema = retrain_json_schema()
    path.write_text(json.dumps(schema, indent=2) + "\n", encoding="utf-8")
    return path


def _with_catalog_metadata(schema: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(schema)
    out.setdefault("properties", {})["metadata"] = deepcopy(_CATALOG_METADATA_SCHEMA)
    return out


def _relax_placeholders(value: Any) -> Any:
    if isinstance(value, list):
        return [_relax_placeholders(item) for item in value]
    if not isinstance(value, dict):
        return value

    out = {key: _relax_placeholders(item) for key, item in value.items()}
    if _placeholder_allowed_here(out):
        return {"anyOf": [out, deepcopy(_PLACEHOLDER_SCHEMA)]}
    if "anyOf" in out:
        any_of = list(out["anyOf"])
        if not any(_is_placeholder_schema(item) for item in any_of):
            any_of.append(deepcopy(_PLACEHOLDER_SCHEMA))
        out["anyOf"] = any_of
    return out


def _placeholder_allowed_here(schema: dict[str, Any]) -> bool:
    if "const" in schema or "enum" in schema:
        return True
    schema_type = schema.get("type")
    if isinstance(schema_type, str):
        return schema_type not in {"object", "array"}
    if isinstance(schema_type, list):
        return any(item not in {"object", "array", "null"} for item in schema_type)
    return False


def _is_placeholder_schema(schema: Any) -> bool:
    return isinstance(schema, dict) and schema.get("pattern") == _PLACEHOLDER_SCHEMA["pattern"]

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from lisai.data.dataset_registry import (
    load_dataset_registry,
    registry_data_format_for_output,
    registry_data_types,
    registry_value_for_data_type,
)
from lisai.infra.paths import Paths

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


@dataclass
class _RegistryDatasetSchema:
    inputs: set[str] = field(default_factory=set)
    targets: set[str] = field(default_factory=set)
    input_formats: dict[str, set[str]] = field(default_factory=dict)
    data_formats: set[str] = field(default_factory=set)


def experiment_json_schema() -> dict:
    return _with_registry_awareness(
        _with_catalog_metadata(ExperimentConfig.model_json_schema()),
    )


def experiment_template_json_schema() -> dict:
    schema = _relax_placeholders(_with_catalog_metadata(ExperimentConfig.model_json_schema()))
    schema = _with_registry_awareness(schema, allow_placeholders=True)
    schema["title"] = "ExperimentTemplateConfig"
    schema["description"] = (
        "Relaxed training config schema for editable templates. Placeholder strings such as "
        "`<dataset_name>` or `${dataset_name}` are allowed and must be replaced before training."
    )
    return schema


def continue_training_json_schema() -> dict:
    return _with_catalog_metadata(ContinueTrainingConfig.model_json_schema())


def retrain_json_schema() -> dict:
    return _with_registry_awareness(
        _with_catalog_metadata(RetrainConfig.model_json_schema()),
    )


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


def _with_registry_awareness(schema: dict[str, Any], *, allow_placeholders: bool = False) -> dict[str, Any]:
    out = deepcopy(schema)
    data_schema = _data_definition_schema(out)
    if data_schema is None:
        return out

    registry = _load_dataset_registry_for_schema()
    summaries = _registry_schema_summaries(registry)
    if not summaries:
        return out

    dataset_names = sorted(summaries)
    data_properties = data_schema.setdefault("properties", {})
    dataset_name_schema = data_properties.get("dataset_name")
    if isinstance(dataset_name_schema, dict):
        _add_string_suggestions(dataset_name_schema, dataset_names)

    conditions = []
    conditions.extend(_known_dataset_conditions(summaries))
    conditions.append(_unknown_dataset_requires_unprepared_condition(dataset_names, allow_placeholders=allow_placeholders))
    out.setdefault("allOf", []).extend(conditions)
    return out


def _load_dataset_registry_for_schema() -> dict[str, dict[str, Any]]:
    try:
        return load_dataset_registry(Paths().dataset_registry_path())
    except Exception:
        return {}


def _data_definition_schema(schema: dict[str, Any]) -> dict[str, Any] | None:
    data_property = schema.get("properties", {}).get("data")
    if not isinstance(data_property, dict):
        return None

    ref = data_property.get("$ref")
    if not isinstance(ref, str) or not ref.startswith("#/$defs/"):
        return data_property

    ref_name = ref.rsplit("/", 1)[-1]
    data_schema = schema.get("$defs", {}).get(ref_name)
    return data_schema if isinstance(data_schema, dict) else None


def _add_string_suggestions(schema: dict[str, Any], values: list[str]) -> None:
    if not values:
        return

    schema["examples"] = values
    any_of = list(schema.get("anyOf", []))
    any_of.insert(
        0,
        {
            "type": "string",
            "enum": values,
            "description": "Registered dataset names available in the current dataset registry.",
        },
    )
    if not any(_is_plain_string_schema(item) for item in any_of):
        any_of.append({"type": "string"})
    schema["anyOf"] = any_of


def _is_plain_string_schema(schema: Any) -> bool:
    return isinstance(schema, dict) and schema.get("type") == "string" and "enum" not in schema


def _registry_schema_summaries(registry: Mapping[str, Any]) -> dict[str, _RegistryDatasetSchema]:
    summaries: dict[str, _RegistryDatasetSchema] = {}
    for dataset_name, info in registry.items():
        if not isinstance(info, Mapping):
            continue
        summary = _registry_schema_summary(info)
        summaries[str(dataset_name)] = summary
    return summaries


def _registry_schema_summary(info: Mapping[str, Any]) -> _RegistryDatasetSchema:
    summary = _RegistryDatasetSchema()
    fallback = info.get("data_format")
    if fallback is not None:
        summary.data_formats.add(str(fallback))

    data_types = registry_data_types(info)
    if not data_types:
        data_types = {None}

    for data_type in data_types:
        outputs = registry_value_for_data_type(info, "outputs", data_type)
        if not isinstance(outputs, list):
            continue

        for output in outputs:
            if not isinstance(output, Mapping):
                continue
            role = output.get("role")
            values = _output_path_values(output)
            if role == "inp":
                for value in values:
                    summary.inputs.add(value)
                    data_format = registry_data_format_for_output(info, data_type, value)
                    if data_format is not None:
                        text = str(data_format)
                        summary.data_formats.add(text)
                        summary.input_formats.setdefault(value, set()).add(text)
            elif role == "gt":
                summary.targets.update(values)

    return summary


def _output_path_values(output: Mapping[str, Any]) -> set[str]:
    path = output.get("path")
    if path is not None:
        path = {str(path)}
    else:
        key = output.get("key")
        path = str(key) if key is not None else set()
    return path


def _known_dataset_conditions(summaries: Mapping[str, _RegistryDatasetSchema]) -> list[dict[str, Any]]:
    return [
        {
            "if": _prepared_dataset_if(dataset_name),
            "then": _prepared_dataset_then(summary),
        }
        for dataset_name, summary in sorted(summaries.items())
    ]


def _prepared_dataset_if(dataset_name: str) -> dict[str, Any]:
    return {
        "required": ["data"],
        "properties": {
            "data": {
                "required": ["dataset_name"],
                "properties": {
                    "dataset_name": {"const": dataset_name},
                    "prep_before": {"not": {"const": False}},
                },
            },
        },
    }


def _prepared_dataset_then(summary: _RegistryDatasetSchema) -> dict[str, Any]:
    data_schema: dict[str, Any] = {"properties": {}, "allOf": []}

    if summary.inputs:
        input_values = sorted(summary.inputs)
        data_schema["properties"]["input"] = _string_enum_schema(input_values)
        data_schema["properties"]["inp"] = _string_enum_schema(input_values)
        data_schema["allOf"].append({"anyOf": [{"required": ["input"]}, {"required": ["inp"]}]})

    if summary.targets:
        target_values = sorted(summary.targets)
        data_schema["properties"]["target"] = _nullable_string_enum_schema(target_values)
        data_schema["properties"]["gt"] = _nullable_string_enum_schema(target_values)
        data_schema["allOf"].append(
            {
                "if": {"properties": {"paired": {"const": True}}, "required": ["paired"]},
                "then": {
                    "anyOf": [
                        {"required": ["target"], "properties": {"target": _string_enum_schema(target_values)}},
                        {"required": ["gt"], "properties": {"gt": _string_enum_schema(target_values)}},
                    ]
                },
            }
        )

    if summary.data_formats:
        data_schema["properties"]["data_format"] = _nullable_string_enum_schema(sorted(summary.data_formats))

    for input_name, formats in sorted(summary.input_formats.items()):
        if not formats:
            continue
        data_schema["allOf"].extend(
            _input_data_format_conditions(input_name=input_name, data_formats=sorted(formats))
        )

    if not data_schema["allOf"]:
        data_schema.pop("allOf")

    return {"properties": {"data": data_schema}}


def _input_data_format_conditions(*, input_name: str, data_formats: list[str]) -> list[dict[str, Any]]:
    format_schema = _nullable_string_enum_schema(data_formats)
    return [
        {
            "if": {"properties": {field_name: {"const": input_name}}, "required": [field_name]},
            "then": {"properties": {"data_format": format_schema}},
        }
        for field_name in ("input", "inp")
    ]


def _unknown_dataset_requires_unprepared_condition(
    dataset_names: list[str],
    *,
    allow_placeholders: bool,
) -> dict[str, Any]:
    dataset_name_schema: dict[str, Any] = {"not": {"enum": dataset_names}}
    if allow_placeholders:
        dataset_name_schema = {
            "allOf": [
                dataset_name_schema,
                {"not": deepcopy(_PLACEHOLDER_SCHEMA)},
            ]
        }

    return {
        "if": {
            "required": ["data"],
            "properties": {
                "data": {
                    "required": ["dataset_name"],
                    "properties": {"dataset_name": dataset_name_schema},
                }
            },
        },
        "then": {
            "properties": {
                "data": {
                    "required": ["prep_before"],
                    "properties": {"prep_before": {"const": False}},
                }
            }
        },
    }


def _string_enum_schema(values: list[str]) -> dict[str, Any]:
    return {
        "type": "string",
        "enum": values,
    }


def _nullable_string_enum_schema(values: list[str]) -> dict[str, Any]:
    return {
        "anyOf": [
            _string_enum_schema(values),
            {"type": "null"},
        ]
    }


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

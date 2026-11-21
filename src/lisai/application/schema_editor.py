"""Small schema-backed field descriptions for a first, task-aware Qt editor.

The JSON Schema remains the source of field types, descriptions and enumerations.
Only *presentation priority* is curated here; this is not a second validation model.
"""
from __future__ import annotations

from dataclasses import dataclass
from io import StringIO
from typing import Any


# Scientific knobs first; keep these lists deliberately short. Other fields remain
# accessible in the raw YAML tab without inventing a second configuration engine.
TASK_FIELDS: dict[str, tuple[str, ...]] = {
    "denoising_hdn": ("betaKL", "supervised"),
    "denoising_care": ("loss",),
    "denoising_unetrcan": ("loss",),
    "upsamp_single_frame": ("sampling_ratio", "upsampling_factor"),
    "upsamp_multiframes": ("temporal_window", "upsampling_factor"),
    "custom": (),
}
TASK_LABELS: dict[str, str] = {
    "denoising_hdn": "HDN denoising",
    "denoising_care": "CARE denoising",
    "denoising_unetrcan": "UNet-RCAN denoising",
    "upsamp_single_frame": "Single-frame upsampling",
    "upsamp_multiframes": "Multi-frame upsampling",
    "custom": "Custom / low-level",
}
TASK_DEFS = {
    "denoising_hdn": "DenoisingHDNTaskSection",
    "denoising_care": "DenoisingCARETaskSection",
    "denoising_unetrcan": "DenoisingUNetRCANTaskSection",
    "upsamp_single_frame": "UpsampSingleFrameTaskSection",
    "upsamp_multiframes": "UpsampMultipleSamplingTaskSection",
    "custom": "CustomTaskSection",
}


@dataclass(frozen=True)
class EditorField:
    path: tuple[str, ...]
    title: str
    description: str
    field_type: str
    choices: tuple[Any, ...] = ()
    default: Any = None


def _ref(schema: dict, definition: dict) -> dict:
    if "$ref" in definition:
        return schema.get("$defs", {}).get(definition["$ref"].split("/")[-1], {})
    return definition


def _field(schema: dict, definition: dict, name: str, path: tuple[str, ...]) -> EditorField:
    info = _ref(schema, definition).get("properties", {}).get(name, {})
    choices = info.get("enum", ())
    if "const" in info:
        choices = (info["const"],)
    kind = info.get("type", "")
    if not kind and info.get("anyOf"):
        kinds = [item.get("type") for item in info["anyOf"] if item.get("type") != "null"]
        if len(kinds) == 1:
            kind = kinds[0]
        if not choices:
            choices = tuple(item["const"] for item in info["anyOf"] if "const" in item)
    return EditorField(path, info.get("title", name.replace("_", " ").capitalize()),
                       info.get("description", ""), kind, tuple(choices), info.get("default"))


def training_editor_fields(settings, task_name: str) -> dict[str, tuple[EditorField, ...]]:
    """Return a small ordered set of schema-derived fields for an existing task.

    No Qt imports, no default insertion and no mutation of the configuration.
    Unknown task names have no generated scientific fields (raw YAML still works).
    """
    from lisai.config.json_schema.experiment import experiment_json_schema

    schema = experiment_json_schema(stg=settings)
    experiment = _ref(schema, schema["properties"]["experiment"])
    data = _ref(schema, schema["properties"]["data"])
    training = _ref(schema, schema["properties"]["training"])
    task_definition = schema.get("$defs", {}).get(TASK_DEFS.get(task_name, ""), {})
    scientific = tuple(_field(schema, task_definition, name, ("experiment", "task", name))
                       for name in TASK_FIELDS.get(task_name, ()))
    return {
        "Experiment": (
            _field(schema, experiment, "exp_name", ("experiment", "exp_name")),
            *scientific,
        ),
        "Dataset": (
            _field(schema, data, "dataset_name", ("data", "dataset_name")),
        ),
        "Optimization": tuple(_field(schema, training, name, ("training", name))
                              for name in ("n_epochs", "batch_size", "learning_rate")),
    }


def update_yaml_field(text: str, path: tuple[str, ...], value: Any) -> str:
    """Round-trip one guided edit; preserve comments/unknown keys/raw YAML.

    ruamel.yaml is loaded only on this edit path, so non-Qt LISAI needs no
    additional dependency. Raw YAML editing continues to work without it.
    """
    if not path:
        raise ValueError("A field path is required.")
    try:
        from ruamel.yaml import YAML
    except ImportError as exc:
        raise RuntimeError("Guided YAML editing requires ruamel.yaml (pip install 'lisai[qt]').") from exc
    engine = YAML(typ="rt")
    engine.preserve_quotes = True
    doc = engine.load(text)
    if not isinstance(doc, dict):
        raise ValueError("Training YAML must be a mapping.")
    node = doc
    for name in path[:-1]:
        child = node.get(name)
        if child is None:
            node[name] = {}
            child = node[name]
        if not isinstance(child, dict):
            raise ValueError(f"Cannot edit {'.'.join(path)}: {name} is not an object.")
        node = child
    node[path[-1]] = value
    out = StringIO()
    engine.dump(doc, out)
    return out.getvalue()


__all__ = ["EditorField", "TASK_LABELS", "TASK_FIELDS", "training_editor_fields", "update_yaml_field"]

"""Catalog-backed training configuration drafts; no Qt and no implicit writes."""
from __future__ import annotations

from pathlib import Path

from lisai.application.config_document import TrainingConfigDocument
from lisai.config.catalog import (
    TrainingConfigEntry,
    build_training_config_from_preset,
    build_training_config_from_template,
    discover_training_configs,
    render_training_config_yaml,
    load_training_config,
)


def list_config_sources(settings, *, task: str | None = None,
                        kind: str | None = None) -> list[TrainingConfigEntry]:
    """Catalog entries include local files and packaged presets/examples/templates."""
    entries = discover_training_configs(stg=settings, kind=kind, task=task)
    # The packaged training examples also include runner sweep files. A sweep
    # is not a standalone experiment YAML and cannot be saved with this editor.
    selected = []
    for entry in entries:
        if entry.kind == "example" and entry.ownership == "built-in":
            body, _metadata, _location = load_training_config(entry.relative_path, stg=settings)
            if not isinstance(body.get("experiment"), dict):
                continue
        selected.append(entry)
    return selected


def open_existing_document(settings, config_ref: str | Path) -> TrainingConfigDocument:
    """A writable training YAML is opened without creating or altering a file."""
    return TrainingConfigDocument.open(settings, config_ref)


def create_document_from_source(
    settings, source_ref: str, *, dataset_selection: str | None = None,
    experiment_name: str | None = None,
) -> tuple[TrainingConfigDocument, list[str]]:
    """Construct an unsaved draft from a *packaged* config source.

    Presets and templates use the exact CLI scaffold builder, before its write
    boundary. Examples are copied verbatim (including their hardcoded dataset
    references) and intentionally do not receive preset-style substitutions.
    """
    entries = {e.relative_path: e for e in list_config_sources(settings)}
    for e in tuple(entries.values()):
        if Path(e.relative_path).suffix.lower() in {".yml", ".yaml"}:
            entries.setdefault(e.relative_path.rsplit(".", 1)[0], e)
    entry = entries.get(source_ref)
    if entry is None:
        raise FileNotFoundError(f"Unknown training configuration source: {source_ref}")
    if entry.ownership != "built-in":
        raise ValueError("For an existing writable configuration, use open_existing_document().")
    source_ref = entry.relative_path  # Resolve optional .yml/.yaml extension once.
    if entry.kind == "preset":
        cfg, warnings = build_training_config_from_preset(
            source_ref, exp_name=experiment_name, dataset_name=dataset_selection, stg=settings)
        document = TrainingConfigDocument.from_text(
            render_training_config_yaml(cfg), source_kind="preset", source_ref=source_ref)
    elif entry.kind == "template":
        cfg, warnings = build_training_config_from_template(
            source_ref, exp_name=experiment_name, dataset_name=dataset_selection, stg=settings)
        document = TrainingConfigDocument.from_text(
            render_training_config_yaml(cfg), source_kind="template", source_ref=source_ref)
    elif entry.kind == "example":
        document = TrainingConfigDocument.from_source(settings, source_ref)
        warnings = [
            "Example dataset and experiment names are preserved exactly. "
            "Review the data and routing sections before training."
        ]
        if dataset_selection is not None or experiment_name is not None:
            warnings.append("Selections were not applied to the example; edit them explicitly.")
    else:
        raise ValueError(f"Cannot create a draft from configuration kind {entry.kind!r}.")
    return document, warnings


__all__ = ["list_config_sources", "open_existing_document", "create_document_from_source"]

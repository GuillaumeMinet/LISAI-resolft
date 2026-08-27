from __future__ import annotations

from pathlib import Path

from lisai.config import load_yaml, save_yaml, settings
from lisai.infra.paths import Paths

from .schema import PromotedModelRegistry, PromotedModelRegistryEntry


def load_promoted_model_registry(*, paths: Paths | None = None) -> PromotedModelRegistry:
    resolved_paths = paths or Paths(settings)
    registry_path = resolved_paths.promoted_model_registry_path()
    if not registry_path.exists():
        return PromotedModelRegistry()
    return PromotedModelRegistry.model_validate(load_yaml(registry_path))


def save_promoted_model_registry(
    registry: PromotedModelRegistry,
    *,
    paths: Paths | None = None,
) -> Path:
    resolved_paths = paths or Paths(settings)
    registry_path = resolved_paths.promoted_model_registry_path()
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    save_yaml(registry.model_dump(mode="json"), registry_path)
    return registry_path


def register_promoted_model(
    *,
    name: str,
    entry: PromotedModelRegistryEntry,
    overwrite: bool = False,
    paths: Paths | None = None,
) -> PromotedModelRegistry:
    registry = load_promoted_model_registry(paths=paths)
    if name in registry.models and not overwrite:
        raise FileExistsError(
            f"Promoted model {name!r} is already registered. Use --overwrite to replace it."
        )
    models = dict(registry.models)
    models[name] = entry
    updated = registry.model_copy(update={"models": models})
    updated = PromotedModelRegistry.model_validate(updated.model_dump(mode="python"))
    save_promoted_model_registry(updated, paths=paths)
    return updated


def promoted_source_run_ids(*, paths: Paths | None = None) -> set[str]:
    registry = load_promoted_model_registry(paths=paths)
    return {entry.source_run_id for entry in registry.models.values()}


def resolve_registered_model_dir(name: str, *, paths: Paths | None = None) -> Path:
    resolved_paths = paths or Paths(settings)
    registry = load_promoted_model_registry(paths=resolved_paths)
    entry = registry.models.get(name)
    if entry is None:
        raise KeyError(f"Unknown promoted model {name!r}.")
    model_dir = resolved_paths.promoted_models_root() / entry.path
    if not model_dir.is_dir():
        raise FileNotFoundError(
            f"Promoted model {name!r} is registered but its directory is missing: {model_dir}"
        )
    return model_dir


__all__ = [
    "load_promoted_model_registry",
    "promoted_source_run_ids",
    "register_promoted_model",
    "resolve_registered_model_dir",
    "save_promoted_model_registry",
]

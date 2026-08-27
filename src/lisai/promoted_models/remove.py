from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from lisai.config import settings
from lisai.infra.paths import Paths

from .registry import load_promoted_model_registry, save_promoted_model_registry
from .schema import PromotedModelRegistry, PromotedModelRegistryEntry


@dataclass(frozen=True)
class RemovedPromotedModel:
    name: str
    model_dir: Path
    entry: PromotedModelRegistryEntry


def remove_promoted_model(
    name: str,
    *,
    paths: Paths | None = None,
) -> RemovedPromotedModel:
    """Remove one local promoted model while leaving exported archives untouched."""
    resolved_paths = paths or Paths(settings)
    registry = load_promoted_model_registry(paths=resolved_paths)
    entry = registry.models.get(name)
    if entry is None:
        raise KeyError(f"Unknown promoted model {name!r}.")

    promoted_root = resolved_paths.promoted_models_root()
    model_dir = promoted_root / entry.path
    if not model_dir.is_dir():
        raise FileNotFoundError(
            f"Promoted model {name!r} is registered but its directory is missing: {model_dir}"
        )

    models = dict(registry.models)
    del models[name]
    updated = PromotedModelRegistry.model_validate(
        registry.model_copy(update={"models": models}).model_dump(mode="python")
    )

    # Rename inside the same filesystem first. If registry persistence fails,
    # restore the directory so local state remains consistent.
    promoted_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="lisai-remove-", dir=promoted_root) as tmp:
        staged_dir = Path(tmp) / model_dir.name
        model_dir.rename(staged_dir)
        try:
            save_promoted_model_registry(updated, paths=resolved_paths)
        except Exception:
            staged_dir.rename(model_dir)
            raise
        shutil.rmtree(staged_dir)

    return RemovedPromotedModel(name=name, model_dir=model_dir, entry=entry)


__all__ = ["RemovedPromotedModel", "remove_promoted_model"]

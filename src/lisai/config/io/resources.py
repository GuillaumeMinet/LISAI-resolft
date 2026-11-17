"""Small read-only config-resource helpers; package assets are never copied to workspaces.

Keep resources as Traversables: an installed wheel typically exposes ordinary files,
while zip importers can expose virtual files. Only materialize within a managed
``as_file`` scope when a downstream API genuinely requires a filesystem path.
"""
from __future__ import annotations

from importlib import resources
from lisai._compat import Traversable
from pathlib import Path
from typing import Iterator

PACKAGE_ROOT = "lisai.resources"

BUILTIN_NAMESPACES = {
    "training": ("presets", "templates", "examples"),
    "inference": ("presets",),
    "preprocess": ("templates",),
}


def resource_directory(*parts: str) -> Traversable:
    return resources.files(PACKAGE_ROOT).joinpath(*parts)


def config_resource_directory(kind: str, namespace: str) -> Traversable:
    if namespace not in BUILTIN_NAMESPACES.get(kind, ()):
        raise ValueError(f"No built-in {kind} {namespace} namespace")
    return resource_directory("configs", kind, namespace)


def candidate_names(value: str | Path, suffixes: tuple[str, ...]) -> tuple[str, ...]:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"Package config name must be a relative path without '..': {value}")
    return (path.as_posix(),) if path.suffix else (path.as_posix(), *(path.with_suffix(s).as_posix() for s in suffixes))


def find_resource(kind: str, namespace: str, name: str | Path,
                  *, suffixes: tuple[str, ...] = (".yml", ".yaml")) -> Traversable | None:
    root = config_resource_directory(kind, namespace)
    for candidate in candidate_names(name, suffixes):
        item = root.joinpath(*Path(candidate).parts)
        if item.is_file():
            return item
    return None


def iter_resource_files(kind: str, namespace: str) -> Iterator[tuple[str, Traversable]]:
    """Enumerate package config files using Traversable APIs (zip-safe)."""
    root = config_resource_directory(kind, namespace)
    if not root.is_dir():
        return
    def walk(directory: Traversable, prefix: str = "") -> Iterator[tuple[str, Traversable]]:
        for item in sorted(directory.iterdir(), key=lambda child: child.name):
            rel = f"{prefix}{item.name}"
            if item.is_dir():
                yield from walk(item, rel + "/")
            elif item.is_file() and item.name.lower().endswith((".yml", ".yaml")):
                yield rel, item
    yield from walk(root)

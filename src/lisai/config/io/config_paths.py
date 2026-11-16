"""Resolve editable configs and read-only package resources by explicit namespace."""
from __future__ import annotations

from dataclasses import dataclass
from importlib.resources.abc import Traversable
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from .resources import BUILTIN_NAMESPACES, find_resource, iter_resource_files

if TYPE_CHECKING:
    from lisai.config.settings import Settings

ConfigKind = Literal["training", "preprocess", "inference"]
ConfigOwnership = Literal["built-in", "local", "explicit"]
ConfigLocation = Path | Traversable

_KIND_LABELS = {"training": "Training", "preprocess": "Preprocess", "inference": "Inference"}


@dataclass(frozen=True)
class ConfigReference:
    location: ConfigLocation
    ownership: ConfigOwnership
    namespace: str | None = None


@dataclass(frozen=True)
class ConfigPathResolver:
    kind: ConfigKind
    stg: Settings

    @property
    def root(self) -> Path:
        if self.kind == "training":
            return self.stg.TRAINING_CONFIG_DIR
        if self.kind == "preprocess":
            return self.stg.PREPROCESS_CONFIG_DIR
        if self.kind == "inference":
            return self.stg.INFERENCE_CONFIG_DIR
        raise ValueError(f"Unknown config kind: {self.kind}")

    @property
    def suffixes(self) -> tuple[str, ...]:
        return tuple(self.stg.CONFIG_SUFFIXES)

    @property
    def label(self) -> str:
        return _KIND_LABELS[self.kind]

    @property
    def default_name(self) -> str | None:
        return self.stg.INFERENCE_DEFAULT_CONFIG_NAME if self.kind == "inference" else None

    def candidate_paths(self, path: str | Path) -> tuple[Path, ...]:
        path = Path(path).expanduser()
        return (path,) if path.suffix else (path, *(path.with_suffix(s) for s in self.suffixes))

    def first_existing_path(self, candidates) -> Path | None:
        for candidate in candidates:
            if candidate.is_file():
                return candidate.resolve()
        return None

    def available(self) -> list[str]:
        available: set[str] = set()
        local_root = self.root / "local"
        if local_root.is_dir():
            for suffix in self.suffixes:
                available.update(f"local/{path.relative_to(local_root).as_posix()}"
                                 for path in local_root.rglob(f"*{suffix}") if path.is_file())
        for namespace in BUILTIN_NAMESPACES.get(self.kind, ()):
            available.update(f"{namespace}/{name}" for name, _ in iter_resource_files(self.kind, namespace))
        # Older checkouts may still contain root-level dataset-specific files.
        # Keep them addressable as explicit paths until guided migration in E.
        if self.kind == "preprocess":
            for suffix in self.suffixes:
                available.update(p.name for p in self.root.glob(f"*{suffix}") if p.is_file())
        return sorted(available)

    def missing_error(self, config_arg: str | Path) -> FileNotFoundError:
        lines = [f"{self.label} config not found: {config_arg}"]
        available = self.available()
        if available:
            lines.append("Available configs:")
            lines.extend(f"  - {name}" for name in available)
        else:
            lines.append(f"No {self.kind} configs found in {self.root / 'local'} or package resources.")
        return FileNotFoundError("\n".join(lines))

    def resolve_reference(self, config_arg: str | Path | None) -> ConfigReference | None:
        if config_arg is None:
            if self.default_name is None:
                return None
            return self.resolve_reference(self.default_name) if self.first_existing_path(
                self.candidate_paths(self.root / self.default_name)
            ) is not None else None

        path = Path(config_arg).expanduser()
        # An explicit file path is authoritative (especially absolute model bundles).
        if path.is_absolute():
            found = self.first_existing_path(self.candidate_paths(path))
            if found is not None:
                return ConfigReference(found, "explicit")
            raise self.missing_error(config_arg)

        namespaces = BUILTIN_NAMESPACES.get(self.kind, ())
        if path.parts and path.parts[0] in namespaces:
            name = Path(*path.parts[1:])
            if not name.parts:
                raise self.missing_error(config_arg)
            found = find_resource(self.kind, path.parts[0], name, suffixes=self.suffixes)
            if found is not None:
                return ConfigReference(found, "built-in", path.parts[0])
            raise self.missing_error(config_arg)

        if path.parts and path.parts[0] == "local":
            if ".." in path.parts or len(path.parts) == 1:
                raise ValueError(f"Local config must be a name within {self.root / 'local'}")
            found = self.first_existing_path(self.candidate_paths(self.root / path))
            if found is not None:
                return ConfigReference(found, "local", "local")
            raise self.missing_error(config_arg)

        # Paths existing relative to the caller are explicit filenames, not a
        # fallback to an unrelated config tree.
        found = self.first_existing_path(self.candidate_paths(path))
        if found is not None:
            return ConfigReference(found, "explicit")

        local = self.first_existing_path(self.candidate_paths(self.root / "local" / path))
        builtins = [(namespace, item) for namespace in namespaces
                    if (item := find_resource(self.kind, namespace, path, suffixes=self.suffixes)) is not None]
        if local is not None and builtins:
            raise ValueError(
                f"Ambiguous {self.kind} config {config_arg!r}: both local and built-in. "
                f"Use 'local/{path.as_posix()}' or an explicit package namespace."
            )
        if len(builtins) > 1:
            names = ", ".join(n for n, _ in builtins)
            raise ValueError(f"Ambiguous {self.kind} config {config_arg!r} in {names}; use a namespace.")
        if local is not None:
            return ConfigReference(local, "local", "local")
        if builtins:
            namespace, resource = builtins[0]
            return ConfigReference(resource, "built-in", namespace)

        # Root-relative legacy files can still be named explicitly while local
        # inference/preprocess configs are migrated (M2 D). They are not built-ins.
        legacy = self.first_existing_path(self.candidate_paths(self.root / path))
        if legacy is not None:
            return ConfigReference(legacy, "explicit")
        raise self.missing_error(config_arg)

    def resolve(self, config_arg: str | Path | None) -> ConfigLocation | None:
        ref = self.resolve_reference(config_arg)
        return None if ref is None else ref.location

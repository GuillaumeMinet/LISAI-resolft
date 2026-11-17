"""Deterministic config names: flat user root, explicit package namespaces, and paths."""
from __future__ import annotations

from dataclasses import dataclass
from lisai._compat import Traversable
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from .resources import BUILTIN_NAMESPACES, find_resource, iter_resource_files

if TYPE_CHECKING:
    from lisai.config.settings import Settings

ConfigKind = Literal["training", "preprocess", "inference"]
ConfigOwnership = Literal["built-in", "local", "explicit"]
ConfigLocation = Path | Traversable

_KIND_LABELS = {"training": "Training", "preprocess": "Preprocess", "inference": "Inference"}


def _explicit_relative(value: str | Path) -> bool:
    """Check the spelling before pathlib discards the leading './'."""
    return isinstance(value, str) and value.startswith(("./", "../", ".\\", "..\\"))


def managed_output_path(value: str | Path, *, root: Path, kind: ConfigKind) -> Path:
    """Resolve an authored output name in its writable root, or an explicit file path.

    Bare/nested names belong to the selected domain root, even when cwd contains
    a same-named file. Only absolute and lexical ./ or ../ are external paths.
    """
    path = Path(value).expanduser()
    if path.is_absolute() or _explicit_relative(value):
        target = path.resolve()
    else:
        if not path.parts or ".." in path.parts:
            raise ValueError(f"Invalid managed config name: {value!r}")
        if path.parts[0] in BUILTIN_NAMESPACES.get(kind, ()):
            raise ValueError(f"Reserved packaged config namespace: {path.parts[0]}/")
        target = (root / path).resolve()
        if not target.is_relative_to(root.resolve()):
            raise ValueError(f"Managed config escapes writable root {root}: {value!r}")
    if not target.suffix:
        target = target.with_suffix(".yml")
    return target


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

    def user_config_paths(self):
        if self.root.is_dir():
            for path in sorted(self.root.rglob("*")):
                if path.is_file() and path.suffix.lower() in self.suffixes:
                    # Do not allow symlinks escaping the writable root to be exposed
                    # as managed/user configs.
                    if path.resolve().is_relative_to(self.root.resolve()):
                        relative = path.relative_to(self.root).as_posix()
                        if relative.split("/", 1)[0] not in BUILTIN_NAMESPACES.get(self.kind, ()):
                            yield relative, path

    def available(self) -> list[str]:
        available = {name for name, _ in self.user_config_paths()}
        for namespace in BUILTIN_NAMESPACES.get(self.kind, ()):
            available.update(f"{namespace}/{name}" for name, _ in iter_resource_files(self.kind, namespace))
        return sorted(available)

    def missing_error(self, config_arg: str | Path) -> FileNotFoundError:
        lines = [f"{self.label} config not found: {config_arg}"]
        available = self.available()
        if available:
            lines.append("Available configs:")
            lines.extend(f"  - {name}" for name in available)
        else:
            lines.append(f"No {self.kind} configs found in {self.root} or package resources.")
        return FileNotFoundError("\n".join(lines))

    def resolve_reference(self, config_arg: str | Path | None) -> ConfigReference | None:
        if config_arg is None:
            if self.default_name is None:
                return None
            return self._user_reference(self.default_name, missing_ok=True)

        path = Path(config_arg).expanduser()
        # Absolute and lexical ./ or ../ mean exactly that file, never a
        # managed name. Do not normalize away explicit markers before checking.
        if path.is_absolute() or _explicit_relative(config_arg):
            found = self.first_existing_path(self.candidate_paths(path))
            if found is not None:
                return ConfigReference(found, "explicit")
            raise self.missing_error(config_arg)

        if not path.parts or ".." in path.parts:
            raise ValueError(f"Invalid managed config name: {config_arg!r}")

        namespaces = BUILTIN_NAMESPACES.get(self.kind, ())
        if path.parts[0] in namespaces:
            if len(path.parts) == 1:
                raise self.missing_error(config_arg)
            found = find_resource(self.kind, path.parts[0], Path(*path.parts[1:]), suffixes=self.suffixes)
            if found is not None:
                return ConfigReference(found, "built-in", path.parts[0])
            raise self.missing_error(config_arg)

        return self._user_reference(path, missing_ok=False)

    def _user_reference(self, name: str | Path, *, missing_ok: bool) -> ConfigReference | None:
        path = Path(name)
        if ".." in path.parts or not path.parts:
            raise ValueError(f"Invalid managed config name: {name!r}")
        root = self.root.resolve()
        candidates = self.candidate_paths(root / path)
        for candidate in candidates:
            # Containment is checked even for symlinks, not just lexical paths.
            if not candidate.resolve().is_relative_to(root):
                raise ValueError(f"Managed config escapes writable root {root}: {name!r}")
        found = self.first_existing_path(candidates)
        if found is not None:
            return ConfigReference(found, "local")
        if missing_ok:
            return None
        raise self.missing_error(name)

    def resolve(self, config_arg: str | Path | None) -> ConfigLocation | None:
        ref = self.resolve_reference(config_arg)
        return None if ref is None else ref.location

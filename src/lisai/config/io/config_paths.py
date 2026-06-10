from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

ConfigKind = Literal["training", "preprocess", "inference"]


_KIND_LABELS = {
    "training": "Training",
    "preprocess": "Preprocess",
    "inference": "Inference",
}


@dataclass(frozen=True)
class ConfigPathResolver:
    kind: ConfigKind
    stg: object | None = None

    @property
    def settings(self):
        if self.stg is not None:
            return self.stg
        from lisai.config.settings import settings

        return settings

    @property
    def root(self) -> Path:
        stg = self.settings
        if self.kind == "training":
            return stg.TRAINING_CONFIG_DIR
        if self.kind == "preprocess":
            return stg.PREPROCESS_CONFIG_DIR
        if self.kind == "inference":
            return stg.INFERENCE_CONFIG_DIR
        raise ValueError(f"Unknown config kind: {self.kind}")

    @property
    def suffixes(self) -> tuple[str, ...]:
        return tuple(self.settings.CONFIG_SUFFIXES)

    @property
    def label(self) -> str:
        return _KIND_LABELS[self.kind]

    @property
    def default_name(self) -> str | None:
        if self.kind == "inference":
            return self.settings.INFERENCE_DEFAULT_CONFIG_NAME
        return None

    def candidate_paths(self, path: str | Path) -> tuple[Path, ...]:
        path = Path(path).expanduser()
        candidates = [path]
        if not path.suffix:
            candidates.extend(path.with_suffix(suffix) for suffix in self.suffixes)
        return tuple(candidates)

    def first_existing_path(self, candidates) -> Path | None:
        for candidate in candidates:
            if candidate.exists():
                return candidate.resolve()
        return None

    def available(self) -> list[str]:
        available: set[str] = set()
        for suffix in self.suffixes:
            available.update(
                path.relative_to(self.root).as_posix()
                for path in self.root.rglob(f"*{suffix}")
                if path.is_file()
            )
        return sorted(available)

    def missing_error(self, config_arg: str | Path) -> FileNotFoundError:
        lines = [f"{self.label} config not found: {config_arg}"]
        available = self.available()
        if available:
            lines.append("Available configs:")
            lines.extend(f"  - {name}" for name in available)
        else:
            lines.append(f"No {self.kind} configs were found under {self.root}.")
        return FileNotFoundError("\n".join(lines))

    def resolve(self, config_arg: str | Path | None) -> Path | None:
        if config_arg is None:
            if self.default_name is None:
                return None
            return self.first_existing_path(
                self.candidate_paths(self.root / self.default_name)
            )

        config_path = Path(config_arg).expanduser()

        resolved = self.first_existing_path(self.candidate_paths(config_path))
        if resolved is not None:
            return resolved

        if not config_path.is_absolute():
            resolved = self.first_existing_path(
                self.candidate_paths(self.root / config_path)
            )
            if resolved is not None:
                return resolved

        raise self.missing_error(config_arg)

"""One toolkit-independent, editable training YAML document.

Guided controls and raw editors use this same buffer.  The document deliberately
owns no Qt objects, file dialogs or training execution state.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import yaml

from .config_editor import (
    managed_training_config_path,
    save_training_yaml_text,
    validate_training_yaml_authoring,
    validate_training_yaml_text,
)
from .schema_editor import update_yaml_field


def _digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


class TrainingConfigDocument:
    """Editable YAML plus the identity of its last successfully saved revision.

    ``source_ref`` is provenance only. A packaged source is never writable; after
    Save As the new writable file becomes ``save_path``. Invalid user text is
    preserved in memory until explicitly corrected/discarded.
    """

    def __init__(
        self,
        yaml_text: str,
        *,
        source_kind: str = "new",
        source_ref: str | None = None,
        save_path: Path | None = None,
        saved_text: str | None = None,
        saved_revision: str | None = None,
    ):
        if source_kind not in {"new", "existing", "preset", "example", "template"}:
            raise ValueError(f"Unknown training configuration source kind: {source_kind}")
        self.yaml_text = str(yaml_text)
        self.source_kind = source_kind
        self.source_ref = source_ref
        self.save_path = Path(save_path) if save_path is not None else None
        self.saved_text = saved_text
        self._saved_revision = saved_revision

    @classmethod
    def from_text(
        cls, text: str, *, source_kind: str = "new", source_ref: str | None = None,
    ) -> "TrainingConfigDocument":
        if source_kind == "existing":
            raise ValueError("Existing documents must be opened from a writable file.")
        return cls(text, source_kind=source_kind, source_ref=source_ref)

    @classmethod
    def open(cls, settings, filename: str | Path) -> "TrainingConfigDocument":
        """Read a user-owned YAML only; reject external and package paths."""
        path = managed_training_config_path(settings, filename)
        content = path.read_bytes()
        text = content.decode("utf-8")
        return cls(text, source_kind="existing", source_ref=str(path),
                   save_path=path, saved_text=text, saved_revision=_digest(content))

    @classmethod
    def from_source(cls, settings, source_ref: str) -> "TrainingConfigDocument":
        """Read a catalog source into an unsaved draft, never modify resources."""
        from lisai.config.io.config_paths import ConfigPathResolver

        reference = ConfigPathResolver("training", stg=settings).resolve_reference(source_ref)
        if reference is None:
            raise FileNotFoundError(source_ref)
        if reference.ownership == "built-in":
            return cls.from_text(reference.location.read_text(encoding="utf-8"),
                                 source_kind={"presets": "preset", "examples": "example",
                                              "templates": "template"}[reference.namespace],
                                 source_ref=source_ref)
        # Explicit absolute files outside the managed root are not editor-owned.
        return cls.open(settings, reference.location)

    @property
    def dirty(self) -> bool:
        return self.save_path is None or self.yaml_text != self.saved_text

    @property
    def saved_revision(self) -> str | None:
        return self._saved_revision

    def replace_text(self, text: str) -> None:
        self.yaml_text = str(text)

    def read_field(self, path: tuple[str, ...]) -> Any:
        try:
            body = yaml.safe_load(self.yaml_text)
        except yaml.YAMLError as exc:
            raise ValueError(f"Invalid training YAML syntax: {exc}") from exc
        if not isinstance(body, dict):
            raise ValueError("Training YAML must be a mapping (object).")
        node: Any = body
        for part in path:
            if not isinstance(node, dict) or part not in node:
                return None
            node = node[part]
        return node

    def update_field(self, path: tuple[str, ...], value: Any) -> None:
        # Assign only on success. A failed guided edit must preserve the raw buffer.
        try:
            updated = update_yaml_field(self.yaml_text, path, value)
        except Exception as exc:
            try:
                from ruamel.yaml.error import YAMLError
            except ImportError:
                raise exc
            if isinstance(exc, YAMLError):
                raise ValueError(f"Invalid training YAML syntax: {exc}") from exc
            raise
        self.yaml_text = updated

    def validate_for_save(self, settings) -> list[str]:
        """Check syntax/authoring structure, allowing incomplete templates."""
        return validate_training_yaml_authoring(self.yaml_text)

    def validate_for_training(self, settings):
        """Full training validation; unfinished drafts cannot launch."""
        warnings = self.validate_for_save(settings)
        if warnings:
            raise ValueError("Training configuration still contains placeholders: " + "; ".join(warnings))
        return validate_training_yaml_text(self.yaml_text, settings)

    def _check_revision(self) -> None:
        assert self.save_path is not None
        try:
            content = self.save_path.read_bytes()
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"Training YAML disappeared on disk: {self.save_path}. Save As to keep your changes."
            ) from exc
        if _digest(content) != self._saved_revision:
            raise RuntimeError(
                f"Training YAML was modified outside this editor: {self.save_path}. "
                "Reload or use Save As; no changes were written."
            )

    def save(self, settings) -> Path:
        """Save only an existing writable document; never overwrite a package."""
        if self.save_path is None:
            raise ValueError("This configuration has no writable file. Use Save As first.")
        target = managed_training_config_path(settings, self.save_path)
        self._check_revision()
        if not self.dirty:
            return target
        self.validate_for_save(settings)
        saved = save_training_yaml_text(
            self.yaml_text, target, settings=settings, overwrite=True,
            validation="authoring", expected_revision=self._saved_revision,
        )
        self._record_saved(saved)
        return saved

    def save_as(self, settings, destination: str | Path) -> Path:
        """Create a new user YAML exclusively; never overwrite an existing file."""
        self.validate_for_save(settings)
        saved = save_training_yaml_text(
            self.yaml_text, destination, settings=settings, overwrite=False,
            validation="authoring",
        )
        self._record_saved(saved)
        return saved

    def _record_saved(self, path: Path) -> None:
        content = path.read_bytes()
        self.save_path = path
        self.yaml_text = content.decode("utf-8")
        self.saved_text = self.yaml_text
        self._saved_revision = _digest(content)
        # The original source_ref remains available for provenance. From this
        # point the file is an existing user document, not a writable preset.
        self.source_kind = "existing"


__all__ = ["TrainingConfigDocument"]

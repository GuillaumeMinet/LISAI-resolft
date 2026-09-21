from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from lisai.config.settings import settings

IfExistsPolicy = Literal["error", "numbered", "overwrite", "reuse"]
OutputFolderAction = Literal["created", "numbered", "overwritten", "reused"]
ParentPolicy = Literal["create", "require"]


@dataclass(frozen=True)
class OutputFolderResolution:
    """Result of preparing an output folder."""

    requested: Path
    path: Path
    action: OutputFolderAction
    requested_existed: bool

    @property
    def redirected(self) -> bool:
        return self.path != self.requested

    def message(
        self,
        *,
        overwrite_hint: str = "Use --overwrite to replace the existing folder.",
    ) -> str:
        """Return the standard user-facing saving message."""
        if self.action == "numbered":
            return (
                f"SAVING: Folder {self.requested} already exists; "
                f"saving to {self.path} instead. {overwrite_hint}"
            )
        if self.action == "overwritten":
            return (
                f"SAVING: Folder {self.requested} already exists; "
                "--overwrite enabled, replacing it."
            )
        if self.action == "reused":
            return f"SAVING: Reusing output folder: {self.path}"
        return f"SAVING: Saving outputs to: {self.path}"


def prepare_output_folder(
    path: Path,
    *,
    if_exists_policy: IfExistsPolicy,
    parent_policy: ParentPolicy = "create",
) -> OutputFolderResolution:
    """Prepare an output directory according to a reusable existence policy."""
    from .folders import ensure_folder

    requested = Path(path)
    if if_exists_policy not in {"error", "numbered", "overwrite", "reuse"}:
        raise ValueError(f"Unknown if_exists_policy: {if_exists_policy}")
    if parent_policy not in {"create", "require"}:
        raise ValueError(f"Unknown parent_policy: {parent_policy}")

    _prepare_parent(requested.parent, parent_policy=parent_policy)

    requested_existed = requested.exists()
    if not requested_existed:
        resolved = ensure_folder(requested, mode="strict")
        return OutputFolderResolution(
            requested=requested,
            path=resolved,
            action="created",
            requested_existed=False,
        )

    if if_exists_policy == "error":
        ensure_folder(requested, mode="strict")

    if if_exists_policy == "overwrite":
        resolved = ensure_folder(requested, mode="overwrite")
        return OutputFolderResolution(
            requested=requested,
            path=resolved,
            action="overwritten",
            requested_existed=True,
        )

    if if_exists_policy == "reuse":
        resolved = ensure_folder(requested, mode="exist_ok")
        return OutputFolderResolution(
            requested=requested,
            path=resolved,
            action="reused",
            requested_existed=True,
        )

    numbered = _next_numbered_path(requested.parent, requested.name)
    resolved = ensure_folder(numbered, mode="strict")
    return OutputFolderResolution(
        requested=requested,
        path=resolved,
        action="numbered",
        requested_existed=True,
    )


def _prepare_parent(parent: Path, *, parent_policy: ParentPolicy) -> None:
    from .folders import ensure_folder

    if parent_policy == "require" and not parent.exists():
        raise FileNotFoundError(f"Parent folder does not exist: {parent}")
    ensure_folder(parent, mode="exist_ok")


def _next_numbered_path(parent: Path, name: str) -> Path:
    fmt = settings.NAMING.exp_name_format
    max_id = 0

    for entry in parent.iterdir():
        if not entry.name.startswith(name):
            continue

        suffix = entry.name[len(name):]
        clean_suffix = re.sub(r"^[^0-9]+", "", suffix)
        if clean_suffix.isdigit():
            max_id = max(max_id, int(clean_suffix))

    return parent / fmt.format(name=name, id=max_id + 1)

from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Literal, Optional

from .constants import MAIN_OUTPUT_KEY

Axes = Literal["YX", "TYX"]
Role = Literal["inp", "gt", "aux"]

@dataclass(frozen=True)
class OutputDecl:
    key: str
    axes: Axes
    role: Role
    data_format_override: Optional[str] = None
    path: Optional[str] = None

@dataclass(frozen=True)
class OutputSpec:
    """
    An OutputSpec declares the structure of the generated dataset so that
    folder creation, saving logic, and registry updates can be handled
    automatically and consistently.

    Each OutputDecl defines:
      - key: logical name of the output (e.g. "inp", "gt", "inp_snr_1", etc.).
             NOTE: 
                - This must match the keys returned by pipeline.process_item()
                - if key==<MAIN_OUTPUT_KEY>, and self.save_at_root==True,
                 we save directly under preprocess/<data_type>/

      - axes: expected array layout ("YX" for 2D images, "TYX" for stacks).
      - role: semantic meaning ("inp", "gt", or "aux"), useful for training logic.
      - data_format_override: optional loader format for outputs whose format differs
        from the dataset-level data_format.
      - path: optional processed-data subfolder override. If omitted, the existing
        key/save_at_root behavior is used. An empty string explicitly saves at root.

    Example:
        OutputSpec(
            outputs=(
                OutputDecl(key="inp", axes="YX", role="inp"),
                OutputDecl(key="gt", axes="YX", role="gt"),
            )
        )

    This will create the folders:
        preprocess/<data_type>/inp/
        preprocess/<data_type>/gt/

    and register the dataset structure as ["inp", "gt"].

    
    """
    outputs: tuple[OutputDecl, ...]
    save_at_root: bool = False

    def output_keys(self) -> list[Optional[str]]:
        # Keys that must match the dict returned by pipeline.process_item()
        return [o.key for o in self.outputs]

    def __post_init__(self):
        if self.save_at_root and MAIN_OUTPUT_KEY not in self.output_keys():
            raise ValueError(f"save_at_root=True requires an output with key='{MAIN_OUTPUT_KEY}'")

        folders = [self.folder_for(o.key) for o in self.outputs]
        duplicates = sorted({folder for folder in folders if folders.count(folder) > 1})
        if duplicates:
            rendered = ["<root>" if folder == "" else folder for folder in duplicates]
            raise ValueError(f"Output folders must be unique; conflicting paths: {rendered}")

    def folder_for(self, key: str) -> str:
        for o in self.outputs:
            if o.key != key:
                continue

            if o.path is not None:
                path = o.path.strip().replace("\\", "/")
                if path:
                    pure = PurePosixPath(path)
                    if pure.is_absolute() or ".." in pure.parts:
                        raise ValueError(
                            f"Output path for '{key}' must be a relative subfolder without '..': {o.path!r}"
                        )
                    return pure.as_posix()
                return ""

            # Historical behavior when no explicit path override is declared.
            if self.save_at_root and key == MAIN_OUTPUT_KEY:
                return ""
            return o.key

        raise KeyError(key)

    def axes_for(self, key: str) -> Axes:
        for o in self.outputs:
            if o.key == key:
                return o.axes
        raise KeyError(key)
    
    def structure_keys(self) -> list[str]:
        # Registry structure: list of output subfolders.
        # Root output is represented by "".
        out: list[str] = []
        for o in self.outputs:
            out.append(self.folder_for(o.key))
        return out

    def output_entries(self) -> list[dict[str, str]]:
        """Registry-ready descriptions of produced outputs."""
        entries: list[dict[str, str]] = []
        for o in self.outputs:
            entry = {
                "key": o.key,
                "path": self.folder_for(o.key),
                "role": o.role,
                "axes": o.axes,
            }
            if o.data_format_override is not None:
                entry["data_format_override"] = o.data_format_override
            entries.append(entry)
        return entries

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

from .constants import MAIN_OUTPUT_KEY

Axes = Literal["YX", "TYX"]
Role = Literal["inp", "gt", "aux"]

@dataclass(frozen=True)
class OutputDecl:
    key: str
    axes: Axes
    role: Role

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

    def folder_for(self, key: str) -> str:
        # If enabled, one specific output is saved at preprocess/<data_type>/ (no subfolder)
        if self.save_at_root and key == MAIN_OUTPUT_KEY:
            return ""

        # Otherwise, folder name == key
        for o in self.outputs:
            if o.key == key:
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
            if self.save_at_root and o.key == MAIN_OUTPUT_KEY:
                out.append("")
            else:
                out.append(o.key)
        return out

    def output_entries(self) -> list[dict[str, str]]:
        """Registry-ready descriptions of produced outputs."""
        return [
            {
                "key": o.key,
                "path": self.folder_for(o.key),
                "role": o.role,
                "axes": o.axes,
            }
            for o in self.outputs
        ]

from __future__ import annotations

from typing import Literal


LegacyDumpRole = Literal["base", "input"]


def resolve_source_subfolders(
    *,
    pipeline_name: str,
    base_subfolder: str = "",
    input_subfolder: str = "",
    dump_subfolder: str = "",
    legacy_dump_role: LegacyDumpRole,
) -> tuple[str, str]:
    """Resolve new source-folder semantics while preserving legacy dump_subfolder behavior."""

    base = base_subfolder or ""
    inp = input_subfolder or ""
    legacy = dump_subfolder or ""

    if not legacy:
        return base, inp

    if legacy_dump_role == "base":
        if base:
            raise ValueError(
                f"{pipeline_name}: use either 'base_subfolder' or legacy 'dump_subfolder', not both. "
                "'dump_subfolder' is a deprecated alias for 'base_subfolder' in this pipeline."
            )
        base = legacy
    elif legacy_dump_role == "input":
        if inp:
            raise ValueError(
                f"{pipeline_name}: use either 'input_subfolder' or legacy 'dump_subfolder', not both. "
                "'dump_subfolder' is a deprecated alias for 'input_subfolder' in this pipeline."
            )
        inp = legacy
    else:  # defensive; callers use the Literal above
        raise ValueError(f"Unsupported legacy dump role: {legacy_dump_role}")

    return base, inp

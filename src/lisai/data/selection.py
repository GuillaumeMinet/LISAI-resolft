from __future__ import annotations

import sys
from collections.abc import Iterable, Sequence

from lisai.infra.cli.selection import find_name_matches, resolve_ambiguous_matches

_DATASET_LIST_HINT = "Use 'lisai datasets list' to inspect available datasets."


def resolve_dataset_name(
    query: str,
    candidates: Iterable[str],
    *,
    stdin=None,
    stdout=None,
    stderr=None,
    help_hint: str = _DATASET_LIST_HINT,
) -> str | None:
    """Resolve an exact or partial dataset name against known candidates."""
    out = sys.stdout if stdout is None else stdout
    err = sys.stderr if stderr is None else stderr
    in_stream = sys.stdin if stdin is None else stdin
    names = sorted(set(candidates), key=str.casefold)
    matches = find_name_matches(query, names)

    if not matches:
        print(f"No matching dataset found for {query!r}.", file=err)
        print(help_hint, file=err)
        return None

    return resolve_ambiguous_matches(
        matches,
        render_matches=_render_dataset_choices,
        heading="Multiple matching datasets found:",
        rerun_hint=help_hint,
        selection_name="dataset",
        stdin=in_stream,
        stdout=out,
        stderr=err,
    )


def _render_dataset_choices(names: Sequence[str]) -> str:
    index_width = max(2, len(str(len(names))))
    rows = [(f"{index:0{index_width}d}", name) for index, name in enumerate(names, start=1)]
    header = ("#", "dataset")
    widths = [
        max(len(header[column]), *(len(row[column]) for row in rows))
        for column in range(2)
    ]
    lines = [
        "  ".join(header[column].ljust(widths[column]) for column in range(2)),
        "  ".join("-" * widths[column] for column in range(2)),
    ]
    lines.extend(
        "  ".join(row[column].ljust(widths[column]) for column in range(2))
        for row in rows
    )
    return "\n".join(lines)


__all__ = ["resolve_dataset_name"]

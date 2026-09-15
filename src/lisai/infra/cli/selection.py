from __future__ import annotations

import sys
from collections.abc import Callable, Sequence
from typing import TypeVar

from .prompts import is_interactive

_T = TypeVar("_T")


def find_name_matches(query: str, candidates: Sequence[str]) -> list[str]:
    """Return exact name matches, or case-insensitive substring matches as fallback."""
    normalized_query = query.strip().casefold()
    if not normalized_query:
        return []

    exact_matches = [
        candidate for candidate in candidates if candidate.casefold() == normalized_query
    ]
    if exact_matches:
        return exact_matches

    return [
        candidate for candidate in candidates if normalized_query in candidate.casefold()
    ]


def resolve_ambiguous_matches(
    matches: Sequence[_T],
    *,
    render_matches: Callable[[Sequence[_T]], str],
    heading: str,
    rerun_hint: str,
    selection_name: str = "item",
    stdin=None,
    stdout=None,
    stderr=None,
) -> _T | None:
    if not matches:
        return None
    if len(matches) == 1:
        return matches[0]

    out = sys.stdout if stdout is None else stdout
    err = sys.stderr if stderr is None else stderr
    in_stream = sys.stdin if stdin is None else stdin

    print(heading, file=out)
    print(render_matches(matches), file=out)

    selected_idx = _prompt_selection_index(
        len(matches),
        stdin=in_stream,
        stdout=out,
        stderr=err,
        selection_name=selection_name,
    )
    if selected_idx is None:
        print(rerun_hint, file=err)
        return None
    return matches[selected_idx]


def _prompt_selection_index(
    count: int,
    *,
    stdin,
    stdout,
    stderr,
    selection_name: str = "item",
) -> int | None:
    if count <= 1:
        return 0 if count == 1 else None

    if not is_interactive(stdin):
        return None

    while True:
        print(
            f"Select {selection_name} number from '#' (for example 01), or press Enter to cancel: ",
            end="",
            file=stdout,
            flush=True,
        )
        answer = stdin.readline()
        if answer == "":
            return None
        choice = answer.strip()
        if choice == "":
            return None
        if not choice.isdigit():
            print("Invalid selection. Enter a number from the '#' column.", file=stderr)
            continue

        selected = int(choice)
        if 1 <= selected <= count:
            return selected - 1
        print(f"Selection out of range. Enter a value between 1 and {count}.", file=stderr)


__all__ = ["find_name_matches", "resolve_ambiguous_matches"]

from __future__ import annotations

import sys
from collections.abc import Callable, Iterable, Sequence
from typing import TypeVar

from .prompts import is_interactive, prompt_yes_no

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


def resolve_partial_name(
    query: str,
    candidates: Iterable[str],
    *,
    entity_name: str,
    help_hint: str,
    column_name: str | None = None,
    confirm_unique_partial: bool = True,
    stdin=None,
    stdout=None,
    stderr=None,
) -> str | None:
    """Resolve an exact or partial human-entered name against known candidates."""
    out = sys.stdout if stdout is None else stdout
    err = sys.stderr if stderr is None else stderr
    in_stream = sys.stdin if stdin is None else stdin

    names = sorted(set(candidates), key=str.casefold)
    matches = find_name_matches(query, names)
    if not matches:
        print(f"No matching {entity_name} found for {query!r}.", file=err)
        print(help_hint, file=err)
        return None

    normalized_query = query.strip().casefold()
    unique_partial = (
        confirm_unique_partial
        and len(matches) == 1
        and matches[0].casefold() != normalized_query
    )
    label = column_name or entity_name
    return resolve_ambiguous_matches(
        matches,
        render_matches=lambda values: _render_name_choices(values, column_name=label),
        heading=f"Multiple matching {entity_name}s found:",
        rerun_hint=help_hint,
        selection_name=label,
        confirm_single=unique_partial,
        single_prompt=lambda name: f"Did you mean {name!r}? [y/N] ",
        stdin=in_stream,
        stdout=out,
        stderr=err,
    )


def _render_name_choices(names: Sequence[str], *, column_name: str) -> str:
    index_width = max(2, len(str(len(names))))
    rows = [
        (f"{index:0{index_width}d}", name)
        for index, name in enumerate(names, start=1)
    ]
    header = ("#", column_name)
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


def resolve_ambiguous_matches(
    matches: Sequence[_T],
    *,
    render_matches: Callable[[Sequence[_T]], str],
    heading: str,
    rerun_hint: str,
    selection_name: str = "item",
    confirm_single: bool = False,
    single_prompt: Callable[[_T], str] | None = None,
    stdin=None,
    stdout=None,
    stderr=None,
) -> _T | None:
    if not matches:
        return None

    out = sys.stdout if stdout is None else stdout
    err = sys.stderr if stderr is None else stderr
    in_stream = sys.stdin if stdin is None else stdin

    if len(matches) == 1:
        selected = matches[0]
        if not confirm_single:
            return selected
        prompt = (
            single_prompt(selected)
            if single_prompt is not None
            else f"Use matching {selection_name}? [y/N] "
        )
        confirmed = prompt_yes_no(
            prompt,
            stdin=in_stream,
            stdout=out,
            require_interactive=True,
        )
        if confirmed:
            return selected
        print(rerun_hint, file=err)
        return None

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


__all__ = ["find_name_matches", "resolve_ambiguous_matches", "resolve_partial_name"]

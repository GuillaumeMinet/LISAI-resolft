from __future__ import annotations

import sys
from collections.abc import Callable


def is_interactive(stream) -> bool:
    """Return whether a CLI stream is attached to an interactive terminal."""
    is_tty = getattr(stream, "isatty", None)
    return callable(is_tty) and bool(is_tty())


def prompt_yes_no(
    prompt: str,
    *,
    stdin=None,
    stdout=None,
    input_fn: Callable[[str], str] | None = None,
    require_interactive: bool = False,
) -> bool | None:
    """Prompt for a yes/no answer.

    Returns ``True`` for yes, ``False`` for any explicit non-yes answer, and
    ``None`` when no answer can be read or interactive input is required but
    unavailable.
    """
    if input_fn is not None:
        try:
            answer = input_fn(prompt)
        except EOFError:
            return None
        return answer.strip().casefold() in {"y", "yes"}

    in_stream = sys.stdin if stdin is None else stdin
    out = sys.stdout if stdout is None else stdout
    if require_interactive and not is_interactive(in_stream):
        return None

    print(prompt, end="", file=out, flush=True)
    answer = in_stream.readline()
    if answer == "":
        return None
    return answer.strip().casefold() in {"y", "yes"}


__all__ = ["is_interactive", "prompt_yes_no"]

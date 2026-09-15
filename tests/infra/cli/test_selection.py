from __future__ import annotations

import io

from lisai.infra.cli.selection import resolve_ambiguous_matches


class InteractiveInput(io.StringIO):
    def isatty(self) -> bool:
        return True


class NonInteractiveInput(io.StringIO):
    def isatty(self) -> bool:
        return False


def _render(items) -> str:
    return "\n".join(f"{index:02d}  {item}" for index, item in enumerate(items, start=1))


def test_resolve_ambiguous_matches_returns_unique_match_without_prompt():
    assert resolve_ambiguous_matches(
        ["only"],
        render_matches=_render,
        heading="Matches:",
        rerun_hint="Try again.",
    ) == "only"


def test_resolve_ambiguous_matches_allows_interactive_selection():
    stdout = io.StringIO()
    stderr = io.StringIO()

    selected = resolve_ambiguous_matches(
        ["first", "second"],
        render_matches=_render,
        heading="Matches:",
        rerun_hint="Try again.",
        selection_name="entry",
        stdin=InteractiveInput("02\n"),
        stdout=stdout,
        stderr=stderr,
    )

    assert selected == "second"
    assert "Matches:" in stdout.getvalue()
    assert "Select entry number" in stdout.getvalue()
    assert stderr.getvalue() == ""


def test_resolve_ambiguous_matches_non_interactive_returns_none_with_hint():
    stdout = io.StringIO()
    stderr = io.StringIO()

    selected = resolve_ambiguous_matches(
        ["first", "second"],
        render_matches=_render,
        heading="Matches:",
        rerun_hint="Try again.",
        stdin=NonInteractiveInput(""),
        stdout=stdout,
        stderr=stderr,
    )

    assert selected is None
    assert "Matches:" in stdout.getvalue()
    assert "Try again." in stderr.getvalue()

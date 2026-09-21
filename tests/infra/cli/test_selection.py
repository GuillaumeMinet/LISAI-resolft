from __future__ import annotations

import io

from lisai.infra.cli.selection import resolve_ambiguous_matches, resolve_partial_name


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


def test_resolve_ambiguous_matches_can_confirm_unique_match():
    stdout = io.StringIO()
    stderr = io.StringIO()

    selected = resolve_ambiguous_matches(
        ["only"],
        render_matches=_render,
        heading="Matches:",
        rerun_hint="Try again.",
        selection_name="entry",
        confirm_single=True,
        single_prompt=lambda item: f"Did you mean {item!r}? [y/N] ",
        stdin=InteractiveInput("y\n"),
        stdout=stdout,
        stderr=stderr,
    )

    assert selected == "only"
    assert "Did you mean 'only'? [y/N]" in stdout.getvalue()
    assert stderr.getvalue() == ""


def test_resolve_ambiguous_matches_rejects_unconfirmed_unique_match():
    stdout = io.StringIO()
    stderr = io.StringIO()

    selected = resolve_ambiguous_matches(
        ["only"],
        render_matches=_render,
        heading="Matches:",
        rerun_hint="Try again.",
        confirm_single=True,
        single_prompt=lambda item: f"Did you mean {item!r}? [y/N] ",
        stdin=InteractiveInput("n\n"),
        stdout=stdout,
        stderr=stderr,
    )

    assert selected is None
    assert "Did you mean 'only'? [y/N]" in stdout.getvalue()
    assert "Try again." in stderr.getvalue()


def test_resolve_ambiguous_matches_does_not_assume_unique_match_non_interactively():
    stdout = io.StringIO()
    stderr = io.StringIO()

    selected = resolve_ambiguous_matches(
        ["only"],
        render_matches=_render,
        heading="Matches:",
        rerun_hint="Try again.",
        confirm_single=True,
        single_prompt=lambda item: f"Did you mean {item!r}? [y/N] ",
        stdin=NonInteractiveInput("y\n"),
        stdout=stdout,
        stderr=stderr,
    )

    assert selected is None
    assert stdout.getvalue() == ""
    assert "Try again." in stderr.getvalue()


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


def test_find_name_matches_prefers_exact_match_over_partial_matches():
    from lisai.infra.cli.selection import find_name_matches

    assert find_name_matches(
        "actin_fixed",
        ["actin_fixed_multi_snr", "actin_fixed", "other"],
    ) == ["actin_fixed"]


def test_find_name_matches_falls_back_to_case_insensitive_partial_matches():
    from lisai.infra.cli.selection import find_name_matches

    assert find_name_matches(
        "ACTIN_FIXED",
        ["actin_fixed_multi_snr", "other", "actin_fixed_pair"],
    ) == ["actin_fixed_multi_snr", "actin_fixed_pair"]


def test_resolve_partial_name_prefers_exact_match_without_prompt():
    stdout = io.StringIO()
    stderr = io.StringIO()

    selected = resolve_partial_name(
        "actin_fixed",
        ["actin_fixed_multi_snr", "actin_fixed"],
        entity_name="dataset",
        help_hint="List datasets.",
        stdin=NonInteractiveInput(""),
        stdout=stdout,
        stderr=stderr,
    )

    assert selected == "actin_fixed"
    assert stdout.getvalue() == ""
    assert stderr.getvalue() == ""


def test_resolve_partial_name_confirms_unique_partial_match():
    stdout = io.StringIO()
    stderr = io.StringIO()

    selected = resolve_partial_name(
        "actin_fixed",
        ["actin_fixed_multi_snr", "vimentin_live"],
        entity_name="dataset",
        help_hint="List datasets.",
        stdin=InteractiveInput("y\n"),
        stdout=stdout,
        stderr=stderr,
    )

    assert selected == "actin_fixed_multi_snr"
    assert "Did you mean 'actin_fixed_multi_snr'? [y/N]" in stdout.getvalue()
    assert stderr.getvalue() == ""


def test_resolve_partial_name_can_skip_unique_partial_confirmation():
    selected = resolve_partial_name(
        "actin_fixed",
        ["actin_fixed_multi_snr", "vimentin_live"],
        entity_name="dataset",
        help_hint="List datasets.",
        confirm_unique_partial=False,
        stdin=NonInteractiveInput(""),
        stdout=io.StringIO(),
        stderr=io.StringIO(),
    )

    assert selected == "actin_fixed_multi_snr"


def test_resolve_partial_name_prompts_for_ambiguous_partial_match():
    stdout = io.StringIO()
    stderr = io.StringIO()

    selected = resolve_partial_name(
        "actin",
        ["actin_denoising", "actin_upsampling"],
        entity_name="promoted model",
        column_name="model",
        help_hint="List models.",
        stdin=InteractiveInput("02\n"),
        stdout=stdout,
        stderr=stderr,
    )

    assert selected == "actin_upsampling"
    assert "Multiple matching promoted models found:" in stdout.getvalue()
    assert "#   model" in stdout.getvalue()
    assert stderr.getvalue() == ""


def test_resolve_partial_name_does_not_accept_unique_partial_noninteractively():
    stdout = io.StringIO()
    stderr = io.StringIO()

    selected = resolve_partial_name(
        "actin",
        ["actin_model", "vimentin_model"],
        entity_name="promoted model",
        column_name="model",
        help_hint="List models.",
        stdin=NonInteractiveInput("y\n"),
        stdout=stdout,
        stderr=stderr,
    )

    assert selected is None
    assert stdout.getvalue() == ""
    assert "List models." in stderr.getvalue()


def test_resolve_partial_name_reports_no_match_with_hint():
    stdout = io.StringIO()
    stderr = io.StringIO()

    selected = resolve_partial_name(
        "missing",
        ["actin_model"],
        entity_name="promoted model",
        column_name="model",
        help_hint="List models.",
        stdout=stdout,
        stderr=stderr,
    )

    assert selected is None
    assert stdout.getvalue() == ""
    assert "No matching promoted model found for 'missing'." in stderr.getvalue()
    assert "List models." in stderr.getvalue()

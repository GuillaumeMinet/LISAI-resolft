from __future__ import annotations

import io

from lisai.data.selection import resolve_dataset_name


class InteractiveInput(io.StringIO):
    def isatty(self) -> bool:
        return True


class NonInteractiveInput(io.StringIO):
    def isatty(self) -> bool:
        return False


def test_resolve_dataset_name_accepts_unique_partial_match():
    selected = resolve_dataset_name(
        "actin_fixed",
        ["actin_fixed_multi_snr", "vimentin_live"],
        stdout=io.StringIO(),
        stderr=io.StringIO(),
    )

    assert selected == "actin_fixed_multi_snr"


def test_resolve_dataset_name_prefers_exact_match():
    selected = resolve_dataset_name(
        "actin_fixed",
        ["actin_fixed_multi_snr", "actin_fixed"],
        stdout=io.StringIO(),
        stderr=io.StringIO(),
    )

    assert selected == "actin_fixed"


def test_resolve_dataset_name_prompts_for_ambiguous_partial_match():
    stdout = io.StringIO()
    stderr = io.StringIO()

    selected = resolve_dataset_name(
        "actin_fixed",
        ["actin_fixed_multi_snr", "actin_fixed_pair"],
        stdin=InteractiveInput("02\n"),
        stdout=stdout,
        stderr=stderr,
    )

    assert selected == "actin_fixed_pair"
    assert "Multiple matching datasets found:" in stdout.getvalue()
    assert "actin_fixed_multi_snr" in stdout.getvalue()
    assert "actin_fixed_pair" in stdout.getvalue()
    assert stderr.getvalue() == ""


def test_resolve_dataset_name_noninteractive_ambiguity_returns_none():
    stdout = io.StringIO()
    stderr = io.StringIO()

    selected = resolve_dataset_name(
        "actin_fixed",
        ["actin_fixed_multi_snr", "actin_fixed_pair"],
        stdin=NonInteractiveInput(""),
        stdout=stdout,
        stderr=stderr,
    )

    assert selected is None
    assert "Multiple matching datasets found:" in stdout.getvalue()
    assert "lisai datasets list" in stderr.getvalue()

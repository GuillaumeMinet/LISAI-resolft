from __future__ import annotations

import io

from lisai.infra.cli.prompts import is_interactive, prompt_yes_no


class InteractiveInput(io.StringIO):
    def isatty(self) -> bool:
        return True


class NonInteractiveInput(io.StringIO):
    def isatty(self) -> bool:
        return False


def test_is_interactive_uses_stream_isatty():
    assert is_interactive(InteractiveInput()) is True
    assert is_interactive(NonInteractiveInput()) is False


def test_prompt_yes_no_reads_explicit_streams():
    stdout = io.StringIO()

    result = prompt_yes_no(
        "Proceed? [y/N] ",
        stdin=InteractiveInput("yes\n"),
        stdout=stdout,
    )

    assert result is True
    assert stdout.getvalue() == "Proceed? [y/N] "


def test_prompt_yes_no_can_require_interactive_input():
    stdout = io.StringIO()

    result = prompt_yes_no(
        "Proceed? [y/N] ",
        stdin=NonInteractiveInput("yes\n"),
        stdout=stdout,
        require_interactive=True,
    )

    assert result is None
    assert stdout.getvalue() == ""


def test_prompt_yes_no_supports_injected_input_function():
    assert prompt_yes_no("Proceed? ", input_fn=lambda _prompt: "Y") is True
    assert prompt_yes_no("Proceed? ", input_fn=lambda _prompt: "no") is False

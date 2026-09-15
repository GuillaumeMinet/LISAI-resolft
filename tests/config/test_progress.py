from __future__ import annotations

from types import SimpleNamespace

from lisai.config.progress import resolve_progress_bar


def test_resolve_progress_bar_prefers_cli_override():
    assert resolve_progress_bar(True, False, stg=SimpleNamespace(LOCAL_PROGRESS_BAR=True)) is False
    assert resolve_progress_bar(False, True, stg=SimpleNamespace(LOCAL_PROGRESS_BAR=False)) is True


def test_resolve_progress_bar_uses_local_config_when_cli_is_absent():
    assert resolve_progress_bar(True, stg=SimpleNamespace(LOCAL_PROGRESS_BAR=False)) is False
    assert resolve_progress_bar(False, stg=SimpleNamespace(LOCAL_PROGRESS_BAR=True)) is True


def test_resolve_progress_bar_falls_back_to_command_default_when_local_is_unset():
    assert resolve_progress_bar(True, stg=SimpleNamespace(LOCAL_PROGRESS_BAR=None)) is True
    assert resolve_progress_bar(False, stg=SimpleNamespace(LOCAL_PROGRESS_BAR=None)) is False

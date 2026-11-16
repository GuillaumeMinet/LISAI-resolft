"""Publication graph helpers must not rediscover a different workspace."""
from __future__ import annotations

from types import SimpleNamespace

from graphs.utils import run_selection
from lisai.config.settings import Settings
from lisai.config.workspace import initialize_workspace
from lisai.infra.paths import Paths


def test_run_discovery_receives_explicit_workspace_context(tmp_path, monkeypatch):
    a, b = tmp_path / "A", tmp_path / "B"
    initialize_workspace(a)
    initialize_workspace(b)
    seen = []

    def scan(*, paths):
        seen.append(paths.workspace_root())
        return SimpleNamespace(runs=())

    monkeypatch.setattr(run_selection, "scan_runs", scan)
    for root in (a, b, a):
        paths = Paths(Settings.from_workspace(root))
        assert run_selection.discover_graph_runs(dataset="demo", model_subfolder="net", paths=paths) == []
    assert seen == [a, b, a]

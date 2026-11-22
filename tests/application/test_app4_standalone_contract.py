"""Qt-free APP-4 boundaries and CLI help remain available in base LISAI."""
from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path


SOURCE = Path(__file__).resolve().parents[2] / "src" / "lisai"


def test_app_namespace_is_inert_without_qt():
    command = [sys.executable, "-c", (
        "import lisai; import lisai.application; import lisai.app; "
        "import sys; assert 'qtpy' not in sys.modules; "
        "assert 'pyqtgraph' not in sys.modules"
    )]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=30,
                               env={**os.environ, "PYTHONPATH": str(SOURCE.parent)})
    assert completed.returncode == 0, completed.stderr


def test_app_help_without_qt_binding():
    command = [sys.executable, "-m", "lisai.app", "--help"]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=30,
                               env={**os.environ, "PYTHONPATH": str(SOURCE.parent)})
    assert completed.returncode == 0, completed.stderr
    assert "--workspace" in completed.stdout
    assert "--config-root" in completed.stdout


def test_app_and_reusable_worker_have_no_improcess_or_engine_duplication():
    for name in ("app/main_window.py", "app/inference_page.py",
                 "qt/inference/worker.py"):
        path = SOURCE / name
        module = ast.parse(path.read_text())
        imported = [node.module for node in ast.walk(module)
                    if isinstance(node, ast.ImportFrom) and node.module]
        assert all(not v.startswith("imswitch") for v in imported)
        if name == "qt/inference/worker.py":
            assert all(v != "lisai.app" and not v.startswith("lisai.app.") for v in imported)


def test_training_workspace_uses_explicit_parent_keyword():
    """Avoid passing the QTabWidget as the optional recordings argument."""
    module = ast.parse((SOURCE / "app/main_window.py").read_text())
    calls = [
        node for node in ast.walk(module)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "TrainingWorkspaceWidget"
    ]
    # Both initial creation and the workspace-change path must use parent=...
    assert len(calls) == 2
    assert all(len(call.args) == 1 for call in calls)
    assert all(any(kw.arg == "parent" for kw in call.keywords) for call in calls)

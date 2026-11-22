"""Headless architectural constraints for the optional APP-2 widgets."""
from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_base_and_inference_widget_namespace_import_without_qt():
    code = (
        "import sys; import lisai; import lisai.application; "
        "import lisai.qt.inference; "
        "assert 'qtpy' not in sys.modules; "
        "assert 'pyqtgraph' not in sys.modules"
    )
    subprocess.run(
        [sys.executable, "-c", code], check=True, cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
    )


def test_reusable_qt_files_do_not_import_hosts_or_create_second_inference_engine():
    paths = [
        ROOT / "src/lisai/qt/inference/controls.py",
        ROOT / "src/lisai/qt/model_manager.py",
    ]
    for path in paths:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.append(node.module)
        assert not any(
            name == "lisai.app" or name.startswith(("lisai.app.", "imswitch", "pyqtgraph"))
            for name in imports
        )
        assert "def predict(" not in source
        assert "ProcessingResult" not in source

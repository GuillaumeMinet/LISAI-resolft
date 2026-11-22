"""APP-2 smoke launcher: reusable inference controls and full-page model manager.

Usage: python scripts/lisai_inference_qt_demo.py E:/lisai
This does NOT execute prediction; image comparison is APP-3.
"""
from __future__ import annotations

import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        print("Usage: python scripts/lisai_inference_qt_demo.py WORKSPACE", file=sys.stderr)
        return 2
    try:
        from qtpy import QtWidgets
    except Exception as exc:
        print(f"Qt is unavailable: {exc}. Install LISAI[qt] and a Qt binding.", file=sys.stderr)
        return 2

    from lisai.config.settings import Settings
    from lisai.qt.inference import InferenceControlsWidget
    from lisai.qt.model_manager import ModelManagerWidget

    settings = Settings.from_workspace(Path(args[0]).expanduser().resolve())
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = QtWidgets.QWidget()
    window.setWindowTitle("LISAI — Reusable inference controls (APP-2 demo)")
    window.resize(820, 640)
    layout = QtWidgets.QVBoxLayout(window)
    pages = QtWidgets.QStackedWidget(window)
    layout.addWidget(pages)

    controls_page = QtWidgets.QWidget(pages)
    form = QtWidgets.QVBoxLayout(controls_page)
    controls = InferenceControlsWidget(settings, controls_page)
    form.addWidget(controls)
    preview = QtWidgets.QPushButton("Validate selection / show explicit overrides", controls_page)
    form.addWidget(preview)
    result = QtWidgets.QPlainTextEdit(controls_page)
    result.setReadOnly(True)
    form.addWidget(result)
    pages.addWidget(controls_page)

    manager = ModelManagerWidget(settings, pages)
    pages.addWidget(manager)
    controls.manageModelsRequested.connect(lambda: pages.setCurrentWidget(manager))
    manager.backRequested.connect(lambda: pages.setCurrentWidget(controls_page))
    manager.modelsChanged.connect(controls.refresh_catalog)
    manager.modelInstalled.connect(lambda name: controls.select_model(name))

    def show_selection():
        try:
            selection = controls.inference_selection()
            result.setPlainText(
                f"Source: {selection.source}\nModel/run: {selection.identifier}\n"
                f"Explicit overrides: {selection.overrides.model_dump(exclude_unset=True)}"
            )
        except Exception as exc:
            result.setPlainText(f"Selection is not ready: {exc}")

    preview.clicked.connect(show_selection)
    window.show()
    return app.exec() if hasattr(app, "exec") else app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())

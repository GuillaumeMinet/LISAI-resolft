"""Launch LISAI's optional standalone Qt GUI with an explicit workspace."""
from __future__ import annotations

import argparse
import sys


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="LISAI standalone training and inference GUI")
    parser.add_argument("--workspace", help="Initialized LISAI workspace directory")
    parser.add_argument("--config-root", help="Optional explicit experiment config directory")
    args = parser.parse_args(argv)
    try:
        from qtpy import QtWidgets
    except (ImportError, RuntimeError) as exc:
        parser.exit(2, "LISAI Qt requires a Qt binding (e.g. PySide6), "
                    "qtpy and pyqtgraph. Install lisai[qt] and a binding.\n"
                    f"Details: {exc}\n")

    from .main_window import LisaiMainWindow, WorkspaceSelectionDialog, load_app_settings

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
    if args.workspace:
        try:
            settings = load_app_settings(args.workspace, config_root=args.config_root)
        except Exception as exc:
            parser.exit(2, f"Invalid LISAI workspace: {exc}\n")
    else:
        chooser = WorkspaceSelectionDialog(config_root=args.config_root)
        if chooser.exec() != QtWidgets.QDialog.Accepted:
            return 0
        settings = chooser.settings
    window = LisaiMainWindow(settings, config_root=args.config_root)
    window.show()
    return int(app.exec())


if __name__ == "__main__":
    raise SystemExit(main())

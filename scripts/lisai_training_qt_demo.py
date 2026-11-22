"""Small ImProcess-free smoke launcher for the reusable LISAI Training workspace.

Requires lisai[qt] plus a supported Qt binding, in an existing full workspace.
"""
from __future__ import annotations

import argparse


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workspace", help="Path to a previously initialized LISAI workspace")
    args = parser.parse_args()

    # Explicit imports after CLI parsing: base `lisai` never requires Qt.
    from qtpy import QtWidgets
    from lisai.config.settings import Settings
    from lisai.qt.training_panel import TrainingDialog

    settings = Settings.from_workspace(args.workspace)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    dialog = TrainingDialog(settings)
    dialog.show()
    return app.exec() if hasattr(app, "exec") else app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())

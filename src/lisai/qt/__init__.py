"""Optional, host-independent LISAI Qt components.

Import specific submodules when needed; ``import lisai`` and
``import lisai.application`` never load Qt or a Qt binding. Host applications
must provide qtpy and a supported Qt binding (PySide6/PyQt5/etc.).
"""

__all__ = ["config_editor", "training_session", "training_panel", "quick_hdn", "training_data", "config_sources"]
# R4-B: TrainingDataWidget and TrainingConfigSourceWidget live in dedicated
# submodules so they can be embedded by a future standalone LISAI shell.

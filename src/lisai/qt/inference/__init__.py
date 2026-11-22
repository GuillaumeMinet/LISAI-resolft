"""Reusable inference Qt controls; no viewer or standalone-host dependencies.

Importing this package does not import Qt until a widget is requested.
"""

__all__ = ["InferenceControlsWidget"]


def __getattr__(name):
    if name == "InferenceControlsWidget":
        from .controls import InferenceControlsWidget

        return InferenceControlsWidget
    raise AttributeError(name)

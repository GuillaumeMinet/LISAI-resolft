"""Reusable inference Qt components (optional, lazily imported).

No image files, model engines, or Qt bindings are imported at package import time.
"""

__all__ = ["InferenceControlsWidget", "ImageViewer2D", "ImageComparisonWidget"]


def __getattr__(name):
    if name == "InferenceControlsWidget":
        from .controls import InferenceControlsWidget

        return InferenceControlsWidget
    if name == "ImageViewer2D":
        from .image_viewer import ImageViewer2D

        return ImageViewer2D
    if name == "ImageComparisonWidget":
        from .comparison import ImageComparisonWidget

        return ImageComparisonWidget
    raise AttributeError(name)

# APP-3 — Reusable YX/TYX image comparison

APP-3 adds two **host-neutral, reusable Qt widgets** over pyqtgraph. It does not
load TIFFs, select models, run inference, or create the standalone app. APP-1
owns TIFF I/O, APP-2 owns model/parameter controls, and APP-4 will compose
these widgets into the standalone Inference tab. **The ImProcess plugin remains
unchanged.**

## Public widgets

```python
from lisai.qt.inference import ImageComparisonWidget, ImageViewer2D
from lisai.evaluation.inference.results import PredictionInfo

comparison = ImageComparisonWidget()
comparison.set_input(raw_tyx, axes="TYX")
comparison.set_prediction(prediction_tyx, info=prediction_info)
comparison.set_frame_index(5)  # original input frame index, 0-based
comparison.invalidate_prediction("Model or options changed")

viewer = ImageViewer2D()
viewer.set_image(image_yx)
viewer.set_levels(0.0, 500.0)  # changes display contrast only
viewer.reset_view()
```

Both components are `QWidget`s and can be embedded by LISAI or ImProcess without
importing `lisai.app`. `ImageViewer2D` accepts **YX** grayscale arrays only; the
comparison accepts **YX** or **TYX** and never transposes the source data. Each
viewer has independent pan/zoom, auto contrast (robust sampled percentiles),
manual intensity levels, and Fit. Display operations do not modify underlying
NumPy pixel values. A frame switch displays only the selected YX image rather
than preloading the entire stack as Qt textures.

The comparison slider is **always indexed by original input frame**, not by the
prediction array. `PredictionInfo.source_frame_indices` provides the mapping.
Missing `drop` edges explicitly show *No prediction for this input frame*;
`black_output_indices` labels the actual black output placeholders, rather than
inventing them or showing a neighboring frame. The mapping validates duplicates,
lengths, bounds, and integer indices. The small `alignment.py` helper is
Qt-free so these correctness rules can be tested headlessly.

Input replacement clears the old prediction. A **host must also call**
`invalidate_prediction()` when model, checkpoint, normalization, other overrides,
or workspace changes. APP-4 will connect those signals and must prevent Save
until the displayed result belongs to the current prediction request.

Zoom is **not synchronized**: the current inference metadata does not generally
establish a spatial transform between input and prediction. Supplying optional
pixel spacing applies that explicit Y/X scale independently within one viewer (`spatial_scale` is the **output pixel spacing**, not an inferred upsampling factor);
no calibration, physical scale bar, cropping transform, or registration is
invented from array dimensions. Z/channel selection remains outside APP-3.

## Try the image viewer without a model

```powershell
pip install -e ".[qt]"
pip install PySide6
python scripts/lisai_image_comparison_qt_demo.py
```

The demo uses synthetic noisy TYX recordings and has buttons to switch between
dropped and black-output edges. It requires neither a workspace nor ImProcess.

Run offscreen tests with a compatible Qt binding:

```powershell
$env:QT_QPA_PLATFORM = 'offscreen'
python -m pytest tests/application/test_app3_alignment.py tests/qt/test_app3_comparison.py
```

Without Qt/pyqtgraph, the alignment tests run and GUI tests skip. `import lisai`,
`import lisai.application`, and `import lisai.qt.inference` remain Qt-free; only
requesting the actual widgets loads optional graphical dependencies.

## Deliberately deferred to APP-4

TIFF browsing, inference workers, prediction/file provenance, Save As, result
invalidation tied to selected model/options, and the two-tab main application.
Test visual orientation and mouse responsiveness on Windows, including a large
real-world microscopy stack. No ImProcess plugin changes belong to APP-3.

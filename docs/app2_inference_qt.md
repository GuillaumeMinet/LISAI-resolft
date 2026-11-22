# APP-2 — Host-independent inference Qt controls and model browser

This milestone adds reusable GUI widgets over the APP-1 host-independent services.
It does **not** implement an image viewer, model execution, or a standalone app
shell; those are APP-3/APP-4. **The ImProcess plugin is unchanged.**

## Components

- `lisai.qt.inference.InferenceControlsWidget(settings)` is an embeddable `QWidget`.
  It selects an installed model or completed training run, checkpoint and
  optional overrides for tiling, cropping, normalization and temporal edges.
  It uses `resolve_inference_selection()` to produce the typed `InferenceSelection`.
  The controls have no Open/Save/Run buttons and don't import LISAI prediction
  engines, pyqtgraph or ImProcess.
- `lisai.qt.model_manager.ModelManagerWidget(settings)` is a full-width,
  embeddable page for local installed models and the remote catalog. A local
  model list and card remain accessible when the online catalog is offline.
  Catalog and download/install actions run in `QThread`, preserving download
  progress, cancellation and the transactional install backend. Jobs stay alive
  when their page is hidden; duplicate downloads into the same workspace are
  rejected. Installation cannot be cancelled once started.

Widgets use **explicit** workspace `Settings`. Switching workspace clears model
identity, rather than reusing a name accidentally. Refresh preserves an old
selection even if it disappears and marks it *unavailable*, never choosing a
different model silently. `inference_selection()` revalidates that an installed
model/run and requested checkpoint still exist before yielding options.

No saved-model defaults are applied by the widget. In particular, the initial
choice *Model settings* means **do not override**, `tiling=auto` means LISAI's
existing saved/default policy and not a GPU-memory heuristic, and manual gain
and offset are forwarded only in manual-normalization mode.

## Embedding

```python
from lisai.qt.inference import InferenceControlsWidget
from lisai.qt.model_manager import ModelManagerWidget

controls = InferenceControlsWidget(settings)
manager = ModelManagerWidget(settings)
controls.manageModelsRequested.connect(show_model_manager_page)
manager.modelsChanged.connect(controls.refresh_catalog)
manager.modelInstalled.connect(controls.select_model)

selection = controls.inference_selection()
# Later, the host calls lisai.predict(selection.model, image,
#                                    settings=settings,
#                                    overrides=selection.overrides)
```

`InferenceControlsWidget` provides `selectionChanged`, `optionsChanged`,
`manageModelsRequested`, `refresh_catalog()`, `set_settings()`, `current_values()`
(alias `get_values()`), `set_values()`, and `inference_selection()`.
The optional UI-value adapter uses R4-4 field names, but intentionally does
**not** mutate ImProcess settings, `params_version` or plugin serialization.

`ModelManagerWidget` exposes `modelInstalled(str)`, `modelsChanged()` and
`backRequested()` so a host can embed it in `QStackedWidget` without opening a
second dialog. `set_settings()` refuses to change workspace while a network or
installation job is in progress.

## Standalone widget smoke launcher

With a Qt binding and the optional LISAI Qt dependencies installed:

```powershell
pip install -e ".[qt]"
pip install PySide6
python scripts/lisai_inference_qt_demo.py E:/lisai
```

The demo shows the selector, validates selections, opens the model-manager page
inside the same window, and returns. It never executes inference.

Tests in `tests/qt/test_app2_inference_controls.py` can run with
`QT_QPA_PLATFORM=offscreen` on a machine with a compatible Qt binding. Base
LISAI imports and `lisai.application` remain usable without Qt installed.

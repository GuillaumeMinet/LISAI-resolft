# APP-4 — Standalone LISAI Qt host

Launch the **optional GUI** from an installed or editable LISAI package:

```powershell
pip install -e ".[qt]"
pip install PySide6
python -m lisai.app --workspace E:/lisai
```

Without `--workspace`, a workspace chooser appears. `--config-root PATH` is an
optional **explicit** config-root override, not implicit developer CLI discovery.
The GUI displays the workspace and shows its effective experiment config root
in the workspace label tooltip. It never sets `LISAI_HOME` or writes CLI selection.
For the normal default layout, workspaces must have been initialized using
`lisai workspace init PATH`. The chooser can also **explicitly** create a full
workspace in an empty directory, after confirmation.

## Two application tabs

**Training** directly embeds the existing R4 `TrainingWorkspaceWidget` with Data,
Configuration, Training and internal Noise Models tools. Switching main tabs
does not close the training subprocess, invalidate an unsaved YAML or restart it.

**Inference** uses APP-2's `InferenceControlsWidget` and `ModelManagerWidget`,
APP-3's `ImageComparisonWidget`, and APP-1's TIFF I/O. Open one **YX** or
explicitly labeled **TYX** TIFF, choose an installed model or completed run,
then Run. The app calls public `lisai.predict(model, array, return_info=True)`
**once** in a Qt worker thread, displays the result on the right, and optionally
saves the in-memory prediction to an atomic TIFF with alignment/provenance
metadata. The raw frame slider indexes **input frames**; missing multi-frame
edge predictions remain empty, not shifted to adjacent timepoints.

Model management is an internal full-page view within the same Inference tab,
not an additional dialog. On return, the TIFF and current model options remain.

## Component ownership

- `lisai.qt.inference.controls`: model/run/options selection, **no viewer**
- `lisai.qt.inference.comparison`: image display, **no models or files**
- `lisai.qt.inference.worker`: QThread calling public `lisai.predict()`
- `lisai.app.inference_page`: standalone Open / Run / Save composition
- `lisai.app.main_window`: optional workspace chooser and two main tabs
- `lisai.qt.training.workspace`: existing embeddable R4 training workflow
- `lisai.application`: validation/catalog/TIFF I/O, remains Qt-independent

The ImProcess plugin stays at R4-4; it will eventually reuse the inference
controls but **not** the standalone file browser/viewer/executor.

## Current limits

- Only unambiguous **YX** and **TYX** TIFF files; no Z/channel guessing.
- Separate pan/zoom and contrast controls; no unverified physical registration.
- No inference cancellation or percentage progress, no GPU model cache/retry.
- App closing/workspace changes are blocked during inference/model management;
  training workspace changes are blocked while training. Closing during training
  requires explicit confirmation and terminates the local subprocess.
- Interactive UI testing requires Windows/Qt; headless tests are skipped without
  a Qt binding. A real promoted HDN and multi-frame model should be smoke-tested
  before treating this as a production image-analysis application.

## APP-4 implementation notes / deviations

- The existing `TrainingWorkspaceWidget` is embedded **without copying** its
  Qt forms; a future ImProcess migration still uses the host-specific dialog.
- The APP-3 comparison widget remains independently embeddable; only the app
  composition owns TIFF dialogs and output paths.
- Model Manager is embedded as another page of the **Inference tab**, not
  another top-level window. Navigating to/from it does not invalidate a valid
  prediction; an actual model/option/catalog change still does.
- A single `InferenceWorker` QThread runs the public NumPy `predict` API.
  The app does not provide an unimplemented Cancel button, silently retry or
  cache model weights across runs.
- Workspace switching reconstructs the training widget and explicitly resets
  inference input/model state. It is not allowed while a training, inference or
  model-download operation is active; unsaved YAML must be handled first.
- The model manager and simple inference controls remain standalone-reusable
  widgets and import neither `lisai.app` nor ImProcess.

**Validation at implementation:** Qt-free application tests, APP-1/APP-3
regressions and broader application/config/inference/Qt-contract tests were
run in the available environment. Offscreen Qt tests exist but cannot execute
without a Qt binding (PySide6/PyQt); the final Windows GUI and real-model smoke
checks remain necessary.

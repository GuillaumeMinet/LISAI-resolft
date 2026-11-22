# LISAI Qt prototype (R3)

`lisai.qt` is an **optional**, host-independent interface over the existing
LISAI backend. The R3 prototype takes ownership of the full training dialog
that previously lived in the ImProcess `lisai_restore.py` plugin. It builds on
R1 (`lisai.application`) and R2 (schema-aware editor / training session).

## Installation

- `pip install -e '.[qt]'` installs `qtpy` and `ruamel.yaml`.
- A Qt binding (PySide6/PyQt5, etc.) must be present in the host environment.
- For training, the existing LISAI `[full]` extra is still required.
- Base `import lisai` and `import lisai.application` never import Qt.

## Embeddable training panel

```python
from lisai.config.settings import Settings
from lisai.qt.training_panel import (
    TrainingPanel, TrainingRecording, show_training_dialog,
)

settings = Settings.from_workspace('E:/lisai')
# With a QApplication already running:
dialog = show_training_dialog(settings)  # TIFF selection works without a host
# Or embed a QWidget: panel = TrainingPanel(settings, recordings=[...])
```

Host-provided recordings use the small `TrainingRecording` contract: a display
name, NumPy array, explicit `('Y','X')` or `('T','Y','X')` axes and a default
checkbox selection. A host needs only to convert its own image result to this
record; the Qt components do not import ImProcess.

The panel offers two paths:

1. **Quick HDN**: select supplied recordings and/or TIFF files, or reuse a
   registered unprepared dataset; set noise model, betaKL, patches, epochs and
   split; prepare a YAML through the existing LISAI `quick_hdn` APIs.
2. **Existing YAML**: select any writable, validated LISAI training YAML,
   including non-LVAE denoising or upsampling configurations. This path uses
   the normal LISAI runner and does not require Quick HDN data/split rules.
   No guided dataset preparation is supplied for those tasks yet.

Both paths use the same embedded **Configuration** view, with task-aware
scientific controls and a complete **Raw YAML** tab. Guided changes retain
comments and unrelated fields through `ruamel.yaml`. Unsaved edits block
launch; Save/Save As perform authoring validation and the final launch performs
full LISAI training validation. Prepared
Quick HDN files may be edited and saved without reimporting the dataset.

`TrainingSession` owns one QProcess per QApplication; closing the modeless
window hides it but does not stop training. The reusable monitor shows logs,
Stop/Force Stop, and the run folder. The process executes
`lisai_runner.training_entrypoint` with explicit workspace/config-root paths.
The ImProcess plugin owns only workspace preferences, available result
conversion, and completion notifications.

## Current limits

- Only unsupervised Quick HDN has a guided *preparation* wizard; other tasks
  use authored YAML. The editor is intentionally not a general JSON-Schema
  renderer for arbitrary nested objects.
- GUI preflight refuses destructive `overwrite: true` for fresh/retrain jobs,
  but permits repeated new training with the same YAML and experiment name.
  Continue training must be explicitly confirmed.
- The model manager remains ImProcess-owned; it has not been moved to Qt.
- A real Windows/Qt interaction test is still required. Automated tests cover
  application/runner contracts, and optional Qt smoke tests skip when no
  Qt binding is installed.

## R4-1: shared configuration document and reusable runs

The existing R3 dialog/layout remains in place. Its two configuration views now
share a single Qt-independent `TrainingConfigDocument` (`lisai.application`):

- `TrainingConfigDocument.open(settings, path)` reads a user YAML and records its
  on-disk revision. `from_source(settings, "templates/base_training")` and other
  packaged sources create **unsaved** drafts; built-in YAMLs are never writable.
- Guided field edits and the complete Raw YAML editor modify the same document
  buffer. Invalid Raw YAML remains editable, and guided controls are disabled
  until syntax can be parsed again.
- **Save As** creates a new user file exclusively. **Save** updates an existing
  writable file, detecting outside modifications and declining to overwrite
  them. This uses the established application YAML persistence helper, with
  round-trip editing to preserve comments.
- **Authoring validation** permits incomplete templates (with warnings about
  remaining placeholders); **training validation** requires complete, resolved
  settings. Launch always revalidates saved YAML. Unfinished drafts can be saved
  but cannot be launched.
- Creating or rerunning a fresh training job never reserves or rejects an
  unnumbered directory at preflight. LISAI's existing `create_run_dir()` creates
  a new numbered folder at runtime; `plan.run_dir` is `None` until the actual
  run is discovered. Continue mode still targets an explicit existing run;
  destructive `overwrite: true` remains blocked for new GUI launches.
- `quick_hdn_training_document(...)` prepares the existing scientifically
  validated Quick HDN configuration **in memory**. The legacy
  `prepare_quick_hdn_training(..., write=True)` remains exclusive-create for
  compatibility; `write=False` can preview even when previous configs/runs
  exist. The existing R3 GUI still uses its original Quick HDN preparation
  path; switching the full workflow to in-memory drafts is R4-B/C work.

The GUI layout, Quick HDN-only guided setup, ImProcess integration, and process
monitor are intentionally unchanged in this first R4 increment. No `lisai.qt`
module is imported by `lisai.application`.

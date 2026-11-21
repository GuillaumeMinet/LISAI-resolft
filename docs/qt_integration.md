# LISAI Qt first pass (R2)

This is an intentionally small, host-independent first extraction from the
ImProcess `lisai_restore.py` plugin. It builds on the R1 `lisai.application`
services; it is **not** a general schema form renderer or a standalone app.

## Installation

`pip install -e '.[qt]'` adds `qtpy` and `ruamel.yaml`. A **Qt binding** (for
example PySide6) must already be provided by the host environment or installed
separately. `pip install lisai` and `import lisai` do not load Qt. To train,
LISAI's existing `[full]` extra is still needed.

## Reusable entry points

```python
from lisai.config.settings import Settings
from lisai.qt.config_editor import TrainingConfigEditorWidget, TrainingConfigEditorDialog
from lisai.qt.training_session import get_training_session
from lisai.qt.training_panel import TrainingMonitorWidget
from lisai.qt.quick_hdn import QuickHDNParametersWidget

settings = Settings.from_workspace('E:/lisai')
editor = TrainingConfigEditorWidget(settings)  # embeddable QWidget
# Or: dialog = TrainingConfigEditorDialog(settings)

session = get_training_session()  # needs QApplication
monitor = TrainingMonitorWidget(session)  # generic QProcess log + Stop controls
quick_hdn_fields = QuickHDNParametersWidget(['my_noise_model'])
```

The editor provides a small curated **Experiment parameters** page from the
existing Pydantic JSON Schema (HDN `betaKL`/`supervised`, single-frame sampling
ratio, temporal window, dataset and optimization parameters). The complete
**Raw YAML** tab remains editable for every field. Both views edit the same text;
guided changes use `ruamel.yaml` to preserve YAML comments, key order and
unexposed fields. Validation/Save/Save As use `lisai.application` rather than a
second schema/resolver. Existing YAMLs are not rewritten unless saved.

Task changes and complex nested parameter combinations are still edited in Raw
YAML; the first version does **not** promise a self-consistent automatic migration
between network architectures. The Quick HDN form remains *unsupervised-only*;
supervised HDN and the other tasks can be configured by YAML and launched via
the existing saved-YAML workflow when their input data is correctly prepared.

`TrainingSession` is application-owned (per QApplication), survives hiding its
window, invokes the existing LISAI runner subprocess, and uses LISAI application
run discovery. `TrainingMonitorWidget` is reusable and does not own the worker.
The Quick HDN parameter widget owns task-specific controls (βKL, noise model,
patching, optimization), while ImProcess still handles selecting/normalizing its
own `ProcessingResult` recordings and owns its own workspace preference.

## Current limitations

- No generalized guided training generation except unsupervised Quick HDN.
- No Qt model-manager migration yet; it remains in ImProcess.
- No UI test/visual validation on the target Windows/Qt environment in this
  first delivery. The package and plugin can be imported without Qt, and
  core/application/schema tests are exercised.

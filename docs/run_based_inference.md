# Predict directly from a saved training run (L3)

`predict()` supports two **distinct** model sources:

- `predict("installed_model_name", ...)`: an installed **promoted** model (unchanged).
- `predict(RunRef(run_id="..."), ...)`: a saved training run in the **explicitly selected workspace** (new).

The second form does not promote, copy, export, or install the training run. It loads the run's saved `config_train.yaml` snapshot, checkpoint, and (for HDN/LVAE) the original workspace noise model.

```python
from lisai import ApplyOverrides, RunRef, predict
from lisai.config.settings import Settings

settings = Settings.from_workspace("D:/LISAI-workspace")
run = RunRef(run_id="01ARZ3NDEKTSV4RRFFQ69G5FAV")  # See `lisai runs list`

# Image is a NumPy YX, TYX or TZYX array. No intermediate TIFF is written.
restored, info = predict(run, image, settings=settings, return_info=True)

# Load the last checkpoint rather than the default best checkpoint.
restored_last = predict(
    run, image, settings=settings,
    overrides=ApplyOverrides(checkpoint={"best_or_last": "last"}),
)

# The same source also accepts a TIFF or directory and returns PredictionResult.
result = predict(run, "D:/images/cell.tif", settings=settings,
                 output_dir="D:/images/restored")
```

The `run_id` is an exact stable ID from run metadata, not the experiment name or a folder name. A duplicate ID is an error, never an interactive prompt. The selected run must have a readable saved configuration, at least one requested checkpoint, and any needed noise-model files. Non-completed runs can still be used if they have a valid checkpoint; ImProcess can filter its picker to completed runs initially.

Run-based inference resolves the same workspace inference defaults and user overrides as the existing `lisai apply --run-id` CLI. Best/last checkpoint resolution and TIFF output policies are shared with that CLI. Changes to the original *editable training YAML* do not alter the saved run snapshot. Existing promoted-model calls remain unchanged.

# APP-1 — Host-independent inference services

This increment adds **only** `lisai.application` functions. There are no Qt widgets,
new inference engines, or changes to `lisai_restore.py`.

## Model/run selection

```python
from lisai.application import resolve_inference_selection

selection = resolve_inference_selection(
    settings=settings,                  # explicit Settings.from_workspace(...)
    source="installed",                # or "run"
    identifier="vim_hdn",              # or completed run ID
    tiling_size="model settings",       # no explicit override
    normalization_mode="manual",
    manual_gain=9.94,
    manual_offset=-7.35,
)

# selection.model is str or RunRef; selection.overrides is ApplyOverrides
prediction, info = predict(
    selection.model, raw_array, settings=settings,
    overrides=selection.overrides, return_info=True,
)
```

The selection checks the selected workspace's registered model or completed run,
and checks checkpoint availability for runs. It does **not** load weights. The
`ApplyOverrides` it returns is sparse: "Model settings" does not overwrite the
model's saved settings. Tiling `auto` retains LISAI's existing saved-policy
semantics (it does not estimate available GPU memory).

## TIFF loading and saving

```python
from lisai.application import (
    InferenceProvenance, load_inference_tiff, save_prediction_tiff,
)

raw, input_info = load_inference_tiff("raw.tif")
prediction, info = predict(
    selection.model, raw, settings=settings,
    overrides=selection.overrides, return_info=True,
)
saved = save_prediction_tiff(
    "prediction.tif", prediction, axes=input_info.axes,
    info=info,
    provenance=InferenceProvenance(
        source_path=input_info.path,
        model=selection.model,
        overrides=selection.overrides,
    ),
)
```

`load_inference_tiff` accepts only a single TIFF series with an **explicit YX or
TYX axis interpretation**. It refuses ZYX, channels, and ambiguous 3-D stacks
(`IYX`/`QYX`) instead of silently guessing that the first axis means time.
A future GUI may add an explicit user-confirmed axes choice.

`save_prediction_tiff` saves the **already-computed** image as float32. It never
calls a model. It embeds the source path, model/run ID, explicit overrides,
LISAI version, source-frame indices, and black-output indices in the TIFF's
JSON ImageDescription, along with its axes. No separate metadata sidecar is
needed. By default, existing targets are never overwritten; use
`overwrite=True` explicitly. The helper does not reproduce CLI output-directory
numbering, perform resampling, or fabricate physical pixel calibration.

The standalone viewer and ImProcess remain responsible for their respective
input-selection and output-presentation policies. This application module has
no Qt/ImProcess dependencies.

## Decisions specific to this implementation

- The single-file prediction helper embeds provenance **inside the TIFF**, to
  avoid a partially written TIFF/JSON pair.
- Saving uses a temporary file in the destination directory and an exclusive
  hard link for no-overwrite installation, so a concurrent file cannot be
  silently replaced. On filesystems that do not support hard links, saving
  reports the operating-system error instead of reverting to an unsafe write.
- Incomplete or ambiguous axis metadata is an error; the first standalone
  viewer will not infer TYX from an arbitrary 3D TIFF.
- No change to `predict()`, inference normalization, model bundles, the
  runner, workspace selection, or the ImProcess plugin is included.

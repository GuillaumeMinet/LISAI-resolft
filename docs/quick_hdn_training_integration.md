# Quick HDN training from Python (L1 + L2)

These helpers prepare **unsupervised HDN training from independent raw recordings** without creating separate preprocessed train/validation folders. They are designed for ImProcess and other Python integrations. Training execution remains in the existing LISAI runner.

Prerequisites: an initialized workspace, LISAI installed with `[full]` for
training, and an existing calibrated HDN GMM noise model (`GMM.npz` and
`norm_prm.json` in the workspace noise-model folder). To generate one from
paired images, see the [noise-model calibration guide](noise_model_calibration.md)
or the runnable, documented
[`scripts/generate_noise_model.py`](../scripts/generate_noise_model.py) example.
Use the saved model name as `noise_model_name` below.

```python
from lisai.config.settings import Settings
from lisai.data import import_training_dataset
from lisai.training.quick_hdn import prepare_quick_hdn_training

settings = Settings.from_workspace("E:/lisai")

# Each path or (name, ndarray) is ONE original recording. No frame splitting.
imported = import_training_dataset(
    ["D:/acquisitions/cell01.tif", "D:/acquisitions/cell02.tif"],
    settings=settings, dataset_name="quick_hdn_cells", data_format="timelapse",
)

arguments = dict(
    settings=settings, dataset_name=imported.dataset_name,
    noise_model_name="my_calibrated_noise",
    experiment_name="my_first_hdn", epochs=20, patch_size=64,
    seed=0,  # file-level split, default 80%/20%/0% train/val/test
)
preview = prepare_quick_hdn_training(**arguments)
print(preview.split_counts, preview.split_filenames)

# After user confirmation, save a new, non-overwriting experiment YAML:
ready = prepare_quick_hdn_training(**arguments, write=True)
print(ready.config_path)
```

To run in an independent Python process (as ImProcess will do):

```python
import subprocess
import sys

subprocess.run(
    [sys.executable, "-m", "lisai_runner.training_entrypoint",
     "--config", str(ready.config_path),
     "--workspace", str(settings.WORKSPACE_ROOT),
     "--config-root", str(settings.CONFIGS_ROOT)],
    check=True,
)
```

## Constraints and decisions

- A timelapse `TYX` is saved as **one TIFF file**. All of its frames belong to the **same** training, validation, or test split. HDN sees the frames individually *inside* the assigned split (`timelapse_prm.context_length: null`).
- At least **two distinct input files** are required (train and validation). Three are required if the optional test fraction is positive. A single 200-frame recording is insufficient.
- Default split ratios are 0.8 train / 0.2 validation / 0 test. The preview uses LISAI's existing split-manifest generator and reports **actual integer file counts** and exact filenames; only the training runtime writes the v1 manifest to its run directory.
- L1 accepts `YX` or `TYX` NumPy arrays, optional `(filename, array)` pairs, and TIFF file paths. Use a named pair when saving an ImProcess result to retain its source name. Raw numeric pixel values are not rescaled. Imported datasets cannot overwrite existing datasets.
- L2 requires the dataset to be registered for training and the selected noise model to have readable `GMM.npz` and `norm_prm.json` resources. It resolves the packaged unsupervised HDN preset and standard LISAI typed training config, without maintaining a second YAML resolver.
- `write=False` is a read-only preview. `write=True` writes a new configuration under `<selected-config-root>/training/improcess/<experiment>.yml`, refusing an existing name. It does not start training or create a run.
- Do **not** call L1 on a single recording in a first-version ImProcess quick-training flow: validate the number of recordings and the intended split *before import*, so invalid input does not leave an unused registered dataset.
- Training normalization currently incorporates train **and** validation statistics in the original loader. Changing this is a separate scientific-correctness fix, not part of L1/L2.

The ImProcess training dialog, process/log monitoring, and Windows/NVIDIA real-training validation are separate subsequent steps (I1/I2/I3).

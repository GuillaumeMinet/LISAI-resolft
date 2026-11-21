# Calibrate an HDN noise model from NumPy arrays

LISAI exposes `create_noise_model()` for applications (including ImProcess) that
already hold paired calibration images in memory. It fits LISAI's existing HDN
conditional-noise GMM and saves the same files as the dataset-based builder.
This is a **calibration** API, not unsupervised HDN training itself.

## Prerequisites and example

Initialize a LISAI workspace and install LISAI with its training/visualization
extras when you want the default plots (`pip install -e ".[full]"` from a
checkout). The fitter also works without Matplotlib or scikit-learn when
`save_gmm_plot=False`; HDN/GMM fitting itself uses NumPy and PyTorch.

```python
from pathlib import Path

import numpy as np
import tifffile

from lisai.config.settings import Settings
from lisai.noise_models import create_noise_model, NoiseModelOptions

settings = Settings.from_workspace("D:/lisai-workspace")

# Already-aligned pairs acquired under the same experimental conditions.
# Each folder contains the corresponding images in matching order.
signal_files = sorted(Path("D:/calibration/reference").glob("*.tif"))
noisy_files = sorted(Path("D:/calibration/noisy").glob("*.tif"))
assert signal_files and len(signal_files) == len(noisy_files)

signal = np.stack([tifffile.imread(p) for p in signal_files])
observation = np.stack([tifffile.imread(p) for p in noisy_files])
assert signal.shape == observation.shape  # verify pair identities as well

result = create_noise_model(
    signal=signal,
    observation=observation,
    name="monalisa_488_calibration",
    settings=settings,
    options=NoiseModelOptions(save_gmm_plot=True),
    device="auto",  # 'cpu', 'cuda' or 'auto'
)
print(result.name, result.save_dir)
print(result.gmm_path, result.norm_prm_path)
```

Alternatively, pass two `(Y, X)` arrays for one pair. `NYX` means *N aligned
pairs*, not a time axis from a moving sample; **do not** use an unrelated
noisy timelapse as the `observation` paired to a single reference image.
Both inputs must have identical shapes and finite real numeric values.
The API copies the input arrays rather than modifying them.

**Pairing and microscopy domain matter.** Reference (high-SNR) and noisy images
must depict the same stable structure, aligned pixel for pixel. The reference
should be as close to a noise-free signal as experimentally feasible; averaging
multiple reference exposures can be performed *before* calling this API.
Calibrate in the intensity domain on which HDN will train: for example, if
training on reconstructed MoNaLISA images, calibrate using reconstructed data,
not raw camera images. Check reconstruction scaling, detector gain, exposure,
laser settings and other acquisition factors before reusing a calibration.
Matching mean/std alone does not correct a changed noise distribution.

## Options and output

`NoiseModelOptions` retains the existing fitter defaults:

| Setting | Default | Meaning |
|---|---:|---|
| `norm_sig_to_obs` | `True` | Match reference statistics to observations |
| `normalize_everything` | `True` | Normalize both arrays for GMM fitting |
| `clip` | `-3.0` | Lower bound before normalization, or `None` |
| `crop_size` | `None` | Optional centered pixel crop (int or `(H, W)`) |
| `histogram_bins` | `256` | Bins in the empirical conditional histogram |
| `gmm_n_gaussian` | `5` | Number of mixture components |
| `gmm_n_coeff` | `4` | Polynomial coefficient count |
| `gmm_n_epochs` | `3000` | Optimizer epochs |
| `gmm_learning_rate` | `0.05` | Fitting learning rate |
| `gmm_batch_size` | `250000` | Fitting batch size |
| `gmm_min_sigma` | `1.0` | Minimum variance parameter used by the existing GMM |
| `save_gmm_plot` | `True` | Write diagnostic PNG and SVG (requires Matplotlib and full plotting stack) |
| `gmm_plot_points` | `15` | Signal values represented in diagnostics |
| `gmm_plot_max_columns` | `4` | Diagnostic subplot columns |

The result folder is resolved through the **explicitly supplied** `Settings`:

```text
<workspace>/noise_models/monalisa_488_calibration/
    GMM.npz
    norm_prm.json
    Histogram.npy
    info.txt
    GMM_plot.png   # if save_gmm_plot=True
    GMM_plot.svg   # if save_gmm_plot=True
```

The returned `NoiseModelResult` includes `name`, `save_dir`, artifact paths,
`n_pairs`, prepared array shapes, and fitted minimum/maximum signal values.
`name` is one directory component; slashes and path traversal are rejected.
A duplicate name raises `FileExistsError` by default. To replace an existing
calibration explicitly, pass `overwrite=True`.

Fitting and plotting run in a private staging directory. A failed operation
normally leaves the prior model intact, and successful replacement removes
obsolete artifacts from the old directory. Directory replacement is **not
fully crash-atomic**: a power loss between renames may leave a hidden sibling
`.NAME.backup-*` directory containing the previous model. Recover manually
only after verifying which model is complete, and never delete the backup
before confirming recovery.

## Use with quick unsupervised HDN training

The files are compatible with LISAI's normal HDN noise-model loader. After
calibrating, use the returned **name** with the quick-training helper:

```python
from lisai.training.quick_hdn import prepare_quick_hdn_training

preview = prepare_quick_hdn_training(
    settings=settings,
    dataset_name="my_registered_training_dataset",
    noise_model_name=result.name,
    experiment_name="hdn_first",
    epochs=20,
    patch_size=64,
)
print(preview.split_counts)
```

The training recordings are **not** the calibration pairs. Quick HDN training
needs a registered training dataset with at least two independent recordings
for train/validation splitting; see
[quick HDN training](quick_hdn_training_integration.md).
The preview checks configuration compatibility but does not validate the
scientific quality of the calibration or actually train a model.

## Limitations and first-version validation

- Array input supports **only YX or NYX**; multiple observations per reference,
  noise-level selection and averaging reference frames inside the builder remain
  available through the legacy `lisai.data.noise_model.build_noise_model` API.
- No interactive histogram inspection or model-comparison UI is provided.
  The plots are saved for manual inspection, as in the legacy workflow.
- File order is the caller's responsibility. Two equal-shaped, mispaired
  images are not automatically detected.
- GMM optimization is stochastic and can depend on PyTorch/CPU/GPU versions.
  For comparisons, control `numpy` and `torch` random seeds and do not expect
  bit-identical weights across all platforms.
- The numerical and workspace tests use small synthetic arrays, **not** real
  microscope data; validate a real calibration and actual HDN training on your
  target Windows/NVIDIA environment before using it experimentally.

For manual validation on a real dataset, inspect the generated `GMM_plot.png`,
check that the selected intensity range is represented, ensure calibration
and training images use matching reconstruction/normalization conventions,
and run a short real HDN training with `noise_model_name=result.name`. See the
[quick HDN training integration notes](quick_hdn_training_integration.md).

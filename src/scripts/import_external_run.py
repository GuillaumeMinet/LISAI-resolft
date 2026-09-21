"""Lightweight editable script for importing one external run into LISAI.

Edit the configuration block below, then run this file from the LISAI environment.
The external checkpoint/config are archived for provenance; LISAI does not try to
interpret or execute them.
"""

from pathlib import Path

from lisai.evaluation import EvalSource
from lisai.runs.external import import_external_run


# -----------------------------------------------------------------------------
# External run
# -----------------------------------------------------------------------------
run_name = "SN2N_Gag"
trained_on = "gag_live_tl_lowON"

checkpoint_path = Path(r"E:\lisai\datasets\training\gag_live_tl_lowON\sn2n\models\last_model\model_9_20_full.pth")
evaluation_data_path = Path(r"E:\lisai\datasets\evaluation\gag_live_denoise_eval\preprocess\recon\predictions")

# Optional external config - copied into external_runs/<run>/config/.
config_path = None

# What the imported predictions were evaluated on.
evaluated_on = EvalSource.dataset("gag_live_denoise_eval")
# evaluated_on = EvalSource.training_split("test")

# Describes how each saved prediction TIFF should be interpreted at import time.
# prediction_format = "image"
# prediction_format = "stack_inp_pred"  # axis 0 = [input, prediction]
prediction_format = "4d"              # shape (1, 1, H, W)

prediction_glob = ("*.tif", "*.tiff")

# Specify only information that cannot be inferred unambiguously from the registry.
# Multi-SNR datasets require snr_idx. Datasets with several inputs/GTs require a choice.
training_data = {
    # "data_type": "recon",
    # "input": "inp_single",
    # "gt": "...",
    # "snr_idx": 0,
}

evaluation_data = {
    # "data_type": "recon",
    # "input": "inp_single",
    # "gt": "gt_avg",
    # "snr_idx": 0,
}

# External runs are assumed to use LISAI's current train/val/test split.
uses_current_split = True

overwrite = True
notes = "SN2N trained on noisy Gag data with original SN2N implementation."


if __name__ == "__main__":
    result = import_external_run(
        run_name=run_name,
        trained_on=trained_on,
        checkpoint_path=checkpoint_path,
        evaluation_data_path=evaluation_data_path,
        evaluated_on=evaluated_on,
        prediction_format=prediction_format,
        prediction_glob=prediction_glob,
        config_path=config_path,
        training_data=training_data,
        evaluation_data=evaluation_data,
        uses_current_split=uses_current_split,
        overwrite=overwrite,
        notes=notes,
    )
    print(f"Imported external run: {result.run_dir}")
    print(f"Imported evaluation: {result.evaluation_dir}")
    print(f"Predictions: {result.prediction_count}")

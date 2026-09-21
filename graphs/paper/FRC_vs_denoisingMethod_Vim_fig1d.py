import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SRC_ROOT):
    if str(path) not in sys.path: sys.path.insert(0, str(path))

import frc
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

from lisai.data.utils import crop_center
from graphs.utils.eval_folder import EvalSource, discover_evaluations, discover_external_evaluations
from graphs.utils.eval_outputs import prepare_comparison_outputs
from graphs.utils.paths import get_saved_graphs_dir


# data selection
dataset_name = "vim_fixed_multi_snr"
model_subfolder = "Denoising"
lisai_run_names = ["UNetRCAN_single_to_avg", "UNet_single_to_avg", "HDN_unsup_KL05", "HDN_sup_KL001"]
external_run_names = ["SN2N_Vim", "N2V_Vim"]
evaluation_source = EvalSource.training_split("test")
checkpoint = "best"

smooth_gt = True
crop_size = None

# figure parameters
unsup_conditions = ["GT", "HDNunsup", "SN2N", "N2V"]
sup_conditions = ["GT", "CARE", "HDNsup", "UnetRCAN"]
show_figure = True
linewidth = 0.7
figure_size = (12, 4)

# saving parameters
save_figure = True
save_folder = get_saved_graphs_dir()
save_title = "Denoising_FRC_vs_DenoisingMethod_Vim.svg"


# discover LISAI + imported external evaluations
lisai_evaluations = discover_evaluations(dataset=dataset_name, model_subfolder=model_subfolder,
                                         run_names=lisai_run_names, source=evaluation_source,
                                         checkpoint=checkpoint)
lisai_by_name = {evaluation.name: evaluation for evaluation in lisai_evaluations}
external_evaluations = discover_external_evaluations(dataset=dataset_name, run_names=external_run_names,
                                                      source=evaluation_source)
external_by_name = {evaluation.name: evaluation for evaluation in external_evaluations}

evaluation_by_condition = {
    "UnetRCAN": lisai_by_name["UNetRCAN_single_to_avg"],
    "CARE": lisai_by_name["UNet_single_to_avg"],
    "HDNunsup": lisai_by_name["HDN_unsup_KL05"],
    "HDNsup": lisai_by_name["HDN_sup_KL001"],
    "SN2N": external_by_name["SN2N_Vim"],
    "N2V": external_by_name["N2V_Vim"],
}

evaluations = list(evaluation_by_condition.values())
comparison = prepare_comparison_outputs(evaluations)

conditions = ["GT", "CARE", "HDNsup", "UnetRCAN", "HDNunsup", "SN2N", "N2V"]
colors_per_cond = {
    "GT": "black",
    "CARE": "#ce727db7",
    "HDNsup": "#7e2954ff",
    "UnetRCAN": "#d55e00ff",
    "HDNunsup": "#2e2585ff",
    "SN2N": "#48bda6ff",
    "N2V": "#337538ff",
}

frc_curves = {cond: [] for cond in conditions}
img_size = None


def add_frc(arr, cond):
    if crop_size is not None: 
        arr = crop_center(arr, crop_size)
    if cond == "GT" and smooth_gt: 
        arr = gaussian_filter(arr, sigma=0.5, radius=3)

    arr = (arr - np.mean(arr)) / np.std(arr)
    arr = arr - np.min(arr)
    arr = frc.util.apply_tukey(arr)

    frc_curve = frc.one_frc(arr)
    if np.isnan(frc_curve).any():
        print(f"Skipping image in {cond} due to NaN values in FRC curve.")
        return False
    frc_curves[cond].append((frc_curve - np.min(frc_curve)) / (np.max(frc_curve) - np.min(frc_curve)))
    return True


# GT is identical across all aligned evaluations: calculate only once
reference_evaluation = evaluation_by_condition["HDNunsup"]
count = 0
for item in comparison.items_for(reference_evaluation):
    gt_frames, _ = item.load_gt_pred()
    for gt in gt_frames:
        img_size = gt.shape
        count += int(add_frc(gt, "GT"))
print(f"GT: #{count} files")


# predictions
for cond, evaluation in evaluation_by_condition.items():
    count = 0
    print(f"\n{cond}: {evaluation.name}")
    print(f"  {evaluation.folder}")
    for item in comparison.items_for(evaluation):
        _, pred_frames = item.load_gt_pred()
        for pred in pred_frames: count += int(add_frc(pred, cond))
    print(f"{cond}: #{count} files")


# average and standard deviation
avg_frc_curves = {}
std_frc_curves = {}
for cond in conditions:
    curves = np.stack(frc_curves[cond])
    avg_frc_curves[cond] = np.mean(curves, axis=0)
    std_frc_curves[cond] = np.std(curves, axis=0)


# plot splitting supervised and unsupervised
scale = 1 / 30 * 1e3
fig, axs = plt.subplots(1, 2, figsize=figure_size)
for ax, panel_conditions in zip(axs, [unsup_conditions, sup_conditions]):
    for cond in panel_conditions:
        color = colors_per_cond[cond]
        frc_curve = avg_frc_curves[cond]
        std_curve = std_frc_curves[cond]
        xs_pix = np.arange(len(frc_curve)) / img_size[0]
        xs_nm_freq = xs_pix * scale
        ax.plot(xs_nm_freq, frc_curve, label=cond, linewidth=linewidth,color=color)
        ax.fill_between(xs_nm_freq, frc_curve - std_curve, frc_curve + std_curve, 
                        alpha=0.5, color=color, linewidth=0)
    ax.legend()
    ax.set_xticks([0, 5, 10, 15])
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Spatial frequency (µm$^{-1}$)")
    ax.set_ylabel("Correlation")

if show_figure: 
    plt.show()
if save_figure: 
    fig.savefig(save_folder / save_title, format="svg", bbox_inches="tight")

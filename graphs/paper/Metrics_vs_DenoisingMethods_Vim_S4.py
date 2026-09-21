import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SRC_ROOT):
    if str(path) not in sys.path: sys.path.insert(0, str(path))

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

from graphs.utils.calculate_metrics import calculate_metrics
from graphs.utils.boxplot import box_plot
from graphs.utils.eval_folder import EvalSource, discover_evaluations, discover_external_evaluations
from graphs.utils.eval_outputs import prepare_comparison_outputs
from graphs.utils.paths import get_saved_graphs_dir


# data selection
dataset_name = "vim_fixed_multi_snr"
model_subfolder = "denoising_legacy"
lisai_run_names = ["UNetRCAN_single_to_avg", "UNet_single_to_avg", "HDN_unsup_KL05", "HDN_sup_KL001"]
external_run_names = ["SN2N_Vim", "N2V_Vim"]
evaluation_source = EvalSource.training_split("test")
checkpoint = "best"

# metrics calculation parameters
smooth_gt = True
use_windowed = True
window_size = 600
range_invariant = False
patch_selection = 0.3

# figure parameters
conditions = ["SN2N", "N2V", "HDN", "CARE", "UnetRCAN", "HDNsup"]
labels = ["SN2N", "N2V", "HDN", "CARE", "UNetRCAN", r"HDN$^{sup}$"]
colors_per_cond = {
    "SN2N": "#48bda6ff",
    "N2V": "#337538ff",
    "HDN": "#2e2585ff",
    "CARE": "#ce727db7",
    "UnetRCAN": "#d55e00ff",
    "HDNsup": "#7e2954ff",
}
colors_list = [colors_per_cond[cond] for cond in conditions]
show_figure = True
figsize = (15, 7)
spaceBetweenSubplots = 0.3
spaceBelowSubplots = 0.2

# Slight gap between unsupervised (first 3) and supervised (last 3) methods.
positions = [0.1, 0.25, 0.4, 0.6, 0.75, 0.9]
box_plot_parameters = {
    "colors": colors_list,
    "widths": 0.1,
    "positions": positions,
    "linewidth": 2,
    "dashed_whiskers": True,
    "showfliers": False,
    "showMeanAndStd": False,
    "showMeanAndStd_pos": "above",
    "showDataPoints": True,
    "dataPoints_size": 10,
    "dataPoints_alpha": 0.7,
    "dataPoints_color": "same",
    "labels_angle": 45,
    "use_mean": True,
    "labels_fontSize": 20,
    "ticks_prms": {"labelsize": 20, "width": 2, "length": 8},
}

# saving parameters
save_figure = True
save_folder = get_saved_graphs_dir()
save_title = "Denoising_PSNR_SSIM_vs_DenoisingMethod_Vim.svg"


# discover LISAI + imported external evaluations
lisai_evaluations = discover_evaluations(dataset=dataset_name, model_subfolder=model_subfolder,
                                         run_names=lisai_run_names, source=evaluation_source,
                                         checkpoint=checkpoint)
lisai_by_name = {evaluation.name: evaluation for evaluation in lisai_evaluations}
external_evaluations = discover_external_evaluations(dataset=dataset_name, run_names=external_run_names,
                                                      source=evaluation_source)
external_by_name = {evaluation.name: evaluation for evaluation in external_evaluations}

evaluation_by_condition = {
    "SN2N": external_by_name["SN2N_Vim"],
    "N2V": external_by_name["N2V_Vim"],
    "HDN": lisai_by_name["HDN_unsup_KL05"],
    "CARE": lisai_by_name["UNet_single_to_avg"],
    "UnetRCAN": lisai_by_name["UNetRCAN_single_to_avg"],
    "HDNsup": lisai_by_name["HDN_sup_KL001"],
}

evaluations = [evaluation_by_condition[cond] for cond in conditions]
comparison = prepare_comparison_outputs(evaluations, gt_reference="dataset")

psnr_values = {cond: [] for cond in conditions}
ssim_values = {cond: [] for cond in conditions}


# metrics calculation
for cond in conditions:
    evaluation = evaluation_by_condition[cond]
    n_images = 0
    n_patches = 0
    print(f"\n{cond}: {evaluation.name}")
    print(f"  {evaluation.folder}")

    for item in comparison.items_for(evaluation):
        gt_frames, pred_frames = comparison.load_gt_pred(item)
        for gt, pred in zip(gt_frames, pred_frames):
            n_images += 1

            if smooth_gt:
                gt = gaussian_filter(gt, sigma=0.6, radius=3)

            gt = (gt - np.mean(gt)) / np.std(gt)
            pred = (pred - np.mean(pred)) / np.std(pred)
            data_range = np.max(gt) - np.min(gt)

            psnr_val, ssim_val, _ = calculate_metrics(gt, pred, data_range, use_windowed,
                                                       window_size, patch_selection, range_invariant)
            psnr_values[cond].extend(psnr_val)
            ssim_values[cond].extend(ssim_val)
            n_patches += len(psnr_val)

    print(f"Used #{n_images} aligned images -> #{n_patches} selected {window_size}x{window_size} patches")


# plot PSNR and SSIM together, ordered as Supplementary Figure S4
fig, axs = plt.subplots(1, 2, figsize=figsize)
fig.subplots_adjust(wspace=spaceBetweenSubplots)
fig.subplots_adjust(bottom=spaceBelowSubplots)

psnrs = [psnr_values[cond] for cond in conditions]
ssims = [ssim_values[cond] for cond in conditions]

fig = box_plot(psnrs, labels, fig=fig, ax=axs[0], ylabel="PSNR (dB)", **box_plot_parameters)
fig = box_plot(ssims, labels, fig=fig, ax=axs[1], ylabel="SSIM", **box_plot_parameters)

for ax in axs:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(2)
    ax.spines["bottom"].set_linewidth(2)

if show_figure: plt.show()
if save_figure: fig.savefig(save_folder / save_title, bbox_inches="tight")

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

from graphs.utils.calculate_metrics import calculate_metrics
from graphs.utils.boxplot import box_plot as new_box_plot
from graphs.utils.eval_folder import EvalSource, discover_evaluations
from graphs.utils.eval_outputs import prepare_comparison_outputs
from graphs.utils.paths import get_saved_graphs_dir

# this dataset contains:
#   - Sampling 0.25, CL1,3,5,7
#   - Sampling 0.5 (CL1 only)
# By default this graphs only compares the different CL for Sampling 0.25
# Change show_s05 to False  to add S=0.5-CL1 condition
show_s05=False

# data selection
dataset_name = "gag_live_tl_highON"
model_subfolder = "Upsamp"

run_names = [
    "CL1_Upsamp2_biggerNet_02",
    "CL3_Upsamp2_biggerNet_03",
    "CL5_Upsamp2_biggerNet_03",
    "CL7_Upsamp2_biggerNet_00",
]

if show_s05:
    run_names.append(
        "CL1_Upsamp05_biggerNet_01"
    )


evaluation_source = EvalSource.training_split("test")
checkpoint = "best"  # "best" or "last"


# saving parameters
save_figure = True
save_folder = get_saved_graphs_dir()
save_title = "Upsamp_PSNR_SSIM_vs_ContextLength_Gag.svg"

# metrics calculation parameters
smooth_gt = False
use_windowed = False
window_size = 600
range_invariant = False
patch_selection = False

# figure parameters
show_figure = True

colors_list = [
    "mediumblue",
    "mediumblue",
    "mediumblue",
    "#0d9188ff",
    "darkred",
]

figsize = (10, 5)
spaceBetweenSubplots = 0.5
spaceBelowSubplots = 0.2

linewidth = 2
labelsize = 20
tick_length = linewidth/2.5

def condition_label(evaluation):
    """
    Use N=<context> for the normal S=0.25 comparison.

    If a different sampling ratio is selected, expose it explicitly.
    """
    context_length = evaluation.context_length
    sampling_ratio = evaluation.sampling_ratio

    if (
        sampling_ratio is not None
        and not np.isclose(sampling_ratio, 0.25)
    ):
        if context_length == 1:
            return f"S={sampling_ratio:g}"

        return f"N={context_length}, S={sampling_ratio:g}"

    return f"N={context_length}"


# find evaluation items
evaluations = discover_evaluations(
    dataset=dataset_name,
    model_subfolder=model_subfolder,
    run_names=run_names,
    source=evaluation_source,
    checkpoint=checkpoint,
)

# Put the regular context-length comparison first, followed by
# any optional non-default sampling conditions.
evaluations.sort(
    key=lambda evaluation: (
        evaluation.sampling_ratio is not None
        and not np.isclose(evaluation.sampling_ratio, 0.25),
        evaluation.context_length,
        evaluation.sampling_ratio or 0.25,
    )
)

conditions = [condition_label(evaluation)for evaluation in evaluations]

# Exact source-item and timepoint alignment across all conditions
comparison = prepare_comparison_outputs(evaluations)


# adapt box positions/colors to the selected conditions
positions = [
    0.1 + 0.15 * i
    for i in range(len(conditions))
]

box_plot_parameters = {
    "colors": colors_list[:len(conditions)],
    "widths": 0.1,
    "positions": positions,
    "linewidth": linewidth,
    "dashed_whiskers": False,
    "showfliers": False,
    "showMeanAndStd": False,
    "showMeanAndStd_pos": "above",
    "showDataPoints": False,
    "dataPoints_size": 10,
    "dataPoints_alpha": 0.7,
    "dataPoints_color": "same",
    "labels_angle": 45,
    "xlabel": "Number of frames",
    "use_mean": True,
    "labels_fontSize": labelsize,
    "ticks_prms": {"labelsize": labelsize,
                   "width": linewidth, 
                   "length": tick_length},
}


# Initialize dictionaries to store metrics for each condition
psnr_values = {cond: [] for cond in conditions}
ssim_values = {cond: [] for cond in conditions}
mse_values = {cond: [] for cond in conditions}


# metrics calculation
for evaluation, cond in zip(evaluations, conditions):
    print(f"{cond}: {evaluation.run.name}")
    print(f"  {evaluation.folder}")

    effective_images = 0

    for item in comparison.items_for(evaluation):
        gt_frames, pred_frames = item.load_gt_pred()

        for gt, pred in zip(gt_frames, pred_frames):
            effective_images += 1

            gt = gt.copy()

            min_pred = np.min(pred)
            gt[gt < min_pred * 1.1] = min_pred * 1.1

            if smooth_gt:
                gt = gaussian_filter(gt,sigma=0.5,radius=3)

            gt = (gt - np.mean(gt)) / np.std(gt)
            pred = (pred - np.mean(pred)) / np.std(pred)

            data_range = np.max(gt) - np.min(gt)

            psnr_val, ssim_val, mse_val = calculate_metrics(
                gt,pred,data_range,use_windowed,window_size,
                patch_selection,range_invariant,
            )

            psnr_values[cond].extend(psnr_val)
            ssim_values[cond].extend(ssim_val)
            mse_values[cond].extend(mse_val)

    print(f"Used #{effective_images} aligned images")


# do plots
fig, axs = plt.subplots(1, 2, figsize=figsize)
fig.subplots_adjust(wspace=spaceBetweenSubplots)
fig.subplots_adjust(bottom=spaceBelowSubplots)

psnrs = [psnr_values[cond] for cond in conditions]
ssims = [ssim_values[cond] for cond in conditions]
mses = [mse_values[cond] for cond in conditions]

fig = new_box_plot(psnrs, conditions,fig=fig,ax=axs[0],
                   ylabel="PSNR (dB)",**box_plot_parameters)
fig = new_box_plot(ssims,conditions,fig=fig,ax=axs[1],
                   ylabel="SSIM",**box_plot_parameters)

for ax in axs:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(linewidth)
    ax.spines["bottom"].set_linewidth(linewidth)


# show
if show_figure:
    plt.show()


# save
if save_figure:
    fig.savefig(save_folder / save_title,bbox_inches="tight")
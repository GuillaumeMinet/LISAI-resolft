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


# data selection
dataset_name = "vim_fixed_multi_snr"
model_subfolder = "Upsamp"

run_names = "all"
evaluation_source = EvalSource.training_split("test")
checkpoint = "last"  # "best" or "last"


# saving parameters
save_figure = True
save_folder = get_saved_graphs_dir()
save_title = "Upsamp_PSNR_SSIM_vsSampling_HighSNR.svg"


# metrics calculation parameters
smooth_gt = True
use_windowed = True
window_size = 600
range_invariant = False
patch_selection = False


# figure parameters
show_figure = True

colors_list = [
    "#a70048ff",
    "#0000ffff",
    "#439c43fb",
]

linewidth = 2
labelsize = 20
tick_length = linewidth/2.5

figsize = (10, 5)
spaceBetweenSubplots = 0.5
spaceBelowSubplots = 0.2


box_plot_parameters = {
    "widths": 0.04,
    "positions": [0.1, 0.2, 0.3],
    "linewidth": linewidth,
    "dashed_whiskers": False,
    "showfliers": False,
    "showMeanAndStd": False,
    "showDataPoints": False,
    "xlabel": None,
    "use_mean": True,
    "labels_fontSize": labelsize,
    "ticks_prms": {
        "labelsize": labelsize,
        "width": linewidth,
        "length": tick_length,
    },
}


# find evaluations
evaluations = discover_evaluations(
    dataset=dataset_name,
    model_subfolder=model_subfolder,
    run_names=run_names,
    source=evaluation_source,
    checkpoint=checkpoint,
)

if any (evaluation.sampling_ratio is None for evaluation in evaluations):
    raise ValueError( "Could not determine the sampling ratio  for one or more runs.")

evaluations.sort(key=lambda evaluation: evaluation.sampling_ratio)

sampling_ratios = [evaluation.sampling_ratio for evaluation in evaluations]

# Align the exact same source images across all sampling ratios
comparison = prepare_comparison_outputs(evaluations)


# Initialize dictionaries to store metrics
psnr_values = {sampling_ratio: []for sampling_ratio in sampling_ratios}
ssim_values = {sampling_ratio: []for sampling_ratio in sampling_ratios}
mse_values = {sampling_ratio: []for sampling_ratio in sampling_ratios}


# metrics calculation
for evaluation, sampling_ratio in zip(evaluations, sampling_ratios):
    print(f"S={sampling_ratio}: {evaluation.run.name}")
    print(f"  {evaluation.folder}")

    n_images = 0

    for item in comparison.items_for(evaluation):
        gt_frames, pred_frames = item.load_gt_pred()

        for gt, pred in zip( gt_frames,pred_frames):
            n_images += 1

            if smooth_gt:
                gt = gaussian_filter(gt,sigma=0.6,radius=3)

            gt = (gt - np.mean(gt)) / np.std(gt)
            pred = (pred - np.mean(pred)) / np.std(pred)

            data_range = np.max(gt) - np.min(gt)

            psnr_val, ssim_val, mse_val = calculate_metrics(
                gt,pred,data_range,use_windowed,window_size,
                patch_selection,range_invariant,
            )

            psnr_values[sampling_ratio].extend(psnr_val)
            ssim_values[sampling_ratio].extend(ssim_val)
            mse_values[sampling_ratio].extend(mse_val)

    print(f"Used #{n_images} aligned images")


# do plots
fig, axs = plt.subplots(1,2,figsize=figsize)
fig.subplots_adjust(wspace=spaceBetweenSubplots)
fig.subplots_adjust(bottom=spaceBelowSubplots)

psnrs = [psnr_values[sampling_ratio] for sampling_ratio in sampling_ratios]
ssims = [ssim_values[sampling_ratio] for sampling_ratio in sampling_ratios]
mses = [mse_values[sampling_ratio] for sampling_ratio in sampling_ratios]


fig = new_box_plot(psnrs,sampling_ratios,fig=fig,ax=axs[0],
                   ylabel="PSNR (dB)",colors=colors_list,**box_plot_parameters)

fig = new_box_plot(ssims,sampling_ratios,fig=fig,ax=axs[1],
                   ylabel="SSIM",colors=colors_list,**box_plot_parameters)


for ax in axs:
    ax.set_xticks([])

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(linewidth)
    ax.spines["bottom"].set_linewidth(linewidth)


axs[0].set_ylim([28, 44])
axs[1].set_ylim([0.65, 0.99])
axs[0].set_yticks([30, 35, 40])

for ax in axs:
    ax.set_xlim([0.03, 0.38])


# show
if show_figure:
    plt.show()


# save
if save_figure:
    fig.savefig(save_folder / save_title,bbox_inches="tight")

import os
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
from graphs.utils.eval_folder import EvalSource,discover_evaluations
from graphs.utils.eval_outputs import prepare_comparison_outputs
from graphs.utils.paths import get_saved_graphs_dir



dataset_name = "mito_live_tl_highON"
model_subfolder = "Upsamp"
run_names = "all"

evaluation_source = EvalSource.training_split("test")
checkpoint = "last" # "best" or "last"

# saving parameters
save_figure = True
save_folder = get_saved_graphs_dir() 
save_title = "Upsamp_PSNR_SSIM_vs_ContextLength_Mitoch.svg"

# metrics calculation parameters
smooth_gt = True
use_windowed = False
window_size = None
range_invariant = False
patch_selection = False


# figure parameters
show_figure = True
colors_list = ["mediumblue","mediumblue","#0d9188ff","mediumblue"]

figsize = (10,5)
spaceBetweenSubplots=0.3
spaceBelowSubplots=0.2

box_plot_parameters = {
    "colors": colors_list,#"mediumblue", # if None, each box will be all black
    "widths": 0.1,
    "positions": [0.1, 0.25, 0.4, 0.55], # positions of the boxes on the x-axis
    "linewidth": 2,
    "dashed_whiskers": False,
    "showfliers": False,
    "showMeanAndStd": False,
    "showMeanAndStd_pos": "above",
    "showDataPoints": False,
    "dataPoints_size": 10,
    "dataPoints_alpha": 0.7,
    "dataPoints_color": 'same',
    "labels_angle": 45,
    "xlabel": "Number of frames",
    "use_mean": True,
    "labels_fontSize": 20,
    "ticks_prms": {"labelsize":20, "width":2,"length":8},
}


# find evaluations items
evaluations = discover_evaluations(
    dataset=dataset_name,
    model_subfolder=model_subfolder,
    run_names=run_names,
    source=evaluation_source,
    checkpoint=checkpoint,
)

evaluations.sort(
    key=lambda evaluation: evaluation.context_length
)

cl_list = [
    f"N={evaluation.context_length}"
    for evaluation in evaluations
]

comparison = prepare_comparison_outputs(evaluations)

# Initialize dictionaries to store metrics for each context length
psnr_values = {cl: [] for cl in cl_list}
ssim_values = {cl: [] for cl in cl_list}
mse_values = {cl: [] for cl in cl_list}

# metrics calculation
for evaluation,cl in zip (evaluations,cl_list):
    print(f"{cl}: {evaluation.run.name}")
    print(f"  {evaluation.folder}")

    for item in comparison.items_for(evaluation):
        gt_frames,pred_frames = item.load_gt_pred()

        for gt,pred in zip(gt_frames,pred_frames):
            if smooth_gt:
                gt = gaussian_filter (gt,sigma = 0.6,radius = 3)

            data_range = np.max(gt) - np.min(gt)
            psnr_val, ssim_val,mse_val = calculate_metrics(
                gt, pred, data_range,use_windowed,window_size,
                patch_selection,range_invariant
            )
            
            psnr_values[cl].extend(psnr_val)
            ssim_values[cl].extend(ssim_val)
            mse_values[cl].extend(mse_val)


# do plots
fig,axs = plt.subplots(1,2,figsize=figsize)
fig.subplots_adjust(wspace=spaceBetweenSubplots)
plt.subplots_adjust(bottom=spaceBelowSubplots)

psnrs = [psnr_values[cl] for cl in cl_list]
ssims = [ssim_values[cl] for cl in cl_list]
mses = [mse_values[cl] for cl in cl_list]

fig=new_box_plot(psnrs,cl_list,fig=fig,ax=axs[0],
                 ylabel="PSNR (dB)",**box_plot_parameters)
fig=new_box_plot(ssims,cl_list,fig=fig,ax=axs[1],
                 ylabel="SSIM",**box_plot_parameters)

for ax in axs:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)


# show
if show_figure:
    plt.show()  

# Saving
if save_figure:
    fig.savefig(os.path.join(save_folder, save_title), bbox_inches='tight')
import os, sys
sys.path.append(os.getcwd())
from lisai.graphs.utils.calculate_metrics import calculate_metrics
from lisai.graphs.utils.boxplot import box_plot

import numpy as np
import matplotlib.pyplot as plt
from tifffile import imread
from matplotlib.lines import Line2D
from pathlib import Path

# Folder containing the image stacks
folder_path = Path(r'\\storage3.ad.scilifelab.se\testalab\Guillaume\01_Projects\DL_monalisa\_paper\Denoising_Technical\Model_comp')
stack_name = "All_in_one.tif"
nn_list = ["SN2N", "N2V", "HDN-unsup", "CARE", "UNet-RCAN", "HDN-sup"]
labels_list = [
    "SN2N",
    "N2V",
    "HDN",
    "CARE",
    "UNetRCAN",
    "HDN$^{sup}$"
]
pred_order = [6, 5, 2, 1, 3, 4]

# metrics calculation parameters
smooth_gt = True
use_windowed = True
window_size = 600
range_invariant = False
patch_selection = 0.3

# figure parameters
colors_list = ['#48bda6ff', '#337538ff', '#2e2585ff', '#cc79a7ff', '#d55e00ff', '#7e2954ff']

figsize = (15,7)
spaceBetweenSubplots=0.3
spaceBelowSubplots=0.2
box_plot_parameters = {
    "colors": colors_list, # if None, each box will be all black
    "widths": 0.1,
    "positions": [0.1, 0.25, 0.4,0.55,0.7,0.85], # positions of the boxes on the x-axis
    "linewidth": 2,
    "dashed_whiskers": True,
    "showfliers": False,
    "showMeanAndStd": False,
    "showMeanAndStd_pos": "above",
    "showDataPoints": True,
    "dataPoints_size": 10,
    "dataPoints_alpha": 0.7,
    "dataPoints_color": 'same',
    "labels_angle": 45,
    "use_mean": True,
    "labels_fontSize": 20,
    "ticks_prms": {"labelsize":20, "width":2,"length":8},
}

show_figure = True

# saving parameters
save_figure = True
save_folder = os.path.join(os.getcwd(), r"src/graphs/saved_graphs")
save_title = "Denoising_PSNR_SSIM_MSE_vs_DenoisingModel.svg"

# load images
stack = imread(folder_path / stack_name)
gts = stack[:, 0]
preds = stack[:, pred_order]

# metrics calculation
psnr_values = {nn: [] for nn in nn_list}
ssim_values = {nn: [] for nn in nn_list}
mse_values = {nn: [] for nn in nn_list}

for i, nn in enumerate(nn_list):
    nn_preds = preds[:, i]
    for j in range(len(nn_preds)):
        pred = nn_preds[j]
        
        gt = gts[j]
        gt = (gt - np.mean(gt))/ np.std(gt)
        pred = (pred - np.mean(pred))/ np.std(pred)
        data_range = np.max(gt) - np.min(gt)
        psnr_val, ssim_val, mse_val = calculate_metrics(
            gt, pred, data_range, use_windowed,
            window_size, patch_selection, range_invariant
        )
        psnr_values[nn].extend(psnr_val)
        ssim_values[nn].extend(ssim_val)
        mse_values[nn].extend(mse_val)

# do plots
fig, axs = plt.subplots(1, 2, figsize=figsize)
fig.subplots_adjust(wspace=spaceBetweenSubplots)
plt.subplots_adjust(bottom=spaceBelowSubplots)

for _ in range(1):
    psnrs = [psnr_values[nn] for nn in nn_list]
    ssims = [ssim_values[nn] for nn in nn_list]
    mses = [mse_values[nn] for nn in nn_list]
    fig = box_plot(psnrs, labels_list, fig=fig, ax=axs[0], ylabel="PSNR (dB)",**box_plot_parameters)
    fig = box_plot(ssims, labels_list, fig=fig, ax=axs[1], ylabel="SSIM",**box_plot_parameters)
    # fig = box_plot(mses, nn_list, fig=fig, ax=axs[2], title="MSE", ylabel="MSE",**box_plot_parameters)

for ax in axs:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)

if show_figure:
    plt.show()

# Saving
if save_figure:
    fig.savefig(os.path.join(save_folder, save_title), bbox_inches='tight')
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from graphs.utils.calculate_metrics import calculate_metrics
from graphs.utils.boxplot import box_plot as new_box_plot

import numpy as np
import matplotlib.pyplot as plt
from tifffile import imread
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter


# data folder
folder_path = r'E:\dl_monalisa\Models\Vim_fixed_mltplSNR_30nm\Upsampling_selected\unpaired'

# SNR names and sampling ratio values
sampling_ratios = [0.25, 0.5, 0.75]

# metrics calculation parameters
smooth_gt = True
use_windowed = True
window_size = 600
range_invariant = False
patch_selection = False

linewidth = 0.7
labelsize = 8
# figure parameters
colors_list = ['#a70048ff', '#0000ffff', '#439c43fb']
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
    "ticks_prms": {"labelsize":8, "width":linewidth,"length":2.5}
}


show_figure = False

# saving parameters
save_figure = True
save_folder = PROJECT_ROOT / "graphs" / "saved_graphs"
save_title = "Upsamp_PSNR_SSIM_vsSampling_HighSNR.svg"



# Initialize dictionaries to store metrics for each SNR and sampling ratio combination
psnr_values = {sampling: [] for sampling in sampling_ratios}
ssim_values = {sampling: [] for sampling in sampling_ratios}
mse_values = {sampling: [] for sampling in sampling_ratios}

# Process each image stack
for file_name in os.listdir(folder_path):
    if file_name.endswith('.tif'):
        file_path = os.path.join(folder_path, file_name)
        img_stack = imread(file_path)

        # Ground truth (first image in the stack)
        gt = img_stack[0]
        if smooth_gt:
            gt = gaussian_filter (gt,sigma = 0.6,radius = 3)
        data_range = np.max(gt) - np.min(gt)

        # Process predictions for each SNR and filling factor
        for j, sampling_ratio in enumerate(sampling_ratios):
            # Get the corresponding prediction index
            pred_index = j + 1 
            pred = img_stack[pred_index]
            
            # Calculate PSNR and SSIM
            psnr_val, ssim_val,mse_val = calculate_metrics(gt, pred, data_range,use_windowed,
                                                window_size,patch_selection,range_invariant)
            psnr_values[sampling_ratio].extend(psnr_val)
            ssim_values[sampling_ratio].extend(ssim_val)
            mse_values[sampling_ratio].extend(mse_val)


# do plots

fig,axs = plt.subplots(1,2,figsize=(1.4, 1.2))
fig.subplots_adjust(wspace=1.2,left=0,right=0.9)

psnrs = [psnr_values[cond] for cond in sampling_ratios]
ssims = [ssim_values[cond] for cond in sampling_ratios]
mses =[mse_values[cond] for cond in sampling_ratios]

fig=new_box_plot(psnrs,sampling_ratios,fig=fig,ax=axs[0],ylabel="PSNR (dB)",
                    colors=colors_list,**box_plot_parameters)

fig=new_box_plot(ssims,sampling_ratios,fig=fig,ax=axs[1],ylabel="SSIM",
                    colors=colors_list,**box_plot_parameters)
    # fig=new_box_plot(mses,snr_names,fig=fig,ax=axs[2],plot_idx=plot_idx,title="MSE",
    #                  colors=colors_list[plot_idx],**box_plot_parameters)

for ax in axs:
    ax.set_xticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(linewidth)
    ax.spines['bottom'].set_linewidth(linewidth)

axs[0].set_ylim([28,44])
axs[1].set_ylim([0.65,0.99])
axs[0].set_yticks([30,35,40])

for ax in axs:
    ax.set_xlim([0.03,0.38])



if show_figure:
    plt.show()

# Saving
if save_figure:
    fig.savefig(os.path.join(save_folder, save_title), bbox_inches='tight')


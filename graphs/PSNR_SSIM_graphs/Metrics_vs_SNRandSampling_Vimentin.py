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
snr_names = [f"High\nSNR", f"Medium\nSNR", "Low\nSNR"] 
sampling_ratios = [0.25, 0.5, 0.75]

# metrics calculation parameters
smooth_gt = False
use_windowed = False
window_size = 600
range_invariant = False
patch_selection = False

# figure parameters
colors_list = ['#a70048ff', '#0000ffff', '#439c43fb']
box_plot_parameters = {
    "mltpl_plots":True,
    "n_plots": len(sampling_ratios),
    "mltpl_displacements": 0.15,
    "widths": 0.08,
    "linewidth": 0.6,
    "dashed_whiskers": False,
    "showfliers": False,
    "showMeanAndStd": False,
    "showMeanAndStd_pos": "above",
    "showDataPoints": False,
    "dataPoints_size": 10,
    "dataPoints_alpha": 0.5,
    "dataPoints_color": 'same',
    "labels_angle": 0,
    "labels_fontSize": 5.5,
    "xlabel": None,
    "use_mean": True,
    "ticks_prms": {"labelsize":5.5, "width":0.7,"length":2},
}


show_figure = True

# saving parameters
save_figure = False
save_folder = PROJECT_ROOT / "graphs" / "saved_graphs"
save_title = "Upsamp_PSNR_SSIM_MSE_vsSNRandSampling.svg"



# Initialize dictionaries to store metrics for each SNR and sampling ratio combination
psnr_values = {snr: {sampling: [] for sampling in sampling_ratios} for snr in snr_names}
ssim_values = {snr: {sampling: [] for sampling in sampling_ratios} for snr in snr_names}
mse_values = {snr: {sampling: [] for sampling in sampling_ratios} for snr in snr_names}

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
        for i, snr in enumerate(snr_names):
            for j, sampling_ratio in enumerate(sampling_ratios):
                # Get the corresponding prediction index
                pred_index = i * len(sampling_ratios) + j + 1
                pred = img_stack[pred_index]
                
                # Calculate PSNR and SSIM
                psnr_val, ssim_val,mse_val = calculate_metrics(gt, pred, data_range,use_windowed,
                                                       window_size,patch_selection,range_invariant)
                psnr_values[snr][sampling_ratio].extend(psnr_val)
                ssim_values[snr][sampling_ratio].extend(ssim_val)
                mse_values[snr][sampling_ratio].extend(mse_val)


# do plots

fig,axs = plt.subplots(2,1,figsize=(5, 5))
# fig.subplots_adjust(hspace=0.4,left=0,right=0.9)

for plot_idx,sampling in enumerate(sampling_ratios,start=0):
    psnrs = [psnr_values[snr][sampling] for snr in snr_names]
    ssims = [ssim_values[snr][sampling] for snr in snr_names]
    mses = [mse_values[snr][sampling] for snr in snr_names]

    fig=new_box_plot(psnrs,snr_names,fig=fig,ax=axs[0],plot_idx=plot_idx,ylabel="PSNR (dB)",
                     colors=colors_list[plot_idx],**box_plot_parameters)
    fig=new_box_plot(ssims,snr_names,fig=fig,ax=axs[1],plot_idx=plot_idx,ylabel="SSIM",
                     colors=colors_list[plot_idx],**box_plot_parameters)
    # fig=new_box_plot(mses,snr_names,fig=fig,ax=axs[2],plot_idx=plot_idx,title="MSE",
    #                  colors=colors_list[plot_idx],**box_plot_parameters)

for ax in axs:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.7)
    ax.spines['bottom'].set_linewidth(0.7)


if show_figure:
    plt.show()

# Saving
if save_figure:
    fig.savefig(os.path.join(save_folder, save_title), bbox_inches='tight')


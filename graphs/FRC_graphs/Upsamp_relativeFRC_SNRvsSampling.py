import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from graphs.utils.eval_folder import get_eval_folder, list_images
from graphs.utils.calculate_metrics import calculate_metrics
from graphs.utils.boxplot import box_plot as new_box_plot

import frc
import numpy as np
import matplotlib.pyplot as plt
from tifffile import imread
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter

folder_path = r'E:\dl_monalisa\Models\Vim_fixed_mltplSNR_30nm\Upsampling_selected\unpaired'

scale = 1/30*1e3 # um-1

snrs_labels = ["High SNR", "Medium SNR", "Low SNR"]
sampling_ratios = [0.25, 0.5, 0.75]


# saving parameters
save_figure = True
save_folder = os.path.join(os.getcwd(), r"src/graphs/saved_graphs")
save_title = "Upsamp_FRC_vsSNRandSampling.svg"


# Process each image stack
files = os.listdir(folder_path)
all_frc_curves = []
for file_idx,file_name in enumerate(files):
    if file_name.endswith('.tif'):
        file_path = os.path.join(folder_path, file_name)
        img_stack = imread(file_path)

        # Ground truth (first image in the stack)
        gt = img_stack[0]
        gt = frc.util.apply_tukey(gt)
        
        frc_curves = []
        # calculate frc curves for each pred
        for i, snr in enumerate(snrs_labels):
            for j, sampling_ratio in enumerate(sampling_ratios):
                pred_index = i * len(sampling_ratios) + j + 1
                pred = img_stack[pred_index]
                pred = frc.util.apply_tukey(pred)
                frc_curve = frc.two_frc(gt,pred)
                frc_curve = (frc_curve - np.min(frc_curve)) / (np.max(frc_curve) - np.min(frc_curve))
                frc_curves.append(frc_curve)
        
        all_frc_curves.append(np.stack(frc_curves))
        
all_frc_curves = np.stack(all_frc_curves)
avg_frc_curves = np.mean(all_frc_curves,axis=0)
std_frc_curves = np.std(all_frc_curves, axis=0)

img_size = gt.shape[0]
fontsize=5.5
colors = ['#a70048ff', '#0000ffff', '#439c43fb']
xticks_positions = [0,5,10,15]
yticks_positions = [0,0.5,1]
ticks_prms={"labelsize":fontsize, "width":0.7,"length":2}
fig, axs = plt.subplots(3, 1, figsize=(1, 3))
for i,snr in enumerate(snrs_labels):
    # axs[i].set_title(snr)
    for j,ff in enumerate(sampling_ratios):
        idx = i * len(sampling_ratios) + j
        frc_curve = avg_frc_curves[idx]
        std_curve = std_frc_curves[idx]

        xs_pix = np.arange(len(frc_curve)) / img_size
        xs_nm_freq = xs_pix * scale
        axs[i].plot(xs_nm_freq, frc_curve,label=ff,linewidth=0.2,color=colors[j])
        axs[i].fill_between(xs_nm_freq, frc_curve - std_curve, frc_curve + std_curve, 
                            color=colors[j],alpha=0.5, linewidth=0)
        
    axs[i].set_xticks(xticks_positions)
    axs[i].set_yticks(yticks_positions)
    axs[i].tick_params(axis='x',which='major',**ticks_prms)
    axs[i].tick_params(axis='y',which='major',**ticks_prms)

    axs[i].set_ylabel("Correlation",fontsize=fontsize)
    axs[i].set_ylim(0, 1)
    axs[i].set_xlim(0,)

axs[-1].set_xlabel("Spatial frequency (µm$^{-1}$)",fontsize=fontsize)

fig.subplots_adjust(hspace=0.4,left=0,right=0.9)

for ax in axs:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.7)
    ax.spines['bottom'].set_linewidth(0.7)

# plt.show()


# Saving
if save_figure:
    fig.savefig(os.path.join(save_folder, save_title), bbox_inches='tight')



# # data folder
# folder_path = r'E:\dl_monalisa\Models\Vim_fixed_mltplSNR_30nm\Upsampling_selected\unpaired'

# # SNR names and sampling ratio values
# snr_names = ["High SNR", "Medium SNR", "Low SNR"] 
# sampling_ratios = [0.25, 0.5, 0.75]

# # metrics calculation parameters
# smooth_gt = True
# use_windowed = True
# window_size = 600
# range_invariant = False
# patch_selection = False

# # figure parameters
# colors_list = ['mediumblue', 'darkred', 'forestgreen']
# box_plot_parameters = {
#     "mltpl_plots":True,
#     "n_plots": len(sampling_ratios),
#     "mltpl_displacements": 0.15,
#     "widths": 0.08,
#     "linewidth": 2,
#     "dashed_whiskers": True,
#     "showfliers": False,
#     "showMeanAndStd": False,
#     "showMeanAndStd_pos": "above",
#     "showDataPoints": True,
#     "dataPoints_size": 10,
#     "dataPoints_alpha": 0.5,
#     "dataPoints_color": 'same',
#     "labels_angle": 0,
#     "xlabel": None,
#     "use_mean": True,
# }


# show_figure = True

# # saving parameters
# save_figure = True
# save_folder = os.path.join(os.getcwd(), r"src/graphs/saved_graphs")
# save_title = "Upsamp_PSNR_SSIM_MSE_vsSNRandSampling_wDataPoints.svg"



# # Initialize dictionaries to store metrics for each SNR and sampling ratio combination
# psnr_values = {snr: {sampling: [] for sampling in sampling_ratios} for snr in snr_names}
# ssim_values = {snr: {sampling: [] for sampling in sampling_ratios} for snr in snr_names}
# mse_values = {snr: {sampling: [] for sampling in sampling_ratios} for snr in snr_names}

# # Process each image stack
# for file_name in os.listdir(folder_path):
#     if file_name.endswith('.tif'):
#         file_path = os.path.join(folder_path, file_name)
#         img_stack = imread(file_path)

#         # Ground truth (first image in the stack)
#         gt = img_stack[0]
#         if smooth_gt:
#             gt = gaussian_filter (gt,sigma = 0.6,radius = 3)
#         data_range = np.max(gt) - np.min(gt)

#         # Process predictions for each SNR and filling factor
#         for i, snr in enumerate(snr_names):
#             for j, sampling_ratio in enumerate(sampling_ratios):
#                 # Get the corresponding prediction index
#                 pred_index = i * len(sampling_ratios) + j + 1
#                 pred = img_stack[pred_index]
                
#                 # Calculate PSNR and SSIM
#                 psnr_val, ssim_val,mse_val = calculate_metrics(gt, pred, data_range,use_windowed,
#                                                        window_size,patch_selection,range_invariant)
#                 psnr_values[snr][sampling_ratio].extend(psnr_val)
#                 ssim_values[snr][sampling_ratio].extend(ssim_val)
#                 mse_values[snr][sampling_ratio].extend(mse_val)


# # do plots

# fig,axs = plt.subplots(1,3,figsize=(15, 5))
# for plot_idx,sampling in enumerate(sampling_ratios,start=0):
#     psnrs = [psnr_values[snr][sampling] for snr in snr_names]
#     ssims = [ssim_values[snr][sampling] for snr in snr_names]
#     mses = [mse_values[snr][sampling] for snr in snr_names]

#     fig=new_box_plot(psnrs,snr_names,fig=fig,ax=axs[0],plot_idx=plot_idx,title="PSNR",
#                      colors=colors_list[plot_idx],**box_plot_parameters)
#     fig=new_box_plot(ssims,snr_names,fig=fig,ax=axs[1],plot_idx=plot_idx,title="SSIM",
#                      colors=colors_list[plot_idx],**box_plot_parameters)
#     fig=new_box_plot(mses,snr_names,fig=fig,ax=axs[2],plot_idx=plot_idx,title="MSE",
#                      colors=colors_list[plot_idx],**box_plot_parameters)
# if show_figure:
#     plt.show()

# # Saving
# if save_figure:
#     fig.savefig(os.path.join(save_folder, save_title), bbox_inches='tight')


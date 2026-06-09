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
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim, mean_squared_error as mse
from tifffile import imread
from matplotlib.lines import Line2D
from lisai.evaluation.metrics import windowed_psnr_2d,windowed_ssim_2d,windowed_mse_2d
from scipy.ndimage import gaussian_filter


# Function to calculate PSNR and SSIM
def calculate_metrics(gt, pred,data_range=None,use_windowed=True,window_size=None,
                      patch_selection=None, range_invariant=False):
    if use_windowed:
        psnr_values = windowed_psnr_2d(gt, pred,size=window_size,
                                       patch_selection=patch_selection,
                                       range_invariant=range_invariant)
        ssim_values = windowed_ssim_2d(gt, pred,size=window_size,
                                       patch_selection=patch_selection)
        mse_values = windowed_mse_2d(gt, pred,size=window_size,
                                       patch_selection=patch_selection)
    else:
        psnr_values = [psnr(gt, pred,data_range=data_range)]
        ssim_values = [ssim(gt, pred,data_range=data_range)]
        mse_values = [mse(gt, pred)]

    return psnr_values, ssim_values, mse_values


# Function to do box plot

def box_plot(nn_list,values,box_width,colors,metric_name,ax):

    data = [values[nn] for nn in ((nn_list))]
    displaced_x = np.linspace(0,len(nn_list),num=len(nn_list))
    box = ax.boxplot(data, positions=displaced_x, widths=box_width,showfliers=False)
    # for item in ['whiskers', 'caps', 'fliers', 'medians','boxes']:
    #     plt.setp(box[item], color=colors[idx]) 
    

    ax.set_title(metric_name)
    ax.set_ylabel(metric_name)
    ax.set_xticklabels(nn_list)

    # legend_lines = [Line2D([0], [0], color=colors[idx], lw=1) for idx in range(len(filling_factors))]
    # ax.legend(legend_lines, [f'Fill Factor {ff}' for ff in filling_factors], loc='upper right',
    #           fontsize=8,frameon=False,edgecolor='black')



blur_gt = False
use_windowed = False
window_size = 128
range_invariant = False
patch_selection = 0.3

# Folder containing the image stacks
folder_path = r'E:\dl_monalisa\Models\Vim_fixed_mltplSNR_30nm\Upsampling_selected\unetvsunetrcanvsbic'

nn_list = ["UNet","UNet-RCAN","BIC"]

psnr_values = {nn:[] for nn in nn_list}
ssim_values = {nn:[] for nn in nn_list}
mse_values = {nn:[] for nn in nn_list}

for i in [0]:
    gt_name = f"img_{i}_gt.tif"
    gt = imread(os.path.join(folder_path,gt_name))
    # plt.figure()
    # plt.imshow(gt,cmap='gray')
    # plt.show()
    if blur_gt:
        gt = gaussian_filter (gt,sigma = 0.6,radius = 3)
    gt = (gt - np.mean(gt)) / np.std(gt)
    data_range = np.max(gt) - np.min(gt)

    pred_stack_name = f"img_{i}_pred.tif"
    pred_stack = imread(os.path.join(folder_path,pred_stack_name))
    
    # fig, axs = plt.subplots(1, 3, figsize=(14, 6))
    for nn_idx,pred in enumerate(pred_stack):
        nn = nn_list[nn_idx]
        pred = (pred - np.mean(pred)) / np.std(pred)
        
        psnr_val, ssim_val,mse_val = calculate_metrics(gt, pred, data_range,use_windowed,
                                            window_size,patch_selection,range_invariant)
        psnr_values[nn].extend(psnr_val)
        ssim_values[nn].extend(ssim_val)
        mse_values[nn].extend(mse_val)
    #     axs[nn_idx].imshow(pred,cmap='gray')
    # plt.show()



offset = 0.15
box_width = 0.08
colors = ['mediumblue', 'darkred', 'forestgreen']
fig, axs = plt.subplots(1, 3, figsize=(18, 6))

box_plot(nn_list,psnr_values,box_width,colors,"PSNR",axs[0])
box_plot(nn_list,ssim_values,box_width,colors,"SSIM",axs[1])
box_plot(nn_list,mse_values,box_width,colors,"MSE",axs[2])

plt.show()

# # Individual plots saving

# from pathlib import Path
# save_folder = Path(r"\\storage3.ad.scilifelab.se\testalab\Guillaume\01_Projects\DL_monalisa\paper\Upsampling_charac")


# # PSNR plot
# fig_psnr, ax_psnr = plt.subplots(figsize=(6, 5))
# box_plot(filling_factors, snr_values, psnr_values, offset, box_width, colors, "PSNR", ax_psnr)
# ax_psnr.legend().set_visible(False)  # Remove legend from the graph
# fig_psnr.savefig(save_folder/"PSNR_plot.svg", format="svg", bbox_inches='tight')

# # SSIM plot
# fig_ssim, ax_ssim = plt.subplots(figsize=(6, 5))
# box_plot(filling_factors, snr_values, ssim_values, offset, box_width, colors, "SSIM", ax_ssim)
# ax_ssim.legend().set_visible(False)  # Remove legend from the graph
# fig_ssim.savefig(save_folder/"SSIM_plot.svg", format="svg", bbox_inches='tight')

# # MSE plot
# fig_mse, ax_mse = plt.subplots(figsize=(6, 5))
# box_plot(filling_factors, snr_values, mse_values, offset, box_width, colors, "MSE", ax_mse)
# ax_mse.legend().set_visible(False)  # Remove legend from the graph
# fig_mse.savefig(save_folder/"MSE_plot.svg", format="svg", bbox_inches='tight')




# # save legend
# fig = plt.figure()
# legend_lines = [Line2D([0], [0], color=colors[idx], lw=1) for idx in range(len(filling_factors))]

# # Create a figure for the legend (without axes)
# fig = plt.figure(figsize=(3, 1)) 
# ax = fig.add_subplot(111)  
# ax.set_axis_off()
# ax.legend(legend_lines, [f'Fill Factor {ff}' for ff in filling_factors],
#           loc='center', framealpha=1, frameon=False, fontsize=15)

# # Remove ticks and labels
# ax.set_xticks([])
# ax.set_yticks([])

# # Save the legend as an SVG
# fig.savefig(save_folder/'legend.svg', format='svg', bbox_inches='tight', pad_inches=0.1)
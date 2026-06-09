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
from scipy.ndimage import gaussian_filter
from pathlib import Path


# data paths
folders_path = Path(r'E:\dl_monalisa\Models\Mito_34nm_timelapses\Upsampling')

foder_names = [
    "10FramesOnly_Upsamp2_RandomPx_UnetRCAN_rg8_rcab12_red16_CharEdge_alpha005",
    "CL3_Upsamp2_RandomPx_UnetRCAN_rg8_rcab12_red16_CharEdge_alpha005",
    "CL5_Upsamp2_RandomPx_UnetRCAN_rg8_rcab12_red16_CharEdge_alpha005",
    "CL7_Upsamp2_RandomPx_UnetRCAN_rg8_rcab12_red16_CharEdge_alpha005"]

evaluation_foder = "evaluation_last"


cl_list = ["N=1","N=3","N=5","N=7"]

# metrics calculation parameters
smooth_gt = True
use_windowed = False
window_size = None
range_invariant = False
patch_selection = False

# figure parameters
figsize = (10,5)
spaceBetweenSubplots=0.3
spaceBelowSubplots=0.2

colors_list = ["mediumblue","mediumblue","#0d9188ff","mediumblue"]


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

show_figure = True

# saving parameters
save_figure = False
save_folder = os.path.join(os.getcwd(), r"src/graphs/saved_graphs")
save_title = "Upsamp_PSNR_SSIM_MSE_vs_ContextLength_Mito.svg"


# Initialize dictionaries to store metrics for each context length
psnr_values = {cl: [] for cl in cl_list}
ssim_values = {cl: [] for cl in cl_list}
mse_values = {cl: [] for cl in cl_list}

# metrics calculation
for folder_idx,cl in  enumerate(cl_list):
    folder = foder_names[folder_idx]
    eval_folder = folders_path/folder/evaluation_foder
    list_files = os.listdir(eval_folder)
    for i,f in enumerate(list_files):
        if f.split(".")[-1] != "tif":
            list_files.pop(i)
    assert len(list_files) %3==0
    n = len(list_files)//3
    for i in range(n):
        gt = imread(eval_folder/f"img_{i}_gt.tif")
        if smooth_gt:
            gt = gaussian_filter (gt,sigma = 0.6,radius = 3)

        pred = imread(eval_folder/f"img_{i}_pred.tif")
        data_range = np.max(gt) - np.min(gt)
        psnr_val, ssim_val,mse_val = calculate_metrics(gt, pred, data_range,use_windowed,
                                                        window_size,patch_selection,range_invariant)
        
        psnr_values[cl].extend(psnr_val)
        ssim_values[cl].extend(ssim_val)
        mse_values[cl].extend(mse_val)

exit()
# do plots
fig,axs = plt.subplots(1,2,figsize=figsize)
fig.subplots_adjust(wspace=spaceBetweenSubplots)
plt.subplots_adjust(bottom=spaceBelowSubplots)

psnrs = [psnr_values[cl] for cl in cl_list]
ssims = [ssim_values[cl] for cl in cl_list]
mses = [mse_values[cl] for cl in cl_list]

fig=new_box_plot(psnrs,cl_list,fig=fig,ax=axs[0],ylabel="PSNR (dB)",**box_plot_parameters)
fig=new_box_plot(ssims,cl_list,fig=fig,ax=axs[1],ylabel="SSIM",**box_plot_parameters)
# fig=new_box_plot(mses,cl_list,fig=fig,ax=axs[2],title="MSE",**box_plot_parameters)


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


save_path = folders_path/ "Metrics_plots_different_CL.svg"
fig.savefig(save_path, format="svg", bbox_inches='tight')
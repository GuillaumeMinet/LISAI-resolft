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
from graphs.utils.eval_folder import get_eval_folder, list_images
import numpy as np
import matplotlib.pyplot as plt
from tifffile import imread
from scipy.ndimage import gaussian_filter


# data paths
folders_path = Path(r'E:\lisai\datasets\vim_live\models\Upsamp_selected')

folder_names = [
    "Fulldataset_CL1_1FramesMax_Upsamp2_smallerNet_clip_modifUpsamp_03",
    "Fulldataset_CL3_3FramesMax_Upsamp2_smallerNet_clip_06",
    "Fulldataset_CL5_5FramesMax_Upsamp2_smallerNet_clip_02",
    "Fulldataset_CL7_7FramesMax_Upsamp2_smallerNet_clip_06"
]

evaluation_folder = "eval_gathered"

max_imgs_per_timelapse = 10

cl_list = [1,3,5,7]
max_cl = max(cl_list)


smooth_gt = True
show_figure = True
use_windowed = False
window_size = 600
patch_selection = False
plot_gt = True
range_invariant = True

# figure parameters
figsize = (10,5)
spaceBetweenSubplots=0.5
spaceBelowSubplots=0.2

colors_list = ["mediumblue","mediumblue","#0d9188ff","mediumblue"]


box_plot_parameters = {
    "colors": colors_list,#"mediumblue", # if None, each box will be all black
    "widths": 0.1,
    "positions": [0.1, 0.25, 0.4, 0.55], #,0.7], # positions of the boxes on the x-axis
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

show_figure = False

# saving parameters
save_figure = True
save_folder = PROJECT_ROOT / "graphs" / "saved_graphs"
save_title = "Upsamp_PSNR_SSIM_MSE_vs_ContextLength_Vim.svg"


# get file idxs first
eval_folder = folders_path/folder_names[0]/evaluation_folder
list_files = list_images(eval_folder)
if len(list_files) % 3 != 0:
    raise ValueError
n_imgs = len(list_files) // 3

# Initialize dictionaries to store metrics for each context length
psnr_values = {cl: [] for cl in cl_list}
ssim_values = {cl: [] for cl in cl_list}
mse_values = {cl: [] for cl in cl_list}

# metrics calculation
for folder_idx,cl in  enumerate(cl_list):
    folder = folder_names[folder_idx]
    print("Folder: ", folder)
    eval_folder = folders_path/folder_names[folder_idx]/evaluation_folder

    # eval_folder = Path(f"{eval_folder}_val")
    count = 0
    for i in range(n_imgs):
        print(f"{count}/{n_imgs}")
        pred_stack = imread(eval_folder/f"img_{i}_pred.tif")
        gt_stack = imread(eval_folder/f"img_{i}_gt.tif")
        t = pred_stack.shape[0]
        toremove = (max_cl - cl)
        n_frames = min(max_imgs_per_timelapse, t - toremove)
        
        start = toremove // 2  
        stop = start + n_frames
        
        print(f"Using #{n_frames} frames from {start} to {stop}")   

        for j in range(start,stop):

            gt = gt_stack[j]
            pred = pred_stack[j]

            if smooth_gt:
                gt = gaussian_filter (gt,sigma = 0.6,radius = 3)

            gt = (gt - np.mean(gt)) / np.std(gt)
            pred = (pred - np.mean(pred)) / np.std(pred)

            data_range = np.max(gt) - np.min(gt)
            psnr_val, ssim_val,mse_val = calculate_metrics(gt, pred, data_range,use_windowed,
                                                        window_size,patch_selection,range_invariant)
        
            psnr_values[cl].extend(psnr_val)
            ssim_values[cl].extend(ssim_val)
            mse_values[cl].extend(mse_val)

        count +=1

# do plots
fig,axs = plt.subplots(1,2,figsize=figsize)
fig.subplots_adjust(wspace=spaceBetweenSubplots)
plt.subplots_adjust(bottom=spaceBelowSubplots)

psnrs = [psnr_values[cl] for cl in cl_list]
ssims = [ssim_values[cl] for cl in cl_list]
mses = [mse_values[cl] for cl in cl_list]

fig=new_box_plot(psnrs,cl_list,fig=fig,ax=axs[0],ylabel="PSNR (dB)",**box_plot_parameters)
fig=new_box_plot(ssims,cl_list,fig=fig,ax=axs[1],ylabel="SSIM",**box_plot_parameters)
# fig=new_box_plot(mses,cl,fig=fig,ax=axs[2],title="MSE",**box_plot_parameters)


for ax in axs:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)


if show_figure:
    plt.show()  

# Saving
if save_figure:
    fig.savefig(save_folder / save_title, bbox_inches='tight')

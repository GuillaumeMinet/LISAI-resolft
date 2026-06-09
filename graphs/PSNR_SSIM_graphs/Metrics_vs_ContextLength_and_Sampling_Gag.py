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
folders_path = Path(r'E:\lisai\datasets\Gag_timelapses\models\Upsamp')

folder_names = [
    "CL1_Upsamp2_biggerNet_02",
    "CL3_Upsamp2_biggerNet_03",
    "CL5_Upsamp2_biggerNet_03",
    "CL7_Upsamp2_biggerNet_00",
    # "CL1_Upsamp05_biggerNet_01",
    # "CL1_Upsamp2_mltpl075_00",
]

evaluation_folder = "evaluation_best"
ambiguity_selector = "last_epoch"

conditions = ["N=1","N=3","N=5", "N=7"]

# metrics calculation parameters
smooth_gt = False
use_windowed = False
window_size = 600
range_invariant = False
patch_selection = False

# figure parameters
figsize = (10,5)
spaceBetweenSubplots=0.5
spaceBelowSubplots=0.2

colors_list = ["mediumblue","mediumblue","mediumblue","#0d9188ff","mediumblue"]


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
save_title = "Upsamp_PSNR_SSIM_MSE_vs_ContextLength_Gag.svg"


# get file idxs first
min_idx = 1e9
max_idx = 0

for folder in folder_names:
    eval_folder = get_eval_folder(folders_path/folder, evaluation_folder, ambiguity_selector)
    # eval_folder = Path(f"{eval_folder}_val")
    list_files = list_images(eval_folder)
    if list_files is None:
        raise RuntimeError("no images found in the eval_folder")
    print(len(list_files))
    if len(list_files) < min_idx:
        min_idx = len(list_files)
    if len(list_files) > max_idx:
        max_idx = len(list_files)

assert min_idx%3==0, "expected to have a folder with inp,gt,pred"
assert max_idx%3==0, "expected to have a folder with inp,gt,pred"

n_min = min_idx // 3
n_max = max_idx // 3



# Initialize dictionaries to store metrics for each context length
psnr_values = {cond: [] for cond in conditions}
ssim_values = {cond: [] for cond in conditions}
mse_values = {cond: [] for cond in conditions}

# metrics calculation
for folder_idx,cond in  enumerate(conditions):
    folder = folder_names[folder_idx]
    print("Folder: ", folder)
    eval_folder = get_eval_folder(folders_path/folder, evaluation_folder, ambiguity_selector)
    # eval_folder = Path(f"{eval_folder}_val")
    n_files = len(list_images(eval_folder)) // 3
    
    start = (n_files - n_min) // 2
    stop = start + n_min
    idxs = np.arange(start,stop,1)
    print(idxs)
    
    effective_files = 0
    for i in (idxs):
        effective_files+=1

        gt = imread(eval_folder/f"img_{i}_gt.tif")
        pred = imread(eval_folder/f"img_{i}_pred.tif")

        min_pred = np.min(pred)
        gt[gt<min_pred*1.1]=min_pred*1.1
        if smooth_gt:
            gt = gaussian_filter (gt,sigma = 0.5,radius = 3)

        gt = (gt - np.mean(gt)) / np.std(gt)
        pred = (pred - np.mean(pred)) / np.std(pred)

        data_range = np.max(gt) - np.min(gt)
        psnr_val, ssim_val,mse_val = calculate_metrics(gt, pred, data_range,use_windowed,
                                                        window_size,patch_selection,range_invariant)
        
        psnr_values[cond].extend(psnr_val)
        ssim_values[cond].extend(ssim_val)
        mse_values[cond].extend(mse_val)
        
    print(f"Used #{effective_files} files")

# do plots
fig,axs = plt.subplots(1,2,figsize=figsize)
fig.subplots_adjust(wspace=spaceBetweenSubplots)
plt.subplots_adjust(bottom=spaceBelowSubplots)

psnrs = [psnr_values[cond] for cond in conditions]
ssims = [ssim_values[cond] for cond in conditions]
mses = [mse_values[cond] for cond in conditions]

fig=new_box_plot(psnrs,conditions,fig=fig,ax=axs[0],ylabel="PSNR (dB)",**box_plot_parameters)
fig=new_box_plot(ssims,conditions,fig=fig,ax=axs[1],ylabel="SSIM",**box_plot_parameters)
# fig=new_box_plot(mses,conditions,fig=fig,ax=axs[2],title="MSE",**box_plot_parameters)


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


save_path = folders_path/ "Metrics_plots_GAG_CL1-3-5_and_S05.svg"
fig.savefig(save_path, format="svg", bbox_inches='tight')
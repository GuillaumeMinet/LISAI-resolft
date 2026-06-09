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
from lisai.data.utils import crop_center, extract_patches

import frc
import numpy as np
import matplotlib.pyplot as plt
from tifffile import imread
from pathlib import Path
import random
from scipy.ndimage import gaussian_filter

from matplotlib.lines import Line2D
from matplotlib.legend import Legend

# data paths
folders_path = Path(r'E:\lisai\datasets\vim_live\models\Upsamp_selected')

folder_names = [
    "Fulldataset_CL1_1FramesMax_Upsamp2_smallerNet_clip_modifUpsamp_03",
    # "Fulldataset_CL3_3FramesMax_Upsamp2_smallerNet_clip_06",
    "Fulldataset_CL5_5FramesMax_Upsamp2_smallerNet_clip_02",
    # "Fulldataset_CL7_7FramesMax_Upsamp2_smallerNet_clip_06"
]

evaluation_folder = "eval_gathered"

max_imgs_per_timelapse = 10

cl_list = [1,5]#"5","7"]
max_cl = max(cl_list)

smooth_gt = True
show_figure = True
plot_gt = True

# saving parameters
save_figure = True
save_folder = PROJECT_ROOT / "graphs" / "saved_graphs"
save_title = "Vim_upsamp_FRCvsCL.svg"
legend_save_title = f"{save_title.split('.')[0]}_legend.svg"



# intialize frc dict
frc_curves = {cl: [] for cl in cl_list}
frc_curves["gt"] = []
avg_frc_curves = {cl: [] for cl in cl_list}
avg_frc_curves["gt"] = []
std_frc_curves = {cl: [] for cl in cl_list}
std_frc_curves["gt"] = []

# gt calculation
eval_folder = folders_path/folder_names[0]/evaluation_folder
list_files = list_images(eval_folder)
if len(list_files) % 3 != 0:
    raise ValueError
n_imgs = len(list_files) // 3

count = 0
for i in range(n_imgs):
    print(f"GT {count}/{n_imgs}")
    
    stack = imread(eval_folder/f"img_{i}_gt.tif")
    t = stack.shape[0]
    toremove = (max_cl - cl_list[0])
    n_frames = min(max_imgs_per_timelapse, t - toremove)
    
    start = toremove // 2  
    stop = start + n_frames

    print(f"Using #{n_frames} frames from {start} to {stop}")

    for j in range(start,stop):
        gt = stack[j]
        if smooth_gt:
                gt = gaussian_filter (gt,sigma = 0.6,radius = 3)
        gt = (gt - np.mean(gt)) / (np.std(gt))
        gt = gt - np.min(gt)
        frc_curve = frc.one_frc(gt)
        if np.isnan(frc_curve).any():
            print(f"Skipping a patch in gt {i} due to NaN values in FRC curve.")
        else:
            frc_curve = (frc_curve - np.min(frc_curve)) / (np.max(frc_curve) - np.min(frc_curve))
            frc_curves["gt"].append(frc_curve)
    count+=1


# prediction FRC
for folder_idx,cl in  enumerate(cl_list):
    count = 0
    eval_folder = folders_path/folder_names[folder_idx]/evaluation_folder

    for i in range(n_imgs):
        
        print(f"Prediction {count}/{n_imgs} in context length = {cl}")

        stack = imread(eval_folder/f"img_{i}_pred.tif")
        t = stack.shape[0]
        toremove = (max_cl - cl)
        n_frames = min(max_imgs_per_timelapse, t - toremove)
    
        start = toremove // 2  
        stop = start + n_frames

        print(f"Using #{n_frames} frames from {start} to {stop}")

        for j in range(start,stop):
            pred = stack[j]
            pred = (pred - np.mean(pred)) / np.std(pred)
            pred = pred - np.min(pred)
            pred = frc.util.apply_tukey(pred)
            frc_curve = frc.one_frc(pred)
            if np.isnan(frc_curve).any():
                print(f"Skipping image {i} due to NaN values in FRC curve.")
            else:
                frc_curve = (frc_curve - np.min(frc_curve)) / (np.max(frc_curve) - np.min(frc_curve))
                frc_curves[cl].append(frc_curve)
        count+=1

print(len(frc_curves["gt"]))
frc_curves["gt"] = np.stack(frc_curves["gt"])
avg_frc_curves["gt"] = np.mean(frc_curves["gt"], axis=0)
std_frc_curves["gt"] = np.std(frc_curves["gt"], axis=0)

for cl in cl_list:
    frc_curves[cl] = np.stack(frc_curves[cl])
    avg_frc_curves[cl] = np.mean(frc_curves[cl], axis=0)
    std_frc_curves[cl] = np.std(frc_curves[cl], axis=0)
    
# colors_list = ['mediumblue', 'darkred','forestgreen','pink',"cyan"]


colors_list = ['mediumblue', "#0d9188ff"] #'darkred','forestgreen','pink']
linewidth = 0.7
fig,ax = plt.subplots(1, 1, figsize=(1.4, 1.4))


# Plot 1frc 
scale = 1/35*1e3
img_size = pred.shape

# gt plot
if plot_gt:
    frc_curve = avg_frc_curves["gt"]
    std_curve = std_frc_curves["gt"]
    xs_pix = np.arange(len(frc_curve)) / img_size[0]
    xs_nm_freq = xs_pix * scale
    ax.plot(xs_nm_freq, frc_curve,label="gt",linewidth=linewidth*0.6,color="black")
    ax.fill_between(xs_nm_freq, frc_curve - std_curve, frc_curve + std_curve, alpha=0.2,
                    color="black",linewidth=0)


# preds plot
for cl in cl_list:
    frc_curve = avg_frc_curves[cl]
    std_curve = std_frc_curves[cl]
    xs_pix = np.arange(len(frc_curve)) / img_size[0]
    xs_nm_freq = xs_pix * scale
    ax.plot(xs_nm_freq, frc_curve,label=cl,linewidth=linewidth,color=colors_list[cl_list.index(cl)])
    ax.fill_between(xs_nm_freq, frc_curve - std_curve, frc_curve + std_curve, alpha=0.3,
                    color=colors_list[cl_list.index(cl)],
                    linewidth=0)


labels_fontSize=8
ticks_prms={"labelsize":10, "width":linewidth,"length":3*linewidth}
# ax.set_title("1FRC")
ax.set_xlabel("Spatial frequency (µm$^{-1}$)",fontsize=labels_fontSize)
ax.set_ylabel("Correlation",fontsize=labels_fontSize)
ax.set_xlim(0, 14)
ax.set_ylim(0, 1)
# ax.legend()

ax.set_xticks([0,5,10])
ax.set_yticks([0,0.5,1])
ax.tick_params(axis='x',which='major',**ticks_prms)
ax.tick_params(axis='y',which='major',**ticks_prms)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_linewidth(linewidth)
plt.gca().spines['bottom'].set_linewidth(linewidth)

if show_figure:
    plt.show()  

# Saving
if save_figure:
    fig.savefig(os.path.join(save_folder, save_title), bbox_inches='tight')

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


from lisai.data.utils import extract_patches
from scipy.ndimage import gaussian_filter

import frc
import numpy as np
import matplotlib.pyplot as plt
from tifffile import imread
from pathlib import Path
import random

from matplotlib.lines import Line2D
from matplotlib.legend import Legend

# data paths
folders_path = Path(r'E:\dl_monalisa\Models\Mito_34nm_timelapses\Upsampling')

folder_names = [
    "10FramesOnly_Upsamp2_RandomPx_UnetRCAN_rg8_rcab12_red16_CharEdge_alpha005",
    # "CL3_Upsamp2_RandomPx_UnetRCAN_rg8_rcab12_red16_CharEdge_alpha005_02",
    # "CL5_Upsamp2_RandomPx_UnetRCAN_rg8_rcab12_red16_CharEdge_alpha005",
    "CL5_Upsamp2_RandomPx_UnetRCAN_rg8_rcab12_red16_CharEdge_alpha005_01",
    # "CL5_Upsamp2_RandomPx_UnetRCAN_rg8_rcab12_red16_CharEdge_alpha005_02",
    ]

evaluation_foder = "evaluation_last"


cl_list = ["N=1", "N=5"]


n_imgs = 100
random_imgs = True
patch_size = 300
show_figure = False
plot_gt = True
smooth_gt = True

# saving parameters
save_figure = True
save_legend = False

save_folder = PROJECT_ROOT / "graphs" / "saved_graphs"
save_title = "Upsamp_FRC_vs_ContextLength_Mito.svg"
legend_save_title = f"{save_title.split('.')[0]}_legend.svg"
save_legend = True

# get file idxs first
min_idx = 1e9
for folder in folder_names:
    eval_folder = folders_path/folder/evaluation_foder
    list_files = os.listdir(eval_folder)
    print(len(list_files))
    if len(list_files) < min_idx:
        min_idx = len(list_files)

assert min_idx%3==0, "expected to have a folder with inp,gt,pred"
n = min_idx//3
if n_imgs is not None and n_imgs < n:
    if random_imgs:
        idxs = random.sample(range(n), n_imgs)
    else:
        idxs = np.arange(n_imgs)
else:
    idxs = np.arange(n)


# intialize frc dict
frc_curves = {cl: [] for cl in cl_list}
frc_curves["gt"] = []
avg_frc_curves = {cl: [] for cl in cl_list}
avg_frc_curves["gt"] = []
std_frc_curves = {cl: [] for cl in cl_list}
std_frc_curves["gt"] = []

# gt calculation
eval_folder = folders_path/folder_names[0]/evaluation_foder
count = 0
for i in idxs:
    print(f"GT {count}/{len(idxs)}")
    count+=1
    gt = imread(eval_folder/f"img_{i}_gt.tif")
    if smooth_gt:
            gt = gaussian_filter (gt,sigma = 0.4,radius = 3)
    gt = (gt - np.mean(gt)) / (np.std(gt))
    gt = gt - np.min(gt)
    patches_gt = extract_patches(gt,patch_size)
    for i in range(patches_gt.shape[0]):
        patch_gt = frc.util.apply_tukey(patches_gt[i])
        frc_curve = frc.one_frc(patch_gt)
        if np.isnan(frc_curve).any():
            print(f"Skipping a patch in gt {i} due to NaN values in FRC curve.")
        else:
            frc_curve = (frc_curve - np.min(frc_curve)) / (np.max(frc_curve) - np.min(frc_curve))
            frc_curves["gt"].append(frc_curve)


# prediction FRC
for folder_idx,cl in  enumerate(cl_list):
    count = 0
    eval_folder = folders_path/folder_names[folder_idx]/evaluation_foder
    for i in idxs:
        print(f"Predictino {count}/{len(idxs)} in {cl} context length")
        count+=1
        pred = imread(eval_folder/f"img_{i}_pred.tif")
        pred = (pred - np.mean(pred)) / np.std(pred)
        pred = pred - np.min(pred)
        patches = extract_patches(pred,patch_size)

        for i in range(patches.shape[0]):
            patch = frc.util.apply_tukey(patches[i])
            frc_curve = frc.one_frc(patch)
            if np.isnan(frc_curve).any():
                print(f"Skipping a patch in image {i} due to NaN values in FRC curve.")
            else:
                frc_curve = (frc_curve - np.min(frc_curve)) / (np.max(frc_curve) - np.min(frc_curve))
                frc_curves[cl].append(frc_curve)

frc_curves["gt"] = np.stack(frc_curves["gt"])
avg_frc_curves["gt"] = np.mean(frc_curves["gt"], axis=0)
std_frc_curves["gt"] = np.std(frc_curves["gt"], axis=0)

for cl in cl_list:
    frc_curves[cl] = np.stack(frc_curves[cl])
    avg_frc_curves[cl] = np.mean(frc_curves[cl], axis=0)
    std_frc_curves[cl] = np.std(frc_curves[cl], axis=0)
    

colors_list = ['mediumblue', "#0d9188ff"] #'darkred','forestgreen','pink']
linewidth = 0.7

scale = 1/34*1e3
img_size = pred.shape
fig,ax = plt.subplots(1, 1, figsize=(1.4, 1.4))

# gt plot
if plot_gt:
    frc_curve = avg_frc_curves["gt"]
    std_curve = std_frc_curves["gt"]
    xs_pix = np.arange(len(frc_curve)) / img_size[0]
    xs_nm_freq = xs_pix * scale
    ax.plot(xs_nm_freq, frc_curve,label="gt",linewidth=linewidth*0.6,color="black")
    ax.fill_between(xs_nm_freq, frc_curve - std_curve, frc_curve + std_curve, alpha=0.2,
                    color="black",linewidth=0)



# Plot 1frc 
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
    print("saved_figure")

# save_path = folders_path/ "Upsamp_Mitoch_Metrics_vs_CL.svg"
# fig.savefig(save_path, format="svg", bbox_inches='tight')

# labels_list = [
#     "N=1",
#     "N=5"
# ]

# if save_legend:
#     custom_lines = [
#         Line2D([0], [0], color=colors_list[i], lw=4, solid_capstyle='butt', label=labels_list[i])
#         for i in range(len(labels_list))
#     ]

#     legend_fig = plt.figure(figsize=(2, 2))
#     legend_ax = legend_fig.add_subplot(111)
#     legend_ax.axis('off')
#     legend = Legend(legend_ax, custom_lines, labels_list, loc='center', frameon=False, fontsize=16, handlelength=1.5, handleheight=1.0)
#     legend_ax.add_artist(legend)

#     legend_fig.savefig(os.path.join(save_folder, legend_save_title), bbox_inches='tight')

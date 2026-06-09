import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
import frc
import numpy as np
import matplotlib.pyplot as plt
from tifffile import imread
from scipy.ndimage import gaussian_filter

folder_path = r'E:\dl_monalisa\Data\Vim_fixed_mltplSNR_30nm\Evaluation\FRC_UpsampResults'

scale = 1/30*1e3 # um-1

list_conditions = ["GT", "S=0.25", "S=0.5", "S=0.75"]
smooth_gt=True

show_figure = True

# saving parameters
save_figure = True
save_folder = PROJECT_ROOT / "graphs" / "saved_graphs"
save_title = "Upsamp_1FRC_Sampling_highSNRonly.svg"


# Process each image stack
files = os.listdir(folder_path)
all_frc_curves = []
for file_idx,file_name in enumerate(files):
    if file_name.endswith('.tif'):
        file_path = os.path.join(folder_path, file_name)
        img_stack = imread(file_path)
        print(img_stack.shape)
        frc_curves = []

        # calculate frc curves for each pred
        for j, cond in enumerate(list_conditions):
            img = img_stack[j]
            # Ground truth (first image in the stack)
            if cond == "GT":
                if smooth_gt:
                    img = gaussian_filter (img,sigma = 0.6,radius = 3)
                img[img<0]=0

            img = (img - np.mean(img)) / np.std(img)
            img = img - np.min(img)
            img = frc.util.apply_tukey(img)
            frc_curve = frc.one_frc(img)
            frc_curve = (frc_curve - np.min(frc_curve)) / (np.max(frc_curve) - np.min(frc_curve))
            frc_curves.append(frc_curve)
        
        all_frc_curves.append(np.stack(frc_curves))
        
all_frc_curves = np.stack(all_frc_curves)
avg_frc_curves = np.mean(all_frc_curves,axis=0)
std_frc_curves = np.std(all_frc_curves, axis=0)

img_size = img.shape[0]
colors = ["black","darkred","mediumblue","forestgreen"]

fig,ax = plt.subplots(1, 1, figsize=(5, 5))

for j,cond in enumerate(list_conditions):
    frc_curve = avg_frc_curves[j]
    std_curve = std_frc_curves[j]

    xs_pix = np.arange(len(frc_curve)) / img_size
    xs_nm_freq = xs_pix * scale
    ax.plot(xs_nm_freq, frc_curve,label=cond,linewidth=1.5,color=colors[j])
    ax.fill_between(xs_nm_freq, frc_curve - std_curve, frc_curve + std_curve, 
                        color=colors[j],alpha=0.3, linewidth=0)
    
labels_fontSize=20
ticks_prms={"labelsize":20, "width":2,"length":8}
ticks_positions=[0,5,10,15]
# ax.set_title("1FRC")
ax.set_xlabel("Spatial frequency (µm$^{-1}$)",fontsize=labels_fontSize)
ax.set_ylabel("Correlation",fontsize=labels_fontSize)
ax.set_xlim(0, 15)
ax.set_ylim(0, 1)
ax.legend()

ax.set_xticks(ticks_positions)
ax.tick_params(axis='x',which='major',**ticks_prms)
ax.tick_params(axis='y',which='major',**ticks_prms)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_linewidth(2)
plt.gca().spines['bottom'].set_linewidth(2)

if show_figure:
    plt.show()

# Saving
if save_figure:
    fig.savefig(os.path.join(save_folder, save_title), bbox_inches='tight')


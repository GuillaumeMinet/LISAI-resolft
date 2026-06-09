import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from lisai.data.utils import crop_center
import frc
import numpy as np
import matplotlib.pyplot as plt
from tifffile import imread
from pathlib import Path
from scipy.ndimage import gaussian_filter

stack_path= r"E:\dl_monalisa\Models\Vim_fixed_mltplSNR_30nm\GTqualityIssue\summary_stack.tif"

stack = imread(stack_path)

conditions = [
    "care_gt1", 
    "care_gt2", 
    "HDN_gt1",
    "HDN_gt2",
    "unetrcan_gt1", 
    "unetrcan_gt2",
    ]

display_order = [0,2,1]

crop_size = 1200
show_figure = True
# saving parameters

save_figure = True
save_folder = PROJECT_ROOT / "graphs" / "saved_graphs"
save_title = "Vim_FRC_differentGT.svg"
colors = ["#33c4b8","#b626aa"]

scale = 1/30*1e3
img_size = stack.shape[1::]
fig,axs = plt.subplots(3,1,figsize=(1, 4))
fig.subplots_adjust(hspace=0.5,left=0,right=0.9)

for i,cond in enumerate(conditions):
    ax_idx = display_order[i//2]
    ax = axs[ax_idx]
    img = stack[i]
    img = (img - np.mean(img)) / np.std(img)
    img = img - np.min(img)
    img = frc.util.apply_tukey(img)
    frc_curve = frc.one_frc(img)
    frc_curve = (frc_curve - np.min(frc_curve)) / (np.max(frc_curve) - np.min(frc_curve)) 
    
    
    xs_pix = np.arange(len(frc_curve)) / img_size[0]
    xs_nm_freq = xs_pix * scale

    ax.plot(xs_nm_freq,frc_curve, label=cond,color=colors[i%2],
            linewidth=1.0)
    # ax.legend()


linewidth = 1.2
labels_fontSize = 9
ticks_prms={"labelsize":labels_fontSize, "width":linewidth,"length":2*linewidth}


axs[-1].set_xlabel("Spatial freq. (µm$^{-1}$)",fontsize=labels_fontSize)
for ax in axs:
    ax.set_xticks([0,5,10,15])
    ax.set_yticks([0,0.5,1.0])
    ax.set_xlim([0,15])
    ax.set_ylim([0,1])
    ax.tick_params(axis='x',which='major',**ticks_prms)
    ax.tick_params(axis='y',which='major',**ticks_prms)
    ax.set_ylabel("Correlation",fontsize=labels_fontSize)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(linewidth)
    ax.spines['bottom'].set_linewidth(linewidth)


# plt.show()

if save_figure:
    fig.savefig(save_folder / save_title, bbox_inches='tight')

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

from graphs.utils.single_lineProfile_fit import perform_fit

raw_data = """
0	2589.254	3600.326	458.45	-0.354561	1233.204	1154.646
1	1652.042	2719.623	674.894	-0.382448	1106.147	1138.886
2	1298.471	2372.985	974.258	-0.392317	1122.837	1186.607
3	1193.466	2328.501	549.355	-0.391648	1186.911	1327.226
4	1438.2	2701.155	784.023	-0.372642	1510.831	1546.734
5	1890.71	3575.149	1043.803	-0.333141	2375.539	1977.095
6	3782.749	5042.279	2732.239	-0.290574	3839.716	2624.043
7	8098.093	6328.166	4177.515	-0.231621	4867.665	3536.706
8	12033.406	7108.675	8618.628	-0.150318	11072.304	4403.758
9	9821.697	6425.357	14028.449	-0.11604	16022.278	4617.31
10	4564.163	4736.597	9167.293	-0.14744	8662.53	3922.375
11	2072.756	3391.732	3496.303	-0.235779	4095.038	2673.41
12	1327.54	2408.731	436.588	-0.317071	2065.685	1823.648
13	1159.374	1654.455	173.05	-0.362974	1501.71	1397.102
14	1254.419	1244.178	347.639	-0.392361	1321.734	1163.519
15	1133.013	1339.661	1796.671	-0.407553	1116.407	976.757
16	721.628	1424.282	1185.759	-0.416491	990.944	873.263
17	500.258	1426.432	13.95	-0.421425	932.32	799.191
18	537.776	1389.695	0	-0.406484	867.456	738.665
"""

def parse_clipboard_data(text):
    """
    Converts tabular text input (e.g. from copy-pasting) into x and y NumPy arrays.
    Assumes tab or space-separated two-column data.
    """
    # Split lines and filter out empty lines
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    # Split each line by whitespace or tab
    data = [list(map(float, line.split())) for line in lines]
    arr = np.array(data)
    return arr

pxSize = 28


arr = parse_clipboard_data(raw_data)
x = arr[:,0] 
x = x * pxSize

fig,axs = plt.subplots(3,1,figsize=(1.2, 4.2))
fig.subplots_adjust(hspace=0.5,left=0,right=0.9)

save_figure = True
save_folder = PROJECT_ROOT / "graphs" / "saved_graphs"
save_title = "Vim_lineProfiles_differentGT.svg"

linewidth = 1.2
labels_fontSize = 9
ticks_prms={"labelsize":labels_fontSize, "width":linewidth,"length":2*linewidth}
colors = ["#33c4b8","#b626aa"]

for i in range(1,arr.shape[1]):
    ax = axs[(i-1)//2]
    color = colors[(i+1)%2]
    y = arr[:,i]
    y=(y-np.min(y))/(np.max(y)-np.min(y))  # Normalize y
    
    plot_curve,fwhm = perform_fit(x,y)
    ax.plot(x,y,color, marker='+',markersize=6,alpha=0.7,linewidth=0)
    ax.plot(plot_curve[0],plot_curve[1],color, linewidth=1)

    f = (-1)**(i%2) * 0.12
    ax.text(350, 0.5+f, f'{fwhm:.0f}nm', color=color,fontsize = labels_fontSize+1)


axs[-1].set_xlabel("x (nm)",fontsize=labels_fontSize)
for ax in axs:
    ax.set_xticks([0,250,500])
    ax.set_yticks([0,0.5,1.0])
    ax.set_xlim([0,600])
    ax.set_ylim([0,1.2])
    ax.tick_params(axis='x',which='major',**ticks_prms)
    ax.tick_params(axis='y',which='major',**ticks_prms)
    ax.set_ylabel("Norm. Fluo",fontsize=labels_fontSize)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(linewidth)
    ax.spines['bottom'].set_linewidth(linewidth)

# plt.show()

if save_figure:
    fig.savefig(save_folder / save_title, bbox_inches='tight')



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
0	-0.24602	-0.21734	-0.18336
1	-0.31687	-0.21726	-0.18175
2	-0.22935	-0.21696	-0.18213
3	-0.17836	-0.21461	-0.18302
4	-0.16414	-0.19221	-0.18277
5	-0.3886	-0.05426	-0.17594
6	-0.00573	0.1945	-0.12212
7	0.98769	0.6845	0.04982
8	1.27034	0.85142	0.08123
9	0.21722	0.60594	0.19938
10	0.70815	0.77221	0.88832
11	1.12892	1.0832	1.99184
12	0.92073	0.96629	1.75552
13	-0.07552	0.48078	0.6544
14	-0.45427	-0.07986	-0.03033
15	-0.52983	-0.20995	-0.17235
16	-0.56018	-0.21655	-0.18507
17	-0.55375	-0.21717	-0.18504
18	-0.37253	-0.21725	-0.18542
19	-0.21827	-0.21731	-0.18431
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

pxSize = 35


arr = parse_clipboard_data(raw_data)
x = arr[:,0] 
x = x * pxSize

fig,ax = plt.subplots(1,1,figsize=(2,2))
save_figure = True
save_folder = PROJECT_ROOT / "graphs" / "saved_graphs"
save_title = "Vim_2filaments_lineprofile.svg"

linewidth = 1.2
labels_fontSize = 9
ticks_prms={"labelsize":labels_fontSize, "width":linewidth,"length":2*linewidth}
colors = ["black","#0d9188ff","#0000cdff"]

for i in range(1,arr.shape[1]):
    color = colors[(i+1)%2]
    y = arr[:,i]
    y=(y-np.min(y))/(np.max(y)-np.min(y))  # Normalize y

    plt.plot(x,y,color=colors[i-1],linewidth=1)


ax.set_xlabel("x (nm)",fontsize=labels_fontSize)
ax.set_xticks([0,200,400,600])
ax.set_yticks([0,0.5,1.0])
ax.set_xlim([0,600])
ax.set_ylim([0,1.05])
ax.tick_params(axis='x',which='major',**ticks_prms)
ax.tick_params(axis='y',which='major',**ticks_prms)
ax.set_ylabel("Norm. Fluo",fontsize=labels_fontSize)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(linewidth)
ax.spines['bottom'].set_linewidth(linewidth)

plt.show()

if save_figure:
    fig.savefig(save_folder / save_title, bbox_inches='tight')


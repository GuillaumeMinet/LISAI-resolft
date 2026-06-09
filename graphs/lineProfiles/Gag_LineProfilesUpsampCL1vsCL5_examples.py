
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
0	0.020807	0.000162	0.001964
1	0.029386	0.000949	0.001639
2	0.057522	0.007715	0.003386
3	0.061656	0.042512	0.012182
4	0.09841	0.118225	0.052234
5	0.202759	0.197348	0.154134
6	0.454581	0.312197	0.297146
7	0.746495	0.434653	0.506278
8	0.842704	0.584617	0.675226
9	0.729523	0.657886	0.649859
10	0.620831	0.634842	0.508191
11	0.467293	0.564021	0.368588
12	0.383966	0.577917	0.355896
13	0.469159	0.728168	0.597371
14	0.710063	0.900228	0.775337
15	0.727325	0.850513	0.680273
16	0.665032	0.595707	0.392287
17	0.401515	0.316994	0.194785
18	0.177729	0.161084	0.110668
19	0.140232	0.089669	0.065899
20	0.135503	0.049673	0.037851
21	0.084554	0.026135	0.023825
22	0.066057	0.020945	0.015152
23	0.101808	0.0274	0.015751
24	0.118338	0.036996	0.019217
25	0.117495	0.030971	0.01334
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

pxSize = 34


arr = parse_clipboard_data(raw_data)
x = arr[:,0] 
x = x * pxSize

fig,ax = plt.subplots(1,1,figsize=(2,2))
save_figure = True
save_folder = PROJECT_ROOT / "graphs" / "saved_graphs"
save_title = "Gag_lineprofiles_CL1vsCL5.svg"

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
ax.set_xticks([0,250,500,750])
ax.set_yticks([0,0.5,1.0])
ax.set_xlim([0,800])
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


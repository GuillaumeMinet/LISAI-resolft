
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
0	-0.355975	-0.223675	-0.26494
1	-0.329927	-0.223575	-0.264686
2	-0.324815	-0.223601	-0.264042
3	-0.33569	-0.222892	-0.262144
4	-0.351139	-0.219962	-0.255277
5	-0.343072	-0.207352	-0.236568
6	-0.265507	-0.158883	-0.189406
7	-0.10433	-0.047718	-0.066768
8	0.103916	0.070466	0.116773
9	0.261486	0.109031	0.208955
10	0.248612	0.080001	0.129475
11	0.081744	0.069848	0.08174
12	0.219745	0.112621	0.154787
13	0.319667	0.108443	0.218575
14	0.230937	-0.013458	0.11604
15	0.12615	-0.126905	-0.045093
16	-0.100926	-0.187186	-0.156693
17	-0.289886	-0.207002	-0.185521
18	-0.304018	-0.217667	-0.186681
19	-0.271721	-0.221101	-0.187499
20	-0.245894	-0.22216	-0.215516
21	-0.290142	-0.222896	-0.244749
22	-0.339646	-0.22323	-0.260543

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
save_title = "Mito_vesicle_lineprofile.svg"

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


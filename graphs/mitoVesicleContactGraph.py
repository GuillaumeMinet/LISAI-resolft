slow = [
    0.131578947,
    0.166666667,
    0.15,
    0.05,
    0.192307692,
]

fast = [
    0.315789474,
    0.444444444,
    0.35,
    0.175,
    0.5,
]


slow = [5,6,3,2,5]
fast = [12,16,7,7,13]


import matplotlib.pyplot as plt
import numpy as np
import os,sys
sys.path.append(os.getcwd())
from lisai.graphs.utils.boxplot import box_plot

fig=plt.figure(figsize=(1.2, 1.5))  # Adjust the figure size to make it smaller
data = [slow, fast]
labels = ['Regular', 'Fast']

colors = ['black','#900400f1']  # Set all boxplot elements to black

fontsize= 5.5
ticks_prms={"labelsize":fontsize, "width":0.7,"length":2}

fig = box_plot(data, labels=labels, fig=fig, use_mean=True,
               colors=colors,
               showDataPoints=True, dataPoints_color='same', 
               dataPoints_size=17, dataPoints_alpha=0.6,
               linewidth=1,positions=[0.25,0.5],widths=0.15,
               ylabel='#MDV-mitoch. contacts ', xlabel=None, title=None,
               labels_fontSize=fontsize,
               ticks_prms=ticks_prms)



saving=True
save_folder = os.path.join(os.getcwd(), r"src/graphs/saved_graphs")
save_title = "3dMitoch_boxplot_slow_fast.svg"

plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_linewidth(0.7)
plt.gca().spines['bottom'].set_linewidth(0.7)

# Change the color of the first x tick label to red
# xticks = plt.gca().get_xticklabels()
# xticks[1].set_color(colors[1])

plt.yticks([0,5,10,15])

if saving:
    fig.savefig(os.path.join(save_folder, save_title), bbox_inches='tight')

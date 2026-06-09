import os,sys
sys.path.append(os.getcwd())
import csv
import numpy as np
import matplotlib.pyplot as plt
from lisai.graphs.utils.lineProfileAnalysis import lineProfileAnalysis

path = r'E:\dl_monalisa\Data\Vim_fixed_mltplSNR_30nm\SNR\examples\data.txt'
pxSize = 0.030 #nm

# saving parameters
save_figure = True
save_folder = os.path.join(os.getcwd(), r"src/graphs/saved_graphs")
save_title = "SNR_LineProfilesExamples.svg"

fig, axes = plt.subplots(4,3,figsize=(10,10),constrained_layout=True)

titles=["High Signal", "Medium Signal", "Low Signal"]*4

fwhms,snrs,used_axes=lineProfileAnalysis(path, pxSize, do_plots=True,show_plots=True,
                        fig=fig, axes=axes.flatten(), start_index=0,min_fwhm=0.040,max_snr=20,
                        replace_empty_with_zeros=True,header=False,titles_list=titles)

# cosmetic

for ax in axes:    
    ...
    # ticks_prms={"labelsize":10, "width":2,"length":8}
    # ax.tick_params(axis='x',which='major',**ticks_prms)
    # ax.tick_params(axis='y',which='major',**ticks_prms)

# Saving
if save_figure:
    fig.savefig(os.path.join(save_folder, save_title), bbox_inches='tight')


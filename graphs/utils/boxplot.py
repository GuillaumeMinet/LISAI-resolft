import numpy as np
import matplotlib.pyplot as plt



def box_plot(datasets,labels,fig=None,ax=None,mltpl_plots=False,n_plots=None,plot_idx=None,figsize=(6.4,4.8),
             use_mean=False,colors=None,showfliers=False,showMeanAndStd=False,reverse_dataset_order=False,custom_order=None,
             showMeanAndStd_pos="inside",positions=None,widths=0.3,mltpl_displacements=0.4,
             ylim=None,xlabel=None, ylabel=None,title=None,labels_angle=0,dashed_whiskers=False,linewidth=2,
             labels_fontSize=None,showMeanAndStd_fontsize=None,ticks_prms=None,
             showDataPoints=True,dataPoints_color='same',dataPoints_size=15,dataPoints_alpha=1):
    

    if mltpl_plots:
        assert n_plots is not None, "n_plots must be specified when mltpl_plots is True"
        assert plot_idx is not None, "plot_idx must be specified when mltpl_plots is True"

    if reverse_dataset_order:
        datasets = datasets[::-1]
        labels = labels[::-1]
    elif custom_order is not None:
        datasets = [datasets[i] for i in custom_order]
        labels = [labels[i] for i in custom_order]

    if positions is None:
        positions = np.arange(1, len(datasets) + 1)

    if fig is None:
        assert ax is None
        fig,ax = plt.subplots(1,1,figsize=figsize)
    elif ax is None:
        ax = fig.gca()
    
    if mltpl_plots:
        displacement = (plot_idx - n_plots//2) * mltpl_displacements
        positions = [pos + displacement for pos in positions]
    
    if colors is None:
        colors = ['black'] * len(datasets)
    elif isinstance(colors, str):
        colors = [colors] * len(datasets)

    if use_mean:
        means = [np.mean(data) for data in datasets]
        box = ax.boxplot(datasets, labels=None, patch_artist=True, showfliers=showfliers,
                        positions=positions, widths=widths, manage_ticks=False, medianprops=dict(color='none'))
        for idx, (mean, pos) in enumerate(zip(means, positions)):
            color = colors[idx]
            ax.plot([pos - widths/2, pos + widths/2], [mean, mean], color=color, linewidth=linewidth, zorder=3)
    else:
        box = ax.boxplot(datasets, labels=None, patch_artist=True, showfliers=showfliers,
                        positions=positions, widths=widths, manage_ticks=False)



    for patch, color in zip(box['boxes'], colors):
        patch.set(facecolor='none', edgecolor=color, linewidth=linewidth)
    for whisker, color in zip(box['whiskers'], [c for c in colors for _ in (0, 1)]):
        if dashed_whiskers:
            whisker.set(color=color, linewidth=linewidth*0.7, linestyle=(0, (3, 3)))
        else:
            whisker.set(color=color, linewidth=linewidth) 
    for cap, color in zip(box['caps'], [c for c in colors for _ in (0, 1)]):
        cap.set(color=color, linewidth=linewidth)
    if not use_mean:
        for median, color in zip(box['medians'], colors):
            median.set(color=color, linewidth=linewidth)
    for flier, color in zip(box['fliers'], colors):
        flier.set(markeredgecolor=color)

    if showDataPoints:        
        for idx, group in enumerate(datasets):
            jitter = np.random.uniform(-0.01, 0.01, size=len(group))  # Add jitter to x-coordinates
            if dataPoints_color == 'same':
                color = colors[idx] if colors else 'black'
            else:
                color = dataPoints_color

            ax.scatter([positions[idx] + j for j in jitter], group, alpha=dataPoints_alpha,
                       color=color, s=dataPoints_size, edgecolors='none')

    if showMeanAndStd:
        data_max = np.max([np.max(data) for data in datasets])
        data_min = np.min([np.min(data) for data in datasets])
        range = data_max - data_min
        for idx, data in enumerate(datasets):
            if use_mean:
                mean = np.mean(data)
            else:
                median = np.median(data)
            std = np.std(data)
            if showMeanAndStd_pos == "above":
                y_pos = data_max + 0.05 * range
            elif showMeanAndStd_pos == "below":
                y_pos = data_min - 0.07 * range
            elif showMeanAndStd_pos == "inside":
                y_pos = np.median(datasets[idx]) + (np.percentile(datasets[idx],75) - np.median(datasets[idx]))/2 - 0.1 # inside box
            else:
                raise ValueError("showMeanAndStd_pos must be 'above', 'inside', or 'below'.")
            
            if showMeanAndStd_fontsize is None:
                fontsize = 8 if showMeanAndStd_pos == "inside" else 10
            else:
                fontsize=showMeanAndStd_fontsize

            ax.text(positions[idx], y_pos, f"{mean:.1f} ± {std:.1f}",
                    ha='center', va='bottom', fontsize=fontsize, fontweight='bold',
                    color=colors[idx] if colors else 'black')
    
    if ylim is None and showMeanAndStd:
        if showMeanAndStd and showMeanAndStd_pos == "above":
            min = data_min - 0.05 * range
            maxi = data_max + 0.12 * range
        elif showMeanAndStd and showMeanAndStd_pos == "below":
            min = data_min - 0.12 * range
            maxi = data_max + 0.05 * range
        elif showMeanAndStd and showMeanAndStd_pos == "inside":
            min = data_min - 0.05 * range
            maxi = data_max + 0.05 * range
        ylim = (min, maxi)
    if ylim is not None:
        ax.set_ylim(ylim)
    if xlabel is not None:
        ax.set_xlabel(xlabel,fontsize=labels_fontSize)
    if ylabel is not None:
        ax.set_ylabel(ylabel,fontsize=labels_fontSize)
    if title is not None:
        ax.set_title(title)
    
    if (mltpl_plots and displacement == 0) or not mltpl_plots:
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=labels_angle)
    
    if ticks_prms is not None:
        ax.tick_params(axis='x',which='major',**ticks_prms)
        ax.tick_params(axis='y',which='major',**ticks_prms)

    return fig


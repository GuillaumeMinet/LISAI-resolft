
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import leastsq
import math

def lineProfileAnalysis(file_path, pxSize, do_plots=False,show_plots=False,calculate_snr=True,
                        fig=None, axes=None, start_index=0,min_fwhm=40,max_snr=20,
                        replace_empty_with_zeros=True,header=True,titles_list=None, 
                        fit_method="gaussian"):
    
    x,y = readLineProfileFile(file_path, pxSize,header,replace_empty_with_zeros)
    num_profiles = y.shape[0]

    if fit_method == "gaussian":
        fwhm_coeff = 2.35
        fit_func = single_gaussian_fit
        plot_func = single_gaussian
    elif fit_method == "lorentzian":
        fwhm_coeff = 2
        fit_func = single_lorentzian_fit
        plot_func = single_lorentzian
    else:
        raise ValueError(f"Unknown fit method {fit_method}")
    
    # Prepare (optional) subplot grid
    if do_plots and axes is None:
        max_cols = 5
        cols = min(max_cols, num_profiles)
        rows = math.ceil(num_profiles / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
        axes = axes.flatten()  # Flatten to easily index even if it's 1D

    fwhms = []
    snrs = []
    used_axes = []

    for i in range(num_profiles):
        idx = start_index + i
        ax = axes[idx] if do_plots else None

        # Remove zero values from the profile
        y2 = y[i]
        mask = np.where(y2 != 0)
        y2 = y2[mask]
        x2 = x[mask]

        # Estimate initial parameters
        peak_pos = x2[np.argmax(y2)]
        max_value = np.max(y2)

        # Fit Gaussian
        fit = leastsq(fit_func, [max_value, peak_pos, 70, 0], args=(x2, y2))
        params = fit[0][:4]  # [c1, mu1, sigma1, cste]

        # FWHM and SNR calculation
        sigma = np.abs(params[2])
        fwhm = fwhm_coeff * sigma

        if calculate_snr:
            bg_mask = (x2 < peak_pos - 3 * sigma) | (x2 > peak_pos + 3 * sigma)
            bg_y = y2[bg_mask]
            bg = np.std(bg_y) if len(bg_y) > 0 else np.nan  # avoid div by 0
            snr = params[0] / (bg) if bg != 0 and not np.isnan(bg) else np.nan

        
        if fwhm < min_fwhm: 
            print(f"Profile {i + 1}: FWHM out of bounds {fwhm}")
            fwhm = np.nan
        if (calculate_snr and snr > max_snr):
            print(f"Profile {i + 1}: SNR out of bounds: {snr}")
            snr = np.nan
        
        fwhms.append(fwhm)
        if calculate_snr:
            snrs.append(snr)

        # optional plotting
        if do_plots:
            # Plot data and fit
            ax.plot(x2, y2, 'ko', markersize=4, label='Data')
            x_fit = np.linspace(x.min(),x.max(),1000)
            ax.plot(x_fit, plot_func(x_fit, params), color='orange', label='Fit')

            # Plot background regions
            if calculate_snr:
                bg_left = peak_pos - 3 * sigma
                bg_right = peak_pos + 3 * sigma
                ax.axvline(bg_left, color='k', linestyle='--', linewidth=1)
                ax.axvline(bg_right, color='k', linestyle='--', linewidth=1)
                # ax.axhline(bg,color='k', linestyle='--', linewidth=1)
                # ax.axhline(-bg,color='k', linestyle='--', linewidth=1)
                ax.hlines(y=bg, xmin=x.min(), xmax=bg_left, color='k', linestyle='--', linewidth=1)
                ax.hlines(y=bg, xmin=bg_right, xmax=x.max(), color='k', linestyle='--', linewidth=1)
                ax.hlines(y=-bg, xmin=x.min(), xmax=bg_left, color='k', linestyle='--', linewidth=1)
                ax.hlines(y=-bg, xmin=bg_right, xmax=x.max(), color='k', linestyle='--', linewidth=1)
                # ax.axhspan(-bg, bg, facecolor='gray', alpha=0.3)

                ax.fill_betweenx(y=[-bg, bg], x1=ax.get_xlim()[0], x2=bg_left, color='gray', alpha=0.2)
                ax.fill_betweenx(y=[-bg, bg], x1=bg_right, x2=ax.get_xlim()[1], color='gray', alpha=0.2)

            # Annotate SNR and FWHM
            if calculate_snr:
                ax.text(0.95, 0.95, f"SNR: {snr:.2f}", ha='right', va='top', transform=ax.transAxes, fontsize=10)
            ax.text(0.95, 0.75, f"FWHM: {fwhm:.2f} nm", ha='right', va='top', transform=ax.transAxes, fontsize=10)

            if titles_list is not None:
                try:
                    ax.set_title(titles_list[i])
                except:
                    ax.set_title(f'Profile {i + 1}')
            else:
                ax.set_title(f'Profile {i + 1}')


            # cosmetic
            linewidth=1.5
            labelsize=12
            ticks_prms={"labelsize":labelsize, "width":linewidth,"length":linewidth*4}
            
            ax.set_xlabel('x (um)',fontsize=labelsize)
            ax.set_ylabel('Intensity',fontsize=labelsize)

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_linewidth(linewidth)
            ax.spines['bottom'].set_linewidth(linewidth)
            ax.tick_params(axis='x',which='major',**ticks_prms)
            ax.tick_params(axis='y',which='major',**ticks_prms)
            used_axes.append(ax)

    
    if do_plots and show_plots:
        for j in range(num_profiles, len(axes)):# Hide unused subplots
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.show()

    return fwhms, snrs, used_axes

def single_gaussian(x, params):
    """
    Computes a single Gaussian function.
    Args:
        x (np.ndarray): The x-coordinates where the Gaussian is evaluated.
        params (tuple): Parameters of the Gaussian in the form (c1, mu1, sigma1, cste)
            - c1: Amplitude of the Gaussian.
            - mu1: Mean (center) of the Gaussian.
            - sigma1: Standard deviation (width) of the Gaussian.
            - cste: Constant offset.
    Returns:
        np.ndarray: The computed Gaussian values at the given x-coordinates.
    """
    (c1, mu1, sigma1, cste) = params
    return c1 * np.exp(- (x - mu1) ** 2.0 / (2.0 * sigma1 ** 2.0)) + cste

def single_gaussian_fit(params, x, y):
    """ Single Gaussian fit."""
    fit = single_gaussian(x, params)
    return fit - y

def single_lorentzian(x, params):
    (c1, mu1, gamma1, cste) = params
    res =c1 * (gamma1**2.0 / ((x - mu1)**2.0 + gamma1**2.0)) + cste
    return res

def single_lorentzian_fit(params, x, y):
    fit = single_lorentzian(x, params)
    return (fit - y)



def readLineProfileFile(file_path, pxSize,header=True,replace_empty_with_zeros=True):
    """    
    Reads a line profile data file and returns the x-coordinates and y-values.
    Args:
        file_path (str): Path to the line profile data file.
        pxSize (float): Pixel size in nanometers.
        header (bool), default = True
            To skip first line
    Returns:
        - x (np.ndarray): Array of x-coordinates scaled by pxSize.
        - y (np.ndarray): 2D array of y-values for each profile.

    """

    x = []
    y_values = []
    # Read data from the CSV file
    with open(file_path, 'r') as file:
        reader = csv.reader(file, delimiter='\t')  # Assuming tab-separated values
        if header:
            _ = next(reader)  # Skip the first row (titles)

        for row in reader:
            x.append(float(row[0]))  # First column is x
            if replace_empty_with_zeros:
                y_row = [float(val) if val.strip() != '' else 0.0 for val in row[1:]] # replace empty with zeros
            else:
                y_row = [float(val) for val in row[1:] if val.strip() != '']  # Convert all y-values, skip empty
            y_values.append(y_row)

    # Convert to numpy arrays
    x = np.array(x) * pxSize
    y = np.array(y_values).T  # Transpose to shape (num_profiles, num_points)
    return x, y
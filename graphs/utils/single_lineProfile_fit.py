
import numpy as np
from scipy.optimize import leastsq

def single_gaussian(x,params):
    (c1, mu1, sigma1, b) = params
    res =   b + c1 * np.exp( - (x - mu1)**2.0 / (2.0 * sigma1**2.0) ) 
    return res


def single_gaussian_fit(params,x,y):
    fit = single_gaussian(x, params)
    return (fit - y)

def single_lorentzian(x, params):
    (c1, mu1, gamma1, b) = params
    res = b + c1 * (gamma1**2.0 / ((x - mu1)**2.0 + gamma1**2.0))
    return res

def single_lorentzian_fit(params, x, y):
    fit = single_lorentzian(x, params)
    return (fit - y)


def perform_fit(x,y,params = None, fit_choice = "lorentz",
                expected_fwhm = 60, expected_bg=0):

    if fit_choice == "gaussian":
        fit_func = single_gaussian_fit
        plot_func = single_gaussian
        fwhm_factor = 2.35
    elif fit_choice == "lorentz":
        fit_func = single_lorentzian_fit
        plot_func = single_lorentzian
        fwhm_factor = 2
    else:
        raise ValueError("unknown fit choice")

    if params is None:
        # Estimate initial parameters
        peak_pos = x[np.argmax(y)]
        max_value = np.max(y)
        params = [
            max_value,
            peak_pos,
            expected_fwhm, 
            expected_bg,
        ]
    
    fit_results = leastsq(fit_func, params, args=(x, y))

    fwhm = np.abs(fit_results[0][2]) * fwhm_factor


    x2 = np.linspace(min(x),max(x),1000)
    plot_curve = [x2, plot_func(x2, fit_results[0])]

    return plot_curve,fwhm
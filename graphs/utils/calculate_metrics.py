
from lisai.evaluation.metrics import windowed_psnr_2d,windowed_ssim_2d,windowed_mse_2d
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim, mean_squared_error as mse

def calculate_metrics(gt, pred,data_range=None,use_windowed=False,window_size=None,
                      patch_selection=None, range_invariant=False):
    """    Calculate PSNR, SSIM, and MSE between ground truth and predicted images."""
    
    if use_windowed:
        psnr_values = windowed_psnr_2d(gt, pred,size=window_size,
                                       patch_selection=patch_selection,
                                       range_invariant=range_invariant)
        ssim_values = windowed_ssim_2d(gt, pred,size=window_size,
                                       patch_selection=patch_selection)
        mse_values = windowed_mse_2d(gt, pred,size=window_size,
                                       patch_selection=patch_selection)
    else:
        psnr_values = [psnr(gt, pred,data_range=data_range)]
        ssim_values = [ssim(gt, pred,data_range=data_range)]
        mse_values = [mse(gt, pred)]

    return psnr_values, ssim_values, mse_values

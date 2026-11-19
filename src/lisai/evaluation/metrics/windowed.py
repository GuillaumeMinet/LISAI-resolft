import numpy as np

from lisai.data.utils import extract_patches, select_patches
from lisai.evaluation.metrics.ra_psnr import RangeInvariantPsnr as ra_psnr


def windowed_mse_2d(ref,test_image,size=128,patch_selection=None):
    """
    Calculate list of non-overlapping windowed mse on a 2d image.

    Parameters:
    -----------
    ref: 2d np.ndarray
        reference image
    test_image: 2d np.ndarray
        image to be tested
    size: integer, default=128
        window size
    patch_selection: float, default=None
        patch selection threshold between 0 and 1.
        If None, no selection done.
    range_invariant: bool, default = True
        to use range invariant psnr calculation
    
    Output
    ------
    list_mse: list
        list of the windowed mse

    """
    test_patches = extract_patches(test_image,patch_size=size)
    ref_patches = extract_patches(ref,patch_size=size)
    if patch_selection:
        test_patches = np.expand_dims(test_patches,axis=1)
        ref_patches = np.expand_dims(ref_patches,axis=1)
        test_patches,ref_patches,num_removed= select_patches(inp_patches=test_patches,
                                                 gt_patches=ref_patches,
                                                 threshold=patch_selection,
                                                 select_on_gt=True)
        test_patches=np.squeeze(test_patches,axis=1)
        ref_patches=np.squeeze(ref_patches,axis=1)

    from skimage.metrics import mean_squared_error as mse

    list_mse = []
    for idx in range(test_patches.shape[0]):
        gt=ref_patches[idx]
        pred=test_patches[idx]
        list_mse.append(mse(gt,pred))

    return list_mse


def windowed_psnr_2d(ref,test_image,size=128,patch_selection=None,range_invariant = True):
    """
    Calculate list of non-overlapping windowed psnrs on a 2d image.

    Parameters:
    -----------
    ref: 2d np.ndarray
        reference image
    test_image: 2d np.ndarray
        image to be tested
    size: integer, default=128
        window size
    patch_selection: float, default=None
        patch selection threshold between 0 and 1.
        If None, no selection done.
    range_invariant: bool, default = True
        to use range invariant psnr calculation
    
    Output
    ------
    list_psnrs: list
        list of the windowed psnrs

    """
    dyn_range = np.max(ref) - np.min(ref)
    test_patches = extract_patches(test_image,patch_size=size)
    ref_patches = extract_patches(ref,patch_size=size)
    if patch_selection:
        test_patches = np.expand_dims(test_patches,axis=1)
        ref_patches = np.expand_dims(ref_patches,axis=1)
        test_patches,ref_patches,num_removed= select_patches(inp_patches=test_patches,
                                                 gt_patches=ref_patches,
                                                 threshold=patch_selection,
                                                 select_on_gt=True,verbose=True)
        test_patches=np.squeeze(test_patches,axis=1)
        ref_patches=np.squeeze(ref_patches,axis=1)

    if not range_invariant:
        from skimage.metrics import peak_signal_noise_ratio as psnr

    list_psnrs = []
    for idx in range(test_patches.shape[0]):
        gt=ref_patches[idx]
        pred=test_patches[idx]
        if range_invariant:
            gt = np.expand_dims(gt,axis=0)
            pred = np.expand_dims(pred,axis=0)
            list_psnrs.append(ra_psnr(gt,pred))
        else:
            list_psnrs.append(psnr(gt,pred,data_range=dyn_range))
    return list_psnrs


def windowed_ssim_2d(ref,test_image,size=128,patch_selection=None):
    """
    Calculate list of windowed ssim on a 2d image.

    Parameters:
    -----------
    ref: 2d np.ndarray
        reference image
    test_image: 2d np.ndarray
        image to be tested
    patch_selection: float, default=None
        patch selection threshold between 0 and 1.
        If None, no selection done.
    
    Output
    ------
    list_ssim: list
        list of the windowed ssim

    """
    
    dyn_range = np.max(ref) - np.min(ref)
    test_patches = extract_patches(test_image,patch_size=size)
    ref_patches = extract_patches(ref,patch_size=size)
    if patch_selection:
        test_patches = np.expand_dims(test_patches,axis=1)
        ref_patches = np.expand_dims(ref_patches,axis=1)
        test_patches,ref_patches,_ = select_patches(inp_patches=test_patches,
                                                 gt_patches=ref_patches,
                                                 threshold=patch_selection,
                                                 select_on_gt=True)
        test_patches=np.squeeze(test_patches,axis=1)
        ref_patches=np.squeeze(ref_patches,axis=1)
    from skimage.metrics import structural_similarity as ssim

    list_ssim = []
    for idx in range(test_patches.shape[0]):
        gt=ref_patches[idx]
        pred=test_patches[idx]
        list_ssim.append(ssim(gt,pred,data_range=dyn_range))

    return list_ssim

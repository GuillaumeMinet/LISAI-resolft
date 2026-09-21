import numpy as np
from scipy.ndimage import shift as ndi_shift
from skimage.registration import phase_cross_correlation
from tifffile import imread, imwrite
from pathlib import Path
def register_stack(
    stack,
    reference_index=0,
    upsample_factor=10,
    interpolation_order=1,
    mode="nearest",
    return_shifts=False,
):
    """
    Register all images in `stack` to the image at `reference_index`
    using skimage.registration.phase_cross_correlation.

    Parameters
    ----------
    stack : np.ndarray
        Array of shape (n_images, height, width).
    reference_index : int
        Index of the reference image, usually the highest-SNR image.
    upsample_factor : int
        Subpixel registration precision. Higher = finer but slower.
        Example: 10 means ~1/10 pixel precision.
    interpolation_order : int
        Interpolation order for scipy.ndimage.shift.
        0 = nearest, 1 = bilinear, 3 = cubic.
    mode : str
        Border handling mode for scipy.ndimage.shift.
    return_shifts : bool
        If True, also return the estimated shifts.

    Returns
    -------
    registered : np.ndarray
        Registered stack with same shape as input.
    shifts : np.ndarray, optional
        Array of shape (n_images, 2), each row = (dy, dx).
        Returned only if return_shifts=True.
    """
    stack = np.asarray(stack)
    if stack.ndim != 3:
        raise ValueError(f"`stack` must have shape (n_images, height, width), got {stack.shape}")

    n_images = stack.shape[0]
    reference = stack[reference_index]
    registered = np.empty_like(stack, dtype=np.float32)
    shifts = np.zeros((n_images, 2), dtype=np.float32)

    for i in range(n_images):
        if i == reference_index:
            registered[i] = stack[i]
            shifts[i] = (0.0, 0.0)
            continue

        shift, error, phasediff = phase_cross_correlation(
            reference_image=reference,
            moving_image=stack[i],
            upsample_factor=upsample_factor,
        )

        dy, dx = shift

        registered[i] = ndi_shift(
            stack[i].astype(np.float32),
            shift=(dy, dx),
            order=interpolation_order,
            mode=mode,
            prefilter=(interpolation_order > 1),
        )
        shifts[i] = (dy, dx)

    if np.issubdtype(stack.dtype, np.integer):
        info = np.iinfo(stack.dtype)
        registered = np.clip(registered, info.min, info.max).astype(stack.dtype)
    else:
        registered = registered.astype(stack.dtype, copy=False)

    if return_shifts:
        return registered, shifts
    return registered


source = Path("/mnt/e/lisai/datasets/vim_fixed/dump/recon/timelapses_gathered/rec_c07_rec_CAM.tiff")
stack = imread(source)
stack_reg = register_stack(stack,reference_index=1,interpolation_order=0)
save_name = source.parent / "registered.tiff"
imwrite(save_name, stack_reg)
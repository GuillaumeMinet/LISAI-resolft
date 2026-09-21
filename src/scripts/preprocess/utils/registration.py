import logging

import numpy as np

# pystackreg imports
from pystackreg import StackReg

# sift with skimage imports ###
from skimage.feature import SIFT, match_descriptors
from skimage.measure import ransac
from skimage.transform import AffineTransform, warp
from tifffile import imread, imsave

logger = logging.getLogger("registration")


def pystackreg_registration(stack: np.array, ref_frame_idx:int=0):
    """

    registration with pystackreg library:
    https://pypi.org/project/pystackreg/

    but using skimage wrap function to apply transformation matrix,
    to avoid cubic spline interpolation, unavoidable with pystackteg,
    and which does weird effect to very low snr data.

    NOTE : kept only translation transformation.
    => faster and no need for anything else for now.

    NOTE: rolling axis part commented for now because for some reasons
    it impacts the registration in a weird way. => TO UNDERSTAND
    Meaning reference is always the first frame of the stack.
     
    """

    assert stack.shape[0] > 1

    logger.info("Starting registration...") 
    
    # move ref_frame_idx in first position
    # if ref_frame_idx != 0:
    #         assert ref_frame_idx < stack.shape[0]
    #         stack = np.roll(stack,shift=-ref_frame_idx,axis=0)

    # perform registration
    tf = StackReg.TRANSLATION
    sr = StackReg(tf)
    reference = 'first'
    tmat = sr.register_stack(stack, axis=0, reference=reference, verbose=True)
    
    registered = np.empty_like(stack)
    for i_img in range(stack.shape[0]):
            # print(i_img)
            logger.debug(f"Starting to align frame: #{i_img}")
            # get skimage's AffineTransform object
            tform = AffineTransform(matrix=tmat[i_img, :, :])

            # transform image using the saved transformation matrix
            registered[i_img, :, :] = warp(stack[i_img, :, :], tform,order=0)

    # re-arrange stack in original position
    # if ref_frame_idx != 0:
    #         registered = np.roll(registered,shift=ref_frame_idx,axis=0)
    
    logger.info("Registration finished.") 
    return registered


def sift_registration(stack:np.array, ref_frame_idx:int = 0 ):
    """

    register with SIFT algorithm (Scale-Invariant Feature Transform),
    using skimage library tools.
    Parameters hardcoded to what seems to work best: tested on fixed 
    vimentin "bleaching stacks".

    """

    assert stack.shape[0] > 1
    
    # Detect and extract features from reference frame

    ref_frame = stack[ref_frame_idx]

    sift = SIFT()
    sift.detect_and_extract(ref_frame)
    keypoints_fixed = sift.keypoints
    descriptors_fixed = sift.descriptors

    logger.info("Found reference descriptors and keypoints.")

    align_stack = np.empty_like(stack)
    for i in range(stack.shape[0]):
        logger.debug(f"Starting to align frame: #{i}")            

        # skip reference frame
        if i == ref_frame_idx:
            align_stack [i,...] = ref_frame
            continue
        
        # Detect and extract features from frame to be aligned
        frame = stack[i]
        sift.detect_and_extract(frame)
        keypoints_moving = sift.keypoints
        descriptors_moving = sift.descriptors

        # Match descriptors between the two images
        matches = match_descriptors(descriptors_fixed, descriptors_moving, cross_check=True)
        
        # Extract matched keypoints
        src = keypoints_moving[matches[:, 1]][:, ::-1]
        dst = keypoints_fixed[matches[:, 0]][:, ::-1]

        # Estimate translation transformation model using RANSAC
        matrix,_ = ransac((src, dst), AffineTransform, min_samples=4,
                        residual_threshold=25, max_trials=100)

        # align 
        align_stack[i,...] = warp(frame, matrix.inverse, output_shape=frame.shape, order=0)

    
    logger.info("Registration finished.") 

    return align_stack



def rearrange(stack,ref_frame_idx):
    """
    Helper function which moves the frames frames @ re_frames_idxs 
    to position 0 in the stack.
    """


### test script
if __name__ == "__main__":
    # path of stack to register
    stack_path = r"E:\dl_monalisa\Data\Actin_fixed_mltplSNR_30nm\dump\timelapses_gathered\c01_rec_scan00_CAM.hdf5.0.reconstruction.tif"
    # save location to save registered stack
    save_path = r"E:\dl_monalisa\Data\Actin_fixed_mltplSNR_30nm\dump\c00_registration_test_sift.tiff"

    stack = imread(stack_path)
    print("Sack loaded. Starting registration...")
    reg_stack = sift_registration(stack)
    imsave(save_path,reg_stack)
    print("Done.")

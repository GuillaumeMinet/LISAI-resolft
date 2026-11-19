import numpy as np


def make_4d(img:np.array,stack_selection_idx:int=None,timelapse_max:int=None):
    """
    Specific data preparation for "apply_model", to make all data 
    consistently 4d, while dealing with "stack_selection_idx" or "timelapse_max".
    """

    if len(img.shape) == 2:
        img = np.expand_dims(img,axis=(0,1))
        timelapse = False
        volumetric = False

    elif len(img.shape) == 3:
        if stack_selection_idx is not None:
            if isinstance(stack_selection_idx,int):
                img = img[stack_selection_idx:stack_selection_idx+1]
            elif isinstance(stack_selection_idx,list):
                img = img[stack_selection_idx]
        elif timelapse_max is not None:
            img = img[:min(timelapse_max,img.shape[0])]
        img = np.expand_dims(img,axis=0)
        timelapse = True
        volumetric = False

    elif len(img.shape) == 4:
        # print(img.shape)
        if stack_selection_idx is not None:
            if isinstance(stack_selection_idx,int):
                img = img[stack_selection_idx:stack_selection_idx+1]
            elif isinstance(stack_selection_idx,list):
                img = img[stack_selection_idx]
        elif timelapse_max is not None:
            img = img[:min(timelapse_max,img.shape[0])]
        timelapse = True
        volumetric = True
        img = np.transpose(img,(1,0,2,3))
    else:
        raise ValueError (f"Expected data to be 2d, 3d or 4d,"
                            f"but data has shape {img.shape}.")
    
    return img, timelapse,volumetric


def inverse_make_4d(img:np.array,volumetric:bool,timelapse:bool,
                    lvae_samples=False):
    """
    To inverse effect of "make_4d" and restore original image size.
    Expecting img to be [Z,T,H,W], or [Samples,Z,T,H,W] if lvae_samples.
    """
    if not timelapse:
        if lvae_samples:
            img = img[:,0,0]
        else:
            img = img[0,0]
            
    elif not volumetric:
        if lvae_samples:
            img = img[:,0]
        else:
            img = img[0]
    
    elif volumetric:
        if lvae_samples:
            img = np.transpose(img, (0, 2, 1, 3, 4))
        else:
            img = np.transpose(img, (1, 0, 2, 3))

    return img

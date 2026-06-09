import glob
import logging
import os
import shutil
from pathlib import Path

import numpy as np
from tifffile import imread, imsave

logger = logging.getLogger("gather")


def gather_timelapses(dir,stack_name_start,stack_name_end,trgt_dir = None, overwrite = False):

    logger = logging.getLogger("gather")
    logger.info("Gathering frames...")
    dir = Path(dir)
    if trgt_dir is None:
        trgt_dir = dir / "timelapses_gathered"
    
    if not os.path.exists(trgt_dir):
        os.makedirs(trgt_dir)
    elif overwrite:
        shutil.rmtree(trgt_dir)
        os.makedirs(trgt_dir)
    else:
        logger.critical("Target directory already exists and overwrite"
                        " is False. Exiting program.")
        exit()

    list_files = sorted(glob.glob(str(dir) + "/*.tif")) \
            + sorted(glob.glob(str(dir) + "/*.tiff"))

    # init stack_name,first_frame_name and empty arrays_list
    stack_name = list_files[0].split("\\")[-1][stack_name_start:stack_name_end]
    first_frame_name = list_files[0].split("\\")[-1]
    arrays_list = []

    # loop over files
    for file in list_files:
        name = file.split("\\")[-1]

        if stack_name != name[stack_name_start:stack_name_end]:
            #saving, only if there is something in the array
            if len(arrays_list) !=0: 
                stack_array = np.stack(arrays_list,axis=2)
                stack_array = np.transpose(stack_array, (2,0,1))
                save_path = trgt_dir / first_frame_name
                imsave(save_path,stack_array)
                logger.debug(f"Saved gathered {first_frame_name}")

            # update stack_name,first_frame_name and empty arrays_list
            stack_name = name[stack_name_start:stack_name_end]
            first_frame_name = name
            arrays_list = []

        # read frame and add it to ongoing stack
        im = np.array(imread(file))
        if len(im.shape) == 3:
            im = im[0,...]
        arrays_list.append(im[:,:])
        
        logger.debug(f"file: {file}, stack_name: {stack_name} ")

    #last file
    if len(arrays_list) !=0: #saving, only if there is something in the array
        stack_array = np.stack(arrays_list,axis=2)
        stack_array = np.transpose(stack_array, (2,0,1))
        save_path = trgt_dir / first_frame_name
        imsave(save_path,stack_array)
        logger.debug(f"saved {first_frame_name}")
    
    logger.info(f"Gathered timelapes sucessfully in {trgt_dir}.")
    
    return trgt_dir


if __name__ == "__main__":
    print("running")
    dir = r"E:\dl_monalisa\Data\Actin_fixed_bleachTimelapses_30nm\dump\rec"
    gather_timelapses(dir,0,3,trgt_dir = None, overwrite = True)
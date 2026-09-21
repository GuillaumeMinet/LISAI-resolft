import math
import os

import cv2 as cv
import h5py as h5
import numpy as np


def resize_stack(stack,target_size):
    new_stack = np.zeros([stack.shape[0],target_size,target_size])
    for i in range (stack.shape[0]):
        new_stack[i,...] = cv.resize(stack[i,...],(target_size,target_size))

    return new_stack

def crop_n_ulens(raw_stack, recon, period, phase, scan_factor=1, n_ulens=59):

    n_scan = raw_stack.shape[0] # number of scanning steps
    n_scan_1d = math.sqrt(n_scan) # number of scanning ssteps in 1 direction
    assert n_scan_1d % 1 == 0 # n_frames should be a perfect square number 
    n_scan_1d = int(n_scan_1d)

    R_raw,C_raw = raw_stack.shape[1:] 
    square_frames = False #to know if we start with a squre size R=C

    if R_raw == C_raw:
        square_frames = True 

    # for now, only implemented on squared size
    assert square_frames == True 

    # we crop as follow: exclude 1st ulens, keep n_ulens
    y_start = round(phase[0] + period[0]/2) 
    x_start = round(phase[1] + period[1]/2)
    y_final = y_start + round(n_ulens*period[0])
    x_final = x_start + round(n_ulens*period[1])
    raw_stack_crop = raw_stack[:,(y_start):(y_final),(x_start):(x_final)]

    #if started with squared frames, we should end with squared frames
    if square_frames: 
        if raw_stack_crop.shape[1]!=raw_stack_crop.shape[2]:
            min_size = min(raw_stack_crop.shape[1],raw_stack_crop.shape[2])
            raw_stack_crop = raw_stack_crop[:,:min_size,:min_size]
            print("Not squared anymore after initial cropping -> Recropping")

    # corresponding cropping on the reconstruction
    if len(recon.shape) == 3:
        recon = recon[0,...]

    n_scan_1d_recon = n_scan_1d * scan_factor #in case of "upsampling", n_scan is different of raw and recon
    
    recon_crop_start = n_scan_1d_recon #skipping 1st ulens
    recon_crop_end = n_scan_1d_recon * (1+n_ulens) # keepping n-ulens
    recon_crop = recon[recon_crop_start:recon_crop_end,recon_crop_start:recon_crop_end]

    return recon_crop, raw_stack_crop


def resize_round_period(stack,period,n_ulens):
    #print(stack.shape[1])
    n_frames = stack.shape[0]
    new_period = int(np.ceil(np.mean(period)))
    #print(new_period)
    new_size = int((2*np.ceil(new_period/np.mean(period) * stack.shape[1]))//2)
    #print(new_size)

    stack_resize=resize_stack(stack,new_size)

    return new_period, stack_resize

def reshape_recon_space(stack, period, n_ulens):

    n_frames = stack.shape[0]
    n_scan = int(math.sqrt(n_frames)) #number of scan step in one direction
    N = stack.shape[1] #image frame size

    stack_4d = np.reshape(stack,[n_scan,n_scan,N,N])
    stack_4d = np.flip(stack_4d, axis=0)

    stack_6d = np.zeros(shape=(n_scan,n_scan,n_ulens,n_ulens,period,period))

    x0 = 0
    y0 = 0

    for i in range(n_scan):
        for j in range(n_scan):

            idx_x = x0
            idx_y = y0
            slice = stack_4d[i,j,...]

            while idx_x <= N - period:
                while idx_y <= N - period:
                    stack_6d[i,j,(idx_y-y0)//period,(idx_x-x0)//period,...]=slice[idx_y:idx_y+period,idx_x:idx_x + period]
                    idx_y += period
                    #print(idx_y)
                #print('out')
                idx_y = y0
                idx_x += period

    stack_6d_swapped = np.transpose(stack_6d,(2,0,3,1,4,5))
    new_stack_4d = np.reshape(stack_6d_swapped,[n_scan*n_ulens,n_scan*n_ulens,period,period])

    return new_stack_4d
    
def shift_scan_space(stack_4d, n_ulens, f=None, crop=False):

    period = stack_4d.shape[-1]

    if f is None:
        f = stack_4d.shape[0]/n_ulens/period

    stack_4d_shifted = np.zeros_like(stack_4d)
    shifts = np.abs(np.round(f * np.linspace (-np.floor(period/2),np.ceil(period/2)-1,period)))
    #print(shifts)
    for x in range(period):
        for y in range(period):
            
                        
            sx = int(shifts[x])
            sy = int(shifts[y])
            #print(sx,sy)

            if x > period/2:
                if y > period/2:
                    stack_4d_shifted[sy:,sx:,y,x]=stack_4d[:stack_4d.shape[0]-sy,:stack_4d.shape[0]-sx,y,x]
                else:
                    stack_4d_shifted[:stack_4d.shape[0]-sy,sx:,y,x]=stack_4d[sy:,:stack_4d.shape[0]-sx,y,x]
            else:
                if y > period/2:
                    stack_4d_shifted[sy:,:stack_4d.shape[0]-sx,y,x]=stack_4d[:stack_4d.shape[0]-sy,sx:,y,x]
                else:
                    stack_4d_shifted[:stack_4d.shape[0]-sy,:stack_4d.shape[0]-sx,y,x]=stack_4d[sy:,sx:,y,x]

            

    

    if crop:
        max_shift = round((period-1)*f)
        stack_4d_shifted = stack_4d_shifted[max_shift:, max_shift:,...]
        px_cropped = max_shift
    else:
        px_cropped = 0
   
    return stack_4d_shifted, px_cropped




def save_hdf5_file(f_dir,f_name,data):
    """
    Save a numpy array (data) as hdf5 dir, at location f_dir with name f_name.
    """
    new_f = h5.File(f_dir / f_name, 'w')
    new_f.create_dataset(f_name, data.shape, data=data)
    new_f.close()


def load_hdf5_file(f_adress):
    """
    Load hdf5 file into a np array, taking the full path location as input.
    """
    f = h5.File(f_adress, 'r')
    data = np.array(f[list(f.keys())[0]])
    f.close()
    return data

def open_hdf5_files(data_dir):
    data_list = list()
    f_list = os.listdir(data_dir)
    for f_name in f_list:
        try:
            f_address = data_dir / f_name
            f = h5.File(f_address, 'r')
            data_list.append(f[list(f.keys())[0]])
            # f.close()
        except:
            print(f"unable to open {f_name}")
    return data_list, f_list


def crop_and_resize(raw_stack,recon, period, phase,n_ulens=59):
    """
    Crop raw and reconstructed to avoid microlens at the border, 
    and resize raw to match reconstructed with simple cv.resize.

    /!\ only for square size input /!\   
    TODO: implement when not squared -> deal with different n_ulens 
    in x and y. 

    inputs:
        - stack of raw data, 3d numpy array 
        - reconstruction image, 2d numpy array
        - period, phase of the pattern, each one = list with [y x] info
        - n_ulens: number of ulens to keep, default = 59
    output:
        - raw and recon data crop and resized, as numpy arrays
    """

    n_scan = raw_stack.shape[0] # number of scanning steps
    n_scan_1d = math.sqrt(n_scan) # number of scanning ssteps in 1 direction

    
    assert n_scan_1d % 1 == 0 # n_frames should be a perfect square number 
    n_scan_1d = int(n_scan_1d)

    R_raw,C_raw = raw_stack.shape[1:] 
    square_frames = False #to know if we start with a squre size R=C

    if R_raw == C_raw:
        square_frames = True 

    # for now, only implemented on squared size
    assert square_frames == True 

    # we crop as follow: exclude 1st ulens, keep n_ulens
    y_start = round(phase[0] + period[0]/2) 
    x_start = round(phase[1] + period[1]/2)
    y_final = y_start + round(n_ulens*period[0])
    x_final = x_start + round(n_ulens*period[1])
    raw_stack_crop = raw_stack[:,(y_start):(y_final),(x_start):(x_final)]

    #if started with squared frames, we should end with squared frames
    if square_frames: 
        if raw_stack_crop.shape[1]!=raw_stack_crop.shape[2]:
            min_size = min(raw_stack_crop.shape[1],raw_stack_crop.shape[2])
            raw_stack_crop = raw_stack_crop[:,:min_size,:min_size]
            print("Not squared anymore after initial cropping -> Recropping")

    # corresponding cropping on the reconstruction
    if len(recon.shape) == 3:
        recon = recon[0,...]
    recon_crop_start = n_scan_1d #skipping 1st ulens
    recon_crop_end = n_scan_1d * (1+n_ulens) # keepping n-ulens
    recon_crop = recon[recon_crop_start:recon_crop_end,recon_crop_start:recon_crop_end]

    # resize stack frames to match recon shape
    print(recon_crop.shape)
    R_final, C_final = recon_crop.shape
    raw_stack_crop_resize=np.zeros(shape=(n_scan,R_final,C_final))
    for i, frame in enumerate(raw_stack_crop):
        raw_stack_crop_resize[i,...]=cv.resize(frame, (R_final,C_final))
    #raw_stack_crop_resize = raw_stack_crop
    
    return recon_crop, raw_stack_crop_resize
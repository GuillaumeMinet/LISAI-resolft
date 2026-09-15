import numpy as np
import torch

from lisai.data.utils import adjust_for_tiling, adjust_img_size, adjust_pred_size, crop_center
from lisai.evaluation.inference.progress import ProgressLike, ensure_progress
from lisai.lib.hdn.forwardpass import forward_pass as lvae_forward_pass


def predict(model:torch.nn.Module, inp:torch.tensor, device=None,
            is_lvae:bool=False,num_samples:int = None,mltpl_of:int = 16,
            tiling_size:int = None,tiling_overlap = 50,upsamp:int=1,
            ch_out = None, progress: ProgressLike | None = None,
            progress_level: int = 0):
    
    """
    Predicts image for given model, adjusting image size with padding,
    and doing tiling if necessary.

    Parameters:
    ----------
    model: torch.nn.Module
        model used for prediction
    inp: torch tensor
        input to pass to model
    device: torch device
        cpu or gpu
    is_lvae: bool, default = False
        if model is a LVAE or not
    num_samples: int, default = None
        Mandator if LVAE: number of samples to computer MMSE
    mltpl_of: int, default = 16
        Minimum multiple of which the size of the input should
        be to not create issues when passing through model.
    tiling_size: int, default = None
        Size of the tiles. If tiling_size > size of inp, no tiling.
        If tiling_size is None, inference runs without tiling.
    tiling_overlap: int, default = 50
        Overlap size in pixel for the tiling.
    upsamp: int, default = 1
        Upsampling factor of model. Leave to 1 if not upsampling task.
    ch_out: int, default = None
        Number of channels of the prediction. If None, same as input channels.

    Returns:
    -------
    outputs: dict
        dictionnary with "prediction" (i.e. mmse for LVAE) and 
        "samples" if lvae, all as numpy arrays.
    """

    if is_lvae:
        assert num_samples is not None, ("for LVAE inference, number of " \
                                         "samples needs to be specified")

    progress = ensure_progress(progress)
    
    original_size = np.array(inp.shape[-2:])
    # print(f"Original input size: {original_size}")
    if inp.shape[-1] % 2 !=0 or inp.shape[-2] % 2 !=0: # ensure that image size is even
        inp = adjust_img_size(inp,mltpl_of=2,mode="crop")

    inp_size = np.array(inp.shape[-2:])

    # no-tiling prediction
    if tiling_size is None or (inp_size < tiling_size).any():
        # define output shape
        output_size = ((inp_size[0]*upsamp,inp_size[1]*upsamp))
        if ch_out is None:
            output_shape = (*inp.shape[:-2],*output_size)
        else:
            output_shape = (*inp.shape[:-3],ch_out,*output_size)

        inp_pad = adjust_img_size(inp,mltpl_of = mltpl_of, mode="pad")
        outputs = make_prediction(model,inp_pad,device,is_lvae,num_samples)
        outputs["prediction"] = crop_center(outputs.get("prediction"),
                                            crop_size = output_size)
        if is_lvae:
            outputs["samples"] = crop_center(outputs.get("samples"),
                                             crop_size = output_size)

    # tiling prediction
    else:
        # adjust tiling size and padding
        inp_pad,tiling_size,overlap,additionnal_padding = adjust_for_tiling(inp,tiling_size,mltpl_of,
                                                        min_overlap=tiling_overlap)
        # imsave("inp_pad.tif", inp_pad.cpu().detach().numpy())
        # print(inp.shape,inp_pad.shape,tiling_size,overlap)

        
        full_tile_size = (tiling_size[0] + overlap[0], tiling_size[1] + overlap[1])
        output_tile_size = (tiling_size[0]*upsamp,tiling_size[1]*upsamp)
        padded_size = inp_pad.shape[-2:]

        offset_y = (overlap[0] + additionnal_padding [0])//2
        offset_x = (overlap[1] + additionnal_padding [1])//2

        # define output shape
        output_size = ((padded_size[0]*upsamp,padded_size[1]*upsamp))
        if ch_out is None:
            output_shape = (*inp.shape[:-2],*output_size)
        else:
            output_shape = (*inp.shape[:-3],ch_out,*output_size)

        # create empty prediction and samples
        prediction = np.zeros(output_shape,dtype='float32')
        if is_lvae:
            samples = np.zeros((num_samples,*output_size),dtype='float32')

        # tiling loop 
        y_iter = range(0,inp_size[0],tiling_size[0])
        x_iter = range(0,inp_size[1],tiling_size[1])
        total_iterations = len(y_iter) * len(x_iter)

        tile_iter = ((y, x) for y in y_iter for x in x_iter)
        for y, x in progress.track(
            tile_iter,
            total=total_iterations,
            desc="Tiling",
            level=progress_level,
            leave=progress_level <= 0,
        ):
            # print(f"\n","x:",x,"y:",y)
            xx = x + full_tile_size[1]
            yy = y + full_tile_size[0]
            patch = inp_pad[...,y:yy,x:xx]
            # print(patch.shape)
            patch_outputs = make_prediction(model,patch,device,is_lvae,num_samples)
            # print(patch_outputs.get("prediction").shape)
            x_pred = (x+offset_x)*upsamp
            y_pred = (y+offset_y)*upsamp
            xx_pred = (x + offset_x + tiling_size[1])*upsamp
            yy_pred = (y + offset_y + tiling_size[0])*upsamp
            prediction[...,y_pred:yy_pred,x_pred:xx_pred] = crop_center(patch_outputs.get("prediction"),
                                                                        crop_size=output_tile_size)
            if is_lvae:
                samples[:,y_pred:yy_pred,x_pred:xx_pred] = crop_center(patch_outputs.get("samples"),
                                                                       crop_size=output_tile_size)

        outputs = {"prediction": prediction}
        if is_lvae: 
            outputs["samples"] = samples
            
    
    # imsave("prediction.tif", outputs["prediction"].astype(np.float32))
    outputs["prediction"] = adjust_pred_size(outputs.get("prediction"),original_size,upsamp)
    if is_lvae: 
        outputs["samples"] = adjust_pred_size(outputs.get("samples"),original_size,upsamp)
    
    return outputs



def make_prediction(model:torch.nn.Module,inp:torch.tensor,device=None,
                    is_lvae:bool=False,num_samples:int = None):
    """
    Feed input in model and outputs prediction.

    Parameters:
    ----------
    model: torch module
        model used for prediction
    inp: torch tensor
        input to pass to model
    device: torch device
        cpu or gpu
    is_lvae: bool, default = False
        if model is a LVAE or not
    num_samples: int, default = None
        Mandator if LVAE: number of samples to computer MMSE

    Returns:
    -------
    outputs: dict
        dictionnary with "prediction" (i.e. mmse for LVAE) and "samples"
        if lvae, all as numpy arrays.
    """
    
    model.eval()
    
    if is_lvae:

        assert num_samples is not None, ("for LVAE inference, number of " \
                                         "samples needs to be specified")
        
        model.mode_pred=True
        
        img_mmse, samples = lvae_predict(inp,num_samples,model,None,device,return_samples=True)

        outputs = {
            "prediction": img_mmse,
            "samples": np.stack(samples,axis=0),
        }
    
    else:
        with torch.no_grad():
            prediction = model(inp)
        outputs = {
            "prediction": prediction.cpu().detach().numpy()
        }
    
    return outputs


def lvae_predict(inp, num_samples, model, gaussian_noise_std, device, return_samples=False):
    """
    Predicts desired number of samples and computes MMSE estimate for HDN model.
    Parameters
    ----------
    inp: torch.tensor
        input to feed into model.
    num_samples: int
        Number of samples to average for computing MMSE estimate.
    model: Ladder VAE object
        Hierarchical DivNoising model.
    gaussian_noise_std: float
        std of Gaussian noise used to corrupty data. For intrinsically noisy data, set to None.
    device: GPU device
    """

    img_height,img_width=inp.shape[0],inp.shape[1]
    if len(inp.shape)==2:
        inp = inp.view(1,1,img_height,img_width)
    inp = inp.to(device=device, dtype=torch.float)
    samples = []
        
    with torch.no_grad():
        for _ in range(num_samples):
            output = lvae_forward_pass(inp,None,device,model,gaussian_noise_std)
            sample = output.get('out_mean').cpu().detach().numpy()
            samples.append(np.squeeze(sample))
    
    img_mmse = np.mean(np.array(samples),axis=0)
    if return_samples:
        return img_mmse, samples
    else:
        return img_mmse


def predict_sample(inp, model, gaussian_noise_std, device):
    """
    Predicts a LVAE sample.
    Parameters
    ----------
    img: torch.tensor
        inp to feed into model.
    model: Ladder VAE object
        Hierarchical DivNoising model.
    gaussian_noise_std: float
        std of Gaussian noise used to corrupty data. For intrinsically noisy data, set to None.
    device: GPU device
    """
    with torch.no_grad():
        outputs = lvae_forward_pass(inp, None, device, model, gaussian_noise_std)
    recon = outputs['out_mean']
    # recon_denormalized = recon*model.data_std+model.data_mean
    # recon_cpu = recon_denormalized.cpu()
    # recon_numpy = recon_cpu.detach().numpy()

    return recon


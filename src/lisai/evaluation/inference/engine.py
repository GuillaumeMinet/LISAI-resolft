from contextlib import nullcontext

import numpy as np
import torch

from lisai.data.utils import adjust_for_tiling, adjust_img_size, adjust_pred_size, crop_center
from lisai.evaluation.inference.progress import ProgressLike, InferenceProgressReporter, ensure_progress
from lisai.lib.hdn.forwardpass import forward_pass as lvae_forward_pass


def predict(model:torch.nn.Module, inp:torch.tensor, device=None,
            is_lvae:bool=False,num_samples:int = None,mltpl_of:int = 16,
            tiling_size:int = None,tiling_overlap = 50,upsamp:int=1,
            ch_out = None, progress: ProgressLike | None = None,
            progress_level: int = 0, save_samples: bool = False,
            sample_batch_size: int = 1, precision: str = "fp32",
            progress_reporter: InferenceProgressReporter | None = None,
            frame_index: int = 1):
    
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
    hdn_options = (dict(save_samples=save_samples, sample_batch_size=sample_batch_size,
                        precision=precision)
                   if is_lvae else {})
    
    if progress_reporter is not None:
        progress_reporter.checkpoint()
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
        outputs = make_prediction(
            model, inp_pad, device, is_lvae, num_samples,
            **(dict(progress_reporter=progress_reporter, frame_index=frame_index,
                    tile_index=1, tile_total=1) if progress_reporter is not None else {}),
            **hdn_options,
        )
        outputs["prediction"] = crop_center(outputs.get("prediction"),
                                            crop_size = output_size)
        if "samples" in outputs:
            outputs["samples"] = crop_center(outputs["samples"], crop_size=output_size)

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
        if is_lvae and save_samples:
            samples = np.zeros((num_samples,*output_size),dtype='float32')

        # tiling loop 
        y_iter = range(0,inp_size[0],tiling_size[0])
        x_iter = range(0,inp_size[1],tiling_size[1])
        total_iterations = len(y_iter) * len(x_iter)

        tile_iter = ((y, x) for y in y_iter for x in x_iter)
        for tile_index, (y, x) in enumerate(progress.track(
            tile_iter,
            total=total_iterations,
            desc="Tiling",
            level=progress_level,
            leave=progress_level <= 0,
        ), start=1):
            # print(f"\n","x:",x,"y:",y)
            xx = x + full_tile_size[1]
            yy = y + full_tile_size[0]
            patch = inp_pad[...,y:yy,x:xx]
            # print(patch.shape)
            patch_outputs = make_prediction(
                model, patch, device, is_lvae, num_samples,
                **(dict(progress_reporter=progress_reporter, frame_index=frame_index,
                        tile_index=tile_index, tile_total=total_iterations)
                   if progress_reporter is not None else {}),
                **hdn_options,
            )
            # print(patch_outputs.get("prediction").shape)
            x_pred = (x+offset_x)*upsamp
            y_pred = (y+offset_y)*upsamp
            xx_pred = (x + offset_x + tiling_size[1])*upsamp
            yy_pred = (y + offset_y + tiling_size[0])*upsamp
            prediction[...,y_pred:yy_pred,x_pred:xx_pred] = crop_center(patch_outputs.get("prediction"),
                                                                        crop_size=output_tile_size)
            if is_lvae and save_samples:
                samples[:,y_pred:yy_pred,x_pred:xx_pred] = crop_center(
                    patch_outputs["samples"], crop_size=output_tile_size)

        outputs = {"prediction": prediction}
        if is_lvae and save_samples:
            outputs["samples"] = samples
            
    
    # imsave("prediction.tif", outputs["prediction"].astype(np.float32))
    outputs["prediction"] = adjust_pred_size(outputs.get("prediction"),original_size,upsamp)
    if "samples" in outputs:
        outputs["samples"] = adjust_pred_size(outputs["samples"],original_size,upsamp)
    
    return outputs



def make_prediction(model: torch.nn.Module, inp: torch.Tensor, device=None,
                    is_lvae: bool = False, num_samples: int | None = None,
                    *, save_samples: bool = False, sample_batch_size: int = 1,
                    precision: str = "fp32",
                    progress_reporter: InferenceProgressReporter | None = None,
                    frame_index: int = 1, tile_index: int = 1,
                    tile_total: int = 1):
    """Predict one padded frame/tile; only collect HDN samples when requested."""
    if progress_reporter is not None:
        progress_reporter.checkpoint()
    was_training = model.training
    original_pred_mode = getattr(model, "mode_pred", None)
    model.eval()
    if is_lvae:
        if num_samples is None:
            raise ValueError("LVAE inference requires lvae_num_samples.")
        try:
            model.mode_pred = True
            result = lvae_predict(inp, num_samples, model, None, device,
                                  return_samples=save_samples,
                                  sample_batch_size=sample_batch_size,
                                  precision=precision,
                                  **(dict(progress_reporter=progress_reporter,
                                          frame_index=frame_index,
                                          tile_index=tile_index, tile_total=tile_total)
                                     if progress_reporter is not None else {}))
            if save_samples:
                mmse, samples = result
                return {"prediction": mmse, "samples": np.stack(samples, axis=0)}
            return {"prediction": result}
        finally:
            if original_pred_mode is not None:
                model.mode_pred = original_pred_mode
            model.train(was_training)

    try:
        with torch.no_grad():
            prediction = model(inp)
        if progress_reporter is not None:
            progress_reporter.advance(
                1, frame_index=frame_index, tile_index=tile_index,
                tile_total=tile_total, sample_index=1,
            )
        return {"prediction": prediction.detach().cpu().numpy()}
    finally:
        model.train(was_training)


def lvae_predict(inp, num_samples, model, gaussian_noise_std, device,
                 return_samples=False, *, sample_batch_size: int = 1,
                 reuse_encoder: bool = True, precision: str = "fp32",
                 progress_reporter: InferenceProgressReporter | None = None,
                 frame_index: int = 1, tile_index: int = 1,
                 tile_total: int = 1):
    """Generate exactly N HDN draws and accumulate an FP32 Monte Carlo mean.

    Cached deterministic features and their stochastic draws are scoped to this
    call (one frame/tile). Batches may change the random-number stream; FP32
    accumulation order can differ slightly from NumPy's historical mean.
    """
    if isinstance(num_samples, bool) or not isinstance(num_samples, int) or num_samples < 1:
        raise ValueError("num_samples must be a positive integer.")
    if (isinstance(sample_batch_size, bool) or not isinstance(sample_batch_size, int)
            or sample_batch_size < 1):
        raise ValueError("sample_batch_size must be a positive integer.")
    if inp.ndim == 2:
        inp = inp[None, None]
    if inp.ndim != 4 or inp.shape[0] != 1:
        raise ValueError("LVAE inference expects one image in NCHW format.")
    resolved_device = torch.device(device) if device is not None else inp.device
    if precision not in ("fp32", "fp16"):
        raise ValueError("HDN inference precision must be fp32 or fp16.")
    if precision == "fp16" and resolved_device.type != "cuda":
        raise ValueError("FP16 HDN inference requires a compatible CUDA device.")
    inp = inp.to(device=resolved_device, dtype=torch.float32)
    collected = [] if return_samples else None
    total = None
    autocast = (torch.autocast(device_type="cuda", dtype=torch.float16)
                if precision == "fp16" else nullcontext())
    with torch.no_grad(), autocast:
        features = (model.prepare_inference_features(inp)
                    if reuse_encoder and hasattr(model, "prepare_inference_features") else None)
        for start in range(0, num_samples, sample_batch_size):
            if progress_reporter is not None:
                progress_reporter.checkpoint()
            batch_size = min(sample_batch_size, num_samples - start)
            batch_input = inp.expand(batch_size, *inp.shape[1:])
            if features is not None:
                model_out = model(batch_input, cached_bottomup=features)
                draw = model_out["out_mean"]
            else:
                draw = lvae_forward_pass(batch_input, None, device, model,
                                         gaussian_noise_std)["out_mean"]
            if draw.shape[0] != batch_size:
                raise RuntimeError("LVAE did not produce the requested number of stochastic draws.")
            if not bool(torch.isfinite(draw).all()):
                raise FloatingPointError("Non-finite HDN prediction under " + precision.upper())
            draw = draw.float()
            # Sum a batch on the GPU before transferring its contribution.
            contribution = draw.sum(dim=0).detach().cpu().numpy()
            total = contribution if total is None else total + contribution
            if collected is not None:
                collected.extend(np.squeeze(sample) for sample in draw.detach().cpu().numpy())
            if progress_reporter is not None:
                progress_reporter.advance(
                    batch_size, frame_index=frame_index,
                    tile_index=tile_index, tile_total=tile_total,
                    sample_index=start + batch_size,
                )
    mmse = np.squeeze(total / np.float32(num_samples)).astype(np.float32, copy=False)
    return (mmse, collected) if return_samples else mmse


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


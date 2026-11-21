"""Lightweight HDN execution choices and CUDA-OOM diagnostics.

Batching is experimental and never autotuned. Available CUDA memory remains
inspectable for diagnostics, but does not control the default batch size.
"""

from __future__ import annotations

from lisai.hdn_performance import HDNPerformanceOptions, InferencePlan, HDNOutOfMemoryError


def effective_inference_shape(image_shape: tuple[int, int], *,
                              tiling_size: int | None, model=None,
                              mltpl_of: int = 16, overlap: int = 50) -> tuple[int, int]:
    """Mirror engine spatial crop/pad and actual overlap-adjusted tile dimensions.

    This is computed after input normalization, optional cropping/downsampling.
    It intentionally does not allocate a full padded image just to plan.
    """
    from lisai.data.utils.tiling import find_best_tile

    h, w = (int(v) for v in image_shape)
    h -= h % 2
    w -= w % 2
    if min(h, w) < 1:
        raise ValueError("Empty effective input after even-dimension cropping.")
    if tiling_size is not None and h >= tiling_size and w >= tiling_size:
        if min(h, w) < 100:
            raise ValueError("tensor smaller than min_tiling_size")
        size = tiling_size - (tiling_size % 2)
        tile_h, _ = find_best_tile(h, 100, size)
        tile_w, _ = find_best_tile(w, 100, size)
        overlap_h = overlap + (mltpl_of - (tile_h + overlap) % mltpl_of)
        overlap_w = overlap + (mltpl_of - (tile_w + overlap) % mltpl_of)
        h, w = tile_h + overlap_h, tile_w + overlap_w
    else:
        h = ((h + mltpl_of - 1) // mltpl_of) * mltpl_of
        w = ((w + mltpl_of - 1) // mltpl_of) * mltpl_of
    if model is not None and callable(getattr(model, "get_padded_size", None)):
        h, w = map(int, model.get_padded_size((h, w)))
    return h, w


def _cuda_fp16_compatible(device) -> bool:
    import torch

    if device.type != "cuda" or not torch.cuda.is_available():
        return False
    try:
        major, _ = torch.cuda.get_device_capability(device)
        return major >= 7  # Tensor-core-era CUDA architecture
    except (RuntimeError, AssertionError, ValueError):
        return False


def _cuda_available_including_cache(device) -> tuple[int | None, int | None, int | None]:
    """Return driver-free, reusable-cache, usable bytes for one CUDA device.

    Unused reserved blocks belong to PyTorch's allocator and can normally be
    reused without requesting more driver memory.  Subtract inactive split
    blocks where reported: fragmentation may make those bytes unreusable.
    This remains an advisory estimate, not a guarantee against CUDA OOM.
    """
    import torch

    try:
        driver_free = int(torch.cuda.mem_get_info(device)[0])
        allocated = int(torch.cuda.memory_allocated(device))
        reserved = int(torch.cuda.memory_reserved(device))
        stats = torch.cuda.memory_stats(device)
        inactive_split = max(0, int(stats.get("inactive_split_bytes.all.current", 0)))
        cache = max(0, reserved - allocated - inactive_split)
        return driver_free, cache, driver_free + cache
    except (RuntimeError, AssertionError, ValueError):
        return None, None, None


def resolve_hdn_plan(*, model, device, image_shape: tuple[int, int],
                     tiling_size: int | None, requested_samples: int,
                     performance: HDNPerformanceOptions | None = None) -> InferencePlan:
    """Resolve an execution plan once, before computing any stochastic draws."""
    import torch

    if isinstance(requested_samples, bool) or not isinstance(requested_samples, int) or requested_samples < 1:
        raise ValueError("HDN inference requires a positive integer sample count.")
    device = torch.device(device)
    # No explicit settings: Auto precision (FP16 where compatible), batch 1.
    # Historical `sample_batch_size: auto` is kept as a batch-1 alias.
    options = performance if performance is not None else HDNPerformanceOptions()
    compatible = _cuda_fp16_compatible(device)
    if options.precision == "fp16" and not compatible:
        raise ValueError("FP16 HDN inference requires a compatible CUDA GPU (compute capability >= 7.0).")
    precision = ("fp16" if compatible else "fp32") if options.precision == "auto" else options.precision
    if device.type == "cpu" and options.sample_batch_size != "auto" and options.sample_batch_size != 1:
        raise ValueError("CPU HDN inference supports sample batch size 1 only.")
    if device.type not in ("cpu", "cuda"):
        if options.precision == "fp16":
            raise ValueError("FP16 HDN inference supports CUDA devices only.")
        precision = "fp32"
    shape = effective_inference_shape(image_shape, tiling_size=tiling_size, model=model)
    free = reusable = available = None
    if device.type == "cuda":
        free, reusable, available = _cuda_available_including_cache(device)
    manual = (options.precision != "auto" or
              options.sample_batch_size not in (1, "auto"))
    if options.sample_batch_size == "auto":
        batch, reason = 1, "legacy 'auto' batch setting resolves to 1 (experimental batching disabled)"
    else:
        batch = min(int(options.sample_batch_size), requested_samples)
        reason = ("experimental explicit sample batch size" if batch > 1
                  else "batch 1: default, no automatic batching")
    return InferencePlan(
        device=str(device), precision=precision,
        reuse_encoder=callable(getattr(model, "prepare_inference_features", None)),
        requested_samples=requested_samples, sample_batch_size=batch,
        tiling_size=tiling_size, effective_input_shape=shape,
        source="manual" if manual else "auto",
        reason=reason,
        free_gpu_memory_bytes=free,
        reusable_cuda_cache_bytes=reusable,
        estimated_usable_gpu_memory_bytes=available,
    )

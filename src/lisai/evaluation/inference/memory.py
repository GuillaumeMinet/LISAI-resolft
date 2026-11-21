"""Optional, non-invasive CUDA memory diagnostics around real stack inference.

Do not use these numbers to select batch sizes or tiling. The CUDA peak
counters are per device/process, so simultaneous GPU work affects results.
"""
from __future__ import annotations

from dataclasses import dataclass

from lisai.evaluation.inference.results import GPUMemoryInfo


@dataclass(frozen=True)
class _Baseline:
    device: object
    allocated: int
    reserved: int
    driver_free: int
    device_total: int


def start_gpu_memory_measurement(device) -> _Baseline | None:
    import torch

    chosen = torch.device(device)
    if chosen.type != "cuda" or not torch.cuda.is_available():
        return None
    # Statistics are captured after model loading, not around checkpoint IO.
    torch.cuda.synchronize(chosen)
    driver_free, total = torch.cuda.mem_get_info(chosen)
    allocated = torch.cuda.memory_allocated(chosen)
    reserved = torch.cuda.memory_reserved(chosen)
    torch.cuda.reset_peak_memory_stats(chosen)
    return _Baseline(chosen, int(allocated), int(reserved), int(driver_free), int(total))


def finish_gpu_memory_measurement(baseline: _Baseline | None) -> GPUMemoryInfo | None:
    if baseline is None:
        return None
    import torch

    device = baseline.device
    torch.cuda.synchronize(device)
    peak_allocated = int(torch.cuda.max_memory_allocated(device))
    peak_reserved = int(torch.cuda.max_memory_reserved(device))
    allocated_after = int(torch.cuda.memory_allocated(device))
    reserved_after = int(torch.cuda.memory_reserved(device))
    driver_free_after, _ = torch.cuda.mem_get_info(device)
    return GPUMemoryInfo(
        device=str(device),
        allocated_before_bytes=baseline.allocated,
        reserved_before_bytes=baseline.reserved,
        driver_free_before_bytes=baseline.driver_free,
        device_total_bytes=baseline.device_total,
        peak_allocated_bytes=peak_allocated,
        peak_reserved_bytes=peak_reserved,
        allocated_after_bytes=allocated_after,
        reserved_after_bytes=reserved_after,
        driver_free_after_bytes=int(driver_free_after),
    )

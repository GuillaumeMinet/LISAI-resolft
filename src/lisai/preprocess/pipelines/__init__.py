from .paired_single_recon import PairedSingleReconPipeline
from .recon_mltpl_snr import ReconMltplSnrPipeline
from .recon_timelapse_simple import ReconTimelapseSimplePipeline
from .single import SingleReconPipeline

PIPELINES_REGISTRY = {
    "single_recon": SingleReconPipeline,
    "paired_single_recon": PairedSingleReconPipeline,
    "recon_timelapse_simple": ReconTimelapseSimplePipeline,
    "recon_mltpl_snr": ReconMltplSnrPipeline,
}

__all__ = ["PIPELINES_REGISTRY"]

from __future__ import annotations

POST_TRAINING_OVERRIDES = {
    "evaluate": {
        "checkpoint": {"best_or_last": "both"},
        "metrics": ["psnr", "ssim"],
        "saving": {"overwrite": True},
    }
}

__all__ = ["POST_TRAINING_OVERRIDES"]

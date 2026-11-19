import torch

from .stable_exp import StableExponential


def _center_crop(tensor: torch.Tensor, size: int) -> torch.Tensor:
    """Center-crop the spatial dimensions using only Torch indexing."""
    height, width = tensor.shape[-2:]
    if height < size or width < size:
        raise ValueError(f"Cannot center-crop {height}x{width} to {size}x{size}.")
    top = (height - size) // 2
    left = (width - size) // 2
    return tensor[..., top : top + size, left : left + size]


class StableLogVar:

    def __init__(self, logvar, enable_stable=True, var_eps=1e-6):
        """
        Args:
            var_eps: var() has this minimum value.
        """
        self._lv = logvar
        self._enable_stable = enable_stable
        self._eps = var_eps

    def get(self):
        if self._enable_stable is False:
            return self._lv

        return torch.log(self.get_var())

    def get_var(self):
        if self._enable_stable is False:
            return torch.exp(self._lv)
        return StableExponential(self._lv).exp() + self._eps

    def get_std(self):
        return torch.sqrt(self.get_var())

    def centercrop_to_size(self, size):
        if self._lv.shape[-1] == size:
            return

        diff = self._lv.shape[-1] - size
        assert diff > 0 and diff % 2 == 0
        self._lv = _center_crop(self._lv, size)


class StableMean:

    def __init__(self, mean):
        self._mean = mean

    def get(self):
        return self._mean

    def centercrop_to_size(self, size):
        if self._mean.shape[-1] == size:
            return

        diff = self._mean.shape[-1] - size
        assert diff > 0 and diff % 2 == 0
        self._mean = _center_crop(self._mean, size)

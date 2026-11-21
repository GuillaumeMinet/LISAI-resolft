import numpy as np
import torch
from torch import nn

from lisai.lib.hdn.likelihoods import GaussianLikelihood, NoiseModelLikelihood
from lisai.lib.hdn.utils import Interpolate, crop_img_tensor, free_bits_kl, pad_img_tensor
from lisai.models.params import LVAEParams

from .lvae_layers import (
    BottomUpDeterministicResBlock,
    BottomUpLayer,
    TopDownDeterministicResBlock,
    TopDownLayer,
)


class LadderVAE(nn.Module):
    def __init__(
        self,
        params: LVAEParams,
        *,
        device,
        norm_prm,
        noise_model=None,
        img_shape=None,
    ):
        super().__init__()
        self.params = params
        self.color_ch = params.color_ch
        self.z_dims = params.resolved_z_dims()
        self.blocks_per_layer = params.blocks_per_layer
        self.n_layers = len(self.z_dims)
        self.stochastic_skip = params.stochastic_skip
        self.n_filters = params.n_filters
        self.norm = params.norm
        self.gr_norm = params.gr_norm
        self.dropout = params.dropout
        self.free_bits = params.free_bits
        self.learn_top_prior = params.learn_top_prior
        self.img_shape = tuple(img_shape)
        self.res_block_type = params.res_block_type
        self.gated = params.gated
        self.device = device
        self.noiseModel = noise_model
        self.mode_pred = params.mode_pred
        self.use_uncond_mode_at = list(params.use_uncond_mode_at)
        self._global_step = 0

        self.data_mean = norm_prm.get("data_mean")
        self.data_std = norm_prm.get("data_std")
        assert self.data_std is not None
        assert self.data_mean is not None
        self.data_mean_gt = norm_prm.get("data_mean_gt", None)
        self.data_std_gt = norm_prm.get("data_std_gt", None)

        if self.noiseModel is None:
            self.likelihood_form = "gaussian"
        else:
            self.likelihood_form = "noise_model"

        self.downsample = [1] * self.n_layers

        self.overall_downscale_factor = np.power(2, sum(self.downsample))
        if not params.no_initial_downscaling:
            self.overall_downscale_factor *= 2

        assert max(self.downsample) <= self.blocks_per_layer
        assert len(self.downsample) == self.n_layers

        nonlin = {
            "relu": nn.ReLU,
            "leakyrelu": nn.LeakyReLU,
            "elu": nn.ELU,
            "selu": nn.SELU,
        }[params.nonlin]

        stride = 1 if params.no_initial_downscaling else 2
        self.first_bottom_up = nn.Sequential(
            nn.Conv2d(self.color_ch, self.n_filters, 5, padding=2, stride=stride),
            nonlin(),
            BottomUpDeterministicResBlock(
                c_in=self.n_filters,
                c_out=self.n_filters,
                nonlin=nonlin,
                norm=self.norm,
                gr_norm=self.gr_norm,
                dropout=self.dropout,
                res_block_type=self.res_block_type,
            ))

        self.top_down_layers = nn.ModuleList([])
        self.bottom_up_layers = nn.ModuleList([])

        for i in range(self.n_layers):
            is_top = i == self.n_layers - 1

            self.bottom_up_layers.append(
                BottomUpLayer(
                    n_res_blocks=self.blocks_per_layer,
                    n_filters=self.n_filters,
                    downsampling_steps=self.downsample[i],
                    nonlin=nonlin,
                    norm=self.norm,
                    gr_norm=self.gr_norm,
                    dropout=self.dropout,
                    res_block_type=self.res_block_type,
                    gated=self.gated,
                ))

            self.top_down_layers.append(
                TopDownLayer(
                    z_dim=self.z_dims[i],
                    n_res_blocks=self.blocks_per_layer,
                    n_filters=self.n_filters,
                    is_top_layer=is_top,
                    downsampling_steps=self.downsample[i],
                    nonlin=nonlin,
                    merge_type=params.merge_type,
                    norm=self.norm,
                    gr_norm=self.gr_norm,
                    dropout=self.dropout,
                    stochastic_skip=self.stochastic_skip,
                    learn_top_prior=self.learn_top_prior,
                    top_prior_param_shape=self.get_top_prior_param_shape(),
                    res_block_type=self.res_block_type,
                    gated=self.gated,
                    analytical_kl=params.analytical_kl,
                ))

        modules = list()
        if not params.no_initial_downscaling:
            modules.append(Interpolate(scale=2))
        for _ in range(self.blocks_per_layer):
            modules.append(
                TopDownDeterministicResBlock(
                    c_in=self.n_filters,
                    c_out=self.n_filters,
                    nonlin=nonlin,
                    norm=self.norm,
                    gr_norm=self.gr_norm,
                    dropout=self.dropout,
                    res_block_type=self.res_block_type,
                    gated=self.gated,
                ))
        self.final_top_down = nn.Sequential(*modules)

        if self.likelihood_form == 'gaussian':
            self.likelihood = GaussianLikelihood(self.n_filters, self.color_ch)
        elif self.likelihood_form == 'noise_model':
            self.likelihood = NoiseModelLikelihood(
                self.n_filters,
                self.color_ch,
                self.data_mean,
                self.data_std,
                noise_model,
                self.data_mean_gt,
                self.data_std_gt,
            )
        else:
            msg = "Unrecognized likelihood '{}'".format(self.likelihood_form)
            raise RuntimeError(msg)

    def increment_global_step(self):
        """Increments global step by 1."""
        self._global_step += 1
        
    @property
    def global_step(self) -> int:
        """Global step."""
        return self._global_step

    def prepare_inference_features(self, x):
        """Compute bottom-up features for *one* inference image/tile.

        No features are stored on the module: the caller owns their lifetime.
        This method must never be used for training or gradient-bearing passes.
        """
        if self.training or not self.mode_pred or torch.is_grad_enabled():
            raise RuntimeError("Cached LVAE features require eval prediction under no_grad.")
        if x.ndim != 4 or x.shape[0] != 1:
            raise ValueError("Cached LVAE inference requires one NCHW input image.")
        return tuple(self.bottomup_pass(self.pad_input(x)))

    def forward(self, x, y=None, *, cached_bottomup=None):

        img_size = x.size()[2:]
        if cached_bottomup is None:
            # Normal training/evaluation forward is unchanged.
            bu_values = self.bottomup_pass(self.pad_input(x))
        else:
            if self.training or not self.mode_pred or torch.is_grad_enabled() or y is not None:
                raise RuntimeError("Cached LVAE features require eval prediction under no_grad.")
            if x.ndim != 4 or len(cached_bottomup) != self.n_layers:
                raise ValueError("Invalid cached LVAE bottom-up features.")
            if any(feature.shape[0] != 1 for feature in cached_bottomup):
                raise ValueError("Cached LVAE features must belong to a single input.")
            bu_values = tuple(feature.expand(x.shape[0], *feature.shape[1:])
                              for feature in cached_bottomup)

        # Top-down inference/generation
        out, td_data = self.topdown_pass(bu_values)

        #Restore original image size
        out = crop_img_tensor(out, img_size)
        
        # Log likelihood and other info (per data point)
        if self.mode_pred is False and y is not None:
            ll, likelihood_info = self.likelihood(out,y)
        else:
            ll, likelihood_info = self.likelihood(out, x)

        if self.mode_pred is False:
            # kl[i] for each i has length batch_size
            # resulting kl shape: (batch_size, layers)
            kl = torch.cat([kl_layer.unsqueeze(1) for kl_layer in td_data['kl']],
                           dim=1)

            kl_sep = kl.sum(1)
            kl_avg_layerwise = kl.mean(0)
            kl_loss = free_bits_kl(kl, self.free_bits).sum()  # sum over layers
            kl = kl_sep.mean()
        else:
            kl = None
            kl_sep = None
            kl_avg_layerwise = None
            kl_loss = None
            kl = None
            
        output = {
            'll': ll,
            'z': td_data['z'],
            'kl': kl,
            'kl_sep': kl_sep,
            'kl_avg_layerwise': kl_avg_layerwise,
            'kl_spatial': td_data['kl_spatial'],
            'kl_loss': kl_loss,
            'logp': td_data['logprob_p'],
            'out_mean': likelihood_info['mean'],
            'out_mode': likelihood_info['mode'],
            'out_sample': likelihood_info['sample'],
            'likelihood_params': likelihood_info['params']
        }
        return output

    def bottomup_pass(self, x):
        # Bottom-up initial layer
        x = self.first_bottom_up(x)

        # Loop from bottom to top layer, store all deterministic nodes we
        # need in the top-down pass
        bu_values = []
        for i in range(self.n_layers):
            x = self.bottom_up_layers[i](x)
            bu_values.append(x)

        return bu_values

    def topdown_pass(self,
                     bu_values=None,
                     n_img_prior=None,
                     mode_layers=None,
                     constant_layers=None,
                     forced_latent=None):

        # Default: no layer is sampled from the distribution's mode
        if mode_layers is None:
            mode_layers = []
        if constant_layers is None:
            constant_layers = []
        prior_experiment = len(mode_layers) > 0 or len(constant_layers) > 0

        # If the bottom-up inference values are not given, don't do
        # inference, sample from prior instead
        inference_mode = bu_values is not None

        # Check consistency of arguments
        if inference_mode != (n_img_prior is None):
            msg = ("Number of images for top-down generation has to be given "
                   "if and only if we're not doing inference")
            raise RuntimeError(msg)
        if inference_mode and prior_experiment:
            msg = ("Prior experiments (e.g. sampling from mode) are not"
                   " compatible with inference mode")
            raise RuntimeError(msg)

        # Sampled latent variables at each layer
        z = [None] * self.n_layers

        # KL divergence of each layer
        kl = [None] * self.n_layers

        # Spatial map of KL divergence for each layer
        kl_spatial = [None] * self.n_layers

        if forced_latent is None:
            forced_latent = [None] * self.n_layers

        # log p(z) where z is the sample in the topdown pass
        logprob_p = 0.

        # Top-down inference/generation loop
        out = out_pre_residual = None
        for i in reversed(range(self.n_layers)):

            # If available, get deterministic node from bottom-up inference
            try:
                bu_value = bu_values[i]
            except TypeError:
                bu_value = None

            # Whether the current layer should be sampled from the mode
            use_mode = i in mode_layers
            constant_out = i in constant_layers
            use_uncond_mode = i in self.use_uncond_mode_at

            # Input for skip connection
            skip_input = out  # TODO or out_pre_residual? or both?

            # Full top-down layer, including sampling and deterministic part
            out, out_pre_residual, aux = self.top_down_layers[i](
                out,
                skip_connection_input=skip_input,
                inference_mode=inference_mode,
                bu_value=bu_value,
                n_img_prior=n_img_prior,
                use_mode=use_mode,
                force_constant_output=constant_out,
                forced_latent=forced_latent[i],
                mode_pred=self.mode_pred,
                use_uncond_mode=use_uncond_mode
            )
            z[i] = aux['z']  # sampled variable at this layer (batch, ch, h, w)
            kl[i] = aux['kl_samplewise']  # (batch, )
            kl_spatial[i] = aux['kl_spatial']  # (batch, h, w)
            if self.mode_pred is False:
                logprob_p += aux['logprob_p'].mean()  # mean over batch
            else:
                logprob_p = None
        # Final top-down layer
        out = self.final_top_down(out)

        data = {
            'z': z,  # list of tensors with shape (batch, ch[i], h[i], w[i])
            'kl': kl,  # list of tensors with shape (batch, )
            'kl_spatial':
                kl_spatial,  # list of tensors w shape (batch, h[i], w[i])
            'logprob_p': logprob_p,  # scalar, mean over batch
        }
        return out, data

    def pad_input(self, x):
        """
        Pads input x so that its sizes are powers of 2
        :param x:
        :return: Padded tensor
        """
        size = self.get_padded_size(x.size())
        x = pad_img_tensor(x, size)
        return x

    def get_padded_size(self, size):
        """
        Returns the smallest size (H, W) of the image with actual size given
        as input, such that H and W are powers of 2.
        :param size: input size, tuple either (N, C, H, w) or (H, W)
        :return: 2-tuple (H, W)
        """

        # Overall downscale factor from input to top layer (power of 2)
        dwnsc = self.overall_downscale_factor

        # Make size argument into (heigth, width)
        if len(size) == 4:
            size = size[2:]
        if len(size) != 2:
            msg = ("input size must be either (N, C, H, W) or (H, W), but it "
                   "has length {} (size={})".format(len(size), size))
            raise RuntimeError(msg)

        # Output smallest powers of 2 that are larger than current sizes
        padded_size = list(((s - 1) // dwnsc + 1) * dwnsc for s in size)

        return padded_size

    def sample_prior(self, n_imgs, mode_layers=None, constant_layers=None):

        # Generate from prior
        out, _ = self.topdown_pass(n_img_prior=n_imgs,
                                   mode_layers=mode_layers,
                                   constant_layers=constant_layers)
        out = crop_img_tensor(out, self.img_shape)

        # Log likelihood and other info (per data point)
        _, likelihood_data = self.likelihood(out, None)

        return likelihood_data['sample']

    def get_top_prior_param_shape(self, n_imgs=1):
        # TODO num channels depends on random variable we're using
        dwnsc = self.overall_downscale_factor
        sz = self.get_padded_size(self.img_shape)
        h = sz[0] // dwnsc
        w = sz[1] // dwnsc
        c = self.z_dims[-1] * 2  # mu and logvar
        top_layer_shape = (n_imgs, c, h, w)
        return top_layer_shape

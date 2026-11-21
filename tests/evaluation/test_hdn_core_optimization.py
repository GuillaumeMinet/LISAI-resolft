"""P1: HDN batching/feature reuse independent of GPU availability."""
import numpy as np
import pytest
import torch

from lisai.evaluation.inference.engine import lvae_predict, make_prediction, predict
from lisai.models.lvae import LadderVAE
from lisai.models.params import LVAEParams


def _model():
    params = LVAEParams(num_latents=1, z_dims=4, blocks_per_layer=1,
                        n_filters=8, gr_norm=1, dropout=0.)
    model = LadderVAE(params, device=torch.device("cpu"),
                       norm_prm={"data_mean": 0., "data_std": 1.}, img_shape=(32, 32))
    model.mode_pred = True
    return model.eval()


def test_cached_batch_one_matches_original_forward_seeded():
    torch.manual_seed(4)
    model = _model()
    image = torch.rand(1, 1, 32, 32)
    torch.manual_seed(42)
    baseline = lvae_predict(image, 3, model, None, "cpu", reuse_encoder=False)
    torch.manual_seed(42)
    cached = lvae_predict(image, 3, model, None, "cpu", reuse_encoder=True)
    np.testing.assert_allclose(cached, baseline, rtol=0, atol=1e-6)


def test_batch_remainder_exact_draws_and_mean_and_optional_sample_storage():
    class CountingModel(torch.nn.Module):
        mode_pred = True
        def __init__(self):
            super().__init__()
            self.counts = []
        def prepare_inference_features(self, x):
            return (x,)
        def forward(self, x, cached_bottomup=None):
            self.counts.append(x.shape[0])
            return {"out_mean": torch.arange(x.shape[0], dtype=x.dtype)
                    .reshape(-1, 1, 1, 1).expand_as(x) + len(self.counts)}

    image = torch.ones((1, 1, 8, 8))
    model = CountingModel()
    mean = lvae_predict(image, 7, model, None, "cpu", sample_batch_size=3)
    assert model.counts == [3, 3, 1]
    assert mean.shape == (8, 8)
    assert float(mean[0, 0]) == pytest.approx((sum((1, 2, 3)) + sum((2, 3, 4)) + 3) / 7)
    model = CountingModel()
    mean_saved, draws = lvae_predict(image, 7, model, None, "cpu", return_samples=True,
                                     sample_batch_size=3)
    assert len(draws) == 7
    np.testing.assert_allclose(mean_saved, np.mean(draws, axis=0))
    outputs = make_prediction(CountingModel(), image, "cpu", is_lvae=True,
                              num_samples=7, save_samples=False, sample_batch_size=3)
    assert "samples" not in outputs


def test_cached_features_rejected_in_training_and_gradient_mode():
    model = _model()
    x = torch.zeros((1, 1, 32, 32))
    with pytest.raises(RuntimeError, match="eval prediction"):
        model.prepare_inference_features(x)
    with torch.no_grad():
        features = model.prepare_inference_features(x)
    model.train()
    with torch.no_grad(), pytest.raises(RuntimeError, match="eval prediction"):
        model(x, cached_bottomup=features)
    model.eval()
    with pytest.raises(RuntimeError, match="eval prediction"):
        model(x, cached_bottomup=features)


def test_tiling_and_sample_saving_shape_regression():
    class DeterministicFakeLVAE(torch.nn.Module):
        mode_pred = True
        def prepare_inference_features(self, x):
            return (x,)
        def forward(self, x, cached_bottomup=None):
            return {"out_mean": x[:, :1] + 1}

    model = DeterministicFakeLVAE()
    x = torch.zeros((1, 1, 256, 256))
    mean_only = predict(model, x, device="cpu", is_lvae=True, num_samples=5,
                        tiling_size=128, sample_batch_size=2)
    assert mean_only["prediction"].shape == (1, 1, 256, 256) or mean_only["prediction"].shape == (256, 256)
    assert "samples" not in mean_only
    full = predict(model, x, device="cpu", is_lvae=True, num_samples=5,
                   tiling_size=128, sample_batch_size=2, save_samples=True)
    assert full["samples"].shape == (5, 256, 256)
    np.testing.assert_allclose(full["prediction"], mean_only["prediction"])


def test_invalid_batches_and_samples():
    model = _model()
    x = torch.zeros(1, 1, 32, 32)
    for n, batch in [(0, 1), (1, 0), (True, 2), (2, True)]:
        with pytest.raises(ValueError, match="positive integer"):
            lvae_predict(x, n, model, None, "cpu", sample_batch_size=batch)


def test_training_forward_retains_gradients():
    model = _model().train()
    model.mode_pred = False
    image = torch.rand((1, 1, 32, 32))
    output = model(image)
    assert output["kl_loss"] is not None
    loss = -output["ll"].mean() + output["kl_loss"]
    loss.backward()
    assert any(param.grad is not None for param in model.parameters())

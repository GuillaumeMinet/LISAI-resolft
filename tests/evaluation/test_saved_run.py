from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import lisai.evaluation.saved_run as saved_run_mod
from lisai.models.params import LVAEParams


class FakePaths:
    def run_dir(self, *, dataset_name, models_subfolder, exp_name):
        return Path('/runs') / dataset_name / models_subfolder / exp_name

    def cfg_train_path(self, *, run_dir):
        return Path(run_dir) / 'config_train.yaml'


def _raw_lvae_config() -> dict:
    return {
        'experiment': {'mode': 'train', 'exp_name': 'exp_a'},
        'routing': {'data_subfolder': 'raw'},
        'data': {
            'dataset_name': 'dataset_a',
            'paired': True,
            'input': 'inp',
            'target': 'gt',
            'patch_size': 64,
            'downsampling': {
                'supervised_training': True,
                'downsamp_factor': 2,
                'downsamp_method': 'blur',
            },
            'timelapse_prm': {'context_length': 1},
        },
        'model': {'architecture': 'lvae', 'parameters': {'num_latents': 4, 'z_dims': 32}},
        'training': {'n_epochs': 1},
        'normalization': {'norm_prm': {'clip': 0}},
        'model_norm_prm': {'data_mean': 1.0},
        'noise_model': {'name': 'noise_A'},
        'saving': {'enabled': True, 'state_dict': True, 'entire_model': True},
    }


def _load_with_raw_config(monkeypatch, raw_cfg: dict, run_dir: Path):
    monkeypatch.setattr(saved_run_mod, 'Paths', lambda _settings: FakePaths())
    monkeypatch.setattr(saved_run_mod, 'load_yaml', lambda path: raw_cfg)
    return saved_run_mod.load_saved_run(run_dir)



def test_load_saved_run_validates_and_extracts_fields(monkeypatch):
    raw_cfg = _raw_lvae_config()
    monkeypatch.setattr(saved_run_mod, 'Paths', lambda _settings: FakePaths())

    run_dir = saved_run_mod.resolve_run_dir(
        dataset_name='dataset_a',
        subfolder='subfolder_a',
        exp_name='exp_a',
    )
    saved_run = _load_with_raw_config(monkeypatch, raw_cfg, run_dir)

    assert run_dir == Path('/runs/dataset_a/subfolder_a/exp_a')
    assert saved_run.run_dir == run_dir
    assert saved_run.experiment_name == 'exp_a'
    assert saved_run.dataset_name == 'dataset_a'
    assert saved_run.data_subfolder == 'raw'
    assert saved_run.model_architecture == 'lvae'
    assert saved_run.is_lvae is True
    assert isinstance(saved_run.model_parameters, LVAEParams)
    assert saved_run.model_parameters.num_latents == 4
    assert saved_run.model_parameters.resolved_z_dims() == [32, 32, 32, 32]
    assert saved_run.data_norm_prm == {'clip': 0}
    assert saved_run.model_norm_prm == {'data_mean': 1.0}
    assert saved_run.noise_model_name == 'noise_A'
    assert saved_run.checkpoint_methods == ('state_dict', 'full_model')
    assert saved_run.patch_size == 64
    assert saved_run.downsamp_factor == 2
    assert saved_run.upsampling_factor == 1
    assert saved_run.context_length == 1
    assert saved_run.default_tiling_size == 300


def test_load_saved_run_prefers_explicit_inference_default_tiling_size(monkeypatch):
    raw_cfg = deepcopy(_raw_lvae_config())
    raw_cfg['inference'] = {'default_tiling_size': 2000}

    saved_run = _load_with_raw_config(monkeypatch, raw_cfg, Path('/runs/dataset_a/HDN/exp_a'))

    assert saved_run.default_tiling_size == 2000


def test_load_saved_run_allows_disabling_default_tiling(monkeypatch):
    raw_cfg = deepcopy(_raw_lvae_config())
    raw_cfg['inference'] = {'default_tiling_size': 'off'}

    saved_run = _load_with_raw_config(monkeypatch, raw_cfg, Path('/runs/dataset_a/HDN/exp_a'))

    assert saved_run.default_tiling_size is None


def test_load_saved_run_uses_legacy_unetrcan_compat_default(monkeypatch):
    raw_cfg = deepcopy(_raw_lvae_config())
    raw_cfg['routing'] = {'data_subfolder': 'raw', 'models_subfolder': 'denoising_legacy'}
    raw_cfg['model'] = {'architecture': 'unet_rcan', 'parameters': {'upsampling_factor': 1}}
    raw_cfg['noise_model'] = None

    saved_run = _load_with_raw_config(
        monkeypatch,
        raw_cfg,
        Path('/runs/dataset_a/models/denoising_legacy/UNetRCAN_single_to_avg'),
    )

    assert saved_run.default_tiling_size == 2000

"""M3-D bundle integration with real LISAI architectures and generated test weights.

These are *not* the published Zenodo weights.  Their purpose is to catch issues
that a mocked forward() or a manifest-only package test cannot detect.
"""
from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
import tifffile
import torch
import yaml

from lisai import (
    ApplyInferenceOverrides,
    ApplyInputOverrides,
    ApplyOverrides,
    InputNormalization,
    predict,
)
from lisai.config.models import ResolvedExperiment
from lisai.infra.paths import Paths
from lisai.models.loader import init_model
from lisai.models.load_nm import load_noise_model_from_paths
from lisai.models.params import LVAEParams, UNetRCANParams
from lisai.promoted_models.install import install_model_archive
from lisai.promoted_models.schema import (
    PromotedModelArtifacts,
    PromotedModelDefinition,
    PromotedModelManifest,
    PromotedModelSource,
    PromotedNoiseModelArtifacts,
    PromotedTrainingData,
)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _bundle(tmp_path: Path, *, architecture: str) -> tuple[str, Path]:
    name = 'm3d_generated_hdn' if architecture == 'lvae' else 'm3d_generated_fiveframe'
    source = tmp_path / f'{name}-source'
    source.mkdir()
    device = torch.device('cpu')
    model_norm = {'data_mean': 10., 'data_std': 4.}
    noise = None
    noise_artifacts = None
    if architecture == 'lvae':
        params = LVAEParams(
            num_latents=1, z_dims=4, blocks_per_layer=1, n_filters=8,
            gr_norm=1, dropout=0.,
        )
        noise_dir = source / 'artifacts' / 'noise_model'
        noise_dir.mkdir(parents=True)
        gmm = noise_dir / 'GMM.npz'
        np.savez(
            gmm,
            min_signal=np.array([0.], dtype=np.float32),
            max_signal=np.array([100.], dtype=np.float32),
            trained_weight=np.array([[0., 0.], [2., 0.], [0., 0.]], dtype=np.float32),
            min_sigma=np.array(1., dtype=np.float32),
        )
        (noise_dir / 'norm_prm.json').write_text(
            json.dumps({'data_mean': 10., 'data_std': 4.}), encoding='utf-8',
        )
        noise, _ = load_noise_model_from_paths(gmm, noise_dir / 'norm_prm.json', device)
        noise_artifacts = PromotedNoiseModelArtifacts(
            name='M3D_GMM', model='artifacts/noise_model/GMM.npz',
            norm_prm='artifacts/noise_model/norm_prm.json',
        )
        task = 'denoising_hdn'
        data = {'dataset_name': 'M3D_mock', 'patch_size': 32}
    else:
        params = UNetRCANParams(
            upsampling_factor=2, upsampling_net='unet',
            UNet_prm={'in_channels': 5, 'out_channels': 5, 'feat': 8, 'depth': 2,
                      'gr_norm': 1, 'ch': 16},
            RCAN_prm={'num_features': 8, 'num_rg': 1, 'num_rcab': 1, 'reduction': 2},
        )
        task = 'upsamp_multiframes'
        data = {'dataset_name': 'M3D_mock', 'patch_size': 32,
                'timelapse_prm': {'context_length': 5}}
    config = {
        'experiment': {'mode': 'train', 'exp_name': 'm3d_fixture', 'task': 'custom'},
        'data': data,
        'model': {'architecture': architecture, 'parameters': params.model_dump()},
        'normalization': {'norm_prm': {'clip': 0}},
        'model_norm_prm': model_norm,
        'noise_model': {'name': 'M3D_GMM'} if noise_artifacts else None,
        'saving': {'state_dict': True, 'entire_model': False},
        'inference': {'default_tiling_size': 'off'},
    }
    cfg = ResolvedExperiment.model_validate(config)
    (source / 'config_train.yaml').write_text(
        yaml.safe_dump(cfg.model_dump(mode='json', exclude_none=True)), encoding='utf-8',
    )
    torch.manual_seed(17)
    model = init_model(
        architecture=architecture, model_prm=params, device=device,
        model_norm_prm=model_norm, noise_model=noise, img_shape=32,
    )
    torch.save(model.state_dict(), source / 'weights.pt')
    (source / 'config_inference.yaml').write_text(
        'apply:\n  saving:\n    lvae_save_samples: false\n', encoding='utf-8',
    )
    artifacts = PromotedModelArtifacts(
        inference_config='config_inference.yaml', noise_model=noise_artifacts,
    )
    manifest = PromotedModelManifest(
        name=name,
        model=PromotedModelDefinition(task=task, architecture=architecture),
        training_data=PromotedTrainingData(dataset='M3D_mock'),
        source=PromotedModelSource(
            run_id='01ARZ3NDEKTSV4RRFFQ69G5FAV', run_name='m3d_fixture',
            run_status='completed', checkpoint_selector='best',
            checkpoint_filename='model_best_state_dict.pt',
        ),
        artifacts=artifacts,
        checksums={
            path.relative_to(source).as_posix(): _sha(path)
            for path in source.rglob('*') if path.is_file()
        },
    )
    (source / 'lisai_model.yaml').write_text(
        yaml.safe_dump(manifest.model_dump(mode='json', exclude_none=True)),
        encoding='utf-8',
    )
    archive_path = tmp_path / f'{name}.lisai.zip'
    with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(p for p in source.rglob('*') if p.is_file()):
            archive.write(path, path.relative_to(source).as_posix())
    return name, archive_path


def _overrides(*, mode='drop', normalization='training', gain=1., offset=0.):
    return ApplyOverrides(
        input=ApplyInputOverrides(normalization=InputNormalization(
            mode=normalization, gain=gain, offset=offset,
        )),
        inference=ApplyInferenceOverrides(
            tiling_size='off', temporal_edge_mode=mode, lvae_num_samples=2,
        ),
    )


@pytest.mark.parametrize('architecture', ['lvae', 'unet_rcan'])
def test_generated_real_architecture_bundle_roundtrip(tmp_path, settings, architecture):
    name, archive = _bundle(tmp_path, architecture=architecture)
    original_archive_hash = _sha(archive)
    installed = install_model_archive(archive, paths=Paths(settings))
    assert installed.model.manifest.name == name
    assert _sha(archive) == original_archive_hash
    assert installed.model.weights_path.exists()
    if architecture == 'lvae':
        assert installed.model.noise_model_path.is_file()
        assert installed.model.noise_model_norm_prm_path.is_file()
        arr = np.random.default_rng(44).uniform(4, 14, (32, 32)).astype('float32')
    else:
        arr = np.random.default_rng(44).uniform(4, 14, (7, 32, 32)).astype('float32')
    source_copy = arr.copy()
    path = tmp_path / f'{name}.tif'
    tifffile.imwrite(path, arr)
    for normalization in ('training', 'manual', 'image'):
        for mode in ('drop', 'black_output', 'black_input') if architecture != 'lvae' else ('drop',):
            options = _overrides(
                mode=mode, normalization=normalization,
                gain=1.3, offset=0.5,
            )
            torch.manual_seed(99)
            result, info = predict(name, arr, settings=settings, overrides=options, return_info=True)
            assert result.dtype == np.float32
            assert np.isfinite(result).all()
            np.testing.assert_array_equal(arr, source_copy)
            if architecture != 'lvae':
                assert result.shape == ((3 if mode == 'drop' else 7), 64, 64)
                assert info.source_frame_indices == (
                    (2, 3, 4) if mode == 'drop' else tuple(range(7))
                )
                if mode == 'black_output':
                    assert np.count_nonzero(result[[0, 1, -2, -1]]) == 0
                else:
                    assert info.black_output_indices == ()
            else:
                assert result.shape == (32, 32)
            torch.manual_seed(99)
            saved = predict(name, path, settings=settings, overrides=options,
                            output_dir=tmp_path / f'prediction-{normalization}-{mode}')
            assert len(saved.prediction_paths) == 1
            assert saved.info_by_path[saved.prediction_paths[0]] == info
            from_file = tifffile.imread(saved.prediction_paths[0])
            np.testing.assert_allclose(from_file, result, rtol=1e-5, atol=1e-4)


def test_generated_bundle_detects_tampered_noise_resources(tmp_path, settings):
    name, archive = _bundle(tmp_path, architecture='lvae')
    tampered = tmp_path / f'tampered-{name}.lisai.zip'
    with zipfile.ZipFile(archive) as source, zipfile.ZipFile(tampered, 'w') as output:
        for info in source.infolist():
            data = source.read(info.filename)
            if info.filename == 'artifacts/noise_model/GMM.npz':
                data = b'invalid'
            output.writestr(info, data)
    with pytest.raises(ValueError, match='SHA256'):
        install_model_archive(tampered, paths=Paths(settings))

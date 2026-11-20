"""L3: direct prediction from a saved run, without promotion."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import tifffile
import torch
import yaml

from lisai import ApplyOverrides, PredictionResult, RunRef, predict
from lisai.config.models import ResolvedExperiment
from lisai.config.models.inference import ApplyInferenceOverrides
from lisai.infra.paths import Paths
from lisai.models.loader import init_model
from lisai.models.load_nm import load_noise_model_from_paths
from lisai.models.params import LVAEParams
from lisai.runs.io import write_run_metadata_atomic
from lisai.runs.lifecycle import create_run_metadata, finalize_run_completed
from lisai.runs.scanner import scan_runs
from lisai.evaluation.defaults import resolve_apply_config
from lisai.evaluation.run_apply_model import run_apply_model


def _run_fixture(settings, *, exp_name='hdn_training_run'):
    """Create a real HDN saved-run snapshot and checkpoints in a real workspace."""
    paths = Paths(settings)
    dataset = 'runref_hdn'
    subfolder = 'lvae'
    run_dir = paths.run_dir(dataset_name=dataset, models_subfolder=subfolder, exp_name=exp_name)
    run_dir.mkdir(parents=True)
    noise_name = 'runref_GMM'
    nm_path = paths.noise_model_path(noiseModel_name=noise_name)
    norm_path = paths.noise_model_norm_prm_path(noiseModel_name=noise_name)
    nm_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        nm_path,
        min_signal=np.array([0.], dtype=np.float32),
        max_signal=np.array([100.], dtype=np.float32),
        trained_weight=np.array([[0., 0.], [2., 0.], [0., 0.]], dtype=np.float32),
        min_sigma=np.array(1., dtype=np.float32),
    )
    norm_path.write_text(json.dumps({'data_mean': 10., 'data_std': 4.}), encoding='utf-8')
    params = LVAEParams(
        num_latents=1, z_dims=4, blocks_per_layer=1, n_filters=8,
        gr_norm=1, dropout=0.,
    )
    cfg = ResolvedExperiment.model_validate({
        'experiment': {'mode': 'train', 'exp_name': exp_name, 'task': 'custom'},
        'data': {'dataset_name': dataset, 'patch_size': 32},
        'model': {'architecture': 'lvae', 'parameters': params.model_dump()},
        'normalization': {'norm_prm': {'clip': 0}},
        'model_norm_prm': {'data_mean': 10., 'data_std': 4.},
        'noise_model': {'name': noise_name},
        'saving': {'state_dict': True, 'entire_model': False},
        'inference': {'default_tiling_size': 'off'},
    })
    paths.cfg_train_path(run_dir=run_dir).write_text(
        yaml.safe_dump(cfg.model_dump(mode='json', exclude_none=True)), encoding='utf-8',
    )
    noise, _ = load_noise_model_from_paths(nm_path, norm_path, torch.device('cpu'))
    torch.manual_seed(21)
    model = init_model(
        architecture='lvae', model_prm=params, device=torch.device('cpu'),
        model_norm_prm={'data_mean': 10., 'data_std': 4.},
        noise_model=noise, img_shape=32,
    )
    checkpoint_dir = paths.checkpoint_path(
        run_dir=run_dir, load_method='state_dict', best_or_last='best',
    ).parent
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_dir / 'model_best_state_dict.pt')
    # Deterministically different last checkpoint, to verify selector forwarding.
    with torch.no_grad():
        for param in model.parameters():
            param.add_(0.001)
    torch.save(model.state_dict(), checkpoint_dir / 'model_last_state_dict.pt')
    meta = create_run_metadata(run_dir, dataset=dataset, paths=paths, model_subfolder=subfolder)
    finalize_run_completed(run_dir)
    return meta.run_id, run_dir, nm_path


def _options(checkpoint='best'):
    return ApplyOverrides(
        checkpoint={'best_or_last': checkpoint},
        inference=ApplyInferenceOverrides(tiling_size='off', lvae_num_samples=2),
        saving={'lvae_save_samples': False},
    )


def test_runref_reexport_and_run_lookup_are_stable(settings):
    """The cleanup keeps public type identity and exact workspace selection."""
    from lisai.api import RunRef as ApiRunRef
    from lisai.model_sources import RunRef as ModelSourceRunRef
    from lisai.runs.lookup import resolve_run_ref

    assert RunRef is ApiRunRef is ModelSourceRunRef
    run_id, run_dir, _ = _run_fixture(settings)
    discovered = resolve_run_ref(RunRef(run_id), paths=Paths(settings))
    assert discovered.run_dir == run_dir
    assert discovered.metadata.run_id == run_id


def test_runref_validates_and_normalizes_id():
    assert RunRef(' 01ARZ3NDEKTSV4RRFFQ69G5FAV ').run_id == '01ARZ3NDEKTSV4RRFFQ69G5FAV'
    with pytest.raises(ValueError, match='nonempty'):
        RunRef(' ')
    with pytest.raises(ValueError, match='nonempty'):
        RunRef(None)  # type: ignore[arg-type]


def test_runref_hdn_array_and_file_match_cli_run_apply(settings, tmp_path):
    run_id, run_dir, _ = _run_fixture(settings)
    assert any(run.metadata.run_id == run_id for run in scan_runs(paths=Paths(settings)).runs)
    raw = np.random.default_rng(12).uniform(1, 20, (32, 32)).astype('float32')
    original = raw.copy()
    src = tmp_path / 'cell.tif'
    tifffile.imwrite(src, raw)
    opts = _options()
    torch.manual_seed(41)
    result, info = predict(RunRef(run_id), raw, settings=settings, overrides=opts, return_info=True)
    assert result.shape == (32, 32)
    assert result.dtype == np.float32
    assert np.isfinite(result).all()
    np.testing.assert_array_equal(raw, original)
    assert info.source_frame_indices == (0,)
    torch.manual_seed(41)
    saved = predict(RunRef(run_id), src, settings=settings, overrides=opts,
                    output_dir=tmp_path / 'from_public_api')
    assert isinstance(saved, PredictionResult)
    assert saved.info_by_path[saved.prediction_paths[0]] == info
    np.testing.assert_allclose(tifffile.imread(saved.prediction_paths[0]), result, atol=1e-4)

    # Existing run-based CLI engine (the same function called by `lisai apply --run-id`).
    cfg = resolve_apply_config(overrides=opts, stg=settings)
    cfg = cfg.model_copy(update={
        'input': cfg.input.model_copy(update={'filters': ['tif', 'tiff']}),
    })
    torch.manual_seed(41)
    legacy = run_apply_model(
        cfg, model_dataset='runref_hdn', model_subfolder='lvae',
        model_name=run_dir.name, data_path=src, paths=Paths(settings),
        return_result=True,
    )
    np.testing.assert_allclose(tifffile.imread(legacy.prediction_paths[0]), result, atol=1e-4)


def test_best_last_checkpoint_selected_by_saved_run(settings, monkeypatch):
    run_id, run_dir, _ = _run_fixture(settings)
    from lisai.evaluation import runtime as runtime_module
    original = runtime_module.initialize_runtime
    selected = []

    def capture(**kwargs):
        instance = original(**kwargs)
        selected.append(instance.checkpoint_path.name)
        return instance

    monkeypatch.setattr(runtime_module, 'initialize_runtime', capture)
    raw = np.full((32, 32), 11., dtype=np.float32)
    predict(RunRef(run_id), raw, settings=settings, overrides=_options('last'))
    predict(RunRef(run_id), raw, settings=settings, overrides=_options('best'))
    assert selected == ['model_last_state_dict.pt', 'model_best_state_dict.pt']


def test_missing_run_wrong_workspace_and_inconsistent_metadata(settings, settings_factory):
    run_id, run_dir, _ = _run_fixture(settings)
    with pytest.raises(FileNotFoundError, match='not found in workspace'):
        predict(RunRef(run_id), np.ones((32, 32)), settings=settings_factory('other'))
    with pytest.raises(FileNotFoundError, match='not found in workspace'):
        predict(RunRef('01ARZ3NDEKTSV4RRFFQ69G5FAV'), np.ones((32, 32)), settings=settings)
    from lisai.runs.io import read_run_metadata
    altered = read_run_metadata(run_dir).model_copy(update={'dataset': 'wrong_dataset'})
    write_run_metadata_atomic(run_dir, altered)
    with pytest.raises(ValueError, match='inconsistent metadata/path'):
        predict(RunRef(run_id), np.ones((32, 32)), settings=settings)


def test_ambiguous_run_ids_fail_without_prompt(settings, monkeypatch):
    run_id, run_dir, _ = _run_fixture(settings)
    from lisai.runs import scanner
    from dataclasses import replace
    original = scanner.scan_runs
    def fake_scan(*, paths):
        valid = original(paths=paths)
        return replace(valid, runs=(valid.runs[0], valid.runs[0]))
    monkeypatch.setattr(scanner, 'scan_runs', fake_scan)
    with pytest.raises(ValueError, match='ambiguous'):
        predict(RunRef(run_id), np.ones((32, 32)), settings=settings)


def test_missing_saved_snapshot_checkpoint_and_noise_model(settings):
    run_id, run_dir, noise_path = _run_fixture(settings)
    raw = np.ones((32, 32), dtype=np.float32)
    config_path = Paths(settings).cfg_train_path(run_dir=run_dir)
    snapshot_text = config_path.read_text()
    config_path.unlink()
    with pytest.raises(FileNotFoundError):
        predict(RunRef(run_id), raw, settings=settings)
    config_path.write_text(snapshot_text)
    for path in (run_dir / 'checkpoints').glob('model_*_state_dict.pt'):
        path.unlink()
    with pytest.raises(FileNotFoundError, match='checkpoint'):
        predict(RunRef(run_id), raw, settings=settings)
    run_id, run_dir, noise_path = _run_fixture(settings, exp_name='hdn_other')
    noise_path.unlink()
    with pytest.raises(FileNotFoundError, match='Noise model'):
        predict(RunRef(run_id), raw, settings=settings)


def test_promoted_name_and_runref_are_distinct(settings):
    with pytest.raises(FileNotFoundError, match='not found in workspace'):
        predict(RunRef('01ARZ3NDEKTSV4RRFFQ69G5FAV'), np.ones((32, 32)), settings=settings)
    with pytest.raises((FileNotFoundError, KeyError, ValueError)) as exc:
        predict('01ARZ3NDEKTSV4RRFFQ69G5FAV', np.ones((32, 32)), settings=settings)
    assert 'promoted' in str(exc.value).lower() or 'installed' in str(exc.value).lower()

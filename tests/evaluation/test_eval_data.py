from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
from tifffile import imwrite

import lisai.evaluation.data as data_mod
from lisai.evaluation.saved_run import SavedTrainingRun
from lisai.models.params import UNetParams
from lisai.preprocess.core.dataset_registry import DatasetRegistry


class FakePaths:
    def __init__(self, root: Path):
        self.root = root

    def dataset_dir(self, *, dataset_name, data_subfolder):
        return self.root / dataset_name / data_subfolder

    def dataset_registry_path(self):
        return self.root / 'dataset_registry.yml'


def _write_preprocess_registry_entry(registry_path: Path, *, dataset_name: str, data_format: str) -> None:
    registry = DatasetRegistry(registry_path)
    registry.update_after_preprocess(
        dataset_name=dataset_name,
        data_type='recon',
        data_format=data_format,
        structure=['inp'],
        outputs=[{'key': 'inp', 'path': 'inp', 'role': 'inp', 'axes': 'YX'}],
        result=SimpleNamespace(n_files=1, n_frames=None, snr_levels=None, timepoints=None),
        split_summary={'counts': {'train': 1, 'val': 0, 'test': 0}},
    )
    registry.save()


def _make_saved_run(data_cfg: dict | None = None, split_manifest: dict | None = None) -> SavedTrainingRun:
    resolved_data_cfg = {
        'dataset_name': 'dataset_a',
        'canonical_load': True,
        'paired': False,
        'input': 'inp',
        'patch_size': 64,
    }
    if data_cfg is not None:
        resolved_data_cfg.update(data_cfg)

    return SavedTrainingRun(
        run_dir=Path('/runs/dataset_a/exp_a'),
        experiment_name='exp_a',
        dataset_name='dataset_a',
        data_subfolder='raw',
        data_cfg=resolved_data_cfg,
        model_architecture='unet',
        model_parameters=UNetParams(),
        data_norm_prm={'clip': 0},
        model_norm_prm={'data_mean': 1.0, 'data_std': 2.0},
        noise_model_name=None,
        checkpoint_methods=('state_dict',),
        patch_size=64,
        downsamp_factor=1,
        upsampling_factor=1,
        context_length=None,
        default_tiling_size=1024,
        split_manifest=split_manifest,
    )


def test_build_eval_source_resolves_data_and_applies_overrides(monkeypatch, tmp_path: Path):
    saved_run = _make_saved_run()
    data_root = tmp_path / 'data'
    data_dir = data_root / 'dataset_a' / 'override_subfolder'
    inp_dir = data_dir / 'inp' / 'val'
    gt_dir = data_dir / 'gt_folder' / 'val'
    inp_dir.mkdir(parents=True)
    gt_dir.mkdir(parents=True)
    imwrite(inp_dir / 'img_a.tif', np.ones((4, 5), dtype=np.float32) * 3)
    imwrite(gt_dir / 'img_a.tif', np.ones((4, 5), dtype=np.float32))
    _write_preprocess_registry_entry(
        data_root / 'dataset_registry.yml',
        dataset_name='dataset_a',
        data_format='single',
    )

    monkeypatch.setattr(data_mod, 'Paths', lambda _settings: FakePaths(data_root))

    source = data_mod.build_eval_source(
        saved_run,
        split='val',
        crop_size=32,
        eval_gt='gt_folder',
        data_prm_update={'subfolder': 'override_subfolder'},
    )

    assert isinstance(source, data_mod.EvalSampleSource)
    cfg = source.config
    assert cfg.data_dir == data_dir
    assert cfg.dataset_info['data_format'] == 'single'
    assert cfg.target == 'gt_folder'
    assert cfg.initial_crop == 32
    assert cfg.split == 'val'
    assert cfg.model_norm_prm['data_mean_gt'] == 0
    assert cfg.model_norm_prm['data_std_gt'] == 1
    assert len(source.items) == 1

    item = source.items[0]
    sample = next(iter(source))
    assert item.sample_name(0) == 'img_a'
    assert sample.x.shape == (1, 4, 5)
    assert sample.y is not None
    assert sample.y.shape == (1, 4, 5)



def test_resolve_eval_data_dir_returns_explicit_data_dir():
    saved_run = _make_saved_run()

    out = data_mod.resolve_eval_data_dir(saved_run, {'data_dir': '/custom/data'})

    assert out == Path('/custom/data')


def test_build_eval_source_streams_mixed_size_inputs(monkeypatch, tmp_path: Path):
    saved_run = _make_saved_run()
    data_dir = tmp_path / 'dataset'
    inp_dir = data_dir / 'inp' / 'test'
    inp_dir.mkdir(parents=True)
    imwrite(inp_dir / 'a_small.tif', np.ones((3, 4), dtype=np.float32))
    imwrite(inp_dir / 'b_large.tif', np.ones((5, 7), dtype=np.float32))

    monkeypatch.setattr(
        data_mod,
        'load_dataset_info',
        lambda path, dataset_name: {'data_format': 'single'},
    )

    source = data_mod.build_eval_source(
        saved_run,
        split='test',
        data_prm_update={'data_dir': str(data_dir)},
    )

    shapes = []
    for sample in source:
        shapes.append(tuple(sample.x.shape))
        assert sample.y is None

    assert shapes == [(1, 3, 4), (1, 5, 7)]


def test_build_eval_source_uses_split_manifest(monkeypatch, tmp_path: Path):
    manifest = {
        'version': 1,
        'splits': {
            'train': [{'input': 'train_a.tif', 'target': None}],
            'val': [{'input': 'val_a.tif', 'target': None}],
            'test': [{'input': 'test_a.tif', 'target': None}],
        },
    }
    saved_run = _make_saved_run(
        data_cfg={
            'prep_before': False,
            'input': 'dump',
        },
        split_manifest=manifest,
    )
    data_dir = tmp_path / 'dataset'
    dump_dir = data_dir / 'dump'
    dump_dir.mkdir(parents=True)
    imwrite(dump_dir / 'train_a.tif', np.ones((3, 4), dtype=np.float32))
    imwrite(dump_dir / 'val_a.tif', np.ones((4, 5), dtype=np.float32))
    imwrite(dump_dir / 'test_a.tif', np.ones((5, 6), dtype=np.float32))

    monkeypatch.setattr(
        data_mod,
        'load_dataset_info',
        lambda path, dataset_name: {'data_format': 'single'},
    )

    source = data_mod.build_eval_source(
        saved_run,
        split='test',
        data_prm_update={'data_dir': str(data_dir)},
    )

    assert len(source.items) == 1
    assert source.items[0].inp_path == dump_dir / 'test_a.tif'
    sample = next(iter(source))
    assert sample.x.shape == (1, 5, 6)
    assert sample.y is None


def test_eval_source_keeps_timelapse_item_and_time_indices(monkeypatch, tmp_path: Path):
    saved_run = _make_saved_run(
        data_cfg={
            'data_format': 'timelapse',
            'timelapse_prm': {'context_length': 3},
        }
    )
    data_dir = tmp_path / 'dataset'
    inp_dir = data_dir / 'inp' / 'test'
    inp_dir.mkdir(parents=True)
    stack = np.arange(5, dtype=np.float32)[:, None, None] * np.ones((5, 4, 5), dtype=np.float32)
    imwrite(inp_dir / 'stack_a.tif', stack)

    monkeypatch.setattr(
        data_mod,
        'load_dataset_info',
        lambda path, dataset_name: {'data_format': 'timelapse'},
    )

    source = data_mod.build_eval_source(
        saved_run,
        split='test',
        data_prm_update={'data_dir': str(data_dir)},
    )

    assert len(source.items) == 1
    item = source.items[0]
    assert item.name == 'stack_a'
    assert item.data_format == 'timelapse'
    assert item.sample_count == 3
    assert item.time_indices == (1, 2, 3)

    indexed_samples = list(item.iter_samples(source.config))
    assert [sample_index for sample_index, _ in indexed_samples] == [0, 1, 2]
    assert [item.sample_time_index(sample_index) for sample_index, _ in indexed_samples] == [1, 2, 3]
    assert [item.sample_name(sample_index) for sample_index, _ in indexed_samples] == [
        'stack_a_1',
        'stack_a_2',
        'stack_a_3',
    ]
    assert [tuple(sample.x.shape) for _, sample in indexed_samples] == [(3, 4, 5)] * 3

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
from tifffile import imwrite

import lisai.evaluation.data as data_mod
from lisai.config.io.yaml import save_yaml
from lisai.evaluation.saved_run import SavedTrainingRun
from lisai.models.params import LVAEParams, UNetParams
from lisai.preprocess.core.dataset_registry import DatasetRegistry


class FakePaths:
    def __init__(self, root: Path):
        self.root = root

    def dataset_dir(self, *, dataset_name, data_subfolder, usage="training"):
        return self.root / dataset_name / data_subfolder

    def dataset_registry_path(self):
        return self.root / 'dataset_registry.yml'

    def dataset_preprocess_dir(self, *, dataset_name, data_type='', usage="training"):
        return self.root / dataset_name / 'preprocess' / data_type


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


def _write_registry_eval_gt(registry_path: Path, *, dataset_name: str, eval_gt: str | None) -> None:
    save_yaml(
        {
            dataset_name: {
                'data_format': 'single',
                'defaults': {
                    'recon': {
                        'input': 'inp',
                        'target': None,
                        'eval_gt': eval_gt,
                    }
                },
            }
        },
        registry_path,
    )


def _write_eval_split(data_dir: Path, *, input_name: str = 'inp', target_names: tuple[str, ...] = ()) -> None:
    inp_dir = data_dir / input_name / 'test'
    inp_dir.mkdir(parents=True)
    imwrite(inp_dir / 'img_a.tif', np.ones((4, 5), dtype=np.float32) * 3)
    for target_name in target_names:
        target_dir = data_dir / target_name / 'test'
        target_dir.mkdir(parents=True)
        imwrite(target_dir / 'img_a.tif', np.ones((4, 5), dtype=np.float32))


def _make_saved_run(
    data_cfg: dict | None = None,
    split_manifest: dict | None = None,
    *,
    model_architecture: str = 'unet',
    model_parameters=None,
    model_norm_prm: dict | None = None,
) -> SavedTrainingRun:
    resolved_data_cfg = {
        'dataset_name': 'dataset_a',
        'canonical_load': True,
        'paired': False,
        'input': 'inp',
        'patch_size': 64,
    }
    if data_cfg is not None:
        resolved_data_cfg.update(data_cfg)

    if model_parameters is None:
        model_parameters = LVAEParams() if model_architecture == 'lvae' else UNetParams()
    if model_norm_prm is None:
        model_norm_prm = {'data_mean': 1.0, 'data_std': 2.0}

    return SavedTrainingRun(
        run_dir=Path('/runs/dataset_a/exp_a'),
        experiment_name='exp_a',
        dataset_name='dataset_a',
        data_subfolder='raw',
        data_cfg=resolved_data_cfg,
        model_architecture=model_architecture,
        model_parameters=model_parameters,
        data_norm_prm={'clip': 0},
        model_norm_prm=model_norm_prm,
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
        data_overrides={'subfolder': 'override_subfolder'},
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


def test_build_eval_source_uses_registry_eval_gt_when_omitted(monkeypatch, tmp_path: Path):
    saved_run = _make_saved_run()
    data_root = tmp_path / 'data'
    data_dir = data_root / 'dataset_a' / 'raw'
    _write_eval_split(data_dir, target_names=('registry_gt',))
    _write_registry_eval_gt(
        data_root / 'dataset_registry.yml',
        dataset_name='dataset_a',
        eval_gt='registry_gt',
    )
    monkeypatch.setattr(data_mod, 'Paths', lambda _settings: FakePaths(data_root))

    source = data_mod.build_eval_source(saved_run, split='test')

    assert source.config.target == 'registry_gt'
    assert source.config.paired is True
    assert source.config.model_norm_prm['data_mean_gt'] == 0
    assert source.config.model_norm_prm['data_std_gt'] == 1
    assert next(iter(source)).y is not None


def test_build_eval_source_replaces_null_gt_norm_with_identity_for_non_lvae(
    monkeypatch,
    tmp_path: Path,
):
    saved_run = _make_saved_run(
        model_norm_prm={
            'data_mean': 1.0,
            'data_std': 2.0,
            'data_mean_gt': None,
            'data_std_gt': None,
        },
    )
    data_root = tmp_path / 'data'
    data_dir = data_root / 'dataset_a' / 'raw'
    _write_eval_split(data_dir, target_names=('registry_gt',))
    _write_registry_eval_gt(
        data_root / 'dataset_registry.yml',
        dataset_name='dataset_a',
        eval_gt='registry_gt',
    )
    monkeypatch.setattr(data_mod, 'Paths', lambda _settings: FakePaths(data_root))

    source = data_mod.build_eval_source(saved_run, split='test')

    assert source.config.model_norm_prm['data_mean_gt'] == 0
    assert source.config.model_norm_prm['data_std_gt'] == 1
    assert next(iter(source)).y is not None


def test_build_eval_source_uses_input_norm_for_unpaired_lvae_injected_eval_gt(
    monkeypatch,
    tmp_path: Path,
):
    saved_run = _make_saved_run(
        model_architecture='lvae',
        model_parameters=LVAEParams(),
        model_norm_prm={
            'data_mean': 0.07035854364676693,
            'data_std': 1.118182888310137,
            'data_mean_gt': None,
            'data_std_gt': None,
        },
    )
    data_root = tmp_path / 'data'
    data_dir = data_root / 'dataset_a' / 'raw'
    _write_eval_split(data_dir, target_names=('registry_gt',))
    _write_registry_eval_gt(
        data_root / 'dataset_registry.yml',
        dataset_name='dataset_a',
        eval_gt='registry_gt',
    )
    monkeypatch.setattr(data_mod, 'Paths', lambda _settings: FakePaths(data_root))

    source = data_mod.build_eval_source(saved_run, split='test')

    assert source.config.paired is True
    assert source.config.model_norm_prm['data_mean_gt'] == saved_run.model_norm_prm[
        'data_mean'
    ]
    assert source.config.model_norm_prm['data_std_gt'] == saved_run.model_norm_prm[
        'data_std'
    ]
    assert next(iter(source)).y is not None


def test_build_eval_source_preserves_explicit_gt_norm_when_eval_gt_is_injected(
    monkeypatch,
    tmp_path: Path,
):
    saved_run = _make_saved_run(
        model_architecture='lvae',
        model_parameters=LVAEParams(),
        model_norm_prm={
            'data_mean': 1.0,
            'data_std': 2.0,
            'data_mean_gt': 10.0,
            'data_std_gt': 5.0,
        },
    )
    data_root = tmp_path / 'data'
    data_dir = data_root / 'dataset_a' / 'raw'
    _write_eval_split(data_dir, target_names=('registry_gt',))
    _write_registry_eval_gt(
        data_root / 'dataset_registry.yml',
        dataset_name='dataset_a',
        eval_gt='registry_gt',
    )
    monkeypatch.setattr(data_mod, 'Paths', lambda _settings: FakePaths(data_root))

    source = data_mod.build_eval_source(saved_run, split='test')

    assert source.config.model_norm_prm['data_mean_gt'] == 10.0
    assert source.config.model_norm_prm['data_std_gt'] == 5.0
    assert next(iter(source)).y is not None


def test_build_eval_source_registry_eval_gt_overrides_training_target(monkeypatch, tmp_path: Path):
    saved_run = _make_saved_run(data_cfg={'paired': True, 'target': 'training_gt'})
    data_root = tmp_path / 'data'
    data_dir = data_root / 'dataset_a' / 'raw'
    _write_eval_split(data_dir, target_names=('registry_gt',))
    _write_registry_eval_gt(
        data_root / 'dataset_registry.yml',
        dataset_name='dataset_a',
        eval_gt='registry_gt',
    )
    monkeypatch.setattr(data_mod, 'Paths', lambda _settings: FakePaths(data_root))

    source = data_mod.build_eval_source(saved_run, split='test')

    assert source.config.target == 'registry_gt'
    assert source.items[0].gt_path == data_dir / 'registry_gt' / 'test' / 'img_a.tif'


def test_build_eval_source_explicit_eval_gt_overrides_registry_and_training_target(monkeypatch, tmp_path: Path):
    saved_run = _make_saved_run(data_cfg={'paired': True, 'target': 'training_gt'})
    data_root = tmp_path / 'data'
    data_dir = data_root / 'dataset_a' / 'raw'
    _write_eval_split(data_dir, target_names=('explicit_gt',))
    _write_registry_eval_gt(
        data_root / 'dataset_registry.yml',
        dataset_name='dataset_a',
        eval_gt='registry_gt',
    )
    monkeypatch.setattr(data_mod, 'Paths', lambda _settings: FakePaths(data_root))

    source = data_mod.build_eval_source(saved_run, split='test', eval_gt='explicit_gt')

    assert source.config.target == 'explicit_gt'
    assert source.items[0].gt_path == data_dir / 'explicit_gt' / 'test' / 'img_a.tif'


def test_build_eval_source_training_eval_gt_token_uses_training_target(monkeypatch, tmp_path: Path):
    saved_run = _make_saved_run(data_cfg={'paired': True, 'target': 'training_gt'})
    data_root = tmp_path / 'data'
    data_dir = data_root / 'dataset_a' / 'raw'
    _write_eval_split(data_dir, target_names=('training_gt',))
    _write_registry_eval_gt(
        data_root / 'dataset_registry.yml',
        dataset_name='dataset_a',
        eval_gt='registry_gt',
    )
    monkeypatch.setattr(data_mod, 'Paths', lambda _settings: FakePaths(data_root))

    source = data_mod.build_eval_source(saved_run, split='test', eval_gt='@training')

    assert source.config.target == 'training_gt'
    assert source.items[0].gt_path == data_dir / 'training_gt' / 'test' / 'img_a.tif'


def test_build_eval_source_none_eval_gt_token_disables_gt(monkeypatch, tmp_path: Path):
    saved_run = _make_saved_run(data_cfg={'paired': True, 'target': 'training_gt'})
    data_root = tmp_path / 'data'
    data_dir = data_root / 'dataset_a' / 'raw'
    _write_eval_split(data_dir)
    _write_registry_eval_gt(
        data_root / 'dataset_registry.yml',
        dataset_name='dataset_a',
        eval_gt='registry_gt',
    )
    monkeypatch.setattr(data_mod, 'Paths', lambda _settings: FakePaths(data_root))

    source = data_mod.build_eval_source(saved_run, split='test', eval_gt='@none')

    assert source.config.paired is False
    assert source.config.target is None
    assert source.items[0].gt_path is None
    assert next(iter(source)).y is None



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
        data_overrides={'data_dir': str(data_dir)},
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
        data_overrides={'data_dir': str(data_dir)},
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
        data_overrides={'data_dir': str(data_dir)},
    )

    assert len(source.items) == 1
    item = source.items[0]
    assert item.name == 'stack_a'
    assert item.data_format == 'timelapse'
    assert item.sample_count == 3
    assert item.time_indices == (1, 2, 3)
    assert item.source_axis == 'time'
    assert item.source_indices == (1, 2, 3)
    assert item.input_id == 'inp/test/stack_a.tif'

    indexed_samples = list(item.iter_samples(source.config))
    assert [sample_index for sample_index, _ in indexed_samples] == [0, 1, 2]
    assert [item.sample_time_index(sample_index) for sample_index, _ in indexed_samples] == [1, 2, 3]
    assert [item.sample_name(sample_index) for sample_index, _ in indexed_samples] == [
        'stack_a_1',
        'stack_a_2',
        'stack_a_3',
    ]
    assert [tuple(sample.x.shape) for _, sample in indexed_samples] == [(3, 4, 5)] * 3


def test_eval_source_marks_shuffled_timelapse_provenance_unknown(monkeypatch, tmp_path: Path):
    saved_run = _make_saved_run(
        data_cfg={
            'data_format': 'timelapse',
            'timelapse_prm': {
                'context_length': 3,
                'timelapse_max_frames': 5,
                'shuffle': True,
            },
        }
    )
    data_dir = tmp_path / 'dataset'
    inp_dir = data_dir / 'inp' / 'test'
    inp_dir.mkdir(parents=True)
    imwrite(inp_dir / 'stack_a.tif', np.ones((8, 4, 5), dtype=np.float32))

    monkeypatch.setattr(
        data_mod,
        'load_dataset_info',
        lambda path, dataset_name: {'data_format': 'timelapse'},
    )

    source = data_mod.build_eval_source(
        saved_run,
        split='test',
        data_overrides={'data_dir': str(data_dir)},
    )

    item = source.items[0]
    assert item.sample_count == 3
    assert item.source_axis == 'time'
    assert item.source_indices == 'unknown'
    assert item.time_indices == (None, None, None)


def test_eval_source_keeps_time_provenance_when_shuffle_does_not_subsample(monkeypatch, tmp_path: Path):
    saved_run = _make_saved_run(
        data_cfg={
            'data_format': 'timelapse',
            'timelapse_prm': {
                'context_length': 3,
                'timelapse_max_frames': 10,
                'shuffle': True,
            },
        }
    )
    data_dir = tmp_path / 'dataset'
    inp_dir = data_dir / 'inp' / 'test'
    inp_dir.mkdir(parents=True)
    imwrite(inp_dir / 'stack_a.tif', np.ones((5, 4, 5), dtype=np.float32))

    monkeypatch.setattr(
        data_mod,
        'load_dataset_info',
        lambda path, dataset_name: {'data_format': 'timelapse'},
    )

    source = data_mod.build_eval_source(
        saved_run,
        split='test',
        data_overrides={'data_dir': str(data_dir)},
    )

    item = source.items[0]
    assert item.source_axis == 'time'
    assert item.source_indices == (1, 2, 3)
    assert item.time_indices == (1, 2, 3)


def test_eval_source_keeps_selected_snr_indices(monkeypatch, tmp_path: Path):
    saved_run = _make_saved_run(
        data_cfg={
            'data_format': 'mltpl_snr',
            'mltpl_snr_prm': {'snr_idx': [4, 1, 3]},
        }
    )
    data_dir = tmp_path / 'dataset'
    inp_dir = data_dir / 'inp' / 'test'
    inp_dir.mkdir(parents=True)
    imwrite(inp_dir / 'stack_a.tif', np.ones((5, 4, 5), dtype=np.float32))

    monkeypatch.setattr(
        data_mod,
        'load_dataset_info',
        lambda path, dataset_name: {'data_format': 'mltpl_snr'},
    )

    source = data_mod.build_eval_source(
        saved_run,
        split='test',
        data_overrides={'data_dir': str(data_dir)},
    )

    item = source.items[0]
    assert item.sample_count == 3
    assert item.source_axis == 'snr'
    assert item.source_indices == (4, 1, 3)
    assert item.time_indices == (None, None, None)


def test_eval_source_resolves_last_snr_index(monkeypatch, tmp_path: Path):
    saved_run = _make_saved_run(
        data_cfg={
            'data_format': 'mltpl_snr',
            'mltpl_snr_prm': {'snr_idx': 'last'},
        }
    )
    data_dir = tmp_path / 'dataset'
    inp_dir = data_dir / 'inp' / 'test'
    inp_dir.mkdir(parents=True)
    imwrite(inp_dir / 'stack_a.tif', np.ones((5, 4, 5), dtype=np.float32))

    monkeypatch.setattr(
        data_mod,
        'load_dataset_info',
        lambda path, dataset_name: {'data_format': 'mltpl_snr'},
    )

    source = data_mod.build_eval_source(
        saved_run,
        split='test',
        data_overrides={'data_dir': str(data_dir)},
    )

    item = source.items[0]
    assert item.source_axis == 'snr'
    assert item.source_indices == (4,)


def test_eval_source_marks_random_snr_provenance_unknown(monkeypatch, tmp_path: Path):
    saved_run = _make_saved_run(
        data_cfg={
            'data_format': 'mltpl_snr',
            'mltpl_snr_prm': {'snr_idx': 'random'},
        }
    )
    data_dir = tmp_path / 'dataset'
    inp_dir = data_dir / 'inp' / 'test'
    inp_dir.mkdir(parents=True)
    imwrite(inp_dir / 'stack_a.tif', np.ones((5, 4, 5), dtype=np.float32))

    monkeypatch.setattr(
        data_mod,
        'load_dataset_info',
        lambda path, dataset_name: {'data_format': 'mltpl_snr'},
    )

    source = data_mod.build_eval_source(
        saved_run,
        split='test',
        data_overrides={'data_dir': str(data_dir)},
    )

    item = source.items[0]
    assert item.source_axis == 'snr'
    assert item.source_indices == 'unknown'


def _write_evaluation_registry_entry(
    registry_path: Path,
    *,
    dataset_name: str,
    eval_gt: str | None = 'gt',
) -> None:
    save_yaml(
        {
            dataset_name: {
                'data_format': 'single',
                'usage': 'evaluation',
                'for_training': False,
                'defaults': {
                    'recon': {
                        'input': 'inp',
                        'target': None,
                        'eval_gt': eval_gt,
                    }
                },
                'outputs': {
                    'recon': [
                        {'key': 'inp', 'path': 'inp', 'role': 'inp', 'axes': 'YX'},
                        {'key': 'gt', 'path': 'gt', 'role': 'gt', 'axes': 'YX'},
                    ]
                },
                'split': {'recon': {'enabled': False}},
            }
        },
        registry_path,
    )


def test_resolve_and_build_independent_evaluation_dataset_uses_all_root_files(
    monkeypatch, tmp_path: Path
):
    saved_run = _make_saved_run(
        data_cfg={'paired': True, 'target': 'training_gt'},
        split_manifest={
            'version': 1,
            'splits': {'test': [{'input': 'must_not_be_used.tif', 'target': None}]},
        },
    )
    data_root = tmp_path / 'data'
    eval_dir = data_root / 'eval_dataset' / 'preprocess' / 'recon'
    (eval_dir / 'inp').mkdir(parents=True)
    (eval_dir / 'gt').mkdir(parents=True)
    imwrite(eval_dir / 'inp' / 'img_a.tif', np.ones((4, 5), dtype=np.float32) * 3)
    imwrite(eval_dir / 'gt' / 'img_a.tif', np.ones((4, 5), dtype=np.float32))
    _write_evaluation_registry_entry(
        data_root / 'dataset_registry.yml',
        dataset_name='eval_dataset',
    )
    monkeypatch.setattr(data_mod, 'Paths', lambda _settings: FakePaths(data_root))

    eval_dataset = data_mod.resolve_evaluation_dataset(saved_run, 'eval_dataset')
    source = data_mod.build_eval_source(
        saved_run,
        split='test',
        evaluation_dataset=eval_dataset,
    )

    assert eval_dataset.data_type == 'recon'
    assert eval_dataset.data_dir == eval_dir
    assert source.use_split is False
    assert source.split_manifest is None
    assert source.config.dataset_name == 'eval_dataset'
    assert source.config.data_dir == eval_dir
    assert source.config.input == 'inp'
    assert source.config.target == 'gt'
    assert len(source.items) == 1
    assert source.items[0].split is None
    assert source.items[0].inp_path == eval_dir / 'inp' / 'img_a.tif'
    assert source.items[0].gt_path == eval_dir / 'gt' / 'img_a.tif'


def test_independent_evaluation_dataset_without_eval_gt_does_not_fall_back_to_training_target(
    monkeypatch, tmp_path: Path
):
    saved_run = _make_saved_run(data_cfg={'paired': True, 'target': 'training_gt'})
    data_root = tmp_path / 'data'
    eval_dir = data_root / 'eval_dataset' / 'preprocess' / 'recon'
    (eval_dir / 'inp').mkdir(parents=True)
    imwrite(eval_dir / 'inp' / 'img_a.tif', np.ones((4, 5), dtype=np.float32))
    _write_evaluation_registry_entry(
        data_root / 'dataset_registry.yml',
        dataset_name='eval_dataset',
        eval_gt=None,
    )
    monkeypatch.setattr(data_mod, 'Paths', lambda _settings: FakePaths(data_root))

    eval_dataset = data_mod.resolve_evaluation_dataset(saved_run, 'eval_dataset')
    source = data_mod.build_eval_source(saved_run, evaluation_dataset=eval_dataset)

    assert source.config.paired is False
    assert source.config.target is None
    assert source.items[0].gt_path is None


def test_resolve_evaluation_dataset_rejects_training_dataset(monkeypatch, tmp_path: Path):
    saved_run = _make_saved_run()
    data_root = tmp_path / 'data'
    data_root.mkdir(parents=True)
    save_yaml(
        {
            'training_dataset': {
                'data_format': 'single',
                'usage': 'training',
                'for_training': True,
                'defaults': {'recon': {'input': 'inp', 'target': None, 'eval_gt': None}},
            }
        },
        data_root / 'dataset_registry.yml',
    )
    monkeypatch.setattr(data_mod, 'Paths', lambda _settings: FakePaths(data_root))

    try:
        data_mod.resolve_evaluation_dataset(saved_run, 'training_dataset')
    except ValueError as exc:
        assert 'usage: evaluation' in str(exc)
    else:
        raise AssertionError('Expected training dataset to be rejected by --on resolution.')

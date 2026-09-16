from __future__ import annotations

import importlib
from pathlib import Path
from types import SimpleNamespace

from lisai.config.models.inference import EvaluateDefaults
from lisai.evaluation.data import EvaluationDatasetSpec
from lisai.evaluation.run_evaluate import (
    _build_evaluation_folder_name,
    _evaluation_dataset_folder,
    _evaluation_metadata,
    _expand_checkpoint_selection,
    _format_eval_gt_for_display,
    _format_tiling_size_for_display,
    _resolved_data_overrides,
)

run_evaluate_mod = importlib.import_module("lisai.evaluation.run_evaluate")


class EmptySampleSource:
    config = SimpleNamespace(target=None)
    items = ()

    def __len__(self):
        return 0

    def iter_items(self):
        return iter(())


def test_build_evaluation_folder_name_uses_requested_epoch_when_explicit():
    assert _build_evaluation_folder_name(
        best_or_last='best',
        requested_epoch=12,
        resolved_epoch=7,
    ) == 'epoch_12'


def test_build_evaluation_folder_name_includes_selector_and_resolved_epoch():
    assert _build_evaluation_folder_name(
        best_or_last='last',
        requested_epoch=None,
        resolved_epoch=100,
    ) == 'last_epoch_100'


def test_build_evaluation_folder_name_falls_back_when_epoch_is_unknown():
    assert _build_evaluation_folder_name(
        best_or_last='best',
        requested_epoch=None,
        resolved_epoch=None,
    ) == 'best'


def test_evaluation_dataset_folder_distinguishes_training_split_and_independent_dataset():
    eval_dataset = EvaluationDatasetSpec(
        name='Gag/Independent',
        data_type='recon',
        data_dir=Path('/tmp/eval'),
        dataset_info={},
        input='inp',
        eval_gt='gt',
        data_format='single',
    )

    assert _evaluation_dataset_folder(evaluation_dataset=None, split='test') == 'training_test'
    assert _evaluation_dataset_folder(evaluation_dataset=eval_dataset, split='test') == 'Gag__Independent'


def test_expand_checkpoint_selection_for_both_creates_two_typed_configs():
    cfg = EvaluateDefaults.model_validate(
        {
            "checkpoint": {"best_or_last": "both"},
            "saving": {"save_folder": "/tmp/eval"},
        }
    )

    expanded = _expand_checkpoint_selection(cfg)

    assert [item.checkpoint.best_or_last for item in expanded] == ["best", "last"]
    assert expanded[0].saving.save_folder == str(Path("/tmp/eval/best"))
    assert expanded[1].saving.save_folder == str(Path("/tmp/eval/last"))
    assert all(isinstance(item, EvaluateDefaults) for item in expanded)
    assert cfg.checkpoint.best_or_last == "both"
    assert cfg.saving.save_folder == "/tmp/eval"


def test_format_eval_gt_for_display_handles_special_values():
    assert _format_eval_gt_for_display(None) == "<none>"
    assert _format_eval_gt_for_display("") == "<root>"
    assert _format_eval_gt_for_display("gt_avg") == "gt_avg"


def test_run_evaluate_default_tiling_policy_reaches_single_evaluation(monkeypatch, tmp_path: Path):
    captured = {}
    saved_run = object()

    monkeypatch.setattr(run_evaluate_mod, "resolve_run_dir", lambda **_: tmp_path)
    monkeypatch.setattr(run_evaluate_mod, "load_saved_run", lambda _run_dir: saved_run)
    monkeypatch.setattr(
        run_evaluate_mod,
        "_run_single_evaluation",
        lambda **kwargs: captured.update(kwargs),
    )

    run_evaluate_mod.run_evaluate(
        EvaluateDefaults(),
        dataset_name="dataset_a",
        model_name="model_a",
    )

    assert captured["cfg"].inference.tiling_size == "auto"


def test_format_tiling_size_for_display_reports_policy_and_effective_size():
    assert _format_tiling_size_for_display("auto", 300) == "300 (auto)"
    assert _format_tiling_size_for_display(None, 512) == "512 (auto)"
    assert _format_tiling_size_for_display(2000, 2000) == "2000"
    assert _format_tiling_size_for_display("off", None) == "off"


def test_run_single_evaluation_reports_effective_auto_tiling_size(monkeypatch, tmp_path: Path, capsys):
    captured = {}
    saved_run = SimpleNamespace(is_lvae=False, upsampling_factor=1)

    def fake_initialize_runtime(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            model=object(),
            device="cpu",
            tiling_size=300,
            resolved_epoch=93,
        )

    cfg = EvaluateDefaults.model_validate(
        {
            "inference": {"tiling_size": "auto", "lvae_num_samples": 20},
            "saving": {"save_folder": str(tmp_path / "eval")},
        }
    )

    monkeypatch.setattr(run_evaluate_mod, "initialize_runtime", fake_initialize_runtime)
    monkeypatch.setattr(run_evaluate_mod, "build_eval_source", lambda *_args, **_kwargs: EmptySampleSource())
    monkeypatch.setattr(run_evaluate_mod, "_evaluation_metadata", lambda **_kwargs: {})

    run_evaluate_mod._run_single_evaluation(
        output_root=tmp_path,
        saved_run=saved_run,
        cfg=cfg,
        progress_bar=True,
    )

    assert captured["tiling_size"] == "auto"
    assert "Tiling size: 300 (auto)" in capsys.readouterr().out


def test_run_evaluate_resolves_independent_dataset_and_separates_output_root(monkeypatch, tmp_path: Path):
    captured = {}
    saved_run = SimpleNamespace()
    eval_dataset = EvaluationDatasetSpec(
        name='eval_dataset',
        data_type='recon',
        data_dir=tmp_path / 'data',
        dataset_info={},
        input='inp',
        eval_gt='gt',
        data_format='single',
    )

    monkeypatch.setattr(run_evaluate_mod, 'resolve_run_dir', lambda **_: tmp_path)
    monkeypatch.setattr(run_evaluate_mod, 'load_saved_run', lambda _run_dir: saved_run)
    monkeypatch.setattr(
        run_evaluate_mod,
        'resolve_evaluation_dataset',
        lambda *_args, **_kwargs: eval_dataset,
    )
    monkeypatch.setattr(
        run_evaluate_mod,
        '_run_single_evaluation',
        lambda **kwargs: captured.update(kwargs),
    )

    run_evaluate_mod.run_evaluate(
        EvaluateDefaults(),
        dataset_name='training_dataset',
        model_name='model_a',
        evaluation_dataset_name='eval_dataset',
    )

    assert captured['output_root'] == tmp_path / 'evaluations'
    assert captured['saved_run'] is saved_run
    assert captured['evaluation_dataset'] is eval_dataset


def test_resolved_data_overrides_adds_timelapse_limit_without_mutating_config():
    cfg = EvaluateDefaults.model_validate(
        {
            "data": {
                "overrides": {"subfolder": "recon"},
                "timelapse_max": 7,
            }
        }
    )

    resolved = _resolved_data_overrides(cfg)

    assert resolved == {
        "subfolder": "recon",
        "timelapse_prm": {"timelapse_max_frames": 7},
    }
    assert cfg.data.overrides == {"subfolder": "recon"}


def test_evaluation_metadata_records_independent_dataset_and_checkpoint(tmp_path: Path):
    saved_run = SimpleNamespace(
        run_dir=tmp_path / 'run',
        experiment_name='model_a_00',
        dataset_name='training_dataset',
    )
    runtime = SimpleNamespace(
        resolved_epoch=42,
        load_method='state_dict',
        checkpoint_path=tmp_path / 'run' / 'model_epoch_42.pth',
        tiling_size=300,
    )
    sample_source = SimpleNamespace(
        config=SimpleNamespace(
            data_dir=tmp_path / 'eval_dataset' / 'preprocess' / 'recon',
            input='inp',
            target='gt',
            resolved_data_format='single',
        )
    )
    eval_dataset = EvaluationDatasetSpec(
        name='eval_dataset',
        data_type='recon',
        data_dir=sample_source.config.data_dir,
        dataset_info={},
        input='inp',
        eval_gt='gt',
        data_format='single',
    )
    cfg = EvaluateDefaults.model_validate(
        {
            'metrics': ['psnr', 'ssim'],
            'inference': {'tiling_size': 'auto'},
        }
    )

    metadata = _evaluation_metadata(
        saved_run=saved_run,
        runtime=runtime,
        sample_source=sample_source,
        cfg=cfg,
        evaluation_dataset=eval_dataset,
    )

    assert metadata['model']['source'] == 'training_run'
    assert metadata['model']['checkpoint']['resolved_epoch'] == 42
    assert metadata['dataset']['name'] == 'eval_dataset'
    assert metadata['dataset']['usage'] == 'evaluation'
    assert metadata['dataset']['split'] is None
    assert metadata['dataset']['eval_gt'] == 'gt'
    assert metadata['evaluation']['metrics'] == ['psnr', 'ssim']
    assert metadata['evaluation']['tiling_size'] == {'requested': 'auto', 'effective': 300}

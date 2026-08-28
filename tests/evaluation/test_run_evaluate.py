from __future__ import annotations

import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from lisai.evaluation.data import EvaluationDatasetSpec
from lisai.evaluation.run_evaluate import (
    _build_evaluation_folder_name,
    _expand_checkpoint_selection,
    _evaluation_dataset_folder,
    _evaluation_metadata,
    _format_eval_gt_for_display,
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


def test_expand_checkpoint_selection_for_both_creates_two_runs():
    options = {
        "best_or_last": "both",
        "epoch_number": None,
        "save_folder": Path("/tmp/eval"),
        "results": {"seed": 1},
    }

    expanded = _expand_checkpoint_selection(options)

    assert [item["best_or_last"] for item in expanded] == ["best", "last"]
    assert expanded[0]["save_folder"] == Path("/tmp/eval/best")
    assert expanded[1]["save_folder"] == Path("/tmp/eval/last")
    assert expanded[0]["results"] == {"seed": 1}
    assert expanded[1]["results"] == {"seed": 1}
    assert expanded[0]["results"] is not options["results"]
    assert expanded[1]["results"] is not options["results"]


def test_format_eval_gt_for_display_handles_special_values():
    assert _format_eval_gt_for_display(None) == "<none>"
    assert _format_eval_gt_for_display("") == "<root>"
    assert _format_eval_gt_for_display("gt_avg") == "gt_avg"


def test_run_evaluate_default_tiling_size_reaches_single_evaluation(monkeypatch, tmp_path: Path):
    captured = {}
    saved_run = object()

    monkeypatch.setattr(run_evaluate_mod, "resolve_run_dir", lambda **_: tmp_path)
    monkeypatch.setattr(run_evaluate_mod, "load_saved_run", lambda _run_dir: saved_run)
    monkeypatch.setattr(
        run_evaluate_mod,
        "_run_single_evaluation",
        lambda **kwargs: captured.update(kwargs["options"]),
    )

    run_evaluate_mod.run_evaluate(dataset_name="dataset_a", model_name="model_a")

    assert captured["tiling_size"] == 2000


def test_run_single_evaluation_reports_runtime_tiling_size(monkeypatch, tmp_path: Path, capsys):
    captured = {}
    saved_run = SimpleNamespace(is_lvae=False, upsampling_factor=1)

    def fake_initialize_runtime(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            model=object(),
            device="cpu",
            tiling_size=2000,
            resolved_epoch=93,
        )

    options = {
        "best_or_last": "best",
        "epoch_number": None,
        "tiling_size": 2000,
        "save_folder": tmp_path / "eval",
        "overwrite": False,
        "split": "test",
        "lvae_num_samples": 20,
        "crop_size": None,
        "eval_gt": None,
        "data_prm_update": None,
        "results": None,
        "ch_out": 1,
        "metrics_list": None,
        "limit_n_imgs": None,
    }

    monkeypatch.setattr(run_evaluate_mod, "initialize_runtime", fake_initialize_runtime)
    monkeypatch.setattr(run_evaluate_mod, "build_eval_source", lambda *_args, **_kwargs: EmptySampleSource())
    monkeypatch.setattr(run_evaluate_mod, "_evaluation_metadata", lambda **_kwargs: {})

    run_evaluate_mod._run_single_evaluation(output_root=tmp_path, saved_run=saved_run, options=options)

    assert captured["tiling_size"] == 2000
    assert "Tiling size: 2000" in capsys.readouterr().out


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
        dataset_name='training_dataset',
        model_name='model_a',
        evaluation_dataset_name='eval_dataset',
    )

    assert captured['output_root'] == tmp_path / 'evaluations'
    assert captured['saved_run'] is saved_run
    assert captured['evaluation_dataset'] is eval_dataset


def test_run_evaluate_rejects_split_with_independent_dataset():
    with pytest.raises(ValueError, match='cannot be combined'):
        run_evaluate_mod.run_evaluate(
            dataset_name='training_dataset',
            model_name='model_a',
            evaluation_dataset_name='eval_dataset',
            split='val',
        )


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
    options = {
        'best_or_last': 'best',
        'epoch_number': None,
        'split': 'test',
        'metrics_list': ['psnr', 'ssim'],
    }

    metadata = _evaluation_metadata(
        saved_run=saved_run,
        runtime=runtime,
        sample_source=sample_source,
        options=options,
        evaluation_dataset=eval_dataset,
    )

    assert metadata['model']['source'] == 'training_run'
    assert metadata['model']['checkpoint']['resolved_epoch'] == 42
    assert metadata['dataset']['name'] == 'eval_dataset'
    assert metadata['dataset']['usage'] == 'evaluation'
    assert metadata['dataset']['split'] is None
    assert metadata['dataset']['eval_gt'] == 'gt'
    assert metadata['evaluation']['metrics'] == ['psnr', 'ssim']

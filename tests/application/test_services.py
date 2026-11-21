"""First extraction of non-Qt application services from ImProcess."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from lisai.application import (
    create_workspace,
    find_new_completed_run,
    get_installed_model_names,
    list_hdn_noise_models,
    list_reusable_training_datasets,
    list_user_training_configs,
    list_usable_training_runs,
    managed_training_config_path,
    plan_for_saved_training_config,
    quick_hdn_training_command,
    save_training_yaml_text,
    validate_training_yaml_text,
    validate_workspace,
)
from lisai.data.import_training import import_training_dataset
from lisai.infra.paths import Paths
from lisai.training.quick_hdn import prepare_quick_hdn_training


def test_workspace_validation_and_creation_are_explicit(tmp_path):
    root = tmp_path / 'lisai'
    with pytest.raises(ValueError, match='does not exist'):
        validate_workspace(root)
    settings = create_workspace(root)
    assert settings.WORKSPACE_ROOT == root.resolve()
    assert validate_workspace(root).WORKSPACE_ROOT == root.resolve()
    with pytest.raises(ValueError, match='not empty'):
        create_workspace(tmp_path)  # unrelated nonempty parent
    # No overwrite on an already-existing workspace.
    assert create_workspace(root).WORKSPACE_ROOT == root.resolve()


def test_services_import_without_qt():
    # A separate interpreter makes this test independent of other test imports.
    script = ('import sys; import lisai.application; '
              'assert "qtpy" not in sys.modules; '
              'assert "imswitch" not in sys.modules')
    result = subprocess.run(
        [sys.executable, '-c', script], capture_output=True, text=True,
        env={**os.environ, 'PYTHONPATH': str(Path(__file__).resolve().parents[2] / 'src')},
    )
    assert result.returncode == 0, result.stderr


@pytest.fixture
def prepared(tmp_path):
    settings = create_workspace(tmp_path / 'root')
    noise_dir = Paths(settings).noise_model_dir(noiseModel_name='calibrated')
    noise_dir.mkdir(parents=True)
    np.savez(noise_dir / 'GMM.npz', min_signal=np.array([0.0]), max_signal=np.array([255.0]),
             trained_weight=np.zeros((3, 2)), min_sigma=np.array([1.0]))
    (noise_dir / 'norm_prm.json').write_text(
        json.dumps({'clip': 0, 'normalize_data': False}), encoding='utf-8')
    import_training_dataset(
        [np.ones((3, 64, 64), dtype=np.uint16), np.ones((3, 64, 64), dtype=np.uint16) * 2],
        settings=settings, dataset_name='mito', data_format='timelapse',
    )
    plan = prepare_quick_hdn_training(
        settings=settings, dataset_name='mito', noise_model_name='calibrated',
        experiment_name='training_a', epochs=2, patch_size=32, write=True,
    )
    return settings, plan


def test_catalogs_and_config_editor(prepared):
    settings, plan = prepared
    assert list_hdn_noise_models(settings) == ['calibrated']
    assert list_reusable_training_datasets(settings) == [
        {'name': 'mito', 'data_format': 'timelapse', 'recording_count': 2},
    ]
    assert any(path == plan.config_path for _name, path in list_user_training_configs(settings))
    assert get_installed_model_names(settings) == []
    assert list_usable_training_runs(settings) == []
    assert find_new_completed_run(settings, dataset_name='mito', experiment_name='training_a',
                                  old_run_ids=set()) is None

    text = plan.config_path.read_text(encoding='utf-8')
    assert validate_training_yaml_text(text, settings).experiment.exp_name == 'training_a'
    saved_path = save_training_yaml_text(text, 'another/experiment.yml', settings=settings)
    assert saved_path.is_file()
    with pytest.raises(FileExistsError):
        save_training_yaml_text(text, saved_path, settings=settings)
    assert save_training_yaml_text(text, saved_path, settings=settings, overwrite=True) == saved_path
    with pytest.raises(ValueError):
        managed_training_config_path(settings, '../outside.yml')
    with pytest.raises(ValueError, match='syntax'):
        validate_training_yaml_text('key: [unterminated', settings)


def test_saved_training_preflight_and_worker_contract(prepared):
    settings, original = prepared
    planned = plan_for_saved_training_config(original.config_path, settings=settings)
    assert planned.dataset_name == 'mito'
    assert planned.experiment_name == 'training_a'
    assert planned.models_subfolder == 'HDN'
    command = quick_hdn_training_command(settings, planned, interpreter='/test/python')
    assert command[:4] == ['/test/python', '-u', '-m', 'lisai_runner.training_entrypoint']
    assert command[command.index('--config') + 1] == str(original.config_path)
    assert command[command.index('--workspace') + 1] == str(settings.WORKSPACE_ROOT)
    assert command[command.index('--config-root') + 1] == str(settings.CONFIGS_ROOT)
    original.config_path.unlink()
    with pytest.raises(FileNotFoundError):
        plan_for_saved_training_config(original.config_path, settings=settings)

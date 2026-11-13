from __future__ import annotations

from pathlib import Path
from string import Formatter

import pytest

from lisai.config import load_yaml
from lisai.config.settings import Settings


DATA_YAML = Path("configs/data_config.yml")
TEMPLATE_VALUES = {
    "sample_id": "sample_name",
    "snr_level": "snr_10",
    "n_timepoints": 5,
    "timepoint": "01",
}


def _build_kwargs(template: str) -> dict[str, object]:
    field_names = {
        field_name
        for _, field_name, _, _ in Formatter().parse(template)
        if field_name
    }
    missing_fields = sorted(field_names - TEMPLATE_VALUES.keys())
    assert not missing_fields, f"Missing sample values for placeholders: {missing_fields}"
    return {field_name: TEMPLATE_VALUES[field_name] for field_name in field_names}


def _format_cases() -> list[tuple[str, str, str]]:
    cfg = load_yaml(DATA_YAML)
    return [
        (fmt, data_type, template)
        for fmt, type_map in cfg["format"].items()
        for data_type, template in type_map.items()
    ]


@pytest.fixture
def settings_obj(settings_factory) -> Settings:
    return settings_factory("single")


@pytest.mark.parametrize(("fmt", "data_type", "template"), _format_cases())
def test_get_data_filename_matches_data_yml(
    settings_obj: Settings, fmt: str, data_type: str, template: str
):
    kwargs = _build_kwargs(template)
    expected = Path(template.format(**kwargs))

    actual = settings_obj.get_data_filename(fmt=fmt, data_type=data_type, **kwargs)

    assert actual == expected


def test_project_run_tracking_timeout_is_configured(settings_obj: Settings):
    assert settings_obj.project_cfg.run_tracking.active_heartbeat_timeout_minutes == 10


def test_project_queue_resource_classes_are_configured(settings_obj: Settings):
    defaults = settings_obj.project_cfg.queue.resource_class_vram_mb
    assert defaults.light == 3000
    assert defaults.medium == 6000
    assert defaults.heavy == 9000
    assert settings_obj.project_cfg.queue.fixed_margin_pct == 0.20
    assert settings_obj.project_cfg.queue.paused is False
    assert settings_obj.project_cfg.queue.max_concurrent_runs_per_gpu == 1


def test_project_recovery_defaults_are_configured(settings_obj: Settings):
    safe_resume = settings_obj.project_cfg.recovery.hdn_safe_resume
    assert safe_resume.enabled is True
    assert safe_resume.auto_use_safe_checkpoint_on_continue is True
    assert safe_resume.drop_optimizer_scheduler_state_on_safe_resume is True
    assert safe_resume.rewind_steps == 2
    assert safe_resume.lr_scale == pytest.approx(0.1)
    assert safe_resume.min_lr == pytest.approx(1.0e-8)
    assert safe_resume.max_compound_steps is None
    assert safe_resume.force_grad_clip_max_norm is None


def test_explicit_settings_are_read_only(monkeypatch, settings_factory):
    def forbid_input(*_args, **_kwargs):
        raise AssertionError("Settings must not ask for input")

    monkeypatch.setattr("builtins.input", forbid_input)
    stg = settings_factory("read_only")
    before = {
        path.relative_to(stg.CONFIGS_ROOT): path.read_bytes()
        for path in stg.CONFIGS_ROOT.rglob("*")
        if path.is_file()
    }

    reloaded = Settings.from_config_dir(stg.CONFIGS_ROOT)
    after = {
        path.relative_to(stg.CONFIGS_ROOT): path.read_bytes()
        for path in stg.CONFIGS_ROOT.rglob("*")
        if path.is_file()
    }
    assert after == before
    assert reloaded.PROJECT_ROOT == stg.CONFIGS_ROOT.parent
    assert not (stg.CONFIGS_ROOT / "inference" / "local").exists()
    assert not (stg.CONFIGS_ROOT / "local_config.yml").read_text().startswith(
        "# yaml-language-server:"
    )


def test_settings_requires_explicit_config_dir(tmp_path):
    with pytest.raises(TypeError):
        Settings()
    with pytest.raises(FileNotFoundError, match="local_config.yml"):
        Settings.from_config_dir(tmp_path / "absent")


def test_two_independent_settings_trees(settings_factory):
    from lisai.infra.paths import Paths

    a = settings_factory("a")
    b = settings_factory("b")
    paths_a, paths_b = Paths(a), Paths(b)

    assert a._ctx.paths.roots.data_dir != b._ctx.paths.roots.data_dir
    assert paths_a.dataset_registry_path() != paths_b.dataset_registry_path()
    assert paths_a.promoted_model_registry_path() != paths_b.promoted_model_registry_path()
    assert paths_a.inference_output_dir(source_name="demo", model_name="x") != (
        paths_b.inference_output_dir(source_name="demo", model_name="x")
    )
    assert paths_a.run_dir(dataset_name="d", models_subfolder="m", exp_name="e") != (
        paths_b.run_dir(dataset_name="d", models_subfolder="m", exp_name="e")
    )
    assert Paths(a).dataset_registry_path() == paths_a.dataset_registry_path()


def test_explicit_settings_preserve_default_legacy_results():
    from lisai.config.settings import load_default_settings, settings as legacy
    from lisai.config.io.resolver import resolve_config_dict
    from lisai.infra.paths import Paths

    explicit = Settings.from_config_dir(legacy.CONFIGS_ROOT)
    reloaded = load_default_settings()
    assert reloaded is not legacy and reloaded is not explicit
    assert explicit._ctx == legacy._ctx

    original, new = Paths(legacy), Paths(explicit)
    assert original.dataset_registry_path() == new.dataset_registry_path()
    assert original.dataset_dir(dataset_name="d", data_subfolder="raw") == new.dataset_dir(
        dataset_name="d", data_subfolder="raw"
    )
    assert original.run_dir(dataset_name="d", models_subfolder="m", exp_name="x") == (
        new.run_dir(dataset_name="d", models_subfolder="m", exp_name="x")
    )
    assert original.tensorboard_dir(dataset_name="d") == new.tensorboard_dir(dataset_name="d")
    assert original.inference_output_dir(source_name="d", model_name="m") == (
        new.inference_output_dir(source_name="d", model_name="m")
    )
    assert original.noise_model_path(noiseModel_name="noise") == (
        new.noise_model_path(noiseModel_name="noise")
    )
    raw = {
        "experiment": {"mode": "train", "exp_name": "smoke"},
        "data": {"dataset_name": "demo", "prep_before": False},
        "model": {"architecture": "unet", "parameters": {}},
    }
    assert resolve_config_dict(raw, stg=legacy) == resolve_config_dict(raw, stg=explicit)


def test_file_config_resolution_uses_selected_context(settings_factory, monkeypatch, tmp_path):
    from lisai.config.io import resolver as resolver_mod
    from lisai.config.io.resolver import resolve_config
    from lisai.config.io.yaml import save_yaml
    from lisai.infra.paths import Paths

    a = settings_factory("resolver_a")
    b = settings_factory("resolver_b")
    calls = []

    def capture_registry(registry_path):
        calls.append(registry_path)
        return {}

    monkeypatch.setattr(resolver_mod, "load_dataset_registry", capture_registry)
    yaml_path = tmp_path / "training.yml"
    save_yaml(
        {
            "experiment": {"mode": "train", "exp_name": "smoke"},
            "data": {"dataset_name": "demo", "prep_before": False},
            "model": {"architecture": "unet", "parameters": {}},
        },
        yaml_path,
    )
    for stg in (a, b, a):
        assert resolve_config(yaml_path, stg=stg).experiment.exp_name == "smoke"
    assert calls == [
        Paths(a).dataset_registry_path(),
        Paths(b).dataset_registry_path(),
        Paths(a).dataset_registry_path(),
    ]

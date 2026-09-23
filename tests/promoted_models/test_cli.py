from __future__ import annotations

import io
from pathlib import Path

from lisai.promoted_models.cli import build_parser as build_models_parser
from lisai.runs.cli import build_parser as build_runs_parser


class InteractiveInput(io.StringIO):
    def isatty(self) -> bool:
        return True


def test_runs_promote_cli_defaults():
    parser = build_runs_parser()
    args = parser.parse_args(["promote", "run_a", "--name", "hdn-vimentin"])

    assert args.run == "run_a"
    assert args.name == "hdn-vimentin"
    assert args.checkpoint == "best"
    assert args.inference_config is None
    assert args.overwrite is False


def test_runs_promote_cli_supports_checkpoint_and_overwrite():
    parser = build_runs_parser()
    args = parser.parse_args(
        ["promote", "run_a", "--name", "hdn-vimentin", "--checkpoint", "last", "--overwrite"]
    )

    assert args.checkpoint == "last"
    assert args.overwrite is True



def test_runs_promote_cli_supports_inference_config():
    parser = build_runs_parser()
    args = parser.parse_args(
        ["promote", "run_a", "--name", "hdn-vimentin", "--inference-config", "hdn_sup"]
    )

    assert args.inference_config == "hdn_sup"

def test_models_cli_contains_library_and_export_commands():
    parser = build_models_parser()
    assert parser.parse_args(["list"]).models_command == "list"
    assert parser.parse_args(["show", "hdn-vimentin"]).name == "hdn-vimentin"
    installed = parser.parse_args(["install", "hdn-vimentin.lisai.zip"])
    assert installed.models_command == "install"
    assert installed.archive == "hdn-vimentin.lisai.zip"
    assert installed.overwrite is False
    removed = parser.parse_args(["remove", "hdn-vimentin"])
    assert removed.models_command == "remove"
    assert removed.name == "hdn-vimentin"
    assert removed.yes is False
    assert parser.parse_args(["remove", "hdn-vimentin", "--yes"]).yes is True
    exported = parser.parse_args(["export", "hdn-vimentin"])
    assert exported.models_command == "export"
    assert exported.name == "hdn-vimentin"
    assert exported.output is None

    set_config = parser.parse_args(["set-config", "hdn-vimentin", "hdn_sup"])
    assert set_config.models_command == "set-config"
    assert set_config.config == "hdn_sup"
    assert set_config.clear is False
    assert parser.parse_args(["set-config", "hdn-vimentin", "--clear"]).clear is True

    catalog = parser.parse_args(["catalog"])
    assert catalog.models_command == "catalog"

    downloaded = parser.parse_args(["download", "hdn-vimentin", "--install"])
    assert downloaded.models_command == "download"
    assert downloaded.name == "hdn-vimentin"
    assert downloaded.install is True
    assert downloaded.overwrite is False


def test_top_level_cli_registers_runs_promote_and_models_export():
    from lisai.cli import build_parser as build_top_level_parser

    promoted = build_top_level_parser().parse_args(
        ["runs", "promote", "run_a", "--name", "hdn-vimentin"]
    )
    assert promoted.command == "runs"
    assert promoted.runs_command == "promote"

    installed = build_top_level_parser().parse_args(
        ["models", "install", "hdn-vimentin.lisai.zip"]
    )
    assert installed.command == "models"
    assert installed.models_command == "install"

    removed = build_top_level_parser().parse_args(["models", "remove", "hdn-vimentin", "--yes"])
    assert removed.command == "models"
    assert removed.models_command == "remove"
    assert removed.yes is True

    exported = build_top_level_parser().parse_args(["models", "export", "hdn-vimentin"])
    assert exported.command == "models"
    assert exported.models_command == "export"


def test_models_remove_cli_requires_confirmation(monkeypatch, capsys):
    from datetime import datetime, timezone
    from types import SimpleNamespace
    import lisai.promoted_models.cli as models_cli
    from lisai.promoted_models.schema import PromotedModelRegistry, PromotedModelRegistryEntry

    registry = PromotedModelRegistry(
        models={
            "demo-model": PromotedModelRegistryEntry(
                source_run_id="01ARZ3NDEKTSV4RRFFQ69G5FAV",
                path="demo-model",
                created_at=datetime(2026, 8, 27, tzinfo=timezone.utc),
            )
        }
    )
    monkeypatch.setattr(models_cli, "load_promoted_model_registry", lambda: registry)
    called = []
    monkeypatch.setattr(models_cli, "remove_promoted_model", lambda name: called.append(name))
    monkeypatch.setattr("builtins.input", lambda prompt: "n")

    result = models_cli.main(["remove", "demo-model"])

    assert result == 0
    assert called == []
    assert "Removal cancelled." in capsys.readouterr().out


def test_models_remove_cli_yes_skips_confirmation(monkeypatch, capsys):
    from datetime import datetime, timezone
    from types import SimpleNamespace
    import lisai.promoted_models.cli as models_cli
    from lisai.promoted_models.schema import PromotedModelRegistry, PromotedModelRegistryEntry

    registry = PromotedModelRegistry(
        models={
            "demo-model": PromotedModelRegistryEntry(
                source_run_id="01ARZ3NDEKTSV4RRFFQ69G5FAV",
                path="demo-model",
                created_at=datetime(2026, 8, 27, tzinfo=timezone.utc),
            )
        }
    )
    monkeypatch.setattr(models_cli, "load_promoted_model_registry", lambda: registry)
    monkeypatch.setattr(
        "builtins.input",
        lambda prompt: (_ for _ in ()).throw(AssertionError("input should not be called")),
    )
    removed = SimpleNamespace(name="demo-model", model_dir="/tmp/demo-model")
    monkeypatch.setattr(models_cli, "remove_promoted_model", lambda name: removed)

    result = models_cli.main(["remove", "demo-model", "--yes"])

    assert result == 0
    assert "Removed model: demo-model" in capsys.readouterr().out


def test_models_sync_cli_confirms_unique_partial_name(monkeypatch, capsys):
    from types import SimpleNamespace
    import lisai.promoted_models.cli as models_cli

    registry = SimpleNamespace(models={"demo-model": object(), "other-model": object()})
    monkeypatch.setattr(models_cli, "load_promoted_model_registry", lambda: registry)
    monkeypatch.setattr("sys.stdin", InteractiveInput("y\n"))
    called = []
    promoted = SimpleNamespace(
        manifest=SimpleNamespace(
            name="demo-model",
            model=SimpleNamespace(task="denoising"),
        )
    )
    monkeypatch.setattr(
        models_cli,
        "sync_promoted_model",
        lambda name: called.append(name) or promoted,
    )

    result = models_cli.main(["sync", "demo"])

    assert result == 0
    assert called == ["demo-model"]
    assert "Did you mean 'demo-model'? [y/N]" in capsys.readouterr().out


def test_models_sync_cli_selects_ambiguous_partial_name(monkeypatch, capsys):
    from types import SimpleNamespace
    import lisai.promoted_models.cli as models_cli

    registry = SimpleNamespace(
        models={"demo-denoising": object(), "demo-upsampling": object()}
    )
    monkeypatch.setattr(models_cli, "load_promoted_model_registry", lambda: registry)
    monkeypatch.setattr("sys.stdin", InteractiveInput("02\n"))
    called = []
    promoted = SimpleNamespace(
        manifest=SimpleNamespace(
            name="demo-upsampling",
            model=SimpleNamespace(task="upsampling"),
        )
    )
    monkeypatch.setattr(
        models_cli,
        "sync_promoted_model",
        lambda name: called.append(name) or promoted,
    )

    result = models_cli.main(["sync", "demo"])

    assert result == 0
    assert called == ["demo-upsampling"]
    output = capsys.readouterr().out
    assert "Multiple matching promoted models found:" in output
    assert "demo-denoising" in output
    assert "demo-upsampling" in output


def test_models_catalog_cli_renders_remote_models(monkeypatch, capsys):
    from types import SimpleNamespace
    import lisai.promoted_models.cli as models_cli

    monkeypatch.setattr(
        models_cli.catalog,
        "list_models",
        lambda: [
            SimpleNamespace(
                name="demo-model",
                task="denoising_hdn",
                description="Demo downloadable model",
            )
        ],
    )

    result = models_cli.main(["catalog"])

    assert result == 0
    output = capsys.readouterr().out
    assert "demo-model" in output
    assert "denoising_hdn" in output
    assert "Demo downloadable model" in output


def test_models_download_cli_prints_short_install_command(monkeypatch, capsys, tmp_path):
    from types import SimpleNamespace
    import lisai.promoted_models.cli as models_cli

    archive = tmp_path / "demo-model.lisai.zip"
    monkeypatch.setattr(
        models_cli,
        "download_model",
        lambda name, overwrite=False: SimpleNamespace(
            name=name,
            archive_path=archive,
            archive_sha256="a" * 64,
            status="downloaded",
        ),
    )

    result = models_cli.main(["download", "demo-model"])

    assert result == 0
    output = capsys.readouterr().out
    assert f"Archive: {archive}" in output
    assert "lisai models install demo-model.lisai.zip" in output


def test_models_download_cli_install_composes_existing_installer(monkeypatch, capsys, tmp_path):
    from types import SimpleNamespace
    import lisai.promoted_models.cli as models_cli

    archive = tmp_path / "demo-model.lisai.zip"
    monkeypatch.setattr(
        models_cli,
        "download_model",
        lambda name, overwrite=False: SimpleNamespace(
            name=name,
            archive_path=archive,
            archive_sha256="a" * 64,
            status="reused",
        ),
    )
    called = []
    monkeypatch.setattr(
        models_cli,
        "install_model_archive",
        lambda path: called.append(path)
        or SimpleNamespace(
            model=SimpleNamespace(
                manifest=SimpleNamespace(name="demo-model"),
                model_dir=tmp_path / "models" / "demo-model",
            )
        ),
    )

    result = models_cli.main(["download", "demo-model", "--install"])

    assert result == 0
    assert called == [archive]
    output = capsys.readouterr().out
    assert "Existing archive checksum verified; reusing it." in output
    assert "Installed model: demo-model" in output


def test_models_download_cli_prompts_before_replacing_checksum_conflict(
    monkeypatch, capsys, tmp_path
):
    from types import SimpleNamespace
    import lisai.promoted_models.cli as models_cli
    from lisai.promoted_models.download import DownloadConflictError

    archive = tmp_path / "demo-model.lisai.zip"
    calls = []

    def fake_download(name, overwrite=False):
        calls.append(overwrite)
        if not overwrite:
            raise DownloadConflictError(
                path=archive,
                expected_sha256="a" * 64,
                actual_sha256="b" * 64,
            )
        return SimpleNamespace(
            name=name,
            archive_path=archive,
            archive_sha256="a" * 64,
            status="overwritten",
        )

    monkeypatch.setattr(models_cli, "download_model", fake_download)
    monkeypatch.setattr("builtins.input", lambda prompt: "y")

    result = models_cli.main(["download", "demo-model"])

    assert result == 0
    assert calls == [False, True]
    assert "Replaced the existing archive after checksum mismatch." in capsys.readouterr().out


def test_models_download_cli_supports_all_install(monkeypatch, capsys, tmp_path):
    from types import SimpleNamespace
    import lisai.promoted_models.cli as models_cli

    models = [
        SimpleNamespace(name="model-a"),
        SimpleNamespace(name="model-b"),
    ]
    monkeypatch.setattr(models_cli.catalog, "list_models", lambda: models)
    monkeypatch.setattr(
        models_cli,
        "load_promoted_model_registry",
        lambda: SimpleNamespace(models={}),
    )
    downloaded = []

    def fake_download(name, overwrite=False):
        downloaded.append(name)
        archive = tmp_path / f"{name}.lisai.zip"
        return SimpleNamespace(
            name=name,
            archive_path=archive,
            archive_sha256="a" * 64,
            status="downloaded",
        )

    monkeypatch.setattr(models_cli, "download_model", fake_download)
    installed = []
    monkeypatch.setattr(
        models_cli,
        "install_model_archive",
        lambda path: installed.append(path.name)
        or SimpleNamespace(
            model=SimpleNamespace(
                manifest=SimpleNamespace(name=path.name.removesuffix(".lisai.zip")),
                model_dir=tmp_path / "models" / path.stem,
            )
        ),
    )

    assert models_cli.main(["download", "--all", "--install"]) == 0
    assert downloaded == ["model-a", "model-b"]
    assert installed == ["model-a.lisai.zip", "model-b.lisai.zip"]
    output = capsys.readouterr().out
    assert "Installed model: model-a" in output
    assert "Installed model: model-b" in output


def test_models_download_cli_all_install_skips_already_installed(monkeypatch, capsys):
    from types import SimpleNamespace
    import lisai.promoted_models.cli as models_cli

    monkeypatch.setattr(
        models_cli.catalog,
        "list_models",
        lambda: [SimpleNamespace(name="model-a"), SimpleNamespace(name="model-b")],
    )
    monkeypatch.setattr(
        models_cli,
        "load_promoted_model_registry",
        lambda: SimpleNamespace(models={"model-a": object()}),
    )
    downloaded = []
    monkeypatch.setattr(
        models_cli,
        "download_model",
        lambda name, overwrite=False: downloaded.append(name)
        or SimpleNamespace(
            name=name,
            archive_path=Path(f"{name}.lisai.zip"),
            archive_sha256="a" * 64,
            status="downloaded",
        ),
    )
    monkeypatch.setattr(
        models_cli,
        "install_model_archive",
        lambda path: SimpleNamespace(
            model=SimpleNamespace(
                manifest=SimpleNamespace(name="model-b"),
                model_dir=Path("models/model-b"),
            )
        ),
    )

    assert models_cli.main(["download", "--all", "--install"]) == 0
    assert downloaded == ["model-b"]
    assert "Model already installed: model-a" in capsys.readouterr().out

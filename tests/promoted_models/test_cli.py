from __future__ import annotations

from lisai.promoted_models.cli import build_parser as build_models_parser
from lisai.runs.cli import build_parser as build_runs_parser


def test_runs_promote_cli_defaults():
    parser = build_runs_parser()
    args = parser.parse_args(["promote", "run_a", "--name", "hdn-vimentin"])

    assert args.run == "run_a"
    assert args.name == "hdn-vimentin"
    assert args.checkpoint == "best"
    assert args.overwrite is False


def test_runs_promote_cli_supports_checkpoint_and_overwrite():
    parser = build_runs_parser()
    args = parser.parse_args(
        ["promote", "run_a", "--name", "hdn-vimentin", "--checkpoint", "last", "--overwrite"]
    )

    assert args.checkpoint == "last"
    assert args.overwrite is True


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

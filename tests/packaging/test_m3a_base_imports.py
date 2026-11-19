"""M3-A: the base installation must not import optional full extras."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from lisai.lib.hdn.stable_dist_params import StableLogVar, StableMean


_BASE_IMPORT_SCRIPT = r'''
import importlib
import importlib.abc
import os
import sys
import tempfile
from pathlib import Path

BLOCKED = {
    "matplotlib", "sklearn", "skimage", "PIL", "torchvision",
    "cv2", "h5py", "tensorboard", "pystackreg",
}


class RejectOptionalImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in BLOCKED:
            raise ModuleNotFoundError("Optional dependency imported during base operation: " + fullname)


sys.meta_path.insert(0, RejectOptionalImports())

# Imports may read installed package resources but must not create/choose workspaces.
for module in (
    "lisai", "lisai.api", "lisai.cli", "lisai.promoted_models.package",
    "lisai.evaluation.run_apply_model", "lisai.models.lvae", "lisai.models.unet_rcan",
):
    importlib.import_module(module)

from lisai import train, predict
from lisai.cli import build_parser, main
from lisai.config.settings import Settings

assert callable(train) and callable(predict)
parser = build_parser()
for argv in (["--help"], ["models", "--help"], ["apply", "--help"],
             ["train", "--help"], ["workspace", "--help"]):
    try:
        parser.parse_args(argv)
    except SystemExit as exc:
        assert exc.code == 0
    else:
        raise AssertionError("--help did not exit")

with tempfile.TemporaryDirectory() as folder:
    workspace = Path(folder) / "workspace"
    assert main(["workspace", "init", str(workspace), "--core-only"]) == 0
    assert (workspace / "configs" / "workspace.yml").is_file()
    old = os.environ.get("LISAI_HOME")
    try:
        os.environ["LISAI_HOME"] = str(workspace)
        assert main(["models", "list"]) == 0
        assert isinstance(Settings.from_workspace(workspace), Settings)
    finally:
        if old is None:
            os.environ.pop("LISAI_HOME", None)
        else:
            os.environ["LISAI_HOME"] = old
print("M3A_BASE_IMPORTS_OK")
'''


def test_base_imports_cli_and_workspace_without_full_deps(tmp_path: Path):
    """Run in a new process so optional packages are not already imported."""
    env = os.environ.copy()
    env.pop("LISAI_HOME", None)
    env.pop("LISAI_CONFIG_ROOT", None)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2] / "src")
    result = subprocess.run(
        [sys.executable, "-c", _BASE_IMPORT_SCRIPT],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "M3A_BASE_IMPORTS_OK" in result.stdout


@pytest.mark.parametrize("shape", [(2, 3, 8, 8), (1, 5, 10, 12)])
def test_stable_distribution_crop_needs_only_torch(shape):
    values = torch.arange(int(torch.tensor(shape).prod()), dtype=torch.float32).reshape(shape)
    center_size = 4
    top = (shape[-2] - center_size) // 2
    left = (shape[-1] - center_size) // 2
    expected = values[..., top : top + center_size, left : left + center_size]

    logvar = StableLogVar(values.clone())
    logvar.centercrop_to_size(center_size)
    torch.testing.assert_close(logvar._lv, expected)

    mean = StableMean(values.clone())
    mean.centercrop_to_size(center_size)
    torch.testing.assert_close(mean.get(), expected)


def test_stable_distribution_rejects_oversize_crop():
    mean = StableMean(torch.zeros(2, 3, 4, 4))
    with pytest.raises(AssertionError):
        mean.centercrop_to_size(6)

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from lisai.promoted_models.schema import (
    PromotedModelDefinition,
    PromotedModelManifest,
    PromotedModelSource,
    PromotedTrainingData,
)


def _manifest(**overrides):
    payload = {
        "name": "hdn-vimentin",
        "created_at": datetime(2026, 8, 24, tzinfo=timezone.utc),
        "model": PromotedModelDefinition(task="denoising_hdn", architecture="lvae"),
        "training_data": PromotedTrainingData(dataset="VimFixed"),
        "source": PromotedModelSource(
            run_id="01ARZ3NDEKTSV4RRFFQ69G5FAV",
            run_name="run_a",
            run_status="stopped",
            checkpoint_selector="best",
            checkpoint_filename="model_best_state_dict.pt",
        ),
    }
    payload.update(overrides)
    return PromotedModelManifest(**payload)


def test_manifest_accepts_stopped_run_and_dataset_name():
    manifest = _manifest()

    assert manifest.source.run_status == "stopped"
    assert manifest.training_data.dataset == "VimFixed"
    assert manifest.artifacts.weights == "weights.pt"
    assert manifest.model_dump(mode="json")["created_at"] == "2026-08-24T00:00:00Z"


def test_manifest_public_name_cannot_be_a_path():
    with pytest.raises(ValidationError, match="path separators"):
        _manifest(name="folder/hdn-vimentin")


def test_manifest_artifact_paths_must_be_safe_relative_paths():
    manifest = _manifest()
    payload = manifest.model_dump(mode="python")
    payload["artifacts"]["weights"] = "../weights.pt"

    with pytest.raises(ValidationError, match="relative"):
        PromotedModelManifest.model_validate(payload)


def test_manifest_checksums_require_sha256_hex():
    manifest = _manifest()
    payload = manifest.model_dump(mode="python")
    payload["checksums"] = {"weights.pt": "a" * 64}
    validated = PromotedModelManifest.model_validate(payload)
    assert validated.checksums["weights.pt"] == "a" * 64

    payload["checksums"] = {"weights.pt": "not-a-sha"}
    with pytest.raises(ValidationError, match="SHA256"):
        PromotedModelManifest.model_validate(payload)

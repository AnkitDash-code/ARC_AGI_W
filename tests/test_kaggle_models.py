from __future__ import annotations

from pathlib import Path
import os

import pytest

from mythos.kaggle_models import MODEL_ENV_KEYS, autodiscover_model_inputs


def _clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in MODEL_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


def test_autodiscover_sets_hrm_repo_and_checkpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_env(monkeypatch)
    hrm_repo = tmp_path / "hrm-code" / "HRM"
    (hrm_repo / "dataset").mkdir(parents=True)
    (hrm_repo / "evaluate.py").write_text("# fake", encoding="utf-8")
    (hrm_repo / "dataset" / "build_arc_dataset.py").write_text("# fake", encoding="utf-8")
    checkpoint = tmp_path / "hrm-checkpoint" / "hrm_model.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"fake")

    try:
        result = autodiscover_model_inputs(tmp_path, apply=True)

        assert result["set"]["HRM_REPO_DIR"] == str(hrm_repo)
        assert result["set"]["HRM_CHECKPOINT_PATH"] == str(checkpoint)
        assert str(hrm_repo) == __import__("os").environ["HRM_REPO_DIR"]
    finally:
        os.environ.pop("HRM_REPO_DIR", None)
        os.environ.pop("HRM_CHECKPOINT_PATH", None)


def test_autodiscover_sets_checkpoint_only_models(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_env(monkeypatch)
    world = tmp_path / "world-model" / "world_model.pt"
    lora = tmp_path / "lora-adapter" / "adapter_lora.pt"
    projection = tmp_path / "ijepa-projection" / "ijepa_projection.pt"
    world.parent.mkdir()
    lora.parent.mkdir()
    projection.parent.mkdir()
    world.write_bytes(b"fake")
    lora.write_bytes(b"fake")
    projection.write_bytes(b"fake")

    result = autodiscover_model_inputs(tmp_path, apply=False)

    assert result["set"]["IJEPA_PROJECTION_CHECKPOINT_PATH"] == str(projection)
    assert result["set"]["WORLD_MODEL_CHECKPOINT_PATH"] == str(world)
    assert result["set"]["TTT_LORA_CHECKPOINT_PATH"] == str(lora)


def test_autodiscover_sets_ijepa_checkpoint_without_code_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_env(monkeypatch)
    checkpoint = tmp_path / "facebook-ijepa-vith14" / "model.safetensors"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"fake")

    result = autodiscover_model_inputs(tmp_path, apply=False)

    assert result["set"]["IJEPA_CHECKPOINT_PATH"] == str(checkpoint)

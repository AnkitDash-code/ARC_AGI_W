from __future__ import annotations

import pytest

from pathlib import Path

from mythos.models import MODEL_SPECS, ModelLoadError, ModelRegistry, _load_model_from_env


def _clear_model_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for spec in MODEL_SPECS:
        monkeypatch.delenv(spec.checkpoint_env, raising=False)
        if spec.repo_env is not None:
            monkeypatch.delenv(spec.repo_env, raising=False)


def test_model_registry_fallback_allows_unconfigured_models(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_model_env(monkeypatch)

    registry = ModelRegistry.from_env(strict=False)

    assert not registry.get("jepa").loaded
    assert not registry.get("hrm_l_module").loaded


def test_model_registry_strict_requires_all_model_env(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_model_env(monkeypatch)

    with pytest.raises(ModelLoadError, match="required for strict model loading"):
        ModelRegistry.from_env(strict=True)


def test_jepa_model_loads_from_checkpoint_without_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_model_env(monkeypatch)
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.write_bytes(b"fake")
    spec = next(item for item in MODEL_SPECS if item.key == "jepa")
    monkeypatch.setenv("IJEPA_CHECKPOINT_PATH", str(checkpoint))

    loaded = _load_model_from_env(spec, strict=True)

    assert loaded.loaded
    assert loaded.repo_dir is None
    assert loaded.checkpoint_path == checkpoint

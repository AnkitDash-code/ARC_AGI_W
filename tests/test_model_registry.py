from __future__ import annotations

import pytest

from mythos.models import MODEL_SPECS, ModelLoadError, ModelRegistry


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

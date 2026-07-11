from __future__ import annotations

from mythos.kaggle_models import (
    download_direct_checkpoint_inputs,
    download_git_code_repositories,
    download_huggingface_model_inputs,
)


def test_huggingface_download_skips_when_no_repo_ids(monkeypatch) -> None:
    for key in (
        "IJEPA_HF_REPO_ID",
        "IJEPA_PROJECTION_HF_REPO_ID",
        "HRM_TEXT_HF_REPO_ID",
        "WORLD_MODEL_HF_REPO_ID",
        "TTT_LORA_HF_REPO_ID",
        "HRM_HF_REPO_ID",
    ):
        monkeypatch.delenv(key, raising=False)

    result = download_huggingface_model_inputs(apply=False)

    assert result["downloaded"] == {}
    assert "HRM_HF_REPO_ID" in result["skipped"]


def test_git_code_download_skips_when_no_urls(monkeypatch) -> None:
    for key in ("IJEPA_GIT_REPO_URL", "HRM_TEXT_GIT_REPO_URL", "HRM_GIT_REPO_URL"):
        monkeypatch.delenv(key, raising=False)

    result = download_git_code_repositories(apply=False)

    assert result["cloned"] == {}
    assert "HRM_GIT_REPO_URL" in result["skipped"]


def test_direct_checkpoint_download_skips_when_no_urls(monkeypatch) -> None:
    for key in (
        "IJEPA_CHECKPOINT_URL",
        "IJEPA_PROJECTION_CHECKPOINT_URL",
        "HRM_TEXT_CHECKPOINT_URL",
        "WORLD_MODEL_CHECKPOINT_URL",
        "TTT_LORA_CHECKPOINT_URL",
        "HRM_CHECKPOINT_URL",
    ):
        monkeypatch.delenv(key, raising=False)

    result = download_direct_checkpoint_inputs(apply=False)

    assert result["downloaded"] == {}
    assert "IJEPA_CHECKPOINT_URL" in result["skipped"]

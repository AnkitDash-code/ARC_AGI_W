"""Kaggle model input auto-discovery.

Kaggle submissions usually mount model code and checkpoints under
`/kaggle/input/<dataset-name>/...`. This module scans those inputs and sets the
environment variables consumed by `mythos.models.ModelRegistry`.
"""

from __future__ import annotations

from pathlib import Path
import os
import subprocess
import urllib.request
from typing import Iterable


CHECKPOINT_SUFFIXES = (".pt", ".pth", ".ckpt", ".bin", ".safetensors", ".tar")


MODEL_ENV_KEYS = (
    "IJEPA_CHECKPOINT_PATH",
    "IJEPA_PROJECTION_CHECKPOINT_PATH",
    "HRM_TEXT_REPO_DIR",
    "HRM_TEXT_CHECKPOINT_PATH",
    "WORLD_MODEL_CHECKPOINT_PATH",
    "TTT_LORA_CHECKPOINT_PATH",
    "HRM_REPO_DIR",
    "HRM_CHECKPOINT_PATH",
)

HF_MODEL_SPECS = (
    {
        "name": "jepa",
        "repo_id_env": "IJEPA_HF_REPO_ID",
        "repo_dir_env": None,
        "checkpoint_env": "IJEPA_CHECKPOINT_PATH",
    },
    {
        "name": "jepa_projection",
        "repo_id_env": "IJEPA_PROJECTION_HF_REPO_ID",
        "repo_dir_env": None,
        "checkpoint_env": "IJEPA_PROJECTION_CHECKPOINT_PATH",
    },
    {
        "name": "hrm_text",
        "repo_id_env": "HRM_TEXT_HF_REPO_ID",
        "repo_dir_env": None,
        "checkpoint_env": "HRM_TEXT_CHECKPOINT_PATH",
    },
    {
        "name": "world_model",
        "repo_id_env": "WORLD_MODEL_HF_REPO_ID",
        "repo_dir_env": None,
        "checkpoint_env": "WORLD_MODEL_CHECKPOINT_PATH",
    },
    {
        "name": "ttt_lora",
        "repo_id_env": "TTT_LORA_HF_REPO_ID",
        "repo_dir_env": None,
        "checkpoint_env": "TTT_LORA_CHECKPOINT_PATH",
    },
    {
        "name": "hrm_l_module",
        "repo_id_env": "HRM_HF_REPO_ID",
        "repo_dir_env": None,
        "checkpoint_env": "HRM_CHECKPOINT_PATH",
    },
)

GIT_REPO_SPECS = (
    {
        "name": "hrm_text",
        "url_env": "HRM_TEXT_GIT_REPO_URL",
        "repo_dir_env": "HRM_TEXT_REPO_DIR",
    },
    {
        "name": "hrm_l_module",
        "url_env": "HRM_GIT_REPO_URL",
        "repo_dir_env": "HRM_REPO_DIR",
    },
)

DIRECT_CHECKPOINT_SPECS = (
    {
        "name": "jepa",
        "url_env": "IJEPA_CHECKPOINT_URL",
        "checkpoint_env": "IJEPA_CHECKPOINT_PATH",
    },
    {
        "name": "jepa_projection",
        "url_env": "IJEPA_PROJECTION_CHECKPOINT_URL",
        "checkpoint_env": "IJEPA_PROJECTION_CHECKPOINT_PATH",
    },
    {
        "name": "hrm_text",
        "url_env": "HRM_TEXT_CHECKPOINT_URL",
        "checkpoint_env": "HRM_TEXT_CHECKPOINT_PATH",
    },
    {
        "name": "world_model",
        "url_env": "WORLD_MODEL_CHECKPOINT_URL",
        "checkpoint_env": "WORLD_MODEL_CHECKPOINT_PATH",
    },
    {
        "name": "ttt_lora",
        "url_env": "TTT_LORA_CHECKPOINT_URL",
        "checkpoint_env": "TTT_LORA_CHECKPOINT_PATH",
    },
    {
        "name": "hrm_l_module",
        "url_env": "HRM_CHECKPOINT_URL",
        "checkpoint_env": "HRM_CHECKPOINT_PATH",
    },
)


def download_git_code_repositories(
    output_root: str | Path = "/kaggle/working/model_code",
    *,
    apply: bool = True,
) -> dict[str, object]:
    """Clone configured Git repos for model code and export repo env vars."""

    result: dict[str, object] = {
        "output_root": str(output_root),
        "cloned": {},
        "skipped": [],
        "errors": {},
    }
    configured = [spec for spec in GIT_REPO_SPECS if os.environ.get(str(spec["url_env"]))]
    if not configured:
        result["skipped"] = [str(spec["url_env"]) for spec in GIT_REPO_SPECS]
        return result

    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    for spec in configured:
        name = str(spec["name"])
        url = os.environ[str(spec["url_env"])]
        repo_dir = root / name
        try:
            if not repo_dir.exists():
                subprocess.run(
                    ["git", "clone", "--depth", "1", url, str(repo_dir)],
                    check=True,
                    capture_output=True,
                    text=True,
                )
            if apply:
                os.environ.setdefault(str(spec["repo_dir_env"]), str(repo_dir))
            result["cloned"][name] = {
                "url": url,
                "repo_dir": str(repo_dir),
                "repo_dir_env": spec["repo_dir_env"],
            }
        except Exception as exc:
            result["errors"][name] = str(exc)
    return result


def download_direct_checkpoint_inputs(
    output_root: str | Path = "/kaggle/working/model_inputs/direct",
    *,
    apply: bool = True,
) -> dict[str, object]:
    """Download configured direct checkpoint URLs and export checkpoint env vars."""

    result: dict[str, object] = {
        "output_root": str(output_root),
        "downloaded": {},
        "skipped": [],
        "errors": {},
    }
    configured = [spec for spec in DIRECT_CHECKPOINT_SPECS if os.environ.get(str(spec["url_env"]))]
    if not configured:
        result["skipped"] = [str(spec["url_env"]) for spec in DIRECT_CHECKPOINT_SPECS]
        return result

    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    for spec in configured:
        name = str(spec["name"])
        url = os.environ[str(spec["url_env"])]
        target = root / name / _filename_from_url(url)
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                urllib.request.urlretrieve(url, target)
            checkpoint_env = str(spec["checkpoint_env"])
            if apply:
                os.environ.setdefault(checkpoint_env, str(target))
            result["downloaded"][name] = {
                "url": url,
                "checkpoint": str(target),
                "checkpoint_env": checkpoint_env,
            }
        except Exception as exc:
            result["errors"][name] = str(exc)
    return result


def download_huggingface_model_inputs(
    output_root: str | Path = "/kaggle/working/model_inputs",
    *,
    apply: bool = True,
) -> dict[str, object]:
    """Download configured Hugging Face model repos before model loading.

    Configure with env vars such as `HRM_HF_REPO_ID`. Optional env vars named
    `<PREFIX>_HF_CHECKPOINT_GLOB` can narrow checkpoint selection, for example
    `HRM_HF_CHECKPOINT_GLOB="*.pt"`.
    """

    result: dict[str, object] = {
        "output_root": str(output_root),
        "downloaded": {},
        "skipped": [],
        "errors": {},
    }

    configured = [spec for spec in HF_MODEL_SPECS if os.environ.get(str(spec["repo_id_env"]))]
    if not configured:
        result["skipped"] = [str(spec["repo_id_env"]) for spec in HF_MODEL_SPECS]
        return result

    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:
        result["errors"]["huggingface_hub"] = (
            "huggingface_hub is not installed or cannot be imported: " + str(exc)
        )
        return result

    output_dir = Path(output_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    for spec in configured:
        name = str(spec["name"])
        repo_id_env = str(spec["repo_id_env"])
        repo_id = os.environ[repo_id_env]
        local_dir = output_dir / name
        glob_env = f"{repo_id_env.removesuffix('_REPO_ID')}_CHECKPOINT_GLOB"
        checkpoint_glob = os.environ.get(glob_env)

        try:
            downloaded = Path(
                snapshot_download(
                    repo_id=repo_id,
                    local_dir=local_dir,
                    local_dir_use_symlinks=False,
                )
            )
            checkpoint = _find_checkpoint_by_glob(downloaded, checkpoint_glob)
            if checkpoint is None:
                result["errors"][name] = (
                    f"downloaded {repo_id} to {downloaded}, but found no checkpoint "
                    f"matching {checkpoint_glob or CHECKPOINT_SUFFIXES}"
                )
                continue

            repo_dir_env = spec["repo_dir_env"]
            checkpoint_env = str(spec["checkpoint_env"])
            if apply:
                if repo_dir_env is not None:
                    os.environ.setdefault(str(repo_dir_env), str(downloaded))
                os.environ.setdefault(checkpoint_env, str(checkpoint))

            result["downloaded"][name] = {
                "repo_id": repo_id,
                "local_dir": str(downloaded),
                "checkpoint": str(checkpoint),
                "repo_dir_env": repo_dir_env,
                "checkpoint_env": checkpoint_env,
            }
        except Exception as exc:
            result["errors"][name] = str(exc)

    return result


def autodiscover_model_inputs(
    input_root: str | Path = "/kaggle/input",
    *,
    apply: bool = True,
) -> dict[str, object]:
    """Find likely model repos/checkpoints and optionally export env vars."""

    root = Path(input_root)
    result: dict[str, object] = {
        "input_root": str(root),
        "exists": root.exists(),
        "set": {},
        "missing": [],
    }
    if not root.exists():
        result["missing"] = list(MODEL_ENV_KEYS)
        return result

    discovered: dict[str, Path] = {}

    hrm_repo = _find_hrm_repo(root)
    hrm_checkpoint = _find_checkpoint(root, include=("hrm",), exclude=("text", "lora", "adapter"))
    if hrm_repo is not None and hrm_checkpoint is not None:
        discovered["HRM_REPO_DIR"] = hrm_repo
        discovered["HRM_CHECKPOINT_PATH"] = hrm_checkpoint

    ijepa_checkpoint = _find_checkpoint(root, include=("ijepa", "i-jepa", "jepa"), exclude=("projection",))
    if ijepa_checkpoint is not None:
        discovered["IJEPA_CHECKPOINT_PATH"] = ijepa_checkpoint

    ijepa_projection_checkpoint = _find_checkpoint(root, include=("ijepa", "projection"), require_all=True)
    if ijepa_projection_checkpoint is not None:
        discovered["IJEPA_PROJECTION_CHECKPOINT_PATH"] = ijepa_projection_checkpoint

    hrm_text_repo = _find_named_repo(root, names=("hrm-text", "hrm_text", "hrmtext"))
    hrm_text_checkpoint = _find_checkpoint(root, include=("hrm", "text"), require_all=True)
    if hrm_text_repo is not None and hrm_text_checkpoint is not None:
        discovered["HRM_TEXT_REPO_DIR"] = hrm_text_repo
        discovered["HRM_TEXT_CHECKPOINT_PATH"] = hrm_text_checkpoint

    world_model_checkpoint = _find_checkpoint(root, include=("world", "transition"))
    if world_model_checkpoint is not None:
        discovered["WORLD_MODEL_CHECKPOINT_PATH"] = world_model_checkpoint

    lora_checkpoint = _find_checkpoint(root, include=("lora", "adapter"))
    if lora_checkpoint is not None:
        discovered["TTT_LORA_CHECKPOINT_PATH"] = lora_checkpoint

    set_values: dict[str, str] = {}
    for key, path in discovered.items():
        if key in os.environ:
            set_values[key] = os.environ[key]
            continue
        if apply:
            os.environ[key] = str(path)
        set_values[key] = str(path)

    result["set"] = set_values
    result["missing"] = [key for key in MODEL_ENV_KEYS if key not in set_values and key not in os.environ]
    return result


def _find_hrm_repo(root: Path) -> Path | None:
    for path in _iter_dirs(root):
        if (path / "evaluate.py").exists() and (path / "dataset" / "build_arc_dataset.py").exists():
            return path
    return None


def _find_named_repo(root: Path, *, names: Iterable[str]) -> Path | None:
    lowered_names = tuple(name.lower() for name in names)
    for path in _iter_dirs(root):
        path_text = path.as_posix().lower()
        if any(name in path_text for name in lowered_names):
            return path
    return None


def _find_checkpoint(
    root: Path,
    *,
    include: Iterable[str],
    exclude: Iterable[str] = (),
    require_all: bool = False,
) -> Path | None:
    include_terms = tuple(term.lower() for term in include)
    exclude_terms = tuple(term.lower() for term in exclude)
    candidates = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in CHECKPOINT_SUFFIXES:
            continue
        path_text = path.as_posix().lower()
        if require_all and not all(term in path_text for term in include_terms):
            continue
        if not require_all and not any(term in path_text for term in include_terms):
            continue
        if any(term in path_text for term in exclude_terms):
            continue
        candidates.append(path)
    if not candidates:
        return None
    return sorted(candidates, key=lambda item: (len(item.as_posix()), item.as_posix()))[0]


def _find_checkpoint_by_glob(root: Path, checkpoint_glob: str | None) -> Path | None:
    if checkpoint_glob:
        candidates = [path for path in root.rglob(checkpoint_glob) if path.is_file()]
    else:
        candidates = [
            path
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in CHECKPOINT_SUFFIXES
        ]
    if not candidates:
        return None
    return sorted(candidates, key=lambda item: (len(item.as_posix()), item.as_posix()))[0]


def _filename_from_url(url: str) -> str:
    filename = url.rstrip("/").split("/")[-1]
    return filename or "checkpoint.pt"


def _iter_dirs(root: Path):
    for path in root.rglob("*"):
        if path.is_dir():
            yield path

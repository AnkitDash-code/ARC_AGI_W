"""Build a single-file Kaggle notebook with the Mythos package embedded."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src" / "mythos"
OUTPUT = ROOT / "project_mythos_kaggle_pipeline_standalone.ipynb"


def _source_lines(source: str) -> list[str]:
    return source.splitlines(keepends=True)


def _markdown(source: str) -> dict[str, object]:
    return {"cell_type": "markdown", "metadata": {}, "source": _source_lines(source)}


def _code(source: str) -> dict[str, object]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _source_lines(source),
    }


def _embedded_files() -> dict[str, str]:
    files: dict[str, str] = {}
    for path in sorted(SRC_ROOT.rglob("*.py")):
        relative = path.relative_to(ROOT).as_posix()
        files[relative] = path.read_text(encoding="utf-8")
    return files


def _bootstrap_cell(files: dict[str, str]) -> str:
    return f"""from pathlib import Path
import os
import sys

EMBEDDED_FILES = {files!r}

EMBED_ROOT = Path(os.environ.get('MYTHOS_EMBED_ROOT', '/kaggle/working/project_mythos_embedded'))
if not EMBED_ROOT.parent.exists():
    EMBED_ROOT = Path.cwd() / 'project_mythos_embedded'

for relative_path, content in EMBEDDED_FILES.items():
    path = EMBED_ROOT / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding='utf-8')

SRC_DIR = EMBED_ROOT / 'src'
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

print('Embedded Mythos package written to:', SRC_DIR)
print('Embedded files:', len(EMBEDDED_FILES))
"""


def build() -> None:
    files = _embedded_files()
    notebook = {
        "cells": [
            _markdown(
                "# Project Mythos Standalone Kaggle Pipeline\n\n"
                "Upload this `.ipynb` by itself. It embeds the current `src/mythos` package, "
                "writes it into `/kaggle/working/project_mythos_embedded/src`, then runs the "
                "plan-aligned ARC pipeline and writes `/kaggle/working/submission.json`.\n\n"
                "Default mode is `pipeline` + `fallback`, which loads any configured model "
                "checkpoints and uses explicit fallback adapters for missing model stages."
            ),
            _markdown("## 1. Bootstrap Embedded Project Mythos Code"),
            _code(_bootstrap_cell(files)),
            _markdown("## 2. Configuration"),
            _code(
                "from pathlib import Path\n"
                "import json\n"
                "import os\n"
                "import time\n\n"
                "DATA_DIR = Path('/kaggle/input/competitions/arc-prize-2026-arc-agi-2')\n"
                "SPLIT = 'test'  # 'training', 'evaluation', or 'test'\n"
                "SOLVER_NAME = 'pipeline'  # 'pipeline', 'baseline', 'fixture', or 'hrm'\n"
                "MODEL_MODE = 'fallback'  # 'fallback' or 'strict'\n"
                "AUTO_DOWNLOAD_GIT_CODE = True\n"
                "AUTO_DOWNLOAD_HF_MODELS = True\n"
                "AUTO_DOWNLOAD_DIRECT_CHECKPOINTS = True\n"
                "AUTO_DISCOVER_MODELS = True\n"
                "OUTPUT_PATH = Path('/kaggle/working/submission.json')\n"
                "RUN_HRM_SMOKE = False\n\n"
                "# Verified public defaults looked up from official sources.\n"
                "os.environ.setdefault('HRM_GIT_REPO_URL', 'https://github.com/sapientinc/HRM.git')\n"
                "os.environ.setdefault('HRM_HF_REPO_ID', 'sapientinc/HRM-checkpoint-ARC-2')\n"
                "os.environ.setdefault('HRM_HF_CHECKPOINT_GLOB', '*.pt')\n"
                "os.environ.setdefault('IJEPA_GIT_REPO_URL', 'https://github.com/facebookresearch/ijepa.git')\n"
                "os.environ.setdefault('IJEPA_CHECKPOINT_URL', 'https://dl.fbaipublicfiles.com/ijepa/IN1K-vit.h.14-300e.pth.tar')\n\n"
                "# Optional real-model inputs. Set these if you have additional public/private model repos.\n"
                "# os.environ['IJEPA_HF_REPO_ID'] = '<org-or-user>/<ijepa-model-repo>'\n"
                "# os.environ['IJEPA_HF_CHECKPOINT_GLOB'] = '*.pt'\n"
                "# os.environ['HRM_TEXT_HF_REPO_ID'] = '<org-or-user>/<hrm-text-model-repo>'\n"
                "# os.environ['WORLD_MODEL_HF_REPO_ID'] = '<org-or-user>/<world-model-repo>'\n"
                "# os.environ['TTT_LORA_HF_REPO_ID'] = '<org-or-user>/<lora-repo>'\n"
                "# os.environ['HRM_HF_REPO_ID'] = '<org-or-user>/<hrm-model-repo>'\n"
                "# os.environ['HRM_HF_CHECKPOINT_GLOB'] = '*.pt'\n\n"
                "# Or set explicit Kaggle input paths when internet/download is unavailable.\n"
                "# os.environ['IJEPA_REPO_DIR'] = '/kaggle/input/<ijepa-code>/ijepa'\n"
                "# os.environ['IJEPA_CHECKPOINT_PATH'] = '/kaggle/input/<jepa-checkpoint>/checkpoint.pt'\n"
                "# os.environ['HRM_TEXT_REPO_DIR'] = '/kaggle/input/<hrm-text-code>/hrm-text'\n"
                "# os.environ['HRM_TEXT_CHECKPOINT_PATH'] = '/kaggle/input/<hrm-text-checkpoint>/checkpoint.pt'\n"
                "# os.environ['WORLD_MODEL_CHECKPOINT_PATH'] = '/kaggle/input/<world-model>/world_model.pt'\n"
                "# os.environ['TTT_LORA_CHECKPOINT_PATH'] = '/kaggle/input/<lora>/lora.pt'\n"
                "# os.environ['HRM_REPO_DIR'] = '/kaggle/input/<hrm-code>/HRM'\n"
                "# os.environ['HRM_CHECKPOINT_PATH'] = '/kaggle/input/<hrm-checkpoint>/checkpoint.pt'\n\n"
                "print('DATA_DIR =', DATA_DIR)\n"
                "print('SPLIT =', SPLIT)\n"
                "print('SOLVER_NAME =', SOLVER_NAME)\n"
                "print('MODEL_MODE =', MODEL_MODE)\n"
                "print('AUTO_DOWNLOAD_GIT_CODE =', AUTO_DOWNLOAD_GIT_CODE)\n"
                "print('AUTO_DOWNLOAD_HF_MODELS =', AUTO_DOWNLOAD_HF_MODELS)\n"
                "print('AUTO_DOWNLOAD_DIRECT_CHECKPOINTS =', AUTO_DOWNLOAD_DIRECT_CHECKPOINTS)\n"
                "print('AUTO_DISCOVER_MODELS =', AUTO_DISCOVER_MODELS)\n"
                "print('OUTPUT_PATH =', OUTPUT_PATH)\n"
            ),
            _markdown("## 3. Import Mythos Runtime"),
            _code(
                "import mythos\n"
                "from mythos.arc import load_challenges\n"
                "from mythos.kaggle_run import resolve_challenge_path, resolve_solution_path\n"
                "from mythos.kaggle_models import (\n"
                "    autodiscover_model_inputs,\n"
                "    download_direct_checkpoint_inputs,\n"
                "    download_git_code_repositories,\n"
                "    download_huggingface_model_inputs,\n"
                ")\n"
                "from mythos.metrics import score_files\n"
                "from mythos.pipeline import PLAN_STAGE_ORDER\n"
                "from mythos.solvers.factory import make_solver\n"
                "from mythos.submission import load_submission, write_submission\n\n"
                "print('Imported mythos from:', mythos.__file__)\n"
                "print('PLAN_STAGE_ORDER =', ' -> '.join(PLAN_STAGE_ORDER))\n"
            ),
            _markdown("## 4. Load ARC Data"),
            _code(
                "challenge_path = resolve_challenge_path(DATA_DIR, SPLIT)\n"
                "solution_path = resolve_solution_path(DATA_DIR, SPLIT)\n"
                "tasks = load_challenges(challenge_path)\n\n"
                "train_examples = sum(len(task.train) for task in tasks.values())\n"
                "test_items = sum(len(task.test) for task in tasks.values())\n\n"
                "print('challenge_path =', challenge_path)\n"
                "print('solution_path =', solution_path)\n"
                "print('tasks =', len(tasks))\n"
                "print('train_examples =', train_examples)\n"
                "print('test_items =', test_items)\n"
            ),
            _markdown("## 5. Load Models / Solver"),
            _code(
                "if AUTO_DOWNLOAD_GIT_CODE:\n"
                "    git_download = download_git_code_repositories(apply=True)\n"
                "    print('git_code_download =')\n"
                "    print(json.dumps(git_download, indent=2))\n\n"
                "if AUTO_DOWNLOAD_HF_MODELS:\n"
                "    hf_download = download_huggingface_model_inputs(apply=True)\n"
                "    print('huggingface_model_download =')\n"
                "    print(json.dumps(hf_download, indent=2))\n\n"
                "if AUTO_DOWNLOAD_DIRECT_CHECKPOINTS:\n"
                "    direct_download = download_direct_checkpoint_inputs(apply=True)\n"
                "    print('direct_checkpoint_download =')\n"
                "    print(json.dumps(direct_download, indent=2))\n\n"
                "if AUTO_DISCOVER_MODELS:\n"
                "    discovery = autodiscover_model_inputs(apply=True)\n"
                "    print('model_autodiscovery =')\n"
                "    print(json.dumps(discovery, indent=2))\n\n"
                "solver = make_solver(SOLVER_NAME, model_mode=MODEL_MODE)\n"
                "print('Loaded solver:', solver.__class__.__name__)\n\n"
                "if hasattr(solver, 'pipeline'):\n"
                "    print('model_registry =')\n"
                "    print(json.dumps(solver.pipeline.model_registry.summary(), indent=2))\n"
            ),
            _markdown("## 6. Run Plan-Aligned Pipeline"),
            _code(
                "started = time.perf_counter()\n"
                "predictions = []\n\n"
                "for index, task in enumerate(tasks.values(), start=1):\n"
                "    prediction = solver.solve(task)\n"
                "    predictions.append(prediction)\n"
                "    if index <= 3 or index == len(tasks):\n"
                "        print(f'{index}/{len(tasks)} solved: {task.id}')\n\n"
                "write_submission(predictions, OUTPUT_PATH)\n"
                "elapsed = time.perf_counter() - started\n\n"
                "print('Wrote submission:', OUTPUT_PATH)\n"
                "print('tasks_predicted =', len(predictions))\n"
                "print('elapsed_seconds =', round(elapsed, 3))\n\n"
                "if hasattr(solver, 'last_trace') and solver.last_trace is not None:\n"
                "    print('last_pipeline_trace =')\n"
                "    print(json.dumps(solver.last_trace.to_dict(), indent=2))\n"
            ),
            _markdown("## 7. Validate Submission and Score When Solutions Exist"),
            _code(
                "submission = load_submission(OUTPUT_PATH)\n"
                "submission_items = sum(len(outputs) for outputs in submission.values())\n\n"
                "print('submission_tasks =', len(submission))\n"
                "print('submission_test_items =', submission_items)\n"
                "print('sample_task_id =', next(iter(submission)))\n\n"
                "if solution_path is not None:\n"
                "    score = score_files(str(OUTPUT_PATH), str(solution_path))\n"
                "    print('score =')\n"
                "    print(json.dumps(score.to_dict(), indent=2, sort_keys=True))\n"
                "else:\n"
                "    print('No solutions file for this split; skipping score.')\n"
            ),
            _markdown("## 8. Optional HRM Smoke Test"),
            _code(
                "if RUN_HRM_SMOKE:\n"
                "    from mythos.hrm_dataset import prepare_hrm_raw_dataset\n"
                "    from mythos.solvers.hrm import HRMEnvironment\n\n"
                "    env = HRMEnvironment.from_env()\n"
                "    env.validate(require_cuda=True)\n"
                "    modules = env.import_modules()\n"
                "    checkpoint = env.load_checkpoint()\n"
                "    raw_dir = prepare_hrm_raw_dataset(tasks.values(), Path('/kaggle/working/hrm_smoke/raw/ARC-AGI-2/data'))\n\n"
                "    print('HRM repo:', env.repo_dir)\n"
                "    print('HRM checkpoint:', env.checkpoint_path)\n"
                "    print('Imported modules:', sorted(modules))\n"
                "    print('Checkpoint type:', type(checkpoint).__name__)\n"
                "    print('Prepared HRM raw data:', raw_dir)\n"
                "else:\n"
                "    print('RUN_HRM_SMOKE is False; skipping HRM smoke test.')\n"
            ),
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUTPUT.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    build()

"""Build a single-file Kaggle notebook with the Mythos package embedded."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src" / "mythos"
# Vendored third-party code that isn't part of the mythos package (keeps its
# own top-level import style, e.g. `import arc_compressor`, unmodified from
# upstream -- see third_party/compress_arc/NOTICE.md), so it gets its own
# sys.path entry rather than being embedded under src/mythos.
THIRD_PARTY_ROOT = ROOT / "third_party"
# agentic_repl is a separate top-level package (see its __init__.py), not
# part of src/mythos, so it's embedded under its own top-level directory
# below rather than under src/. models/ is excluded: that holds only the
# (unrun) staging script + docs for the multi-GB GGUF artifact, which is
# mounted from a Kaggle Dataset at runtime, never embedded as source.
AGENTIC_REPL_ROOT = ROOT / "agentic_repl"
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
    for path in sorted(THIRD_PARTY_ROOT.rglob("*")):
        if path.is_file() and path.suffix in (".py", ".md") or path.name == "LICENSE":
            relative = path.relative_to(ROOT).as_posix()
            files[relative] = path.read_text(encoding="utf-8")
    for path in sorted(AGENTIC_REPL_ROOT.rglob("*.py")):
        if path.relative_to(AGENTIC_REPL_ROOT).parts[0] == "models":
            continue  # staging script only; the model itself is a mounted Kaggle Dataset
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

# Vendored third-party code (e.g. CompressARC) keeps its own top-level
# import style (`import arc_compressor`, not a package), so its directory
# goes on sys.path directly rather than under src/.
COMPRESS_ARC_DIR = EMBED_ROOT / 'third_party' / 'compress_arc'
if COMPRESS_ARC_DIR.is_dir() and str(COMPRESS_ARC_DIR) not in sys.path:
    sys.path.insert(0, str(COMPRESS_ARC_DIR))

# agentic_repl is embedded as its own top-level package (agentic_repl/, a
# sibling of src/), so EMBED_ROOT itself -- not EMBED_ROOT/src -- needs to be
# on sys.path for `import agentic_repl` to resolve.
if (EMBED_ROOT / 'agentic_repl').is_dir() and str(EMBED_ROOT) not in sys.path:
    sys.path.insert(0, str(EMBED_ROOT))

# The agentic-REPL code-LLM's weights are pre-staged as a Kaggle Dataset
# (see agentic_repl/models/README.md), never downloaded live -- internet is
# disabled on scored reruns. Autodiscover the mounted .gguf file the same
# way the HRM checkpoint path is set explicitly above, so LlamaCppClient
# just reads MYTHOS_AGENTIC_MODEL_PATH without any further wiring.
if 'MYTHOS_AGENTIC_MODEL_PATH' not in os.environ:
    _kaggle_input = Path('/kaggle/input')
    if _kaggle_input.is_dir():
        # Confirmed directly (not assumed): private dataset inputs mount at
        # /kaggle/input/datasets/<owner>/<slug>/, not the flatter
        # /kaggle/input/<slug>/ this repo's HRM checkpoint paths assume --
        # check both rather than trust either blindly.
        _gguf_candidates = sorted(_kaggle_input.glob('*/*.gguf')) or sorted(
            _kaggle_input.glob('datasets/*/*/*.gguf')
        )
        if _gguf_candidates:
            os.environ['MYTHOS_AGENTIC_MODEL_PATH'] = str(_gguf_candidates[0])

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
                "Default mode is `pipeline` + `fallback`, with internet downloads disabled. "
                "It autodiscovers pre-staged Kaggle inputs when present and otherwise uses "
                "explicit fallback adapters for missing model stages."
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
                "# ARCHITECTURE pass: hyperparameter tuning alone plateaued (v29-v37: stable,\n"
                "# decreasing loss but 0 exact matches regardless of steps/rank/lr). Testing two\n"
                "# real architectural changes together -- Genie background-consistency loss (was\n"
                "# implemented in mythos.losses but never wired into any training loop) and wider\n"
                "# LoRA target modules (MLP layers, not just attention) -- against the evaluation\n"
                "# split's known solutions before spending the daily submission quota again.\n"
                "SPLIT = 'test'  # 'evaluation' was used for diagnostic runs against known solutions (v43-v49); 'test' is the real submission split\n"
                "# agentic_repl accuracy validation: point at the staged, solution-known public\n"
                "# ARC-AGI-2 dataset instead of the competition's own split. None = use DATA_DIR/SPLIT above.\n"
                "BENCHMARK_ARC_AGI_2_SPLIT = 'training'  # 'training' (1000 tasks) or 'evaluation' (120 tasks) or None\n"
                "BENCHMARK_DATA_DIR = Path('/kaggle/input/agentic-repl-arc-agi-2-data')\n"
                "os.environ.setdefault('MYTHOS_TTT_STEPS', '100')  # validated in v49: 2-view ensemble at steps=100 each beat steps=200 single-view at roughly matched total TTT compute\n"
                "os.environ.setdefault('MYTHOS_TTT_ENSEMBLE', '0,4')  # v49-validated: identity + mirror_horizontal views, voted -- real (modest) accuracy gain over single-view\n"
                "os.environ.setdefault('MYTHOS_TTT_NUM_AUG', '8')\n"
                "os.environ.setdefault('MYTHOS_TTT_RANK', '16')\n"
                "os.environ.setdefault('MYTHOS_TTT_GENIE_WEIGHT', '0.01')\n"
                "os.environ.setdefault('MYTHOS_TTT_LR', '1e-4')\n"
                "os.environ.setdefault('MYTHOS_TTT_BATCH_SIZE', '2')\n"
                "SOLVER_NAME = 'agentic_repl'  # 'pipeline', 'baseline', 'fixture', 'hrm', or 'agentic_repl'\n"
                "# BENCHMARK: agentic_repl against the real, solution-known ARC-AGI-2 training\n"
                "# split (BENCHMARK_ARC_AGI_2_SPLIT above), 30 tasks per pass -- v52 smoke test\n"
                "# (2 tasks) confirmed the model loads and the solver runs end to end on real\n"
                "# L4 hardware; this measures actual accuracy against mythos's own documented\n"
                "# symbolic-solver baseline (25/1000 training tasks). Remove/raise both this and\n"
                "# BENCHMARK_ARC_AGI_2_SPLIT=None for a real competition submission.\n"
                "os.environ.setdefault('MYTHOS_MAX_TASKS', '30')\n"
                "MODEL_MODE = 'fallback'  # 'fallback' or 'strict' (irrelevant for SOLVER_NAME='hrm', which bypasses ModelRegistry)\n"
                "# Kaggle competition reruns are internet-disabled, so the HRM repo + checkpoint\n"
                "# are pre-staged as a Kaggle Dataset (ankitdash24/hrm-arc2-checkpoint) instead of\n"
                "# downloaded live -- verified working end-to-end first with AUTO_DOWNLOAD+internet\n"
                "# in a dev run, then pinned here for the actually-submittable configuration.\n"
                "AUTO_DOWNLOAD_GIT_CODE = False\n"
                "AUTO_DOWNLOAD_HF_MODELS = False\n"
                "AUTO_DOWNLOAD_DIRECT_CHECKPOINTS = False\n"
                "AUTO_DISCOVER_MODELS = True\n"
                "os.environ.setdefault('HRM_REPO_DIR', '/kaggle/input/hrm-arc2-checkpoint/hrm-repo')\n"
                "os.environ.setdefault('HRM_CHECKPOINT_PATH', '/kaggle/input/hrm-arc2-checkpoint/hrm-checkpoint/checkpoint')\n"
                "OUTPUT_PATH = Path('/kaggle/working/submission.json')\n"
                "RUN_HRM_SMOKE = False\n\n"
                "# Training/checkpoint-producing stages. Keep all False for final rerun unless needed.\n"
                "RUN_TRAINING_STAGES = False\n"
                "TRAIN_JEPA_PROJECTION = False\n"
                "TRAIN_WORLD_MODEL = False\n"
                "RUN_TTT_SMOKE = False\n"
                "ENABLE_REAL_HRM_INFERENCE = True\n"
                "ENABLE_REAL_JEPA = False\n"
                "ENABLE_HRM_TEXT = False\n"
                "ENABLE_TTT_IN_PIPELINE = True  # drives MYTHOS_ENABLE_TTT for SOLVER_NAME='hrm' too, not just the pipeline solver\n"
                "CHECKPOINT_DIR = Path('/kaggle/working/mythos_checkpoints')\n"
                "IJEPA_PROJECTION_OUTPUT = CHECKPOINT_DIR / 'ijepa_projection.pt'\n"
                "WORLD_MODEL_OUTPUT = CHECKPOINT_DIR / 'world_model.pt'\n"
                "TTT_LORA_OUTPUT = CHECKPOINT_DIR / 'ttt_lora_smoke.pt'\n"
                "JEPA_PROJECTION_STEPS = 200\n"
                "WORLD_MODEL_STEPS = 300\n"
                "TTT_SMOKE_STEPS = 50\n\n"
                "os.environ['MYTHOS_ENABLE_REAL_HRM'] = '1' if ENABLE_REAL_HRM_INFERENCE else '0'\n"
                "os.environ['MYTHOS_ENABLE_REAL_JEPA'] = '1' if ENABLE_REAL_JEPA else '0'\n"
                "os.environ['MYTHOS_ENABLE_HRM_TEXT'] = '1' if ENABLE_HRM_TEXT else '0'\n"
                "os.environ['MYTHOS_ENABLE_TTT'] = '1' if ENABLE_TTT_IN_PIPELINE else '0'\n"
                "# HRM's own eval path logs to Weights & Biases; without this it can block on an\n"
                "# interactive API-key prompt in a non-interactive kernel and hang out the session.\n"
                "os.environ.setdefault('WANDB_MODE', 'offline')\n"
                "# HRM's public checkpoint was trained for 8-GPU distributed batches; keep eval batches\n"
                "# small since this pipeline calls HRM's eval loop with world_size=1 on Kaggle's GPU(s).\n"
                "os.environ.setdefault('HRM_GLOBAL_BATCH_SIZE', '32')\n\n"
                "# Verified public defaults looked up from official sources.\n"
                "os.environ.setdefault('HRM_GIT_REPO_URL', 'https://github.com/sapientinc/HRM.git')\n"
                "os.environ.setdefault('HRM_HF_REPO_ID', 'sapientinc/HRM-checkpoint-ARC-2')\n"
                "os.environ.setdefault('HRM_HF_CHECKPOINT_GLOB', 'checkpoint')\n"
                "os.environ.setdefault('IJEPA_HF_REPO_ID', 'facebook/ijepa_vith14_1k')\n"
                "os.environ.setdefault('IJEPA_HF_CHECKPOINT_GLOB', 'model.safetensors')\n"
                "os.environ.setdefault('HRM_TEXT_HF_REPO_ID', 'sapientinc/HRM-Text-1B')\n"
                "os.environ.setdefault('HRM_TEXT_HF_CHECKPOINT_GLOB', 'model.safetensors')\n\n"
                "# Components intentionally left unset because no verified public checkpoint ID was found.\n"
                "# Train/fine-tune these and add your own dataset/HF IDs when available:\n"
                "# - IJEPA_PROJECTION_HF_REPO_ID / IJEPA_PROJECTION_CHECKPOINT_PATH\n"
                "# - WORLD_MODEL_HF_REPO_ID / WORLD_MODEL_CHECKPOINT_PATH\n"
                "# - TTT_LORA_HF_REPO_ID / TTT_LORA_CHECKPOINT_PATH\n\n"
                "# For real transformers I-JEPA, stage the Hugging Face snapshot as a Kaggle Dataset\n"
                "# and set IJEPA_CHECKPOINT_PATH to any file inside that snapshot, or let autodiscovery find it.\n\n"
                "# Optional real-model inputs. Set these if you have additional public/private model repos.\n"
                "# os.environ['IJEPA_HF_REPO_ID'] = '<org-or-user>/<ijepa-model-repo>'\n"
                "# os.environ['IJEPA_HF_CHECKPOINT_GLOB'] = '*.pt'\n"
                "# os.environ['IJEPA_PROJECTION_HF_REPO_ID'] = '<org-or-user>/<projection-repo>'\n"
                "# os.environ['HRM_TEXT_HF_REPO_ID'] = '<org-or-user>/<hrm-text-model-repo>'\n"
                "# os.environ['WORLD_MODEL_HF_REPO_ID'] = '<org-or-user>/<world-model-repo>'\n"
                "# os.environ['TTT_LORA_HF_REPO_ID'] = '<org-or-user>/<lora-repo>'\n"
                "# os.environ['HRM_HF_REPO_ID'] = '<org-or-user>/<hrm-model-repo>'\n"
                "# os.environ['HRM_HF_CHECKPOINT_GLOB'] = '*.pt'\n\n"
                "# Or set explicit Kaggle input paths when internet/download is unavailable.\n"
                "# os.environ['IJEPA_CHECKPOINT_PATH'] = '/kaggle/input/<ijepa-hf-snapshot>/model.safetensors'\n"
                "# os.environ['IJEPA_PROJECTION_CHECKPOINT_PATH'] = '/kaggle/input/<projection>/ijepa_projection.pt'\n"
                "# os.environ['HRM_TEXT_REPO_DIR'] = '/kaggle/input/<hrm-text-code>/hrm-text'\n"
                "# os.environ['HRM_TEXT_CHECKPOINT_PATH'] = '/kaggle/input/<hrm-text-checkpoint>/checkpoint.pt'\n"
                "# os.environ['WORLD_MODEL_CHECKPOINT_PATH'] = '/kaggle/input/<world-model>/world_model.pt'\n"
                "# os.environ['TTT_LORA_CHECKPOINT_PATH'] = '/kaggle/input/<lora>/lora.pt'\n"
                "# os.environ['HRM_REPO_DIR'] = '/kaggle/input/<hrm-code>/HRM'\n"
                "# os.environ['HRM_CHECKPOINT_PATH'] = '/kaggle/input/<hrm-checkpoint>/checkpoint.pt'\n\n"
                "# HRM uses flash-attn in its attention path. For final offline reruns, pre-stage a wheel\n"
                "# built against Kaggle's CUDA/PyTorch image instead of building from source at submission time.\n\n"
                "print('DATA_DIR =', DATA_DIR)\n"
                "print('SPLIT =', SPLIT)\n"
                "print('SOLVER_NAME =', SOLVER_NAME)\n"
                "print('MODEL_MODE =', MODEL_MODE)\n"
                "print('AUTO_DOWNLOAD_GIT_CODE =', AUTO_DOWNLOAD_GIT_CODE)\n"
                "print('AUTO_DOWNLOAD_HF_MODELS =', AUTO_DOWNLOAD_HF_MODELS)\n"
                "print('AUTO_DOWNLOAD_DIRECT_CHECKPOINTS =', AUTO_DOWNLOAD_DIRECT_CHECKPOINTS)\n"
                "print('AUTO_DISCOVER_MODELS =', AUTO_DISCOVER_MODELS)\n"
                "print('OUTPUT_PATH =', OUTPUT_PATH)\n"
                "print('RUN_TRAINING_STAGES =', RUN_TRAINING_STAGES)\n"
                "print('ENABLE_REAL_HRM_INFERENCE =', ENABLE_REAL_HRM_INFERENCE)\n"
                "print('ENABLE_REAL_JEPA =', ENABLE_REAL_JEPA)\n"
                "print('ENABLE_HRM_TEXT =', ENABLE_HRM_TEXT)\n"
                "print('ENABLE_TTT_IN_PIPELINE =', ENABLE_TTT_IN_PIPELINE)\n"
                "print('MYTHOS_MAX_TASKS =', os.environ.get('MYTHOS_MAX_TASKS'))\n"
                "print('MYTHOS_TTT_STEPS =', os.environ.get('MYTHOS_TTT_STEPS'))\n"
                "print('MYTHOS_TTT_NUM_AUG =', os.environ.get('MYTHOS_TTT_NUM_AUG'))\n"
                "print('MYTHOS_TTT_RANK =', os.environ.get('MYTHOS_TTT_RANK'))\n"
                "print('MYTHOS_TTT_LR =', os.environ.get('MYTHOS_TTT_LR'))\n"
                "print('MYTHOS_TTT_BATCH_SIZE =', os.environ.get('MYTHOS_TTT_BATCH_SIZE'))\n"
                "print('MYTHOS_TTT_GENIE_WEIGHT =', os.environ.get('MYTHOS_TTT_GENIE_WEIGHT'))\n"
                "print('MYTHOS_TTT_ENSEMBLE =', os.environ.get('MYTHOS_TTT_ENSEMBLE'))\n"
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
                "from mythos.arc import load_solutions\n"
                "from mythos.metrics import score_files, score_submission_data\n"
                "from mythos.pipeline import PLAN_STAGE_ORDER\n"
                "from mythos.solvers.factory import make_solver\n"
                "from mythos.submission import load_submission, write_submission\n\n"
                "from mythos.training import (\n"
                "    JepaProjectionConfig,\n"
                "    WorldModelConfig,\n"
                "    run_ttt_lora_smoke,\n"
                "    train_jepa_projection,\n"
                "    train_world_model,\n"
                ")\n\n"
                "print('Imported mythos from:', mythos.__file__)\n"
                "print('PLAN_STAGE_ORDER =', ' -> '.join(PLAN_STAGE_ORDER))\n"
            ),
            _markdown(
                "## 4. Load ARC Data\n\n"
                "If BENCHMARK_ARC_AGI_2_SPLIT is set, load the staged public ARC-AGI-2 "
                "data (with known solutions) instead of the competition's own split -- "
                "for measuring real accuracy, not producing a submission."
            ),
            _code(
                "if BENCHMARK_ARC_AGI_2_SPLIT:\n"
                "    _bench_dir = BENCHMARK_DATA_DIR\n"
                "    if not _bench_dir.is_dir():\n"
                "        _nested = sorted(Path('/kaggle/input').glob('datasets/*/agentic-repl-arc-agi-2-data'))\n"
                "        if _nested:\n"
                "            _bench_dir = _nested[0]\n"
                "    challenge_path = _bench_dir / f'{BENCHMARK_ARC_AGI_2_SPLIT}_challenges.json'\n"
                "    solution_path = _bench_dir / f'{BENCHMARK_ARC_AGI_2_SPLIT}_solutions.json'\n"
                "else:\n"
                "    challenge_path = resolve_challenge_path(DATA_DIR, SPLIT)\n"
                "    solution_path = resolve_solution_path(DATA_DIR, SPLIT)\n"
                "tasks = load_challenges(challenge_path)\n\n"
                "_max_tasks = os.environ.get('MYTHOS_MAX_TASKS')\n"
                "if _max_tasks:\n"
                "    tasks = dict(list(tasks.items())[: int(_max_tasks)])\n"
                "    print(f'MYTHOS_MAX_TASKS set: truncated to {len(tasks)} task(s) for a fast iteration pass')\n\n"
                "train_examples = sum(len(task.train) for task in tasks.values())\n"
                "test_items = sum(len(task.test) for task in tasks.values())\n\n"
                "print('challenge_path =', challenge_path)\n"
                "print('solution_path =', solution_path)\n"
                "print('tasks =', len(tasks))\n"
                "print('train_examples =', train_examples)\n"
                "print('test_items =', test_items)\n"
            ),
            _markdown(
                "## 4b. Install agentic_repl Runtime Dependencies (offline wheel)\n\n"
                "`AgenticReplSolver`'s real backend (`LlamaCppClient`) needs "
                "`llama-cpp-python`, which Kaggle doesn't preinstall. A live `pip "
                "install` doesn't work here at all: Kaggle's L4 sessions hard-enforce "
                "no internet regardless of kernel-metadata.json's enable_internet "
                "setting (confirmed directly -- a live install attempt failed with DNS "
                "resolution errors even with enable_internet=true). A "
                "transformers-native GGUF loader was considered as an install-free "
                "alternative, but transformers doesn't support the qwen3moe "
                "architecture for GGUF loading yet.\n\n"
                "Fix: a prebuilt CUDA wheel (`llama_cpp_python-0.3.31-py3-none-"
                "manylinux_2_35_x86_64.whl`, cu125 -- CUDA is runtime-backward-"
                "compatible, so this runs fine against Kaggle's newer CUDA), staged as "
                "the small Kaggle Dataset `agentic-repl-llama-cpp-wheel` and installed "
                "offline via `pip install --no-index`. This has to run before "
                "`make_solver()` below, not after like the HRM dependency cell, since "
                "AgenticReplSolver's constructor loads the model eagerly (unlike "
                "HRMSolver, which defers its heavy imports)."
            ),
            _code(
                "agentic_setup_ok = True\n\n"
                "if SOLVER_NAME == 'agentic_repl':\n"
                "    import subprocess\n"
                "    try:\n"
                "        import llama_cpp\n"
                "        print('llama_cpp already importable:', llama_cpp.__file__)\n"
                "    except ImportError:\n"
                "        wheel_dir_candidates = (\n"
                "            list(Path('/kaggle/input/agentic-repl-llama-cpp-wheel').glob('*'))\n"
                "            + list(Path('/kaggle/input').glob('datasets/*/agentic-repl-llama-cpp-wheel/*'))\n"
                "        )\n"
                "        wheel_files = [p for p in wheel_dir_candidates if p.suffix == '.whl']\n"
                "        print('found wheel files:', wheel_files)\n"
                "        if not wheel_files:\n"
                "            print('WARNING: no staged llama-cpp-python wheel found under /kaggle/input')\n"
                "            agentic_setup_ok = False\n"
                "        else:\n"
                "            install = subprocess.run(\n"
                "                [sys.executable, '-m', 'pip', 'install', '-v', '--no-index',\n"
                "                 '--find-links', str(wheel_files[0].parent), 'llama-cpp-python'],\n"
                "                capture_output=True, text=True, timeout=300,\n"
                "            )\n"
                "            print('--- pip stdout (tail) ---')\n"
                "            print(install.stdout[-3000:])\n"
                "            if install.returncode != 0:\n"
                "                print('--- pip stderr (tail) ---')\n"
                "                print(install.stderr[-3000:])\n"
                "                agentic_setup_ok = False\n"
                "            else:\n"
                "                try:\n"
                "                    import llama_cpp\n"
                "                    print('llama_cpp installed and importable:', llama_cpp.__file__)\n"
                "                except ImportError as exc:\n"
                "                    print('WARNING: pip install succeeded but import still fails:', repr(exc))\n"
                "                    agentic_setup_ok = False\n\n"
                "    if not agentic_setup_ok:\n"
                "        print('Disabling agentic_repl for this run; falling back to SOLVER_NAME=pipeline.')\n"
                "        SOLVER_NAME = 'pipeline'\n\n"
                "print('agentic_setup_ok =', agentic_setup_ok)\n"
                "print('SOLVER_NAME (post agentic_repl setup) =', SOLVER_NAME)\n"
            ),
            _markdown("## 5. Download Models, Optionally Train Local Stages, Then Load Solver"),
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
                "training_results = {}\n"
                "if RUN_TRAINING_STAGES:\n"
                "    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)\n"
                "    if TRAIN_JEPA_PROJECTION:\n"
                "        projection_result = train_jepa_projection(\n"
                "            tasks.values(),\n"
                "            checkpoint_path=IJEPA_PROJECTION_OUTPUT,\n"
                "            config=JepaProjectionConfig(input_dim=1280, output_dim=768),\n"
                "            steps=JEPA_PROJECTION_STEPS,\n"
                "            lr=1e-3,\n"
                "        )\n"
                "        os.environ['IJEPA_PROJECTION_CHECKPOINT_PATH'] = str(IJEPA_PROJECTION_OUTPUT)\n"
                "        training_results['jepa_projection'] = projection_result.to_dict()\n"
                "    if TRAIN_WORLD_MODEL:\n"
                "        world_result = train_world_model(\n"
                "            tasks.values(),\n"
                "            checkpoint_path=WORLD_MODEL_OUTPUT,\n"
                "            config=WorldModelConfig(z_dim=768, rule_dim=4, hidden_dim=3072),\n"
                "            steps=WORLD_MODEL_STEPS,\n"
                "            lr=1e-3,\n"
                "        )\n"
                "        os.environ['WORLD_MODEL_CHECKPOINT_PATH'] = str(WORLD_MODEL_OUTPUT)\n"
                "        training_results['world_model'] = world_result.to_dict()\n"
                "    if RUN_TTT_SMOKE:\n"
                "        ttt_result = run_ttt_lora_smoke(\n"
                "            rank=16,\n"
                "            steps=TTT_SMOKE_STEPS,\n"
                "            checkpoint_path=TTT_LORA_OUTPUT,\n"
                "        )\n"
                "        os.environ['TTT_LORA_CHECKPOINT_PATH'] = str(TTT_LORA_OUTPUT)\n"
                "        training_results['ttt_lora_smoke'] = ttt_result.to_dict()\n"
                "else:\n"
                "    print('RUN_TRAINING_STAGES is False; using downloaded/discovered checkpoints only.')\n\n"
                "if training_results:\n"
                "    print('training_results =')\n"
                "    print(json.dumps(training_results, indent=2, sort_keys=True))\n\n"
                "solver = make_solver(SOLVER_NAME, model_mode=MODEL_MODE)\n"
                "print('Loaded solver:', solver.__class__.__name__)\n\n"
                "if hasattr(solver, 'pipeline'):\n"
                "    print('model_registry =')\n"
                "    print(json.dumps(solver.pipeline.model_registry.summary(), indent=2))\n"
            ),
            _markdown(
                "## 5b. Install HRM Runtime Dependencies\n\n"
                "The external HRM repo needs packages Kaggle's base image doesn't ship "
                "(`flash-attn` in particular has no fallback attention path). This cell "
                "installs them best-effort; if anything critical fails, it disables real "
                "HRM inference and falls back to the deterministic pipeline solver rather "
                "than risk a hung or crashed GPU session."
            ),
            _code(
                "hrm_setup_ok = True\n"
                "hrm_setup_log = []\n\n"
                "if SOLVER_NAME == 'hrm':\n"
                "    import subprocess\n"
                "    import torch as _torch_probe\n\n"
                "    print('torch =', _torch_probe.__version__, '| cuda =', _torch_probe.version.cuda,\n"
                "          '| cuda_available =', _torch_probe.cuda.is_available(),\n"
                "          '| device_count =', _torch_probe.cuda.device_count())\n"
                "    if _torch_probe.cuda.is_available():\n"
                "        print('gpu_name =', _torch_probe.cuda.get_device_name(0))\n\n"
                "    hrm_dependencies = [\n"
                "        'einops', 'tqdm', 'coolname', 'pydantic', 'argdantic', 'wandb',\n"
                "        'omegaconf', 'hydra-core', 'huggingface_hub', 'pyyaml',\n"
                "        # adam-atan2's setup.py uses the legacy setup_requires=['setuptools_scm']\n"
                "        # auto-fetch path, which pulls a setuptools_scm version whose own\n"
                "        # vcs_versioning dependency doesn't resolve through that legacy mechanism.\n"
                "        # Installing setuptools_scm normally first lets adam-atan2's build find it\n"
                "        # already satisfied and skip the broken auto-fetch.\n"
                "        'setuptools_scm', 'adam-atan2',\n"
                "    ]\n"
                "    # Kaggle's scored rerun is internet-disabled, so PyPI is unreachable too --\n"
                "    # confirmed by a real run: pydantic/pyyaml/etc already ship in Kaggle's base\n"
                "    # image (installs no-op fine offline), but coolname/argdantic/hydra-core/\n"
                "    # setuptools_scm/adam-atan2 don't and failed outright with no internet. Their\n"
                "    # wheels (pure-python only; platform-specific ones like pydantic-core were\n"
                "    # deliberately excluded since those packages are already present) are\n"
                "    # pre-staged in the same dataset as the checkpoint.\n"
                "    wheelhouse = Path('/kaggle/input/hrm-arc2-checkpoint/wheels')\n"
                "    print('wheelhouse =', wheelhouse, '| is_dir =', wheelhouse.is_dir())\n"
                "    hrm_dataset_root = Path('/kaggle/input/hrm-arc2-checkpoint')\n"
                "    if hrm_dataset_root.is_dir():\n"
                "        print('hrm-arc2-checkpoint dataset contents:', sorted(p.name for p in hrm_dataset_root.iterdir()))\n"
                "    else:\n"
                "        print('hrm-arc2-checkpoint dataset root not found; listing /kaggle/input:')\n"
                "        print(sorted(p.name for p in Path('/kaggle/input').iterdir()) if Path('/kaggle/input').is_dir() else 'no /kaggle/input')\n"
                "    offline_install_args = ['--no-index', '--find-links', str(wheelhouse)] if wheelhouse.is_dir() else []\n\n"
                "    def _resolve_install_target(package):\n"
                "        # Resolve to the exact staged .whl instead of relying on pip's --find-links\n"
                "        # name-based matching: confirmed on a real run that Kaggle's pip 24.1.2 fails\n"
                "        # to match 'adam-atan2' against a staged sdist by name. The glob is restricted\n"
                "        # to *.whl files specifically -- Kaggle auto-extracts uploaded .tar.gz archives\n"
                "        # into a same-named bare directory (confirmed: an old 'adam_atan2-0.0.3/' dir\n"
                "        # from a superseded dataset version was still present and, unfiltered, sorted\n"
                "        # ahead of the real wheel and got picked instead), so only .whl is ever staged.\n"
                "        if not wheelhouse.is_dir():\n"
                "            return package\n"
                "        normalized = package.replace('-', '_')\n"
                "        candidates = sorted(wheelhouse.glob(f'{normalized}-*.whl')) or sorted(wheelhouse.glob(f'{package}-*.whl'))\n"
                "        return str(candidates[0]) if candidates else package\n\n"
                "    failed_packages = []\n"
                "    for package in hrm_dependencies:\n"
                "        # Install one at a time: a single bad package must not take down the\n"
                "        # whole batch and hide which of the others would have installed fine.\n"
                "        try:\n"
                "            install = subprocess.run(\n"
                "                # -v: pip swallows the build subprocess's own traceback by default\n"
                "                # (even without -q) and only shows its own generic wrapper error;\n"
                "                # -v is what actually surfaces why a legacy setup.py build failed.\n"
                "                [sys.executable, '-m', 'pip', 'install', '-v', *offline_install_args, _resolve_install_target(package)],\n"
                "                capture_output=True, text=True, timeout=300,\n"
                "            )\n"
                "            hrm_setup_log.append({'step': f'pip_install:{package}', 'returncode': install.returncode})\n"
                "            if install.returncode != 0:\n"
                "                failed_packages.append(package)\n"
                "                print(f'WARNING: failed to install {package}:')\n"
                "                print('--- stdout (tail) ---')\n"
                "                print(install.stdout[-4000:])\n"
                "                print('--- stderr (tail) ---')\n"
                "                print(install.stderr[-4000:])\n"
                "        except subprocess.TimeoutExpired:\n"
                "            failed_packages.append(package)\n"
                "            hrm_setup_log.append({'step': f'pip_install:{package}', 'error': 'timed out after 300s'})\n"
                "            print(f'WARNING: installing {package} timed out after 300s')\n"
                "    if failed_packages:\n"
                "        hrm_setup_ok = False\n"
                "        print('WARNING: HRM dependency install failed for:', failed_packages)\n\n"
                "    if hrm_setup_ok:\n"
                "        # flash-attn has no prebuilt wheel matching Kaggle's exact torch build and a\n"
                "        # from-source compile was observed taking 100+ minutes without finishing (nvcc\n"
                "        # compiling many CUDA template instantiations, no fast path available). HRM's\n"
                "        # Attention.forward() only calls flash_attn_func(q=, k=, v=, causal=) with plain\n"
                "        # [batch, seq_len, heads, head_dim] tensors -- torch's own built-in\n"
                "        # scaled_dot_product_attention implements the same math and ships with the stock\n"
                "        # torch Kaggle already has installed, so no extra install/compile is needed at all.\n"
                "        import types\n\n"
                "        def _flash_attn_func(q, k, v, causal=False, softmax_scale=None, dropout_p=0.0, **_ignored):\n"
                "            q_, k_, v_ = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)\n"
                "            num_heads, num_kv_heads = q_.shape[1], k_.shape[1]\n"
                "            if num_kv_heads and num_kv_heads != num_heads:\n"
                "                repeat = num_heads // num_kv_heads\n"
                "                k_ = k_.repeat_interleave(repeat, dim=1)\n"
                "                v_ = v_.repeat_interleave(repeat, dim=1)\n"
                "            out = _torch_probe.nn.functional.scaled_dot_product_attention(\n"
                "                q_, k_, v_, dropout_p=dropout_p, is_causal=causal, scale=softmax_scale,\n"
                "            )\n"
                "            # .contiguous(): .transpose() is a view (stride swap only); real\n"
                "            # flash_attn_func returns memory already laid out this way, and the\n"
                "            # model's own code does a plain .view() right after this call, which\n"
                "            # requires contiguous memory (confirmed failure: 'Cannot view a tensor\n"
                "            # with shape ... and strides ...' -- that's the exact non-contiguous\n"
                "            # symptom -- .reshape() would also work but .contiguous() matches what\n"
                "            # the real function actually hands back).\n"
                "            return out.transpose(1, 2).contiguous()\n\n"
                "        _flash_attn_shim = types.ModuleType('flash_attn')\n"
                "        _flash_attn_shim.flash_attn_func = _flash_attn_func\n"
                "        sys.modules['flash_attn'] = _flash_attn_shim\n"
                "        hrm_setup_log.append({'step': 'flash_attn_shim', 'note': 'using torch.nn.functional.scaled_dot_product_attention, no compile'})\n"
                "        print('Installed flash_attn shim backed by scaled_dot_product_attention (no compile needed)')\n\n"
                "        # adam_atan2's compiled CUDA/C++ backend (adam_atan2_backend) is only built\n"
                "        # when torch is present at pip-build time; our offline wheel was built without\n"
                "        # torch available, so it's missing and 'import pretrain' fails outright (it\n"
                "        # imports AdamATan2 unconditionally to build the optimizer in init_train_state,\n"
                "        # even though .step() is never called in this eval-only flow). Provide a\n"
                "        # faithful pure-PyTorch implementation of the same fused update instead of a\n"
                "        # whole extra Kaggle round-trip just to compile one C extension.\n"
                "        def _adam_atan2_cuda_impl_(params, grads, exp_avgs, exp_avg_sqs, state_steps, lr, beta1, beta2, weight_decay):\n"
                "            for param, grad, exp_avg, exp_avg_sq, step in zip(params, grads, exp_avgs, exp_avg_sqs, state_steps):\n"
                "                if weight_decay != 0:\n"
                "                    param.mul_(1 - lr * weight_decay)\n"
                "                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)\n"
                "                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)\n"
                "                bias_correction1 = 1 - beta1 ** step.item()\n"
                "                bias_correction2 = 1 - beta2 ** step.item()\n"
                "                numerator = exp_avg / bias_correction1\n"
                "                denominator = (exp_avg_sq / bias_correction2).sqrt()\n"
                "                param.add_(_torch_probe.atan2(numerator, denominator), alpha=-lr)\n\n"
                "        _adam_atan2_backend_shim = types.ModuleType('adam_atan2_backend')\n"
                "        _adam_atan2_backend_shim.adam_atan2_cuda_impl_ = _adam_atan2_cuda_impl_\n"
                "        sys.modules['adam_atan2_backend'] = _adam_atan2_backend_shim\n"
                "        hrm_setup_log.append({'step': 'adam_atan2_backend_shim', 'note': 'pure-PyTorch AdamATan2 update, no compile'})\n"
                "        print('Installed adam_atan2_backend shim (pure PyTorch, no compile needed)')\n\n"
                "    if not hrm_setup_ok:\n"
                "        print('Disabling real HRM inference for this run; falling back to SOLVER_NAME=pipeline.')\n"
                "        SOLVER_NAME = 'pipeline'\n"
                "        ENABLE_REAL_HRM_INFERENCE = False\n"
                "        os.environ['MYTHOS_ENABLE_REAL_HRM'] = '0'\n"
                "        solver = make_solver(SOLVER_NAME, model_mode=MODEL_MODE)\n\n"
                "print('hrm_setup_ok =', hrm_setup_ok)\n"
                "print('hrm_setup_log =', json.dumps(hrm_setup_log, indent=2))\n"
                "print('SOLVER_NAME (post-setup) =', SOLVER_NAME)\n"
            ),
            _markdown("## 6. Run Plan-Aligned Pipeline"),
            _code(
                "from mythos.arc import copy_grid\n"
                "from mythos.solvers.base import make_prediction\n"
                "from mythos.solvers.baseline import BaselineSolver\n\n"
                "def solve_with_fallback(solver, fallback_solver, task):\n"
                "    # A Kaggle rerun must always produce a submission.json; one task raising\n"
                "    # must never abort the loop and discard every already-solved prediction.\n"
                "    try:\n"
                "        return solver.solve(task)\n"
                "    except Exception as exc:\n"
                "        print(f'WARNING: {task.id} failed with {solver.__class__.__name__}: {exc!r}; using baseline fallback')\n"
                "    try:\n"
                "        return fallback_solver.solve(task)\n"
                "    except Exception as exc:\n"
                "        print(f'WARNING: {task.id} baseline fallback also failed: {exc!r}; using trivial prediction')\n"
                "    attempts = [(copy_grid(example.input), [[0]]) for example in task.test]\n"
                "    return make_prediction(task, attempts)\n\n"
                "started = time.perf_counter()\n"
                "predictions = []\n"
                "fallback_solver = solver if isinstance(solver, BaselineSolver) else BaselineSolver()\n\n"
                "if SOLVER_NAME == 'hrm':\n"
                "    from mythos.solvers.hrm import HRMEnvironment, HRMInferenceRunner, HRMTTTRunner, TTTConfig\n"
                "    from mythos.solvers.symbolic import SymbolicSolver\n"
                "    from mythos.kaggle_run import parse_ensemble_transforms, solve_symbolic_first\n"
                "    symbolic_predictions, remaining_tasks = solve_symbolic_first(list(tasks.values()))\n"
                "    print(f'symbolic solver: {len(symbolic_predictions)}/{len(tasks)} tasks solved with a train-verified rule; {len(remaining_tasks)} sent to HRM')\n"
                "    try:\n"
                "        env = HRMEnvironment.from_env()\n"
                "        env.validate(require_cuda=True)\n"
                "        if os.environ.get('MYTHOS_ENABLE_TTT') == '1':\n"
                "            hrm_runner = HRMTTTRunner(\n"
                "                env,\n"
                "                ttt=TTTConfig(\n"
                "                    rank=int(os.environ.get('MYTHOS_TTT_RANK', '16')),\n"
                "                    steps=int(os.environ.get('MYTHOS_TTT_STEPS', '20')),\n"
                "                    lr=float(os.environ.get('MYTHOS_TTT_LR', '1e-3')),\n"
                "                    batch_size=int(os.environ.get('MYTHOS_TTT_BATCH_SIZE', '2')),\n"
                "                    genie_weight=float(os.environ.get('MYTHOS_TTT_GENIE_WEIGHT', '0.1')),\n"
                "                    ensemble_transforms=parse_ensemble_transforms(os.environ.get('MYTHOS_TTT_ENSEMBLE', '0')),\n"
                "                ),\n"
                "                num_aug=int(os.environ.get('MYTHOS_TTT_NUM_AUG', '0')),\n"
                "            )\n"
                "        else:\n"
                "            hrm_runner = HRMInferenceRunner(env)\n"
                "        hrm_predictions = hrm_runner.solve_tasks(remaining_tasks) if remaining_tasks else []\n"
                "        print(f'Batched HRM solved {len(hrm_predictions)} task(s)')\n"
                "    except Exception as exc:\n"
                "        import traceback\n"
                "        print(f'WARNING: HRM batch run failed: {exc!r}; using baseline fallback for all tasks')\n"
                "        print('--- full traceback ---')\n"
                "        traceback.print_exc()\n"
                "        if hasattr(exc, 'stdout') and exc.stdout:\n"
                "            print('--- subprocess stdout (tail) ---')\n"
                "            print(exc.stdout[-4000:])\n"
                "        if hasattr(exc, 'stderr') and exc.stderr:\n"
                "            print('--- subprocess stderr (tail) ---')\n"
                "            print(exc.stderr[-4000:])\n"
                "        hrm_predictions = [fallback_solver.solve(task) for task in remaining_tasks]\n"
                "    hrm_predictions_by_id = {prediction.task_id: prediction for prediction in hrm_predictions}\n"
                "    predictions = [symbolic_predictions.get(task.id) or hrm_predictions_by_id[task.id] for task in tasks.values()]\n"
                "else:\n"
                "    for index, task in enumerate(tasks.values(), start=1):\n"
                "        prediction = solve_with_fallback(solver, fallback_solver, task)\n"
                "        predictions.append(prediction)\n"
                "        if index <= 3 or index == len(tasks):\n"
                "            print(f'{index}/{len(tasks)} solved: {task.id}')\n\n"
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
                "    if _max_tasks:\n"
                "        # score_files requires every solved task to be present (correct for a\n"
                "        # real full run); a MYTHOS_MAX_TASKS test run only solved a subset, so\n"
                "        # score just that subset instead of letting it raise on the rest.\n"
                "        all_solutions = load_solutions(solution_path)\n"
                "        partial_solutions = {task_id: all_solutions[task_id] for task_id in submission if task_id in all_solutions}\n"
                "        score = score_submission_data(submission, partial_solutions)\n"
                "        print(f'score (partial: {len(partial_solutions)}/{len(all_solutions)} tasks, MYTHOS_MAX_TASKS set) =')\n"
                "    else:\n"
                "        score = score_files(str(OUTPUT_PATH), str(solution_path))\n"
                "        print('score =')\n"
                "    print(json.dumps(score.to_dict(), indent=2, sort_keys=True))\n\n"
                "    # DIAGNOSTIC: print predicted grids next to the true answer for two\n"
                "    # representative tasks, to see the real failure mode directly instead of\n"
                "    # guessing from aggregate metrics alone (cell accuracy alone can't say\n"
                "    # whether it's a wrong output shape, systematically wrong colors, or\n"
                "    # something else). One task has train output shapes that disagree\n"
                "    # (output_shape_hint defers to the model's own EOS markers); one has\n"
                "    # train output shapes that all agree (shape is never in question, so any\n"
                "    # remaining failure is purely about predicted grid content).\n"
                "    solutions_for_diagnostic = partial_solutions if _max_tasks else load_solutions(solution_path)\n"
                "    candidate_ids = [tid for tid in submission if tid in solutions_for_diagnostic]\n"
                "    def _train_shapes_agree(task_id):\n"
                "        shapes = {(len(ex.output), len(ex.output[0])) for ex in tasks[task_id].train if ex.output is not None}\n"
                "        return len(shapes) == 1\n"
                "    diagnostic_task_ids = []\n"
                "    variable_shape_id = next((tid for tid in candidate_ids if not _train_shapes_agree(tid)), None)\n"
                "    fixed_shape_id = next((tid for tid in candidate_ids if _train_shapes_agree(tid)), None)\n"
                "    if variable_shape_id is not None:\n"
                "        diagnostic_task_ids.append(('variable-train-shape', variable_shape_id))\n"
                "    if fixed_shape_id is not None:\n"
                "        diagnostic_task_ids.append(('fixed-train-shape', fixed_shape_id))\n"
                "    for label, diagnostic_task_id in diagnostic_task_ids:\n"
                "        truth = solutions_for_diagnostic[diagnostic_task_id]\n"
                "        preds_for_task = submission[diagnostic_task_id]\n"
                "        source_task = tasks[diagnostic_task_id]\n"
                "        print(f'--- diagnostic ({label}): task {diagnostic_task_id} ---')\n"
                "        for demo_index, example in enumerate(source_task.train):\n"
                "            print(f'train[{demo_index}] input =', example.input)\n"
                "            print(f'train[{demo_index}] output =', example.output)\n"
                "        for item_index, (prediction, truth_grid) in enumerate(zip(preds_for_task, truth)):\n"
                "            print(f'test[{item_index}] input =', source_task.test[item_index].input)\n"
                "            print(f'test[{item_index}] attempt_1 =', prediction.attempt_1)\n"
                "            print(f'test[{item_index}] attempt_2 =', prediction.attempt_2)\n"
                "            print(f'test[{item_index}] true output =', truth_grid)\n"
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

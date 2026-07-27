"""Build a minimal standalone notebook to measure CompressARC's real per-step
timing on a Kaggle T4 -- a diagnostic experiment, separate from the main HRM
pipeline notebook/kernel so it can't disturb that kernel's state.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src" / "mythos"
THIRD_PARTY_ROOT = ROOT / "third_party"
OUT_DIR = ROOT / "compress_arc_experiment"
OUTPUT = OUT_DIR / "compress_arc_timing_test.ipynb"


def _source_lines(source: str) -> list[str]:
    return source.splitlines(keepends=True)


def _markdown(source: str) -> dict[str, object]:
    return {"cell_type": "markdown", "metadata": {}, "source": _source_lines(source)}


def _code(source: str) -> dict[str, object]:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": _source_lines(source)}


def _embedded_files() -> dict[str, str]:
    files: dict[str, str] = {}
    for path in sorted(SRC_ROOT.rglob("*.py")):
        files[path.relative_to(ROOT).as_posix()] = path.read_text(encoding="utf-8")
    for path in sorted(THIRD_PARTY_ROOT.rglob("*")):
        if path.is_file() and (path.suffix in (".py", ".md") or path.name == "LICENSE"):
            files[path.relative_to(ROOT).as_posix()] = path.read_text(encoding="utf-8")
    return files


def build() -> None:
    files = _embedded_files()
    bootstrap = f"""from pathlib import Path
import sys

EMBEDDED_FILES = {files!r}

EMBED_ROOT = Path('/kaggle/working/compress_arc_embedded')
for relative_path, content in EMBEDDED_FILES.items():
    path = EMBED_ROOT / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding='utf-8')

SRC_DIR = EMBED_ROOT / 'src'
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
COMPRESS_ARC_DIR = EMBED_ROOT / 'third_party' / 'compress_arc'
if str(COMPRESS_ARC_DIR) not in sys.path:
    sys.path.insert(0, str(COMPRESS_ARC_DIR))

print('Embedded files:', len(EMBEDDED_FILES))
"""

    experiment = """import time
import torch

from mythos.arc import attach_solutions, load_challenges, load_solutions
from mythos.kaggle_run import resolve_challenge_path, resolve_solution_path

print('CUDA available:', torch.cuda.is_available())
print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')
print('Total VRAM (GB):', torch.cuda.get_device_properties(0).total_memory / 1e9)

DATA_DIR = '/kaggle/input/competitions/arc-prize-2026-arc-agi-2'
challenge_path = resolve_challenge_path(DATA_DIR, 'evaluation')
solution_path = resolve_solution_path(DATA_DIR, 'evaluation')
challenges = load_challenges(challenge_path)
solutions = load_solutions(solution_path)
tasks = attach_solutions(challenges, solutions)

# Per-task GPU memory measurement, mirroring CompressARC's own
# parallel_train.py (\"we run 2 steps of every puzzle to determine how much
# memory each puzzle uses\") -- memory footprint stabilizes fast, so this
# needs far fewer steps than the earlier accuracy/timing experiment. Sample
# a spread of real tasks so grid-size effects on memory show up.
TASK_IDS = list(tasks)[:20]
MEMORY_PROBE_STEPS = 3

import preprocessing
import arc_compressor
import train as compress_arc_train
import solution_selection

results = []
for task_id in TASK_IDS:
    task = tasks[task_id]
    problem = {
        'train': [{'input': ex.input, 'output': ex.output} for ex in task.train],
        'test': [{'input': ex.input} for ex in task.test],
    }
    torch.manual_seed(0)
    try:
        compress_task = preprocessing.Task(task_id, problem, None)
        torch.cuda.reset_peak_memory_stats()
        model = arc_compressor.ARCCompressor(compress_task)
        optimizer = torch.optim.Adam(model.weights_list, lr=0.01, betas=(0.5, 0.9))
        logger = solution_selection.Logger(compress_task)
        logger.solution_most_frequent = tuple(((0, 0), (0, 0)) for _ in range(compress_task.n_test))
        logger.solution_second_most_frequent = tuple(((0, 0), (0, 0)) for _ in range(compress_task.n_test))
        t0 = time.perf_counter()
        for step in range(MEMORY_PROBE_STEPS):
            compress_arc_train.take_step(compress_task, model, optimizer, step, logger)
        elapsed = time.perf_counter() - t0
        peak_mem_gb = torch.cuda.max_memory_allocated() / 1e9
        grid_shape = (compress_task.n_x, compress_task.n_y, compress_task.n_examples)
        results.append((task_id, peak_mem_gb, elapsed / MEMORY_PROBE_STEPS, grid_shape))
        print(f'{task_id}: peak_mem={peak_mem_gb:.3f}GB, {elapsed/MEMORY_PROBE_STEPS*1000:.0f}ms/step, (n_x,n_y,n_examples)={grid_shape}')
        del model, optimizer, logger, compress_task
        torch.cuda.empty_cache()
    except Exception as exc:
        print(f'{task_id}: FAILED: {exc!r}')

mems = [r[1] for r in results]
if mems:
    print()
    print(f'n={len(mems)}, min={min(mems):.3f}GB, max={max(mems):.3f}GB, mean={sum(mems)/len(mems):.3f}GB')
    total_vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    safety_margin_gb = 2.0
    usable_gb = total_vram_gb - safety_margin_gb
    for label, mem in [('max (worst-case)', max(mems)), ('mean', sum(mems)/len(mems))]:
        concurrency = int(usable_gb // mem) if mem > 0 else 0
        print(f'Estimated safe concurrency using {label} per-task memory ({mem:.3f}GB): {concurrency} tasks in parallel on {total_vram_gb:.1f}GB T4')
"""

    notebook = {
        "cells": [
            _markdown(
                "# CompressARC Timing Experiment\n\n"
                "Diagnostic-only: measures real per-step wall-clock time on a Kaggle T4 for "
                "the vendored CompressARC solver (see third_party/compress_arc/NOTICE.md), "
                "against a small fixed set of real evaluation-split tasks with known solutions. "
                "Not a submission notebook."
            ),
            _code(bootstrap),
            _code(experiment),
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    print(f"Wrote {OUTPUT}")
    print(f"Embedded files: {len(files)}")


if __name__ == "__main__":
    build()

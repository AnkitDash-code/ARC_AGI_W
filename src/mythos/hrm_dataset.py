"""Dataset-preparation glue for the external HRM checkout."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Iterable

from mythos.arc import ArcTask, ArcValidationError, Grid, require_test_outputs


def default_run_dir() -> Path:
    import os

    # Must be absolute: build_hrm_dataset() invokes HRM's dataset builder with
    # cwd=repo_dir (the external HRM checkout), while the caller that later reads
    # the built dataset back (HRMInferenceRunner._run_external_evaluate, running in
    # the notebook/CLI's own process) has a different cwd. A relative path here
    # resolves to two different real locations across those processes -- confirmed
    # by a real run: the builder wrote train/dataset.json under repo_dir, but the
    # dataloader looked for it relative to the notebook's cwd and got FileNotFoundError.
    return Path(os.environ.get("MYTHOS_RUN_DIR", "runs")).resolve()


def prepare_hrm_raw_dataset(
    tasks: Iterable[ArcTask],
    output_dir: str | Path,
    *,
    allow_dummy_test_outputs: bool = False,
) -> Path:
    """Write tasks into the directory shape HRM's ARC dataset builder expects."""

    task_list = list(tasks)
    if not allow_dummy_test_outputs:
        require_test_outputs(task_list)

    raw_data_dir = Path(output_dir)
    eval_dir = raw_data_dir / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)

    for task in task_list:
        raw_task = {
            "train": [
                {"input": example.input, "output": example.output}
                for example in task.train
            ],
            "test": [
                {
                    "input": example.input,
                    "output": example.output
                    if example.output is not None
                    else _dummy_output_like(example.input),
                }
                for example in task.test
            ],
        }
        with (eval_dir / f"{task.id}.json").open("w", encoding="utf-8") as handle:
            json.dump(raw_task, handle, indent=2)
            handle.write("\n")
    return raw_data_dir


def _dummy_output_like(grid: Grid) -> Grid:
    return [[0 for _ in row] for row in grid]


def build_hrm_dataset(
    *,
    hrm_repo_dir: str | Path,
    raw_data_dir: str | Path,
    output_dir: str | Path,
    num_aug: int = 0,
) -> subprocess.CompletedProcess[str]:
    """Invoke HRM's own ARC dataset builder against a prepared raw-data directory."""

    repo_dir = Path(hrm_repo_dir)
    script = repo_dir / "dataset" / "build_arc_dataset.py"
    if not script.exists():
        raise ArcValidationError(f"HRM dataset builder not found: {script}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # build_arc_dataset.py's DataProcessConfig.dataset_dirs is a List[str] Pydantic
    # field defaulting to ["dataset/raw-data/ARC-AGI/data", "dataset/raw-data/ConceptARC/corpus"],
    # resolved relative to the subprocess's cwd. Passing --dataset-dirs <path> on the CLI
    # does not override this list -- verified against a real run: it silently kept
    # scanning the unmodified default and crashed with FileNotFoundError. Sidestep the CLI
    # entirely by giving the subprocess a writable cwd that already has the default paths
    # satisfied, decoupled from repo_dir -- which is read-only when HRM_REPO_DIR points at
    # a Kaggle Dataset mount, confirmed by a real run failing with OSError(30, 'Read-only
    # file system') when this used to write directly under repo_dir. The script itself is
    # still invoked from its real (possibly read-only) location via an absolute path.
    build_cwd = output_path.parent / "hrm_build_cwd"
    default_arc_dir = build_cwd / "dataset" / "raw-data" / "ARC-AGI" / "data"
    default_concept_dir = build_cwd / "dataset" / "raw-data" / "ConceptARC" / "corpus"
    if default_arc_dir.is_symlink() or default_arc_dir.is_file():
        default_arc_dir.unlink()
    elif default_arc_dir.exists():
        shutil.rmtree(default_arc_dir)
    default_arc_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(Path(raw_data_dir), default_arc_dir)
    default_concept_dir.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        str(script.resolve()),
        "--output-dir",
        str(output_path),
        "--num-aug",
        str(num_aug),
    ]
    return subprocess.run(
        command,
        cwd=build_cwd,
        check=True,
        capture_output=True,
        text=True,
    )

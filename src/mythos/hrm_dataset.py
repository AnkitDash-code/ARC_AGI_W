"""Dataset-preparation glue for the external HRM checkout."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from typing import Iterable

from mythos.arc import ArcTask, ArcValidationError, Grid, require_test_outputs


def default_run_dir() -> Path:
    import os

    return Path(os.environ.get("MYTHOS_RUN_DIR", "runs"))


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
    command = [
        sys.executable,
        str(script),
        "--dataset-dirs",
        str(Path(raw_data_dir)),
        "--output-dir",
        str(output_path),
        "--num-aug",
        str(num_aug),
    ]
    return subprocess.run(
        command,
        cwd=repo_dir,
        check=True,
        capture_output=True,
        text=True,
    )

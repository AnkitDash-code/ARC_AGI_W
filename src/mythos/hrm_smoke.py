"""CLI smoke test for the external HRM runtime."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from mythos.arc import ArcValidationError, attach_solutions, load_challenges, load_solutions
from mythos.hrm_dataset import build_hrm_dataset, default_run_dir, prepare_hrm_raw_dataset
from mythos.solvers.hrm import HRMEnvironment, HRMEnvironmentError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Smoke-test external HRM integration.")
    parser.add_argument("--task", required=True, help="ARC-style challenge JSON for the smoke run.")
    parser.add_argument("--solutions", help="Optional solution JSON if --task omits test outputs.")
    parser.add_argument("--run-dir", default=None, help="Output directory for smoke artifacts.")
    parser.add_argument("--num-aug", type=int, default=0, help="HRM dataset-builder augmentation count.")
    parser.add_argument(
        "--skip-dataset-build",
        action="store_true",
        help="Only write the HRM raw-data layout; do not invoke HRM's dataset builder.",
    )
    args = parser.parse_args(argv)

    started = time.perf_counter()
    try:
        tasks = load_challenges(args.task)
        if args.solutions:
            tasks = attach_solutions(tasks, load_solutions(args.solutions))

        env = HRMEnvironment.from_env()
        env.validate(require_cuda=True)
        modules = env.import_modules()

        torch = HRMEnvironment._import_torch()
        torch.cuda.reset_peak_memory_stats()
        checkpoint = env.load_checkpoint()

        run_dir = Path(args.run_dir) if args.run_dir else default_run_dir() / "hrm_smoke"
        raw_dir = prepare_hrm_raw_dataset(tasks.values(), run_dir / "raw" / "ARC-AGI-2" / "data")

        dataset_build = None
        if not args.skip_dataset_build:
            result = build_hrm_dataset(
                hrm_repo_dir=env.repo_dir,
                raw_data_dir=raw_dir,
                output_dir=run_dir / "data" / "arc-2-smoke",
                num_aug=args.num_aug,
            )
            dataset_build = {
                "returncode": result.returncode,
                "stdout_tail": result.stdout[-2000:],
                "stderr_tail": result.stderr[-2000:],
            }
    except (ArcValidationError, HRMEnvironmentError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    summary = {
        "tasks": len(tasks),
        "hrm_repo_dir": str(env.repo_dir),
        "checkpoint_path": str(env.checkpoint_path),
        "checkpoint_type": type(checkpoint).__name__,
        "imported_modules": sorted(modules),
        "raw_data_dir": str(raw_dir),
        "dataset_build": dataset_build,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "cuda_peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

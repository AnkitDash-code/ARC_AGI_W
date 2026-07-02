# Project Mythos

Base implementation for testing ARC-style data loading, fixture solvers, scoring, submission generation, and an external HRM smoke path.

## Setup

```bash
python -m pip install -e ".[dev]"
```

## Local commands

```bash
python -m mythos.validate data/toy/challenges.json
python -m mythos.solve --solver pipeline --challenges data/toy/challenges.json --out runs/submission.json
python -m mythos.score --pred runs/submission.json --solutions data/toy/solutions.json
python -m pytest
```

The plan-aligned pipeline can load external model checkpoints from environment variables:

```bash
export IJEPA_REPO_DIR=/path/to/ijepa
export IJEPA_CHECKPOINT_PATH=/path/to/jepa.pt
export HRM_TEXT_REPO_DIR=/path/to/hrm-text
export HRM_TEXT_CHECKPOINT_PATH=/path/to/hrm_text.pt
export WORLD_MODEL_CHECKPOINT_PATH=/path/to/world_model.pt
export TTT_LORA_CHECKPOINT_PATH=/path/to/lora.pt
export HRM_REPO_DIR=/path/to/HRM
export HRM_CHECKPOINT_PATH=/path/to/hrm.pt
```

Use `--model-mode strict` to require every planned model checkpoint before running. Default `fallback` mode loads any configured checkpoints and keeps placeholder adapters active for missing models.

## Kaggle smoke submission

For the ARC Prize 2026 Kaggle dataset mounted at `/kaggle/input/competitions/arc-prize-2026-arc-agi-2`, run:

```bash
python -m mythos.kaggle_run \
  --data-dir /kaggle/input/competitions/arc-prize-2026-arc-agi-2 \
  --split test \
  --solver pipeline \
  --model-mode fallback \
  --out /kaggle/working/submission.json
```

The `pipeline` solver follows the master-plan stage order: ingest, JEPA encode adapter, HRM-Text planning adapter, world-model simulation adapter, TTT/LoRA adaptation adapter, HRM L-module execution adapter, and decode/output. The real-model stages are explicit placeholders today; execution uses the baseline fallback until HRM prediction is wired.

The same flow is available as a notebook in `project_mythos_kaggle_pipeline.ipynb`.

To validate locally on the public evaluation split:

```bash
python -m mythos.kaggle_run \
  --data-dir /kaggle/input/competitions/arc-prize-2026-arc-agi-2 \
  --split evaluation \
  --solver pipeline \
  --out /kaggle/working/evaluation_submission.json \
  --score
```

## HRM smoke path

The HRM integration is intentionally external. Set these in a Kaggle/Linux CUDA environment:

```bash
export HRM_REPO_DIR=/path/to/HRM
export HRM_CHECKPOINT_PATH=/path/to/checkpoint.pt
python -m mythos.hrm_smoke --task data/toy/challenges.json
```

The smoke command validates the ARC task, checks the HRM checkout and checkpoint, imports HRM modules, loads the checkpoint with PyTorch, writes an HRM-compatible raw dataset layout, and optionally invokes HRM's dataset builder. Real prediction wiring stays behind the `HRMSolver` adapter until the upstream runtime is available.

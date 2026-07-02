#!/usr/bin/env bash
set -euo pipefail

python -m mythos.kaggle_run \
  --data-dir /kaggle/input/competitions/arc-prize-2026-arc-agi-2 \
  --split test \
  --solver pipeline \
  --model-mode fallback \
  --out /kaggle/working/submission.json

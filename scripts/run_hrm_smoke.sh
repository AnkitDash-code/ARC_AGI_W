#!/usr/bin/env bash
set -euo pipefail

python -m mythos.hrm_smoke --task "${1:-data/toy/challenges.json}"

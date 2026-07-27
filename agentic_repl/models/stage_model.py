"""Download and verify the agentic-REPL code-LLM GGUF artifact for Kaggle staging.

This does NOT upload anything to Kaggle -- run `kaggle datasets create`
yourself once the local directory is ready (see agentic_repl/models/README.md).
Kept as a plain script, not wired into any CI, because publishing a new
Kaggle Dataset is a deliberate, external, one-time action, not something that
should happen as a side effect of building or testing this package.

Note: the GGUF model itself has already been staged as
ankitdash24/qwen3-coder-30b-a3b-instruct-gguf (18.6GB, size- and
sha256-verified against Hugging Face's published hash). This script is kept
for reference/reproducibility, not because staging needs to happen again.

Usage:
    python agentic_repl/models/stage_model.py --out-dir <local-dir>
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys
import urllib.request

MODEL_REPO = "unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF"
MODEL_FILE = "Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf"
MODEL_URL = f"https://huggingface.co/{MODEL_REPO}/resolve/main/{MODEL_FILE}"

# Confirmed via a real staging run (matches Hugging Face's own X-Linked-ETag
# for this file).
EXPECTED_SHA256 = "fadc3e5f8d42bf7e894a785b05082e47daee4df26680389817e2093056f088ad"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True, help="Directory to stage the model file into.")
    parser.add_argument("--url", default=MODEL_URL, help="Override the GGUF download URL.")
    parser.add_argument("--filename", default=MODEL_FILE, help="Output filename.")
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    destination = out_dir / args.filename

    print(f"Downloading {args.url} -> {destination}", file=sys.stderr)
    urllib.request.urlretrieve(args.url, destination)

    checksum = _sha256(destination)
    print(f"sha256: {checksum}")
    if EXPECTED_SHA256 and checksum != EXPECTED_SHA256:
        print("ERROR: checksum mismatch -- delete and re-download before staging.", file=sys.stderr)
        return 2

    size_gb = destination.stat().st_size / 1e9
    print(
        f"Staged {destination} ({size_gb:.2f} GB). "
        f"Next: kaggle datasets create -p {out_dir} --dir-mode zip",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

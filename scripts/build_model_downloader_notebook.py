"""Build a small, CPU-only, internet-enabled Kaggle notebook that fetches the
agentic-REPL code-LLM GGUF from Hugging Face and publishes it as a Kaggle
Dataset -- entirely on Kaggle's own infrastructure, no local re-upload of a
~18.6GB file.

This is a one-time staging step (see agentic_repl/models/README.md), not part
of the actual solve pipeline, so it's a separate generated notebook from
scripts/build_standalone_notebook.py with its own kernel-metadata.

Kaggle Secrets don't survive an API-based `kaggle kernels push` (confirmed:
https://www.kaggle.com/discussions/product-feedback/666571), so this
notebook authenticates by reading a Kaggle API token staged as its own small
private Dataset (ankitdash24/agentic-repl-kaggle-token) rather than a Secret.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KERNEL_DIR = ROOT / "kaggle_kernels" / "model_downloader"
OUTPUT = KERNEL_DIR / "model_downloader.ipynb"
KERNEL_METADATA_OUTPUT = KERNEL_DIR / "kernel-metadata.json"
DOWNLOADER_KERNEL_ID = "ankitdash24/agentic-repl-model-downloader"

MODEL_REPO = "unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF"
MODEL_FILE = "Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf"
MODEL_URL = f"https://huggingface.co/{MODEL_REPO}/resolve/main/{MODEL_FILE}"
TARGET_DATASET_ID = "ankitdash24/qwen3-coder-30b-a3b-instruct-gguf"
TARGET_DATASET_TITLE = "qwen3-coder-30b-a3b-instruct-gguf"
CREDENTIALS_DATASET_SLUG = "agentic-repl-kaggle-token"
HF_TOKEN_DATASET_SLUG = "agentic-repl-hf-token"


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


def build() -> None:
    notebook = {
        "cells": [
            _markdown(
                "# Agentic REPL model staging: download + publish\n\n"
                f"Downloads `{MODEL_FILE}` from `{MODEL_REPO}` on Hugging Face "
                "and publishes it as the Kaggle Dataset "
                f"`{TARGET_DATASET_ID}`. CPU-only, internet-enabled, no GPU "
                "quota consumed. One-time staging step -- see "
                "agentic_repl/models/README.md."
            ),
            _markdown("## 1. Authenticate the Kaggle CLI from the staged token dataset"),
            _code(
                "from pathlib import Path\n"
                "import shutil\n"
                "import subprocess\n"
                "import sys\n\n"
                "INPUT_ROOT = Path('/kaggle/input')\n"
                "print('/kaggle/input contents:')\n"
                "for entry in sorted(INPUT_ROOT.rglob('*')):\n"
                "    print(' ', entry)\n\n"
                "def find_input_file(dataset_slug, filename):\n"
                "    # Confirmed directly: private dataset inputs can mount at either\n"
                "    # /kaggle/input/<slug>/<file> or the nested\n"
                "    # /kaggle/input/datasets/<owner>/<slug>/<file> -- check both rather\n"
                "    # than assume either.\n"
                "    flat = INPUT_ROOT / dataset_slug / filename\n"
                "    if flat.is_file():\n"
                "        return flat\n"
                "    nested = sorted(INPUT_ROOT.glob(f'datasets/*/{dataset_slug}/{filename}'))\n"
                "    if nested:\n"
                "        return nested[0]\n"
                "    return None\n\n"
                f"CREDENTIALS_SRC = find_input_file({CREDENTIALS_DATASET_SLUG!r}, 'kaggle.json')\n"
                "if CREDENTIALS_SRC is None:\n"
                "    raise SystemExit(\n"
                "        'No kaggle.json found under /kaggle/input -- the credentials dataset '\n"
                "        'is not mounted. Check dataset_sources in kernel-metadata.json and that '\n"
                "        'the dataset is attached to this kernel.'\n"
                "    )\n"
                "print('Using Kaggle credentials from:', CREDENTIALS_SRC)\n\n"
                "KAGGLE_CONFIG_DIR = Path.home() / '.kaggle'\n"
                "KAGGLE_CONFIG_DIR.mkdir(parents=True, exist_ok=True)\n"
                "shutil.copy(CREDENTIALS_SRC, KAGGLE_CONFIG_DIR / 'kaggle.json')\n"
                "(KAGGLE_CONFIG_DIR / 'kaggle.json').chmod(0o600)\n"
                "print('Wrote Kaggle credentials to', KAGGLE_CONFIG_DIR / 'kaggle.json')\n\n"
                "subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'kaggle'], check=True)\n"
                "print('kaggle CLI ready')\n\n"
                f"HF_TOKEN_SRC = find_input_file({HF_TOKEN_DATASET_SLUG!r}, 'hf_token.txt')\n"
                "HF_TOKEN = HF_TOKEN_SRC.read_text(encoding='utf-8').strip() if HF_TOKEN_SRC else None\n"
                "if HF_TOKEN:\n"
                "    print('Found HF token -- downloads will be authenticated (higher rate limit).')\n"
                "else:\n"
                "    print('No HF token staged -- downloading anonymously (may be rate-limited/slow).')\n"
            ),
            _markdown(
                "## 2. Check free disk space before downloading\n\n"
                "`/kaggle/working` is capped around ~21GB total, too tight for an "
                "~18.6GB file plus margin (confirmed directly: 20.9GB free on a "
                "fresh session). Download to `/kaggle/tmp` instead, which reports "
                "a larger ephemeral quota."
            ),
            _code(
                "WORKDIR = Path('/kaggle/working')\n"
                "DOWNLOAD_DIR = Path('/kaggle/tmp/model')\n"
                "DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)\n"
                f"REQUIRED_BYTES = int({MODEL_FILE!r} and 18.6e9 * 1.15)  # +15% safety margin\n"
                "total, used, free = shutil.disk_usage(DOWNLOAD_DIR)\n"
                "print(f'/kaggle/tmp: total={total/1e9:.1f}GB used={used/1e9:.1f}GB free={free/1e9:.1f}GB')\n"
                "print(f'required (with margin) = {REQUIRED_BYTES/1e9:.1f}GB')\n\n"
                "if free < REQUIRED_BYTES:\n"
                "    raise SystemExit(\n"
                "        f'Not enough free disk in /kaggle/tmp ({free/1e9:.1f}GB free, '\n"
                "        f'need ~{REQUIRED_BYTES/1e9:.1f}GB) -- aborting before a partial download. '\n"
                "        'See agentic_repl/models/README.md for a smaller-model fallback.'\n"
                "    )\n"
            ),
            _markdown(
                "## 3. Download the GGUF from Hugging Face (resumable, size-verified)\n\n"
                "Uses HF_TOKEN (if staged) via an Authorization header -- anonymous "
                "requests are explicitly rate-limited by Hugging Face "
                "(`X-HF-Warning: unauthenticated ... enable higher rate limits and "
                "faster downloads`, confirmed directly against this exact URL).\n\n"
                "**Resumable by design, not just retried**: the redirect this URL "
                "resolves to is a presigned S3-style link valid for exactly "
                "`X-Amz-Expires=3600` seconds (confirmed directly in the response "
                "headers) -- a first version of this cell hit that expiry mid-transfer "
                "(achieved throughput was only ~5MB/s, so the full 18.6GB transfer "
                "took longer than the URL's 1-hour validity), the connection closed "
                "early, and the code published the truncated ~17.5GB result anyway "
                "since nothing checked the final size against Content-Length. Fixed by "
                "re-resolving a fresh presigned URL (via a Range request against the "
                "*original* HF URL, not the expired redirect) on every retry, and by "
                "hard-failing before publishing if the final size doesn't match."
            ),
            _code(
                "import time\n"
                "import urllib.error\n"
                "import urllib.request\n\n"
                f"MODEL_URL = {MODEL_URL!r}\n"
                f"MODEL_FILE = {MODEL_FILE!r}\n"
                "MODEL_DIR = DOWNLOAD_DIR\n"
                "destination = MODEL_DIR / MODEL_FILE\n"
                "CHUNK_SIZE = 8 * 1024 * 1024\n"
                "MAX_ATTEMPTS = 10\n"
                "SOCKET_TIMEOUT_S = 60\n\n"
                "auth_headers = {'Authorization': f'Bearer {HF_TOKEN}'} if HF_TOKEN else {}\n\n"
                "expected_size = None\n"
                "started = time.perf_counter()\n"
                "for attempt in range(1, MAX_ATTEMPTS + 1):\n"
                "    existing = destination.stat().st_size if destination.exists() else 0\n"
                "    headers = dict(auth_headers)\n"
                "    if existing:\n"
                "        headers['Range'] = f'bytes={existing}-'\n"
                "    request = urllib.request.Request(MODEL_URL, headers=headers)\n"
                "    try:\n"
                "        with urllib.request.urlopen(request, timeout=SOCKET_TIMEOUT_S) as response:\n"
                "            if existing and response.status == 206:\n"
                "                mode, base = 'ab', existing\n"
                "                content_range_total = response.headers.get('Content-Range', '').rsplit('/', 1)[-1]\n"
                "                expected_size = int(content_range_total) if content_range_total.isdigit() else expected_size\n"
                "            else:\n"
                "                mode, base = 'wb', 0  # server ignored/rejected Range -- restart clean\n"
                "                expected_size = int(response.headers.get('Content-Length', 0)) or expected_size\n"
                "            print(f'attempt {attempt}: status={response.status} mode={mode} resume_from={base} expected_size={expected_size}')\n"
                "            downloaded = base\n"
                "            next_report_at = downloaded\n"
                "            with destination.open(mode) as out_file:\n"
                "                while True:\n"
                "                    chunk = response.read(CHUNK_SIZE)\n"
                "                    if not chunk:\n"
                "                        break\n"
                "                    out_file.write(chunk)\n"
                "                    downloaded += len(chunk)\n"
                "                    if downloaded >= next_report_at:\n"
                "                        elapsed_so_far = time.perf_counter() - started\n"
                "                        rate_mb_s = (downloaded / 1e6) / elapsed_so_far if elapsed_so_far > 0 else 0\n"
                "                        pct = 100 * downloaded / expected_size if expected_size else 0\n"
                "                        print(f'{downloaded/1e9:.2f}/{(expected_size or 0)/1e9:.2f} GB ({pct:.1f}%) -- {rate_mb_s:.1f} MB/s avg')\n"
                "                        next_report_at = downloaded + 500 * 1024 * 1024  # every ~500MB\n"
                "    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:\n"
                "        print(f'attempt {attempt} failed: {exc!r} -- will retry, resuming from current file size')\n"
                "        continue\n"
                "    final_size = destination.stat().st_size\n"
                "    print(f'attempt {attempt} ended with {final_size/1e9:.2f} GB on disk')\n"
                "    if expected_size and final_size >= expected_size:\n"
                "        break\n"
                "else:\n"
                "    raise SystemExit(\n"
                "        f'Download did not complete after {MAX_ATTEMPTS} attempts '\n"
                "        f'({destination.stat().st_size if destination.exists() else 0} bytes on disk, '\n"
                "        f'expected {expected_size}). Aborting before publishing an incomplete file.'\n"
                "    )\n\n"
                "elapsed = time.perf_counter() - started\n"
                "final_size = destination.stat().st_size\n"
                "if expected_size and final_size != expected_size:\n"
                "    raise SystemExit(\n"
                "        f'Downloaded size {final_size} != expected {expected_size} -- '\n"
                "        'aborting before publishing a corrupt/truncated file.'\n"
                "    )\n"
                "size_gb = final_size / 1e9\n"
                "print(f'Downloaded {destination} ({size_gb:.2f} GB, verified against Content-Length) '\n"
                "      f'in {elapsed:.0f}s ({(size_gb*1000)/elapsed:.1f} MB/s average)')\n"
            ),
            _markdown("## 4. Verify checksum"),
            _code(
                "import hashlib\n\n"
                "digest = hashlib.sha256()\n"
                "with destination.open('rb') as handle:\n"
                "    for chunk in iter(lambda: handle.read(1 << 20), b''):\n"
                "        digest.update(chunk)\n"
                "print('sha256:', digest.hexdigest())\n"
            ),
            _markdown("## 5. Publish as a Kaggle Dataset"),
            _code(
                "import json as json_module\n\n"
                f"TARGET_DATASET_ID = {TARGET_DATASET_ID!r}\n"
                f"TARGET_DATASET_TITLE = {TARGET_DATASET_TITLE!r}\n"
                "metadata = {\n"
                "    'title': TARGET_DATASET_TITLE,\n"
                "    'id': TARGET_DATASET_ID,\n"
                "    'licenses': [{'name': 'unknown'}],\n"
                "}\n"
                "(MODEL_DIR / 'dataset-metadata.json').write_text(json_module.dumps(metadata, indent=2))\n\n"
                "check = subprocess.run(\n"
                "    ['kaggle', 'datasets', 'status', TARGET_DATASET_ID],\n"
                "    capture_output=True, text=True,\n"
                ")\n"
                "dataset_exists = check.returncode == 0\n"
                "print('dataset_exists =', dataset_exists)\n"
                "if dataset_exists:\n"
                "    result = subprocess.run(\n"
                "        ['kaggle', 'datasets', 'version', '-p', str(MODEL_DIR), '-m', 'update', '-r', 'zip'],\n"
                "        capture_output=True, text=True,\n"
                "    )\n"
                "else:\n"
                "    result = subprocess.run(\n"
                "        ['kaggle', 'datasets', 'create', '-p', str(MODEL_DIR), '-r', 'zip'],\n"
                "        capture_output=True, text=True,\n"
                "    )\n"
                "print('--- stdout ---')\n"
                "print(result.stdout)\n"
                "print('--- stderr ---')\n"
                "print(result.stderr)\n"
                "result.check_returncode()\n"
                "print('Published dataset:', TARGET_DATASET_ID)\n"
            ),
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
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
    KERNEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    print(f"Wrote {OUTPUT}")

    kernel_metadata = {
        "id": DOWNLOADER_KERNEL_ID,
        "title": "Agentic REPL Model Downloader",
        "code_file": "model_downloader.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": False,
        "enable_tpu": False,
        "enable_internet": True,
        "dataset_sources": [
            f"ankitdash24/{CREDENTIALS_DATASET_SLUG}",
            f"ankitdash24/{HF_TOKEN_DATASET_SLUG}",
        ],
        "competition_sources": [],
        "kernel_sources": [],
    }
    KERNEL_METADATA_OUTPUT.write_text(json.dumps(kernel_metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {KERNEL_METADATA_OUTPUT}")


if __name__ == "__main__":
    build()

"""Build a small, CPU-only, internet-enabled Kaggle notebook that fetches the
llama-cpp-python cu125 prebuilt wheel from GitHub Releases and publishes it as
a Kaggle Dataset -- entirely on Kaggle's own infrastructure.

Local downloads of this exact file were tried first and abandoned: sustained
throughput from this network to GitHub's release-assets CDN (Azure Blob
backed) was ~15KB/s (confirmed via two separate connection attempts, each
projecting 28+ hours to complete), while the same network reached 12.5MB/s
against Cloudflare in the same few minutes -- a CDN-specific bottleneck, not
a general connectivity problem. Kaggle's datacenter network already proved
excellent throughput for a similarly large Hugging Face download (14.2MB/s
for the 18.6GB GGUF model, once the earlier auth/resume issues were fixed),
so this mirrors that same "fetch via Kaggle, not the dev laptop" pattern
instead of continuing to fight the slow path.

Reuses the same credentials-dataset self-publish pattern as
scripts/build_model_downloader_notebook.py (see that file's docstring for why
Kaggle Secrets don't work for an API-pushed kernel).
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KERNEL_DIR = ROOT / "kaggle_kernels" / "wheel_downloader"
OUTPUT = KERNEL_DIR / "wheel_downloader.ipynb"
KERNEL_METADATA_OUTPUT = KERNEL_DIR / "kernel-metadata.json"
DOWNLOADER_KERNEL_ID = "ankitdash24/agentic-repl-wheel-downloader"

WHEEL_URL = (
    "https://github.com/abetlen/llama-cpp-python/releases/download/"
    "v0.3.31-cu125/llama_cpp_python-0.3.31-py3-none-manylinux_2_35_x86_64.whl"
)
WHEEL_FILE = "llama_cpp_python-0.3.31-py3-none-manylinux_2_35_x86_64.whl"
EXPECTED_SIZE_BYTES = 1832490513  # confirmed via HEAD request against WHEEL_URL

# llama-cpp-python's one pure-python dependency not already on Kaggle's image
# (confirmed via a real offline-install attempt: numpy/typing_extensions were
# already satisfied, diskcache was not -- "ERROR: Could not find a version
# that satisfies the requirement diskcache>=5.6.1 ... No matching
# distribution found", since --no-index blocks PyPI entirely).
EXTRA_PYPI_PACKAGES = ["diskcache"]
TARGET_DATASET_ID = "ankitdash24/agentic-repl-llama-cpp-wheel"
TARGET_DATASET_TITLE = "agentic-repl-llama-cpp-wheel"
CREDENTIALS_DATASET_SLUG = "agentic-repl-kaggle-token"


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
                "# Agentic REPL wheel staging: download + publish\n\n"
                f"Downloads `{WHEEL_FILE}` from GitHub Releases and publishes it as "
                f"the Kaggle Dataset `{TARGET_DATASET_ID}`. CPU-only, "
                "internet-enabled, no GPU quota consumed."
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
                "    flat = INPUT_ROOT / dataset_slug / filename\n"
                "    if flat.is_file():\n"
                "        return flat\n"
                "    nested = sorted(INPUT_ROOT.glob(f'datasets/*/{dataset_slug}/{filename}'))\n"
                "    return nested[0] if nested else None\n\n"
                f"CREDENTIALS_SRC = find_input_file({CREDENTIALS_DATASET_SLUG!r}, 'kaggle.json')\n"
                "if CREDENTIALS_SRC is None:\n"
                "    raise SystemExit('No kaggle.json found under /kaggle/input.')\n"
                "print('Using Kaggle credentials from:', CREDENTIALS_SRC)\n\n"
                "KAGGLE_CONFIG_DIR = Path.home() / '.kaggle'\n"
                "KAGGLE_CONFIG_DIR.mkdir(parents=True, exist_ok=True)\n"
                "shutil.copy(CREDENTIALS_SRC, KAGGLE_CONFIG_DIR / 'kaggle.json')\n"
                "(KAGGLE_CONFIG_DIR / 'kaggle.json').chmod(0o600)\n"
                "subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'kaggle'], check=True)\n"
                "print('kaggle CLI ready')\n"
            ),
            _markdown("## 2. Download the wheel (resumable, size-verified)"),
            _code(
                "import time\n"
                "import urllib.error\n"
                "import urllib.request\n\n"
                f"WHEEL_URL = {WHEEL_URL!r}\n"
                f"WHEEL_FILE = {WHEEL_FILE!r}\n"
                f"EXPECTED_SIZE_BYTES = {EXPECTED_SIZE_BYTES}\n"
                "WHEEL_DIR = Path('/kaggle/tmp/wheel')\n"
                "WHEEL_DIR.mkdir(parents=True, exist_ok=True)\n"
                "destination = WHEEL_DIR / WHEEL_FILE\n"
                "CHUNK_SIZE = 8 * 1024 * 1024\n"
                "MAX_ATTEMPTS = 10\n\n"
                "started = time.perf_counter()\n"
                "for attempt in range(1, MAX_ATTEMPTS + 1):\n"
                "    existing = destination.stat().st_size if destination.exists() else 0\n"
                "    headers = {'Range': f'bytes={existing}-'} if existing else {}\n"
                "    request = urllib.request.Request(WHEEL_URL, headers=headers)\n"
                "    try:\n"
                "        with urllib.request.urlopen(request, timeout=60) as response:\n"
                "            mode = 'ab' if existing and response.status == 206 else 'wb'\n"
                "            downloaded = existing if mode == 'ab' else 0\n"
                "            print(f'attempt {attempt}: status={response.status} mode={mode} resume_from={downloaded}')\n"
                "            with destination.open(mode) as out_file:\n"
                "                while True:\n"
                "                    chunk = response.read(CHUNK_SIZE)\n"
                "                    if not chunk:\n"
                "                        break\n"
                "                    out_file.write(chunk)\n"
                "                    downloaded += len(chunk)\n"
                "    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:\n"
                "        print(f'attempt {attempt} failed: {exc!r} -- retrying')\n"
                "        continue\n"
                "    final_size = destination.stat().st_size\n"
                "    print(f'attempt {attempt} ended with {final_size} bytes on disk')\n"
                "    if final_size >= EXPECTED_SIZE_BYTES:\n"
                "        break\n"
                "else:\n"
                "    raise SystemExit(f'Download incomplete after {MAX_ATTEMPTS} attempts.')\n\n"
                "elapsed = time.perf_counter() - started\n"
                "final_size = destination.stat().st_size\n"
                "if final_size != EXPECTED_SIZE_BYTES:\n"
                "    raise SystemExit(f'size {final_size} != expected {EXPECTED_SIZE_BYTES} -- aborting.')\n"
                "print(f'Downloaded {destination} ({final_size/1e9:.2f} GB) in {elapsed:.0f}s '\n"
                "      f'({(final_size/1e6)/elapsed:.1f} MB/s average)')\n"
            ),
            _markdown("## 3. Verify it's a real, complete wheel (valid zip)"),
            _code(
                "import zipfile\n\n"
                "with zipfile.ZipFile(destination) as archive:\n"
                "    bad_file = archive.testzip()\n"
                "    if bad_file is not None:\n"
                "        raise SystemExit(f'Corrupt member in wheel zip: {bad_file}')\n"
                "    names = archive.namelist()\n"
                "print(f'Valid zip archive, {len(names)} entries, e.g.:', names[:5])\n"
            ),
            _markdown(
                "## 4. Download extra pure-Python dependencies not already on Kaggle"
            ),
            _code(
                f"EXTRA_PACKAGES = {EXTRA_PYPI_PACKAGES!r}\n"
                "for package in EXTRA_PACKAGES:\n"
                "    result = subprocess.run(\n"
                "        [sys.executable, '-m', 'pip', 'download', package, '--no-deps', '-d', str(WHEEL_DIR)],\n"
                "        capture_output=True, text=True, timeout=120,\n"
                "    )\n"
                "    print(f'--- pip download {package} ---')\n"
                "    print(result.stdout[-1500:])\n"
                "    if result.returncode != 0:\n"
                "        print(result.stderr[-1500:])\n"
                "    result.check_returncode()\n"
                "print('files in WHEEL_DIR now:', sorted(p.name for p in WHEEL_DIR.glob('*.whl')))\n"
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
                "(WHEEL_DIR / 'dataset-metadata.json').write_text(json_module.dumps(metadata, indent=2))\n\n"
                "check = subprocess.run(['kaggle', 'datasets', 'status', TARGET_DATASET_ID], capture_output=True, text=True)\n"
                "dataset_exists = check.returncode == 0\n"
                "print('dataset_exists =', dataset_exists)\n"
                "cmd = (\n"
                "    ['kaggle', 'datasets', 'version', '-p', str(WHEEL_DIR), '-m', 'update']\n"
                "    if dataset_exists else\n"
                "    ['kaggle', 'datasets', 'create', '-p', str(WHEEL_DIR)]\n"
                ")\n"
                "result = subprocess.run(cmd, capture_output=True, text=True)\n"
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
        "title": "Agentic REPL Wheel Downloader",
        "code_file": "wheel_downloader.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": False,
        "enable_tpu": False,
        "enable_internet": True,
        "dataset_sources": [f"ankitdash24/{CREDENTIALS_DATASET_SLUG}"],
        "competition_sources": [],
        "kernel_sources": [],
    }
    KERNEL_METADATA_OUTPUT.write_text(json.dumps(kernel_metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {KERNEL_METADATA_OUTPUT}")


if __name__ == "__main__":
    build()

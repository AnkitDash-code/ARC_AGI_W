# Agentic REPL model staging

`AgenticReplSolver` calls a local quantized code-LLM through
`agentic_repl.llm.client.LlamaCppClient`. Kaggle competition reruns are
internet-disabled, so both the model *and* its inference library have to be
pre-staged as Kaggle Datasets and mounted at submission time -- the same
pattern this repo already uses for the HRM checkpoint
(`ankitdash24/hrm-arc2-checkpoint`, see the main `README.md`'s "Architecture
as of the last commit" section).

## Confirmed hardware

The real ARC Prize 2026 scoring sandbox is **4x NVIDIA L4 (96GB total VRAM)**,
12h wall-clock for 240 tasks (confirmed directly, and consistent with public
retrospectives on the 2025 competition's infrastructure).

**Important lesson from the 2025 competition** (NVARC, the Kaggle Grandmaster
winning team): they explicitly did *not* reach for the biggest/smartest model
-- they won with a small (~4B), fine-tuned model plus test-time training,
because heavier LLM reasoning didn't fit the tight per-task time budget
(~3 min/task average across 240 tasks in 12h, shared with every other solver
stage). Generation latency per candidate, multiplied by N candidates and K
refinement rounds, has to fit that budget -- so **generation speed matters at
least as much as raw model capability** when picking a model here.

**Kaggle's L4 sessions hard-enforce no internet, confirmed directly**: setting
`enable_internet: true` in `kernel-metadata.json` did not help -- a live `pip
install` attempt on a real L4 run failed with DNS resolution errors regardless.
This isn't just competition-rules guidance ("No Internet - All L4 sessions
must have internet disabled"), it's actually enforced platform-side. Every
dependency, not just the model, has to be staged offline.

## Chosen model

**`Qwen3-Coder-30B-A3B-Instruct`**, GGUF `Q4_K_M` quantization (~18.6GB,
`unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF` -- no official Qwen-published GGUF
exists for this variant). Sparse MoE, ~3B *active* parameters per token, so it
generates at roughly the speed of a much smaller dense model (matching the
"speed matters" lesson above). Comfortably fits a single L4 (24GB) with room
for KV cache, let alone the 96GB pool. Already staged and verified: published
as `ankitdash24/qwen3-coder-30b-a3b-instruct-gguf` (18.6GB byte-exact and
sha256-verified against Hugging Face's own published hash before publishing --
see git history of `scripts/build_model_downloader_notebook.py` for a first
version that silently published a truncated file, and why that's no longer
possible: the download is resumable and the final size is checked before
anything gets published).

**Why not a bigger/newer model**: current frontier open-weight models (e.g.
GLM-5.2 at ~744B total params, Kimi K2.x, DeepSeek-V4) are all too large to
fit 96GB even quantized, and per the lesson above would likely be too slow for
this competition's time budget regardless. `Qwen3-Coder-Next` (80B total, same
~3B active params, ~48.5GB at Q4_K_M) is a documented upgrade path if this
proves too weak, but its Hugging Face release ships only as a single ~48.5GB
file or four ~15GB shards, tight against Kaggle's ~20GB `/kaggle/working`
session disk cap for in-Kaggle staging.

## Inference stack: llama-cpp-python via an offline-staged wheel

Two alternatives were tried and dropped first:

- **transformers' native GGUF loader** (`AutoModelForCausalLM.from_pretrained(
  ..., gguf_file=...)`) needs no install at all -- `transformers`/`torch`/
  `accelerate` already ship on Kaggle's GPU image (confirmed: already used
  elsewhere in this codebase, `mythos.jepa_encoder`/`mythos.text_reasoning`/
  `mythos.kaggle_models`, with no install step). But transformers explicitly
  does not support the `qwen3moe` architecture for GGUF loading yet
  (confirmed error: `"GGUF model with architecture qwen3moe is not supported
  yet"`), so this is a dead end for this exact model.
- **vLLM** would give better multi-GPU throughput, but has a documented,
  currently-open compatibility issue running inside Kaggle notebooks above
  v0.10 -- an avoidable risk given this project's own history of losing real
  time to Kaggle-environment fragility (the HRM flash-attn saga, see main
  `README.md`).
- **Building llama-cpp-python from source** (the older, commonly-referenced
  Kaggle pattern: stage the source sdist + `scikit-build-core`/`ninja` as a
  wheelhouse, compile at offline-install time) risks the same kind of
  multi-hour CUDA compile this repo already hit once with flash-attn.

**What's actually used**: a **prebuilt CUDA wheel** from
`abetlen.github.io/llama-cpp-python/whl/cu125/` --
`llama_cpp_python-0.3.31-py3-none-manylinux_2_35_x86_64.whl`. Newer
llama-cpp-python releases are `py3-none` (pure ctypes bindings around a
compiled `.so`, not a Python-version-specific extension), and CUDA is
runtime-backward-compatible, so a cu125-built wheel runs fine against
Kaggle's newer CUDA (12.8, confirmed via `torch.version.cuda` on a real run).
No compilation, no Python-version matching needed. Staged as the small Kaggle
Dataset `ankitdash24/agentic-repl-llama-cpp-wheel` (~118MB, downloaded and
uploaded directly from a local machine -- small enough that the whole
Kaggle-notebook-downloader dance used for the 18.6GB model wasn't worth it)
and installed offline in the submission notebook via `pip install --no-index
--find-links`.

`llama-cpp-python` supports splitting a model's layers across multiple CUDA
devices (`tensor_split`) if using all 4 L4s for one model instance is ever
worth the added complexity; running independent single-GPU instances across
the 4 GPUs to solve different tasks in parallel is the simpler option and
likely the better fit for this workload (many independent short generations,
not one huge context).

## Staging steps (already done once; here for reproducibility)

1. **Model**: `python agentic_repl/models/stage_model.py --out-dir <dir>`
   downloads the GGUF locally (or use a Kaggle downloader notebook -- see
   `scripts/build_model_downloader_notebook.py` for the internet-enabled,
   CPU-only, self-publishing pattern used for the first upload), then
   `kaggle datasets create -p <dir>` (or `version` to update) publishes it.
2. **Inference wheel**: download
   `https://github.com/abetlen/llama-cpp-python/releases/download/v0.3.31-cu125/llama_cpp_python-0.3.31-py3-none-manylinux_2_35_x86_64.whl`
   locally, then `kaggle datasets create -p <dir-containing-just-the-whl>`
   to publish as `ankitdash24/agentic-repl-llama-cpp-wheel`.
3. Both datasets are listed in the main `kernel-metadata.json`'s
   `dataset_sources`. `MYTHOS_AGENTIC_MODEL_PATH` doesn't need to be set
   explicitly -- the bootstrap cell in `scripts/build_standalone_notebook.py`
   autodiscovers the first mounted `*.gguf` file (checking both the flat
   `/kaggle/input/<slug>/` and nested `/kaggle/input/datasets/<owner>/<slug>/`
   mount layouts -- confirmed directly that Kaggle uses either depending on
   context, not consistently one or the other).
4. Set `SOLVER_NAME = 'agentic_repl'` in the generated notebook.

## Local dev

None of the above is needed for local development or tests.
`tests/test_agentic_repl.py` and `scripts/benchmark_agentic_repl.py
--llm-client stub` use `FakeLLMClient` and never touch a real model, a GPU,
or the network.

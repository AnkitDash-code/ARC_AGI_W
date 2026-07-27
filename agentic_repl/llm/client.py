"""LLM client abstraction: the solver depends only on this Protocol, never on
which backend is actually generating code.

FakeLLMClient is a deterministic stand-in for tests (no network, no GPU, no
real model -- see tests/test_agentic_repl.py). LlamaCppClient is the real
backend: a local quantized code-LLM served via llama-cpp-python, loaded from
a GGUF file staged as a Kaggle Dataset for internet-disabled competition
reruns (see agentic_repl/models/README.md).

A transformers-native GGUF loader (AutoModelForCausalLM(..., gguf_file=...))
was considered first since transformers ships on Kaggle with no install
needed at all -- but transformers explicitly does not support the qwen3moe
architecture for GGUF loading yet (confirmed: "GGUF model with architecture
qwen3moe is not supported yet"), so it's not viable for this exact model.
llama-cpp-python needs an offline-staged wheel instead: Kaggle's L4 sessions
hard-enforce no internet regardless of kernel-metadata.json's enable_internet
setting (confirmed directly -- a live pip install attempt failed with DNS
resolution errors even with enable_internet=true), and building from source
risks the same multi-hour CUDA compile this repo already hit once with
flash-attn. The fix: a prebuilt CUDA wheel (llama-cpp-python's newer releases
are `py3-none-manylinux_*` -- pure ctypes bindings around a compiled .so, no
Python-version-specific extension, and CUDA is runtime-backward-compatible)
staged as a small Kaggle Dataset and installed offline via `pip install
--no-index --find-links`.
"""

from __future__ import annotations

import os
from typing import Protocol


class LLMClient(Protocol):
    def generate(self, prompt: str, *, n: int, temperature: float = 0.7) -> list[str]:
        """Return up to n candidate completions for prompt."""


class FakeLLMClient:
    """Deterministic stand-in: cycles through pre-scripted responses.

    Each generate() call advances a cursor through `responses`, so a
    multi-round refinement loop sees later scripted responses instead of the
    first one repeating forever -- lets tests script "wrong code, then the
    fix" without depending on any real model.
    """

    def __init__(self, responses: list[str]) -> None:
        if not responses:
            raise ValueError("FakeLLMClient needs at least one scripted response")
        self._responses = list(responses)
        self._cursor = 0

    def generate(self, prompt: str, *, n: int, temperature: float = 0.7) -> list[str]:
        del prompt, temperature  # unused: the fake ignores prompt content by design
        batch = []
        for _ in range(n):
            batch.append(self._responses[self._cursor % len(self._responses)])
            self._cursor += 1
        return batch


class LlamaCppClient:
    """Real backend: a local GGUF code model served via llama-cpp-python.

    Model path resolves from the `model_path` argument, else the
    MYTHOS_AGENTIC_MODEL_PATH env var (mirrors this repo's existing
    env-var-driven checkpoint loading, e.g. HRM_CHECKPOINT_PATH), so the
    same code runs against a local file during development and a mounted
    Kaggle Dataset during a real submission. Importing llama_cpp is deferred
    to __init__ so importing this module -- or selecting any other solver
    via mythos.solvers.factory -- never requires the dependency installed.
    """

    def __init__(
        self,
        model_path: str | None = None,
        *,
        n_gpu_layers: int = -1,
        n_ctx: int = 32768,
    ) -> None:
        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise RuntimeError(
                "LlamaCppClient requires the 'llama-cpp-python' package: "
                "pip install -e '.[agentic]'"
            ) from exc

        resolved_path = model_path or os.environ.get("MYTHOS_AGENTIC_MODEL_PATH")
        if not resolved_path:
            raise RuntimeError(
                "no model_path given and MYTHOS_AGENTIC_MODEL_PATH is not set "
                "(see agentic_repl/models/README.md)"
            )
        self._llama = Llama(
            model_path=resolved_path,
            n_gpu_layers=n_gpu_layers,
            n_ctx=n_ctx,
            verbose=False,
        )

    def generate(self, prompt: str, *, n: int, temperature: float = 0.7) -> list[str]:
        # create_chat_completion (not create_completion) so llama.cpp applies
        # the GGUF's embedded chat template -- Qwen3-Coder-Instruct was
        # fine-tuned to expect its chat format, and a first real run against
        # it via raw create_completion verified 0/30 candidates on real
        # ARC-AGI-2 training tasks, consistent with the model not being
        # prompted the way it was tuned to respond to.
        completions = []
        for _ in range(n):
            result = self._llama.create_chat_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1024,
                temperature=temperature,
            )
            completions.append(result["choices"][0]["message"]["content"] or "")
        return completions

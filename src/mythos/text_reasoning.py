"""Optional HRM-Text rule generation for ARC tasks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from mythos.arc import ArcTask, Grid
from mythos.features import DEFAULT_RULE_DIM


class HRMTextError(RuntimeError):
    """Raised when optional HRM-Text inference cannot run."""


@dataclass(frozen=True)
class HRMTextRuleResult:
    description: str
    vector: tuple[float, ...]
    model_root: str


def generate_hrm_text_rule(
    task: ArcTask,
    model_path: str | Path,
    *,
    max_new_tokens: int = 96,
    device: str | None = None,
    vector_dim: int = DEFAULT_RULE_DIM,
) -> HRMTextRuleResult:
    """Run a Hugging Face text model to produce a rule description/vector."""

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:  # pragma: no cover - optional dependency.
        raise HRMTextError("transformers and torch are required for HRM-Text inference") from exc

    root = _model_root(model_path)
    selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    try:
        tokenizer = AutoTokenizer.from_pretrained(root, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            root,
            trust_remote_code=True,
            torch_dtype=torch.float16 if selected_device == "cuda" else torch.float32,
        ).to(selected_device)
    except Exception as exc:  # pragma: no cover - depends on external model files.
        raise HRMTextError(f"failed to load HRM-Text model from {root}: {exc}") from exc

    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    prompt = format_arc_rule_prompt(task)
    encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).to(selected_device)
    try:
        with torch.no_grad():
            generated = model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
            )
            forward = model(
                generated,
                output_hidden_states=True,
                use_cache=False,
            )
    except Exception as exc:  # pragma: no cover - depends on external model behavior.
        raise HRMTextError(f"HRM-Text forward/generation failed: {exc}") from exc

    generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
    description = _extract_answer(prompt, generated_text)
    vector = _hidden_state_to_vector(forward.hidden_states[-1][0], vector_dim)
    return HRMTextRuleResult(description=description, vector=vector, model_root=str(root))


def format_arc_rule_prompt(task: ArcTask) -> str:
    examples = []
    for index, example in enumerate(task.train):
        examples.append(
            f"Train {index} input:\n{_grid_text(example.input)}\n"
            f"Train {index} output:\n{_grid_text(example.output or example.input)}"
        )
    tests = []
    for index, example in enumerate(task.test):
        tests.append(f"Test {index} input:\n{_grid_text(example.input)}")
    return (
        "You are solving an ARC abstract reasoning task. Infer the rule from "
        "the train pairs and describe the transformation in one concise sentence.\n\n"
        + "\n\n".join(examples)
        + "\n\n"
        + "\n\n".join(tests)
        + "\n\nRule:"
    )


def _model_root(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate.parent if candidate.is_file() else candidate


def _grid_text(grid: Grid) -> str:
    return "\n".join(" ".join(str(cell) for cell in row) for row in grid)


def _extract_answer(prompt: str, generated_text: str) -> str:
    if generated_text.startswith(prompt):
        generated_text = generated_text[len(prompt) :]
    answer = generated_text.strip()
    return answer or "No HRM-Text rule text generated."


def _hidden_state_to_vector(hidden_state, dim: int) -> tuple[float, ...]:  # type: ignore[no-untyped-def]
    if dim <= 0:
        raise ValueError("vector_dim must be positive")
    pooled = hidden_state.float().mean(dim=0).detach().cpu()
    if pooled.numel() < dim:
        values = pooled.tolist() + [0.0 for _ in range(dim - pooled.numel())]
    else:
        chunk_size = max(1, pooled.numel() // dim)
        values = []
        for index in range(dim):
            start = index * chunk_size
            end = pooled.numel() if index == dim - 1 else min(pooled.numel(), start + chunk_size)
            values.append(float(pooled[start:end].mean()))
    return _normalize(values[:dim])


def _normalize(values: Sequence[float]) -> tuple[float, ...]:
    norm = sum(value * value for value in values) ** 0.5
    if norm == 0.0:
        return tuple(round(value, 6) for value in values)
    return tuple(round(value / norm, 6) for value in values)

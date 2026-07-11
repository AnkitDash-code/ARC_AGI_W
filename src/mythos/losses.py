"""Loss functions for Mythos adaptation experiments."""

from __future__ import annotations

from typing import Sequence

from mythos.arc import Grid


def genie_background_consistency_loss(
    logits,
    input_grid: Grid,
    *,
    preserve_mask: Sequence[Sequence[bool]] | None = None,
):  # type: ignore[no-untyped-def]
    """Penalize changing cells that should remain visually consistent.

    `logits` may be shaped `[H, W, 10]`, `[1, H, W, 10]`, or `[10, H, W]`.
    The target color for preserved cells is the original input-grid color.
    """

    torch = _torch()
    prepared = _prepare_logits(logits)
    height = min(prepared.shape[0], len(input_grid))
    width = min(prepared.shape[1], len(input_grid[0]))
    mask = preserve_mask or _default_preserve_mask(input_grid)

    selected_logits = []
    selected_targets = []
    for row in range(height):
        for col in range(width):
            if row < len(mask) and col < len(mask[row]) and mask[row][col]:
                selected_logits.append(prepared[row, col])
                selected_targets.append(int(input_grid[row][col]))

    if not selected_logits:
        return prepared.sum() * 0.0

    logits_tensor = torch.stack(selected_logits, dim=0)
    target_tensor = torch.tensor(selected_targets, dtype=torch.long, device=prepared.device)
    return torch.nn.functional.cross_entropy(logits_tensor, target_tensor)


def background_preservation_mask(input_grid: Grid, output_grid: Grid | None = None) -> tuple[tuple[bool, ...], ...]:
    """Return cells that should be preserved for consistency regularization."""

    if output_grid is not None and _same_shape(input_grid, output_grid):
        return tuple(
            tuple(input_grid[row][col] == output_grid[row][col] for col in range(len(input_grid[0])))
            for row in range(len(input_grid))
        )
    return _default_preserve_mask(input_grid)


def _default_preserve_mask(input_grid: Grid) -> tuple[tuple[bool, ...], ...]:
    background = _dominant_color(input_grid)
    return tuple(tuple(cell == background for cell in row) for row in input_grid)


def _prepare_logits(logits):  # type: ignore[no-untyped-def]
    torch = _torch()
    tensor = logits if hasattr(logits, "shape") else torch.as_tensor(logits)
    if tensor.ndim == 4:
        if tensor.shape[0] != 1:
            raise ValueError("batched consistency loss expects batch size 1")
        tensor = tensor[0]
    if tensor.ndim != 3:
        raise ValueError("logits must have rank 3 or rank 4")
    if tensor.shape[0] == 10 and tensor.shape[-1] != 10:
        tensor = tensor.permute(1, 2, 0)
    if tensor.shape[-1] != 10:
        raise ValueError("logits must have 10 color channels")
    return tensor.float()


def _same_shape(left: Grid, right: Grid) -> bool:
    return len(left) == len(right) and len(left[0]) == len(right[0])


def _dominant_color(grid: Grid) -> int:
    counts: dict[int, int] = {}
    for row in grid:
        for cell in row:
            counts[cell] = counts.get(cell, 0) + 1
    return max(counts.items(), key=lambda item: item[1])[0]


def _torch():
    try:
        import torch
    except Exception as exc:  # pragma: no cover - depends on optional torch install.
        raise RuntimeError("PyTorch is required for Mythos loss functions") from exc
    return torch

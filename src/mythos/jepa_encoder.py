"""Optional transformers-native I-JEPA image encoder for ARC grids."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from mythos.arc import Grid

ARC_PALETTE: dict[int, tuple[int, int, int]] = {
    0: (0, 0, 0),
    1: (0, 116, 217),
    2: (255, 65, 54),
    3: (46, 204, 64),
    4: (255, 220, 0),
    5: (170, 170, 170),
    6: (240, 18, 190),
    7: (255, 133, 27),
    8: (127, 219, 255),
    9: (135, 12, 37),
}


class JepaEncodingError(RuntimeError):
    """Raised when optional I-JEPA image encoding cannot run."""


def grid_to_rgb_image(grid: Grid, *, cell_px: int = 8, output_size: int = 224):
    """Rasterize an ARC grid to the RGB image shape expected by ViT-H/14."""

    try:
        from PIL import Image
    except Exception as exc:  # pragma: no cover - optional dependency.
        raise JepaEncodingError("Pillow is required for real I-JEPA grid rasterization") from exc

    if cell_px <= 0:
        raise ValueError("cell_px must be positive")
    height = len(grid)
    width = len(grid[0])
    image = Image.new("RGB", (width * cell_px, height * cell_px))
    for row_index, row in enumerate(grid):
        for col_index, color in enumerate(row):
            image.paste(
                ARC_PALETTE[int(color)],
                (
                    col_index * cell_px,
                    row_index * cell_px,
                    (col_index + 1) * cell_px,
                    (row_index + 1) * cell_px,
                ),
            )
    resampling = getattr(Image, "Resampling", Image)
    return image.resize((output_size, output_size), resampling.NEAREST)


@dataclass
class JepaImageEncoder:
    """Frozen transformers I-JEPA feature extractor."""

    model_root: Path
    device: str
    processor: object
    model: object

    @classmethod
    def from_path(cls, model_path: str | Path, *, device: str | None = None) -> "JepaImageEncoder":
        try:
            import torch
            from transformers import AutoModel, AutoProcessor
        except Exception as exc:  # pragma: no cover - optional dependency.
            raise JepaEncodingError("transformers, torch, and Pillow are required for real I-JEPA") from exc

        root = _model_root(model_path)
        selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        try:
            processor = AutoProcessor.from_pretrained(root, local_files_only=True)
            model = AutoModel.from_pretrained(root, local_files_only=True).to(selected_device)
            model.eval()
            for parameter in model.parameters():
                parameter.requires_grad = False
        except Exception as exc:  # pragma: no cover - depends on external model files.
            raise JepaEncodingError(f"failed to load transformers I-JEPA model from {root}: {exc}") from exc
        return cls(model_root=root, device=selected_device, processor=processor, model=model)

    def encode_grid(self, grid: Grid) -> tuple[float, ...]:
        try:
            import torch
        except Exception as exc:  # pragma: no cover - optional dependency.
            raise JepaEncodingError("torch is required for real I-JEPA") from exc

        image = grid_to_rgb_image(grid)
        encoded = self.processor(images=image, return_tensors="pt")
        encoded = {
            key: value.to(self.device) if hasattr(value, "to") else value
            for key, value in encoded.items()
        }
        with torch.no_grad():
            outputs = self.model(**encoded)
        hidden = getattr(outputs, "last_hidden_state", None)
        if hidden is None:
            raise JepaEncodingError("I-JEPA model output did not include last_hidden_state")
        pooled = hidden.mean(dim=1)[0].detach().float().cpu().tolist()
        return tuple(round(float(value), 6) for value in pooled)


def _model_root(model_path: str | Path) -> Path:
    path = Path(model_path)
    return path.parent if path.is_file() else path

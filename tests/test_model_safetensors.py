from __future__ import annotations

from pathlib import Path

from mythos.models import _load_torch_checkpoint


def test_safetensors_checkpoint_is_registered_as_lazy_handle(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.write_bytes(b"fake")

    loaded = _load_torch_checkpoint(checkpoint)

    assert loaded["path"] == str(checkpoint)
    assert loaded["format"] == "safetensors"
    assert loaded["lazy"] is True

from __future__ import annotations

import pytest

from mythos.jepa_encoder import ARC_PALETTE, grid_to_rgb_image


def test_grid_to_rgb_image_rasterizes_arc_palette() -> None:
    pytest.importorskip("PIL")

    image = grid_to_rgb_image([[0, 1], [2, 3]], cell_px=4, output_size=8)

    assert image.size == (8, 8)
    assert image.getpixel((0, 0)) == ARC_PALETTE[0]
    assert image.getpixel((5, 1)) == ARC_PALETTE[1]
    assert image.getpixel((1, 5)) == ARC_PALETTE[2]
    assert image.getpixel((5, 5)) == ARC_PALETTE[3]

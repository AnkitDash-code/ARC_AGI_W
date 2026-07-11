from __future__ import annotations

import torch

from mythos.losses import background_preservation_mask, genie_background_consistency_loss


def test_background_preservation_mask_uses_matching_cells_when_output_available() -> None:
    input_grid = [[0, 1], [0, 2]]
    output_grid = [[0, 3], [0, 2]]

    mask = background_preservation_mask(input_grid, output_grid)

    assert mask == ((True, False), (True, True))


def test_genie_background_consistency_loss_penalizes_wrong_preserved_color() -> None:
    input_grid = [[0, 1], [0, 1]]
    mask = ((True, False), (True, False))
    good_logits = torch.zeros(2, 2, 10)
    bad_logits = torch.zeros(2, 2, 10)
    good_logits[:, :, 0] = 5.0
    bad_logits[:, :, 2] = 5.0

    good_loss = genie_background_consistency_loss(good_logits, input_grid, preserve_mask=mask)
    bad_loss = genie_background_consistency_loss(bad_logits, input_grid, preserve_mask=mask)

    assert good_loss < bad_loss

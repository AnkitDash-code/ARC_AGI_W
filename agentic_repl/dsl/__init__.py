"""DSL surface exposed to the code-generating LLM.

Re-exports mythos's verified grid-primitive modules (mythos.objects,
mythos.symmetry, mythos.augment) as a curated, flat namespace: these
functions already carry docstrings written for a human reader (see each
module's docstring in src/mythos/), so this module doesn't re-document them
-- it just picks the stable subset worth exposing and collects them into
DSL_FUNCTIONS/DSL_CLASSES so agentic_repl.dsl.catalog can generate the
LLM-facing reference directly from the real callables (never a hand-copied,
driftable description) and agentic_repl.repl can build the restricted exec()
namespace from the same list.
"""

from __future__ import annotations

from mythos.arc import copy_grid, grid_shape
from mythos.augment import NUM_TRANSFORMS, forward_transform, inverse_transform, transform_name
from mythos.objects import (
    ArcObject,
    crop_to_object,
    d4_signature_variants,
    dominant_grid_color,
    segment_objects,
)
from mythos.symmetry import (
    crop,
    find_occlusion_color_candidates,
    hole_bbox,
    hole_cells_for_color,
    repair_grid,
    verify_symmetry,
)

DSL_FUNCTIONS = (
    segment_objects,
    crop_to_object,
    dominant_grid_color,
    d4_signature_variants,
    find_occlusion_color_candidates,
    repair_grid,
    hole_bbox,
    crop,
    hole_cells_for_color,
    verify_symmetry,
    forward_transform,
    inverse_transform,
    transform_name,
    grid_shape,
    copy_grid,
)

DSL_CLASSES = (ArcObject,)

PUBLIC_NAMESPACE: dict[str, object] = {fn.__name__: fn for fn in DSL_FUNCTIONS}
PUBLIC_NAMESPACE.update({cls.__name__: cls for cls in DSL_CLASSES})
PUBLIC_NAMESPACE["NUM_TRANSFORMS"] = NUM_TRANSFORMS

__all__ = [
    "DSL_FUNCTIONS",
    "DSL_CLASSES",
    "PUBLIC_NAMESPACE",
    *PUBLIC_NAMESPACE.keys(),
]

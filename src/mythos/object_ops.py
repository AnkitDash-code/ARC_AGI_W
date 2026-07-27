"""Object-level transform candidates for verified program search.

Each `find_*_transform` function takes a whole ArcTask and tries to build
ONE transform (Grid -> Grid) that reproduces every train pair exactly; if
it can, it returns that transform ready to apply to test inputs, else None.
This mirrors FixtureSolver's verify-before-trust pattern, just operating
over segmented objects instead of raw grid transforms -- the substrate a
one-line rule like "move this fragment into its matching slot" actually
needs, instead of reasoning over 900 flat pixels.
"""

from __future__ import annotations

from typing import Callable

from mythos.arc import ArcTask, Grid, grid_equal
from mythos.objects import ArcObject, crop_to_object, d4_signature_variants, segment_objects

Transform = Callable[[Grid], Grid]

_BACKGROUND = 0


def find_fragment_to_slot_transform(task: ArcTask) -> Transform | None:
    """Move/copy scattered fragments into shape-matching occluded slots.

    Targets the "jigsaw" ARC pattern: one or more small colored objects sit
    apart from a solid-colored placeholder region shaped like a rotated or
    mirrored copy of one of them; the true output pastes each fragment's
    colors into its matching placeholder, possibly leaving the original
    fragment in place or removing it. Both variants are tried and verified
    against every train pair before being trusted; an ambiguous match (two
    donors equally fit one slot) makes a variant refuse rather than guess.
    """

    if any(example.output is None for example in task.train):
        return None

    # Slot colors are not restricted to solid rectangles here (unlike
    # mythos.symmetry's occlusion detector): a slot mirrors a donor
    # fragment's own silhouette, which can be any shape. Verification
    # against every train pair is what filters out the false positives a
    # broad candidate set like this necessarily includes.
    slot_colors: set[int] = set()
    for example in task.train:
        slot_colors.update(color for row in example.input for color in row if color != _BACKGROUND)

    for slot_color in slot_colors:
        for keep_donor in (False, True):
            transform = _fragment_transform(slot_color, keep_donor)
            if _verifies_on_train(transform, task):
                return transform
    return None


def _fragment_transform(slot_color: int, keep_donor: bool) -> Transform:
    def transform(grid: Grid) -> Grid:
        result = _place_fragments(grid, slot_color, keep_donor)
        if result is None:
            raise _TransformFailed(f"no unambiguous fragment placement for slot color {slot_color}")
        return result

    return transform


def _place_fragments(grid: Grid, slot_color: int, keep_donor: bool) -> Grid | None:
    all_objects = segment_objects(grid, background=_BACKGROUND, connectivity=8, univalued=False)
    slots = [obj for obj in all_objects if obj.dominant_color == slot_color and obj.is_univalued]
    donors = [obj for obj in all_objects if obj.dominant_color != slot_color]
    if not slots or not donors:
        return None

    result = [row[:] for row in grid]
    matched_donor_indices: set[int] = set()
    for slot in slots:
        matches = [
            index
            for index, donor in enumerate(donors)
            if index not in matched_donor_indices and _donor_fits_slot(donor, slot)
        ]
        if len(matches) != 1:
            return None  # no fit, or an ambiguous multi-way fit -- refuse
        donor_index = matches[0]
        matched_donor_indices.add(donor_index)
        if not _paste_donor_into_slot(result, donors[donor_index], slot):
            return None
        if not keep_donor:
            for r, c in donors[donor_index].cells:
                result[r][c] = _BACKGROUND
    return result


def _donor_fits_slot(donor: ArcObject, slot: ArcObject) -> bool:
    return slot.shape_signature in d4_signature_variants(donor.shape_signature)


def _paste_donor_into_slot(result: Grid, donor: ArcObject, slot: ArcObject) -> bool:
    target_shape = slot.shape_signature
    top, left, _, _ = slot.bbox
    for variant in _colored_d4_variants(donor.colored_signature):
        variant_shape = frozenset((r, c) for r, c, _ in variant)
        if variant_shape != target_shape:
            continue
        for r, c, color in variant:
            result[top + r][left + c] = color
        return True
    return False


def _colored_d4_variants(colored_signature: frozenset[tuple[int, int, int]]) -> list[frozenset[tuple[int, int, int]]]:
    variants: list[frozenset[tuple[int, int, int]]] = []
    current = colored_signature
    for _ in range(4):
        variants.append(_normalize_colored(current))
        variants.append(_normalize_colored(frozenset((r, -c, color) for r, c, color in current)))
        current = frozenset((c, -r, color) for r, c, color in current)  # rotate 90 degrees
    return variants


def _normalize_colored(cells: frozenset[tuple[int, int, int]]) -> frozenset[tuple[int, int, int]]:
    min_r = min(r for r, _, _ in cells)
    min_c = min(c for _, c, _ in cells)
    return frozenset((r - min_r, c - min_c, color) for r, c, color in cells)


def find_crop_to_selected_object_transform(task: ArcTask) -> Transform | None:
    """Output is one train-consistently-selected object, cropped to its bounding box.

    Common ARC pattern family: "find the largest/smallest/uniquely-colored/
    uniquely-shaped object and output just that." Each selection rule is
    tried and verified against every train pair.
    """

    if any(example.output is None for example in task.train):
        return None

    for connectivity in (4, 8):
        for univalued in (True, False):
            for selector in _OBJECT_SELECTORS:
                transform = _crop_to_selected_transform(connectivity, univalued, selector)
                if _verifies_on_train(transform, task):
                    return transform
    return None


def _crop_to_selected_transform(connectivity: int, univalued: bool, selector: Callable[[list[ArcObject]], ArcObject | None]) -> Transform:
    def transform(grid: Grid) -> Grid:
        objects = segment_objects(grid, background=_BACKGROUND, connectivity=connectivity, univalued=univalued)
        selected = selector(objects)
        if selected is None:
            raise _TransformFailed("no object matched the selection rule")
        return crop_to_object(grid, selected, background=_BACKGROUND)

    return transform


def _largest_object(objects: list[ArcObject]) -> ArcObject | None:
    return max(objects, key=lambda obj: obj.size) if objects else None


def _smallest_object(objects: list[ArcObject]) -> ArcObject | None:
    return min(objects, key=lambda obj: obj.size) if objects else None


def _uniquely_colored_object(objects: list[ArcObject]) -> ArcObject | None:
    counts: dict[int, int] = {}
    for obj in objects:
        counts[obj.dominant_color] = counts.get(obj.dominant_color, 0) + 1
    unique = [obj for obj in objects if counts[obj.dominant_color] == 1]
    return unique[0] if len(unique) == 1 else None


def _uniquely_shaped_object(objects: list[ArcObject]) -> ArcObject | None:
    counts: dict[frozenset, int] = {}
    for obj in objects:
        counts[obj.shape_signature] = counts.get(obj.shape_signature, 0) + 1
    unique = [obj for obj in objects if counts[obj.shape_signature] == 1]
    return unique[0] if len(unique) == 1 else None


_OBJECT_SELECTORS = (_largest_object, _smallest_object, _uniquely_colored_object, _uniquely_shaped_object)


def find_fill_enclosed_regions_transform(task: ArcTask) -> Transform | None:
    """Fill background pockets fully enclosed by non-background cells.

    Common ARC pattern: closed outlines get their interior flood-filled
    with a consistent color. Tries a handful of train-derived fill-color
    rules (fixed color, bordering object's own color) and verifies each.
    """

    if any(example.output is None for example in task.train):
        return None

    palette: set[int] = set()
    for example in task.train:
        for row in example.output:
            palette.update(row)

    for fill_color in sorted(palette):
        transform = _fill_enclosed_transform(fill_color)
        if _verifies_on_train(transform, task):
            return transform

    transform = _fill_enclosed_transform(None)  # fill with the enclosing object's own color
    if _verifies_on_train(transform, task):
        return transform
    return None


def _fill_enclosed_transform(fill_color: int | None) -> Transform:
    def transform(grid: Grid) -> Grid:
        return _fill_enclosed_regions(grid, fill_color)

    return transform


def _fill_enclosed_regions(grid: Grid, fill_color: int | None) -> Grid:
    height = len(grid)
    width = len(grid[0]) if height else 0
    background_components = segment_objects(grid, background=None, connectivity=4, univalued=True)
    result = [row[:] for row in grid]
    for component in background_components:
        if component.dominant_color != _BACKGROUND:
            continue
        if _touches_border(component.cells, height, width):
            continue
        color = fill_color if fill_color is not None else _surrounding_color(grid, component.cells, height, width)
        if color is None:
            continue
        for r, c in component.cells:
            result[r][c] = color
    return result


def _touches_border(cells: frozenset[tuple[int, int]], height: int, width: int) -> bool:
    return any(r in (0, height - 1) or c in (0, width - 1) for r, c in cells)


def _surrounding_color(grid: Grid, cells: frozenset[tuple[int, int]], height: int, width: int) -> int | None:
    neighbor_colors: set[int] = set()
    for r, c in cells:
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                neighbor_colors.add(grid[nr][nc])
    return neighbor_colors.pop() if len(neighbor_colors) == 1 else None


class _TransformFailed(Exception):
    pass


def _verifies_on_train(transform: Transform, task: ArcTask) -> bool:
    for example in task.train:
        if example.output is None:
            return False
        try:
            predicted = transform(example.input)
        except _TransformFailed:
            return False
        if not grid_equal(predicted, example.output):
            return False
    return True


ALL_OBJECT_TRANSFORM_FINDERS = (
    find_fragment_to_slot_transform,
    find_crop_to_selected_object_transform,
    find_fill_enclosed_regions_transform,
)

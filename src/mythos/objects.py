"""Object-centric grid decomposition: connected-component segmentation.

Raw grid tokens are a poor substrate for compositional rules ("move this
fragment into its matching slot" is a one-line rule over objects and a
near-impossible one over 900 flat pixels). This module segments a grid into
connected-component objects with shape/color/position attributes, so
downstream primitives (mythos.object_ops) can operate over objects instead
of raw cells.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from mythos.arc import Grid

Cell = tuple[int, int]


@dataclass(frozen=True)
class ArcObject:
    """A connected group of non-background cells, in original grid coordinates."""

    cells: frozenset[Cell]
    colors: tuple[tuple[Cell, int], ...]  # (cell, color) pairs, sorted by cell

    @property
    def color_map(self) -> dict[Cell, int]:
        return dict(self.colors)

    @property
    def size(self) -> int:
        return len(self.cells)

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        """(top, left, height, width)."""
        rows = [r for r, _ in self.cells]
        cols = [c for _, c in self.cells]
        top, bottom = min(rows), max(rows)
        left, right = min(cols), max(cols)
        return top, left, bottom - top + 1, right - left + 1

    @property
    def dominant_color(self) -> int:
        counts = Counter(color for _, color in self.colors)
        return counts.most_common(1)[0][0]

    @property
    def is_univalued(self) -> bool:
        return len({color for _, color in self.colors}) == 1

    @property
    def shape_signature(self) -> frozenset[Cell]:
        """Cell offsets relative to the bounding-box origin, ignoring color.

        Two objects with the same signature have the same silhouette,
        independent of position -- the basis for shape-matching primitives
        like "find where this fragment's outline fits".
        """
        top, left, _, _ = self.bbox
        return frozenset((r - top, c - left) for r, c in self.cells)

    @property
    def colored_signature(self) -> frozenset[tuple[Cell, int]]:
        """Like shape_signature, but including each cell's color."""
        top, left, _, _ = self.bbox
        color_map = self.color_map
        return frozenset((r - top, c - left, color_map[(r, c)]) for r, c in self.cells)

    def translated(self, dr: int, dc: int) -> "ArcObject":
        return ArcObject(
            cells=frozenset((r + dr, c + dc) for r, c in self.cells),
            colors=tuple(sorted(((r + dr, c + dc), color) for (r, c), color in self.colors)),
        )


def segment_objects(
    grid: Grid,
    *,
    background: int | None = 0,
    connectivity: int = 4,
    univalued: bool = True,
) -> list[ArcObject]:
    """Segment a grid into connected-component objects.

    background: color treated as empty space (excluded from all objects).
        None means every cell (including the majority/background color) is
        segmented -- rarely useful, but supported for completeness.
    connectivity: 4 (edge-adjacent) or 8 (edge+diagonal-adjacent).
    univalued: if True, a component only connects cells of the *same* color
        (mirrors arc-dsl's univalued objects()); if False, any adjacent
        non-background cells join one object regardless of color (useful for
        multicolor fragments like the ones in task 16b78196).
    """

    if connectivity not in (4, 8):
        raise ValueError(f"connectivity must be 4 or 8, got {connectivity}")

    height = len(grid)
    width = len(grid[0]) if height else 0
    visited = [[False] * width for _ in range(height)]
    offsets = _NEIGHBOR_OFFSETS[connectivity]

    objects: list[ArcObject] = []
    for start_r in range(height):
        for start_c in range(width):
            if visited[start_r][start_c]:
                continue
            start_color = grid[start_r][start_c]
            if background is not None and start_color == background:
                visited[start_r][start_c] = True
                continue

            stack = [(start_r, start_c)]
            visited[start_r][start_c] = True
            component: list[Cell] = []
            while stack:
                r, c = stack.pop()
                component.append((r, c))
                cell_color = grid[r][c]
                for dr, dc in offsets:
                    nr, nc = r + dr, c + dc
                    if not (0 <= nr < height and 0 <= nc < width) or visited[nr][nc]:
                        continue
                    neighbor_color = grid[nr][nc]
                    if background is not None and neighbor_color == background:
                        continue
                    if univalued and neighbor_color != cell_color:
                        continue
                    visited[nr][nc] = True
                    stack.append((nr, nc))

            colors = tuple(sorted((cell, grid[cell[0]][cell[1]]) for cell in component))
            objects.append(ArcObject(cells=frozenset(component), colors=colors))
    return objects


_NEIGHBOR_OFFSETS = {
    4: ((-1, 0), (1, 0), (0, -1), (0, 1)),
    8: ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)),
}


def crop_to_object(grid: Grid, obj: ArcObject, *, background: int = 0) -> Grid:
    """Return the object's bounding box as a standalone grid."""

    top, left, height, width = obj.bbox
    color_map = obj.color_map
    return [
        [color_map.get((top + r, left + c), background) for c in range(width)]
        for r in range(height)
    ]


def dominant_grid_color(grid: Grid) -> int:
    counts = Counter(cell for row in grid for cell in row)
    return counts.most_common(1)[0][0]


def d4_signature_variants(signature: frozenset[Cell]) -> list[frozenset[Cell]]:
    """All 8 dihedral-group transforms of a shape signature, each renormalized to (0,0).

    Used to match a donor fragment's silhouette against a target slot's
    silhouette allowing for the donor having been rotated/reflected before
    being placed -- a common ARC pattern ("fit this piece into its outline,
    turned whichever way makes it fit").
    """

    variants: list[frozenset[Cell]] = []
    current = signature
    for _ in range(4):
        variants.append(_normalize_signature(current))
        variants.append(_normalize_signature(frozenset((r, -c) for r, c in current)))
        current = frozenset((c, -r) for r, c in current)  # rotate 90 degrees
    return variants


def _normalize_signature(cells: frozenset[Cell]) -> frozenset[Cell]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return frozenset((r - min_r, c - min_c) for r, c in cells)

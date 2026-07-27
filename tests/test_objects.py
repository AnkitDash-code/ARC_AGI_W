from mythos.objects import crop_to_object, dominant_grid_color, segment_objects


def test_segment_objects_splits_disconnected_same_color_regions() -> None:
    grid = [
        [1, 1, 0, 2],
        [0, 0, 0, 2],
        [3, 0, 1, 1],
    ]
    objects = segment_objects(grid, background=0, connectivity=4, univalued=True)
    sizes = sorted(obj.size for obj in objects)
    assert sizes == [1, 2, 2, 2]


def test_segment_objects_diagonal_connectivity_joins_diagonal_cells() -> None:
    grid = [
        [1, 0],
        [0, 1],
    ]
    objects_4 = segment_objects(grid, background=0, connectivity=4)
    objects_8 = segment_objects(grid, background=0, connectivity=8)
    assert len(objects_4) == 2
    assert len(objects_8) == 1


def test_shape_signature_is_position_independent() -> None:
    grid_a = [[0, 0, 0], [0, 5, 5], [0, 0, 5]]
    grid_b = [[5, 5, 0], [0, 5, 0], [0, 0, 0]]
    obj_a = segment_objects(grid_a, background=0)[0]
    obj_b = segment_objects(grid_b, background=0)[0]
    assert obj_a.shape_signature == obj_b.shape_signature


def test_crop_to_object_extracts_bounding_box() -> None:
    grid = [
        [0, 0, 0, 0],
        [0, 3, 3, 0],
        [0, 3, 0, 0],
    ]
    obj = segment_objects(grid, background=0, univalued=False)[0]
    assert crop_to_object(grid, obj) == [[3, 3], [3, 0]]


def test_dominant_grid_color() -> None:
    grid = [[0, 0, 1], [0, 0, 0]]
    assert dominant_grid_color(grid) == 0

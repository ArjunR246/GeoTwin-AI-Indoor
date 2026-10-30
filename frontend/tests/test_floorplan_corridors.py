import numpy as np
import pytest

from geotwin.floorplan.corridors import (
    Corridor,
    corridor_mask,
    detect_corridors,
)
from geotwin.floorplan.rooms import Room


def make_room(
    room_id: str,
    width: int,
    height: int,
    area: float = 5000.0,
) -> Room:
    """Create a deterministic Room test object."""
    return Room(
        id=room_id,
        polygon=(
            (0, 0),
            (width, 0),
            (width, height),
            (0, height),
        ),
        area=area,
        centroid=(width / 2, height / 2),
        bounding_box=(0, 0, width, height),
        aspect_ratio=width / height,
    )


def test_detect_corridors_returns_list():
    rooms = [
        make_room("room_1", 300, 50),
        make_room("room_2", 100, 100),
    ]

    corridors = detect_corridors(rooms)

    assert isinstance(corridors, list)


def test_elongated_room_is_detected_as_corridor():
    rooms = [
        make_room("room_1", 600, 100),
    ]

    corridors = detect_corridors(
        rooms,
        min_aspect_ratio=3.0,
    )

    assert len(corridors) == 1
    assert isinstance(corridors[0], Corridor)


def test_normal_room_is_not_detected_as_corridor():
    rooms = [
        make_room("room_1", 200, 180),
    ]

    corridors = detect_corridors(
        rooms,
        min_aspect_ratio=3.0,
    )

    assert len(corridors) == 0


def test_vertical_elongated_room_is_detected():
    rooms = [
        make_room("room_1", 80, 500),
    ]

    corridors = detect_corridors(
        rooms,
        min_aspect_ratio=3.0,
    )

    assert len(corridors) == 1


def test_corridor_ids_are_unique():
    rooms = [
        make_room("room_1", 600, 100),
        make_room("room_2", 500, 100),
    ]

    corridors = detect_corridors(
        rooms,
        min_aspect_ratio=3.0,
    )

    ids = [corridor.id for corridor in corridors]

    assert len(ids) == len(set(ids))


def test_corridor_geometry_is_preserved():
    room = make_room(
        "room_1",
        600,
        100,
        area=60000,
    )

    corridors = detect_corridors(
        [room],
        min_aspect_ratio=3.0,
    )

    corridor = corridors[0]

    assert corridor.polygon == room.polygon
    assert corridor.area == room.area
    assert corridor.centroid == room.centroid
    assert corridor.bounding_box == room.bounding_box


def test_max_width_filters_wide_regions():
    rooms = [
        make_room("room_1", 600, 150),
    ]

    corridors = detect_corridors(
        rooms,
        min_aspect_ratio=3.0,
        max_width=100,
    )

    assert len(corridors) == 0


def test_min_area_filters_small_regions():
    rooms = [
        make_room(
            "room_1",
            600,
            100,
            area=500,
        ),
    ]

    corridors = detect_corridors(
        rooms,
        min_aspect_ratio=3.0,
        min_area=1000,
    )

    assert len(corridors) == 0


def test_invalid_rooms_argument_is_rejected():
    with pytest.raises(TypeError):
        detect_corridors(None)


def test_invalid_aspect_ratio_is_rejected():
    with pytest.raises(ValueError):
        detect_corridors(
            [],
            min_aspect_ratio=0,
        )


def test_invalid_max_width_is_rejected():
    with pytest.raises(ValueError):
        detect_corridors(
            [],
            max_width=0,
        )


def test_corridor_mask_has_correct_shape():
    room = make_room("room_1", 600, 100)

    corridors = detect_corridors(
        [room],
        min_aspect_ratio=3.0,
    )

    mask = corridor_mask(
        (800, 1200),
        corridors,
    )

    assert isinstance(mask, np.ndarray)
    assert mask.shape == (800, 1200)
    assert mask.dtype == np.uint8


def test_corridor_mask_contains_corridor_pixels():
    room = make_room("room_1", 600, 100)

    corridors = detect_corridors(
        [room],
        min_aspect_ratio=3.0,
    )

    mask = corridor_mask(
        (800, 1200),
        corridors,
    )

    assert np.count_nonzero(mask) > 0


def test_empty_corridor_list_produces_empty_mask():
    mask = corridor_mask(
        (100, 100),
        [],
    )

    assert mask.shape == (100, 100)
    assert np.count_nonzero(mask) == 0
from pathlib import Path

import numpy as np
import pytest

from geotwin.floorplan.loader import load_floor_plan
from geotwin.floorplan.preprocessing import preprocess_floor_plan
from geotwin.floorplan.rooms import Room, detect_rooms


FIXTURE = (
    Path(__file__).parent
    / "fixtures"
    / "floorplans"
    / "simple_floorplan.png"
)


@pytest.fixture
def processed_floor_plan():
    image = load_floor_plan(FIXTURE)
    return preprocess_floor_plan(image)


def test_detect_rooms_returns_list(processed_floor_plan):
    rooms = detect_rooms(
        processed_floor_plan["binary"],
    )

    assert isinstance(rooms, list)


def test_detected_objects_are_rooms(processed_floor_plan):
    rooms = detect_rooms(
        processed_floor_plan["binary"],
    )

    assert all(isinstance(room, Room) for room in rooms)


def test_rooms_have_valid_ids(processed_floor_plan):
    rooms = detect_rooms(
        processed_floor_plan["binary"],
    )

    ids = [room.id for room in rooms]

    assert len(ids) == len(set(ids))
    assert all(room_id.startswith("room_") for room_id in ids)


def test_rooms_have_positive_area(processed_floor_plan):
    rooms = detect_rooms(
        processed_floor_plan["binary"],
    )

    assert all(room.area > 0 for room in rooms)


def test_rooms_have_valid_centroids(processed_floor_plan):
    rooms = detect_rooms(
        processed_floor_plan["binary"],
    )

    height, width = processed_floor_plan["binary"].shape

    for room in rooms:
        x, y = room.centroid

        assert 0 <= x < width
        assert 0 <= y < height


def test_rooms_have_valid_bounding_boxes(processed_floor_plan):
    rooms = detect_rooms(
        processed_floor_plan["binary"],
    )

    height, width = processed_floor_plan["binary"].shape

    for room in rooms:
        x, y, room_width, room_height = room.bounding_box

        assert 0 <= x < width
        assert 0 <= y < height
        assert room_width > 0
        assert room_height > 0
        assert x + room_width <= width
        assert y + room_height <= height


def test_room_polygon_has_at_least_three_points(processed_floor_plan):
    rooms = detect_rooms(
        processed_floor_plan["binary"],
    )

    for room in rooms:
        assert len(room.polygon) >= 3


def test_room_aspect_ratio_is_positive(processed_floor_plan):
    rooms = detect_rooms(
        processed_floor_plan["binary"],
    )

    assert all(room.aspect_ratio > 0 for room in rooms)


def test_empty_image_is_rejected():
    with pytest.raises(ValueError):
        detect_rooms(np.array([], dtype=np.uint8))


def test_multichannel_image_is_rejected():
    image = np.zeros(
        (100, 100, 3),
        dtype=np.uint8,
    )

    with pytest.raises(ValueError):
        detect_rooms(image)
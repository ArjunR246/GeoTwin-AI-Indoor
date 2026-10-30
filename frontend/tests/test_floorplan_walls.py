from pathlib import Path

import numpy as np
import pytest

from geotwin.floorplan.loader import load_floor_plan
from geotwin.floorplan.preprocessing import preprocess_floor_plan
from geotwin.floorplan.walls import WallSegment, detect_walls


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


def test_detect_walls_returns_list(processed_floor_plan):
    walls = detect_walls(
        processed_floor_plan["binary"],
        processed_floor_plan["edges"],
    )

    assert isinstance(walls, list)
    assert len(walls) > 0


def test_detected_objects_are_wall_segments(processed_floor_plan):
    walls = detect_walls(
        processed_floor_plan["binary"],
        processed_floor_plan["edges"],
    )

    assert all(isinstance(wall, WallSegment) for wall in walls)


def test_wall_orientation_is_valid(processed_floor_plan):
    walls = detect_walls(
        processed_floor_plan["binary"],
        processed_floor_plan["edges"],
    )

    orientations = {wall.orientation for wall in walls}

    assert orientations <= {"horizontal", "vertical"}


def test_both_wall_orientations_are_detected(processed_floor_plan):
    walls = detect_walls(
        processed_floor_plan["binary"],
        processed_floor_plan["edges"],
    )

    orientations = {wall.orientation for wall in walls}

    assert "horizontal" in orientations
    assert "vertical" in orientations


def test_wall_lengths_are_positive(processed_floor_plan):
    walls = detect_walls(
        processed_floor_plan["binary"],
        processed_floor_plan["edges"],
    )

    assert all(wall.length > 0 for wall in walls)


def test_wall_coordinates_are_valid(processed_floor_plan):
    walls = detect_walls(
        processed_floor_plan["binary"],
        processed_floor_plan["edges"],
    )

    height, width = processed_floor_plan["binary"].shape

    for wall in walls:
        assert 0 <= wall.x1 < width
        assert 0 <= wall.x2 < width
        assert 0 <= wall.y1 < height
        assert 0 <= wall.y2 < height


def test_midpoint_is_correct():
    wall = WallSegment(
        x1=10,
        y1=20,
        x2=110,
        y2=20,
        orientation="horizontal",
        length=100.0,
    )

    assert wall.midpoint == (60.0, 20.0)


def test_empty_binary_is_rejected():
    with pytest.raises(ValueError):
        detect_walls(np.array([], dtype=np.uint8))
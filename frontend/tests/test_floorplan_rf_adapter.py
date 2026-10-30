import cv2
import numpy as np
import pytest

from geotwin.floorplan.coordinates import CoordinateTransform
from geotwin.floorplan.pipeline import process_floor_plan
from geotwin.floorplan.rf_adapter import (
    floor_plan_to_rf_geometry,
    transform_wall,
)
from geotwin.floorplan.walls import WallSegment


def make_floor_plan(
    width=400,
    height=300,
):
    image = np.full(
        (height, width, 3),
        255,
        dtype=np.uint8,
    )

    cv2.rectangle(
        image,
        (10, 10),
        (width - 10, height - 10),
        (0, 0, 0),
        4,
    )

    cv2.line(
        image,
        (width // 2, 10),
        (width // 2, height - 10),
        (0, 0, 0),
        4,
    )

    cv2.line(
        image,
        (width // 2, height // 2),
        (width - 10, height // 2),
        (0, 0, 0),
        4,
    )

    return image


def test_transform_wall_converts_coordinates():
    wall = WallSegment(
        x1=0,
        y1=0,
        x2=100,
        y2=0,
        orientation="horizontal",
        length=100.0,
    )

    transform = CoordinateTransform(
        image_width_px=1000,
        image_height_px=500,
        physical_width_m=20.0,
        physical_height_m=10.0,
    )

    result = transform_wall(
        wall,
        transform,
    )

    assert result.x1 == pytest.approx(0.0)
    assert result.y1 == pytest.approx(0.0)
    assert result.x2 == pytest.approx(2.0)
    assert result.y2 == pytest.approx(0.0)
    assert result.length == pytest.approx(2.0)


def test_floor_plan_to_rf_geometry_returns_physical_positions():
    image = make_floor_plan()

    result = process_floor_plan(
        image,
        physical_width_m=20.0,
        physical_height_m=15.0,
    )

    (
        receiver_positions,
        valid_ap_positions,
        blocked_positions,
        physical_walls,
    ) = floor_plan_to_rf_geometry(
        building=result.building,
        coordinates=result.coordinates,
        cell_size_m=1.0,
    )

    assert receiver_positions
    assert valid_ap_positions
    assert physical_walls

    for position in receiver_positions:
        assert 0.0 <= position[0] <= 20.0
        assert 0.0 <= position[1] <= 15.0

    for position in valid_ap_positions:
        assert 0.0 <= position[0] <= 20.0
        assert 0.0 <= position[1] <= 15.0


def test_floor_plan_to_rf_geometry_requires_positive_cell_size():
    image = make_floor_plan()

    result = process_floor_plan(
        image,
        physical_width_m=20.0,
        physical_height_m=15.0,
    )

    with pytest.raises(ValueError):
        floor_plan_to_rf_geometry(
            building=result.building,
            coordinates=result.coordinates,
            cell_size_m=0.0,
        )


def test_floor_plan_to_rf_geometry_requires_building():
    transform = CoordinateTransform(
        image_width_px=100,
        image_height_px=100,
        physical_width_m=10.0,
        physical_height_m=10.0,
    )

    with pytest.raises(TypeError):
        floor_plan_to_rf_geometry(
            building=None,
            coordinates=transform,
            cell_size_m=1.0,
        )
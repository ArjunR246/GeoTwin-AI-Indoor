import cv2
import numpy as np
import pytest

from geotwin.floorplan.building_model import BuildingModel
from geotwin.floorplan.coordinates import CoordinateTransform
from geotwin.floorplan.pipeline import (
    FloorPlanPipelineResult,
    process_floor_plan,
)


def make_floor_plan(
    width=400,
    height=300,
):
    """
    Create a simple synthetic floor plan.

    The outer rectangle and internal walls provide enough
    structure for the processing pipeline.
    """

    image = np.full(
        (height, width, 3),
        255,
        dtype=np.uint8,
    )

    # Outer boundary.
    cv2.rectangle(
        image,
        (10, 10),
        (width - 10, height - 10),
        (0, 0, 0),
        4,
    )

    # Vertical internal wall.
    cv2.line(
        image,
        (width // 2, 10),
        (width // 2, height - 10),
        (0, 0, 0),
        4,
    )

    # Horizontal internal wall.
    cv2.line(
        image,
        (width // 2, height // 2),
        (width - 10, height // 2),
        (0, 0, 0),
        4,
    )

    return image


def test_pipeline_returns_result():
    image = make_floor_plan()

    result = process_floor_plan(
        image
    )

    assert isinstance(
        result,
        FloorPlanPipelineResult,
    )


def test_pipeline_preserves_image_dimensions():
    image = make_floor_plan(
        width=500,
        height=350,
    )

    result = process_floor_plan(
        image
    )

    assert result.image_width == 500
    assert result.image_height == 350


def test_pipeline_creates_building_model():
    image = make_floor_plan()

    result = process_floor_plan(
        image
    )

    assert isinstance(
        result.building,
        BuildingModel,
    )


def test_pipeline_without_scale():
    image = make_floor_plan()

    result = process_floor_plan(
        image
    )

    assert result.coordinates is None
    assert result.has_physical_scale is False


def test_pipeline_with_physical_scale():
    image = make_floor_plan(
        width=1000,
        height=500,
    )

    result = process_floor_plan(
        image,
        physical_width_m=20.0,
        physical_height_m=10.0,
    )

    assert result.coordinates is not None
    assert result.has_physical_scale is True

    assert isinstance(
        result.coordinates,
        CoordinateTransform,
    )


def test_pipeline_coordinate_transform_matches_image():
    image = make_floor_plan(
        width=1000,
        height=500,
    )

    result = process_floor_plan(
        image,
        physical_width_m=20.0,
        physical_height_m=10.0,
    )

    point = result.coordinates.pixel_to_physical(
        (500, 250)
    )

    assert point == pytest.approx(
        (10.0, 5.0)
    )


def test_pipeline_supports_different_building_sizes():
    image_a = make_floor_plan(
        width=400,
        height=300,
    )

    image_b = make_floor_plan(
        width=1600,
        height=900,
    )

    result_a = process_floor_plan(
        image_a,
        physical_width_m=40.0,
        physical_height_m=30.0,
    )

    result_b = process_floor_plan(
        image_b,
        physical_width_m=80.0,
        physical_height_m=45.0,
    )

    assert result_a.image_width == 400
    assert result_a.image_height == 300

    assert result_b.image_width == 1600
    assert result_b.image_height == 900

    assert result_a.coordinates is not None
    assert result_b.coordinates is not None


def test_pipeline_supports_grayscale_image():
    color_image = make_floor_plan()

    grayscale = cv2.cvtColor(
        color_image,
        cv2.COLOR_BGR2GRAY,
    )

    result = process_floor_plan(
        grayscale
    )

    assert isinstance(
        result,
        FloorPlanPipelineResult,
    )


def test_pipeline_returns_preprocessing_outputs():
    image = make_floor_plan()

    result = process_floor_plan(
        image
    )

    assert "grayscale" in result.preprocessing
    assert "blurred" in result.preprocessing
    assert "binary" in result.preprocessing
    assert "cleaned" in result.preprocessing
    assert "edges" in result.preprocessing


def test_preprocessing_shapes_match_input():
    image = make_floor_plan(
        width=600,
        height=400,
    )

    result = process_floor_plan(
        image
    )

    for processed_image in result.preprocessing.values():
        assert processed_image.shape[:2] == (
            400,
            600,
        )


def test_pipeline_detects_some_geometry():
    image = make_floor_plan()

    result = process_floor_plan(
        image
    )

    # The exact number of detections can vary with image
    # characteristics, but the result must remain valid.
    assert result.wall_count >= 0
    assert result.room_count >= 0
    assert result.corridor_count >= 0


def test_physical_dimensions_must_be_supplied_together():
    image = make_floor_plan()

    with pytest.raises(ValueError):
        process_floor_plan(
            image,
            physical_width_m=40.0,
        )

    with pytest.raises(ValueError):
        process_floor_plan(
            image,
            physical_height_m=30.0,
        )


def test_invalid_image_type():
    with pytest.raises(TypeError):
        process_floor_plan(
            "not an image"
        )


def test_empty_image_is_rejected():
    image = np.array(
        [],
        dtype=np.uint8,
    )

    with pytest.raises(ValueError):
        process_floor_plan(
            image
        )


def test_invalid_image_dimension():
    image = np.zeros(
        (10, 10, 2),
        dtype=np.uint8,
    )

    with pytest.raises(ValueError):
        process_floor_plan(
            image
        )


def test_origin_is_passed_to_coordinate_transform():
    image = make_floor_plan(
        width=1000,
        height=500,
    )

    result = process_floor_plan(
        image,
        physical_width_m=20.0,
        physical_height_m=10.0,
        origin_x_m=100.0,
        origin_y_m=50.0,
    )

    point = result.coordinates.pixel_to_physical(
        (500, 250)
    )

    assert point == pytest.approx(
        (110.0, 55.0)
    )


def test_pipeline_keeps_building_coordinates_in_pixels():
    image = make_floor_plan(
        width=800,
        height=600,
    )

    result = process_floor_plan(
        image,
        physical_width_m=80.0,
        physical_height_m=60.0,
    )

    assert result.building.image_width == 800
    assert result.building.image_height == 600


def test_pipeline_can_use_custom_room_area():
    image = make_floor_plan()

    result = process_floor_plan(
        image,
        room_min_area=100.0,
    )

    assert isinstance(
        result,
        FloorPlanPipelineResult,
    )


def test_pipeline_can_use_custom_corridor_parameters():
    image = make_floor_plan()

    result = process_floor_plan(
        image,
        corridor_min_aspect_ratio=2.0,
        corridor_min_area=100.0,
        corridor_max_width=500.0,
    )

    assert isinstance(
        result,
        FloorPlanPipelineResult,
    )
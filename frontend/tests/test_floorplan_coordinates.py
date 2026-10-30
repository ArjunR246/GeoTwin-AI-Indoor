import pytest

from geotwin.floorplan.coordinates import (
    CoordinateTransform,
    create_coordinate_transform,
)


@pytest.fixture
def transform():
    return CoordinateTransform(
        image_width_px=1000,
        image_height_px=500,
        physical_width_m=20.0,
        physical_height_m=10.0,
    )


def test_coordinate_transform_creation(transform):
    assert transform.image_width_px == 1000
    assert transform.image_height_px == 500
    assert transform.physical_width_m == 20.0
    assert transform.physical_height_m == 10.0


def test_metres_per_pixel(transform):
    assert transform.metres_per_pixel_x == pytest.approx(0.02)
    assert transform.metres_per_pixel_y == pytest.approx(0.02)


def test_pixels_per_metre(transform):
    assert transform.pixels_per_metre_x == pytest.approx(50.0)
    assert transform.pixels_per_metre_y == pytest.approx(50.0)


def test_uniform_scale(transform):
    assert transform.is_uniform_scale is True


def test_pixel_to_physical(transform):
    result = transform.pixel_to_physical(
        (500, 250)
    )

    assert result == pytest.approx(
        (10.0, 5.0)
    )


def test_physical_to_pixel(transform):
    result = transform.physical_to_pixel(
        (10.0, 5.0)
    )

    assert result == pytest.approx(
        (500.0, 250.0)
    )


def test_round_trip_pixel_conversion(transform):
    original = (275.0, 125.0)

    physical = transform.pixel_to_physical(
        original
    )

    restored = transform.physical_to_pixel(
        physical
    )

    assert restored == pytest.approx(
        original
    )


def test_round_trip_physical_conversion(transform):
    original = (7.5, 3.25)

    pixels = transform.physical_to_pixel(
        original
    )

    restored = transform.pixel_to_physical(
        pixels
    )

    assert restored == pytest.approx(
        original
    )


def test_pixel_distance_to_metres(transform):
    distance = transform.pixel_distance_to_metres(
        300,
        400,
    )

    assert distance == pytest.approx(10.0)


def test_metres_distance_to_pixels(transform):
    distance = transform.metres_distance_to_pixels(
        6,
        8,
    )

    assert distance == pytest.approx(500.0)


def test_pixel_segment_length_to_metres(transform):
    length = transform.pixel_segment_length_to_metres(
        (0, 0),
        (300, 400),
    )

    assert length == pytest.approx(10.0)


def test_physical_segment_length_to_pixels(transform):
    length = transform.physical_segment_length_to_pixels(
        (0, 0),
        (6, 8),
    )

    assert length == pytest.approx(500.0)


def test_origin_offset():
    transform = CoordinateTransform(
        image_width_px=1000,
        image_height_px=500,
        physical_width_m=20.0,
        physical_height_m=10.0,
        origin_x_m=100.0,
        origin_y_m=50.0,
    )

    result = transform.pixel_to_physical(
        (500, 250)
    )

    assert result == pytest.approx(
        (110.0, 55.0)
    )


def test_non_uniform_scale():
    transform = CoordinateTransform(
        image_width_px=1000,
        image_height_px=500,
        physical_width_m=20.0,
        physical_height_m=5.0,
    )

    assert transform.is_uniform_scale is False

    result = transform.pixel_to_physical(
        (500, 250)
    )

    assert result == pytest.approx(
        (10.0, 2.5)
    )


def test_factory_function():
    transform = create_coordinate_transform(
        image_width_px=1000,
        image_height_px=500,
        physical_width_m=20.0,
        physical_height_m=10.0,
    )

    assert isinstance(
        transform,
        CoordinateTransform,
    )


def test_invalid_image_width():
    with pytest.raises(ValueError):
        CoordinateTransform(
            image_width_px=0,
            image_height_px=500,
            physical_width_m=20.0,
            physical_height_m=10.0,
        )


def test_invalid_image_height():
    with pytest.raises(ValueError):
        CoordinateTransform(
            image_width_px=1000,
            image_height_px=0,
            physical_width_m=20.0,
            physical_height_m=10.0,
        )


def test_invalid_physical_width():
    with pytest.raises(ValueError):
        CoordinateTransform(
            image_width_px=1000,
            image_height_px=500,
            physical_width_m=0,
            physical_height_m=10.0,
        )


def test_invalid_physical_height():
    with pytest.raises(ValueError):
        CoordinateTransform(
            image_width_px=1000,
            image_height_px=500,
            physical_width_m=20.0,
            physical_height_m=0,
        )


def test_invalid_point_type(transform):
    with pytest.raises(TypeError):
        transform.pixel_to_physical(
            "invalid"
        )


def test_invalid_point_length(transform):
    with pytest.raises(ValueError):
        transform.pixel_to_physical(
            (10,)
        )


def test_invalid_point_coordinates(transform):
    with pytest.raises(TypeError):
        transform.pixel_to_physical(
            ("x", 10)
        )


def test_negative_pixel_distance_x(transform):
    with pytest.raises(ValueError):
        transform.pixel_distance_to_metres(
            -1,
            10,
        )


def test_negative_pixel_distance_y(transform):
    with pytest.raises(ValueError):
        transform.pixel_distance_to_metres(
            10,
            -1,
        )


def test_negative_metres_distance_x(transform):
    with pytest.raises(ValueError):
        transform.metres_distance_to_pixels(
            -1,
            10,
        )


def test_negative_metres_distance_y(transform):
    with pytest.raises(ValueError):
        transform.metres_distance_to_pixels(
            10,
            -1,
        )
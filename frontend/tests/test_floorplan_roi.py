import numpy as np
import pytest

from geotwin.floorplan.pipeline import process_floor_plan


def test_roi_records_original_dimensions_and_offset():
    image = np.full((100, 120, 3), 255, dtype=np.uint8)

    result = process_floor_plan(
        image,
        roi=(10, 20, 60, 50),
    )

    assert result.image_width == 120
    assert result.image_height == 100
    assert result.roi == (10, 20, 60, 50)
    assert result.coordinate_offset == (10, 20)

    assert result.preprocessing["binary"].shape == (50, 60)


def test_roi_none_preserves_full_image():
    image = np.full((100, 120, 3), 255, dtype=np.uint8)

    result = process_floor_plan(image)

    assert result.roi is None
    assert result.coordinate_offset == (0, 0)
    assert result.preprocessing["binary"].shape == (100, 120)


@pytest.mark.parametrize(
    "roi",
    [
        (-1, 0, 20, 20),
        (0, -1, 20, 20),
        (0, 0, 0, 20),
        (0, 0, 20, 0),
        (110, 0, 20, 20),
        (0, 90, 20, 20),
    ],
)
def test_invalid_roi_is_rejected(roi):
    image = np.full((100, 120, 3), 255, dtype=np.uint8)

    with pytest.raises(ValueError):
        process_floor_plan(image, roi=roi)
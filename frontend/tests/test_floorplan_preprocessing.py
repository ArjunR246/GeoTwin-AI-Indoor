from pathlib import Path

import cv2
import numpy as np

from geotwin.floorplan.loader import load_floor_plan
from geotwin.floorplan.preprocessing import (
    preprocess_floor_plan,
    to_grayscale,
)


FIXTURE = (
    Path(__file__).parent
    / "fixtures"
    / "floorplans"
    / "simple_floorplan.png"
)


def test_to_grayscale():
    image = load_floor_plan(FIXTURE)

    grayscale = to_grayscale(image)

    assert isinstance(grayscale, np.ndarray)
    assert grayscale.ndim == 2
    assert grayscale.shape == image.shape[:2]
    assert grayscale.dtype == np.uint8


def test_grayscale_input_is_supported():
    image = load_floor_plan(FIXTURE)

    grayscale = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    result = to_grayscale(grayscale)

    assert result.ndim == 2
    assert result.shape == grayscale.shape
    assert np.array_equal(result, grayscale)


def test_preprocessing_returns_all_stages():
    image = load_floor_plan(FIXTURE)

    result = preprocess_floor_plan(image)

    expected_keys = {
        "grayscale",
        "blurred",
        "binary",
        "cleaned",
        "edges",
    }

    assert set(result.keys()) == expected_keys


def test_preprocessing_preserves_image_dimensions():
    image = load_floor_plan(FIXTURE)

    result = preprocess_floor_plan(image)

    expected_shape = image.shape[:2]

    for stage in result.values():
        assert stage.shape == expected_shape


def test_preprocessing_outputs_uint8_images():
    image = load_floor_plan(FIXTURE)

    result = preprocess_floor_plan(image)

    for stage in result.values():
        assert stage.dtype == np.uint8


def test_preprocessing_produces_non_empty_output():
    image = load_floor_plan(FIXTURE)

    result = preprocess_floor_plan(image)

    for name, stage in result.items():
        assert stage.size > 0, f"{name} output is empty."


def test_edge_detection_produces_edges():
    image = load_floor_plan(FIXTURE)

    result = preprocess_floor_plan(image)

    edges = result["edges"]

    assert np.count_nonzero(edges) > 0
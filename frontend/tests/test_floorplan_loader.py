from pathlib import Path

import cv2
import numpy as np
import pymupdf
import pytest

from geotwin.floorplan.loader import (
    FloorPlanLoadError,
    load_floor_plan,
)


FIXTURE = (
    Path(__file__).parent
    / "fixtures"
    / "floorplans"
    / "simple_floorplan.png"
)


def test_floor_plan_loads():
    image = load_floor_plan(FIXTURE)

    assert image is not None
    assert image.size > 0
    assert len(image.shape) == 3
    assert image.shape[2] == 3


def test_missing_floor_plan_is_rejected(tmp_path):
    missing_file = tmp_path / "missing.png"

    with pytest.raises(FloorPlanLoadError):
        load_floor_plan(missing_file)


def test_unsupported_extension_is_rejected(tmp_path):
    invalid_file = tmp_path / "floor_plan.txt"

    invalid_file.write_text(
        "not an image"
    )

    with pytest.raises(FloorPlanLoadError):
        load_floor_plan(invalid_file)


def test_corrupt_image_is_rejected(tmp_path):
    corrupt_file = tmp_path / "corrupt.png"

    corrupt_file.write_bytes(
        b"this is not a valid image"
    )

    with pytest.raises(FloorPlanLoadError):
        load_floor_plan(corrupt_file)


def test_loaded_image_is_valid_opencv_image():
    image = load_floor_plan(FIXTURE)

    assert image.dtype.name == "uint8"
    assert image.shape[0] == 800
    assert image.shape[1] == 1200


def test_jpg_floor_plan_loads(tmp_path):
    image = cv2.imread(
        str(FIXTURE),
        cv2.IMREAD_COLOR,
    )

    jpg_file = tmp_path / "floor_plan.jpg"

    success = cv2.imwrite(
        str(jpg_file),
        image,
    )

    assert success is True

    loaded = load_floor_plan(jpg_file)

    assert loaded is not None
    assert loaded.size > 0
    assert loaded.dtype == np.uint8
    assert loaded.ndim == 3
    assert loaded.shape[2] == 3


def test_jpeg_floor_plan_loads(tmp_path):
    image = cv2.imread(
        str(FIXTURE),
        cv2.IMREAD_COLOR,
    )

    jpeg_file = tmp_path / "floor_plan.jpeg"

    success = cv2.imwrite(
        str(jpeg_file),
        image,
    )

    assert success is True

    loaded = load_floor_plan(jpeg_file)

    assert loaded is not None
    assert loaded.size > 0
    assert loaded.dtype == np.uint8
    assert loaded.ndim == 3
    assert loaded.shape[2] == 3


def test_pdf_floor_plan_loads(tmp_path):
    pdf_file = tmp_path / "floor_plan.pdf"

    document = pymupdf.open()

    page = document.new_page(
        width=600,
        height=400,
    )

    page.insert_text(
        (50, 50),
        "GeoTwin AI Indoor Test Floor Plan",
    )

    page.draw_rect(
        pymupdf.Rect(
            50,
            80,
            550,
            350,
        ),
        color=(0, 0, 0),
        width=3,
    )

    page.draw_line(
        pymupdf.Point(300, 80),
        pymupdf.Point(300, 350),
        color=(0, 0, 0),
        width=3,
    )

    document.save(
        str(pdf_file)
    )

    document.close()

    loaded = load_floor_plan(pdf_file)

    assert loaded is not None
    assert loaded.size > 0
    assert loaded.dtype == np.uint8
    assert loaded.ndim == 3
    assert loaded.shape[2] == 3


def test_invalid_pdf_is_rejected(tmp_path):
    """
    A PDF containing no usable floor-plan rendering should
    eventually be rejected by the loader.

    Since PyMuPDF cannot save a zero-page PDF, use a file that
    is not a valid PDF instead.
    """

    pdf_file = tmp_path / "invalid.pdf"

    pdf_file.write_bytes(
        b"this is not a valid pdf"
    )

    with pytest.raises(FloorPlanLoadError):
        load_floor_plan(pdf_file)


def test_corrupt_pdf_is_rejected(tmp_path):
    pdf_file = tmp_path / "corrupt.pdf"

    pdf_file.write_bytes(
        b"not really a pdf document"
    )

    with pytest.raises(FloorPlanLoadError):
        load_floor_plan(pdf_file)


def test_pdf_uses_first_page(tmp_path):
    pdf_file = tmp_path / "multi_page.pdf"

    document = pymupdf.open()

    document.new_page(
        width=400,
        height=300,
    )

    document.new_page(
        width=800,
        height=600,
    )

    document.save(
        str(pdf_file)
    )

    document.close()

    loaded = load_floor_plan(pdf_file)

    assert loaded is not None
    assert loaded.size > 0

    # PyMuPDF rounds rendered dimensions to whole pixels.
    expected_width = round(
        400 * 150 / 72
    )

    expected_height = round(
        300 * 150 / 72
    )

    # PyMuPDF's pixmap rasterization can round upward,
    # so allow the adjacent pixel.
    assert abs(
        loaded.shape[1] - expected_width
    ) <= 1

    assert abs(
        loaded.shape[0] - expected_height
    ) <= 1
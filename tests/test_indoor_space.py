"""Tests for detect_indoor_space: corridors count as usable, outside things do not."""

import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "frontend/src"))

from geotwin.floorplan.indoor import detect_indoor_space
from geotwin.floorplan.pipeline import process_floor_plan
from geotwin.floorplan.preprocessing import preprocess_floor_plan


def draw_building(scale: float = 1.0):
    """Two rooms with doors onto an open corridor, an entrance gap in the
    outer wall, and a separate 'EXIT' sign box outside the building."""
    s = lambda v: int(round(v * scale))
    image = np.full((s(700), s(1000), 3), 255, np.uint8)
    t = max(3, s(8))
    black = (0, 0, 0)
    # Outer wall with an entrance gap on the left side (y 380..440).
    cv2.line(image, (s(100), s(100)), (s(800), s(100)), black, t)
    cv2.line(image, (s(100), s(600)), (s(800), s(600)), black, t)
    cv2.line(image, (s(800), s(100)), (s(800), s(600)), black, t)
    cv2.line(image, (s(100), s(100)), (s(100), s(380)), black, t)
    cv2.line(image, (s(100), s(440)), (s(100), s(600)), black, t)
    # Corridor wall between rooms (top) and corridor (bottom), with two doors.
    cv2.line(image, (s(100), s(350)), (s(250), s(350)), black, t)
    cv2.line(image, (s(310), s(350)), (s(550), s(350)), black, t)
    cv2.line(image, (s(610), s(350)), (s(800), s(350)), black, t)
    cv2.line(image, (s(450), s(100)), (s(450), s(350)), black, t)
    # A sign box outside the building (100 px away), with text in it.
    cv2.rectangle(image, (s(900), s(300)), (s(990), s(380)), black, t)
    cv2.putText(image, "EXIT", (s(910), s(350)), cv2.FONT_HERSHEY_SIMPLEX, 0.7 * scale, black, max(1, s(2)))
    return image, s


def usable(mask, s, x, y):
    return bool(mask[s(y), s(x)])


def test_open_corridor_rooms_inside_and_outside_areas():
    image, s = draw_building()
    mask = detect_indoor_space(preprocess_floor_plan(image)["binary"])
    assert usable(mask, s, 300, 220)        # room 1
    assert usable(mask, s, 650, 220)        # room 2
    assert usable(mask, s, 450, 480)        # open corridor
    assert not usable(mask, s, 50, 50)      # outside, top-left
    assert not usable(mask, s, 900, 600)    # outside, bottom-right
    assert not usable(mask, s, 945, 330)    # inside the EXIT sign box
    assert not usable(mask, s, 450, 100)    # on the outer wall itself


def test_same_result_at_different_image_sizes():
    for scale in (0.6, 2.0):
        image, s = draw_building(scale)
        mask = detect_indoor_space(preprocess_floor_plan(image)["binary"])
        assert usable(mask, s, 450, 480)
        assert not usable(mask, s, 945, 330)


def test_pipeline_returns_indoor_mask_matching_image_size():
    image, _ = draw_building()
    result = process_floor_plan(image)
    assert result.indoor_mask is not None
    assert result.indoor_mask.shape == image.shape[:2]
    assert 0.2 < (result.indoor_mask > 0).mean() < 0.8


def test_simple_fixture_corridor_is_usable():
    image = cv2.imread(str(PROJECT_ROOT / "frontend/tests/fixtures/floorplans/simple_floorplan.png"))
    mask = detect_indoor_space(preprocess_floor_plan(image)["binary"])
    assert mask[400, 600] > 0      # middle of the corridor
    assert mask[20, 20] == 0       # white margin outside the building

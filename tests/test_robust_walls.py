"""Tests for detect_walls_robust: real walls are found, text/furniture is ignored."""

import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "frontend/src"))

from geotwin.floorplan.preprocessing import preprocess_floor_plan
from geotwin.floorplan.walls import detect_walls_robust

FIXTURES = PROJECT_ROOT / "frontend/tests/fixtures/floorplans"


def draw_plan(scale: float = 1.0):
    """A small plan with 6 walls, a text label, desks and a door arc."""
    s = lambda v: int(round(v * scale))
    image = np.full((s(600), s(800), 3), 255, np.uint8)
    t = max(3, s(8))
    walls = [
        (s(50), s(50), s(750), s(50)), (s(50), s(550), s(750), s(550)),
        (s(50), s(50), s(50), s(550)), (s(750), s(50), s(750), s(550)),
        (s(400), s(50), s(400), s(550)), (s(50), s(300), s(400), s(300)),
    ]
    for x1, y1, x2, y2 in walls:
        cv2.line(image, (x1, y1), (x2, y2), (0, 0, 0), t)
    cv2.putText(image, "COMPUTER LAB", (s(450), s(200)), cv2.FONT_HERSHEY_SIMPLEX,
                scale, (0, 0, 0), max(1, s(2)))
    for row in range(4):                       # a grid of thin desks
        for col in range(5):
            x, y = s(460 + col * 55), s(300 + row * 50)
            cv2.rectangle(image, (x, y), (x + s(35), y + s(20)), (0, 0, 0), max(1, s(2)))
    cv2.ellipse(image, (s(400), s(400)), (s(40), s(40)), 0, 0, 90, (0, 0, 0), max(1, s(2)))
    return image, walls, t


def distance_to_walls(point, walls):
    px, py = point
    best = np.inf
    for x1, y1, x2, y2 in walls:
        cx = min(max(px, min(x1, x2)), max(x1, x2))
        cy = min(max(py, min(y1, y2)), max(y1, y2))
        best = min(best, np.hypot(px - cx, py - cy))
    return best


def detect(image):
    return detect_walls_robust(preprocess_floor_plan(image)["binary"])


def test_finds_real_walls_and_ignores_text_and_desks():
    image, true_walls, t = draw_plan()
    found = detect(image)
    assert len(found) == len(true_walls)
    # Every detected wall lies on a real wall (no walls on text or desks).
    for wall in found:
        assert distance_to_walls(wall.midpoint, true_walls) <= t


def test_same_result_at_different_image_sizes():
    small = detect(draw_plan(scale=0.75)[0])
    large = detect(draw_plan(scale=2.0)[0])
    assert len(small) == len(large)


def test_simple_fixture_has_exactly_its_eight_walls():
    image = cv2.imread(str(FIXTURES / "simple_floorplan.png"))
    walls = detect(image)
    assert len(walls) == 8
    assert {w.orientation for w in walls} == {"horizontal", "vertical"}


def test_inverted_binary_mask_is_handled():
    # If the mask arrives with the colours flipped (lines white, background
    # black), the detector flips it back before looking for walls.
    image, true_walls, _ = draw_plan()
    binary = preprocess_floor_plan(image)["binary"]
    found = detect_walls_robust(255 - binary)
    assert len(found) == len(true_walls)

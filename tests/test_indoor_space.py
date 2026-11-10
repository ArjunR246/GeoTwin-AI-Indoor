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


# --- Outside pockets and AP edge margin ---------------------------------------

from geotwin.floorplan.indoor import interior_distance_m, remove_background_pockets
from geotwin.floorplan.pipeline import process_floor_plan as _process
from geotwin.floorplan.rf_adapter import floor_plan_to_rf_geometry


def photo_like_plan():
    """Beige building on a green background. A dark callout line and a sign
    box close off a pocket of green outside the right wall."""
    image = np.full((700, 1000, 3), (60, 120, 60), np.uint8)        # green (BGR)
    cv2.rectangle(image, (100, 100), (800, 600), (170, 210, 225), -1)  # beige floor
    black = (30, 30, 30)
    cv2.rectangle(image, (100, 100), (800, 600), black, 8)
    cv2.line(image, (450, 100), (450, 600), black, 8)
    # pocket outside the right wall, closed by a line and a sign box
    cv2.line(image, (800, 250), (990, 250), black, 6)
    cv2.line(image, (800, 450), (990, 450), black, 6)
    cv2.line(image, (990, 250), (990, 450), black, 6)
    return image


def test_background_coloured_pocket_is_removed():
    image = photo_like_plan()
    mask = _process(image).indoor_mask
    assert mask[350, 300] > 0          # inside the building
    assert mask[350, 650] > 0          # other room
    assert mask[350, 900] == 0         # green pocket outside the right wall


def test_white_plans_are_not_changed_by_colour_rule():
    image, _ = draw_building()
    binary = preprocess_floor_plan(image)["binary"]
    indoor = detect_indoor_space(binary)
    assert np.array_equal(remove_background_pockets(image, binary, indoor), indoor)


def test_aps_are_kept_away_from_the_outer_edge():
    image = photo_like_plan()
    result = _process(image, physical_width_m=50.0, physical_height_m=35.0)
    receivers, ap_spots, _, _ = floor_plan_to_rf_geometry(
        result.building, result.coordinates, 1.0, indoor_mask=result.indoor_mask
    )
    assert 0 < len(ap_spots) < len(receivers)
    # Building runs from x = 5 m to 40 m; AP spots keep a 1 m margin.
    assert min(x for x, _ in ap_spots) >= 6.0
    assert max(x for x, _ in ap_spots) <= 39.0
    # Coverage is still measured right up to the walls.
    assert min(x for x, _ in receivers) < 6.0


def test_interior_distance_grows_towards_the_middle():
    indoor = np.zeros((100, 100), np.uint8)
    indoor[10:90, 10:90] = 255
    distance = interior_distance_m(indoor, metres_per_pixel=0.1)
    assert distance[50, 50] > distance[12, 50] > 0
    assert distance[5, 5] == 0


def test_cells_whose_centre_lands_on_a_line_still_count():
    """Lines running exactly through cell centres (desk rows, text, wall
    lines) must not remove those cells from the usable area."""
    image = np.full((600, 1000, 3), 255, np.uint8)
    cv2.rectangle(image, (50, 50), (950, 550), (0, 0, 0), 8)
    # 1 m cells are 20 px here (50 m x 30 m image); centres sit at y = 10, 30, 50, ...
    for y in range(110, 550, 40):
        cv2.line(image, (50, y), (950, y), (0, 0, 0), 2)     # thin lines through centres
    result = process_floor_plan(image, physical_width_m=50.0, physical_height_m=30.0)
    receivers, _, _, _ = floor_plan_to_rf_geometry(
        result.building, result.coordinates, 1.0, indoor_mask=result.indoor_mask
    )
    cells_inside_building = 44 * 24          # roughly the 45 m x 25 m interior
    assert len(receivers) > 0.85 * cells_inside_building

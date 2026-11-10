"""Tests for the fast AP optimizer: same numbers as the heatmap, optimal results."""

import itertools
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for import_path in (PROJECT_ROOT, PROJECT_ROOT / "backend", PROJECT_ROOT / "frontend/src"):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from services.fast_placement import (
    PlacedAccessPoint,
    assign_channels,
    build_signal_table,
    choose_ap_spots,
    layout_score,
    optimize_placement,
    plan_access_points,
)
from services.floorplan_coverage import load_prediction_artifacts, predict_heatmap

ARTIFACTS = load_prediction_artifacts()


def wall(x1, y1, x2, y2, loss=6.0):
    return {"start": {"x": x1, "y": y1}, "end": {"x": x2, "y": y2}, "attenuation_db": loss}


def small_building():
    """A 24 m x 12 m floor split into rooms by walls with door gaps."""
    walls = [
        wall(8, 0, 8, 5), wall(8, 7, 8, 12),          # vertical wall with a door
        wall(16, 0, 16, 9),                           # vertical wall, open at bottom
        wall(0, 6, 5, 6), wall(10, 6, 16, 6),         # horizontal walls with gaps
        wall(18, 3, 24, 3, loss=12.0),                # a heavier wall
    ]
    points = [(x + 0.5, y + 0.5) for x in range(24) for y in range(12)]
    return points, walls


def test_table_matches_heatmap_exactly():
    points, walls = small_building()
    table = build_signal_table(points, points, walls, ARTIFACTS, 20.0, 5.0)
    rng = np.random.default_rng(0)
    for _ in range(5):
        layout = list(rng.choice(len(points), size=3, replace=False))
        aps = [PlacedAccessPoint(*points[i], channel=1) for i in layout]
        heatmap = predict_heatmap(aps, points, walls, ARTIFACTS, 20.0, 5.0).rssi_dbm
        assert np.allclose(table[layout].max(axis=0), heatmap, atol=1e-4)


def test_non_square_table_matches_square_table():
    points, walls = small_building()
    full = build_signal_table(points, points, walls, ARTIFACTS, 20.0, 5.0)
    rows = [0, 17, 100, 250]
    part = build_signal_table([points[i] for i in rows], points, walls, ARTIFACTS, 20.0, 5.0)
    assert np.allclose(full[rows], part, atol=1e-4)


def test_two_ap_result_is_the_true_optimum():
    points, walls = small_building()
    table = build_signal_table(points, points, walls, ARTIFACTS, 20.0, 5.0)
    threshold = float(np.percentile(table, 40))       # makes coverage non-trivial
    best = max(
        layout_score(table, pair, threshold)[0]
        for pair in itertools.combinations(range(len(points)), 2)
    )
    result = optimize_placement(table, points, 2, threshold)
    assert result.covered_points == best


def test_overlapping_aps_get_different_channels():
    table = np.array([
        [-40, -50, -60, -95],
        [-50, -40, -55, -95],
        [-95, -95, -95, -40],
    ], dtype=np.float32)
    channels = assign_channels(table, [0, 1, 2])
    assert channels[0] != channels[1]                 # heard in the same places
    assert set(channels) <= {1, 6, 11}


def test_ap_spots_are_thinned_only_for_very_fine_grids():
    coarse = [(x, y) for x in range(40) for y in range(40)]          # 1,600 points
    fine = [(x / 2, y / 2) for x in range(120) for y in range(120)]  # 14,400 points
    assert len(choose_ap_spots(coarse)) == len(coarse)
    assert len(choose_ap_spots(fine)) <= 4000


def test_plan_access_points_returns_requested_count():
    points, walls = small_building()
    result, aps = plan_access_points(points, walls, ARTIFACTS, 3, 20.0, 5.0, -67.0)
    assert len(aps) == 3
    assert len({(ap.x, ap.y) for ap in aps}) == 3
    assert 0 <= result.coverage_percent <= 100

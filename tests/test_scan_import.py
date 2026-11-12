"""Tests for importing WiFiman scans: colour decoding, alignment and bands."""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.scan_import import (
    WIFIMAN_LEGEND,
    DecodedScan,
    band_summary,
    combine_scans,
    walked_area_summary,
    decode_scan,
    fit_alignment,
    warp_scan_to_plan,
)


def legend_colour(dbm: float) -> tuple[int, int, int]:
    """BGR colour WiFiman uses for a given dBm (full saturation and brightness)."""
    hues = [h for _, h in WIFIMAN_LEGEND]
    values = [d for d, _ in WIFIMAN_LEGEND]
    hue = float(np.interp(dbm, values, hues)) % 180
    pixel = np.uint8([[[hue, 230, 230]]])
    return tuple(int(c) for c in cv2.cvtColor(pixel, cv2.COLOR_HSV2BGR)[0, 0])


def synthetic_scan(dbm_left=-55.0, dbm_right=-78.0, grid=True):
    """Black image with a measured corridor (two strengths), a grey walked-only
    area, thin grid lines like WiFiman draws, and a small stray icon."""
    image = np.zeros((400, 600, 3), np.uint8)
    image[150:250, 50:300] = legend_colour(dbm_left)
    image[150:250, 300:550] = legend_colour(dbm_right)
    image[250:380, 400:550] = (140, 140, 140)          # walked, no reading
    if grid:
        image[:, ::8] = (20, 20, 20)
        image[::8, :] = (20, 20, 20)
    cv2.circle(image, (30, 30), 4, legend_colour(-40.0), -1)   # stray icon
    return image


def test_decodes_colours_to_dbm():
    scan = decode_scan(synthetic_scan())
    left = np.nanmedian(scan.rssi_dbm[150:250, 50:300])
    right = np.nanmedian(scan.rssi_dbm[150:250, 300:550])
    assert abs(left - (-55.0)) <= 2.5
    assert abs(right - (-78.0)) <= 2.5


def test_grid_lines_do_not_remove_measured_cells():
    with_grid = decode_scan(synthetic_scan(grid=True))
    without_grid = decode_scan(synthetic_scan(grid=False))
    # Grid lines take ~23% of pixels; the rest must still count as measured.
    assert with_grid.measured.sum() > 0.7 * without_grid.measured.sum()


def test_walked_area_and_stray_icons():
    scan = decode_scan(synthetic_scan())
    assert scan.scanned[300:370, 410:540].mean() > 0.6     # grey area counted as walked
    assert not scan.measured[300:370, 410:540].any()       # ... but not as measured
    assert not scan.measured[25:36, 25:36].any()           # small icon ignored


def test_alignment_recovers_rotation_scale_and_shift():
    angle = np.radians(37)
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    scan_points = np.array([[50, 40], [550, 120], [300, 380], [100, 300]], float)
    plan_points = scan_points @ (0.6 * rotation).T + [200, 150]
    alignment = fit_alignment(scan_points[:3], plan_points[:3])
    assert abs(alignment.scale - 0.6) < 1e-6
    assert abs(alignment.rotation_deg - 37) < 1e-6
    assert alignment.residual_px < 1e-6


def test_points_in_wrong_order_give_large_error():
    scan_points = [(50, 40), (550, 120), (300, 380)]
    plan_points = [(50, 40), (300, 380), (550, 120)]       # 2 and 3 swapped
    assert fit_alignment(scan_points, plan_points).residual_px > 50


def test_alignment_needs_three_points():
    with pytest.raises(ValueError):
        fit_alignment([(0, 0), (1, 1)], [(0, 0), (1, 1)])


def test_warp_keeps_values_and_scales_area():
    scan = decode_scan(synthetic_scan())
    alignment = fit_alignment([(0, 0), (100, 0), (0, 100)], [(10, 10), (60, 10), (10, 60)])
    moved = warp_scan_to_plan(scan, alignment, (400, 600))
    assert abs(np.nanmedian(moved.rssi_dbm) - np.nanmedian(scan.rssi_dbm)) < 1.0
    ratio = moved.measured.sum() / scan.measured.sum()
    assert 0.2 < ratio < 0.3                               # scale 0.5 -> area x 0.25


def test_band_summary_adds_up_and_sorts_values():
    summary = band_summary(np.array([-45, -60, -70, -80, -90, np.nan]))
    percents = {band["band"]: band["percent"] for band in summary}
    assert percents == {"Excellent": 20.0, "Good": 20.0, "Fair": 20.0, "Poor": 40.0}


def test_walked_area_summary_counts_dead_zones():
    scan = decode_scan(synthetic_scan())
    rows = {row["band"]: row["percent"] for row in walked_area_summary(scan)}
    assert abs(sum(rows.values()) - 100) < 1e-6
    # The grey block is about 3/8 of the walked area in the synthetic scan.
    assert 25 < rows["Dead zone"] < 50


def make(values, walked):
    values = np.array(values, float)
    measured = np.isfinite(values)
    return DecodedScan(rssi_dbm=values, measured=measured, scanned=np.array(walked) | measured)


def test_combine_scans_averages_overlap_and_keeps_dead_zones():
    nan = np.nan
    a = make([[-60, nan, nan]], [[True, True, False]])
    b = make([[-70, -80, nan]], [[True, True, True]])
    combined = combine_scans([a, b])
    assert combined.rssi_dbm[0, 0] == -65            # average where both measured
    assert combined.rssi_dbm[0, 1] == -80            # signal from scan b wins over a's dead zone
    assert combined.scanned[0, 2] and not combined.measured[0, 2]   # dead zone


def test_combine_scans_rejects_different_floorplans():
    with pytest.raises(ValueError):
        combine_scans([make([[-60]], [[True]]), make([[-60, -60]], [[True, True]])])

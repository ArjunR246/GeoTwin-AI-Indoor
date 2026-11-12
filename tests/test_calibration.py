"""Tests for calibrating the radio model against measurements."""

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for import_path in (PROJECT_ROOT, PROJECT_ROOT / "backend", PROJECT_ROOT / "frontend/src"):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from services.calibration import (
    DEAD_ZONE_DBM,
    band_shares,
    fit_calibration,
    path_features,
    sample_walked_cells,
)


def wall(x1, y1, x2, y2):
    return {"start": {"x": x1, "y": y1}, "end": {"x": x2, "y": y2}, "attenuation_db": 6.0}


WALLS = [wall(10, 0, 10, 8), wall(20, 4, 20, 12), wall(0, 8, 8, 8), wall(22, 8, 30, 8)]
TRUE_APS = [(5.0, 3.0), (25.0, 10.0)]
CANDIDATES = [(x + 0.5, y + 0.5) for x in range(0, 30, 2) for y in range(0, 12, 2)]


def synthetic_measurements(exponent=3.0, wall_loss=4.0, noise=1.5, seed=0):
    """A corridor walk (y = 6) plus a branch, measured from two known APs."""
    rng = np.random.default_rng(seed)
    cells = np.array([(x + 0.5, 6.0) for x in range(30)] + [(15.0, y + 0.5) for y in range(12)])
    distance, crossings = path_features(TRUE_APS, cells, WALLS)
    signal = (-35 - 10 * exponent * np.log10(distance) - wall_loss * crossings).max(0)
    values = signal + rng.normal(0, noise, len(cells))
    values[values < DEAD_ZONE_DBM] = np.nan          # phone shows no signal
    return cells, values


def test_recovers_known_physics_and_beats_guessing():
    cells, values = synthetic_measurements()
    model = fit_calibration(cells, values, CANDIDATES, WALLS, ap_count=2, block_m=4.0)
    assert abs(model.exponent - 3.0) <= 0.6
    assert abs(model.wall_loss_db - 4.0) <= 3.0
    assert model.fit_error_db < 3.0
    assert model.is_useful
    # Each true AP has an estimated AP within 4 m.
    for x, y in TRUE_APS:
        assert min(np.hypot(x - ex, y - ey) for ex, ey in model.current_aps) <= 4.0


def test_outcome_range_contains_best_fit():
    cells, values = synthetic_measurements()
    model = fit_calibration(cells, values, CANDIDATES, WALLS, ap_count=2, block_m=4.0)
    distance, crossings = path_features(CANDIDATES, cells, WALLS)
    layout = [CANDIDATES.index(p) for p in model.current_aps]
    low, high = model.outcome_range(distance, crossings, layout, -67.0)
    best = 100 * (model.signal(distance[layout], crossings[layout]).max(0) >= -67).mean()
    assert low - 1e-6 <= best <= high + 1e-6


def test_needs_enough_measurements():
    cells = np.array([(x, 0.0) for x in range(20)])
    with pytest.raises(ValueError):
        fit_calibration(cells, np.full(20, np.nan), CANDIDATES, WALLS, ap_count=2)


def test_sample_walked_cells_marks_dead_zones():
    rssi = np.full((40, 60), np.nan)
    measured = np.zeros((40, 60), bool)
    walked = np.zeros((40, 60), bool)
    walked[10:20, :] = True                    # a corridor exactly one cell row high
    measured[10:20, :30] = True                # signal only in the left half
    rssi[measured] = -60.0
    cells, values = sample_walked_cells(rssi, measured, walked, width_m=6.0, height_m=4.0)
    assert len(cells) == 6
    assert np.allclose(values[:3], -60.0) and np.isnan(values[3:]).all()


def test_band_shares_count_missing_as_dead():
    shares = band_shares(np.array([-45, -60, -70, -80, np.nan]))
    assert shares["Dead zone"] == 20.0 and shares["Excellent"] == 20.0
    assert shares["meets_threshold"] == 40.0

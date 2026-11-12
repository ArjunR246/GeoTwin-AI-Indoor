"""TX power and frequency must change predictions (the model alone ignores them)."""

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for import_path in (PROJECT_ROOT, PROJECT_ROOT / "backend", PROJECT_ROOT / "frontend/src"):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from services.fast_placement import PlacedAccessPoint, build_signal_table
from services.floorplan_coverage import (
    load_prediction_artifacts,
    predict_heatmap,
    radio_settings_adjustment_db,
)

ARTIFACTS = load_prediction_artifacts()
POINTS = [(x + 0.5, y + 0.5) for x in range(20) for y in range(10)]
WALLS = [{"start": {"x": 10, "y": 0}, "end": {"x": 10, "y": 7}, "attenuation_db": 6.0}]
APS = [PlacedAccessPoint(3.0, 4.0, 1), PlacedAccessPoint(16.0, 6.0, 6)]


def heatmap(tx, freq):
    return predict_heatmap(APS, POINTS, WALLS, ARTIFACTS, tx, freq).rssi_dbm


def test_no_change_at_training_settings():
    assert radio_settings_adjustment_db(20.0, 5.0) == 0.0


def test_tx_power_shifts_signal_one_to_one():
    assert np.allclose(heatmap(30.0, 5.0) - heatmap(20.0, 5.0), 10.0, atol=1e-4)


def test_lower_frequency_is_stronger():
    gain = heatmap(20.0, 2.4) - heatmap(20.0, 5.0)
    assert np.allclose(gain, 20 * np.log10(5.0 / 2.4), atol=1e-4)     # about +6.4 dB
    assert np.all(heatmap(20.0, 6.0) < heatmap(20.0, 5.0))


def test_optimizer_table_still_matches_heatmap_at_other_settings():
    table = build_signal_table(POINTS, POINTS, WALLS, ARTIFACTS, 14.0, 2.4)
    rows = [POINTS.index((3.5, 4.5)), POINTS.index((16.5, 6.5))]
    aps = [PlacedAccessPoint(*POINTS[i], channel=1) for i in rows]
    expected = predict_heatmap(aps, POINTS, WALLS, ARTIFACTS, 14.0, 2.4).rssi_dbm
    assert np.allclose(table[rows].max(axis=0), expected, atol=1e-3)


def test_invalid_frequency_rejected():
    with pytest.raises(ValueError):
        radio_settings_adjustment_db(20.0, 0.0)

"""Check the fine-grid smooth heatmap used for drawing."""

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for import_path in (PROJECT_ROOT, PROJECT_ROOT / "backend", PROJECT_ROOT / "frontend/src"):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from services.floorplan_coverage import load_prediction_artifacts, smooth_heatmap_grid


class FakeAP:
    def __init__(self, x, y):
        self.x, self.y, self.channel = x, y, 1


def test_smooth_grid_shape_mask_and_values():
    # Receivers only cover the left half (x < 5) of a 10 m x 6 m floor.
    receivers = [(x + 0.5, y + 0.5) for x in range(5) for y in range(6)]
    x_values, y_values, grid = smooth_heatmap_grid(
        access_points=[FakeAP(2.5, 3.0)],
        receiver_positions=receivers,
        walls=[],
        artifacts=load_prediction_artifacts(),
        tx_power_dbm=20.0,
        frequency_ghz=5.0,
        width_m=10.0,
        height_m=6.0,
        cell_size_m=1.0,
    )

    assert grid.shape == (len(y_values), len(x_values))
    left = grid[:, x_values < 5.0]
    right = grid[:, x_values > 5.0]
    assert np.isfinite(left).all()        # usable space is drawn
    assert np.isnan(right).all()          # outside the receivers stays empty
    assert -100 < np.nanmin(grid) < np.nanmax(grid) < 0   # sensible dBm values

    # Signal should be strongest near the AP and weaker far from it.
    near = grid[np.argmin(abs(y_values - 3.0)), np.argmin(abs(x_values - 2.5))]
    far = grid[np.argmin(abs(y_values - 0.2)), np.argmin(abs(x_values - 0.2))]
    assert near > far

"""Bridge uploaded-floorplan geometry to the trained RSSI predictor."""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from xgboost import XGBRegressor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = PROJECT_ROOT / "backend"
FRONTEND_SRC = PROJECT_ROOT / "frontend/src"

for import_path in (BACKEND_ROOT, FRONTEND_SRC):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from utils.calibration import apply_residual_calibration, uncertainty_half_widths
from utils.rssi_features import MODEL_FEATURES, build_rssi_features, feature_frame
from utils.wall_intersections import wall_attenuation_db
import pandas as pd
from scipy import ndimage
from scipy.spatial import cKDTree


# The trained model only ever saw these settings (see backend/data), so it
# ignores the TX power and frequency inputs. Their effect is added with the
# standard physics instead: received power rises 1:1 with transmit power, and
# free-space path loss changes by 20*log10(f / f_trained).
MODEL_TRAINED_TX_POWER_DBM = 20.0
MODEL_TRAINED_FREQUENCY_GHZ = 5.0


def radio_settings_adjustment_db(tx_power_dbm: float, frequency_ghz: float) -> float:
    """dB to add to model predictions for settings that differ from training."""
    if frequency_ghz <= 0:
        raise ValueError("frequency_ghz must be positive.")
    return float(
        (tx_power_dbm - MODEL_TRAINED_TX_POWER_DBM)
        - 20.0 * np.log10(frequency_ghz / MODEL_TRAINED_FREQUENCY_GHZ)
    )


@dataclass(frozen=True)
class PredictionArtifacts:
    """Loaded predictor, uncertainty profile, and optional field calibration."""

    model: XGBRegressor
    uncertainty_profile: dict[str, Any]
    calibration_model: Any | None


@dataclass(frozen=True)
class HeatmapPrediction:
    """Backend predictions for every valid receiver location."""

    rssi_dbm: np.ndarray
    uncertainty_db: np.ndarray
    serving_ap_indexes: np.ndarray


def load_prediction_artifacts() -> PredictionArtifacts:
    """Load the trained backend model and optional real-survey correction."""
    model = XGBRegressor()
    model.load_model(BACKEND_ROOT / "models/rssi_xgb.json")

    uncertainty_path = BACKEND_ROOT / "models/uncertainty.json"
    with uncertainty_path.open(encoding="utf-8") as file:
        uncertainty_profile = json.load(file)

    calibration_path = BACKEND_ROOT / "models/field_calibration.pkl"
    calibration_model = joblib.load(calibration_path) if calibration_path.exists() else None
    return PredictionArtifacts(model, uncertainty_profile, calibration_model)


def floorplan_walls_to_backend(
    physical_walls: Sequence[object], wall_loss_db: float
) -> list[dict[str, object]]:
    """Convert vision-extracted wall segments to the backend twin schema."""
    return [
        {
            "start": {"x": float(wall.x1), "y": float(wall.y1)},
            "end": {"x": float(wall.x2), "y": float(wall.y2)},
            "material": "drywall",
            "attenuation_db": float(wall_loss_db),
        }
        for wall in physical_walls
    ]


def count_wall_crossings_fast(
    access_point: tuple[float, float],
    receiver_positions: Sequence[tuple[float, float]],
    walls: Sequence[Mapping[str, object]],
) -> tuple[np.ndarray, np.ndarray]:
    """Count wall crossings from one AP to every receiver at once.

    This gives the same answer as utils.wall_intersections.count_intersections
    (Shapely ``crosses``), but checks all receivers against all walls in one
    NumPy calculation instead of one pair at a time.

    Returns:
        (crossing_counts, attenuation_sums), one value per receiver.
    """
    receivers = np.asarray(receiver_positions, dtype=float)  # shape (R, 2)
    if len(walls) == 0:
        zeros = np.zeros(len(receivers))
        return zeros.astype(int), zeros

    # Wall start points, end points and losses, shape (W, 2), (W, 2), (W,)
    wall_start = np.array([[w["start"]["x"], w["start"]["y"]] for w in walls], dtype=float)
    wall_end = np.array([[w["end"]["x"], w["end"]["y"]] for w in walls], dtype=float)
    wall_loss = np.array([wall_attenuation_db(w) for w in walls], dtype=float)

    ap = np.asarray(access_point, dtype=float)  # shape (2,)

    def side(origin, direction_end, point):
        """Cross product: >0 if point is left of the line, <0 if right, 0 if on it."""
        return ((direction_end[..., 0] - origin[..., 0]) * (point[..., 1] - origin[..., 1])
                - (direction_end[..., 1] - origin[..., 1]) * (point[..., 0] - origin[..., 0]))

    # Add axes so every receiver (rows) is compared with every wall (columns).
    rx = receivers[:, None, :]          # (R, 1, 2)
    ws = wall_start[None, :, :]         # (1, W, 2)
    we = wall_end[None, :, :]           # (1, W, 2)
    ap_b = np.broadcast_to(ap, rx.shape)

    # Do the wall's two ends lie on opposite sides of the signal path?
    o1 = side(ap_b, rx, ws)
    o2 = side(ap_b, rx, we)
    # Do the path's two ends (AP and receiver) lie on opposite sides of the wall?
    o3 = side(ws, we, np.broadcast_to(ap, ws.shape))
    o4 = side(ws, we, rx)

    # A tiny tolerance ignores paths that only graze the exact end of a wall.
    eps = 1e-9
    crosses = ((o1 * o2) < -eps) & ((o3 * o4) < -eps)   # (R, W) True/False table

    counts = crosses.sum(axis=1)
    attenuation = crosses.astype(float) @ wall_loss
    return counts.astype(int), attenuation


def predict_heatmap(
    access_points: Sequence[object],
    receiver_positions: Sequence[tuple[float, float]],
    walls: Sequence[Mapping[str, object]],
    artifacts: PredictionArtifacts,
    tx_power_dbm: float,
    frequency_ghz: float,
) -> HeatmapPrediction:
    """Predict max-serving RSSI and uncertainty for every receiver position."""
    if not access_points:
        raise ValueError("At least one optimized access point is required.")
    if not receiver_positions:
        raise ValueError("At least one receiver position is required.")

    receivers = np.asarray(receiver_positions, dtype=float)
    feature_blocks = []
    for access_point in access_points:
        ap_xy = (float(access_point.x), float(access_point.y))
        counts, losses = count_wall_crossings_fast(ap_xy, receiver_positions, walls)
        distances = np.hypot(receivers[:, 0] - ap_xy[0], receivers[:, 1] - ap_xy[1])
        feature_blocks.append(pd.DataFrame({
            "distance": np.maximum(distances, 0.1),
            "num_walls_intersected": counts,
            "wall_attenuation_sum": losses,
            "tx_power_dbm": float(tx_power_dbm),
            "frequency_ghz": float(frequency_ghz),
        }))

    features = pd.concat(feature_blocks, ignore_index=True)[MODEL_FEATURES]
    base_predictions = artifacts.model.predict(features)
    predictions = apply_residual_calibration(
        features, base_predictions, artifacts.calibration_model
    ) + radio_settings_adjustment_db(tx_power_dbm, frequency_ghz)
    uncertainty = uncertainty_half_widths(features, artifacts.uncertainty_profile)

    receiver_count = len(receiver_positions)
    prediction_matrix = predictions.reshape(len(access_points), receiver_count)
    uncertainty_matrix = uncertainty.reshape(len(access_points), receiver_count)
    serving_ap_indexes = np.argmax(prediction_matrix, axis=0)
    receiver_indexes = np.arange(receiver_count)

    return HeatmapPrediction(
        rssi_dbm=prediction_matrix[serving_ap_indexes, receiver_indexes],
        uncertainty_db=uncertainty_matrix[serving_ap_indexes, receiver_indexes],
        serving_ap_indexes=serving_ap_indexes,
    )


def smooth_heatmap_grid(
    access_points: Sequence[object],
    receiver_positions: Sequence[tuple[float, float]],
    walls: Sequence[Mapping[str, object]],
    artifacts: PredictionArtifacts,
    tx_power_dbm: float,
    frequency_ghz: float,
    width_m: float,
    height_m: float,
    cell_size_m: float,
    display_step_m: float = 0.25,
    smoothing_m: float = 1.0,
    max_points: int = 60_000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Predict RSSI on a fine display grid and soften it for a smooth heatmap.

    The coarse receiver grid is still used for coverage numbers. This finer
    grid is only for drawing, so the picture looks continuous instead of blocky.

    Returns:
        (x_values, y_values, rssi_grid). rssi_grid is NaN outside the building.
    """
    # 1. Choose the display step. Keep the number of points bounded so very
    #    large floors do not become slow.
    step = max(display_step_m, float(np.sqrt(width_m * height_m / max_points)))
    x_values = np.arange(step / 2, width_m, step)
    y_values = np.arange(step / 2, height_m, step)
    grid_x, grid_y = np.meshgrid(x_values, y_values)

    # 2. Only draw where the floorplan has usable space: a fine point is
    #    "inside" if it falls in the square cell of a coarse receiver point.
    #    p=np.inf measures distance as max(|dx|, |dy|), i.e. a square.
    tree = cKDTree(np.asarray(receiver_positions, dtype=float))
    distances, _ = tree.query(
        np.column_stack((grid_x.ravel(), grid_y.ravel())), p=np.inf
    )
    inside = (distances <= cell_size_m / 2).reshape(grid_x.shape)

    # 3. Predict RSSI for every inside point, in chunks to limit memory use.
    points = list(zip(grid_x[inside], grid_y[inside]))
    values = np.empty(len(points))
    chunk = 3000
    for start in range(0, len(points), chunk):
        values[start:start + chunk] = predict_heatmap(
            access_points, points[start:start + chunk], walls,
            artifacts, tx_power_dbm, frequency_ghz,
        ).rssi_dbm

    rssi_grid = np.full(grid_x.shape, np.nan)
    rssi_grid[inside] = values

    # 4. Soften sharp steps with a Gaussian blur that ignores outside points
    #    (blur the values and the mask separately, then divide).
    sigma = smoothing_m / step
    if sigma > 0:
        blurred_values = ndimage.gaussian_filter(np.where(inside, rssi_grid, 0.0), sigma)
        blurred_weights = ndimage.gaussian_filter(inside.astype(float), sigma)
        rssi_grid = np.where(inside, blurred_values / np.maximum(blurred_weights, 1e-6), np.nan)

    return x_values, y_values, rssi_grid


def heatmap_grid(
    receiver_positions: Sequence[tuple[float, float]], values: Sequence[float]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert sparse receiver values into a regularly indexed heatmap grid."""
    if len(receiver_positions) != len(values):
        raise ValueError("receiver_positions and values must have equal length.")

    x_values = np.array(sorted({float(point[0]) for point in receiver_positions}))
    y_values = np.array(sorted({float(point[1]) for point in receiver_positions}))
    grid = np.full((len(y_values), len(x_values)), np.nan, dtype=float)
    x_indexes = {value: index for index, value in enumerate(x_values)}
    y_indexes = {value: index for index, value in enumerate(y_values)}

    for (x, y), value in zip(receiver_positions, values):
        grid[y_indexes[float(y)], x_indexes[float(x)]] = float(value)

    return x_values, y_values, grid


def grid_edges(values: np.ndarray) -> np.ndarray:
    """Return cell edges for heatmap values positioned at cell centres."""
    if len(values) == 1:
        return np.array([values[0] - 0.5, values[0] + 0.5])
    step = float(np.median(np.diff(values)))
    return np.concatenate((
        [values[0] - step / 2],
        (values[:-1] + values[1:]) / 2,
        [values[-1] + step / 2],
    ))


def coverage_metrics(
    rssi_dbm: Sequence[float], minimum_rssi_dbm: float
) -> dict[str, float]:
    """Summarize the backend-predicted coverage for the uploaded floorplan."""
    values = np.asarray(rssi_dbm, dtype=float)
    if len(values) == 0:
        raise ValueError("At least one RSSI prediction is required.")
    covered = values >= minimum_rssi_dbm
    return {
        "coverage_percent": float(covered.mean() * 100),
        "average_rssi_dbm": float(values.mean()),
        "minimum_rssi_dbm": float(values.min()),
        "dead_zone_percent": float((~covered).mean() * 100),
    }


def optimize_access_points(
    receiver_positions: Sequence[tuple[float, float]],
    valid_ap_positions: Sequence[tuple[float, float]],
    blocked_positions: Sequence[tuple[float, float]],
    physical_walls: Sequence[object],
    ap_count: int,
    tx_power_dbm: float,
    frequency_ghz: float,
    minimum_rssi_dbm: float,
    wall_loss_db: float,
):
    """Use the frontend's physical optimizer to find deployment locations."""
    from geotwin.optimization.nsga2 import NSGA2Config, create_access_points, optimize

    if ap_count < 1:
        raise ValueError("ap_count must be at least one.")
    if not valid_ap_positions:
        raise ValueError("No valid AP positions were extracted from the floorplan.")

    config = NSGA2Config(
        population_size=14,
        generations=6,
        crossover_probability=0.9,
        mutation_probability=0.12,
        min_ap_count=ap_count,
        max_ap_count=ap_count,
        random_seed=42,
        minimum_rssi_dbm=minimum_rssi_dbm,
        minimum_sinr_db=20.0,
        noise_floor_dbm=-95.0,
        frequency_hz=frequency_ghz * 1e9,
        tx_power_dbm=tx_power_dbm,
        antenna_gain_dbi=3.0,
        available_channels=(1, 6, 11),
        reference_distance_m=1.0,
        path_loss_exponent=3.0,
        wall_material="concrete",
        number_of_walls=0,
        obstacle_loss_db=6.0,
        same_channel_factor=1.0,
        adjacent_channel_factor=0.5,
        non_overlapping_factor=0.0,
    )
    config.validate()
    result = optimize(
        receiver_positions=receiver_positions,
        valid_positions=valid_ap_positions,
        blocked_positions=blocked_positions,
        config=config,
        floor_plan_walls=physical_walls,
        floor_plan_wall_loss_db=wall_loss_db,
    )
    if not result.pareto_front:
        raise RuntimeError("The AP optimizer returned no feasible solution.")

    best_solution = min(
        result.pareto_front,
        key=lambda item: (item.objectives[0], item.objectives[1], item.objectives[2]),
    )
    return best_solution, create_access_points(best_solution, config)

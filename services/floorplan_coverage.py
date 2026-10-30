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
from utils.rssi_features import build_rssi_features, feature_frame


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

    feature_rows = []
    for access_point in access_points:
        for receiver in receiver_positions:
            feature_rows.append(
                build_rssi_features(
                    (float(access_point.x), float(access_point.y)),
                    receiver,
                    walls,
                    tx_power_dbm,
                    frequency_ghz,
                )
            )

    features = feature_frame(feature_rows)
    base_predictions = artifacts.model.predict(features)
    predictions = apply_residual_calibration(
        features, base_predictions, artifacts.calibration_model
    )
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

"""Uncertainty, field calibration, and active-survey utilities for RSSI models."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from utils.rssi_features import MODEL_FEATURES


DEFAULT_CONFIDENCE_LEVEL = 0.90
MIN_GROUP_SAMPLES = 30
MIN_CALIBRATION_SAMPLES = 5
CALIBRATION_FEATURES = ["base_rssi_dbm", *MODEL_FEATURES]


def _absolute_residual_quantile(residuals: np.ndarray, confidence_level: float) -> float:
    return float(np.quantile(np.abs(residuals), confidence_level))


def build_uncertainty_profile(
    actual: Sequence[float],
    predicted: Sequence[float],
    features: pd.DataFrame,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    min_group_samples: int = MIN_GROUP_SAMPLES,
) -> dict[str, object]:
    """Create prediction half-widths from held-out absolute residuals.

    The global interval is always available. Wall-count-specific intervals are
    used only when the validation group has enough samples to be meaningful.
    """
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between zero and one.")
    if "num_walls_intersected" not in features:
        raise ValueError("features must contain num_walls_intersected.")

    actual_values = np.asarray(actual, dtype=float)
    predicted_values = np.asarray(predicted, dtype=float)
    if len(actual_values) == 0 or len(actual_values) != len(predicted_values):
        raise ValueError("actual and predicted values must have equal non-zero length.")
    if len(features) != len(actual_values):
        raise ValueError("features must have one row per prediction.")

    residuals = actual_values - predicted_values
    by_wall_count: dict[str, dict[str, float | int]] = {}
    wall_counts = features["num_walls_intersected"].to_numpy()

    for wall_count in sorted(np.unique(wall_counts)):
        group_residuals = residuals[wall_counts == wall_count]
        if len(group_residuals) >= min_group_samples:
            by_wall_count[str(int(wall_count))] = {
                "sample_count": int(len(group_residuals)),
                "half_width_db": _absolute_residual_quantile(
                    group_residuals, confidence_level
                ),
            }

    return {
        "method": "held_out_absolute_residual_quantile",
        "confidence_level": float(confidence_level),
        "validation_samples": int(len(actual_values)),
        "global_half_width_db": _absolute_residual_quantile(
            residuals, confidence_level
        ),
        "by_wall_count": by_wall_count,
    }


def uncertainty_half_widths(
    features: pd.DataFrame, uncertainty_profile: Mapping[str, object]
) -> np.ndarray:
    """Return the calibrated interval half-width for every feature row."""
    if "num_walls_intersected" not in features:
        raise ValueError("features must contain num_walls_intersected.")

    global_half_width = float(uncertainty_profile["global_half_width_db"])
    result = np.full(len(features), global_half_width, dtype=float)
    groups = uncertainty_profile.get("by_wall_count", {})

    for wall_count, group in groups.items():
        mask = features["num_walls_intersected"].to_numpy() == int(wall_count)
        result[mask] = float(group["half_width_db"])

    return result


def calibration_input_frame(
    features: pd.DataFrame, base_predictions: Sequence[float]
) -> pd.DataFrame:
    """Create residual-model inputs from model features and base predictions."""
    missing_columns = set(MODEL_FEATURES).difference(features.columns)
    if missing_columns:
        raise ValueError(f"Missing model features: {sorted(missing_columns)}")

    base_values = np.asarray(base_predictions, dtype=float)
    if len(base_values) != len(features):
        raise ValueError("base_predictions must align with features.")

    calibration_inputs = features.loc[:, MODEL_FEATURES].copy()
    calibration_inputs.insert(0, "base_rssi_dbm", base_values)
    return calibration_inputs.loc[:, CALIBRATION_FEATURES]


def fit_residual_calibration(
    features: pd.DataFrame,
    observed_rssi_dbm: Sequence[float],
    base_predictions: Sequence[float],
    min_samples: int = MIN_CALIBRATION_SAMPLES,
) -> tuple[Pipeline, dict[str, object]]:
    """Fit a regularized field-data residual correction model.

    This deliberately models only the gap between the synthetic/base model and
    field measurements. It avoids retraining the XGBoost model on a tiny,
    potentially biased survey.
    """
    if len(features) < min_samples:
        raise ValueError(f"At least {min_samples} field measurements are required.")

    calibration_inputs = calibration_input_frame(features, base_predictions)
    observed = np.asarray(observed_rssi_dbm, dtype=float)
    base_values = np.asarray(base_predictions, dtype=float)
    if len(observed) != len(features):
        raise ValueError("observed_rssi_dbm must align with features.")

    residuals = observed - base_values
    calibration_model = Pipeline([
        ("scale", StandardScaler()),
        ("ridge", Ridge(alpha=2.0)),
    ])
    calibration_model.fit(calibration_inputs, residuals)
    corrected_predictions = base_values + calibration_model.predict(calibration_inputs)

    report: dict[str, object] = {
        "method": "ridge_residual_correction",
        "measurement_count": int(len(features)),
        "base_mae_db": float(mean_absolute_error(observed, base_values)),
        "calibrated_training_mae_db": float(
            mean_absolute_error(observed, corrected_predictions)
        ),
    }

    if len(features) >= 10:
        cross_validation = KFold(
            n_splits=min(5, len(features)), shuffle=True, random_state=42
        )
        cross_validated_residuals = cross_val_predict(
            Pipeline([
                ("scale", StandardScaler()),
                ("ridge", Ridge(alpha=2.0)),
            ]),
            calibration_inputs,
            residuals,
            cv=cross_validation,
        )
        report["calibrated_cross_validated_mae_db"] = float(
            mean_absolute_error(observed, base_values + cross_validated_residuals)
        )

    return calibration_model, report


def apply_residual_calibration(
    features: pd.DataFrame,
    base_predictions: Sequence[float],
    calibration_model: Pipeline | None,
) -> np.ndarray:
    """Apply a fitted residual correction, or return base values unchanged."""
    base_values = np.asarray(base_predictions, dtype=float)
    if calibration_model is None:
        return base_values
    corrections = calibration_model.predict(
        calibration_input_frame(features, base_values)
    )
    return base_values + corrections


def receiver_grid_from_twin(
    digital_twin: Mapping[str, object], resolution: float
) -> list[tuple[float, float]]:
    """Build candidate receiver locations in every valid receiver zone."""
    if resolution <= 0:
        raise ValueError("resolution must be positive.")

    zones = [
        Polygon([(point["x"], point["y"]) for point in zone])
        for zone in digital_twin.get("valid_receiver_zones", [])
    ]
    if not zones:
        raise ValueError("digital_twin must define valid_receiver_zones.")

    min_x = min(zone.bounds[0] for zone in zones)
    min_y = min(zone.bounds[1] for zone in zones)
    max_x = max(zone.bounds[2] for zone in zones)
    max_y = max(zone.bounds[3] for zone in zones)
    candidates: list[tuple[float, float]] = []

    for x in np.arange(min_x, max_x + resolution / 2, resolution):
        for y in np.arange(min_y, max_y + resolution / 2, resolution):
            point = Point(float(x), float(y))
            if any(zone.covers(point) for zone in zones):
                candidates.append((float(x), float(y)))

    return candidates


def recommend_survey_points(
    candidate_locations: Sequence[tuple[float, float]],
    candidate_features: pd.DataFrame,
    predicted_rssi_dbm: Sequence[float],
    uncertainty_profile: Mapping[str, object],
    existing_locations: Sequence[tuple[float, float]] = (),
    max_recommendations: int = 10,
    min_separation_m: float = 2.0,
) -> list[dict[str, float | int | None]]:
    """Greedily choose high-uncertainty, spatially diverse survey points."""
    if max_recommendations <= 0 or min_separation_m <= 0:
        raise ValueError("max_recommendations and min_separation_m must be positive.")
    if len(candidate_locations) != len(candidate_features):
        raise ValueError("candidate_locations must align with candidate_features.")

    predictions = np.asarray(predicted_rssi_dbm, dtype=float)
    if len(predictions) != len(candidate_locations):
        raise ValueError("predicted_rssi_dbm must align with candidate_locations.")

    uncertainty = uncertainty_half_widths(candidate_features, uncertainty_profile)
    wall_counts = candidate_features["num_walls_intersected"].to_numpy(dtype=float)
    base_priority = uncertainty * (1.0 + 0.12 * wall_counts)
    remaining = list(range(len(candidate_locations)))
    anchors = list(existing_locations)
    recommendations: list[dict[str, float | int | None]] = []

    while remaining and len(recommendations) < max_recommendations:
        scores: list[tuple[float, float | None, int]] = []
        for index in remaining:
            x, y = candidate_locations[index]
            if anchors:
                nearest_distance = min(
                    float(math.hypot(x - anchor_x, y - anchor_y))
                    for anchor_x, anchor_y in anchors
                )
                novelty = min(1.0, nearest_distance / min_separation_m)
            else:
                nearest_distance = None
                novelty = 1.0
            score = float(base_priority[index] * (0.25 + 0.75 * novelty))
            scores.append((score, nearest_distance, index))

        score, nearest_distance, selected_index = max(scores, key=lambda item: item[0])
        if nearest_distance is not None and nearest_distance < min_separation_m * 0.25:
            break

        x, y = candidate_locations[selected_index]
        recommendations.append({
            "x": x,
            "y": y,
            "predicted_rssi_dbm": float(predictions[selected_index]),
            "uncertainty_db": float(uncertainty[selected_index]),
            "num_walls_intersected": int(wall_counts[selected_index]),
            "wall_attenuation_sum": float(
                candidate_features.iloc[selected_index]["wall_attenuation_sum"]
            ),
            "information_score": score,
            "nearest_survey_distance_m": nearest_distance,
        })
        anchors.append((x, y))
        remaining.remove(selected_index)

    return recommendations

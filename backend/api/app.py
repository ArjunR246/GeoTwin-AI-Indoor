"""HTTP interface for RSSI prediction and measurement-driven calibration."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from xgboost import XGBRegressor

from api.schemas import (
    CalibrationFitResponse,
    CalibrationStatusResponse,
    FieldMeasurementRequest,
    FieldMeasurementResponse,
    PredictRSSIRequest,
    PredictRSSIResponse,
    RSSIPrediction,
    SurveyRecommendationRequest,
    SurveyRecommendationResponse,
)
from utils.calibration import (
    MIN_CALIBRATION_SAMPLES,
    apply_residual_calibration,
    fit_residual_calibration,
    recommend_survey_points,
    receiver_grid_from_twin,
    uncertainty_half_widths,
)
from utils.rssi_features import MODEL_FEATURES, build_rssi_features, feature_frame
from utils.wall_intersections import count_intersections


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models/rssi_xgb.json"
UNCERTAINTY_PATH = PROJECT_ROOT / "models/uncertainty.json"
CALIBRATION_MODEL_PATH = PROJECT_ROOT / "models/field_calibration.pkl"
CALIBRATION_REPORT_PATH = PROJECT_ROOT / "models/calibration_report.json"
FIELD_MEASUREMENTS_PATH = PROJECT_ROOT / "data/processed/field_measurements.csv"

MEASUREMENT_COLUMNS = [
    "measurement_id",
    "measured_at",
    "layout_id",
    "ap_x",
    "ap_y",
    "rx_x",
    "rx_y",
    "observed_rssi_dbm",
    *MODEL_FEATURES,
]
FALLBACK_UNCERTAINTY_PROFILE = {
    "method": "fallback",
    "confidence_level": 0.90,
    "validation_samples": 0,
    "global_half_width_db": 8.0,
    "by_wall_count": {},
}

app = FastAPI(
    title="GeoTwin AI Indoor RSSI API",
    description=(
        "Predict RSSI with uncertainty, collect field observations, and "
        "prioritize the next high-value survey measurements."
    ),
)

model: XGBRegressor | None = None
uncertainty_profile: dict[str, Any] = FALLBACK_UNCERTAINTY_PROFILE.copy()
calibration_model: Any = None
calibration_report: dict[str, Any] = {}


def _read_json(path: Path, default: dict[str, Any]) -> dict[str, Any]:
    if not path.exists():
        return default
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def _field_measurements() -> pd.DataFrame:
    if not FIELD_MEASUREMENTS_PATH.exists():
        return pd.DataFrame(columns=MEASUREMENT_COLUMNS)
    measurements = pd.read_csv(FIELD_MEASUREMENTS_PATH)
    missing_columns = set(MEASUREMENT_COLUMNS).difference(measurements.columns)
    if missing_columns:
        raise ValueError(
            f"Field measurement store is missing columns: {sorted(missing_columns)}"
        )
    return measurements


def _require_model() -> XGBRegressor:
    if model is None:
        raise HTTPException(status_code=503, detail="RSSI model has not loaded.")
    return model


def _predict(feature_rows: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return base, calibrated, and uncertainty predictions for feature rows."""
    active_model = _require_model()
    base_predictions = active_model.predict(feature_rows)
    calibrated_predictions = apply_residual_calibration(
        feature_rows, base_predictions, calibration_model
    )
    half_widths = uncertainty_half_widths(feature_rows, uncertainty_profile)
    return base_predictions, calibrated_predictions, half_widths


def _measurement_record(
    request: FieldMeasurementRequest, feature_values: dict[str, float | int], measurement_id: int
) -> dict[str, object]:
    measured_at = request.measured_at or datetime.now(timezone.utc)
    return {
        "measurement_id": measurement_id,
        "measured_at": measured_at.isoformat(),
        "layout_id": request.layout_id,
        "ap_x": request.ap.x,
        "ap_y": request.ap.y,
        "rx_x": request.receiver.x,
        "rx_y": request.receiver.y,
        "observed_rssi_dbm": request.observed_rssi_dbm,
        **feature_values,
    }


def _fit_saved_calibration() -> dict[str, Any]:
    """Fit and persist a field residual correction from all recorded samples."""
    global calibration_model, calibration_report

    measurements = _field_measurements()
    if len(measurements) < MIN_CALIBRATION_SAMPLES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"At least {MIN_CALIBRATION_SAMPLES} field measurements are required; "
                f"found {len(measurements)}."
            ),
        )

    features = measurements.loc[:, MODEL_FEATURES]
    base_predictions = _require_model().predict(features)
    fitted_model, report = fit_residual_calibration(
        features,
        measurements["observed_rssi_dbm"],
        base_predictions,
    )
    calibration_report = {
        **report,
        "fitted_at": datetime.now(timezone.utc).isoformat(),
    }
    CALIBRATION_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(fitted_model, CALIBRATION_MODEL_PATH)
    with CALIBRATION_REPORT_PATH.open("w", encoding="utf-8") as file:
        json.dump(calibration_report, file, indent=2)
    calibration_model = fitted_model
    return calibration_report


def wall_features(ap: dict[str, float], rx: dict[str, float], walls: list[dict[str, object]]):
    """Backward-compatible wall feature helper used by existing consumers."""
    return count_intersections((ap["x"], ap["y"]), (rx["x"], rx["y"]), walls)


@app.on_event("startup")
def load_model() -> None:
    """Load the base predictor and any previously fitted calibration state."""
    global model, uncertainty_profile, calibration_model, calibration_report

    loaded_model = XGBRegressor()
    loaded_model.load_model(MODEL_PATH)
    model = loaded_model
    uncertainty_profile = _read_json(
        UNCERTAINTY_PATH, FALLBACK_UNCERTAINTY_PROFILE.copy()
    )
    calibration_report = _read_json(CALIBRATION_REPORT_PATH, {})
    calibration_model = (
        joblib.load(CALIBRATION_MODEL_PATH)
        if CALIBRATION_MODEL_PATH.exists()
        else None
    )


@app.get("/health")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "calibrated": calibration_model is not None,
    }


@app.post("/predict_rssi", response_model=PredictRSSIResponse)
def predict_rssi(request: PredictRSSIRequest) -> PredictRSSIResponse:
    walls = request.digital_twin.get("walls", [])
    feature_rows = [
        build_rssi_features(
            {"x": request.ap.x, "y": request.ap.y},
            {"x": receiver.x, "y": receiver.y},
            walls,
            request.tx_power_dbm,
            request.frequency_ghz,
        )
        for receiver in request.receivers
    ]
    features = feature_frame(feature_rows)
    _, predictions, half_widths = _predict(features)

    return PredictRSSIResponse(predictions=[
        RSSIPrediction(
            x=receiver.x,
            y=receiver.y,
            rssi_dbm=float(predictions[index]),
            uncertainty_db=float(half_widths[index]),
            lower_bound_dbm=float(predictions[index] - half_widths[index]),
            upper_bound_dbm=float(predictions[index] + half_widths[index]),
            calibrated=calibration_model is not None,
        )
        for index, receiver in enumerate(request.receivers)
    ])


@app.get("/calibration/status", response_model=CalibrationStatusResponse)
def calibration_status() -> CalibrationStatusResponse:
    return CalibrationStatusResponse(
        calibrated=calibration_model is not None,
        field_measurement_count=int(len(_field_measurements())),
        min_measurements_required=MIN_CALIBRATION_SAMPLES,
        uncertainty_profile=uncertainty_profile,
        calibration_report=calibration_report,
    )


@app.post(
    "/calibration/measurements", response_model=FieldMeasurementResponse, status_code=201
)
def record_field_measurement(request: FieldMeasurementRequest) -> FieldMeasurementResponse:
    measurements = _field_measurements()
    walls = request.digital_twin.get("walls", [])
    feature_values = build_rssi_features(
        {"x": request.ap.x, "y": request.ap.y},
        {"x": request.receiver.x, "y": request.receiver.y},
        walls,
        request.tx_power_dbm,
        request.frequency_ghz,
    )
    features = feature_frame([feature_values])
    base_predictions, calibrated_predictions, _ = _predict(features)
    measurement_id = (
        int(measurements["measurement_id"].max()) + 1 if len(measurements) else 1
    )
    updated_measurements = pd.concat([
        measurements,
        pd.DataFrame([_measurement_record(request, feature_values, measurement_id)]),
    ], ignore_index=True)
    FIELD_MEASUREMENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    updated_measurements.to_csv(FIELD_MEASUREMENTS_PATH, index=False)

    return FieldMeasurementResponse(
        measurement_id=measurement_id,
        base_prediction_dbm=float(base_predictions[0]),
        calibrated_prediction_dbm=float(calibrated_predictions[0]),
        observed_rssi_dbm=request.observed_rssi_dbm,
        residual_db=float(request.observed_rssi_dbm - base_predictions[0]),
        measurements_ready_for_fit=len(updated_measurements) >= MIN_CALIBRATION_SAMPLES,
    )


@app.post("/calibration/fit", response_model=CalibrationFitResponse)
def fit_calibration() -> CalibrationFitResponse:
    report = _fit_saved_calibration()
    return CalibrationFitResponse(calibrated=True, report=report)


@app.post(
    "/calibration/recommendations", response_model=SurveyRecommendationResponse
)
def survey_recommendations(
    request: SurveyRecommendationRequest,
) -> SurveyRecommendationResponse:
    walls = request.digital_twin.get("walls", [])
    candidate_locations = receiver_grid_from_twin(
        request.digital_twin, request.candidate_resolution_m
    )
    feature_rows = [
        build_rssi_features(
            {"x": request.ap.x, "y": request.ap.y},
            {"x": x, "y": y},
            walls,
            request.tx_power_dbm,
            request.frequency_ghz,
        )
        for x, y in candidate_locations
    ]
    features = feature_frame(feature_rows)
    _, predictions, _ = _predict(features)

    measurements = _field_measurements()
    layout_measurements = measurements[measurements["layout_id"] == request.layout_id]
    existing_locations = list(zip(
        layout_measurements["rx_x"], layout_measurements["rx_y"]
    ))
    ranked = recommend_survey_points(
        candidate_locations,
        features,
        predictions,
        uncertainty_profile,
        existing_locations=existing_locations,
        max_recommendations=request.max_recommendations,
        min_separation_m=request.min_separation_m,
    )

    confidence = float(uncertainty_profile["confidence_level"]) * 100
    for recommendation in ranked:
        recommendation["lower_bound_dbm"] = (
            recommendation["predicted_rssi_dbm"] - recommendation["uncertainty_db"]
        )
        recommendation["upper_bound_dbm"] = (
            recommendation["predicted_rssi_dbm"] + recommendation["uncertainty_db"]
        )
        recommendation["rationale"] = (
            f"{confidence:.0f}% interval ±{recommendation['uncertainty_db']:.2f} dB; "
            f"path crosses {recommendation['num_walls_intersected']} wall(s) with "
            f"{recommendation['wall_attenuation_sum']:.1f} dB estimated loss."
        )

    return SurveyRecommendationResponse(recommendations=ranked)

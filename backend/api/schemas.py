from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class Point(BaseModel):
    x: float
    y: float


class AP(BaseModel):
    x: float
    y: float


class PredictRSSIRequest(BaseModel):
    ap: AP
    receivers: List[Point]
    digital_twin: Dict[str, Any]
    tx_power_dbm: float = 20.0
    frequency_ghz: float = 5.0


class RSSIPrediction(BaseModel):
    x: float
    y: float
    rssi_dbm: float
    uncertainty_db: float
    lower_bound_dbm: float
    upper_bound_dbm: float
    calibrated: bool


class PredictRSSIResponse(BaseModel):
    predictions: List[RSSIPrediction]


class FieldMeasurementRequest(BaseModel):
    ap: AP
    receiver: Point
    observed_rssi_dbm: float
    digital_twin: Dict[str, Any]
    layout_id: str = "custom"
    tx_power_dbm: float = 20.0
    frequency_ghz: float = 5.0
    measured_at: Optional[datetime] = None


class FieldMeasurementResponse(BaseModel):
    measurement_id: int
    base_prediction_dbm: float
    calibrated_prediction_dbm: float
    observed_rssi_dbm: float
    residual_db: float
    measurements_ready_for_fit: bool


class CalibrationFitResponse(BaseModel):
    calibrated: bool
    report: Dict[str, Any]


class CalibrationStatusResponse(BaseModel):
    calibrated: bool
    field_measurement_count: int
    min_measurements_required: int
    uncertainty_profile: Dict[str, Any]
    calibration_report: Dict[str, Any]


class SurveyRecommendationRequest(BaseModel):
    ap: AP
    digital_twin: Dict[str, Any]
    layout_id: str = "custom"
    tx_power_dbm: float = 20.0
    frequency_ghz: float = 5.0
    candidate_resolution_m: float = 1.0
    max_recommendations: int = 10
    min_separation_m: float = 2.0


class SurveyRecommendation(BaseModel):
    x: float
    y: float
    predicted_rssi_dbm: float
    uncertainty_db: float
    lower_bound_dbm: float
    upper_bound_dbm: float
    num_walls_intersected: int
    wall_attenuation_sum: float
    information_score: float
    nearest_survey_distance_m: Optional[float]
    rationale: str


class SurveyRecommendationResponse(BaseModel):
    recommendations: List[SurveyRecommendation]

import importlib
import json
import tempfile
import unittest
from pathlib import Path

from xgboost import XGBRegressor

from api.schemas import (
    AP,
    FieldMeasurementRequest,
    Point,
    PredictRSSIRequest,
    SurveyRecommendationRequest,
)


api = importlib.import_module("api.app")
PROJECT_ROOT = Path(__file__).resolve().parents[1]


class PredictorCalibrationWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with (PROJECT_ROOT / "data/digital_twins/floor1.json").open(encoding="utf-8") as file:
            cls.twin = json.load(file)

        cls.original_state = {
            "model": api.model,
            "uncertainty_profile": api.uncertainty_profile,
            "calibration_model": api.calibration_model,
            "calibration_report": api.calibration_report,
            "measurements_path": api.FIELD_MEASUREMENTS_PATH,
            "calibration_model_path": api.CALIBRATION_MODEL_PATH,
            "calibration_report_path": api.CALIBRATION_REPORT_PATH,
        }
        cls.temporary_directory = tempfile.TemporaryDirectory()
        temporary_path = Path(cls.temporary_directory.name)
        api.FIELD_MEASUREMENTS_PATH = temporary_path / "field_measurements.csv"
        api.CALIBRATION_MODEL_PATH = temporary_path / "field_calibration.pkl"
        api.CALIBRATION_REPORT_PATH = temporary_path / "calibration_report.json"
        api.calibration_model = None
        api.calibration_report = {}
        api.uncertainty_profile = json.loads(
            (PROJECT_ROOT / "models/uncertainty.json").read_text(encoding="utf-8")
        )
        api.model = XGBRegressor()
        api.model.load_model(PROJECT_ROOT / "models/rssi_xgb.json")

    @classmethod
    def tearDownClass(cls):
        for key, value in cls.original_state.items():
            attribute = {
                "measurements_path": "FIELD_MEASUREMENTS_PATH",
                "calibration_model_path": "CALIBRATION_MODEL_PATH",
                "calibration_report_path": "CALIBRATION_REPORT_PATH",
            }.get(key, key)
            setattr(api, attribute, value)
        cls.temporary_directory.cleanup()

    def _prediction_request(self, receiver: Point) -> PredictRSSIRequest:
        return PredictRSSIRequest(
            ap=AP(x=2.0, y=4.0),
            receivers=[receiver],
            digital_twin=self.twin,
        )

    def test_uncertainty_recommendation_and_field_calibration_workflow(self):
        initial_prediction = api.predict_rssi(self._prediction_request(Point(x=24.0, y=4.0)))
        initial_item = initial_prediction.predictions[0]
        self.assertGreater(initial_item.uncertainty_db, 0.0)
        self.assertFalse(initial_item.calibrated)
        self.assertAlmostEqual(
            initial_item.upper_bound_dbm - initial_item.rssi_dbm,
            initial_item.uncertainty_db,
        )

        recommendation_response = api.survey_recommendations(
            SurveyRecommendationRequest(
                ap=AP(x=2.0, y=4.0),
                digital_twin=self.twin,
                layout_id="floor1_office",
                candidate_resolution_m=5.0,
                max_recommendations=3,
                min_separation_m=3.0,
            )
        )
        self.assertEqual(len(recommendation_response.recommendations), 3)
        self.assertTrue(all(
            item.uncertainty_db > 0 for item in recommendation_response.recommendations
        ))

        measurement_points = [
            Point(x=4.0, y=3.0),
            Point(x=10.0, y=3.0),
            Point(x=16.0, y=3.0),
            Point(x=22.0, y=3.0),
            Point(x=10.0, y=15.0),
        ]
        for index, receiver in enumerate(measurement_points):
            base_prediction = api.predict_rssi(self._prediction_request(receiver))
            measurement = api.record_field_measurement(
                FieldMeasurementRequest(
                    ap=AP(x=2.0, y=4.0),
                    receiver=receiver,
                    observed_rssi_dbm=base_prediction.predictions[0].rssi_dbm + 2.0,
                    digital_twin=self.twin,
                    layout_id="floor1_office",
                )
            )
            self.assertEqual(measurement.measurement_id, index + 1)

        fit_result = api.fit_calibration()
        self.assertTrue(fit_result.calibrated)
        self.assertEqual(fit_result.report["measurement_count"], 5)
        self.assertTrue(api.CALIBRATION_MODEL_PATH.exists())

        calibrated_prediction = api.predict_rssi(
            self._prediction_request(Point(x=24.0, y=4.0))
        )
        self.assertTrue(calibrated_prediction.predictions[0].calibrated)
        status = api.calibration_status()
        self.assertEqual(status.field_measurement_count, 5)
        self.assertTrue(status.calibrated)


if __name__ == "__main__":
    unittest.main()

# GeoTwin AI Indoor

GeoTwin AI Indoor is a measurement-driven Wi-Fi RSSI digital twin. It predicts
coverage through realistic indoor partitions, quantifies prediction uncertainty,
and uses real surveys to direct the next most useful measurements and calibrate
the model without discarding the physics-informed synthetic baseline.

## What the calibration loop does

1. Predict RSSI together with a 90% uncertainty interval.
2. Rank receiver locations where a field measurement has the greatest expected
   information value: high uncertainty, difficult NLOS paths, and far from
   locations already surveyed.
3. Store observed RSSI measurements from a phone, laptop, or ESP32 survey.
4. Fit a regularized residual-correction model once at least five measurements
   are available.
5. Apply that correction to future predictions while retaining the original
   XGBoost model as the stable base model.

The first uncertainty profile is built from held-out validation residuals. Once
you begin collecting real measurements, use the calibration report to assess
whether the field correction reduces error for the target building.

## Run it

From the project root:

```bash
venv/bin/python data/generate_dataset.py
venv/bin/python models/train_xgboost.py
venv/bin/python -m unittest discover -v
venv/bin/uvicorn api.app:app --reload
```

Open `http://127.0.0.1:8000/docs` for the interactive API interface.

## Calibration API

All request bodies include the relevant `digital_twin`, so the service can
derive the correct distance, crossed walls, and material attenuation.

| Endpoint | Purpose |
| --- | --- |
| `POST /predict_rssi` | RSSI, uncertainty half-width, lower/upper bounds, and calibration state for receiver points. |
| `POST /calibration/recommendations` | Next best field-survey points for an AP and digital twin. |
| `POST /calibration/measurements` | Save one observed RSSI measurement. |
| `POST /calibration/fit` | Fit and persist the field residual correction after five or more measurements. |
| `GET /calibration/status` | Field-measurement count, uncertainty profile, and calibration report. |

Suggested field workflow:

```text
recommendations → measure RSSI at selected locations → measurements (repeat)
                → calibration/fit → predict_rssi / status
```

`/calibration/recommendations` accepts `candidate_resolution_m`,
`max_recommendations`, and `min_separation_m`; start with a 1–2 m grid and a
2–3 m minimum separation. Include a consistent `layout_id` when sending
measurements so recommendations can avoid already surveyed locations on that
layout.

## Saved artifacts

- `models/rssi_xgb.json` — base RSSI predictor.
- `models/uncertainty.json` — held-out residual uncertainty profile.
- `data/processed/field_measurements.csv` — created only after real field data
  is submitted; do not replace it with synthetic measurements.
- `models/field_calibration.pkl` and `models/calibration_report.json` — created
  after a successful field calibration.

## Current synthetic baseline

The included data generator samples two realistic multi-room layouts and
models 0–5 wall intersections. The current held-out synthetic test result is
RMSE 4.07 dB, MAE 3.25 dB, R² 0.9131, with a 90% validation-derived interval
of ±6.63 dB. These values are a baseline only; report separate field metrics
after calibration rather than treating synthetic performance as deployment
accuracy.

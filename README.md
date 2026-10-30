# GeoTwin AI Indoor — Combined Application

This folder combines the supplied floorplan-analysis frontend with the trained
RSSI backend.

## What happens in the UI

1. Upload a single PNG, JPG, BMP, or TIFF floorplan.
2. Set the physical floor dimensions and number of APs.
3. Click **Generate AP plan**.
4. GeoTwin extracts walls and usable space, optimizes AP positions, and overlays
   the backend's RSSI heatmap directly on the uploaded plan.
5. Download the final PNG and review the recommended AP coordinates and channels.

The frontend's computer-vision and NSGA-II modules determine the feasible AP
locations. The copied backend's trained XGBoost model generates the final RSSI
map from distance and detected wall crossings. If real field calibration has
been created through `backend/api`, it is automatically applied to the map.

## Start the UI

On macOS/Linux:

```bash
cd /Users/ashwath/Desktop/GeoTwin_AI_Indoor_Combined
bash run.sh
```

The unified app requires Python 3.11 or later. If Python 3.11 is not your
default interpreter, provide it explicitly, for example
`PYTHON_BIN=/opt/anaconda3/bin/python3.11 bash run.sh`.

Or, after installing dependencies manually:

```bash
streamlit run app.py
```

Streamlit prints the local browser address, normally `http://localhost:8501`.

## Project layout

- `app.py` — clean, unified Streamlit UI.
- `services/` — bridge from extracted floorplan geometry to the trained model.
- `frontend/` — supplied floorplan vision, RF, and optimization code.
- `backend/` — trained XGBoost predictor, uncertainty profile, calibration API,
  digital twins, and tests.

The original projects in Downloads and Desktop remain unchanged.

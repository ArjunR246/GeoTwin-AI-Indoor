import json
import sys
from pathlib import Path

from xgboost import XGBRegressor
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from shapely.geometry import Point
from shapely.geometry import Polygon

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.wall_intersections import count_intersections


# ==================================================
# CONFIG
# ==================================================

MODEL_PATH = PROJECT_ROOT / "models/rssi_xgb.json"

DIGITAL_TWIN = PROJECT_ROOT / "data/digital_twins/floor1.json"

OUTPUT_PNG = PROJECT_ROOT / "reports/coverage_heatmap.png"

GRID_RESOLUTION = 0.25


# ==================================================
# LOAD MODEL
# ==================================================

model = XGBRegressor()
model.load_model(MODEL_PATH)


# ==================================================
# LOAD DIGITAL TWIN
# ==================================================

with open(DIGITAL_TWIN) as f:
    twin = json.load(f)

walls = twin["walls"]

receiver_zone = Polygon(
    [
        (p["x"], p["y"])
        for p in twin["valid_receiver_zones"][0]
    ]
)


# ==================================================
# AP LOCATION
# ==================================================
AP_X = 2.5
AP_Y = 5.0
plt.annotate(
    "AP",
    (AP_X, AP_Y)
)

# ==================================================
# WALL FEATURES
# ==================================================

def wall_features(ap, rx):
    return count_intersections(ap, rx, walls)


# ==================================================
# GRID GENERATION
# ==================================================

minx, miny, maxx, maxy = receiver_zone.bounds

xs = np.arange(
    minx,
    maxx,
    GRID_RESOLUTION
)

ys = np.arange(
    miny,
    maxy,
    GRID_RESOLUTION
)

heatmap = []


# ==================================================
# PREDICT RSSI
# ==================================================

for y in ys:

    row = []

    for x in xs:

        p = Point(x, y)

        if not receiver_zone.contains(p):

            row.append(np.nan)

            continue

        distance = np.sqrt(
            (x - AP_X) ** 2 +
            (y - AP_Y) ** 2
        )

        walls_count, wall_loss = wall_features(
            (AP_X, AP_Y),
            (x, y)
        )

        features = pd.DataFrame([{
            "distance": distance,
            "num_walls_intersected": walls_count,
            "wall_attenuation_sum": wall_loss,
            "tx_power_dbm": 20,
            "frequency_ghz": 5.0
        }])

        pred = model.predict(features)[0]

        row.append(pred)

    heatmap.append(row)

heatmap = np.array(heatmap)


# ==================================================
# PLOT
# ==================================================

plt.figure(figsize=(8, 8))

plt.imshow(
    heatmap,
    origin="lower",
    extent=[
        minx,
        maxx,
        miny,
        maxy
    ]
)
for wall in walls:
    plt.plot(
        [wall["start"]["x"], wall["end"]["x"]],
        [wall["start"]["y"], wall["end"]["y"]],
        linewidth=3,
        color="black"
    )

plt.colorbar(
    label="Predicted RSSI (dBm)"
)

plt.scatter(
    AP_X,
    AP_Y,
    marker="*",
    s=250
)

plt.title(
    "GeoTwin AI Indoor RSSI Coverage"
)

plt.xlabel("X (m)")
plt.ylabel("Y (m)")

plt.tight_layout()

plt.savefig(
    OUTPUT_PNG,
    dpi=300
)

plt.show()

print(
    "Saved:",
    OUTPUT_PNG
)

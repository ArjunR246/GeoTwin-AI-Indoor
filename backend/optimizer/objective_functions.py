import json
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from utils.wall_intersections import count_intersections

MODEL_PATH = "models/rssi_xgb.json"
TWIN_PATH = "data/digital_twins/floor1.json"

TX_POWER_DBM = 20.0
FREQUENCY_GHZ = 5.0

RSSI_COVERAGE_THRESHOLD = -67.0


def load_model():
    """Load the trained XGBoost RSSI model."""
    model = XGBRegressor()
    model.load_model(MODEL_PATH)
    return model


def load_digital_twin():
    """Load the digital twin."""
    with open(TWIN_PATH, "r") as f:
        return json.load(f)


def wall_features(ap_x, ap_y, rx_x, rx_y, walls):
    """
    Calculate the number of walls crossed and total wall attenuation
    between an AP and receiver.
    """
    return count_intersections((ap_x, ap_y), (rx_x, rx_y), walls)


def generate_receiver_grid(
    digital_twin,
    resolution=0.5
):
    """
    Generate receiver locations throughout the valid receiver zones.
    """
    zones = digital_twin.get("valid_receiver_zones", [])

    if not zones:
        raise ValueError("No valid receiver zones found.")

    # Get overall bounds
    min_x = min(
        point["x"]
        for zone in zones
        for point in zone
    )

    max_x = max(
        point["x"]
        for zone in zones
        for point in zone
    )

    min_y = min(
        point["y"]
        for zone in zones
        for point in zone
    )

    max_y = max(
        point["y"]
        for zone in zones
        for point in zone
    )

    receivers = []

    for x in np.arange(min_x, max_x + resolution, resolution):
        for y in np.arange(min_y, max_y + resolution, resolution):

            # Current floor1 uses a rectangular receiver zone.
            # Check whether the point belongs to any zone.
            from shapely.geometry import Point, Polygon

            point = Point(x, y)

            for zone in zones:
                polygon = Polygon([
                    (p["x"], p["y"])
                    for p in zone
                ])

                if polygon.contains(point) or polygon.touches(point):
                    receivers.append((float(x), float(y)))
                    break

    return receivers


def predict_grid_rssi(ap_x, ap_y, receivers, digital_twin, model):
    """
    Predict RSSI at every receiver location for a given AP position.
    """
    walls = digital_twin.get("walls", [])

    features = []

    for rx_x, rx_y in receivers:

        distance = np.sqrt(
            (ap_x - rx_x) ** 2 +
            (ap_y - rx_y) ** 2
        )

        # Avoid zero-distance numerical issues.
        distance = max(distance, 0.1)

        wall_count, wall_loss = wall_features(
            ap_x,
            ap_y,
            rx_x,
            rx_y,
            walls
        )

        features.append({
            "distance": distance,
            "num_walls_intersected": wall_count,
            "wall_attenuation_sum": wall_loss,
            "tx_power_dbm": TX_POWER_DBM,
            "frequency_ghz": FREQUENCY_GHZ
        })

    feature_df = pd.DataFrame(features)

    predictions = model.predict(feature_df)

    return np.array(predictions)


def calculate_coverage(rssi_values):
    """
    Percentage of receiver points with RSSI above the
    coverage threshold.
    """
    covered = np.sum(
        rssi_values >= RSSI_COVERAGE_THRESHOLD
    )

    total = len(rssi_values)

    if total == 0:
        return 0.0

    return (covered / total) * 100.0


def calculate_average_rssi(rssi_values):
    """Calculate average RSSI across the receiver grid."""
    if len(rssi_values) == 0:
        return -100.0

    return float(np.mean(rssi_values))


def evaluate_ap_position(
    ap_x,
    ap_y,
    receivers,
    digital_twin,
    model
):
    """
    Evaluate one AP position.

    Returns:
        coverage percentage
        average RSSI
        minimum RSSI
    """
    rssi_values = predict_grid_rssi(
        ap_x,
        ap_y,
        receivers,
        digital_twin,
        model
    )

    coverage = calculate_coverage(rssi_values)
    average_rssi = calculate_average_rssi(rssi_values)
    minimum_rssi = float(np.min(rssi_values))

    return {
        "coverage": coverage,
        "average_rssi": average_rssi,
        "minimum_rssi": minimum_rssi,
        "rssi_values": rssi_values
    }

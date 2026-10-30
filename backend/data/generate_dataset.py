"""Generate reproducible synthetic RSSI samples from the indoor twins."""

from __future__ import annotations

import json
import math
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.rssi_features import build_rssi_features


DEFAULT_TWIN_PATHS = (
    PROJECT_ROOT / "data/digital_twins/floor1.json",
    PROJECT_ROOT / "data/digital_twins/floor2.json",
)
OUTPUT_CSV = PROJECT_ROOT / "data/processed/train.csv"

NUM_SAMPLES = 12_000
RANDOM_SEED = 42

TX_POWER_DBM = 20.0
FREQUENCY_GHZ = 5.0
D0 = 1.0
PL_D0 = 40.0
PATH_LOSS_EXPONENT = 2.4
X_SIGMA = 4.0


def build_zone_polygon(zone: Sequence[dict[str, float]]) -> Polygon:
    """Convert a digital-twin zone to a valid Shapely polygon."""
    polygon = Polygon([(point["x"], point["y"]) for point in zone])
    if not polygon.is_valid or polygon.area <= 0:
        raise ValueError("Digital-twin zones must be valid polygons with area.")
    return polygon


def random_point_in_polygon(polygon: Polygon, rng: np.random.Generator) -> Point:
    """Uniformly sample a point from a polygon's bounding box by rejection."""
    min_x, min_y, max_x, max_y = polygon.bounds

    for _ in range(10_000):
        point = Point(rng.uniform(min_x, max_x), rng.uniform(min_y, max_y))
        if polygon.contains(point):
            return point

    raise RuntimeError("Could not sample a point inside a valid zone.")


def random_point_in_zones(
    zones: Sequence[Polygon], rng: np.random.Generator
) -> Point:
    """Sample a zone by area, then uniformly sample a point inside it."""
    if not zones:
        raise ValueError("At least one valid zone is required.")

    areas = np.array([zone.area for zone in zones], dtype=float)
    zone_index = int(rng.choice(len(zones), p=areas / areas.sum()))
    return random_point_in_polygon(zones[zone_index], rng)


def simulate_rssi(
    distance: float,
    wall_loss: float,
    rng: np.random.Generator,
    tx_power: float = TX_POWER_DBM,
    path_loss_exponent: float = PATH_LOSS_EXPONENT,
    sigma: float = X_SIGMA,
) -> float:
    """Apply log-distance loss, wall loss, and independent shadow fading."""
    distance = max(distance, D0)
    path_loss = PL_D0 + 10 * path_loss_exponent * math.log10(distance / D0)
    shadowing = rng.normal(loc=0.0, scale=sigma)
    return float(tx_power - path_loss - wall_loss + shadowing)


def load_layout(twin_path: Path) -> dict[str, object]:
    """Load a digital twin and prepare the polygons needed for sampling."""
    with twin_path.open(encoding="utf-8") as file:
        twin = json.load(file)

    ap_zones = [build_zone_polygon(zone) for zone in twin.get("valid_ap_zones", [])]
    receiver_zones = [
        build_zone_polygon(zone) for zone in twin.get("valid_receiver_zones", [])
    ]
    if not ap_zones or not receiver_zones:
        raise ValueError(f"{twin_path} must define AP and receiver zones.")

    return {
        "id": twin.get("id", twin_path.stem),
        "walls": twin.get("walls", []),
        "ap_zones": ap_zones,
        "receiver_zones": receiver_zones,
    }


def generate_dataset(
    twin_paths: Sequence[Path] = DEFAULT_TWIN_PATHS,
    num_samples: int = NUM_SAMPLES,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Generate balanced samples across all provided indoor layouts."""
    if num_samples <= 0:
        raise ValueError("num_samples must be positive.")
    if not twin_paths:
        raise ValueError("At least one digital-twin path is required.")

    layouts = [load_layout(Path(path)) for path in twin_paths]
    rng = np.random.default_rng(seed)
    layout_indexes = np.resize(np.arange(len(layouts)), num_samples)
    rng.shuffle(layout_indexes)

    rows: list[dict[str, float | int | str]] = []
    for layout_index in layout_indexes:
        layout = layouts[int(layout_index)]
        ap = random_point_in_zones(layout["ap_zones"], rng)
        rx = random_point_in_zones(layout["receiver_zones"], rng)
        features = build_rssi_features(
            (ap.x, ap.y),
            (rx.x, rx.y),
            layout["walls"],
            TX_POWER_DBM,
            FREQUENCY_GHZ,
        )
        distance = float(features["distance"])
        wall_loss = float(features["wall_attenuation_sum"])

        rows.append({
            "layout_id": str(layout["id"]),
            "ap_x": ap.x,
            "ap_y": ap.y,
            "rx_x": rx.x,
            "rx_y": rx.y,
            **features,
            "rssi_dbm": simulate_rssi(distance, wall_loss, rng),
        })

    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    dataset = generate_dataset()
    dataset.to_csv(OUTPUT_CSV, index=False)

    print(f"Generated {len(dataset)} samples in {OUTPUT_CSV}")
    print("Samples by layout:")
    print(dataset["layout_id"].value_counts().sort_index().to_string())
    print("Wall crossings:")
    print(dataset["num_walls_intersected"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()

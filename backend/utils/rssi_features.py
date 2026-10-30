"""Construct the RSSI feature contract shared by training and serving."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import pandas as pd

from utils.wall_intersections import count_intersections


MODEL_FEATURES = [
    "distance",
    "num_walls_intersected",
    "wall_attenuation_sum",
    "tx_power_dbm",
    "frequency_ghz",
]


def _coordinates(point: Sequence[float] | Mapping[str, float]) -> tuple[float, float]:
    if isinstance(point, Mapping):
        return float(point["x"]), float(point["y"])
    if len(point) != 2:
        raise ValueError("Points must contain exactly two coordinates.")
    return float(point[0]), float(point[1])


def build_rssi_features(
    ap: Sequence[float] | Mapping[str, float],
    receiver: Sequence[float] | Mapping[str, float],
    walls: Sequence[Mapping[str, object]],
    tx_power_dbm: float = 20.0,
    frequency_ghz: float = 5.0,
) -> dict[str, float | int]:
    """Return the exact feature row expected by the RSSI model."""
    ap_x, ap_y = _coordinates(ap)
    receiver_x, receiver_y = _coordinates(receiver)
    wall_count, wall_loss = count_intersections(
        (ap_x, ap_y), (receiver_x, receiver_y), walls
    )

    return {
        "distance": max(math.hypot(receiver_x - ap_x, receiver_y - ap_y), 0.1),
        "num_walls_intersected": wall_count,
        "wall_attenuation_sum": wall_loss,
        "tx_power_dbm": float(tx_power_dbm),
        "frequency_ghz": float(frequency_ghz),
    }


def feature_frame(feature_rows: Sequence[Mapping[str, float | int]]) -> pd.DataFrame:
    """Create a model-ready frame with deterministic column ordering."""
    return pd.DataFrame(feature_rows, columns=MODEL_FEATURES)

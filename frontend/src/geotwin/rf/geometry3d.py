"""Basic 3D geometry utilities for indoor RF calculations."""

from __future__ import annotations

from math import isfinite, sqrt
from typing import TypeAlias


Point3D: TypeAlias = tuple[float, float, float]


def validate_point3d(
    point: Point3D,
    name: str = "point",
) -> None:
    """Validate a three-dimensional point."""
    if not isinstance(point, (tuple, list)) or len(point) != 3:
        raise ValueError(
            f"{name} must contain exactly three coordinates."
        )

    if not all(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and isfinite(value)
        for value in point
    ):
        raise ValueError(
            f"{name} coordinates must be finite numbers."
        )


def distance_3d(
    point_a: Point3D,
    point_b: Point3D,
) -> float:
    """Calculate Euclidean distance between two 3D points, in metres."""
    validate_point3d(point_a, "point_a")
    validate_point3d(point_b, "point_b")

    dx = point_a[0] - point_b[0]
    dy = point_a[1] - point_b[1]
    dz = point_a[2] - point_b[2]

    return sqrt(dx * dx + dy * dy + dz * dz)
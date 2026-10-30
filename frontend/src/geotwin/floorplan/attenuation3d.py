"""3D wall-crossing and RF attenuation utilities.

Wall crossings are detected in the horizontal (x, y) plane. The
intersection height is then calculated along the 3D transmitter-to-
receiver path. A wall contributes attenuation only when the path crosses
the wall segment below its top.

Important:
- Transmitter, receiver, and wall endpoints must use the same x/y
  coordinate system.
- Wall heights and transmitter/receiver z coordinates are in metres.
- Wall3D is assumed to extend vertically from z=0 to wall.height_m.
"""

from dataclasses import dataclass
from math import isfinite
from typing import Sequence, Tuple

from geotwin.floorplan.geometry3d import Wall3D

Point3D = Tuple[float, float, float]


@dataclass(frozen=True)
class WallAttenuation3D:
    """Result of a 3D wall-attenuation calculation."""

    crossings: int
    attenuation_db: float


def _validate_point(point: Point3D, name: str) -> None:
    """Validate a three-dimensional point."""
    if not isinstance(point, (tuple, list)) or len(point) != 3:
        raise ValueError(f"{name} must contain exactly three coordinates.")

    for coordinate in point:
        if isinstance(coordinate, bool):
            raise ValueError(f"{name} coordinates must be finite numbers.")

        try:
            valid = isfinite(coordinate)
        except TypeError as exc:
            raise ValueError(
                f"{name} coordinates must be finite numbers."
            ) from exc

        if not valid:
            raise ValueError(f"{name} coordinates must be finite numbers.")


def _intersection_parameters_2d(
    start: Point3D,
    end: Point3D,
    wall: Wall3D,
    tolerance: float,
) -> float | None:
    """Return the path parameter t for an XY intersection, if one exists.

    The transmitter-to-receiver path is P + t(R-P).
    The wall segment is Q + u(S-Q).

    Collinear overlap is not counted as a crossing because it does not
    define one unique intersection point.
    """
    px, py = start[0], start[1]
    rx, ry = end[0] - px, end[1] - py

    qx, qy = wall.start
    sx = wall.end[0] - qx
    sy = wall.end[1] - qy

    denominator = rx * sy - ry * sx

    # A vertical 3D path with no horizontal displacement, or a
    # parallel/collinear horizontal path, has no unique XY crossing.
    if abs(denominator) <= tolerance:
        return None

    qpx, qpy = qx - px, qy - py

    t = (qpx * sy - qpy * sx) / denominator
    u = (qpx * ry - qpy * rx) / denominator

    if not (-tolerance <= t <= 1.0 + tolerance):
        return None

    if not (-tolerance <= u <= 1.0 + tolerance):
        return None

    # Clamp small floating-point deviations at segment endpoints.
    return min(1.0, max(0.0, t))


def count_wall_crossings_3d(
    start: Point3D,
    end: Point3D,
    walls: Sequence[Wall3D],
    tolerance: float = 1e-9,
) -> int:
    """Count walls intersected by the 3D path below each wall's top.

    Each Wall3D is treated as a vertical surface extending from z=0
    to z=wall.height_m. Each wall segment contributes at most one
    crossing. Duplicate wall segments are not automatically deduplicated.
    """
    _validate_point(start, "start")
    _validate_point(end, "end")

    if not isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be a finite, non-negative number.")

    if start == end:
        return 0

    crossings = 0

    for wall in walls:
        if not isinstance(wall, Wall3D):
            raise TypeError("walls must contain only Wall3D instances.")

        if not isfinite(wall.height_m) or wall.height_m < 0:
            raise ValueError(
                f"Wall {wall.id!r} must have a finite, non-negative height."
            )

        t = _intersection_parameters_2d(
            start=start,
            end=end,
            wall=wall,
            tolerance=tolerance,
        )

        if t is None:
            continue

        # Interpolate the ray height at its XY intersection with the wall.
        intersection_z = start[2] + t * (end[2] - start[2])

        if -tolerance <= intersection_z <= wall.height_m + tolerance:
            crossings += 1

    return crossings


def calculate_path_attenuation_3d(
    transmitter: Point3D,
    receiver: Point3D,
    walls: Sequence[Wall3D],
    wall_loss_db: float,
    tolerance: float = 1e-9,
) -> WallAttenuation3D:
    """Calculate total wall attenuation for a 3D transmitter-receiver path."""
    if isinstance(wall_loss_db, bool) or not isfinite(wall_loss_db):
        raise ValueError("wall_loss_db must be a finite, non-negative number.")

    if wall_loss_db < 0:
        raise ValueError("wall_loss_db must be a finite, non-negative number.")

    crossings = count_wall_crossings_3d(
        start=transmitter,
        end=receiver,
        walls=walls,
        tolerance=tolerance,
    )

    return WallAttenuation3D(
        crossings=crossings,
        attenuation_db=crossings * wall_loss_db,
    )


def calculate_wall_attenuation_3d(
    transmitter: Point3D,
    receiver: Point3D,
    walls: Sequence[Wall3D],
    wall_loss_db: float,
    tolerance: float = 1e-9,
) -> float:
    """Return total wall attenuation in dB for a 3D path."""
    return calculate_path_attenuation_3d(
        transmitter=transmitter,
        receiver=receiver,
        walls=walls,
        wall_loss_db=wall_loss_db,
        tolerance=tolerance,
    ).attenuation_db
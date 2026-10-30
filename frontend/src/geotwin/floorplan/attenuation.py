from dataclasses import dataclass
from math import hypot

import numpy as np

from .digital_twin import DigitalTwinGrid
from .walls import WallSegment


Point = tuple[float, float]


@dataclass(frozen=True)
class WallCrossingResult:
    """Result of tracing a signal path through detected walls."""

    crossings: int
    attenuation_db: float


def _orientation(point_a: Point, point_b: Point, point_c: Point) -> float:
    """
    Calculate the orientation/cross-product value of three points.
    """
    return (
        (point_b[0] - point_a[0]) * (point_c[1] - point_a[1])
        - (point_b[1] - point_a[1]) * (point_c[0] - point_a[0])
    )


def _on_segment(
    point_a: Point,
    point_b: Point,
    point_c: Point,
    tolerance: float = 1e-9,
) -> bool:
    """Return whether point_b lies on segment a-c."""
    return (
        min(point_a[0], point_c[0]) - tolerance
        <= point_b[0]
        <= max(point_a[0], point_c[0]) + tolerance
        and
        min(point_a[1], point_c[1]) - tolerance
        <= point_b[1]
        <= max(point_a[1], point_c[1]) + tolerance
    )


def segments_intersect(
    segment_a_start: Point,
    segment_a_end: Point,
    segment_b_start: Point,
    segment_b_end: Point,
    tolerance: float = 1e-9,
) -> bool:
    """
    Determine whether two line segments intersect.

    Handles:
        - normal crossings
        - touching endpoints
        - collinear overlap
    """
    if tolerance < 0:
        raise ValueError("tolerance cannot be negative.")

    o1 = _orientation(
        segment_a_start,
        segment_a_end,
        segment_b_start,
    )

    o2 = _orientation(
        segment_a_start,
        segment_a_end,
        segment_b_end,
    )

    o3 = _orientation(
        segment_b_start,
        segment_b_end,
        segment_a_start,
    )

    o4 = _orientation(
        segment_b_start,
        segment_b_end,
        segment_a_end,
    )

    # General intersection.
    if (
        ((o1 > tolerance and o2 < -tolerance)
         or (o1 < -tolerance and o2 > tolerance))
        and
        ((o3 > tolerance and o4 < -tolerance)
         or (o3 < -tolerance and o4 > tolerance))
    ):
        return True

    # Special collinear/touching cases.
    if abs(o1) <= tolerance and _on_segment(
        segment_a_start,
        segment_b_start,
        segment_a_end,
        tolerance,
    ):
        return True

    if abs(o2) <= tolerance and _on_segment(
        segment_a_start,
        segment_b_end,
        segment_a_end,
        tolerance,
    ):
        return True

    if abs(o3) <= tolerance and _on_segment(
        segment_b_start,
        segment_a_start,
        segment_b_end,
        tolerance,
    ):
        return True

    if abs(o4) <= tolerance and _on_segment(
        segment_b_start,
        segment_a_end,
        segment_b_end,
        tolerance,
    ):
        return True

    return False


def count_wall_crossings(
    start: Point,
    end: Point,
    walls: tuple[WallSegment, ...] | list[WallSegment],
    tolerance: float = 1e-9,
) -> int:
    """
    Count the number of detected walls crossed by a path.

    Args:
        start:
            Signal source point.

        end:
            Signal destination point.

        walls:
            Detected wall segments.

        tolerance:
            Numerical tolerance used for geometric intersection.

    Returns:
        Number of wall segments intersected by the path.
    """
    if len(start) != 2 or len(end) != 2:
        raise ValueError("Points must contain exactly two coordinates.")

    if tolerance < 0:
        raise ValueError("tolerance cannot be negative.")

    if start == end:
        return 0

    crossings = 0

    for wall in walls:
        wall_start = (
            float(wall.x1),
            float(wall.y1),
        )

        wall_end = (
            float(wall.x2),
            float(wall.y2),
        )

        if segments_intersect(
            start,
            end,
            wall_start,
            wall_end,
            tolerance,
        ):
            crossings += 1

    return crossings


def calculate_path_attenuation(
    start: Point,
    end: Point,
    walls: tuple[WallSegment, ...] | list[WallSegment],
    wall_loss_db: float,
    tolerance: float = 1e-9,
) -> WallCrossingResult:
    """
    Calculate attenuation caused by walls along a path.

    Attenuation model:

        attenuation = wall_crossings × wall_loss_db
    """
    if wall_loss_db < 0:
        raise ValueError("wall_loss_db cannot be negative.")

    crossings = count_wall_crossings(
        start,
        end,
        walls,
        tolerance,
    )

    attenuation_db = crossings * wall_loss_db

    return WallCrossingResult(
        crossings=crossings,
        attenuation_db=float(attenuation_db),
    )

def calculate_wall_attenuation(
    transmitter: Point,
    receiver: Point,
    walls: tuple[WallSegment, ...] | list[WallSegment],
    wall_loss_db: float,
    tolerance: float = 1e-9,
) -> float:
    """
    Calculate additional RF attenuation between a transmitter
    and receiver caused by detected floor-plan walls.

    The attenuation is calculated as:

        wall_attenuation =
            number_of_wall_crossings × wall_loss_db

    This function is intentionally independent of the RF engine.
    It only converts floor-plan geometry into an attenuation value.
    """

    if wall_loss_db < 0:
        raise ValueError(
            "wall_loss_db cannot be negative."
        )

    crossings = count_wall_crossings(
        start=transmitter,
        end=receiver,
        walls=walls,
        tolerance=tolerance,
    )

    return float(crossings * wall_loss_db)

def build_attenuation_map(
    grid: DigitalTwinGrid,
    transmitter: Point,
    walls: tuple[WallSegment, ...] | list[WallSegment],
    wall_loss_db: float,
    tolerance: float = 1e-9,
) -> np.ndarray:
    """
    Build an automatic wall-attenuation map for a digital-twin grid.

    Each grid cell contains the wall attenuation between the transmitter
    and the center of that cell.

    Returns:
        2-D NumPy array with shape:
            (grid.rows, grid.columns)

        Values are attenuation in dB.
    """
    if not isinstance(grid, DigitalTwinGrid):
        raise TypeError(
            "grid must be a DigitalTwinGrid."
        )

    if len(transmitter) != 2:
        raise ValueError(
            "transmitter must contain exactly two coordinates."
        )

    if wall_loss_db < 0:
        raise ValueError(
            "wall_loss_db cannot be negative."
        )

    attenuation = np.zeros(
        (grid.rows, grid.columns),
        dtype=np.float64,
    )

    for cell in grid.cells:
        result = calculate_path_attenuation(
            transmitter,
            cell.center,
            walls,
            wall_loss_db,
            tolerance,
        )

        attenuation[
            cell.row,
            cell.column,
        ] = result.attenuation_db

    return attenuation


def build_wall_crossing_map(
    grid: DigitalTwinGrid,
    transmitter: Point,
    walls: tuple[WallSegment, ...] | list[WallSegment],
    tolerance: float = 1e-9,
) -> np.ndarray:
    """
    Build a map containing the number of walls crossed to each cell.
    """
    if not isinstance(grid, DigitalTwinGrid):
        raise TypeError(
            "grid must be a DigitalTwinGrid."
        )

    if len(transmitter) != 2:
        raise ValueError(
            "transmitter must contain exactly two coordinates."
        )

    crossing_map = np.zeros(
        (grid.rows, grid.columns),
        dtype=np.int32,
    )

    for cell in grid.cells:
        crossing_map[
            cell.row,
            cell.column,
        ] = count_wall_crossings(
            transmitter,
            cell.center,
            walls,
            tolerance,
        )

    return crossing_map
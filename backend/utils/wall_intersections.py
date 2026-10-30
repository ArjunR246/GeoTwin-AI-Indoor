"""Wall-crossing features shared by dataset generation and inference."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from shapely.geometry import LineString


DEFAULT_WALL_ATTENUATION_DB = {
    "drywall": 4.0,
    "glass": 3.0,
    "wood": 5.0,
    "brick": 8.0,
    "concrete": 15.0,
}


def wall_attenuation_db(wall: Mapping[str, object]) -> float:
    """Return a wall's explicit attenuation or its material default."""
    if "attenuation_db" in wall:
        return float(wall["attenuation_db"])

    material = str(wall.get("material", "")).lower()
    return DEFAULT_WALL_ATTENUATION_DB.get(material, 5.0)


def _point_coordinates(point: Sequence[float]) -> tuple[float, float]:
    if len(point) != 2:
        raise ValueError("Points must contain exactly two coordinates.")
    return float(point[0]), float(point[1])


def _wall_line(wall: Mapping[str, object]) -> LineString:
    """Build a line from the digital-twin wall schema."""
    try:
        start = wall["start"]
        end = wall["end"]
        start_xy = (float(start["x"]), float(start["y"]))
        end_xy = (float(end["x"]), float(end["y"]))
    except (KeyError, TypeError) as exc:
        raise ValueError("Each wall needs start/end x and y coordinates.") from exc

    if start_xy == end_xy:
        raise ValueError("Wall start and end coordinates must differ.")

    return LineString([start_xy, end_xy])


def count_intersections(
    ap: Sequence[float],
    rx: Sequence[float],
    walls: Sequence[Mapping[str, object]],
) -> tuple[int, float]:
    """Count walls crossed by a signal path and sum their attenuation.

    ``LineString.crosses`` intentionally excludes a path that only touches a
    wall endpoint or overlaps the wall. Those edge cases do not represent a
    signal travelling through a partition and would otherwise inflate loss at
    room corners or along a wall.
    """
    signal_path = LineString([_point_coordinates(ap), _point_coordinates(rx)])

    if signal_path.length == 0:
        return 0, 0.0

    count = 0
    attenuation_sum = 0.0

    for wall in walls:
        if signal_path.crosses(_wall_line(wall)):
            count += 1
            attenuation_sum += wall_attenuation_db(wall)

    return count, attenuation_sum

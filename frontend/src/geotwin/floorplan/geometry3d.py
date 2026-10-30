"""
3D geometry representations for the GeoTwin AI Indoor project.

This module converts the existing 2D BuildingModel into a simple
3D geometric representation.

Important:
- The source floorplan currently uses image/pixel coordinates.
- The z-axis is measured in metres.
- The default ceiling height is 3.2 metres.
- x and y remain in source coordinate units until physical scaling
  is applied using a known floorplan dimension.

This module provides geometric representations, not a watertight
3D mesh or a complete RF propagation model.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot, isfinite
from typing import Optional, Tuple

from geotwin.floorplan.building_model import BuildingModel
from geotwin.floorplan.corridors import Corridor
from geotwin.floorplan.rooms import Room
from geotwin.floorplan.walls import WallSegment


Point2D = Tuple[float, float]
Point3D = Tuple[float, float, float]
Polygon2D = Tuple[Point2D, ...]


def _validate_positive_finite(value: float, name: str) -> None:
    """Validate that a dimension is finite and strictly positive."""
    if not isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite number.")


def _validate_point(point: Point2D, name: str = "point") -> None:
    """Validate a two-dimensional point."""
    if len(point) != 2:
        raise ValueError(f"{name} must contain exactly two coordinates.")

    if not all(isfinite(float(value)) for value in point):
        raise ValueError(f"{name} coordinates must be finite.")


def _validate_polygon(polygon: Polygon2D, name: str = "polygon") -> None:
    """Validate a polygon containing at least three finite points."""
    if len(polygon) < 3:
        raise ValueError(f"{name} must contain at least three points.")

    for point in polygon:
        _validate_point(point, name)


@dataclass(frozen=True)
class Wall3D:
    """
    3D representation of a wall segment.

    x and y use the same coordinate units as the source WallSegment.
    z and height_m use metres.
    """

    id: str
    start: Point2D
    end: Point2D
    height_m: float = 3.2
    thickness_m: Optional[float] = None
    material: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("Wall id must not be empty.")

        _validate_point(self.start, "wall start")
        _validate_point(self.end, "wall end")
        _validate_positive_finite(self.height_m, "height_m")

        if self.start == self.end:
            raise ValueError("Wall start and end points must be different.")

        if self.thickness_m is not None:
            _validate_positive_finite(
                self.thickness_m,
                "thickness_m",
            )

    @property
    def length(self) -> float:
        """
        Return wall length in source coordinate units.

        This is not necessarily a length in metres.
        """
        return hypot(
            self.end[0] - self.start[0],
            self.end[1] - self.start[1],
        )

    @property
    def bottom_vertices(self) -> Tuple[Point3D, Point3D]:
        """Return the two endpoints at floor level."""
        return (
            (self.start[0], self.start[1], 0.0),
            (self.end[0], self.end[1], 0.0),
        )

    @property
    def top_vertices(self) -> Tuple[Point3D, Point3D]:
        """Return the two endpoints at the top of the wall."""
        return (
            (self.start[0], self.start[1], self.height_m),
            (self.end[0], self.end[1], self.height_m),
        )

    @property
    def vertices(self) -> Tuple[Point3D, ...]:
        """
        Return four corner vertices describing a vertical wall surface.

        Vertex order:
        start-bottom, end-bottom, end-top, start-top.
        """
        return (
            self.bottom_vertices[0],
            self.bottom_vertices[1],
            self.top_vertices[1],
            self.top_vertices[0],
        )


@dataclass(frozen=True)
class Room3D:
    """3D floor and ceiling representation of a room."""

    id: str
    polygon: Polygon2D
    floor_z_m: float = 0.0
    ceiling_height_m: float = 3.2

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("Room id must not be empty.")

        _validate_polygon(self.polygon, "room polygon")

        if not isfinite(self.floor_z_m):
            raise ValueError("floor_z_m must be finite.")

        _validate_positive_finite(
            self.ceiling_height_m,
            "ceiling_height_m",
        )

    @property
    def ceiling_z_m(self) -> float:
        """Return the absolute ceiling height."""
        return self.floor_z_m + self.ceiling_height_m

    @property
    def floor_vertices(self) -> Tuple[Point3D, ...]:
        """Return room polygon vertices at floor level."""
        return tuple(
            (x, y, self.floor_z_m)
            for x, y in self.polygon
        )

    @property
    def ceiling_vertices(self) -> Tuple[Point3D, ...]:
        """Return room polygon vertices at ceiling level."""
        return tuple(
            (x, y, self.ceiling_z_m)
            for x, y in self.polygon
        )

    @property
    def wall_vertices(self) -> Tuple[Tuple[Point3D, ...], ...]:
        """
        Return vertical side surfaces around the room polygon.

        These are geometric surfaces, not a watertight solid mesh.
        """
        result = []
        count = len(self.polygon)

        for index in range(count):
            start = self.polygon[index]
            end = self.polygon[(index + 1) % count]

            result.append(
                (
                    (start[0], start[1], self.floor_z_m),
                    (end[0], end[1], self.floor_z_m),
                    (end[0], end[1], self.ceiling_z_m),
                    (start[0], start[1], self.ceiling_z_m),
                )
            )

        return tuple(result)


@dataclass(frozen=True)
class Corridor3D:
    """3D floor and ceiling representation of a corridor."""

    id: str
    polygon: Polygon2D
    floor_z_m: float = 0.0
    ceiling_height_m: float = 3.2

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("Corridor id must not be empty.")

        _validate_polygon(self.polygon, "corridor polygon")

        if not isfinite(self.floor_z_m):
            raise ValueError("floor_z_m must be finite.")

        _validate_positive_finite(
            self.ceiling_height_m,
            "ceiling_height_m",
        )

    @property
    def ceiling_z_m(self) -> float:
        """Return the absolute ceiling height."""
        return self.floor_z_m + self.ceiling_height_m

    @property
    def floor_vertices(self) -> Tuple[Point3D, ...]:
        """Return corridor polygon vertices at floor level."""
        return tuple(
            (x, y, self.floor_z_m)
            for x, y in self.polygon
        )

    @property
    def ceiling_vertices(self) -> Tuple[Point3D, ...]:
        """Return corridor polygon vertices at ceiling level."""
        return tuple(
            (x, y, self.ceiling_z_m)
            for x, y in self.polygon
        )


@dataclass(frozen=True)
class Building3D:
    """3D geometric representation of a single building floor."""

    floor_id: str
    ceiling_height_m: float
    walls: Tuple[Wall3D, ...]
    rooms: Tuple[Room3D, ...]
    corridors: Tuple[Corridor3D, ...]

    def __post_init__(self) -> None:
        if not self.floor_id:
            raise ValueError("floor_id must not be empty.")

        _validate_positive_finite(
            self.ceiling_height_m,
            "ceiling_height_m",
        )

        if not isinstance(self.walls, tuple):
            raise TypeError("walls must be a tuple.")

        if not isinstance(self.rooms, tuple):
            raise TypeError("rooms must be a tuple.")

        if not isinstance(self.corridors, tuple):
            raise TypeError("corridors must be a tuple.")

        if not all(isinstance(wall, Wall3D) for wall in self.walls):
            raise TypeError("All walls must be Wall3D objects.")

        if not all(isinstance(room, Room3D) for room in self.rooms):
            raise TypeError("All rooms must be Room3D objects.")

        if not all(
            isinstance(corridor, Corridor3D)
            for corridor in self.corridors
        ):
            raise TypeError(
                "All corridors must be Corridor3D objects."
            )

    @property
    def wall_count(self) -> int:
        return len(self.walls)

    @property
    def room_count(self) -> int:
        return len(self.rooms)

    @property
    def corridor_count(self) -> int:
        return len(self.corridors)


def polygon_from_object(
    obj: Room | Corridor,
) -> Polygon2D:
    """
    Convert an existing room/corridor polygon into float coordinates.

    The coordinates are preserved numerically; they are not scaled
    from pixels into metres.
    """
    polygon = tuple(
        (float(point[0]), float(point[1]))
        for point in obj.polygon
    )

    _validate_polygon(polygon)
    return polygon


def build_building_3d(
    building_model: BuildingModel,
    ceiling_height_m: float = 3.2,
    floor_id: str = "floor_3",
    wall_thickness_m: Optional[float] = None,
) -> Building3D:
    """
    Convert the current 2D BuildingModel to a simple 3D representation.

    The source wall/room/corridor x/y coordinates remain unchanged
    numerically. Supply calibrated physical coordinates in a later
    pipeline stage before interpreting horizontal distances in metres.

    Args:
        building_model: Existing 2D BuildingModel.
        ceiling_height_m: Height of the floor, in metres.
        floor_id: Identifier for this building floor.
        wall_thickness_m: Optional assumed wall thickness, in metres.

    Returns:
        Building3D containing converted walls, rooms, and corridors.
    """
    if not isinstance(building_model, BuildingModel):
        raise TypeError("building_model must be a BuildingModel.")

    if not floor_id:
        raise ValueError("floor_id must not be empty.")

    _validate_positive_finite(
        ceiling_height_m,
        "ceiling_height_m",
    )

    if wall_thickness_m is not None:
        _validate_positive_finite(
            wall_thickness_m,
            "wall_thickness_m",
        )

    walls = tuple(
        Wall3D(
            id=f"wall_{index}",
            start=(float(wall.x1), float(wall.y1)),
            end=(float(wall.x2), float(wall.y2)),
            height_m=ceiling_height_m,
            thickness_m=wall_thickness_m,
        )
        for index, wall in enumerate(building_model.walls)
    )

    rooms = tuple(
        Room3D(
            id=room.id,
            polygon=polygon_from_object(room),
            ceiling_height_m=ceiling_height_m,
        )
        for room in building_model.rooms
    )

    corridors = tuple(
        Corridor3D(
            id=corridor.id,
            polygon=polygon_from_object(corridor),
            ceiling_height_m=ceiling_height_m,
        )
        for corridor in building_model.corridors
    )

    return Building3D(
        floor_id=floor_id,
        ceiling_height_m=ceiling_height_m,
        walls=walls,
        rooms=rooms,
        corridors=corridors,
    )
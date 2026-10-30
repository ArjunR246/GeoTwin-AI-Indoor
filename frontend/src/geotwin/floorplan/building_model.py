from dataclasses import dataclass

from .corridors import Corridor
from .rooms import Room
from .walls import WallSegment


@dataclass(frozen=True)
class BuildingModel:
    """
    Unified geometric representation extracted from a floor plan.

    Coordinates are currently represented in image/pixel space.
    Physical-world scaling will be added later.
    """

    image_width: int
    image_height: int
    walls: tuple[WallSegment, ...]
    rooms: tuple[Room, ...]
    corridors: tuple[Corridor, ...]

    def __post_init__(self) -> None:
        """Validate the building model."""
        if self.image_width <= 0:
            raise ValueError("image_width must be positive.")

        if self.image_height <= 0:
            raise ValueError("image_height must be positive.")

        if not isinstance(self.walls, tuple):
            raise TypeError("walls must be a tuple.")

        if not isinstance(self.rooms, tuple):
            raise TypeError("rooms must be a tuple.")

        if not isinstance(self.corridors, tuple):
            raise TypeError("corridors must be a tuple.")

        if not all(
            isinstance(wall, WallSegment)
            for wall in self.walls
        ):
            raise TypeError(
                "Every wall must be a WallSegment."
            )

        if not all(
            isinstance(room, Room)
            for room in self.rooms
        ):
            raise TypeError(
                "Every room must be a Room."
            )

        if not all(
            isinstance(corridor, Corridor)
            for corridor in self.corridors
        ):
            raise TypeError(
                "Every corridor must be a Corridor."
            )

    @property
    def wall_count(self) -> int:
        """Return the number of detected walls."""
        return len(self.walls)

    @property
    def room_count(self) -> int:
        """Return the number of detected rooms."""
        return len(self.rooms)

    @property
    def corridor_count(self) -> int:
        """Return the number of detected corridors."""
        return len(self.corridors)

    @property
    def image_shape(self) -> tuple[int, int]:
        """Return image shape as (height, width)."""
        return self.image_height, self.image_width

    @property
    def total_room_area(self) -> float:
        """Return the total detected room area in pixels²."""
        return sum(room.area for room in self.rooms)

    @property
    def total_corridor_area(self) -> float:
        """Return the total detected corridor area in pixels²."""
        return sum(
            corridor.area
            for corridor in self.corridors
        )


def build_building_model(
    image_width: int,
    image_height: int,
    walls: list[WallSegment],
    rooms: list[Room],
    corridors: list[Corridor],
) -> BuildingModel:
    """
    Build a unified BuildingModel from floor-plan detections.

    Args:
        image_width:
            Width of the source floor-plan image in pixels.

        image_height:
            Height of the source floor-plan image in pixels.

        walls:
            Detected wall segments.

        rooms:
            Detected enclosed spaces.

        corridors:
            Detected corridor candidates.

    Returns:
        A validated BuildingModel.
    """
    if image_width <= 0:
        raise ValueError(
            "image_width must be positive."
        )

    if image_height <= 0:
        raise ValueError(
            "image_height must be positive."
        )

    if not isinstance(walls, list):
        raise TypeError("walls must be a list.")

    if not isinstance(rooms, list):
        raise TypeError("rooms must be a list.")

    if not isinstance(corridors, list):
        raise TypeError("corridors must be a list.")

    return BuildingModel(
        image_width=image_width,
        image_height=image_height,
        walls=tuple(walls),
        rooms=tuple(rooms),
        corridors=tuple(corridors),
    )
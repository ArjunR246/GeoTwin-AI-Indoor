from dataclasses import dataclass

import cv2
import numpy as np

from .rooms import Room


@dataclass(frozen=True)
class Corridor:
    """Represents a corridor-like enclosed region in a floor plan."""

    id: str
    polygon: tuple[tuple[int, int], ...]
    area: float
    centroid: tuple[float, float]
    bounding_box: tuple[int, int, int, int]
    aspect_ratio: float

    @property
    def width(self) -> int:
        """Return corridor bounding-box width."""
        return self.bounding_box[2]

    @property
    def height(self) -> int:
        """Return corridor bounding-box height."""
        return self.bounding_box[3]

    @property
    def length(self) -> int:
        """Return the longer bounding-box dimension."""
        return max(self.width, self.height)

    @property
    def narrow_dimension(self) -> int:
        """Return the shorter bounding-box dimension."""
        return min(self.width, self.height)


def _room_to_corridor(
    room: Room,
    corridor_index: int,
) -> Corridor:
    """Convert a Room candidate into a Corridor."""
    return Corridor(
        id=f"corridor_{corridor_index}",
        polygon=room.polygon,
        area=room.area,
        centroid=room.centroid,
        bounding_box=room.bounding_box,
        aspect_ratio=room.aspect_ratio,
    )


def _is_corridor_candidate(
    room: Room,
    min_aspect_ratio: float,
    max_width: float | None,
    min_area: float,
) -> bool:
    """
    Determine whether an enclosed region is corridor-like.

    A corridor is initially identified using geometric characteristics:
        - sufficiently elongated
        - sufficiently large
        - optionally limited in width
    """
    if room.area < min_area:
        return False

    if room.aspect_ratio < min_aspect_ratio:
        return False

    if max_width is not None:
        narrow_dimension = min(room.width, room.height)

        if narrow_dimension > max_width:
            return False

    return True


def detect_corridors(
    rooms: list[Room],
    min_aspect_ratio: float = 3.0,
    max_width: float | None = None,
    min_area: float = 1000.0,
) -> list[Corridor]:
    """
    Detect corridor-like spaces from already detected enclosed regions.

    Args:
        rooms:
            Candidate enclosed regions returned by detect_rooms().

        min_aspect_ratio:
            Minimum length-to-width ratio required for a corridor.

        max_width:
            Optional maximum corridor width in pixels.

        min_area:
            Minimum corridor area.

    Returns:
        List of detected Corridor objects.
    """
    if not isinstance(rooms, list):
        raise TypeError("rooms must be a list.")

    if min_aspect_ratio <= 0:
        raise ValueError("min_aspect_ratio must be positive.")

    if max_width is not None and max_width <= 0:
        raise ValueError("max_width must be positive.")

    if min_area < 0:
        raise ValueError("min_area cannot be negative.")

    corridors: list[Corridor] = []

    for room in rooms:
        if not isinstance(room, Room):
            raise TypeError("Every item in rooms must be a Room.")

        # Normalize the aspect ratio so both horizontal and vertical
        # elongated spaces are treated equally.
        normalized_ratio = max(
            room.width / room.height,
            room.height / room.width,
        )

        candidate_room = Room(
            id=room.id,
            polygon=room.polygon,
            area=room.area,
            centroid=room.centroid,
            bounding_box=room.bounding_box,
            aspect_ratio=float(normalized_ratio),
        )

        if not _is_corridor_candidate(
            candidate_room,
            min_aspect_ratio=min_aspect_ratio,
            max_width=max_width,
            min_area=min_area,
        ):
            continue

        corridors.append(
            _room_to_corridor(
                candidate_room,
                len(corridors) + 1,
            )
        )

    return corridors


def corridor_mask(
    image_shape: tuple[int, int],
    corridors: list[Corridor],
) -> np.ndarray:
    """
    Create a binary mask containing the detected corridors.

    White pixels represent corridor regions.

    Args:
        image_shape:
            Tuple containing (height, width).

        corridors:
            Detected corridor objects.

    Returns:
        NumPy uint8 binary mask.
    """
    if len(image_shape) != 2:
        raise ValueError(
            "image_shape must contain (height, width)."
        )

    height, width = image_shape

    if height <= 0 or width <= 0:
        raise ValueError(
            "Image dimensions must be positive."
        )

    if not isinstance(corridors, list):
        raise TypeError(
            "corridors must be a list."
        )

    mask = np.zeros(
        (height, width),
        dtype=np.uint8,
    )

    for corridor in corridors:
        if not isinstance(corridor, Corridor):
            raise TypeError(
                "Every item in corridors must be a Corridor."
            )

        polygon = np.asarray(
            corridor.polygon,
            dtype=np.int32,
        )

        if polygon.shape[0] < 3:
            continue

        cv2.fillPoly(
            mask,
            [polygon],
            255,
        )

    return mask
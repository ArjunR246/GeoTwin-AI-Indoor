
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class Room:
    """Represents an enclosed region detected in a floor plan."""

    id: str
    polygon: tuple[tuple[int, int], ...]
    area: float
    centroid: tuple[float, float]
    bounding_box: tuple[int, int, int, int]
    aspect_ratio: float

    @property
    def width(self) -> int:
        """Return room bounding-box width."""
        return self.bounding_box[2]

    @property
    def height(self) -> int:
        """Return room bounding-box height."""
        return self.bounding_box[3]


def _validate_binary(binary: np.ndarray) -> None:
    """Validate the floor-plan image."""
    if not isinstance(binary, np.ndarray):
        raise TypeError("Binary floor-plan image must be a NumPy array.")

    if binary.size == 0:
        raise ValueError("Binary floor-plan image is empty.")

    if binary.ndim != 2:
        raise ValueError("Binary floor-plan image must be single-channel.")

    if binary.dtype != np.uint8:
        raise TypeError("Binary floor-plan image must have dtype uint8.")


def _find_enclosed_regions(binary: np.ndarray) -> np.ndarray:
    """
    Find enclosed light regions surrounded by dark floor-plan structures.

    Small gaps in walls are closed before flood filling. The original
    thresholded image is retained for the final enclosed-region mask.
    """
    _, thresholded = cv2.threshold(
        binary,
        127,
        255,
        cv2.THRESH_BINARY,
    )

    # Close small breaks in wall strokes. Keep this kernel deliberately
    # small so nearby architectural spaces are less likely to merge.
    wall_mask = cv2.bitwise_not(thresholded)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (3, 3),
    )

    closed_walls = cv2.morphologyEx(
        wall_mask,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=1,
    )

    flood_input = cv2.bitwise_not(closed_walls)

    # Flood-fill the exterior from the top-left corner. Use a padded
    # mask and seed from every border-connected light pixel to avoid
    # relying on a single corner being clear.
    flood = flood_input.copy()

    flood_mask = np.zeros(
        (flood.shape[0] + 2, flood.shape[1] + 2),
        dtype=np.uint8,
    )

    height, width = flood.shape

    seeds = (
        [(x, 0) for x in range(width)]
        + [(x, height - 1) for x in range(width)]
        + [(0, y) for y in range(height)]
        + [(width - 1, y) for y in range(height)]
    )

    for seed_x, seed_y in seeds:
        if flood[seed_y, seed_x] == 255:
            cv2.floodFill(
                flood,
                flood_mask,
                (seed_x, seed_y),
                128,
            )

    # Only return original light pixels belonging to regions that
    # remain enclosed after closing small wall gaps.
    enclosed = np.where(
        (thresholded == 255) & (flood != 128),
        255,
        0,
    ).astype(np.uint8)

    return enclosed


def _contour_to_room(
    contour: np.ndarray,
    room_index: int,
) -> Room | None:
    """Convert a contour into a Room object."""
    area = float(cv2.contourArea(contour))

    if area <= 0:
        return None

    perimeter = cv2.arcLength(contour, True)

    polygon = cv2.approxPolyDP(
        contour,
        0.005 * perimeter,
        True,
    )

    if len(polygon) < 3:
        return None

    points = tuple(
        (int(point[0][0]), int(point[0][1]))
        for point in polygon
    )

    x, y, width, height = cv2.boundingRect(contour)

    if width <= 0 or height <= 0:
        return None

    moments = cv2.moments(contour)

    if moments["m00"] != 0:
        centroid_x = moments["m10"] / moments["m00"]
        centroid_y = moments["m01"] / moments["m00"]
    else:
        centroid_x = x + width / 2
        centroid_y = y + height / 2

    return Room(
        id=f"room_{room_index}",
        polygon=points,
        area=area,
        centroid=(float(centroid_x), float(centroid_y)),
        bounding_box=(x, y, width, height),
        aspect_ratio=float(width / height),
    )


def detect_rooms(
    binary: np.ndarray,
    min_area: float = 1000.0,
    max_area_ratio: float = 0.95,
) -> list[Room]:
    """
    Detect enclosed regions that may represent rooms.

    Args:
        binary:
            Preprocessed single-channel floor-plan image.
        min_area:
            Minimum enclosed-region area in pixels.
        max_area_ratio:
            Maximum allowed region area as a fraction of image area.

    Returns:
        Candidate Room objects sorted approximately top-to-bottom.
    """
    _validate_binary(binary)

    if min_area < 0:
        raise ValueError("min_area cannot be negative.")

    if not 0 < max_area_ratio <= 1:
        raise ValueError("max_area_ratio must be in the interval (0, 1].")

    height, width = binary.shape
    image_area = float(height * width)
    max_area = image_area * max_area_ratio

    enclosed = _find_enclosed_regions(binary)

    contours, _ = cv2.findContours(
        enclosed,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    candidates: list[Room] = []

    for contour in contours:
        area = float(cv2.contourArea(contour))

        if area < min_area or area > max_area:
            continue

        x, y, box_width, box_height = cv2.boundingRect(contour)

        # Reject regions touching the image border: they may be exterior
        # background regions rather than enclosed architectural spaces.
        if (
            x <= 0
            or y <= 0
            or x + box_width >= width
            or y + box_height >= height
        ):
            continue

        room = _contour_to_room(
            contour,
            len(candidates) + 1,
        )

        if room is not None:
            candidates.append(room)

    candidates.sort(
        key=lambda room: (
            round(room.centroid[1] / 20),
            room.centroid[0],
        )
    )

    return [
        Room(
            id=f"room_{index}",
            polygon=room.polygon,
            area=room.area,
            centroid=room.centroid,
            bounding_box=room.bounding_box,
            aspect_ratio=room.aspect_ratio,
        )
        for index, room in enumerate(candidates, start=1)
    ]


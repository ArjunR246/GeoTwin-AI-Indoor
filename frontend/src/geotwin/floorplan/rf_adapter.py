"""
GeoTwin AI Indoor - Floor-plan to RF adapter.

Converts floor-plan geometry from pixel coordinates into the
physical coordinate system used by the RF and optimization engines.
"""

from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np

from .building_model import BuildingModel
from .coordinates import CoordinateTransform
from .walls import WallSegment


Point = tuple[float, float]


def transform_wall(
    wall: WallSegment,
    coordinates: CoordinateTransform,
) -> WallSegment:
    """
    Convert one wall from pixel coordinates to physical coordinates.

    The RF engine operates in metres, while floor-plan detection
    operates in pixels.

    WallSegment is reused so it remains compatible with the
    existing wall-crossing and attenuation functions.
    """

    start = coordinates.pixel_to_physical(
        (wall.x1, wall.y1)
    )

    end = coordinates.pixel_to_physical(
        (wall.x2, wall.y2)
    )

    return WallSegment(
        x1=start[0],
        y1=start[1],
        x2=end[0],
        y2=end[1],
        orientation=wall.orientation,
        length=coordinates.pixel_segment_length_to_metres(
            (wall.x1, wall.y1),
            (wall.x2, wall.y2),
        ),
    )


def transform_walls(
    walls: Sequence[WallSegment],
    coordinates: CoordinateTransform,
) -> tuple[WallSegment, ...]:
    """
    Convert all detected walls from pixels to metres.
    """

    return tuple(
        transform_wall(
            wall,
            coordinates,
        )
        for wall in walls
    )


def floor_plan_to_rf_geometry(
    building: BuildingModel,
    coordinates: CoordinateTransform,
    cell_size_m: float,
    indoor_mask: np.ndarray | None = None,
    ap_edge_margin_m: float = 1.0,
    min_indoor_share: float = 0.3,
) -> tuple[
    list[Point],
    list[Point],
    list[Point],
    tuple[WallSegment, ...],
]:
    """
    Convert a BuildingModel into RF/optimization geometry.

    Returns:

        receiver_positions
        valid_ap_positions
        blocked_positions
        physical_walls

    Coordinates returned by this function are in metres.
    """

    if not isinstance(building, BuildingModel):
        raise TypeError(
            "building must be a BuildingModel."
        )

    if not isinstance(
        coordinates,
        CoordinateTransform,
    ):
        raise TypeError(
            "coordinates must be a CoordinateTransform."
        )

    if cell_size_m <= 0:
        raise ValueError(
            "cell_size_m must be greater than zero."
        )

    from .digital_twin import build_digital_twin_grid

    # DigitalTwinGrid itself remains in pixel space.
    # We use the physical scale only when exporting positions.
    cell_size_px_x = (
        cell_size_m
        * coordinates.pixels_per_metre_x
    )

    cell_size_px_y = (
        cell_size_m
        * coordinates.pixels_per_metre_y
    )

    # A rectangular RF grid requires one cell size.
    # Use the smaller pixel dimension so the requested physical
    # resolution is not exceeded in either direction.
    cell_size_px = min(
        cell_size_px_x,
        cell_size_px_y,
    )

    grid = build_digital_twin_grid(
        model=building,
        cell_size=cell_size_px,
    )

    edge_distance = None
    if indoor_mask is not None:
        from .indoor import interior_distance_m
        metres_per_pixel = 1.0 / min(
            coordinates.pixels_per_metre_x,
            coordinates.pixels_per_metre_y,
        )
        edge_distance = interior_distance_m(indoor_mask, metres_per_pixel)
        # Share of indoor pixels in a cell-sized window around every pixel.
        window = max(1, int(round(cell_size_px)))
        indoor_share = cv2.boxFilter(
            (indoor_mask > 0).astype(np.float32), -1, (window, window),
            normalize=True, borderType=cv2.BORDER_CONSTANT,
        )

    receiver_positions: list[Point] = []
    valid_ap_positions: list[Point] = []
    blocked_positions: list[Point] = []

    for cell in grid.cells:
        physical_position = coordinates.pixel_to_physical(
            cell.center
        )

        ap_allowed = True
        if indoor_mask is not None:
            # Usable if the cell centre lies in detected indoor space
            # (rooms, corridors and halls), see indoor.py.
            column = min(int(cell.center[0]), indoor_mask.shape[1] - 1)
            row = min(int(cell.center[1]), indoor_mask.shape[0] - 1)
            # A cell counts if enough of its area is indoor. Checking only
            # the centre pixel dropped cells whose centre happened to land
            # on a wall line, text or a desk outline.
            usable = indoor_share[row, column] >= min_indoor_share
            # APs are not placed right at the building's outer edge.
            ap_allowed = edge_distance[row, column] >= ap_edge_margin_m
        else:
            # Original behaviour: only enclosed rooms and corridors.
            usable = cell.occupied

        if usable:
            receiver_positions.append(
                physical_position
            )
            if ap_allowed:
                valid_ap_positions.append(
                    physical_position
                )
        else:
            blocked_positions.append(
                physical_position
            )

    physical_walls = transform_walls(
        building.walls,
        coordinates,
    )

    if not receiver_positions:
        raise ValueError(
            "Floor plan produced no occupied receiver positions."
        )

    if not valid_ap_positions:
        raise ValueError(
            "Floor plan produced no valid AP positions."
        )

    return (
        receiver_positions,
        valid_ap_positions,
        blocked_positions,
        physical_walls,
    )


__all__ = [
    "transform_wall",
    "transform_walls",
    "floor_plan_to_rf_geometry",
]
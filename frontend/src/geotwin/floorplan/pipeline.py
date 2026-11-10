"""
GeoTwin AI Indoor - Floor-plan processing pipeline.

Converts a floor-plan image into a unified BuildingModel and,
when physical dimensions are available, a CoordinateTransform.

Features:
    - Image validation
    - Optional region-of-interest (ROI) cropping
    - Original-image coordinate preservation
    - Wall detection
    - Enclosed-region detection
    - Corridor candidate detection
    - Removal of corridor candidates from the room collection
    - Optional physical coordinate transformation

ROI format:
    (x, y, width, height)

All geometry stored in BuildingModel uses coordinates relative
to the original, uncropped image.

Physical dimensions, when supplied, describe the original image
coordinate system.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import TypeAlias

import numpy as np

from .building_model import (
    BuildingModel,
    build_building_model,
)
from .coordinates import CoordinateTransform
from .corridors import Corridor, detect_corridors
from .loader import load_floor_plan
from .preprocessing import preprocess_floor_plan
from .rooms import Room, detect_rooms
from .walls import WallSegment, detect_walls, detect_walls_robust


ROI: TypeAlias = tuple[int, int, int, int]


@dataclass(frozen=True)
class FloorPlanPipelineResult:
    """
    Complete result of floor-plan processing.

    Attributes:
        image_width:
            Width of the original image in pixels.

        image_height:
            Height of the original image in pixels.

        building:
            Extracted building geometry in original-image coordinates.

        coordinates:
            Optional transformation between original-image pixels
            and physical coordinates.

        preprocessing:
            Intermediate preprocessing images. When an ROI is used,
            these images correspond to the cropped image.

        roi:
            Validated ROI expressed as (x, y, width, height), or None
            when the complete image was processed.

        coordinate_offset:
            (x, y) offset of the cropped image relative to the
            original image.
    """

    image_width: int
    image_height: int

    building: BuildingModel

    coordinates: CoordinateTransform | None

    preprocessing: dict[str, np.ndarray]

    roi: ROI | None = None

    coordinate_offset: tuple[int, int] = (0, 0)

    @property
    def has_physical_scale(self) -> bool:
        """Return True when physical dimensions were supplied."""
        return self.coordinates is not None

    @property
    def wall_count(self) -> int:
        """Return the number of detected walls."""
        return self.building.wall_count

    @property
    def room_count(self) -> int:
        """Return the number of detected rooms."""
        return self.building.room_count

    @property
    def corridor_count(self) -> int:
        """Return the number of detected corridors."""
        return self.building.corridor_count


def _validate_image(image: np.ndarray) -> None:
    """
    Validate an input floor-plan image.

    Supported formats:
        - Two-dimensional grayscale images
        - Three-dimensional images with 1, 3, or 4 channels
    """

    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a NumPy array.")

    if image.size == 0:
        raise ValueError("image cannot be empty.")

    if image.ndim not in (2, 3):
        raise ValueError(
            "image must be a 2D grayscale or 3D color image."
        )

    if image.ndim == 3 and image.shape[2] not in (1, 3, 4):
        raise ValueError(
            "image must have 1, 3, or 4 channels."
        )


def _validate_roi(
    roi: ROI,
    image_width: int,
    image_height: int,
) -> ROI:
    """
    Validate an ROI expressed as (x, y, width, height).

    Coordinates must be integers, and the complete ROI must lie
    inside the original image.
    """

    if not isinstance(roi, (tuple, list)) or len(roi) != 4:
        raise ValueError(
            "roi must contain (x, y, width, height)."
        )

    values: list[int] = []

    for value in roi:
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, np.integer)
        ):
            raise TypeError("ROI coordinates must be integers.")

        values.append(int(value))

    x, y, width, height = values

    if x < 0 or y < 0:
        raise ValueError("ROI x and y must be non-negative.")

    if width <= 0 or height <= 0:
        raise ValueError(
            "ROI width and height must be positive."
        )

    if x + width > image_width or y + height > image_height:
        raise ValueError(
            "ROI extends beyond the input image boundaries."
        )

    return x, y, width, height


def _offset_region(
    region: Room | Corridor,
    offset_x: int,
    offset_y: int,
) -> Room | Corridor:
    """
    Translate a room or corridor from cropped-image coordinates
    into original-image coordinates.
    """

    polygon = tuple(
        (int(x + offset_x), int(y + offset_y))
        for x, y in region.polygon
    )

    centroid = (
        float(region.centroid[0] + offset_x),
        float(region.centroid[1] + offset_y),
    )

    box_x, box_y, box_width, box_height = region.bounding_box

    bounding_box = (
        int(box_x + offset_x),
        int(box_y + offset_y),
        int(box_width),
        int(box_height),
    )

    return replace(
        region,
        polygon=polygon,
        centroid=centroid,
        bounding_box=bounding_box,
    )


def _offset_wall(
    wall: WallSegment,
    offset_x: int,
    offset_y: int,
) -> WallSegment:
    """
    Translate a wall segment from cropped-image coordinates
    into original-image coordinates.
    """

    return replace(
        wall,
        x1=wall.x1 + offset_x,
        y1=wall.y1 + offset_y,
        x2=wall.x2 + offset_x,
        y2=wall.y2 + offset_y,
    )


def _validate_physical_dimensions(
    physical_width_m: float | None,
    physical_height_m: float | None,
) -> None:
    """Validate optional physical dimensions."""

    if (
        physical_width_m is None
        and physical_height_m is not None
    ):
        raise ValueError(
            "physical_width_m must be provided when "
            "physical_height_m is provided."
        )

    if (
        physical_width_m is not None
        and physical_height_m is None
    ):
        raise ValueError(
            "physical_height_m must be provided when "
            "physical_width_m is provided."
        )

    if physical_width_m is not None:
        if (
            not np.isfinite(physical_width_m)
            or physical_width_m <= 0
        ):
            raise ValueError(
                "physical_width_m must be finite and positive."
            )

    if physical_height_m is not None:
        if (
            not np.isfinite(physical_height_m)
            or physical_height_m <= 0
        ):
            raise ValueError(
                "physical_height_m must be finite and positive."
            )


def process_floor_plan(
    image: np.ndarray,
    *,
    roi: ROI | None = None,
    physical_width_m: float | None = None,
    physical_height_m: float | None = None,
    origin_x_m: float = 0.0,
    origin_y_m: float = 0.0,
    room_min_area: float = 1000.0,
    room_max_area_ratio: float = 0.95,
    corridor_min_aspect_ratio: float = 3.0,
    corridor_max_width: float | None = None,
    corridor_min_area: float = 1000.0,
) -> FloorPlanPipelineResult:
    """
    Process a floor-plan image.

    Args:
        image:
            Grayscale or color floor-plan image.

        roi:
            Optional region of interest in original-image pixels,
            expressed as (x, y, width, height).

        physical_width_m:
            Physical width of the original image in metres.

        physical_height_m:
            Physical height of the original image in metres.

        origin_x_m:
            Physical X-coordinate of the original image origin.

        origin_y_m:
            Physical Y-coordinate of the original image origin.

        room_min_area:
            Minimum enclosed-region area in pixels.

        room_max_area_ratio:
            Maximum enclosed-region area as a fraction of the
            image being processed.

        corridor_min_aspect_ratio:
            Minimum length-to-width ratio for corridor candidates.

        corridor_max_width:
            Optional maximum corridor width in pixels.

        corridor_min_area:
            Minimum corridor area in pixels.

    Returns:
        FloorPlanPipelineResult.

    Notes:
        Geometry is translated back into original-image coordinates
        when an ROI is used.

        The ROI only restricts the image-processing area. It does not
        automatically identify the exact building footprint.

        Physical dimensions describe the original image, not the crop.
    """

    # ---------------------------------------------------------
    # 1. VALIDATION
    # ---------------------------------------------------------

    _validate_image(image)

    _validate_physical_dimensions(
        physical_width_m,
        physical_height_m,
    )

    if not np.isfinite(origin_x_m) or not np.isfinite(origin_y_m):
        raise ValueError(
            "Physical coordinate origins must be finite."
        )

    if not np.isfinite(room_min_area) or room_min_area < 0:
        raise ValueError(
            "room_min_area must be finite and non-negative."
        )

    if (
        not np.isfinite(room_max_area_ratio)
        or not 0 < room_max_area_ratio <= 1
    ):
        raise ValueError(
            "room_max_area_ratio must be in the interval (0, 1]."
        )

    if (
        not np.isfinite(corridor_min_aspect_ratio)
        or corridor_min_aspect_ratio <= 0
    ):
        raise ValueError(
            "corridor_min_aspect_ratio must be finite and positive."
        )

    if corridor_max_width is not None:
        if (
            not np.isfinite(corridor_max_width)
            or corridor_max_width <= 0
        ):
            raise ValueError(
                "corridor_max_width must be finite and positive."
            )

    if (
        not np.isfinite(corridor_min_area)
        or corridor_min_area < 0
    ):
        raise ValueError(
            "corridor_min_area must be finite and non-negative."
        )

    image_height, image_width = image.shape[:2]

    # ---------------------------------------------------------
    # 2. APPLY OPTIONAL REGION OF INTEREST
    # ---------------------------------------------------------

    validated_roi: ROI | None = None
    offset_x = 0
    offset_y = 0

    if roi is None:
        processing_image = image
    else:
        validated_roi = _validate_roi(
            roi,
            image_width=image_width,
            image_height=image_height,
        )

        offset_x, offset_y, roi_width, roi_height = validated_roi

        processing_image = image[
            offset_y:offset_y + roi_height,
            offset_x:offset_x + roi_width,
        ]

    # ---------------------------------------------------------
    # 3. PREPROCESSING
    # ---------------------------------------------------------

    preprocessing = preprocess_floor_plan(processing_image)

    cleaned = preprocessing["cleaned"]
    edges = preprocessing["edges"]

    # ---------------------------------------------------------
    # 4. WALL DETECTION
    # ---------------------------------------------------------

    # detect_walls_robust ignores text, furniture and other small symbols,
    # and reports each wall once. The older detect_walls(cleaned, edges) is
    # still available in walls.py for comparison.
    walls = detect_walls_robust(preprocessing["binary"])

    # ---------------------------------------------------------
    # 5. ENCLOSED-REGION / ROOM CANDIDATE DETECTION
    # ---------------------------------------------------------

    # The binary image is used for room candidates because previous
    # diagnostics showed that the cleaned image merged or removed
    # many enclosed regions on this floor plan.
    room_detection_image = preprocessing["binary"]

    detected_regions = detect_rooms(
        room_detection_image,
        min_area=room_min_area,
        max_area_ratio=room_max_area_ratio,
    )

    # ---------------------------------------------------------
    # 6. CORRIDOR CANDIDATE DETECTION
    # ---------------------------------------------------------

    corridors = detect_corridors(
        detected_regions,
        min_aspect_ratio=corridor_min_aspect_ratio,
        max_width=corridor_max_width,
        min_area=corridor_min_area,
    )

    # ---------------------------------------------------------
    # 7. REMOVE CORRIDORS FROM ROOM CANDIDATES
    # ---------------------------------------------------------

    # Corridor IDs are assigned independently by detect_corridors(),
    # so corridor_2 does not necessarily correspond to room_2.
    # Match using the original polygon instead of replacing ID text.
    corridor_polygons = {
        corridor.polygon
        for corridor in corridors
    }

    rooms = [
        room
        for room in detected_regions
        if room.polygon not in corridor_polygons
    ]

    # ---------------------------------------------------------
    # 8. RESTORE ORIGINAL IMAGE COORDINATES
    # ---------------------------------------------------------

    if validated_roi is not None:
        walls = [
            _offset_wall(wall, offset_x, offset_y)
            for wall in walls
        ]

        rooms = [
            _offset_region(room, offset_x, offset_y)
            for room in rooms
        ]

        corridors = [
            _offset_region(corridor, offset_x, offset_y)
            for corridor in corridors
        ]

    # ---------------------------------------------------------
    # 9. BUILD UNIFIED BUILDING MODEL
    # ---------------------------------------------------------

    building = build_building_model(
        image_width=image_width,
        image_height=image_height,
        walls=walls,
        rooms=rooms,
        corridors=corridors,
    )

    # ---------------------------------------------------------
    # 10. OPTIONAL PHYSICAL COORDINATE TRANSFORMATION
    # ---------------------------------------------------------

    coordinates = None

    if (
        physical_width_m is not None
        and physical_height_m is not None
    ):
        coordinates = CoordinateTransform(
            image_width_px=image_width,
            image_height_px=image_height,
            physical_width_m=physical_width_m,
            physical_height_m=physical_height_m,
            origin_x_m=origin_x_m,
            origin_y_m=origin_y_m,
        )

    # ---------------------------------------------------------
    # 11. RETURN PIPELINE RESULT
    # ---------------------------------------------------------

    return FloorPlanPipelineResult(
        image_width=image_width,
        image_height=image_height,
        building=building,
        coordinates=coordinates,
        preprocessing=preprocessing,
        roi=validated_roi,
        coordinate_offset=(offset_x, offset_y),
    )


def process_floor_plan_file(
    path: str | Path,
    *,
    roi: ROI | None = None,
    physical_width_m: float | None = None,
    physical_height_m: float | None = None,
    origin_x_m: float = 0.0,
    origin_y_m: float = 0.0,
    room_min_area: float = 1000.0,
    room_max_area_ratio: float = 0.95,
    corridor_min_aspect_ratio: float = 3.0,
    corridor_max_width: float | None = None,
    corridor_min_area: float = 1000.0,
) -> FloorPlanPipelineResult:
    """
    Load and process a floor-plan image from disk.

    Args:
        path:
            Path to the floor-plan image.

        roi:
            Optional ROI expressed as (x, y, width, height)
            in original-image coordinates.

        physical_width_m:
            Physical width of the original image in metres.

        physical_height_m:
            Physical height of the original image in metres.

        origin_x_m:
            Physical X-coordinate of the original image origin.

        origin_y_m:
            Physical Y-coordinate of the original image origin.

        room_min_area:
            Minimum enclosed-region area in pixels.

        room_max_area_ratio:
            Maximum enclosed-region area ratio.

        corridor_min_aspect_ratio:
            Minimum corridor aspect ratio.

        corridor_max_width:
            Optional maximum corridor width in pixels.

        corridor_min_area:
            Minimum corridor area in pixels.

    Returns:
        FloorPlanPipelineResult.
    """

    image = load_floor_plan(path)

    return process_floor_plan(
        image,
        roi=roi,
        physical_width_m=physical_width_m,
        physical_height_m=physical_height_m,
        origin_x_m=origin_x_m,
        origin_y_m=origin_y_m,
        room_min_area=room_min_area,
        room_max_area_ratio=room_max_area_ratio,
        corridor_min_aspect_ratio=corridor_min_aspect_ratio,
        corridor_max_width=corridor_max_width,
        corridor_min_area=corridor_min_area,
    )


__all__ = [
    "ROI",
    "FloorPlanPipelineResult",
    "process_floor_plan",
    "process_floor_plan_file",
]
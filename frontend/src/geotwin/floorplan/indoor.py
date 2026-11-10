"""
GeoTwin AI Indoor - Indoor (usable) space detection.

The room detector only finds fully enclosed regions. Open corridors and halls
are often not enclosed, so they were left out, while closed shapes outside the
building (signs, legend boxes, callouts) were wrongly counted as rooms.

detect_indoor_space() instead finds everything inside the building outline:

    1. Make a mask of dark strokes and drop small blobs (text, symbols).
    2. Thicken the strokes so door and window gaps are temporarily sealed.
    3. Flood-fill from the image border: everything reached is outside.
    4. Undo the thickening on the outside region, so the building outline
       is not shrunk.
    5. Inside = not outside. Keep the main building (and any other large
       parts), drop small separate islands such as signs. Limitation:
       something drawn very close to the building (within the sealing
       distance) is merged with it.
    6. Safety net: an enclosed room found by the room detector is added back
       if it lies within the building's overall outline (convex hull).
    7. Remove the wall pixels themselves.

remove_background_pockets() then removes outside areas that arrows or signs
closed off, using colour (see its docstring).

All sizes are relative to the image, so the same settings work for small and
large floorplans.
"""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

from .walls import remove_small_components

Polygon = Sequence[tuple[int, int]]


def detect_indoor_space(
    binary: np.ndarray,
    room_polygons: Sequence[Polygon] = (),
    gap_ratio: float = 0.03,
    min_component_ratio: float = 0.06,
    keep_ratio: float = 0.1,
) -> np.ndarray:
    """Return a uint8 mask (255 = usable indoor space) for a floor plan.

    Args:
        binary:
            Thresholded floor plan (preprocessing["binary"]): dark lines are 0.
        room_polygons:
            Polygons of rooms/corridors from the room detector, used only as a
            safety net so properly enclosed rooms are never lost.
        gap_ratio:
            Openings up to about 2 x gap_ratio x (longer image side) wide,
            such as doors and windows, are sealed while finding the outside.
        min_component_ratio:
            Blobs smaller than this fraction of the longer side are ignored
            (text, furniture, symbols).
        keep_ratio:
            Separate building parts smaller than this fraction of the largest
            part are treated as non-building (signs, legends) and dropped.
    """
    if binary is None or binary.size == 0:
        raise ValueError("Binary floor-plan image is empty.")
    if binary.ndim != 2:
        raise ValueError("Binary floor-plan image must be single-channel.")

    height, width = binary.shape
    longer_side = max(height, width)

    # 1. Dark strokes, without text and small symbols.
    ink = np.where(binary < 128, 255, 0).astype(np.uint8)
    if ink.mean() > 127:
        ink = cv2.bitwise_not(ink)
    ink = remove_small_components(ink, min_component_ratio * longer_side)

    # 2. Seal doors and other gaps by thickening the strokes.
    radius = max(2, int(gap_ratio * longer_side))
    disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    sealed = cv2.dilate(ink, disk)

    # 3. Flood-fill the free space from a 1-pixel frame around the image.
    free = cv2.copyMakeBorder(
        cv2.bitwise_not(sealed), 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=255
    )
    fill_mask = np.zeros((height + 4, width + 4), np.uint8)
    cv2.floodFill(free, fill_mask, (0, 0), 128)
    outside = np.where(free[1:-1, 1:-1] == 128, 255, 0).astype(np.uint8)

    # 4. Grow the outside back by the same amount it was shrunk in step 2.
    outside = cv2.dilate(outside, disk)

    # 5. Building footprint (walls included); keep only the large parts.
    #    Note: anything drawn closer to the building than the sealing distance
    #    (about 2 x gap_ratio of the image size) is merged with it.
    footprint = np.where(outside == 0, 255, 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(footprint, 8)
    if count > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        keep = np.zeros(count, dtype=bool)
        keep[1:] = areas >= keep_ratio * areas.max()
        footprint = np.where(keep[labels], 255, 0).astype(np.uint8)

    # 6. Safety net: add back enclosed rooms that lie inside the building outline.
    points = cv2.findNonZero(footprint)
    if points is not None and room_polygons:
        hull = np.zeros_like(footprint)
        cv2.fillPoly(hull, [cv2.convexHull(points)], 255)
        for polygon in room_polygons:
            if len(polygon) < 3:
                continue
            room = np.zeros_like(footprint)
            cv2.fillPoly(room, [np.asarray(polygon, dtype=np.int32)], 255)
            area = int((room > 0).sum())
            if area and int(((room > 0) & (hull > 0)).sum()) >= 0.9 * area:
                footprint = cv2.bitwise_or(footprint, room)

    # 7. Usable space = footprint minus the walls/lines themselves.
    return np.where((footprint > 0) & (ink == 0), 255, 0).astype(np.uint8)


def remove_background_pockets(
    image: np.ndarray,
    binary: np.ndarray,
    indoor: np.ndarray,
    min_contrast: float = 25.0,
    edge_ratio: float = 0.07,
) -> np.ndarray:
    """Remove outside "pockets" that were mistaken for indoor space.

    Arrows, callouts and signs drawn on the background can cut the outside
    into closed pockets next to the building. Those pockets still have the
    background's colour (e.g. the green behind a photographed plan), while
    floors inside the building have a different colour.

    A piece of indoor space is removed when:
      * the outside and the building floor clearly differ in colour (if the
        plan is, say, white outside and white inside, nothing is removed);
      * the piece lies near the building edge (within edge_ratio of the
        image size from the definite outside), so symbols deep inside the
        building are never touched; and
      * most of its pixels are closer in colour to the outside than to the
        building floor.
    """
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    height, width = binary.shape
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    ink = binary < 128
    outside = (indoor == 0) & ~ink

    # Typical outside colour, sampled along the image border.
    band = np.zeros((height, width), dtype=bool)
    b = max(3, int(0.02 * max(height, width)))
    band[:b] = band[-b:] = True
    band[:, :b] = band[:, -b:] = True
    outside_pixels = lab[outside & band]
    inside_pixels = lab[indoor > 0]
    if len(outside_pixels) < 100 or len(inside_pixels) < 100:
        return indoor
    outside_colour = np.median(outside_pixels, axis=0)
    inside_colour = np.median(inside_pixels, axis=0)
    if np.linalg.norm(outside_colour - inside_colour) < min_contrast:
        return indoor

    closer_to_outside = (
        np.linalg.norm(lab - outside_colour, axis=2)
        < np.linalg.norm(lab - inside_colour, axis=2)
    )
    distance_to_outside = cv2.distanceTransform(
        (~outside).astype(np.uint8), cv2.DIST_L2, 3
    )
    near_edge = distance_to_outside <= edge_ratio * max(height, width)

    pieces = ((indoor > 0) & ~ink).astype(np.uint8)
    count, labels = cv2.connectedComponents(pieces, connectivity=4)
    result = indoor.copy()
    for index in range(1, count):
        piece = labels == index
        if near_edge[piece].any() and closer_to_outside[piece].mean() > 0.5:
            result[piece] = 0

    return result



def interior_distance_m(indoor: np.ndarray, metres_per_pixel: float) -> np.ndarray:
    """Distance (metres) from each pixel to the building's outer edge.

    Wall lines inside the building are first closed up, so the distance is
    measured to the outside or the image border, not to the nearest wall.
    """
    height, width = indoor.shape
    thickness = max(3, int(0.01 * max(height, width)))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (thickness, thickness))
    building = cv2.morphologyEx(indoor, cv2.MORPH_CLOSE, kernel)
    building = cv2.copyMakeBorder(building, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    distance = cv2.distanceTransform((building > 0).astype(np.uint8), cv2.DIST_L2, 5)
    return distance[1:-1, 1:-1] * metres_per_pixel


__all__ = ["detect_indoor_space", "remove_background_pockets", "interior_distance_m"]

from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np


Orientation = Literal["horizontal", "vertical"]


@dataclass(frozen=True)
class WallSegment:
    """Represents a detected straight wall segment in image coordinates."""

    x1: int
    y1: int
    x2: int
    y2: int
    orientation: Orientation
    length: float

    @property
    def midpoint(self) -> tuple[float, float]:
        """Return the midpoint of the wall segment."""
        return (
            (self.x1 + self.x2) / 2.0,
            (self.y1 + self.y2) / 2.0,
        )


def _normalize_line(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
) -> tuple[int, int, int, int, Orientation]:
    """
    Normalize a line so horizontal/vertical segments have
    a consistent coordinate ordering.
    """
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)

    if dx >= dy:
        orientation: Orientation = "horizontal"

        if x1 > x2:
            x1, x2 = x2, x1
            y1, y2 = y2, y1
    else:
        orientation = "vertical"

        if y1 > y2:
            x1, x2 = x2, x1
            y1, y2 = y2, y1

    return x1, y1, x2, y2, orientation


def _line_length(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
) -> float:
    """Calculate Euclidean line length."""
    return float(np.hypot(x2 - x1, y2 - y1))


def _extract_morphological_lines(
    binary: np.ndarray,
    orientation: Orientation,
    min_length: int,
) -> np.ndarray:
    """
    Extract horizontal or vertical architectural lines
    using morphological operations.
    """
    if binary.ndim != 2:
        raise ValueError("Wall detection requires a grayscale/binary image.")

    if binary.size == 0:
        raise ValueError("Wall detection input is empty.")

    height, width = binary.shape

    if orientation == "horizontal":
        kernel_length = max(10, width // 20)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (kernel_length, 1),
        )
    else:
        kernel_length = max(10, height // 20)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (1, kernel_length),
        )

    lines = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        kernel,
    )

    # Remove tiny isolated structures.
    if min_length > 1:
        if orientation == "horizontal":
            small_kernel = cv2.getStructuringElement(
                cv2.MORPH_RECT,
                (min_length, 1),
            )
        else:
            small_kernel = cv2.getStructuringElement(
                cv2.MORPH_RECT,
                (1, min_length),
            )

        lines = cv2.morphologyEx(
            lines,
            cv2.MORPH_OPEN,
            small_kernel,
        )

    return lines


def _segments_from_mask(
    mask: np.ndarray,
    orientation: Orientation,
    min_length: float,
) -> list[WallSegment]:
    """Convert a binary line mask into wall segments."""
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    segments: list[WallSegment] = []

    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)

        if orientation == "horizontal":
            if width < min_length:
                continue

            x1 = x
            y1 = y + height // 2
            x2 = x + width - 1
            y2 = y1

        else:
            if height < min_length:
                continue

            x1 = x + width // 2
            y1 = y
            x2 = x1
            y2 = y + height - 1

        x1, y1, x2, y2, normalized_orientation = _normalize_line(
            x1,
            y1,
            x2,
            y2,
        )

        segments.append(
            WallSegment(
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                orientation=normalized_orientation,
                length=_line_length(x1, y1, x2, y2),
            )
        )

    return segments


def _detect_hough_lines(
    edges: np.ndarray,
    min_length: float,
) -> list[WallSegment]:
    """Detect strong horizontal/vertical lines using Hough transform."""
    if edges.ndim != 2:
        raise ValueError("Hough wall detection requires a single-channel image.")

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=50,
        minLineLength=max(20, int(min_length)),
        maxLineGap=10,
    )

    if lines is None:
        return []

    segments: list[WallSegment] = []

    for line in lines:
        values = np.asarray(line).reshape(-1)

        if values.size != 4:
            continue

        x1, y1, x2, y2 = map(int, values)

        dx = abs(x2 - x1)
        dy = abs(y2 - y1)

        # Only retain predominantly horizontal/vertical lines.
        if dx >= dy * 3:
            orientation: Orientation = "horizontal"
        elif dy >= dx * 3:
            orientation = "vertical"
        else:
            continue

        x1, y1, x2, y2, orientation = _normalize_line(
            x1,
            y1,
            x2,
            y2,
        )

        length = _line_length(x1, y1, x2, y2)

        if length < min_length:
              continue

        segments.append(
            WallSegment(
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                orientation=orientation,
                length=length,
            )
        )

    return segments


def _deduplicate_segments(
    segments: list[WallSegment],
    coordinate_tolerance: int = 5,
) -> list[WallSegment]:
    """
    Remove nearly identical wall detections.

    This is intentionally conservative. More advanced wall merging
    will be implemented after we validate the first detector.
    """
    unique: list[WallSegment] = []

    for candidate in segments:
        duplicate = False

        for existing in unique:
            if candidate.orientation != existing.orientation:
                continue

            if candidate.orientation == "horizontal":
                same_axis = abs(candidate.y1 - existing.y1) <= coordinate_tolerance
                overlap = min(candidate.x2, existing.x2) >= max(
                    candidate.x1,
                    existing.x1,
                )
            else:
                same_axis = abs(candidate.x1 - existing.x1) <= coordinate_tolerance
                overlap = min(candidate.y2, existing.y2) >= max(
                    candidate.y1,
                    existing.y1,
                )

            if same_axis and overlap:
                duplicate = True
                break

        if not duplicate:
            unique.append(candidate)

    return unique


def detect_walls(
    binary: np.ndarray,
    edges: np.ndarray | None = None,
    min_length: float = 30.0,
) -> list[WallSegment]:
    """
    Detect predominantly horizontal and vertical wall segments.

    Args:
        binary:
            Binary floor-plan image.
        edges:
            Optional edge image. If provided, Hough detection is also used.
        min_length:
            Minimum wall segment length in pixels.

    Returns:
        List of detected WallSegment objects.
    """
    if binary is None or binary.size == 0:
        raise ValueError("Binary floor-plan image is empty.")

    if binary.ndim != 2:
        raise ValueError("Binary floor-plan image must be single-channel.")

    # Ensure a clean binary representation.
    _, clean_binary = cv2.threshold(
        binary,
        127,
        255,
        cv2.THRESH_BINARY,
    )

    horizontal_mask = _extract_morphological_lines(
        clean_binary,
        "horizontal",
        int(min_length),
    )

    vertical_mask = _extract_morphological_lines(
        clean_binary,
        "vertical",
        int(min_length),
    )

    segments = []

    segments.extend(
        _segments_from_mask(
            horizontal_mask,
            "horizontal",
            min_length,
        )
    )

    segments.extend(
        _segments_from_mask(
            vertical_mask,
            "vertical",
            min_length,
        )
    )

    # Hough detection adds robustness when walls are slightly broken.
    if edges is not None:
        if edges.size == 0:
            raise ValueError("Edge image is empty.")

        segments.extend(
            _detect_hough_lines(
                edges,
                min_length,
            )
        )

    return _deduplicate_segments(segments)

# ---------------------------------------------------------------------------
# Robust wall detection
# ---------------------------------------------------------------------------
#
# detect_walls() above treats every long dark line as a wall, so text, desks,
# door arcs and stair lines are often reported as walls, and thick walls are
# reported twice (once per edge). detect_walls_robust() removes that clutter
# first. Every setting is relative to the image (its size, or the measured
# wall thickness), so the same code works for small and large floorplans.


def estimate_wall_thickness(ink: np.ndarray) -> float:
    """Estimate wall thickness in pixels from a mask of dark strokes.

    The distance transform gives, for every stroke pixel, its distance to the
    nearest background pixel. At the centre of a stroke this is half the
    stroke's thickness. Walls are usually the boldest lines on a plan, so the
    90th percentile of those centre values approximates wall thickness.
    """
    distance = cv2.distanceTransform(ink, cv2.DIST_L2, 3)
    centre = (distance >= cv2.dilate(distance, np.ones((3, 3), np.uint8))) & (distance > 0)
    values = distance[centre]
    if values.size == 0:
        return 3.0
    return float(max(2.0, 2.0 * np.percentile(values, 90)))


def remove_small_components(ink: np.ndarray, min_extent: float) -> np.ndarray:
    """Remove isolated blobs (letters, furniture, symbols) smaller than min_extent px.

    Walls connect into long structures, while each letter or desk is a small
    separate blob. A blob is kept only if its width or height reaches min_extent.
    """
    count, labels, stats, _ = cv2.connectedComponentsWithStats(ink, 8)
    keep = np.zeros(count, dtype=bool)
    extent = np.maximum(stats[1:, cv2.CC_STAT_WIDTH], stats[1:, cv2.CC_STAT_HEIGHT])
    keep[1:] = extent >= min_extent
    return np.where(keep[labels], 255, 0).astype(np.uint8)


def merge_collinear_segments(
    segments: list[WallSegment],
    offset_tolerance: float,
    gap_tolerance: float,
) -> list[WallSegment]:
    """Join pieces of the same wall: same orientation, almost the same line,
    and overlapping or separated by a small gap."""
    merged: list[WallSegment] = []
    for orientation in ("horizontal", "vertical"):
        lines = sorted(
            (s.y1, s.x1, s.x2) if orientation == "horizontal" else (s.x1, s.y1, s.y2)
            for s in segments
            if s.orientation == orientation
        )
        groups: list[list[float]] = []   # [position, start, end]
        for position, start, end in lines:
            for group in groups:
                if (
                    abs(group[0] - position) <= offset_tolerance
                    and start <= group[2] + gap_tolerance
                    and end >= group[1] - gap_tolerance
                ):
                    old_length = group[2] - group[1] + 1
                    new_length = end - start + 1
                    group[0] = (group[0] * old_length + position * new_length) / (old_length + new_length)
                    group[1] = min(group[1], start)
                    group[2] = max(group[2], end)
                    break
            else:
                groups.append([position, start, end])

        for position, start, end in groups:
            p = int(round(position))
            if orientation == "horizontal":
                coords = (int(start), p, int(end), p)
            else:
                coords = (p, int(start), p, int(end))
            x1, y1, x2, y2, normalized = _normalize_line(*coords)
            merged.append(
                WallSegment(x1, y1, x2, y2, normalized, _line_length(x1, y1, x2, y2))
            )
    return merged


def detect_walls_robust(
    binary: np.ndarray,
    min_length_ratio: float = 0.03,
    min_component_ratio: float = 0.06,
    thin_stroke_ratio: float = 0.5,
) -> list[WallSegment]:
    """Detect horizontal and vertical walls while ignoring text and furniture.

    Args:
        binary:
            Thresholded floor plan (preprocessing["binary"]): dark lines are 0,
            background is 255.
        min_length_ratio:
            Shortest wall to report, as a fraction of the image's longer side.
        min_component_ratio:
            Blobs smaller than this fraction of the image's longer side are
            treated as text or furniture and removed.
        thin_stroke_ratio:
            Strokes thinner than this fraction of the measured wall thickness
            are treated as furniture/annotation lines and removed.

    Steps:
        1. Make a mask of dark strokes (white = ink). Flip it if the plan uses
           light lines on a dark background.
        2. Remove small separate blobs: letters, desks, chairs, symbols.
        3. Measure the wall thickness, then erase strokes much thinner than it.
        4. Keep only long horizontal / vertical runs (morphological opening).
        5. Turn each run into one centre-line segment and merge pieces that
           belong to the same wall.
    """
    if binary is None or binary.size == 0:
        raise ValueError("Binary floor-plan image is empty.")
    if binary.ndim != 2:
        raise ValueError("Binary floor-plan image must be single-channel.")

    height, width = binary.shape
    longer_side = max(height, width)

    # 1. Ink mask.
    ink = np.where(binary < 128, 255, 0).astype(np.uint8)
    if ink.mean() > 127:
        ink = cv2.bitwise_not(ink)

    # 2. Remove text, furniture and symbols.
    ink = remove_small_components(ink, min_component_ratio * longer_side)

    # 3. Remove strokes much thinner than walls.
    thickness = estimate_wall_thickness(ink)
    kernel_size = max(2, int(round(thin_stroke_ratio * thickness)))
    walls_only = cv2.morphologyEx(
        ink, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size)),
    )

    # 4 and 5. Long straight runs -> centre-line segments.
    min_length = max(15, int(min_length_ratio * longer_side))
    segments: list[WallSegment] = []
    for orientation in ("horizontal", "vertical"):
        shape = (min_length, 1) if orientation == "horizontal" else (1, min_length)
        mask = cv2.morphologyEx(
            walls_only, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, shape)
        )
        count, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        for index in range(1, count):
            x, y, w, h = (int(v) for v in stats[index, :4])
            along, across = (w, h) if orientation == "horizontal" else (h, w)
            # Very thick runs are filled areas (dark blocks, hatching), not walls.
            if along < min_length or across > 4 * thickness + 2:
                continue
            if orientation == "horizontal":
                coords = (x, y + h // 2, x + w - 1, y + h // 2)
            else:
                coords = (x + w // 2, y, x + w // 2, y + h - 1)
            x1, y1, x2, y2, normalized = _normalize_line(*coords)
            segments.append(
                WallSegment(x1, y1, x2, y2, normalized, _line_length(x1, y1, x2, y2))
            )

    return merge_collinear_segments(
        segments,
        offset_tolerance=max(3.0, thickness),
        gap_tolerance=max(2.0, thickness / 2),
    )

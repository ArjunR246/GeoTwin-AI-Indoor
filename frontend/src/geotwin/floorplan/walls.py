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
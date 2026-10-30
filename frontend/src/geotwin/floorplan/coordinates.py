"""
GeoTwin AI Indoor - Floor-plan coordinate transformation.

Provides conversion between floor-plan pixel coordinates and
physical building coordinates.

The transformation supports:
    pixel coordinates <-> physical coordinates

Physical dimensions are normally expressed in metres.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from typing import Tuple


Point = Tuple[float, float]


@dataclass(frozen=True)
class CoordinateTransform:
    """
    Convert floor-plan pixel coordinates into physical coordinates.

    Coordinate convention:

        Pixel:
            x increases to the right
            y increases downward

        Physical:
            x increases to the right
            y increases downward

    The orientation is intentionally kept identical to image
    coordinates so that geometry extracted from OpenCV can be
    transformed without changing its shape.

    Args:
        image_width_px:
            Width of the floor-plan image in pixels.

        image_height_px:
            Height of the floor-plan image in pixels.

        physical_width_m:
            Real-world width represented by the floor plan.

        physical_height_m:
            Real-world height represented by the floor plan.

        origin_x_m:
            Optional physical X-coordinate of the image origin.

        origin_y_m:
            Optional physical Y-coordinate of the image origin.
    """

    image_width_px: int
    image_height_px: int

    physical_width_m: float
    physical_height_m: float

    origin_x_m: float = 0.0
    origin_y_m: float = 0.0

    def __post_init__(self) -> None:
        """Validate transformation parameters."""

        if self.image_width_px <= 0:
            raise ValueError(
                "image_width_px must be positive."
            )

        if self.image_height_px <= 0:
            raise ValueError(
                "image_height_px must be positive."
            )

        if self.physical_width_m <= 0:
            raise ValueError(
                "physical_width_m must be positive."
            )

        if self.physical_height_m <= 0:
            raise ValueError(
                "physical_height_m must be positive."
            )

    @property
    def metres_per_pixel_x(self) -> float:
        """Return physical width represented by one X pixel."""
        return (
            self.physical_width_m
            / self.image_width_px
        )

    @property
    def metres_per_pixel_y(self) -> float:
        """Return physical height represented by one Y pixel."""
        return (
            self.physical_height_m
            / self.image_height_px
        )

    @property
    def pixels_per_metre_x(self) -> float:
        """Return number of X pixels per physical metre."""
        return (
            self.image_width_px
            / self.physical_width_m
        )

    @property
    def pixels_per_metre_y(self) -> float:
        """Return number of Y pixels per physical metre."""
        return (
            self.image_height_px
            / self.physical_height_m
        )

    @property
    def is_uniform_scale(self) -> bool:
        """
        Return whether X and Y use the same scale.

        A uniform scale is normally expected for an undistorted
        floor-plan image.
        """
        return abs(
            self.metres_per_pixel_x
            - self.metres_per_pixel_y
        ) < 1e-12

    def pixel_to_physical(
        self,
        point: Point,
    ) -> Point:
        """
        Convert a pixel coordinate to a physical coordinate.

        Example:

            (500, 250) pixels
                ↓
            (10.0, 5.0) metres
        """

        self._validate_point(point)

        x_px, y_px = point

        x_m = (
            self.origin_x_m
            + x_px * self.metres_per_pixel_x
        )

        y_m = (
            self.origin_y_m
            + y_px * self.metres_per_pixel_y
        )

        return (
            float(x_m),
            float(y_m),
        )

    def physical_to_pixel(
        self,
        point: Point,
    ) -> Point:
        """
        Convert a physical coordinate to a pixel coordinate.
        """

        self._validate_point(point)

        x_m, y_m = point

        x_px = (
            x_m - self.origin_x_m
        ) * self.pixels_per_metre_x

        y_px = (
            y_m - self.origin_y_m
        ) * self.pixels_per_metre_y

        return (
            float(x_px),
            float(y_px),
        )

    def pixel_distance_to_metres(
        self,
        distance_x_px: float,
        distance_y_px: float,
    ) -> float:
        """
        Convert a pixel-space displacement into physical metres.

        Both X and Y scaling are respected.
        """

        if distance_x_px < 0:
            raise ValueError(
                "distance_x_px cannot be negative."
            )

        if distance_y_px < 0:
            raise ValueError(
                "distance_y_px cannot be negative."
            )

        physical_x = (
            distance_x_px
            * self.metres_per_pixel_x
        )

        physical_y = (
            distance_y_px
            * self.metres_per_pixel_y
        )

        return float(
            hypot(
                physical_x,
                physical_y,
            )
        )

    def metres_distance_to_pixels(
        self,
        distance_x_m: float,
        distance_y_m: float,
    ) -> float:
        """
        Convert a physical displacement into pixel distance.

        Both X and Y scaling are respected.
        """

        if distance_x_m < 0:
            raise ValueError(
                "distance_x_m cannot be negative."
            )

        if distance_y_m < 0:
            raise ValueError(
                "distance_y_m cannot be negative."
            )

        pixel_x = (
            distance_x_m
            * self.pixels_per_metre_x
        )

        pixel_y = (
            distance_y_m
            * self.pixels_per_metre_y
        )

        return float(
            hypot(
                pixel_x,
                pixel_y,
            )
        )

    def pixel_segment_length_to_metres(
        self,
        start: Point,
        end: Point,
    ) -> float:
        """Calculate physical length of a pixel-space segment."""

        self._validate_point(start)
        self._validate_point(end)

        dx_px = abs(
            end[0] - start[0]
        )

        dy_px = abs(
            end[1] - start[1]
        )

        return self.pixel_distance_to_metres(
            dx_px,
            dy_px,
        )

    def physical_segment_length_to_pixels(
        self,
        start: Point,
        end: Point,
    ) -> float:
        """Calculate pixel length of a physical segment."""

        self._validate_point(start)
        self._validate_point(end)

        dx_m = abs(
            end[0] - start[0]
        )

        dy_m = abs(
            end[1] - start[1]
        )

        return self.metres_distance_to_pixels(
            dx_m,
            dy_m,
        )

    @staticmethod
    def _validate_point(
        point: Point,
    ) -> None:
        """Validate a 2D coordinate."""

        if not isinstance(point, (tuple, list)):
            raise TypeError(
                "Point must be a tuple or list."
            )

        if len(point) != 2:
            raise ValueError(
                "Point must contain exactly two coordinates."
            )

        try:
            float(point[0])
            float(point[1])
        except (TypeError, ValueError):
            raise TypeError(
                "Point coordinates must be numeric."
            ) from None


def create_coordinate_transform(
    image_width_px: int,
    image_height_px: int,
    physical_width_m: float,
    physical_height_m: float,
    origin_x_m: float = 0.0,
    origin_y_m: float = 0.0,
) -> CoordinateTransform:
    """
    Create a validated coordinate transformation.

    This factory function provides a simple public API for
    applications that do not need to instantiate the dataclass
    directly.
    """

    return CoordinateTransform(
        image_width_px=image_width_px,
        image_height_px=image_height_px,
        physical_width_m=physical_width_m,
        physical_height_m=physical_height_m,
        origin_x_m=origin_x_m,
        origin_y_m=origin_y_m,
    )
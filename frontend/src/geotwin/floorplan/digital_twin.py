from dataclasses import dataclass

import cv2
import numpy as np

from .building_model import BuildingModel


@dataclass(frozen=True)
class DigitalTwinCell:
    """Represents one cell in the automatically generated digital twin."""

    row: int
    column: int

    x: float
    y: float

    width: float
    height: float

    occupied: bool
    room_id: str | None
    corridor_id: str | None

    @property
    def center(self) -> tuple[float, float]:
        """Return the cell center coordinate."""
        return (
            self.x + self.width / 2.0,
            self.y + self.height / 2.0,
        )


@dataclass(frozen=True)
class DigitalTwinGrid:
    """Regular grid generated from a BuildingModel."""

    rows: int
    columns: int

    cell_width: float
    cell_height: float

    cells: tuple[DigitalTwinCell, ...]

    def __post_init__(self) -> None:
        """Validate the generated grid."""
        if self.rows <= 0:
            raise ValueError("rows must be positive.")

        if self.columns <= 0:
            raise ValueError("columns must be positive.")

        if self.cell_width <= 0:
            raise ValueError("cell_width must be positive.")

        if self.cell_height <= 0:
            raise ValueError("cell_height must be positive.")

        expected_cells = self.rows * self.columns

        if len(self.cells) != expected_cells:
            raise ValueError(
                "Number of cells does not match rows × columns."
            )

    @property
    def cell_count(self) -> int:
        """Return the total number of grid cells."""
        return len(self.cells)

    @property
    def occupied_cells(self) -> tuple[DigitalTwinCell, ...]:
        """Return cells that belong to the detected building."""
        return tuple(
            cell for cell in self.cells
            if cell.occupied
        )

    @property
    def free_cells(self) -> tuple[DigitalTwinCell, ...]:
        """Return cells outside the detected building."""
        return tuple(
            cell for cell in self.cells
            if not cell.occupied
        )

    @property
    def occupied_count(self) -> int:
        """Return the number of occupied cells."""
        return len(self.occupied_cells)

    @property
    def free_count(self) -> int:
        """Return the number of free cells."""
        return len(self.free_cells)

    def cell_at(
        self,
        row: int,
        column: int,
    ) -> DigitalTwinCell:
        """Return a specific grid cell."""
        if not 0 <= row < self.rows:
            raise IndexError("row is outside the grid.")

        if not 0 <= column < self.columns:
            raise IndexError("column is outside the grid.")

        index = row * self.columns + column

        return self.cells[index]


def _point_in_polygon(
    point: tuple[float, float],
    polygon: tuple[tuple[int, int], ...],
) -> bool:
    """Return whether a point lies inside a polygon."""
    if len(polygon) < 3:
        return False

    contour = np.asarray(
        polygon,
        dtype=np.float32,
    )

    result = cv2.pointPolygonTest(
        contour,
        point,
        False,
    )

    return result >= 0


def _find_room_at_point(
    model: BuildingModel,
    point: tuple[float, float],
) -> str | None:
    """Return the room containing a point, if any."""
    for room in model.rooms:
        if _point_in_polygon(point, room.polygon):
            return room.id

    return None


def _find_corridor_at_point(
    model: BuildingModel,
    point: tuple[float, float],
) -> str | None:
    """Return the corridor containing a point, if any."""
    for corridor in model.corridors:
        if _point_in_polygon(point, corridor.polygon):
            return corridor.id

    return None


def _point_near_wall(
    model: BuildingModel,
    point: tuple[float, float],
    tolerance: float,
) -> bool:
    """
    Determine whether a point lies close to a detected wall.

    This helper is intentionally simple for now.
    A precise wall-crossing model will be added later.
    """
    px, py = point

    for wall in model.walls:
        if wall.orientation == "horizontal":
            if wall.x1 <= px <= wall.x2:
                if abs(py - wall.y1) <= tolerance:
                    return True

        else:
            if wall.y1 <= py <= wall.y2:
                if abs(px - wall.x1) <= tolerance:
                    return True

    return False


def build_digital_twin_grid(
    model: BuildingModel,
    cell_size: float,
    wall_tolerance: float | None = None,
) -> DigitalTwinGrid:
    """
    Automatically generate a digital-twin grid from a BuildingModel.

    Coordinates remain in image/pixel space.

    Args:
        model:
            Unified building geometry model.

        cell_size:
            Grid-cell size in image pixels.

        wall_tolerance:
            Optional distance used when evaluating wall proximity.

    Returns:
        DigitalTwinGrid containing every generated cell.
    """
    if not isinstance(model, BuildingModel):
        raise TypeError(
            "model must be a BuildingModel."
        )

    if cell_size <= 0:
        raise ValueError(
            "cell_size must be positive."
        )

    if wall_tolerance is None:
        wall_tolerance = max(1.0, cell_size * 0.1)

    if wall_tolerance < 0:
        raise ValueError(
            "wall_tolerance cannot be negative."
        )

    rows = int(
        np.ceil(model.image_height / cell_size)
    )

    columns = int(
        np.ceil(model.image_width / cell_size)
    )

    cells: list[DigitalTwinCell] = []

    for row in range(rows):
        for column in range(columns):
            x = column * cell_size
            y = row * cell_size

            width = min(
                cell_size,
                model.image_width - x,
            )

            height = min(
                cell_size,
                model.image_height - y,
            )

            center = (
                x + width / 2.0,
                y + height / 2.0,
            )

            room_id = _find_room_at_point(
                model,
                center,
            )

            corridor_id = _find_corridor_at_point(
                model,
                center,
            )

            # A cell is considered occupied if it belongs to
            # a detected room or corridor.
            occupied = (
                room_id is not None
                or corridor_id is not None
            )

            # Evaluate wall proximity now so that the information
            # is available for later attenuation work.
            _point_near_wall(
                model,
                center,
                wall_tolerance,
            )

            cells.append(
                DigitalTwinCell(
                    row=row,
                    column=column,
                    x=float(x),
                    y=float(y),
                    width=float(width),
                    height=float(height),
                    occupied=occupied,
                    room_id=room_id,
                    corridor_id=corridor_id,
                )
            )

    return DigitalTwinGrid(
        rows=rows,
        columns=columns,
        cell_width=float(cell_size),
        cell_height=float(cell_size),
        cells=tuple(cells),
    )
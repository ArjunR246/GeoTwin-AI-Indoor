import pytest

from geotwin.floorplan.building_model import (
    BuildingModel,
)
from geotwin.floorplan.corridors import Corridor
from geotwin.floorplan.digital_twin import (
    DigitalTwinCell,
    DigitalTwinGrid,
    build_digital_twin_grid,
)
from geotwin.floorplan.rooms import Room
from geotwin.floorplan.walls import WallSegment


@pytest.fixture
def wall():
    return WallSegment(
        x1=0,
        y1=0,
        x2=100,
        y2=0,
        orientation="horizontal",
        length=100.0,
    )


@pytest.fixture
def room():
    return Room(
        id="room_1",
        polygon=(
            (10, 10),
            (100, 10),
            (100, 100),
            (10, 100),
        ),
        area=8100.0,
        centroid=(55.0, 55.0),
        bounding_box=(10, 10, 90, 90),
        aspect_ratio=1.0,
    )


@pytest.fixture
def corridor():
    return Corridor(
        id="corridor_1",
        polygon=(
            (10, 200),
            (500, 200),
            (500, 300),
            (10, 300),
        ),
        area=49000.0,
        centroid=(255.0, 250.0),
        bounding_box=(10, 200, 490, 100),
        aspect_ratio=4.9,
    )


@pytest.fixture
def model(wall, room, corridor):
    return BuildingModel(
        image_width=1200,
        image_height=800,
        walls=(wall,),
        rooms=(room,),
        corridors=(corridor,),
    )


def test_digital_twin_cell_center():
    cell = DigitalTwinCell(
        row=2,
        column=3,
        x=30.0,
        y=20.0,
        width=10.0,
        height=10.0,
        occupied=True,
        room_id="room_1",
        corridor_id=None,
    )

    assert cell.center == (35.0, 25.0)


def test_grid_creation():
    cells = tuple(
        DigitalTwinCell(
            row=row,
            column=column,
            x=float(column * 10),
            y=float(row * 10),
            width=10.0,
            height=10.0,
            occupied=False,
            room_id=None,
            corridor_id=None,
        )
        for row in range(2)
        for column in range(3)
    )

    grid = DigitalTwinGrid(
        rows=2,
        columns=3,
        cell_width=10.0,
        cell_height=10.0,
        cells=cells,
    )

    assert grid.rows == 2
    assert grid.columns == 3
    assert grid.cell_count == 6


def test_grid_automatically_generates_expected_dimensions(model):
    grid = build_digital_twin_grid(
        model,
        cell_size=100.0,
    )

    assert grid.rows == 8
    assert grid.columns == 12
    assert grid.cell_count == 96


def test_grid_contains_occupied_cells(model):
    grid = build_digital_twin_grid(
        model,
        cell_size=10.0,
    )

    assert grid.occupied_count > 0


def test_grid_cell_at_returns_correct_cell(model):
    grid = build_digital_twin_grid(
        model,
        cell_size=100.0,
    )

    cell = grid.cell_at(0, 0)

    assert cell.row == 0
    assert cell.column == 0


def test_grid_cell_room_assignment(model):
    grid = build_digital_twin_grid(
        model,
        cell_size=10.0,
    )

    room_cells = [
        cell
        for cell in grid.cells
        if cell.room_id == "room_1"
    ]

    assert len(room_cells) > 0


def test_grid_corridor_assignment(model):
    grid = build_digital_twin_grid(
        model,
        cell_size=10.0,
    )

    corridor_cells = [
        cell
        for cell in grid.cells
        if cell.corridor_id == "corridor_1"
    ]

    assert len(corridor_cells) > 0


def test_occupied_and_free_counts_add_up(model):
    grid = build_digital_twin_grid(
        model,
        cell_size=20.0,
    )

    assert (
        grid.occupied_count
        + grid.free_count
        == grid.cell_count
    )


def test_last_cells_can_have_smaller_dimensions(model):
    grid = build_digital_twin_grid(
        model,
        cell_size=300.0,
    )

    last_column_cell = grid.cell_at(
        0,
        grid.columns - 1,
    )

    assert last_column_cell.width <= 300.0


def test_invalid_cell_size_is_rejected(model):
    with pytest.raises(ValueError):
        build_digital_twin_grid(
            model,
            cell_size=0,
        )


def test_invalid_model_is_rejected():
    with pytest.raises(TypeError):
        build_digital_twin_grid(
            None,
            cell_size=10.0,
        )


def test_grid_requires_positive_rows():
    with pytest.raises(ValueError):
        DigitalTwinGrid(
            rows=0,
            columns=1,
            cell_width=1.0,
            cell_height=1.0,
            cells=(),
        )


def test_grid_requires_positive_columns():
    with pytest.raises(ValueError):
        DigitalTwinGrid(
            rows=1,
            columns=0,
            cell_width=1.0,
            cell_height=1.0,
            cells=(),
        )


def test_grid_cell_count_must_match_dimensions():
    with pytest.raises(ValueError):
        DigitalTwinGrid(
            rows=2,
            columns=2,
            cell_width=10.0,
            cell_height=10.0,
            cells=(),
        )


def test_invalid_cell_lookup_is_rejected(model):
    grid = build_digital_twin_grid(
        model,
        cell_size=100.0,
    )

    with pytest.raises(IndexError):
        grid.cell_at(-1, 0)

    with pytest.raises(IndexError):
        grid.cell_at(0, grid.columns)


def test_negative_wall_tolerance_is_rejected(model):
    with pytest.raises(ValueError):
        build_digital_twin_grid(
            model,
            cell_size=10.0,
            wall_tolerance=-1.0,
        )
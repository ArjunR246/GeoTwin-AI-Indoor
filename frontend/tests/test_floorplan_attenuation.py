import numpy as np
import pytest

from geotwin.floorplan.attenuation import (
    WallCrossingResult,
    build_attenuation_map,
    build_wall_crossing_map,
    calculate_path_attenuation,
    calculate_wall_attenuation,
    count_wall_crossings,
    segments_intersect,
)
from geotwin.floorplan.building_model import BuildingModel
from geotwin.floorplan.digital_twin import build_digital_twin_grid
from geotwin.floorplan.walls import WallSegment


def make_horizontal_wall(
    y: int,
    x1: int = 0,
    x2: int = 100,
) -> WallSegment:
    return WallSegment(
        x1=x1,
        y1=y,
        x2=x2,
        y2=y,
        orientation="horizontal",
        length=float(abs(x2 - x1)),
    )


def make_vertical_wall(
    x: int,
    y1: int = 0,
    y2: int = 100,
) -> WallSegment:
    return WallSegment(
        x1=x,
        y1=y1,
        x2=x,
        y2=y2,
        orientation="vertical",
        length=float(abs(y2 - y1)),
    )


def make_grid(
    width: int = 100,
    height: int = 100,
    cell_size: float = 10.0,
):
    model = BuildingModel(
        image_width=width,
        image_height=height,
        walls=(),
        rooms=(),
        corridors=(),
    )

    return build_digital_twin_grid(
        model,
        cell_size=cell_size,
    )


def test_intersecting_segments():
    assert segments_intersect(
        (0, 0),
        (100, 100),
        (0, 100),
        (100, 0),
    )


def test_non_intersecting_segments():
    assert not segments_intersect(
        (0, 0),
        (100, 0),
        (0, 100),
        (100, 100),
    )


def test_touching_segments_count_as_intersection():
    assert segments_intersect(
        (0, 0),
        (100, 0),
        (100, 0),
        (100, 100),
    )


def test_collinear_overlapping_segments():
    assert segments_intersect(
        (0, 0),
        (100, 0),
        (50, 0),
        (150, 0),
    )


def test_identical_segments_intersect():
    assert segments_intersect(
        (0, 0),
        (100, 0),
        (0, 0),
        (100, 0),
    )


def test_horizontal_wall_crossing():
    wall = make_horizontal_wall(50)

    crossings = count_wall_crossings(
        (50, 10),
        (50, 90),
        [wall],
    )

    assert crossings == 1


def test_vertical_wall_crossing():
    wall = make_vertical_wall(50)

    crossings = count_wall_crossings(
        (10, 50),
        (90, 50),
        [wall],
    )

    assert crossings == 1


def test_no_wall_crossing():
    wall = make_horizontal_wall(50)

    crossings = count_wall_crossings(
        (10, 10),
        (90, 10),
        [wall],
    )

    assert crossings == 0


def test_multiple_wall_crossings():
    walls = [
        make_horizontal_wall(30),
        make_horizontal_wall(60),
        make_horizontal_wall(90),
    ]

    crossings = count_wall_crossings(
        (50, 10),
        (50, 100),
        walls,
    )

    assert crossings == 3


def test_same_start_and_end_has_zero_crossings():
    wall = make_horizontal_wall(50)

    crossings = count_wall_crossings(
        (10, 10),
        (10, 10),
        [wall],
    )

    assert crossings == 0


def test_path_attenuation():
    walls = [
        make_horizontal_wall(30),
        make_horizontal_wall(60),
    ]

    result = calculate_path_attenuation(
        (50, 10),
        (50, 90),
        walls,
        wall_loss_db=5.0,
    )

    assert isinstance(result, WallCrossingResult)
    assert result.crossings == 2
    assert result.attenuation_db == 10.0


def test_zero_wall_loss():
    wall = make_horizontal_wall(50)

    result = calculate_path_attenuation(
        (50, 10),
        (50, 90),
        [wall],
        wall_loss_db=0.0,
    )

    assert result.crossings == 1
    assert result.attenuation_db == 0.0


def test_attenuation_map_shape():
    grid = make_grid()

    wall = make_horizontal_wall(50)

    attenuation = build_attenuation_map(
        grid,
        transmitter=(50, 10),
        walls=[wall],
        wall_loss_db=5.0,
    )

    assert isinstance(attenuation, np.ndarray)
    assert attenuation.shape == (
        grid.rows,
        grid.columns,
    )


def test_attenuation_map_dtype():
    grid = make_grid()

    attenuation = build_attenuation_map(
        grid,
        transmitter=(10, 10),
        walls=[],
        wall_loss_db=5.0,
    )

    assert attenuation.dtype == np.float64


def test_attenuation_map_without_walls_is_zero():
    grid = make_grid()

    attenuation = build_attenuation_map(
        grid,
        transmitter=(10, 10),
        walls=[],
        wall_loss_db=5.0,
    )

    assert np.all(attenuation == 0.0)


def test_attenuation_map_contains_wall_loss():
    grid = make_grid(
        width=100,
        height=100,
        cell_size=10,
    )

    wall = make_horizontal_wall(
        y=50,
        x1=0,
        x2=100,
    )

    attenuation = build_attenuation_map(
        grid,
        transmitter=(50, 10),
        walls=[wall],
        wall_loss_db=5.0,
    )

    # Cells above the wall should not cross it.
    assert attenuation[0, 5] == 0.0

    # Cells below the wall should cross it.
    assert attenuation[9, 5] == 5.0


def test_wall_crossing_map_shape():
    grid = make_grid()

    crossing_map = build_wall_crossing_map(
        grid,
        transmitter=(10, 10),
        walls=[],
    )

    assert crossing_map.shape == (
        grid.rows,
        grid.columns,
    )


def test_wall_crossing_map_without_walls_is_zero():
    grid = make_grid()

    crossing_map = build_wall_crossing_map(
        grid,
        transmitter=(10, 10),
        walls=[],
    )

    assert np.all(crossing_map == 0)


def test_invalid_wall_loss_is_rejected():
    grid = make_grid()

    with pytest.raises(ValueError):
        build_attenuation_map(
            grid,
            transmitter=(10, 10),
            walls=[],
            wall_loss_db=-1.0,
        )


def test_invalid_grid_is_rejected():
    with pytest.raises(TypeError):
        build_attenuation_map(
            None,
            transmitter=(10, 10),
            walls=[],
            wall_loss_db=5.0,
        )


def test_negative_tolerance_is_rejected():
    wall = make_horizontal_wall(50)

    with pytest.raises(ValueError):
        count_wall_crossings(
            (10, 10),
            (10, 90),
            [wall],
            tolerance=-1.0,
        )


def test_invalid_transmitter_is_rejected():
    grid = make_grid()

    with pytest.raises(ValueError):
        build_attenuation_map(
            grid,
            transmitter=(10,),
            walls=[],
            wall_loss_db=5.0,
        )

def test_calculate_wall_attenuation():
    wall = make_horizontal_wall(50)

    attenuation = calculate_wall_attenuation(
        transmitter=(50, 10),
        receiver=(50, 90),
        walls=[wall],
        wall_loss_db=5.0,
    )

    assert attenuation == 5.0


def test_calculate_wall_attenuation_multiple_walls():
    walls = [
        make_horizontal_wall(30),
        make_horizontal_wall(60),
        make_horizontal_wall(90),
    ]

    attenuation = calculate_wall_attenuation(
        transmitter=(50, 10),
        receiver=(50, 100),
        walls=walls,
        wall_loss_db=5.0,
    )

    assert attenuation == 15.0


def test_calculate_wall_attenuation_without_walls():
    attenuation = calculate_wall_attenuation(
        transmitter=(10, 10),
        receiver=(90, 90),
        walls=[],
        wall_loss_db=5.0,
    )

    assert attenuation == 0.0


def test_calculate_wall_attenuation_rejects_negative_loss():
    with pytest.raises(ValueError):
        calculate_wall_attenuation(
            transmitter=(10, 10),
            receiver=(90, 90),
            walls=[],
            wall_loss_db=-5.0,
        )
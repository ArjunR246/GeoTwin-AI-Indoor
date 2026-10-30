import pytest

from geotwin.floorplan.attenuation3d import (
    calculate_path_attenuation_3d,
    calculate_wall_attenuation_3d,
    count_wall_crossings_3d,
)
from geotwin.floorplan.geometry3d import Wall3D


@pytest.fixture
def wall():
    return Wall3D(
        id="wall_1",
        start=(5.0, 0.0),
        end=(5.0, 10.0),
        height_m=2.5,
    )


def test_path_crossing_below_wall_top_counts_crossing(wall):
    assert count_wall_crossings_3d(
        start=(0.0, 5.0, 1.0),
        end=(10.0, 5.0, 1.0),
        walls=[wall],
    ) == 1


def test_path_passing_above_wall_does_not_count(wall):
    assert count_wall_crossings_3d(
        start=(0.0, 5.0, 3.0),
        end=(10.0, 5.0, 3.0),
        walls=[wall],
    ) == 0


def test_sloping_path_uses_height_at_intersection(wall):
    # The path crosses x=5 halfway through. Its height there is 1.5 m.
    assert count_wall_crossings_3d(
        start=(0.0, 5.0, 3.0),
        end=(10.0, 5.0, 0.0),
        walls=[wall],
    ) == 1


def test_path_missing_wall_segment_does_not_count(wall):
    assert count_wall_crossings_3d(
        start=(0.0, 15.0, 1.0),
        end=(10.0, 15.0, 1.0),
        walls=[wall],
    ) == 0


def test_multiple_walls_are_counted():
    walls = [
        Wall3D(
            id="wall_1",
            start=(3.0, 0.0),
            end=(3.0, 10.0),
            height_m=2.5,
        ),
        Wall3D(
            id="wall_2",
            start=(7.0, 0.0),
            end=(7.0, 10.0),
            height_m=2.5,
        ),
    ]

    result = calculate_path_attenuation_3d(
        transmitter=(0.0, 5.0, 1.0),
        receiver=(10.0, 5.0, 1.0),
        walls=walls,
        wall_loss_db=5.0,
    )

    assert result.crossings == 2
    assert result.attenuation_db == pytest.approx(10.0)


def test_wall_attenuation_returns_zero_when_no_wall_is_crossed(wall):
    assert calculate_wall_attenuation_3d(
        transmitter=(0.0, 5.0, 3.0),
        receiver=(10.0, 5.0, 3.0),
        walls=[wall],
        wall_loss_db=5.0,
    ) == pytest.approx(0.0)


def test_negative_wall_loss_is_rejected(wall):
    with pytest.raises(ValueError):
        calculate_wall_attenuation_3d(
            transmitter=(0.0, 5.0, 1.0),
            receiver=(10.0, 5.0, 1.0),
            walls=[wall],
            wall_loss_db=-2.0,
        )


def test_invalid_point_is_rejected(wall):
    with pytest.raises(ValueError):
        count_wall_crossings_3d(
            start=(0.0, 5.0),
            end=(10.0, 5.0, 1.0),
            walls=[wall],
        )


def test_identical_endpoints_do_not_cross_walls(wall):
    point = (1.0, 1.0, 1.0)

    assert count_wall_crossings_3d(
        start=point,
        end=point,
        walls=[wall],
    ) == 0
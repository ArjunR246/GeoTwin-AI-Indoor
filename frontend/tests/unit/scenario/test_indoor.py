import pytest

from geotwin.scenario.indoor import (
    IndoorScenario,
    create_default_scenario,
)


def test_default_scenario():
    scenario = create_default_scenario()

    assert scenario.width_m == 20.0
    assert scenario.length_m == 15.0
    assert scenario.grid_resolution_m == 0.5
    assert scenario.floors == 1
    assert scenario.wall_material == "concrete"
    assert scenario.wall_thickness_m == 0.2


def test_grid_point_count():
    scenario = IndoorScenario(
        width_m=20.0,
        length_m=15.0,
        grid_resolution_m=0.5,
    )

    assert scenario.total_grid_points == 1271


def test_grid_contains_origin():
    scenario = create_default_scenario()

    assert (0.0, 0.0) in scenario.grid_positions


def test_grid_contains_building_corner():
    scenario = create_default_scenario()

    assert (20.0, 15.0) in scenario.grid_positions


def test_valid_ap_positions():
    scenario = create_default_scenario()

    assert (
        scenario.valid_ap_positions
        == scenario.grid_positions
    )


def test_no_blocked_positions_initially():
    scenario = create_default_scenario()

    assert scenario.blocked_positions == []


def test_receiver_positions_cover_grid():
    scenario = create_default_scenario()

    assert (
        scenario.receiver_positions
        == scenario.grid_positions
    )


def test_invalid_dimensions():
    scenario = IndoorScenario(
        width_m=0.0,
        length_m=15.0,
    )

    with pytest.raises(ValueError):
        scenario.validate()


def test_invalid_grid_resolution():
    scenario = IndoorScenario(
        width_m=20.0,
        length_m=15.0,
        grid_resolution_m=0.0,
    )

    with pytest.raises(ValueError):
        scenario.validate()


def test_invalid_floor_count():
    scenario = IndoorScenario(
        floors=0,
    )

    with pytest.raises(ValueError):
        scenario.validate()
import pytest

from geotwin.rf.propagation import calculate_propagation_loss


def test_propagation_without_walls_or_obstacles():
    result = calculate_propagation_loss(
        frequency_hz=2.4e9,
        distance_m=10.0,
        reference_distance_m=1.0,
        path_loss_exponent=3.0,
        number_of_walls=0,
        obstacle_loss_db=0.0,
    )

    assert result.path_loss_db > 0
    assert result.wall_loss_db == 0.0
    assert result.obstacle_loss_db == 0.0
    assert result.total_loss_db == pytest.approx(
        result.path_loss_db
    )


def test_propagation_with_walls():
    result = calculate_propagation_loss(
        frequency_hz=2.4e9,
        distance_m=10.0,
        reference_distance_m=1.0,
        path_loss_exponent=3.0,
        wall_material="concrete",
        number_of_walls=2,
        obstacle_loss_db=0.0,
    )

    assert result.wall_loss_db == 10.0
    assert result.total_loss_db == pytest.approx(
        result.path_loss_db + 10.0
    )


def test_propagation_with_obstacles():
    result = calculate_propagation_loss(
        frequency_hz=2.4e9,
        distance_m=10.0,
        obstacle_loss_db=4.0,
    )

    assert result.obstacle_loss_db == 4.0
    assert result.total_loss_db == pytest.approx(
        result.path_loss_db + 4.0
    )


def test_total_propagation_loss():
    result = calculate_propagation_loss(
        frequency_hz=2.4e9,
        distance_m=10.0,
        wall_material="brick",
        number_of_walls=2,
        obstacle_loss_db=3.0,
    )

    expected = (
        result.path_loss_db
        + result.wall_loss_db
        + 3.0
    )

    assert result.total_loss_db == pytest.approx(expected)


def test_negative_obstacle_loss():
    with pytest.raises(ValueError):
        calculate_propagation_loss(
            frequency_hz=2.4e9,
            distance_m=10.0,
            obstacle_loss_db=-1.0,
        )
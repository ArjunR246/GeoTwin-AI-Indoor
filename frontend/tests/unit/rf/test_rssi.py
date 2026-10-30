import pytest

from geotwin.rf.ap import AccessPoint
from geotwin.rf.rssi import calculate_rssi


@pytest.fixture
def ap():
    return AccessPoint(
        position=(0.0, 0.0),
        frequency_hz=2.4e9,
        tx_power_dbm=20.0,
        antenna_gain_dbi=3.0,
    )


def test_rssi_is_calculated(ap):
    rssi = calculate_rssi(
        ap=ap,
        distance_m=10.0,
    )

    assert rssi < ap.tx_power_dbm


def test_receiver_gain_increases_rssi(ap):
    rssi_without_gain = calculate_rssi(
        ap=ap,
        distance_m=10.0,
        receiver_gain_dbi=0.0,
    )

    rssi_with_gain = calculate_rssi(
        ap=ap,
        distance_m=10.0,
        receiver_gain_dbi=5.0,
    )

    assert rssi_with_gain == pytest.approx(
        rssi_without_gain + 5.0
    )


def test_wall_loss_reduces_rssi(ap):
    rssi_without_wall = calculate_rssi(
        ap=ap,
        distance_m=10.0,
        number_of_walls=0,
    )

    rssi_with_wall = calculate_rssi(
        ap=ap,
        distance_m=10.0,
        wall_material="concrete",
        number_of_walls=1,
    )

    assert rssi_with_wall == pytest.approx(
        rssi_without_wall - 5.0
    )


def test_obstacle_loss_reduces_rssi(ap):
    rssi_without_obstacle = calculate_rssi(
        ap=ap,
        distance_m=10.0,
        obstacle_loss_db=0.0,
    )

    rssi_with_obstacle = calculate_rssi(
        ap=ap,
        distance_m=10.0,
        obstacle_loss_db=4.0,
    )

    assert rssi_with_obstacle == pytest.approx(
        rssi_without_obstacle - 4.0
    )


def test_receiver_gain_validation(ap):
    with pytest.raises(ValueError):
        calculate_rssi(
            ap=ap,
            distance_m=10.0,
            receiver_gain_dbi=-21.0,
        )


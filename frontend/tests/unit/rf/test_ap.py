import pytest

from geotwin.rf.ap import AccessPoint


def test_access_point_creation():
    ap = AccessPoint(
        position=(5.0, 7.5),
        frequency_hz=2.4e9,
        tx_power_dbm=20.0,
        antenna_gain_dbi=3.0,
    )

    assert ap.position == (5.0, 7.5)
    assert ap.frequency_hz == 2.4e9
    assert ap.tx_power_dbm == 20.0
    assert ap.antenna_gain_dbi == 3.0


def test_access_point_coordinates():
    ap = AccessPoint(
        position=(5.0, 7.5),
        frequency_hz=2.4e9,
        tx_power_dbm=20.0,
        antenna_gain_dbi=3.0,
    )

    assert ap.x == 5.0
    assert ap.y == 7.5


def test_invalid_frequency():
    with pytest.raises(ValueError):
        AccessPoint(
            position=(5.0, 7.5),
            frequency_hz=0,
            tx_power_dbm=20.0,
            antenna_gain_dbi=3.0,
        )


def test_invalid_position():
    with pytest.raises(ValueError):
        AccessPoint(
            position=(5.0,),
            frequency_hz=2.4e9,
            tx_power_dbm=20.0,
            antenna_gain_dbi=3.0,
        )


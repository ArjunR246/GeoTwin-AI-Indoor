import pytest

from geotwin.rf.snr import calculate_snr


def test_snr_calculation():
    snr = calculate_snr(
        rssi_dbm=-60.0,
        noise_floor_dbm=-95.0,
    )

    assert snr == pytest.approx(35.0)


def test_snr_with_different_noise():
    snr = calculate_snr(
        rssi_dbm=-60.0,
        noise_floor_dbm=-90.0,
    )

    assert snr == pytest.approx(30.0)


def test_better_rssi_produces_better_snr():
    weak_snr = calculate_snr(
        rssi_dbm=-70.0,
        noise_floor_dbm=-95.0,
    )

    strong_snr = calculate_snr(
        rssi_dbm=-50.0,
        noise_floor_dbm=-95.0,
    )

    assert strong_snr > weak_snr


def test_higher_noise_reduces_snr():
    low_noise_snr = calculate_snr(
        rssi_dbm=-60.0,
        noise_floor_dbm=-95.0,
    )

    high_noise_snr = calculate_snr(
        rssi_dbm=-60.0,
        noise_floor_dbm=-85.0,
    )

    assert high_noise_snr < low_noise_snr


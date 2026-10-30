import pytest

from geotwin.rf.interference import (
    calculate_channel_aware_interference_dbm,
    calculate_interference_dbm,
    calculate_sinr_db,
    calculate_weighted_interference_dbm,
    channel_overlap_factor,
    dbm_to_mw,
    mw_to_dbm,
    sum_power_dbm,
)


def test_dbm_to_mw():
    assert dbm_to_mw(0.0) == pytest.approx(1.0)


def test_mw_to_dbm():
    assert mw_to_dbm(1.0) == pytest.approx(0.0)


def test_dbm_conversion_round_trip():
    power_dbm = -60.0

    power_mw = dbm_to_mw(power_dbm)
    result = mw_to_dbm(power_mw)

    assert result == pytest.approx(power_dbm)


def test_single_power_sum():
    result = sum_power_dbm([-60.0])

    assert result == pytest.approx(-60.0)


def test_multiple_power_sum():
    result = sum_power_dbm([-60.0, -60.0])

    # Two equal powers produce approximately 3.01 dB more power.
    assert result == pytest.approx(-56.9897, abs=0.001)


def test_interference_calculation():
    result = calculate_interference_dbm(
        [-60.0, -70.0]
    )

    assert result > -60.0
    assert result < -59.0


def test_sinr_without_interference():
    result = calculate_sinr_db(
        signal_power_dbm=-60.0,
        interference_powers_dbm=[],
        noise_floor_dbm=-95.0,
    )

    assert result == pytest.approx(35.0)


def test_sinr_with_interference():
    without_interference = calculate_sinr_db(
        signal_power_dbm=-60.0,
        interference_powers_dbm=[],
        noise_floor_dbm=-95.0,
    )

    with_interference = calculate_sinr_db(
        signal_power_dbm=-60.0,
        interference_powers_dbm=[-70.0],
        noise_floor_dbm=-95.0,
    )

    assert with_interference < without_interference


def test_stronger_interference_reduces_sinr():
    weak_interference = calculate_sinr_db(
        signal_power_dbm=-60.0,
        interference_powers_dbm=[-80.0],
        noise_floor_dbm=-95.0,
    )

    strong_interference = calculate_sinr_db(
        signal_power_dbm=-60.0,
        interference_powers_dbm=[-65.0],
        noise_floor_dbm=-95.0,
    )

    assert strong_interference < weak_interference


def test_same_channel_overlap():
    result = channel_overlap_factor(
        desired_channel=1,
        interfering_channel=1,
    )

    assert result == pytest.approx(1.0)


def test_adjacent_channel_overlap():
    result = channel_overlap_factor(
        desired_channel=1,
        interfering_channel=2,
    )

    assert result == pytest.approx(0.5)


def test_non_overlapping_channel():
    result = channel_overlap_factor(
        desired_channel=1,
        interfering_channel=6,
    )

    assert result == pytest.approx(0.0)


def test_weighted_interference_same_channel():
    result = calculate_weighted_interference_dbm(
        interfering_powers_dbm=[-60.0],
        overlap_factors=[1.0],
    )

    assert result == pytest.approx(-60.0)


def test_weighted_interference_zero_overlap():
    result = calculate_weighted_interference_dbm(
        interfering_powers_dbm=[-60.0],
        overlap_factors=[0.0],
    )

    assert result == float("-inf")


def test_channel_aware_interference():
    result = calculate_channel_aware_interference_dbm(
        interfering_powers_dbm=[-60.0, -60.0],
        desired_channel=1,
        interfering_channels=[1, 6],
    )

    # Only the same-channel AP contributes.
    assert result == pytest.approx(-60.0)


def test_channel_aware_interference_multiple_same_channel():
    result = calculate_channel_aware_interference_dbm(
        interfering_powers_dbm=[-60.0, -60.0],
        desired_channel=1,
        interfering_channels=[1, 1],
    )

    assert result == pytest.approx(-56.9897, abs=0.001)


def test_mismatched_power_and_channel_lists():
    with pytest.raises(ValueError):
        calculate_channel_aware_interference_dbm(
            interfering_powers_dbm=[-60.0, -70.0],
            desired_channel=1,
            interfering_channels=[1],
        )
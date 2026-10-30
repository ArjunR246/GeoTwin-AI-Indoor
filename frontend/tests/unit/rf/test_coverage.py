import pytest

from geotwin.rf.coverage import (
    calculate_coverage,
    evaluate_grid,
    evaluate_receiver,
)
from geotwin.rf.ap import AccessPoint

def test_evaluate_grid():
    aps = [
        AccessPoint(
            position=(5.0, 5.0),
            frequency_hz=2.4e9,
            tx_power_dbm=20.0,
            antenna_gain_dbi=3.0,
            channel=1,
        )
    ]

    positions = [
        (1.0, 1.0),
        (2.0, 2.0),
        (3.0, 3.0),
        (4.0, 4.0),
    ]

    result = evaluate_grid(
        receiver_positions=positions,
        access_points=aps,
    )

    assert result.total_points == 4
    assert len(result.receiver_results) == 4
    assert result.coverage_percent >= 0.0
    assert result.coverage_percent <= 100.0


def test_grid_metrics_are_calculated():
    aps = [
        AccessPoint(
            position=(5.0, 5.0),
            frequency_hz=2.4e9,
            tx_power_dbm=20.0,
            antenna_gain_dbi=3.0,
            channel=1,
        )
    ]

    positions = [
        (1.0, 1.0),
        (5.0, 5.0),
        (10.0, 10.0),
    ]

    result = evaluate_grid(
        receiver_positions=positions,
        access_points=aps,
    )

    assert result.average_rssi_dbm > -100.0
    assert result.minimum_rssi_dbm <= result.average_rssi_dbm

    assert result.average_sinr_db > 0.0
    assert result.minimum_sinr_db <= result.average_sinr_db


def test_grid_coverage_count():
    aps = [
        AccessPoint(
            position=(5.0, 5.0),
            frequency_hz=2.4e9,
            tx_power_dbm=20.0,
            antenna_gain_dbi=3.0,
            channel=1,
        )
    ]

    positions = [
        (5.0, 5.0),
        (5.5, 5.0),
        (6.0, 5.0),
    ]

    result = evaluate_grid(
        receiver_positions=positions,
        access_points=aps,
        minimum_rssi_dbm=-67.0,
    )

    assert result.covered_points <= result.total_points


def test_grid_without_receivers():
    aps = [
        AccessPoint(
            position=(5.0, 5.0),
            frequency_hz=2.4e9,
            tx_power_dbm=20.0,
            antenna_gain_dbi=3.0,
            channel=1,
        )
    ]

    with pytest.raises(ValueError):
        evaluate_grid(
            receiver_positions=[],
            access_points=aps,
        )


def test_grid_without_access_points():
    positions = [
        (1.0, 1.0),
        (2.0, 2.0),
    ]

    with pytest.raises(ValueError):
        evaluate_grid(
            receiver_positions=positions,
            access_points=[],
        )

def test_multi_ap_receiver_evaluation():
    aps = [
        AccessPoint(
            position=(0.0, 0.0),
            frequency_hz=2.4e9,
            tx_power_dbm=20.0,
            antenna_gain_dbi=3.0,
            channel=1,
        ),
        AccessPoint(
            position=(15.0, 0.0),
            frequency_hz=2.4e9,
            tx_power_dbm=20.0,
            antenna_gain_dbi=3.0,
            channel=6,
        ),
    ]

    result = evaluate_receiver(
        receiver_position=(3.0, 0.0),
        access_points=aps,
    )

    assert result.serving_ap_index == 0
    assert result.serving_rssi_dbm > -100.0
    assert result.sinr_db > 0.0


def test_strongest_ap_is_selected():
    aps = [
        AccessPoint(
            position=(2.0, 0.0),
            frequency_hz=2.4e9,
            tx_power_dbm=20.0,
            antenna_gain_dbi=3.0,
            channel=1,
        ),
        AccessPoint(
            position=(15.0, 0.0),
            frequency_hz=2.4e9,
            tx_power_dbm=20.0,
            antenna_gain_dbi=3.0,
            channel=1,
        ),
    ]

    result = evaluate_receiver(
        receiver_position=(3.0, 0.0),
        access_points=aps,
    )

    assert result.serving_ap_index == 0


def test_same_channel_creates_more_interference():
    aps_same_channel = [
        AccessPoint(
            position=(0.0, 0.0),
            frequency_hz=2.4e9,
            tx_power_dbm=20.0,
            antenna_gain_dbi=3.0,
            channel=1,
        ),
        AccessPoint(
            position=(8.0, 0.0),
            frequency_hz=2.4e9,
            tx_power_dbm=20.0,
            antenna_gain_dbi=3.0,
            channel=1,
        ),
    ]

    result = evaluate_receiver(
        receiver_position=(3.0, 0.0),
        access_points=aps_same_channel,
    )

    assert result.interference_dbm > float("-inf")


def test_non_overlapping_channel_reduces_interference():
    aps = [
        AccessPoint(
            position=(0.0, 0.0),
            frequency_hz=2.4e9,
            tx_power_dbm=20.0,
            antenna_gain_dbi=3.0,
            channel=1,
        ),
        AccessPoint(
            position=(8.0, 0.0),
            frequency_hz=2.4e9,
            tx_power_dbm=20.0,
            antenna_gain_dbi=3.0,
            channel=6,
        ),
    ]

    result = evaluate_receiver(
        receiver_position=(3.0, 0.0),
        access_points=aps,
    )

    assert result.interference_dbm == float("-inf")


def test_receiver_requires_ap():
    with pytest.raises(ValueError):
        evaluate_receiver(
            receiver_position=(5.0, 5.0),
            access_points=[],
        )


def test_full_coverage():
    result = calculate_coverage(
        rssi_values=[-50, -55, -60, -65],
        minimum_rssi_dbm=-67,
    )

    assert result.total_points == 4
    assert result.covered_points == 4
    assert result.coverage_percent == pytest.approx(100.0)
    assert result.dead_zone_percent == pytest.approx(0.0)


def test_partial_coverage():
    result = calculate_coverage(
        rssi_values=[-50, -60, -70, -80],
        minimum_rssi_dbm=-67,
    )

    assert result.total_points == 4
    assert result.covered_points == 2
    assert result.coverage_percent == pytest.approx(50.0)
    assert result.dead_zone_percent == pytest.approx(50.0)


def test_no_coverage():
    result = calculate_coverage(
        rssi_values=[-70, -75, -80],
        minimum_rssi_dbm=-67,
    )

    assert result.covered_points == 0
    assert result.coverage_percent == pytest.approx(0.0)
    assert result.dead_zone_percent == pytest.approx(100.0)


def test_snr_threshold():
    result = calculate_coverage(
        rssi_values=[-60, -60, -60],
        minimum_rssi_dbm=-67,
        snr_values=[25, 20, 15],
        minimum_snr_db=20,
    )

    assert result.covered_points == 2
    assert result.coverage_percent == pytest.approx(
        66.6666666667
    )


def test_rssi_and_snr_both_required():
    result = calculate_coverage(
        rssi_values=[-60, -60, -70],
        minimum_rssi_dbm=-67,
        snr_values=[25, 15, 25],
        minimum_snr_db=20,
    )

    assert result.covered_points == 1


def test_empty_measurements():
    with pytest.raises(ValueError):
        calculate_coverage(
            rssi_values=[],
            minimum_rssi_dbm=-67,
        )


def test_mismatched_rssi_snr():
    with pytest.raises(ValueError):
        calculate_coverage(
            rssi_values=[-60, -65],
            minimum_rssi_dbm=-67,
            snr_values=[20],
            minimum_snr_db=20,
        )
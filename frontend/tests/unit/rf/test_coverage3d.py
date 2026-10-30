
import pytest

from geotwin.rf.ap import AccessPoint3D
from geotwin.rf.coverage import evaluate_receiver_3d


from geotwin.floorplan.geometry3d import Wall3D



@pytest.fixture
def ap():
    return AccessPoint3D(
        position=(0.0, 0.0, 2.5),
        frequency_hz=2.4e9,
        tx_power_dbm=20.0,
        antenna_gain_dbi=2.0,
        channel=1,
    )


def test_evaluate_receiver_3d_single_ap(ap):
    result = evaluate_receiver_3d(
        receiver_position=(3.0, 4.0, 2.5),
        access_points=[ap],
    )

    assert result.position == (3.0, 4.0, 2.5)
    assert result.serving_ap_index == 0
    assert result.serving_rssi_dbm < 22.0
    assert result.interference_dbm == float("-inf")
    assert isinstance(result.covered, bool)


def test_vertical_separation_affects_rssi(ap):
    same_height = evaluate_receiver_3d(
        receiver_position=(3.0, 4.0, 2.5),
        access_points=[ap],
    )
    different_height = evaluate_receiver_3d(
        receiver_position=(3.0, 4.0, 5.5),
        access_points=[ap],
    )

    assert different_height.serving_rssi_dbm < same_height.serving_rssi_dbm


def test_strongest_ap_is_selected(ap):
    farther_ap = AccessPoint3D(
        position=(20.0, 20.0, 2.5),
        frequency_hz=2.4e9,
        tx_power_dbm=20.0,
        antenna_gain_dbi=2.0,
        channel=6,
    )

    result = evaluate_receiver_3d(
        receiver_position=(1.0, 1.0, 2.5),
        access_points=[farther_ap, ap],
    )

    assert result.serving_ap_index == 1


def test_receiver_requires_three_coordinates(ap):
    with pytest.raises(ValueError):
        evaluate_receiver_3d(
            receiver_position=(1.0, 2.0),
            access_points=[ap],
        )


def test_receiver_rejects_non_finite_coordinates(ap):
    with pytest.raises(ValueError):
        evaluate_receiver_3d(
            receiver_position=(1.0, float("nan"), 2.0),
            access_points=[ap],
        )


def test_receiver_requires_access_points():
    with pytest.raises(ValueError):
        evaluate_receiver_3d(
            receiver_position=(1.0, 2.0, 3.0),
            access_points=[],
        )




def test_3d_wall_attenuation_reduces_serving_rssi():
    ap = AccessPoint3D(
        position=(0.0, 5.0, 2.0),
        frequency_hz=2.4e9,
        tx_power_dbm=20.0,
        antenna_gain_dbi=0.0,
        antenna_type="omnidirectional",
        channel=1,
        channel_width_mhz=20,
    )

    wall = Wall3D(
        id="wall_1",
        start=(5.0, 0.0),
        end=(5.0, 10.0),
        height_m=2.5,
    )

    receiver = (10.0, 5.0, 1.0)

    baseline = evaluate_receiver_3d(
        receiver_position=receiver,
        access_points=[ap],
    )

    obstructed = evaluate_receiver_3d(
        receiver_position=receiver,
        access_points=[ap],
        floor_plan_walls_3d=[wall],
        floor_plan_wall_loss_db=6.0,
    )

    assert obstructed.serving_rssi_dbm == pytest.approx(
        baseline.serving_rssi_dbm - 6.0
    )


def test_3d_wall_above_receiver_path_does_not_reduce_rssi():
    ap = AccessPoint3D(
        position=(0.0, 5.0, 3.0),
        frequency_hz=2.4e9,
        tx_power_dbm=20.0,
        antenna_gain_dbi=0.0,
        antenna_type="omnidirectional",
        channel=1,
        channel_width_mhz=20,
    )

    wall = Wall3D(
        id="short_wall",
        start=(5.0, 0.0),
        end=(5.0, 10.0),
        height_m=1.0,
    )

    receiver = (10.0, 5.0, 3.0)

    baseline = evaluate_receiver_3d(
        receiver_position=receiver,
        access_points=[ap],
    )

    unobstructed = evaluate_receiver_3d(
        receiver_position=receiver,
        access_points=[ap],
        floor_plan_walls_3d=[wall],
        floor_plan_wall_loss_db=6.0,
    )

    assert unobstructed.serving_rssi_dbm == pytest.approx(
        baseline.serving_rssi_dbm
    )


def test_3d_wall_model_rejects_uniform_wall_count():
    ap = AccessPoint3D(
        position=(0.0, 5.0, 2.0),
        frequency_hz=2.4e9,
        tx_power_dbm=20.0,
        antenna_gain_dbi=0.0,
        antenna_type="omnidirectional",
        channel=1,
        channel_width_mhz=20,
    )

    wall = Wall3D(
        id="wall_1",
        start=(5.0, 0.0),
        end=(5.0, 10.0),
        height_m=2.5,
    )

    with pytest.raises(ValueError, match="double-counting"):
        evaluate_receiver_3d(
            receiver_position=(10.0, 5.0, 1.0),
            access_points=[ap],
            number_of_walls=1,
            floor_plan_walls_3d=[wall],
            floor_plan_wall_loss_db=6.0,
        )
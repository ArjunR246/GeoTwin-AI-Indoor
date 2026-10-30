import pytest

from geotwin.floorplan.walls import WallSegment
from geotwin.rf.ap import AccessPoint
from geotwin.rf.coverage import (
    evaluate_grid,
    evaluate_receiver,
)


def make_wall(y: int) -> WallSegment:
    return WallSegment(
        x1=0,
        y1=y,
        x2=100,
        y2=y,
        orientation="horizontal",
        length=100.0,
    )


def make_ap(x: float, y: float) -> AccessPoint:
    return AccessPoint(
        position=(x, y),
        frequency_hz=2.4e9,
        tx_power_dbm=20.0,
        antenna_gain_dbi=2.0,
        channel=1,
    )


def test_floor_plan_attenuation_reduces_rssi():
    ap = make_ap(50, 10)

    without_walls = evaluate_receiver(
        receiver_position=(50, 90),
        access_points=[ap],
    )

    with_wall = evaluate_receiver(
        receiver_position=(50, 90),
        access_points=[ap],
        floor_plan_walls=[make_wall(50)],
        floor_plan_wall_loss_db=5.0,
    )

    assert with_wall.serving_rssi_dbm < (
        without_walls.serving_rssi_dbm
    )


def test_no_floor_plan_preserves_existing_rssi():
    ap = make_ap(50, 10)

    baseline = evaluate_receiver(
        receiver_position=(50, 90),
        access_points=[ap],
    )

    explicit_empty = evaluate_receiver(
        receiver_position=(50, 90),
        access_points=[ap],
        floor_plan_walls=[],
        floor_plan_wall_loss_db=5.0,
    )

    assert (
        explicit_empty.serving_rssi_dbm
        == baseline.serving_rssi_dbm
    )


def test_two_walls_apply_twice():
    ap = make_ap(50, 10)

    baseline = evaluate_receiver(
        receiver_position=(50, 100),
        access_points=[ap],
    )

    attenuated = evaluate_receiver(
        receiver_position=(50, 100),
        access_points=[ap],
        floor_plan_walls=[
            make_wall(30),
            make_wall(60),
        ],
        floor_plan_wall_loss_db=5.0,
    )

    assert (
        attenuated.serving_rssi_dbm
        == pytest.approx(
            baseline.serving_rssi_dbm - 10.0
        )
    )

def test_floor_plan_can_change_serving_ap():
    ap_1 = make_ap(50, 10)
    ap_2 = make_ap(50, 35)

    wall = make_wall(25)

    result = evaluate_receiver(
        receiver_position=(50, 40),
        access_points=[ap_1, ap_2],
        floor_plan_walls=[wall],
        floor_plan_wall_loss_db=20.0,
    )

    assert result.serving_ap_index == 1


def test_negative_floor_plan_wall_loss_is_rejected():
    ap = make_ap(50, 10)

    with pytest.raises(ValueError):
        evaluate_receiver(
            receiver_position=(50, 90),
            access_points=[ap],
            floor_plan_walls=[make_wall(50)],
            floor_plan_wall_loss_db=-1.0,
        )


def test_grid_accepts_floor_plan_walls():
    ap = make_ap(50, 10)

    result = evaluate_grid(
        receiver_positions=[
            (50, 20),
            (50, 40),
            (50, 80),
        ],
        access_points=[ap],
        floor_plan_walls=[make_wall(50)],
        floor_plan_wall_loss_db=5.0,
    )

    assert result.total_points == 3
    assert len(result.receiver_results) == 3


def test_grid_without_floor_plan_preserves_operation():
    ap = make_ap(50, 10)

    result = evaluate_grid(
        receiver_positions=[
            (20, 20),
            (50, 50),
            (80, 80),
        ],
        access_points=[ap],
    )

    assert result.total_points == 3
    assert result.coverage_percent >= 0.0
    assert result.coverage_percent <= 100.0


def test_grid_rejects_negative_floor_plan_wall_loss():
    ap = make_ap(50, 10)

    with pytest.raises(ValueError):
        evaluate_grid(
            receiver_positions=[(50, 90)],
            access_points=[ap],
            floor_plan_walls=[make_wall(50)],
            floor_plan_wall_loss_db=-1.0,
        )
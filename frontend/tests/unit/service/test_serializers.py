from geotwin.models.geotwin_result import (
    APResult,
    GeoTwinResult,
    ParetoSolution,
    RFResult,
)
from geotwin.service.serializers import (
    serialize_ap_result,
    serialize_geotwin_result,
    serialize_pareto_solution,
    serialize_rf_result,
)


def make_result() -> GeoTwinResult:
    ap1 = APResult(
        position=(5.0, 3.0),
        channel=1,
    )

    ap2 = APResult(
        position=(5.5, 12.0),
        channel=11,
    )

    ap3 = APResult(
        position=(16.5, 8.0),
        channel=6,
    )

    rf = RFResult(
        coverage_percent=87.49,
        dead_zone_percent=12.51,
        average_rssi_dbm=-60.41,
        minimum_rssi_dbm=-71.29,
        average_sinr_db=34.59,
        minimum_sinr_db=23.71,
    )

    pareto = ParetoSolution(
        ap_count=3,
        coverage_percent=87.49,
        average_rssi_dbm=-60.41,
        minimum_rssi_dbm=-71.29,
        average_sinr_db=34.59,
        minimum_sinr_db=23.71,
        access_points=(ap1, ap2, ap3),
    )

    return GeoTwinResult(
        building_width_m=20.0,
        building_length_m=15.0,
        grid_resolution_m=0.5,
        floors=1,
        total_grid_points=1271,
        access_points=(ap1, ap2, ap3),
        ap_count=3,
        rf=rf,
        pareto_solutions=(pareto,),
        generations_completed=50,
    )


def test_serialize_ap_result():
    ap = APResult(
        position=(5.0, 3.0),
        channel=1,
    )

    result = serialize_ap_result(ap)

    assert result == {
        "position": {
            "x_m": 5.0,
            "y_m": 3.0,
        },
        "channel": 1,
    }


def test_serialize_rf_result():
    rf = RFResult(
        coverage_percent=87.49,
        dead_zone_percent=12.51,
        average_rssi_dbm=-60.41,
        minimum_rssi_dbm=-71.29,
        average_sinr_db=34.59,
        minimum_sinr_db=23.71,
    )

    result = serialize_rf_result(rf)

    assert result["coverage_percent"] == 87.49
    assert result["dead_zone_percent"] == 12.51
    assert result["average_rssi_dbm"] == -60.41
    assert result["minimum_rssi_dbm"] == -71.29
    assert result["average_sinr_db"] == 34.59
    assert result["minimum_sinr_db"] == 23.71


def test_serialize_pareto_solution():
    result = make_result()

    serialized = serialize_pareto_solution(
        result.pareto_solutions[0]
    )

    assert serialized["ap_count"] == 3
    assert serialized["coverage_percent"] == 87.49
    assert serialized["average_sinr_db"] == 34.59

    assert len(
        serialized["access_points"]
    ) == 3

    assert serialized["access_points"][0]["channel"] == 1


def test_serialize_geotwin_result_scenario():
    result = make_result()

    serialized = serialize_geotwin_result(result)

    assert serialized["scenario"] == {
        "building_width_m": 20.0,
        "building_length_m": 15.0,
        "grid_resolution_m": 0.5,
        "floors": 1,
        "total_grid_points": 1271,
    }


def test_serialize_geotwin_result_best_configuration():
    result = make_result()

    serialized = serialize_geotwin_result(result)

    best = serialized["best_configuration"]

    assert best["ap_count"] == 3
    assert len(best["access_points"]) == 3

    assert best["access_points"][0] == {
        "position": {
            "x_m": 5.0,
            "y_m": 3.0,
        },
        "channel": 1,
    }


def test_serialize_geotwin_result_rf():
    result = make_result()

    serialized = serialize_geotwin_result(result)

    assert serialized["rf"]["coverage_percent"] == 87.49
    assert serialized["rf"]["average_rssi_dbm"] == -60.41
    assert serialized["rf"]["average_sinr_db"] == 34.59


def test_serialize_geotwin_result_optimization():
    result = make_result()

    serialized = serialize_geotwin_result(result)

    assert serialized["optimization"] == {
        "generations_completed": 50,
    }


def test_serialize_geotwin_result_pareto_front():
    result = make_result()

    serialized = serialize_geotwin_result(result)

    pareto = serialized["pareto_solutions"]

    assert len(pareto) == 1
    assert pareto[0]["ap_count"] == 3
    assert pareto[0]["coverage_percent"] == 87.49
    assert len(pareto[0]["access_points"]) == 3


def test_serialized_result_has_public_sections_only():
    result = make_result()

    serialized = serialize_geotwin_result(result)

    assert set(serialized.keys()) == {
        "scenario",
        "best_configuration",
        "rf",
        "pareto_solutions",
        "optimization",
    }
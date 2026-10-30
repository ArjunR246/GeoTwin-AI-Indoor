"""
GeoTwin AI Indoor - API Tests

Tests the FastAPI HTTP boundary without running the
full RF/NSGA-II optimization pipeline.
"""

from fastapi.testclient import TestClient

from geotwin.api.app import app
from geotwin.models.geotwin_result import (
    APResult,
    GeoTwinResult,
    ParetoSolution,
    RFResult,
)


client = TestClient(app)


# ======================================================================
# TEST FIXTURE
# ======================================================================

def make_test_result() -> GeoTwinResult:
    """Create a deterministic GeoTwinResult for API tests."""

    ap1 = APResult(
        position=(5.0, 3.0),
        channel=1,
    )

    ap2 = APResult(
        position=(15.0, 10.0),
        channel=6,
    )

    rf = RFResult(
        coverage_percent=90.0,
        dead_zone_percent=10.0,
        average_rssi_dbm=-60.0,
        minimum_rssi_dbm=-72.0,
        average_sinr_db=30.0,
        minimum_sinr_db=20.0,
    )

    pareto = ParetoSolution(
        ap_count=2,
        coverage_percent=90.0,
        average_rssi_dbm=-60.0,
        minimum_rssi_dbm=-72.0,
        average_sinr_db=30.0,
        minimum_sinr_db=20.0,
        access_points=(ap1, ap2),
    )

    return GeoTwinResult(
        building_width_m=20.0,
        building_length_m=15.0,
        grid_resolution_m=0.5,
        floors=1,
        total_grid_points=1271,
        access_points=(ap1, ap2),
        ap_count=2,
        rf=rf,
        pareto_solutions=(pareto,),
        generations_completed=10,
    )


# ======================================================================
# HEALTH ENDPOINT
# ======================================================================

def test_health_endpoint():

    response = client.get("/health")

    assert response.status_code == 200

    assert response.json() == {
        "status": "ok",
        "service": "GeoTwin AI Indoor",
    }


# ======================================================================
# ROOT ENDPOINT
# ======================================================================

def test_root_endpoint():

    response = client.get("/")

    assert response.status_code == 200

    data = response.json()

    assert data["name"] == "GeoTwin AI Indoor API"
    assert data["version"] == "0.1.0"
    assert data["status"] == "running"


# ======================================================================
# OPTIMIZE ENDPOINT
# ======================================================================

def test_optimize_endpoint(monkeypatch):

    test_result = make_test_result()

    def fake_run():
        return test_result

    monkeypatch.setattr(
        "geotwin.api.app.service.run",
        fake_run,
    )

    response = client.post("/optimize")

    assert response.status_code == 200

    data = response.json()

    assert set(data.keys()) == {
        "scenario",
        "best_configuration",
        "rf",
        "pareto_solutions",
        "optimization",
    }


def test_optimize_endpoint_scenario(monkeypatch):

    test_result = make_test_result()

    monkeypatch.setattr(
        "geotwin.api.app.service.run",
        lambda: test_result,
    )

    response = client.post("/optimize")

    assert response.status_code == 200

    data = response.json()

    assert data["scenario"] == {
        "building_width_m": 20.0,
        "building_length_m": 15.0,
        "grid_resolution_m": 0.5,
        "floors": 1,
        "total_grid_points": 1271,
    }


def test_optimize_endpoint_rf_metrics(monkeypatch):

    test_result = make_test_result()

    monkeypatch.setattr(
        "geotwin.api.app.service.run",
        lambda: test_result,
    )

    response = client.post("/optimize")

    assert response.status_code == 200

    data = response.json()

    assert data["rf"]["coverage_percent"] == 90.0
    assert data["rf"]["dead_zone_percent"] == 10.0
    assert data["rf"]["average_rssi_dbm"] == -60.0
    assert data["rf"]["minimum_rssi_dbm"] == -72.0
    assert data["rf"]["average_sinr_db"] == 30.0
    assert data["rf"]["minimum_sinr_db"] == 20.0


def test_optimize_endpoint_best_configuration(monkeypatch):

    test_result = make_test_result()

    monkeypatch.setattr(
        "geotwin.api.app.service.run",
        lambda: test_result,
    )

    response = client.post("/optimize")

    assert response.status_code == 200

    data = response.json()

    best = data["best_configuration"]

    assert best["ap_count"] == 2

    assert len(best["access_points"]) == 2

    assert best["access_points"][0] == {
        "position": {
            "x_m": 5.0,
            "y_m": 3.0,
        },
        "channel": 1,
    }


def test_optimize_endpoint_pareto_solutions(monkeypatch):

    test_result = make_test_result()

    monkeypatch.setattr(
        "geotwin.api.app.service.run",
        lambda: test_result,
    )

    response = client.post("/optimize")

    assert response.status_code == 200

    data = response.json()

    pareto = data["pareto_solutions"]

    assert len(pareto) == 1
    assert pareto[0]["ap_count"] == 2
    assert pareto[0]["coverage_percent"] == 90.0
    assert pareto[0]["average_rssi_dbm"] == -60.0
    assert pareto[0]["minimum_rssi_dbm"] == -72.0
    assert pareto[0]["average_sinr_db"] == 30.0
    assert pareto[0]["minimum_sinr_db"] == 20.0
    assert len(pareto[0]["access_points"]) == 2


def test_optimize_endpoint_optimization(monkeypatch):

    test_result = make_test_result()

    monkeypatch.setattr(
        "geotwin.api.app.service.run",
        lambda: test_result,
    )

    response = client.post("/optimize")

    assert response.status_code == 200

    data = response.json()

    assert data["optimization"] == {
        "generations_completed": 10,
    }


# ======================================================================
# RESULT SUMMARY ENDPOINT
# ======================================================================

def test_result_summary_endpoint(monkeypatch):

    test_result = make_test_result()

    monkeypatch.setattr(
        "geotwin.api.app.service.run",
        lambda: test_result,
    )

    response = client.get("/result/summary")

    assert response.status_code == 200

    data = response.json()

    assert set(data.keys()) == {
        "building",
        "best_solution",
        "rf",
        "optimization",
    }


def test_result_summary_building(monkeypatch):

    test_result = make_test_result()

    monkeypatch.setattr(
        "geotwin.api.app.service.run",
        lambda: test_result,
    )

    response = client.get("/result/summary")

    assert response.status_code == 200

    data = response.json()

    assert data["building"] == {
        "width_m": 20.0,
        "length_m": 15.0,
        "grid_resolution_m": 0.5,
        "floors": 1,
    }


def test_result_summary_best_solution(monkeypatch):

    test_result = make_test_result()

    monkeypatch.setattr(
        "geotwin.api.app.service.run",
        lambda: test_result,
    )

    response = client.get("/result/summary")

    assert response.status_code == 200

    data = response.json()

    best = data["best_solution"]

    assert best["ap_count"] == 2

    assert best["positions"] == [
        [5.0, 3.0],
        [15.0, 10.0],
    ]

    assert best["channels"] == [1, 6]


def test_result_summary_rf(monkeypatch):

    test_result = make_test_result()

    monkeypatch.setattr(
        "geotwin.api.app.service.run",
        lambda: test_result,
    )

    response = client.get("/result/summary")

    assert response.status_code == 200

    data = response.json()

    assert data["rf"]["coverage_percent"] == 90.0
    assert data["rf"]["dead_zone_percent"] == 10.0
    assert data["rf"]["average_rssi_dbm"] == -60.0
    assert data["rf"]["minimum_rssi_dbm"] == -72.0
    assert data["rf"]["average_sinr_db"] == 30.0
    assert data["rf"]["minimum_sinr_db"] == 20.0


def test_result_summary_optimization(monkeypatch):

    test_result = make_test_result()

    monkeypatch.setattr(
        "geotwin.api.app.service.run",
        lambda: test_result,
    )

    response = client.get("/result/summary")

    assert response.status_code == 200

    data = response.json()

    assert data["optimization"] == {
        "generations_completed": 10,
        "pareto_solutions": 1,
    }


# ======================================================================
# ERROR HANDLING
# ======================================================================

def test_optimize_returns_500_on_service_failure(monkeypatch):

    def failing_run():
        raise RuntimeError("test failure")

    monkeypatch.setattr(
        "geotwin.api.app.service.run",
        failing_run,
    )

    response = client.post("/optimize")

    assert response.status_code == 500

    assert response.json()["detail"] == "test failure"


def test_result_summary_returns_500_on_service_failure(monkeypatch):

    def failing_run():
        raise RuntimeError("test failure")

    monkeypatch.setattr(
        "geotwin.api.app.service.run",
        failing_run,
    )

    response = client.get("/result/summary")

    assert response.status_code == 500

    assert response.json()["detail"] == "test failure"


# ======================================================================
# HTTP METHOD VALIDATION
# ======================================================================

def test_optimize_requires_post():

    response = client.get("/optimize")

    assert response.status_code == 405


def test_result_summary_requires_get():

    response = client.post("/result/summary")

    assert response.status_code == 405


from geotwin.optimization.nsga2 import NSGA2Config
from geotwin.pipeline import GeoTwinResult
from geotwin.scenario.indoor import IndoorScenario
from geotwin.service.geotwin_service import GeoTwinService


def create_test_config() -> NSGA2Config:
    return NSGA2Config(
        population_size=10,
        generations=3,
        min_ap_count=1,
        max_ap_count=3,
        random_seed=42,
    )


def test_service_returns_geotwin_result():

    service = GeoTwinService()

    result = service.run(
        config=create_test_config()
    )

    assert isinstance(
        result,
        GeoTwinResult,
    )


def test_service_returns_best_solution():

    service = GeoTwinService()

    result = service.run(
        config=create_test_config()
    )

    assert result.best_solution is not None

    assert result.best_solution.ap_count >= 1

    assert len(
        result.best_solution.positions
    ) == len(
        result.best_solution.channels
    )


def test_service_returns_rf_evaluation():

    service = GeoTwinService()

    result = service.run(
        config=create_test_config()
    )

    evaluation = result.rf_evaluation

    assert evaluation.total_points > 0

    assert 0.0 <= evaluation.coverage_percent <= 100.0

    assert 0.0 <= evaluation.dead_zone_percent <= 100.0


def test_service_preserves_custom_scenario():

    service = GeoTwinService()

    scenario = IndoorScenario(
        width_m=10.0,
        length_m=10.0,
        grid_resolution_m=1.0,
        floors=1,
        wall_material="concrete",
        wall_thickness_m=0.2,
    )

    result = service.run(
        scenario=scenario,
        config=create_test_config(),
    )

    assert result.scenario == scenario

    assert (
        result.rf_evaluation.total_points
        == scenario.total_grid_points
    )


def test_service_is_reproducible():

    service = GeoTwinService()

    config = create_test_config()

    result_one = service.run(
        config=config
    )

    result_two = service.run(
        config=config
    )

    assert (
        result_one.best_solution.positions
        == result_two.best_solution.positions
    )

    assert (
        result_one.best_solution.channels
        == result_two.best_solution.channels
    )

    assert (
        result_one.best_solution.objectives
        == result_two.best_solution.objectives
    )


def test_service_channel_assignments_are_valid():

    service = GeoTwinService()

    config = create_test_config()

    result = service.run(
        config=config
    )

    best = result.best_solution

    assert all(
        channel in config.available_channels
        for channel in best.channels
    )

    assert len(best.positions) == len(best.channels)


def test_service_pareto_front_exists():

    service = GeoTwinService()

    result = service.run(
        config=create_test_config()
    )

    assert result.optimization.pareto_front

    assert (
        result.optimization.generations_completed
        == 3
    )
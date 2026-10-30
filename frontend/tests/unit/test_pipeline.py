from geotwin.optimization.nsga2 import NSGA2Config
from geotwin.pipeline import GeoTwinResult, run_geotwin
from geotwin.scenario.indoor import IndoorScenario


def create_test_scenario() -> IndoorScenario:
    return IndoorScenario(
        width_m=20.0,
        length_m=15.0,
        grid_resolution_m=0.5,
        floors=1,
        wall_material="concrete",
        wall_thickness_m=0.2,
    )


def create_test_config() -> NSGA2Config:
    return NSGA2Config(
        population_size=10,
        generations=3,
        min_ap_count=1,
        max_ap_count=3,
        random_seed=42,
    )


def test_geotwin_pipeline_returns_result_model():

    scenario = create_test_scenario()
    config = create_test_config()

    result = run_geotwin(
        scenario=scenario,
        config=config,
    )

    assert isinstance(result, GeoTwinResult)


def test_geotwin_pipeline_preserves_scenario():

    scenario = create_test_scenario()
    config = create_test_config()

    result = run_geotwin(
        scenario=scenario,
        config=config,
    )

    assert result.scenario == scenario


def test_geotwin_pipeline_returns_optimization_result():

    scenario = create_test_scenario()
    config = create_test_config()

    result = run_geotwin(
        scenario=scenario,
        config=config,
    )

    assert result.optimization.generations_completed == 3

    assert len(
        result.optimization.history
    ) == 3

    assert result.optimization.pareto_front


def test_geotwin_pipeline_best_solution_is_valid():

    scenario = create_test_scenario()
    config = create_test_config()

    result = run_geotwin(
        scenario=scenario,
        config=config,
    )

    best = result.best_solution

    assert best.ap_count >= 1
    assert best.ap_count <= 3

    assert len(best.positions) == best.ap_count
    assert len(best.channels) == best.ap_count

    assert len(best.positions) == len(
        set(best.positions)
    )

    assert all(
        position in scenario.valid_ap_positions
        for position in best.positions
    )

    assert all(
        channel in config.available_channels
        for channel in best.channels
    )


def test_geotwin_pipeline_rf_evaluation_matches_scenario():

    scenario = create_test_scenario()
    config = create_test_config()

    result = run_geotwin(
        scenario=scenario,
        config=config,
    )

    evaluation = result.rf_evaluation

    assert evaluation.total_points == (
        scenario.total_grid_points
    )

    assert evaluation.covered_points >= 0

    assert evaluation.covered_points <= (
        evaluation.total_points
    )

    assert 0.0 <= evaluation.coverage_percent <= 100.0

    assert 0.0 <= evaluation.dead_zone_percent <= 100.0

    assert (
        evaluation.coverage_percent
        + evaluation.dead_zone_percent
        == 100.0
    )

    assert evaluation.average_rssi_dbm <= 0.0

    assert evaluation.minimum_rssi_dbm <= (
        evaluation.average_rssi_dbm
    )

    assert evaluation.minimum_sinr_db <= (
        evaluation.average_sinr_db
    )


def test_geotwin_pipeline_best_solution_has_objectives():

    scenario = create_test_scenario()
    config = create_test_config()

    result = run_geotwin(
        scenario=scenario,
        config=config,
    )

    best = result.best_solution

    assert len(best.objectives) == 3

    coverage_objective = best.objectives[0]
    ap_count_objective = best.objectives[1]

    assert coverage_objective <= 0.0
    assert ap_count_objective == best.ap_count


def test_geotwin_pipeline_is_reproducible():

    scenario = create_test_scenario()
    config = create_test_config()

    result_one = run_geotwin(
        scenario=scenario,
        config=config,
    )

    result_two = run_geotwin(
        scenario=scenario,
        config=config,
    )

    best_one = result_one.best_solution
    best_two = result_two.best_solution

    assert best_one.positions == best_two.positions
    assert best_one.channels == best_two.channels
    assert best_one.objectives == best_two.objectives

    assert (
        result_one.rf_evaluation.coverage_percent
        == result_two.rf_evaluation.coverage_percent
    )

    assert (
        result_one.rf_evaluation.average_sinr_db
        == result_two.rf_evaluation.average_sinr_db
    )


def test_geotwin_pipeline_channel_position_pairing():

    scenario = create_test_scenario()
    config = create_test_config()

    result = run_geotwin(
        scenario=scenario,
        config=config,
    )

    best = result.best_solution

    configuration = best.ap_configuration

    assert len(configuration) == best.ap_count

    for position, channel in configuration:

        assert position in scenario.valid_ap_positions
        assert channel in config.available_channels


def test_geotwin_pipeline_rf_results_match_best_solution():

    scenario = create_test_scenario()
    config = create_test_config()

    result = run_geotwin(
        scenario=scenario,
        config=config,
    )

    best = result.best_solution

    # The first objective is negative coverage.
    expected_coverage = -best.objectives[0]

    actual_coverage = (
        result.rf_evaluation.coverage_percent
    )

    assert actual_coverage == expected_coverage
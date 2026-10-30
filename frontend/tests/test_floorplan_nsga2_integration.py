import pytest

from geotwin.floorplan.pipeline import process_floor_plan
from geotwin.floorplan.rf_adapter import floor_plan_to_rf_geometry
from geotwin.optimization.nsga2 import NSGA2Config, optimize
from geotwin.rf.ap import AccessPoint
from geotwin.rf.coverage import evaluate_grid


FLOOR_PLAN_PATH = (
    "tests/fixtures/floorplans/simple_floorplan.png"
)

PHYSICAL_WIDTH_M = 20.0
PHYSICAL_HEIGHT_M = 15.0


def load_floor_plan_geometry():
    """
    Process the fixture floor plan and convert it into
    physical RF geometry.
    """

    result = process_floor_plan(
        __import__("cv2").imread(
            FLOOR_PLAN_PATH
        ),
        physical_width_m=PHYSICAL_WIDTH_M,
        physical_height_m=PHYSICAL_HEIGHT_M,
    )

    assert result.coordinates is not None

    return result


# ============================================================
# FLOOR PLAN → RF GEOMETRY
# ============================================================


def test_floor_plan_produces_rf_geometry():
    """Floor-plan processing must produce usable RF geometry."""

    result = load_floor_plan_geometry()

    (
        receiver_positions,
        valid_ap_positions,
        blocked_positions,
        physical_walls,
    ) = floor_plan_to_rf_geometry(
        building=result.building,
        coordinates=result.coordinates,
        cell_size_m=0.5,
    )

    assert receiver_positions
    assert valid_ap_positions
    assert physical_walls

    assert len(receiver_positions) > 0
    assert len(valid_ap_positions) > 0
    assert len(blocked_positions) > 0


def test_floor_plan_rf_geometry_is_in_physical_coordinates():
    """
    RF geometry must be expressed in metres rather than pixels.
    """

    result = load_floor_plan_geometry()

    (
        receiver_positions,
        valid_ap_positions,
        blocked_positions,
        physical_walls,
    ) = floor_plan_to_rf_geometry(
        building=result.building,
        coordinates=result.coordinates,
        cell_size_m=0.5,
    )

    for position in receiver_positions:
        assert 0.0 <= position[0] <= PHYSICAL_WIDTH_M
        assert 0.0 <= position[1] <= PHYSICAL_HEIGHT_M

    for position in valid_ap_positions:
        assert 0.0 <= position[0] <= PHYSICAL_WIDTH_M
        assert 0.0 <= position[1] <= PHYSICAL_HEIGHT_M

    for position in blocked_positions:
        assert 0.0 <= position[0] <= PHYSICAL_WIDTH_M
        assert 0.0 <= position[1] <= PHYSICAL_HEIGHT_M


def test_floor_plan_walls_are_transformed_to_physical_space():
    """Detected walls must be transformed into physical coordinates."""

    result = load_floor_plan_geometry()

    (
        _receiver_positions,
        _valid_ap_positions,
        _blocked_positions,
        physical_walls,
    ) = floor_plan_to_rf_geometry(
        building=result.building,
        coordinates=result.coordinates,
        cell_size_m=0.5,
    )

    assert len(physical_walls) == result.wall_count

    for wall in physical_walls:
        assert 0.0 <= wall.x1 <= PHYSICAL_WIDTH_M
        assert 0.0 <= wall.x2 <= PHYSICAL_WIDTH_M
        assert 0.0 <= wall.y1 <= PHYSICAL_HEIGHT_M
        assert 0.0 <= wall.y2 <= PHYSICAL_HEIGHT_M


# ============================================================
# FLOOR PLAN → NSGA-II
# ============================================================


def test_nsga2_accepts_floor_plan_valid_positions():
    """
    NSGA-II must be able to use valid AP positions generated
    from the floor plan.
    """

    result = load_floor_plan_geometry()

    (
        receiver_positions,
        valid_ap_positions,
        blocked_positions,
        physical_walls,
    ) = floor_plan_to_rf_geometry(
        building=result.building,
        coordinates=result.coordinates,
        cell_size_m=0.5,
    )

    config = NSGA2Config(
        population_size=10,
        generations=2,
        min_ap_count=1,
        max_ap_count=2,
        minimum_sinr_db=None,
        random_seed=42,
    )

    optimization_result = optimize(
        receiver_positions=receiver_positions,
        valid_positions=valid_ap_positions,
        blocked_positions=blocked_positions,
        config=config,
    )

    assert optimization_result.pareto_front
    assert optimization_result.generations_completed == 2

    for individual in optimization_result.pareto_front:
        assert individual.positions
        assert len(individual.positions) == len(
            individual.channels
        )

        for position in individual.positions:
            assert position in valid_ap_positions

        for position in individual.positions:
            assert position not in blocked_positions


# ============================================================
# NSGA-II → CHANNEL-AWARE RF
# ============================================================


def test_optimized_solution_can_be_evaluated_by_rf_engine():
    """
    An NSGA-II solution generated from the floor plan must be
    compatible with the production RF evaluation engine.
    """

    result = load_floor_plan_geometry()

    (
        receiver_positions,
        valid_ap_positions,
        blocked_positions,
        physical_walls,
    ) = floor_plan_to_rf_geometry(
        building=result.building,
        coordinates=result.coordinates,
        cell_size_m=0.5,
    )

    config = NSGA2Config(
        population_size=10,
        generations=2,
        min_ap_count=1,
        max_ap_count=2,
        minimum_sinr_db=None,
        random_seed=42,
    )

    optimization_result = optimize(
        receiver_positions=receiver_positions,
        valid_positions=valid_ap_positions,
        blocked_positions=blocked_positions,
        config=config,
    )

    solution = optimization_result.pareto_front[0]

    access_points = [
        AccessPoint(
            position=position,
            frequency_hz=2.4e9,
            tx_power_dbm=20.0,
            antenna_gain_dbi=3.0,
            channel=channel,
        )
        for position, channel in zip(
            solution.positions,
            solution.channels,
        )
    ]

    evaluation = evaluate_grid(
        receiver_positions=receiver_positions,
        access_points=access_points,
        noise_floor_dbm=-95.0,
        minimum_rssi_dbm=-67.0,
        minimum_sinr_db=None,
        reference_distance_m=1.0,
        path_loss_exponent=3.0,
        wall_material="concrete",
        number_of_walls=0,
        obstacle_loss_db=0.0,
        same_channel_factor=1.0,
        adjacent_channel_factor=0.5,
        non_overlapping_factor=0.0,
        floor_plan_walls=physical_walls,
        floor_plan_wall_loss_db=6.0,
    )

    assert evaluation.total_points == len(
        receiver_positions
    )

    assert evaluation.total_points > 0

    assert 0.0 <= evaluation.coverage_percent <= 100.0

    assert evaluation.minimum_rssi_dbm <= (
        evaluation.average_rssi_dbm
    )

    assert evaluation.minimum_sinr_db <= (
        evaluation.average_sinr_db
    )

    assert len(evaluation.receiver_results) == (
        len(receiver_positions)
    )


# ============================================================
# BLOCKED POSITION PROTECTION
# ============================================================


def test_nsga2_never_places_ap_in_blocked_position():
    """
    AP locations produced by NSGA-II must never come from
    floor-plan blocked positions.
    """

    result = load_floor_plan_geometry()

    (
        receiver_positions,
        valid_ap_positions,
        blocked_positions,
        _physical_walls,
    ) = floor_plan_to_rf_geometry(
        building=result.building,
        coordinates=result.coordinates,
        cell_size_m=0.5,
    )

    config = NSGA2Config(
        population_size=12,
        generations=3,
        min_ap_count=1,
        max_ap_count=3,
        minimum_sinr_db=None,
        random_seed=123,
    )

    optimization_result = optimize(
        receiver_positions=receiver_positions,
        valid_positions=valid_ap_positions,
        blocked_positions=blocked_positions,
        config=config,
    )

    for individual in optimization_result.pareto_front:

        for position in individual.positions:

            assert position in valid_ap_positions

            assert position not in blocked_positions


# ============================================================
# CHANNEL VALIDATION
# ============================================================


def test_optimized_solution_contains_valid_channels():
    """Every optimized AP must have a valid Wi-Fi channel."""

    result = load_floor_plan_geometry()

    (
        receiver_positions,
        valid_ap_positions,
        blocked_positions,
        _physical_walls,
    ) = floor_plan_to_rf_geometry(
        building=result.building,
        coordinates=result.coordinates,
        cell_size_m=0.5,
    )

    config = NSGA2Config(
        population_size=10,
        generations=2,
        min_ap_count=1,
        max_ap_count=3,
        minimum_sinr_db=None,
        available_channels=(1, 6, 11),
        random_seed=42,
    )

    optimization_result = optimize(
        receiver_positions=receiver_positions,
        valid_positions=valid_ap_positions,
        blocked_positions=blocked_positions,
        config=config,
    )

    for individual in optimization_result.pareto_front:

        assert len(individual.positions) == len(
            individual.channels
        )

        assert all(
            channel in (1, 6, 11)
            for channel in individual.channels
        )


# ============================================================
# COMPLETE INTEGRATION
# ============================================================


def test_complete_floor_plan_to_rf_pipeline():
    """
    Full integration test:

        floor plan
        → building model
        → physical RF geometry
        → NSGA-II
        → channel-aware RF evaluation
    """

    result = load_floor_plan_geometry()

    assert result.wall_count > 0
    assert result.room_count > 0

    (
        receiver_positions,
        valid_ap_positions,
        blocked_positions,
        physical_walls,
    ) = floor_plan_to_rf_geometry(
        building=result.building,
        coordinates=result.coordinates,
        cell_size_m=0.5,
    )

    config = NSGA2Config(
        population_size=8,
        generations=2,
        min_ap_count=1,
        max_ap_count=2,
        minimum_sinr_db=None,
        available_channels=(1, 6, 11),
        random_seed=99,
    )

    optimization_result = optimize(
        receiver_positions=receiver_positions,
        valid_positions=valid_ap_positions,
        blocked_positions=blocked_positions,
        config=config,
    )

    assert optimization_result.pareto_front

    solution = optimization_result.pareto_front[0]

    access_points = [
        AccessPoint(
            position=position,
            frequency_hz=2.4e9,
            tx_power_dbm=20.0,
            antenna_gain_dbi=3.0,
            channel=channel,
        )
        for position, channel in zip(
            solution.positions,
            solution.channels,
        )
    ]

    evaluation = evaluate_grid(
        receiver_positions=receiver_positions,
        access_points=access_points,
        noise_floor_dbm=-95.0,
        minimum_rssi_dbm=-67.0,
        minimum_sinr_db=None,
        reference_distance_m=1.0,
        path_loss_exponent=3.0,
        wall_material="concrete",
        number_of_walls=0,
        obstacle_loss_db=0.0,
        same_channel_factor=1.0,
        adjacent_channel_factor=0.5,
        non_overlapping_factor=0.0,
        floor_plan_walls=physical_walls,
        floor_plan_wall_loss_db=6.0,
    )

    assert evaluation.total_points > 0
    assert evaluation.covered_points >= 0
    assert evaluation.covered_points <= evaluation.total_points

    assert 0.0 <= evaluation.coverage_percent <= 100.0

    assert len(evaluation.receiver_results) == (
        evaluation.total_points
    )
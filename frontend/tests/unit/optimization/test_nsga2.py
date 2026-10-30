import random

import pytest

from geotwin.floorplan.walls import WallSegment
from geotwin.optimization.nsga2 import (
    Individual,
    NSGA2Config,
    OptimizationResult,
    calculate_crowding_distance,
    create_offspring,
    crossover,
    dominates,
    evaluate_individual,
    fast_non_dominated_sort,
    initialize_population,
    mutate,
    optimize,
    tournament_select,
)


VALID_POSITIONS = [
    (0.0, 0.0),
    (0.5, 0.0),
    (1.0, 0.0),
    (0.0, 0.5),
    (0.5, 0.5),
    (1.0, 0.5),
]


def make_individual(
    positions,
    objectives,
):
    return Individual(
        positions=tuple(positions),
        objectives=tuple(objectives),
    )


def make_wall():
    return WallSegment(
        x1=1,
        y1=-2,
        x2=1,
        y2=2,
        orientation="vertical",
        length=4.0,
    )


# ============================================================
# End-to-End NSGA-II
# ============================================================


def test_initialize_population():
    config = NSGA2Config(
        population_size=10,
        min_ap_count=2,
        max_ap_count=3,
        random_seed=42,
    )

    rng = random.Random(42)

    population = initialize_population(
        valid_positions=VALID_POSITIONS,
        config=config,
        rng=rng,
    )

    assert len(population) == 10

    for individual in population:
        assert 2 <= individual.ap_count <= 3

        assert len(individual.positions) == len(
            set(individual.positions)
        )

        assert all(
            position in VALID_POSITIONS
            for position in individual.positions
        )


def test_invalid_nsga2_configuration():
    config = NSGA2Config(
        population_size=1,
    )

    with pytest.raises(ValueError):
        config.validate()


def test_evaluate_individual_end_to_end():
    config = NSGA2Config(
        min_ap_count=1,
        max_ap_count=2,
        minimum_sinr_db=None,
    )

    individual = Individual(
        positions=(
            (0.0, 0.0),
        )
    )

    receiver_positions = [
        (0.5, 0.0),
        (1.0, 0.0),
        (0.5, 0.5),
        (1.0, 0.5),
    ]

    evaluation = evaluate_individual(
        individual=individual,
        receiver_positions=receiver_positions,
        valid_positions=VALID_POSITIONS,
        blocked_positions=[],
        config=config,
    )

    assert evaluation.total_points == 4
    assert len(individual.objectives) == 3


def test_end_to_end_nsga2_optimization():
    config = NSGA2Config(
        population_size=10,
        generations=5,
        min_ap_count=1,
        max_ap_count=3,
        crossover_probability=0.9,
        mutation_probability=0.1,
        minimum_sinr_db=None,
        random_seed=42,
    )

    receiver_positions = [
        (0.0, 0.0),
        (0.5, 0.0),
        (1.0, 0.0),
        (0.0, 0.5),
        (0.5, 0.5),
        (1.0, 0.5),
    ]

    result = optimize(
        receiver_positions=receiver_positions,
        valid_positions=VALID_POSITIONS,
        blocked_positions=[],
        config=config,
    )

    assert isinstance(
        result,
        OptimizationResult,
    )

    assert result.generations_completed == 5

    assert len(result.pareto_front) > 0

    assert len(result.history) == 5

    for individual in result.pareto_front:
        assert len(individual.objectives) == 3


def test_nsga2_is_reproducible_with_seed():
    config = NSGA2Config(
        population_size=10,
        generations=3,
        min_ap_count=1,
        max_ap_count=3,
        minimum_sinr_db=None,
        random_seed=123,
    )

    receiver_positions = [
        (0.0, 0.0),
        (0.5, 0.0),
        (1.0, 0.0),
        (0.0, 0.5),
        (0.5, 0.5),
        (1.0, 0.5),
    ]

    result_one = optimize(
        receiver_positions=receiver_positions,
        valid_positions=VALID_POSITIONS,
        config=config,
    )

    result_two = optimize(
        receiver_positions=receiver_positions,
        valid_positions=VALID_POSITIONS,
        config=config,
    )

    objectives_one = [
        individual.objectives
        for individual in result_one.pareto_front
    ]

    objectives_two = [
        individual.objectives
        for individual in result_two.pareto_front
    ]

    assert objectives_one == objectives_two


# ============================================================
# Floor-plan Wall Integration
# ============================================================


def test_evaluate_individual_accepts_floor_plan_walls():
    config = NSGA2Config(
        min_ap_count=1,
        max_ap_count=1,
        minimum_sinr_db=None,
    )

    individual_without_wall = Individual(
        positions=(
            (0.0, 0.0),
        )
    )

    individual_with_wall = Individual(
        positions=(
            (0.0, 0.0),
        )
    )

    receiver_positions = [
        (2.0, 0.0),
        (4.0, 0.0),
    ]

    evaluation_without_wall = evaluate_individual(
        individual=individual_without_wall,
        receiver_positions=receiver_positions,
        valid_positions=[
            (0.0, 0.0),
        ],
        blocked_positions=[],
        config=config,
    )

    evaluation_with_wall = evaluate_individual(
        individual=individual_with_wall,
        receiver_positions=receiver_positions,
        valid_positions=[
            (0.0, 0.0),
        ],
        blocked_positions=[],
        config=config,
        floor_plan_walls=(make_wall(),),
        floor_plan_wall_loss_db=10.0,
    )

    assert (
        evaluation_with_wall.average_rssi_dbm
        < evaluation_without_wall.average_rssi_dbm
    )


def test_evaluate_individual_rejects_negative_floor_plan_wall_loss():
    config = NSGA2Config(
        min_ap_count=1,
        max_ap_count=1,
        minimum_sinr_db=None,
    )

    individual = Individual(
        positions=(
            (0.0, 0.0),
        )
    )

    with pytest.raises(ValueError):
        evaluate_individual(
            individual=individual,
            receiver_positions=[
                (2.0, 0.0),
            ],
            valid_positions=[
                (0.0, 0.0),
            ],
            blocked_positions=[],
            config=config,
            floor_plan_walls=(make_wall(),),
            floor_plan_wall_loss_db=-5.0,
        )


def test_optimize_accepts_floor_plan_walls():
    config = NSGA2Config(
        population_size=6,
        generations=2,
        min_ap_count=1,
        max_ap_count=1,
        minimum_sinr_db=None,
        random_seed=42,
    )

    receiver_positions = [
        (0.0, 0.0),
        (2.0, 0.0),
        (4.0, 0.0),
    ]

    valid_positions = [
        (0.0, 0.0),
        (2.0, 0.0),
        (4.0, 0.0),
    ]

    wall = make_wall()

    result = optimize(
        receiver_positions=receiver_positions,
        valid_positions=valid_positions,
        blocked_positions=[],
        config=config,
        floor_plan_walls=(wall,),
        floor_plan_wall_loss_db=10.0,
    )

    assert isinstance(
        result,
        OptimizationResult,
    )

    assert result.pareto_front

    assert result.generations_completed == 2


def test_optimize_rejects_negative_floor_plan_wall_loss():
    config = NSGA2Config(
        population_size=6,
        generations=2,
        min_ap_count=1,
        max_ap_count=1,
        minimum_sinr_db=None,
    )

    with pytest.raises(ValueError):
        optimize(
            receiver_positions=[
                (0.0, 0.0),
            ],
            valid_positions=[
                (0.0, 0.0),
            ],
            config=config,
            floor_plan_wall_loss_db=-1.0,
        )


# ============================================================
# Individual
# ============================================================


def test_individual_creation():
    individual = make_individual(
        positions=[
            (1.0, 1.0),
            (2.0, 2.0),
        ],
        objectives=[
            -90.0,
            2.0,
            -60.0,
        ],
    )

    assert individual.ap_count == 2
    assert individual.objectives == (
        -90.0,
        2.0,
        -60.0,
    )
    assert individual.rank is None
    assert individual.crowding_distance == 0.0


def test_individual_ap_count():
    individual = make_individual(
        positions=[
            (1.0, 1.0),
            (2.0, 2.0),
            (3.0, 3.0),
        ],
        objectives=[
            -95.0,
            3.0,
            -55.0,
        ],
    )

    assert individual.ap_count == 3


# ============================================================
# Tournament Selection
# ============================================================


def test_tournament_selection_prefers_better_rank():
    better = make_individual(
        positions=[(0.0, 0.0)],
        objectives=[-95.0, 2.0, -70.0],
    )

    worse = make_individual(
        positions=[(1.0, 1.0)],
        objectives=[-80.0, 4.0, -50.0],
    )

    better.rank = 0
    worse.rank = 1

    selected = tournament_select(
        [better, worse],
        rng=random.Random(10),
    )

    assert selected is better


def test_tournament_selection_prefers_crowding_distance():
    first = make_individual(
        positions=[(0.0, 0.0)],
        objectives=[-90.0, 2.0, -60.0],
    )

    second = make_individual(
        positions=[(1.0, 1.0)],
        objectives=[-90.0, 2.0, -60.0],
    )

    first.rank = 0
    second.rank = 0

    first.crowding_distance = 10.0
    second.crowding_distance = 2.0

    selected = tournament_select(
        [first, second],
        rng=random.Random(10),
    )

    assert selected is first


# ============================================================
# Crossover
# ============================================================


def test_crossover_creates_two_children():
    parent_one = make_individual(
        positions=[
            (0.0, 0.0),
            (0.5, 0.0),
        ],
        objectives=[],
    )

    parent_two = make_individual(
        positions=[
            (1.0, 0.0),
            (0.0, 0.5),
        ],
        objectives=[],
    )

    child_one, child_two = crossover(
        parent_one,
        parent_two,
        VALID_POSITIONS,
        rng=random.Random(5),
    )

    assert len(child_one.positions) == 2
    assert len(child_two.positions) == 2

    assert all(
        position in VALID_POSITIONS
        for position in child_one.positions
    )

    assert all(
        position in VALID_POSITIONS
        for position in child_two.positions
    )


def test_crossover_does_not_create_duplicates():
    parent_one = make_individual(
        positions=[
            (0.0, 0.0),
            (0.5, 0.0),
        ],
        objectives=[],
    )

    parent_two = make_individual(
        positions=[
            (0.5, 0.0),
            (1.0, 0.0),
        ],
        objectives=[],
    )

    child_one, child_two = crossover(
        parent_one,
        parent_two,
        VALID_POSITIONS,
        rng=random.Random(2),
    )

    assert len(child_one.positions) == len(
        set(child_one.positions)
    )

    assert len(child_two.positions) == len(
        set(child_two.positions)
    )


# ============================================================
# Mutation
# ============================================================


def test_mutation_probability_zero_keeps_positions():
    individual = make_individual(
        positions=[
            (0.0, 0.0),
            (0.5, 0.0),
        ],
        objectives=[],
    )

    mutated = mutate(
        individual,
        VALID_POSITIONS,
        mutation_probability=0.0,
        rng=random.Random(1),
    )

    assert mutated.positions == individual.positions


def test_mutation_probability_one_changes_possible_position():
    individual = make_individual(
        positions=[
            (0.0, 0.0),
            (0.5, 0.0),
        ],
        objectives=[],
    )

    mutated = mutate(
        individual,
        VALID_POSITIONS,
        mutation_probability=1.0,
        rng=random.Random(4),
    )

    assert all(
        position in VALID_POSITIONS
        for position in mutated.positions
    )

    assert len(mutated.positions) == len(
        set(mutated.positions)
    )


def test_invalid_mutation_probability():
    individual = make_individual(
        positions=[(0.0, 0.0)],
        objectives=[],
    )

    with pytest.raises(ValueError):
        mutate(
            individual,
            VALID_POSITIONS,
            mutation_probability=1.5,
        )


# ============================================================
# Offspring Generation
# ============================================================


def test_create_offspring():
    population = [
        make_individual(
            positions=[
                (0.0, 0.0),
                (0.5, 0.0),
            ],
            objectives=[-90.0, 2.0, -60.0],
        ),
        make_individual(
            positions=[
                (1.0, 0.0),
                (0.0, 0.5),
            ],
            objectives=[-85.0, 2.0, -55.0],
        ),
        make_individual(
            positions=[
                (0.5, 0.5),
                (1.0, 0.5),
            ],
            objectives=[-80.0, 2.0, -50.0],
        ),
    ]

    fronts = fast_non_dominated_sort(
        population
    )

    for front in fronts:
        calculate_crowding_distance(front)

    offspring = create_offspring(
        population=population,
        valid_positions=VALID_POSITIONS,
        offspring_count=6,
        crossover_probability=0.9,
        mutation_probability=0.2,
        rng=random.Random(42),
    )

    assert len(offspring) == 6

    for child in offspring:
        assert child.positions

        assert all(
            position in VALID_POSITIONS
            for position in child.positions
        )

        assert len(child.positions) == len(
            set(child.positions)
        )


def test_create_offspring_invalid_population():
    with pytest.raises(ValueError):
        create_offspring(
            population=[],
            valid_positions=VALID_POSITIONS,
            offspring_count=2,
        )


def test_create_offspring_invalid_count():
    population = [
        make_individual(
            positions=[(0.0, 0.0)],
            objectives=[-90.0, 1.0, -60.0],
        )
    ]

    with pytest.raises(ValueError):
        create_offspring(
            population=population,
            valid_positions=VALID_POSITIONS,
            offspring_count=0,
        )


# ============================================================
# Dominance
# ============================================================


def test_dominance():
    better = make_individual(
        positions=[(1.0, 1.0)],
        objectives=[
            -95.0,
            2.0,
            -70.0,
        ],
    )

    worse = make_individual(
        positions=[(2.0, 2.0)],
        objectives=[
            -90.0,
            3.0,
            -60.0,
        ],
    )

    assert dominates(better, worse)
    assert not dominates(worse, better)


def test_equal_individuals_do_not_dominate():
    first = make_individual(
        positions=[(1.0, 1.0)],
        objectives=[
            -90.0,
            2.0,
            -60.0,
        ],
    )

    second = make_individual(
        positions=[(2.0, 2.0)],
        objectives=[
            -90.0,
            2.0,
            -60.0,
        ],
    )

    assert not dominates(first, second)
    assert not dominates(second, first)


def test_partial_tradeoff_does_not_dominate():
    first = make_individual(
        positions=[(1.0, 1.0)],
        objectives=[
            -95.0,
            4.0,
            -60.0,
        ],
    )

    second = make_individual(
        positions=[(2.0, 2.0)],
        objectives=[
            -90.0,
            2.0,
            -60.0,
        ],
    )

    assert not dominates(first, second)
    assert not dominates(second, first)


def test_dominance_requires_objectives():
    first = make_individual(
        positions=[(1.0, 1.0)],
        objectives=[],
    )

    second = make_individual(
        positions=[(2.0, 2.0)],
        objectives=[
            -90.0,
            2.0,
            -60.0,
        ],
    )

    with pytest.raises(ValueError):
        dominates(first, second)


def test_dominance_requires_same_objective_count():
    first = make_individual(
        positions=[(1.0, 1.0)],
        objectives=[
            -90.0,
            2.0,
        ],
    )

    second = make_individual(
        positions=[(2.0, 2.0)],
        objectives=[
            -90.0,
            2.0,
            -60.0,
        ],
    )

    with pytest.raises(ValueError):
        dominates(first, second)


# ============================================================
# Non-dominated Sorting
# ============================================================


def test_non_dominated_sort():
    first = make_individual(
        positions=[(1.0, 1.0)],
        objectives=[
            -95.0,
            2.0,
            -70.0,
        ],
    )

    second = make_individual(
        positions=[(2.0, 2.0)],
        objectives=[
            -90.0,
            3.0,
            -60.0,
        ],
    )

    third = make_individual(
        positions=[(3.0, 3.0)],
        objectives=[
            -80.0,
            4.0,
            -50.0,
        ],
    )

    fronts = fast_non_dominated_sort(
        [first, second, third]
    )

    assert len(fronts) == 3

    assert fronts[0] == [first]
    assert fronts[1] == [second]
    assert fronts[2] == [third]

    assert first.rank == 0
    assert second.rank == 1
    assert third.rank == 2


def test_multiple_non_dominated_solutions():
    first = make_individual(
        positions=[(1.0, 1.0)],
        objectives=[
            -95.0,
            5.0,
            -50.0,
        ],
    )

    second = make_individual(
        positions=[(2.0, 2.0)],
        objectives=[
            -90.0,
            3.0,
            -60.0,
        ],
    )

    third = make_individual(
        positions=[(3.0, 3.0)],
        objectives=[
            -80.0,
            2.0,
            -70.0,
        ],
    )

    fronts = fast_non_dominated_sort(
        [first, second, third]
    )

    assert len(fronts) == 1
    assert len(fronts[0]) == 3

    assert all(
        individual.rank == 0
        for individual in fronts[0]
    )


def test_empty_population():
    fronts = fast_non_dominated_sort([])

    assert fronts == []


# ============================================================
# Crowding Distance
# ============================================================


def test_crowding_distance_two_points():
    first = make_individual(
        positions=[(1.0, 1.0)],
        objectives=[
            -95.0,
            2.0,
            -70.0,
        ],
    )

    second = make_individual(
        positions=[(2.0, 2.0)],
        objectives=[
            -90.0,
            3.0,
            -60.0,
        ],
    )

    calculate_crowding_distance(
        [first, second]
    )

    assert first.crowding_distance == float(
        "inf"
    )
    assert second.crowding_distance == float(
        "inf"
    )


def test_crowding_distance_three_points():
    first = make_individual(
        positions=[(1.0, 1.0)],
        objectives=[
            -95.0,
            2.0,
            -70.0,
        ],
    )

    middle = make_individual(
        positions=[(2.0, 2.0)],
        objectives=[
            -90.0,
            3.0,
            -60.0,
        ],
    )

    last = make_individual(
        positions=[(3.0, 3.0)],
        objectives=[
            -80.0,
            4.0,
            -50.0,
        ],
    )

    calculate_crowding_distance(
        [first, middle, last]
    )

    assert first.crowding_distance == float(
        "inf"
    )
    assert last.crowding_distance == float(
        "inf"
    )
    assert middle.crowding_distance > 0.0


def test_crowding_distance_empty_front():
    calculate_crowding_distance([])


def test_crowding_distance_same_objective_values():
    first = make_individual(
        positions=[(1.0, 1.0)],
        objectives=[
            -90.0,
            2.0,
            -60.0,
        ],
    )

    second = make_individual(
        positions=[(2.0, 2.0)],
        objectives=[
            -90.0,
            2.0,
            -60.0,
        ],
    )

    third = make_individual(
        positions=[(3.0, 3.0)],
        objectives=[
            -90.0,
            2.0,
            -60.0,
        ],
    )

    calculate_crowding_distance(
        [first, second, third]
    )

    assert first.crowding_distance == float(
        "inf"
    )
    assert third.crowding_distance == float(
        "inf"
    )


# ============================================================
# Channel-Aware Regression Tests
# ============================================================


def test_individual_channels_match_positions():
    """Every AP must have exactly one associated channel."""

    individual = Individual(
        positions=(
            (0.0, 0.0),
            (0.5, 0.0),
            (1.0, 0.0),
        ),
        channels=(1, 6, 11),
    )

    assert len(individual.positions) == len(
        individual.channels
    )

    assert individual.channels == (
        1,
        6,
        11,
    )

    assert all(
        channel in (1, 6, 11)
        for channel in individual.channels
    )


def test_channel_aware_crossover_preserves_position_channel_pairing():
    """Crossover must keep each AP position paired with its channel."""

    parent_one = Individual(
        positions=(
            (0.0, 0.0),
            (0.5, 0.0),
        ),
        channels=(1, 6),
    )

    parent_two = Individual(
        positions=(
            (1.0, 0.0),
            (0.0, 0.5),
        ),
        channels=(11, 1),
    )

    child_one, child_two = crossover(
        parent_one,
        parent_two,
        VALID_POSITIONS,
        rng=random.Random(5),
    )

    assert len(child_one.positions) == len(
        child_one.channels
    )

    assert len(child_two.positions) == len(
        child_two.channels
    )

    assert all(
        channel in (1, 6, 11)
        for channel in child_one.channels
    )

    assert all(
        channel in (1, 6, 11)
        for channel in child_two.channels
    )

    valid_genes = {
        ((0.0, 0.0), 1),
        ((0.5, 0.0), 6),
        ((1.0, 0.0), 11),
        ((0.0, 0.5), 1),
    }

    child_genes = (
        set(zip(child_one.positions, child_one.channels))
        | set(zip(child_two.positions, child_two.channels))
    )

    assert child_genes <= valid_genes


def test_channel_mutation_changes_channel():
    """Channel mutation must be able to change an AP channel."""

    individual = Individual(
        positions=(
            (0.0, 0.0),
            (0.5, 0.0),
        ),
        channels=(1, 6),
    )

    mutated = mutate(
        individual=individual,
        valid_positions=VALID_POSITIONS,
        mutation_probability=1.0,
        rng=random.Random(4),
    )

    assert len(mutated.positions) == len(
        mutated.channels
    )

    assert all(
        channel in (1, 6, 11)
        for channel in mutated.channels
    )

    assert any(
        original != new
        for original, new in zip(
            individual.channels,
            mutated.channels,
        )
    )


def test_invalid_channel_is_rejected():
    """Production evaluation must reject unsupported channels."""

    config = NSGA2Config(
        min_ap_count=1,
        max_ap_count=2,
        minimum_sinr_db=None,
        available_channels=(1, 6, 11),
    )

    individual = Individual(
        positions=(
            (0.0, 0.0),
        ),
        channels=(999,),
    )

    with pytest.raises(ValueError):
        evaluate_individual(
            individual=individual,
            receiver_positions=[
                (0.5, 0.0),
                (1.0, 0.0),
            ],
            valid_positions=VALID_POSITIONS,
            blocked_positions=[],
            config=config,
        )


def test_channel_assignment_affects_sinr():
    """
    Different channel assignments must affect RF interference/SINR.
    """

    config = NSGA2Config(
        min_ap_count=2,
        max_ap_count=2,
        minimum_sinr_db=None,
        available_channels=(1, 6, 11),
        same_channel_factor=1.0,
        adjacent_channel_factor=0.5,
        non_overlapping_factor=0.0,
    )

    positions = (
        (0.0, 0.0),
        (1.0, 0.0),
    )

    receiver_positions = [
        (0.5, 0.0),
        (1.0, 0.5),
    ]

    same_channel = Individual(
        positions=positions,
        channels=(1, 1),
    )

    different_channels = Individual(
        positions=positions,
        channels=(1, 6),
    )

    same_channel_evaluation = evaluate_individual(
        individual=same_channel,
        receiver_positions=receiver_positions,
        valid_positions=VALID_POSITIONS,
        blocked_positions=[],
        config=config,
    )

    different_channel_evaluation = evaluate_individual(
        individual=different_channels,
        receiver_positions=receiver_positions,
        valid_positions=VALID_POSITIONS,
        blocked_positions=[],
        config=config,
    )

    assert (
        different_channel_evaluation.average_sinr_db
        > same_channel_evaluation.average_sinr_db
    )
"""
GeoTwin AI Indoor - Channel-Aware NSGA-II Experiment

Purpose
-------
Validate whether NSGA-II can optimize AP placement together with
channel assignment before modifying the production NSGA-II individual
representation.

The experiment compares:

1. Fixed all-Channel-1 AP assignment
2. Channel-aware NSGA-II search using channels 1, 6 and 11

The optimizer searches over:
    - AP positions
    - AP channels

Objectives:
    1. Maximize coverage
    2. Minimize AP count
    3. Minimize interference

NOTE:
This is an experiment/prototype. The production nsga2.py is not modified.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Sequence, Tuple

from geotwin.scenario.indoor import IndoorScenario
from geotwin.rf.ap import AccessPoint
from geotwin.rf.coverage import GridEvaluation, evaluate_grid
from geotwin.optimization.objectives import calculate_objectives
from geotwin.optimization.constraints import (
    ConstraintConfig,
    validate_configuration,
)


Position = Tuple[float, float]
ObjectiveVector = Tuple[float, float, float]


# ======================================================================
# CONFIGURATION
# ======================================================================

POPULATION_SIZE = 50
GENERATIONS = 50

CROSSOVER_PROBABILITY = 0.9
MUTATION_PROBABILITY = 0.1

MIN_AP_COUNT = 1
MAX_AP_COUNT = 5

RANDOM_SEED = 42

AVAILABLE_CHANNELS = (1, 6, 11)

MINIMUM_RSSI_DBM = -67.0
MINIMUM_SINR_DB = 20.0
NOISE_FLOOR_DBM = -95.0

FREQUENCY_HZ = 2.4e9
TX_POWER_DBM = 20.0
ANTENNA_GAIN_DBI = 3.0

REFERENCE_DISTANCE_M = 1.0
PATH_LOSS_EXPONENT = 3.0

WALL_MATERIAL = "concrete"
NUMBER_OF_WALLS = 4
OBSTACLE_LOSS_DB = 6.0

SAME_CHANNEL_FACTOR = 1.0
ADJACENT_CHANNEL_FACTOR = 0.5
NON_OVERLAPPING_FACTOR = 0.0


# ======================================================================
# DATA STRUCTURES
# ======================================================================

@dataclass
class ChannelAwareIndividual:
    """
    One candidate solution containing AP positions and channel assignment.
    """

    positions: Tuple[Position, ...]
    channels: Tuple[int, ...]

    objectives: ObjectiveVector = ()
    rank: int | None = None
    crowding_distance: float = 0.0

    @property
    def ap_count(self) -> int:
        return len(self.positions)


@dataclass
class ChannelAwareResult:
    pareto_front: List[ChannelAwareIndividual]
    generations_completed: int
    history: List[dict]


# ======================================================================
# ACCESS POINT CREATION
# ======================================================================

def create_access_points(
    individual: ChannelAwareIndividual,
) -> List[AccessPoint]:
    """
    Convert an individual into RF AccessPoint objects.
    """

    if len(individual.positions) != len(individual.channels):
        raise ValueError(
            "Number of positions must equal number of channels."
        )

    access_points = []

    for position, channel in zip(
        individual.positions,
        individual.channels,
    ):
        access_points.append(
            AccessPoint(
                position=position,
                frequency_hz=FREQUENCY_HZ,
                tx_power_dbm=TX_POWER_DBM,
                antenna_gain_dbi=ANTENNA_GAIN_DBI,
                channel=channel,
            )
        )

    return access_points


# ======================================================================
# VALIDATION
# ======================================================================

def validate_individual(
    individual: ChannelAwareIndividual,
    valid_positions: Sequence[Position],
    blocked_positions: Sequence[Position],
) -> None:
    """
    Validate AP positions and channel assignment.
    """

    if not individual.positions:
        raise ValueError("Individual must contain at least one AP.")

    if len(individual.positions) != len(individual.channels):
        raise ValueError(
            "Each AP must have exactly one channel."
        )

    constraint_config = ConstraintConfig(
        min_ap_count=MIN_AP_COUNT,
        max_ap_count=MAX_AP_COUNT,
    )

    result = validate_configuration(
        positions=individual.positions,
        valid_positions=valid_positions,
        blocked_positions=blocked_positions,
        config=constraint_config,
    )

    if not result.valid:
        raise ValueError(
            "Invalid individual: "
            + "; ".join(result.errors)
        )

    for channel in individual.channels:
        if channel not in AVAILABLE_CHANNELS:
            raise ValueError(
                f"Unsupported channel: {channel}. "
                f"Allowed channels: {AVAILABLE_CHANNELS}"
            )


# ======================================================================
# EVALUATION
# ======================================================================

def evaluate_individual(
    individual: ChannelAwareIndividual,
    receiver_positions: Sequence[Position],
    valid_positions: Sequence[Position],
    blocked_positions: Sequence[Position],
) -> GridEvaluation:
    """
    Evaluate one channel-aware individual.
    """

    validate_individual(
        individual=individual,
        valid_positions=valid_positions,
        blocked_positions=blocked_positions,
    )

    access_points = create_access_points(individual)

    evaluation = evaluate_grid(
        receiver_positions=receiver_positions,
        access_points=access_points,

        noise_floor_dbm=NOISE_FLOOR_DBM,
        minimum_rssi_dbm=MINIMUM_RSSI_DBM,
        minimum_sinr_db=MINIMUM_SINR_DB,

        reference_distance_m=REFERENCE_DISTANCE_M,
        path_loss_exponent=PATH_LOSS_EXPONENT,

        wall_material=WALL_MATERIAL,
        number_of_walls=NUMBER_OF_WALLS,
        obstacle_loss_db=OBSTACLE_LOSS_DB,

        same_channel_factor=SAME_CHANNEL_FACTOR,
        adjacent_channel_factor=ADJACENT_CHANNEL_FACTOR,
        non_overlapping_factor=NON_OVERLAPPING_FACTOR,
    )

    objective_result = calculate_objectives(
        evaluation=evaluation,
        ap_count=individual.ap_count,
    )

    individual.objectives = objective_result.values

    return evaluation


# ======================================================================
# RANDOM INDIVIDUAL
# ======================================================================

def random_individual(
    valid_positions: Sequence[Position],
    rng: random.Random,
) -> ChannelAwareIndividual:
    """
    Generate a random AP configuration with random channels.
    """

    maximum_ap_count = min(
        MAX_AP_COUNT,
        len(valid_positions),
    )

    ap_count = rng.randint(
        MIN_AP_COUNT,
        maximum_ap_count,
    )

    positions = tuple(
        rng.sample(
            list(valid_positions),
            ap_count,
        )
    )

    channels = tuple(
        rng.choice(
            AVAILABLE_CHANNELS
        )
        for _ in range(ap_count)
    )

    return ChannelAwareIndividual(
        positions=positions,
        channels=channels,
    )


# ======================================================================
# INITIAL POPULATION
# ======================================================================

def initialize_population(
    valid_positions: Sequence[Position],
    rng: random.Random,
) -> List[ChannelAwareIndividual]:
    """
    Create initial population.
    """

    return [
        random_individual(
            valid_positions,
            rng,
        )
        for _ in range(POPULATION_SIZE)
    ]


# ======================================================================
# DOMINATION
# ======================================================================

def dominates(
    first: ChannelAwareIndividual,
    second: ChannelAwareIndividual,
) -> bool:
    """
    All objectives are minimization objectives.
    """

    if not first.objectives:
        raise ValueError(
            "First individual has no objectives."
        )

    if not second.objectives:
        raise ValueError(
            "Second individual has no objectives."
        )

    if len(first.objectives) != len(second.objectives):
        raise ValueError(
            "Objective vectors must have equal length."
        )

    no_worse = all(
        a <= b
        for a, b in zip(
            first.objectives,
            second.objectives,
        )
    )

    strictly_better = any(
        a < b
        for a, b in zip(
            first.objectives,
            second.objectives,
        )
    )

    return no_worse and strictly_better


# ======================================================================
# NON-DOMINATED SORT
# ======================================================================

def fast_non_dominated_sort(
    population: Sequence[ChannelAwareIndividual],
) -> List[List[ChannelAwareIndividual]]:
    """
    Fast non-dominated sorting.
    """

    if not population:
        return []

    domination_count = {
        id(individual): 0
        for individual in population
    }

    dominated_solutions = {
        id(individual): []
        for individual in population
    }

    fronts: List[List[ChannelAwareIndividual]] = [[]]

    for individual in population:

        for other in population:

            if individual is other:
                continue

            if dominates(individual, other):

                dominated_solutions[
                    id(individual)
                ].append(other)

            elif dominates(other, individual):

                domination_count[
                    id(individual)
                ] += 1

        if domination_count[
            id(individual)
        ] == 0:

            individual.rank = 0
            fronts[0].append(individual)

    current = 0

    while current < len(fronts):

        next_front = []

        for individual in fronts[current]:

            for dominated in dominated_solutions[
                id(individual)
            ]:

                domination_count[
                    id(dominated)
                ] -= 1

                if domination_count[
                    id(dominated)
                ] == 0:

                    dominated.rank = current + 1

                    next_front.append(
                        dominated
                    )

        if next_front:
            fronts.append(next_front)

        current += 1

    return fronts


# ======================================================================
# CROWDING DISTANCE
# ======================================================================

def calculate_crowding_distance(
    front: Sequence[ChannelAwareIndividual],
) -> None:
    """
    Calculate crowding distance for a Pareto front.
    """

    if not front:
        return

    for individual in front:
        individual.crowding_distance = 0.0

    if len(front) <= 2:

        for individual in front:
            individual.crowding_distance = float("inf")

        return

    objective_count = len(
        front[0].objectives
    )

    for objective_index in range(
        objective_count
    ):

        sorted_front = sorted(
            front,
            key=lambda individual:
            individual.objectives[
                objective_index
            ],
        )

        sorted_front[
            0
        ].crowding_distance = float("inf")

        sorted_front[
            -1
        ].crowding_distance = float("inf")

        minimum = sorted_front[
            0
        ].objectives[
            objective_index
        ]

        maximum = sorted_front[
            -1
        ].objectives[
            objective_index
        ]

        objective_range = maximum - minimum

        if objective_range == 0:
            continue

        for index in range(
            1,
            len(sorted_front) - 1,
        ):

            if (
                sorted_front[index]
                .crowding_distance
                == float("inf")
            ):
                continue

            previous_value = (
                sorted_front[index - 1]
                .objectives[objective_index]
            )

            next_value = (
                sorted_front[index + 1]
                .objectives[objective_index]
            )

            sorted_front[index].crowding_distance += (
                next_value - previous_value
            ) / objective_range


# ======================================================================
# TOURNAMENT SELECTION
# ======================================================================

def tournament_select(
    population: Sequence[ChannelAwareIndividual],
    rng: random.Random,
) -> ChannelAwareIndividual:
    """
    Binary tournament selection.
    """

    first, second = rng.sample(
        list(population),
        2,
    )

    first_rank = (
        first.rank
        if first.rank is not None
        else float("inf")
    )

    second_rank = (
        second.rank
        if second.rank is not None
        else float("inf")
    )

    if first_rank < second_rank:
        return first

    if second_rank < first_rank:
        return second

    if (
        first.crowding_distance
        > second.crowding_distance
    ):
        return first

    if (
        second.crowding_distance
        > first.crowding_distance
    ):
        return second

    return rng.choice(
        [first, second]
    )


# ======================================================================
# CHANNEL-AWARE CROSSOVER
# ======================================================================

def crossover(
    parent_one: ChannelAwareIndividual,
    parent_two: ChannelAwareIndividual,
    valid_positions: Sequence[Position],
    rng: random.Random,
) -> Tuple[
    ChannelAwareIndividual,
    ChannelAwareIndividual,
]:
    """
    Crossover both AP positions and channel assignments.

    Position and channel remain paired.
    """

    parent_one_genes = list(
        zip(
            parent_one.positions,
            parent_one.channels,
        )
    )

    parent_two_genes = list(
        zip(
            parent_two.positions,
            parent_two.channels,
        )
    )

    combined = (
        parent_one_genes
        + parent_two_genes
    )

    valid_set = set(valid_positions)

    unique_genes = []

    seen_positions = set()

    for position, channel in combined:

        if position not in valid_set:
            continue

        if position in seen_positions:
            continue

        if channel not in AVAILABLE_CHANNELS:
            continue

        seen_positions.add(position)

        unique_genes.append(
            (position, channel)
        )

    rng.shuffle(unique_genes)

    child_one_count = min(
        parent_one.ap_count,
        len(unique_genes),
    )

    child_two_count = min(
        parent_two.ap_count,
        len(unique_genes),
    )

    child_one_genes = unique_genes[
        :child_one_count
    ]

    child_two_genes = unique_genes[
        :child_two_count
    ]

    if not child_one_genes:

        position = rng.choice(
            list(valid_positions)
        )

        child_one_genes = [
            (
                position,
                rng.choice(AVAILABLE_CHANNELS),
            )
        ]

    if not child_two_genes:

        position = rng.choice(
            list(valid_positions)
        )

        child_two_genes = [
            (
                position,
                rng.choice(AVAILABLE_CHANNELS),
            )
        ]

    child_one = ChannelAwareIndividual(
        positions=tuple(
            gene[0]
            for gene in child_one_genes
        ),
        channels=tuple(
            gene[1]
            for gene in child_one_genes
        ),
    )

    child_two = ChannelAwareIndividual(
        positions=tuple(
            gene[0]
            for gene in child_two_genes
        ),
        channels=tuple(
            gene[1]
            for gene in child_two_genes
        ),
    )

    return child_one, child_two


# ======================================================================
# MUTATION
# ======================================================================

def mutate(
    individual: ChannelAwareIndividual,
    valid_positions: Sequence[Position],
    rng: random.Random,
) -> ChannelAwareIndividual:
    """
    Mutate both AP position and channel.

    Position mutation:
        Move AP to another valid grid point.

    Channel mutation:
        Change AP to another allowed channel.
    """

    positions = list(
        individual.positions
    )

    channels = list(
        individual.channels
    )

    for index in range(len(positions)):

        if rng.random() <= MUTATION_PROBABILITY:

            available_positions = [
                position
                for position in valid_positions
                if position not in positions
                or position == positions[index]
            ]

            if available_positions:

                positions[index] = rng.choice(
                    available_positions
                )

        if rng.random() <= MUTATION_PROBABILITY:

            alternative_channels = [
                channel
                for channel in AVAILABLE_CHANNELS
                if channel != channels[index]
            ]

            if alternative_channels:

                channels[index] = rng.choice(
                    alternative_channels
                )

    return ChannelAwareIndividual(
        positions=tuple(positions),
        channels=tuple(channels),
    )


# ======================================================================
# OFFSPRING
# ======================================================================

def create_offspring(
    population: Sequence[ChannelAwareIndividual],
    valid_positions: Sequence[Position],
    rng: random.Random,
) -> List[ChannelAwareIndividual]:

    offspring = []

    while len(offspring) < POPULATION_SIZE:

        parent_one = tournament_select(
            population,
            rng,
        )

        parent_two = tournament_select(
            population,
            rng,
        )

        if rng.random() <= CROSSOVER_PROBABILITY:

            child_one, child_two = crossover(
                parent_one,
                parent_two,
                valid_positions,
                rng,
            )

        else:

            child_one = ChannelAwareIndividual(
                positions=parent_one.positions,
                channels=parent_one.channels,
            )

            child_two = ChannelAwareIndividual(
                positions=parent_two.positions,
                channels=parent_two.channels,
            )

        child_one = mutate(
            child_one,
            valid_positions,
            rng,
        )

        child_two = mutate(
            child_two,
            valid_positions,
            rng,
        )

        offspring.append(child_one)

        if len(offspring) < POPULATION_SIZE:
            offspring.append(child_two)

    return offspring


# ======================================================================
# ENVIRONMENTAL SELECTION
# ======================================================================

def select_next_generation(
    combined_population: Sequence[ChannelAwareIndividual],
) -> List[ChannelAwareIndividual]:

    fronts = fast_non_dominated_sort(
        combined_population
    )

    selected = []

    for front in fronts:

        calculate_crowding_distance(front)

        if (
            len(selected)
            + len(front)
            <= POPULATION_SIZE
        ):

            selected.extend(front)

        else:

            remaining = (
                POPULATION_SIZE
                - len(selected)
            )

            sorted_front = sorted(
                front,
                key=lambda individual:
                individual.crowding_distance,
                reverse=True,
            )

            selected.extend(
                sorted_front[:remaining]
            )

            break

    return selected


# ======================================================================
# OPTIMIZATION
# ======================================================================

def optimize_channel_aware(
    receiver_positions: Sequence[Position],
    valid_positions: Sequence[Position],
    blocked_positions: Sequence[Position],
) -> ChannelAwareResult:
    """
    Run channel-aware NSGA-II.
    """

    rng = random.Random(
        RANDOM_SEED
    )

    population = initialize_population(
        valid_positions,
        rng,
    )

    evaluate_population(
        population,
        receiver_positions,
        valid_positions,
        blocked_positions,
    )

    history = []

    for generation in range(
        GENERATIONS
    ):

        fronts = fast_non_dominated_sort(
            population
        )

        for front in fronts:
            calculate_crowding_distance(front)

        offspring = create_offspring(
            population,
            valid_positions,
            rng,
        )

        evaluate_population(
            offspring,
            receiver_positions,
            valid_positions,
            blocked_positions,
        )

        combined = (
            list(population)
            + offspring
        )

        population = select_next_generation(
            combined
        )

        current_fronts = fast_non_dominated_sort(
            population
        )

        best_front = (
            current_fronts[0]
            if current_fronts
            else []
        )

        history.append(
            {
                "generation": generation + 1,
                "population": len(population),
                "pareto": len(best_front),
            }
        )

    final_fronts = fast_non_dominated_sort(
        population
    )

    pareto_front = (
        final_fronts[0]
        if final_fronts
        else []
    )

    for front in final_fronts:
        calculate_crowding_distance(front)

    return ChannelAwareResult(
        pareto_front=pareto_front,
        generations_completed=GENERATIONS,
        history=history,
    )


# ======================================================================
# POPULATION EVALUATION
# ======================================================================

def evaluate_population(
    population: Sequence[ChannelAwareIndividual],
    receiver_positions: Sequence[Position],
    valid_positions: Sequence[Position],
    blocked_positions: Sequence[Position],
) -> None:

    for individual in population:

        evaluate_individual(
            individual=individual,
            receiver_positions=receiver_positions,
            valid_positions=valid_positions,
            blocked_positions=blocked_positions,
        )


# ======================================================================
# FIXED BASELINE
# ======================================================================

def evaluate_fixed_configuration(
    positions: Sequence[Position],
    channels: Sequence[int],
    receiver_positions: Sequence[Position],
    valid_positions: Sequence[Position],
    blocked_positions: Sequence[Position],
) -> tuple[
    ChannelAwareIndividual,
    GridEvaluation,
]:
    """
    Evaluate a fixed AP/channel configuration.
    """

    individual = ChannelAwareIndividual(
        positions=tuple(positions),
        channels=tuple(channels),
    )

    evaluation = evaluate_individual(
        individual=individual,
        receiver_positions=receiver_positions,
        valid_positions=valid_positions,
        blocked_positions=blocked_positions,
    )

    return individual, evaluation


# ======================================================================
# PRINT EVALUATION
# ======================================================================

def print_evaluation(
    label: str,
    individual: ChannelAwareIndividual,
    evaluation: GridEvaluation,
) -> None:

    print()
    print("-" * 100)
    print(label)
    print("-" * 100)

    print(
        f"AP count              : "
        f"{individual.ap_count}"
    )

    print(
        f"AP positions          : "
        f"{individual.positions}"
    )

    print(
        f"Channels              : "
        f"{individual.channels}"
    )

    print(
        f"Covered points        : "
        f"{evaluation.covered_points}/"
        f"{evaluation.total_points}"
    )

    print(
        f"Coverage              : "
        f"{evaluation.coverage_percent:.2f}%"
    )

    print(
        f"Dead zone             : "
        f"{evaluation.dead_zone_percent:.2f}%"
    )

    print(
        f"Average RSSI          : "
        f"{evaluation.average_rssi_dbm:.2f} dBm"
    )

    print(
        f"Minimum RSSI          : "
        f"{evaluation.minimum_rssi_dbm:.2f} dBm"
    )

    print(
        f"Average SINR          : "
        f"{evaluation.average_sinr_db:.2f} dB"
    )

    print(
        f"Minimum SINR          : "
        f"{evaluation.minimum_sinr_db:.2f} dB"
    )

    print(
        f"Average interference  : "
        f"{evaluation.average_interference_dbm:.2f} dBm"
    )

    print(
        f"Objectives            : "
        f"{individual.objectives}"
    )


# ======================================================================
# UNIQUE PARETO SOLUTIONS
# ======================================================================

def unique_solutions(
    pareto_front: Sequence[ChannelAwareIndividual],
) -> List[ChannelAwareIndividual]:

    seen = set()
    unique = []

    for individual in pareto_front:

        key = (
            individual.positions,
            individual.channels,
        )

        if key in seen:
            continue

        seen.add(key)
        unique.append(individual)

    return unique


# ======================================================================
# MAIN
# ======================================================================

def main() -> None:

    print("=" * 100)
    print(
        "GeoTwin AI Indoor - "
        "Channel-Aware NSGA-II Experiment"
    )
    print("=" * 100)

    # --------------------------------------------------------------
    # Scenario
    # --------------------------------------------------------------

    scenario = IndoorScenario(
        width_m=20.0,
        length_m=15.0,
        grid_resolution_m=0.5,
        floors=1,
        wall_material="concrete",
        wall_thickness_m=0.20,
    )

    scenario.validate()

    valid_positions = scenario.valid_ap_positions
    receiver_positions = scenario.receiver_positions
    blocked_positions = scenario.blocked_positions

    print()
    print("[SCENARIO]")

    print(
        f"Building width       : "
        f"{scenario.width_m} m"
    )

    print(
        f"Building length      : "
        f"{scenario.length_m} m"
    )

    print(
        f"Grid resolution      : "
        f"{scenario.grid_resolution_m} m"
    )

    print(
        f"Floors               : "
        f"{scenario.floors}"
    )

    print(
        f"Wall material        : "
        f"{scenario.wall_material}"
    )

    print(
        f"Wall thickness       : "
        f"{scenario.wall_thickness_m:.2f} m"
    )

    print()
    print("[GRID]")

    print(
        f"Total grid points    : "
        f"{scenario.total_grid_points}"
    )

    print(
        f"Valid AP positions   : "
        f"{len(valid_positions)}"
    )

    print(
        f"Receiver positions   : "
        f"{len(receiver_positions)}"
    )

    print(
        f"Blocked positions    : "
        f"{len(blocked_positions)}"
    )

    print()
    print("[CHANNEL-AWARE NSGA-II CONFIGURATION]")

    print(
        f"Population size      : "
        f"{POPULATION_SIZE}"
    )

    print(
        f"Generations          : "
        f"{GENERATIONS}"
    )

    print(
        f"Crossover probability: "
        f"{CROSSOVER_PROBABILITY}"
    )

    print(
        f"Mutation probability : "
        f"{MUTATION_PROBABILITY}"
    )

    print(
        f"AP count range       : "
        f"{MIN_AP_COUNT} - {MAX_AP_COUNT}"
    )

    print(
        f"Available channels   : "
        f"{AVAILABLE_CHANNELS}"
    )

    print(
        f"Random seed          : "
        f"{RANDOM_SEED}"
    )

    # --------------------------------------------------------------
    # Baseline configurations
    # --------------------------------------------------------------

    baseline_positions = (
        (2.0, 2.0),
        (18.0, 13.0),
        (10.0, 7.5),
        (18.0, 2.0),
        (2.0, 13.0),
    )

    print()
    print("=" * 100)
    print("BASELINE COMPARISON")
    print("=" * 100)

    baseline_results = {}

    for ap_count in range(
        1,
        MAX_AP_COUNT + 1,
    ):

        positions = baseline_positions[:ap_count]

        all_one_channels = tuple(
            1
            for _ in range(ap_count)
        )

        reuse_channels = tuple(
            AVAILABLE_CHANNELS[
                index % len(AVAILABLE_CHANNELS)
            ]
            for index in range(ap_count)
        )

        baseline_all_one, evaluation_all_one = (
            evaluate_fixed_configuration(
                positions=positions,
                channels=all_one_channels,
                receiver_positions=receiver_positions,
                valid_positions=valid_positions,
                blocked_positions=blocked_positions,
            )
        )

        baseline_reuse, evaluation_reuse = (
            evaluate_fixed_configuration(
                positions=positions,
                channels=reuse_channels,
                receiver_positions=receiver_positions,
                valid_positions=valid_positions,
                blocked_positions=blocked_positions,
            )
        )

        baseline_results[ap_count] = {
            "all_one": (
                baseline_all_one,
                evaluation_all_one,
            ),
            "reuse": (
                baseline_reuse,
                evaluation_reuse,
            ),
        }

        print_evaluation(
            f"{ap_count} AP - All Channel 1",
            baseline_all_one,
            evaluation_all_one,
        )

        print_evaluation(
            f"{ap_count} AP - 1-6-11 Reuse",
            baseline_reuse,
            evaluation_reuse,
        )

    # --------------------------------------------------------------
    # Run channel-aware NSGA-II
    # --------------------------------------------------------------

    print()
    print("=" * 100)
    print("STARTING CHANNEL-AWARE NSGA-II")
    print("=" * 100)

    print()
    print(
        "Searching AP positions AND channel assignments..."
    )

    result = optimize_channel_aware(
        receiver_positions=receiver_positions,
        valid_positions=valid_positions,
        blocked_positions=blocked_positions,
    )

    print()
    print("=" * 100)
    print("CHANNEL-AWARE NSGA-II COMPLETE")
    print("=" * 100)

    print(
        f"Generations completed : "
        f"{result.generations_completed}"
    )

    print(
        f"Pareto solutions      : "
        f"{len(result.pareto_front)}"
    )

    # --------------------------------------------------------------
    # Pareto front
    # --------------------------------------------------------------

    unique_pareto = unique_solutions(
        result.pareto_front
    )

    print()
    print("=" * 100)
    print("UNIQUE CHANNEL-AWARE PARETO SOLUTIONS")
    print("=" * 100)

    print(
        f"Unique configurations : "
        f"{len(unique_pareto)}"
    )

    for index, individual in enumerate(
        unique_pareto,
        start=1,
    ):

        print()
        print(
            f"Solution {index}"
        )

        print(
            f"  AP count       : "
            f"{individual.ap_count}"
        )

        print(
            f"  Positions      : "
            f"{individual.positions}"
        )

        print(
            f"  Channels       : "
            f"{individual.channels}"
        )

        print(
            f"  Objectives     : "
            f"{individual.objectives}"
        )

        print(
            f"  Pareto rank    : "
            f"{individual.rank}"
        )

        print(
            f"  Crowding       : "
            f"{individual.crowding_distance}"
        )

    # --------------------------------------------------------------
    # Evaluate Pareto solutions again
    # --------------------------------------------------------------

    print()
    print("=" * 100)
    print("PARETO PERFORMANCE")
    print("=" * 100)

    evaluated_pareto = []

    for individual in unique_pareto:

        evaluation = evaluate_individual(
            individual=individual,
            receiver_positions=receiver_positions,
            valid_positions=valid_positions,
            blocked_positions=blocked_positions,
        )

        evaluated_pareto.append(
            (
                individual,
                evaluation,
            )
        )

    # --------------------------------------------------------------
    # Best coverage
    # --------------------------------------------------------------

    if evaluated_pareto:

        best_coverage = max(
            evaluated_pareto,
            key=lambda item:
            item[1].coverage_percent,
        )

        best_sinr = max(
            evaluated_pareto,
            key=lambda item:
            item[1].average_sinr_db,
        )

        print()
        print("BEST COVERAGE SOLUTION")

        print(
            f"Coverage       : "
            f"{best_coverage[1].coverage_percent:.2f}%"
        )

        print(
            f"AP count       : "
            f"{best_coverage[0].ap_count}"
        )

        print(
            f"Positions      : "
            f"{best_coverage[0].positions}"
        )

        print(
            f"Channels       : "
            f"{best_coverage[0].channels}"
        )

        print(
            f"Average SINR   : "
            f"{best_coverage[1].average_sinr_db:.2f} dB"
        )

        print()
        print("BEST AVERAGE SINR SOLUTION")

        print(
            f"Average SINR   : "
            f"{best_sinr[1].average_sinr_db:.2f} dB"
        )

        print(
            f"Coverage       : "
            f"{best_sinr[1].coverage_percent:.2f}%"
        )

        print(
            f"AP count       : "
            f"{best_sinr[0].ap_count}"
        )

        print(
            f"Positions      : "
            f"{best_sinr[0].positions}"
        )

        print(
            f"Channels       : "
            f"{best_sinr[0].channels}"
        )

    # --------------------------------------------------------------
    # Generation history
    # --------------------------------------------------------------

    print()
    print("=" * 100)
    print("OPTIMIZATION HISTORY")
    print("=" * 100)

    for entry in result.history:

        print(
            f"Generation "
            f"{entry['generation']:3d}: "
            f"population="
            f"{entry['population']:3d}, "
            f"pareto="
            f"{entry['pareto']:3d}"
        )

    # --------------------------------------------------------------
    # Final conclusion
    # --------------------------------------------------------------

    print()
    print("=" * 100)
    print("CHANNEL-AWARE NSGA-II CONCLUSION")
    print("=" * 100)

    if evaluated_pareto:

        best = max(
            evaluated_pareto,
            key=lambda item:
            item[1].coverage_percent,
        )

        print()
        print(
            "Best channel-aware solution:"
        )

        print(
            f"  AP count       : "
            f"{best[0].ap_count}"
        )

        print(
            f"  Coverage       : "
            f"{best[1].coverage_percent:.2f}%"
        )

        print(
            f"  Average RSSI   : "
            f"{best[1].average_rssi_dbm:.2f} dBm"
        )

        print(
            f"  Average SINR   : "
            f"{best[1].average_sinr_db:.2f} dB"
        )

        print(
            f"  Channels       : "
            f"{best[0].channels}"
        )

        print(
            f"  Positions      : "
            f"{best[0].positions}"
        )

        print()
        print(
            "✓ NSGA-II successfully optimized "
            "AP positions and channel assignments."
        )

        print(
            "✓ Channel assignment is now part of "
            "the optimization search space."
        )

        print()
        print(
            "Next step:"
        )

        print(
            "Integrate channel assignment into the "
            "production nsga2.py Individual representation."
        )

    print()
    print(
        "=" * 100
    )

    print(
        "Channel-Aware NSGA-II Experiment Complete"
    )

    print(
        "=" * 100
    )


if __name__ == "__main__":
    main()
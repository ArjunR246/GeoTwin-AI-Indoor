import random

from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

from geotwin.floorplan.walls import WallSegment
from geotwin.optimization.constraints import (
    ConstraintConfig,
    validate_configuration,
)
from geotwin.optimization.objectives import (
    calculate_objectives,
)
from geotwin.rf.ap import AccessPoint
from geotwin.rf.coverage import (
    GridEvaluation,
    evaluate_grid,
)


ObjectiveVector = Tuple[float, ...]
Position = Tuple[float, float]


@dataclass(frozen=True)
class NSGA2Config:
    """Configuration for the end-to-end channel-aware NSGA-II optimizer."""

    population_size: int = 50
    generations: int = 50

    crossover_probability: float = 0.9
    mutation_probability: float = 0.1

    min_ap_count: int = 1
    max_ap_count: int = 5

    random_seed: int | None = 42

    minimum_rssi_dbm: float = -67.0
    minimum_sinr_db: float | None = 20.0
    noise_floor_dbm: float = -95.0

    frequency_hz: float = 2.4e9
    tx_power_dbm: float = 20.0
    antenna_gain_dbi: float = 3.0

    # Legacy/default channel.
    # Used when channels are not explicitly supplied.
    channel: int = 1

    # Channels available to the optimizer.
    available_channels: Tuple[int, ...] = (1, 6, 11)

    reference_distance_m: float = 1.0
    path_loss_exponent: float = 3.0

    wall_material: str = "concrete"
    number_of_walls: int = 0
    obstacle_loss_db: float = 0.0

    same_channel_factor: float = 1.0
    adjacent_channel_factor: float = 0.5
    non_overlapping_factor: float = 0.0

    def validate(self) -> None:
        """Validate optimizer configuration."""

        if self.population_size < 2:
            raise ValueError(
                "population_size must be at least 2."
            )

        if self.generations <= 0:
            raise ValueError(
                "generations must be greater than zero."
            )

        if not 0.0 <= self.crossover_probability <= 1.0:
            raise ValueError(
                "crossover_probability must be between 0 and 1."
            )

        if not 0.0 <= self.mutation_probability <= 1.0:
            raise ValueError(
                "mutation_probability must be between 0 and 1."
            )

        if self.min_ap_count <= 0:
            raise ValueError(
                "min_ap_count must be greater than zero."
            )

        if self.max_ap_count < self.min_ap_count:
            raise ValueError(
                "max_ap_count must be >= min_ap_count."
            )

        if self.frequency_hz <= 0:
            raise ValueError(
                "frequency_hz must be greater than zero."
            )

        if self.tx_power_dbm < 0:
            raise ValueError(
                "tx_power_dbm cannot be negative."
            )

        if self.reference_distance_m <= 0:
            raise ValueError(
                "reference_distance_m must be greater than zero."
            )

        if self.path_loss_exponent <= 0:
            raise ValueError(
                "path_loss_exponent must be greater than zero."
            )

        if not self.available_channels:
            raise ValueError(
                "available_channels cannot be empty."
            )

        if len(set(self.available_channels)) != len(
            self.available_channels
        ):
            raise ValueError(
                "available_channels must not contain duplicates."
            )

        if self.channel not in self.available_channels:
            raise ValueError(
                "Default channel must be present in available_channels."
            )


@dataclass
class Individual:
    """
    One candidate AP configuration.

    Each AP is represented by a position and a channel.

    The pairing is positional:

        positions[i] <-> channels[i]

    Objectives are minimization objectives.
    """

    positions: Tuple[Position, ...]

    channels: Tuple[int, ...] = field(default_factory=tuple)

    objectives: ObjectiveVector = field(default_factory=tuple)

    rank: int | None = None
    crowding_distance: float = 0.0

    def __post_init__(self) -> None:
        """
        Validate and initialize channel assignments.

        Existing callers that only provide positions receive
        the legacy default channel 1 for every AP.
        """

        if len(self.channels) == 0 and len(self.positions) > 0:
            self.channels = tuple(
                1 for _ in self.positions
            )

        if len(self.positions) != len(self.channels):
            raise ValueError(
                "positions and channels must have the same length."
            )

    @property
    def ap_count(self) -> int:
        """Return the number of APs in this candidate."""
        return len(self.positions)

    @property
    def ap_configuration(
        self,
    ) -> Tuple[Tuple[Position, int], ...]:
        """
        Return position/channel pairs.

        Example:

            (
                ((5.0, 5.0), 1),
                ((10.0, 10.0), 6),
            )
        """

        return tuple(
            zip(
                self.positions,
                self.channels,
            )
        )


@dataclass
class OptimizationResult:
    """Final result returned by the NSGA-II optimizer."""

    pareto_front: List[Individual]
    generations_completed: int
    history: List[dict] = field(default_factory=list)


def canonicalize_individual(
    individual: Individual,
) -> Individual:
    """
    Sort APs by position while preserving position/channel pairing.

    This prevents the optimizer from treating:

        position A -> channel 1
        position B -> channel 6

    and

        position B -> channel 6
        position A -> channel 1

    as different configurations.
    """

    paired = list(
        zip(
            individual.positions,
            individual.channels,
        )
    )

    paired.sort(
        key=lambda item: (
            item[0][0],
            item[0][1],
        )
    )

    positions = tuple(
        item[0]
        for item in paired
    )

    channels = tuple(
        item[1]
        for item in paired
    )

    return Individual(
        positions=positions,
        channels=channels,
        objectives=individual.objectives,
        rank=individual.rank,
        crowding_distance=individual.crowding_distance,
    )


def individual_key(
    individual: Individual,
) -> Tuple[Tuple[Position, int], ...]:
    """Return a hashable canonical representation."""

    canonical = canonicalize_individual(
        individual
    )

    return canonical.ap_configuration


def dominates(
    first: Individual,
    second: Individual,
) -> bool:
    """
    Determine whether `first` dominates `second`.

    All objectives are minimization objectives.
    """

    if not first.objectives:
        raise ValueError(
            "First individual has no objective values."
        )

    if not second.objectives:
        raise ValueError(
            "Second individual has no objective values."
        )

    if len(first.objectives) != len(
        second.objectives
    ):
        raise ValueError(
            "Individuals must have the same number of objectives."
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


def tournament_select(
    population: Sequence[Individual],
    rng: random.Random | None = None,
) -> Individual:
    """Select one individual using binary tournament selection."""

    if not population:
        raise ValueError(
            "Population cannot be empty."
        )

    if rng is None:
        rng = random.Random()

    if len(population) == 1:
        return population[0]

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


def crossover(
    parent_one: Individual,
    parent_two: Individual,
    valid_positions: Sequence[Position],
    available_channels: Sequence[int] = (1, 6, 11),
    rng: random.Random | None = None,
) -> Tuple[Individual, Individual]:
    """
    Crossover AP positions and their corresponding channels.

    Position/channel pairs are treated as one genetic unit.
    """

    if not valid_positions:
        raise ValueError(
            "Valid positions cannot be empty."
        )

    if not available_channels:
        raise ValueError(
            "Available channels cannot be empty."
        )

    if rng is None:
        rng = random.Random()

    valid_set = set(
        valid_positions
    )

    parent_one_pairs = [
        pair
        for pair in parent_one.ap_configuration
        if pair[0] in valid_set
    ]

    parent_two_pairs = [
        pair
        for pair in parent_two.ap_configuration
        if pair[0] in valid_set
    ]

    combined_pairs = (
        parent_one_pairs
        + parent_two_pairs
    )

    unique_pairs = []

    seen_positions = set()

    for position, channel in combined_pairs:

        if position in seen_positions:
            continue

        if channel not in available_channels:
            channel = rng.choice(
                list(available_channels)
            )

        seen_positions.add(position)

        unique_pairs.append(
            (position, channel)
        )

    rng.shuffle(
        unique_pairs
    )

    child_one_count = min(
        parent_one.ap_count,
        len(unique_pairs),
    )

    child_two_count = min(
        parent_two.ap_count,
        len(unique_pairs),
    )

    child_one_pairs = unique_pairs[
        :child_one_count
    ]

    child_two_pairs = unique_pairs[
        :child_two_count
    ]

    if not child_one_pairs:

        position = rng.choice(
            list(valid_positions)
        )

        channel = rng.choice(
            list(available_channels)
        )

        child_one_pairs = [
            (position, channel)
        ]

    if not child_two_pairs:

        position = rng.choice(
            list(valid_positions)
        )

        channel = rng.choice(
            list(available_channels)
        )

        child_two_pairs = [
            (position, channel)
        ]

    child_one = Individual(
        positions=tuple(
            pair[0]
            for pair in child_one_pairs
        ),
        channels=tuple(
            pair[1]
            for pair in child_one_pairs
        ),
    )

    child_two = Individual(
        positions=tuple(
            pair[0]
            for pair in child_two_pairs
        ),
        channels=tuple(
            pair[1]
            for pair in child_two_pairs
        ),
    )

    return (
        canonicalize_individual(
            child_one
        ),
        canonicalize_individual(
            child_two
        ),
    )


def mutate(
    individual: Individual,
    valid_positions: Sequence[Position],
    available_channels: Sequence[int] = (1, 6, 11),
    mutation_probability: float = 0.1,
    rng: random.Random | None = None,
) -> Individual:
    """
    Mutate both AP positions and channel assignments.
    """

    if not valid_positions:
        raise ValueError(
            "Valid positions cannot be empty."
        )

    if not available_channels:
        raise ValueError(
            "Available channels cannot be empty."
        )

    if not 0.0 <= mutation_probability <= 1.0:
        raise ValueError(
            "Mutation probability must be between 0 and 1."
        )

    if rng is None:
        rng = random.Random()

    positions = list(
        individual.positions
    )

    channels = list(
        individual.channels
    )

    if not positions:
        return Individual(
            positions=tuple(),
            channels=tuple(),
        )

    valid_positions_list = list(
        valid_positions
    )

    for index in range(
        len(positions)
    ):

        # Position mutation.
        if rng.random() <= mutation_probability:

            current_position = positions[
                index
            ]

            available_positions = [
                position
                for position in valid_positions_list
                if position not in positions
                or position == current_position
            ]

            if available_positions:

                positions[index] = rng.choice(
                    available_positions
                )

        # Channel mutation.
        if rng.random() <= mutation_probability:

            current_channel = channels[
                index
            ]

            possible_channels = [
                channel
                for channel in available_channels
                if channel != current_channel
            ]

            if possible_channels:

                channels[index] = rng.choice(
                    possible_channels
                )

    # Remove duplicate positions while preserving
    # their associated channels.
    unique_pairs = []

    seen_positions = set()

    for position, channel in zip(
        positions,
        channels,
    ):

        if position in seen_positions:
            continue

        seen_positions.add(
            position
        )

        unique_pairs.append(
            (position, channel)
        )

    mutated = Individual(
        positions=tuple(
            pair[0]
            for pair in unique_pairs
        ),
        channels=tuple(
            pair[1]
            for pair in unique_pairs
        ),
    )

    return canonicalize_individual(
        mutated
    )


def create_offspring(
    population: Sequence[Individual],
    valid_positions: Sequence[Position],
    offspring_count: int,
    available_channels: Sequence[int] = (1, 6, 11),
    crossover_probability: float = 0.9,
    mutation_probability: float = 0.1,
    rng: random.Random | None = None,
) -> List[Individual]:
    """
    Create a new offspring population.

    Pipeline:

        tournament selection
              ↓
          crossover
              ↓
           mutation
              ↓
      position/channel pair
    """

    if not population:
        raise ValueError(
            "Population cannot be empty."
        )

    if not valid_positions:
        raise ValueError(
            "Valid positions cannot be empty."
        )

    if not available_channels:
        raise ValueError(
            "Available channels cannot be empty."
        )

    if offspring_count <= 0:
        raise ValueError(
            "offspring_count must be greater than zero."
        )

    if not 0.0 <= crossover_probability <= 1.0:
        raise ValueError(
            "crossover_probability must be between 0 and 1."
        )

    if not 0.0 <= mutation_probability <= 1.0:
        raise ValueError(
            "mutation_probability must be between 0 and 1."
        )

    if rng is None:
        rng = random.Random()

    offspring: List[Individual] = []

    while len(offspring) < offspring_count:

        parent_one = tournament_select(
            population,
            rng=rng,
        )

        parent_two = tournament_select(
            population,
            rng=rng,
        )

        if (
            rng.random()
            <= crossover_probability
        ):

            child_one, child_two = crossover(
                parent_one=parent_one,
                parent_two=parent_two,
                valid_positions=valid_positions,
                available_channels=available_channels,
                rng=rng,
            )

        else:

            child_one = Individual(
                positions=parent_one.positions,
                channels=parent_one.channels,
            )

            child_two = Individual(
                positions=parent_two.positions,
                channels=parent_two.channels,
            )

        child_one = mutate(
            individual=child_one,
            valid_positions=valid_positions,
            available_channels=available_channels,
            mutation_probability=mutation_probability,
            rng=rng,
        )

        child_two = mutate(
            individual=child_two,
            valid_positions=valid_positions,
            available_channels=available_channels,
            mutation_probability=mutation_probability,
            rng=rng,
        )

        offspring.append(
            child_one
        )

        if len(offspring) < offspring_count:

            offspring.append(
                child_two
            )

    return offspring


def fast_non_dominated_sort(
    population: Sequence[Individual],
) -> List[List[Individual]]:
    """Perform fast non-dominated sorting."""

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

    fronts: List[List[Individual]] = [
        []
    ]

    for individual in population:

        for other in population:

            if individual is other:
                continue

            if dominates(
                individual,
                other,
            ):

                dominated_solutions[
                    id(individual)
                ].append(
                    other
                )

            elif dominates(
                other,
                individual,
            ):

                domination_count[
                    id(individual)
                ] += 1

        if (
            domination_count[
                id(individual)
            ]
            == 0
        ):

            individual.rank = 0
            fronts[0].append(
                individual
            )

    current_front_index = 0

    while current_front_index < len(fronts):

        next_front: List[Individual] = []

        for individual in fronts[
            current_front_index
        ]:

            for dominated in dominated_solutions[
                id(individual)
            ]:

                domination_count[
                    id(dominated)
                ] -= 1

                if (
                    domination_count[
                        id(dominated)
                    ]
                    == 0
                ):

                    dominated.rank = (
                        current_front_index
                        + 1
                    )

                    next_front.append(
                        dominated
                    )

        if next_front:

            fronts.append(
                next_front
            )

        current_front_index += 1

    return fronts


def calculate_crowding_distance(
    front: Sequence[Individual],
) -> None:
    """Calculate crowding distance for one Pareto front."""

    if not front:
        return

    for individual in front:
        individual.crowding_distance = 0.0

    if len(front) <= 2:

        for individual in front:
            individual.crowding_distance = float(
                "inf"
            )

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
        ].crowding_distance = float(
            "inf"
        )

        sorted_front[
            -1
        ].crowding_distance = float(
            "inf"
        )

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

        objective_range = (
            maximum - minimum
        )

        if objective_range == 0:
            continue

        for index in range(
            1,
            len(sorted_front) - 1,
        ):

            if (
                sorted_front[
                    index
                ].crowding_distance
                == float("inf")
            ):
                continue

            previous_value = (
                sorted_front[
                    index - 1
                ].objectives[
                    objective_index
                ]
            )

            next_value = (
                sorted_front[
                    index + 1
                ].objectives[
                    objective_index
                ]
            )

            sorted_front[
                index
            ].crowding_distance += (
                next_value
                - previous_value
            ) / objective_range


def initialize_population(
    valid_positions: Sequence[Position],
    config: NSGA2Config,
    rng: random.Random,
) -> List[Individual]:
    """Generate an initial population of channel-aware configurations."""

    if not valid_positions:
        raise ValueError(
            "valid_positions cannot be empty."
        )

    maximum_ap_count = min(
        config.max_ap_count,
        len(valid_positions),
    )

    if config.min_ap_count > maximum_ap_count:
        raise ValueError(
            "Not enough valid positions for min_ap_count."
        )

    population: List[Individual] = []

    valid_positions_list = list(
        valid_positions
    )

    for _ in range(
        config.population_size
    ):

        ap_count = rng.randint(
            config.min_ap_count,
            maximum_ap_count,
        )

        positions = tuple(
            rng.sample(
                valid_positions_list,
                ap_count,
            )
        )

        channels = tuple(
            rng.choice(
                config.available_channels
            )
            for _ in range(
                ap_count
            )
        )

        individual = Individual(
            positions=positions,
            channels=channels,
        )

        population.append(
            canonicalize_individual(
                individual
            )
        )

    return population


def create_access_points(
    individual: Individual,
    config: NSGA2Config,
) -> List[AccessPoint]:
    """Create RF AccessPoint objects with individual channels."""

    if len(individual.positions) != len(
        individual.channels
    ):
        raise ValueError(
            "Each AP position must have a corresponding channel."
        )

    return [
        AccessPoint(
            position=position,
            frequency_hz=config.frequency_hz,
            tx_power_dbm=config.tx_power_dbm,
            antenna_gain_dbi=config.antenna_gain_dbi,
            channel=channel,
        )
        for position, channel in zip(
            individual.positions,
            individual.channels,
        )
    ]


def evaluate_individual(
    individual: Individual,
    receiver_positions: Sequence[Position],
    valid_positions: Sequence[Position],
    blocked_positions: Sequence[Position],
    config: NSGA2Config,
    floor_plan_walls: Sequence[WallSegment] | None = None,
    floor_plan_wall_loss_db: float = 0.0,
) -> GridEvaluation:
    """
    Evaluate one channel-aware NSGA-II individual.

    Optional floor-plan walls provide additional physical
    attenuation based on detected wall crossings.
    """

    constraint_config = ConstraintConfig(
        min_ap_count=config.min_ap_count,
        max_ap_count=config.max_ap_count,
    )

    constraint_result = validate_configuration(
        positions=individual.positions,
        valid_positions=valid_positions,
        blocked_positions=blocked_positions,
        config=constraint_config,
    )

    if not constraint_result.valid:
        raise ValueError(
            "Invalid individual: "
            + "; ".join(
                constraint_result.errors
            )
        )

    for channel in individual.channels:

        if channel not in config.available_channels:

            raise ValueError(
                f"Invalid channel {channel}. "
                f"Available channels: "
                f"{config.available_channels}"
            )

    if floor_plan_wall_loss_db < 0:
        raise ValueError(
            "floor_plan_wall_loss_db cannot be negative."
        )

    access_points = create_access_points(
        individual,
        config,
    )

    evaluation = evaluate_grid(
        receiver_positions=receiver_positions,
        access_points=access_points,
        noise_floor_dbm=config.noise_floor_dbm,
        minimum_rssi_dbm=config.minimum_rssi_dbm,
        minimum_sinr_db=config.minimum_sinr_db,
        reference_distance_m=config.reference_distance_m,
        path_loss_exponent=config.path_loss_exponent,
        wall_material=config.wall_material,
        number_of_walls=config.number_of_walls,
        obstacle_loss_db=config.obstacle_loss_db,
        same_channel_factor=config.same_channel_factor,
        adjacent_channel_factor=config.adjacent_channel_factor,
        non_overlapping_factor=config.non_overlapping_factor,
        floor_plan_walls=floor_plan_walls,
        floor_plan_wall_loss_db=floor_plan_wall_loss_db,
    )

    objectives = calculate_objectives(
        evaluation=evaluation,
        ap_count=individual.ap_count,
    )

    individual.objectives = objectives.values

    return evaluation


def evaluate_population(
    population: Sequence[Individual],
    receiver_positions: Sequence[Position],
    valid_positions: Sequence[Position],
    blocked_positions: Sequence[Position],
    config: NSGA2Config,
    floor_plan_walls: Sequence[WallSegment] | None = None,
    floor_plan_wall_loss_db: float = 0.0,
) -> None:
    """Evaluate every individual in the population."""

    for individual in population:

        evaluate_individual(
            individual=individual,
            receiver_positions=receiver_positions,
            valid_positions=valid_positions,
            blocked_positions=blocked_positions,
            config=config,
            floor_plan_walls=floor_plan_walls,
            floor_plan_wall_loss_db=floor_plan_wall_loss_db,
        )


def select_next_generation(
    combined_population: Sequence[Individual],
    population_size: int,
) -> List[Individual]:
    """Perform NSGA-II elitist environmental selection."""

    if population_size <= 0:
        raise ValueError(
            "population_size must be greater than zero."
        )

    fronts = fast_non_dominated_sort(
        combined_population
    )

    selected: List[Individual] = []

    for front in fronts:

        calculate_crowding_distance(
            front
        )

        if (
            len(selected)
            + len(front)
            <= population_size
        ):

            selected.extend(
                front
            )

            continue

        remaining = (
            population_size
            - len(selected)
        )

        sorted_front = sorted(
            front,
            key=lambda individual:
            individual.crowding_distance,
            reverse=True,
        )

        selected.extend(
            sorted_front[
                :remaining
            ]
        )

        break

    return selected


def deduplicate_population(
    population: Sequence[Individual],
) -> List[Individual]:
    """
    Remove exact duplicate AP/channel configurations.

    This is primarily useful for reporting and for avoiding
    repeated equivalent solutions caused by AP ordering.
    """

    unique: List[Individual] = []
    seen = set()

    for individual in population:

        key = individual_key(
            individual
        )

        if key in seen:
            continue

        seen.add(key)

        unique.append(
            individual
        )

    return unique


def optimize(
    receiver_positions: Sequence[Position],
    valid_positions: Sequence[Position],
    blocked_positions: Sequence[Position] = (),
    config: NSGA2Config | None = None,
    floor_plan_walls: Sequence[WallSegment] | None = None,
    floor_plan_wall_loss_db: float = 0.0,
) -> OptimizationResult:
    """
    Run the complete channel-aware NSGA-II optimization.

    The search space includes:

        1. AP count
        2. AP positions
        3. AP channel assignments

    Optional floor-plan walls are propagated through every
    candidate evaluation so that optimization is performed
    using wall-aware RF propagation.
    """

    if config is None:
        config = NSGA2Config()

    config.validate()

    if not receiver_positions:
        raise ValueError(
            "receiver_positions cannot be empty."
        )

    if not valid_positions:
        raise ValueError(
            "valid_positions cannot be empty."
        )

    if floor_plan_wall_loss_db < 0:
        raise ValueError(
            "floor_plan_wall_loss_db cannot be negative."
        )

    rng = random.Random(
        config.random_seed
    )

    population = initialize_population(
        valid_positions=valid_positions,
        config=config,
        rng=rng,
    )

    evaluate_population(
        population=population,
        receiver_positions=receiver_positions,
        valid_positions=valid_positions,
        blocked_positions=blocked_positions,
        config=config,
        floor_plan_walls=floor_plan_walls,
        floor_plan_wall_loss_db=floor_plan_wall_loss_db,
    )

    history: List[dict] = []

    for generation in range(
        config.generations
    ):

        fronts = fast_non_dominated_sort(
            population
        )

        for front in fronts:

            calculate_crowding_distance(
                front
            )

        offspring = create_offspring(
            population=population,
            valid_positions=valid_positions,
            available_channels=config.available_channels,
            offspring_count=config.population_size,
            crossover_probability=config.crossover_probability,
            mutation_probability=config.mutation_probability,
            rng=rng,
        )

        evaluate_population(
            population=offspring,
            receiver_positions=receiver_positions,
            valid_positions=valid_positions,
            blocked_positions=blocked_positions,
            config=config,
            floor_plan_walls=floor_plan_walls,
            floor_plan_wall_loss_db=floor_plan_wall_loss_db,
        )

        combined_population = (
            list(population)
            + offspring
        )

        population = select_next_generation(
            combined_population=combined_population,
            population_size=config.population_size,
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
                "population_size": len(
                    population
                ),
                "pareto_size": len(
                    best_front
                ),
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

        calculate_crowding_distance(
            front
        )

    # Remove equivalent configurations for the final result.
    pareto_front = deduplicate_population(
        pareto_front
    )

    return OptimizationResult(
        pareto_front=pareto_front,
        generations_completed=config.generations,
        history=history,
    )
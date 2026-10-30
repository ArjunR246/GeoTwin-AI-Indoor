"""
GeoTwin AI Indoor
NSGA-II End-to-End Optimization Demo

Pipeline:

    Indoor Scenario
          ↓
    Candidate AP positions
          ↓
    NSGA-II
          ↓
    Multi-AP RF evaluation
          ↓
    Coverage / AP count / interference
          ↓
    Pareto-optimal solutions
"""

from collections import defaultdict

from geotwin.scenario.indoor import create_default_scenario
from geotwin.optimization.nsga2 import (
    NSGA2Config,
    optimize,
)


# ============================================================
# DISPLAY HELPERS
# ============================================================


def print_separator(title: str = "") -> None:
    print()
    print("=" * 70)

    if title:
        print(title)
        print("=" * 70)


# ============================================================
# CONFIGURATION DISPLAY
# ============================================================


def print_configuration(
    scenario,
    config: NSGA2Config,
) -> None:

    print_separator(
        "GeoTwin AI Indoor - NSGA-II Optimization Demo"
    )

    print("\n[SCENARIO]")

    print(
        f"Building width       : "
        f"{scenario.width_m:.1f} m"
    )

    print(
        f"Building length      : "
        f"{scenario.length_m:.1f} m"
    )

    print(
        f"Grid resolution      : "
        f"{scenario.grid_resolution_m:.1f} m"
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

    print("\n[GRID]")

    print(
        f"Total grid points    : "
        f"{scenario.total_grid_points}"
    )

    print(
        f"Valid AP positions   : "
        f"{len(scenario.valid_ap_positions)}"
    )

    print(
        f"Receiver positions   : "
        f"{len(scenario.receiver_positions)}"
    )

    print(
        f"Blocked positions    : "
        f"{len(scenario.blocked_positions)}"
    )

    print("\n[NSGA-II CONFIGURATION]")

    print(
        f"Population size      : "
        f"{config.population_size}"
    )

    print(
        f"Generations          : "
        f"{config.generations}"
    )

    print(
        f"Crossover probability: "
        f"{config.crossover_probability}"
    )

    print(
        f"Mutation probability : "
        f"{config.mutation_probability}"
    )

    print(
        f"AP count range       : "
        f"{config.min_ap_count} - "
        f"{config.max_ap_count}"
    )

    print(
        f"Random seed          : "
        f"{config.random_seed}"
    )

    print("\n[RF CONFIGURATION]")

    print(
        f"Frequency            : "
        f"{config.frequency_hz / 1e9:.1f} GHz"
    )

    print(
        f"Transmit power       : "
        f"{config.tx_power_dbm:.1f} dBm"
    )

    print(
        f"Antenna gain         : "
        f"{config.antenna_gain_dbi:.1f} dBi"
    )

    print(
        f"Channel              : "
        f"{config.channel}"
    )

    print(
        f"Minimum RSSI         : "
        f"{config.minimum_rssi_dbm:.1f} dBm"
    )

    print(
        f"Minimum SINR         : "
        f"{config.minimum_sinr_db:.1f} dB"
    )

    print(
        f"Noise floor          : "
        f"{config.noise_floor_dbm:.1f} dBm"
    )

    print(
        f"Reference distance   : "
        f"{config.reference_distance_m:.1f} m"
    )

    print(
        f"Path-loss exponent   : "
        f"{config.path_loss_exponent:.1f}"
    )

    print(
        f"Number of walls      : "
        f"{config.number_of_walls}"
    )

    print(
        f"Obstacle loss        : "
        f"{config.obstacle_loss_db:.1f} dB"
    )

    print(
        f"Wall material        : "
        f"{config.wall_material}"
    )


# ============================================================
# PARETO FRONT
# ============================================================


def print_pareto_front(
    pareto_front,
) -> None:

    print_separator(
        "PARETO FRONT"
    )

    if not pareto_front:
        print("No Pareto solutions found.")
        return

    for index, individual in enumerate(
        pareto_front,
        start=1,
    ):

        print(
            f"\nSolution {index}"
        )

        print(
            f"  AP count       : "
            f"{individual.ap_count}"
        )

        print(
            f"  Objectives     : "
            f"{individual.objectives}"
        )

        print(
            f"  AP positions   : "
            f"{individual.positions}"
        )

        print(
            f"  Pareto rank    : "
            f"{individual.rank}"
        )

        print(
            f"  Crowding       : "
            f"{individual.crowding_distance}"
        )


# ============================================================
# UNIQUE PARETO SUMMARY
# ============================================================


def print_pareto_summary(
    pareto_front,
) -> None:

    print_separator(
        "UNIQUE PARETO SUMMARY"
    )

    if not pareto_front:
        print("No Pareto solutions found.")
        return

    unique = {}

    for individual in pareto_front:

        key = (
            individual.ap_count,
            tuple(individual.positions),
            tuple(individual.objectives),
        )

        unique[key] = individual

    unique_solutions = list(
        unique.values()
    )

    unique_solutions.sort(
        key=lambda individual: (
            individual.ap_count,
            individual.objectives,
        )
    )

    print(
        f"Unique Pareto configurations : "
        f"{len(unique_solutions)}"
    )

    print()

    for index, individual in enumerate(
        unique_solutions,
        start=1,
    ):

        coverage = -individual.objectives[0]

        ap_count = individual.objectives[1]

        interference = individual.objectives[2]

        print(
            f"Solution {index}"
        )

        print(
            f"  Coverage         : "
            f"{coverage:.2f}%"
        )

        print(
            f"  AP count         : "
            f"{int(ap_count)}"
        )

        print(
            f"  Interference     : "
            f"{interference:.8e} mW"
        )

        print(
            f"  AP positions     : "
            f"{individual.positions}"
        )

        print()


# ============================================================
# GROUP BY AP COUNT
# ============================================================


def print_ap_count_summary(
    pareto_front,
) -> None:

    print_separator(
        "PARETO SOLUTIONS BY AP COUNT"
    )

    if not pareto_front:
        return

    grouped = defaultdict(list)

    for individual in pareto_front:
        grouped[
            individual.ap_count
        ].append(individual)

    for ap_count in sorted(
        grouped
    ):

        solutions = grouped[
            ap_count
        ]

        best_coverage = max(
            -solution.objectives[0]
            for solution in solutions
        )

        minimum_interference = min(
            solution.objectives[2]
            for solution in solutions
        )

        print(
            f"AP count {ap_count}: "
            f"{len(solutions)} solutions"
        )

        print(
            f"  Best coverage       : "
            f"{best_coverage:.2f}%"
        )

        print(
            f"  Minimum interference: "
            f"{minimum_interference:.8e} mW"
        )


# ============================================================
# OBJECTIVE RANGE
# ============================================================


def print_objective_ranges(
    pareto_front,
) -> None:

    print_separator(
        "PARETO OBJECTIVE RANGES"
    )

    if not pareto_front:
        return

    coverage_values = [
        -individual.objectives[0]
        for individual in pareto_front
    ]

    ap_counts = [
        individual.objectives[1]
        for individual in pareto_front
    ]

    interference_values = [
        individual.objectives[2]
        for individual in pareto_front
    ]

    print(
        f"Coverage range       : "
        f"{min(coverage_values):.2f}% "
        f"to "
        f"{max(coverage_values):.2f}%"
    )

    print(
        f"AP count range       : "
        f"{int(min(ap_counts))} "
        f"to "
        f"{int(max(ap_counts))}"
    )

    print(
        f"Interference range   : "
        f"{min(interference_values):.8e} "
        f"to "
        f"{max(interference_values):.8e} mW"
    )


# ============================================================
# OPTIMIZATION HISTORY
# ============================================================


def print_history(
    history,
) -> None:

    print_separator(
        "OPTIMIZATION HISTORY"
    )

    for entry in history:

        print(
            f"Generation "
            f"{entry['generation']:3d}: "
            f"population="
            f"{entry['population_size']:3d}, "
            f"pareto="
            f"{entry['pareto_size']:3d}"
        )


# ============================================================
# MAIN
# ============================================================


def main() -> None:

    # --------------------------------------------------------
    # Create scenario
    # --------------------------------------------------------

    scenario = create_default_scenario()

    # --------------------------------------------------------
    # Configure NSGA-II
    # --------------------------------------------------------

    config = NSGA2Config(

        population_size=50,

        generations=50,

        crossover_probability=0.9,

        mutation_probability=0.1,

        min_ap_count=1,

        max_ap_count=5,

        random_seed=42,

        # ----------------------------------------------------
        # RF
        # ----------------------------------------------------

        frequency_hz=2.4e9,

        tx_power_dbm=20.0,

        antenna_gain_dbi=3.0,

        channel=1,

        # ----------------------------------------------------
        # Coverage
        # ----------------------------------------------------

        minimum_rssi_dbm=-67.0,

        minimum_sinr_db=20.0,

        noise_floor_dbm=-95.0,

        # ----------------------------------------------------
        # Propagation
        # ----------------------------------------------------

        reference_distance_m=1.0,

        path_loss_exponent=3.0,

        # ----------------------------------------------------
        # Synthetic indoor attenuation
        # ----------------------------------------------------

        wall_material="concrete",

        number_of_walls=4,

        obstacle_loss_db=6.0,

        # ----------------------------------------------------
        # Channel overlap
        # ----------------------------------------------------

        same_channel_factor=1.0,

        adjacent_channel_factor=0.5,

        non_overlapping_factor=0.0,
    )

    # --------------------------------------------------------
    # Display configuration
    # --------------------------------------------------------

    print_configuration(
        scenario,
        config,
    )

    # --------------------------------------------------------
    # Start optimization
    # --------------------------------------------------------

    print_separator(
        "Starting NSGA-II optimization..."
    )

    result = optimize(
        receiver_positions=(
            scenario.receiver_positions
        ),
        valid_positions=(
            scenario.valid_ap_positions
        ),
        blocked_positions=(
            scenario.blocked_positions
        ),
        config=config,
    )

    # --------------------------------------------------------
    # Optimization complete
    # --------------------------------------------------------

    print_separator(
        "Optimization complete"
    )

    print(
        f"Generations completed : "
        f"{result.generations_completed}"
    )

    print(
        f"Pareto solutions      : "
        f"{len(result.pareto_front)}"
    )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print_pareto_front(
        result.pareto_front
    )

    print_pareto_summary(
        result.pareto_front
    )

    print_ap_count_summary(
        result.pareto_front
    )

    print_objective_ranges(
        result.pareto_front
    )

    print_history(
        result.history
    )

    # --------------------------------------------------------
    # Finished
    # --------------------------------------------------------

    print_separator(
        "GeoTwin AI Indoor NSGA-II demo finished."
    )


if __name__ == "__main__":
    main()
"""
GeoTwin AI Indoor - Production Channel-Aware NSGA-II Runner.

Runs the production NSGA-II optimizer using the same RF and
optimization configuration as the validated channel-aware experiment.
"""

from __future__ import annotations

from geotwin.scenario.indoor import IndoorScenario
from geotwin.optimization.nsga2 import (
    NSGA2Config,
    optimize,
    evaluate_individual,
)


def main() -> None:
    print("=" * 100)
    print("GeoTwin AI Indoor - Production Channel-Aware NSGA-II")
    print("=" * 100)

    # ==================================================================
    # SCENARIO
    # ==================================================================

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
    print(f"Building width       : {scenario.width_m} m")
    print(f"Building length      : {scenario.length_m} m")
    print(f"Grid resolution      : {scenario.grid_resolution_m} m")
    print(f"Floors               : {scenario.floors}")
    print(f"Wall material        : {scenario.wall_material}")
    print(f"Wall thickness       : {scenario.wall_thickness_m:.2f} m")

    print()
    print("[GRID]")
    print(f"Total grid points    : {scenario.total_grid_points}")
    print(f"Valid AP positions   : {len(valid_positions)}")
    print(f"Receiver positions   : {len(receiver_positions)}")
    print(f"Blocked positions    : {len(blocked_positions)}")

    # ==================================================================
    # PRODUCTION NSGA-II CONFIGURATION
    # ==================================================================

    config = NSGA2Config(
        population_size=50,
        generations=50,
        crossover_probability=0.9,
        mutation_probability=0.1,

        min_ap_count=1,
        max_ap_count=5,

        random_seed=42,

        minimum_rssi_dbm=-67.0,
        minimum_sinr_db=20.0,
        noise_floor_dbm=-95.0,

        frequency_hz=2.4e9,
        tx_power_dbm=20.0,
        antenna_gain_dbi=3.0,

        reference_distance_m=1.0,
        path_loss_exponent=3.0,

        wall_material="concrete",
        number_of_walls=4,
        obstacle_loss_db=6.0,

        available_channels=(1, 6, 11),

        same_channel_factor=1.0,
        adjacent_channel_factor=0.5,
        non_overlapping_factor=0.0,
    )

    print()
    print("[PRODUCTION NSGA-II CONFIGURATION]")
    print(f"Population size      : {config.population_size}")
    print(f"Generations          : {config.generations}")
    print(f"Crossover probability: {config.crossover_probability}")
    print(f"Mutation probability : {config.mutation_probability}")
    print(f"AP count range       : {config.min_ap_count} - {config.max_ap_count}")
    print(f"Available channels   : {config.available_channels}")
    print(f"Random seed          : {config.random_seed}")

    print()
    print("=" * 100)
    print("STARTING PRODUCTION CHANNEL-AWARE NSGA-II")
    print("=" * 100)

    # ==================================================================
    # PRODUCTION OPTIMIZATION
    # ==================================================================

    result = optimize(
        valid_positions=valid_positions,
        receiver_positions=receiver_positions,
        blocked_positions=blocked_positions,
        config=config,
    )

    print()
    print("=" * 100)
    print("PRODUCTION OPTIMIZATION COMPLETE")
    print("=" * 100)

    print(f"Generations completed : {result.generations_completed}")
    print(f"Pareto solutions      : {len(result.pareto_front)}")

    # ==================================================================
    # EVALUATE PARETO SOLUTIONS
    # ==================================================================

    evaluated_solutions = []

    for individual in result.pareto_front:

        evaluation = evaluate_individual(
            individual=individual,
            receiver_positions=receiver_positions,
            valid_positions=valid_positions,
            blocked_positions=blocked_positions,
            config=config,
        )

        evaluated_solutions.append(
            (individual, evaluation)
        )

    print()
    print("=" * 100)
    print("PRODUCTION PARETO SOLUTIONS")
    print("=" * 100)

    for index, (individual, evaluation) in enumerate(
        evaluated_solutions,
        start=1,
    ):
        print()
        print(f"Solution {index}")
        print(f"  AP count       : {individual.ap_count}")
        print(f"  Positions      : {individual.positions}")
        print(f"  Channels       : {individual.channels}")
        print(f"  Coverage       : {evaluation.coverage_percent:.2f}%")
        print(f"  Average RSSI   : {evaluation.average_rssi_dbm:.2f} dBm")
        print(f"  Minimum RSSI   : {evaluation.minimum_rssi_dbm:.2f} dBm")
        print(f"  Average SINR   : {evaluation.average_sinr_db:.2f} dB")
        print(f"  Minimum SINR   : {evaluation.minimum_sinr_db:.2f} dB")

    # ==================================================================
    # BEST COVERAGE SOLUTION
    # ==================================================================

    best_individual, best_evaluation = max(
        evaluated_solutions,
        key=lambda item: item[1].coverage_percent,
    )

    print()
    print("=" * 100)
    print("BEST PRODUCTION SOLUTION")
    print("=" * 100)

    print(f"AP count       : {best_individual.ap_count}")
    print(f"Coverage       : {best_evaluation.coverage_percent:.2f}%")
    print(f"Average RSSI   : {best_evaluation.average_rssi_dbm:.2f} dBm")
    print(f"Minimum RSSI   : {best_evaluation.minimum_rssi_dbm:.2f} dBm")
    print(f"Average SINR   : {best_evaluation.average_sinr_db:.2f} dB")
    print(f"Minimum SINR   : {best_evaluation.minimum_sinr_db:.2f} dB")
    print(f"Channels       : {best_individual.channels}")
    print(f"Positions      : {best_individual.positions}")

    # ==================================================================
    # EXPERIMENTAL REFERENCE
    # ==================================================================

    print()
    print("=" * 100)
    print("PRODUCTION VALIDATION REFERENCE")
    print("=" * 100)

    print("Validated channel-aware experiment:")
    print("  Coverage       : 82.38%")
    print("  Average SINR   : 34.05 dB")
    print("  AP count       : 3")
    print("  Channels       : (6, 11, 1)")
    print(
        "  Positions      : "
        "((13.5, 14.0), (5.5, 9.0), (13.0, 4.0))"
    )

    print()
    print("Production result:")
    print(f"  Coverage       : {best_evaluation.coverage_percent:.2f}%")
    print(f"  Average SINR   : {best_evaluation.average_sinr_db:.2f} dB")
    print(f"  AP count       : {best_individual.ap_count}")
    print(f"  Channels       : {best_individual.channels}")
    print(f"  Positions      : {best_individual.positions}")

    # ==================================================================
    # VALIDATION INTERPRETATION
    # ==================================================================

    coverage_difference = (
        best_evaluation.coverage_percent - 82.38
    )

    sinr_difference = (
        best_evaluation.average_sinr_db - 34.05
    )

    print()
    print("=" * 100)
    print("VALIDATION SUMMARY")
    print("=" * 100)

    print(
        f"Coverage difference vs experiment : "
        f"{coverage_difference:+.2f} percentage points"
    )

    print(
        f"SINR difference vs experiment     : "
        f"{sinr_difference:+.2f} dB"
    )

    if best_individual.ap_count == 3:
        print("AP count validation               : PASS")
    else:
        print("AP count validation               : DIFFERENT PARETO CHOICE")

    if best_individual.channels:
        print("Channel assignment                : PRESENT")
    else:
        print("Channel assignment                : MISSING")

    print()
    print("=" * 100)
    print("Production channel-aware validation complete.")
    print("=" * 100)


if __name__ == "__main__":
    main()
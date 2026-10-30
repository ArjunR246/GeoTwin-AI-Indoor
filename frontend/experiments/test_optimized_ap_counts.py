"""
GeoTwin AI Indoor
Optimized AP Count Experiment

Runs NSGA-II separately for exactly:
    1 AP
    2 APs
    3 APs
    4 APs
    5 APs

This allows comparison of the best placement achievable for
each AP count.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from geotwin.optimization.nsga2 import (
    NSGA2Config,
    optimize,
)
from geotwin.scenario.indoor import IndoorScenario


Position = Tuple[float, float]


@dataclass(frozen=True)
class APCountResult:
    ap_count: int
    coverage: float
    dead_zone: float
    average_rssi: float
    minimum_rssi: float
    average_sinr: float
    minimum_sinr: float
    interference: float
    objectives: Tuple[float, ...]
    positions: Tuple[Position, ...]


def run_for_ap_count(
    scenario: IndoorScenario,
    ap_count: int,
) -> APCountResult:

    config = NSGA2Config(
        population_size=30,
        generations=30,

        crossover_probability=0.9,
        mutation_probability=0.1,

        min_ap_count=ap_count,
        max_ap_count=ap_count,

        random_seed=42,

        minimum_rssi_dbm=-67.0,
        minimum_sinr_db=20.0,
        noise_floor_dbm=-95.0,

        frequency_hz=2.4e9,
        tx_power_dbm=20.0,
        antenna_gain_dbi=3.0,
        channel=1,

        reference_distance_m=1.0,
        path_loss_exponent=3.0,

        wall_material="concrete",
        number_of_walls=4,
        obstacle_loss_db=6.0,

        same_channel_factor=1.0,
        adjacent_channel_factor=0.5,
        non_overlapping_factor=0.0,
    )

    result = optimize(
        receiver_positions=scenario.receiver_positions,
        valid_positions=scenario.valid_ap_positions,
        blocked_positions=scenario.blocked_positions,
        config=config,
    )

    if not result.pareto_front:
        raise RuntimeError(
            f"No Pareto solution found for {ap_count} AP(s)."
        )

    # Since AP count is fixed, select the solution with:
    # 1. maximum coverage
    # 2. then lower interference
    best = min(
        result.pareto_front,
        key=lambda individual: (
            individual.objectives[0],
            individual.objectives[2],
        ),
    )

    # Re-evaluate the selected individual to obtain RF metrics.
    from geotwin.optimization.nsga2 import evaluate_individual

    evaluation = evaluate_individual(
        individual=best,
        receiver_positions=scenario.receiver_positions,
        valid_positions=scenario.valid_ap_positions,
        blocked_positions=scenario.blocked_positions,
        config=config,
    )

    return APCountResult(
        ap_count=ap_count,
        coverage=evaluation.coverage_percent,
        dead_zone=evaluation.dead_zone_percent,
        average_rssi=evaluation.average_rssi_dbm,
        minimum_rssi=evaluation.minimum_rssi_dbm,
        average_sinr=evaluation.average_sinr_db,
        minimum_sinr=evaluation.minimum_sinr_db,
        interference=evaluation.average_interference_dbm,
        objectives=best.objectives,
        positions=best.positions,
    )


def format_interference(
    value: float,
) -> str:

    if value == float("-inf"):
        return "-inf"

    return f"{value:.2f} dBm"


def print_result(
    result: APCountResult,
) -> None:

    print()
    print("-" * 90)

    print(
        f"AP COUNT: {result.ap_count}"
    )

    print(
        f"Coverage              : "
        f"{result.coverage:.2f}%"
    )

    print(
        f"Dead zone             : "
        f"{result.dead_zone:.2f}%"
    )

    print(
        f"Average RSSI          : "
        f"{result.average_rssi:.2f} dBm"
    )

    print(
        f"Minimum RSSI          : "
        f"{result.minimum_rssi:.2f} dBm"
    )

    print(
        f"Average SINR          : "
        f"{result.average_sinr:.2f} dB"
    )

    print(
        f"Minimum SINR          : "
        f"{result.minimum_sinr:.2f} dB"
    )

    print(
        f"Average interference  : "
        f"{format_interference(result.interference)}"
    )

    print(
        f"Objectives            : "
        f"{result.objectives}"
    )

    print(
        f"AP positions          : "
        f"{result.positions}"
    )


def print_summary(
    results: list[APCountResult],
) -> None:

    print()
    print("=" * 120)
    print("OPTIMIZED AP COUNT SUMMARY")
    print("=" * 120)

    print()

    header = (
        f"{'APs':>4} | "
        f"{'Coverage':>10} | "
        f"{'Dead Zone':>10} | "
        f"{'Avg RSSI':>12} | "
        f"{'Min RSSI':>12} | "
        f"{'Avg SINR':>12} | "
        f"{'Min SINR':>12} | "
        f"{'Interference':>15}"
    )

    print(header)
    print("-" * len(header))

    for result in results:

        print(
            f"{result.ap_count:>4} | "
            f"{result.coverage:>9.2f}% | "
            f"{result.dead_zone:>9.2f}% | "
            f"{result.average_rssi:>10.2f} dBm | "
            f"{result.minimum_rssi:>10.2f} dBm | "
            f"{result.average_sinr:>10.2f} dB | "
            f"{result.minimum_sinr:>10.2f} dB | "
            f"{format_interference(result.interference):>15}"
        )


def print_best_configuration(
    results: list[APCountResult],
) -> None:

    if not results:
        return

    best_coverage = max(
        results,
        key=lambda result: result.coverage,
    )

    best_sinr = max(
        results,
        key=lambda result: result.average_sinr,
    )

    print()
    print("=" * 100)
    print("BEST CONFIGURATIONS")
    print("=" * 100)

    print()

    print(
        f"Highest coverage:"
    )
    print(
        f"  AP count : {best_coverage.ap_count}"
    )
    print(
        f"  Coverage : {best_coverage.coverage:.2f}%"
    )
    print(
        f"  Positions: {best_coverage.positions}"
    )

    print()

    print(
        f"Highest average SINR:"
    )
    print(
        f"  AP count : {best_sinr.ap_count}"
    )
    print(
        f"  SINR     : {best_sinr.average_sinr:.2f} dB"
    )
    print(
        f"  Positions: {best_sinr.positions}"
    )


def main() -> None:

    print()
    print("=" * 100)
    print("GeoTwin AI Indoor - Optimized AP Count Experiment")
    print("=" * 100)

    scenario = IndoorScenario(
        width_m=20.0,
        length_m=15.0,
        grid_resolution_m=0.5,
        floors=1,
        wall_material="concrete",
        wall_thickness_m=0.2,
    )

    scenario.validate()

    print()
    print("[SCENARIO]")
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
        f"Grid points          : "
        f"{scenario.total_grid_points}"
    )

    print()
    print(
        "Each AP count will be optimized independently."
    )

    results: list[APCountResult] = []

    for ap_count in range(1, 6):

        print()
        print("=" * 100)
        print(
            f"Optimizing exactly {ap_count} "
            f"AP{'s' if ap_count != 1 else ''}..."
        )
        print("=" * 100)

        result = run_for_ap_count(
            scenario=scenario,
            ap_count=ap_count,
        )

        results.append(result)

        print_result(result)

    print_summary(results)

    print_best_configuration(results)

    print()
    print("=" * 100)
    print("Optimized AP Count Experiment Complete")
    print("=" * 100)


if __name__ == "__main__":
    main()
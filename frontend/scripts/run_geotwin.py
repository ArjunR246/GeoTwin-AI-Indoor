"""
GeoTwin AI Indoor - Complete Production Pipeline
"""

from pathlib import Path

from geotwin.optimization.nsga2 import NSGA2Config
from geotwin.pipeline import run_geotwin
from geotwin.scenario.indoor import IndoorScenario


def main() -> None:
    print("=" * 80)
    print("GeoTwin AI Indoor - Complete Production Pipeline")
    print("=" * 80)

    scenario = IndoorScenario(
        width_m=20.0,
        length_m=15.0,
        grid_resolution_m=0.5,
        floors=1,
        wall_material="concrete",
        wall_thickness_m=0.2,
    )

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
        available_channels=(1, 6, 11),
        reference_distance_m=1.0,
        path_loss_exponent=3.0,
        wall_material="concrete",
        number_of_walls=4,
        obstacle_loss_db=6.0,
        same_channel_factor=1.0,
        adjacent_channel_factor=0.5,
        non_overlapping_factor=0.0,
    )

    print()
    print("[SCENARIO]")
    print(f"Building          : {scenario.width_m} m x {scenario.length_m} m")
    print(f"Grid resolution   : {scenario.grid_resolution_m} m")
    print(f"Grid points       : {scenario.total_grid_points}")
    print(f"Floors            : {scenario.floors}")

    print()
    print("[OPTIMIZATION]")
    print(f"Population        : {config.population_size}")
    print(f"Generations       : {config.generations}")
    print(f"AP count          : {config.min_ap_count} - {config.max_ap_count}")
    print(f"Channels          : {config.available_channels}")
    print(f"Random seed       : {config.random_seed}")

    print()
    print("=" * 80)
    print("RUNNING GEOTWIN PIPELINE")
    print("=" * 80)

    result = run_geotwin(
        scenario=scenario,
        config=config,
    )

    best = result.best_solution
    evaluation = result.rf_evaluation

    print()
    print("=" * 80)
    print("GEOTWIN OPTIMIZATION COMPLETE")
    print("=" * 80)

    print()
    print("[BEST SOLUTION]")
    print(f"AP count          : {best.ap_count}")
    print(f"Positions         : {best.positions}")
    print(f"Channels          : {best.channels}")

    print()
    print("[RF PERFORMANCE]")
    print(f"Coverage          : {evaluation.coverage_percent:.2f}%")
    print(f"Dead zone         : {evaluation.dead_zone_percent:.2f}%")
    print(f"Average RSSI      : {evaluation.average_rssi_dbm:.2f} dBm")
    print(f"Minimum RSSI      : {evaluation.minimum_rssi_dbm:.2f} dBm")
    print(f"Average SINR      : {evaluation.average_sinr_db:.2f} dB")
    print(f"Minimum SINR      : {evaluation.minimum_sinr_db:.2f} dB")

    print()
    print("[PARETO FRONT]")
    print(f"Solutions         : {len(result.optimization.pareto_front)}")

    for index, individual in enumerate(
        result.optimization.pareto_front,
        start=1,
    ):
        print(
            f"  {index}. "
            f"{individual.ap_count} AP | "
            f"Coverage={-individual.objectives[0]:.2f}% | "
            f"Channels={individual.channels}"
        )

    outputs_dir = Path("outputs")

    print()
    print("[OUTPUTS]")
    print(f"Directory         : {outputs_dir.resolve()}")

    existing_outputs = sorted(outputs_dir.glob("*.png"))

    if existing_outputs:
        for output in existing_outputs:
            print(f"  ✓ {output.name}")
    else:
        print("  No visualization files found.")

    print()
    print("=" * 80)
    print("GEOTWIN AI INDOOR PIPELINE COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
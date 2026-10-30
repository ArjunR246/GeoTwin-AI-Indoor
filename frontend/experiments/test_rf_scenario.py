"""
GeoTwin AI Indoor
Multi-AP RF Diagnostic

Purpose:
    Validate the RF evaluation pipeline using two access points.

This diagnostic checks:

    1. Multi-AP serving AP selection
    2. Interference calculation
    3. SINR degradation due to interference
    4. Coverage percentage
    5. RSSI statistics
    6. Interference statistics
    7. SINR statistics

Scenario:
    Building      : 20 m × 15 m
    Grid          : 0.5 m
    Frequency     : 2.4 GHz
    AP count      : 2
    Channel       : 1 for both APs

Using the same channel intentionally creates
co-channel interference.
"""

from math import isinf, isnan

from geotwin.scenario.indoor import (
    create_default_scenario,
)

from geotwin.optimization.nsga2 import (
    NSGA2Config,
    Individual,
    evaluate_individual,
)


def print_separator(title: str = "") -> None:
    """Print a formatted section separator."""

    print("\n" + "=" * 70)

    if title:
        print(title)
        print("=" * 70)


def main() -> None:
    """Run the multi-AP RF diagnostic."""

    # =========================================================
    # 1. Create indoor scenario
    # =========================================================

    scenario = create_default_scenario()

    print_separator(
        "GeoTwin AI Indoor - Multi-AP RF Diagnostic"
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

    # =========================================================
    # 2. Get grid
    # =========================================================

    valid_ap_positions = (
        scenario.valid_ap_positions
    )

    receiver_positions = (
        scenario.receiver_positions
    )

    blocked_positions = (
        scenario.blocked_positions
    )

    print("\n[GRID]")

    print(
        f"Total grid points    : "
        f"{scenario.total_grid_points}"
    )

    print(
        f"Valid AP positions   : "
        f"{len(valid_ap_positions)}"
    )

    print(
        f"Receiver positions   : "
        f"{len(receiver_positions)}"
    )

    print(
        f"Blocked positions    : "
        f"{len(blocked_positions)}"
    )

    # =========================================================
    # 3. RF configuration
    # =========================================================

    config = NSGA2Config(
        min_ap_count=2,
        max_ap_count=2,

        # NSGA-II values are irrelevant for this diagnostic,
        # but they must remain valid.
        population_size=10,
        generations=1,

        # -----------------------------------------------------
        # RF configuration
        # -----------------------------------------------------

        frequency_hz=2.4e9,
        tx_power_dbm=20.0,
        antenna_gain_dbi=3.0,

        # Both APs use channel 1 intentionally.
        # This allows us to observe co-channel interference.
        channel=1,

        # -----------------------------------------------------
        # Coverage thresholds
        # -----------------------------------------------------

        minimum_rssi_dbm=-67.0,
        minimum_sinr_db=20.0,
        noise_floor_dbm=-95.0,

        # -----------------------------------------------------
        # Propagation
        # -----------------------------------------------------

        reference_distance_m=1.0,
        path_loss_exponent=3.0,

        # -----------------------------------------------------
        # Current synthetic scenario
        # -----------------------------------------------------

        wall_material="concrete",
        number_of_walls=0,
        obstacle_loss_db=0.0,

        # -----------------------------------------------------
        # Channel overlap
        # -----------------------------------------------------

        same_channel_factor=1.0,
        adjacent_channel_factor=0.5,
        non_overlapping_factor=0.0,
    )

    # =========================================================
    # 4. Define TWO APs
    # =========================================================
    #
    # AP 1 is placed near the bottom-left.
    #
    # AP 2 is placed near the top-right.
    #
    # Because both use channel 1, they can interfere with
    # each other where their signals overlap.
    # =========================================================

    ap_positions = (
        (2.0, 2.0),
        (18.0, 13.0),
    )

    individual = Individual(
        positions=ap_positions,
    )

    print("\n[ACCESS POINTS]")

    for index, position in enumerate(
        individual.positions
    ):
        print(
            f"AP {index + 1}              : "
            f"{position}"
        )

    print(
        f"AP count             : "
        f"{individual.ap_count}"
    )

    print(
        f"Channel               : "
        f"{config.channel}"
    )

    # =========================================================
    # 5. Validate AP positions
    # =========================================================

    print("\n[AP POSITION VALIDATION]")

    for index, position in enumerate(
        individual.positions
    ):

        if position in valid_ap_positions:

            print(
                f"AP {index + 1}              : "
                f"VALID"
            )

        else:

            print(
                f"AP {index + 1}              : "
                f"INVALID"
            )

            raise ValueError(
                f"AP position {position} "
                "is not a valid AP position."
            )

    # =========================================================
    # 6. Run RF evaluation
    # =========================================================

    print_separator(
        "Running Multi-AP RF Evaluation..."
    )

    evaluation = evaluate_individual(
        individual=individual,

        receiver_positions=receiver_positions,

        valid_positions=valid_ap_positions,

        blocked_positions=blocked_positions,

        config=config,
    )

    print("RF evaluation completed.")

    # =========================================================
    # 7. Basic evaluation summary
    # =========================================================

    print_separator(
        "RF EVALUATION SUMMARY"
    )

    print(
        f"Total receiver points : "
        f"{evaluation.total_points}"
    )

    print(
        f"Covered points        : "
        f"{evaluation.covered_points}"
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

    # =========================================================
    # 8. Objective values
    # =========================================================

    print_separator(
        "NSGA-II OBJECTIVES"
    )

    print(
        f"Objectives             : "
        f"{individual.objectives}"
    )

    print(
        "Objective interpretation:"
    )

    print(
        "  Objective 1 = negative coverage"
    )

    print(
        "  Objective 2 = AP count"
    )

    print(
        "  Objective 3 = interference metric"
    )

    # =========================================================
    # 9. Analyze receiver-level results
    # =========================================================

    receiver_results = (
        evaluation.receiver_results
    )

    print_separator(
        "RECEIVER-LEVEL RF ANALYSIS"
    )

    print(
        f"Receiver evaluations  : "
        f"{len(receiver_results)}"
    )

    # ---------------------------------------------------------
    # Serving AP distribution
    # ---------------------------------------------------------

    serving_ap_counts = {}

    for result in receiver_results:

        index = result.serving_ap_index

        serving_ap_counts[index] = (
            serving_ap_counts.get(index, 0)
            + 1
        )

    print("\n[SERVING AP DISTRIBUTION]")

    for index in sorted(
        serving_ap_counts
    ):

        count = serving_ap_counts[index]

        percentage = (
            count
            / len(receiver_results)
            * 100.0
        )

        print(
            f"AP {index + 1}: "
            f"{count} receivers "
            f"({percentage:.2f}%)"
        )

    # =========================================================
    # 10. Interference analysis
    # =========================================================

    interference_values = []

    no_interference_count = 0

    finite_interference_count = 0

    nan_interference_count = 0

    for result in receiver_results:

        value = result.interference_dbm

        if isnan(value):

            nan_interference_count += 1

        elif isinf(value):

            no_interference_count += 1

        else:

            finite_interference_count += 1

            interference_values.append(
                value
            )

    print("\n[INTERFERENCE ANALYSIS]")

    print(
        f"Receivers with finite "
        f"interference : "
        f"{finite_interference_count}"
    )

    print(
        f"Receivers with no "
        f"interference : "
        f"{no_interference_count}"
    )

    print(
        f"Receivers with NaN "
        f"interference : "
        f"{nan_interference_count}"
    )

    if interference_values:

        average_interference = (
            sum(interference_values)
            / len(interference_values)
        )

        minimum_interference = min(
            interference_values
        )

        maximum_interference = max(
            interference_values
        )

        print(
            f"Average interference : "
            f"{average_interference:.2f} dBm"
        )

        print(
            f"Minimum interference : "
            f"{minimum_interference:.2f} dBm"
        )

        print(
            f"Maximum interference : "
            f"{maximum_interference:.2f} dBm"
        )

    else:

        print(
            "No finite interference "
            "values were produced."
        )

    # =========================================================
    # 11. SINR analysis
    # =========================================================

    sinr_values = []

    nan_sinr_count = 0

    infinite_sinr_count = 0

    for result in receiver_results:

        value = result.sinr_db

        if isnan(value):

            nan_sinr_count += 1

        elif isinf(value):

            infinite_sinr_count += 1

        else:

            sinr_values.append(value)

    print("\n[SINR ANALYSIS]")

    print(
        f"Finite SINR values   : "
        f"{len(sinr_values)}"
    )

    print(
        f"NaN SINR values      : "
        f"{nan_sinr_count}"
    )

    print(
        f"Infinite SINR values : "
        f"{infinite_sinr_count}"
    )

    if sinr_values:

        print(
            f"Average SINR         : "
            f"{sum(sinr_values) / len(sinr_values):.2f} dB"
        )

        print(
            f"Minimum SINR         : "
            f"{min(sinr_values):.2f} dB"
        )

        print(
            f"Maximum SINR         : "
            f"{max(sinr_values):.2f} dB"
        )

    # =========================================================
    # 12. Print selected receiver samples
    # =========================================================

    print_separator(
        "SAMPLE RECEIVER RESULTS"
    )

    sample_indices = [
        0,
        len(receiver_results) // 4,
        len(receiver_results) // 2,
        (3 * len(receiver_results)) // 4,
        len(receiver_results) - 1,
    ]

    printed_indices = set()

    for index in sample_indices:

        if index in printed_indices:
            continue

        printed_indices.add(index)

        result = receiver_results[index]

        print(
            f"\nReceiver {index + 1}"
        )

        print(
            f"  Position        : "
            f"{result.position}"
        )

        print(
            f"  Serving AP      : "
            f"AP {result.serving_ap_index + 1}"
        )

        print(
            f"  Serving RSSI    : "
            f"{result.serving_rssi_dbm:.2f} dBm"
        )

        print(
            f"  Interference    : "
            f"{result.interference_dbm} dBm"
        )

        print(
            f"  SINR            : "
            f"{result.sinr_db:.2f} dB"
        )

        print(
            f"  Covered         : "
            f"{result.covered}"
        )

    # =========================================================
    # 13. Final diagnostic conclusion
    # =========================================================

    print_separator(
        "DIAGNOSTIC CONCLUSION"
    )

    if finite_interference_count > 0:

        print(
            "✓ Finite interference is being "
            "generated by the two-AP scenario."
        )

    else:

        print(
            "⚠ No finite interference values "
            "were detected."
        )

    if len(sinr_values) > 0:

        print(
            "✓ Finite SINR values are available."
        )

    else:

        print(
            "⚠ No finite SINR values detected."
        )

    if evaluation.coverage_percent < 100.0:

        print(
            "✓ Coverage is below 100%, giving "
            "the optimizer a meaningful coverage signal."
        )

    else:

        print(
            "⚠ Coverage is still 100%."
        )

        print(
            "  The synthetic environment may still "
            "be too easy for optimization."
        )

    print(
        "\nNext step:"
    )

    print(
        "Validate these multi-AP RF results "
        "before modifying the NSGA-II objective layer."
    )

    print_separator(
        "Multi-AP RF Diagnostic Complete"
    )


if __name__ == "__main__":
    main()
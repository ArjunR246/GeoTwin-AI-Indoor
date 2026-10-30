"""
GeoTwin AI Indoor - Channel Scaling Experiment

Purpose
-------
Evaluate how different Wi-Fi channel assignments affect multi-AP
indoor RF performance.

The experiment keeps AP positions fixed and compares several
channel-assignment strategies for 1-5 AP configurations.

Metrics:
    - Coverage
    - Dead zone
    - Average RSSI
    - Minimum RSSI
    - Average SINR
    - Minimum SINR
    - Average interference
    - Minimum interference
    - Maximum interference
    - NSGA-II objective values

This experiment uses the existing GeoTwin RF engine and does not
modify the NSGA-II optimizer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from geotwin.rf.ap import AccessPoint
from geotwin.rf.coverage import GridEvaluation, evaluate_grid
from geotwin.optimization.objectives import calculate_objectives
from geotwin.scenario.indoor import IndoorScenario


Position = tuple[float, float]


# ============================================================================
# RF CONFIGURATION
# ============================================================================

FREQUENCY_HZ = 2.4e9
TX_POWER_DBM = 20.0
ANTENNA_GAIN_DBI = 3.0

MINIMUM_RSSI_DBM = -67.0
MINIMUM_SINR_DB = 20.0
NOISE_FLOOR_DBM = -95.0

REFERENCE_DISTANCE_M = 1.0
PATH_LOSS_EXPONENT = 3.0

NUMBER_OF_WALLS = 4
OBSTACLE_LOSS_DB = 6.0
WALL_MATERIAL = "concrete"

# Channel overlap model already implemented in the RF engine.
SAME_CHANNEL_FACTOR = 1.0
ADJACENT_CHANNEL_FACTOR = 0.5
NON_OVERLAPPING_FACTOR = 0.0


# ============================================================================
# AP CONFIGURATIONS
# ============================================================================

AP_CONFIGURATIONS: dict[int, tuple[Position, ...]] = {
    1: (
        (10.0, 7.5),
    ),
    2: (
        (2.0, 2.0),
        (18.0, 13.0),
    ),
    3: (
        (2.0, 2.0),
        (10.0, 7.5),
        (18.0, 13.0),
    ),
    4: (
        (2.0, 2.0),
        (18.0, 2.0),
        (2.0, 13.0),
        (18.0, 13.0),
    ),
    5: (
        (2.0, 2.0),
        (18.0, 2.0),
        (10.0, 7.5),
        (2.0, 13.0),
        (18.0, 13.0),
    ),
}


# ============================================================================
# CHANNEL STRATEGIES
# ============================================================================

CHANNEL_STRATEGIES: dict[str, tuple[int, ...]] = {
    # Every AP operates on channel 1.
    "All APs - Channel 1": (
        1,
        1,
        1,
        1,
        1,
    ),

    # Traditional 2.4 GHz non-overlapping channel pattern.
    "1-6-11 Reuse": (
        1,
        6,
        11,
        1,
        6,
    ),

    # Alternating channels.
    "1-11 Alternating": (
        1,
        11,
        1,
        11,
        1,
    ),

    # Adjacent-channel pattern.
    "1-6 Alternating": (
        1,
        6,
        1,
        6,
        1,
    ),
}


# ============================================================================
# RESULT DATA STRUCTURE
# ============================================================================

@dataclass(frozen=True)
class ChannelExperimentResult:
    """Stores the result of one AP/channel configuration."""

    ap_count: int
    strategy_name: str
    channels: tuple[int, ...]

    evaluation: GridEvaluation
    objectives: tuple[float, float, float]

    minimum_interference_dbm: float
    maximum_interference_dbm: float


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_access_points(
    positions: Sequence[Position],
    channels: Sequence[int],
) -> list[AccessPoint]:
    """
    Create AccessPoint objects using the supplied positions and channels.
    """

    if len(positions) != len(channels):
        raise ValueError(
            "Number of AP positions must match number of channels."
        )

    access_points: list[AccessPoint] = []

    for position, channel in zip(positions, channels):
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


def calculate_interference_statistics(
    evaluation: GridEvaluation,
) -> tuple[float, float, float]:
    """
    Calculate average, minimum and maximum interference.

    The RF engine represents no interference using -inf dBm.
    """

    values = [
        result.interference_dbm
        for result in evaluation.receiver_results
        if result.interference_dbm != float("-inf")
    ]

    if not values:
        return (
            float("-inf"),
            float("-inf"),
            float("-inf"),
        )

    return (
        sum(values) / len(values),
        min(values),
        max(values),
    )


def print_scenario_information(
    scenario: IndoorScenario,
) -> None:
    """Print scenario configuration."""

    print()
    print("=" * 100)
    print("GeoTwin AI Indoor - Channel Scaling Experiment")
    print("=" * 100)

    print()
    print("[SCENARIO]")
    print(
        f"Building width       : {scenario.width_m:.1f} m"
    )
    print(
        f"Building length      : {scenario.length_m:.1f} m"
    )
    print(
        f"Grid resolution      : {scenario.grid_resolution_m:.1f} m"
    )
    print(
        f"Floors               : {scenario.floors}"
    )
    print(
        f"Wall material        : {scenario.wall_material}"
    )
    print(
        f"Wall thickness       : {scenario.wall_thickness_m:.2f} m"
    )

    print()
    print("[GRID]")
    print(
        f"Total grid points    : {scenario.total_grid_points}"
    )
    print(
        f"Valid AP positions   : {len(scenario.valid_ap_positions)}"
    )
    print(
        f"Receiver positions   : {len(scenario.receiver_positions)}"
    )
    print(
        f"Blocked positions    : {len(scenario.blocked_positions)}"
    )


def print_rf_configuration() -> None:
    """Print RF configuration."""

    print()
    print("[RF CONFIGURATION]")
    print(
        f"Frequency            : {FREQUENCY_HZ / 1e9:.1f} GHz"
    )
    print(
        f"Transmit power       : {TX_POWER_DBM:.1f} dBm"
    )
    print(
        f"Antenna gain         : {ANTENNA_GAIN_DBI:.1f} dBi"
    )
    print(
        f"Minimum RSSI         : {MINIMUM_RSSI_DBM:.1f} dBm"
    )
    print(
        f"Minimum SINR         : {MINIMUM_SINR_DB:.1f} dB"
    )
    print(
        f"Noise floor          : {NOISE_FLOOR_DBM:.1f} dBm"
    )
    print(
        f"Reference distance   : {REFERENCE_DISTANCE_M:.1f} m"
    )
    print(
        f"Path-loss exponent   : {PATH_LOSS_EXPONENT:.1f}"
    )
    print(
        f"Number of walls      : {NUMBER_OF_WALLS}"
    )
    print(
        f"Obstacle loss        : {OBSTACLE_LOSS_DB:.1f} dB"
    )
    print(
        f"Wall material        : {WALL_MATERIAL}"
    )

    print()
    print("[CHANNEL MODEL]")
    print(
        f"Same-channel factor      : {SAME_CHANNEL_FACTOR:.2f}"
    )
    print(
        f"Adjacent-channel factor  : {ADJACENT_CHANNEL_FACTOR:.2f}"
    )
    print(
        f"Non-overlapping factor   : {NON_OVERLAPPING_FACTOR:.2f}"
    )


def print_ap_configuration(
    positions: Sequence[Position],
) -> None:
    """Print AP positions."""

    print()
    print("[AP CONFIGURATION]")
    print(
        f"AP count : {len(positions)}"
    )

    for index, position in enumerate(
        positions,
        start=1,
    ):
        print(
            f"AP {index:<2}    : {position}"
        )


def evaluate_configuration(
    scenario: IndoorScenario,
    positions: Sequence[Position],
    channels: Sequence[int],
    strategy_name: str,
) -> ChannelExperimentResult:
    """
    Evaluate one AP placement and channel assignment.
    """

    if len(positions) != len(channels):
        raise ValueError(
            "Positions and channels must have the same length."
        )

    access_points = create_access_points(
        positions=positions,
        channels=channels,
    )

    evaluation = evaluate_grid(
        receiver_positions=scenario.receiver_positions,
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

    objectives = calculate_objectives(
        evaluation=evaluation,
        ap_count=len(positions),
    )

    (
        average_interference_dbm,
        minimum_interference_dbm,
        maximum_interference_dbm,
    ) = calculate_interference_statistics(
        evaluation
    )

    # average_interference_dbm is already represented by the RF layer.
    # It is intentionally not recomputed here.
    _ = average_interference_dbm

    return ChannelExperimentResult(
        ap_count=len(positions),
        strategy_name=strategy_name,
        channels=tuple(channels),
        evaluation=evaluation,
        objectives=objectives.values,
        minimum_interference_dbm=minimum_interference_dbm,
        maximum_interference_dbm=maximum_interference_dbm,
    )


def print_result(
    result: ChannelExperimentResult,
) -> None:
    """Print detailed results for one experiment."""

    evaluation = result.evaluation

    average_interference_dbm = (
        evaluation.average_interference_dbm
    )

    print()
    print("-" * 100)
    print(
        f"AP COUNT: {result.ap_count} | "
        f"STRATEGY: {result.strategy_name}"
    )
    print("-" * 100)

    print(
        f"Channels              : {result.channels}"
    )

    print(
        f"Covered points        : "
        f"{evaluation.covered_points}/{evaluation.total_points}"
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
        f"{average_interference_dbm:.2f} dBm"
    )

    print(
        f"Minimum interference  : "
        f"{result.minimum_interference_dbm:.2f} dBm"
    )

    print(
        f"Maximum interference  : "
        f"{result.maximum_interference_dbm:.2f} dBm"
    )

    print()
    print("[NSGA-II OBJECTIVES]")

    print(
        f"Objectives            : "
        f"{result.objectives}"
    )

    print(
        f"Objective 1 "
        f"(negative coverage)   : "
        f"{result.objectives[0]:.6f}"
    )

    print(
        f"Objective 2 "
        f"(AP count)            : "
        f"{result.objectives[1]:.2f}"
    )

    print(
        f"Objective 3 "
        f"(interference)        : "
        f"{result.objectives[2]:.8e}"
    )

    print(
        f"Interference objective: "
        f"{result.objectives[2]:.8e} mW"
    )


def print_summary(
    results: Sequence[ChannelExperimentResult],
) -> None:
    """Print compact comparison table."""

    print()
    print("=" * 120)
    print("CHANNEL SCALING SUMMARY")
    print("=" * 120)

    print()

    header = (
        f"{'APs':>4} | "
        f"{'Strategy':<24} | "
        f"{'Channels':<15} | "
        f"{'Coverage':>9} | "
        f"{'Avg RSSI':>11} | "
        f"{'Avg SINR':>10} | "
        f"{'Avg Int.':>12}"
    )

    print(header)
    print("-" * len(header))

    for result in results:

        evaluation = result.evaluation

        average_interference = (
            evaluation.average_interference_dbm
        )

        print(
            f"{result.ap_count:>4} | "
            f"{result.strategy_name:<24} | "
            f"{str(result.channels):<15} | "
            f"{evaluation.coverage_percent:>8.2f}% | "
            f"{evaluation.average_rssi_dbm:>10.2f} | "
            f"{evaluation.average_sinr_db:>9.2f} | "
            f"{average_interference:>11.2f}"
        )


def print_best_strategy_by_ap_count(
    results: Sequence[ChannelExperimentResult],
) -> None:
    """Determine the best strategy for every AP count."""

    print()
    print("=" * 100)
    print("BEST CHANNEL STRATEGY BY AP COUNT")
    print("=" * 100)

    ap_counts = sorted(
        {
            result.ap_count
            for result in results
        }
    )

    for ap_count in ap_counts:

        candidates = [
            result
            for result in results
            if result.ap_count == ap_count
        ]

        # Primary metric: coverage.
        # Secondary metric: average SINR.
        best = max(
            candidates,
            key=lambda result: (
                result.evaluation.coverage_percent,
                result.evaluation.average_sinr_db,
            ),
        )

        print()
        print(
            f"AP count : {ap_count}"
        )
        print(
            f"Best strategy : {best.strategy_name}"
        )
        print(
            f"Channels      : {best.channels}"
        )
        print(
            f"Coverage      : "
            f"{best.evaluation.coverage_percent:.2f}%"
        )
        print(
            f"Average SINR  : "
            f"{best.evaluation.average_sinr_db:.2f} dB"
        )
        print(
            f"Average RSSI  : "
            f"{best.evaluation.average_rssi_dbm:.2f} dBm"
        )
        print(
            f"Interference  : "
            f"{best.evaluation.average_interference_dbm:.2f} dBm"
        )


def print_channel_effect_analysis(
    results: Sequence[ChannelExperimentResult],
) -> None:
    """Print the effect of channel assignment."""

    print()
    print("=" * 100)
    print("CHANNEL EFFECT ANALYSIS")
    print("=" * 100)

    print()
    print(
        "Interpretation:"
    )

    print(
        "  - Same-channel APs should produce the strongest interference."
    )

    print(
        "  - Non-overlapping channels should reduce co-channel interference."
    )

    print(
        "  - Reduced interference should generally improve SINR."
    )

    print(
        "  - Coverage depends on BOTH RSSI and the SINR threshold."
    )

    print(
        "  - Therefore, a channel strategy can improve coverage even"
    )

    print(
        "    when AP positions and transmit power remain unchanged."
    )

    print()

    # Compare the all-channel-1 strategy with the 1-6-11 strategy.
    for ap_count in sorted(
        {
            result.ap_count
            for result in results
        }
    ):

        same_channel = next(
            (
                result
                for result in results
                if result.ap_count == ap_count
                and result.strategy_name
                == "All APs - Channel 1"
            ),
            None,
        )

        reuse_strategy = next(
            (
                result
                for result in results
                if result.ap_count == ap_count
                and result.strategy_name
                == "1-6-11 Reuse"
            ),
            None,
        )

        if same_channel is None or reuse_strategy is None:
            continue

        coverage_change = (
            reuse_strategy.evaluation.coverage_percent
            - same_channel.evaluation.coverage_percent
        )

        sinr_change = (
            reuse_strategy.evaluation.average_sinr_db
            - same_channel.evaluation.average_sinr_db
        )

        print(
            f"{ap_count} AP(s):"
        )

        print(
            f"  Coverage change "
            f"(1-6-11 vs all-1): "
            f"{coverage_change:+.2f} percentage points"
        )

        print(
            f"  Average SINR change: "
            f"{sinr_change:+.2f} dB"
        )

        print()


def main() -> None:
    """Run the complete channel scaling experiment."""

    scenario = IndoorScenario(
        width_m=20.0,
        length_m=15.0,
        grid_resolution_m=0.5,
        floors=1,
        wall_material="concrete",
        wall_thickness_m=0.2,
    )

    scenario.validate()

    print_scenario_information(
        scenario
    )

    print_rf_configuration()

    # ----------------------------------------------------------------------
    # Validate AP configurations
    # ----------------------------------------------------------------------

    print()
    print("[AP CONFIGURATION VALIDATION]")

    for ap_count, positions in AP_CONFIGURATIONS.items():

        print(
            f"{ap_count} AP"
            f"{'s' if ap_count > 1 else ''} : "
            f"{positions}"
        )

        valid_positions = set(
            scenario.valid_ap_positions
        )

        for index, position in enumerate(
            positions,
            start=1,
        ):

            status = (
                "VALID"
                if position in valid_positions
                else "INVALID"
            )

            print(
                f"    AP {index}: "
                f"{position} -> {status}"
            )

    # ----------------------------------------------------------------------
    # Run experiment
    # ----------------------------------------------------------------------

    print()
    print("=" * 100)
    print("Running channel scaling experiment...")
    print("=" * 100)

    results: list[ChannelExperimentResult] = []

    for ap_count in sorted(
        AP_CONFIGURATIONS
    ):

        positions = AP_CONFIGURATIONS[
            ap_count
        ]

        for strategy_name, channel_template in (
            CHANNEL_STRATEGIES.items()
        ):

            channels = channel_template[
                :ap_count
            ]

            result = evaluate_configuration(
                scenario=scenario,
                positions=positions,
                channels=channels,
                strategy_name=strategy_name,
            )

            results.append(
                result
            )

            print_result(
                result
            )

    # ----------------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------------

    print_summary(
        results
    )

    print_best_strategy_by_ap_count(
        results
    )

    print_channel_effect_analysis(
        results
    )

    # ----------------------------------------------------------------------
    # Final conclusion
    # ----------------------------------------------------------------------

    print()
    print("=" * 100)
    print("CHANNEL SCALING EXPERIMENT CONCLUSION")
    print("=" * 100)

    print()

    same_channel_results = [
        result
        for result in results
        if result.strategy_name
        == "All APs - Channel 1"
    ]

    reuse_results = [
        result
        for result in results
        if result.strategy_name
        == "1-6-11 Reuse"
    ]

    if same_channel_results and reuse_results:

        same_channel_average_sinr = (
            sum(
                result.evaluation.average_sinr_db
                for result in same_channel_results
            )
            / len(same_channel_results)
        )

        reuse_average_sinr = (
            sum(
                result.evaluation.average_sinr_db
                for result in reuse_results
            )
            / len(reuse_results)
        )

        if reuse_average_sinr > same_channel_average_sinr:

            print(
                "✓ Channel reuse produced a higher average SINR "
                "across the tested configurations."
            )

        else:

            print(
                "⚠ Channel reuse did not produce a higher average "
                "SINR under the current RF configuration."
            )

    print()
    print(
        "Next step:"
    )
    print(
        "Use these results to validate channel-aware optimization "
        "before modifying the NSGA-II individual representation."
    )

    print()
    print("=" * 100)
    print("Channel Scaling Experiment Complete")
    print("=" * 100)


if __name__ == "__main__":
    main()
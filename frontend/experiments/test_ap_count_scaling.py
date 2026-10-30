"""
GeoTwin AI Indoor
AP Count Scaling Experiment

Evaluates fixed AP configurations with 1-5 APs and compares:

    - Coverage
    - Dead-zone percentage
    - Average RSSI
    - Minimum RSSI
    - Average SINR
    - Minimum SINR
    - Average interference
    - Minimum interference
    - Maximum interference
    - NSGA-II objective values

This experiment uses the existing GeoTwin RF evaluation layer directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

from geotwin.optimization.objectives import calculate_objectives
from geotwin.rf.ap import AccessPoint
from geotwin.rf.coverage import GridEvaluation, evaluate_grid
from geotwin.scenario.indoor import IndoorScenario


Position = Tuple[float, float]


# ============================================================================
# CONFIGURATION
# ============================================================================

FREQUENCY_HZ = 2.4e9
TX_POWER_DBM = 20.0
ANTENNA_GAIN_DBI = 3.0
CHANNEL = 1

MINIMUM_RSSI_DBM = -67.0
MINIMUM_SINR_DB = 20.0
NOISE_FLOOR_DBM = -95.0

REFERENCE_DISTANCE_M = 1.0
PATH_LOSS_EXPONENT = 3.0

NUMBER_OF_WALLS = 4
OBSTACLE_LOSS_DB = 6.0
WALL_MATERIAL = "concrete"

SAME_CHANNEL_FACTOR = 1.0
ADJACENT_CHANNEL_FACTOR = 0.5
NON_OVERLAPPING_FACTOR = 0.0


# ============================================================================
# AP CONFIGURATIONS
# ============================================================================

AP_CONFIGURATIONS: dict[int, Tuple[Position, ...]] = {
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
# RESULT DATA STRUCTURE
# ============================================================================

@dataclass(frozen=True)
class ScalingResult:
    """Summary of one AP-count evaluation."""

    ap_count: int

    positions: Tuple[Position, ...]

    evaluation: GridEvaluation

    average_interference_dbm: float
    minimum_interference_dbm: float
    maximum_interference_dbm: float

    interference_values_mw: Tuple[float, ...]

    objective_values: Tuple[float, float, float]


# ============================================================================
# HELPERS
# ============================================================================

def create_access_points(
    positions: Sequence[Position],
) -> list[AccessPoint]:
    """Create AccessPoint objects using the experiment RF configuration."""

    return [
        AccessPoint(
            position=position,
            frequency_hz=FREQUENCY_HZ,
            tx_power_dbm=TX_POWER_DBM,
            antenna_gain_dbi=ANTENNA_GAIN_DBI,
            channel=CHANNEL,
        )
        for position in positions
    ]


def dbm_to_mw(
    power_dbm: float,
) -> float:
    """Convert dBm to milliwatts."""

    if power_dbm == float("-inf"):
        return 0.0

    return 10.0 ** (power_dbm / 10.0)


def extract_interference_values(
    evaluation: GridEvaluation,
) -> Tuple[float, ...]:
    """
    Extract finite interference values from receiver evaluations.

    -inf means no measurable interference and is converted to 0 mW.
    """

    return tuple(
        result.interference_dbm
        for result in evaluation.receiver_results
    )


def calculate_interference_statistics(
    evaluation: GridEvaluation,
) -> tuple[
    float,
    float,
    float,
    Tuple[float, ...],
]:
    """
    Calculate interference statistics.

    Important:

    Interference is physically a power quantity, so averaging dBm
    values directly is not physically correct.

    The RF layer currently exposes average_interference_dbm as an
    existing diagnostic property. This experiment reports that value
    for consistency with the current system.

    Minimum and maximum are calculated directly from finite
    receiver-level dBm values.
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
            tuple(),
        )

    average_dbm = evaluation.average_interference_dbm

    minimum_dbm = min(values)
    maximum_dbm = max(values)

    values_mw = tuple(
        dbm_to_mw(value)
        for value in values
    )

    return (
        average_dbm,
        minimum_dbm,
        maximum_dbm,
        values_mw,
    )


def validate_ap_configuration(
    scenario: IndoorScenario,
    positions: Sequence[Position],
) -> list[bool]:
    """Check whether every AP position is valid."""

    valid_positions = set(
        scenario.valid_ap_positions
    )

    blocked_positions = set(
        scenario.blocked_positions
    )

    return [
        position in valid_positions
        and position not in blocked_positions
        for position in positions
    ]


def evaluate_configuration(
    scenario: IndoorScenario,
    positions: Sequence[Position],
) -> ScalingResult:
    """Evaluate one fixed AP configuration."""

    if not positions:
        raise ValueError(
            "At least one AP position is required."
        )

    validity = validate_ap_configuration(
        scenario,
        positions,
    )

    if not all(validity):
        raise ValueError(
            "One or more AP positions are invalid."
        )

    access_points = create_access_points(
        positions
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

    (
        average_interference_dbm,
        minimum_interference_dbm,
        maximum_interference_dbm,
        interference_values_mw,
    ) = calculate_interference_statistics(
        evaluation
    )

    objectives = calculate_objectives(
        evaluation=evaluation,
        ap_count=len(positions),
    )

    return ScalingResult(
        ap_count=len(positions),
        positions=tuple(positions),
        evaluation=evaluation,
        average_interference_dbm=average_interference_dbm,
        minimum_interference_dbm=minimum_interference_dbm,
        maximum_interference_dbm=maximum_interference_dbm,
        interference_values_mw=interference_values_mw,
        objective_values=objectives.values,
    )


# ============================================================================
# DISPLAY HELPERS
# ============================================================================

def format_dbm(
    value: float,
) -> str:
    """Format a dBm value for console output."""

    if value == float("-inf"):
        return "-inf"

    return f"{value:.2f} dBm"


def format_mw(
    value_dbm: float,
) -> str:
    """Convert dBm to mW and format."""

    if value_dbm == float("-inf"):
        return "0.00000000e+00 mW"

    return f"{dbm_to_mw(value_dbm):.8e} mW"


def print_configuration(
    scenario: IndoorScenario,
) -> None:

    print()
    print("=" * 100)
    print("GeoTwin AI Indoor - AP Count Scaling Experiment")
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
        f"Channel              : {CHANNEL}"
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


def print_ap_validation(
    scenario: IndoorScenario,
) -> None:

    print()
    print("[AP CONFIGURATION VALIDATION]")

    valid_positions = set(
        scenario.valid_ap_positions
    )

    blocked_positions = set(
        scenario.blocked_positions
    )

    for ap_count, positions in AP_CONFIGURATIONS.items():

        print(
            f"{ap_count} AP"
            f"{'s' if ap_count > 1 else ''} : "
            f"{positions}"
        )

        for index, position in enumerate(
            positions,
            start=1,
        ):

            is_valid = (
                position in valid_positions
                and position not in blocked_positions
            )

            status = (
                "VALID"
                if is_valid
                else "INVALID"
            )

            print(
                f"    AP {index}: "
                f"{position} -> {status}"
            )


def print_result(
    result: ScalingResult,
) -> None:

    evaluation = result.evaluation

    print()
    print("-" * 100)

    print(
        f"AP COUNT: {result.ap_count}"
    )

    print(
        f"AP positions          : {result.positions}"
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
        f"{format_dbm(result.average_interference_dbm)}"
    )

    print(
        f"Minimum interference  : "
        f"{format_dbm(result.minimum_interference_dbm)}"
    )

    print(
        f"Maximum interference  : "
        f"{format_dbm(result.maximum_interference_dbm)}"
    )

    print()

    print("[NSGA-II OBJECTIVES]")

    coverage_objective = result.objective_values[0]
    ap_objective = result.objective_values[1]
    interference_objective = result.objective_values[2]

    print(
        f"Objectives             : "
        f"{result.objective_values}"
    )

    print(
        f"Objective 1 "
        f"(negative coverage)   : "
        f"{coverage_objective:.4f}"
    )

    print(
        f"Objective 2 "
        f"(AP count)             : "
        f"{ap_objective:.1f}"
    )

    print(
        f"Objective 3 "
        f"(interference)         : "
        f"{interference_objective:.4f}"
    )

    print(
        f"Interference objective : "
        f"{format_mw(result.average_interference_dbm)}"
    )


# ============================================================================
# COMPARISON TABLE
# ============================================================================

def print_comparison_table(
    results: Sequence[ScalingResult],
) -> None:

    print()
    print("=" * 120)
    print("AP COUNT SCALING SUMMARY")
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
        f"{'Avg Int.':>12}"
    )

    print(header)
    print("-" * len(header))

    for result in results:

        evaluation = result.evaluation

        average_interference = (
            format_dbm(
                result.average_interference_dbm
            )
        )

        print(
            f"{result.ap_count:>4} | "
            f"{evaluation.coverage_percent:>9.2f}% | "
            f"{evaluation.dead_zone_percent:>9.2f}% | "
            f"{evaluation.average_rssi_dbm:>10.2f} dBm | "
            f"{evaluation.minimum_rssi_dbm:>10.2f} dBm | "
            f"{evaluation.average_sinr_db:>10.2f} dB | "
            f"{evaluation.minimum_sinr_db:>10.2f} dB | "
            f"{average_interference:>12}"
        )


def print_objective_comparison(
    results: Sequence[ScalingResult],
) -> None:

    print()
    print("=" * 100)
    print("NSGA-II OBJECTIVE COMPARISON")
    print("=" * 100)

    print()

    header = (
        f"{'APs':>4} | "
        f"{'Coverage Obj.':>16} | "
        f"{'AP Count Obj.':>14} | "
        f"{'Interference Obj.':>20}"
    )

    print(header)
    print("-" * len(header))

    for result in results:

        coverage_objective = (
            result.objective_values[0]
        )

        ap_count_objective = (
            result.objective_values[1]
        )

        interference_objective = (
            result.objective_values[2]
        )

        print(
            f"{result.ap_count:>4} | "
            f"{coverage_objective:>16.6f} | "
            f"{ap_count_objective:>14.2f} | "
            f"{interference_objective:>20.6f}"
        )


# ============================================================================
# INTERPRETATION
# ============================================================================

def print_interpretation(
    results: Sequence[ScalingResult],
) -> None:

    print()
    print("=" * 100)
    print("SCALING INTERPRETATION")
    print("=" * 100)

    if not results:
        return

    best_coverage = max(
        results,
        key=lambda result:
        result.evaluation.coverage_percent,
    )

    best_sinr = max(
        results,
        key=lambda result:
        result.evaluation.average_sinr_db,
    )

    lowest_interference = min(
        results,
        key=lambda result:
        (
            result.average_interference_dbm
            if result.average_interference_dbm
            != float("-inf")
            else float("-inf")
        ),
    )

    print()

    print(
        f"Best coverage configuration : "
        f"{best_coverage.ap_count} AP(s)"
    )

    print(
        f"Best coverage               : "
        f"{best_coverage.evaluation.coverage_percent:.2f}%"
    )

    print()

    print(
        f"Best average SINR           : "
        f"{best_sinr.ap_count} AP(s)"
    )

    print(
        f"Best average SINR           : "
        f"{best_sinr.evaluation.average_sinr_db:.2f} dB"
    )

    print()

    print(
        f"Lowest reported interference: "
        f"{lowest_interference.ap_count} AP(s)"
    )

    print(
        f"Interference                : "
        f"{format_dbm(lowest_interference.average_interference_dbm)}"
    )

    print()

    print(
        "Interpretation:"
    )

    print(
        "  - Increasing AP count can improve RSSI and spatial coverage."
    )

    print(
        "  - Same-channel APs also introduce additional interference."
    )

    print(
        "  - SINR therefore represents the trade-off between stronger "
        "coverage and increased interference."
    )

    print(
        "  - NSGA-II can use these competing objectives to search for "
        "efficient AP deployments."
    )


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:

    scenario = IndoorScenario(
        width_m=20.0,
        length_m=15.0,
        grid_resolution_m=0.5,
        floors=1,
        wall_material="concrete",
        wall_thickness_m=0.2,
    )

    scenario.validate()

    print_configuration(
        scenario
    )

    print_ap_validation(
        scenario
    )

    print()
    print("=" * 100)
    print("Running AP-count scaling experiment...")
    print("=" * 100)

    results: list[ScalingResult] = []

    for ap_count in sorted(
        AP_CONFIGURATIONS
    ):

        positions = AP_CONFIGURATIONS[
            ap_count
        ]

        result = evaluate_configuration(
            scenario=scenario,
            positions=positions,
        )

        results.append(
            result
        )

        print_result(
            result
        )

    print_comparison_table(
        results
    )

    print_objective_comparison(
        results
    )

    print_interpretation(
        results
    )

    print()
    print("=" * 100)
    print("AP Count Scaling Experiment Complete")
    print("=" * 100)


if __name__ == "__main__":
    main()
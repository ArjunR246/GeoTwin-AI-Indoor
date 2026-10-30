"""
GeoTwin AI Indoor - NSGA-II Optimization Visualization

Visualizes the production channel-aware NSGA-II Pareto solutions.
The script uses the same production RF evaluation pipeline and does
not modify the optimizer.
"""

from pathlib import Path

import matplotlib.pyplot as plt

from geotwin.rf.ap import AccessPoint
from geotwin.rf.coverage import evaluate_grid
from geotwin.scenario.indoor import IndoorScenario


# ============================================================
# CONFIGURATION
# ============================================================

OUTPUT_DIR = Path("outputs")

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


# Production Pareto configurations from the validated run.
# Duplicate equivalent configurations are removed.

PARETO_CONFIGURATIONS = [
    {
        "positions": ((12.0, 7.0),),
        "channels": (6,),
    },
    {
        "positions": ((5.0, 3.0), (5.5, 12.0), (16.5, 8.0)),
        "channels": (1, 11, 6),
    },
    {
        "positions": ((6.0, 8.5), (15.5, 5.0)),
        "channels": (11, 6),
    },
    {
        "positions": ((6.0, 8.5),),
        "channels": (11,),
    },
    {
        "positions": ((6.0, 8.5),),
        "channels": (6,),
    },
    {
        "positions": ((6.0, 8.5),),
        "channels": (1,),
    },
    {
        "positions": ((9.5, 6.0),),
        "channels": (6,),
    },
]


# ============================================================
# ACCESS POINT CREATION
# ============================================================

def create_access_points(positions, channels):
    """Create RF AccessPoint objects."""

    return [
        AccessPoint(
            position=position,
            frequency_hz=FREQUENCY_HZ,
            tx_power_dbm=TX_POWER_DBM,
            antenna_gain_dbi=ANTENNA_GAIN_DBI,
            channel=channel,
        )
        for position, channel in zip(
            positions,
            channels,
        )
    ]


# ============================================================
# EVALUATE CONFIGURATION
# ============================================================

def evaluate_configuration(
    scenario,
    positions,
    channels,
):
    """Evaluate one Pareto configuration using the real RF engine."""

    access_points = create_access_points(
        positions,
        channels,
    )

    return evaluate_grid(
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


# ============================================================
# PARETO DATA
# ============================================================

def collect_results(scenario):
    """Evaluate all unique Pareto configurations."""

    results = []

    for configuration in PARETO_CONFIGURATIONS:

        positions = configuration["positions"]
        channels = configuration["channels"]

        evaluation = evaluate_configuration(
            scenario,
            positions,
            channels,
        )

        results.append(
            {
                "ap_count": len(positions),
                "coverage": evaluation.coverage_percent,
                "average_sinr": evaluation.average_sinr_db,
                "minimum_sinr": evaluation.minimum_sinr_db,
                "average_rssi": evaluation.average_rssi_dbm,
                "minimum_rssi": evaluation.minimum_rssi_dbm,
                "positions": positions,
                "channels": channels,
            }
        )

    return results


# ============================================================
# PRINT RESULTS
# ============================================================

def print_results(results):

    print()
    print("=" * 80)
    print("PRODUCTION PARETO DATA")
    print("=" * 80)

    for index, result in enumerate(
        results,
        start=1,
    ):

        print()
        print(f"Solution {index}")
        print(
            f"  AP count       : "
            f"{result['ap_count']}"
        )
        print(
            f"  Coverage       : "
            f"{result['coverage']:.2f}%"
        )
        print(
            f"  Average RSSI   : "
            f"{result['average_rssi']:.2f} dBm"
        )
        print(
            f"  Average SINR   : "
            f"{result['average_sinr']:.2f} dB"
        )
        print(
            f"  Minimum SINR   : "
            f"{result['minimum_sinr']:.2f} dB"
        )
        print(
            f"  Channels       : "
            f"{result['channels']}"
        )


# ============================================================
# PLOT 1
# COVERAGE VS AP COUNT
# ============================================================

def plot_coverage_vs_ap_count(results):

    fig, ax = plt.subplots(
        figsize=(10, 7)
    )

    for result in results:

        ax.scatter(
            result["ap_count"],
            result["coverage"],
            s=130,
        )

    # Highlight production best solution.
    best = max(
        results,
        key=lambda item: item["coverage"],
    )

    ax.scatter(
        best["ap_count"],
        best["coverage"],
        s=280,
        marker="*",
        edgecolors="black",
        linewidths=1.5,
        label="Best coverage",
        zorder=10,
    )

    ax.annotate(
        f"{best['coverage']:.2f}% coverage\n"
        f"{best['ap_count']} APs",
        (
            best["ap_count"],
            best["coverage"],
        ),
        xytext=(12, 12),
        textcoords="offset points",
        fontsize=10,
    )

    ax.set_xlabel(
        "Number of APs"
    )

    ax.set_ylabel(
        "Coverage (%)"
    )

    ax.set_title(
        "GeoTwin AI Indoor — Coverage vs AP Count",
        fontsize=15,
        fontweight="bold",
    )

    ax.set_xticks(
        sorted(
            set(
                r["ap_count"]
                for r in results
            )
        )
    )

    ax.grid(
        True,
        alpha=0.25,
    )

    ax.legend()

    fig.tight_layout()

    path = (
        OUTPUT_DIR
        / "pareto_coverage_vs_ap_count.png"
    )

    fig.savefig(
        path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"Created: {path}"
    )


# ============================================================
# PLOT 2
# SINR VS AP COUNT
# ============================================================

def plot_sinr_vs_ap_count(results):

    fig, ax = plt.subplots(
        figsize=(10, 7)
    )

    for result in results:

        ax.scatter(
            result["ap_count"],
            result["average_sinr"],
            s=130,
        )

    best = max(
        results,
        key=lambda item: item["average_sinr"],
    )

    ax.scatter(
        best["ap_count"],
        best["average_sinr"],
        s=280,
        marker="*",
        edgecolors="black",
        linewidths=1.5,
        label="Best average SINR",
        zorder=10,
    )

    ax.axhline(
        MINIMUM_SINR_DB,
        linestyle="--",
        linewidth=1.5,
        label=f"SINR threshold ({MINIMUM_SINR_DB:.0f} dB)",
    )

    ax.annotate(
        f"{best['average_sinr']:.2f} dB\n"
        f"{best['ap_count']} APs",
        (
            best["ap_count"],
            best["average_sinr"],
        ),
        xytext=(12, 12),
        textcoords="offset points",
        fontsize=10,
    )

    ax.set_xlabel(
        "Number of APs"
    )

    ax.set_ylabel(
        "Average SINR (dB)"
    )

    ax.set_title(
        "GeoTwin AI Indoor — SINR vs AP Count",
        fontsize=15,
        fontweight="bold",
    )

    ax.set_xticks(
        sorted(
            set(
                r["ap_count"]
                for r in results
            )
        )
    )

    ax.grid(
        True,
        alpha=0.25,
    )

    ax.legend()

    fig.tight_layout()

    path = (
        OUTPUT_DIR
        / "pareto_sinr_vs_ap_count.png"
    )

    fig.savefig(
        path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"Created: {path}"
    )


# ============================================================
# PLOT 3
# COVERAGE VS SINR
# ============================================================

def plot_coverage_vs_sinr(results):

    fig, ax = plt.subplots(
        figsize=(10, 7)
    )

    for result in results:

        ax.scatter(
            result["coverage"],
            result["average_sinr"],
            s=140,
        )

        ax.annotate(
            f"{result['ap_count']} AP",
            (
                result["coverage"],
                result["average_sinr"],
            ),
            xytext=(7, 7),
            textcoords="offset points",
            fontsize=9,
        )

    best = max(
        results,
        key=lambda item: item["coverage"],
    )

    ax.scatter(
        best["coverage"],
        best["average_sinr"],
        s=300,
        marker="*",
        edgecolors="black",
        linewidths=1.5,
        label="Production best",
        zorder=10,
    )

    ax.set_xlabel(
        "Coverage (%)"
    )

    ax.set_ylabel(
        "Average SINR (dB)"
    )

    ax.set_title(
        "GeoTwin AI Indoor — Coverage vs Average SINR",
        fontsize=15,
        fontweight="bold",
    )

    ax.grid(
        True,
        alpha=0.25,
    )

    ax.legend()

    fig.tight_layout()

    path = (
        OUTPUT_DIR
        / "pareto_coverage_vs_sinr.png"
    )

    fig.savefig(
        path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"Created: {path}"
    )


# ============================================================
# PLOT 4
# PARETO FRONT
# ============================================================

def plot_pareto_front(results):

    # Determine non-dominated solutions using:
    # maximize coverage
    # maximize average SINR
    # minimize AP count

    pareto = []

    for candidate in results:

        dominated = False

        for other in results:

            if candidate is other:
                continue

            no_worse = (
                other["coverage"]
                >= candidate["coverage"]
                and
                other["average_sinr"]
                >= candidate["average_sinr"]
                and
                other["ap_count"]
                <= candidate["ap_count"]
            )

            strictly_better = (
                other["coverage"]
                > candidate["coverage"]
                or
                other["average_sinr"]
                > candidate["average_sinr"]
                or
                other["ap_count"]
                < candidate["ap_count"]
            )

            if no_worse and strictly_better:
                dominated = True
                break

        if not dominated:
            pareto.append(candidate)

    pareto.sort(
        key=lambda item: item["coverage"]
    )

    fig, ax = plt.subplots(
        figsize=(10, 7)
    )

    for result in results:

        is_pareto = result in pareto

        ax.scatter(
            result["coverage"],
            result["average_sinr"],
            s=180 if is_pareto else 90,
            marker="o",
            edgecolors="black" if is_pareto else None,
            linewidths=1.2 if is_pareto else 0,
            label="Pareto solution"
            if is_pareto
            and not any(
                h.get_label()
                == "Pareto solution"
                for h in ax.collections
            )
            else None,
        )

        ax.annotate(
            f"{result['ap_count']} AP",
            (
                result["coverage"],
                result["average_sinr"],
            ),
            xytext=(7, 7),
            textcoords="offset points",
            fontsize=9,
        )

    if len(pareto) >= 2:

        ax.plot(
            [
                item["coverage"]
                for item in pareto
            ],
            [
                item["average_sinr"]
                for item in pareto
            ],
            linestyle="--",
            linewidth=1.5,
            label="Pareto front",
        )

    ax.set_xlabel(
        "Coverage (%)"
    )

    ax.set_ylabel(
        "Average SINR (dB)"
    )

    ax.set_title(
        "GeoTwin AI Indoor — NSGA-II Pareto Front",
        fontsize=15,
        fontweight="bold",
    )

    ax.grid(
        True,
        alpha=0.25,
    )

    handles, labels = ax.get_legend_handles_labels()

    if handles:
        unique = dict(
            zip(labels, handles)
        )

        ax.legend(
            unique.values(),
            unique.keys(),
        )

    fig.tight_layout()

    path = (
        OUTPUT_DIR
        / "pareto_front.png"
    )

    fig.savefig(
        path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"Created: {path}"
    )

    return pareto


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print(
        "GeoTwin AI Indoor - NSGA-II Optimization Visualization"
    )
    print("=" * 80)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

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
        f"Building: "
        f"{scenario.width_m} m x "
        f"{scenario.length_m} m"
    )

    print(
        f"Grid points: "
        f"{scenario.total_grid_points}"
    )

    print()
    print(
        "Evaluating production Pareto configurations..."
    )

    results = collect_results(
        scenario
    )

    print_results(
        results
    )

    print()
    print(
        "Generating optimization plots..."
    )

    plot_coverage_vs_ap_count(
        results
    )

    plot_sinr_vs_ap_count(
        results
    )

    plot_coverage_vs_sinr(
        results
    )

    pareto = plot_pareto_front(
        results
    )

    print()
    print(
        "=" * 80
    )
    print(
        "PARETO VISUALIZATION COMPLETE"
    )
    print(
        "=" * 80
    )

    print()
    print(
        f"Non-dominated solutions: "
        f"{len(pareto)}"
    )

    for result in pareto:

        print(
            f"  {result['ap_count']} APs | "
            f"{result['coverage']:.2f}% coverage | "
            f"{result['average_sinr']:.2f} dB SINR"
        )

    print()
    print(
        f"Output directory: "
        f"{OUTPUT_DIR.resolve()}"
    )


if __name__ == "__main__":
    main()
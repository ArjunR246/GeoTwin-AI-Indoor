"""
GeoTwin AI Indoor - Floor-Plan-Aware RF Heatmap Generator

Loads a real floor-plan image, converts the detected geometry into
physical RF coordinates, evaluates the production RF engine, and
generates RSSI, SINR, interference, and coverage maps.
"""

from pathlib import Path

import matplotlib

# Use a non-GUI backend so visualization generation works in
# automated tests, CI environments, and machines without Tk/Tcl.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from geotwin.floorplan.pipeline import process_floor_plan_file
from geotwin.floorplan.rf_adapter import floor_plan_to_rf_geometry
from geotwin.rf.ap import AccessPoint
from geotwin.rf.coverage import evaluate_grid


# ============================================================
# CONFIGURATION
# ============================================================

OUTPUT_DIR = Path("outputs")

FLOOR_PLAN_PATH = Path(
    "tests/fixtures/floorplans/simple_floorplan.png"
)

PHYSICAL_WIDTH_M = 20.0
PHYSICAL_HEIGHT_M = 15.0

CELL_SIZE_M = 0.5

MINIMUM_RSSI_DBM = -67.0
MINIMUM_SINR_DB = 20.0
NOISE_FLOOR_DBM = -95.0

FREQUENCY_HZ = 2.4e9
TX_POWER_DBM = 20.0
ANTENNA_GAIN_DBI = 3.0

REFERENCE_DISTANCE_M = 1.0
PATH_LOSS_EXPONENT = 3.0

WALL_MATERIAL = "concrete"
OBSTACLE_LOSS_DB = 6.0

FLOOR_PLAN_WALL_LOSS_DB = 6.0

SAME_CHANNEL_FACTOR = 1.0
ADJACENT_CHANNEL_FACTOR = 0.5
NON_OVERLAPPING_FACTOR = 0.0


# ============================================================
# PRODUCTION AP CONFIGURATION
# ============================================================

AP_CONFIGURATION = (
    ((5.0, 3.0), 1),
    ((5.5, 12.0), 11),
    ((16.5, 8.0), 6),
)


# ============================================================
# CREATE ACCESS POINTS
# ============================================================

def create_access_points():
    """Create production AccessPoint objects."""

    return [
        AccessPoint(
            position=position,
            frequency_hz=FREQUENCY_HZ,
            tx_power_dbm=TX_POWER_DBM,
            antenna_gain_dbi=ANTENNA_GAIN_DBI,
            channel=channel,
        )
        for position, channel in AP_CONFIGURATION
    ]


# ============================================================
# FLOOR-PLAN PROCESSING
# ============================================================

def load_floor_plan_geometry():
    """
    Process the floor plan and convert its geometry to RF space.

    Returns:
        pipeline_result:
            Complete floor-plan processing result.

        receiver_positions:
            Physical receiver/grid positions.

        valid_ap_positions:
            Physical positions where APs may be placed.

        blocked_positions:
            Physical blocked positions.

        physical_walls:
            Detected walls transformed into metres.
    """

    pipeline_result = process_floor_plan_file(
        FLOOR_PLAN_PATH,
        physical_width_m=PHYSICAL_WIDTH_M,
        physical_height_m=PHYSICAL_HEIGHT_M,
    )

    if not pipeline_result.has_physical_scale:
        raise ValueError(
            "Floor plan must have physical dimensions for RF evaluation."
        )

    (
        receiver_positions,
        valid_ap_positions,
        blocked_positions,
        physical_walls,
    ) = floor_plan_to_rf_geometry(
        building=pipeline_result.building,
        coordinates=pipeline_result.coordinates,
        cell_size_m=CELL_SIZE_M,
    )

    return (
        pipeline_result,
        receiver_positions,
        valid_ap_positions,
        blocked_positions,
        physical_walls,
    )


# ============================================================
# EVALUATION
# ============================================================

def evaluate_production_solution(
    receiver_positions,
    physical_walls,
):
    """Run the production RF evaluation using floor-plan geometry."""

    access_points = create_access_points()

    evaluation = evaluate_grid(
        receiver_positions=receiver_positions,
        access_points=access_points,

        noise_floor_dbm=NOISE_FLOOR_DBM,
        minimum_rssi_dbm=MINIMUM_RSSI_DBM,
        minimum_sinr_db=MINIMUM_SINR_DB,

        reference_distance_m=REFERENCE_DISTANCE_M,
        path_loss_exponent=PATH_LOSS_EXPONENT,

        wall_material=WALL_MATERIAL,

        # Keep the legacy wall model disabled here.
        # Actual floor-plan wall attenuation is supplied below
        # through floor_plan_walls.
        number_of_walls=0,
        obstacle_loss_db=OBSTACLE_LOSS_DB,

        same_channel_factor=SAME_CHANNEL_FACTOR,
        adjacent_channel_factor=ADJACENT_CHANNEL_FACTOR,
        non_overlapping_factor=NON_OVERLAPPING_FACTOR,

        floor_plan_walls=physical_walls,
        floor_plan_wall_loss_db=FLOOR_PLAN_WALL_LOSS_DB,
    )

    return access_points, evaluation


# ============================================================
# COMMON HEATMAP
# ============================================================

def create_heatmap(
    evaluation,
    access_points,
    values,
    title,
    colorbar_label,
    filename,
    threshold=None,
):
    """Create an RF heatmap from receiver evaluation data."""

    if not values:
        raise ValueError(
            "Heatmap values cannot be empty."
        )

    x = [
        result.position[0]
        for result in evaluation.receiver_results
    ]

    y = [
        result.position[1]
        for result in evaluation.receiver_results
    ]

    fig, ax = plt.subplots(
        figsize=(11, 8)
    )

    scatter = ax.scatter(
        x,
        y,
        c=values,
        s=28,
        marker="s",
        cmap="turbo",
        norm=Normalize(
            vmin=min(values),
            vmax=max(values),
        ),
    )

    colorbar = fig.colorbar(
        scatter,
        ax=ax,
    )

    colorbar.set_label(
        colorbar_label,
        fontsize=11,
    )

    # AP markers.
    for index, ap in enumerate(
        access_points,
        start=1,
    ):
        ax.scatter(
            ap.x,
            ap.y,
            s=260,
            marker="^",
            edgecolors="black",
            linewidths=1.5,
            facecolors="white",
            zorder=10,
        )

        ax.text(
            ap.x,
            ap.y + 0.45,
            f"AP {index}\nCh {ap.channel}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            zorder=11,
        )

    if threshold is not None:
        ax.text(
            0.02,
            0.02,
            f"Threshold: {threshold}",
            transform=ax.transAxes,
            fontsize=10,
            bbox=dict(
                boxstyle="round",
                facecolor="white",
                alpha=0.85,
            ),
        )

    ax.set_xlim(
        0,
        PHYSICAL_WIDTH_M,
    )

    ax.set_ylim(
        0,
        PHYSICAL_HEIGHT_M,
    )

    ax.set_aspect("equal")

    ax.set_xlabel("X position (m)")
    ax.set_ylabel("Y position (m)")

    ax.set_title(
        title,
        fontsize=15,
        fontweight="bold",
    )

    ax.grid(
        True,
        alpha=0.2,
    )

    fig.tight_layout()

    output_path = OUTPUT_DIR / filename

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"Created: {output_path}"
    )


# ============================================================
# COVERAGE MAP
# ============================================================

def create_coverage_map(
    evaluation,
    access_points,
):
    """Create covered/dead-zone visualization."""

    x = [
        result.position[0]
        for result in evaluation.receiver_results
    ]

    y = [
        result.position[1]
        for result in evaluation.receiver_results
    ]

    values = [
        1 if result.covered else 0
        for result in evaluation.receiver_results
    ]

    fig, ax = plt.subplots(
        figsize=(11, 8)
    )

    scatter = ax.scatter(
        x,
        y,
        c=values,
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
        s=28,
        marker="s",
    )

    colorbar = fig.colorbar(
        scatter,
        ax=ax,
        ticks=[0, 1],
    )

    colorbar.ax.set_yticklabels(
        [
            "Dead zone",
            "Covered",
        ]
    )

    for index, ap in enumerate(
        access_points,
        start=1,
    ):
        ax.scatter(
            ap.x,
            ap.y,
            s=260,
            marker="^",
            edgecolors="black",
            linewidths=1.5,
            facecolors="white",
            zorder=10,
        )

        ax.text(
            ap.x,
            ap.y + 0.45,
            f"AP {index}\nCh {ap.channel}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            zorder=11,
        )

    ax.set_xlim(
        0,
        PHYSICAL_WIDTH_M,
    )

    ax.set_ylim(
        0,
        PHYSICAL_HEIGHT_M,
    )

    ax.set_aspect("equal")

    ax.set_xlabel("X position (m)")
    ax.set_ylabel("Y position (m)")

    ax.set_title(
        (
            "GeoTwin AI Indoor - Floor-Plan-Aware Coverage Map "
            f"({evaluation.coverage_percent:.2f}% coverage)"
        ),
        fontsize=15,
        fontweight="bold",
    )

    ax.grid(
        True,
        alpha=0.2,
    )

    fig.tight_layout()

    output_path = (
        OUTPUT_DIR
        / "coverage_map.png"
    )

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"Created: {output_path}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 75)
    print(
        "GeoTwin AI Indoor - Floor-Plan-Aware RF Heatmap Generator"
    )
    print("=" * 75)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not FLOOR_PLAN_PATH.exists():
        raise FileNotFoundError(
            f"Floor-plan file not found: {FLOOR_PLAN_PATH}"
        )

    print()
    print("[FLOOR PLAN]")
    print(
        f"Input: {FLOOR_PLAN_PATH}"
    )

    (
        pipeline_result,
        receiver_positions,
        valid_ap_positions,
        blocked_positions,
        physical_walls,
    ) = load_floor_plan_geometry()

    print(
        f"Image size: "
        f"{pipeline_result.image_width} x "
        f"{pipeline_result.image_height} px"
    )

    print(
        f"Physical size: "
        f"{PHYSICAL_WIDTH_M} m x "
        f"{PHYSICAL_HEIGHT_M} m"
    )

    print(
        f"Detected walls: "
        f"{pipeline_result.wall_count}"
    )

    print(
        f"Detected rooms: "
        f"{pipeline_result.room_count}"
    )

    print(
        f"Detected corridors: "
        f"{pipeline_result.corridor_count}"
    )

    print(
        f"RF receiver positions: "
        f"{len(receiver_positions)}"
    )

    print(
        f"Valid AP positions: "
        f"{len(valid_ap_positions)}"
    )

    print(
        f"Blocked positions: "
        f"{len(blocked_positions)}"
    )

    print(
        f"Physical walls: "
        f"{len(physical_walls)}"
    )

    print()
    print("[PRODUCTION AP CONFIGURATION]")

    for index, (position, channel) in enumerate(
        AP_CONFIGURATION,
        start=1,
    ):
        print(
            f"AP {index}: "
            f"position={position}, "
            f"channel={channel}"
        )

    print()
    print(
        "Running floor-plan-aware RF evaluation..."
    )

    (
        access_points,
        evaluation,
    ) = evaluate_production_solution(
        receiver_positions=receiver_positions,
        physical_walls=physical_walls,
    )

    print()
    print("[RF RESULTS]")

    print(
        f"Coverage       : "
        f"{evaluation.coverage_percent:.2f}%"
    )

    print(
        f"Average RSSI   : "
        f"{evaluation.average_rssi_dbm:.2f} dBm"
    )

    print(
        f"Minimum RSSI   : "
        f"{evaluation.minimum_rssi_dbm:.2f} dBm"
    )

    print(
        f"Average SINR   : "
        f"{evaluation.average_sinr_db:.2f} dB"
    )

    print(
        f"Minimum SINR   : "
        f"{evaluation.minimum_sinr_db:.2f} dB"
    )

    print()
    print(
        "Generating floor-plan-aware visualizations..."
    )

    rssi_values = [
        result.serving_rssi_dbm
        for result in evaluation.receiver_results
    ]

    sinr_values = [
        result.sinr_db
        for result in evaluation.receiver_results
    ]

    interference_values = [
        (
            result.interference_dbm
            if result.interference_dbm != float("-inf")
            else -120.0
        )
        for result in evaluation.receiver_results
    ]

    create_coverage_map(
        evaluation,
        access_points,
    )

    create_heatmap(
        evaluation=evaluation,
        access_points=access_points,
        values=rssi_values,
        title=(
            "GeoTwin AI Indoor - Floor-Plan-Aware RSSI Heatmap"
        ),
        colorbar_label="Serving RSSI (dBm)",
        filename="rssi_heatmap.png",
        threshold=(
            f"{MINIMUM_RSSI_DBM:.1f} dBm"
        ),
    )

    create_heatmap(
        evaluation=evaluation,
        access_points=access_points,
        values=sinr_values,
        title=(
            "GeoTwin AI Indoor - Floor-Plan-Aware SINR Heatmap"
        ),
        colorbar_label="SINR (dB)",
        filename="sinr_heatmap.png",
        threshold=(
            f"{MINIMUM_SINR_DB:.1f} dB"
        ),
    )

    create_heatmap(
        evaluation=evaluation,
        access_points=access_points,
        values=interference_values,
        title=(
            "GeoTwin AI Indoor - Floor-Plan-Aware "
            "Interference Heatmap"
        ),
        colorbar_label="Interference (dBm)",
        filename="interference_heatmap.png",
    )

    print()
    print("=" * 75)
    print(
        "Floor-plan-aware visualization generation complete."
    )
    print("=" * 75)

    print()
    print(
        f"Output directory: "
        f"{OUTPUT_DIR.resolve()}"
    )


if __name__ == "__main__":
    main()
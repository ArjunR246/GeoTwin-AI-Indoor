"""
GeoTwin AI Indoor - Week 7 Streamlit Dashboard

Complete dashboard for:

    Floor Plan Upload
          ↓
    OpenCV Floor-plan Processing
          ↓
    Wall / Room / Corridor Detection
          ↓
    Physical RF Geometry
          ↓
    Wall-aware RF Evaluation
          ↓
    Channel-aware NSGA-II Optimization
          ↓
    Optimization Results
          ↓
    Floor-plan-aware RSSI / SINR / Coverage / Interference Maps
          ↓
    Optimization History
"""

from __future__ import annotations

import sys
from pathlib import Path
from tempfile import NamedTemporaryFile

import cv2
import matplotlib

# Non-GUI backend for Streamlit/server environments.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))


# ============================================================
# GEOTWIN IMPORTS
# ============================================================

from geotwin.floorplan.pipeline import (
    process_floor_plan_file,
)

from geotwin.floorplan.rf_adapter import (
    floor_plan_to_rf_geometry,
)

from geotwin.optimization.nsga2 import (
    NSGA2Config,
    create_access_points,
    optimize,
)

from geotwin.rf.coverage import (
    evaluate_grid,
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="GeoTwin AI Indoor",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# SESSION STATE
# ============================================================

if "pipeline_result" not in st.session_state:
    st.session_state.pipeline_result = None

if "receiver_positions" not in st.session_state:
    st.session_state.receiver_positions = None

if "valid_ap_positions" not in st.session_state:
    st.session_state.valid_ap_positions = None

if "blocked_positions" not in st.session_state:
    st.session_state.blocked_positions = None

if "physical_walls" not in st.session_state:
    st.session_state.physical_walls = None

if "optimization_result" not in st.session_state:
    st.session_state.optimization_result = None

if "best_solution" not in st.session_state:
    st.session_state.best_solution = None

if "rf_evaluation" not in st.session_state:
    st.session_state.rf_evaluation = None


# ============================================================
# HELPER FUNCTIONS
# ============================================================


def decode_uploaded_image(uploaded_file) -> np.ndarray:
    """
    Decode the uploaded image into RGB NumPy format.

    Streamlit UploadedFile is converted directly from its
    in-memory bytes, so no permanent file is required.
    """

    if uploaded_file is None:
        raise ValueError(
            "No uploaded floor-plan image was provided."
        )

    image_bytes = uploaded_file.getvalue()

    if not image_bytes:
        raise ValueError(
            "Uploaded floor-plan image is empty."
        )

    image_array = np.frombuffer(
        image_bytes,
        dtype=np.uint8,
    )

    image_bgr = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR,
    )

    if image_bgr is None:
        raise ValueError(
            "OpenCV could not decode the uploaded floor plan."
        )

    image_rgb = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2RGB,
    )

    return image_rgb


def _get_receiver_value(
    result,
    metric: str,
) -> float:
    """
    Extract a numerical RF value from one ReceiverEvaluation.
    """

    if metric == "rssi":
        return float(
            result.serving_rssi_dbm
        )

    if metric == "sinr":
        return float(
            result.sinr_db
        )

    if metric == "interference":
        value = float(
            result.interference_dbm
        )

        if np.isneginf(value):
            return -120.0

        if not np.isfinite(value):
            return -120.0

        return value

    raise ValueError(
        f"Unsupported RF metric: {metric}"
    )


def _receiver_grid(
    receiver_results,
    metric: str,
):
    """
    Convert receiver evaluation points into a 2D physical grid.

    The RF adapter generates receiver positions in physical metres.
    Missing cells are left as NaN.
    """

    if not receiver_results:
        raise ValueError(
            "No receiver results are available."
        )

    points = []

    for result in receiver_results:

        x = float(
            result.position[0]
        )

        y = float(
            result.position[1]
        )

        value = _get_receiver_value(
            result,
            metric,
        )

        if not np.isfinite(value):
            continue

        points.append(
            (
                x,
                y,
                value,
            )
        )

    if not points:
        raise ValueError(
            "No finite RF values are available."
        )

    x_values = np.array(
        sorted(
            {
                point[0]
                for point in points
            }
        ),
        dtype=float,
    )

    y_values = np.array(
        sorted(
            {
                point[1]
                for point in points
            }
        ),
        dtype=float,
    )

    x_index = {
        value: index
        for index, value in enumerate(
            x_values
        )
    }

    y_index = {
        value: index
        for index, value in enumerate(
            y_values
        )
    }

    grid = np.full(
        (
            len(y_values),
            len(x_values),
        ),
        np.nan,
        dtype=float,
    )

    for x, y, value in points:

        grid[
            y_index[y],
            x_index[x],
        ] = value

    return (
        x_values,
        y_values,
        grid,
    )


def _calculate_grid_edges(
    values: np.ndarray,
):
    """
    Calculate cell edges from RF grid centre coordinates.
    """

    if len(values) == 1:

        half_step = 0.25

        return np.array(
            [
                values[0] - half_step,
                values[0] + half_step,
            ]
        )

    differences = np.diff(
        values
    )

    step = float(
        np.median(
            differences
        )
    )

    if step <= 0:
        step = 0.5

    interior = (
        values[:-1]
        + values[1:]
    ) / 2.0

    edges = np.concatenate(
        [
            [
                values[0] - step / 2.0
            ],
            interior,
            [
                values[-1] + step / 2.0
            ],
        ]
    )

    return edges


def _draw_floor_plan_background(
    ax,
    image_rgb: np.ndarray,
    physical_width_m: float,
    physical_height_m: float,
):
    """
    Draw the original uploaded floor plan.

    CoordinateTransform intentionally keeps the same coordinate
    orientation as the image-processing geometry. Therefore
    origin='lower' is used so pixel row 0 corresponds to the
    physical Y=0 side of the RF coordinate system.
    """

    ax.imshow(
        image_rgb,
        extent=[
            0.0,
            physical_width_m,
            0.0,
            physical_height_m,
        ],
        origin="lower",
        aspect="auto",
        zorder=0,
    )


def _draw_physical_walls(
    ax,
    physical_walls,
):
    """
    Draw detected floor-plan walls in physical coordinates.
    """

    if not physical_walls:
        return

    for wall in physical_walls:

        ax.plot(
            [
                float(wall.x1),
                float(wall.x2),
            ],
            [
                float(wall.y1),
                float(wall.y2),
            ],
            color="black",
            linewidth=2.5,
            solid_capstyle="round",
            zorder=10,
        )


def _draw_access_points(
    ax,
    best_solution,
):
    """
    Draw optimized AP positions and channel assignments.
    """

    if best_solution is None:
        return

    for index, (
        position,
        channel,
    ) in enumerate(
        best_solution.ap_configuration,
        start=1,
    ):

        x = float(
            position[0]
        )

        y = float(
            position[1]
        )

        # AP marker.
        ax.scatter(
            x,
            y,
            s=280,
            marker="^",
            facecolors="white",
            edgecolors="black",
            linewidths=2.0,
            zorder=20,
        )

        # AP label.
        ax.annotate(
            (
                f"AP {index}\n"
                f"Ch {channel}"
            ),
            xy=(
                x,
                y,
            ),
            xytext=(
                0,
                15,
            ),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color="black",
            bbox=dict(
                boxstyle="round,pad=0.25",
                facecolor="white",
                edgecolor="black",
                alpha=0.88,
            ),
            zorder=21,
        )


def _style_rf_axes(
    ax,
    physical_width_m: float,
    physical_height_m: float,
    title: str,
):
    """
    Apply common RF visualization formatting.
    """

    ax.set_xlim(
        0.0,
        physical_width_m,
    )

    ax.set_ylim(
        0.0,
        physical_height_m,
    )

    ax.set_xlabel(
        "X position (m)",
        fontsize=11,
    )

    ax.set_ylabel(
        "Y position (m)",
        fontsize=11,
    )

    ax.set_title(
        title,
        fontsize=16,
        fontweight="bold",
        pad=12,
    )

    ax.set_aspect(
        "equal",
        adjustable="box",
    )

    ax.grid(
        False
    )


def plot_floor_plan_heatmap(
    *,
    image_rgb: np.ndarray,
    receiver_results,
    physical_walls,
    best_solution,
    physical_width_m: float,
    physical_height_m: float,
    metric: str,
    title: str,
    colorbar_label: str,
    cmap: str = "turbo",
    alpha: float = 0.60,
):
    """
    Create a floor-plan-aware RF heatmap.

    Layers:

        1. Original floor plan
        2. Computed RF heatmap
        3. Detected physical walls
        4. Optimized AP locations
    """

    (
        x_values,
        y_values,
        grid,
    ) = _receiver_grid(
        receiver_results,
        metric,
    )

    x_edges = _calculate_grid_edges(
        x_values
    )

    y_edges = _calculate_grid_edges(
        y_values
    )

    masked_grid = np.ma.masked_invalid(
        grid
    )

    fig, ax = plt.subplots(
        figsize=(13, 8),
        dpi=120,
    )

    # --------------------------------------------------------
    # FLOOR PLAN
    # --------------------------------------------------------

    _draw_floor_plan_background(
        ax=ax,
        image_rgb=image_rgb,
        physical_width_m=physical_width_m,
        physical_height_m=physical_height_m,
    )

    # --------------------------------------------------------
    # RF HEATMAP
    # --------------------------------------------------------

    heatmap = ax.pcolormesh(
        x_edges,
        y_edges,
        masked_grid,
        cmap=cmap,
        shading="auto",
        alpha=alpha,
        zorder=3,
    )

    # --------------------------------------------------------
    # COLORBAR
    # --------------------------------------------------------

    colorbar = fig.colorbar(
        heatmap,
        ax=ax,
        pad=0.02,
    )

    colorbar.set_label(
        colorbar_label,
        fontsize=11,
    )

    # --------------------------------------------------------
    # WALLS
    # --------------------------------------------------------

    _draw_physical_walls(
        ax=ax,
        physical_walls=physical_walls,
    )

    # --------------------------------------------------------
    # APs
    # --------------------------------------------------------

    _draw_access_points(
        ax=ax,
        best_solution=best_solution,
    )

    # --------------------------------------------------------
    # AXES
    # --------------------------------------------------------

    _style_rf_axes(
        ax=ax,
        physical_width_m=physical_width_m,
        physical_height_m=physical_height_m,
        title=title,
    )

    fig.tight_layout()

    return fig


def plot_floor_plan_coverage(
    *,
    image_rgb: np.ndarray,
    receiver_results,
    physical_walls,
    best_solution,
    physical_width_m: float,
    physical_height_m: float,
    coverage_percent: float,
):
    """
    Create a floor-plan-aware coverage/dead-zone map.
    """

    if not receiver_results:
        raise ValueError(
            "No receiver results are available."
        )

    x_values = sorted(
        {
            float(
                result.position[0]
            )
            for result in receiver_results
        }
    )

    y_values = sorted(
        {
            float(
                result.position[1]
            )
            for result in receiver_results
        }
    )

    x_values = np.array(
        x_values,
        dtype=float,
    )

    y_values = np.array(
        y_values,
        dtype=float,
    )

    x_index = {
        value: index
        for index, value in enumerate(
            x_values
        )
    }

    y_index = {
        value: index
        for index, value in enumerate(
            y_values
        )
    }

    grid = np.full(
        (
            len(y_values),
            len(x_values),
        ),
        np.nan,
        dtype=float,
    )

    for result in receiver_results:

        x = float(
            result.position[0]
        )

        y = float(
            result.position[1]
        )

        grid[
            y_index[y],
            x_index[x],
        ] = (
            1.0
            if result.covered
            else 0.0
        )

    x_edges = _calculate_grid_edges(
        x_values
    )

    y_edges = _calculate_grid_edges(
        y_values
    )

    masked_grid = np.ma.masked_invalid(
        grid
    )

    fig, ax = plt.subplots(
        figsize=(13, 8),
        dpi=120,
    )

    # --------------------------------------------------------
    # FLOOR PLAN
    # --------------------------------------------------------

    _draw_floor_plan_background(
        ax=ax,
        image_rgb=image_rgb,
        physical_width_m=physical_width_m,
        physical_height_m=physical_height_m,
    )

    # --------------------------------------------------------
    # COVERAGE MAP
    # --------------------------------------------------------

    heatmap = ax.pcolormesh(
        x_edges,
        y_edges,
        masked_grid,
        cmap="RdYlGn",
        vmin=0.0,
        vmax=1.0,
        shading="auto",
        alpha=0.48,
        zorder=3,
    )

    # --------------------------------------------------------
    # COLORBAR
    # --------------------------------------------------------

    colorbar = fig.colorbar(
        heatmap,
        ax=ax,
        pad=0.02,
        ticks=[
            0.0,
            1.0,
        ],
    )

    colorbar.ax.set_yticklabels(
        [
            "Dead Zone",
            "Covered",
        ]
    )

    colorbar.set_label(
        "Coverage Status",
        fontsize=11,
    )

    # --------------------------------------------------------
    # WALLS
    # --------------------------------------------------------

    _draw_physical_walls(
        ax=ax,
        physical_walls=physical_walls,
    )

    # --------------------------------------------------------
    # APs
    # --------------------------------------------------------

    _draw_access_points(
        ax=ax,
        best_solution=best_solution,
    )

    # --------------------------------------------------------
    # AXES
    # --------------------------------------------------------

    _style_rf_axes(
        ax=ax,
        physical_width_m=physical_width_m,
        physical_height_m=physical_height_m,
        title=(
            "GeoTwin AI Indoor - "
            f"Floor-Plan Coverage Map "
            f"({coverage_percent:.2f}%)"
        ),
    )

    fig.tight_layout()

    return fig


# ============================================================
# TITLE
# ============================================================

st.title(
    "📡 GeoTwin AI Indoor"
)

st.markdown(
    """
### AI-powered indoor RF digital twin

Upload a floor plan, automatically extract its geometry,
convert it into physical RF coordinates, evaluate wall-aware
wireless propagation, and optimize access-point placement using
channel-aware NSGA-II.
"""
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header(
    "⚙️ Configuration"
)


# ------------------------------------------------------------
# PHYSICAL BUILDING
# ------------------------------------------------------------

physical_width_m = st.sidebar.number_input(
    "Building width (m)",
    min_value=0.1,
    value=20.0,
    step=0.5,
)

physical_height_m = st.sidebar.number_input(
    "Building height / length (m)",
    min_value=0.1,
    value=15.0,
    step=0.5,
)

cell_size_m = st.sidebar.number_input(
    "RF grid resolution (m)",
    min_value=0.1,
    value=0.5,
    step=0.1,
)


st.sidebar.divider()


# ------------------------------------------------------------
# RF PARAMETERS
# ------------------------------------------------------------

st.sidebar.subheader(
    "RF Parameters"
)

minimum_rssi_dbm = st.sidebar.number_input(
    "Minimum RSSI (dBm)",
    value=-67.0,
    step=1.0,
)

minimum_sinr_db = st.sidebar.number_input(
    "Minimum SINR (dB)",
    value=20.0,
    step=1.0,
)

noise_floor_dbm = st.sidebar.number_input(
    "Noise floor (dBm)",
    value=-95.0,
    step=1.0,
)

frequency_hz = st.sidebar.number_input(
    "Frequency (Hz)",
    value=2.4e9,
    step=1.0e8,
    format="%.0f",
)

tx_power_dbm = st.sidebar.number_input(
    "TX power (dBm)",
    value=20.0,
    step=1.0,
)

antenna_gain_dbi = st.sidebar.number_input(
    "Antenna gain (dBi)",
    value=3.0,
    step=0.5,
)

path_loss_exponent = st.sidebar.number_input(
    "Path-loss exponent",
    min_value=0.1,
    value=3.0,
    step=0.1,
)

wall_loss_db = st.sidebar.number_input(
    "Floor-plan wall loss (dB)",
    min_value=0.0,
    value=6.0,
    step=1.0,
)


st.sidebar.divider()


# ------------------------------------------------------------
# NSGA-II PARAMETERS
# ------------------------------------------------------------

st.sidebar.subheader(
    "NSGA-II"
)

population_size = st.sidebar.number_input(
    "Population size",
    min_value=2,
    value=10,
    step=1,
)

generations = st.sidebar.number_input(
    "Generations",
    min_value=1,
    value=3,
    step=1,
)

min_ap_count = st.sidebar.number_input(
    "Minimum AP count",
    min_value=1,
    value=1,
    step=1,
)

max_ap_count = st.sidebar.number_input(
    "Maximum AP count",
    min_value=1,
    value=5,
    step=1,
)

random_seed = st.sidebar.number_input(
    "Random seed",
    value=42,
    step=1,
)

st.sidebar.caption(
    "Channels: 1, 6, 11"
)


# ============================================================
# FLOOR PLAN UPLOAD
# ============================================================

st.header(
    "1️⃣ Floor Plan"
)

uploaded_file = st.file_uploader(
    "Upload a floor-plan image",
    type=[
        "png",
        "jpg",
        "jpeg",
        "bmp",
        "tif",
        "tiff",
    ],
)


if uploaded_file is not None:

    # --------------------------------------------------------
    # SHOW ORIGINAL FLOOR PLAN
    # --------------------------------------------------------

    st.image(
        uploaded_file,
        caption="Uploaded floor plan",
        width="stretch",
    )

    process_button = st.button(
        "🔍 Process Floor Plan",
        type="primary",
        width="stretch",
    )

    if process_button:

        temporary_path = None

        try:

            suffix = Path(
                uploaded_file.name
            ).suffix

            if not suffix:
                suffix = ".png"

            with NamedTemporaryFile(
                delete=False,
                suffix=suffix,
            ) as temporary_file:

                temporary_file.write(
                    uploaded_file.getbuffer()
                )

                temporary_path = Path(
                    temporary_file.name
                )

            with st.spinner(
                "Processing floor plan..."
            ):

                pipeline_result = (
                    process_floor_plan_file(
                        temporary_path,
                        physical_width_m=(
                            physical_width_m
                        ),
                        physical_height_m=(
                            physical_height_m
                        ),
                    )
                )

                (
                    receiver_positions,
                    valid_ap_positions,
                    blocked_positions,
                    physical_walls,
                ) = floor_plan_to_rf_geometry(
                    building=(
                        pipeline_result.building
                    ),
                    coordinates=(
                        pipeline_result.coordinates
                    ),
                    cell_size_m=cell_size_m,
                )

            # ------------------------------------------------
            # SAVE PIPELINE STATE
            # ------------------------------------------------

            st.session_state.pipeline_result = (
                pipeline_result
            )

            st.session_state.receiver_positions = (
                receiver_positions
            )

            st.session_state.valid_ap_positions = (
                valid_ap_positions
            )

            st.session_state.blocked_positions = (
                blocked_positions
            )

            st.session_state.physical_walls = (
                physical_walls
            )

            # Reset optimization results because a new
            # floor plan has been processed.
            st.session_state.optimization_result = (
                None
            )

            st.session_state.best_solution = (
                None
            )

            st.session_state.rf_evaluation = (
                None
            )

            st.success(
                "Floor plan processed successfully."
            )

        except Exception as exc:

            st.error(
                f"Floor-plan processing failed: {exc}"
            )

        finally:

            if (
                temporary_path is not None
                and temporary_path.exists()
            ):

                temporary_path.unlink(
                    missing_ok=True
                )


# ============================================================
# FLOOR PLAN INFORMATION
# ============================================================

pipeline_result = (
    st.session_state.pipeline_result
)


if pipeline_result is not None:

    st.header(
        "2️⃣ Floor Plan Analysis"
    )

    col1, col2, col3, col4 = (
        st.columns(4)
    )

    col1.metric(
        "Image Width",
        f"{pipeline_result.image_width} px",
    )

    col2.metric(
        "Image Height",
        f"{pipeline_result.image_height} px",
    )

    col3.metric(
        "Detected Walls",
        pipeline_result.wall_count,
    )

    col4.metric(
        "Detected Rooms",
        pipeline_result.room_count,
    )

    col5, col6, col7, col8 = (
        st.columns(4)
    )

    col5.metric(
        "Corridors",
        pipeline_result.corridor_count,
    )

    col6.metric(
        "RF Receiver Points",
        len(
            st.session_state.receiver_positions
        ),
    )

    col7.metric(
        "Valid AP Positions",
        len(
            st.session_state.valid_ap_positions
        ),
    )

    col8.metric(
        "Physical Walls",
        len(
            st.session_state.physical_walls
        ),
    )


# ============================================================
# RF + OPTIMIZATION
# ============================================================

if pipeline_result is not None:

    st.header(
        "3️⃣ Optimization"
    )

    optimize_button = st.button(
        "🚀 Run NSGA-II Optimization",
        type="primary",
        width="stretch",
    )

    if optimize_button:

        try:

            receiver_positions = (
                st.session_state.receiver_positions
            )

            valid_ap_positions = (
                st.session_state.valid_ap_positions
            )

            blocked_positions = (
                st.session_state.blocked_positions
            )

            physical_walls = (
                st.session_state.physical_walls
            )

            # ------------------------------------------------
            # NSGA-II CONFIGURATION
            # ------------------------------------------------

            config = NSGA2Config(
                population_size=int(
                    population_size
                ),
                generations=int(
                    generations
                ),
                crossover_probability=0.9,
                mutation_probability=0.1,
                min_ap_count=int(
                    min_ap_count
                ),
                max_ap_count=int(
                    max_ap_count
                ),
                random_seed=int(
                    random_seed
                ),
                minimum_rssi_dbm=(
                    minimum_rssi_dbm
                ),
                minimum_sinr_db=(
                    minimum_sinr_db
                ),
                noise_floor_dbm=(
                    noise_floor_dbm
                ),
                frequency_hz=(
                    frequency_hz
                ),
                tx_power_dbm=(
                    tx_power_dbm
                ),
                antenna_gain_dbi=(
                    antenna_gain_dbi
                ),
                available_channels=(
                    1,
                    6,
                    11,
                ),
                reference_distance_m=1.0,
                path_loss_exponent=(
                    path_loss_exponent
                ),
                wall_material="concrete",

                # Disable the old global wall model.
                # Actual detected floor-plan walls are passed
                # through floor_plan_walls below.
                number_of_walls=0,

                obstacle_loss_db=6.0,

                same_channel_factor=1.0,
                adjacent_channel_factor=0.5,
                non_overlapping_factor=0.0,
            )

            config.validate()

            # ------------------------------------------------
            # RUN OPTIMIZATION
            # ------------------------------------------------

            with st.spinner(
                "Running channel-aware NSGA-II..."
            ):

                optimization_result = optimize(
                    receiver_positions=(
                        receiver_positions
                    ),
                    valid_positions=(
                        valid_ap_positions
                    ),
                    blocked_positions=(
                        blocked_positions
                    ),
                    config=config,
                    floor_plan_walls=(
                        physical_walls
                    ),
                    floor_plan_wall_loss_db=(
                        wall_loss_db
                    ),
                )

            if not optimization_result.pareto_front:

                raise RuntimeError(
                    "Optimization returned an empty Pareto front."
                )

            # ------------------------------------------------
            # SELECT BEST SOLUTION
            # ------------------------------------------------

            best_solution = min(
                optimization_result.pareto_front,
                key=lambda individual: (
                    individual.objectives[0],
                    individual.objectives[1],
                    individual.objectives[2],
                ),
            )

            # ------------------------------------------------
            # FINAL RF EVALUATION
            # ------------------------------------------------

            access_points = create_access_points(
                best_solution,
                config,
            )

            rf_evaluation = evaluate_grid(
                receiver_positions=(
                    receiver_positions
                ),
                access_points=(
                    access_points
                ),
                noise_floor_dbm=(
                    noise_floor_dbm
                ),
                minimum_rssi_dbm=(
                    minimum_rssi_dbm
                ),
                minimum_sinr_db=(
                    minimum_sinr_db
                ),
                reference_distance_m=1.0,
                path_loss_exponent=(
                    path_loss_exponent
                ),
                wall_material="concrete",

                # Legacy wall model disabled.
                number_of_walls=0,

                obstacle_loss_db=6.0,

                same_channel_factor=1.0,
                adjacent_channel_factor=0.5,
                non_overlapping_factor=0.0,

                # Actual detected physical walls.
                floor_plan_walls=(
                    physical_walls
                ),

                floor_plan_wall_loss_db=(
                    wall_loss_db
                ),
            )

            # ------------------------------------------------
            # SAVE RESULTS
            # ------------------------------------------------

            st.session_state.optimization_result = (
                optimization_result
            )

            st.session_state.best_solution = (
                best_solution
            )

            st.session_state.rf_evaluation = (
                rf_evaluation
            )

            st.success(
                "NSGA-II optimization completed."
            )

        except Exception as exc:

            st.error(
                f"Optimization failed: {exc}"
            )


# ============================================================
# RESULTS
# ============================================================

optimization_result = (
    st.session_state.optimization_result
)

best_solution = (
    st.session_state.best_solution
)

rf_evaluation = (
    st.session_state.rf_evaluation
)


if (
    optimization_result is not None
    and best_solution is not None
    and rf_evaluation is not None
):

    st.header(
        "4️⃣ Optimization Results"
    )

    # --------------------------------------------------------
    # MAIN METRICS
    # --------------------------------------------------------

    col1, col2, col3, col4 = (
        st.columns(4)
    )

    col1.metric(
        "Coverage",
        (
            f"{rf_evaluation.coverage_percent:.2f}%"
        ),
    )

    col2.metric(
        "Dead Zone",
        (
            f"{rf_evaluation.dead_zone_percent:.2f}%"
        ),
    )

    col3.metric(
        "Average RSSI",
        (
            f"{rf_evaluation.average_rssi_dbm:.2f} dBm"
        ),
    )

    col4.metric(
        "Average SINR",
        (
            f"{rf_evaluation.average_sinr_db:.2f} dB"
        ),
    )

    col5, col6, col7, col8 = (
        st.columns(4)
    )

    col5.metric(
        "Minimum RSSI",
        (
            f"{rf_evaluation.minimum_rssi_dbm:.2f} dBm"
        ),
    )

    col6.metric(
        "Minimum SINR",
        (
            f"{rf_evaluation.minimum_sinr_db:.2f} dB"
        ),
    )

    col7.metric(
        "Best AP Count",
        best_solution.ap_count,
    )

    col8.metric(
        "Pareto Solutions",
        len(
            optimization_result.pareto_front
        ),
    )

    # --------------------------------------------------------
    # BEST AP CONFIGURATION
    # --------------------------------------------------------

    st.subheader(
        "📍 Best AP Configuration"
    )

    ap_rows = []

    for index, (
        position,
        channel,
    ) in enumerate(
        best_solution.ap_configuration,
        start=1,
    ):

        ap_rows.append(
            {
                "AP": index,
                "X (m)": round(
                    float(position[0]),
                    4,
                ),
                "Y (m)": round(
                    float(position[1]),
                    4,
                ),
                "Channel": channel,
            }
        )

    st.dataframe(
        ap_rows,
        width="stretch",
        hide_index=True,
    )

    # --------------------------------------------------------
    # PARETO FRONT
    # --------------------------------------------------------

    st.subheader(
        "📊 Pareto Front"
    )

    pareto_rows = []

    for index, individual in enumerate(
        optimization_result.pareto_front,
        start=1,
    ):

        pareto_rows.append(
            {
                "Solution": index,
                "AP Count": individual.ap_count,
                "Coverage (%)": round(
                    -individual.objectives[0],
                    4,
                ),
                "Interference Objective": round(
                    individual.objectives[2],
                    6,
                ),
                "Channels": str(
                    individual.channels
                ),
            }
        )

    st.dataframe(
        pareto_rows,
        width="stretch",
        hide_index=True,
    )


# ============================================================
# RF VISUALIZATION
# ============================================================

if (
    rf_evaluation is not None
    and best_solution is not None
    and st.session_state.physical_walls is not None
    and uploaded_file is not None
):

    st.header(
        "5️⃣ RF Visualization"
    )

    # --------------------------------------------------------
    # LOAD ORIGINAL FLOOR PLAN
    # --------------------------------------------------------

    try:

        uploaded_image_rgb = (
            decode_uploaded_image(
                uploaded_file
            )
        )

    except Exception as exc:

        st.error(
            f"Unable to decode floor plan for RF visualization: {exc}"
        )

        uploaded_image_rgb = None


    if uploaded_image_rgb is not None:

        receiver_results = (
            rf_evaluation.receiver_results
        )

        physical_walls = (
            st.session_state.physical_walls
        )

        # ====================================================
        # RSSI HEATMAP
        # ====================================================

        st.subheader(
            "📶 RSSI Heatmap"
        )

        st.caption(
            "Computed RSSI over the detected floor-plan "
            "geometry with physical walls shown in black."
        )

        try:

            fig = plot_floor_plan_heatmap(
                image_rgb=(
                    uploaded_image_rgb
                ),
                receiver_results=(
                    receiver_results
                ),
                physical_walls=(
                    physical_walls
                ),
                best_solution=(
                    best_solution
                ),
                physical_width_m=(
                    physical_width_m
                ),
                physical_height_m=(
                    physical_height_m
                ),
                metric="rssi",
                title=(
                    "GeoTwin AI Indoor - "
                    "Floor-Plan RSSI Heatmap"
                ),
                colorbar_label="RSSI (dBm)",
                cmap="turbo",
                alpha=0.60,
            )

            st.pyplot(
                fig,
                width="stretch",
            )

            plt.close(fig)

        except Exception as exc:

            st.error(
                f"RSSI visualization failed: {exc}"
            )


        # ====================================================
        # SINR HEATMAP
        # ====================================================

        st.subheader(
            "📡 SINR Heatmap"
        )

        st.caption(
            "Computed SINR over the same physical RF grid."
        )

        try:

            fig = plot_floor_plan_heatmap(
                image_rgb=(
                    uploaded_image_rgb
                ),
                receiver_results=(
                    receiver_results
                ),
                physical_walls=(
                    physical_walls
                ),
                best_solution=(
                    best_solution
                ),
                physical_width_m=(
                    physical_width_m
                ),
                physical_height_m=(
                    physical_height_m
                ),
                metric="sinr",
                title=(
                    "GeoTwin AI Indoor - "
                    "Floor-Plan SINR Heatmap"
                ),
                colorbar_label="SINR (dB)",
                cmap="turbo",
                alpha=0.60,
            )

            st.pyplot(
                fig,
                width="stretch",
            )

            plt.close(fig)

        except Exception as exc:

            st.error(
                f"SINR visualization failed: {exc}"
            )


        # ====================================================
        # COVERAGE MAP
        # ====================================================

        st.subheader(
            "🟢 Coverage / Dead-Zone Map"
        )

        st.caption(
            "Green = RF coverage satisfies the RSSI and SINR "
            "thresholds. Red = dead zone."
        )

        try:

            fig = plot_floor_plan_coverage(
                image_rgb=(
                    uploaded_image_rgb
                ),
                receiver_results=(
                    receiver_results
                ),
                physical_walls=(
                    physical_walls
                ),
                best_solution=(
                    best_solution
                ),
                physical_width_m=(
                    physical_width_m
                ),
                physical_height_m=(
                    physical_height_m
                ),
                coverage_percent=(
                    rf_evaluation.coverage_percent
                ),
            )

            st.pyplot(
                fig,
                width="stretch",
            )

            plt.close(fig)

        except Exception as exc:

            st.error(
                f"Coverage visualization failed: {exc}"
            )


        # ====================================================
        # INTERFERENCE HEATMAP
        # ====================================================

        st.subheader(
            "⚡ Interference Heatmap"
        )

        st.caption(
            "Computed channel-aware interference over the "
            "floor-plan RF grid."
        )

        try:

            fig = plot_floor_plan_heatmap(
                image_rgb=(
                    uploaded_image_rgb
                ),
                receiver_results=(
                    receiver_results
                ),
                physical_walls=(
                    physical_walls
                ),
                best_solution=(
                    best_solution
                ),
                physical_width_m=(
                    physical_width_m
                ),
                physical_height_m=(
                    physical_height_m
                ),
                metric="interference",
                title=(
                    "GeoTwin AI Indoor - "
                    "Floor-Plan Interference Heatmap"
                ),
                colorbar_label=(
                    "Interference (dBm)"
                ),
                cmap="turbo",
                alpha=0.60,
            )

            st.pyplot(
                fig,
                width="stretch",
            )

            plt.close(fig)

        except Exception as exc:

            st.error(
                f"Interference visualization failed: {exc}"
            )


# ============================================================
# OPTIMIZATION HISTORY
# ============================================================

if optimization_result is not None:

    st.header(
        "6️⃣ Optimization History"
    )

    history = (
        optimization_result.history
    )

    if history:

        generations_history = [
            item["generation"]
            for item in history
        ]

        pareto_sizes = [
            item["pareto_size"]
            for item in history
        ]

        fig, ax = plt.subplots(
            figsize=(13, 7),
            dpi=120,
        )

        ax.plot(
            generations_history,
            pareto_sizes,
            marker="o",
            linewidth=2,
        )

        ax.set_xlabel(
            "Generation"
        )

        ax.set_ylabel(
            "Pareto Front Size"
        )

        ax.set_title(
            "NSGA-II Pareto Front Evolution",
            fontsize=16,
            fontweight="bold",
        )

        ax.grid(
            True,
            alpha=0.25,
        )

        fig.tight_layout()

        st.pyplot(
            fig,
            width="stretch",
        )

        plt.close(fig)

        # ----------------------------------------------------
        # HISTORY TABLE
        # ----------------------------------------------------

        with st.expander(
            "View Optimization History Data"
        ):

            st.dataframe(
                history,
                width="stretch",
                hide_index=True,
            )

    else:

        st.info(
            "No optimization history is available."
        )


# ============================================================
# WALL ATTENUATION INFORMATION
# ============================================================

if (
    pipeline_result is not None
    and st.session_state.physical_walls is not None
):

    st.header(
        "7️⃣ Floor-Plan Wall Attenuation"
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Detected Physical Walls",
        len(
            st.session_state.physical_walls
        ),
    )

    col2.metric(
        "Wall Loss / Crossing",
        f"{wall_loss_db:.1f} dB",
    )

    col3.metric(
        "Propagation Model",
        "Wall-aware",
    )

    st.info(
        "The RF evaluation uses the detected physical floor-plan "
        "walls rather than the legacy global wall-count model. "
        "Each RF path is evaluated against the supplied wall "
        "geometry."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "GeoTwin AI Indoor • Week 7 • Automated Digital Twin "
    "• Wall-aware RF • Channel-aware NSGA-II"
)
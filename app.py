"""Single-screen GeoTwin UI for floorplan upload and AP placement planning."""

from __future__ import annotations

import sys
from io import BytesIO
from pathlib import Path
from tempfile import NamedTemporaryFile

import cv2
import matplotlib
import numpy as np
import streamlit as st

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent
for import_path in (PROJECT_ROOT / "frontend/src", PROJECT_ROOT / "backend"):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from geotwin.floorplan.pipeline import process_floor_plan_file
from geotwin.floorplan.rf_adapter import floor_plan_to_rf_geometry
from services.floorplan_coverage import (
    coverage_metrics,
    floorplan_walls_to_backend,
    grid_edges,
    heatmap_grid,
    load_prediction_artifacts,
    optimize_access_points,
    predict_heatmap,
    smooth_heatmap_grid,
)


st.set_page_config(
    page_title="GeoTwin AI Indoor",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner=False)
def get_prediction_artifacts():
    """Cache the model across Streamlit reruns."""
    return load_prediction_artifacts()


def decode_image(uploaded_file) -> np.ndarray:
    """Decode an uploaded floorplan into an RGB image for plotting."""
    image = cv2.imdecode(
        np.frombuffer(uploaded_file.getvalue(), dtype=np.uint8), cv2.IMREAD_COLOR
    )
    if image is None:
        raise ValueError("The uploaded file is not a supported floorplan image.")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def generate_plan(
    uploaded_file,
    building_width_m: float,
    building_height_m: float,
    grid_resolution_m: float,
    ap_count: int,
    tx_power_dbm: float,
    frequency_ghz: float,
    coverage_threshold_dbm: float,
    wall_loss_db: float,
):
    """Process one plan, optimize AP placement, then predict model RSSI."""
    suffix = Path(uploaded_file.name).suffix or ".png"
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(delete=False, suffix=suffix) as temporary_file:
            temporary_file.write(uploaded_file.getbuffer())
            temporary_path = Path(temporary_file.name)

        pipeline_result = process_floor_plan_file(
            temporary_path,
            physical_width_m=building_width_m,
            physical_height_m=building_height_m,
        )
        (
            receiver_positions,
            valid_ap_positions,
            blocked_positions,
            physical_walls,
        ) = floor_plan_to_rf_geometry(
            building=pipeline_result.building,
            coordinates=pipeline_result.coordinates,
            cell_size_m=grid_resolution_m,
        )
        best_solution, access_points = optimize_access_points(
            receiver_positions=receiver_positions,
            valid_ap_positions=valid_ap_positions,
            blocked_positions=blocked_positions,
            physical_walls=physical_walls,
            ap_count=ap_count,
            tx_power_dbm=tx_power_dbm,
            frequency_ghz=frequency_ghz,
            minimum_rssi_dbm=coverage_threshold_dbm,
            wall_loss_db=wall_loss_db,
        )
        walls = floorplan_walls_to_backend(physical_walls, wall_loss_db)
        prediction = predict_heatmap(
            access_points=access_points,
            receiver_positions=receiver_positions,
            walls=walls,
            artifacts=get_prediction_artifacts(),
            tx_power_dbm=tx_power_dbm,
            frequency_ghz=frequency_ghz,
        )
        smooth_x, smooth_y, smooth_rssi = smooth_heatmap_grid(
            access_points=access_points,
            receiver_positions=receiver_positions,
            walls=walls,
            artifacts=get_prediction_artifacts(),
            tx_power_dbm=tx_power_dbm,
            frequency_ghz=frequency_ghz,
            width_m=building_width_m,
            height_m=building_height_m,
            cell_size_m=grid_resolution_m,
        )
        return {
            "image_rgb": decode_image(uploaded_file),
            "smooth_heatmap": (smooth_x, smooth_y, smooth_rssi),
            "pipeline_result": pipeline_result,
            "receiver_positions": receiver_positions,
            "physical_walls": physical_walls,
            "best_solution": best_solution,
            "access_points": access_points,
            "prediction": prediction,
            "metrics": coverage_metrics(prediction.rssi_dbm, coverage_threshold_dbm),
            "coverage_threshold_dbm": coverage_threshold_dbm,
        }
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def render_heatmap(
    result: dict[str, object], width_m: float, height_m: float, show_walls: bool = False
):
    """Render the final AI heatmap directly over the uploaded floorplan."""
    x_values, y_values, rssi_grid = result["smooth_heatmap"]
    fig, axis = plt.subplots(figsize=(13, 8), dpi=150)
    axis.imshow(
        result["image_rgb"],
        extent=[0, width_m, height_m, 0],
        origin="upper",
        aspect="auto",
        zorder=0,
    )
    step = float(x_values[1] - x_values[0]) if len(x_values) > 1 else 1.0
    heatmap = axis.imshow(
        np.ma.masked_invalid(rssi_grid),
        extent=[0, len(x_values) * step, len(y_values) * step, 0],
        origin="upper",
        cmap="turbo",
        vmin=-90,
        vmax=-25,
        alpha=0.6,
        interpolation="bilinear",
        zorder=2,
    )
    contours = axis.contour(
        x_values, y_values, np.ma.masked_invalid(rssi_grid),
        levels=sorted({-75, result["coverage_threshold_dbm"], -60, -50, -40}),
        colors="white", linewidths=1.0, linestyles="solid", zorder=3,
    )
    axis.clabel(contours, fmt="%d dBm", fontsize=7)
    if show_walls:
        for wall in result["physical_walls"]:
            axis.plot(
                [wall.x1, wall.x2], [wall.y1, wall.y2],
                color="#172033", linewidth=1.0, alpha=0.7, zorder=4,
            )
    for index, access_point in enumerate(result["access_points"], start=1):
        axis.scatter(
            access_point.x, access_point.y, marker="*", s=300,
            color="#ffffff", edgecolor="#111827", linewidth=1.4, zorder=6,
        )
        axis.annotate(
            f"AP {index}\nCh {access_point.channel}",
            (access_point.x, access_point.y), xytext=(0, 14),
            textcoords="offset points", ha="center", fontsize=9,
            fontweight="bold", color="#111827",
            bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "none", "alpha": 0.9},
            zorder=7,
        )
    colorbar = fig.colorbar(heatmap, ax=axis, pad=0.015)
    colorbar.set_label("Predicted RSSI (dBm)")
    axis.set(
        title="Optimized AP Placement and AI-Predicted RSSI",
        xlabel="X position (m)", ylabel="Y position (m, from top edge)",
        xlim=(0, width_m), ylim=(height_m, 0),
    )
    axis.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    return fig


def figure_bytes(figure) -> bytes:
    buffer = BytesIO()
    figure.savefig(buffer, format="png", dpi=200, bbox_inches="tight")
    return buffer.getvalue()


st.markdown(
    """
    <style>
      .block-container {max-width: 1420px; padding-top: 2rem;}
      [data-testid="stMetric"] {background: #f8fafc; border: 1px solid #e2e8f0;
        border-radius: 12px; padding: 12px;}
    </style>
    """,
    unsafe_allow_html=True,
)

if "plan_result" not in st.session_state:
    st.session_state.plan_result = None

st.title("📡 GeoTwin AI Indoor")
st.caption(
    "Upload one floorplan. GeoTwin extracts the structure, recommends access-point "
    "locations, and overlays an uncertainty-aware RSSI heatmap on the original plan."
)

with st.sidebar:
    st.header("Plan settings")
    building_width_m = st.number_input("Floor width (m)", min_value=1.0, value=20.0, step=1.0)
    building_height_m = st.number_input("Floor height (m)", min_value=1.0, value=15.0, step=1.0)
    ap_count = st.slider("Number of APs", min_value=1, max_value=4, value=2)
    grid_resolution_m = st.select_slider(
        "Heatmap detail", options=[0.5, 1.0, 1.5, 2.0], value=1.0,
        help="Smaller cells provide more detail but take longer to optimize.",
    )
    with st.expander("Advanced RF settings"):
        tx_power_dbm = st.number_input("TX power (dBm)", value=20.0, step=1.0)
        frequency_ghz = st.selectbox("Frequency (GHz)", options=[2.4, 5.0, 6.0], index=1)
        coverage_threshold_dbm = st.number_input("Coverage threshold (dBm)", value=-67.0, step=1.0)
        wall_loss_db = st.number_input("Detected-wall loss (dB)", min_value=0.0, value=6.0, step=0.5)
    st.divider()
    st.caption("The final map is generated by the trained XGBoost RSSI model. If a field calibration exists in `backend/models/`, it is applied automatically.")

uploaded_file = st.file_uploader(
    "Upload a single floorplan", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"]
)

if uploaded_file is None:
    st.info("Start by uploading one clear, top-down floorplan image.")
else:
    preview, action = st.columns([1.5, 1])
    with preview:
        st.image(uploaded_file, caption="Uploaded floorplan", width="stretch")
    with action:
        st.subheader("Ready to plan")
        st.write("GeoTwin will detect walls, find valid AP points, optimize their placement, and create the final heatmap.")
        if st.button("Generate AP plan", type="primary", width="stretch"):
            try:
                with st.spinner("Extracting geometry, optimizing AP locations, and predicting coverage..."):
                    st.session_state.plan_result = generate_plan(
                        uploaded_file, building_width_m, building_height_m,
                        grid_resolution_m, ap_count, tx_power_dbm, frequency_ghz,
                        coverage_threshold_dbm, wall_loss_db,
                    )
                st.success("Your AP plan is ready.")
            except Exception as error:
                st.session_state.plan_result = None
                st.error(f"Unable to generate the AP plan: {error}")

result = st.session_state.plan_result
if result is not None:
    st.divider()
    metrics = result["metrics"]
    mean_uncertainty = float(np.mean(result["prediction"].uncertainty_db))
    metric_columns = st.columns(4)
    metric_columns[0].metric("Coverage", f"{metrics['coverage_percent']:.1f}%")
    metric_columns[1].metric("Average RSSI", f"{metrics['average_rssi_dbm']:.1f} dBm")
    metric_columns[2].metric("Dead zones", f"{metrics['dead_zone_percent']:.1f}%")
    metric_columns[3].metric("Typical uncertainty", f"±{mean_uncertainty:.1f} dB")

    show_walls = st.checkbox("Show detected walls", value=False)
    figure = render_heatmap(result, building_width_m, building_height_m, show_walls)
    st.pyplot(figure, width="stretch")
    st.download_button(
        "Download heatmap PNG", data=figure_bytes(figure),
        file_name="geotwin_ap_heatmap.png", mime="image/png",
    )
    plt.close(figure)

    st.subheader("Recommended AP locations")
    st.dataframe([
        {
            "AP": index,
            "X (m)": round(float(access_point.x), 2),
            "Y (m)": round(float(access_point.y), 2),
            "Channel": access_point.channel,
        }
        for index, access_point in enumerate(result["access_points"], start=1)
    ], hide_index=True, width="stretch")

    with st.expander("Analysis details"):
        pipeline = result["pipeline_result"]
        detail_columns = st.columns(4)
        detail_columns[0].metric("Detected walls", pipeline.wall_count)
        detail_columns[1].metric("Detected rooms", pipeline.room_count)
        detail_columns[2].metric("Receiver points", len(result["receiver_positions"]))
        detail_columns[3].metric("APs selected", len(result["access_points"]))
        st.caption(
            "AP placement is optimized against the extracted physical geometry. The displayed RSSI map is generated by the trained backend predictor using the detected wall crossings."
        )

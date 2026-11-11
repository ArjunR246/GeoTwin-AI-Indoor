"""Single-screen GeoTwin UI for floorplan upload and AP placement planning."""

from __future__ import annotations

import json
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
from services.fast_placement import plan_access_points
from services.scan_import import (
    band_overlay,
    combine_scans,
    decode_scan,
    draw_points,
    fit_alignment,
    walked_area_summary,
    warp_scan_to_plan,
)
from services.floorplan_coverage import (
    coverage_metrics,
    floorplan_walls_to_backend,
    grid_edges,
    heatmap_grid,
    load_prediction_artifacts,
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
            indoor_mask=pipeline_result.indoor_mask,
        )
        walls = floorplan_walls_to_backend(physical_walls, wall_loss_db)
        best_solution, access_points = plan_access_points(
            receiver_positions=receiver_positions,
            walls=walls,
            artifacts=get_prediction_artifacts(),
            ap_count=ap_count,
            tx_power_dbm=tx_power_dbm,
            frequency_ghz=frequency_ghz,
            threshold_dbm=coverage_threshold_dbm,
            allowed_ap_positions=valid_ap_positions,
        )
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


# ---------------------------------------------------------------------------
# Measured scan (WiFiman) import
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def decode_scan_bytes(data: bytes):
    """Decode an uploaded WiFiman image once per file."""
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("The scan image could not be read.")
    return image, decode_scan(image)


def clicked_point(component_value, natural_size, state_key):
    """Convert a click to image pixels; return None if it was already used."""
    if not component_value:
        return None
    last = st.session_state.setdefault("last_clicks", {})
    stamp = component_value.get("unix_time")
    if last.get(state_key) == stamp:
        return None
    last[state_key] = stamp
    natural_width, natural_height = natural_size
    shown_width = component_value.get("width") or natural_width
    shown_height = component_value.get("height") or natural_height
    return (
        float(component_value["x"]) * natural_width / shown_width,
        float(component_value["y"]) * natural_height / shown_height,
    )


def band_bar_figure(summary):
    """One 100% bar split into the signal bands (and dead zones)."""
    figure, axis = plt.subplots(figsize=(10, 1.6), dpi=120)
    left = 0.0
    for band in summary:
        width = band["percent"]
        if width <= 0:
            continue
        axis.barh(0, width, left=left, color=band["colour"], edgecolor="white", linewidth=2)
        if width >= 7:
            text_colour = "#0b0b0b" if band["band"] == "Good" else "white"
            axis.text(left + width / 2, 0, f"{band['band']} {width:.0f}%",
                      ha="center", va="center", color=text_colour, fontsize=9, fontweight="bold")
        left += width
    axis.set_xlim(0, 100)
    axis.set_yticks([])
    axis.set_xlabel("Share of walked area (%)")
    for side in ("top", "right", "left"):
        axis.spines[side].set_visible(False)
    figure.tight_layout()
    return figure


def apply_saved_alignments(data, scan_files, store):
    """Fill clicked points from a saved .json (one scan or a set of scans)."""
    entries = data["scans"] if isinstance(data, dict) and "scans" in data else [data]
    names = {scan_file.name: scan_file for scan_file in scan_files}
    applied = 0
    for entry in entries:
        target = names.get(entry.get("scan_file"))
        if target is None and entry.get("scan_size"):
            # Names can change when files are shared; the image size does not.
            same_size = [
                scan_file for scan_file in scan_files
                if list(image_size(scan_file)) == list(entry["scan_size"])
            ]
            if len(same_size) == 1:
                target = same_size[0]
        if target is None and len(entries) == 1 and len(scan_files) == 1:
            target = scan_files[0]
        if target is None:
            continue
        points = store.setdefault(scan_key(target), {"scan": [], "plan": []})
        points["scan"] = [tuple(map(float, p)) for p in entry["scan_points"]]
        points["plan"] = [tuple(map(float, p)) for p in entry["plan_points"]]
        applied += 1
    return applied


def image_size(scan_file) -> tuple[int, int]:
    """(width, height) of an uploaded scan image."""
    image, _ = decode_scan_bytes(scan_file.getvalue())
    return image.shape[1], image.shape[0]


def scan_key(scan_file) -> str:
    return f"{scan_file.name}-{len(scan_file.getvalue())}"


def align_one_scan(scan_file, plan_bgr, streamlit_image_coordinates, store):
    """Click-to-align UI for one scan. Returns (aligned scan, alignment) or None."""
    try:
        scan_bgr, decoded = decode_scan_bytes(scan_file.getvalue())
    except ValueError as error:
        st.error(str(error))
        return None
    if not (decoded.measured.any() or decoded.scanned.any()):
        st.warning("No WiFiman scan colours were found in this image.")
        return None

    key = scan_key(scan_file)
    points = store.setdefault(key, {"scan": [], "plan": []})
    st.markdown(
        "Click a point on the scan (left), then the same place on the floorplan "
        "(right). Repeat for at least **3** points far apart, such as corridor "
        "ends, junctions or corners. The numbers must match."
    )
    scan_column, plan_column = st.columns(2)
    with scan_column:
        st.caption(f"Scan: {len(points['scan'])} point(s)")
        value = streamlit_image_coordinates(
            cv2.cvtColor(draw_points(scan_bgr, points["scan"]), cv2.COLOR_BGR2RGB),
            width="stretch", key=f"scan_click_{key}",
        )
        point = clicked_point(value, (scan_bgr.shape[1], scan_bgr.shape[0]), f"scan_{key}")
        if point is not None and len(points["scan"]) <= len(points["plan"]):
            points["scan"].append(point)
            st.rerun()
    with plan_column:
        st.caption(f"Floorplan: {len(points['plan'])} point(s)")
        value = streamlit_image_coordinates(
            cv2.cvtColor(draw_points(plan_bgr, points["plan"]), cv2.COLOR_BGR2RGB),
            width="stretch", key=f"plan_click_{key}",
        )
        point = clicked_point(value, (plan_bgr.shape[1], plan_bgr.shape[0]), f"plan_{key}")
        if point is not None and len(points["plan"]) < len(points["scan"]):
            points["plan"].append(point)
            st.rerun()

    undo_column, clear_column, _ = st.columns([1, 1, 3])
    if undo_column.button("Undo last point", key=f"undo_{key}"):
        longer = "scan" if len(points["scan"]) > len(points["plan"]) else "plan"
        if points[longer]:
            points[longer].pop()
        st.rerun()
    if clear_column.button("Clear points", key=f"clear_{key}"):
        points["scan"].clear()
        points["plan"].clear()
        st.rerun()

    pairs = min(len(points["scan"]), len(points["plan"]))
    if pairs < 3:
        st.info(f"{pairs} of 3 point pairs done.")
        return None

    alignment = fit_alignment(points["scan"][:pairs], points["plan"][:pairs])
    if alignment.residual_px > 0.03 * float(np.hypot(*plan_bgr.shape[:2])):
        st.warning(
            f"The points don't fit well (average error {alignment.residual_px:.0f} px). "
            "Check that each numbered point marks the same place on both images."
        )
    aligned = warp_scan_to_plan(decoded, alignment, plan_bgr.shape[:2])
    st.image(
        cv2.cvtColor(band_overlay(plan_bgr, aligned), cv2.COLOR_BGR2RGB),
        caption=(f"This scan on the floorplan (rotation {alignment.rotation_deg:.0f}°, "
                 f"average point error {alignment.residual_px:.1f} px)."),
        width="stretch",
    )
    return aligned, points["scan"][:pairs], points["plan"][:pairs]


def render_scan_section(floorplan_file) -> None:
    """Upload WiFiman scans, align each to the floorplan, show the combined map."""
    try:
        from streamlit_image_coordinates import streamlit_image_coordinates
    except ImportError:
        st.warning("Install `streamlit-image-coordinates` to align scans: `pip install streamlit-image-coordinates`.")
        return

    scan_files = st.file_uploader(
        "Upload WiFiman scan images (use WiFiman's share button for the full scan)",
        type=["png", "jpg", "jpeg"], accept_multiple_files=True, key="scan_upload",
    )
    if not scan_files:
        st.caption("Measured scans are optional. They let you compare real measurements with GeoTwin's predictions.")
        return

    store = st.session_state.setdefault("scan_points", {})
    saved_files = st.file_uploader(
        "Optional: load saved alignments (.json)", type=["json"],
        accept_multiple_files=True, key="load_alignments",
    )
    loaded = st.session_state.setdefault("loaded_alignment_ids", set())
    for saved in saved_files or []:
        if saved.file_id in loaded:
            continue
        try:
            applied = apply_saved_alignments(json.loads(saved.getvalue()), scan_files, store)
        except (KeyError, TypeError, ValueError, AttributeError):
            st.error(f"{saved.name} is not a GeoTwin scan alignment.")
            continue
        loaded.add(saved.file_id)
        if applied == 0:
            st.warning(f"{saved.name} didn't match any uploaded scan (scan file names differ).")
        else:
            st.rerun()

    plan_bgr = cv2.cvtColor(decode_image(floorplan_file), cv2.COLOR_RGB2BGR)
    aligned_scans = []
    for scan_file in scan_files:
        points = store.get(scan_key(scan_file), {"scan": [], "plan": []})
        done = min(len(points["scan"]), len(points["plan"])) >= 3
        with st.expander(f"{'✅' if done else '⬜'} {scan_file.name}", expanded=not done):
            result = align_one_scan(scan_file, plan_bgr, streamlit_image_coordinates, store)
        if result is not None:
            aligned_scans.append((scan_file.name, image_size(scan_file), *result))

    if not aligned_scans:
        return

    st.download_button(
        "Save all alignments (.json)",
        data=json.dumps({"scans": [
            {"scan_file": name, "scan_size": list(size),
             "scan_points": scan_points, "plan_points": plan_points}
            for name, size, _, scan_points, plan_points in aligned_scans
        ]}, indent=2),
        file_name="scan_alignments.json", mime="application/json",
    )

    combined = combine_scans([scan for _, _, scan, _, _ in aligned_scans])
    st.markdown(f"**Measured Wi-Fi on the floor ({len(aligned_scans)} scan(s) combined)**")
    st.image(
        cv2.cvtColor(band_overlay(plan_bgr, combined), cv2.COLOR_BGR2RGB),
        caption="Coloured = measured signal band. Dark grey = dead zone (walked, no signal).",
        width="stretch",
    )
    summary = walked_area_summary(combined)
    figure = band_bar_figure(summary)
    st.pyplot(figure, width="stretch")
    plt.close(figure)

    walked = combined.scanned | combined.measured
    readings = combined.rssi_dbm[combined.measured]
    good_enough = 100 * (readings >= -67).sum() / max(1, walked.sum())
    columns = st.columns(3)
    columns[0].metric("Meets −67 dBm", f"{good_enough:.0f}% of walked area")
    columns[1].metric("Dead zones", f"{summary[-1]['percent']:.0f}% of walked area")
    columns[2].metric("Median signal (where any)", f"{np.median(readings):.0f} dBm" if readings.size else "—")

    st.dataframe([
        {"Scan": name, **{row["band"]: f"{row['percent']:.0f}%" for row in walked_area_summary(scan)}}
        for name, _, scan, _, _ in aligned_scans
    ], hide_index=True, width="stretch")
    st.caption("Signal values are read from WiFiman's colours, so they are accurate to about one legend step (a few dB).")

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
    ap_count = st.slider("Number of APs", min_value=1, max_value=12, value=2)
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
            "AP placement is optimized directly against the trained backend predictor that draws this RSSI map, so the coverage shown is what the optimizer maximized."
        )

if uploaded_file is not None:
    st.divider()
    st.subheader("Measured scans (WiFiman)")
    render_scan_section(uploaded_file)

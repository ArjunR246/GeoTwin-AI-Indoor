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
    no_ap_zones_m: tuple = (),
):
    """Process one plan, optimize AP placement, then predict model RSSI.

    no_ap_zones_m: rectangles (x0, y0, x1, y1) in metres where APs may not go.
    """
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
        valid_ap_positions = [
            point for point in valid_ap_positions
            if not any(x0 <= point[0] <= x1 and y0 <= point[1] <= y1
                       for x0, y0, x1, y1 in no_ap_zones_m)
        ]
        if len(valid_ap_positions) < ap_count:
            raise ValueError("The no-AP zones leave too few places for the requested APs.")
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
            "valid_ap_positions": valid_ap_positions,
            "walls": walls,
            "width_m": building_width_m,
            "height_m": building_height_m,
            "no_ap_zones_m": no_ap_zones_m,
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
        saved_size = entry.get("scan_size")
        if target is None and saved_size:
            # Names change when files are shared, and sharing apps may resize
            # images, but the shape (width / height) stays the same.
            same_shape = [
                scan_file for scan_file in scan_files
                if same_aspect(image_size(scan_file), saved_size)
            ]
            if len(same_shape) == 1:
                target = same_shape[0]
        if target is None and len(entries) == 1 and len(scan_files) == 1:
            target = scan_files[0]
        if target is None:
            continue
        if entry.get("scan_size") and list(image_size(target)) != list(entry["scan_size"]):
            # Points only fit the exact image they were made on.
            width, height = image_size(target)
            st.warning(
                f"This alignment was made for a {entry['scan_size'][0]}×{entry['scan_size'][1]} image, "
                f"but {target.name} is {width}×{height}. Upload the original (unresized) scan image."
            )
            continue
        scale_x = scale_y = 1.0
        if saved_size:
            width, height = image_size(target)
            if not same_aspect((width, height), saved_size):
                st.warning(
                    f"{target.name} has a different shape from the scan this alignment "
                    "was made for. Use the original WiFiman image or align it again."
                )
                continue
            scale_x, scale_y = width / saved_size[0], height / saved_size[1]
        points = store.setdefault(scan_key(target), {"scan": [], "plan": []})
        points["scan"] = [(float(x) * scale_x, float(y) * scale_y) for x, y in entry["scan_points"]]
        points["plan"] = [tuple(map(float, p)) for p in entry["plan_points"]]
        applied += 1
    return applied


def same_aspect(size_a, size_b, tolerance: float = 0.02) -> bool:
    """True if two (width, height) sizes have the same shape (within 2%)."""
    return abs(size_a[0] / size_a[1] - size_b[0] / size_b[1]) <= tolerance * size_b[0] / size_b[1]


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
        return None

    scan_files = st.file_uploader(
        "Upload WiFiman scan images (use WiFiman's share button for the full scan)",
        type=["png", "jpg", "jpeg"], accept_multiple_files=True, key="scan_upload",
    )
    if not scan_files:
        st.caption("Measured scans are optional. They let you compare real measurements with GeoTwin's predictions.")
        return None

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
        return None

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
    return combined, aligned_scans

# ---------------------------------------------------------------------------
# No-AP zones
# ---------------------------------------------------------------------------

def zones_in_metres(zones_px, image_shape, width_m, height_m):
    """Convert rectangles from floorplan pixels to metres."""
    height_px, width_px = image_shape[:2]
    return tuple(
        (min(x0, x1) * width_m / width_px, min(y0, y1) * height_m / height_px,
         max(x0, x1) * width_m / width_px, max(y0, y1) * height_m / height_px)
        for x0, y0, x1, y1 in zones_px
    )


def draw_zones(image_bgr, zones_px, pending=None):
    out = image_bgr.copy()
    thickness = max(3, int(0.004 * max(out.shape[:2])))
    for x0, y0, x1, y1 in zones_px:
        p0, p1 = (int(x0), int(y0)), (int(x1), int(y1))
        shade = out.copy()
        cv2.rectangle(shade, p0, p1, (60, 60, 210), -1)
        out = cv2.addWeighted(shade, 0.25, out, 0.75, 0)
        cv2.rectangle(out, p0, p1, (40, 40, 200), thickness)
        cv2.line(out, p0, p1, (40, 40, 200), thickness)
        cv2.line(out, (p0[0], p1[1]), (p1[0], p0[1]), (40, 40, 200), thickness)
    if pending is not None:
        cv2.circle(out, (int(pending[0]), int(pending[1])), 3 * thickness, (40, 40, 200), -1)
    return out


def render_zone_section(floorplan_file) -> list:
    """Click two opposite corners per rectangle to mark areas where APs can't go."""
    zones = st.session_state.setdefault("no_ap_zones_px", [])
    editing = st.session_state.get("zone_editing", False)
    with st.expander(f"No-AP zones (optional) — {len(zones)} marked", expanded=editing):
        st.caption(
            "Mark areas where access points must not be installed (for example labs). "
            "Click two opposite corners of each rectangle. Coverage is still measured "
            "inside these areas; only AP placement is blocked."
        )
        try:
            from streamlit_image_coordinates import streamlit_image_coordinates
        except ImportError:
            st.warning("Install `streamlit-image-coordinates` to draw zones.")
            return zones
        plan_bgr = cv2.cvtColor(decode_image(floorplan_file), cv2.COLOR_RGB2BGR)
        pending = st.session_state.get("zone_corner")
        value = streamlit_image_coordinates(
            cv2.cvtColor(draw_zones(plan_bgr, zones, pending), cv2.COLOR_BGR2RGB),
            width="stretch", key="zone_click",
        )
        point = clicked_point(value, (plan_bgr.shape[1], plan_bgr.shape[0]), "zone")
        if point is not None:
            st.session_state["zone_editing"] = True      # keep this box open
            if pending is None:
                st.session_state["zone_corner"] = point
            else:
                zones.append((*pending, *point))
                st.session_state["zone_corner"] = None
            st.rerun()

        undo, clear, save, load = st.columns(4)
        if undo.button("Undo last zone", key="zone_undo"):
            st.session_state["zone_editing"] = True
            if st.session_state.get("zone_corner") is not None:
                st.session_state["zone_corner"] = None
            elif zones:
                zones.pop()
            st.rerun()
        if clear.button("Clear zones", key="zone_clear"):
            zones.clear()
            st.session_state["zone_corner"] = None
            st.rerun()
        save.download_button(
            "Save zones (.json)", data=json.dumps({"no_ap_zones_px": zones}),
            file_name="no_ap_zones.json", mime="application/json", key="zone_save",
        )
        uploaded = load.file_uploader("Load zones", type=["json"], key="zone_load",
                                      label_visibility="collapsed")
        if uploaded is not None and st.session_state.get("zone_loaded") != uploaded.file_id:
            try:
                zones[:] = [tuple(map(float, z)) for z in json.loads(uploaded.getvalue())["no_ap_zones_px"]]
                st.session_state["zone_loaded"] = uploaded.file_id
                st.rerun()
            except (KeyError, TypeError, ValueError):
                st.error("That file is not a GeoTwin no-AP zone file.")
    return zones


# ---------------------------------------------------------------------------
# Calibration and comparison
# ---------------------------------------------------------------------------

BAND_COLOURS = {"Excellent": "#0ca30c", "Good": "#fab219", "Fair": "#ec835a",
                "Poor": "#d03b3b", "Dead zone": "#3d3d3a"}


TODAY_COLOUR = "#2a78d6"
GEOTWIN_COLOUR = "#1baf7a"


def _style_column_axis(axis):
    axis.set_ylim(0, 100)
    axis.set_ylabel("Share of walked area (%)")
    axis.grid(axis="y", color="#e1e0d9", linewidth=0.8)
    axis.set_axisbelow(True)
    for side in ("top", "right", "left"):
        axis.spines[side].set_visible(False)
    axis.tick_params(axis="y", length=0, colors="#52514e")


def _label_columns(axis, bars):
    for bar in bars:
        height = bar.get_height()
        axis.text(bar.get_x() + bar.get_width() / 2, height + 1.5, f"{height:.0f}%",
                  ha="center", va="bottom", fontsize=10, fontweight="bold", color="#0b0b0b")


def headline_columns_figure(today, geotwin, threshold, geotwin_label):
    """Two side-by-side panels: meets threshold and dead zone, today vs GeoTwin."""
    figure, axes = plt.subplots(1, 2, figsize=(9, 4.2), dpi=120, sharey=True)
    panels = (
        (f"Meets {threshold:.0f} dBm (higher is better)", "meets_threshold"),
        ("Dead zone (lower is better)", "Dead zone"),
    )
    for axis, (title, key) in zip(axes, panels):
        bars = axis.bar([0, 1], [today[key], geotwin[key]], width=0.55,
                        color=[TODAY_COLOUR, GEOTWIN_COLOUR], edgecolor="white", linewidth=2)
        _label_columns(axis, bars)
        axis.set_xticks([0, 1], ["Today\n(measured)", geotwin_label])
        axis.set_title(title, fontsize=11, loc="left", color="#0b0b0b")
        _style_column_axis(axis)
    axes[1].set_ylabel("")
    figure.tight_layout()
    return figure


def band_columns_figure(today, geotwin, geotwin_label):
    """Grouped vertical columns per signal band, today vs GeoTwin."""
    bands = list(BAND_COLOURS)
    x = np.arange(len(bands))
    width = 0.36
    figure, axis = plt.subplots(figsize=(10, 4.4), dpi=120)
    for offset, shares, colour, label in (
        (-width / 2, today, TODAY_COLOUR, "Today (measured)"),
        (width / 2, geotwin, GEOTWIN_COLOUR, geotwin_label.replace("\n", " ")),
    ):
        bars = axis.bar(x + offset, [shares[b] for b in bands], width=width, color=colour,
                        edgecolor="white", linewidth=2, label=label)
        _label_columns(axis, bars)
    axis.set_xticks(x, bands)
    _style_column_axis(axis)
    axis.legend(frameon=False, loc="upper left", ncol=2)
    figure.tight_layout()
    return figure


def comparison_figure(rows):
    """Stacked 100% bars, one per scenario."""
    figure, axis = plt.subplots(figsize=(11, 1.0 + 1.1 * len(rows)), dpi=120)
    for y, (label, shares) in zip(range(len(rows) - 1, -1, -1), rows):
        left = 0.0
        for band, colour in BAND_COLOURS.items():
            width = shares[band]
            if width <= 0.4:
                continue
            axis.barh(y, width, left=left, color=colour, edgecolor="white", linewidth=2, height=0.6)
            if width >= 7:
                axis.text(left + width / 2, y, f"{band}\n{width:.0f}%", ha="center", va="center",
                          fontsize=8.5, fontweight="bold",
                          color="#0b0b0b" if band == "Good" else "white")
            left += width
    axis.set_yticks(range(len(rows) - 1, -1, -1))
    axis.set_yticklabels([label for label, _ in rows])
    axis.set_xlim(0, 100)
    axis.set_xlabel("Share of walked area (%)")
    for side in ("top", "right", "left"):
        axis.spines[side].set_visible(False)
    figure.tight_layout()
    return figure


def ap_curve_figure(counts, walked, walked_range, floor, floor_range, today, threshold):
    figure, axis = plt.subplots(figsize=(8, 4), dpi=120)
    axis.fill_between(counts, [r[0] for r in walked_range], [r[1] for r in walked_range],
                      color="#2a6fdb", alpha=0.15, linewidth=0)
    axis.fill_between(counts, [r[0] for r in floor_range], [r[1] for r in floor_range],
                      color="#7a5195", alpha=0.15, linewidth=0)
    axis.plot(counts, walked, "-o", color="#2a6fdb", lw=2, ms=7, label="Walked (measured) area")
    axis.plot(counts, floor, "-o", color="#7a5195", lw=2, ms=7, label="Whole floor")
    axis.axhline(today, color="#d03b3b", ls="--", lw=1.5)
    axis.text(counts[-1], today + 1.5, "today, measured", color="#d03b3b", ha="right", fontsize=9)
    axis.set_xticks(counts)
    axis.set_xlabel("Number of APs (GeoTwin layout)")
    axis.set_ylabel(f"Area meeting {threshold:.0f} dBm (%)")
    axis.set_ylim(0, 100)
    axis.grid(axis="y", color="#e1e0d9")
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    axis.legend(frameon=False, loc="upper left")
    figure.tight_layout()
    return figure


def calibrated_map_figure(result, receivers, rssi, geotwin_aps, current_aps, threshold):
    x_values, y_values, grid = heatmap_grid([tuple(r) for r in receivers], rssi)
    figure, axis = plt.subplots(figsize=(10, 10 * result["height_m"] / result["width_m"] * 0.75), dpi=120)
    axis.imshow(result["image_rgb"], extent=[0, result["width_m"], result["height_m"], 0], aspect="auto")
    image = axis.imshow(
        np.ma.masked_invalid(grid), cmap="turbo", vmin=-90, vmax=-30, alpha=0.6,
        extent=[grid_edges(x_values)[0], grid_edges(x_values)[-1],
                grid_edges(y_values)[-1], grid_edges(y_values)[0]],
        interpolation="bilinear", aspect="auto",
    )
    axis.contour(x_values, y_values, np.ma.masked_invalid(grid), levels=[threshold],
                 colors="white", linewidths=1.2)
    for number, (x, y) in enumerate(geotwin_aps, start=1):
        axis.scatter(x, y, marker="*", s=260, color="white", edgecolor="#111827", zorder=5)
        axis.text(x, y - 1.2, f"G{number}", ha="center", fontsize=8, fontweight="bold",
                  bbox={"boxstyle": "round,pad=0.15", "fc": "white", "alpha": 0.85}, zorder=6)
    for x, y in current_aps:
        axis.scatter(x, y, marker="X", s=110, color="#d03b3b", edgecolor="white", zorder=5)
    axis.set(xlim=(0, result["width_m"]), ylim=(result["height_m"], 0),
             xlabel="X position (m)", ylabel="Y position (m, from top edge)")
    figure.colorbar(image, ax=axis, pad=0.015, label="Calibrated RSSI (dBm)")
    figure.tight_layout()
    return figure


def render_calibration_section(combined, aligned_scans, result) -> None:
    """Calibrate a radio model on measured scans and compare layouts."""
    from services.calibration import band_shares, fit_calibration, path_features, sample_walked_cells
    from services.fast_placement import assign_channels, choose_ap_spots, optimize_placement

    if result is None:
        st.info("Generate an AP plan above first (with the building's real size). "
                "Calibration uses the same floor geometry.")
        return
    threshold = result["coverage_threshold_dbm"]
    st.caption(
        "Fits signal = P0 − 10·n·log₁₀(distance) − wall loss × walls crossed to your measurements, "
        "and estimates where the current APs are (outside no-AP zones). P0 is fixed at −35 dBm, "
        "a typical value 1 m from an AP."
    )
    current_count = st.number_input("Number of APs in the building today", 1, 30, 6, key="current_aps")

    key = json.dumps([
        [name, size, sp, pp] for name, size, _, sp, pp in aligned_scans
    ] + [result["width_m"], result["height_m"], len(result["valid_ap_positions"]),
         list(result["no_ap_zones_m"]), int(current_count)])
    state = st.session_state.get("calibration")
    if state is not None and state["key"] != key:
        state = None
    if st.button("Calibrate with measured scans", type="primary"):
        with st.spinner("Fitting the model to your measurements..."):
            plan_shape = combined.rssi_dbm.shape
            cells, values = sample_walked_cells(
                combined.rssi_dbm, combined.measured, combined.scanned | combined.measured,
                result["width_m"], result["height_m"],
            )
            candidates = [result["valid_ap_positions"][i]
                          for i in choose_ap_spots(result["valid_ap_positions"], 1500)]
            try:
                model = fit_calibration(cells, values, candidates, result["walls"], int(current_count))
            except ValueError as error:
                st.error(str(error))
                return
            receivers = np.asarray(result["receiver_positions"], dtype=float)
            floor_dist, floor_walls = path_features(candidates, receivers, result["walls"])
            cell_dist, cell_walls = path_features(candidates, cells, result["walls"])
            state = {
                "key": key, "model": model, "cells": cells, "values": values,
                "candidates": candidates, "receivers": receivers,
                "floor": (floor_dist, floor_walls), "walk": (cell_dist, cell_walls),
            }
            st.session_state["calibration"] = state
    if state is None:
        return

    model = state["model"]
    columns = st.columns(4)
    columns[0].metric("Distance exponent n", f"{model.exponent:.1f}")
    columns[1].metric("Loss per wall", f"{model.wall_loss_db:.1f} dB")
    columns[2].metric("Error on unseen cells", f"{model.test_error_db:.1f} dB",
                      delta=f"{model.test_error_db - model.baseline_error_db:+.1f} dB vs guessing",
                      delta_color="inverse")
    columns[3].metric("Dead / signal cells right", f"{model.test_dead_accuracy:.0%} / {model.test_signal_accuracy:.0%}")
    if not model.is_useful:
        st.warning(
            "The calibrated model is not clearly better than guessing the average on cells it "
            "didn't see. Treat the comparison below as unreliable; more scans would help."
        )
    for note in model.notes:
        st.caption(note)
    st.caption(
        f"Based on {model.measured_cells} walked cells with signal and {model.dead_cells} dead-zone "
        "cells (1 m each). Ranges below cover all fits that match the measurements almost as well."
    )

    candidates, values = state["candidates"], state["values"]
    floor_dist, floor_walls = state["floor"]
    cell_dist, cell_walls = state["walk"]
    floor_table = model.signal(floor_dist, floor_walls)
    max_aps = min(len(candidates), max(int(current_count) + 8, 14))
    plan_count = st.number_input("GeoTwin plan: number of APs", 1, max_aps,
                                 int(current_count), key="geotwin_count")

    def plan(count):
        placement = optimize_placement(floor_table, candidates, int(count), threshold)
        return [candidates.index(p) for p in placement.positions]

    current_layout = [candidates.index(p) for p in model.current_aps]
    geotwin_layout = plan(plan_count)
    walk_current = model.signal(cell_dist[current_layout], cell_walls[current_layout]).max(0)
    walk_geotwin = model.signal(cell_dist[geotwin_layout], cell_walls[geotwin_layout]).max(0)
    rows = [
        ("Today (measured)", band_shares(values, threshold)),
        (f"Estimated current {len(current_layout)} APs\n(calibrated prediction)", band_shares(walk_current, threshold)),
        (f"GeoTwin, {plan_count} APs\n(calibrated prediction)", band_shares(walk_geotwin, threshold)),
    ]
    today_shares, geotwin_shares = rows[0][1], rows[2][1]
    geotwin_label = f"GeoTwin\n({plan_count} APs, predicted)"
    st.markdown("**Before vs after: walked area**")
    tiles = st.columns(2)
    tiles[0].metric(f"Meets {threshold:.0f} dBm",
                    f"{today_shares['meets_threshold']:.0f}% → {geotwin_shares['meets_threshold']:.0f}%")
    tiles[1].metric("Dead zone",
                    f"{today_shares['Dead zone']:.0f}% → {geotwin_shares['Dead zone']:.0f}%")
    figure = headline_columns_figure(today_shares, geotwin_shares, threshold, geotwin_label)
    st.pyplot(figure, width="stretch")
    st.download_button("Download chart PNG", data=figure_bytes(figure),
                       file_name="geotwin_headline_bars.png", mime="image/png", key="dl_headline")
    plt.close(figure)

    st.markdown("**Signal bands: today vs GeoTwin**")
    figure = band_columns_figure(today_shares, geotwin_shares, geotwin_label)
    st.pyplot(figure, width="stretch")
    st.download_button("Download chart PNG", data=figure_bytes(figure),
                       file_name="geotwin_band_bars.png", mime="image/png", key="dl_bands")
    plt.close(figure)

    with st.expander("Details: full breakdown including the estimated current APs"):
        figure = comparison_figure(rows)
        st.pyplot(figure, width="stretch")
        st.download_button("Download chart PNG", data=figure_bytes(figure),
                           file_name="geotwin_before_after.png", mime="image/png", key="dl_compare")
        plt.close(figure)

    low, high = model.outcome_range(cell_dist, cell_walls, geotwin_layout, threshold)
    floor_low, floor_high = model.outcome_range(floor_dist, floor_walls, geotwin_layout, threshold)
    st.dataframe([
        {"Scenario": label.replace("\n", " "), f"Meets {threshold:.0f} dBm": f"{shares['meets_threshold']:.0f}%",
         "Dead zone": f"{shares['Dead zone']:.0f}%"}
        for label, shares in rows
    ], hide_index=True, width="stretch")
    st.caption(
        f"GeoTwin {plan_count}-AP plan: {low:.0f}–{high:.0f}% of the walked area and "
        f"{floor_low:.0f}–{floor_high:.0f}% of the whole floor meet {threshold:.0f} dBm across plausible fits. "
        "Predictions, not measurements: confirm with a new scan after changes."
    )

    if st.checkbox("Show how many APs are needed", key="show_curve"):
        with st.spinner("Planning layouts for different AP counts..."):
            start = max(1, int(current_count))
            counts = list(range(start, max_aps + 1, 2))
            walked, walked_range, floor, floor_range = [], [], [], []
            for count in counts:
                layout = plan(count)
                walked.append(float(100 * (model.signal(cell_dist[layout], cell_walls[layout]).max(0) >= threshold).mean()))
                walked_range.append(model.outcome_range(cell_dist, cell_walls, layout, threshold))
                floor.append(float(100 * (floor_table[layout].max(0) >= threshold).mean()))
                floor_range.append(model.outcome_range(floor_dist, floor_walls, layout, threshold))
        figure = ap_curve_figure(counts, walked, walked_range, floor, floor_range,
                                 rows[0][1]["meets_threshold"], threshold)
        st.pyplot(figure, width="stretch")
        st.download_button("Download chart PNG", data=figure_bytes(figure),
                           file_name="geotwin_aps_needed.png", mime="image/png", key="dl_curve")
        plt.close(figure)

    figure = calibrated_map_figure(
        result, state["receivers"], floor_table[geotwin_layout].max(0),
        [candidates[i] for i in geotwin_layout], model.current_aps, threshold,
    )
    st.markdown("**Calibrated map of the GeoTwin plan** (★ GeoTwin APs, ✕ estimated current APs, white line = threshold)")
    st.pyplot(figure, width="stretch")
    plt.close(figure)
    channels = assign_channels(floor_table, geotwin_layout)
    st.dataframe([
        {"AP": f"G{i}", "X (m)": round(candidates[j][0], 2), "Y (m)": round(candidates[j][1], 2), "Channel": c}
        for i, (j, c) in enumerate(zip(geotwin_layout, channels), start=1)
    ], hide_index=True, width="stretch")
    st.caption("Estimated current AP positions are approximate: they are inferred from the measurements, not confirmed.")


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
        st.caption(
            "The AI model was trained at 20 dBm and 5 GHz. Other TX powers and "
            "frequencies are applied with standard radio physics (1 dB per dB of "
            "TX power; free-space loss for frequency). Extra wall loss at higher "
            "frequencies is not modelled."
        )
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
                    zones_m = zones_in_metres(
                        st.session_state.get("no_ap_zones_px", []),
                        decode_image(uploaded_file).shape,
                        building_width_m, building_height_m,
                    )
                    st.session_state.plan_result = generate_plan(
                        uploaded_file, building_width_m, building_height_m,
                        grid_resolution_m, ap_count, tx_power_dbm, frequency_ghz,
                        coverage_threshold_dbm, wall_loss_db, zones_m,
                    )
                st.success("Your AP plan is ready.")
            except Exception as error:
                st.session_state.plan_result = None
                st.error(f"Unable to generate the AP plan: {error}")
    render_zone_section(uploaded_file)

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
    scans = render_scan_section(uploaded_file)
    if scans is not None:
        st.divider()
        st.subheader("Calibrate and compare with GeoTwin")
        render_calibration_section(scans[0], scans[1], st.session_state.get("plan_result"))

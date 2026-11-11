"""Import measured Wi-Fi scans (WiFiman Signal Mapper images) onto a floorplan.

WiFiman shares a scan as an image: grey cells are areas the LiDAR scan
walked, coloured cells are where Wi-Fi signal was measured. This module:

1. decode_scan()        turns the colours back into dBm using WiFiman's legend;
2. fit_alignment()      finds the rotation, scale and shift that map the scan
                        onto the floorplan from >= 3 matching point pairs;
3. warp_scan_to_plan()  moves the decoded values into floorplan pixels;
4. band_summary()       splits measured values into signal-strength bands.

Colours are matched by hue (the position on the red -> yellow -> green wheel),
because WiFiman darkens cells with its grid lines: brightness changes, hue
does not. Values are accurate to about one legend step (a few dB).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np

# WiFiman legend: (dBm, OpenCV hue 0-179). Read from the app's colour bar
# (red -80 ... orange -70 ... yellow -50 ... green -40 ... teal -30).
# Red wraps around hue 0, so hues above 150 are treated as negative.
WIFIMAN_LEGEND: tuple[tuple[float, float], ...] = (
    (-85.0, 1), (-82.5, 7), (-80.0, 9), (-77.5, 11), (-75.0, 12),
    (-72.5, 14), (-70.0, 15), (-67.5, 17), (-65.0, 18), (-62.5, 20),
    (-60.0, 22), (-57.5, 24), (-55.0, 26), (-52.5, 30), (-50.0, 34),
    (-47.5, 38), (-45.0, 43), (-42.5, 47), (-40.0, 50), (-37.5, 54),
    (-35.0, 60), (-32.5, 68), (-30.0, 76), (-27.5, 83), (-25.0, 86),
)

# Signal bands used in every chart: (name, label, lower dBm, upper dBm, colour).
SIGNAL_BANDS: tuple[tuple[str, str, float, float, str], ...] = (
    ("Excellent", "≥ −50 dBm", -50.0, 999.0, "#0ca30c"),
    ("Good", "−50 to −67 dBm", -67.0, -50.0, "#fab219"),
    ("Fair", "−67 to −75 dBm", -75.0, -67.0, "#ec835a"),
    ("Poor", "< −75 dBm", -999.0, -75.0, "#d03b3b"),
)


# Walked by the scanner but no Wi-Fi signal recorded (WiFiman's grey cells).
DEAD_ZONE: tuple[str, str, str] = ("Dead zone", "no signal", "#3d3d3a")


@dataclass(frozen=True)
class DecodedScan:
    """A scan image turned into numbers (all arrays have the image's shape)."""

    rssi_dbm: np.ndarray        # float, NaN where no signal was recorded
    measured: np.ndarray        # bool, Wi-Fi signal recorded
    scanned: np.ndarray         # bool, walked area (signal or dead zone)


@dataclass(frozen=True)
class Alignment:
    """Similarity transform (rotation + uniform scale + shift) scan -> plan."""

    matrix: np.ndarray          # 2x3, maps scan pixels to plan pixels
    scale: float                # plan pixels per scan pixel
    rotation_deg: float
    residual_px: float          # mean distance between moved scan points and plan points


# ---------------------------------------------------------------------------
# 1. Decoding
# ---------------------------------------------------------------------------

def _hue_to_dbm(hue: np.ndarray) -> np.ndarray:
    dbm = np.array([d for d, _ in WIFIMAN_LEGEND])
    hues = np.array([h for _, h in WIFIMAN_LEGEND], dtype=float)
    return np.interp(hue, hues, dbm)


def decode_scan(image_bgr: np.ndarray, min_blob_ratio: float = 0.0005) -> DecodedScan:
    """Decode a WiFiman scan image into dBm values.

    Measured cells are saturated colours inside the legend's hue range.
    Scanned-but-unmeasured cells are mid grey. Small separate blobs (icons,
    text) and long thin horizontal bars (the legend itself, if a screenshot
    is uploaded) are ignored.
    """
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("Scan image is empty.")
    if image_bgr.ndim == 2:
        image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)
    height, width = image_bgr.shape[:2]
    hsv = cv2.cvtColor(image_bgr[..., :3], cv2.COLOR_BGR2HSV)
    hue = hsv[..., 0].astype(float)
    hue = np.where(hue > 150, hue - 180, hue)
    saturation, value = hsv[..., 1], hsv[..., 2]

    max_hue = WIFIMAN_LEGEND[-1][1] + 4
    measured = (saturation > 90) & (value > 70) & (hue >= -10) & (hue <= max_hue)
    grey = (saturation < 45) & (value > 85) & (value < 215)

    def drop_clutter(mask: np.ndarray) -> np.ndarray:
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
        keep = np.zeros(count, dtype=bool)
        area = stats[1:, cv2.CC_STAT_AREA]
        w, h = stats[1:, cv2.CC_STAT_WIDTH], stats[1:, cv2.CC_STAT_HEIGHT]
        legend_bar = (w > 0.7 * width) & (h < 0.08 * height)
        keep[1:] = (area >= min_blob_ratio * height * width) & ~legend_bar
        return keep[labels]

    # Cells are separated by thin grid lines, so join neighbouring cells
    # before deciding what is a real area and what is a stray icon.
    def joined(mask: np.ndarray) -> np.ndarray:
        return cv2.morphologyEx(
            mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)
        ) > 0

    measured_pixels = drop_clutter(joined(measured)) & measured
    values = _hue_to_dbm(hue[measured_pixels])

    # Fill the thin grid lines between cells, so areas stay solid when the
    # scan is resized onto the floorplan. Each filled pixel takes the value
    # of the nearest measured pixel.
    measured = joined(measured_pixels)
    scanned = drop_clutter(joined(grey | measured_pixels)) | measured
    _, nearest = cv2.distanceTransformWithLabels(
        (~measured_pixels).astype(np.uint8), cv2.DIST_L2, 3,
        labelType=cv2.DIST_LABEL_PIXEL,
    )
    by_label = np.concatenate(([np.nan], values))   # labels follow raster order
    rssi = np.where(measured, by_label[nearest], np.nan)
    return DecodedScan(rssi_dbm=rssi, measured=measured, scanned=scanned)


# ---------------------------------------------------------------------------
# 2. Alignment
# ---------------------------------------------------------------------------

def fit_alignment(
    scan_points: Sequence[tuple[float, float]],
    plan_points: Sequence[tuple[float, float]],
) -> Alignment:
    """Least-squares rotation + uniform scale + shift (Umeyama's method).

    Needs at least 3 matching point pairs, clicked in the same order on the
    scan and on the plan. Mirroring is not allowed: a LiDAR scan is never a
    mirror image of the building.
    """
    src = np.asarray(scan_points, dtype=float)
    dst = np.asarray(plan_points, dtype=float)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 2:
        raise ValueError("Scan and plan points must be matching lists of (x, y).")
    if len(src) < 3:
        raise ValueError("At least 3 matching point pairs are needed.")

    src_mean, dst_mean = src.mean(axis=0), dst.mean(axis=0)
    src_c, dst_c = src - src_mean, dst - dst_mean
    src_var = (src_c ** 2).sum() / len(src)
    if src_var < 1e-9:
        raise ValueError("Scan points are all in the same place.")

    covariance = dst_c.T @ src_c / len(src)
    u, singular, vt = np.linalg.svd(covariance)
    sign = np.eye(2)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        sign[1, 1] = -1                     # forbid reflection
    rotation = u @ sign @ vt
    scale = float(np.trace(np.diag(singular) @ sign) / src_var)
    shift = dst_mean - scale * rotation @ src_mean

    matrix = np.hstack([scale * rotation, shift[:, None]])
    moved = src @ matrix[:, :2].T + matrix[:, 2]
    residual = float(np.linalg.norm(moved - dst, axis=1).mean())
    angle = float(np.degrees(np.arctan2(rotation[1, 0], rotation[0, 0])))
    return Alignment(matrix=matrix, scale=scale, rotation_deg=angle, residual_px=residual)


def warp_scan_to_plan(
    scan: DecodedScan, alignment: Alignment, plan_shape: tuple[int, int]
) -> DecodedScan:
    """Move a decoded scan into floorplan pixel coordinates."""
    height, width = plan_shape[:2]
    size = (width, height)
    rssi = cv2.warpAffine(
        np.nan_to_num(scan.rssi_dbm, nan=0.0).astype(np.float32), alignment.matrix, size,
        flags=cv2.INTER_NEAREST, borderValue=0.0,
    )
    measured = cv2.warpAffine(
        scan.measured.astype(np.uint8), alignment.matrix, size,
        flags=cv2.INTER_NEAREST, borderValue=0,
    ) > 0
    scanned = cv2.warpAffine(
        scan.scanned.astype(np.uint8), alignment.matrix, size,
        flags=cv2.INTER_NEAREST, borderValue=0,
    ) > 0
    rssi = np.where(measured, rssi, np.nan)
    return DecodedScan(rssi_dbm=rssi, measured=measured, scanned=scanned | measured)


# ---------------------------------------------------------------------------
# 3. Bands and pictures
# ---------------------------------------------------------------------------

def band_summary(values_dbm: np.ndarray) -> list[dict[str, object]]:
    """Share of values (in %) falling in each signal band."""
    values = np.asarray(values_dbm, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return [{"band": n, "range": r, "percent": 0.0, "colour": c} for n, r, _, _, c in SIGNAL_BANDS]
    return [
        {
            "band": name,
            "range": label,
            "percent": float(100 * ((values >= low) & (values < high)).mean()),
            "colour": colour,
        }
        for name, label, low, high, colour in SIGNAL_BANDS
    ]


def walked_area_summary(scan: DecodedScan) -> list[dict[str, object]]:
    """Share (in %) of the whole walked area in each band, dead zones included.

    WiFiman shows walked areas without any signal in grey; those count as a
    fifth band, "Dead zone", so the percentages describe everywhere walked.
    """
    walked = int((scan.scanned | scan.measured).sum())
    if walked == 0:
        return band_summary(np.array([])) + [
            {"band": DEAD_ZONE[0], "range": DEAD_ZONE[1], "percent": 0.0, "colour": DEAD_ZONE[2]}
        ]
    signal_share = scan.measured.sum() / walked
    rows = band_summary(scan.rssi_dbm[scan.measured])
    for row in rows:
        row["percent"] = float(row["percent"] * signal_share)
    rows.append({
        "band": DEAD_ZONE[0], "range": DEAD_ZONE[1],
        "percent": float(100 * (1 - signal_share)), "colour": DEAD_ZONE[2],
    })
    return rows


def combine_scans(scans: Sequence[DecodedScan]) -> DecodedScan:
    """Merge several scans already aligned onto the same floorplan.

    Where scans overlap, signal values are averaged. A spot counts as having
    signal if any scan recorded signal there; it is a dead zone only if it
    was walked and no scan recorded signal.
    """
    if not scans:
        raise ValueError("At least one scan is needed.")
    shape = scans[0].rssi_dbm.shape
    if any(scan.rssi_dbm.shape != shape for scan in scans):
        raise ValueError("All scans must be aligned to the same floorplan.")
    total = np.zeros(shape)
    count = np.zeros(shape)
    for scan in scans:
        total[scan.measured] += scan.rssi_dbm[scan.measured]
        count[scan.measured] += 1
    measured = count > 0
    rssi = np.full(shape, np.nan)
    rssi[measured] = total[measured] / count[measured]
    scanned = np.logical_or.reduce([scan.scanned | scan.measured for scan in scans])
    return DecodedScan(rssi_dbm=rssi, measured=measured, scanned=scanned)


def _hex_to_bgr(colour: str) -> tuple[int, int, int]:
    r, g, b = (int(colour[i:i + 2], 16) for i in (1, 3, 5))
    return b, g, r


def band_overlay(
    base_bgr: np.ndarray, scan: DecodedScan, opacity: float = 0.75
) -> np.ndarray:
    """Draw signal bands, and dead zones in dark grey, over an image."""
    out = base_bgr.copy()
    layer = out.copy()
    dead = scan.scanned & ~scan.measured
    layer[dead] = _hex_to_bgr(DEAD_ZONE[2])
    values = scan.rssi_dbm
    for _, _, low, high, colour in SIGNAL_BANDS:
        mask = scan.measured & (values >= low) & (values < high)
        layer[mask] = _hex_to_bgr(colour)
    painted = scan.scanned | scan.measured
    out[painted] = (opacity * layer[painted] + (1 - opacity) * out[painted]).astype(np.uint8)
    return out


def draw_points(image_bgr: np.ndarray, points: Sequence[tuple[float, float]]) -> np.ndarray:
    """Numbered markers for clicked points."""
    out = image_bgr.copy()
    radius = max(6, int(0.012 * max(out.shape[:2])))
    for number, (x, y) in enumerate(points, start=1):
        centre = (int(round(x)), int(round(y)))
        cv2.circle(out, centre, radius, (255, 255, 255), -1)
        cv2.circle(out, centre, radius, (200, 40, 200), max(2, radius // 4))
        cv2.putText(
            out, str(number), (centre[0] - radius // 2, centre[1] + radius // 2),
            cv2.FONT_HERSHEY_SIMPLEX, radius / 14, (40, 40, 40), max(1, radius // 6),
        )
    return out


__all__ = [
    "Alignment", "DEAD_ZONE", "DecodedScan", "SIGNAL_BANDS", "WIFIMAN_LEGEND",
    "band_overlay", "band_summary", "combine_scans", "decode_scan", "draw_points",
    "fit_alignment", "walked_area_summary", "warp_scan_to_plan",
]

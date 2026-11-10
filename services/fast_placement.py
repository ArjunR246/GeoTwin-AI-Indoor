"""Fast, model-consistent AP placement.

Why this exists
---------------
The original optimizer (NSGA-II in frontend/src/geotwin/optimization) scores
each candidate layout with its own physics formula, recomputing every
AP -> receiver signal and every wall crossing for every layout it tries.
That is slow, so the app could only afford ~100 layouts, and its formula
differs from the XGBoost model that draws the heatmap.

This module:
  1. Precomputes a signal table once: RSSI from every allowed AP spot to every
     receiver point, using the SAME XGBoost model and wall features as the
     heatmap (services.floorplan_coverage.predict_heatmap).
  2. Scores a layout by looking up rows of that table - exactly the values the
     heatmap would show, without recomputation.
  3. Searches with greedy placement + local search (try moving each AP to every
     allowed spot), from several starting layouts.
  4. Assigns channels to reduce co-channel overlap.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from services.floorplan_coverage import PredictionArtifacts
from utils.calibration import apply_residual_calibration
from utils.rssi_features import MODEL_FEATURES
from utils.wall_intersections import wall_attenuation_db


# ---------------------------------------------------------------------------
# 1. Signal table
# ---------------------------------------------------------------------------

def _split_walls(walls: Sequence[Mapping[str, object]]):
    """Return arrays for horizontal walls, vertical walls and any other walls."""
    horizontal, vertical, other = [], [], []
    for wall in walls:
        x1, y1 = float(wall["start"]["x"]), float(wall["start"]["y"])
        x2, y2 = float(wall["end"]["x"]), float(wall["end"]["y"])
        loss = wall_attenuation_db(wall)
        if y1 == y2:
            horizontal.append((y1, min(x1, x2), max(x1, x2), loss))
        elif x1 == x2:
            vertical.append((x1, min(y1, y2), max(y1, y2), loss))
        else:
            other.append(wall)
    as_array = lambda rows: np.array(rows, dtype=float).reshape(-1, 4)
    return as_array(horizontal), as_array(vertical), other


def _axis_crossings(ap, targets, lines, axis):
    """Crossings of paths ap->target with axis-aligned walls.

    For a horizontal wall at height c spanning [lo, hi] (axis=1), the path
    crosses it when c lies strictly between the two y values and, at that
    height, the path's x lies strictly inside (lo, hi). Vertical walls are the
    same with x and y swapped. Touching an end or running along a wall does
    not count, matching Shapely's ``crosses`` used by the original code.
    """
    count = np.zeros(len(targets))
    loss = np.zeros(len(targets))
    if len(lines) == 0:
        return count, loss
    a_main, a_other = ap[axis], ap[1 - axis]
    t_main = targets[:, axis][:, None]          # (R, 1)
    t_other = targets[:, 1 - axis][:, None]
    c, lo, hi, w = (lines[:, i][None, :] for i in range(4))   # (1, W)
    eps = 1e-9
    between = ((a_main - c) * (t_main - c)) < -eps            # strictly opposite sides
    with np.errstate(divide="ignore", invalid="ignore"):
        fraction = (c - a_main) / (t_main - a_main)
        at_wall = a_other + fraction * (t_other - a_other)
    inside = (at_wall > lo + eps) & (at_wall < hi - eps)
    crosses = between & inside
    return crosses.sum(axis=1), crosses.astype(float) @ lines[:, 3]


def wall_features_from(ap, targets, walls_split):
    """(crossing counts, attenuation sums) from one AP to all targets."""
    horizontal, vertical, other = walls_split
    count_h, loss_h = _axis_crossings(ap, targets, horizontal, axis=1)
    count_v, loss_v = _axis_crossings(ap, targets, vertical, axis=0)
    count, loss = count_h + count_v, loss_h + loss_v
    if other:
        from services.floorplan_coverage import count_wall_crossings_fast
        extra_count, extra_loss = count_wall_crossings_fast(tuple(ap), targets, other)
        count, loss = count + extra_count, loss + extra_loss
    return count.astype(int), loss


def build_signal_table(
    ap_positions: Sequence[tuple[float, float]],
    receiver_positions: Sequence[tuple[float, float]],
    walls: Sequence[Mapping[str, object]],
    artifacts: PredictionArtifacts,
    tx_power_dbm: float,
    frequency_ghz: float,
) -> np.ndarray:
    """RSSI table[i, j]: signal (dBm) from an AP at ap_positions[i] to
    receiver_positions[j], computed exactly as the heatmap computes it.

    When AP spots and receivers are the same points, distance and wall
    crossings are the same in both directions, so only half the table is
    computed and then mirrored.
    """
    aps = np.asarray(ap_positions, dtype=float)
    receivers = np.asarray(receiver_positions, dtype=float)
    symmetric = aps.shape == receivers.shape and np.array_equal(aps, receivers)
    walls_split = _split_walls(walls)
    table = np.zeros((len(aps), len(receivers)), dtype=np.float32)
    batch_rows, batch_cells = [], []

    def flush():
        if not batch_rows:
            return
        frame = pd.concat(batch_rows, ignore_index=True)[MODEL_FEATURES]
        values = apply_residual_calibration(
            frame, artifacts.model.predict(frame), artifacts.calibration_model
        ).astype(np.float32)
        start = 0
        for i, js in batch_cells:
            chunk = values[start:start + len(js)]
            table[i, js] = chunk
            if symmetric:
                table[js, i] = chunk
            start += len(js)
        batch_rows.clear()
        batch_cells.clear()

    for i in range(len(aps)):
        js = np.arange(i, len(receivers)) if symmetric else np.arange(len(receivers))
        targets = receivers[js]
        counts, losses = wall_features_from(aps[i], targets, walls_split)
        distances = np.hypot(targets[:, 0] - aps[i, 0], targets[:, 1] - aps[i, 1])
        batch_rows.append(pd.DataFrame({
            "distance": np.maximum(distances, 0.1),
            "num_walls_intersected": counts,
            "wall_attenuation_sum": losses,
            "tx_power_dbm": float(tx_power_dbm),
            "frequency_ghz": float(frequency_ghz),
        }))
        batch_cells.append((i, js))
        if sum(len(js) for _, js in batch_cells) > 400_000:
            flush()
    flush()
    return table


def choose_ap_spots(
    positions: Sequence[tuple[float, float]],
    max_spots: int = 4000,
) -> list[int]:
    """Indexes of allowed AP spots, thinned only for very fine grids.

    The table has (AP spots x receivers) entries. With the default 1 m grid
    every receiver point is also an AP spot. For finer grids (e.g. 0.5 m on a
    large floor) every 2nd row/column of points is used as an AP spot, which
    keeps memory bounded; receivers are never thinned.
    """
    points = np.asarray(positions, dtype=float)
    if len(points) <= max_spots:
        return list(range(len(points)))
    step = int(np.ceil(np.sqrt(len(points) / max_spots)))
    xs, ys = np.unique(points[:, 0]), np.unique(points[:, 1])
    x_rank = {x: i for i, x in enumerate(xs)}
    y_rank = {y: i for i, y in enumerate(ys)}
    return [
        i for i, (x, y) in enumerate(points)
        if x_rank[x] % step == 0 and y_rank[y] % step == 0
    ]


# ---------------------------------------------------------------------------
# 2. Scoring
# ---------------------------------------------------------------------------

def layout_score(table: np.ndarray, layout: Sequence[int], threshold_dbm: float):
    """(covered points, mean best RSSI) for APs placed at the given rows."""
    best = table[list(layout)].max(axis=0)
    return int((best >= threshold_dbm).sum()), float(best.mean())


def _better(a, b) -> bool:
    """Compare scores: more covered points first, then stronger mean signal."""
    return a[0] > b[0] or (a[0] == b[0] and a[1] > b[1] + 1e-9)


# ---------------------------------------------------------------------------
# 3. Search
# ---------------------------------------------------------------------------

def _improve(table, layout, threshold_dbm, max_rounds=50):
    """Local search: repeatedly move one AP to the best spot, given the others."""
    layout = list(layout)
    covered_by_row = table >= threshold_dbm
    score = layout_score(table, layout, threshold_dbm)
    for _ in range(max_rounds):
        improved = False
        for k in range(len(layout)):
            others = [layout[m] for m in range(len(layout)) if m != k]
            if others:
                best_others = table[others].max(axis=0)
            else:
                best_others = np.full(table.shape[1], -np.inf, dtype=table.dtype)
            covered_others = best_others >= threshold_dbm
            # Score every candidate spot for AP k at once.
            covered = (covered_by_row | covered_others).sum(axis=1)
            mean_signal = np.maximum(table, best_others).mean(axis=1)
            candidate = int(np.lexsort((mean_signal, covered))[-1])
            new_score = (int(covered[candidate]), float(mean_signal[candidate]))
            if candidate not in layout and _better(new_score, score):
                layout[k], score, improved = candidate, new_score, True
        if not improved:
            break
    return layout, score


def _greedy(table, ap_count, threshold_dbm, first=None):
    """Add APs one at a time, each at the spot that adds the most coverage."""
    covered_by_row = table >= threshold_dbm
    layout = [] if first is None else [first]
    best = np.full(table.shape[1], -np.inf, dtype=table.dtype)
    if first is not None:
        best = table[first].copy()
    while len(layout) < ap_count:
        covered = (covered_by_row | (best >= threshold_dbm)).sum(axis=1)
        mean_signal = np.maximum(table, best).mean(axis=1)
        order = np.lexsort((mean_signal, covered))[::-1]
        choice = next(int(i) for i in order if int(i) not in layout)
        layout.append(choice)
        best = np.maximum(best, table[choice])
    return layout


@dataclass(frozen=True)
class PlacementResult:
    positions: list[tuple[float, float]]
    channels: list[int]
    covered_points: int
    coverage_percent: float
    mean_rssi_dbm: float
    layouts_tried: int


def optimize_placement(
    table: np.ndarray,
    ap_positions: Sequence[tuple[float, float]],
    ap_count: int,
    threshold_dbm: float,
    channels: Sequence[int] = (1, 6, 11),
    restarts: int = 8,
    seed: int = 42,
) -> PlacementResult:
    """Find AP spots that maximise heatmap coverage, then assign channels."""
    n = table.shape[0]
    if not 1 <= ap_count <= n:
        raise ValueError("ap_count must be between 1 and the number of allowed spots.")
    rng = np.random.default_rng(seed)

    starts = [_greedy(table, ap_count, threshold_dbm)]
    for _ in range(restarts):
        starts.append(_greedy(table, ap_count, threshold_dbm, first=int(rng.integers(n))))

    best_layout, best_score = None, (-1, -np.inf)
    for start in starts:
        layout, score = _improve(table, start, threshold_dbm)
        if _better(score, best_score):
            best_layout, best_score = layout, score

    chosen_channels = assign_channels(table, best_layout, channels)
    return PlacementResult(
        positions=[tuple(map(float, ap_positions[i])) for i in best_layout],
        channels=chosen_channels,
        covered_points=best_score[0],
        coverage_percent=100.0 * best_score[0] / table.shape[1],
        mean_rssi_dbm=best_score[1],
        layouts_tried=len(starts) * n * ap_count,   # approximate, for reporting
    )


# ---------------------------------------------------------------------------
# 4. Channels
# ---------------------------------------------------------------------------

def assign_channels(
    table: np.ndarray,
    layout: Sequence[int],
    channels: Sequence[int] = (1, 6, 11),
    hearing_dbm: float = -82.0,
) -> list[int]:
    """Give APs channels so that APs heard in the same places differ.

    Two APs 'overlap' at a point if both are received above hearing_dbm
    (a common Wi-Fi carrier-sense level). APs are coloured greedily, most
    overlapping first, choosing the channel with the least shared overlap.
    """
    k = len(layout)
    heard = table[list(layout)] >= hearing_dbm          # (k, R)
    overlap = heard.astype(np.int32) @ heard.T.astype(np.int32)
    np.fill_diagonal(overlap, 0)
    assigned = [None] * k
    for ap in np.argsort(-overlap.sum(axis=1)):
        cost = {
            c: sum(overlap[ap, other] for other in range(k) if assigned[other] == c)
            for c in channels
        }
        assigned[ap] = min(channels, key=lambda c: (cost[c], channels.index(c)))
    return [int(c) for c in assigned]


# ---------------------------------------------------------------------------
# 5. Entry point used by the app
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PlacedAccessPoint:
    """Minimal AP record with the attributes the app and heatmap use."""

    x: float
    y: float
    channel: int


def plan_access_points(
    receiver_positions: Sequence[tuple[float, float]],
    walls: Sequence[Mapping[str, object]],
    artifacts: PredictionArtifacts,
    ap_count: int,
    tx_power_dbm: float,
    frequency_ghz: float,
    threshold_dbm: float,
) -> tuple[PlacementResult, list[PlacedAccessPoint]]:
    """Build the signal table, optimise placement and return AP records."""
    if not receiver_positions:
        raise ValueError("At least one receiver position is required.")
    spot_indexes = choose_ap_spots(receiver_positions)
    if ap_count > len(spot_indexes):
        raise ValueError("More APs requested than there are allowed AP spots.")
    ap_positions = [receiver_positions[i] for i in spot_indexes]
    table = build_signal_table(
        ap_positions, receiver_positions, walls, artifacts, tx_power_dbm, frequency_ghz
    )
    result = optimize_placement(table, ap_positions, ap_count, threshold_dbm)
    access_points = [
        PlacedAccessPoint(x=x, y=y, channel=channel)
        for (x, y), channel in zip(result.positions, result.channels)
    ]
    return result, access_points

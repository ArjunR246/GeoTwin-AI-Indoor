"""Calibrate a simple radio model against measured Wi-Fi scans.

The trained AI model was built from simulated data, so its absolute signal
levels may not match a real building. This module fits a standard, easy to
explain model to real measurements:

    signal (dBm) = P0 - 10 * n * log10(distance_m) - wall_loss_db * walls_crossed

P0 is the signal 1 m from an AP (fixed at a realistic -35 dBm, because
corridor-only measurements cannot tell P0 and AP distance apart), n is how fast
signal fades with distance, and wall_loss_db is the loss per wall crossed.
The current AP positions are unknown, so they are estimated at the same time.

Reliability is checked honestly: the corridor is split into blocks, the model
is fitted on half of them and tested on the other half, and the error is
compared with simply guessing the average measured value.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from services.fast_placement import _split_walls, wall_features_from

DEAD_ZONE_DBM = -85.0          # below this, a phone shows "no signal"
DEFAULT_P0_DBM = -35.0         # typical signal 1 m from an enterprise AP


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def sample_walked_cells(
    rssi_dbm: np.ndarray,
    measured: np.ndarray,
    walked: np.ndarray,
    width_m: float,
    height_m: float,
    cell_m: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Turn a measured map (floorplan pixels) into cell centres in metres.

    A cell counts as walked if at least 30% of it was walked. Its value is the
    median measured signal if at least half of it had signal, otherwise NaN
    (a dead zone).
    """
    height_px, width_px = walked.shape
    cells, values = [], []
    for y in np.arange(cell_m / 2, height_m, cell_m):
        y0 = int((y - cell_m / 2) * height_px / height_m)
        y1 = max(y0 + 1, int((y + cell_m / 2) * height_px / height_m))
        for x in np.arange(cell_m / 2, width_m, cell_m):
            x0 = int((x - cell_m / 2) * width_px / width_m)
            x1 = max(x0 + 1, int((x + cell_m / 2) * width_px / width_m))
            if walked[y0:y1, x0:x1].mean() < 0.3:
                continue
            signal = measured[y0:y1, x0:x1]
            cells.append((float(x), float(y)))
            values.append(
                float(np.nanmedian(rssi_dbm[y0:y1, x0:x1][signal]))
                if signal.mean() >= 0.5 else np.nan
            )
    return np.array(cells, dtype=float).reshape(-1, 2), np.array(values, dtype=float)


def path_features(
    sources: Sequence[tuple[float, float]],
    targets: Sequence[tuple[float, float]],
    walls: Sequence[Mapping[str, object]],
) -> tuple[np.ndarray, np.ndarray]:
    """Distance (m, at least 1) and wall crossings for every source-target pair."""
    targets = np.asarray(targets, dtype=float)
    split = _split_walls(walls)
    distance = np.empty((len(sources), len(targets)), dtype=np.float32)
    crossings = np.empty((len(sources), len(targets)), dtype=np.float32)
    for i, source in enumerate(np.asarray(sources, dtype=float)):
        count, _ = wall_features_from(source, targets, split)
        crossings[i] = count
        distance[i] = np.maximum(np.hypot(targets[:, 0] - source[0], targets[:, 1] - source[1]), 1.0)
    return distance, crossings


# ---------------------------------------------------------------------------
# Model and fitting
# ---------------------------------------------------------------------------

@dataclass
class CalibratedModel:
    """Fitted parameters plus how well they predict unseen measurements."""

    p0_dbm: float
    exponent: float
    wall_loss_db: float
    current_aps: list[tuple[float, float]]
    fit_error_db: float                 # mean absolute error on all measured cells
    test_error_db: float                # ... on held-out cells (fitted without them)
    baseline_error_db: float            # ... if you always guess the average
    test_dead_accuracy: float           # held-out dead cells predicted below -85 dBm
    test_signal_accuracy: float         # held-out signal cells predicted above -85 dBm
    measured_cells: int = 0
    dead_cells: int = 0
    notes: list[str] = field(default_factory=list)
    # (exponent, wall loss) pairs that fit almost as well as the best one
    # (cost within 10%). Used to show a range instead of a single number.
    plausible: list[tuple[float, float]] = field(default_factory=list)

    @property
    def is_useful(self) -> bool:
        """Clearly better than guessing on cells it never saw."""
        return self.test_error_db < 0.9 * self.baseline_error_db

    def signal(self, distance: np.ndarray, crossings: np.ndarray,
               exponent: float | None = None, wall_loss_db: float | None = None) -> np.ndarray:
        exponent = self.exponent if exponent is None else exponent
        wall_loss_db = self.wall_loss_db if wall_loss_db is None else wall_loss_db
        return (self.p0_dbm - 10 * exponent * np.log10(distance)
                - wall_loss_db * crossings).astype(np.float32)

    def outcome_range(self, distance: np.ndarray, crossings: np.ndarray,
                      layout: Sequence[int], threshold_dbm: float) -> tuple[float, float]:
        """Lowest and highest % meeting the threshold across plausible fits."""
        pairs = self.plausible or [(self.exponent, self.wall_loss_db)]
        shares = [
            float(100 * (self.signal(distance[list(layout)], crossings[list(layout)], e, w)
                         .max(0) >= threshold_dbm).mean())
            for e, w in pairs
        ]
        return min(shares), max(shares)


def _loss(distance_log, crossings, exponent, wall_loss):
    return -10 * exponent * distance_log - wall_loss * crossings


def _cost(predicted, values):
    """Squared error on signal cells + penalty when dead cells look alive."""
    signal = np.isfinite(values)
    cost = ((predicted[..., signal] - values[signal]) ** 2).sum(-1)
    return cost + (np.maximum(0, predicted[..., ~signal] - DEAD_ZONE_DBM) ** 2).sum(-1)


def _search_layout(loss, p0, values, ap_count, rng, iterations=8):
    """Local search: move one AP at a time to the spot that lowers the cost most."""
    layout = list(rng.choice(loss.shape[0], ap_count, replace=False))
    best = None
    for _ in range(iterations):
        changed = False
        for slot in range(ap_count):
            others = [ap for i, ap in enumerate(layout) if i != slot]
            rest = loss[others].max(0) if others else np.full(loss.shape[1], -np.inf, np.float32)
            costs = _cost(np.maximum(loss, rest) + p0, values)
            choice = int(costs.argmin())
            if best is None or costs[choice] < best - 1e-6:
                layout[slot], best, changed = choice, float(costs[choice]), True
        if not changed:
            break
    return layout, best


def _fit(distance, crossings, values, ap_count, p0, seeds):
    log_distance = np.log10(distance)
    exponents = np.arange(1.6, 5.01, 0.2)
    wall_losses = np.arange(0.0, 12.01, 0.5)
    best = None
    for seed in range(seeds):
        rng = np.random.default_rng(seed)
        exponent, wall_loss = 3.0, 3.0
        for _ in range(3):                     # alternate: layout, then physics
            loss = _loss(log_distance, crossings, exponent, wall_loss)
            layout, _ = _search_layout(loss, p0, values, ap_count, rng)
            ld, cr = log_distance[layout], crossings[layout]
            grid = [
                (_cost(_loss(ld, cr, e, w).max(0) + p0, values), e, w)
                for e in exponents for w in wall_losses
            ]
            cost, exponent, wall_loss = min(grid, key=lambda item: item[0])
        if best is None or cost < best[0]:
            plausible = [(float(e), float(w)) for c, e, w in grid if c <= 1.1 * cost]
            best = (cost, float(exponent), float(wall_loss), layout, plausible)
    return best


def _errors(distance, crossings, values, p0, exponent, wall_loss, layout):
    predicted = _loss(np.log10(distance[layout]), crossings[layout], exponent, wall_loss).max(0) + p0
    signal = np.isfinite(values)
    error = float(np.abs(predicted[signal] - values[signal]).mean()) if signal.any() else float("nan")
    dead_ok = float((predicted[~signal] < DEAD_ZONE_DBM).mean()) if (~signal).any() else float("nan")
    signal_ok = float((predicted[signal] >= DEAD_ZONE_DBM).mean()) if signal.any() else float("nan")
    return error, dead_ok, signal_ok


def fit_calibration(
    cells: np.ndarray,
    values: np.ndarray,
    candidates: Sequence[tuple[float, float]],
    walls: Sequence[Mapping[str, object]],
    ap_count: int,
    p0_dbm: float = DEFAULT_P0_DBM,
    block_m: float = 6.0,
    seeds: int = 3,
) -> CalibratedModel:
    """Fit the model and estimate current AP positions from measured cells.

    cells:      (N, 2) cell centres in metres (from sample_walked_cells)
    values:     (N,) measured dBm, NaN for dead zones
    candidates: where the current APs could be (e.g. rooms outside no-AP zones)
    """
    cells = np.asarray(cells, dtype=float)
    values = np.asarray(values, dtype=float)
    signal = np.isfinite(values)
    if signal.sum() < 10:
        raise ValueError("Need at least 10 measured cells with signal to calibrate.")
    if len(candidates) < ap_count:
        raise ValueError("Fewer allowed AP spots than the number of current APs.")

    distance, crossings = path_features(candidates, cells, walls)

    # Held-out check: fit on alternating corridor blocks, test on the others.
    block = ((cells[:, 0] // block_m).astype(int) + (cells[:, 1] // block_m).astype(int)) % 2
    train, test = block == 0, block == 1
    notes = []
    if np.isfinite(values[train]).sum() >= 5 and np.isfinite(values[test]).sum() >= 5:
        _, e, w, layout, _ = _fit(distance[:, train], crossings[:, train], values[train], ap_count, p0_dbm, seeds)
        test_error, dead_ok, signal_ok = _errors(
            distance[:, test], crossings[:, test], values[test], p0_dbm, e, w, layout)
        baseline = float(np.abs(values[test][np.isfinite(values[test])]
                                - np.nanmean(values[train])).mean())
    else:
        test_error = dead_ok = signal_ok = baseline = float("nan")
        notes.append("Too few measured cells to hold some back for testing.")

    _, exponent, wall_loss, layout, plausible = _fit(distance, crossings, values, ap_count, p0_dbm, seeds)
    fit_error, _, _ = _errors(distance, crossings, values, p0_dbm, exponent, wall_loss, layout)
    candidates = [tuple(map(float, c)) for c in candidates]
    return CalibratedModel(
        p0_dbm=p0_dbm, exponent=exponent, wall_loss_db=wall_loss,
        current_aps=[candidates[i] for i in layout],
        fit_error_db=fit_error, test_error_db=test_error, baseline_error_db=baseline,
        test_dead_accuracy=dead_ok, test_signal_accuracy=signal_ok,
        measured_cells=int(signal.sum()), dead_cells=int((~signal).sum()), notes=notes,
        plausible=plausible,
    )


def band_shares(values: np.ndarray, threshold_dbm: float = -67.0) -> dict[str, float]:
    """Percent of cells in each band; NaN or below -85 dBm counts as dead zone."""
    v = np.where(np.isfinite(values), values, -999.0)
    return {
        "Excellent": float(100 * (v >= -50).mean()),
        "Good": float(100 * ((v >= -67) & (v < -50)).mean()),
        "Fair": float(100 * ((v >= -75) & (v < -67)).mean()),
        "Poor": float(100 * ((v >= DEAD_ZONE_DBM) & (v < -75)).mean()),
        "Dead zone": float(100 * (v < DEAD_ZONE_DBM).mean()),
        "meets_threshold": float(100 * (v >= threshold_dbm).mean()),
    }


__all__ = [
    "CalibratedModel", "DEAD_ZONE_DBM", "DEFAULT_P0_DBM", "band_shares",
    "fit_calibration", "path_features", "sample_walked_cells",
]

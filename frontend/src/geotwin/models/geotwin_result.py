from dataclasses import dataclass
from typing import Tuple


Position = Tuple[float, float]


@dataclass(frozen=True)
class APResult:
    """Final optimized access-point configuration."""

    position: Position
    channel: int


@dataclass(frozen=True)
class RFResult:
    """RF performance of the optimized configuration."""

    coverage_percent: float
    dead_zone_percent: float

    average_rssi_dbm: float
    minimum_rssi_dbm: float

    average_sinr_db: float
    minimum_sinr_db: float


@dataclass(frozen=True)
class ParetoSolution:
    """One meaningful Pareto-optimal configuration."""

    ap_count: int
    coverage_percent: float
    average_rssi_dbm: float
    minimum_rssi_dbm: float
    average_sinr_db: float
    minimum_sinr_db: float

    access_points: Tuple[APResult, ...]


@dataclass(frozen=True)
class GeoTwinResult:
    """
    Consolidated result produced by the GeoTwin AI Indoor pipeline.

    This object is intended to become the single source of truth
    for visualization, reporting, APIs, and the future UI.
    """

    # ------------------------------------------------------------
    # SCENARIO
    # ------------------------------------------------------------

    building_width_m: float
    building_length_m: float
    grid_resolution_m: float
    floors: int
    total_grid_points: int

    # ------------------------------------------------------------
    # BEST OPTIMIZED CONFIGURATION
    # ------------------------------------------------------------

    access_points: Tuple[APResult, ...]
    ap_count: int

    # ------------------------------------------------------------
    # RF PERFORMANCE
    # ------------------------------------------------------------

    rf: RFResult

    # ------------------------------------------------------------
    # PARETO FRONT
    # ------------------------------------------------------------

    pareto_solutions: Tuple[ParetoSolution, ...]

    # ------------------------------------------------------------
    # OPTIMIZATION
    # ------------------------------------------------------------

    generations_completed: int

    @property
    def coverage_percent(self) -> float:
        """Return optimized coverage."""
        return self.rf.coverage_percent

    @property
    def average_rssi_dbm(self) -> float:
        """Return optimized average RSSI."""
        return self.rf.average_rssi_dbm

    @property
    def average_sinr_db(self) -> float:
        """Return optimized average SINR."""
        return self.rf.average_sinr_db
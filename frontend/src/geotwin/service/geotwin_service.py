"""
GeoTwin AI Indoor - Service Layer

Framework-independent application service for running
the GeoTwin AI Indoor pipeline.
"""

from __future__ import annotations

from geotwin.optimization.nsga2 import NSGA2Config
from geotwin.pipeline import GeoTwinResult, run_geotwin
from geotwin.scenario.indoor import IndoorScenario


class GeoTwinService:
    """
    Application service for the GeoTwin AI Indoor engine.

    This layer keeps application-facing execution separate
    from the RF and optimization implementation.
    """

    def run(
        self,
        scenario: IndoorScenario | None = None,
        config: NSGA2Config | None = None,
    ) -> GeoTwinResult:
        """
        Execute the complete GeoTwin pipeline.

        Parameters
        ----------
        scenario:
            Indoor building scenario. If omitted, the default
            scenario is used.

        config:
            NSGA-II and RF configuration. If omitted, the
            default optimizer configuration is used.

        Returns
        -------
        GeoTwinResult
            Complete scenario, optimization, best solution,
            and RF evaluation.
        """

        if scenario is None:
            scenario = IndoorScenario()

        if config is None:
            config = NSGA2Config()

        return run_geotwin(
            scenario=scenario,
            config=config,
        )
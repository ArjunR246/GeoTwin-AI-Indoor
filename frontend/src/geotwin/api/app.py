"""
GeoTwin AI Indoor - FastAPI Application

HTTP API boundary for the GeoTwin AI Indoor engine.

Architecture:

    HTTP Request
        ↓
    FastAPI
        ↓
    GeoTwinService
        ↓
    GeoTwin Pipeline
        ↓
    NSGA-II + RF Engine
        ↓
    GeoTwinResult
        ↓
    Serializer
        ↓
    JSON Response
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException

from geotwin.service.geotwin_service import GeoTwinService
from geotwin.service.serializers import serialize_geotwin_result


# ======================================================================
# APPLICATION
# ======================================================================

app = FastAPI(
    title="GeoTwin AI Indoor API",
    description=(
        "API for indoor RF coverage analysis, "
        "channel-aware AP placement, and NSGA-II optimization."
    ),
    version="0.1.0",
)


# ======================================================================
# SERVICE
# ======================================================================

service = GeoTwinService()


# ======================================================================
# HEALTH CHECK
# ======================================================================

@app.get("/health")
def health_check() -> dict[str, str]:
    """
    Check whether the GeoTwin API is running.
    """

    return {
        "status": "ok",
        "service": "GeoTwin AI Indoor",
    }


# ======================================================================
# API INFORMATION
# ======================================================================

@app.get("/")
def root() -> dict[str, str]:
    """
    Return basic API information.
    """

    return {
        "name": "GeoTwin AI Indoor API",
        "version": "0.1.0",
        "status": "running",
    }


# ======================================================================
# RUN DEFAULT OPTIMIZATION
# ======================================================================

@app.post("/optimize")
def optimize() -> dict[str, Any]:
    """
    Run the GeoTwin AI Indoor optimization pipeline.

    Uses the default IndoorScenario and NSGA2Config.

    Returns:
        JSON-compatible GeoTwin result.
    """

    try:
        result = service.run()

        return serialize_geotwin_result(
            result
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc


# ======================================================================
# RESULT SUMMARY
# ======================================================================

@app.get("/result/summary")
def result_summary() -> dict[str, Any]:
    """
    Run the optimization and return a compact result summary.

    The endpoint supports the public GeoTwinResult representation
    returned by the current API/service architecture.
    """

    try:
        result = service.run()

        # ------------------------------------------------------------
        # BUILDING / SCENARIO
        # ------------------------------------------------------------

        building = {
            "width_m": result.building_width_m,
            "length_m": result.building_length_m,
            "grid_resolution_m": result.grid_resolution_m,
            "floors": result.floors,
        }

        # ------------------------------------------------------------
        # BEST SOLUTION
        # ------------------------------------------------------------

        best_solution = {
            "ap_count": result.ap_count,
            "positions": [
                ap.position
                for ap in result.access_points
            ],
            "channels": [
                ap.channel
                for ap in result.access_points
            ],
        }

        # ------------------------------------------------------------
        # RF
        # ------------------------------------------------------------

        rf = {
            "coverage_percent": (
                result.rf.coverage_percent
            ),
            "dead_zone_percent": (
                result.rf.dead_zone_percent
            ),
            "average_rssi_dbm": (
                result.rf.average_rssi_dbm
            ),
            "minimum_rssi_dbm": (
                result.rf.minimum_rssi_dbm
            ),
            "average_sinr_db": (
                result.rf.average_sinr_db
            ),
            "minimum_sinr_db": (
                result.rf.minimum_sinr_db
            ),
        }

        # ------------------------------------------------------------
        # OPTIMIZATION
        # ------------------------------------------------------------

        optimization = {
            "generations_completed": (
                result.generations_completed
            ),
            "pareto_solutions": len(
                result.pareto_solutions
            ),
        }

        # ------------------------------------------------------------
        # PUBLIC SUMMARY
        # ------------------------------------------------------------

        return {
            "building": building,
            "best_solution": best_solution,
            "rf": rf,
            "optimization": optimization,
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc
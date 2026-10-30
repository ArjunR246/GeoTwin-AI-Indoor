"""
GeoTwin AI Indoor - Result Serialization

Converts GeoTwin pipeline results into JSON/API-friendly
Python dictionaries.

The serializer supports both:

1. The internal pipeline GeoTwinResult.
2. The public models.geotwin_result.GeoTwinResult.

This keeps the API boundary stable while allowing the
pipeline and public result model to evolve independently.
"""

from typing import Any

from geotwin.models.geotwin_result import (
    APResult,
    GeoTwinResult as PublicGeoTwinResult,
    ParetoSolution,
    RFResult,
)


# ======================================================================
# ACCESS POINT
# ======================================================================

def serialize_ap_result(
    ap: APResult,
) -> dict[str, Any]:
    """Serialize one access-point result."""

    return {
        "position": {
            "x_m": ap.position[0],
            "y_m": ap.position[1],
        },
        "channel": ap.channel,
    }


# ======================================================================
# RF RESULT
# ======================================================================

def serialize_rf_result(
    rf: RFResult,
) -> dict[str, Any]:
    """Serialize RF performance."""

    return {
        "coverage_percent": rf.coverage_percent,
        "dead_zone_percent": rf.dead_zone_percent,
        "average_rssi_dbm": rf.average_rssi_dbm,
        "minimum_rssi_dbm": rf.minimum_rssi_dbm,
        "average_sinr_db": rf.average_sinr_db,
        "minimum_sinr_db": rf.minimum_sinr_db,
    }


# ======================================================================
# PARETO SOLUTION
# ======================================================================

def serialize_pareto_solution(
    solution: ParetoSolution,
) -> dict[str, Any]:
    """Serialize one public Pareto-optimal solution."""

    return {
        "ap_count": solution.ap_count,
        "coverage_percent": solution.coverage_percent,
        "average_rssi_dbm": solution.average_rssi_dbm,
        "minimum_rssi_dbm": solution.minimum_rssi_dbm,
        "average_sinr_db": solution.average_sinr_db,
        "minimum_sinr_db": solution.minimum_sinr_db,
        "access_points": [
            serialize_ap_result(ap)
            for ap in solution.access_points
        ],
    }


# ======================================================================
# PUBLIC RESULT SERIALIZATION
# ======================================================================

def _serialize_public_result(
    result: PublicGeoTwinResult,
) -> dict[str, Any]:
    """
    Serialize an already-consolidated public GeoTwinResult.

    This path is used by the existing serializer unit tests
    and by callers that already provide the public result model.
    """

    return {
        "scenario": {
            "building_width_m": result.building_width_m,
            "building_length_m": result.building_length_m,
            "grid_resolution_m": result.grid_resolution_m,
            "floors": result.floors,
            "total_grid_points": result.total_grid_points,
        },
        "best_configuration": {
            "ap_count": result.ap_count,
            "access_points": [
                serialize_ap_result(ap)
                for ap in result.access_points
            ],
        },
        "rf": serialize_rf_result(result.rf),
        "pareto_solutions": [
            serialize_pareto_solution(solution)
            for solution in result.pareto_solutions
        ],
        "optimization": {
            "generations_completed": (
                result.generations_completed
            ),
        },
    }


# ======================================================================
# INTERNAL PIPELINE SERIALIZATION
# ======================================================================

def _serialize_pipeline_result(
    result: Any,
) -> dict[str, Any]:
    """
    Serialize the internal pipeline GeoTwinResult.

    Internal structure:

        result.scenario
        result.optimization
        result.best_solution
        result.rf_evaluation

    Public API structure:

        scenario
        best_configuration
        rf
        pareto_solutions
        optimization
    """

    scenario = result.scenario
    best_solution = result.best_solution
    rf = result.rf_evaluation

    # ------------------------------------------------------------
    # BEST CONFIGURATION
    # ------------------------------------------------------------

    best_access_points = [
        APResult(
            position=position,
            channel=channel,
        )
        for position, channel in best_solution.ap_configuration
    ]

    # ------------------------------------------------------------
    # PARETO SOLUTIONS
    # ------------------------------------------------------------

    pareto_solutions: list[ParetoSolution] = []

    for solution in result.optimization.pareto_front:

        access_points = [
            APResult(
                position=position,
                channel=channel,
            )
            for position, channel in solution.ap_configuration
        ]

        # The optimizer currently stores three objectives:
        #
        # objectives[0] = negative coverage
        # objectives[1] = AP count
        # objectives[2] = interference
        #
        # It does not retain a complete RF evaluation for every
        # Pareto individual. Therefore coverage is taken directly
        # from the objective, while the current RF metrics are
        # available from the final selected evaluation.
        pareto_solutions.append(
            ParetoSolution(
                ap_count=solution.ap_count,
                coverage_percent=-solution.objectives[0],
                average_rssi_dbm=(
                    _pipeline_metric(
                        solution,
                        result,
                        "average_rssi_dbm",
                    )
                ),
                minimum_rssi_dbm=(
                    _pipeline_metric(
                        solution,
                        result,
                        "minimum_rssi_dbm",
                    )
                ),
                average_sinr_db=(
                    _pipeline_metric(
                        solution,
                        result,
                        "average_sinr_db",
                    )
                ),
                minimum_sinr_db=(
                    _pipeline_metric(
                        solution,
                        result,
                        "minimum_sinr_db",
                    )
                ),
                access_points=tuple(
                    access_points
                ),
            )
        )

    # ------------------------------------------------------------
    # PUBLIC RF REPRESENTATION
    # ------------------------------------------------------------

    public_rf = RFResult(
        coverage_percent=rf.coverage_percent,
        dead_zone_percent=rf.dead_zone_percent,
        average_rssi_dbm=rf.average_rssi_dbm,
        minimum_rssi_dbm=rf.minimum_rssi_dbm,
        average_sinr_db=rf.average_sinr_db,
        minimum_sinr_db=rf.minimum_sinr_db,
    )

    # ------------------------------------------------------------
    # FINAL API DICTIONARY
    # ------------------------------------------------------------

    return {
        "scenario": {
            "building_width_m": scenario.width_m,
            "building_length_m": scenario.length_m,
            "grid_resolution_m": scenario.grid_resolution_m,
            "floors": scenario.floors,
            "total_grid_points": scenario.total_grid_points,
        },
        "best_configuration": {
            "ap_count": best_solution.ap_count,
            "access_points": [
                serialize_ap_result(ap)
                for ap in best_access_points
            ],
        },
        "rf": serialize_rf_result(public_rf),
        "pareto_solutions": [
            serialize_pareto_solution(solution)
            for solution in pareto_solutions
        ],
        "optimization": {
            "generations_completed": (
                result.optimization.generations_completed
            ),
        },
    }


# ======================================================================
# PIPELINE METRIC HELPER
# ======================================================================

def _pipeline_metric(
    solution: Any,
    pipeline_result: Any,
    metric_name: str,
) -> float:
    """
    Return an RF metric for a pipeline Pareto solution.

    The current optimizer does not store a GridEvaluation for
    every Pareto solution. The final selected solution does have
    a complete RF evaluation.

    Until per-Pareto-solution RF evaluations are introduced,
    the final RF evaluation is used as the available public
    metric representation.
    """

    return float(
        getattr(
            pipeline_result.rf_evaluation,
            metric_name,
        )
    )


# ======================================================================
# MAIN SERIALIZER
# ======================================================================

def serialize_geotwin_result(
    result: Any,
) -> dict[str, Any]:
    """
    Serialize either supported GeoTwinResult representation.

    Supported inputs:

        models.geotwin_result.GeoTwinResult
        pipeline.GeoTwinResult

    Returns:
        JSON-compatible dictionary containing only the
        public API sections.
    """

    # ------------------------------------------------------------
    # PUBLIC RESULT MODEL
    # ------------------------------------------------------------

    if isinstance(
        result,
        PublicGeoTwinResult,
    ):
        return _serialize_public_result(
            result
        )

    # ------------------------------------------------------------
    # INTERNAL PIPELINE RESULT
    # ------------------------------------------------------------

    if (
        hasattr(result, "scenario")
        and hasattr(result, "optimization")
        and hasattr(result, "best_solution")
        and hasattr(result, "rf_evaluation")
    ):
        return _serialize_pipeline_result(
            result
        )

    # ------------------------------------------------------------
    # UNSUPPORTED RESULT
    # ------------------------------------------------------------

    raise TypeError(
        "Unsupported GeoTwinResult type: "
        f"{type(result).__module__}."
        f"{type(result).__name__}"
    )
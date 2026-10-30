"""
GeoTwin AI Indoor - Production Pipeline

End-to-end integration layer connecting:

    IndoorScenario
        ↓
    Channel-aware NSGA-II
        ↓
    Best Pareto solution
        ↓
    RF grid evaluation
        ↓
    GeoTwin pipeline result
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from geotwin.optimization.nsga2 import (
    Individual,
    NSGA2Config,
    OptimizationResult,
    evaluate_individual,
    optimize,
)
from geotwin.rf.coverage import GridEvaluation
from geotwin.scenario.indoor import IndoorScenario


@dataclass(frozen=True)
class GeoTwinResult:
    """
    Final result of the GeoTwin AI Indoor pipeline.
    """

    scenario: IndoorScenario
    optimization: OptimizationResult
    best_solution: Individual
    rf_evaluation: GridEvaluation


def select_best_solution(
    pareto_front: list[Individual],
) -> Individual:
    """
    Select the strongest practical solution from the Pareto front.

    Current NSGA-II objective definition:

        objectives[0] = negative coverage
        objectives[1] = AP count
        objectives[2] = interference

    Since NSGA-II uses minimization objectives, the preferred
    solution is:

        1. Highest coverage
        2. Lowest AP count
        3. Lowest interference

    The objective tuple is used directly so that this helper
    remains consistent with the optimizer's actual objective
    definition.
    """

    if not pareto_front:
        raise ValueError(
            "Pareto front cannot be empty."
        )

    return min(
        pareto_front,
        key=lambda individual: (
            individual.objectives[0],
            individual.objectives[1],
            individual.objectives[2],
        ),
    )


def run_geotwin(
    scenario: Optional[IndoorScenario] = None,
    config: Optional[NSGA2Config] = None,
) -> GeoTwinResult:
    """
    Run the complete GeoTwin AI Indoor workflow.

    The pipeline:

        1. Creates/uses an indoor scenario.
        2. Runs production channel-aware NSGA-II.
        3. Selects the best Pareto solution.
        4. Performs a final RF evaluation.
        5. Returns all results in one object.
    """

    # ------------------------------------------------------------
    # 1. CREATE / VALIDATE SCENARIO
    # ------------------------------------------------------------

    if scenario is None:
        scenario = IndoorScenario()

    scenario.validate()

    # ------------------------------------------------------------
    # 2. CREATE OPTIMIZATION CONFIGURATION
    # ------------------------------------------------------------

    if config is None:
        config = NSGA2Config()

    # ------------------------------------------------------------
    # 3. RUN NSGA-II OPTIMIZATION
    # ------------------------------------------------------------

    optimization = optimize(
        receiver_positions=scenario.receiver_positions,
        valid_positions=scenario.valid_ap_positions,
        blocked_positions=scenario.blocked_positions,
        config=config,
    )

    # ------------------------------------------------------------
    # 4. VALIDATE PARETO FRONT
    # ------------------------------------------------------------

    if not optimization.pareto_front:
        raise RuntimeError(
            "NSGA-II returned an empty Pareto front."
        )

    # ------------------------------------------------------------
    # 5. SELECT BEST PRACTICAL SOLUTION
    # ------------------------------------------------------------

    best_solution = select_best_solution(
        optimization.pareto_front
    )

    # ------------------------------------------------------------
    # 6. FINAL RF EVALUATION
    # ------------------------------------------------------------

    rf_evaluation = evaluate_individual(
        individual=best_solution,
        receiver_positions=scenario.receiver_positions,
        valid_positions=scenario.valid_ap_positions,
        blocked_positions=scenario.blocked_positions,
        config=config,
    )

    # ------------------------------------------------------------
    # 7. RETURN COMPLETE PIPELINE RESULT
    # ------------------------------------------------------------

    return GeoTwinResult(
        scenario=scenario,
        optimization=optimization,
        best_solution=best_solution,
        rf_evaluation=rf_evaluation,
    )
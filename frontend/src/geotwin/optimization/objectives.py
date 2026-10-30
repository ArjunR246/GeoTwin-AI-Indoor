from dataclasses import dataclass
import math

from geotwin.rf.coverage import GridEvaluation


@dataclass(frozen=True)
class ObjectiveValues:
    """
    Objective values used by the NSGA-II optimizer.

    All objectives are minimization objectives.

    Objective order:
        1. Coverage
        2. AP count
        3. Interference power

    Interference is represented as power in mW rather than dBm
    because mW provides a physically meaningful minimization
    objective:

        lower power = lower interference = better
    """

    coverage: float
    ap_count: int
    interference: float

    @property
    def values(self) -> tuple[float, float, float]:
        """Return objectives in NSGA-II order."""

        return (
            float(self.coverage),
            float(self.ap_count),
            float(self.interference),
        )


def coverage_objective(
    coverage_percent: float,
) -> float:
    """
    Convert coverage maximization into a minimization objective.

    Example:
        95% coverage -> -95

    Higher coverage is better, therefore the value is negated.
    """

    if not isinstance(
        coverage_percent,
        (int, float),
    ):
        raise ValueError(
            "Coverage must be numeric."
        )

    coverage_percent = float(
        coverage_percent
    )

    if not math.isfinite(
        coverage_percent
    ):
        raise ValueError(
            "Coverage must be finite."
        )

    if not 0.0 <= coverage_percent <= 100.0:
        raise ValueError(
            "Coverage must be between 0 and 100 percent."
        )

    return -coverage_percent


def ap_count_objective(
    ap_count: int,
) -> float:
    """
    Minimize the number of access points.
    """

    if isinstance(ap_count, bool):
        raise ValueError(
            "AP count must be an integer."
        )

    if not isinstance(
        ap_count,
        int,
    ):
        raise ValueError(
            "AP count must be an integer."
        )

    if ap_count <= 0:
        raise ValueError(
            "AP count must be greater than zero."
        )

    return float(ap_count)


def interference_dbm_to_mw(
    interference_dbm: float,
) -> float:
    """
    Convert interference power from dBm to mW.

    Formula:

        P_mW = 10 ** (P_dBm / 10)

    Examples:

        -60 dBm -> 0.001 mW
        -50 dBm -> 0.01 mW
        -40 dBm -> 0.1 mW

    The RF layer uses -inf dBm to represent no measurable
    interference. That is converted to exactly 0.0 mW.
    """

    if not isinstance(
        interference_dbm,
        (int, float),
    ):
        raise ValueError(
            "Interference must be numeric."
        )

    interference_dbm = float(
        interference_dbm
    )

    if math.isnan(
        interference_dbm
    ):
        raise ValueError(
            "Interference cannot be NaN."
        )

    if interference_dbm == float(
        "-inf"
    ):
        return 0.0

    if interference_dbm == float(
        "inf"
    ):
        return float("inf")

    return 10.0 ** (
        interference_dbm / 10.0
    )


def interference_objective(
    interference_dbm: float,
) -> float:
    """
    Convert RF interference from dBm into the NSGA-II
    interference objective.

    The objective is expressed in mW.

    This is important because NSGA-II minimizes objectives.

    Physical interpretation:

        lower interference power = better

    Therefore:

        -inf dBm -> 0.0 mW
        -70 dBm  -> 0.0001 mW
        -60 dBm  -> 0.001 mW
        -50 dBm  -> 0.01 mW

    The returned value is an optimization metric in mW.
    """

    return interference_dbm_to_mw(
        interference_dbm
    )


def calculate_objectives(
    evaluation: GridEvaluation,
    ap_count: int,
) -> ObjectiveValues:
    """
    Calculate all NSGA-II objectives from a grid RF evaluation.

    Objective 1:
        maximize coverage

    Objective 2:
        minimize AP count

    Objective 3:
        minimize interference power
    """

    return ObjectiveValues(
        coverage=coverage_objective(
            evaluation.coverage_percent
        ),
        ap_count=ap_count_objective(
            ap_count
        ),
        interference=interference_objective(
            evaluation.average_interference_dbm
        ),
    )
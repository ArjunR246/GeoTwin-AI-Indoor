import math

import pytest

from geotwin.optimization.objectives import (
    ObjectiveValues,
    ap_count_objective,
    calculate_objectives,
    coverage_objective,
    interference_dbm_to_mw,
    interference_objective,
)


# ============================================================
# COVERAGE OBJECTIVE
# ============================================================


def test_coverage_objective():
    assert coverage_objective(95.0) == -95.0


def test_zero_coverage_objective():
    assert coverage_objective(0.0) == 0.0


def test_full_coverage_objective():
    assert coverage_objective(100.0) == -100.0


def test_invalid_coverage():
    with pytest.raises(ValueError):
        coverage_objective(101.0)


def test_negative_coverage():
    with pytest.raises(ValueError):
        coverage_objective(-1.0)


def test_non_numeric_coverage():
    with pytest.raises(ValueError):
        coverage_objective("95")


def test_nan_coverage():
    with pytest.raises(ValueError):
        coverage_objective(float("nan"))


def test_infinite_coverage():
    with pytest.raises(ValueError):
        coverage_objective(float("inf"))


# ============================================================
# AP COUNT OBJECTIVE
# ============================================================


def test_ap_count_objective():
    assert ap_count_objective(4) == 4.0


def test_invalid_ap_count():
    with pytest.raises(ValueError):
        ap_count_objective(0)


def test_negative_ap_count():
    with pytest.raises(ValueError):
        ap_count_objective(-1)


def test_non_integer_ap_count():
    with pytest.raises(ValueError):
        ap_count_objective(2.0)


def test_boolean_ap_count():
    with pytest.raises(ValueError):
        ap_count_objective(True)


# ============================================================
# dBm -> mW CONVERSION
# ============================================================


def test_dbm_to_mw_zero_dbm():
    result = interference_dbm_to_mw(0.0)

    assert result == pytest.approx(1.0)


def test_dbm_to_mw_negative_10_dbm():
    result = interference_dbm_to_mw(-10.0)

    assert result == pytest.approx(0.1)


def test_dbm_to_mw_negative_20_dbm():
    result = interference_dbm_to_mw(-20.0)

    assert result == pytest.approx(0.01)


def test_dbm_to_mw_negative_30_dbm():
    result = interference_dbm_to_mw(-30.0)

    assert result == pytest.approx(0.001)


def test_dbm_to_mw_negative_40_dbm():
    result = interference_dbm_to_mw(-40.0)

    assert result == pytest.approx(0.0001)


def test_dbm_to_mw_negative_50_dbm():
    result = interference_dbm_to_mw(-50.0)

    assert result == pytest.approx(0.00001)


def test_dbm_to_mw_negative_60_dbm():
    result = interference_dbm_to_mw(-60.0)

    assert result == pytest.approx(0.000001)


def test_dbm_to_mw_no_interference():
    """
    The RF layer represents no measurable interference
    as -inf dBm.

    Optimization representation:

        -inf dBm -> 0.0 mW
    """

    result = interference_dbm_to_mw(
        float("-inf")
    )

    assert result == 0.0
    assert math.isfinite(result)


def test_dbm_to_mw_positive_infinity():
    result = interference_dbm_to_mw(
        float("inf")
    )

    assert result == float("inf")


def test_dbm_to_mw_nan():
    with pytest.raises(ValueError):
        interference_dbm_to_mw(
            float("nan")
        )


def test_dbm_to_mw_non_numeric():
    with pytest.raises(ValueError):
        interference_dbm_to_mw(
            "invalid"
        )


# ============================================================
# INTERFERENCE OBJECTIVE
# ============================================================


def test_interference_objective():
    """
    -60 dBm corresponds to:

        10^(-60/10)
        = 1e-6 mW
    """

    result = interference_objective(
        -60.0
    )

    assert result == pytest.approx(
        1e-6
    )


def test_no_interference_objective():
    """
    No measurable interference should be the
    best possible minimization value.
    """

    result = interference_objective(
        float("-inf")
    )

    assert result == 0.0


def test_stronger_interference_has_larger_objective():
    """
    -50 dBm represents more interference power
    than -60 dBm.

    Therefore the minimization objective must be larger.
    """

    weak = interference_objective(
        -60.0
    )

    strong = interference_objective(
        -50.0
    )

    assert strong > weak


def test_weaker_interference_has_smaller_objective():
    weak = interference_objective(
        -70.0
    )

    strong = interference_objective(
        -50.0
    )

    assert weak < strong


def test_interference_objective_ordering():
    """
    Verify the physically correct optimization ordering:

        no interference
            <
        -70 dBm
            <
        -60 dBm
            <
        -50 dBm
    """

    no_interference = interference_objective(
        float("-inf")
    )

    minus_70 = interference_objective(
        -70.0
    )

    minus_60 = interference_objective(
        -60.0
    )

    minus_50 = interference_objective(
        -50.0
    )

    assert (
        no_interference
        < minus_70
        < minus_60
        < minus_50
    )


# ============================================================
# OBJECTIVE VALUES
# ============================================================


def test_objective_values():
    objectives = ObjectiveValues(
        coverage=-95.0,
        ap_count=3,
        interference=1e-6,
    )

    assert objectives.values == (
        -95.0,
        3.0,
        1e-6,
    )


def test_objective_values_ap_count_conversion():
    objectives = ObjectiveValues(
        coverage=-90.0,
        ap_count=2,
        interference=1e-5,
    )

    assert objectives.values[1] == 2.0


def test_objective_values_are_minimization_values():
    objectives = ObjectiveValues(
        coverage=-95.0,
        ap_count=3,
        interference=1e-6,
    )

    assert objectives.coverage == -95.0
    assert objectives.ap_count == 3
    assert objectives.interference == 1e-6


# ============================================================
# CALCULATE OBJECTIVES
# ============================================================


def test_calculate_objectives_with_finite_interference():
    """
    Verify complete objective calculation with finite
    RF interference.
    """

    evaluation = type(
        "MockEvaluation",
        (),
        {
            "coverage_percent": 95.0,
            "average_interference_dbm": -60.0,
        },
    )()

    result = calculate_objectives(
        evaluation=evaluation,
        ap_count=3,
    )

    assert result.coverage == -95.0
    assert result.ap_count == 3

    assert result.interference == pytest.approx(
        1e-6
    )


def test_calculate_objectives_with_no_interference():
    """
    Critical edge case:

        RF layer:
            -inf dBm

        Optimization layer:
            0.0 mW
    """

    evaluation = type(
        "MockEvaluation",
        (),
        {
            "coverage_percent": 100.0,
            "average_interference_dbm": float(
                "-inf"
            ),
        },
    )()

    result = calculate_objectives(
        evaluation=evaluation,
        ap_count=1,
    )

    assert result.coverage == -100.0
    assert result.ap_count == 1
    assert result.interference == 0.0


def test_calculate_objectives_interference_ordering():
    """
    Complete end-to-end check that stronger physical
    interference produces a worse optimization value.
    """

    weak_evaluation = type(
        "MockEvaluation",
        (),
        {
            "coverage_percent": 90.0,
            "average_interference_dbm": -70.0,
        },
    )()

    strong_evaluation = type(
        "MockEvaluation",
        (),
        {
            "coverage_percent": 90.0,
            "average_interference_dbm": -50.0,
        },
    )()

    weak = calculate_objectives(
        evaluation=weak_evaluation,
        ap_count=2,
    )

    strong = calculate_objectives(
        evaluation=strong_evaluation,
        ap_count=2,
    )

    assert (
        weak.interference
        < strong.interference
    )
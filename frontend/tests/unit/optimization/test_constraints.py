import pytest

from geotwin.optimization.constraints import (
    ConstraintConfig,
    validate_ap_count,
    validate_configuration,
    validate_no_duplicate_positions,
    validate_not_blocked,
    validate_positions,
)


VALID_GRID = [
    (0.0, 0.0),
    (0.5, 0.0),
    (1.0, 0.0),
    (0.0, 0.5),
    (0.5, 0.5),
    (1.0, 0.5),
]


def test_valid_ap_count():
    config = ConstraintConfig(
        min_ap_count=2,
        max_ap_count=5,
    )

    validate_ap_count(
        ap_count=3,
        config=config,
    )


def test_ap_count_below_minimum():
    config = ConstraintConfig(
        min_ap_count=2,
        max_ap_count=5,
    )

    with pytest.raises(ValueError):
        validate_ap_count(
            ap_count=1,
            config=config,
        )


def test_ap_count_above_maximum():
    config = ConstraintConfig(
        min_ap_count=2,
        max_ap_count=5,
    )

    with pytest.raises(ValueError):
        validate_ap_count(
            ap_count=6,
            config=config,
        )


def test_position_is_valid():
    positions = [
        (0.0, 0.0),
        (0.5, 0.5),
    ]

    validate_positions(
        positions=positions,
        valid_positions=VALID_GRID,
    )


def test_position_outside_valid_grid():
    positions = [
        (2.0, 2.0),
    ]

    with pytest.raises(ValueError):
        validate_positions(
            positions=positions,
            valid_positions=VALID_GRID,
        )


def test_duplicate_positions_are_rejected():
    positions = [
        (0.0, 0.0),
        (0.0, 0.0),
    ]

    with pytest.raises(ValueError):
        validate_no_duplicate_positions(
            positions=positions,
        )


def test_unique_positions_are_valid():
    positions = [
        (0.0, 0.0),
        (0.5, 0.5),
    ]

    validate_no_duplicate_positions(
        positions=positions,
    )


def test_blocked_position_is_rejected():
    positions = [
        (0.5, 0.5),
    ]

    blocked_positions = [
        (0.5, 0.5),
    ]

    with pytest.raises(ValueError):
        validate_not_blocked(
            positions=positions,
            blocked_positions=blocked_positions,
        )


def test_unblocked_position_is_valid():
    positions = [
        (0.0, 0.0),
    ]

    blocked_positions = [
        (0.5, 0.5),
    ]

    validate_not_blocked(
        positions=positions,
        blocked_positions=blocked_positions,
    )


def test_valid_configuration():
    positions = [
        (0.0, 0.0),
        (1.0, 0.0),
    ]

    config = ConstraintConfig(
        min_ap_count=1,
        max_ap_count=4,
    )

    result = validate_configuration(
        positions=positions,
        valid_positions=VALID_GRID,
        blocked_positions=[(0.5, 0.5)],
        config=config,
    )

    assert result.valid is True
    assert result.errors == ()
    assert result.violation_count == 0


def test_invalid_configuration():
    positions = [
        (0.0, 0.0),
        (0.0, 0.0),
    ]

    config = ConstraintConfig(
        min_ap_count=1,
        max_ap_count=4,
    )

    result = validate_configuration(
        positions=positions,
        valid_positions=VALID_GRID,
        blocked_positions=[],
        config=config,
    )

    assert result.valid is False
    assert result.violation_count == 1
    assert len(result.errors) == 1


def test_multiple_constraint_violations():
    positions = [
        (2.0, 2.0),
        (2.0, 2.0),
        (0.5, 0.5),
    ]

    config = ConstraintConfig(
        min_ap_count=1,
        max_ap_count=2,
    )

    result = validate_configuration(
        positions=positions,
        valid_positions=VALID_GRID,
        blocked_positions=[(0.5, 0.5)],
        config=config,
    )

    assert result.valid is False

    # Too many APs + invalid position + duplicate position
    # + blocked position.
    assert result.violation_count == 4
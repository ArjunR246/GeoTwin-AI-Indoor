from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple


Position = Tuple[float, float]


@dataclass(frozen=True)
class ConstraintConfig:
    """Configuration for AP placement constraints."""

    min_ap_count: int = 1
    max_ap_count: int = 10


@dataclass(frozen=True)
class ConstraintResult:
    """Result of validating an AP configuration."""

    valid: bool
    errors: Tuple[str, ...]

    @property
    def violation_count(self) -> int:
        """Number of violated constraints."""
        return len(self.errors)


def validate_ap_count(
    ap_count: int,
    config: ConstraintConfig,
) -> None:
    """Validate the number of APs."""

    if ap_count < config.min_ap_count:
        raise ValueError(
            f"AP count {ap_count} is below minimum "
            f"{config.min_ap_count}."
        )

    if ap_count > config.max_ap_count:
        raise ValueError(
            f"AP count {ap_count} exceeds maximum "
            f"{config.max_ap_count}."
        )


def validate_positions(
    positions: Sequence[Position],
    valid_positions: Iterable[Position],
) -> None:
    """
    Ensure every AP position exists in the valid candidate grid.
    """

    valid_set = set(valid_positions)

    for position in positions:
        if position not in valid_set:
            raise ValueError(
                f"AP position {position} is not a valid "
                "candidate position."
            )


def validate_no_duplicate_positions(
    positions: Sequence[Position],
) -> None:
    """Ensure that two APs do not occupy the same position."""

    if len(set(positions)) != len(positions):
        raise ValueError(
            "Multiple APs cannot occupy the same position."
        )


def validate_not_blocked(
    positions: Sequence[Position],
    blocked_positions: Iterable[Position],
) -> None:
    """
    Ensure APs are not placed on walls or obstacles.
    """

    blocked_set = set(blocked_positions)

    for position in positions:
        if position in blocked_set:
            raise ValueError(
                f"AP position {position} is blocked."
            )


def validate_configuration(
    positions: Sequence[Position],
    valid_positions: Iterable[Position],
    blocked_positions: Iterable[Position] = (),
    config: ConstraintConfig | None = None,
) -> ConstraintResult:
    """
    Validate an entire AP configuration.

    Returns a ConstraintResult instead of immediately raising an
    exception, which is useful for evolutionary optimization.
    """

    if config is None:
        config = ConstraintConfig()

    errors = []

    try:
        validate_ap_count(
            ap_count=len(positions),
            config=config,
        )
    except ValueError as exc:
        errors.append(str(exc))

    try:
        validate_positions(
            positions=positions,
            valid_positions=valid_positions,
        )
    except ValueError as exc:
        errors.append(str(exc))

    try:
        validate_no_duplicate_positions(
            positions=positions,
        )
    except ValueError as exc:
        errors.append(str(exc))

    try:
        validate_not_blocked(
            positions=positions,
            blocked_positions=blocked_positions,
        )
    except ValueError as exc:
        errors.append(str(exc))

    return ConstraintResult(
        valid=not errors,
        errors=tuple(errors),
    )
import math

import pytest

from geotwin.rf.path_loss import (
    free_space_path_loss,
    path_loss,
)


def test_free_space_path_loss():
    loss = free_space_path_loss(
        frequency_hz=2.4e9,
        distance_m=1.0,
    )

    assert loss == pytest.approx(40.05, abs=0.1)


def test_path_loss_at_reference_distance():
    reference_loss = free_space_path_loss(
        frequency_hz=2.4e9,
        distance_m=1.0,
    )

    calculated_loss = path_loss(
        frequency_hz=2.4e9,
        distance_m=1.0,
        reference_distance_m=1.0,
        path_loss_exponent=3.0,
    )

    assert calculated_loss == pytest.approx(reference_loss)


def test_path_loss_increases_with_distance():
    loss_2m = path_loss(
        frequency_hz=2.4e9,
        distance_m=2.0,
        reference_distance_m=1.0,
        path_loss_exponent=3.0,
    )

    loss_10m = path_loss(
        frequency_hz=2.4e9,
        distance_m=10.0,
        reference_distance_m=1.0,
        path_loss_exponent=3.0,
    )

    assert loss_10m > loss_2m


def test_path_loss_matches_log_distance_model():
    frequency = 2.4e9
    distance = 10.0
    reference_distance = 1.0
    exponent = 3.0

    reference_loss = free_space_path_loss(
        frequency,
        reference_distance,
    )

    expected = reference_loss + (
        10.0
        * exponent
        * math.log10(distance / reference_distance)
    )

    calculated = path_loss(
        frequency,
        distance,
        reference_distance,
        exponent,
    )

    assert calculated == pytest.approx(expected)


def test_invalid_frequency():
    with pytest.raises(ValueError):
        path_loss(
            frequency_hz=0,
            distance_m=10.0,
        )


def test_invalid_distance():
    with pytest.raises(ValueError):
        path_loss(
            frequency_hz=2.4e9,
            distance_m=0,
        )


def test_invalid_reference_distance():
    with pytest.raises(ValueError):
        path_loss(
            frequency_hz=2.4e9,
            distance_m=10.0,
            reference_distance_m=0,
        )


def test_distance_below_reference():
    with pytest.raises(ValueError):
        path_loss(
            frequency_hz=2.4e9,
            distance_m=0.5,
            reference_distance_m=1.0,
        )


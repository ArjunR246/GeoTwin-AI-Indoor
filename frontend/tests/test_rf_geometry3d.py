import pytest

from geotwin.rf.geometry3d import distance_3d


def test_distance_3d_same_point():
    assert distance_3d(
        (1.0, 2.0, 3.0),
        (1.0, 2.0, 3.0),
    ) == 0.0


def test_distance_3d_horizontal():
    assert distance_3d(
        (0.0, 0.0, 2.0),
        (3.0, 4.0, 2.0),
    ) == pytest.approx(5.0)


def test_distance_3d_vertical():
    assert distance_3d(
        (1.0, 2.0, 1.0),
        (1.0, 2.0, 4.0),
    ) == pytest.approx(3.0)


def test_distance_3d_combined_displacement():
    assert distance_3d(
        (5.0, 4.0, 2.7),
        (8.0, 8.0, 1.0),
    ) == pytest.approx(5.2811, abs=1e-3)


@pytest.mark.parametrize(
    "point",
    [
        (1.0, 2.0),
        (1.0, 2.0, 3.0, 4.0),
        (1.0, float("inf"), 3.0),
        (1.0, float("nan"), 3.0),
        (1.0, "invalid", 3.0),
    ],
)
def test_distance_3d_rejects_invalid_points(point):
    with pytest.raises(ValueError):
        distance_3d(point, (0.0, 0.0, 0.0))
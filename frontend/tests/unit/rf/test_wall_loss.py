import pytest

from geotwin.rf.wall_loss import (
    DEFAULT_MATERIALS,
    WallMaterial,
    get_wall_loss,
)


def test_wall_material_creation():
    material = WallMaterial("concrete", 5.0)

    assert material.name == "concrete"
    assert material.attenuation_db == 5.0


def test_concrete_wall_loss():
    assert get_wall_loss("concrete", 1) == 5.0


def test_multiple_walls():
    assert get_wall_loss("concrete", 3) == 15.0


def test_case_insensitive_material():
    assert get_wall_loss("CONCRETE", 2) == 10.0


def test_zero_walls():
    assert get_wall_loss("concrete", 0) == 0.0


def test_negative_wall_count():
    with pytest.raises(ValueError):
        get_wall_loss("concrete", -1)


def test_unknown_material():
    with pytest.raises(ValueError):
        get_wall_loss("glass", 1)


def test_negative_attenuation():
    with pytest.raises(ValueError):
        WallMaterial("invalid", -1.0)


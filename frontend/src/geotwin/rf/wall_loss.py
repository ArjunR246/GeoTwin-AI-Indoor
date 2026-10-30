from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class WallMaterial:
    """RF attenuation properties of an indoor wall/material."""

    name: str
    attenuation_db: float

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Material name cannot be empty.")

        if self.attenuation_db < 0:
            raise ValueError("Wall attenuation cannot be negative.")


DEFAULT_MATERIALS: Dict[str, WallMaterial] = {
    "concrete": WallMaterial("concrete", 5.0),
    "brick": WallMaterial("brick", 4.0),
    "drywall": WallMaterial("drywall", 2.0),
    "door": WallMaterial("door", 2.0),
}


def get_wall_loss(
    material: str,
    number_of_walls: int = 1,
) -> float:
    """
    Calculate total wall attenuation.

    Parameters
    ----------
    material:
        Material name from DEFAULT_MATERIALS.
    number_of_walls:
        Number of walls crossed by the RF signal.

    Returns
    -------
    float
        Total attenuation in dB.
    """

    if number_of_walls < 0:
        raise ValueError("Number of walls cannot be negative.")

    material_key = material.lower()

    if material_key not in DEFAULT_MATERIALS:
        raise ValueError(f"Unknown wall material: {material}")

    return DEFAULT_MATERIALS[material_key].attenuation_db * number_of_walls


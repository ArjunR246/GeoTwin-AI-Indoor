from dataclasses import dataclass

from geotwin.rf.path_loss import path_loss
from geotwin.rf.wall_loss import get_wall_loss


@dataclass(frozen=True)
class PropagationResult:
    """Result of indoor RF propagation-loss calculation."""

    path_loss_db: float
    wall_loss_db: float
    obstacle_loss_db: float

    @property
    def total_loss_db(self) -> float:
        """Total propagation loss in dB."""
        return (
            self.path_loss_db
            + self.wall_loss_db
            + self.obstacle_loss_db
        )


def calculate_propagation_loss(
    frequency_hz: float,
    distance_m: float,
    reference_distance_m: float = 1.0,
    path_loss_exponent: float = 3.0,
    wall_material: str = "concrete",
    number_of_walls: int = 0,
    obstacle_loss_db: float = 0.0,
) -> PropagationResult:
    """
    Calculate indoor RF propagation losses.

    Total loss is:

        L_total = PL + L_walls + L_obstacles
    """

    if obstacle_loss_db < 0:
        raise ValueError("Obstacle loss cannot be negative.")

    calculated_path_loss = path_loss(
        frequency_hz=frequency_hz,
        distance_m=distance_m,
        reference_distance_m=reference_distance_m,
        path_loss_exponent=path_loss_exponent,
    )

    calculated_wall_loss = get_wall_loss(
        material=wall_material,
        number_of_walls=number_of_walls,
    )

    return PropagationResult(
        path_loss_db=calculated_path_loss,
        wall_loss_db=calculated_wall_loss,
        obstacle_loss_db=obstacle_loss_db,
    )


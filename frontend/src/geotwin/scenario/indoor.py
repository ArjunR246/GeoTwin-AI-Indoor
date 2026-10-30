from dataclasses import dataclass
from typing import List, Tuple


Position = Tuple[float, float]


@dataclass(frozen=True)
class IndoorScenario:
    """
    Synthetic indoor building scenario for GeoTwin AI Indoor.

    Coordinates are represented in metres using an X-Y system.
    """

    width_m: float = 20.0
    length_m: float = 15.0
    grid_resolution_m: float = 0.5

    floors: int = 1

    wall_material: str = "concrete"
    wall_thickness_m: float = 0.2

    def validate(self) -> None:
        """Validate scenario parameters."""

        if self.width_m <= 0:
            raise ValueError(
                "width_m must be greater than zero."
            )

        if self.length_m <= 0:
            raise ValueError(
                "length_m must be greater than zero."
            )

        if self.grid_resolution_m <= 0:
            raise ValueError(
                "grid_resolution_m must be greater than zero."
            )

        if self.floors <= 0:
            raise ValueError(
                "floors must be greater than zero."
            )

        if self.wall_thickness_m < 0:
            raise ValueError(
                "wall_thickness_m cannot be negative."
            )

    @property
    def grid_positions(self) -> List[Position]:
        """
        Generate the complete indoor grid.

        The boundary is included when it falls exactly
        on the configured grid resolution.
        """

        self.validate()

        positions: List[Position] = []

        x_steps = int(
            round(
                self.width_m
                / self.grid_resolution_m
            )
        )

        y_steps = int(
            round(
                self.length_m
                / self.grid_resolution_m
            )
        )

        for x_index in range(
            x_steps + 1
        ):
            x = round(
                x_index
                * self.grid_resolution_m,
                6,
            )

            for y_index in range(
                y_steps + 1
            ):
                y = round(
                    y_index
                    * self.grid_resolution_m,
                    6,
                )

                positions.append(
                    (x, y)
                )

        return positions

    @property
    def valid_ap_positions(self) -> List[Position]:
        """
        Return positions where APs can currently be placed.

        In this first synthetic scenario there are no
        internal obstacles or blocked regions, so every
        grid point is valid.
        """

        return list(
            self.grid_positions
        )

    @property
    def blocked_positions(self) -> List[Position]:
        """
        Return blocked grid positions.

        The initial synthetic scenario contains no
        internal blocked positions.
        """

        return []

    @property
    def receiver_positions(self) -> List[Position]:
        """
        Return receiver/evaluation positions.

        Initially every grid point is evaluated.
        """

        return list(
            self.grid_positions
        )

    @property
    def total_grid_points(self) -> int:
        """Return the number of grid points."""

        return len(
            self.grid_positions
        )


def create_default_scenario() -> IndoorScenario:
    """
    Create the initial 20 m × 15 m GeoTwin scenario.
    """

    return IndoorScenario(
        width_m=20.0,
        length_m=15.0,
        grid_resolution_m=0.5,
        floors=1,
        wall_material="concrete",
        wall_thickness_m=0.2,
    )
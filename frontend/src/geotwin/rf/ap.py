from dataclasses import dataclass
from typing import Tuple
import math


@dataclass
class AccessPoint:
    """Represents a Wi-Fi access point in the indoor environment."""

    position: Tuple[float, float]
    frequency_hz: float
    tx_power_dbm: float
    antenna_gain_dbi: float
    antenna_type: str = "omnidirectional"
    channel: int = 1
    channel_width_mhz: float = 20.0
    height_m: float = 2.5

    def __post_init__(self) -> None:
        if len(self.position) != 2:
            raise ValueError("AP position must contain exactly two coordinates.")

        if self.frequency_hz <= 0:
            raise ValueError("Frequency must be greater than zero.")

        if self.tx_power_dbm < -100:
            raise ValueError("Transmit power is unrealistically low.")

        if self.antenna_gain_dbi < -20:
            raise ValueError("Antenna gain is unrealistically low.")

        if self.channel_width_mhz <= 0:
            raise ValueError("Channel width must be greater than zero.")

        if self.height_m < 0:
            raise ValueError("AP height cannot be negative.")

    @property
    def x(self) -> float:
        return self.position[0]

    @property
    def y(self) -> float:
        return self.position[1]

@dataclass
class AccessPoint3D:
    """Represents a Wi-Fi access point in a 3D indoor environment."""

    position: Tuple[float, float, float]
    frequency_hz: float
    tx_power_dbm: float
    antenna_gain_dbi: float
    antenna_type: str = "omnidirectional"
    channel: int = 1
    channel_width_mhz: float = 20.0

    def __post_init__(self) -> None:
        if len(self.position) != 3:
            raise ValueError(
                "3D AP position must contain exactly three coordinates."
            )

        if not all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
            for value in self.position
        ):
            raise ValueError("AP coordinates must be finite numbers.")

        if self.frequency_hz <= 0:
            raise ValueError("Frequency must be greater than zero.")

        if self.tx_power_dbm < -100:
            raise ValueError("Transmit power is unrealistically low.")

        if self.antenna_gain_dbi < -20:
            raise ValueError("Antenna gain is unrealistically low.")

        if self.channel_width_mhz <= 0:
            raise ValueError("Channel width must be greater than zero.")

        if self.channel <= 0:
            raise ValueError("Channel must be greater than zero.")

    @property
    def x(self) -> float:
        return self.position[0]

    @property
    def y(self) -> float:
        return self.position[1]

    @property
    def z(self) -> float:
        return self.position[2]



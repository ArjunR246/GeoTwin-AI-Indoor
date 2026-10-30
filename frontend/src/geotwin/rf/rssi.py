import math

from geotwin.rf.ap import AccessPoint, AccessPoint3D
from geotwin.rf.propagation import calculate_propagation_loss


def calculate_rssi(
    ap: AccessPoint,
    distance_m: float,
    receiver_gain_dbi: float = 0.0,
    reference_distance_m: float = 1.0,
    path_loss_exponent: float = 3.0,
    wall_material: str = "concrete",
    number_of_walls: int = 0,
    obstacle_loss_db: float = 0.0,
) -> float:
    """Calculate received signal strength (RSSI) in dBm."""

    if receiver_gain_dbi < -20:
        raise ValueError(
            "Receiver antenna gain is unrealistically low."
        )

    propagation = calculate_propagation_loss(
        frequency_hz=ap.frequency_hz,
        distance_m=distance_m,
        reference_distance_m=reference_distance_m,
        path_loss_exponent=path_loss_exponent,
        wall_material=wall_material,
        number_of_walls=number_of_walls,
        obstacle_loss_db=obstacle_loss_db,
    )

    return (
        ap.tx_power_dbm
        + ap.antenna_gain_dbi
        + receiver_gain_dbi
        - propagation.total_loss_db
    )


def calculate_rssi_3d(
    ap: AccessPoint3D,
    receiver_position: tuple[float, float, float],
    receiver_gain_dbi: float = 0.0,
    reference_distance_m: float = 1.0,
    path_loss_exponent: float = 3.0,
    wall_material: str = "concrete",
    number_of_walls: int = 0,
    obstacle_loss_db: float = 0.0,
) -> float:
    """Calculate RSSI using the 3D distance between AP and receiver."""

    if (
        not isinstance(receiver_position, (tuple, list))
        or len(receiver_position) != 3
    ):
        raise ValueError(
            "Receiver position must contain exactly three coordinates."
        )

    if not all(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        for value in receiver_position
    ):
        raise ValueError(
            "Receiver coordinates must be finite numbers."
        )

    if receiver_gain_dbi < -20:
        raise ValueError(
            "Receiver antenna gain is unrealistically low."
        )

    distance_m = math.sqrt(
        (ap.x - receiver_position[0]) ** 2
        + (ap.y - receiver_position[1]) ** 2
        + (ap.z - receiver_position[2]) ** 2
    )

    distance_m = max(distance_m, reference_distance_m)

    propagation = calculate_propagation_loss(
        frequency_hz=ap.frequency_hz,
        distance_m=distance_m,
        reference_distance_m=reference_distance_m,
        path_loss_exponent=path_loss_exponent,
        wall_material=wall_material,
        number_of_walls=number_of_walls,
        obstacle_loss_db=obstacle_loss_db,
    )

    return (
        ap.tx_power_dbm
        + ap.antenna_gain_dbi
        + receiver_gain_dbi
        - propagation.total_loss_db
    )


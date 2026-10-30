import math


SPEED_OF_LIGHT_M_S = 299_792_458.0


def free_space_path_loss(
    frequency_hz: float,
    distance_m: float,
) -> float:
    """
    Calculate free-space path loss in dB.

    Parameters
    ----------
    frequency_hz:
        Signal frequency in Hz.
    distance_m:
        Propagation distance in metres.
    """

    if frequency_hz <= 0:
        raise ValueError("Frequency must be greater than zero.")

    if distance_m <= 0:
        raise ValueError("Distance must be greater than zero.")

    wavelength_m = SPEED_OF_LIGHT_M_S / frequency_hz

    return 20.0 * math.log10(
        (4.0 * math.pi * distance_m) / wavelength_m
    )


def path_loss(
    frequency_hz: float,
    distance_m: float,
    reference_distance_m: float = 1.0,
    path_loss_exponent: float = 3.0,
) -> float:
    """
    Calculate indoor path loss using the log-distance model.

    PL(d) = PL(d0) + 10*n*log10(d/d0)
    """

    if frequency_hz <= 0:
        raise ValueError("Frequency must be greater than zero.")

    if distance_m <= 0:
        raise ValueError("Distance must be greater than zero.")

    if reference_distance_m <= 0:
        raise ValueError("Reference distance must be greater than zero.")

    if path_loss_exponent <= 0:
        raise ValueError("Path-loss exponent must be greater than zero.")

    if distance_m < reference_distance_m:
        raise ValueError(
            "Distance cannot be smaller than the reference distance."
        )

    reference_loss_db = free_space_path_loss(
        frequency_hz,
        reference_distance_m,
    )

    distance_factor = math.log10(
        distance_m / reference_distance_m
    )

    return reference_loss_db + (
        10.0 * path_loss_exponent * distance_factor
    )


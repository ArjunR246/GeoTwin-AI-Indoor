import math
from typing import Iterable


def dbm_to_mw(power_dbm: float) -> float:
    """Convert power from dBm to milliwatts."""
    return 10.0 ** (power_dbm / 10.0)


def mw_to_dbm(power_mw: float) -> float:
    """Convert power from milliwatts to dBm."""

    if power_mw <= 0:
        raise ValueError("Power must be greater than zero.")

    return 10.0 * math.log10(power_mw)


def sum_power_dbm(power_levels_dbm: Iterable[float]) -> float:
    """
    Sum multiple power levels correctly in the linear power domain.

    Parameters
    ----------
    power_levels_dbm:
        Iterable of power levels in dBm.

    Returns
    -------
    float
        Total power in dBm.

    Notes
    -----
    dBm values must not be added directly.
    """

    powers = list(power_levels_dbm)

    if not powers:
        return float("-inf")

    total_mw = sum(
        dbm_to_mw(power_dbm)
        for power_dbm in powers
    )

    return mw_to_dbm(total_mw)

def calculate_sinr_from_interference_dbm(
    signal_power_dbm: float,
    interference_dbm: float,
    noise_floor_dbm: float = -95.0,
) -> float:
    """
    Calculate SINR using already-computed total interference power.
    """

    signal_mw = dbm_to_mw(signal_power_dbm)
    noise_mw = dbm_to_mw(noise_floor_dbm)

    if interference_dbm == float("-inf"):
        interference_mw = 0.0
    else:
        interference_mw = dbm_to_mw(interference_dbm)

    sinr_linear = signal_mw / (
        interference_mw + noise_mw
    )

    return 10.0 * math.log10(sinr_linear)


def calculate_interference_dbm(
    interfering_powers_dbm: Iterable[float],
) -> float:
    """Calculate total interference power in dBm."""

    return sum_power_dbm(interfering_powers_dbm)


def channel_overlap_factor(
    desired_channel: int,
    interfering_channel: int,
    same_channel_factor: float = 1.0,
    adjacent_channel_factor: float = 0.5,
    non_overlapping_factor: float = 0.0,
) -> float:
    """
    Determine how strongly an interfering channel contributes.

    Returns
    -------
    float
        Interference weighting factor.

    Interpretation
    --------------
    1.0 -> full co-channel interference
    0.5 -> partial/adjacent-channel interference
    0.0 -> negligible/non-overlapping interference
    """

    if not 0.0 <= same_channel_factor <= 1.0:
        raise ValueError("same_channel_factor must be between 0 and 1.")

    if not 0.0 <= adjacent_channel_factor <= 1.0:
        raise ValueError(
            "adjacent_channel_factor must be between 0 and 1."
        )

    if not 0.0 <= non_overlapping_factor <= 1.0:
        raise ValueError(
            "non_overlapping_factor must be between 0 and 1."
        )

    if desired_channel <= 0 or interfering_channel <= 0:
        raise ValueError("Channels must be positive integers.")

    channel_difference = abs(
        desired_channel - interfering_channel
    )

    if channel_difference == 0:
        return same_channel_factor

    if channel_difference == 1:
        return adjacent_channel_factor

    return non_overlapping_factor


def calculate_weighted_interference_dbm(
    interfering_powers_dbm: Iterable[float],
    overlap_factors: Iterable[float],
) -> float:
    """
    Calculate interference after applying channel-overlap factors.

    Each interfering signal is converted to linear power,
    multiplied by its overlap factor, and then summed.
    """

    powers = list(interfering_powers_dbm)
    factors = list(overlap_factors)

    if len(powers) != len(factors):
        raise ValueError(
            "Interfering powers and overlap factors "
            "must have the same length."
        )

    if not powers:
        return float("-inf")

    weighted_powers_mw = []

    for power_dbm, factor in zip(powers, factors):
        if not 0.0 <= factor <= 1.0:
            raise ValueError(
                "Overlap factors must be between 0 and 1."
            )

        weighted_powers_mw.append(
            dbm_to_mw(power_dbm) * factor
        )

    total_interference_mw = sum(weighted_powers_mw)

    if total_interference_mw <= 0:
        return float("-inf")

    return mw_to_dbm(total_interference_mw)


def calculate_channel_aware_interference_dbm(
    interfering_powers_dbm: Iterable[float],
    desired_channel: int,
    interfering_channels: Iterable[int],
    same_channel_factor: float = 1.0,
    adjacent_channel_factor: float = 0.5,
    non_overlapping_factor: float = 0.0,
) -> float:
    """
    Calculate total interference while considering channel overlap.
    """

    powers = list(interfering_powers_dbm)
    channels = list(interfering_channels)

    if len(powers) != len(channels):
        raise ValueError(
            "Interfering powers and channels "
            "must have the same length."
        )

    factors = [
        channel_overlap_factor(
            desired_channel=desired_channel,
            interfering_channel=channel,
            same_channel_factor=same_channel_factor,
            adjacent_channel_factor=adjacent_channel_factor,
            non_overlapping_factor=non_overlapping_factor,
        )
        for channel in channels
    ]

    return calculate_weighted_interference_dbm(
        powers,
        factors,
    )


def calculate_sinr_db(
    signal_power_dbm: float,
    interference_powers_dbm: Iterable[float],
    noise_floor_dbm: float = -95.0,
) -> float:
    """
    Calculate SINR in dB.

    SINR = Signal / (Interference + Noise)
    """

    signal_mw = dbm_to_mw(signal_power_dbm)
    noise_mw = dbm_to_mw(noise_floor_dbm)

    interference_powers = list(interference_powers_dbm)

    interference_mw = sum(
        dbm_to_mw(power_dbm)
        for power_dbm in interference_powers
    )

    sinr_linear = signal_mw / (
        interference_mw + noise_mw
    )

    return 10.0 * math.log10(sinr_linear)
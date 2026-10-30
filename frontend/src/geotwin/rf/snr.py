def calculate_snr(
    rssi_dbm: float,
    noise_floor_dbm: float = -95.0,
) -> float:
    """
    Calculate signal-to-noise ratio in dB.

    SNR = RSSI - Noise Floor
    """

    return rssi_dbm - noise_floor_dbm


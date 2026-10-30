from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class BuildingConfig:
    width_m: float = 20.0
    length_m: float = 15.0
    grid_resolution_m: float = 0.5
    floors: int = 1
    coordinate_system: str = "xy_metres"


@dataclass
class APConfig:
    position: Tuple[float, float] = (10.0, 7.5)
    frequency_hz: float = 2.4e9
    tx_power_dbm: float = 20.0
    antenna_gain_dbi: float = 3.0
    antenna_type: str = "omnidirectional"
    channel: int = 1
    channel_width_mhz: float = 20.0
    height_m: float = 2.5


@dataclass
class RFConfig:
    reference_distance_m: float = 1.0
    path_loss_exponent: float = 3.0
    wall_attenuation_db: float = 5.0
    obstacle_attenuation_db: float = 2.0
    noise_floor_dbm: float = -95.0
    receiver_sensitivity_dbm: float = -90.0


@dataclass
class CoverageConfig:
    minimum_rssi_dbm: float = -67.0
    coverage_target_percent: float = 90.0
    dead_zone_target_percent: float = 10.0
    snr_target_db: float = 20.0


@dataclass
class NSGA2Config:
    population_size: int = 100
    generations: int = 100
    crossover_probability: float = 0.9
    mutation_probability: float = 0.1
    min_ap_count: int = 1
    max_ap_count: int = 10


@dataclass
class GeoTwinConfig:
    building: BuildingConfig = field(default_factory=BuildingConfig)
    ap: APConfig = field(default_factory=APConfig)
    rf: RFConfig = field(default_factory=RFConfig)
    coverage: CoverageConfig = field(default_factory=CoverageConfig)
    nsga2: NSGA2Config = field(default_factory=NSGA2Config)
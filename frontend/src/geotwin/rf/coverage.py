from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

from geotwin.floorplan.attenuation import calculate_wall_attenuation
from geotwin.floorplan.walls import WallSegment
from geotwin.rf.ap import AccessPoint
from geotwin.rf.interference import (
    calculate_channel_aware_interference_dbm,
    calculate_sinr_from_interference_dbm,
)
from geotwin.rf.rssi import calculate_rssi
from geotwin.rf.ap import AccessPoint3D
from geotwin.rf.rssi import calculate_rssi_3d
import math
from geotwin.floorplan.geometry3d import Wall3D
from geotwin.floorplan.attenuation3d import calculate_wall_attenuation_3d


@dataclass(frozen=True)
class CoverageResult:
    """Summary of indoor RF coverage."""

    total_points: int
    covered_points: int
    coverage_percent: float
    dead_zone_percent: float


@dataclass(frozen=True)
class ReceiverEvaluation:
    """RF evaluation at one receiver/grid point."""

    position: Tuple[float, float]
    serving_ap_index: int
    serving_rssi_dbm: float
    interference_dbm: float
    sinr_db: float
    covered: bool


@dataclass(frozen=True)
class GridEvaluation:
    """Aggregated RF evaluation across an indoor receiver grid."""

    total_points: int
    covered_points: int
    coverage_percent: float
    dead_zone_percent: float
    average_rssi_dbm: float
    minimum_rssi_dbm: float
    average_sinr_db: float
    minimum_sinr_db: float
    receiver_results: Tuple[ReceiverEvaluation, ...]

    @property
    def average_interference_dbm(self) -> float:
        """Average interference power across receivers."""

        valid_values = [
            result.interference_dbm
            for result in self.receiver_results
            if result.interference_dbm != float("-inf")
        ]

        if not valid_values:
            return float("-inf")

        return sum(valid_values) / len(valid_values)


def calculate_coverage(
    rssi_values: Iterable[float],
    minimum_rssi_dbm: float = -67.0,
    snr_values: Optional[Iterable[float]] = None,
    minimum_snr_db: Optional[float] = None,
) -> CoverageResult:
    """
    Calculate indoor coverage from RSSI measurements.

    A point is considered covered when it satisfies the RSSI
    threshold and, if supplied, the SNR threshold.
    """

    rssi_list = list(rssi_values)

    if not rssi_list:
        raise ValueError("At least one RSSI measurement is required.")

    if snr_values is not None:
        snr_list = list(snr_values)

        if len(snr_list) != len(rssi_list):
            raise ValueError(
                "RSSI and SNR lists must have the same length."
            )
    else:
        snr_list = None

    covered_points = 0

    for index, rssi in enumerate(rssi_list):
        rssi_ok = rssi >= minimum_rssi_dbm

        if snr_list is not None and minimum_snr_db is not None:
            snr_ok = snr_list[index] >= minimum_snr_db
        else:
            snr_ok = True

        if rssi_ok and snr_ok:
            covered_points += 1

    total_points = len(rssi_list)

    coverage_percent = (
        covered_points / total_points
    ) * 100.0

    dead_zone_percent = 100.0 - coverage_percent

    return CoverageResult(
        total_points=total_points,
        covered_points=covered_points,
        coverage_percent=coverage_percent,
        dead_zone_percent=dead_zone_percent,
    )


def evaluate_receiver(
    receiver_position: Tuple[float, float],
    access_points: Sequence[AccessPoint],
    receiver_gain_dbi: float = 0.0,
    reference_distance_m: float = 1.0,
    path_loss_exponent: float = 3.0,
    wall_material: str = "concrete",
    number_of_walls: int = 0,
    obstacle_loss_db: float = 0.0,
    noise_floor_dbm: float = -95.0,
    minimum_rssi_dbm: float = -67.0,
    minimum_sinr_db: Optional[float] = None,
    same_channel_factor: float = 1.0,
    adjacent_channel_factor: float = 0.5,
    non_overlapping_factor: float = 0.0,
    floor_plan_walls: Optional[Sequence[WallSegment]] = None,
    floor_plan_wall_loss_db: float = 0.0,
) -> ReceiverEvaluation:
    """
    Evaluate RF performance at one receiver/grid position.

    The strongest received AP is selected as the serving AP.
    All other APs are treated as potential interferers.

    When floor_plan_walls are supplied, wall attenuation is
    calculated independently for every AP -> receiver path.
    """

    if len(receiver_position) != 2:
        raise ValueError(
            "Receiver position must contain exactly two coordinates."
        )

    if not access_points:
        raise ValueError(
            "At least one access point is required."
        )

    if minimum_sinr_db is not None and not isinstance(
        minimum_sinr_db,
        (int, float),
    ):
        raise ValueError(
            "minimum_sinr_db must be numeric or None."
        )

    if floor_plan_wall_loss_db < 0:
        raise ValueError(
            "floor_plan_wall_loss_db cannot be negative."
        )

    rssi_values = []

    for ap in access_points:
        dx = receiver_position[0] - ap.x
        dy = receiver_position[1] - ap.y

        distance_m = (dx**2 + dy**2) ** 0.5

        # The RF path-loss model requires distance >= reference distance.
        distance_m = max(
            distance_m,
            reference_distance_m,
        )

        rssi = calculate_rssi(
            ap=ap,
            distance_m=distance_m,
            receiver_gain_dbi=receiver_gain_dbi,
            reference_distance_m=reference_distance_m,
            path_loss_exponent=path_loss_exponent,
            wall_material=wall_material,
            number_of_walls=number_of_walls,
            obstacle_loss_db=obstacle_loss_db,
        )

        # Optional automatic floor-plan attenuation.
        #
        # Every AP gets its own wall-crossing calculation.
        if floor_plan_walls:
            wall_attenuation_db = calculate_wall_attenuation(
                transmitter=(ap.x, ap.y),
                receiver=receiver_position,
                walls=floor_plan_walls,
                wall_loss_db=floor_plan_wall_loss_db,
            )

            rssi -= wall_attenuation_db

        rssi_values.append(rssi)

    # Strongest AP becomes serving AP.
    serving_ap_index = max(
        range(len(rssi_values)),
        key=rssi_values.__getitem__,
    )

    serving_rssi_dbm = rssi_values[serving_ap_index]

    interfering_powers = []
    interfering_channels = []

    serving_ap = access_points[serving_ap_index]

    for index, (ap, rssi) in enumerate(
        zip(access_points, rssi_values)
    ):
        if index == serving_ap_index:
            continue

        interfering_powers.append(rssi)
        interfering_channels.append(ap.channel)

    if interfering_powers:
        interference_dbm = (
            calculate_channel_aware_interference_dbm(
                interfering_powers_dbm=interfering_powers,
                desired_channel=serving_ap.channel,
                interfering_channels=interfering_channels,
                same_channel_factor=same_channel_factor,
                adjacent_channel_factor=adjacent_channel_factor,
                non_overlapping_factor=non_overlapping_factor,
            )
        )
    else:
        interference_dbm = float("-inf")

    sinr_db = calculate_sinr_from_interference_dbm(
        signal_power_dbm=serving_rssi_dbm,
        interference_dbm=interference_dbm,
        noise_floor_dbm=noise_floor_dbm,
    )

    rssi_ok = serving_rssi_dbm >= minimum_rssi_dbm

    if minimum_sinr_db is None:
        sinr_ok = True
    else:
        sinr_ok = sinr_db >= minimum_sinr_db

    covered = rssi_ok and sinr_ok

    return ReceiverEvaluation(
        position=receiver_position,
        serving_ap_index=serving_ap_index,
        serving_rssi_dbm=serving_rssi_dbm,
        interference_dbm=interference_dbm,
        sinr_db=sinr_db,
        covered=covered,
    )


def evaluate_grid(
    receiver_positions: Iterable[Tuple[float, float]],
    access_points: Sequence[AccessPoint],
    receiver_gain_dbi: float = 0.0,
    reference_distance_m: float = 1.0,
    path_loss_exponent: float = 3.0,
    wall_material: str = "concrete",
    number_of_walls: int = 0,
    obstacle_loss_db: float = 0.0,
    noise_floor_dbm: float = -95.0,
    minimum_rssi_dbm: float = -67.0,
    minimum_sinr_db: Optional[float] = None,
    same_channel_factor: float = 1.0,
    adjacent_channel_factor: float = 0.5,
    non_overlapping_factor: float = 0.0,
    floor_plan_walls: Optional[Sequence[WallSegment]] = None,
    floor_plan_wall_loss_db: float = 0.0,
) -> GridEvaluation:
    """
    Evaluate RF performance across multiple receiver/grid points.

    Each receiver position is evaluated independently using
    evaluate_receiver().

    Optional floor-plan walls provide AP-specific automatic
    wall attenuation.
    """

    positions = list(receiver_positions)

    if not positions:
        raise ValueError(
            "At least one receiver position is required."
        )

    if not access_points:
        raise ValueError(
            "At least one access point is required."
        )

    if floor_plan_wall_loss_db < 0:
        raise ValueError(
            "floor_plan_wall_loss_db cannot be negative."
        )

    receiver_results = tuple(
        evaluate_receiver(
            receiver_position=position,
            access_points=access_points,
            receiver_gain_dbi=receiver_gain_dbi,
            reference_distance_m=reference_distance_m,
            path_loss_exponent=path_loss_exponent,
            wall_material=wall_material,
            number_of_walls=number_of_walls,
            obstacle_loss_db=obstacle_loss_db,
            noise_floor_dbm=noise_floor_dbm,
            minimum_rssi_dbm=minimum_rssi_dbm,
            minimum_sinr_db=minimum_sinr_db,
            same_channel_factor=same_channel_factor,
            adjacent_channel_factor=adjacent_channel_factor,
            non_overlapping_factor=non_overlapping_factor,
            floor_plan_walls=floor_plan_walls,
            floor_plan_wall_loss_db=floor_plan_wall_loss_db,
        )
        for position in positions
    )

    total_points = len(receiver_results)

    covered_points = sum(
        result.covered
        for result in receiver_results
    )

    coverage_percent = (
        covered_points / total_points
    ) * 100.0

    dead_zone_percent = 100.0 - coverage_percent

    rssi_values = [
        result.serving_rssi_dbm
        for result in receiver_results
    ]

    sinr_values = [
        result.sinr_db
        for result in receiver_results
    ]

    average_rssi_dbm = (
        sum(rssi_values) / len(rssi_values)
    )

    minimum_rssi_dbm = min(rssi_values)

    average_sinr_db = (
        sum(sinr_values) / len(sinr_values)
    )

    minimum_sinr_db = min(sinr_values)

    return GridEvaluation(
        total_points=total_points,
        covered_points=covered_points,
        coverage_percent=coverage_percent,
        dead_zone_percent=dead_zone_percent,
        average_rssi_dbm=average_rssi_dbm,
        minimum_rssi_dbm=minimum_rssi_dbm,
        average_sinr_db=average_sinr_db,
        minimum_sinr_db=minimum_sinr_db,
        receiver_results=receiver_results,
    )

@dataclass(frozen=True)
class ReceiverEvaluation3D:
    """RF evaluation at a single 3D receiver position."""

    position: Tuple[float, float, float]
    serving_ap_index: int
    serving_rssi_dbm: float
    interference_dbm: float
    sinr_db: float
    covered: bool


def evaluate_receiver_3d(
    receiver_position: Tuple[float, float, float],
    access_points: Sequence[AccessPoint3D],
    receiver_gain_dbi: float = 0.0,
    reference_distance_m: float = 1.0,
    path_loss_exponent: float = 3.0,
    wall_material: str = "concrete",
    number_of_walls: int = 0,
    obstacle_loss_db: float = 0.0,
    noise_floor_dbm: float = -95.0,
    minimum_rssi_dbm: float = -67.0,
    minimum_sinr_db: Optional[float] = None,
    same_channel_factor: float = 1.0,
    adjacent_channel_factor: float = 0.5,
    non_overlapping_factor: float = 0.0,
    floor_plan_walls_3d: Optional[Sequence[Wall3D]] = None,
    floor_plan_wall_loss_db: float = 0.0,
) -> ReceiverEvaluation3D:
    """Evaluate RSSI, interference, SINR, and coverage in 3D.

    When floor_plan_walls_3d is supplied, wall attenuation is calculated
    separately for each AP-to-receiver path.

    To prevent double-counting, explicit 3D wall attenuation cannot be
    combined with the uniform number_of_walls attenuation model.
    """

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
        raise ValueError("Receiver coordinates must be finite numbers.")

    if not access_points:
        raise ValueError("At least one access point is required.")

    if minimum_sinr_db is not None and (
        not isinstance(minimum_sinr_db, (int, float))
        or isinstance(minimum_sinr_db, bool)
        or not math.isfinite(minimum_sinr_db)
    ):
        raise ValueError("minimum_sinr_db must be finite and numeric or None.")

    if (
        isinstance(floor_plan_wall_loss_db, bool)
        or not isinstance(floor_plan_wall_loss_db, (int, float))
        or not math.isfinite(floor_plan_wall_loss_db)
        or floor_plan_wall_loss_db < 0
    ):
        raise ValueError(
            "floor_plan_wall_loss_db must be finite and non-negative."
        )

    if floor_plan_walls_3d is not None and number_of_walls != 0:
        raise ValueError(
            "Use either number_of_walls or floor_plan_walls_3d, "
            "not both, to avoid double-counting wall attenuation."
        )

    walls = tuple(floor_plan_walls_3d or ())

    # Calculate the signal from every AP, including its own wall losses.
    rssi_values = []

    for ap in access_points:
        rssi = calculate_rssi_3d(
            ap=ap,
            receiver_position=receiver_position,
            receiver_gain_dbi=receiver_gain_dbi,
            reference_distance_m=reference_distance_m,
            path_loss_exponent=path_loss_exponent,
            wall_material=wall_material,
            number_of_walls=number_of_walls,
            obstacle_loss_db=obstacle_loss_db,
        )

        if walls and floor_plan_wall_loss_db > 0:
            wall_loss = calculate_wall_attenuation_3d(
                transmitter=ap.position,
                receiver=receiver_position,
                walls=walls,
                wall_loss_db=floor_plan_wall_loss_db,
            )
            rssi -= wall_loss

        rssi_values.append(rssi)

    serving_ap_index = max(
        range(len(rssi_values)),
        key=rssi_values.__getitem__,
    )

    serving_rssi_dbm = rssi_values[serving_ap_index]
    serving_ap = access_points[serving_ap_index]

    interfering_powers = [
        rssi
        for index, rssi in enumerate(rssi_values)
        if index != serving_ap_index
    ]

    interfering_channels = [
        ap.channel
        for index, ap in enumerate(access_points)
        if index != serving_ap_index
    ]

    if interfering_powers:
        interference_dbm = calculate_channel_aware_interference_dbm(
            interfering_powers_dbm=interfering_powers,
            desired_channel=serving_ap.channel,
            interfering_channels=interfering_channels,
            same_channel_factor=same_channel_factor,
            adjacent_channel_factor=adjacent_channel_factor,
            non_overlapping_factor=non_overlapping_factor,
        )
    else:
        interference_dbm = float("-inf")

    sinr_db = calculate_sinr_from_interference_dbm(
        signal_power_dbm=serving_rssi_dbm,
        interference_dbm=interference_dbm,
        noise_floor_dbm=noise_floor_dbm,
    )

    rssi_ok = serving_rssi_dbm >= minimum_rssi_dbm
    sinr_ok = (
        minimum_sinr_db is None or sinr_db >= minimum_sinr_db
    )

    return ReceiverEvaluation3D(
        position=tuple(float(value) for value in receiver_position),
        serving_ap_index=serving_ap_index,
        serving_rssi_dbm=serving_rssi_dbm,
        interference_dbm=interference_dbm,
        sinr_db=sinr_db,
        covered=rssi_ok and sinr_ok,
    )


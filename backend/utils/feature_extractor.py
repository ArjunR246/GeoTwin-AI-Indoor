from utils.geometry import euclidean_distance
from utils.wall_intersections import count_intersections

def extract_features(ap, rx, twin):
    distance = euclidean_distance(
        ap[0], ap[1],
        rx[0], rx[1]
    )

    walls = twin["walls"]

    count, attn = count_intersections(
        ap,
        rx,
        walls
    )

    return {
        "distance": distance,
        "num_walls_intersected": count,
        "wall_attenuation_sum": attn
    }
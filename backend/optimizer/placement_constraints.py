import json
from shapely.geometry import Point, Polygon


def load_digital_twin(path="data/digital_twins/floor1.json"):
    """Load the digital twin JSON file."""
    with open(path, "r") as f:
        return json.load(f)


def build_valid_ap_polygons(digital_twin):
    """Convert valid AP zones into Shapely polygons."""
    polygons = []

    for zone in digital_twin.get("valid_ap_zones", []):
        coordinates = [(point["x"], point["y"]) for point in zone]

        if len(coordinates) >= 3:
            polygons.append(Polygon(coordinates))

    return polygons


def is_valid_ap_location(x, y, polygons):
    """Return True if the AP lies inside a valid AP zone."""
    point = Point(x, y)

    return any(
        polygon.contains(point) or polygon.touches(point)
        for polygon in polygons
    )


def get_ap_bounds(digital_twin):
    """
    Get the bounding box of all valid AP zones.

    Returns:
        (min_x, max_x, min_y, max_y)
    """
    polygons = build_valid_ap_polygons(digital_twin)

    if not polygons:
        raise ValueError("No valid AP zones found in digital twin.")

    min_x = min(polygon.bounds[0] for polygon in polygons)
    min_y = min(polygon.bounds[1] for polygon in polygons)
    max_x = max(polygon.bounds[2] for polygon in polygons)
    max_y = max(polygon.bounds[3] for polygon in polygons)

    return min_x, max_x, min_y, max_y
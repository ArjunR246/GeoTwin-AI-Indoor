from pathlib import Path

import cv2
import numpy as np

from geotwin.floorplan.pipeline import process_floor_plan_file

image_path = Path("data/floorplans/bmsce_third_floor_original.jpeg")
output_dir = Path("outputs/geometry_validation")
output_dir.mkdir(parents=True, exist_ok=True)

image = cv2.imread(str(image_path))

if image is None:
    raise FileNotFoundError(f"Could not load floor plan: {image_path}")

result = process_floor_plan_file(image_path)

overlay = image.copy()
room_mask = np.zeros(image.shape[:2], dtype=np.uint8)
corridor_mask = np.zeros(image.shape[:2], dtype=np.uint8)

def polygon_to_cv(polygon):
    points = np.asarray(polygon, dtype=np.float64)

    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(
            f"Expected polygon with shape (N, 2), got {points.shape}"
        )

    return np.rint(points).astype(np.int32).reshape((-1, 1, 2))

for room in result.building.rooms:
    points = polygon_to_cv(room.polygon)
    cv2.fillPoly(room_mask, [points], 255)
    cv2.polylines(overlay, [points], True, (0, 200, 0), 2)

    x, y = np.rint(room.centroid).astype(int)
    cv2.putText(
        overlay,
        room.id,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.35,
        (0, 100, 0),
        1,
        cv2.LINE_AA,
    )

for corridor in result.building.corridors:
    points = polygon_to_cv(corridor.polygon)
    cv2.fillPoly(corridor_mask, [points], 255)
    cv2.polylines(overlay, [points], True, (255, 0, 255), 2)

    x, y = np.rint(corridor.centroid).astype(int)
    cv2.putText(
        overlay,
        corridor.id,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.35,
        (180, 0, 180),
        1,
        cv2.LINE_AA,
    )

room_overlay = image.copy()
room_overlay[room_mask > 0] = (
    0.65 * room_overlay[room_mask > 0]
    + 0.35 * np.array([0, 220, 0])
).astype(np.uint8)

corridor_overlay = image.copy()
corridor_overlay[corridor_mask > 0] = (
    0.65 * corridor_overlay[corridor_mask > 0]
    + 0.35 * np.array([255, 0, 255])
).astype(np.uint8)

cv2.imwrite(str(output_dir / "rooms_overlay.png"), room_overlay)
cv2.imwrite(str(output_dir / "corridors_overlay.png"), corridor_overlay)
cv2.imwrite(str(output_dir / "all_geometry_overlay.png"), overlay)
cv2.imwrite(str(output_dir / "room_mask.png"), room_mask)
cv2.imwrite(str(output_dir / "corridor_mask.png"), corridor_mask)

print(f"Rooms: {result.room_count}")
print(f"Corridors: {result.corridor_count}")
print(f"Walls: {result.wall_count}")
print(f"Output directory: {output_dir.resolve()}")

def describe_region(region):
    points = np.asarray(region.polygon, dtype=np.float64)

    if points.ndim != 2 or points.shape[1] != 2:
        print(f"{region.id}: invalid polygon shape {points.shape}")
        return

    x_min, y_min = points.min(axis=0)
    x_max, y_max = points.max(axis=0)

    print(
        f"{region.id}: "
        f"area={region.area:.1f}, "
        f"centroid={region.centroid}, "
        f"bbox=({x_min:.1f}, {y_min:.1f}, "
        f"{x_max:.1f}, {y_max:.1f}), "
        f"polygon_vertices={len(points)}"
    )


print("\nRoom candidates:")
for room in result.building.rooms:
    describe_region(room)

print("\nCorridor candidates:")
for corridor in result.building.corridors:
    describe_region(corridor)
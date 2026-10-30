from pathlib import Path

import cv2
import numpy as np

from geotwin.floorplan.preprocessing import preprocess_floor_plan
from geotwin.floorplan.rooms import detect_rooms

image_path = Path("data/floorplans/bmsce_third_floor_original.jpeg")
output_dir = Path("outputs/step1_room_diagnostics")
output_dir.mkdir(parents=True, exist_ok=True)

image = cv2.imread(str(image_path))

if image is None:
    raise FileNotFoundError(f"Could not load {image_path}")

processed = preprocess_floor_plan(image)

print(f"Original dimensions: {image.shape[1]} x {image.shape[0]}")

for name, result in processed.items():
    print(
        f"{name}: shape={result.shape}, "
        f"dtype={result.dtype}, "
        f"min={result.min()}, max={result.max()}, "
        f"unique_values={len(np.unique(result))}"
    )
    cv2.imwrite(str(output_dir / f"{name}.png"), result)

for source_name in ("binary", "cleaned"):
    source = processed[source_name]

    for minimum_area in (250, 500, 1000, 2000):
        for maximum_ratio in (0.90, 0.95, 0.99):
            rooms = detect_rooms(
                source,
                min_area=minimum_area,
                max_area_ratio=maximum_ratio,
            )
            print(
                f"source={source_name:7s} "
                f"min_area={minimum_area:4d} "
                f"max_ratio={maximum_ratio:.2f} "
                f"candidates={len(rooms)}"
            )

print(f"\nDiagnostic images saved to: {output_dir.resolve()}")

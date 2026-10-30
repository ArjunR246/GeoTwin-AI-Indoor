from pathlib import Path

import cv2
import numpy as np

from geotwin.floorplan.loader import load_floor_plan
from geotwin.floorplan.preprocessing import preprocess_floor_plan
from geotwin.floorplan.rooms import detect_rooms


IMAGE_PATH = Path(
    "data/floorplans/bmsce_third_floor_original.jpeg"
)

OUTPUT_DIR = Path("outputs/room_diagnostics")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    image = load_floor_plan(IMAGE_PATH)
    stages = preprocess_floor_plan(image)

    # Baseline: current detector.
    rooms = detect_rooms(stages["binary"])

    print(f"Current detector: {len(rooms)} room candidates")

    # Save the existing room detector's input.
    cv2.imwrite(
        str(OUTPUT_DIR / "baseline_binary.png"),
        stages["binary"],
    )

    # Experiment: build a wall mask from the thresholded image.
    gray = stages["grayscale"]

    # Dark architectural and printed lines become white.
    _, dark_lines = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )

    # Close small gaps in potential wall boundaries.
    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (15, 1),
    )
    vertical_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (1, 15),
    )

    horizontal_lines = cv2.morphologyEx(
        dark_lines,
        cv2.MORPH_CLOSE,
        horizontal_kernel,
    )
    vertical_lines = cv2.morphologyEx(
        dark_lines,
        cv2.MORPH_CLOSE,
        vertical_kernel,
    )

    wall_mask = cv2.bitwise_or(
        horizontal_lines,
        vertical_lines,
    )

    # Close remaining small gaps in the combined line mask.
    wall_mask = cv2.morphologyEx(
        wall_mask,
        cv2.MORPH_CLOSE,
        np.ones((3, 3), dtype=np.uint8),
        iterations=2,
    )

    # Candidate free-space image.
    free_space = cv2.bitwise_not(wall_mask)

    # Save all diagnostic images.
    cv2.imwrite(
        str(OUTPUT_DIR / "dark_lines.png"),
        dark_lines,
    )
    cv2.imwrite(
        str(OUTPUT_DIR / "wall_mask.png"),
        wall_mask,
    )
    cv2.imwrite(
        str(OUTPUT_DIR / "free_space.png"),
        free_space,
    )

    # Label connected free-space regions.
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        free_space,
        connectivity=8,
    )

    print(f"Connected components including background: {count}")

    component_visualization = cv2.cvtColor(
        stages["grayscale"],
        cv2.COLOR_GRAY2BGR,
    )

    candidates = 0

    for label in range(1, count):
        x = stats[label, cv2.CC_STAT_LEFT]
        y = stats[label, cv2.CC_STAT_TOP]
        width = stats[label, cv2.CC_STAT_WIDTH]
        height = stats[label, cv2.CC_STAT_HEIGHT]
        area = stats[label, cv2.CC_STAT_AREA]

        # Broad diagnostic filters only—not final room rules.
        if area < 1000:
            continue

        if width < 20 or height < 20:
            continue

        candidates += 1

        cv2.rectangle(
            component_visualization,
            (x, y),
            (x + width, y + height),
            (0, 0, 255),
            2,
        )

    cv2.imwrite(
        str(OUTPUT_DIR / "component_candidates.png"),
        component_visualization,
    )

    print(f"Broad connected-component candidates: {candidates}")
    print(f"Diagnostic images saved to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
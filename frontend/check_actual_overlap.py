from pathlib import Path

import cv2
import numpy as np
import pandas as pd


IMAGE_PATH = Path(
    "outputs/room_diagnostics/cropped_floorplan.png"
)

CSV_PATH = Path(
    "outputs/room_diagnostics_v2/room_candidates.csv"
)

TARGET_IDS = [467, 494, 527]


def main():
    image = cv2.imread(str(IMAGE_PATH))

    if image is None:
        raise FileNotFoundError(IMAGE_PATH)

    candidates = pd.read_csv(CSV_PATH)

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    gray = cv2.GaussianBlur(
        gray,
        (3, 3),
        0,
    )

    _, dark_lines = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (3, 3),
    )

    closed_walls = cv2.morphologyEx(
        dark_lines,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2,
    )

    free_space = cv2.bitwise_not(closed_walls)

    count, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            free_space,
            connectivity=8,
        )
    )

    masks = {}

    for component_id in TARGET_IDS:
        if component_id >= count:
            raise ValueError(
                f"Component {component_id} does not exist."
            )

        masks[component_id] = labels == component_id

    print("\nActual pixel-mask intersections")
    print("-" * 45)

    for i, component_a in enumerate(TARGET_IDS):
        for component_b in TARGET_IDS[i + 1:]:
            intersection = np.count_nonzero(
                masks[component_a] & masks[component_b]
            )

            print(
                f"{component_a} vs {component_b}: "
                f"{intersection} overlapping pixels"
            )

    print("\nBounding-box intersections")
    print("-" * 45)

    for i, component_a in enumerate(TARGET_IDS):
        row_a = candidates[
            candidates["component_id"] == component_a
        ].iloc[0]

        ax1 = int(row_a["x"])
        ay1 = int(row_a["y"])
        ax2 = ax1 + int(row_a["width"])
        ay2 = ay1 + int(row_a["height"])

        for component_b in TARGET_IDS[i + 1:]:
            row_b = candidates[
                candidates["component_id"] == component_b
            ].iloc[0]

            bx1 = int(row_b["x"])
            by1 = int(row_b["y"])
            bx2 = bx1 + int(row_b["width"])
            by2 = by1 + int(row_b["height"])

            width = max(
                0,
                min(ax2, bx2) - max(ax1, bx1),
            )

            height = max(
                0,
                min(ay2, by2) - max(ay1, by1),
            )

            print(
                f"{component_a} vs {component_b}: "
                f"{width * height} overlapping "
                "bounding-box pixels"
            )


if __name__ == "__main__":
    main()
from pathlib import Path
import csv

import cv2
import numpy as np


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

INPUT_PATH = Path(
    "outputs/room_diagnostics/cropped_floorplan.png"
)

OUTPUT_DIR = Path(
    "outputs/room_diagnostics_v2"
)

# Candidate filtering thresholds, in pixels.
MIN_AREA = 500
MAX_AREA_RATIO = 0.12

# Morphological kernel used to close small gaps.
CLOSING_KERNEL_SIZE = 3
CLOSING_ITERATIONS = 2


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    image = cv2.imread(str(INPUT_PATH))

    if image is None:
        raise FileNotFoundError(
            f"Cannot load image: {INPUT_PATH}"
        )

    height, width = image.shape[:2]
    image_area = height * width

    print(f"Input dimensions: {width} x {height}")

    # --------------------------------------------------
    # 1. GRAYSCALE AND BINARIZATION
    # --------------------------------------------------

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    gray = cv2.GaussianBlur(
        gray,
        (3, 3),
        0,
    )

    # Dark architectural lines become white.
    _, dark_lines = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )

    # --------------------------------------------------
    # 2. EMPHASIZE CONNECTED WALL STRUCTURES
    # --------------------------------------------------

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (
            CLOSING_KERNEL_SIZE,
            CLOSING_KERNEL_SIZE,
        ),
    )

    closed_walls = cv2.morphologyEx(
        dark_lines,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=CLOSING_ITERATIONS,
    )

    # --------------------------------------------------
    # 3. FIND ENCLOSED FREE-SPACE COMPONENTS
    # --------------------------------------------------

    # Invert so free space is white and wall structures are black.
    free_space = cv2.bitwise_not(closed_walls)

    # Label connected white regions.
    count, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            free_space,
            connectivity=8,
        )
    )

    # Ignore the background component at index 0.
    max_area = image_area * MAX_AREA_RATIO

    candidates = []

    for label in range(1, count):
        x, y, w, h, area = stats[label]

        # Exclude tiny fragments.
        if area < MIN_AREA:
            continue

        # Exclude unusually large open regions.
        if area > max_area:
            continue

        # Exclude components touching the image border.
        # These are often exterior background regions.
        touches_border = (
            x == 0
            or y == 0
            or x + w >= width
            or y + h >= height
        )

        if touches_border:
            continue

        # Discard tiny or degenerate bounding boxes.
        if w < 10 or h < 10:
            continue

        candidates.append(
            {
                "component_id": label,
                "x": int(x),
                "y": int(y),
                "width": int(w),
                "height": int(h),
                "area": int(area),
                "centroid_x": float(centroids[label][0]),
                "centroid_y": float(centroids[label][1]),
                "aspect_ratio": float(w / h),
            }
        )

    # Sort top-to-bottom, then left-to-right.
    candidates.sort(
        key=lambda item: (
            item["centroid_y"],
            item["centroid_x"],
        )
    )

    # --------------------------------------------------
    # 4. VISUALIZE CANDIDATES
    # --------------------------------------------------

    overlay = image.copy()

    for index, candidate in enumerate(
        candidates,
        start=1,
    ):
        x = candidate["x"]
        y = candidate["y"]
        w = candidate["width"]
        h = candidate["height"]

        cv2.rectangle(
            overlay,
            (x, y),
            (x + w, y + h),
            (0, 0, 255),
            2,
        )

        cv2.putText(
            overlay,
            str(index),
            (x + 3, y + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            1,
            cv2.LINE_AA,
        )

    # Create a separate view of component labels.
    component_view = np.zeros_like(image)

    for candidate in candidates:
        label = candidate["component_id"]

        component_view[labels == label] = (
            255,
            255,
            255,
        )

    # --------------------------------------------------
    # 5. SAVE OUTPUTS
    # --------------------------------------------------

    outputs = {
        "dark_lines.png": dark_lines,
        "closed_walls.png": closed_walls,
        "free_space.png": free_space,
        "room_candidates.png": overlay,
        "candidate_components.png": component_view,
    }

    for filename, output_image in outputs.items():
        output_path = OUTPUT_DIR / filename

        if not cv2.imwrite(
            str(output_path),
            output_image,
        ):
            raise RuntimeError(
                f"Failed to save {output_path}"
            )

    csv_path = OUTPUT_DIR / "room_candidates.csv"

    fieldnames = [
        "component_id",
        "x",
        "y",
        "width",
        "height",
        "area",
        "centroid_x",
        "centroid_y",
        "aspect_ratio",
    ]

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(candidates)

    print()
    print("Room-detection experiment completed.")
    print(f"Connected components: {count - 1}")
    print(f"Candidate regions: {len(candidates)}")
    print(f"Output directory: {OUTPUT_DIR.resolve()}")
    print(f"CSV summary: {csv_path.resolve()}")


if __name__ == "__main__":
    main()
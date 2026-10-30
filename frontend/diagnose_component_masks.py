from pathlib import Path

import cv2
import numpy as np
import pandas as pd


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

INPUT_PATH = Path(
    "outputs/room_diagnostics/cropped_floorplan.png"
)

CSV_PATH = Path(
    "outputs/room_diagnostics_v2/room_candidates.csv"
)

OUTPUT_DIR = Path(
    "outputs/room_diagnostics_v5"
)

TARGET_COMPONENTS = [467, 494, 527]


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    image = cv2.imread(str(INPUT_PATH))

    if image is None:
        raise FileNotFoundError(
            f"Cannot load floor plan: {INPUT_PATH}"
        )

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

    free_space = cv2.bitwise_not(
        closed_walls
    )

    count, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            free_space,
            connectivity=8,
        )
    )

    print(f"Image dimensions: {image.shape[1]} x {image.shape[0]}")
    print(f"Connected components: {count - 1}")

    for component_id in TARGET_COMPONENTS:
        matches = candidates[
            candidates["component_id"] == component_id
        ]

        if matches.empty:
            print(
                f"\nComponent {component_id}: "
                "not found in the CSV."
            )
            continue

        row = matches.iloc[0]

        x = int(row["x"])
        y = int(row["y"])
        w = int(row["width"])
        h = int(row["height"])

        # Confirm that the requested label exists.
        if component_id >= count:
            print(
                f"\nComponent {component_id}: "
                "label is outside the connected-component range."
            )
            continue

        actual_mask = (
            labels[y:y + h, x:x + w] == component_id
        )

        # Save the original image crop.
        original_crop = image[
            y:y + h,
            x:x + w,
        ]

        cv2.imwrite(
            str(
                OUTPUT_DIR
                / f"component_{component_id}_original.png"
            ),
            original_crop,
        )

        # Save the exact component mask.
        mask_crop = (
            actual_mask.astype(np.uint8) * 255
        )

        cv2.imwrite(
            str(
                OUTPUT_DIR
                / f"component_{component_id}_mask.png"
            ),
            mask_crop,
        )

        # Overlay only the actual component pixels.
        overlay = original_crop.copy()

        component_pixels = actual_mask

        green = np.zeros_like(overlay)
        green[:, :] = (0, 255, 0)

        overlay[component_pixels] = cv2.addWeighted(
            overlay[component_pixels],
            0.45,
            green[component_pixels],
            0.55,
            0,
        )

        cv2.imwrite(
            str(
                OUTPUT_DIR
                / f"component_{component_id}_overlay.png"
            ),
            overlay,
        )

        actual_pixel_count = int(
            np.count_nonzero(actual_mask)
        )

        print(
            f"\nComponent {component_id}"
        )
        print(
            f"  Bounding box: x={x}, y={y}, "
            f"width={w}, height={h}"
        )
        print(
            f"  CSV area: {int(row['area'])}"
        )
        print(
            f"  Actual component pixels in box: "
            f"{actual_pixel_count}"
        )
        print(
            f"  Centroid: "
            f"({centroids[component_id][0]:.2f}, "
            f"{centroids[component_id][1]:.2f})"
        )

    print(
        f"\nDiagnostic images saved to: "
        f"{OUTPUT_DIR.resolve()}"
    )


if __name__ == "__main__":
    main()
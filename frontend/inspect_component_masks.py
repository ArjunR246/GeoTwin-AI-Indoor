from pathlib import Path

import cv2
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent

CANDIDATE_CSV = (
    ROOT
    / "outputs"
    / "room_diagnostics_v2"
    / "room_candidates.csv"
)

IMAGE_PATH = (
    ROOT
    / "outputs"
    / "room_diagnostics"
    / "cropped_floorplan.png"
)

OUTPUT_DIR = ROOT / "outputs" / "room_diagnostics_v4"

TARGET_IDS = [467, 494, 527]


def main():
    image = cv2.imread(str(IMAGE_PATH))

    if image is None:
        raise FileNotFoundError(
            f"Could not load image: {IMAGE_PATH}"
        )

    df = pd.read_csv(CANDIDATE_CSV)

    required = {
        "component_id",
        "x",
        "y",
        "width",
        "height",
        "area",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing CSV columns: {sorted(missing)}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    height, width = image.shape[:2]

    overlay = image.copy()

    for component_id in TARGET_IDS:
        matches = df[df["component_id"] == component_id]

        if matches.empty:
            print(
                f"Component {component_id} not found in CSV."
            )
            continue

        row = matches.iloc[0]

        x = int(row["x"])
        y = int(row["y"])
        w = int(row["width"])
        h = int(row["height"])

        x2 = x + w
        y2 = y + h

        if not (
            0 <= x < width
            and 0 <= y < height
            and x2 <= width
            and y2 <= height
        ):
            print(
                f"Component {component_id} has invalid bounds."
            )
            continue

        # Crop each candidate's bounding rectangle.
        crop = image[y:y2, x:x2].copy()

        crop_path = (
            OUTPUT_DIR
            / f"component_{component_id}.png"
        )

        cv2.imwrite(str(crop_path), crop)

        # Draw candidate rectangle on the overview.
        cv2.rectangle(
            overlay,
            (x, y),
            (x2, y2),
            (0, 255, 0),
            2,
        )

        cv2.putText(
            overlay,
            str(component_id),
            (x, max(15, y - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

        print(
            f"Component {component_id}: "
            f"position=({x}, {y}), "
            f"size={w}x{h}, "
            f"component area={row['area']}"
        )

    overlay_path = OUTPUT_DIR / "target_components.png"

    cv2.imwrite(
        str(overlay_path),
        overlay,
    )

    print(f"\nSaved diagnostic images to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
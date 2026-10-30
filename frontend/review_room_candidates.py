from pathlib import Path

import cv2
import pandas as pd


INPUT_PATH = Path(
    "outputs/room_diagnostics/cropped_floorplan.png"
)

CSV_PATH = Path(
    "outputs/room_diagnostics_v3/candidate_classification.csv"
)

OUTPUT_DIR = Path(
    "outputs/room_diagnostics_v6"
)


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

    candidates = pd.read_csv(CSV_PATH)

    candidates = candidates[
        candidates["passes_basic_filter"].astype(bool)
    ].copy()

    candidates = candidates.sort_values(
        ["y", "x"]
    )

    overlay = image.copy()

    for _, row in candidates.iterrows():
        component_id = int(row["component_id"])

        x = int(row["x"])
        y = int(row["y"])
        width = int(row["width"])
        height = int(row["height"])

        cv2.rectangle(
            overlay,
            (x, y),
            (x + width, y + height),
            (0, 255, 0),
            2,
        )

        cv2.putText(
            overlay,
            str(component_id),
            (x + 3, y + 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 0, 0),
            1,
            cv2.LINE_AA,
        )

    output_path = (
        OUTPUT_DIR / "all_room_candidates.png"
    )

    if not cv2.imwrite(
        str(output_path),
        overlay,
    ):
        raise RuntimeError(
            f"Could not save image: {output_path}"
        )

    print(f"Candidates displayed: {len(candidates)}")
    print(f"Saved image: {output_path.resolve()}")


if __name__ == "__main__":
    main()
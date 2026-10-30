from pathlib import Path

import cv2
import pandas as pd


ROOT = Path(__file__).resolve().parent

CROP_PATH = (
    ROOT
    / "outputs"
    / "room_diagnostics"
    / "cropped_floorplan.png"
)

CSV_PATH = (
    ROOT
    / "outputs"
    / "room_diagnostics_v2"
    / "room_candidates.csv"
)

OUTPUT_DIR = ROOT / "outputs" / "room_diagnostics_v3"

MIN_AREA = 1000
MIN_WIDTH = 35
MIN_HEIGHT = 30
MAX_ASPECT_RATIO = 5.0


def main():
    image = cv2.imread(str(CROP_PATH))

    if image is None:
        raise FileNotFoundError(
            f"Could not load cropped floor plan: {CROP_PATH}"
        )

    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"Could not load candidate CSV: {CSV_PATH}"
        )

    candidates = pd.read_csv(CSV_PATH)

    required_columns = {
        "component_id",
        "x",
        "y",
        "width",
        "height",
        "area",
        "aspect_ratio",
    }

    missing = required_columns - set(candidates.columns)

    if missing:
        raise ValueError(
            f"Candidate CSV is missing columns: {sorted(missing)}"
        )

    height, width = image.shape[:2]

    # Validate bounding boxes against the cropped image.
    candidates["inside_image"] = (
        (candidates["x"] >= 0)
        & (candidates["y"] >= 0)
        & (candidates["width"] > 0)
        & (candidates["height"] > 0)
        & (
            candidates["x"] + candidates["width"]
            <= width
        )
        & (
            candidates["y"] + candidates["height"]
            <= height
        )
    )

    # This is only a preliminary geometric filter.
    # It does not establish whether a candidate is a real room.
    candidates["passes_basic_filter"] = (
        candidates["inside_image"]
        & (candidates["area"] >= MIN_AREA)
        & (candidates["width"] >= MIN_WIDTH)
        & (candidates["height"] >= MIN_HEIGHT)
        & (candidates["aspect_ratio"] <= MAX_ASPECT_RATIO)
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    overview = image.copy()
    filtered = image.copy()

    for _, row in candidates.iterrows():
        x = int(row["x"])
        y = int(row["y"])
        w = int(row["width"])
        h = int(row["height"])
        component_id = int(row["component_id"])

        if not (
            0 <= x < width
            and 0 <= y < height
            and w > 0
            and h > 0
            and x + w <= width
            and y + h <= height
        ):
            continue

        color = (
            (0, 180, 0)
            if row["passes_basic_filter"]
            else (0, 0, 255)
        )

        # Overview: all candidates.
        cv2.rectangle(
            overview,
            (x, y),
            (x + w, y + h),
            color,
            2,
        )

        cv2.putText(
            overview,
            str(component_id),
            (x + 2, max(15, y + 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )

        # Filtered view: candidates passing the basic filter.
        if row["passes_basic_filter"]:
            cv2.rectangle(
                filtered,
                (x, y),
                (x + w, y + h),
                (0, 180, 0),
                2,
            )

            cv2.putText(
                filtered,
                str(component_id),
                (x + 2, max(15, y + 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 180, 0),
                1,
                cv2.LINE_AA,
            )

    overview_path = OUTPUT_DIR / "all_candidates.png"
    filtered_path = OUTPUT_DIR / "basic_filter_candidates.png"
    csv_output_path = OUTPUT_DIR / "candidate_classification.csv"

    if not cv2.imwrite(str(overview_path), overview):
        raise RuntimeError("Failed to save all-candidates image.")

    if not cv2.imwrite(str(filtered_path), filtered):
        raise RuntimeError("Failed to save filtered-candidates image.")

    candidates.to_csv(csv_output_path, index=False)

    print(f"Input dimensions: {width} x {height}")
    print(f"Total candidates: {len(candidates)}")
    print(
        "Candidates passing basic filter:",
        int(candidates["passes_basic_filter"].sum()),
    )
    print(
        "Candidates rejected by basic filter:",
        int((~candidates["passes_basic_filter"]).sum()),
    )
    print(f"\nSaved outputs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
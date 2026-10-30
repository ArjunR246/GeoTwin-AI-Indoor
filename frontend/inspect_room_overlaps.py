from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent

CSV_PATH = (
    ROOT
    / "outputs"
    / "room_diagnostics_v3"
    / "candidate_classification.csv"
)

OUTPUT_PATH = (
    ROOT
    / "outputs"
    / "room_diagnostics_v3"
    / "overlapping_candidates.csv"
)


def intersection_area(a, b):
    """Return the intersection area of two bounding boxes."""

    left = max(a["x"], b["x"])
    top = max(a["y"], b["y"])

    right = min(
        a["x"] + a["width"],
        b["x"] + b["width"],
    )

    bottom = min(
        a["y"] + a["height"],
        b["y"] + b["height"],
    )

    width = max(0, right - left)
    height = max(0, bottom - top)

    return width * height


def main():
    if not CSV_PATH.exists():
        raise FileNotFoundError(CSV_PATH)

    df = pd.read_csv(CSV_PATH)

    candidates = df[df["passes_basic_filter"]].to_dict(
        orient="records"
    )

    overlaps = []

    for i, a in enumerate(candidates):
        for b in candidates[i + 1:]:
            area = intersection_area(a, b)

            if area <= 0:
                continue

            area_a = a["width"] * a["height"]
            area_b = b["width"] * b["height"]

            smaller_area = min(area_a, area_b)

            overlap_ratio = (
                area / smaller_area
                if smaller_area > 0
                else 0
            )

            overlaps.append({
                "component_a": int(a["component_id"]),
                "component_b": int(b["component_id"]),
                "intersection_area": int(area),
                "overlap_ratio_of_smaller_box": round(
                    overlap_ratio, 3
                ),
            })

    result = pd.DataFrame(
        overlaps,
        columns=[
            "component_a",
            "component_b",
            "intersection_area",
            "overlap_ratio_of_smaller_box",
        ],
    )

    if not result.empty:
        result = result.sort_values(
            "overlap_ratio_of_smaller_box",
            ascending=False,
        )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(OUTPUT_PATH, index=False)

    print(f"Candidates checked: {len(candidates)}")
    print(f"Overlapping pairs: {len(result)}")
    print(f"Saved: {OUTPUT_PATH}")

    if not result.empty:
        print("\nTop 20 overlapping pairs:")
        print(result.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
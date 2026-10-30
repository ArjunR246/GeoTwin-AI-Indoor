from pathlib import Path

import cv2


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

INPUT_PATH = Path(
    "data/floorplans/bmsce_third_floor_original.jpeg"
)

OUTPUT_DIR = Path("outputs/room_diagnostics")

# Initial crop for the building layout.
# Image dimensions: 1291 x 1218.
#
# These are provisional boundaries; inspect the preview
# before using the crop for room detection.

X_MIN = 80
Y_MIN = 0
X_MAX = 1190
Y_MAX = 975


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    image = cv2.imread(str(INPUT_PATH))

    if image is None:
        raise FileNotFoundError(
            f"Could not load image: {INPUT_PATH}"
        )

    height, width = image.shape[:2]

    print(f"Original image: {width} x {height}")

    if not (
        0 <= X_MIN < X_MAX <= width
        and 0 <= Y_MIN < Y_MAX <= height
    ):
        raise ValueError(
            "Crop boundaries exceed the image dimensions."
        )

    # Extract the provisional floor-plan crop.
    cropped = image[
        Y_MIN:Y_MAX,
        X_MIN:X_MAX,
    ]

    # Create a preview of the crop on the original image.
    preview = image.copy()

    cv2.rectangle(
        preview,
        (X_MIN, Y_MIN),
        (X_MAX - 1, Y_MAX - 1),
        (0, 0, 255),
        3,
    )

    crop_path = OUTPUT_DIR / "cropped_floorplan.png"
    preview_path = OUTPUT_DIR / "crop_preview.png"

    if not cv2.imwrite(str(crop_path), cropped):
        raise RuntimeError(
            f"Could not save cropped image: {crop_path}"
        )

    if not cv2.imwrite(str(preview_path), preview):
        raise RuntimeError(
            f"Could not save crop preview: {preview_path}"
        )

    print(f"Cropped dimensions: {cropped.shape[1]} x {cropped.shape[0]}")
    print(f"Saved crop: {crop_path.resolve()}")
    print(f"Saved preview: {preview_path.resolve()}")


if __name__ == "__main__":
    main()
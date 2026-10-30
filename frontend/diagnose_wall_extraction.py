from pathlib import Path

import cv2
import numpy as np


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

INPUT_PATH = Path(
    "outputs/room_diagnostics/cropped_floorplan.png"
)

OUTPUT_DIR = Path(
    "outputs/wall_extraction_diagnostics"
)

# These are experimental parameters, in pixels.
HORIZONTAL_KERNEL_LENGTH = 35
VERTICAL_KERNEL_LENGTH = 35

# Optional closing to connect small gaps.
CLOSING_KERNEL_SIZE = 3
CLOSING_ITERATIONS = 1


def save_image(path: Path, image: np.ndarray) -> None:
    """Save an image or raise an error."""
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(
            f"Failed to save image: {path}"
        )


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    image = cv2.imread(str(INPUT_PATH))

    if image is None:
        raise FileNotFoundError(
            f"Cannot load floor plan: {INPUT_PATH}"
        )

    print(
        f"Input dimensions: "
        f"{image.shape[1]} x {image.shape[0]}"
    )

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

    # Dark structures become white.
    _, binary = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )

    # --------------------------------------------------
    # 2. EXTRACT HORIZONTAL LINES
    # --------------------------------------------------

    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (
            HORIZONTAL_KERNEL_LENGTH,
            1,
        ),
    )

    horizontal = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        horizontal_kernel,
    )

    # --------------------------------------------------
    # 3. EXTRACT VERTICAL LINES
    # --------------------------------------------------

    vertical_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (
            1,
            VERTICAL_KERNEL_LENGTH,
        ),
    )

    vertical = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        vertical_kernel,
    )

    # --------------------------------------------------
    # 4. COMBINE LINE CANDIDATES
    # --------------------------------------------------

    combined = cv2.bitwise_or(
        horizontal,
        vertical,
    )

    # Close small gaps without aggressively merging
    # neighbouring structures.
    closing_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (
            CLOSING_KERNEL_SIZE,
            CLOSING_KERNEL_SIZE,
        ),
    )

    combined_closed = cv2.morphologyEx(
        combined,
        cv2.MORPH_CLOSE,
        closing_kernel,
        iterations=CLOSING_ITERATIONS,
    )

    # --------------------------------------------------
    # 5. SAVE DIAGNOSTIC IMAGES
    # --------------------------------------------------

    outputs = {
        "binary.png": binary,
        "horizontal_lines.png": horizontal,
        "vertical_lines.png": vertical,
        "combined_lines.png": combined,
        "combined_lines_closed.png": combined_closed,
    }

    for filename, output in outputs.items():
        save_image(
            OUTPUT_DIR / filename,
            output,
        )

    # --------------------------------------------------
    # 6. REPORT
    # --------------------------------------------------

    print("\nWall-extraction experiment completed.")

    print(
        f"Horizontal pixels: "
        f"{cv2.countNonZero(horizontal)}"
    )

    print(
        f"Vertical pixels: "
        f"{cv2.countNonZero(vertical)}"
    )

    print(
        f"Combined pixels: "
        f"{cv2.countNonZero(combined_closed)}"
    )

    print(
        f"Output directory: "
        f"{OUTPUT_DIR.resolve()}"
    )


if __name__ == "__main__":
    main()
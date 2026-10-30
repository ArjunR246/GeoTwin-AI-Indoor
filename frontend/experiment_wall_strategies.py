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
    "outputs/wall_strategy_experiment"
)

KERNEL_LENGTHS = [15, 25, 35]

# Used only to connect very small gaps.
CLOSING_KERNEL_SIZE = 3
CLOSING_ITERATIONS = 1


def save_image(path: Path, image: np.ndarray) -> None:
    """Save an image and report failures."""
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Could not save image: {path}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(str(INPUT_PATH))

    if image is None:
        raise FileNotFoundError(
            f"Could not load floor plan: {INPUT_PATH}"
        )

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    gray = cv2.GaussianBlur(
        gray,
        (3, 3),
        0,
    )

    # Dark lines become white.
    _, binary = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )

    save_image(OUTPUT_DIR / "binary.png", binary)

    closing_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (
            CLOSING_KERNEL_SIZE,
            CLOSING_KERNEL_SIZE,
        ),
    )

    for length in KERNEL_LENGTHS:
        # ------------------------------------------
        # 1. Directional opening
        # ------------------------------------------

        horizontal_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (length, 1),
        )

        vertical_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (1, length),
        )

        horizontal_open = cv2.morphologyEx(
            binary,
            cv2.MORPH_OPEN,
            horizontal_kernel,
        )

        vertical_open = cv2.morphologyEx(
            binary,
            cv2.MORPH_OPEN,
            vertical_kernel,
        )

        opened = cv2.bitwise_or(
            horizontal_open,
            vertical_open,
        )

        # ------------------------------------------
        # 2. Directional closing
        # ------------------------------------------

        horizontal_close = cv2.morphologyEx(
            binary,
            cv2.MORPH_CLOSE,
            horizontal_kernel,
        )

        vertical_close = cv2.morphologyEx(
            binary,
            cv2.MORPH_CLOSE,
            vertical_kernel,
        )

        closed = cv2.bitwise_or(
            horizontal_close,
            vertical_close,
        )

        # ------------------------------------------
        # 3. Close small gaps in opening candidates
        # ------------------------------------------

        opened_closed = cv2.morphologyEx(
            opened,
            cv2.MORPH_CLOSE,
            closing_kernel,
            iterations=CLOSING_ITERATIONS,
        )

        # ------------------------------------------
        # 4. Save masks
        # ------------------------------------------

        outputs = {
            f"horizontal_open_{length}.png":
                horizontal_open,

            f"vertical_open_{length}.png":
                vertical_open,

            f"opened_combined_{length}.png":
                opened,

            f"horizontal_close_{length}.png":
                horizontal_close,

            f"vertical_close_{length}.png":
                vertical_close,

            f"closed_combined_{length}.png":
                closed,

            f"opened_then_closed_{length}.png":
                opened_closed,
        }

        for filename, mask in outputs.items():
            save_image(OUTPUT_DIR / filename, mask)

        # ------------------------------------------
        # 5. Visualize opening candidates
        # ------------------------------------------

        overlay = image.copy()

        # Red indicates candidate pixels.
        overlay[opened > 0] = (0, 0, 255)

        comparison = cv2.addWeighted(
            image,
            0.75,
            overlay,
            0.25,
            0,
        )

        save_image(
            OUTPUT_DIR / f"opening_overlay_{length}.png",
            comparison,
        )

        # ------------------------------------------
        # 6. Report pixel counts
        # ------------------------------------------

        print(f"\nKernel length: {length}")

        print(
            "  Horizontal opening:",
            cv2.countNonZero(horizontal_open),
        )

        print(
            "  Vertical opening:",
            cv2.countNonZero(vertical_open),
        )

        print(
            "  Combined opening:",
            cv2.countNonZero(opened),
        )

        print(
            "  Combined closing:",
            cv2.countNonZero(closed),
        )

        print(
            "  Opening followed by small closing:",
            cv2.countNonZero(opened_closed),
        )

    print("\nExperiment completed.")
    print(f"Results: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
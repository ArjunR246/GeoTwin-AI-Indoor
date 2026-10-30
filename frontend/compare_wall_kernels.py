from pathlib import Path

import cv2


INPUT_PATH = Path(
    "outputs/room_diagnostics/cropped_floorplan.png"
)

OUTPUT_DIR = Path(
    "outputs/wall_kernel_comparison"
)

KERNEL_LENGTHS = [15, 25, 35]


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

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    gray = cv2.GaussianBlur(
        gray,
        (3, 3),
        0,
    )

    _, binary = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )

    for length in KERNEL_LENGTHS:
        horizontal_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (length, 1),
        )

        vertical_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (1, length),
        )

        horizontal = cv2.morphologyEx(
            binary,
            cv2.MORPH_OPEN,
            horizontal_kernel,
        )

        vertical = cv2.morphologyEx(
            binary,
            cv2.MORPH_OPEN,
            vertical_kernel,
        )

        combined = cv2.bitwise_or(
            horizontal,
            vertical,
        )

        cv2.imwrite(
            str(
                OUTPUT_DIR
                / f"horizontal_{length}.png"
            ),
            horizontal,
        )

        cv2.imwrite(
            str(
                OUTPUT_DIR
                / f"vertical_{length}.png"
            ),
            vertical,
        )

        cv2.imwrite(
            str(
                OUTPUT_DIR
                / f"combined_{length}.png"
            ),
            combined,
        )

        print(
            f"Kernel {length}: "
            f"horizontal={cv2.countNonZero(horizontal)}, "
            f"vertical={cv2.countNonZero(vertical)}, "
            f"combined={cv2.countNonZero(combined)}"
        )

    print(
        f"\nSaved comparison images to: "
        f"{OUTPUT_DIR.resolve()}"
    )


if __name__ == "__main__":
    main()
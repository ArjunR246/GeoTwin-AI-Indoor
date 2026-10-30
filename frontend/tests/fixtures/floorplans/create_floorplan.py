from pathlib import Path

import cv2
import numpy as np


OUTPUT = Path(__file__).parent / "simple_floorplan.png"


def create_floorplan() -> None:
    # White background.
    image = np.full((800, 1200, 3), 255, dtype=np.uint8)

    # Wall thickness.
    thickness = 8

    # Outer boundary.
    cv2.rectangle(
        image,
        (50, 50),
        (1150, 750),
        (0, 0, 0),
        thickness,
    )

    # Vertical walls.
    cv2.line(
        image,
        (600, 50),
        (600, 300),
        (0, 0, 0),
        thickness,
    )

    cv2.line(
        image,
        (600, 500),
        (600, 750),
        (0, 0, 0),
        thickness,
    )

    # Horizontal walls around corridor.
    cv2.line(
        image,
        (50, 300),
        (1150, 300),
        (0, 0, 0),
        thickness,
    )

    cv2.line(
        image,
        (50, 500),
        (1150, 500),
        (0, 0, 0),
        thickness,
    )

    # Room labels.
    font = cv2.FONT_HERSHEY_SIMPLEX

    cv2.putText(
        image,
        "ROOM 1",
        (220, 180),
        font,
        1.2,
        (0, 0, 0),
        2,
    )

    cv2.putText(
        image,
        "ROOM 2",
        (770, 180),
        font,
        1.2,
        (0, 0, 0),
        2,
    )

    cv2.putText(
        image,
        "CORRIDOR",
        (470, 420),
        font,
        1.2,
        (0, 0, 0),
        2,
    )

    cv2.putText(
        image,
        "ROOM 3",
        (220, 650),
        font,
        1.2,
        (0, 0, 0),
        2,
    )

    cv2.putText(
        image,
        "ROOM 4",
        (770, 650),
        font,
        1.2,
        (0, 0, 0),
        2,
    )

    if not cv2.imwrite(str(OUTPUT), image):
        raise RuntimeError("Failed to create floor-plan fixture.")

    print(f"Created: {OUTPUT}")


if __name__ == "__main__":
    create_floorplan()
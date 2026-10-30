from pathlib import Path

import cv2
import numpy as np

from geotwin.floorplan.pipeline import process_floor_plan


IMAGE_PATH = Path(
    "data/floorplans/bmsce_third_floor_original.jpeg"
)

OUTPUT_DIR = Path("outputs/floorplan_inspection")

# ROI format: (x, y, width, height)
# Initial building-focused crop; coordinates refer to the original image.
ROI = (100, 0, 1075, 970)


def main():
    if not IMAGE_PATH.exists():
        raise FileNotFoundError(
            f"Floorplan image not found: {IMAGE_PATH}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(str(IMAGE_PATH))

    if image is None:
        raise ValueError(
            f"OpenCV could not read: {IMAGE_PATH}"
        )

    image_height, image_width = image.shape[:2]

    print(
        f"Original image dimensions: "
        f"{image_width} x {image_height}"
    )

    x, y, roi_width, roi_height = ROI

    if (
        x < 0
        or y < 0
        or roi_width <= 0
        or roi_height <= 0
        or x + roi_width > image_width
        or y + roi_height > image_height
    ):
        raise ValueError(
            f"ROI {ROI} is outside the image dimensions "
            f"{image_width} x {image_height}"
        )

    # Save the original-coordinate ROI preview.
    roi_image = image[
        y:y + roi_height,
        x:x + roi_width,
    ].copy()

    roi_path = OUTPUT_DIR / "building_roi_preview.png"

    if not cv2.imwrite(str(roi_path), roi_image):
        raise RuntimeError(
            f"Could not save ROI preview: {roi_path}"
        )

    print(f"ROI: {ROI}")
    print(
        f"Cropped image dimensions: "
        f"{roi_image.shape[1]} x {roi_image.shape[0]}"
    )

    # Process the original image using the ROI.
    # The pipeline returns geometry mapped to original image coordinates.
    result = process_floor_plan(
        image,
        roi=ROI,
    )

    print("\nFloorplan extraction summary")
    print("-" * 40)
    print(f"Detected walls:     {result.wall_count}")
    print(f"Detected rooms:     {result.room_count}")
    print(f"Detected corridors: {result.corridor_count}")
    print(f"Physical scale:     {result.has_physical_scale}")

    print(f"Applied ROI:        {result.roi}")
    print(f"Coordinate offset:  {result.coordinate_offset}")

    # Save preprocessing stages generated from the cropped image.
    for name, output in result.preprocessing.items():
        output_path = OUTPUT_DIR / f"{name}.png"

        if not cv2.imwrite(str(output_path), output):
            raise RuntimeError(
                f"Could not save preprocessing image: {output_path}"
            )

        print(f"Saved: {output_path}")

    # Visualize wall segments on the original image.
    wall_visualization = image.copy()

    for wall in result.building.walls:
        cv2.line(
            wall_visualization,
            (int(wall.x1), int(wall.y1)),
            (int(wall.x2), int(wall.y2)),
            (0, 0, 255),
            2,
        )

    wall_path = OUTPUT_DIR / "detected_walls.png"

    if not cv2.imwrite(str(wall_path), wall_visualization):
        raise RuntimeError(
            f"Could not save wall visualization: {wall_path}"
        )

    print(f"Saved: {wall_path}")

    # Visualize detected room and corridor polygons.
    geometry_visualization = image.copy()

    for room in result.building.rooms:
        polygon = np.asarray(
            room.polygon,
            dtype=np.int32,
        ).reshape((-1, 1, 2))

        cv2.polylines(
            geometry_visualization,
            [polygon],
            isClosed=True,
            color=(0, 255, 0),
            thickness=2,
        )

        cx, cy = room.centroid

        cv2.putText(
            geometry_visualization,
            str(room.id),
            (int(cx), int(cy)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (0, 128, 0),
            1,
            cv2.LINE_AA,
        )

    for corridor in result.building.corridors:
        polygon = np.asarray(
            corridor.polygon,
            dtype=np.int32,
        ).reshape((-1, 1, 2))

        cv2.polylines(
            geometry_visualization,
            [polygon],
            isClosed=True,
            color=(255, 0, 0),
            thickness=2,
        )

    geometry_path = OUTPUT_DIR / "detected_rooms_corridors.png"

    if not cv2.imwrite(
        str(geometry_path),
        geometry_visualization,
    ):
        raise RuntimeError(
            f"Could not save geometry visualization: {geometry_path}"
        )

    print(f"Saved: {geometry_path}")

    print("\nDiagnostic output directory:")
    print(OUTPUT_DIR.resolve())


if __name__ == "__main__":
    main()

"""
GeoTwin AI Indoor - Floor-plan loader.

Supports common raster floor-plan formats and PDF floor plans.

Supported formats:
    PNG
    JPG / JPEG
    BMP
    TIFF
    PDF

All loaded floor plans are returned as BGR NumPy arrays so that
they are directly compatible with the OpenCV preprocessing pipeline.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


SUPPORTED_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
    ".pdf",
}


IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
}


class FloorPlanLoadError(ValueError):
    """Raised when a floor-plan file cannot be loaded or validated."""


def load_floor_plan(path: str | Path) -> np.ndarray:
    """
    Load a floor-plan from disk.

    Raster image files are decoded using OpenCV.

    PDF floor plans are rendered from the first PDF page using
    PyMuPDF and converted into a BGR NumPy array.

    Args:
        path:
            Path to the floor-plan file.

    Returns:
        Floor-plan image as a BGR uint8 NumPy array.

    Raises:
        FloorPlanLoadError:
            If the file does not exist, is not a file,
            has an unsupported extension, cannot be decoded,
            or a PDF cannot be rendered.
    """

    image_path = Path(path)

    # ---------------------------------------------------------
    # 1. PATH VALIDATION
    # ---------------------------------------------------------

    if not image_path.exists():
        raise FloorPlanLoadError(
            f"Floor-plan file does not exist: {image_path}"
        )

    if not image_path.is_file():
        raise FloorPlanLoadError(
            f"Floor-plan path is not a file: {image_path}"
        )

    extension = image_path.suffix.lower()

    # ---------------------------------------------------------
    # 2. EXTENSION VALIDATION
    # ---------------------------------------------------------

    if extension not in SUPPORTED_EXTENSIONS:
        raise FloorPlanLoadError(
            f"Unsupported floor-plan format: "
            f"{image_path.suffix}. "
            f"Supported formats: "
            f"{sorted(SUPPORTED_EXTENSIONS)}"
        )

    # ---------------------------------------------------------
    # 3. PDF LOADING
    # ---------------------------------------------------------

    if extension == ".pdf":
        return _load_pdf(image_path)

    # ---------------------------------------------------------
    # 4. RASTER IMAGE LOADING
    # ---------------------------------------------------------

    if extension in IMAGE_EXTENSIONS:
        return _load_image(image_path)

    # Defensive fallback.
    raise FloorPlanLoadError(
        f"Unsupported floor-plan format: {extension}"
    )


def _load_image(
    image_path: Path,
) -> np.ndarray:
    """
    Load a raster floor-plan image using OpenCV.

    Returns:
        BGR uint8 NumPy array.
    """

    image = cv2.imread(
        str(image_path),
        cv2.IMREAD_COLOR,
    )

    if image is None:
        raise FloorPlanLoadError(
            f"Unable to decode floor-plan image: {image_path}"
        )

    if image.size == 0:
        raise FloorPlanLoadError(
            f"Floor-plan image is empty: {image_path}"
        )

    if image.ndim != 3 or image.shape[2] != 3:
        raise FloorPlanLoadError(
            f"Decoded floor-plan image has an invalid "
            f"shape: {image.shape}"
        )

    return image


def _load_pdf(
    pdf_path: Path,
) -> np.ndarray:
    """
    Render the first page of a PDF floor plan.

    PyMuPDF is used because it can render PDF pages directly
    without requiring an external PDF renderer.

    The rendered PDF page is converted from RGB/RGBA into
    the BGR format expected by OpenCV.
    """

    try:
        import pymupdf
    except ImportError as exc:
        raise FloorPlanLoadError(
            "PDF support requires PyMuPDF. "
            "Install it with: python -m pip install PyMuPDF"
        ) from exc

    try:
        document = pymupdf.open(str(pdf_path))
    except Exception as exc:
        raise FloorPlanLoadError(
            f"Unable to open PDF floor plan: {pdf_path}"
        ) from exc

    try:
        if document.page_count == 0:
            raise FloorPlanLoadError(
                f"PDF floor plan contains no pages: {pdf_path}"
            )

        page = document.load_page(0)

        # Render at approximately 150 DPI.
        # PDF coordinates are based on 72 DPI.
        zoom = 150.0 / 72.0

        matrix = pymupdf.Matrix(
              zoom,
              zoom,
            )
        

        pixmap = page.get_pixmap(
            matrix=matrix,
            alpha=False,
        )

        if pixmap.width <= 0 or pixmap.height <= 0:
            raise FloorPlanLoadError(
                f"Rendered PDF page is empty: {pdf_path}"
            )

        image = np.frombuffer(
            pixmap.samples,
            dtype=np.uint8,
        )

        image = image.reshape(
            pixmap.height,
            pixmap.width,
            pixmap.n,
        )

        if pixmap.n == 3:
            image = cv2.cvtColor(
                image,
                cv2.COLOR_RGB2BGR,
            )

        elif pixmap.n == 4:
            image = cv2.cvtColor(
                image,
                cv2.COLOR_RGBA2BGR,
            )

        else:
            raise FloorPlanLoadError(
                f"Unsupported rendered PDF channel count: "
                f"{pixmap.n}"
            )

        if image.size == 0:
            raise FloorPlanLoadError(
                f"Rendered PDF floor plan is empty: {pdf_path}"
            )

        return np.ascontiguousarray(
            image,
            dtype=np.uint8,
        )

    except FloorPlanLoadError:
        raise

    except Exception as exc:
        raise FloorPlanLoadError(
            f"Unable to render PDF floor plan: {pdf_path}"
        ) from exc

    finally:
        document.close()


__all__ = [
    "SUPPORTED_EXTENSIONS",
    "IMAGE_EXTENSIONS",
    "FloorPlanLoadError",
    "load_floor_plan",
]
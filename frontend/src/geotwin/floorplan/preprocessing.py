import cv2
import numpy as np


def _validate_image(image: np.ndarray) -> None:
    """Validate that the input is a non-empty NumPy image."""
    if not isinstance(image, np.ndarray):
        raise TypeError("Input image must be a NumPy array.")

    if image.size == 0:
        raise ValueError("Input image is empty.")


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Convert an OpenCV image to grayscale.

    Supports both:
        - BGR images
        - already-grayscale images
    """
    _validate_image(image)

    if image.ndim == 2:
        return image.copy()

    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    raise ValueError(
        "Input image must be either grayscale or a 3-channel BGR image."
    )


def preprocess_floor_plan(
    image: np.ndarray,
) -> dict[str, np.ndarray]:
    """
    Perform initial OpenCV preprocessing of a floor-plan image.

    Pipeline:
        BGR image
            ↓
        grayscale
            ↓
        Gaussian blur
            ↓
        adaptive threshold
            ↓
        morphological closing
            ↓
        Canny edges

    Returns:
        Dictionary containing:
            grayscale
            blurred
            binary
            cleaned
            edges
    """
    grayscale = to_grayscale(image)

    # Reduce small image/scanning noise.
    blurred = cv2.GaussianBlur(
        grayscale,
        (5, 5),
        0,
    )

    # Convert the floor plan into a binary representation.
    binary = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11,
    )

    # Close small gaps in walls and architectural lines.
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (3, 3),
    )

    cleaned = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2,
    )

    # Detect edges for the upcoming wall-detection stage.
    edges = cv2.Canny(
        cleaned,
        50,
        150,
    )

    return {
        "grayscale": grayscale,
        "blurred": blurred,
        "binary": binary,
        "cleaned": cleaned,
        "edges": edges,
    }
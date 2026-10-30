from pathlib import Path

import cv2
import pytest

from scripts.generate_heatmap import main as generate_heatmaps
from scripts.generate_optimization_plots import main as generate_optimization_plots


OUTPUT_DIR = Path("outputs")


HEATMAP_OUTPUTS = (
    "coverage_map.png",
    "rssi_heatmap.png",
    "sinr_heatmap.png",
    "interference_heatmap.png",
)

OPTIMIZATION_OUTPUTS = (
    "pareto_coverage_vs_ap_count.png",
    "pareto_sinr_vs_ap_count.png",
    "pareto_coverage_vs_sinr.png",
    "pareto_front.png",
)


@pytest.mark.parametrize(
    "filename",
    HEATMAP_OUTPUTS,
)
def test_heatmap_output_exists(filename):
    """
    Verify that every required RF visualization is generated.
    """

    generate_heatmaps()

    output_file = OUTPUT_DIR / filename

    assert output_file.exists()
    assert output_file.is_file()
    assert output_file.stat().st_size > 0


@pytest.mark.parametrize(
    "filename",
    HEATMAP_OUTPUTS,
)
def test_heatmap_output_is_valid_png(filename):
    """
    Verify that generated heatmaps can be decoded by OpenCV.
    """

    generate_heatmaps()

    output_file = OUTPUT_DIR / filename

    image = cv2.imread(
        str(output_file),
        cv2.IMREAD_COLOR,
    )

    assert image is not None
    assert image.size > 0

    height, width = image.shape[:2]

    assert width > 0
    assert height > 0


@pytest.mark.parametrize(
    "filename",
    OPTIMIZATION_OUTPUTS,
)
def test_optimization_plot_exists(filename):
    """
    Verify that every required NSGA-II visualization is generated.
    """

    generate_optimization_plots()

    output_file = OUTPUT_DIR / filename

    assert output_file.exists()
    assert output_file.is_file()
    assert output_file.stat().st_size > 0


@pytest.mark.parametrize(
    "filename",
    OPTIMIZATION_OUTPUTS,
)
def test_optimization_plot_is_valid_png(filename):
    """
    Verify that generated optimization plots can be decoded.
    """

    generate_optimization_plots()

    output_file = OUTPUT_DIR / filename

    image = cv2.imread(
        str(output_file),
        cv2.IMREAD_COLOR,
    )

    assert image is not None
    assert image.size > 0

    height, width = image.shape[:2]

    assert width > 0
    assert height > 0


def test_all_required_visualization_outputs_exist():
    """
    Verify the complete visualization artifact set.
    """

    generate_heatmaps()
    generate_optimization_plots()

    expected_outputs = (
        HEATMAP_OUTPUTS
        + OPTIMIZATION_OUTPUTS
    )

    for filename in expected_outputs:

        output_file = OUTPUT_DIR / filename

        assert output_file.exists(), (
            f"Missing visualization: {output_file}"
        )

        assert output_file.stat().st_size > 0, (
            f"Visualization is empty: {output_file}"
        )
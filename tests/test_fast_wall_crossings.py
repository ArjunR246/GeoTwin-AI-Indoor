"""Check that the fast wall-crossing counter matches the original Shapely one."""

import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for import_path in (PROJECT_ROOT, PROJECT_ROOT / "backend", PROJECT_ROOT / "frontend/src"):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from services.floorplan_coverage import count_wall_crossings_fast
from utils.wall_intersections import count_intersections


def wall(x1, y1, x2, y2, loss=6.0):
    return {"start": {"x": x1, "y": y1}, "end": {"x": x2, "y": y2}, "attenuation_db": loss}


def test_simple_cases():
    walls = [wall(5, 0, 5, 10, loss=6.0), wall(0, 5, 10, 5, loss=4.0)]
    ap = (2.0, 2.0)
    receivers = [
        (3.0, 3.0),   # same room: no walls crossed
        (8.0, 2.0),   # through the vertical wall only
        (2.0, 8.0),   # through the horizontal wall only
        (8.0, 8.0),   # through both walls
    ]
    counts, losses = count_wall_crossings_fast(ap, receivers, walls)
    assert list(counts) == [0, 1, 1, 2]
    assert list(losses) == [0.0, 6.0, 4.0, 10.0]


def test_no_walls():
    counts, losses = count_wall_crossings_fast((0.0, 0.0), [(1.0, 1.0), (2.0, 2.0)], [])
    assert list(counts) == [0, 0]
    assert list(losses) == [0.0, 0.0]


def test_matches_original_on_random_layout():
    rng = random.Random(42)
    walls = []
    for _ in range(40):
        if rng.random() < 0.5:
            y = rng.uniform(0, 30)
            x1, x2 = sorted((rng.uniform(0, 40), rng.uniform(0, 40)))
            walls.append(wall(x1, y, x2 + 0.5, y))
        else:
            x = rng.uniform(0, 40)
            y1, y2 = sorted((rng.uniform(0, 30), rng.uniform(0, 30)))
            walls.append(wall(x, y1, x, y2 + 0.5))
    receivers = [(rng.uniform(0, 40), rng.uniform(0, 30)) for _ in range(300)]

    for _ in range(5):
        ap = (rng.uniform(0, 40), rng.uniform(0, 30))
        counts, losses = count_wall_crossings_fast(ap, receivers, walls)
        for receiver, count, loss in zip(receivers, counts, losses):
            expected_count, expected_loss = count_intersections(ap, receiver, walls)
            assert count == expected_count
            assert abs(loss - expected_loss) < 1e-9
"""Tests for the figure code. Run: make test"""

import sys
import tempfile
import unittest
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from maps import make_map  # noqa: E402
from search import ALGORITHMS  # noqa: E402
from visualize import Canvas, explored_cells, make_figures  # noqa: E402


class TestRecording(unittest.TestCase):
    def test_recorded_cells_match_expanded_count(self):
        # explored_cells raises if the recording and the counter disagree
        for kind in ("random", "maze", "nopath", "uwall", "terrain", "city"):
            for seed in range(3):
                case = make_map(kind, 30, seed)
                for name in ALGORITHMS:
                    with self.subTest(kind=kind, seed=seed, algorithm=name):
                        result, order = explored_cells(name, case.grid, case.start, case.goal)
                        self.assertEqual(len(order), result.expanded)
                        self.assertEqual(len(set(order)), len(order))   # each cell once
                        self.assertEqual(order[0], case.grid.index(case.start))

    def test_recording_does_not_change_result(self):
        case = make_map("random", 40, 1, weighted=True)
        for name, fn in ALGORITHMS.items():
            plain = fn(case.grid, case.start, case.goal)
            recorded, _ = explored_cells(name, case.grid, case.start, case.goal)
            self.assertEqual((plain.cost, plain.expanded, plain.path),
                             (recorded.cost, recorded.expanded, recorded.path))


class TestPng(unittest.TestCase):
    def test_png_is_valid(self):
        c = Canvas(7, 5)
        c.rect(1, 1, 3, 2, (255, 0, 0))
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "x.png"
            c.save_png(p)
            data = p.read_bytes()
        self.assertEqual(data[:8], b"\x89PNG\r\n\x1a\n")
        self.assertEqual(int.from_bytes(data[16:20], "big"), 7)    # width
        self.assertEqual(int.from_bytes(data[20:24], "big"), 5)    # height
        idat = data.index(b"IDAT")
        length = int.from_bytes(data[idat - 4:idat], "big")
        raw = zlib.decompress(data[idat + 4:idat + 4 + length])
        self.assertEqual(len(raw), 5 * (1 + 7 * 3))
        # pixel (x=1, y=1) is red
        self.assertEqual(raw[1 * 22 + 1 + 1 * 3:1 * 22 + 1 + 2 * 3], bytes((255, 0, 0)))

    def test_make_figures_writes_gallery(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "figures"
            make_figures(out, size=20, log=lambda *a: None)
            pngs = sorted((out / "maps").glob("*.png"))
            self.assertEqual(len(pngs), 8)
            text = (out / "README.md").read_text()
            for p in pngs:
                self.assertIn(f"maps/{p.name}", text)


if __name__ == "__main__":
    unittest.main()

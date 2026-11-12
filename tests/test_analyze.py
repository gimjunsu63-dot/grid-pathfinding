"""Tests for the chart helpers and the analysis script. Run: make test"""

import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import analyze  # noqa: E402
import bench  # noqa: E402
from svgchart import LinePlot, bar_chart, log_ticks, nice_ticks, tint  # noqa: E402


class TestChartHelpers(unittest.TestCase):
    def test_nice_ticks_cover_range(self):
        for lo, hi in ((0, 7), (0, 0.33), (3, 97), (0, 31903)):
            t = nice_ticks(lo, hi)
            self.assertLessEqual(t[0], lo)
            self.assertGreaterEqual(t[-1], hi * 0.999)
            steps = {round(b - a, 9) for a, b in zip(t, t[1:])}
            self.assertEqual(len(steps), 1)

    def test_log_ticks(self):
        self.assertEqual(log_ticks(1, 1000), [1, 10, 100, 1000])
        self.assertIn(20, log_ticks(10, 100))

    def test_tint(self):
        self.assertEqual(tint("#000000", 1), "#ffffff")
        self.assertEqual(tint("#2a78d6", 0), "#2a78d6")

    def test_svg_is_well_formed(self):
        p = LinePlot("t", "x", "y", xlog=True, ylog=True)
        p.line([1, 10, 100], [2, 20, 200], "#2a78d6", "a", lo=[1, 10, 100], hi=[3, 30, 300])
        p.line([1, 10, 100], [1, 5, 50], "#eb6834", "b <&>")
        ET.fromstring(p.render())
        ET.fromstring(bar_chart("t", ["a", "b\nc"], [dict(label="s", color="#000", values=[1, 2])], "y"))

    def test_slope(self):
        self.assertAlmostEqual(analyze.slope([1, 10, 100], [3, 30, 300]), 1.0)
        self.assertAlmostEqual(analyze.slope([1, 10, 100], [1, 100, 10000]), 2.0)


class TestAnalyze(unittest.TestCase):
    def test_full_pipeline_on_quick_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            res = Path(tmp) / "results"
            out = Path(tmp) / "figures"
            bench.run([*bench.EXPERIMENTS, "noise"], res, n_maps=2, repeats=2, quick=True,
                      log=lambda *a: None)
            analyze.analyze(res, out, log=lambda *a: None)
            charts = sorted(p.name for p in (out / "charts").glob("*.svg"))
            for name in ("A_time.svg", "B_share.svg", "C_time.svg", "D_bfs_cost.svg", "E_share.svg",
                         "decomposition.svg", "per_cell.svg", "heuristic_accuracy.svg", "noise.svg",
                         "T_expanded.svg", "W_tradeoff.svg", "H_time.svg", "BI_expanded.svg"):
                self.assertIn(name, charts)
            for p in (out / "charts").glob("*.svg"):
                ET.fromstring(p.read_text())
            text = (out / "RESULTS.md").read_text()
            for name in charts:
                self.assertIn(f"charts/{name}", text)

    def test_code_lines_skips_docstring_and_comments(self):
        def f(x):
            """doc
            string"""
            # comment

            return x + 1
        self.assertEqual(analyze.code_lines(f), 2)


if __name__ == "__main__":
    unittest.main()

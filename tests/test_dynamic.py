"""Tests for the dynamic-traffic experiment and landmark placement. Run: make test"""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import bench  # noqa: E402
from grid import WALL  # noqa: E402
from landmarks import (INF, OracleHeuristic, accuracy_stats, landmark_heuristic,  # noqa: E402
                       remaining_to_goal)
from maps import change_jams, make_map  # noqa: E402
from search import astar, dijkstra  # noqa: E402


def jammed_city(seed, size=40):
    return make_map("city", size, seed, endpoints="random", weighted=True, local_cost=1,
                    jam_arterials=True, jam_ratio=0.2)


class TestChangeJams(unittest.TestCase):
    def test_new_jams_only_raise_costs(self):
        case = jammed_city(1)
        before = list(case.grid.costs)
        after = change_jams(case.grid, 1, appear=0.3).costs
        self.assertEqual(case.grid.costs, before)                    # input grid untouched
        self.assertTrue(all(b >= a for a, b in zip(before, after)))
        self.assertTrue(any(b > a for a, b in zip(before, after)))
        self.assertTrue(all((a == WALL) == (b == WALL) for a, b in zip(before, after)))

    def test_cleared_jams_only_lower_costs(self):
        case = jammed_city(2)
        before = case.grid.costs
        after = change_jams(case.grid, 2, clear=0.5).costs
        self.assertTrue(all(b <= a for a, b in zip(before, after)))
        self.assertTrue(any(b < a for a, b in zip(before, after)))
        self.assertTrue(all((a == WALL) == (b == WALL) for a, b in zip(before, after)))

    def test_bigger_change_contains_smaller_one(self):
        g = jammed_city(3).grid
        small = change_jams(g, 3, appear=0.1).costs
        big = change_jams(g, 3, appear=0.3).costs
        for a, s, b in zip(g.costs, small, big):
            if s != a:
                self.assertEqual(b, s)

    def test_same_seed_same_change(self):
        g = jammed_city(4).grid
        self.assertEqual(change_jams(g, 9, appear=0.2, clear=0.2).costs,
                         change_jams(g, 9, appear=0.2, clear=0.2).costs)


class TestOldLandmarks(unittest.TestCase):
    def test_old_landmarks_stay_safe_when_jams_appear(self):
        """Costs only rose: old estimates never overestimate, stay consistent, A* stays optimal."""
        for seed in range(4):
            case = jammed_city(seed)
            new = change_jams(case.grid, seed, appear=0.5)
            h = landmark_heuristic(case.grid, case.start, 4)          # measured BEFORE the change
            t = new.index(case.goal)
            rem = remaining_to_goal(new, t)
            for v, r in enumerate(rem):
                if r == INF:
                    continue
                self.assertLessEqual(h(v, t), r)
                for u, step in new.neighbors(v):                    # consistency with the NEW costs
                    self.assertLessEqual(h(v, t), step + h(u, t))
            self.assertEqual(astar(new, case.start, case.goal, heuristic=h).cost,
                             dijkstra(new, case.start, case.goal).cost)
            self.assertEqual(accuracy_stats(h, t, rem)[1], 0)

    def test_old_landmarks_can_overestimate_after_jams_clear(self):
        found = False
        for seed in range(10):
            case = jammed_city(seed)
            new = change_jams(case.grid, seed, clear=0.5)
            h = landmark_heuristic(case.grid, case.start, 4)
            t = new.index(case.goal)
            if accuracy_stats(h, t, remaining_to_goal(new, t))[1] > 0:
                found = True
                break
        self.assertTrue(found, "clearing half the jams should make some old estimate too high")

    def test_overestimate_share_of_the_ruler(self):
        case = jammed_city(5)
        t = case.grid.index(case.goal)
        rem = remaining_to_goal(case.grid, t)
        self.assertEqual(accuracy_stats(OracleHeuristic(rem, t, 1.0), t, rem), (1.0, 0.0))
        self.assertEqual(accuracy_stats(OracleHeuristic(rem, t, 1.2), t, rem)[1], 1.0)


class TestBenchRules(unittest.TestCase):
    def test_dynamic_condition_keeps_the_old_map(self):
        case, params = bench.build_map(dict(bench.dyn(appear=0.3), size=30), 2026)
        self.assertTrue(hasattr(case.grid, "before"))
        self.assertEqual(params["jam_up"], 0.3)
        old, new = case.grid.before.costs, case.grid.costs
        self.assertTrue(all(b >= a for a, b in zip(old, new)))

    def test_old_landmarks_must_be_optimal_unless_jams_cleared(self):
        longer = SimpleNamespace(found=True, cost=11)
        rose = SimpleNamespace(kind="city", seed=1, params={"jam_up": 0.3, "jam_down": 0.0})
        fell = SimpleNamespace(kind="city", seed=1, params={"jam_up": 0.0, "jam_down": 0.3})
        with self.assertRaises(AssertionError):
            bench.check("alt_old", longer, 10, rose, "exact_if_costs_rise")
        bench.check("alt_old", longer, 10, fell, "exact_if_costs_rise")   # allowed, recorded as wrong

    def test_placements_all_run_and_stay_optimal(self):
        case = make_map("maze", 31, 7, endpoints="random")
        best = dijkstra(case.grid, case.start, case.goal).cost
        for placement in ("farthest", "corners", "random"):
            h = landmark_heuristic(case.grid, case.start, 4, placement=placement)
            self.assertEqual(len(h), 4)
            self.assertEqual(astar(case.grid, case.start, case.goal, heuristic=h).cost, best)


if __name__ == "__main__":
    unittest.main()

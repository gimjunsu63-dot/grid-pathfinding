"""Tests for the extra algorithms: plain-Python heap, bidirectional Dijkstra.

Run: make test
"""

import heapq
import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import binheap  # noqa: E402
from bidirectional import bidirectional_dijkstra  # noqa: E402
from grid import Grid, is_valid_path, path_cost  # noqa: E402
from maps import make_map  # noqa: E402
from search import astar, best_first, dijkstra  # noqa: E402
from test_search import random_grid, reference_cost  # noqa: E402


class TestBinaryHeap(unittest.TestCase):
    def test_sorts_like_heapq(self):
        rng = random.Random(1)
        for _ in range(200):
            items = [(rng.randint(0, 20), rng.randint(0, 5), i) for i in range(rng.randint(0, 60))]
            h = []
            for it in items:
                binheap.push(h, it)
            out = [binheap.pop(h) for _ in range(len(items))]
            self.assertEqual(out, sorted(items))

    def test_interleaved_matches_heapq(self):
        rng = random.Random(2)
        a, b = [], []
        for i in range(2000):
            if a and rng.random() < 0.4:
                self.assertEqual(binheap.pop(a), heapq.heappop(b))
            else:
                item = (rng.random(), i)
                binheap.push(a, item)
                heapq.heappush(b, item)
        self.assertEqual(sorted(a), sorted(b))

    def test_empty_pop_raises(self):
        with self.assertRaises(IndexError):
            binheap.pop([])

    def test_search_identical_with_either_heap(self):
        rng = random.Random(3)
        for _ in range(30):
            grid = random_grid(rng, 25, 25, 0.25, weighted=rng.random() < 0.5)
            for use_h in (False, True):
                a = best_first(grid, (0, 0), (24, 24), use_heuristic=use_h, heap_impl="heapq")
                b = best_first(grid, (0, 0), (24, 24), use_heuristic=use_h, heap_impl="python")
                self.assertEqual((a.cost, a.path, a.expanded, a.stale, a.max_frontier),
                                 (b.cost, b.path, b.expanded, b.stale, b.max_frontier))

    def test_bad_heap_rejected(self):
        with self.assertRaises(ValueError):
            dijkstra(Grid(2, 2, [1] * 4), (0, 0), (1, 1), heap_impl="nope")


class TestBidirectional(unittest.TestCase):
    def test_against_reference(self):
        rng = random.Random(4)
        for case in range(300):
            h, w = rng.randint(1, 12), rng.randint(1, 12)
            grid = random_grid(rng, h, w, rng.choice([0.0, 0.1, 0.2, 0.3, 0.4]), case % 2 == 1)
            start = (rng.randrange(h), rng.randrange(w))
            goal = (rng.randrange(h), rng.randrange(w))
            expected = reference_cost(grid, start, goal)
            with self.subTest(case=case):
                r = bidirectional_dijkstra(grid, start, goal)
                self.assertEqual(r.cost, expected)
                if expected is not None:
                    self.assertTrue(is_valid_path(grid, r.path, start, goal))
                    self.assertEqual(path_cost(grid, r.path), r.cost)

    def test_every_map_kind(self):
        for kind in ("random", "maze", "nopath", "uwall", "terrain", "city"):
            for seed in range(3):
                case = make_map(kind, 41, seed)
                ref = dijkstra(case.grid, case.start, case.goal)
                r = bidirectional_dijkstra(case.grid, case.start, case.goal)
                with self.subTest(kind=kind, seed=seed):
                    self.assertEqual(r.cost, ref.cost)
                    if r.found:
                        self.assertTrue(is_valid_path(case.grid, r.path, case.start, case.goal))

    def test_sealed_goal_found_fast(self):
        case = make_map("nopath", 60, 1)
        r = bidirectional_dijkstra(case.grid, case.start, case.goal)
        self.assertIsNone(r.path)
        self.assertLess(r.expanded, 10)

    def test_trivial_cases(self):
        grid = Grid.from_strings(["..", ".#"])
        self.assertEqual(bidirectional_dijkstra(grid, (0, 0), (0, 0)).cost, 0)
        self.assertIsNone(bidirectional_dijkstra(grid, (0, 0), (1, 1)).path)
        self.assertEqual(bidirectional_dijkstra(grid, (0, 1), (1, 0)).cost, 2)

    def test_python_heap_same_result(self):
        case = make_map("random", 50, 2, weighted=True)
        a = bidirectional_dijkstra(case.grid, case.start, case.goal)
        b = bidirectional_dijkstra(case.grid, case.start, case.goal, heap_impl="python")
        self.assertEqual((a.cost, a.expanded), (b.cost, b.expanded))

    def test_fewer_cells_than_dijkstra_on_open_map(self):
        grid = Grid(60, 60, [1] * 3600)
        a = bidirectional_dijkstra(grid, (0, 0), (59, 59))
        d = dijkstra(grid, (0, 0), (59, 59))
        self.assertEqual(a.cost, d.cost)
        self.assertLess(a.expanded, d.expanded)


class TestWeightedAstar(unittest.TestCase):
    def test_cost_within_weight_bound(self):
        rng = random.Random(5)
        for _ in range(100):
            grid = random_grid(rng, 25, 25, 0.25, weighted=True)
            opt = dijkstra(grid, (0, 0), (24, 24)).cost
            for wgt in (1.5, 2, 3, 5):
                r = astar(grid, (0, 0), (24, 24), weight=wgt)
                self.assertEqual(r.cost is None, opt is None)
                if opt is not None:
                    self.assertGreaterEqual(r.cost, opt)
                    self.assertLessEqual(r.cost, wgt * opt)


if __name__ == "__main__":
    unittest.main()

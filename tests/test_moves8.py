"""Tests for 8-directional movement, the octile heuristic and Jump Point Search.

Run: make test
"""

import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bidirectional import bidirectional_dijkstra  # noqa: E402
from grid import WALL, Grid, Grid8, is_valid_path, octile, path_cost  # noqa: E402
from jps import jump_point_search  # noqa: E402
from maps import make_map  # noqa: E402
from search import astar, bfs, dijkstra  # noqa: E402

MOVES8 = ((-1, 0, 10), (1, 0, 10), (0, -1, 10), (0, 1, 10),
          (-1, -1, 14), (-1, 1, 14), (1, -1, 14), (1, 1, 14))


def reference_cost8(grid, start, goal):
    """Optimal 8-direction cost by repeated relaxation, written without Grid8.neighbors."""
    if not grid.is_open(start) or not grid.is_open(goal):
        return None
    inf = float("inf")
    dist = {start: 0}
    changed = True
    while changed:
        changed = False
        for (r, c), d in list(dist.items()):
            for dr, dc, step in MOVES8:
                q = (r + dr, c + dc)
                if not grid.is_open(q):
                    continue
                if dr and dc and not (grid.is_open((r + dr, c)) and grid.is_open((r, c + dc))):
                    continue
                nd = d + step * grid.costs[grid.index(q)]
                if nd < dist.get(q, inf):
                    dist[q] = nd
                    changed = True
    return dist.get(goal)


def random_grid8(rng, h, w, wall_ratio, weighted):
    costs = [WALL if rng.random() < wall_ratio else (rng.randint(1, 9) if weighted else 1)
             for _ in range(h * w)]
    return Grid8(h, w, costs)


class TestOctile(unittest.TestCase):
    """The octile distance: these are the tests for the function you write."""

    def setUp(self):
        self.g = Grid8(20, 20, [1] * 400)

    def d(self, a, b):
        return octile(self.g, self.g.index(a), self.g.index(b))

    def test_same_cell(self):
        self.assertEqual(self.d((5, 5), (5, 5)), 0)

    def test_straight(self):
        self.assertEqual(self.d((0, 0), (0, 4)), 40)
        self.assertEqual(self.d((7, 3), (2, 3)), 50)

    def test_diagonal(self):
        self.assertEqual(self.d((0, 0), (3, 3)), 42)
        self.assertEqual(self.d((9, 9), (6, 12)), 42)

    def test_mixed(self):
        self.assertEqual(self.d((0, 0), (3, 5)), 62)
        self.assertEqual(self.d((10, 2), (4, 0)), 2 * 14 + 4 * 10)

    def test_symmetric(self):
        rng = random.Random(1)
        for _ in range(200):
            a = (rng.randrange(20), rng.randrange(20))
            b = (rng.randrange(20), rng.randrange(20))
            self.assertEqual(self.d(a, b), self.d(b, a))

    def test_never_more_than_real_cost(self):
        # admissible: octile <= true optimal cost on any 8-dir map
        rng = random.Random(2)
        for _ in range(100):
            g = random_grid8(rng, 12, 12, 0.25, weighted=rng.random() < 0.5)
            s, t = (0, 0), (11, 11)
            opt = reference_cost8(g, s, t)
            if opt is not None and g.is_open(s):
                self.assertLessEqual(octile(g, g.index(s), g.index(t)), opt)


class TestGrid8(unittest.TestCase):
    def test_no_corner_cutting(self):
        g = Grid8.from_grid(Grid.from_strings([".#", ".."]))
        self.assertNotIn(1, [n for n, _ in g.neighbors(g.index((1, 0)))])     # (0,1) is the wall
        nbrs = dict(g.neighbors(g.index((0, 0))))
        self.assertNotIn(g.index((1, 1)), nbrs)                                 # squeezes past the wall
        self.assertEqual(nbrs[g.index((1, 0))], 10)

    def test_diagonal_cost(self):
        g = Grid8.from_grid(Grid.from_strings(["...", ".3.", "..."]))
        nbrs = dict(g.neighbors(g.index((0, 0))))
        self.assertEqual(nbrs[g.index((1, 1))], 14 * 3)
        self.assertEqual(g.move_cost(g.index((0, 0)), g.index((1, 1))), 42)

    def test_algorithms_against_reference(self):
        rng = random.Random(3)
        for case in range(250):
            h, w = rng.randint(1, 10), rng.randint(1, 10)
            g = random_grid8(rng, h, w, rng.choice([0.0, 0.15, 0.3]), weighted=case % 2 == 1)
            s = (rng.randrange(h), rng.randrange(w))
            t = (rng.randrange(h), rng.randrange(w))
            expected = reference_cost8(g, s, t)
            for name, fn in (("dijkstra", dijkstra), ("astar", astar), ("bidir", bidirectional_dijkstra)):
                with self.subTest(case=case, algorithm=name):
                    r = fn(g, s, t)
                    self.assertEqual(r.cost, expected)
                    if expected is not None:
                        self.assertTrue(is_valid_path(g, r.path, s, t))
                        self.assertEqual(path_cost(g, r.path), r.cost)
            r = bfs(g, s, t)
            if expected is not None:
                self.assertGreaterEqual(r.cost, expected)
                self.assertTrue(is_valid_path(g, r.path, s, t))

    def test_bfs_not_optimal_even_with_unit_costs(self):
        # fewest moves != cheapest when diagonal steps cost more
        case = make_map("random", 200, 2026, wall_ratio=0.2)
        g = Grid8.from_grid(case.grid)
        self.assertGreater(bfs(g, case.start, case.goal).cost, dijkstra(g, case.start, case.goal).cost)

    def test_same_connectivity_as_4dir(self):
        for seed in range(5):
            case = make_map("nopath", 40, seed)
            self.assertIsNone(dijkstra(Grid8.from_grid(case.grid), case.start, case.goal).path)


class TestJPS(unittest.TestCase):
    def test_matches_dijkstra_on_random_maps(self):
        rng = random.Random(4)
        for case in range(1500):
            h, w = rng.randint(1, 20), rng.randint(1, 20)
            g = random_grid8(rng, h, w, rng.choice([0.0, 0.1, 0.2, 0.3, 0.4]), weighted=False)
            s = (rng.randrange(h), rng.randrange(w))
            t = (rng.randrange(h), rng.randrange(w))
            ref = dijkstra(g, s, t)
            with self.subTest(case=case):
                r = jump_point_search(g, s, t)
                self.assertEqual(r.cost, ref.cost)
                if r.found:
                    self.assertTrue(is_valid_path(g, r.path, s, t))
                    self.assertEqual(path_cost(g, r.path), r.cost)

    def test_map_kinds(self):
        for kind, params in (("random", {}), ("maze", {}), ("uwall", {}), ("city", {"weighted": False}),
                             ("nopath", {})):
            case = make_map(kind, 61, 7, **params)
            g = Grid8.from_grid(case.grid)
            with self.subTest(kind=kind):
                self.assertEqual(jump_point_search(g, case.start, case.goal).cost,
                                 dijkstra(g, case.start, case.goal).cost)

    def test_open_grid_needs_few_jump_points(self):
        g = Grid8(50, 50, [1] * 2500)
        r = jump_point_search(g, (0, 0), (49, 49))
        self.assertEqual(r.cost, 49 * 14)
        self.assertLessEqual(r.expanded, 3)
        self.assertGreater(r.scanned, 49)

    def test_rejects_weighted_and_4dir(self):
        with self.assertRaises(ValueError):
            jump_point_search(Grid8(2, 2, [1, 2, 1, 1]), (0, 0), (1, 1))
        with self.assertRaises(ValueError):
            jump_point_search(Grid(2, 2, [1] * 4), (0, 0), (1, 1))

    def test_trivial(self):
        g = Grid8(3, 3, [1] * 9)
        self.assertEqual(jump_point_search(g, (1, 1), (1, 1)).cost, 0)


if __name__ == "__main__":
    unittest.main()

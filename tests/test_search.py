"""Unit tests for grid search. Standard library unittest only.

Run: make test

Three layers:
  1. hand-made maps whose answer is known
  2. random maps checked against an independent reference (Bellman-Ford style
     relaxation that shares no code with the algorithms under test)
  3. random maps checked for agreement between the three algorithms
"""

import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from grid import WALL, Grid, is_valid_path, path_cost  # noqa: E402
from search import ALGORITHMS, TIE_BREAKS, astar, best_first, bfs, dijkstra  # noqa: E402

SEED = 2026


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def random_grid(rng, height, width, wall_ratio, weighted):
    costs = []
    for _ in range(height * width):
        if rng.random() < wall_ratio:
            costs.append(WALL)
        else:
            costs.append(rng.randint(1, 9) if weighted else 1)
    return Grid(height, width, costs)


def reference_cost(grid, start, goal):
    """Optimal cost by repeated edge relaxation. Slow but obviously correct.

    Returns None if goal is unreachable.
    """
    if not grid.is_open(start) or not grid.is_open(goal):
        return None
    inf = float("inf")
    n = grid.height * grid.width
    dist = [inf] * n
    dist[grid.index(start)] = 0
    changed = True
    while changed:
        changed = False
        for r in range(grid.height):
            for c in range(grid.width):
                if not grid.is_open((r, c)):
                    continue
                d = dist[grid.index((r, c))]
                if d == inf:
                    continue
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    q = (r + dr, c + dc)
                    if grid.is_open(q):
                        nd = d + grid.costs[grid.index(q)]
                        if nd < dist[grid.index(q)]:
                            dist[grid.index(q)] = nd
                            changed = True
    d = dist[grid.index(goal)]
    return None if d == inf else d


def reachable_count(grid, start):
    """Number of open cells reachable from start (flood fill)."""
    if not grid.is_open(start):
        return 0
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            q = (r + dr, c + dc)
            if grid.is_open(q) and q not in seen:
                seen.add(q)
                stack.append(q)
    return len(seen)


# ---------------------------------------------------------------------------
# 1. hand-made cases
# ---------------------------------------------------------------------------


class TestHandMade(unittest.TestCase):
    def check_all(self, rows, start, goal, expected_cost):
        grid = Grid.from_strings(rows)
        for name, fn in ALGORITHMS.items():
            with self.subTest(algorithm=name):
                r = fn(grid, start, goal)
                if expected_cost is None:
                    self.assertIsNone(r.path)
                    self.assertIsNone(r.cost)
                else:
                    self.assertEqual(r.cost, expected_cost)
                    self.assertTrue(is_valid_path(grid, r.path, start, goal))
                    self.assertEqual(path_cost(grid, r.path), r.cost)

    def test_open_grid(self):
        self.check_all(["...", "...", "..."], (0, 0), (2, 2), 4)

    def test_wall_forces_detour(self):
        self.check_all([".#.", ".#.", "..."], (0, 0), (0, 2), 6)

    def test_start_equals_goal(self):
        self.check_all(["..", ".."], (1, 1), (1, 1), 0)

    def test_goal_enclosed(self):
        self.check_all(["...", ".##", ".#."], (0, 0), (2, 2), None)

    def test_start_on_wall(self):
        self.check_all(["#.", ".."], (0, 0), (1, 1), None)

    def test_goal_on_wall(self):
        self.check_all(["..", ".#"], (0, 0), (1, 1), None)

    def test_single_cell(self):
        self.check_all(["."], (0, 0), (0, 0), 0)

    def test_bfs_not_optimal_with_weights(self):
        grid = Grid.from_strings([".99999.", ".#####.", "......."])
        start, goal = (0, 0), (0, 6)
        self.assertEqual(bfs(grid, start, goal).cost, 46)       # 6 moves through the 9s
        self.assertEqual(dijkstra(grid, start, goal).cost, 10)  # 10 moves around
        self.assertEqual(astar(grid, start, goal).cost, 10)

    def test_bfs_finds_fewest_moves(self):
        grid = Grid.from_strings([".99999.", ".#####.", "......."])
        r = bfs(grid, (0, 0), (0, 6))
        self.assertEqual(len(r.path) - 1, 6)


# ---------------------------------------------------------------------------
# 2. random maps vs. independent reference
# ---------------------------------------------------------------------------


class TestAgainstReference(unittest.TestCase):
    def test_random_maps(self):
        rng = random.Random(SEED)
        for case in range(300):
            h = rng.randint(1, 12)
            w = rng.randint(1, 12)
            weighted = case % 2 == 1
            grid = random_grid(rng, h, w, rng.choice([0.0, 0.1, 0.2, 0.3, 0.4]), weighted)
            start = (rng.randrange(h), rng.randrange(w))
            goal = (rng.randrange(h), rng.randrange(w))
            expected = reference_cost(grid, start, goal)
            with self.subTest(case=case, weighted=weighted):
                for name in ("dijkstra", "astar"):
                    r = ALGORITHMS[name](grid, start, goal)
                    self.assertEqual(r.cost, expected, name)
                    if expected is not None:
                        self.assertTrue(is_valid_path(grid, r.path, start, goal), name)
                        self.assertEqual(path_cost(grid, r.path), r.cost, name)
                r = bfs(grid, start, goal)
                if expected is None:
                    self.assertIsNone(r.path)
                else:
                    self.assertTrue(is_valid_path(grid, r.path, start, goal))
                    self.assertEqual(path_cost(grid, r.path), r.cost)
                    if weighted:
                        self.assertGreaterEqual(r.cost, expected)
                    else:
                        self.assertEqual(r.cost, expected)


# ---------------------------------------------------------------------------
# 3. agreement between algorithms and counter sanity
# ---------------------------------------------------------------------------


class TestAgreement(unittest.TestCase):
    def setUp(self):
        self.rng = random.Random(SEED + 1)

    def test_tie_breaks_keep_optimal_cost(self):
        for _ in range(100):
            grid = random_grid(self.rng, 20, 20, 0.25, weighted=True)
            start, goal = (0, 0), (19, 19)
            costs = {astar(grid, start, goal, tie_break=tb).cost for tb in TIE_BREAKS}
            costs.add(dijkstra(grid, start, goal).cost)
            self.assertEqual(len(costs), 1)

    def test_no_path_expands_whole_component(self):
        # goal walled off: every algorithm must visit every reachable cell
        for _ in range(50):
            grid = random_grid(self.rng, 15, 15, 0.2, weighted=self.rng.random() < 0.5)
            goal = (14, 14)
            for q in ((13, 14), (14, 13)):
                grid.costs[grid.index(q)] = WALL
            grid.costs[grid.index(goal)] = 1
            grid.costs[grid.index((0, 0))] = 1
            n = reachable_count(grid, (0, 0))
            for name, fn in ALGORITHMS.items():
                r = fn(grid, (0, 0), goal)
                self.assertIsNone(r.path, name)
                self.assertEqual(r.expanded, n, name)

    def test_unit_cost_bfs_and_dijkstra_same_cost(self):
        for _ in range(100):
            grid = random_grid(self.rng, 25, 25, 0.25, weighted=False)
            a = bfs(grid, (0, 0), (24, 24))
            b = dijkstra(grid, (0, 0), (24, 24))
            self.assertEqual(a.cost, b.cost)

    def test_astar_expands_no_more_than_dijkstra_on_open_grid(self):
        grid = Grid(30, 30, [1] * 900)
        a = astar(grid, (0, 0), (29, 29))
        d = dijkstra(grid, (0, 0), (29, 29))
        self.assertEqual(a.cost, d.cost)
        self.assertLess(a.expanded, d.expanded)

    def test_counters_are_sane(self):
        for _ in range(50):
            grid = random_grid(self.rng, 15, 15, 0.2, weighted=True)
            n = reachable_count(grid, (0, 0))
            for name, fn in ALGORITHMS.items():
                r = fn(grid, (0, 0), (14, 14))
                self.assertGreaterEqual(r.stale, 0, name)
                self.assertLessEqual(r.expanded, n, name)
                self.assertGreaterEqual(r.max_frontier, 0, name)
            self.assertEqual(bfs(grid, (0, 0), (14, 14)).stale, 0)

    def test_weight_one_equals_astar(self):
        grid = random_grid(self.rng, 20, 20, 0.2, weighted=True)
        a = astar(grid, (0, 0), (19, 19))
        b = best_first(grid, (0, 0), (19, 19), use_heuristic=True, weight=1.0)
        self.assertEqual((a.cost, a.expanded), (b.cost, b.expanded))

    def test_bad_tie_break_rejected(self):
        grid = Grid(2, 2, [1] * 4)
        with self.assertRaises(ValueError):
            best_first(grid, (0, 0), (1, 1), tie_break="nope")


if __name__ == "__main__":
    unittest.main()

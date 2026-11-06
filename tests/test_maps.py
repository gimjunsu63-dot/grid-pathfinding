"""Tests for the map generators and the benchmark runner.

Run: make test
"""

import csv
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import bench  # noqa: E402
from grid import WALL  # noqa: E402
from maps import (MAP_KINDS, component, is_connected, make_map,  # noqa: E402
                  maze_map, random_map)
from search import ALGORITHMS, astar, dijkstra  # noqa: E402


def neighbor_pairs(grid):
    w = grid.width
    for r in range(grid.height):
        for c in range(w):
            i = r * w + c
            if c + 1 < w:
                yield i, i + 1
            if r + 1 < grid.height:
                yield i, i + w


class TestEveryKind(unittest.TestCase):
    def test_deterministic(self):
        for kind in MAP_KINDS:
            with self.subTest(kind=kind):
                a = make_map(kind, 41, seed=7)
                b = make_map(kind, 41, seed=7)
                c = make_map(kind, 41, seed=8)
                self.assertEqual(a.grid.costs, b.grid.costs)
                self.assertEqual((a.start, a.goal), (b.start, b.goal))
                if kind != "uwall":      # uwall has no randomness unless wall_ratio > 0
                    self.assertNotEqual(a.grid.costs, c.grid.costs)
        a = make_map("uwall", 41, seed=7, wall_ratio=0.1)
        c = make_map("uwall", 41, seed=8, wall_ratio=0.1)
        self.assertNotEqual(a.grid.costs, c.grid.costs)

    def test_endpoints_open_and_costs_in_range(self):
        for kind in MAP_KINDS:
            for seed in range(5):
                with self.subTest(kind=kind, seed=seed):
                    case = make_map(kind, 41, seed)
                    self.assertTrue(case.grid.is_open(case.start))
                    self.assertTrue(case.grid.is_open(case.goal))
                    for v in case.grid.costs:
                        self.assertTrue(v == WALL or 1 <= v <= 9)

    def test_connected_except_nopath(self):
        for kind in MAP_KINDS:
            for seed in range(5):
                with self.subTest(kind=kind, seed=seed):
                    case = make_map(kind, 41, seed)
                    connected = is_connected(case.grid, case.start, case.goal)
                    self.assertEqual(connected, kind != "nopath")

    def test_every_algorithm_agrees_with_connectivity(self):
        for kind in MAP_KINDS:
            case = make_map(kind, 41, seed=3)
            optimal = dijkstra(case.grid, case.start, case.goal).cost
            for name, fn in ALGORITHMS.items():
                with self.subTest(kind=kind, algorithm=name):
                    r = fn(case.grid, case.start, case.goal)
                    self.assertEqual(r.found, kind != "nopath")
                    if r.found:
                        self.assertGreaterEqual(r.cost, optimal)


class TestRandom(unittest.TestCase):
    def test_wall_ratio_roughly_respected(self):
        for p in (0.0, 0.1, 0.2, 0.3):
            case = random_map(100, 100, seed=1, wall_ratio=p)
            walls = sum(1 for v in case.grid.costs if v == WALL) / 10000
            self.assertAlmostEqual(walls, p, delta=0.02)

    def test_unit_and_weighted_share_walls(self):
        for seed in range(5):
            a = random_map(60, 60, seed, 0.25, weighted=False)
            b = random_map(60, 60, seed, 0.25, weighted=True)
            self.assertEqual([v == WALL for v in a.grid.costs], [v == WALL for v in b.grid.costs])
            self.assertEqual(a.attempts, b.attempts)
            self.assertTrue(any(v > 1 for v in b.grid.costs))

    def test_attempts_counted(self):
        # 35% walls disconnects often, so some seeds need several draws
        # (above ~41% walls a random grid almost never connects: percolation threshold)
        attempts = [random_map(30, 30, s, 0.35).attempts for s in range(20)]
        self.assertTrue(any(a > 1 for a in attempts))


class TestMaze(unittest.TestCase):
    def test_perfect_maze_is_a_tree(self):
        # spanning tree: open cells all connected, edges = cells - 1
        for size in (21, 40):
            case = maze_map(size, size, seed=5)
            g = case.grid
            open_cells = [i for i, v in enumerate(g.costs) if v != WALL]
            edges = sum(1 for a, b in neighbor_pairs(g) if g.costs[a] and g.costs[b])
            self.assertEqual(edges, len(open_cells) - 1)
            self.assertEqual(len(component(g, case.start)), len(open_cells))

    def test_loops_add_edges(self):
        a = maze_map(41, 41, seed=5)
        b = maze_map(41, 41, seed=5, loop_ratio=0.2)
        ea = sum(1 for x, y in neighbor_pairs(a.grid) if a.grid.costs[x] and a.grid.costs[y])
        eb = sum(1 for x, y in neighbor_pairs(b.grid) if b.grid.costs[x] and b.grid.costs[y])
        self.assertGreater(eb, ea)

    def test_large_maze_no_recursion_error(self):
        case = maze_map(401, 401, seed=1)
        self.assertTrue(is_connected(case.grid, case.start, case.goal))


class TestSpecialKinds(unittest.TestCase):
    def test_uwall_blocks_straight_line(self):
        case = make_map("uwall", 61, seed=1)
        optimal = dijkstra(case.grid, case.start, case.goal).cost
        straight = abs(case.goal[1] - case.start[1])
        self.assertGreater(optimal, straight)

    def test_uwall_traps_astar(self):
        # inside the cup A* expands far more cells than the path length
        case = make_map("uwall", 61, seed=1)
        r = astar(case.grid, case.start, case.goal)
        self.assertGreater(r.expanded, 3 * len(r.path))

    def test_terrain_is_clustered(self):
        case = make_map("terrain", 80, seed=2)
        g = case.grid
        diffs = [abs(g.costs[a] - g.costs[b]) for a, b in neighbor_pairs(g)]
        # independent 1..9 costs would average about 2.96
        self.assertLess(sum(diffs) / len(diffs), 1.0)
        self.assertNotIn(WALL, g.costs)

    def test_city_buildings_and_roads(self):
        case = make_map("city", 43, seed=1, block=6, closure_ratio=0.0)
        g = case.grid
        self.assertEqual(g.costs[g.index((3, 3))], WALL)        # inside a block
        self.assertNotEqual(g.costs[g.index((0, 3))], WALL)     # road
        self.assertEqual(g.costs[g.index((0, 3))], 1)           # arterial (first road)
        self.assertIn(g.costs[g.index((7, 3))], (3, 9))         # local road

    def test_random_endpoints(self):
        for seed in range(10):
            case = make_map("random", 50, seed, endpoints="random")
            self.assertTrue(is_connected(case.grid, case.start, case.goal))
            d = abs(case.start[0] - case.goal[0]) + abs(case.start[1] - case.goal[1])
            self.assertGreaterEqual(d, 25)

    def test_bad_arguments(self):
        with self.assertRaises(ValueError):
            make_map("nope", 10, 1)
        with self.assertRaises(ValueError):
            make_map("nopath", 10, 1, endpoints="random")


class TestBench(unittest.TestCase):
    def test_quick_run_writes_consistent_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            bench.run(["A", "C", "noise"], Path(tmp), n_maps=2, repeats=2, quick=True,
                      log=lambda *a: None)
            with open(Path(tmp) / "A.csv") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 4 * 2 * 2 * 3)   # conditions x maps x repeats x algorithms
            for row in rows:
                self.assertEqual(row["found"], "1")
                if row["algorithm"] in ("dijkstra", "astar"):
                    self.assertEqual(row["cost"], row["optimal_cost"])
                else:
                    self.assertGreaterEqual(int(row["cost"]), int(row["optimal_cost"]))
            # every algorithm runs first in some repetition
            firsts = {r["algorithm"] for r in rows if r["order"] == "0"}
            self.assertEqual(firsts, {"bfs", "dijkstra"})   # 2 repeats -> 2 rotations
            self.assertTrue((Path(tmp) / "A.env.json").exists())
            self.assertTrue((Path(tmp) / "noise.csv").exists())

    def test_check_catches_wrong_cost(self):
        case = make_map("random", 20, 1)
        r = dijkstra(case.grid, case.start, case.goal)
        with self.assertRaises(AssertionError):
            bench.check("dijkstra", r, r.cost - 1, case)


if __name__ == "__main__":
    unittest.main()

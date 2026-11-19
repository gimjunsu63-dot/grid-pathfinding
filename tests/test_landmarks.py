"""Tests for the landmark heuristic, the accuracy ruler and the accuracy measure.

Run: make test
"""

import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from grid import WALL, Grid  # noqa: E402
from landmarks import (INF, PLACEMENTS, LandmarkHeuristic, OracleHeuristic, accuracy,  # noqa: E402
                       choose_landmarks, distances_from, landmark_heuristic, remaining_to_goal)
from maps import component, make_map  # noqa: E402
from search import astar, dijkstra  # noqa: E402
from test_search import random_grid, reference_cost  # noqa: E402


def random_case(rng, weighted):
    while True:
        g = random_grid(rng, 14, 14, 0.25, weighted)
        open_cells = [i for i, c in enumerate(g.costs) if c != WALL]
        if len(open_cells) > 10:
            return g, open_cells


class TestDistances(unittest.TestCase):
    def test_distances_from_matches_reference(self):
        rng = random.Random(1)
        for _ in range(40):
            g, cells = random_case(rng, weighted=True)
            src = rng.choice(cells)
            dist = distances_from(g, src)
            for v in rng.sample(cells, 8):
                ref = reference_cost(g, g.pos(src), g.pos(v))
                self.assertEqual(dist[v], INF if ref is None else ref)

    def test_remaining_uses_entry_costs_correctly(self):
        # d(v -> goal) differs from d(goal -> v) when costs differ: check against reference
        rng = random.Random(2)
        for _ in range(40):
            g, cells = random_case(rng, weighted=True)
            t = rng.choice(cells)
            rem = remaining_to_goal(g, t)
            for v in rng.sample(cells, 8):
                ref = reference_cost(g, g.pos(v), g.pos(t))
                self.assertEqual(rem[v], INF if ref is None else ref)

    def test_two_cell_example(self):
        g = Grid.from_strings(["19"])
        self.assertEqual(remaining_to_goal(g, 1)[0], 9)     # 1 -> 9 pays 9
        self.assertEqual(remaining_to_goal(g, 0)[1], 1)     # 9 -> 1 pays 1


class TestLandmarkHeuristic(unittest.TestCase):
    def test_never_overestimates_and_is_consistent(self):
        rng = random.Random(3)
        for case in range(60):
            g, cells = random_case(rng, weighted=case % 2 == 0)
            start = g.pos(rng.choice(cells))
            area = component(g, start)
            for placement in PLACEMENTS:
                marks, dists = choose_landmarks(g, rng.randint(1, 6), area, placement, seed=case)
                h = LandmarkHeuristic(g, marks, dists)
                t = rng.choice(sorted(area))
                rem = remaining_to_goal(g, t)
                for v in area:
                    with self.subTest(case=case, placement=placement, v=v):
                        self.assertLessEqual(h(v, t), rem[v])               # admissible
                        for nxt, step in g.neighbors(v):
                            self.assertLessEqual(h(v, t), step + h(nxt, t))  # consistent

    def test_astar_with_landmarks_is_optimal(self):
        for kind, params in (("random", {"weighted": True}), ("maze", {}), ("terrain", {}),
                             ("city", {"weighted": True}), ("random", {})):
            for seed in range(3):
                case = make_map(kind, 41, seed, endpoints="random", **params)
                h = landmark_heuristic(case.grid, case.start, 4)
                with self.subTest(kind=kind, seed=seed):
                    self.assertEqual(astar(case.grid, case.start, case.goal, heuristic=h).cost,
                                     dijkstra(case.grid, case.start, case.goal).cost)

    def test_landmark_at_goal_is_exact(self):
        case = make_map("random", 30, 1, weighted=True)
        g = case.grid
        t = g.index(case.goal)
        h = LandmarkHeuristic(g, [t], [distances_from(g, t)])
        rem = remaining_to_goal(g, t)
        for v in component(g, case.start):
            self.assertEqual(h(v, t), rem[v])

    def test_more_farthest_landmarks_never_less_accurate(self):
        case = make_map("random", 40, 2, weighted=True, endpoints="random")
        g = case.grid
        t = g.index(case.goal)
        rem = remaining_to_goal(g, t)
        accs = [accuracy(landmark_heuristic(g, case.start, n), t, rem) for n in (1, 2, 4, 8)]
        self.assertEqual(accs, sorted(accs))

    def test_bad_placement_rejected(self):
        g = Grid(3, 3, [1] * 9)
        with self.assertRaises(ValueError):
            choose_landmarks(g, 2, set(range(9)), placement="nope")


class TestRulerAndAccuracy(unittest.TestCase):
    def setUp(self):
        self.case = make_map("random", 40, 3, weighted=True, endpoints="random")
        self.g = self.case.grid
        self.t = self.g.index(self.case.goal)
        self.rem = remaining_to_goal(self.g, self.t)

    def test_ruler_accuracy_equals_k(self):
        for k in (0, 0.5, 0.9, 1.0):
            acc = accuracy(OracleHeuristic(self.rem, self.t, k), self.t, self.rem)
            self.assertAlmostEqual(acc, k)

    def test_dijkstra_and_manhattan_accuracy_range(self):
        self.assertEqual(accuracy(lambda v, t: 0, self.t, self.rem), 0)
        m = accuracy(self.g.heuristic, self.t, self.rem)
        self.assertGreater(m, 0)
        self.assertLess(m, 1)

    def test_more_accurate_ruler_expands_fewer_cells(self):
        s, goal = self.case.start, self.case.goal
        cells = [astar(self.g, s, goal, heuristic=OracleHeuristic(self.rem, self.t, k)).expanded
                 for k in (0.25, 0.5, 0.8, 1.0)]
        self.assertEqual(cells, sorted(cells, reverse=True))

    def test_ruler_built_for_other_goal_rejected(self):
        h = OracleHeuristic(self.rem, self.t, 1.0)
        with self.assertRaises(ValueError):
            h(0, self.t + 1)

    def test_overestimating_ruler_stays_within_bound(self):
        s, goal = self.case.start, self.case.goal
        opt = dijkstra(self.g, s, goal).cost
        for k in (1.2, 1.5, 3.0):
            r = astar(self.g, s, goal, heuristic=OracleHeuristic(self.rem, self.t, k))
            self.assertLessEqual(r.cost, k * opt)


if __name__ == "__main__":
    unittest.main()
